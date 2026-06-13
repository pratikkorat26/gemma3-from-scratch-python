from collections import deque
import time
from typing import Deque, Dict, Iterator, List, Optional, Union

import torch

from .config import EngineConfig, SamplingConfig
from .kv import KVBlockManager
from .sampling import apply_repetition_penalty_, sample_next_token
from .types import EngineStats, GenerateRequest, GenerationResult, RequestState, StreamEvent
from gemma3.tokenizer import apply_chat_template


class LLMEngine:
    """
    Single-device scheduler with explicit prefill/decode phases.

    - Prefill phase: one full-prompt step per request, then move request to decode queue.
    - Decode phase: token-level round-robin over active requests.
    """

    def __init__(self, runtime: object, config: EngineConfig):
        self.runtime = runtime
        self.config = config
        self.capacity = KVBlockManager(
            max_kv_cache_tokens=self.config.max_kv_cache_tokens,
            block_size=self.config.kv_block_size,
            num_blocks=self.config.num_kv_blocks,
        )
        self.paged_kv_caches = self.runtime.model.init_paged_kv_caches(
            num_blocks=self.capacity.num_blocks,
            block_size=self.capacity.block_size,
            device=self.runtime.device,
        )

    def _encode_prompt(self, prompt: str) -> List[int]:
        if self.config.use_instruct_model:
            prompt = apply_chat_template(prompt)
        return self.runtime.tokenizer.encode(prompt)

    def _build_request(
        self,
        request_id: int,
        prompt: str,
        sampling: Optional[SamplingConfig],
        max_new_tokens: Optional[int],
        created_at_s: float,
    ) -> RequestState:
        return RequestState.from_prompt(
            request_id=request_id,
            prompt_token_ids=self._encode_prompt(prompt),
            sampling=sampling or self.config.sampling,
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens is not None
                else self.config.max_new_tokens
            ),
            eos_token_id=self.runtime.tokenizer.eos_token_id,
            created_at_s=created_at_s,
        )

    def _context_limit(self) -> int:
        return int(self.runtime.model.context_length)

    def _release_request_capacity(self, request: RequestState) -> None:
        self.capacity.release(request)

    def _make_sampling_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        generator = torch.Generator(device=self.runtime.device)
        generator.manual_seed(int(seed))
        return generator

    def _finish_request(self, request: RequestState, reason: str, error_message: Optional[str] = None) -> None:
        request.status = "finished" if reason != "error" else "error"
        request.phase = request.status
        request.stop_reason = reason
        request.error_message = error_message
        request.finished_at_s = time.perf_counter()
        self._release_request_capacity(request)

    def _init_request(self, request: RequestState) -> None:
        request.status = "active"
        request.phase = "prefill"
        request.generated_ids = []
        request.all_token_ids = list(request.prompt_token_ids)
        request.seen_token_ids = set(request.prompt_token_ids)
        request.text_chunks = []
        request.past_kv = None
        request.block_table = []
        request.prompt_cursor = 0
        request.num_computed_tokens = 0
        request.live_kv_tokens = 0
        request.stop_reason = None
        request.error_message = None
        request.sampling_seed = getattr(request.sampling, "seed", None)
        request.sampling_generator = self._make_sampling_generator(request.sampling_seed)
        request.prefill_time_s = 0.0
        request.prefill_steps = 0
        request.decode_time_s = 0.0
        request.decode_steps = 0
        request.current_input = self._next_prefill_chunk(request)

    def _make_result(self, request: RequestState) -> GenerationResult:
        first_scheduled = request.first_scheduled_at_s if request.first_scheduled_at_s is not None else request.created_at_s
        finished = request.finished_at_s if request.finished_at_s is not None else time.perf_counter()
        queue_wait_s = max(0.0, first_scheduled - request.created_at_s)
        total_latency_s = max(0.0, finished - request.created_at_s)
        model_time_s = request.prefill_time_s + request.decode_time_s
        model_tokens_per_s = 0.0 if model_time_s <= 0 else len(request.generated_ids) / model_time_s
        text_ids = list(request.generated_ids)
        if (
            request.stop_reason == "eos"
            and request.eos_token_id is not None
            and text_ids
            and text_ids[-1] == request.eos_token_id
        ):
            text_ids = text_ids[:-1]

        return GenerationResult(
            request_id=request.request_id,
            text=self.runtime.tokenizer.decode(text_ids) if text_ids else "",
            token_ids=list(request.generated_ids),
            stop_reason=request.stop_reason or "error",
            error_message=request.error_message,
            queue_wait_s=queue_wait_s,
            prefill_s=request.prefill_time_s,
            decode_s=request.decode_time_s,
            total_latency_s=total_latency_s,
            model_tokens_per_s=model_tokens_per_s,
            prefill_steps=request.prefill_steps,
            decode_steps=request.decode_steps,
        )

    def _prefill_chunk_size(self, request: RequestState) -> int:
        chunk_size = self.config.prefill_chunk_size
        if chunk_size is None or chunk_size <= 0:
            return max(1, len(request.prompt_token_ids) - request.prompt_cursor)
        return max(1, int(chunk_size))

    def _next_prefill_chunk(self, request: RequestState) -> Optional[torch.Tensor]:
        if request.prompt_cursor >= len(request.prompt_token_ids):
            return None
        chunk_size = self._prefill_chunk_size(request)
        chunk_ids = request.prompt_token_ids[request.prompt_cursor : request.prompt_cursor + chunk_size]
        if not chunk_ids:
            return None
        return torch.tensor(chunk_ids, device=self.runtime.device).unsqueeze(0)

    def _prefill_complete(self, request: RequestState) -> bool:
        return request.prompt_cursor >= len(request.prompt_token_ids)

    def _build_block_tables(self, requests: List[RequestState]) -> torch.Tensor:
        max_blocks = max(len(request.block_table) for request in requests)
        block_tables = torch.full(
            (len(requests), max_blocks),
            fill_value=-1,
            dtype=torch.long,
            device=self.runtime.device,
        )
        for row_idx, request in enumerate(requests):
            if request.block_table:
                block_tables[row_idx, : len(request.block_table)] = torch.tensor(
                    request.block_table,
                    dtype=torch.long,
                    device=self.runtime.device,
                )
        return block_tables

    def _prepare_paged_batch(
        self,
        requests: List[RequestState],
        *,
        defer_on_capacity: bool,
    ) -> tuple[List[RequestState], List[RequestState]]:
        eligible: List[RequestState] = []
        blocked: List[RequestState] = []
        for request in requests:
            if request.current_input is None:
                self._finish_request(request, reason="error", error_message="request tensors are not initialized")
                continue
            if len(request.all_token_ids) >= self._context_limit():
                self._finish_request(request, reason="context_limit")
                continue
            target_tokens = request.live_kv_tokens + int(request.current_input.shape[1])
            if not self.capacity.ensure_capacity(request, target_tokens):
                if defer_on_capacity:
                    blocked.append(request)
                else:
                    self._finish_request(
                        request,
                        reason="capacity_exceeded",
                        error_message="KV cache capacity exceeded",
                    )
                continue
            eligible.append(request)
        return eligible, blocked

    def _sample_next_tokens(self, logits: torch.Tensor, requests: List[RequestState]) -> torch.Tensor:
        with torch.inference_mode():
            next_logits = logits[:, -1, :].clone()
            next_logits = apply_repetition_penalty_(
                next_logits,
                [request.seen_token_ids for request in requests],
                penalty=requests[0].sampling.repetition_penalty,
            )
            if all(request.sampling_generator is None for request in requests):
                return sample_next_token(
                    next_logits,
                    temperature=requests[0].sampling.temperature,
                    top_p=requests[0].sampling.top_p,
                    top_k=requests[0].sampling.top_k,
                )

            sampled_tokens = [
                sample_next_token(
                    next_logits[row_idx : row_idx + 1],
                    temperature=request.sampling.temperature,
                    top_p=request.sampling.top_p,
                    top_k=request.sampling.top_k,
                    generator=request.sampling_generator,
                )
                for row_idx, request in enumerate(requests)
            ]
            return torch.cat(sampled_tokens, dim=0)

    def _forward_paged_batch(self, requests: List[RequestState]) -> torch.Tensor:
        eligible, _ = self._prepare_paged_batch(requests, defer_on_capacity=False)
        if not eligible:
            return torch.empty(0, 1, 0, device=self.runtime.device)

        current_input = torch.cat([request.current_input for request in eligible], dim=0)
        block_tables = self._build_block_tables(eligible)
        kv_lens = torch.tensor(
            [request.live_kv_tokens for request in eligible],
            dtype=torch.long,
            device=self.runtime.device,
        )

        with torch.inference_mode():
            logits = self.runtime.model(
                current_input,
                block_tables=block_tables,
                kv_lens=kv_lens,
                paged_kv_caches=self.paged_kv_caches,
            )

        for request in eligible:
            request.live_kv_tokens += int(request.current_input.shape[1])
            request.num_computed_tokens = request.live_kv_tokens
        return logits

    def _record_next_token(self, request: RequestState, next_token: torch.Tensor, *, emit_text: bool) -> None:
        next_id = int(next_token.item())
        request.generated_ids.append(next_id)

        if request.eos_token_id is not None and next_id == request.eos_token_id:
            self._finish_request(request, reason="eos")
            return

        request.all_token_ids.append(next_id)
        request.seen_token_ids.add(next_id)
        request.current_input = next_token
        if emit_text:
            request.text_chunks.append(self.runtime.tokenizer.decode([next_id]))

        if len(request.generated_ids) >= request.max_new_tokens:
            self._finish_request(request, reason="max_new_tokens")
            return

        request.phase = "decode"

    def _run_one_step(self, request: RequestState, phase: str, *, emit_text: bool = False) -> None:
        if len(request.all_token_ids) >= self._context_limit():
            self._finish_request(request, reason="context_limit")
            return

        started = time.perf_counter()
        logits = self._forward_paged_batch([request])
        if request.status in ("finished", "error"):
            return

        next_token = self._sample_next_tokens(logits, [request])
        elapsed = time.perf_counter() - started

        if phase == "prefill":
            request.prefill_time_s += elapsed
        else:
            request.decode_time_s += elapsed
        request.decode_steps += 1

        self._record_next_token(request, next_token, emit_text=emit_text)

    def _run_prefill_chunk(self, request: RequestState) -> Optional[torch.Tensor]:
        if request.current_input is None:
            request.current_input = self._next_prefill_chunk(request)
        if request.current_input is None:
            return None

        started = time.perf_counter()
        logits = self._forward_paged_batch([request])
        if request.status in ("finished", "error"):
            return None

        elapsed = time.perf_counter() - started
        request.prefill_time_s += elapsed
        request.prefill_steps += 1
        request.prompt_cursor += int(request.current_input.shape[1])

        if self._prefill_complete(request):
            return logits

        request.current_input = self._next_prefill_chunk(request)
        request.phase = "prefill"
        return None

    def _sampling_key(self, request: RequestState):
        return (
            request.sampling.temperature,
            request.sampling.top_p,
            request.sampling.top_k,
            request.sampling.repetition_penalty,
        )

    def _select_decode_batch(self, decode_queue: Deque[RequestState]) -> List[RequestState]:
        max_batch = max(1, int(self.config.max_decode_batch_size))
        selection_window = max(max_batch, int(self.config.decode_selection_window))

        window: List[RequestState] = []
        while decode_queue and len(window) < selection_window:
            window.append(decode_queue.popleft())

        cohorts: Dict[tuple, List[int]] = {}
        for idx, request in enumerate(window):
            cohort_key = (len(request.all_token_ids), self._sampling_key(request))
            cohorts.setdefault(cohort_key, []).append(idx)

        best_indices: List[int] = []
        best_first_idx = len(window)
        for indices in cohorts.values():
            if len(indices) > len(best_indices):
                best_indices = indices
                best_first_idx = indices[0]
                continue
            if len(indices) == len(best_indices) and indices and indices[0] < best_first_idx:
                best_indices = indices
                best_first_idx = indices[0]

        selected_positions = set(best_indices[:max_batch])
        batch = [request for idx, request in enumerate(window) if idx in selected_positions]
        remaining = [request for idx, request in enumerate(window) if idx not in selected_positions]

        for request in reversed(remaining):
            decode_queue.appendleft(request)
        return batch

    def _run_decode_batch(self, batch: List[RequestState]) -> None:
        eligible, blocked = self._prepare_paged_batch(batch, defer_on_capacity=True)
        if not eligible and blocked:
            self._finish_request(
                blocked[0],
                reason="capacity_exceeded",
                error_message="KV cache capacity exceeded",
            )
            return
        if not eligible:
            return

        current_input = torch.cat([request.current_input for request in eligible], dim=0)
        block_tables = self._build_block_tables(eligible)
        kv_lens = torch.tensor(
            [request.live_kv_tokens for request in eligible],
            dtype=torch.long,
            device=self.runtime.device,
        )

        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.runtime.model(
                current_input,
                block_tables=block_tables,
                kv_lens=kv_lens,
                paged_kv_caches=self.paged_kv_caches,
            )
            next_tokens = self._sample_next_tokens(logits, eligible)
        elapsed = time.perf_counter() - started
        per_request_elapsed = elapsed / len(eligible)

        for batch_idx, request in enumerate(eligible):
            request.live_kv_tokens += int(request.current_input.shape[1])
            request.num_computed_tokens = request.live_kv_tokens
            request.decode_time_s += per_request_elapsed
            request.decode_steps += 1
            self._record_next_token(request, next_tokens[batch_idx : batch_idx + 1], emit_text=False)

    def generate_many(
        self,
        prompts: List[str],
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[GenerationResult]:
        if not prompts:
            return []

        if self.config.num_prefill_workers != 1 or self.config.num_decode_workers != 1:
            raise ValueError("Single-device mode supports only num_prefill_workers=1 and num_decode_workers=1")

        prefill_queue: Deque[RequestState] = deque()
        decode_queue: Deque[RequestState] = deque()
        results: Dict[int, GenerationResult] = {}
        ordered_ids: List[int] = []

        for request_id, prompt in enumerate(prompts):
            request = self._build_request(
                request_id=request_id,
                prompt=prompt,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                created_at_s=time.perf_counter(),
            )
            ordered_ids.append(request_id)
            prefill_queue.append(request)

        while prefill_queue or decode_queue:
            if prefill_queue and len(decode_queue) < max(1, int(self.config.max_decode_batch_size)):
                request = prefill_queue.popleft()
                try:
                    if request.max_new_tokens <= 0:
                        self._finish_request(request, reason="max_new_tokens")
                    elif len(request.prompt_token_ids) >= self._context_limit():
                        self._finish_request(request, reason="context_limit")
                    else:
                        if request.first_scheduled_at_s is None:
                            if not self.capacity.can_allocate_for(request, len(request.prompt_token_ids)):
                                prefill_queue.append(request)
                                if decode_queue:
                                    continue
                                self._finish_request(
                                    request,
                                    reason="capacity_exceeded",
                                    error_message="KV cache capacity exceeded",
                                )
                                results[request.request_id] = self._make_result(request)
                                continue
                            request.first_scheduled_at_s = time.perf_counter()
                            self._init_request(request)
                        request.phase = "prefill"
                        final_prefill_logits = self._run_prefill_chunk(request)
                        if request.status not in ("finished", "error") and final_prefill_logits is not None:
                            next_token = self._sample_next_tokens(final_prefill_logits, [request])
                            self._record_next_token(request, next_token, emit_text=False)
                except Exception as exc:
                    self._finish_request(request, reason="error", error_message=str(exc))

                if request.status in ("finished", "error"):
                    results[request.request_id] = self._make_result(request)
                elif not self._prefill_complete(request):
                    prefill_queue.append(request)
                else:
                    decode_queue.append(request)
                continue

            if decode_queue:
                batch = self._select_decode_batch(decode_queue)
                try:
                    self._run_decode_batch(batch)
                except Exception as exc:
                    for request in batch:
                        if request.status in ("finished", "error"):
                            continue
                        self._finish_request(request, reason="error", error_message=str(exc))

                for request in batch:
                    if request.status in ("finished", "error"):
                        results[request.request_id] = self._make_result(request)
                    else:
                        decode_queue.append(request)

        return [results[request_id] for request_id in ordered_ids]

    def generate_stream_events(
        self,
        prompt: str,
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[StreamEvent]:
        request = self._build_request(
            request_id=0,
            prompt=prompt,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            created_at_s=time.perf_counter(),
        )
        request.first_scheduled_at_s = time.perf_counter()

        if request.max_new_tokens <= 0:
            self._finish_request(request, reason="max_new_tokens")
            yield StreamEvent(kind="done", stop_reason=request.stop_reason)
            return

        if len(request.prompt_token_ids) >= self._context_limit():
            self._finish_request(
                request,
                reason="context_limit",
                error_message="context length limit exceeded",
            )
            yield StreamEvent(
                kind="done",
                stop_reason=request.stop_reason,
                error_message=request.error_message,
            )
            return

        if not self.capacity.can_allocate_for(request, len(request.prompt_token_ids)):
            self._finish_request(
                request,
                reason="capacity_exceeded",
                error_message="KV cache capacity exceeded",
            )
            yield StreamEvent(
                kind="done",
                stop_reason=request.stop_reason,
                error_message=request.error_message,
            )
            return

        self._init_request(request)

        try:
            final_prefill_logits = None
            while request.status not in ("finished", "error") and not self._prefill_complete(request):
                final_prefill_logits = self._run_prefill_chunk(request)

            if request.status not in ("finished", "error") and final_prefill_logits is not None:
                next_token = self._sample_next_tokens(final_prefill_logits, [request])
                self._record_next_token(request, next_token, emit_text=True)
                if request.text_chunks:
                    yield StreamEvent(
                        kind="text",
                        text=request.text_chunks[-1],
                        token_id=request.generated_ids[-1],
                        generated_token_count=len(request.generated_ids),
                    )

            while request.status not in ("finished", "error"):
                previous_chunks = len(request.text_chunks)
                self._run_one_step(request, phase="decode", emit_text=True)
                if len(request.text_chunks) > previous_chunks:
                    yield StreamEvent(
                        kind="text",
                        text=request.text_chunks[-1],
                        token_id=request.generated_ids[-1],
                        generated_token_count=len(request.generated_ids),
                    )
        except Exception as exc:
            self._finish_request(request, reason="error", error_message=str(exc))

        yield StreamEvent(
            kind="done",
            generated_token_count=len(request.generated_ids),
            stop_reason=request.stop_reason,
            error_message=request.error_message,
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        for event in self.generate_stream_events(
            prompt=prompt,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
        ):
            if event.kind == "text" and event.text:
                yield event.text

    def stream(self, request: GenerateRequest) -> Iterator[StreamEvent]:
        return self.generate_stream_events(
            prompt=request.prompt,
            sampling=request.sampling,
            max_new_tokens=request.max_new_tokens,
        )

    def generate(
        self,
        prompt: Union[str, GenerateRequest],
        *,
        sampling: Optional[SamplingConfig] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Union[str, GenerationResult]:
        if isinstance(prompt, GenerateRequest):
            return self.generate_many(
                [prompt.prompt],
                sampling=prompt.sampling,
                max_new_tokens=prompt.max_new_tokens,
            )[0]

        result = self.generate_many(
            [prompt],
            sampling=sampling,
            max_new_tokens=max_new_tokens,
        )[0]
        if result.error_message is not None:
            raise RuntimeError(result.error_message)
        return result.text

    def abort(self, request_id: str) -> None:
        # The current single-process scheduler is synchronous, so there is no
        # persistent in-flight registry to cancel yet. The method is part of the
        # stable engine contract for the queued worker architecture.
        return None

    def stats(self) -> EngineStats:
        return EngineStats(
            kv_blocks_total=self.capacity.num_blocks,
            kv_blocks_used=getattr(self.capacity, "allocated_blocks", 0),
        )

    def shutdown(self) -> None:
        return None
