"""Synchronous, single-device paged-model execution primitives.

This module deliberately contains no queueing or asyncio code.  A scheduler is
the sole caller and therefore the sole owner of model and KV-cache mutation.
"""

import time
from typing import List, Optional

import torch

from gemma3.tokenizer import apply_chat_template

from .config import EngineConfig, SamplingConfig
from .kv import KVBlockManager
from .sampling import apply_repetition_penalty_, sample_next_token
from .types import GenerateResult, RequestState


class PagedModelExecutor:
    def __init__(self, runtime: object, config: EngineConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.capacity = KVBlockManager(
            max_kv_cache_tokens=config.max_kv_cache_tokens,
            block_size=config.kv_block_size,
            num_blocks=config.num_kv_blocks,
            enable_prefix_cache=config.enable_prefix_cache,
            max_prefix_cache_entries=config.max_prefix_cache_entries,
        )
        self.paged_kv_caches = runtime.model.init_paged_kv_caches(
            num_blocks=self.capacity.num_blocks,
            block_size=self.capacity.block_size,
            device=runtime.device,
        )
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0

    def build_request(self, request_id: str, prompt: str, sampling: Optional[SamplingConfig], max_new_tokens: Optional[int], created_at_s: float) -> RequestState:
        if self.config.use_instruct_model:
            prompt = apply_chat_template(prompt)
        return RequestState.from_prompt(
            request_id=request_id,
            prompt_token_ids=self.runtime.tokenizer.encode(prompt),
            sampling=sampling or self.config.sampling,
            max_new_tokens=self.config.max_new_tokens if max_new_tokens is None else max_new_tokens,
            eos_token_id=self.runtime.tokenizer.eos_token_id,
            created_at_s=created_at_s,
        )

    def context_limit(self) -> int:
        return int(self.runtime.model.context_length)

    def can_admit(self, request: RequestState) -> bool:
        return self.capacity.can_allocate_for(request, len(request.prompt_token_ids))

    def initialize(self, request: RequestState) -> None:
        request.status = "active"
        request.generated_ids = []
        request.all_token_ids = list(request.prompt_token_ids)
        request.seen_token_ids = set(request.prompt_token_ids)
        request.block_table = []
        cached_tokens = self.capacity.init_request_from_cache(request, scope="default")
        if self.capacity.prefix_cache_enabled:
            if cached_tokens:
                self.prefix_cache_hits += 1
            else:
                self.prefix_cache_misses += 1
        if cached_tokens == len(request.prompt_token_ids) and cached_tokens:
            cached_tokens -= 1
        request.prompt_cursor = request.live_kv_tokens = cached_tokens
        request.stop_reason = request.error_message = None
        request.sampling_generator = self._sampling_generator(getattr(request.sampling, "seed", None))
        request.prefill_time_s = request.decode_time_s = 0.0
        request.prefill_steps = request.decode_steps = 0
        request.current_input = self._next_prefill_chunk(request)

    def finish(self, request: RequestState, reason: str, error_message: Optional[str] = None) -> None:
        request.status = "error" if reason == "error" else "finished"
        request.stop_reason, request.error_message = reason, error_message
        request.finished_at_s = time.perf_counter()
        self.capacity.release(request)

    def result(self, request: RequestState) -> GenerateResult:
        first = request.first_scheduled_at_s or request.created_at_s
        finished = request.finished_at_s or time.perf_counter()
        text_ids = list(request.generated_ids)
        if request.stop_reason == "eos" and text_ids and text_ids[-1] == request.eos_token_id:
            text_ids.pop()
        model_s = request.prefill_time_s + request.decode_time_s
        return GenerateResult(
            request_id=request.request_id,
            text=self.runtime.tokenizer.decode(text_ids) if text_ids else "",
            token_ids=list(request.generated_ids),
            stop_reason=request.stop_reason or "error",
            error_message=request.error_message,
            queue_wait_s=max(0.0, first - request.created_at_s),
            prefill_s=request.prefill_time_s,
            decode_s=request.decode_time_s,
            total_latency_s=max(0.0, finished - request.created_at_s),
            model_tokens_per_s=0.0 if model_s <= 0 else len(request.generated_ids) / model_s,
            prefill_steps=request.prefill_steps,
            decode_steps=request.decode_steps,
        )

    def prefill(self, request: RequestState) -> bool:
        """Run one prefill chunk; return true once its first token is sampled."""
        if request.current_input is None:
            request.current_input = self._next_prefill_chunk(request)
        if request.current_input is None:
            return False
        started = time.perf_counter()
        logits = self._forward([request], defer_on_capacity=False)
        if request.status in ("finished", "error"):
            return False
        request.prefill_time_s += time.perf_counter() - started
        request.prefill_steps += 1
        request.prompt_cursor += int(request.current_input.shape[1])
        if request.prompt_cursor < len(request.prompt_token_ids):
            request.current_input = self._next_prefill_chunk(request)
            return False
        self.capacity.cache_prefix("default", request.prompt_token_ids, request.block_table)
        self._record_token(request, self._sample(logits, [request])[0:1])
        return True

    def decode(self, requests: List[RequestState]) -> List[RequestState]:
        eligible = self._eligible(requests, defer_on_capacity=True)
        if not eligible:
            return []
        started = time.perf_counter()
        logits = self._forward(eligible, defer_on_capacity=True, already_eligible=True)
        tokens = self._sample(logits, eligible)
        elapsed = (time.perf_counter() - started) / len(eligible)
        for index, request in enumerate(eligible):
            request.decode_time_s += elapsed
            request.decode_steps += 1
            self._record_token(request, tokens[index:index + 1])
        return eligible

    def token_text(self, request: RequestState) -> str:
        if not request.generated_ids or request.generated_ids[-1] == request.eos_token_id:
            return ""
        return self.runtime.tokenizer.decode([request.generated_ids[-1]])

    def _sampling_generator(self, seed: Optional[int]):
        if seed is None:
            return None
        generator = torch.Generator(device=self.runtime.device)
        generator.manual_seed(int(seed))
        return generator

    def _next_prefill_chunk(self, request: RequestState):
        if request.prompt_cursor >= len(request.prompt_token_ids):
            return None
        size = self.config.prefill_chunk_size or len(request.prompt_token_ids) - request.prompt_cursor
        ids = request.prompt_token_ids[request.prompt_cursor:request.prompt_cursor + max(1, int(size))]
        return torch.tensor(ids, device=self.runtime.device).unsqueeze(0) if ids else None

    def _eligible(self, requests: List[RequestState], *, defer_on_capacity: bool) -> List[RequestState]:
        eligible = []
        for request in requests:
            if request.current_input is None:
                self.finish(request, "error", "request tensors are not initialized")
            elif len(request.all_token_ids) >= self.context_limit():
                self.finish(request, "context_limit")
            elif self.capacity.ensure_capacity(request, request.live_kv_tokens + int(request.current_input.shape[1])):
                eligible.append(request)
            elif not defer_on_capacity:
                self.finish(request, "capacity_exceeded", "KV cache capacity exceeded")
        return eligible

    def _forward(self, requests: List[RequestState], *, defer_on_capacity: bool, already_eligible: bool = False) -> torch.Tensor:
        eligible = requests if already_eligible else self._eligible(requests, defer_on_capacity=defer_on_capacity)
        if not eligible:
            return torch.empty(0, 1, 0, device=self.runtime.device)
        inputs = torch.cat([request.current_input for request in eligible], dim=0)
        width = max(len(request.block_table) for request in eligible)
        tables = torch.full((len(eligible), width), -1, dtype=torch.long, device=self.runtime.device)
        for row, request in enumerate(eligible):
            tables[row, :len(request.block_table)] = torch.tensor(request.block_table, dtype=torch.long, device=self.runtime.device)
        lengths = torch.tensor([request.live_kv_tokens for request in eligible], dtype=torch.long, device=self.runtime.device)
        with torch.inference_mode():
            logits = self.runtime.model(inputs, block_tables=tables, kv_lens=lengths, paged_kv_caches=self.paged_kv_caches)
        for request in eligible:
            request.live_kv_tokens += int(request.current_input.shape[1])
        return logits

    def _sample(self, logits: torch.Tensor, requests: List[RequestState]) -> torch.Tensor:
        next_logits = logits[:, -1, :].clone()
        next_logits = apply_repetition_penalty_(next_logits, [r.seen_token_ids for r in requests], penalty=requests[0].sampling.repetition_penalty)
        if all(r.sampling_generator is None for r in requests):
            return sample_next_token(next_logits, temperature=requests[0].sampling.temperature, top_p=requests[0].sampling.top_p, top_k=requests[0].sampling.top_k)
        return torch.cat([sample_next_token(next_logits[i:i + 1], temperature=r.sampling.temperature, top_p=r.sampling.top_p, top_k=r.sampling.top_k, generator=r.sampling_generator) for i, r in enumerate(requests)])

    def _record_token(self, request: RequestState, token: torch.Tensor) -> None:
        next_id = int(token.item())
        request.generated_ids.append(next_id)
        if request.eos_token_id is not None and next_id == request.eos_token_id:
            self.finish(request, "eos")
            return
        request.all_token_ids.append(next_id)
        request.seen_token_ids.add(next_id)
        request.current_input = token
        if len(request.generated_ids) >= request.max_new_tokens:
            self.finish(request, "max_new_tokens")
