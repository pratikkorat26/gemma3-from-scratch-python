"""Async request lifecycle scheduler for one paged model executor."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Deque, Dict, List, Optional

from inference.executor import PagedModelExecutor
from inference.types import EngineStats, GenerateRequest, GenerateResult, RequestState, StopReason, StreamEvent

from .admission import FifoAdmissionPolicy
from .batch import DecodeBatchSelector


@dataclass
class ScheduledRequest:
    request: RequestState
    stream: bool
    events: "asyncio.Queue[StreamEvent]" = field(default_factory=asyncio.Queue)
    result: "asyncio.Future[GenerateResult]" = field(default=None)  # type: ignore[assignment]
    cancel_requested: bool = False

    @property
    def request_id(self) -> str:
        return str(self.request.request_id)


class AsyncScheduler:
    def __init__(self, executor: PagedModelExecutor, config) -> None:
        self.executor = executor
        self.config = config
        self.admission = FifoAdmissionPolicy(max_concurrent_requests=config.max_concurrent_requests)
        self.selector = DecodeBatchSelector(max_batch_size=max(1, int(config.decode_batch_size)), selection_window=config.decode_selection_window)
        self.waiting: Deque[ScheduledRequest] = deque()
        self.prefill: Deque[ScheduledRequest] = deque()
        self.decode: Deque[ScheduledRequest] = deque()
        self.active: Dict[str, ScheduledRequest] = {}
        self.known: Dict[str, ScheduledRequest] = {}
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self.completed = self.cancelled = self.failed = 0
        self.last_prefill_batch_size = self.last_decode_batch_size = 0
        self.last_queue_wait_ms = self.last_prefill_latency_ms = self.last_decode_step_latency_ms = 0.0

    async def submit(self, request: GenerateRequest, *, stream: bool) -> ScheduledRequest:
        if self._closed:
            raise RuntimeError("inference scheduler is shut down")
        request_id = str(request.request_id)
        if request_id in self.known:
            raise ValueError(f"duplicate request_id: {request_id}")
        now = time.perf_counter()
        state = self.executor.build_request(request_id, request.prompt, request.sampling, request.max_new_tokens, now)
        timeout = request.timeout_s if request.timeout_s is not None else self.config.request_timeout_s
        state.deadline_s = None if timeout is None else now + float(timeout)
        online = ScheduledRequest(request=state, stream=stream, result=asyncio.get_running_loop().create_future())
        if len(self.known) >= max(1, int(self.config.max_queue_size)):
            self._reject(online, StopReason.QUEUE_FULL.value, "scheduler queue is full")
            return online
        self.known[request_id] = online
        self.waiting.append(online)
        self._ensure_task()
        self._wake.set()
        return online

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        online = await self.submit(request, stream=False)
        return await online.result

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        online = await self.submit(request, stream=True)
        try:
            while True:
                event = await online.events.get()
                yield event
                if event.kind == "done":
                    return
        finally:
            if not online.result.done():
                await self.abort(online.request_id)

    async def abort(self, request_id: str) -> None:
        online = self.known.get(str(request_id))
        if online is not None and not online.result.done():
            online.cancel_requested = True
            self._wake.set()

    async def shutdown(self) -> None:
        self._closed = True
        for online in list(self.known.values()):
            if not online.result.done():
                self._complete(online, StopReason.CANCELLED.value, "scheduler shut down")
        self.waiting.clear()
        self.prefill.clear()
        self.decode.clear()
        self.active.clear()
        self._wake.set()
        if self._task is not None:
            await self._task

    def stats(self) -> EngineStats:
        return EngineStats(
            active_requests=len(self.active), queued_requests=len(self.waiting),
            kv_blocks_total=self.executor.capacity.num_blocks,
            kv_blocks_used=self.executor.capacity.allocated_blocks,
            prefill_batch_size=self.last_prefill_batch_size, decode_batch_size=self.last_decode_batch_size,
            completed_requests=self.completed, cancelled_requests=self.cancelled, failed_requests=self.failed,
            queue_wait_ms=self.last_queue_wait_ms, prefill_latency_ms=self.last_prefill_latency_ms,
            decode_step_latency_ms=self.last_decode_step_latency_ms,
            prefix_cache_enabled=self.executor.capacity.prefix_cache_enabled,
            prefix_cache_entries=self.executor.capacity.prefix_cache_size,
            prefix_cache_hits=self.executor.prefix_cache_hits, prefix_cache_misses=self.executor.prefix_cache_misses,
        )

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="llm-inference-scheduler")

    async def _run(self) -> None:
        while not self._closed or self._has_work():
            if not self._has_work():
                self._wake.clear()
                await self._wake.wait()
                continue
            progressed = self._admit()
            progressed = self._expire_or_cancel() or progressed
            if self.prefill:
                progressed = self._run_prefill() or progressed
            elif self.decode:
                progressed = self._run_decode() or progressed
            if not progressed:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0)

    def _has_work(self) -> bool:
        return bool(self.waiting or self.prefill or self.decode or self.active)

    def _admit(self) -> bool:
        changed = False
        while self.waiting:
            online = self.waiting[0]
            request = online.request
            if online.cancel_requested:
                self.waiting.popleft(); self._complete(online, StopReason.CANCELLED.value); changed = True; continue
            if self._expired(request):
                self.waiting.popleft(); self._complete(online, StopReason.TIMEOUT.value, "request timed out"); changed = True; continue
            if not self.admission.decide(active_requests=len(self.active), request=request).admitted:
                break
            self.waiting.popleft()
            if request.max_new_tokens <= 0:
                self._complete(online, StopReason.MAX_NEW_TOKENS.value); changed = True; continue
            if len(request.prompt_token_ids) >= self.executor.context_limit():
                self._complete(online, StopReason.CONTEXT_LIMIT.value, "context length limit exceeded"); changed = True; continue
            if not self.executor.can_admit(request):
                if self.active or self.prefill or self.decode:
                    self.waiting.appendleft(online)
                    break
                self._complete(online, StopReason.CAPACITY_EXCEEDED.value, "KV cache capacity exceeded"); changed = True; continue
            request.first_scheduled_at_s = time.perf_counter()
            self.last_queue_wait_ms = (request.first_scheduled_at_s - request.created_at_s) * 1000
            self.executor.initialize(request)
            self.active[online.request_id] = online
            self.prefill.append(online)
            changed = True
        return changed

    def _expire_or_cancel(self) -> bool:
        changed = False
        for online in list(self.active.values()):
            if online.cancel_requested:
                self._complete(online, StopReason.CANCELLED.value); changed = True
            elif self._expired(online.request):
                self._complete(online, StopReason.TIMEOUT.value, "request timed out"); changed = True
        return changed

    def _run_prefill(self) -> bool:
        online = self.prefill.popleft()
        if online.result.done():
            return True
        before = len(online.request.generated_ids)
        started = time.perf_counter()
        try:
            completed_prefill = self.executor.prefill(online.request)
        except Exception as exc:
            self._complete(online, StopReason.ERROR.value, str(exc)); return True
        self.last_prefill_batch_size, self.last_prefill_latency_ms = 1, (time.perf_counter() - started) * 1000
        self._emit_token(online, before)
        if online.request.status in ("finished", "error"):
            self._complete(online)
        elif completed_prefill:
            self.decode.append(online)
        else:
            self.prefill.append(online)
        return True

    def _run_decode(self) -> bool:
        candidates: Deque[RequestState] = deque()
        by_id: Dict[str, ScheduledRequest] = {}
        while self.decode:
            online = self.decode.popleft()
            if online.result.done():
                continue
            if online.cancel_requested:
                self._complete(online, StopReason.CANCELLED.value); continue
            candidates.append(online.request); by_id[online.request_id] = online
        batch_states = self.selector.select(candidates)
        selected: List[ScheduledRequest] = []
        deferred: List[ScheduledRequest] = []
        batch_tokens = 0
        max_batch_tokens = max(1, int(self.config.max_batch_tokens))
        for state in batch_states:
            online = by_id[str(state.request_id)]
            token_count = int(state.current_input.shape[1]) if state.current_input is not None else 0
            if selected and batch_tokens + token_count > max_batch_tokens:
                deferred.append(online)
                continue
            selected.append(online)
            batch_tokens += token_count
        self.decode.extend(deferred)
        for state in candidates:
            self.decode.append(by_id[str(state.request_id)])
        if not selected:
            return bool(candidates)
        before = {online.request_id: len(online.request.generated_ids) for online in selected}
        started = time.perf_counter()
        try:
            eligible = self.executor.decode([online.request for online in selected])
        except Exception as exc:
            for online in selected:
                self._complete(online, StopReason.ERROR.value, str(exc))
            return True
        self.last_decode_batch_size = len(eligible)
        self.last_decode_step_latency_ms = (time.perf_counter() - started) * 1000
        eligible_ids = {str(request.request_id) for request in eligible}
        if not eligible:
            blocked = selected.pop(0)
            self._complete(blocked, StopReason.CAPACITY_EXCEEDED.value, "KV cache capacity exceeded")
        for online in selected:
            self._emit_token(online, before[online.request_id])
            if online.request.status in ("finished", "error"):
                self._complete(online)
            elif online.request_id in eligible_ids:
                self.decode.append(online)
            else:
                self.decode.append(online)
        return True

    def _emit_token(self, online: ScheduledRequest, previous_count: int) -> None:
        request = online.request
        if online.stream and len(request.generated_ids) > previous_count:
            text = self.executor.token_text(request)
            if text:
                online.events.put_nowait(StreamEvent(kind="text", text=text, token_id=request.generated_ids[-1], generated_token_count=len(request.generated_ids)))

    def _complete(self, online: ScheduledRequest, reason: Optional[str] = None, error: Optional[str] = None) -> None:
        if online.result.done():
            return
        request = online.request
        if reason is not None and request.status not in ("finished", "error"):
            if reason == StopReason.CANCELLED.value:
                request.status, request.stop_reason, request.error_message, request.finished_at_s = "cancelled", reason, error, time.perf_counter()
                self.executor.capacity.release(request)
            else:
                self.executor.finish(request, reason, error)
        self.active.pop(online.request_id, None)
        self.known.pop(online.request_id, None)
        result = self.executor.result(request)
        if result.stop_reason == StopReason.CANCELLED.value:
            self.cancelled += 1
        elif request.status == "error" or result.error_message is not None:
            self.failed += 1
        else:
            self.completed += 1
        online.events.put_nowait(StreamEvent(kind="done", generated_token_count=len(request.generated_ids), stop_reason=result.stop_reason, error_message=result.error_message))
        online.result.set_result(result)

    @staticmethod
    def _expired(request: RequestState) -> bool:
        return request.deadline_s is not None and time.perf_counter() >= request.deadline_s

    def _reject(self, online: ScheduledRequest, reason: str, error: str) -> None:
        self.executor.finish(online.request, reason, error)
        online.events.put_nowait(StreamEvent(kind="done", stop_reason=reason, error_message=error))
        online.result.set_result(self.executor.result(online.request))
        self.failed += 1
