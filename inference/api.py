from typing import Iterator, Optional, Protocol, runtime_checkable

import torch

from .types import EngineStats, GenerateRequest, GenerateResult, ModelInfo, StreamGenerateEvent


@runtime_checkable
class ModelBackend(Protocol):
    context_length: int

    def init_paged_kv_caches(
        self,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
    ):
        ...

    def __call__(self, input_ids, **kwargs):
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    def generate(self, request: GenerateRequest) -> GenerateResult:
        ...

    def stream(self, request: GenerateRequest) -> Iterator[StreamGenerateEvent]:
        ...

    def abort(self, request_id: str) -> None:
        ...

    def stats(self) -> EngineStats:
        ...

    def shutdown(self) -> None:
        ...


class MetricsSink(Protocol):
    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        ...

    def observe(self, name: str, value: float, **labels: str) -> None:
        ...

    def gauge(self, name: str, value: float, **labels: str) -> None:
        ...


class Clock(Protocol):
    def now(self) -> float:
        ...


class RuntimeProvider(Protocol):
    def load(self) -> ModelBackend:
        ...

    def ready(self) -> bool:
        ...

    def warmup(self) -> Optional[ModelInfo]:
        ...
