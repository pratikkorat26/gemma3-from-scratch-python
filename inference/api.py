from typing import AsyncIterator, Protocol, runtime_checkable

from .types import EngineStats, GenerateRequest, GenerateResult, StreamEvent


@runtime_checkable
class InferenceEngine(Protocol):
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        ...

    def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        ...

    async def abort(self, request_id: str) -> None:
        ...

    def stats(self) -> EngineStats:
        ...

    async def shutdown(self) -> None:
        ...
