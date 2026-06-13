from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

import torch

from .config import SamplingConfig, SamplingParams


class RequestStatus(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"
    ERROR = "error"


class RequestPhase(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    ERROR = "error"


class StopReason(str, Enum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONTEXT_LIMIT = "context_limit"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    TIMEOUT = "timeout"
    ERROR = "error"


class StreamEventKind(str, Enum):
    TEXT = "text"
    DONE = "done"


LayerCache = Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    prompt: str
    sampling: Optional[SamplingConfig] = None
    max_new_tokens: Optional[int] = None


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    context_length: int
    device: str


@dataclass(frozen=True)
class EngineStats:
    active_requests: int = 0
    queued_requests: int = 0
    kv_blocks_total: int = 0
    kv_blocks_used: int = 0


@dataclass
class RequestState:
    request_id: int
    prompt_token_ids: List[int]
    sampling: SamplingConfig
    max_new_tokens: int
    eos_token_id: Optional[int]
    generated_ids: List[int] = field(default_factory=list)
    all_token_ids: List[int] = field(default_factory=list)
    seen_token_ids: Set[int] = field(default_factory=set)
    past_kv: LayerCache = None
    current_input: Optional[torch.Tensor] = None
    text_chunks: List[str] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    prompt_cursor: int = 0
    num_computed_tokens: int = 0
    live_kv_tokens: int = 0
    status: str = RequestStatus.QUEUED.value
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None
    sampling_seed: Optional[int] = None
    sampling_generator: Optional[torch.Generator] = None
    created_at_s: float = 0.0
    first_scheduled_at_s: Optional[float] = None
    finished_at_s: Optional[float] = None
    prefill_time_s: float = 0.0
    prefill_steps: int = 0
    decode_time_s: float = 0.0
    decode_steps: int = 0
    phase: str = RequestPhase.QUEUED.value

    @classmethod
    def from_prompt(
        cls,
        request_id: int,
        prompt_token_ids: List[int],
        sampling: SamplingConfig,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        created_at_s: float,
    ) -> "RequestState":
        return cls(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            created_at_s=created_at_s,
        )

    def should_stop(self) -> bool:
        if len(self.generated_ids) >= self.max_new_tokens:
            return True
        if self.eos_token_id is None:
            return False
        return bool(self.generated_ids and self.generated_ids[-1] == self.eos_token_id)


@dataclass
class GenerationResult:
    request_id: int
    text: str
    token_ids: List[int]
    stop_reason: str
    error_message: Optional[str] = None
    queue_wait_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    total_latency_s: float = 0.0
    model_tokens_per_s: float = 0.0
    prefill_steps: int = 0
    decode_steps: int = 0


GenerateResult = GenerationResult


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    text: str = ""
    token_id: Optional[int] = None
    generated_token_count: int = 0
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None


StreamGenerateEvent = StreamEvent


__all__ = [
    "EngineStats",
    "GenerateRequest",
    "GenerateResult",
    "GenerationResult",
    "LayerCache",
    "ModelInfo",
    "RequestPhase",
    "RequestState",
    "RequestStatus",
    "SamplingParams",
    "StopReason",
    "StreamEvent",
    "StreamEventKind",
    "StreamGenerateEvent",
]
