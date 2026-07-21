from dataclasses import dataclass, field
from typing import List, Optional

from inference.config import SamplingConfig


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    messages: List[ChatMessage]
    model: str
    max_tokens: Optional[int] = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stop: Optional[List[str]] = None
    stream: bool = False
    include_usage: bool = False


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ChatCompletionResult:
    request_id: str
    model: str
    content: str
    finish_reason: str
    usage: Usage
    created: int


@dataclass(frozen=True)
class ChatCompletionEvent:
    request_id: str
    model: str
    created: int
    role: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    done: bool = False
    error: Optional[dict] = None
