from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupResult:
    model_id: str
    context_length: int
    device: str
