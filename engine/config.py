from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


@dataclass(frozen=True)
class EngineConfig:
    choose_model: str = "270m"
    use_instruct_model: bool = True
    max_new_tokens: int = 180
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    num_prefill_workers: int = 1
    num_decode_workers: int = 1
    max_decode_batch_size: int = 4
