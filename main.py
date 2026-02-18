from dataclasses import dataclass

import torch

from engine import EngineConfig, GemmaRuntime, LLMEngine, SamplingConfig, get_device


@dataclass(frozen=True)
class RunConfig:
    prompt: str = "Give me a short introduction to large language models."
    choose_model: str = "270m"
    use_instruct_model: bool = True
    max_new_tokens: int = 180
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


def calc_gpu_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024 / 1024:.2f} GB"


def main() -> None:
    run = RunConfig()
    device = get_device()

    runtime = GemmaRuntime(
        choose_model=run.choose_model,
        use_instruct_model=run.use_instruct_model,
        device=device,
    )
    engine = LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            choose_model=run.choose_model,
            use_instruct_model=run.use_instruct_model,
            max_new_tokens=run.max_new_tokens,
            sampling=SamplingConfig(
                temperature=run.temperature,
                top_p=run.top_p,
                top_k=run.top_k,
                repetition_penalty=run.repetition_penalty,
            ),
        ),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output = engine.generate(prompt=run.prompt)
    print(output)

    if torch.cuda.is_available():
        print(f"\nGPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()
