from dataclasses import dataclass

import torch

from engine import EngineConfig, GemmaRuntime, LLMEngine, SamplingConfig, get_device


@dataclass(frozen=True)
class RunConfig:
    choose_model: str = "270m"
    use_instruct_model: bool = True
    prompt: str = "Give me a short introduction to large language models."
    max_new_tokens: int = 180
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


def calc_gpu_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024 / 1024 / 1024:.2f} GB"


def main() -> None:
    run = RunConfig()
    sampling = SamplingConfig(
        temperature=run.temperature,
        top_p=run.top_p,
        top_k=run.top_k,
        repetition_penalty=run.repetition_penalty,
    )
    runtime = GemmaRuntime(
        choose_model=run.choose_model,
        use_instruct_model=run.use_instruct_model,
        device=get_device(),
    )
    engine = LLMEngine(
        runtime=runtime,
        config=EngineConfig(
            choose_model=run.choose_model,
            use_instruct_model=run.use_instruct_model,
            max_new_tokens=run.max_new_tokens,
            sampling=sampling,
        ),
    )

    input_token_ids = engine._encode_prompt(run.prompt)
    print(runtime.tokenizer.decode(input_token_ids))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for text in engine.generate_stream(
        run.prompt,
        sampling=sampling,
        max_new_tokens=run.max_new_tokens,
    ):
        print(text, end="", flush=True)

    if torch.cuda.is_available():
        print(f"\n\nGPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()
