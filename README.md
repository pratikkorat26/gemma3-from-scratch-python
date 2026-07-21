# Gemma 3 270M: Minimal vLLM-Style Inference

A readable, single-process PyTorch inference stack for **Gemma-3-270M / 270M-IT**, with paged KV cache, continuous decode batching, optional prefix caching, and an OpenAI-compatible FastAPI server.

Default branch: **`main`**.

## Purpose

- Decoder-only inference with **block-based paged KV**
- **Continuous multi-request scheduling** (decode-centric)
- **OpenAI-compatible** local chat API with readiness, stats, and metrics
- Small enough to read end-to-end; structured like a production serving path

## Quick start

```bash
python -m pip install -e ".[dev,benchmark]"

# CLI generation
python main.py --prompt "Say hello in one short sentence." --max-new-tokens 64
# or: gemma3-generate

# OpenAI-compatible server (default 127.0.0.1:8000)
python -m adapters.openai.run
# or: gemma3-serve

python query_fastapi.py --stream --prompt "Give me one short line about LLM inference."
```

Tune with CLI flags or `GEMMA_API_*` env vars:

```bash
python -m adapters.openai.run --host 0.0.0.0 --port 8080 --device cpu --default-max-tokens 64
GEMMA_API_MAX_KV_CACHE_TOKENS=65536 python -m adapters.openai.run
```

Model access: approve Gemma checkpoints on Hugging Face (`google/gemma-3-270m` / `google/gemma-3-270m-it`). Local folders `gemma-3-270m/` / `gemma-3-270m-it/` are reused when present.

## Architecture (summary)

```text
adapters/openai  →  app  →  inference (LLMEngine)  →  runtime  →  gemma3
     HTTP              service         scheduler+executor         load          tensors
```

| Layer | Path | Role |
|-------|------|------|
| HTTP | [`adapters/openai/`](adapters/openai/) | FastAPI routes, schemas, SSE — no `torch` |
| App | [`app/`](app/) | Prompting, validation, metrics, chat orchestration |
| Inference | [`inference/`](inference/) | `LLMEngine`, `AsyncScheduler`, `PagedModelExecutor`, KV, sampling |
| Runtime | [`runtime/`](runtime/) | Device/dtype policy, weight download, tokenizer |
| Model | [`gemma3/`](gemma3/) | 18-layer Gemma3, GQA attention, paged KV tensors, RoPE, MLP |

**Design rule:** inference is deep; HTTP is thin. Enforced by [`tests/architecture/`](tests/architecture/).

Full diagrams and lifecycle details: **[`docs/architecture.md`](docs/architecture.md)**.

### Request path

1. `POST /v1/chat/completions` → `ChatCompletionService`
2. Prompt build → `LLMEngine.generate` / `stream`
3. `AsyncScheduler`: admit → **prefill** (one request, optional chunks) → **decode** cohorts
4. `PagedModelExecutor`: grow KV blocks, forward, sample last-token logits
5. JSON or SSE response

### Performance notes

- **Default device:** CUDA if available, otherwise **CPU** (MPS is opt-in; currently slower for this paged path)
- **Dtype:** `bfloat16` on CUDA, **`float32` on CPU/MPS**
- **`torch.compile`:** CUDA only (`GEMMA_DISABLE_COMPILE=1` to skip)
- Last-token LM head only; intermediate prefill chunks skip `out_head`
- Microbench: `python scripts/benchmark_decode_throughput.py --max-new-tokens 64`

## Engine configuration

| Setting | Meaning |
|---------|---------|
| `max_kv_cache_tokens` / `kv_block_size` / `num_kv_blocks` | Paged KV budget |
| `enable_prefix_cache` / `max_prefix_cache_entries` | Opt-in prefix KV reuse (default off) |
| `max_queue_size` / `max_concurrent_requests` | Admission limits |
| `decode_batch_size` / `max_batch_tokens` | Decode cohort caps |
| `prefill_chunk_size` | Max tokens per prefill step (`None` = whole remainder) |
| `request_timeout_s` | Per-request timeout |

Continuous batching is **decode-centric**: prefill is one-at-a-time (optionally chunked); decode batches share sequence length + sampling params.

Public engine API:

- `await generate(request) -> GenerateResult`
- `stream(request) -> AsyncIterator[StreamEvent]`
- `await abort(request_id)` / `stats()` / `await shutdown()`

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Liveness |
| `GET /readyz` | Ready after service load |
| `GET /v1/models` | Model listing |
| `GET /stats` | JSON counters (+ prefix-cache fields) |
| `GET /metrics` | Prometheus text |
| `POST /v1/chat/completions` | Chat (stream or JSON) |

Chat controls: `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop`, `stream`, `stream_options.include_usage`.

## Project layout

```text
gemma3/           model + paged KV + RoPE + MLP
inference/        engine, executor, scheduler, kv/, sampling/
runtime/          GemmaRuntime, device/dtype
app/              ChatCompletionService
adapters/openai/  FastAPI OpenAI adapter
config/           ServerSettings / RuntimeSettings
scripts/          benches (decode, concurrent, chunked prefill)
tests/            architecture, contract, unit, opt-in real-engine
docs/architecture.md
main.py           CLI entry (gemma3-generate)
```

## Validation

```bash
python -m pytest -q
```

Coverage includes import boundaries, engine contract, continuous batching, prefix cache, OpenAI shapes, and runtime config.

Opt-in real model tests:

```bash
RUN_REAL_ENGINE_TESTS=1 python -m pytest -q tests/test_llmengine_regression_real.py
RUN_REAL_ENGINE_TESTS=1 RUN_REAL_ENGINE_LOAD_TESTS=1 python -m pytest -q tests/test_llmengine_load_real.py
```

## Scope / non-goals

Single-device, single-process prototype. Not a full vLLM clone: no batched prefill, no multi-GPU, no CUDA-graph-heavy serving, limited prefix-cache tenant isolation (`scope` hard-coded to `"default"`).
