# Gemma 3 270M: Readable Minimal Implementation

This repository is a clean, minimal PyTorch implementation of Gemma-3-style inference, built to be easy to read and extend.

## Purpose

Provide a small, understandable codebase for:
- decoder-only model inference with block-based paged KV cache
- continuous multi-request scheduling with prefill/decode batching
- an OpenAI-compatible local chat API with readiness-aware startup

## Technical Scope

- Model: `gemma-3-270m` and `gemma-3-270m-it` checkpoints
- Current runtime target: `choose_model="270m"`
- Architecture: 18-layer decoder with mixed `sliding_attention` + `full_attention`
- Attention stack: GQA, RoPE (local/global bases), causal + optional sliding-window mask
- Normalization/MLP: Gemma-style RMSNorm and gated feedforward (`down(gelu(gate(x)) * up(x))`)
- Engine: single-device prefill/decode scheduler with round-robin decode, decode-step batching, and block-based paged KV allocation
- API: FastAPI app compatible with `POST /v1/chat/completions` plus `GET /healthz` and `GET /readyz`

## Install

```bash
python -m pip install torch tokenizers safetensors huggingface_hub fastapi uvicorn pydantic starlette
```

Or install the local package metadata:

```bash
python -m pip install -e ".[dev,benchmark]"
```

Model access note:
- You need approved access to Gemma checkpoints (for example `google/gemma-3-270m` / `google/gemma-3-270m-it`) before first download.
- If local model files already exist in `gemma-3-270m/` or `gemma-3-270m-it/`, they are reused.

Optional benchmark dependency:

```bash
python -m pip install pandas
```

## Run

Direct generation:

```bash
python main.py
# or: gemma3-generate
```

OpenAI-like API server:

```bash
python -m adapters.openai.run
# or: gemma3-serve
```

The API binds to `127.0.0.1:8000` by default. Use CLI flags or `GEMMA_API_*` environment variables to tune it:

```bash
python -m adapters.openai.run --host 0.0.0.0 --port 8080 --device cpu --default-max-tokens 64
GEMMA_API_MAX_KV_CACHE_TOKENS=65536 python -m adapters.openai.run
```

Query the API:

```bash
python query_fastapi.py --stream --prompt "Give me one short line about LLM inference."
```

## Project Structure

- `gemma3/`: model components, paged KV storage, RoPE, feedforward, tokenizer template, and weights mapping
- `inference/`: async engine facade, paged model executor, request scheduler, KV allocation, sampling, and batching policies
- `runtime/`: model/tokenizer loading, device resolution, and runtime provider
- `app/`: serving/application layer: request validation, metrics, logging, and chat completion orchestration
- `adapters/openai/`: FastAPI/OpenAI compatibility adapter: schemas, routes, mapping, SSE, and HTTP errors
- `config/`: typed runtime/server settings and environment/CLI parsing
- `main.py`: direct local generation flow
- `scripts/`: chunked-prefill demos and concurrency benchmarks
- `tests/`: architecture boundaries, engine contracts, continuous batching, prefix cache, OpenAI API shapes, and opt-in real-engine suites

## Architecture Boundaries

The core design rule is: the inference core is deep, external interfaces are thin.

- `gemma3/` contains tensor/model code only and does not import serving, runtime, FastAPI, or OpenAI API modules.
- `inference/` owns generation lifecycle, scheduling, KV cache management, sampling, stop reasons, and engine contracts. It does not import FastAPI, OpenAI schemas, app services, or runtime loaders.
- `runtime/` builds ready model/tokenizer/backend objects from local files or Hugging Face artifacts. It does not import HTTP or app code.
- `app/` coordinates service behavior around the engine without depending on FastAPI or OpenAI schemas.
- `adapters/openai/` is the replaceable HTTP compatibility layer and must not import `torch` or construct models directly.

Architecture regression tests in `tests/architecture/` enforce these dependency rules.

## Engine Configuration

The scheduler now allocates KV memory in blocks as requests grow, rather than reserving an entire request budget up front.

- `max_kv_cache_tokens`: total KV token budget available to the engine
- `kv_block_size`: size of each KV allocation block
- `num_kv_blocks`: optional explicit number of blocks; if omitted, it is derived from `max_kv_cache_tokens // kv_block_size`
- `enable_prefix_cache`: opt-in prompt-prefix KV cache (default `false`)
- `max_prefix_cache_entries`: maximum number of cached prefixes when prefix caching is enabled (default `64`)
- `max_queue_size`: maximum in-flight requests owned by the online scheduler
- `max_concurrent_requests`: maximum active requests admitted for model execution
- `decode_batch_size`: maximum online decode batch size
- `max_batch_tokens`: maximum token count in a decode batch
- `prefill_chunk_size`: optional max tokens per prefill step (`None` = whole remaining prompt)
- `request_timeout_s`: optional per-request timeout default

This keeps the engine readable while matching the basic vLLM-style idea: admit requests cheaply, grow cache usage incrementally, and free blocks immediately when a request finishes. Continuous batching is decode-centric: prefill runs one request at a time (optionally chunked), then decode cohorts share a step.

When `enable_prefix_cache` is `true`, completed prompt prefixes are stored in a reference-counted cache. Subsequent requests with a matching prefix reuse the cached KV blocks and skip the corresponding prefill work. Reuse is performed at whole-block granularity, and cached blocks remain allocated until the entry is evicted by LRU or the process exits. The cache is keyed by a `scope` string that is currently hard-coded to `"default"`; this is the seam for future tenant isolation.

The public engine API is asynchronous and request-oriented:

- `await generate(request) -> GenerateResult`
- `stream(request) -> AsyncIterator[StreamEvent]`
- `await abort(request_id) -> None`
- `stats() -> EngineStats`
- `await shutdown() -> None`

`LLMEngine` is a thin composition facade. `AsyncScheduler` owns admission, queues, cancellation, deadlines, streaming, and metrics; `PagedModelExecutor` exclusively owns synchronous model, sampling, and KV-cache mutation. Both run on the ASGI event loop, so model steps remain serialized per process.

## API Operations

- `GET /healthz`: lightweight liveness probe
- `GET /readyz`: readiness probe; returns `200` only after `ChatCompletionService` has loaded successfully
- `GET /v1/models`: OpenAI-style local model listing
- `GET /stats`: JSON request/token/latency counters, plus prefix-cache state (`prefix_cache_enabled`, `prefix_cache_entries`, `prefix_cache_hits`, `prefix_cache_misses`)
- `GET /metrics`: dependency-free Prometheus-style text metrics, including `gemma_prefix_cache_*` counters when prefix caching is enabled
- `POST /v1/chat/completions`: returns `503` if service startup failed or is not yet complete

The API initializes `ChatCompletionService` during startup, awaits the shared async engine, shuts it down during application teardown, enforces request/message size limits, and returns generic client-facing errors while logging details server-side.

Supported chat request controls include `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop`, `stream`, and `stream_options.include_usage`.

## Request Flow

```mermaid
flowchart TD
    A[Client] --> B[FastAPI app: adapters/openai/routes.py]
    B --> C[startup: build ChatCompletionService]
    B --> D[GET /healthz]
    B --> E[GET /readyz]
    B --> F{POST /v1/chat/completions}
    F -->|stream=false| G[create_chat_completion]
    F -->|stream=true| H[stream_chat_completion]
    G --> I[messages_to_gemma_prompt]
    H --> I
    I --> J[LLMEngine.generate / stream]
    J --> K[AsyncScheduler admission]
    K --> L[PagedModelExecutor prefill + prefix cache]
    L --> M[model forward with shared paged KV caches]
    M --> N[sample_next_token + repetition penalty]
    N --> O{stop? eos / max_new_tokens / context_limit / capacity}
    O -->|no| P[decode: select cohort, grow block tables, run batched decode]
    P --> M
    O -->|yes| Q[release KV blocks and build result]
    Q -->|non-stream| R[JSON chat.completion]
    Q -->|stream| S["SSE chat.completion.chunk + [DONE]"]
```

## Validation

```bash
python -m pytest -q
# or: python -m unittest discover -s tests -q
```

Tests cover import boundaries, the `InferenceEngine` contract, continuous batching / decode cohort selection, paged KV isolation and capacity reuse, prefix-cache unit + integration behavior, OpenAI response shapes, and runtime config parsing.

Real `LLMEngine` regression tests (uses actual model/runtime, opt-in):

```bash
RUN_REAL_ENGINE_TESTS=1 python -m unittest -q tests/test_llmengine_regression_real.py
```

Real `LLMEngine` load tests (opt-in and heavier):

```bash
RUN_REAL_ENGINE_TESTS=1 RUN_REAL_ENGINE_LOAD_TESTS=1 python -m unittest -q tests/test_llmengine_load_real.py
```

Load threshold overrides (optional):
- `LOAD_MAX_ERROR_RATE`
- `LOAD_MAX_P95_TOTAL_S`
- `LOAD_MIN_THROUGHPUT_TPS`
