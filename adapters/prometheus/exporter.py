def metrics_to_prometheus_text(snapshot: dict) -> str:
    model = snapshot["model"]
    return "\n".join(
        [
            "# HELP gemma_requests_total Total chat completion requests.",
            "# TYPE gemma_requests_total counter",
            f'gemma_requests_total{{model="{model}"}} {snapshot["requests_total"]}',
            f'gemma_requests_failed_total{{model="{model}"}} {snapshot["requests_failed"]}',
            f'gemma_prompt_tokens_total{{model="{model}"}} {snapshot["tokens_prompt_total"]}',
            f'gemma_completion_tokens_total{{model="{model}"}} {snapshot["tokens_completion_total"]}',
            f'gemma_generation_last_latency_seconds{{model="{model}"}} {snapshot["latency_s"]["last"]}',
            "",
        ]
    )
