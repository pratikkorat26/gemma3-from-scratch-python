import unittest
from typing import Iterator

from inference import GenerateRequest, GenerationResult, InferenceEngine, StreamEvent


class FakeInferenceEngine:
    def generate(self, request: GenerateRequest) -> GenerationResult:
        return GenerationResult(
            request_id=0,
            text=f"echo:{request.prompt}",
            token_ids=[1],
            stop_reason="max_new_tokens",
        )

    def stream(self, request: GenerateRequest) -> Iterator[StreamEvent]:
        yield StreamEvent(kind="text", text=request.prompt, token_id=1, generated_token_count=1)
        yield StreamEvent(kind="done", generated_token_count=1, stop_reason="max_new_tokens")

    def abort(self, request_id: str) -> None:
        return None

    def stats(self):
        return {}

    def shutdown(self) -> None:
        return None


class InferenceEngineContractTests(unittest.TestCase):
    def test_fake_engine_matches_runtime_protocol(self):
        engine = FakeInferenceEngine()
        self.assertIsInstance(engine, InferenceEngine)
        result = engine.generate(GenerateRequest(request_id="req-1", prompt="hello"))
        self.assertEqual(result.text, "echo:hello")
        events = list(engine.stream(GenerateRequest(request_id="req-1", prompt="hi")))
        self.assertEqual(events[-1].kind, "done")


if __name__ == "__main__":
    unittest.main()
