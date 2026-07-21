import unittest

from inference import GenerateRequest, GenerateResult, InferenceEngine, StreamEvent


class FakeInferenceEngine:
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        return GenerateResult(request_id=request.request_id, text=f"echo:{request.prompt}", token_ids=[1], stop_reason="max_new_tokens")
    async def stream(self, request: GenerateRequest):
        yield StreamEvent(kind="text", text=request.prompt, token_id=1, generated_token_count=1)
        yield StreamEvent(kind="done", generated_token_count=1, stop_reason="max_new_tokens")
    async def abort(self, request_id: str) -> None: return None
    def stats(self): return {}
    async def shutdown(self) -> None: return None


class InferenceEngineContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_engine_matches_async_protocol(self):
        engine = FakeInferenceEngine()
        self.assertIsInstance(engine, InferenceEngine)
        self.assertEqual((await engine.generate(GenerateRequest("req", "hello"))).text, "echo:hello")
        events = [event async for event in engine.stream(GenerateRequest("req", "hi"))]
        self.assertEqual(events[-1].kind, "done")
