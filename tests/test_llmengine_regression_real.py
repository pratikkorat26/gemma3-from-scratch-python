import asyncio
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from inference import GenerateRequest, SamplingConfig
from real_engine_test_utils import build_engine, real_engine_skip_reason


def sampling(): return SamplingConfig(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0)


async def generate_all(engine, prompts, max_new_tokens):
    return await asyncio.gather(*[engine.generate(GenerateRequest(str(index), prompt, sampling(), max_new_tokens)) for index, prompt in enumerate(prompts)])


@unittest.skipIf(bool(real_engine_skip_reason()), real_engine_skip_reason())
class LLMEngineRealRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()
        cls.engine_batch = build_engine(max_new_tokens=16, decode_batch_size=4)
        cls.engine_single = build_engine(max_new_tokens=16, decode_batch_size=1)
        cls.prompts = ["Give one short sentence about caching in LLM inference.", "Explain batching in one concise sentence.", "Write one line about prompt templating.", "Describe top-k sampling in plain words."]
    @classmethod
    def tearDownClass(cls):
        cls.loop.run_until_complete(cls.engine_batch.shutdown())
        cls.loop.run_until_complete(cls.engine_single.shutdown())
        cls.loop.close()
    def run_async(self, awaitable):
        return self.loop.run_until_complete(awaitable)
    def test_batch_and_single_request_consistency(self):
        batched = self.run_async(generate_all(self.engine_batch, self.prompts, 12))
        singles = [self.run_async(self.engine_batch.generate(GenerateRequest(f"single-{i}", prompt, sampling(), 12))) for i, prompt in enumerate(self.prompts)]
        for batch, single in zip(batched, singles): self.assertEqual(batch.token_ids, single.token_ids)
    def test_decode_batch_size_invariance(self):
        for left, right in zip(self.run_async(generate_all(self.engine_batch, self.prompts, 10)), self.run_async(generate_all(self.engine_single, self.prompts, 10))): self.assertEqual(left.token_ids, right.token_ids)
    def test_stream_matches_non_stream(self):
        prompt = "Share one practical testing tip for inference engines."
        result = self.run_async(self.engine_batch.generate(GenerateRequest("plain", prompt, sampling(), 12)))
        async def collect():
            return [event async for event in self.engine_batch.stream(GenerateRequest("stream", prompt, sampling(), 12))]
        events = self.run_async(collect())
        self.assertEqual(result.text, "".join(event.text for event in events if event.kind == "text"))
    def test_zero_max_new_tokens(self):
        result = self.run_async(self.engine_batch.generate(GenerateRequest("zero", "Return nothing.", sampling(), 0)))
        self.assertEqual(result.stop_reason, "max_new_tokens")
