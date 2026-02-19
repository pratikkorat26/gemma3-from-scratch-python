import unittest

import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import EngineConfig, SamplingConfig
from engine.scheduler import LLMEngine


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text):
        return [int(text)]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{token_id}>" for token_id in ids)


class FakeModel:
    def __init__(self, fail_on_token=None):
        self.context_length = 1024
        self.fail_on_token = fail_on_token
        self.call_last_tokens = []

    def __call__(self, input_ids, past_kv=None, use_cache=True):
        batch_size = int(input_ids.shape[0])
        last_tokens = [int(input_ids[idx, -1].item()) for idx in range(batch_size)]
        self.call_last_tokens.extend(last_tokens)
        for last_token in last_tokens:
            if self.fail_on_token is not None and last_token == self.fail_on_token:
                raise RuntimeError(f"forced failure on token {last_token}")

        vocab_size = 2048
        logits = torch.full((batch_size, input_ids.shape[1], vocab_size), -1e9)
        cache_values = torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32)
        for idx, last_token in enumerate(last_tokens):
            next_token = last_token + 10
            logits[idx, -1, next_token] = 0.0
            cache_values[idx, 0, 0, 0] = float(last_token)
        next_cache = [(cache_values, cache_values.clone())]
        return logits, next_cache


class FakeModelWithStrictCacheIsolation:
    def __init__(self):
        self.context_length = 1024

    def __call__(self, input_ids, past_kv=None, use_cache=True):
        batch_size = int(input_ids.shape[0])
        vocab_size = 4096
        logits = torch.full((batch_size, input_ids.shape[1], vocab_size), -1e9)
        cache_values = torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32)

        for idx in range(batch_size):
            current_token = int(input_ids[idx, -1].item())
            if past_kv is not None:
                cache_signature = int(past_kv[0][0][idx, 0, 0, 0].item())
                expected_signature = current_token - 1
                if cache_signature != expected_signature:
                    raise RuntimeError(
                        f"KV cache leakage detected: cache={cache_signature}, expected={expected_signature}"
                    )

            next_token = current_token + 1
            logits[idx, -1, next_token] = 0.0
            cache_values[idx, 0, 0, 0] = float(current_token)

        next_cache = [(cache_values, cache_values.clone())]
        return logits, next_cache


class FakeRuntime:
    def __init__(self, fail_on_token=None):
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel(fail_on_token=fail_on_token)


class FakeRuntimeStrictCache:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.model = FakeModelWithStrictCacheIsolation()


class RoundRobinSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.config = EngineConfig(
            choose_model="270m",
            use_instruct_model=False,
            max_new_tokens=2,
            num_prefill_workers=1,
            num_decode_workers=1,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            ),
        )

    def test_round_robin_token_order(self):
        runtime = FakeRuntime()
        engine = LLMEngine(runtime=runtime, config=self.config)

        results = engine.generate_many(["1", "2", "3"])

        self.assertEqual(len(runtime.model.call_last_tokens), 6)
        self.assertEqual(sorted(runtime.model.call_last_tokens), [1, 2, 3, 11, 12, 13])
        self.assertEqual([result.token_ids for result in results], [[11, 21], [12, 22], [13, 23]])
        self.assertEqual([result.stop_reason for result in results], ["max_new_tokens"] * 3)

    def test_error_isolation(self):
        runtime = FakeRuntime(fail_on_token=2)
        engine = LLMEngine(runtime=runtime, config=self.config)

        results = engine.generate_many(["1", "2", "3"])

        self.assertEqual(results[1].stop_reason, "error")
        self.assertIn("forced failure", results[1].error_message or "")
        self.assertEqual(results[0].stop_reason, "max_new_tokens")
        self.assertEqual(results[2].stop_reason, "max_new_tokens")

    def test_kv_cache_isolation_between_requests(self):
        runtime = FakeRuntimeStrictCache()
        engine = LLMEngine(runtime=runtime, config=self.config)

        results = engine.generate_many(["100", "200", "300"])

        for result in results:
            self.assertEqual(result.stop_reason, "max_new_tokens")
            self.assertIsNone(result.error_message)

    def test_zero_max_new_tokens_respected(self):
        runtime = FakeRuntime()
        engine = LLMEngine(runtime=runtime, config=self.config)

        result = engine.generate_many(["1"], max_new_tokens=0)[0]

        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertEqual(result.token_ids, [])
        self.assertEqual(result.text, "")


if __name__ == "__main__":
    unittest.main()
