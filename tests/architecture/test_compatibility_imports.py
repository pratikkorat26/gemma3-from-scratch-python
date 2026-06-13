import unittest


class CompatibilityImportTests(unittest.TestCase):
    def test_engine_imports_remain_available(self):
        from engine import EngineConfig, GemmaRuntime, LLMEngine, SamplingConfig
        from engine.scheduler import KVBlockManager

        self.assertIsNotNone(EngineConfig)
        self.assertIsNotNone(GemmaRuntime)
        self.assertIsNotNone(KVBlockManager)
        self.assertIsNotNone(LLMEngine)
        self.assertIsNotNone(SamplingConfig)

    def test_openai_api_imports_remain_available(self):
        from openai_api.app import create_app
        from openai_api.schemas import ChatCompletionRequest, SUPPORTED_MODEL
        from openai_api.service import ChatCompletionService

        self.assertEqual(SUPPORTED_MODEL, "gemma-3-270m-it")
        self.assertIsNotNone(ChatCompletionRequest)
        self.assertIsNotNone(ChatCompletionService)
        self.assertIsNotNone(create_app)


if __name__ == "__main__":
    unittest.main()
