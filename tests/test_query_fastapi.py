import unittest

import query_fastapi
from query_fastapi import (
    RequestTrace,
    _extract_assistant_text,
    _parse_server_trace,
    run_non_stream,
)


class QueryFastAPITests(unittest.TestCase):
    def test_extract_assistant_text_handles_missing_content(self):
        payload = {"choices": [{"message": {"content": None}}]}
        self.assertEqual(_extract_assistant_text(payload), "")

    def test_parse_server_trace_handles_missing_header(self):
        self.assertIsNone(_parse_server_trace({}))

    def test_parse_server_trace_accepts_valid_json(self):
        trace = _parse_server_trace(
            {
                "X-Trace-Data": "{\"trace_id\":\"trace_1\",\"events\":[{\"name\":\"service.response_built\",\"ts_unix_s\":0.0}]}",
            }
        )
        self.assertIsNotNone(trace)
        self.assertEqual(trace["trace_id"], "trace_1")

    def test_run_non_stream_populates_trace(self):
        trace = RequestTrace(trace_id="trace_2", mode="chat")

        def fake_post(url, payload, trace_id=None):
            self.assertEqual(trace_id, "trace_2")
            return (
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Plain answer."},
                            "finish_reason": "stop",
                        }
                    ]
                },
                {
                    "X-Trace-Data": "{\"trace_id\":\"trace_2\",\"component\":\"server\",\"events\":[{\"name\":\"service.response_built\",\"ts_unix_s\":0.0}]}",
                },
            )

        original_post = query_fastapi._post_chat_completion_with_meta
        query_fastapi._post_chat_completion_with_meta = fake_post
        try:
            run_non_stream(
                "http://localhost:8000/v1/chat/completions",
                "gemma-3-270m-it",
                "Hello",
                64,
                0.0,
                0.9,
                trace=trace,
            )
        finally:
            query_fastapi._post_chat_completion_with_meta = original_post

        self.assertEqual(trace.final_text, "Plain answer.")
        event_names = [event["name"] for event in trace.events]
        self.assertIn("client.http_response_received", event_names)
        self.assertIn("service.response_built", event_names)


if __name__ == "__main__":
    unittest.main()
