import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestTrace:
    trace_id: str
    mode: str
    events: list[dict] = field(default_factory=list)
    final_text: str = ""
    round_trips: int = 0

    def add_event(self, name: str, **data) -> None:
        event = {
            "name": name,
            "ts_unix_s": time.time(),
        }
        if data:
            event["data"] = data
        self.events.append(event)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "mode": self.mode,
            "events": self.events,
            "summary": {
                "round_trips": self.round_trips,
                "final_text": self.final_text,
            },
        }


def _handle_http_error(exc: urllib.error.HTTPError) -> None:
    try:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body)
        detail = parsed.get("detail", body)
    except Exception:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
    print(f"Error {exc.code}: {detail}", file=sys.stderr)


def _post_chat_completion_with_meta(url: str, payload: dict, *, trace_id: Optional[str] = None) -> tuple[dict, dict]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if trace_id:
        headers["X-Trace-Id"] = trace_id
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        body = response.read().decode("utf-8")
        response_headers = dict(response.info().items())
    return json.loads(body), response_headers


def _parse_server_trace(headers: dict) -> Optional[dict]:
    raw = headers.get("X-Trace-Data")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_assistant_text(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content") or ""


def run_non_stream(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    *,
    trace: Optional[RequestTrace] = None,
) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    try:
        if trace is not None:
            trace.add_event("client.request_started", prompt=prompt)
            trace.add_event("client.http_request_sent", round_trip=1, message_count=1)
        parsed, headers = _post_chat_completion_with_meta(url, payload, trace_id=(trace.trace_id if trace else None))
        if trace is not None:
            trace.round_trips = 1
            trace.add_event(
                "client.http_response_received",
                round_trip=1,
                finish_reason=parsed.get("choices", [{}])[0].get("finish_reason"),
            )
            server_trace = _parse_server_trace(headers)
            if server_trace is not None:
                trace.add_event("server.trace_received", event_count=len(server_trace.get("events", [])))
                trace.events.extend(server_trace.get("events", []))
            trace.final_text = _extract_assistant_text(parsed)
            trace.add_event("client.final_text_ready", text_chars=len(trace.final_text))
        else:
            print(_extract_assistant_text(parsed))
    except urllib.error.HTTPError as exc:
        _handle_http_error(exc)
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(f"Connection failed: {exc.reason}", file=sys.stderr)
        raise SystemExit(1)


def run_stream(url: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_part = line[len("data: ") :]
                if data_part == "[DONE]":
                    break
                event = json.loads(data_part)
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content")
                if text:
                    print(text, end="", flush=True)
        print()
    except urllib.error.HTTPError as exc:
        _handle_http_error(exc)
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(f"Connection failed: {exc.reason}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local FastAPI OpenAI-like endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default="gemma-3-270m-it")
    parser.add_argument("--prompt", default="Give me one short line about LLM inference.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--trace-request", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    trace = None
    if args.trace_request:
        if args.stream:
            print("--trace-request is not supported together with --stream", file=sys.stderr)
            raise SystemExit(1)
        trace = RequestTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:24]}",
            mode="chat",
        )
    if args.stream:
        run_stream(url, args.model, args.prompt, args.max_tokens, args.temperature, args.top_p)
    else:
        run_non_stream(url, args.model, args.prompt, args.max_tokens, args.temperature, args.top_p, trace=trace)
        if trace is not None:
            print(json.dumps(trace.to_dict(), indent=2))


if __name__ == "__main__":
    main()
