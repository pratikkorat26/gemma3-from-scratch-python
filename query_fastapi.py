import argparse
import json
import sys
import urllib.error
import urllib.request


def _handle_http_error(exc: urllib.error.HTTPError) -> None:
    try:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body)
        detail = parsed.get("detail", body)
    except Exception:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
    print(f"Error {exc.code}: {detail}", file=sys.stderr)


def run_non_stream(url: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
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
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        print(parsed["choices"][0]["message"]["content"])
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
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    if args.stream:
        run_stream(url, args.model, args.prompt, args.max_tokens, args.temperature, args.top_p)
    else:
        run_non_stream(url, args.model, args.prompt, args.max_tokens, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
