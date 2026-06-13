import json

from app.types import ChatCompletionEvent


def event_to_sse(event: ChatCompletionEvent) -> str:
    if event.done:
        return "data: [DONE]\n\n"
    if event.error is not None:
        return f"data: {json.dumps({'error': event.error})}\n\n"
    if event.usage is not None:
        payload = {
            "id": event.request_id,
            "object": "chat.completion.chunk",
            "created": event.created,
            "model": event.model,
            "choices": [],
            "usage": event.usage.dict(),
        }
        return f"data: {json.dumps(payload)}\n\n"
    if event.finish_reason is not None:
        payload = {
            "id": event.request_id,
            "object": "chat.completion.chunk",
            "created": event.created,
            "model": event.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": event.finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"
    delta = {}
    if event.role is not None:
        delta["role"] = event.role
    if event.content is not None:
        delta["content"] = event.content
    payload = {
        "id": event.request_id,
        "object": "chat.completion.chunk",
        "created": event.created,
        "model": event.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"
