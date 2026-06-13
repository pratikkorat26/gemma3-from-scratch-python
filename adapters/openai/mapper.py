from app.types import ChatCompletionRequest as AppChatCompletionRequest
from app.types import ChatCompletionResult, ChatMessage
from inference import SamplingConfig

from .schemas import ChatCompletionRequest


def to_app_request(request: ChatCompletionRequest) -> AppChatCompletionRequest:
    stop = request.stop
    if isinstance(stop, str):
        stop = [stop]
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    return AppChatCompletionRequest(
        model=request.model,
        messages=[ChatMessage(role=message.role, content=message.content) for message in request.messages],
        max_tokens=request.max_tokens,
        sampling=SamplingConfig(
            temperature=float(request.temperature) if request.temperature is not None else 0.8,
            top_p=float(request.top_p) if request.top_p is not None else 0.9,
            top_k=int(request.top_k) if request.top_k is not None else 50,
            repetition_penalty=float(request.repetition_penalty)
            if request.repetition_penalty is not None
            else 1.1,
            seed=request.seed,
        ),
        stop=stop,
        stream=request.stream,
        include_usage=include_usage,
    )


def chat_completion_to_openai(result: ChatCompletionResult) -> dict:
    return {
        "id": result.request_id,
        "object": "chat.completion",
        "created": result.created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": result.usage.dict(),
    }
