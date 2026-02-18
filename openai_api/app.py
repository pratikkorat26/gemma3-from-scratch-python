from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .schemas import ChatCompletionRequest
from .service import ChatCompletionService

app = FastAPI(title="Gemma OpenAI-like API", version="0.1.0")
_service_lock = Lock()


def get_service(fastapi_app: FastAPI) -> ChatCompletionService:
    if hasattr(fastapi_app.state, "service"):
        return fastapi_app.state.service
    with _service_lock:
        if not hasattr(fastapi_app.state, "service"):
            fastapi_app.state.service = ChatCompletionService()
    return fastapi_app.state.service


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    service = get_service(request.app)
    try:
        if payload.stream:
            return StreamingResponse(
                service.stream_chat_completion(payload),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        response = await run_in_threadpool(service.create_chat_completion, payload)
        return JSONResponse(response)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
