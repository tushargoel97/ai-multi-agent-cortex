import asyncio
import json
import logging
import re
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services import model_manager as mm

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    top_p: float | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None


def inference_error(error: Exception) -> HTTPException:
    message = str(error)
    if "context window" in message.lower() and "exceed" in message.lower():
        return HTTPException(
            400,
            "The request exceeds the local model context window. Shorten the "
            "conversation or increase this model's context length.",
        )
    return HTTPException(500, "Local inference failed.")


@router.get("/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": model["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cortex-local",
            }
            for model in mm.list_downloaded()
        ],
    }


_TOOLCALL_TAG = re.compile(
    r"<tool_call>\s*(\{.*?\}|\[.*?\])\s*</tool_call>", re.DOTALL
)


def _json_or_none(value: str):
    value = value.strip().strip("`").strip()
    try:
        return json.loads(value)
    except ValueError:
        match = re.search(r"\{.*\}|\[.*\]", value, re.DOTALL)
        try:
            return json.loads(match.group(0)) if match else None
        except ValueError:
            return None


def _parse_tool_calls(text: str, tool_names: set[str] | None) -> list | None:
    candidates = _TOOLCALL_TAG.findall(text) or [
        re.sub(r"^\s*\[TOOL_CALLS\]", "", text).strip()
    ]
    calls = []
    for candidate in candidates:
        parsed = _json_or_none(candidate)
        for item in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            arguments = item.get("arguments", item.get("parameters"))
            if not name or tool_names is not None and name not in tool_names:
                continue
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": (
                            arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments or {})
                        ),
                    },
                }
            )
    return calls or None


def _recover_tool_calls(result: dict, tool_names: set[str] | None) -> None:
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return
    if message.get("tool_calls") or not isinstance(message.get("content"), str):
        return
    calls = _parse_tool_calls(message["content"], tool_names)
    if calls:
        message["tool_calls"] = calls
        message["content"] = None
        choice["finish_reason"] = "tool_calls"


def _message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def _request_kwargs(body: ChatRequest) -> tuple[dict, set[str] | None]:
    messages = []
    for source in body.messages:
        message = {
            "role": source.role,
            "content": (
                _message_content(source.content)
                if source.content is not None
                else None
            ),
        }
        for field in ("tool_calls", "tool_call_id", "name"):
            if value := getattr(source, field):
                message[field] = value
        messages.append(message)
    kwargs = {
        "messages": messages,
        "temperature": body.temperature,
        "max_tokens": body.max_tokens or 1024,
    }
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p
    if not body.tools:
        return kwargs, None
    kwargs["tools"] = body.tools
    kwargs["tool_choice"] = body.tool_choice or "auto"
    names = {
        tool["function"]["name"]
        for tool in body.tools
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name")
    }
    return kwargs, names


def _normalize(result: dict, model_name: str | None, *, chunk: bool = False) -> dict:
    result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
    result.setdefault(
        "object", "chat.completion.chunk" if chunk else "chat.completion"
    )
    result.setdefault("created", int(time.time()))
    result["model"] = model_name or "local"
    return result


async def _complete(requested: str | None, kwargs: dict) -> tuple[dict, str | None]:
    try:
        async with mm.model_session(requested) as (llm, model_name):
            if llm is None:
                raise HTTPException(503, "No model is currently loaded.")
            result = await asyncio.to_thread(
                lambda: llm.create_chat_completion(**kwargs)
            )
            return result, model_name
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except Exception as error:
        logger.exception("Local inference failed")
        raise inference_error(error) from error


def _completed_stream(result: dict):
    choice = result["choices"][0]
    message = choice["message"]
    delta = {"role": "assistant"}
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": index, **tool_call}
            for index, tool_call in enumerate(message["tool_calls"])
        ]
    base = {key: result[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    payload = {
        **base,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    yield f"data: {json.dumps(payload)}\n\n"
    finish = choice.get("finish_reason") or "stop"
    payload = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }
    yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


async def _live_stream(requested: str | None, kwargs: dict):
    try:
        async with mm.model_session(requested) as (llm, model_name):
            if llm is None:
                yield _stream_error("No model is currently loaded.")
                yield "data: [DONE]\n\n"
                return
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()
            stop = threading.Event()

            def producer():
                iterator = None
                try:
                    iterator = llm.create_chat_completion(**kwargs, stream=True)
                    for chunk in iterator:
                        if stop.is_set():
                            break
                        loop.call_soon_threadsafe(
                            queue.put_nowait, _normalize(chunk, model_name, chunk=True)
                        )
                except Exception as error:
                    loop.call_soon_threadsafe(queue.put_nowait, error)
                finally:
                    if iterator is not None and hasattr(iterator, "close"):
                        iterator.close()
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            worker = asyncio.create_task(
                asyncio.to_thread(producer),
                name=f"local-inference:{model_name or 'local'}",
            )
            try:
                while (item := await queue.get()) is not sentinel:
                    if isinstance(item, Exception):
                        yield _stream_error(inference_error(item).detail)
                        continue
                    yield f"data: {json.dumps(item)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                stop.set()
                await asyncio.shield(worker)
    except FileNotFoundError as error:
        yield _stream_error(str(error))
        yield "data: [DONE]\n\n"


def _stream_error(message: str) -> str:
    return f"data: {json.dumps({'error': {'message': message}})}\n\n"


@router.post("/chat/completions")
async def chat_completions(body: ChatRequest):
    requested = body.model
    if requested and not mm.is_downloaded(requested):
        raise HTTPException(
            404,
            f"Model '{requested}' not downloaded. Download it via /api/v1/admin/download first.",
        )
    if requested is None and mm.get_llm() is None:
        raise HTTPException(503, "No model is currently loaded.")

    kwargs, tool_names = _request_kwargs(body)
    if body.tools:
        result, model_name = await _complete(requested, kwargs)
        _recover_tool_calls(result, tool_names)
        result = _normalize(result, model_name)
        return (
            StreamingResponse(_completed_stream(result), media_type="text/event-stream")
            if body.stream
            else result
        )
    if not body.stream:
        result, model_name = await _complete(requested, kwargs)
        return _normalize(result, model_name)
    return StreamingResponse(
        _live_stream(requested, kwargs), media_type="text/event-stream"
    )
