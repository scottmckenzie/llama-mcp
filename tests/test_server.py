import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from mcp import Client

from llama_mcp.server import create_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


Handler = Callable[[httpx.Request], httpx.Response]


def fake_llama(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://llama.test", transport=httpx.MockTransport(handler))


async def call(
    handler: Handler, tool: str, arguments: dict[str, Any] | None = None, default_model: str | None = None
) -> Any:
    async with Client(create_server(fake_llama(handler), default_model=default_model)) as client:
        result = await client.call_tool(tool, arguments or {})
    return result


async def test_lists_tools() -> None:
    async with Client(create_server(fake_llama(lambda r: httpx.Response(500)))) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools.tools} == {"chat", "complete", "tokenize", "models", "health"}


async def test_chat_sends_system_and_prompt() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi there"}}]})

    result = await call(handler, "chat", {"prompt": "hello", "system": "be brief", "max_tokens": 16})
    assert not result.is_error
    assert result.structured_content == {"result": "hi there"}
    assert seen[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert seen[0]["max_tokens"] == 16
    assert "temperature" not in seen[0]  # llama-server's --temp applies
    assert "model" not in seen[0]  # a single-model llama-server needs none


async def test_chat_sends_history_between_system_and_prompt() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "fixed"}}]})

    history = [{"role": "user", "content": "write f"}, {"role": "assistant", "content": "def f(): ..."}]
    result = await call(handler, "chat", {"prompt": "fix f", "system": "be brief", "history": history})
    assert result.structured_content == {"result": "fixed"}
    assert seen[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "write f"},
        {"role": "assistant", "content": "def f(): ..."},
        {"role": "user", "content": "fix f"},
    ]


async def test_chat_rejects_unknown_history_role() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an invalid call must not reach llama-server")

    result = await call(handler, "chat", {"prompt": "hi", "history": [{"role": "system", "content": "x"}]})
    assert result.is_error


async def test_complete() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/completion"
        body = json.loads(request.content)
        assert body["prompt"] == "Once upon"
        assert body["temperature"] == 0.2
        return httpx.Response(200, json={"content": " a time"})

    result = await call(handler, "complete", {"prompt": "Once upon", "temperature": 0.2})
    assert result.structured_content == {"result": " a time"}


async def test_tokenize() -> None:
    result = await call(lambda r: httpx.Response(200, json={"tokens": [1, 2, 3]}), "tokenize", {"text": "abc"})
    assert result.structured_content == {"result": [1, 2, 3]}


async def test_models_single() -> None:
    result = await call(lambda r: httpx.Response(200, json={"data": [{"id": "qwen.gguf"}]}), "models")
    assert result.structured_content == {"result": [{"id": "qwen.gguf", "status": "loaded"}]}


async def test_models_router() -> None:
    data = {"data": [{"id": "a", "status": {"value": "loaded"}}, {"id": "b", "status": {"value": "unloaded"}}]}
    result = await call(lambda r: httpx.Response(200, json=data), "models")
    assert result.structured_content == {"result": [{"id": "a", "status": "loaded"}, {"id": "b", "status": "unloaded"}]}


async def test_model_argument_and_default() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"tokens": [1]})

    await call(handler, "tokenize", {"text": "x"}, default_model="fallback")
    await call(handler, "tokenize", {"text": "x", "model": "chosen"}, default_model="fallback")
    assert seen == ["fallback", "chosen"]


async def test_health() -> None:
    result = await call(lambda r: httpx.Response(200, json={"status": "ok"}), "health")
    assert result.structured_content == {"result": "ok"}


async def test_http_error_is_tool_error() -> None:
    result = await call(lambda r: httpx.Response(503, text="Loading model"), "health")
    assert result.is_error
    assert "503" in str(result.content)


async def test_unreachable_is_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    result = await call(handler, "health")
    assert result.is_error
    assert "cannot reach llama-server" in str(result.content)
