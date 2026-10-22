"""MCP server exposing a llama.cpp server (llama-server) as tools."""

from typing import Any, Literal, TypedDict

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

INSTRUCTIONS = (
    "Tools for a local llama.cpp model. Use `chat` to send a prompt to the local model; the server keeps "
    "no conversation state, so pass earlier turns (including the model's own replies) in `history` to "
    "continue a conversation. Use `complete` for raw text continuation, and `health` or `models` to check "
    "what is loaded. "
    "When llama-server runs in router mode with several models, pass `model` (an id from `models`)."
)


class Turn(TypedDict):
    """An earlier message in a `chat` conversation."""

    role: Literal["user", "assistant"]
    content: str


def create_server(client: httpx.AsyncClient, default_model: str | None = None) -> MCPServer:
    """Build the MCP server; `client` has llama-server's base URL (and any auth header) set.

    `default_model` is sent when a tool call names no model. Leave it None for a single-model llama-server;
    a router-mode llama-server (several models) rejects requests that name none.
    """
    server = MCPServer("llama-mcp", instructions=INSTRUCTIONS)

    def with_model(body: dict[str, Any], model: str | None) -> dict[str, Any]:
        if (chosen := model or default_model) is not None:
            body["model"] = chosen
        return body

    async def request(method: str, path: str, json: dict[str, Any] | None = None) -> Any:
        try:
            response = await client.request(method, path, json=json)
        except httpx.HTTPError as e:
            raise ToolError(f"cannot reach llama-server at {client.base_url}: {e}") from e
        if response.is_error:
            raise ToolError(f"llama-server returned {response.status_code} for {path}: {response.text}")
        return response.json()

    @server.tool()
    async def chat(
        prompt: str,
        system: str | None = None,
        history: list[Turn] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Send a prompt to the local model through its chat template and return the reply.

        Each call is independent. To continue a conversation, pass the earlier turns in `history`, oldest
        first, with the model's previous replies as `assistant` turns; `prompt` is the new user message.
        Leave `temperature` unset to use llama-server's own sampling settings.
        """
        messages = [{"role": "system", "content": system}] if system is not None else []
        messages += [{"role": turn["role"], "content": turn["content"]} for turn in history or []]
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        data = await request("POST", "/v1/chat/completions", with_model(body, model))
        return str(data["choices"][0]["message"]["content"])

    @server.tool()
    async def complete(
        prompt: str, n_predict: int = 256, temperature: float | None = None, model: str | None = None
    ) -> str:
        """Continue raw text with the local model, with no chat template applied.

        Leave `temperature` unset to use llama-server's own sampling settings.
        """
        body: dict[str, Any] = {"prompt": prompt, "n_predict": n_predict}
        if temperature is not None:
            body["temperature"] = temperature
        data = await request("POST", "/completion", with_model(body, model))
        return str(data["content"])

    @server.tool()
    async def tokenize(text: str, model: str | None = None) -> list[int]:
        """Return the model's token ids for `text`."""
        data = await request("POST", "/tokenize", with_model({"content": text}, model))
        return [int(t) for t in data["tokens"]]

    @server.tool()
    async def models() -> list[dict[str, str]]:
        """List the models llama-server offers, with their status.

        A single-model llama-server always reports `loaded`. In router mode a model is `unloaded` until
        a request names it, and the first such request waits while it loads.
        """
        data = await request("GET", "/v1/models")
        return [{"id": str(m["id"]), "status": str(m.get("status", {}).get("value", "loaded"))} for m in data["data"]]

    @server.tool()
    async def health() -> str:
        """Report whether llama-server is up and has a model loaded."""
        data = await request("GET", "/health")
        return str(data.get("status", "unknown"))

    return server
