from typing import Any

import httpx
import pytest

from llama_mcp import cli


class FakeServer:
    def __init__(self, client: httpx.AsyncClient, default_model: str | None) -> None:
        self.client = client
        self.default_model = default_model
        self.runs: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.runs.append((args, kwargs))


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch) -> list[FakeServer]:
    """Servers `cli.main` built, so it never starts a real one."""
    built: list[FakeServer] = []

    def create_server(client: httpx.AsyncClient, default_model: str | None = None) -> FakeServer:
        built.append(FakeServer(client, default_model))
        return built[-1]

    monkeypatch.setattr(cli, "create_server", create_server)
    for name in ("LLAMA_URL", "LLAMA_MODEL", "LLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return built


def test_defaults_to_local_server_over_stdio(servers: list[FakeServer]) -> None:
    cli.main([])
    [server] = servers
    assert server.client.base_url == httpx.URL("http://localhost:8080")
    assert server.client.timeout == httpx.Timeout(300.0)
    assert "authorization" not in server.client.headers
    assert server.default_model is None
    assert server.runs == [(("stdio",), {})]


def test_reads_environment(servers: list[FakeServer], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_URL", "http://gpu.lan:9000")
    monkeypatch.setenv("LLAMA_MODEL", "qwen")
    monkeypatch.setenv("LLAMA_API_KEY", "secret")
    cli.main([])
    [server] = servers
    assert server.client.base_url == httpx.URL("http://gpu.lan:9000")
    assert server.client.headers["authorization"] == "Bearer secret"
    assert server.default_model == "qwen"


def test_flags_override_environment(servers: list[FakeServer], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_URL", "http://gpu.lan:9000")
    monkeypatch.setenv("LLAMA_MODEL", "qwen")
    cli.main(["--url", "http://other:8081", "--model", "llama", "--timeout", "5"])
    [server] = servers
    assert server.client.base_url == httpx.URL("http://other:8081")
    assert server.client.timeout == httpx.Timeout(5.0)
    assert server.default_model == "llama"


def test_http_serves_on_localhost(servers: list[FakeServer]) -> None:
    cli.main(["--http", "8765"])
    assert servers[0].runs == [(("streamable-http",), {"host": "127.0.0.1", "port": 8765})]
