import argparse
import os

import httpx

from llama_mcp.server import create_server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MCP server for a llama.cpp server")
    parser.add_argument(
        "--url",
        default=os.environ.get("LLAMA_URL", "http://localhost:8080"),
        help="llama-server base URL (env LLAMA_URL; default %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLAMA_MODEL"),
        help="model id to use when a tool call names none; needed for a router-mode llama-server (env LLAMA_MODEL)",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="request timeout in seconds (default %(default)s)")
    parser.add_argument(
        "--http",
        metavar="PORT",
        type=int,
        help="serve streamable HTTP on 127.0.0.1:PORT (path /mcp) instead of stdio",
    )
    args = parser.parse_args(argv)

    headers = {}
    if api_key := os.environ.get("LLAMA_API_KEY"):  # matches llama-server --api-key
        headers["Authorization"] = f"Bearer {api_key}"
    client = httpx.AsyncClient(base_url=args.url, headers=headers, timeout=args.timeout)
    server = create_server(client, default_model=args.model)
    if args.http is None:
        server.run("stdio")
    else:
        server.run("streamable-http", host="127.0.0.1", port=args.http)
