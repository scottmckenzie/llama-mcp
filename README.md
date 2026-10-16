# llama-mcp

An MCP server that exposes a [llama.cpp](https://github.com/ggml-org/llama.cpp) server (`llama-server`) as tools, so an MCP host such as Claude Code can hand work to a local model.

## Tools

| Tool       | llama-server endpoint        | Purpose                                         |
|------------|------------------------------|-------------------------------------------------|
| `chat`     | `POST /v1/chat/completions`  | Prompt (plus optional system prompt) → reply    |
| `complete` | `POST /completion`           | Raw text continuation, no chat template         |
| `tokenize` | `POST /tokenize`             | Text → token ids                                |
| `models`   | `GET /v1/models`             | Model ids and whether each is loaded            |
| `health`   | `GET /health`                | `ok` once a model is loaded                     |

## Usage

Start llama-server, then register the MCP server with Claude Code:

```sh
llama-server -m model.gguf --port 8080
claude mcp add llama -- uv run --project ~/Projects/llama-mcp llama-mcp --url http://localhost:8080
```

Configuration:

- `--url` / `LLAMA_URL`: llama-server base URL (default `http://localhost:8080`)
- `--model` / `LLAMA_MODEL`: model id to use when a tool call names none. Required when llama-server runs in
  router mode (started without `-m`/`-hf`, offering several models); `chat`, `complete` and `tokenize` also
  take a per-call `model` argument.
- `LLAMA_API_KEY`: sent as a bearer token, for a llama-server started with `--api-key`
- `--timeout`: request timeout in seconds (default 300)
- `--http PORT`: serve streamable HTTP at `http://127.0.0.1:PORT/mcp` instead of stdio, for testing with
  MCP Inspector or `claude mcp add --transport http`

## Development

```sh
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyrefly check
```
