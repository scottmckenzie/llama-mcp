# llama-mcp

MCP server that exposes a llama.cpp `llama-server` as tools over stdio. The tool list, configuration and
Claude Code registration command are in `README.md`.

## Layout

- `src/llama_mcp/server.py`: `create_server(client)` registers every tool on an `MCPServer`. It takes an
  `httpx.AsyncClient` already pointed at llama-server, so tests can pass one backed by `httpx.MockTransport`.
- `src/llama_mcp/cli.py`: the `llama-mcp` entry point. It reads `--url`/`LLAMA_URL`, `LLAMA_API_KEY` and
  `--timeout`, then runs the server over stdio.
- `tests/test_server.py`: drives the server in-process with `mcp.Client(server)` against a fake llama-server.

## Workflow

Use uv for everything (`uv add`, `uv sync`, `uv run ...`); never call pip, python or pytest directly.
Before finishing a change, run all of these and make sure they pass:

- `uv run coverage run -m pytest && uv run coverage report` (fails below 80%, lines and branches)
- `uv run ruff check && uv run ruff format --check`
- `uv run pyrefly check` (strict preset)

CI (`.github/workflows/ci.yml`) runs the same checks plus `uv build`, with tests on Python 3.12 and 3.14;
keep `requires-python` and that matrix in sync.

## Development Standards

### Git Workflow
- Branch names: `feature/description` or `fix/description`
- Commit messages: Follow conventional commits
- PR required before merge
- All CI/CD checks must pass
- Minimum 1 approval required

### GitHub Actions
- Pin every action to a full commit SHA, with the release tag in a comment:
  `uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1`. Never use a tag or branch
  (`@v7`, `@main`), which can be moved to different code
- Look up a release's SHA with `gh api repos/<owner>/<repo>/commits/<tag> --jq .sha`, and update the
  SHA and the comment together

### Testing Requirements
- Minimum 80% code coverage
- All critical paths must have tests

## Gotchas

- This uses **mcp 2.x**. `FastMCP` and `mcp.server.fastmcp` are gone; the class is
  `mcp.server.mcpserver.MCPServer`, and the in-process test client is `mcp.Client`. Examples written for
  v1 will not work. Check `.venv/lib/python3.12/site-packages/mcp/` when unsure of the API.
- stdout is the MCP transport. Never `print` from server code, because anything written to stdout corrupts
  the protocol. Logging goes to stderr.
- Raise `ToolError` for llama-server failures (connection errors, HTTP status errors), so the client gets
  an error result and the server keeps running.
- Leave sampling parameters (`temperature` and similar) out of requests unless the caller sets them. The
  user tunes sampling with llama-server flags (`--temp`, `--top-p`, ...), and sending a default would
  override those flags.
- Async tests use the anyio pytest plugin (`pytestmark = pytest.mark.anyio` with an `anyio_backend`
  fixture); pytest-asyncio is not installed.
- llama-server has two modes. A single-model server needs no `model` field. A router-mode server (no
  `-m`/`-hf`, several models) rejects requests without one ("model name is missing from the request") and
  loads a model the first time a request names it. Support both: send `model` only when the caller or
  `--model` gives one.
- `mcp.Client(StdioServerParameters(...))` passes only a minimal environment to the subprocess, so
  `LLAMA_*` variables in your shell don't reach it. Pass `--url`/`--model` as arguments, or set `env=`.
- To test against real hardware, llama-server usually runs at `http://127.0.0.1:8080`. Check
  `curl -s localhost:8080/health` first, and spawn `uv run llama-mcp` through
  `mcp.Client(StdioServerParameters(...))` to exercise the real entry point. Loading a router model
  can take minutes and a lot of VRAM, so test with the one already `loaded` (see `models`).
