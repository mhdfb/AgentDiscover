"""The MCP service: the only route from an agent to the database and the evaluator.
Endpoint: http://127.0.0.1:<port>/mcp/<session-token> (streamable HTTP, stateless).
Run: python -m agentdiscover.server.app --problem <name> --backend <backend>
"""
from __future__ import annotations

import argparse
import re
import time
from typing import Any, Literal

import anyio
import anyio.to_thread
import mcp.types as types
from mcp.server.mcpserver import Context, MCPServer

# Claude Code's HTTP client drops a tool call that receives no bytes for ~5 minutes,
# so submit_candidate sends a progress notification this often while it waits.
KEEPALIVE_SECONDS = 30
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from .. import config
from . import auth, dynamic_tools, tools_meta, tools_search
from .tools_search import ToolError

SEARCH_TOOLS = {"read_graph", "submit_candidate", "add_idea", "write_lesson"}
META_TOOLS = {"read_graph", "write_meta_lesson", "add_tool", "write_strategy"}

_problem: str = ""


# ---------------------------------------------------------------------------
# The server: per-connection tool list, per-call attribution
# ---------------------------------------------------------------------------


class AgentDiscoverServer(MCPServer):
    async def list_tools(self) -> list[types.Tool]:
        every = await super().list_tools()
        try:
            ctx = auth.require()
        except auth.AuthError:
            return []
        if ctx.kind == "meta":
            return [t for t in every if t.name in META_TOOLS]
        builtins = [t for t in every if t.name in SEARCH_TOOLS]
        rows = await anyio.to_thread.run_sync(dynamic_tools.branch_tool_rows, ctx.agent_key)
        return builtins + dynamic_tools.as_mcp_tools(rows)

    async def call_tool(self, name: str, arguments: dict[str, Any], context=None):
        def _text(msg: str) -> types.CallToolResult:
            return types.CallToolResult(content=[types.TextContent(type="text", text=msg)])

        try:
            ctx = auth.require()
        except auth.AuthError as e:
            return _text(f"error: {e}")

        allowed = META_TOOLS if ctx.kind == "meta" else SEARCH_TOOLS
        if name in (SEARCH_TOOLS | META_TOOLS) and name not in allowed:
            return _text(
                f"error: {name} is not available to {ctx.kind} callers."
            )
        if name not in (SEARCH_TOOLS | META_TOOLS):
            try:
                out = await anyio.to_thread.run_sync(dynamic_tools.call, name, arguments)
                return _text(out)
            except ToolError as e:
                return _text(f"error: {e}")
        return await super().call_tool(name, arguments, context)


server = AgentDiscoverServer(
    name="agentdiscover",
    instructions=(
        "The AgentDiscover history service. Explore with read_graph; write only "
        "through the specific tools. Every call is attributed to your session."
    ),
)


# ---------------------------------------------------------------------------
# Built-in tools. Docstrings and signatures are the agent-facing contract.
# Sync work runs in threads so it never blocks the event loop.
# ---------------------------------------------------------------------------


class Parent(BaseModel):
    genome_hash: str = Field(description="genome_hash of an existing candidate")
    kind: Literal["mutation", "crossover", "repair", "inspired"] = Field(
        description="how this candidate came from that parent: mutation = changed one "
        "parent; crossover = combined two or more; repair = fixed a flaw in one; "
        "inspired = read it, did not reuse the code"
    )


class Relation(BaseModel):
    name: str = Field(description="name of an existing idea to link to")
    how: str = Field(description="one short sentence saying how the two ideas relate")


@server.tool()
async def read_graph(cypher: str) -> str:
    """Run any read-only Cypher query against the shared history graph and get the
    rows as JSON. The schema is in your instructions. Writes are rejected by the
    database itself — use the specific write tools instead."""
    try:
        return await anyio.to_thread.run_sync(
            tools_search.read_graph, cypher, config.READ_ROW_LIMIT
        )
    except ToolError as e:
        return f"error: {e}"


@server.tool()
async def add_idea(name: str, description: str,
                   related_to: list[Relation] | None = None) -> str:
    """Create an idea (or declare an existing one you will reuse) before submitting a
    candidate built on it. `name` is a short kebab-case handle (reused across
    candidates); `description` says what the idea is; `related_to` links it to
    existing ideas."""
    try:
        rels = [r.model_dump() for r in (related_to or [])]
        return await anyio.to_thread.run_sync(
            tools_search.add_idea, name, description, rels
        )
    except ToolError as e:
        return f"error: {e}"


async def _heartbeat(ctx: Context, started: float) -> None:
    while True:
        await anyio.sleep(KEEPALIVE_SECONDS)
        elapsed = int(time.monotonic() - started)
        await ctx.report_progress(elapsed, None, f"evaluation running, {elapsed}s elapsed")


@server.tool()
async def submit_candidate(idea: str, linguistic_prediction: str,
                           fitness_prediction: float, ctx: Context,
                           parents: list[Parent] | None = None,
                           resources: list[str] | None = None) -> str:
    """Submit the current solution.py in your worktree as a candidate. The server
    reads the file itself, evaluates it in a locked sandbox, records everything, and
    returns the score — the call blocks until then; never poll.

    `idea`: the idea this candidate uses (must exist — add_idea first or reuse one).
    `linguistic_prediction`: one or two sentences, written BEFORE evaluation, saying
    what you expect and why. `fitness_prediction`: the number you expect.
    `parents`: the candidates this one came from, each with a kind.
    `resources`: names of provided resource files that shaped this candidate."""
    ps = [p.model_dump() for p in (parents or [])]
    async with anyio.create_task_group() as tg:
        tg.start_soon(_heartbeat, ctx, time.monotonic())
        try:
            result = await anyio.to_thread.run_sync(
                tools_search.submit_candidate, idea, linguistic_prediction,
                fitness_prediction, ps, resources or []
            )
        except ToolError as e:
            result = f"error: {e}"
        finally:
            tg.cancel_scope.cancel()
    return result


@server.tool()
async def write_lesson(text: str) -> str:
    """After your last candidate: write one lesson for this whole session — what you
    tried, what the scores taught you, what the next session should know, and the work
    that never became a candidate (only submitted ones are in the graph). Then stop."""
    try:
        return await anyio.to_thread.run_sync(tools_search.write_lesson, text)
    except ToolError as e:
        return f"error: {e}"


@server.tool()
async def write_meta_lesson(session_id: str, text: str) -> str:
    """Meta pass only. Store your judgment of how well one session searched —
    independent of the score it happened to get — on that Session node."""
    try:
        return await anyio.to_thread.run_sync(tools_meta.write_meta_lesson, session_id, text)
    except ToolError as e:
        return f"error: {e}"


@server.tool()
async def add_tool(branch: str, name: str, description: str,
                   kind: Literal["cypher"], code: str,
                   params: dict[str, Any] | None = None) -> str:
    """Meta pass only. Store a SAVED QUERY for ONE branch's future sessions. They
    discover it exactly like a built-in tool: `name`, `description`, and `params`
    (a JSON Schema object) are what the agent sees; the Cypher is not. It runs
    server-side in a read transaction; the call's arguments become its query
    parameters. Executable code is not accepted here — search agents share code
    through their worktree tools/ folder, which they can test and you cannot."""
    try:
        return await anyio.to_thread.run_sync(
            tools_meta.add_tool, branch, name, description, kind, code, params
        )
    except ToolError as e:
        return f"error: {e}"


@server.tool()
async def write_strategy(branch: str, strategy: str, diff_summary: str) -> str:
    """Meta pass only. Store ONE branch's new strategy — written to the next fresh
    agent of that branch, every claim backed by evidence from the sessions, never
    naming iterations, parents, or genomes. `diff_summary`: one or two lines on what
    changed versus the previous version and why."""
    try:
        return await anyio.to_thread.run_sync(
            tools_meta.write_strategy, branch, strategy, diff_summary
        )
    except ToolError as e:
        return f"error: {e}"


@server.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "problem": _problem})


# ---------------------------------------------------------------------------
# Transport: /mcp/<token> → token contextvar + /mcp
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"^/mcp/([A-Za-z0-9_\-]{8,128})/?$")


class TokenPathMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            m = _TOKEN_RE.match(scope.get("path", ""))
            if m:
                auth.current_token.set(m.group(1))
                scope = dict(scope)
                scope["path"] = "/mcp"
                scope["raw_path"] = b"/mcp"
        await self.app(scope, receive, send)


def build_app(problem: str, backend: str):
    global _problem
    _problem = problem
    auth.configure(problem)
    tools_search.configure(problem, backend)
    dynamic_tools.configure(problem, backend)
    # Responses must be streamed (SSE), not plain JSON, or the heartbeat has no channel.
    inner = server.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, json_response=False
    )
    return TokenPathMiddleware(inner)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="agentdiscover.server.app")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    app = build_app(args.problem, args.backend)
    uvicorn.run(app, host="127.0.0.1", port=config.SERVICE_PORT, log_level="warning")


if __name__ == "__main__":
    main()
