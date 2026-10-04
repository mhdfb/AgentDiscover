"""Meta-written tools, offered per branch to its search agents like built-in tools.
Cypher tools run server-side in a read transaction; other kinds run in a sandbox that
holds only the caller's session token. Each call is recorded as a USED_TOOL edge.
"""
from __future__ import annotations

import json
from pathlib import Path

import mcp.types as types
from neo4j.exceptions import ClientError

from .. import config, db, runconfig, sandbox
from . import auth
from .tools_search import ToolError

_problem: str = ""
_backend: str = "none"

def tool_limits() -> config.Limits:
    """Sandbox limits for a tool call. Built per call: the timeout is per-run config."""
    return config.Limits(
        wall_seconds=int(runconfig.get(_problem, "tool_timeout")),
        mem_gb=2.0, cpus=1.0, nproc=128, fsize_mb=64,
    )

_OUTPUT_CAP = 20_000  # characters of tool output returned to the agent


def configure(problem: str, backend: str) -> None:
    global _problem, _backend
    _problem = problem
    _backend = backend


def branch_tool_rows(agent_key: str) -> list[dict]:
    return db.run_read(
        """
        MATCH (a:Agent {key: $k})-[:HAS_TOOL]->(t:Tool)
        RETURN t.name AS name, t.description AS description,
               t.kind AS kind, t.code AS code, t.params AS params
        """,
        k=agent_key,
    )


def as_mcp_tools(rows: list[dict]) -> list[types.Tool]:
    tools = []
    for r in rows:
        try:
            schema = json.loads(r["params"]) if r.get("params") else None
        except ValueError:
            schema = None
        tools.append(types.Tool(
            name=r["name"],
            description="[meta-written; lower priority — prefer the built-in tools "
                        "when they suffice] " + (r["description"] or ""),
            input_schema=schema or {"type": "object", "properties": {}},
        ))
    return tools


def call(name: str, arguments: dict) -> str:
    ctx = auth.require("search")
    rows = [r for r in branch_tool_rows(ctx.agent_key) if r["name"] == name]
    if not rows:
        raise ToolError(f"no tool named {name!r} for your branch.")
    tool = rows[0]
    arguments = arguments or {}

    if tool["kind"] == "cypher":
        try:
            data = db.run_read(tool["code"], **arguments)
        except ClientError as e:
            output = f"error from meta-written tool '{name}': {e.message}"
        else:
            output = json.dumps(data[:config.READ_ROW_LIMIT], default=str, ensure_ascii=False)
    else:
        output = _run_in_sandbox(ctx, tool, arguments)

    _record_use(ctx, name)
    if len(output) > _OUTPUT_CAP:
        output = output[:_OUTPUT_CAP] + "\n... (output truncated)"
    return output + f"\n\n(meta-written tool '{name}' finished.)"


def _run_in_sandbox(ctx: auth.AuthContext, tool: dict, arguments: dict) -> str:
    ext = "py" if tool["kind"] == "python" else "sh"
    staging = config.runs_dir(_problem) / "tools" / ctx.agent_key.replace("/", "__")
    staging.mkdir(parents=True, exist_ok=True)
    script = staging / f"{tool['name']}.{ext}"
    script.write_text(tool["code"])

    mount = sandbox.Mount(staging, "/tool", ro=True)
    spath = f"{sandbox.cpath(_backend, mount)}/{script.name}"
    interp = ["python3"] if tool["kind"] == "python" else ["/bin/sh"]
    limits = tool_limits()
    spec = sandbox.SandboxSpec(
        backend=_backend,
        image=sandbox.prepare_image(_backend, runconfig.get(_problem, "platform_image"),
                                    config.repo_root() / "runs" / ".images"),
        command=[*interp, spath, json.dumps(arguments)],
        mounts=[mount],
        env={
            "HOME": "/tmp",
            "AGENTDISCOVER_SERVICE": config.service_url(),
            "AGENTDISCOVER_TOKEN": ctx.token,
        },
        network="host",   # so the tool can call the MCP service back
        limits=limits,
        name=f"agentdiscover-tool-{tool['name'][:20]}-{ctx.iteration}",
        extra_args=(["--tmpfs", "/tmp"] if _backend in ("docker", "podman") else []),
    )
    result = sandbox.run(spec, wall_seconds=limits.wall_seconds)
    if result.timed_out:
        return f"meta-written tool '{tool['name']}' timed out after {limits.wall_seconds}s."
    if result.returncode != 0:
        return (f"meta-written tool '{tool['name']}' failed (exit {result.returncode}):\n"
                + (result.stderr or result.stdout)[-1000:])
    return result.stdout


def _record_use(ctx: auth.AuthContext, name: str) -> None:
    db.run_write(
        """
        MATCH (s:Session {id: $sid}), (t:Tool {name: $name, branch: $branch})
        MERGE (s)-[:USED_TOOL]->(t)
        """,
        sid=ctx.session_id, name=name, branch=ctx.agent_key,
    )
