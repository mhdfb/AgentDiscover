"""Exposure edges: what a session actually fetched and read, parsed from its log by
code, never written by the agent. Best-effort: only genome hashes in results are seen.

  (:Session)-[:RETRIEVED {via}]->(:Candidate)   via = query (read_graph) | tool
  (:Session)-[:READ]->(:Resource)               provided files the session opened
"""
from __future__ import annotations

import json
import re

from .. import db

_HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")

_BUILTIN_WRITE_TOOLS = {"submit_candidate", "add_idea", "write_lesson"}


def _mcp_tool_name(full: str) -> str:
    """Bare tool name: claude logs mcp__<server>__<tool>, codex <server>.<tool>."""
    for sep in ("__", "."):
        if sep in full:
            full = full.split(sep)[-1]
    return full


def _claude_blocks(raw: str):
    """Yield (kind, tool_name, text) for tool_use / tool_result blocks."""
    id_to_tool: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        msg = d.get("message", {})
        for block in msg.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if d.get("type") == "assistant" and block.get("type") == "tool_use":
                tool = _mcp_tool_name(block.get("name", ""))
                id_to_tool[block.get("id", "")] = tool
                yield ("use", tool, json.dumps(block.get("input", {}), default=str))
            elif d.get("type") == "user" and block.get("type") == "tool_result":
                tool = id_to_tool.get(block.get("tool_use_id", ""), "")
                yield ("result", tool, json.dumps(block.get("content", ""), default=str))


def _codex_blocks(raw: str):
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        item = d.get("item", {})
        if d.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        itype = item.get("type", "")
        if itype == "mcp_tool_call":
            tool = _mcp_tool_name(item.get("tool", "") or item.get("name", ""))
            yield ("use", tool, json.dumps(item.get("arguments", ""), default=str))
            yield ("result", tool, json.dumps(item.get("result", ""), default=str))
        elif itype == "command_execution":
            yield ("use", "shell", json.dumps(item.get("command", ""), default=str))


def extract(raw: str, harness: str, resource_names: set[str]) -> tuple[dict[str, str], set[str]]:
    """Returns ({genome_hash: via}, {resource_name})."""
    blocks = _claude_blocks(raw) if harness == "claude" else \
        _codex_blocks(raw) if harness == "codex" else iter(())

    retrieved: dict[str, str] = {}
    read: set[str] = set()
    for kind, tool, text in blocks:
        if kind == "result" and tool and tool not in _BUILTIN_WRITE_TOOLS:
            via = "query" if tool == "read_graph" else "tool"
            for h in _HASH_RE.findall(text):
                retrieved.setdefault(h, via)
        for name in resource_names:
            if name in text:
                read.add(name)
    return retrieved, read


def write_edges(session_id: str, retrieved: dict[str, str], read: set[str]) -> None:
    for h, via in retrieved.items():
        db.run_write(
            """
            MATCH (s:Session {id: $sid}), (c:Candidate {genome_hash: $h})
            MERGE (s)-[r:RETRIEVED]->(c) SET r.via = $via
            """,
            sid=session_id, h=h, via=via,
        )
    for name in read:
        db.run_write(
            """
            MATCH (s:Session {id: $sid}), (r:Resource {name: $n})
            MERGE (s)-[:READ]->(r)
            """,
            sid=session_id, n=name,
        )
