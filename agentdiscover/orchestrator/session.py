"""Session open/close — one open and one close per agent per iteration.

open: render the instruction file and briefing, create the Session node, write the MCP
config. close: store the agent's reasoning and exposure edges, stamp finished_at.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
from pathlib import Path

from .. import config, db, runconfig
from . import briefing, exposure, resources, runlog
from .launch import web_rule

# ---------------------------------------------------------------------------
# Instruction file
# ---------------------------------------------------------------------------

BIAS_MARKER = "## Strategy biases"
INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md")


def _problem_text(problem: str) -> str:
    """PROBLEM.md up to the bias marker; the rest is not shown to the agent."""
    md = (config.problem_dir(problem) / "PROBLEM.md").read_text()
    return md.split(BIAS_MARKER, 1)[0].strip()


def current_guidance(agent_key: str) -> tuple[str, str]:
    rows = db.run_read(
        """
        MATCH (a:Agent {key: $k})-[:HAS_GUIDANCE]->(g:Guidance)
        WHERE NOT ()-[:SUPERSEDES]->(g)
        RETURN g.id AS id, g.strategy AS strategy
        """,
        k=agent_key,
    )
    if not rows:
        raise SystemExit(f"agent {agent_key} has no guidance — setup_agent seeds one; run setup first")
    return rows[0]["id"], rows[0]["strategy"]


def _first_doc_line(path: Path) -> str:
    """First non-empty line of a docstring or comment — the one-line summary.
    Agents may leave binary files in tools/; an unreadable file gets no doc line."""
    try:
        for ln in path.read_text(errors="strict").splitlines()[:15]:
            t = ln.strip().lstrip("#").strip().strip('\'"')
            if t and not t.startswith(("import ", "from ", "!", "-*-")):
                return t[:110]
    except (OSError, UnicodeDecodeError):
        pass
    return ""


def _folder_listing(worktree: Path) -> str:
    """The live contents of tools/ and skills/, rendered into the instruction file."""
    parts = []
    for folder, label in (("tools", "Reusable code left by earlier sessions of your "
                                    "branch — read before rebuilding anything"),
                          ("skills", "Notes from the meta agent on how your branch "
                                     "searches — worth a look before you plan")):
        files = sorted(f for f in (worktree / folder).glob("*") if f.is_file()
                       and f.name != ".gitkeep")
        if not files:
            continue
        lines = [f"`{folder}/` — {label}:"]
        for f in files:
            doc = _first_doc_line(f)
            lines.append(f"- `{folder}/{f.name}`" + (f" — {doc}" if doc else ""))
        parts.append("\n".join(lines))
    return "\n\n".join(parts) if parts else "(both folders are empty so far)"


def render_instructions(problem: str, worktree: Path, harness: str,
                        strategy: str) -> None:
    """Static sections first, the strategy last, so the model provider's prompt cache
    can reuse the unchanging prefix between iterations."""
    template = (config.repo_root() / "harness-instructions" / "instructions.md").read_text()
    text = (template
            .replace("{{PROBLEM}}", _problem_text(problem))
            .replace("{{SCHEMA}}", config.schema_path().read_text().strip())
            .replace("{{FOLDER_LISTING}}", _folder_listing(worktree))
            .replace("{{STRATEGY}}", strategy.strip()))
    target = "CLAUDE.md" if harness == "claude" else "AGENTS.md"
    for name in INSTRUCTION_FILES:
        if name != target:
            (worktree / name).unlink(missing_ok=True)
    (worktree / target).write_text(text)


# ---------------------------------------------------------------------------
# MCP config (per harness: Claude reads JSON, Codex TOML)
# ---------------------------------------------------------------------------


def write_mcp_config(worktree: Path, harness: str, token: str) -> None:
    url = f"{config.service_url()}/mcp/{token}"
    if harness != "codex":
        # claude reads .mcp.json from the cwd; everything but codex gets the JSON form.
        (worktree / ".mcp.json").write_text(json.dumps({
            "mcpServers": {"agentdiscover": {"type": "http", "url": url}}
        }, indent=2) + "\n")
    else:
        codex_home = worktree / ".codex"
        codex_home.mkdir(exist_ok=True)
        (codex_home / "config.toml").write_text(
            "[mcp_servers.agentdiscover]\n"
            f'url = "{url}"\n'
            "startup_timeout_sec = 30\n"
            # generous: submit_candidate blocks for the whole sandboxed evaluation
            "tool_timeout_sec = 900\n"
        )


# ---------------------------------------------------------------------------
# open / close
# ---------------------------------------------------------------------------


def open_session(problem: str, user: str, name: str, harness: str, model: str,
                 iteration: int, n_agents: int,
                 rel_iteration: int = 0, total_iterations: int = 0) -> dict:
    agent_key = config.agent_key(user, name)
    worktree = config.worktree_path(problem, user, name)
    guidance_id, strategy = current_guidance(agent_key)

    render_instructions(problem, worktree, harness, strategy)
    brief = briefing.build(agent_key, iteration, n_agents, problem,
                           rel_iteration, total_iterations)

    token = secrets.token_urlsafe(32)
    session_id = f"{agent_key}/{iteration}"
    db.run_write(
        """
        CREATE (s:Session {
            id: $sid, iteration: $iteration, model: $model,
            briefing: $briefing, started_at: datetime(), token_hash: $th })
        WITH s
        MATCH (a:Agent {key: $agent_key})   MERGE (a)-[:RAN]->(s)
        WITH s
        MATCH (g:Guidance {id: $guidance})  MERGE (s)-[:UNDER_GUIDANCE]->(g)
        """,
        sid=session_id, iteration=iteration, model=model, briefing=brief,
        th=hashlib.sha256(token.encode()).hexdigest(),
        agent_key=agent_key, guidance=guidance_id,
    )
    write_mcp_config(worktree, harness, token)

    # Same source the service enforces, so the prompt never asks for more than it allows.
    proposals = int(runconfig.get(problem, "proposals"))
    # Say "submit", never "propose": a reply that calls no tool ends the session, so
    # the prompt must name the tool call and state the cost of answering in prose.
    prompt = (f"Submit up to {proposals} candidates with the submit_candidate tool.\n\n"
              "Do NOT reply with a plan or a summary of what you intend to do: a reply "
              "that calls no tool ends your session immediately, and a session that "
              "submits nothing is wasted. Explore as much as you need, then edit "
              "solution.py and submit.\n\n"
              f"{web_rule(bool(runconfig.get(problem, 'web_access')))}\n\n"
              f"{brief}")
    (worktree / ".prompt.txt").write_text(prompt)

    jsonl = worktree / "agent-logs" / f"iter-{iteration}.jsonl"
    return {"session_id": session_id, "jsonl": str(jsonl),
            "prompt_file": str(worktree / ".prompt.txt"), "guidance": guidance_id}


# --- thinking_text extraction ---


def _extract_claude(text: str) -> str:
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "assistant":
            continue
        for block in d.get("message", {}).get("content", []):
            if block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
    return "\n\n".join(p for p in parts if p)


def _extract_codex(text: str) -> str:
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "item.completed":
            continue
        item = d.get("item", {})
        if item.get("type") in ("agent_message", "reasoning"):
            parts.append(item.get("text", ""))
    return "\n\n".join(p for p in parts if p)


def extract_thinking_text(jsonl_path: Path, harness: str) -> str:
    if not jsonl_path.exists():
        return ""
    raw = jsonl_path.read_text()
    if harness == "claude":
        return _extract_claude(raw)
    if harness == "codex":
        return _extract_codex(raw)
    return raw  # unknown harness: already plain text


def commit_worktree(worktree: Path, message: str) -> None:
    """Version the worktree after every session and meta pass. Best effort: a commit
    that fails (nothing changed, or no git) must never fail the run."""
    try:
        subprocess.run(["git", "-C", str(worktree), "add", "-A"],
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-q", "-m", message],
                       check=False, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        pass


def close_session(problem: str, session_id: str, jsonl_path: str, harness: str,
                  exit_status: str) -> None:
    # When `open` failed the launcher has no session id: log and return, do not crash.
    if not session_id or not jsonl_path:
        runlog.append(problem, f"[close] skipped — no session to close "
                               f"(id={session_id!r}, status={exit_status})")
        return
    path = Path(jsonl_path)
    thinking = extract_thinking_text(path, harness)
    raw = path.read_text() if path.exists() else ""
    retrieved, read = exposure.extract(raw, harness, resources.resource_names(problem))

    db.run_write(
        """
        MATCH (s:Session {id: $sid})
        SET s.thinking_text = $thinking,
            s.jsonl_path    = $jsonl,
            s.finished_at   = datetime()
        """,
        sid=session_id, thinking=thinking, jsonl=jsonl_path,
    )
    exposure.write_edges(session_id, retrieved, read)
    commit_worktree(path.parent.parent, f"session {session_id} ({exit_status})")

    stats = db.run_read(
        """
        MATCH (s:Session {id: $sid})
        OPTIONAL MATCH (s)-[:PRODUCED]->(c:Candidate)
        RETURN count(c) AS candidates, max(c.fitness) AS best,
               s.lesson_learned IS NOT NULL AS has_lesson
        """,
        sid=session_id,
    )[0]
    line = (f"[{session_id}] candidates={stats['candidates']} "
            f"best={stats['best'] if stats['best'] is not None else '-'} "
            f"retrieved={len(retrieved)} read={len(read)} status={exit_status}")
    if not stats["has_lesson"]:
        line += " lesson=MISSING"
    runlog.append(problem, line)
    print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.session")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("open")
    p.add_argument("--problem", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--harness", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--iteration", type=int, required=True)
    # Accepted and ignored; run.sh still passes it.
    p.add_argument("--bias-idx", type=int, default=0)
    p.add_argument("--num-agents", type=int, required=True)
    # Position within this invocation; 0 means unknown.
    p.add_argument("--rel-iteration", type=int, default=0)
    p.add_argument("--total-iterations", type=int, default=0)

    p = sub.add_parser("close")
    p.add_argument("--problem", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--harness", required=True)
    p.add_argument("--status", default="ok")

    args = parser.parse_args()
    if args.cmd == "open":
        out = open_session(args.problem, args.user, args.name, args.harness,
                           args.model, args.iteration, args.num_agents,
                           args.rel_iteration, args.total_iterations)
        for k, v in out.items():
            print(f"{k}={v}")
    else:
        close_session(args.problem, args.session_id, args.jsonl, args.harness, args.status)


if __name__ == "__main__":
    main()
