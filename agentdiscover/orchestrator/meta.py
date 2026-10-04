"""The meta pass: runs once per block, after every agent finished META_EVERY iterations.
The meta agent is untrusted: its sandbox gets a scratch home (/work), each branch's
agent-logs read-only, each branch's skills/ writable, and a meta-scoped MCP token."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path

from .. import config, db, runconfig, sandbox
from . import quota, runlog
from .launch import (AGENT_SETTINGS, copy_host_auth, effort_args, last_session_uuid,
                     web_args, web_rule)
from .session import BIAS_MARKER, write_mcp_config

META_RESUME_PROMPT = ("Your meta pass was interrupted by the API's usage window running "
                      "out; it has reset. Continue exactly where you left off — the "
                      "lessons, tools and strategies you already wrote are recorded.")


# ---------------------------------------------------------------------------
# Briefing: fixed queries that orient the pass
# ---------------------------------------------------------------------------


def _block_stats(agent_key: str, lo: int, hi: int) -> dict:
    rows = db.run_read(
        """
        MATCH (a:Agent {key: $k})-[:RAN]->(s:Session)
        WHERE s.iteration > $lo AND s.iteration <= $hi
        OPTIONAL MATCH (s)-[:PRODUCED]->(c:Candidate)
        RETURN count(DISTINCT s) AS sessions, count(c) AS candidates,
               max(c.fitness) AS best, avg(c.fitness) AS mean
        """,
        k=agent_key, lo=lo, hi=hi,
    )
    return rows[0] if rows else {"sessions": 0, "candidates": 0, "best": None, "mean": None}


def build_briefing(branches: list[str], meta_every: int) -> str:
    def fmt(x):
        return "-" if x is None else f"{x:.4g}"

    lines = []
    for key in branches:
        top = db.run_read(
            "MATCH (:Agent {key: $k})-[:RAN]->(s:Session) RETURN coalesce(max(s.iteration), 0) AS m",
            k=key,
        )[0]["m"]
        cur = _block_stats(key, top - meta_every, top)
        prev = _block_stats(key, top - 2 * meta_every, top - meta_every)
        lines.append(
            f"- {key}: this block {cur['sessions']} sessions, {cur['candidates']} candidates, "
            f"best {fmt(cur['best'])}, mean {fmt(cur['mean'])} "
            f"(previous block: best {fmt(prev['best'])}, mean {fmt(prev['mean'])})"
        )
    return "\n".join(lines)


def _branch_paths(problem: str, user: str, name: str, backend: str) -> tuple[str, str]:
    """(logs_path, skills_path) as the meta agent sees them: container paths under a
    container backend, host paths under `none`."""
    wt = config.worktree_path(problem, user, name)
    if backend == "none":
        return str(wt / "agent-logs"), str(wt / "skills")
    return f"/agent-logs/{wt.name}", f"/skills/{wt.name}"


def render_meta_md(problem: str, user: str, agent_names: list[str],
                   meta_every: int, backend: str = "none") -> Path:
    template = (config.repo_root() / "harness-instructions" / "meta.md").read_text()
    problem_md = (config.problem_dir(problem) / "PROBLEM.md").read_text()
    problem_text = problem_md.split(BIAS_MARKER, 1)[0].strip()
    branches = [config.agent_key(user, n) for n in agent_names]
    def _line(n: str) -> str:
        logs, skills = _branch_paths(problem, user, n, backend)
        return (f"- `{config.agent_key(user, n)}` — session logs (read-only): `{logs}/` — "
                f"write its skills to: `{skills}/`")
    branch_lines = "\n".join(_line(n) for n in agent_names)
    text = (template
            .replace("{{PROBLEM}}", problem_text)
            .replace("{{SCHEMA}}", config.schema_path().read_text().strip())
            .replace("{{BRANCHES}}", branch_lines)
            .replace("{{BRIEFING}}", build_briefing(branches, meta_every))
            .replace("{{META_EVERY}}", str(meta_every))
            .replace("{{PROPOSALS}}", str(runconfig.get(problem, "proposals"))))
    out = config.runs_dir(problem) / "meta.md"
    out.write_text(text)
    return out


# ---------------------------------------------------------------------------
# The pass itself
# ---------------------------------------------------------------------------


def _guidance_heads(branches: list[str]) -> dict[str, str]:
    heads = {}
    for key in branches:
        rows = db.run_read(
            """
            MATCH (:Agent {key: $k})-[:HAS_GUIDANCE]->(g:Guidance)
            WHERE NOT ()-[:SUPERSEDES]->(g)
            RETURN g.id AS id
            """,
            k=key,
        )
        heads[key] = rows[0]["id"] if rows else "(none)"
    return heads


def _tool_count() -> int:
    return db.run_read("MATCH (t:Tool) RETURN count(t) AS n")[0]["n"]


def meta_pass(problem: str, backend: str, user: str, harness: str, model: str,
              agent_names: list[str], meta_every: int) -> None:
    branches = [config.agent_key(user, n) for n in agent_names]
    meta_md = render_meta_md(problem, user, agent_names, meta_every, backend)

    rd = config.runs_dir(problem)
    meta_home = rd / "meta-home"
    (meta_home / ".home").mkdir(parents=True, exist_ok=True)
    (meta_home / "agent-logs").mkdir(exist_ok=True)  # worktree shape for copy_host_auth

    token = secrets.token_urlsafe(32)
    pass_id = f"meta-{int(time.time())}"
    meta_file = rd / "meta-pass.json"
    meta_file.write_text(json.dumps({
        "token_hash": hashlib.sha256(token.encode()).hexdigest(), "id": pass_id,
    }))
    write_mcp_config(meta_home, harness, token)

    mounts = [sandbox.Mount(meta_home, "/work", ro=False)]
    for name in agent_names:
        wt = config.worktree_path(problem, user, name)
        logs = wt / "agent-logs"
        if logs.is_dir():
            mounts.append(sandbox.Mount(logs, f"/agent-logs/{wt.name}", ro=True))
        # Only skills/ is mounted rw: the meta agent can write nothing else in a worktree.
        skills = wt / "skills"
        skills.mkdir(exist_ok=True)
        mounts.append(sandbox.Mount(skills, f"/skills/{wt.name}", ro=False))

    work = sandbox.cpath(backend, mounts[0])
    system_prompt = meta_md.read_text()
    prompt = ("Run the meta pass now. Review this block's sessions and candidates "
              "through read_graph and the mounted agent-logs, then write each "
              "branch's meta lessons, its skills files, any saved queries, and its "
              "strategy, as your instructions describe. "
              + web_rule(bool(runconfig.get(problem, "web_access"))))
    web = web_args(harness, bool(runconfig.get(problem, "web_access")))
    if harness == "claude":
        argv = ["claude", "-p", "--model", model, "--dangerously-skip-permissions",
                "--mcp-config", f"{work}/.mcp.json", "--strict-mcp-config",
                "--settings", AGENT_SETTINGS,   # no auto-memory: see launch.py
                "--append-system-prompt", system_prompt, *web, *effort_args(harness),
                "--output-format", "stream-json", "--verbose", prompt]
    elif harness == "codex":
        argv = ["codex", "exec", "--model", model, *web, *effort_args(harness),
                "--dangerously-bypass-approvals-and-sandbox", "--json",
                f"{system_prompt}\n\n{prompt}"]
    elif harness == "mock":
        import shutil

        shutil.copy(config.repo_root() / "tests" / "mock_meta.py",
                    meta_home / ".mock_meta.py")
        argv = ["python3", f"{work}/.mock_meta.py", prompt]
    else:
        raise SystemExit(f"unknown meta harness {harness!r}")

    import os

    env = {
        "AGENTDISCOVER_SERVICE": config.service_url(),
        "MCP_TIMEOUT": "60000",
        "MCP_TOOL_TIMEOUT": "300000",
    }
    if harness == "codex":
        env["CODEX_HOME"] = f"{work}/.codex"
        if os.environ.get("OPENAI_API_KEY"):
            env["CODEX_API_KEY"] = os.environ["OPENAI_API_KEY"]   # see launch.run_agent
    if harness == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

    heads_before = _guidance_heads(branches)
    tools_before = _tool_count()

    image = sandbox.prepare_image(backend, runconfig.get(problem, "platform_image"),
                                  rd.parent / ".images")
    wall = int(runconfig.get(problem, "meta_timeout"))
    (rd / "meta-logs").mkdir(exist_ok=True)
    log_path = rd / "meta-logs" / f"{pass_id}.jsonl"

    # Resume a pass cut off by the API usage window after the reset (bounded).
    resume, resumes = None, 0
    while True:
        command = argv if resume is None else \
            argv[:-1] + ["--resume", resume, META_RESUME_PROMPT]  # claude: prompt is last
        spec = sandbox.SandboxSpec(
            backend=backend, image=image, command=command,
            mounts=mounts + copy_host_auth(meta_home, harness, backend,
                                           keep_state=bool(resume)),
            env=env, workdir=work, network="host", limits=None,
            name=f"agentdiscover-meta-{problem}", home_dir=meta_home / ".home",
        )
        with log_path.open("ab") as log:
            result = sandbox.run(spec, wall_seconds=wall, capture=False,
                                 stdout_file=log, stderr_file=log)
        if harness != "claude" or resumes >= 5:
            break
        try:
            rc = quota.after_session(problem, log_path, f"meta pass {pass_id}")
        except Exception as e:  # noqa: BLE001 — a broken check must never block a run
            print(f"[quota] check error ({e}) — proceeding")
            break
        resume = last_session_uuid(log_path)
        if rc != 3 or resume is None:
            break
        resumes += 1
        runlog.append(problem, f"[meta {pass_id}] resuming after the quota reset "
                               f"(resume {resumes})")
    meta_file.unlink(missing_ok=True)  # the meta token dies with the pass

    heads_after = _guidance_heads(branches)
    changed = [k for k in branches if heads_after[k] != heads_before[k]]
    kept = [k for k in branches if k not in changed]
    status = "timeout" if result.timed_out else \
        ("ok" if result.returncode == 0 else f"exit-{result.returncode}")
    line = (f"[meta {pass_id}] status={status} tools+{_tool_count() - tools_before} "
            f"changed={','.join(changed) or '-'} kept={','.join(kept) or '-'} "
            f"guidance={json.dumps(heads_after)}")
    runlog.append(problem, line)
    print(line)

    # Commit what the pass left in each branch's skills/; agents never run git.
    from .session import commit_worktree

    for name in agent_names:
        commit_worktree(config.worktree_path(problem, user, name),
                        f"meta pass {pass_id}: skills")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.meta")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--agents", required=True,
                        help='space-separated "name:harness:model" entries')
    parser.add_argument("--meta-every", type=int, default=50)
    args = parser.parse_args()
    names = [entry.split(":")[0] for entry in args.agents.split()]
    meta_pass(args.problem, args.backend, args.user, args.harness, args.model,
              names, args.meta_every)


if __name__ == "__main__":
    main()
