"""Start one agent session in its sandbox and enforce its wall clock. The container gets
its worktree (rw, /work), the problem's resources (ro, /resources), at most one LLM key
and the MCP config. The outcome is printed as `status=...`."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .. import config, runconfig, sandbox
# Reuse the evaluator's installer so the agent and the evaluation import the same build.
from ..server.evaluator import _ensure_data, _ensure_deps
from .resources import SUPPORT_CONTAINER_DIR, resources_dir, support_dir

# Same path as in the evaluation sandbox.
DEPS_CONTAINER_DIR = "/deps"


def agent_wants_deps(problem: str) -> bool:
    """Whether the evaluator's dependencies are mounted into the agent's sandbox too.
    Opt-in via `[agent] mount_deps = true` in problem.toml."""
    return bool(config.load_problem_config(problem).get("agent", {}).get("mount_deps", False))


def agent_wants_data(problem: str) -> bool:
    """Whether the fetched data set is mounted read-only into the agent's sandbox too
    (/data, AGENTDISCOVER_DATA_DIR). Opt-in via `[agent] mount_data = true`."""
    return bool(config.load_problem_config(problem).get("agent", {}).get("mount_data", False))

# Claude Code auto-memory is on only under a container backend, where the operator's
# projects/ is shadowed. Off for the meta pass and for backend=none.
AGENT_SETTINGS = json.dumps({"autoMemoryEnabled": False})
MEMORY_SETTINGS = json.dumps({"autoMemoryEnabled": True})

# Entries of the operator's ~/.claude an agent may see. The directory is mounted whole
# (the harness refreshes .credentials.json in place); all other entries are shadowed.
_CLAUDE_STATE_KEEP = frozenset({
    ".credentials.json", "settings.json", "settings.local.json",
    "plugins", "statsig", "cache", "ide", "mcp-needs-auth-cache.json", ".last-cleanup",
})


RESUME_PROMPT = ("Your session was interrupted by the API's usage window running out; it "
                 "has reset. Continue exactly where you left off — the candidates you "
                 "already submitted are recorded.")


def web_args(harness: str, web: bool) -> list[str]:
    """Harness flags that grant or remove the built-in web tools (runconfig web_access).
    Only the tools are removed: Bash keeps network access. The claude flag is variadic,
    so another option must follow it, never the prompt. codex search is on by default."""
    if harness == "claude":
        return [] if web else ["--disallowedTools", "WebSearch,WebFetch"]
    if harness == "codex":
        return ["-c", f'web_search="{"live" if web else "disabled"}"']
    return []


def web_rule(web: bool) -> str:
    """The prompt's sentence on internet use. With web_access off it is the only guard
    on Bash, which keeps network access (see web_args)."""
    if web:
        return "You may search the internet if you want."
    return ("Never reach out to the internet in any way (web search, curl, wget, "
            "git clone, or otherwise) to look up other results for this problem.")


def effort_args(harness: str) -> list[str]:
    """Reasoning effort from AGENTDISCOVER_EFFORT: `--effort` for claude (low|medium|high|
    xhigh|max), `model_reasoning_effort` for codex (minimal|low|medium|high|xhigh).
    Unset means the harness's own default."""
    effort = (os.environ.get("AGENTDISCOVER_EFFORT") or "").strip()
    if harness == "claude" and effort:
        return ["--effort", effort]
    if harness == "codex" and effort:
        return ["-c", f'model_reasoning_effort="{effort}"']
    return []


def harness_argv(harness: str, model: str, prompt: str, work: str = "/work",
                 resume: str | None = None, web: bool = False,
                 memory: bool = False) -> list[str]:
    if harness == "claude":
        # --strict-mcp-config: load only this session's MCP server, no user-level ones.
        argv = ["claude", "-p", "--model", model, "--dangerously-skip-permissions",
                "--mcp-config", f"{work}/.mcp.json", "--strict-mcp-config",
                "--settings", MEMORY_SETTINGS if memory else AGENT_SETTINGS,
                *web_args(harness, web), *effort_args(harness),
                "--output-format", "stream-json", "--verbose"]
        if resume:
            return argv + ["--resume", resume, RESUME_PROMPT]
        return argv + [prompt]
    if resume:
        raise SystemExit(f"harness {harness!r} cannot resume a session")
    if harness == "codex":
        # codex silently truncates AGENTS.md at project_doc_max_bytes (32 KiB by default);
        # the strategy sits at the end of the file, so the limit is raised.
        return ["codex", "exec", "--model", model, *web_args(harness, web),
                *effort_args(harness), "-c", "project_doc_max_bytes=131072",
                "--dangerously-bypass-approvals-and-sandbox", "--json", prompt]
    if harness == "mock":
        return ["python3", f"{work}/.mock_agent.py", prompt]
    raise SystemExit(f"unknown harness {harness!r} (supported: claude, codex)")


def copy_host_auth(worktree: Path, harness: str, backend: str,
                   keep_state: bool = False) -> list[sandbox.Mount]:
    """Bring the host's harness login into the sandbox without mounting $HOME; returns
    the extra mounts. `worktree` is the state dir mounted at /work; its .home/ becomes
    the container's $HOME. keep_state keeps the previous launch's shadowed harness state
    (needed to resume). An API key in the environment wins over the machine's login."""
    mounts: list[sandbox.Mount] = []
    home = Path.home()
    cont_home = sandbox.home_path(backend, worktree / ".home")
    if harness == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        if (home / ".claude").is_dir():
            if backend == "none":
                # No container: a symlink stands in for the mount (no isolation here).
                # Whatever a container run left in the way is replaced.
                link = worktree / ".home" / ".claude"
                target = (home / ".claude").resolve()
                if link.is_symlink() and link.resolve() == target:
                    pass
                else:
                    if link.is_symlink() or link.is_file():
                        link.unlink()
                    elif link.is_dir():
                        shutil.rmtree(link)
                    link.symlink_to(home / ".claude")
            else:
                # The mountpoint must exist on the host first: Apptainer does not create
                # a bind destination inside the bound container home.
                (worktree / ".home" / ".claude").mkdir(parents=True, exist_ok=True)
                mounts.append(sandbox.Mount(home / ".claude", f"{cont_home}/.claude", ro=False))
                # Nested binds come AFTER the directory they sit in (see build_argv).
                mounts += _shadow_operator_state(worktree, home / ".claude",
                                                 f"{cont_home}/.claude", keep_state)
        if (home / ".claude.json").is_file():
            shutil.copy(home / ".claude.json", worktree / ".home" / ".claude.json")
    if harness == "codex":
        src = home / ".codex" / "auth.json"
        if os.environ.get("OPENAI_API_KEY"):
            (worktree / ".codex" / "auth.json").unlink(missing_ok=True)
        elif src.is_file():
            (worktree / ".codex").mkdir(exist_ok=True)
            shutil.copy(src, worktree / ".codex" / "auth.json")
    return mounts


def _shadow_operator_state(worktree: Path, host_claude: Path, cont_claude: str,
                           keep: bool = False) -> list[sandbox.Mount]:
    """Hide everything in the operator's ~/.claude except _CLAUDE_STATE_KEEP behind
    blank, writable per-session mountpoints under the worktree's .home. They are emptied
    at every launch unless `keep` (a resumed session needs its transcript); projects/ is
    always kept, since it holds the branch's auto-memory."""
    shadow = worktree / ".home" / ".claude-shadow"
    if not keep and shadow.is_dir():
        for entry in shadow.iterdir():
            if entry.name == "projects":
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
    shadow.mkdir(parents=True, exist_ok=True)
    mounts: list[sandbox.Mount] = []
    for entry in sorted(host_claude.iterdir()):
        if entry.name in _CLAUDE_STATE_KEEP:
            continue
        blank = shadow / entry.name
        if entry.is_dir():
            blank.mkdir(exist_ok=True)
        else:
            blank.touch()
        mounts.append(sandbox.Mount(blank, f"{cont_claude}/{entry.name}", ro=False))
    return mounts


def last_session_uuid(jsonl: Path) -> str | None:
    """The harness's conversation id from the transcript, as `--resume` needs it."""
    try:
        lines = jsonl.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            sid = json.loads(line).get("session_id")
        except ValueError:
            continue
        if sid:
            return sid
    return None


def run_agent(problem: str, backend: str, worktree: Path, harness: str,
              model: str, prompt: str, jsonl: Path, wall_seconds: int,
              resume: str | None = None) -> str:
    image = sandbox.prepare_image(backend, runconfig.get(problem, "platform_image"),
                                  config.repo_root() / "runs" / ".images")

    mounts = [sandbox.Mount(worktree, "/work", ro=False)]
    res_dir = resources_dir(problem)
    if res_dir.is_dir() and any(res_dir.iterdir()):
        mounts.append(sandbox.Mount(res_dir, "/resources", ro=True))
    sup_dir = support_dir(problem)
    has_support = sup_dir.is_dir() and any(sup_dir.iterdir())
    support_mount = None
    if has_support:
        support_mount = sandbox.Mount(sup_dir, SUPPORT_CONTAINER_DIR, ro=True)
        mounts.append(support_mount)

    # Build the problem's declared libraries now: the agent needs them before it submits.
    deps_mount = None
    if agent_wants_deps(problem):
        deps = _ensure_deps(problem, backend, image)
        if deps is None:
            raise SystemExit(
                f"problems/{problem}/problem.toml sets [agent] mount_deps = true, but the "
                "problem has no evaluator/requirements.txt to install (or ships its own "
                "Dockerfile, which carries its dependencies itself). Add the file or drop "
                "the flag — launching without it would give the agent an environment the "
                "evaluator does not have."
            )
        deps_mount = sandbox.Mount(deps, DEPS_CONTAINER_DIR, ro=True)
        mounts.append(deps_mount)
    data_mount = None
    if agent_wants_data(problem):
        data = _ensure_data(problem, backend, image, _ensure_deps(problem, backend, image))
        if data is None:
            raise SystemExit(
                f"problems/{problem}/problem.toml sets [agent] mount_data = true, but the "
                "problem has no evaluator/fetch.sh to fetch a data set with. Add the "
                "script or drop the flag."
            )
        data_mount = sandbox.Mount(data, "/data", ro=True)
        mounts.append(data_mount)

    mounts += copy_host_auth(worktree, harness, backend, keep_state=bool(resume))

    work = sandbox.cpath(backend, mounts[0])
    if harness == "mock":
        shutil.copy(config.repo_root() / "tests" / "mock_agent.py",
                    worktree / ".mock_agent.py")
    env = {
        "AGENTDISCOVER_SERVICE": config.service_url(),
        # submit_candidate blocks for the whole evaluation, so both MCP timeouts must
        # exceed the evaluation's wall_seconds.
        "MCP_TIMEOUT": str((config.problem_limits(problem).wall_seconds + 600) * 1000),
        "MCP_TOOL_TIMEOUT": str((config.problem_limits(problem).wall_seconds + 600) * 1000),
        # Long Bash waits: under `claude -p` an agent that goes idle to wait for a
        # background job ends its session, so one blocking wait must be able to span it.
        "BASH_MAX_TIMEOUT_MS": "3600000",
        "BASH_DEFAULT_TIMEOUT_MS": "600000",
    }
    # Use cpath, not the constants: under the `none` backend the container paths do not
    # exist. Support modules come first so they win over same-named dependency packages.
    pythonpath = [sandbox.cpath(backend, m)
                  for m in (support_mount, deps_mount) if m is not None]
    if pythonpath:
        env["PYTHONPATH"] = ":".join(pythonpath)
    if data_mount is not None:
        env["AGENTDISCOVER_DATA_DIR"] = sandbox.cpath(backend, data_mount)
    if harness == "codex":
        env["CODEX_HOME"] = f"{work}/.codex"
        if os.environ.get("OPENAI_API_KEY"):
            # codex exec reads CODEX_API_KEY and ignores OPENAI_API_KEY.
            env["CODEX_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if harness == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
        env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
    # The session's own GPU (agent_gpus), kept apart from the evaluator's.
    gpu = config.problem_limits(problem).gpu
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(runconfig.get(problem, "agent_gpus"))

    spec = sandbox.SandboxSpec(
        backend=backend,
        image=image,
        command=harness_argv(harness, model, prompt, work, resume=resume,
                             web=bool(runconfig.get(problem, "web_access")),
                             memory=backend != "none"),
        mounts=mounts,
        env=env,
        workdir=work,
        network="host",
        limits=None,           # no resource limits for the agent, only fs/key isolation
        name=f"agentdiscover-agent-{worktree.name}",
        home_dir=worktree / ".home",
        gpu=gpu,
    )
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    # A resumed session appends to the same transcript: one file per iteration.
    with jsonl.open("ab" if resume else "wb") as log:
        result = sandbox.run(spec, wall_seconds=wall_seconds, capture=False,
                             stdout_file=log, stderr_file=log)
    if result.timed_out:
        return "timeout"
    return "ok" if result.returncode == 0 else f"exit-{result.returncode}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.launch")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue this iteration's interrupted conversation")
    args = parser.parse_args()

    worktree = config.worktree_path(args.problem, args.user, args.name)
    prompt = (worktree / ".prompt.txt").read_text()
    jsonl = worktree / "agent-logs" / f"iter-{args.iteration}.jsonl"
    resume = None
    if args.resume:
        resume = last_session_uuid(jsonl)
        if not resume:
            print("status=exit-1")
            raise SystemExit(f"cannot resume: no session id found in {jsonl}")
        print(f"note: resuming conversation {resume}", flush=True)
    # Raise the timeout if needed so the session can fit its own workload.
    configured = int(runconfig.get(args.problem, "agent_timeout"))
    wall = runconfig.required_agent_timeout(args.problem)
    if wall > configured:
        print(f"note: AGENT_TIMEOUT raised {configured}s -> {wall}s to fit "
              f"{runconfig.get(args.problem, 'proposals')} candidates at "
              f"{config.problem_limits(args.problem).solve_seconds}s each", flush=True)
    status = run_agent(args.problem, args.backend, worktree, args.harness,
                       args.model, prompt, jsonl, wall, resume=resume)
    print(f"status={status}")


if __name__ == "__main__":
    main()
