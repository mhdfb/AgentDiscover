"""Per-agent setup: find or create the agent, seed its guidance, create its worktree.

The worktree is a fresh git repository: the main repository's history must not be
reachable from it, or the agent could read the evaluator out of git.
Prints one line for run.sh:  start_iteration=<n> worktree=<created|reused>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .. import config, db


def seed_strategy_text() -> str:
    path = config.repo_root() / "harness-instructions" / "seed_strategy.md"
    return path.read_text()


def register_agent(user: str, name: str, worktree: Path) -> str:
    """Find or create the agent (MERGE). Returns its key."""
    key = config.agent_key(user, name)
    db.run_write(
        """
        MERGE (u:User {name: $user})
        MERGE (a:Agent {key: $key})
          ON CREATE SET a.name = $name
        SET a.worktree = $worktree
        MERGE (u)-[:OWNS]->(a)
        """,
        user=user, key=key, name=name, worktree=str(worktree),
    )
    return key


def ensure_seed_guidance(agent_key: str) -> None:
    """Create the seed guidance node if the agent has none; session open needs one."""
    has = db.run_read(
        "MATCH (:Agent {key: $k})-[:HAS_GUIDANCE]->(g:Guidance) RETURN count(g) AS n",
        k=agent_key,
    )
    if has and has[0]["n"] > 0:
        return
    db.run_write(
        """
        MATCH (a:Agent {key: $k})
        CREATE (g:Guidance {id: $id, strategy: $strategy,
                            diff_summary: "seed guidance; no meta pass has run yet"})
        MERGE (a)-[:HAS_GUIDANCE]->(g)
        """,
        k=agent_key, id=f"{agent_key}/g0", strategy=seed_strategy_text(),
    )


def start_iteration(agent_key: str) -> int:
    """Next iteration number, from the database rather than the filesystem."""
    rows = db.run_read(
        """
        MATCH (a:Agent {key: $k})-[:RAN]->(s:Session)
        RETURN coalesce(max(s.iteration), 0) + 1 AS start_iteration
        """,
        k=agent_key,
    )
    return rows[0]["start_iteration"] if rows else 1


def ensure_worktree(problem: str, worktree: Path) -> bool:
    """Create the worktree if absent; reuse it otherwise. Returns True if created."""
    if worktree.exists():
        for sub in ("agent-logs", ".home", "tools", "skills"):
            (worktree / sub).mkdir(exist_ok=True)
        return False
    worktree.mkdir(parents=True)
    shutil.copy(config.problem_dir(problem) / "solution.py", worktree / "solution.py")
    (worktree / "agent-logs").mkdir()
    (worktree / ".home").mkdir()
    # tools/: code left by this branch's sessions; skills/: notes from the meta agent.
    (worktree / "tools").mkdir()
    (worktree / "skills").mkdir()
    # .mcp.json carries the per-session token — never into version control.
    (worktree / ".gitignore").write_text(
        "agent-logs/\n.home/\n.briefing.txt\n.prompt.txt\n.mcp.json\n.codex/\n")

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(worktree), *args], check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "agent@agentdiscover.local")
    git("config", "user.name", "agentdiscover agent")
    git("add", "-A")
    git("commit", "-q", "-m", "seed: initial worktree")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.setup_agent")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    worktree = config.worktree_path(args.problem, args.user, args.name)
    key = register_agent(args.user, args.name, worktree)
    ensure_seed_guidance(key)
    created = ensure_worktree(args.problem, worktree)
    start = start_iteration(key)
    print(f"start_iteration={start} worktree={'created' if created else 'reused'}")


if __name__ == "__main__":
    main()
