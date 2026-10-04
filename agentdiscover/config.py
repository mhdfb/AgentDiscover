"""Central configuration: environment, paths, ports, and per-problem limits.

Read only on the trusted side (orchestrator + MCP service); agents never import this.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """The repository root: AGENTDISCOVER_ROOT if set, else the parent of this package."""
    env = os.environ.get("AGENTDISCOVER_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def problem_dir(problem: str) -> Path:
    return repo_root() / "problems" / problem


def runs_dir(problem: str) -> Path:
    return repo_root() / "runs" / problem


def worktree_path(problem: str, user: str, agent_name: str) -> Path:
    # "__" separates user and agent name, so neither may contain it.
    return runs_dir(problem) / f"agent-{user}__{agent_name}"


def agent_key(user: str, agent_name: str) -> str:
    return f"{user}/{agent_name}"


def schema_path() -> Path:
    return repo_root() / "schema.cypher"


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

NEO4J_BOLT_PORT = int(os.environ.get("NEO4J_PORT", "7687"))
NEO4J_HTTP_PORT = int(os.environ.get("NEO4J_HTTP_PORT", "7474"))
SERVICE_PORT = int(os.environ.get("AGENTDISCOVER_PORT", "8900"))


def neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", f"bolt://127.0.0.1:{NEO4J_BOLT_PORT}")


def neo4j_auth() -> tuple[str, str]:
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise SystemExit(
            "NEO4J_PASSWORD is not set. Put it in .env (see .env.example) — the service "
            "holds it; agents never do."
        )
    return ("neo4j", password)


def service_url() -> str:
    """Base URL of the MCP service as seen from inside a sandbox (host networking)."""
    return os.environ.get("AGENTDISCOVER_SERVICE", f"http://127.0.0.1:{SERVICE_PORT}")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# The platform image is a per-run setting; see runconfig.py.

# Official Neo4j community image, pinned to the 5.26 LTS line.
NEO4J_IMAGE = os.environ.get("NEO4J_IMAGE", "docker.io/library/neo4j:5.26-community")


def eval_image_ref(problem: str) -> str:
    """The published evaluator image for a problem (used by pull-only backends)."""
    default = f"ghcr.io/mhdfb/agentdiscover-eval-{problem}:latest"
    return os.environ.get("AGENTDISCOVER_EVAL_IMAGE", default)


# ---------------------------------------------------------------------------
# Limits — the cage around untrusted code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    wall_seconds: int = 300   # whole-evaluation wall clock
    mem_gb: float = 4.0       # address-space / container memory cap
    cpus: float = 2.0         # container cpu cap; cpu-seconds = wall * cpus for ulimit
    nproc: int = 256          # max processes
    fsize_mb: int = 256       # max size of any file written

    # Seconds the genome may spend inside solve(); wall_seconds caps the whole
    # evaluation. None means the evaluator bundle keeps its own default.
    solve_seconds: int | None = None

    # The evaluation needs the host's NVIDIA GPU. No `ulimit -v` is applied then: a
    # CUDA context reserves far more virtual address space than any sane mem_gb.
    gpu: bool = False


# Session, meta-pass and tool wall clocks are per-run settings; see runconfig.py.

READ_ROW_LIMIT = 200          # read_graph result cap


def load_problem_config(problem: str) -> dict:
    """Parse the optional problems/<p>/problem.toml ({} if absent). Recognized keys:
    [evaluator] image (a prebuilt image ref); [limits] wall_seconds, mem_gb, cpus,
    nproc, fsize_mb, solve_seconds, gpu."""
    path = problem_dir(problem) / "problem.toml"
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def problem_limits(problem: str) -> Limits:
    cfg = load_problem_config(problem).get("limits", {})
    base = Limits()
    return Limits(
        wall_seconds=int(cfg.get("wall_seconds", base.wall_seconds)),
        mem_gb=float(cfg.get("mem_gb", base.mem_gb)),
        cpus=float(cfg.get("cpus", base.cpus)),
        nproc=int(cfg.get("nproc", base.nproc)),
        solve_seconds=(int(cfg["solve_seconds"]) if "solve_seconds" in cfg else None),
        fsize_mb=int(cfg.get("fsize_mb", base.fsize_mb)),
        gpu=bool(cfg.get("gpu", base.gpu)),
    )
