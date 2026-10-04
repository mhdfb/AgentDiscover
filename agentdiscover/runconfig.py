"""Per-run settings: read from the environment once (`SETTINGS`), published to the
database at run start, and read back from there by every process, because the MCP
service is long-lived. Anything needed to reach the database stays in `config.py`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from . import db


@dataclass(frozen=True)
class Setting:
    """One setting: how it is named, where it comes from, and what it means."""
    name: str                 # property name on the :RunConfig node
    env: str                  # environment variable read at publish time
    cast: Callable[[str], Any]
    default: Any
    doc: str


def _flag(raw: str) -> bool:
    """Parse a boolean setting. Strict: an unrecognized value raises ValueError."""
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError("expected one of 1/0, true/false, yes/no, on/off")


# The only declaration of the run's settings; adding one means adding a row here.
SETTINGS: tuple[Setting, ...] = (
    Setting("proposals", "AGENTDISCOVER_PROPOSALS", int, 2,
            "candidates each search session is asked to submit"),
    Setting("meta_every", "META_EVERY", int, 50,
            "iterations between meta passes"),
    Setting("agent_timeout", "AGENT_TIMEOUT", int, 1800,
            "wall-clock seconds for one search session"),
    Setting("meta_timeout", "META_TIMEOUT", int, 1800,
            "wall-clock seconds for one meta pass"),
    Setting("tool_timeout", "AGENTDISCOVER_TOOL_TIMEOUT", int, 120,
            "wall-clock seconds for one meta-written tool call"),
    Setting("platform_image", "AGENTDISCOVER_IMAGE", str, "ghcr.io/mhdfb/agentdiscover:latest",
            "image used for agents, meta-written tools, and Dockerfile-less evaluators"),
    Setting("web_access", "AGENTDISCOVER_WEB_ACCESS", _flag, False,
            "give search agents and meta passes the harness's web search/fetch tools"),
    # GPU problems only. Two different devices keep the agent's own test runs off the
    # GPU whose timings are the score.
    Setting("eval_gpus", "AGENTDISCOVER_EVAL_GPUS", str, "0",
            "CUDA_VISIBLE_DEVICES inside evaluations of a GPU problem"),
    Setting("agent_gpus", "AGENTDISCOVER_AGENT_GPUS", str, "0",
            "CUDA_VISIBLE_DEVICES inside search sessions of a GPU problem"),
)

_BY_NAME = {s.name: s for s in SETTINGS}


def from_env() -> dict[str, Any]:
    """Read every setting from the environment, with defaults; exits on a bad value."""
    out: dict[str, Any] = {}
    for s in SETTINGS:
        raw = os.environ.get(s.env)
        if raw is None or raw == "":
            out[s.name] = s.default
            continue
        try:
            out[s.name] = s.cast(raw)
        except (TypeError, ValueError) as e:
            raise SystemExit(
                f"{s.env}={raw!r} is not a valid {s.cast.__name__} ({s.doc}): {e}"
            ) from e
    return out


def publish(problem: str) -> dict[str, Any]:
    """Write this run's settings to the database, replacing any previous run's.
    Returns what was written."""
    values = from_env()
    db.run_write(
        "MERGE (r:RunConfig {problem: $problem}) "
        "SET r += $values, r.updated_at = datetime()",
        problem=problem, values=values,
    )
    return values


def load(problem: str) -> dict[str, Any]:
    """Read the published settings. Raises RuntimeError if none were published; it must
    not fall back to defaults silently."""
    rows = db.run_read(
        "MATCH (r:RunConfig {problem: $problem}) RETURN properties(r) AS p",
        problem=problem,
    )
    if not rows:
        raise RuntimeError(
            f"no RunConfig published for problem {problem!r}. The orchestrator writes it "
            "at run start (agentdiscover.orchestrator.services ensure); a run that reaches "
            "this point without one would silently use defaults instead of your settings."
        )
    props = rows[0]["p"]
    # Tolerate a node written by an older revision that lacks a newly added setting.
    return {s.name: props.get(s.name, s.default) for s in SETTINGS}


def get(problem: str, name: str) -> Any:
    """One published setting, by name."""
    if name not in _BY_NAME:
        raise KeyError(f"unknown setting {name!r}; known: {sorted(_BY_NAME)}")
    return load(problem)[name]


def summary(values: dict[str, Any]) -> str:
    """A one-line rendering of the settings, for run.log."""
    return " ".join(f"{k}={values[k]}" for k in (s.name for s in SETTINGS))


# Rough per-candidate overhead beyond the evaluation: thinking, writing, tool calls.
_THINK_SECONDS_PER_CANDIDATE = 300
_SESSION_STARTUP_SECONDS = 1800


def required_agent_timeout(problem: str) -> int:
    """The smallest session wall clock that fits the work: `proposals` candidates, each
    blocking for up to `solve_seconds`. Never below the configured agent_timeout."""
    from . import config

    cfg = load(problem)
    solve = config.problem_limits(problem).solve_seconds
    if solve is None:
        return int(cfg["agent_timeout"])
    per_candidate = solve * 1.1 + _THINK_SECONDS_PER_CANDIDATE
    needed = int(cfg["proposals"] * per_candidate) + _SESSION_STARTUP_SECONDS
    return max(int(cfg["agent_timeout"]), needed)
