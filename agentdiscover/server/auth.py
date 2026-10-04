"""Call attribution: resolves the bearer token in the URL path (/mcp/<token>) to a
search session or a meta pass. Only sha256 fingerprints of tokens are stored (on the
Session node, or in runs/<problem>/meta-pass.json), never the raw token.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from .. import config, db

# Set per request by the ASGI middleware in app.py.
current_token: ContextVar[str | None] = ContextVar("agentdiscover_token", default=None)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class AuthContext:
    kind: str                 # "search" | "meta"
    token: str                # raw token (forwarded to meta-written tools, never stored)
    session_id: str | None    # search only
    agent_key: str | None     # search only
    worktree: Path | None     # search only
    iteration: int | None     # search only
    meta_pass_id: str | None  # meta only


class AuthError(Exception):
    """Raised when a call cannot be attributed. The message is agent-safe."""


_cache: dict[str, tuple[float, AuthContext]] = {}
_CACHE_TTL = 30.0

_problem: str | None = None


def configure(problem: str) -> None:
    global _problem
    _problem = problem


def _meta_pass_file() -> Path:
    assert _problem is not None
    return config.runs_dir(_problem) / "meta-pass.json"


def resolve(token: str | None) -> AuthContext:
    if not token:
        raise AuthError(
            "no session token in the request URL — the MCP endpoint is /mcp/<token>"
        )
    now = time.monotonic()
    cached = _cache.get(token)
    if cached and cached[0] > now:
        return cached[1]

    h = token_hash(token)

    rows = db.run_read(
        """
        MATCH (a:Agent)-[:RAN]->(s:Session {token_hash: $h})
        WHERE s.finished_at IS NULL
        RETURN s.id AS session_id, s.iteration AS iteration,
               a.key AS agent_key, a.worktree AS worktree
        """,
        h=h,
    )
    if rows:
        r = rows[0]
        ctx = AuthContext(
            kind="search", token=token,
            session_id=r["session_id"], agent_key=r["agent_key"],
            worktree=Path(r["worktree"]), iteration=r["iteration"],
            meta_pass_id=None,
        )
        _cache[token] = (now + _CACHE_TTL, ctx)
        return ctx

    meta_file = _meta_pass_file()
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except ValueError:
            meta = {}
        if meta.get("token_hash") == h:
            ctx = AuthContext(
                kind="meta", token=token, session_id=None, agent_key=None,
                worktree=None, iteration=None,
                meta_pass_id=meta.get("id", "meta"),
            )
            _cache[token] = (now + _CACHE_TTL, ctx)
            return ctx

    raise AuthError(
        "unknown or expired session token — this session may already be closed"
    )


def require(kind: str | None = None) -> AuthContext:
    """The auth context of the current request; optionally require search/meta."""
    ctx = resolve(current_token.get())
    if kind is not None and ctx.kind != kind:
        raise AuthError(
            f"this tool is available to {kind} callers only (you are {ctx.kind})"
        )
    return ctx
