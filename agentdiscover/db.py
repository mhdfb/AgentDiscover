"""Neo4j driver helpers for the trusted side (orchestrator + MCP service).

Agents never connect to the database; they go through the MCP service.
"""
from __future__ import annotations

import time
from pathlib import Path

from neo4j import Driver, GraphDatabase

from . import config

_driver: Driver | None = None


def get_driver() -> Driver:
    """The process-wide driver (it pools connections internally)."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.neo4j_uri(), auth=config.neo4j_auth(),
            notifications_min_severity="OFF",  # server hints are noise in run logs
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def wait_until_ready(timeout: float = 240.0) -> None:
    """Block until the database answers; exit on timeout or a rejected password."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            get_driver().verify_connectivity()
            return
        except Exception as e:  # noqa: BLE001 — retry anything until the deadline
            close_driver()
            # Never retry an auth rejection: retries trip Neo4j's rate limiter.
            if "Security" in (getattr(e, "code", "") or ""):
                raise SystemExit(
                    f"Neo4j is up but rejected the configured password: {e}\n"
                    "NEO4J_PASSWORD in .env does not match this database. Either set it "
                    "back to the password the database was created with, or start the "
                    "problem over by deleting runs/<problem>/ (the database lives "
                    "there). A first run that failed before initialization finished "
                    "also leaves the database in this state — delete runs/<problem>/ "
                    "and run again."
                )
            last_err = e
            time.sleep(2.0)
    raise SystemExit(f"Neo4j did not become ready within {timeout:.0f}s: {last_err}")


def run_read(cypher: str, **params):
    """Run a query in a read transaction; returns a list of record dicts.
    Neo4j itself rejects any write clause here."""
    with get_driver().session() as session:
        return session.execute_read(
            lambda tx: [r.data() for r in tx.run(cypher, **params)]
        )


def run_write(cypher: str, **params):
    """Run a query in a write transaction; returns a list of record dicts."""
    with get_driver().session() as session:
        return session.execute_write(
            lambda tx: [r.data() for r in tx.run(cypher, **params)]
        )


def apply_schema(path: Path | None = None) -> int:
    """Apply schema.cypher (idempotent); returns the number of statements run."""
    path = path or config.schema_path()
    # Strip comment lines before splitting on ';': a comment may contain a semicolon.
    lines = [ln for ln in path.read_text().splitlines() if not ln.strip().startswith("//")]
    statements = [stmt for chunk in "\n".join(lines).split(";") if (stmt := chunk.strip())]
    with get_driver().session() as session:
        for stmt in statements:
            session.run(stmt).consume()
    return len(statements)
