"""The meta agent's three write tools. Meta-token callers only."""
from __future__ import annotations

import json
import re

from neo4j.exceptions import ClientError

from .. import db
from . import auth
from .tools_search import ToolError

# Meta-written tools are saved read-only queries; executable code is not accepted.
TOOL_KINDS = ("cypher",)
TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")

# Saved queries are validated when stored, while the meta agent can still fix them.
_MARKUP_RE = re.compile(r"</|<parameter\b|<invoke\b|<antml", re.I)
_WRITE_CLAUSE_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.I)
_PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _static_cypher_checks(code: str, params: dict) -> None:
    """The checks that need no database. Raises ToolError with the reason."""
    if _MARKUP_RE.search(code):
        raise ToolError(
            "code contains tool-call markup ('</', '<parameter', ...): the query text was "
            "corrupted when the call was made. Nothing was stored — send the Cypher "
            "alone as code, and the JSON Schema as its own params argument."
        )
    hit = _WRITE_CLAUSE_RE.search(code)
    if hit:
        raise ToolError(
            f"saved queries are read-only and {hit.group(0).upper()!r} is a write "
            "clause. Nothing was stored."
        )
    declared = set(params.get("properties", {}))
    missing = sorted(set(_PARAM_RE.findall(code)) - declared)
    if missing:
        raise ToolError(
            f"the query uses parameter(s) {missing} that params does not declare, so "
            "no agent could ever pass them and every call would fail. Nothing was "
            "stored — declare them under params.properties."
        )


def _validate_cypher(code: str, params: dict) -> None:
    _static_cypher_checks(code, params)
    try:
        # EXPLAIN plans the query without running it, and needs no parameter values.
        db.run_read("EXPLAIN " + code.strip())
    except ClientError as e:
        raise ToolError(
            f"Neo4j rejected the query, so nothing was stored: {e.message} "
            "Fix the Cypher and call add_tool again."
        ) from None

# Names a meta-written tool may not take.
BUILTIN_TOOL_NAMES = {
    "read_graph", "submit_candidate", "add_idea", "write_lesson",
    "write_meta_lesson", "add_tool", "write_strategy", "get_schema",
}


def write_meta_lesson(session_id: str, text: str) -> str:
    auth.require("meta")
    text = (text or "").strip()
    if not text:
        raise ToolError("write_meta_lesson needs a non-empty judgment of how the session searched.")
    rows = db.run_write(
        """
        MATCH (s:Session {id: $sid})
        SET s.meta_lesson = $text
        RETURN s.id AS id
        """,
        sid=session_id, text=text,
    )
    if not rows:
        raise ToolError(
            f"session {session_id!r} does not exist — session ids look like "
            "'<user>/<agent>/<iteration>' (list them with read_graph)."
        )
    return (
        f"Meta lesson stored on {session_id}. When this branch's sessions are all "
        "reviewed, write the branch's tools with add_tool, then its strategy."
    )


def _require_branch(branch: str) -> None:
    if not db.run_read("MATCH (a:Agent {key: $k}) RETURN 1", k=branch):
        known = db.run_read("MATCH (a:Agent) RETURN a.key AS key")
        raise ToolError(
            f"branch {branch!r} is not an agent key. Known branches: "
            + ", ".join(r["key"] for r in known)
        )


def add_tool(branch: str, name: str, description: str, kind: str,
             code: str, params: dict | None) -> str:
    auth.require("meta")
    _require_branch(branch)
    if not TOOL_NAME_RE.match(name or ""):
        raise ToolError("tool name must match ^[a-z][a-z0-9_]{2,39}$ (it becomes an MCP tool name).")
    if name in BUILTIN_TOOL_NAMES:
        raise ToolError(f"{name!r} is a built-in tool name — pick another.")
    if kind not in TOOL_KINDS:
        raise ToolError(
            "kind must be 'cypher' — meta-written tools are saved queries only. "
            "Executable code is the search agents' to write (worktree tools/); "
            "methodological guidance goes in the skills files, not a tool."
        )
    if not (description or "").strip() or not (code or "").strip():
        raise ToolError("add_tool needs a non-empty description and code.")
    params = params or {"type": "object", "properties": {}}
    if not isinstance(params, dict) or params.get("type") != "object" \
            or not isinstance(params.get("properties", {}), dict):
        raise ToolError(
            'params must be a JSON Schema object: {"type": "object", "properties": {...}} '
            "— it becomes the tool's input schema, shown to the agent like a built-in's."
        )
    if kind == "cypher":
        # Arguments are passed as query parameters, so names must be plain identifiers.
        for p in params.get("properties", {}):
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p):
                raise ToolError(f"cypher tool parameter {p!r} is not a valid parameter name.")
        _validate_cypher(code, params)

    db.run_write(
        """
        MERGE (t:Tool {name: $name, branch: $branch})
        SET t.description = $description, t.code = $code, t.kind = $kind,
            t.params = $params
        WITH t
        MATCH (a:Agent {key: $branch})
        MERGE (a)-[:HAS_TOOL]->(t)
        """,
        name=name, branch=branch, description=description.strip(),
        code=code, kind=kind, params=json.dumps(params),
    )
    return (
        f"Tool '{name}' stored for branch {branch} (the query parsed and planned "
        "cleanly) — its next sessions will be offered it automatically. Write more "
        "tools, or the branch's strategy with write_strategy."
    )


def write_strategy(branch: str, strategy: str, diff_summary: str) -> str:
    auth.require("meta")
    _require_branch(branch)
    strategy = (strategy or "").strip()
    if not strategy:
        raise ToolError("write_strategy needs the strategy text.")
    diff_summary = (diff_summary or "").strip()
    if not diff_summary:
        raise ToolError(
            "write_strategy needs a diff_summary — one or two lines on what changed "
            "versus the previous strategy, and on what evidence."
        )

    def _tx(tx):
        head = tx.run(
            """
            MATCH (a:Agent {key: $branch})-[:HAS_GUIDANCE]->(g:Guidance)
            WHERE NOT ()-[:SUPERSEDES]->(g)
            RETURN g.id AS id
            """,
            branch=branch,
        ).single()
        count = tx.run(
            "MATCH (:Agent {key: $branch})-[:HAS_GUIDANCE]->(g:Guidance) "
            "RETURN count(g) AS n",
            branch=branch,
        ).single()["n"]
        new_id = f"{branch}/g{count}"
        tx.run(
            """
            CREATE (g:Guidance {id: $id, strategy: $strategy, diff_summary: $summary})
            WITH g
            MATCH (a:Agent {key: $branch})
            MERGE (a)-[:HAS_GUIDANCE]->(g)
            """,
            id=new_id, strategy=strategy, summary=diff_summary, branch=branch,
        )
        if head:
            tx.run(
                """
                MATCH (g:Guidance {id: $new}), (old:Guidance {id: $old})
                MERGE (g)-[:SUPERSEDES]->(old)
                """,
                new=new_id, old=head["id"],
            )
        return new_id

    with db.get_driver().session() as session:
        new_id = session.execute_write(_tx)
    return (
        f"Strategy stored as guidance {new_id} for {branch}; its next session runs "
        "under it. Move to the next branch, or you are done."
    )
