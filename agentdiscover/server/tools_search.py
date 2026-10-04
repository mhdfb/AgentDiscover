"""The built-in search-agent tools and their write logic.

Every write is a specific tool, never free Cypher; read_graph runs in a read transaction.
A ToolError's message goes straight back to the agent; anything else is a real bug.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from neo4j.exceptions import ClientError, ConstraintError

from .. import db, runconfig
from . import auth, evaluator

VALID_KINDS = ("mutation", "crossover", "repair", "inspired")

# Candidate properties the evaluator's metric keys must never overwrite.
RESERVED_CANDIDATE_PROPS = {
    "genome", "genome_hash", "fitness", "eval_status",
    "linguistic_prediction", "fitness_prediction",
}


class ToolError(Exception):
    """Agent-visible argument/state error."""


_problem: str = ""
_backend: str = "none"


def configure(problem: str, backend: str) -> None:
    global _problem, _backend
    _problem = problem
    _backend = backend


def _time_left(ctx) -> str:
    """Sentence stating the session's remaining wall clock, or "" if it is unknown."""
    try:
        rows = db.run_read(
            "MATCH (s:Session {id: $sid}) "
            "RETURN duration.inSeconds(s.started_at, datetime()).seconds AS elapsed",
            sid=ctx.session_id,
        )
        if not rows or rows[0]["elapsed"] is None:
            return ""
        left = runconfig.required_agent_timeout(_problem) - int(rows[0]["elapsed"])
        if left <= 0:
            return " Your session's wall clock is spent — write the lesson now."
        return (f" About {left / 3600:.1f} h of this session's wall clock remain"
                f" ({left} s).")
    except Exception:  # noqa: BLE001 — a missing clock must never fail a submission
        return ""


def _genome_lock(genome_hash: str) -> threading.Lock:
    """One lock per genome, held across check-evaluate-write so it is evaluated once.
    Process-local: relies on every submission going through this one service process."""
    with _locks_guard:
        return _locks.setdefault(genome_hash, threading.Lock())


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _proposals_target() -> int:
    """Candidate cap per session. Read from the database, not this process's environment,
    because the service outlives the run that started it."""
    return int(runconfig.get(_problem, "proposals"))


# ---------------------------------------------------------------------------
# read_graph — available to search and meta callers alike
# ---------------------------------------------------------------------------


def read_graph(cypher: str, row_limit: int) -> str:
    ctx = auth.require()
    try:
        rows = db.run_read(cypher)
    except ClientError as e:
        code = e.code or ""
        if "WriteInReadAccessMode" in code or "AccessMode" in code:
            raise ToolError(
                "read_graph refuses writes — the query ran inside a read transaction "
                "and Neo4j rejected it. Writes happen only through the write tools."
            ) from None
        raise ToolError(f"Cypher error: {e.message}") from None

    truncated = len(rows) > row_limit
    rows = rows[:row_limit]
    payload = json.dumps(rows, default=str, ensure_ascii=False)

    n = len(rows)
    if ctx.kind == "meta":
        steer = f"{n} rows." + (" (truncated at the row limit — narrow the query.)" if truncated else "")
    elif n == 0:
        steer = "0 rows. Adjust the query, or propose a candidate when you are ready."
    else:
        steer = f"{n} rows." + \
            (" Truncated at the row limit — narrow the query." if truncated else "") + \
            " Explore more, or propose a candidate when you are ready."
    return payload + "\n\n" + steer


# ---------------------------------------------------------------------------
# add_idea
# ---------------------------------------------------------------------------


def add_idea(name: str, description: str, related_to: list[dict[str, str]] | None) -> str:
    auth.require("search")
    name = (name or "").strip()
    description = (description or "").strip()
    if not name or not description:
        raise ToolError("add_idea needs a non-empty `name` and `description`.")
    related_to = related_to or []
    for rel in related_to:
        if not rel.get("name") or not rel.get("how"):
            raise ToolError(
                "each related_to entry needs `name` (an existing idea) and `how` "
                "(one short sentence linking the two)."
            )

    missing = [
        rel["name"] for rel in related_to
        if not db.run_read("MATCH (i:Idea {name: $n}) RETURN 1", n=rel["name"])
    ]
    if missing:
        raise ToolError(
            f"related_to names not in the database: {missing}. "
            "Create them first, or check the spelling with read_graph."
        )

    def _tx(tx):
        existed = bool(tx.run("MATCH (i:Idea {name: $n}) RETURN 1", n=name).single())
        tx.run(
            "MERGE (i:Idea {name: $n}) ON CREATE SET i.description = $d",
            n=name, d=description,
        )
        for rel in related_to:
            tx.run(
                """
                MATCH (a:Idea {name: $a}), (b:Idea {name: $b})
                MERGE (a)-[r:RELATED_TO]->(b) SET r.how = $how
                """,
                a=name, b=rel["name"], how=rel["how"],
            )
        return existed

    with db.get_driver().session() as session:
        existed = session.execute_write(_tx)

    if existed:
        return (
            f"Idea '{name}' already existed — reusing it (its original description stands). "
            "Link it to a candidate when you submit."
        )
    return f"Idea '{name}' created. Link it to a candidate when you submit."


# ---------------------------------------------------------------------------
# submit_candidate
# ---------------------------------------------------------------------------


def _validate_submission(idea: str, parents: list[dict], resources: list[str]) -> None:
    if not db.run_read("MATCH (i:Idea {name: $n}) RETURN 1", n=idea):
        raise ToolError(
            f"idea '{idea}' is not in the database. Create it with add_idea first, "
            "or reuse an existing idea's exact name (find them with read_graph)."
        )
    for p in parents:
        h, kind = p.get("genome_hash", ""), p.get("kind", "")
        if kind not in VALID_KINDS:
            raise ToolError(
                f"parent kind {kind!r} is not one of {VALID_KINDS}."
            )
        if not db.run_read("MATCH (c:Candidate {genome_hash: $h}) RETURN 1", h=h):
            raise ToolError(
                f"parent genome_hash {h!r} is not in the database — declare only "
                "candidates you actually retrieved."
            )
    for name in resources:
        if not db.run_read("MATCH (r:Resource {name: $n}) RETURN 1", n=name):
            known = db.run_read("MATCH (r:Resource) RETURN r.name AS name")
            raise ToolError(
                f"resource {name!r} does not exist. Known resources: "
                + (", ".join(r["name"] for r in known) or "(none for this problem)")
            )


def _property_value(v: Any) -> Any:
    """Coerce one metric to a Neo4j property value: scalars and flat single-typed lists
    pass through, anything else becomes JSON. None stays None (property absent)."""
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        items = list(v)
        if not items:
            return items
        # bool first: bool is a subclass of int in Python but not a number in Neo4j.
        if all(isinstance(x, bool) for x in items):
            return items
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items):
            # One numeric type per list: promote to float if the list mixes them.
            return [float(x) for x in items] if any(isinstance(x, float) for x in items) \
                else items
        if all(isinstance(x, str) for x in items):
            return items
    return json.dumps(v, default=str)


def _sanitize_metrics(metrics: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k in RESERVED_CANDIDATE_PROPS or k == "score":
            continue
        out[str(k)] = _property_value(v)
    return out


def submit_candidate(
    idea: str,
    linguistic_prediction: str,
    fitness_prediction: float,
    parents: list[dict] | None,
    resources: list[str] | None,
) -> str:
    ctx = auth.require("search")
    parents = parents or []
    resources = resources or []
    if not (linguistic_prediction or "").strip():
        raise ToolError("linguistic_prediction must be a non-empty sentence, written before evaluation.")
    try:
        fitness_prediction = float(fitness_prediction)
    except (TypeError, ValueError):
        raise ToolError("fitness_prediction must be a number.") from None

    solution = ctx.worktree / "solution.py"
    if not solution.exists():
        raise ToolError("solution.py not found in your worktree — nothing to submit.")
    genome = solution.read_text()
    genome_hash = evaluator_hash(genome)

    _validate_submission(idea, parents, resources)

    # The duplicate check, evaluation and write must all stay under this genome's lock.
    with _genome_lock(genome_hash):
        dup = db.run_read(
            "MATCH (c:Candidate {genome_hash: $h}) RETURN c.fitness AS fitness", h=genome_hash
        )
        if dup:
            return _steer_after_submit(ctx, dup[0]["fitness"], fitness_prediction,
                                       duplicate=True, counted=False)

        try:
            metrics = evaluator.evaluate(_problem, _backend, genome, genome_hash)
        except evaluator.EvaluatorSetupError as e:
            raise ToolError(
                f"the evaluation environment is unavailable: {e} Nothing was recorded. "
                "Retry once at most; if it persists, continue the session and note it in "
                "your lesson."
            ) from None
        score = metrics.pop("score", 0.0)
        try:
            fitness = float(score)
        except (TypeError, ValueError):
            fitness = 0.0
        eval_status = "failed" if metrics.get("stage") in ("timeout", "harness_error") else "done"

        def _tx(tx):
            tx.run(
                """
                CREATE (c:Candidate {
                    genome_hash: $h, genome: $genome,
                    linguistic_prediction: $lp, fitness_prediction: $fp,
                    fitness: $fitness, eval_status: $status,
                    created_at: datetime() })
                SET c += $metrics
                WITH c
                MATCH (s:Session {id: $session_id})
                MERGE (s)-[:PRODUCED]->(c)
                WITH c
                MATCH (i:Idea {name: $idea})
                MERGE (c)-[:USES_IDEA]->(i)
                """,
                h=genome_hash, genome=genome, lp=linguistic_prediction,
                fp=fitness_prediction, fitness=fitness, status=eval_status,
                metrics=_sanitize_metrics(metrics),
                session_id=ctx.session_id, idea=idea,
            )
            for p in parents:
                tx.run(
                    """
                    MATCH (c:Candidate {genome_hash: $h}), (p:Candidate {genome_hash: $ph})
                    MERGE (c)-[r:DERIVED_FROM]->(p) SET r.kind = $kind
                    """,
                    h=genome_hash, ph=p["genome_hash"], kind=p["kind"],
                )
            for name in resources:
                tx.run(
                    """
                    MATCH (c:Candidate {genome_hash: $h}), (r:Resource {name: $n})
                    MERGE (c)-[:USED_CONTEXT]->(r)
                    """,
                    h=genome_hash, n=name,
                )

        try:
            with db.get_driver().session() as session:
                session.execute_write(_tx)
        except ConstraintError:
            # Another process wrote this genome first; treat this call as a duplicate.
            dup = db.run_read(
                "MATCH (c:Candidate {genome_hash: $h}) RETURN c.fitness AS fitness", h=genome_hash
            )
            stored = dup[0]["fitness"] if dup else fitness
            return _steer_after_submit(ctx, stored, fitness_prediction,
                                       duplicate=True, counted=False)

    failure_note = ""
    if eval_status == "failed" or metrics.get("stage") in ("error", "invalid", "timeout"):
        failure_note = (f" EVALUATION FAILED ({metrics.get('stage')}): "
                        f"{str(metrics.get('error', '')).strip()[:500] or 'no diagnostic captured'}."
                        " Fix the cause before your next candidate — a failed submission"
                        " still counts toward your quota.")
    return _steer_after_submit(ctx, fitness, fitness_prediction,
                               duplicate=False, counted=True, note=failure_note,
                               genome_hash=genome_hash,
                               program_output=str(metrics.get("program_output") or ""))


def _noise_phrase(genome_hash: str) -> str:
    """Sentence saying the candidate is within the graph's median mutation step of its
    parent, or "" when it is not or there is too little data."""
    rows = db.run_read(
        """
        MATCH (c:Candidate {genome_hash: $h})-[:DERIVED_FROM]->(p:Candidate)
        WHERE c.objective IS NOT NULL AND p.objective IS NOT NULL
        RETURN c.objective AS obj, min(abs(c.objective - p.objective)) AS d
        """,
        h=genome_hash,
    )
    if not rows or rows[0]["d"] is None:
        return ""
    scale = db.run_read(
        """
        MATCH (c:Candidate)-[e:DERIVED_FROM {kind: 'mutation'}]->(p:Candidate)
        WHERE c.objective IS NOT NULL AND p.objective IS NOT NULL
        WITH abs(c.objective - p.objective) AS step ORDER BY step
        WITH collect(step) AS steps WHERE size(steps) >= 5
        RETURN steps[size(steps)/2] AS median
        """
    )
    if not scale or scale[0]["median"] is None:
        return ""
    d, med = rows[0]["d"], scale[0]["median"]
    if med > 0 and d <= med:
        return (f"This differs from its parent by {d:.2g} — at or below the median "
                f"mutation step in this graph ({med:.2g}); a difference this small is "
                "as likely noise as improvement. ")
    return ""


def _objective_phrase(genome_hash: str, prediction: float | None = None) -> tuple[str, bool]:
    """The raw task objective at full precision, with the gap to the target.

    Returns (phrase, whether it carried `prediction`); phrase is "" with no objective."""
    rows = db.run_read(
        """
        MATCH (c:Candidate {genome_hash: $h})
        RETURN c.objective AS obj, c.objective_name AS name,
               c.objective_target AS target, c.objective_direction AS direction
        """,
        h=genome_hash,
    )
    if not rows or rows[0]["obj"] is None:
        return "", False
    r = rows[0]
    name = r["name"] or "objective"
    phrase = f"{name} = {r['obj']:.9g}"
    target, direction = r["target"], r["direction"]
    if target:
        gap = (r["obj"] - target) / target * 100.0
        if direction == "min":
            beat = r["obj"] < target
        else:
            beat = r["obj"] > target
            gap = -gap
        # Say "target", never "record", to match the briefing's wording.
        phrase += (f" — past the target of {target:.9g}, which is a bar to beat clearly, "
                   "not a finish line" if beat
                   else f" (target {target:.9g}, {abs(gap):.3f}% away)")
    took = prediction is not None
    if took:
        phrase += f"; you predicted {prediction:.9g}"
    return phrase + ". ", took


def _session_evidence(ctx: auth.AuthContext, genome_hash: str = "") -> str:
    """This session's earlier scores and the best in the database, with no verdict.
    Excludes candidate `genome_hash`. Uses the objective where published, else fitness."""
    try:
        rows = db.run_read(
            """
            MATCH (s:Session {id: $sid})-[:PRODUCED]->(c:Candidate)
            WITH c ORDER BY c.created_at, c.genome_hash
            RETURN c.objective AS obj, c.fitness AS fit, c.objective_direction AS d,
                   c.eval_status AS st, c.genome_hash AS h,
                   c.fitness_prediction AS pred
            """,
            sid=ctx.session_id,
        )
        if not rows:
            return ""

        # Use objective or fitness throughout, never both: their scales differ.
        use_obj = any(r["obj"] is not None for r in rows)
        direction = (next((r["d"] for r in rows if r["d"]), None) or "max") if use_obj else "max"
        pick = min if direction == "min" else max

        def value(r):
            # A failed candidate has fitness 0.0 and no objective; it is not a score.
            if r["st"] != "done":
                return None
            return r["obj"] if use_obj else r["fit"]

        def prediction(r, v):
            """The candidate's prediction, which is in the same units as `value`."""
            p = r["pred"]
            return p if p is not None and v is not None else None

        scored = [(r["h"], value(r), prediction(r, value(r))) for r in rows]
        scored = [(h, v, p) for h, v, p in scored if v is not None]
        if not scored:
            return ""
        best_here = pick(v for _, v, _ in scored)

        earlier = [(v, p) for h, v, p in scored if h != genome_hash]
        if not earlier:
            line = " Your first scored candidate this session."
        else:
            shown = earlier[-5:]
            prefix = "… " if len(earlier) > len(shown) else ""
            body = ", ".join(
                f"{v:.9g} (predicted {p:.9g})" if p is not None else f"{v:.9g}"
                for v, p in shown)
            line = (f" Earlier this session: {prefix}{body}."
                    f" Best this session: {best_here:.9g}.")

        field = "c.objective" if use_obj else "c.fitness"
        g = db.run_read(
            f"MATCH (c:Candidate) WHERE {field} IS NOT NULL AND c.eval_status = 'done' "
            f"RETURN {'min' if direction == 'min' else 'max'}({field}) AS best"
        )
        if g and g[0]["best"] is not None:
            line += f" Best in the database: {g[0]['best']:.9g}."
        return line
    except Exception:  # noqa: BLE001 — evidence is a courtesy, never a failure
        return ""


def _steer_after_submit(ctx: auth.AuthContext, fitness, prediction: float,
                        *, duplicate: bool, counted: bool, note: str = "",
                        genome_hash: str = "", program_output: str = "") -> str:
    rows = db.run_read(
        "MATCH (s:Session {id: $sid})-[:PRODUCED]->(c:Candidate) RETURN count(c) AS n",
        sid=ctx.session_id,
    )
    n = rows[0]["n"] if rows else 0
    target = _proposals_target()

    if duplicate:
        head = (
            f"duplicate: this exact genome is already in the database with fitness "
            f"{fitness}. It was not evaluated again and does not count as a new "
            f"candidate — change solution.py and submit again."
        )
    else:
        obj_phrase, took_pred = _objective_phrase(genome_hash, prediction)
        head = (f"{obj_phrase}fitness {fitness}"
                f"{'' if took_pred else f' (you predicted {prediction})'}"
                f".{note} {_noise_phrase(genome_hash)}")

    if counted or not duplicate:
        if n >= target:
            tail = (f" That was candidate {n} of at most {target} — the cap. Now write "
                    "the lesson for this session with write_lesson — include the work "
                    "that never became a candidate, since only submitted ones are in "
                    "the graph — then stop.")
        else:
            tail = (f" That was candidate {n} of at most {target}."
                    f"{_session_evidence(ctx, genome_hash)}"
                    " You can read the graph again, and use or add to your persistent"
                    " tools/ folder."
                    " Work on your next candidate: refine what shows promise, or if "
                    "these numbers say you are stuck, switch to a different idea. Stop "
                    "early only if you have tried different ideas and none worked — then "
                    f"write the lesson.{_time_left(ctx)}{_cache_advice(ctx, n)}")
    else:
        tail = f" You have submitted {n} of at most {target} candidates."
    if program_output.strip():
        tail += ("\n\n--- your program's output (tail) ---\n" + program_output.strip())
    return head + tail


# The cache-cost advice is never given before this many candidates.
CACHE_ADVICE_MIN_CANDIDATES = 3
# Cache read and write prices as multiples of the base input price (read is a default).
_CACHE_READ = 0.1
_CACHE_WRITE = {"5m": 1.25, "1h": 2.0, "openai": 1.0}


def _claude_usage(path: Path) -> list[dict]:
    """Per-call usage records from a Claude Code stream-json log, in order."""
    msgs: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") != "assistant":
                    continue
                m = d.get("message") or {}
                # stream-json repeats one API message per content block; keep the first.
                msgs.setdefault(m.get("id", ""), {**(m.get("usage") or {}), "model": m.get("model") or ""})
    except OSError:
        return []
    return list(msgs.values())


def _codex_usage(worktree: Path) -> list[dict]:
    """Per-call usage records from the worktree's newest codex rollout, in order.
    Assumes the newest rollout is the running session."""
    try:
        rollouts = sorted((worktree / ".codex" / "sessions").rglob("rollout-*.jsonl"),
                          key=lambda p: p.stat().st_mtime)
    except OSError:
        return []
    if not rollouts:
        return []
    usage: list[dict] = []
    model = ""
    try:
        with open(rollouts[-1], encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                p = d.get("payload") or {}
                if d.get("type") == "turn_context" and p.get("model"):
                    model = p["model"]
                if p.get("type") == "token_count" and (p.get("info") or {}).get("last_token_usage"):
                    usage.append({**p["info"]["last_token_usage"], "model": model})
    except OSError:
        return []
    return usage


def _cache_advice(ctx: auth.AuthContext, n: int) -> str:
    """Advice to restart when the next candidate costs more here than in a fresh session.

    A restart is cheaper when k * read * (L - c) > write_multiplier * i, with c = tokens
    cached across sessions, i = tokens a fresh session rewrites, L = current context and
    k = model calls per candidate. Returns "" when too early or no log can be read."""
    if n < CACHE_ADVICE_MIN_CANDIDATES or not ctx.worktree or ctx.iteration is None:
        return ""
    read = _CACHE_READ
    usage = _claude_usage(ctx.worktree / "agent-logs" / f"iter-{ctx.iteration}.jsonl")
    if usage:
        first, last = usage[0], usage[-1]
        c = first.get("cache_read_input_tokens", 0)
        i = first.get("cache_creation_input_tokens", 0)
        ttl = "1h" if (first.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") else "5m"
        L = (last.get("cache_read_input_tokens", 0) + last.get("cache_creation_input_tokens", 0)
             + last.get("input_tokens", 0))
    else:
        usage = _codex_usage(ctx.worktree)
        if not usage:
            return ""
        first, last = usage[0], usage[-1]
        c = first.get("cached_input_tokens", 0)          # input_tokens is the whole
        i = first.get("input_tokens", 0) - c             # context, cached part included
        ttl = "openai"
        L = last.get("input_tokens", 0)
    # Use the model's own cached-read ratio when its price is known.
    from ..orchestrator.spend import price_of
    price = price_of(last.get("model", ""))
    if price:
        read = price["cached"] / price["input"]
    k = len(usage) / n
    if k * read * (L - c) <= _CACHE_WRITE[ttl] * i:
        return ""
    return (" The token cost of continuing this session will probably be higher than the "
            "cost of starting a new one (this session's context has grown to "
            f"{L:,} tokens; a fresh session would start from {c + i:,}). Continue only if "
            "you have a great idea that is very likely to give a good result; otherwise "
            "write the lesson and terminate, and you will get another chance to try in "
            "a new session.")


def evaluator_hash(genome: str) -> str:
    import hashlib

    return hashlib.sha256((genome.rstrip() + "\n").encode()).hexdigest()


# ---------------------------------------------------------------------------
# write_lesson
# ---------------------------------------------------------------------------


def write_lesson(text: str) -> str:
    ctx = auth.require("search")
    text = (text or "").strip()
    if not text:
        raise ToolError("write_lesson needs a non-empty lesson covering this session's candidates.")
    db.run_write(
        "MATCH (s:Session {id: $sid}) SET s.lesson_learned = $text",
        sid=ctx.session_id, text=text,
    )
    return "Lesson stored for this session. You are done — stop now."
