"""Scalar evaluator for the LABS problem.

Called by evaluate.sh with the solution's path as argv[1]; prints a single JSON line
to stdout. Runs inside the evaluation sandbox — no network, hard resource limits —
so it can assume nothing about the machine beyond python3.

Design notes:
- `_run_solve` runs solve() in a fresh subprocess so an agent can't accidentally leak
  state across attempts (caches, RNG, module globals).
- Shape verification rejects malformed returns before scoring — the anti-gaming hook;
  extend per problem with any cheap sanity check worth having.
"""
import json
import os
import math
import subprocess
import sys
import time

# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 60.0))
from pathlib import Path

# ----------------------------------------------------------------------------
# Constants. TUNE_ME per problem.
# ----------------------------------------------------------------------------
N = 60                          # sequence length; must match solution.py
MF_SCALE = 10.0                 # score = min(merit_factor / MF_SCALE, 1.0)


def _run_solve(solution_dir: Path, timeout: float = 60.0) -> tuple[list[int], float]:
    """Run solve() in a fresh subprocess. Return (sequence, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': list(v), 'time': dt}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"solve() ran past its {timeout:.0f}s budget and was killed. Keep its "
            "internal time budgets safely below the evaluator's limit."
        ) from None
    if out.returncode != 0:
        # A killed process (negative exit code, or 134/137/139) usually means the
        # memory cap or another resource limit — and it leaves NO stderr, so say
        # what happened instead of returning an empty message.
        detail = (out.stderr or "").strip() or (out.stdout or "").strip()[-500:]
        reason = f"exit code {out.returncode}"
        if out.returncode < 0 or out.returncode in (134, 137, 139):
            reason += " (killed — likely the memory cap or another resource limit)"
        raise RuntimeError(f"solve failed, {reason}: {detail[:500] or 'no output produced'}")
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            return data["value"], data["time"]
    raise RuntimeError("solve produced no JSON line")


def _verify_shape(seq) -> str | None:
    """Return None if seq is a valid ±1 sequence of length N, else an error string."""
    if not isinstance(seq, list):
        return f"expected list, got {type(seq).__name__}"
    if len(seq) != N:
        return f"expected length {N}, got {len(seq)}"
    bad = [x for x in seq if x not in (-1, 1)]
    if bad:
        return f"non-±1 values present (first offender: {bad[0]!r})"
    return None


def _merit_factor(seq: list[int]) -> float:
    n = len(seq)
    total = 0
    for k in range(1, n):
        c = 0
        for i in range(n - k):
            c += seq[i] * seq[i + k]
        total += c * c
    if total == 0:
        return float("inf")
    return n * n / (2.0 * total)


def _score(mf: float) -> float:
    if not math.isfinite(mf):
        # A non-finite merit factor is a broken genome, not a perfect one. This used to
        # return 1.0, which handed a NaN-producing solution the best possible score.
        return 0.0
    return max(0.0, min(1.0, mf / MF_SCALE))


def evaluate(solution_path: Path) -> dict:
    """Single-stage evaluation. Returns a dict suitable for JSON dump."""
    t0 = time.perf_counter()
    try:
        seq, solve_time = _run_solve(solution_path.parent, timeout=SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    shape_err = _verify_shape(seq)
    if shape_err is not None:
        return {"score": 0.0, "stage": "invalid", "error": shape_err,
                "time": time.perf_counter() - t0}

    mf = _merit_factor(seq)
    return {
        "score": _score(mf),
        "stage": "full",
        # The three keys the briefing and the steering messages read. fitness is a
        # normalised ratio and compresses near the record; these carry the number the
        # agent actually reasons about, at full precision.
        "objective": mf,
        "objective_name": "F",
        "objective_direction": "max",
        "merit_factor": mf,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
