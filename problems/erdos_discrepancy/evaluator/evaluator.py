"""Evaluator for the Erdős discrepancy problem (C=2).

Called by evaluate.sh with the solution's path as argv[1]; prints one JSON line.

Scoring follows DeepMind's published verification code for this problem: the score is
the length of the longest PREFIX whose discrepancy is at most 2.

The check is exact and incremental. Position m belongs to the homogeneous progression of
common difference d exactly when d divides m, so extending the sequence by one term only
touches the running sums of the divisors of m. Total work is O(N * sqrt(N)) — a full
1160-term sequence verifies in milliseconds, and no floating point is involved.

The only number given to the agent is the proven optimum, 1160. Published competitor
results live in paper/benchmark.md; see the note beside OPTIMUM below for why they, and
the multiplicative-function hint, are kept out of everything the agent reads.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_DISCREPANCY = 2
OPTIMUM = 1160                 # C(2), certified by Konev & Lisitsa (2014)
# AlphaEvolve's published results (200 unhinted, 380 hinted toward multiplicative
# functions) are recorded in paper/benchmark.md and deliberately kept out of this file:
# anything returned here reaches the agent, and a competitor's number anchors the search.
# The hint itself is kept out of PROBLEM.md for the same reason — supplying it would make
# our result comparable only to their hinted 380, never to their unhinted 200.
# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 120.0))
MAX_RETURNED = 200_000        # refuse absurdly long returns rather than hang


def _run_solve(solution_dir: Path, timeout: float) -> tuple[list, float]:
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
    """Return None if seq is a valid ±1 sequence, else an error string."""
    if not isinstance(seq, list):
        return f"expected list, got {type(seq).__name__}"
    if not seq:
        return "empty sequence"
    if len(seq) > MAX_RETURNED:
        return f"sequence longer than {MAX_RETURNED} terms"
    for i, x in enumerate(seq):
        if x not in (-1, 1):
            return f"non-±1 value at index {i}: {x!r}"
    return None


def _divisors(m: int) -> list[int]:
    ds, d = [], 1
    while d * d <= m:
        if m % d == 0:
            ds.append(d)
            if d != m // d:
                ds.append(m // d)
        d += 1
    return ds


def longest_good_prefix(seq: list[int]) -> int:
    """Length of the longest prefix with discrepancy <= MAX_DISCREPANCY."""
    sums: dict[int, int] = {}
    for m in range(1, len(seq) + 1):
        v = seq[m - 1]
        for d in _divisors(m):
            s = sums.get(d, 0) + v
            if abs(s) > MAX_DISCREPANCY:
                return m - 1
            sums[d] = s
    return len(seq)


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        seq, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    shape_err = _verify_shape(seq)
    if shape_err is not None:
        return {"score": 0.0, "stage": "invalid", "error": shape_err,
                "time": time.perf_counter() - t0}

    length = longest_good_prefix(seq)
    return {
        # No upper clamp: the divisor is a RECORD, not a proven bound, so a
        # candidate that beats it must score above 1.0. Clamping here made a
        # new record indistinguishable from merely matching one, and destroyed
        # the ordering among record-beating candidates.
        "score": length / OPTIMUM,
        "stage": "full",
        # The three keys the briefing and the steering messages read. fitness is a
        # normalised ratio and compresses near the record; these carry the number the
        # agent actually reasons about, at full precision.
        "objective": length,
        "objective_name": "length",
        "objective_direction": "max",
        "objective_target": OPTIMUM,
        "length": length,
        "returned_length": len(seq),
        "optimum": OPTIMUM,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
