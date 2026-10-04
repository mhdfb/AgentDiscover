"""Evaluator for the sum-difference problem (lower bound on C).

Called by evaluate.sh with the solution's path as argv[1]; prints one JSON line.

Scoring is exact integer arithmetic: the sumset and difference set are built as Python
sets, so |A+A| and |A-A| are counted with no floating point anywhere. Only the final
ratio log|A-A| / log|A+A| is a float, and it certifies C >= ratio.

Reference values (see PROBLEM.md): AlphaEvolve reached about 1.21 with no human hints;
the best known bound is log(1+sqrt(2))/log(2) = 1.2715...
"""
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

MAX_SIZE = 4000
MAX_ABS = 10 ** 12
# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 120.0))

BEST_KNOWN = math.log(1 + math.sqrt(2)) / math.log(2)   # 1.2715533...
ALPHAEVOLVE = 1.21


def _run_solve(solution_dir: Path, timeout: float) -> tuple[list, float]:
    """Run solve() in a fresh subprocess. Return (set_as_list, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': [int(x) for x in v], 'time': dt}))\n"
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


def _verify_shape(a) -> str | None:
    """Return None if a is a valid set of distinct bounded integers, else an error."""
    if not isinstance(a, list):
        return f"expected list, got {type(a).__name__}"
    if len(a) < 2:
        return "need at least 2 elements"
    if len(a) > MAX_SIZE:
        return f"at most {MAX_SIZE} elements allowed, got {len(a)}"
    seen = set()
    for i, x in enumerate(a):
        if not isinstance(x, int) or isinstance(x, bool):
            return f"non-integer element at index {i}: {x!r}"
        if abs(x) > MAX_ABS:
            return f"element at index {i} exceeds |a| <= {MAX_ABS}: {x!r}"
        if x in seen:
            return f"duplicate element {x!r} — A is a set, deduplicate before returning"
        seen.add(x)
    return None


def ratio_of(a: list[int]) -> tuple[float, int, int]:
    """Return (log|A-A|/log|A+A|, |A+A|, |A-A|), all counts exact."""
    sums = {x + y for x in a for y in a}
    diffs = {x - y for x in a for y in a}
    n_sum, n_diff = len(sums), len(diffs)
    if n_sum < 2:
        return 0.0, n_sum, n_diff
    return math.log(n_diff) / math.log(n_sum), n_sum, n_diff


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        a, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    shape_err = _verify_shape(a)
    if shape_err is not None:
        return {"score": 0.0, "stage": "invalid", "error": shape_err,
                "time": time.perf_counter() - t0}

    ratio, n_sum, n_diff = ratio_of(a)
    return {
        # No upper clamp: the divisor is a RECORD, not a proven bound, so a
        # candidate that beats it must score above 1.0. Clamping here made a
        # new record indistinguishable from merely matching one, and destroyed
        # the ordering among record-beating candidates.
        "score": max(0.0, ratio / BEST_KNOWN),
        "stage": "full",
        # The three keys the briefing and the steering messages read. fitness is a
        # normalised ratio and compresses near the record; these carry the number the
        # agent actually reasons about, at full precision.
        "objective": ratio,
        "objective_name": "C",
        "objective_direction": "max",
        "objective_target": BEST_KNOWN,
        "c_lower_bound": ratio,
        "set_size": len(a),
        "sumset_size": n_sum,
        "diffset_size": n_diff,
        "best_known": BEST_KNOWN,
        "alphaevolve": ALPHAEVOLVE,
        "beats_alphaevolve": ratio > ALPHAEVOLVE,
        "beats_best_known": ratio > BEST_KNOWN,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
