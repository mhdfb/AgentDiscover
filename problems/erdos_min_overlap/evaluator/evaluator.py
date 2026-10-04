"""Evaluator for the Erdős minimum overlap problem (upper bound on C).

Called by evaluate.sh with the solution's path as argv[1]; prints one JSON line.

The genome returns heights f_0..f_{n-1} in [0,1] of a step function on [-1,1] with equal
steps of width h = 2/n, subject to the mass constraint int f = 1, i.e. h*sum(f) = 1, i.e.
sum(f) = n/2. The partner g = 1 - f is forced.

The cross-correlation of two step functions on a common grid is piecewise LINEAR in the
shift, with breakpoints at multiples of h, so its supremum over all real shifts is attained
at a breakpoint and equals

    overlap = h * max_k  sum_i f_i * g_{i+k}          (k = -(n-1) .. (n-1))

This is an exact evaluation of the true objective at the submitted step function, not a
numerical approximation of it, so the value is a genuine certified upper bound on C.

Sanity check: f == 1/2 everywhere satisfies the constraint and gives, at shift 0,
h * n * (1/4) = (2/n) * n/4 = 1/2 — the classical trivial bound.

Reference values (see PROBLEM.md): 0.379005 proven lower bound (White 2022), 0.380876 best
construction (Yuksekgonul et al. Jan 2026), 0.380924 AlphaEvolve, 0.5 for the flat function.
"""
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MAX_N = 4000
# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 120.0))
MASS_TOL = 1e-6

PROVEN_LOWER = 0.379005    # White 2022 — nothing can go below this
RECORD = 0.380876          # Yuksekgonul et al., Jan 2026
ALPHAEVOLVE = 0.380924
BEST_CLAUDE = 0.38089      # CORAL on Claude Opus 4.6
TRIVIAL = 0.5


def _run_solve(solution_dir: Path, timeout: float) -> tuple[list, float]:
    """Run solve() in a fresh subprocess. Return (heights, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': [float(x) for x in v], 'time': dt}))\n"
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


def _verify_shape(f) -> str | None:
    """Return None if f is a valid admissible step function, else an error string."""
    if not isinstance(f, list):
        return f"expected list, got {type(f).__name__}"
    n = len(f)
    if n < 2:
        return "need at least 2 step heights"
    if n > MAX_N:
        return f"at most {MAX_N} step heights allowed, got {n}"
    total = 0.0
    for i, x in enumerate(f):
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return f"non-finite height at index {i}: {x!r}"
        if x < 0.0 or x > 1.0:
            return f"height at index {i} is {x!r}; f must take values in [0, 1]"
        total += x
    required = n / 2.0
    if abs(total - required) > MASS_TOL:
        return (f"mass constraint violated: the integral of f must be 1, so with n={n} steps "
                f"of width 2/n the heights must sum to exactly n/2 = {required}. "
                f"Yours sum to {total!r} (off by {total - required:+.3e}).")
    return None


def overlap_bound(f: list[float]) -> float:
    """Exact sup over shifts of the cross-correlation of f with g = 1 - f."""
    n = len(f)
    h = 2.0 / n
    a = np.asarray(f, dtype=np.float64)
    g = 1.0 - a
    # np.correlate(a, g, "full")[k] runs over every integer shift of the two grids.
    corr = np.correlate(a, g, mode="full")
    return float(h * corr.max())


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        f, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    shape_err = _verify_shape(f)
    if shape_err is not None:
        return {"score": 0.0, "stage": "invalid", "error": shape_err,
                "time": time.perf_counter() - t0}

    value = overlap_bound(f)
    if not math.isfinite(value) or value < PROVEN_LOWER:
        return {"score": 0.0, "stage": "invalid",
                "error": (f"computed overlap {value!r} is below the proven lower bound "
                          f"{PROVEN_LOWER} — the construction or the arithmetic is wrong, "
                          "since no admissible f can go below it"),
                "time": time.perf_counter() - t0}

    return {
        # No upper clamp: the divisor is a RECORD, not a proven bound, so a
        # candidate that beats it must score above 1.0. Clamping here made a
        # new record indistinguishable from merely matching one, and destroyed
        # the ordering among record-beating candidates.
        "score": max(0.0, RECORD / value),
        "stage": "full",
        # The three keys the briefing and the steering messages read. fitness is a
        # normalised ratio and compresses near the record; these carry the number the
        # agent actually reasons about, at full precision.
        "objective": value,
        "objective_name": "C",
        "objective_direction": "min",
        "objective_target": RECORD,
        "overlap": value,
        "n_intervals": len(f),
        "proven_lower_bound": PROVEN_LOWER,
        "record": RECORD,
        "alphaevolve": ALPHAEVOLVE,
        "best_claude": BEST_CLAUDE,
        "trivial": TRIVIAL,
        "beats_alphaevolve": value < ALPHAEVOLVE,
        "beats_best_claude": value < BEST_CLAUDE,
        "beats_record": value < RECORD,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
