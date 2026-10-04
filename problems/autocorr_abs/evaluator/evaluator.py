"""Evaluator for the third autocorrelation inequality (upper bound on C).

Called by evaluate.sh with the solution's path as argv[1]; prints one JSON line.

The genome returns heights f_0..f_{n-1} of a step function on [-1/4, 1/4] with equal steps
of width h = (1/2)/n. Heights may be negative. The autoconvolution of a step function is
exactly piecewise LINEAR with breakpoints at multiples of h, and its node values are h*c_k
where c is the discrete autoconvolution of the heights, padded with a zero at each end. A
piecewise linear function attains the maximum of its absolute value at a node, so

    max |f * f|  =  h * max_k |c_k|
    (int f)^2    =  (h * sum f)^2

    C(f) = h*max|c| / (h*sum f)^2 = max|c| / (h * (sum f)^2) = 2n * max|c| / (sum f)^2

using h = 1/(2n). Every power of h cancels except that factor, so the result is an exact
evaluation of the true objective at the submitted step function — a genuine certified upper
bound on C, with no quadrature to exploit.

Sanity check: f == 1 with n = 1 gives c = [1], sum f = 1, so C = 2*1*1/1 = 2, which is the
known value for the indicator of the interval.

Reference values (see PROBLEM.md): 1.4557 is the record (AlphaEvolve, matched by CORAL on
Claude Opus 4.6); 1.459 EvoX; 1.4930 ThetaEvolve's 8B model; 2.0 for the indicator.
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
MIN_MASS = 1e-12

RECORD = 1.4557            # AlphaEvolve; matched by CORAL on Claude Opus 4.6
BEST_CLAUDE = 1.4557       # CORAL, Claude Opus 4.6
EVOX = 1.459
THETAEVOLVE_8B = 1.4930
INDICATOR = 2.0


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
    """Return None if f is a valid step function with non-zero mass, else an error."""
    if not isinstance(f, list):
        return f"expected list, got {type(f).__name__}"
    if len(f) < 2:
        return "need at least 2 step heights"
    if len(f) > MAX_N:
        return f"at most {MAX_N} step heights allowed, got {len(f)}"
    total = 0.0
    for i, x in enumerate(f):
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return f"non-finite height at index {i}: {x!r}"
        total += x
    if abs(total) < MIN_MASS:
        return ("the heights sum to (almost) zero, so the integral of f vanishes and the "
                "ratio is undefined — the denominator (int f)^2 must be non-zero")
    return None


def c_upper_bound(f: list[float]) -> float:
    """Exact C of the step function with heights f. See module docstring."""
    n = len(f)
    a = np.asarray(f, dtype=np.float64)
    c = np.convolve(a, a)                     # length 2n-1
    peak = float(np.abs(c).max())
    total = float(a.sum())
    return 2.0 * n * peak / (total * total)


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

    value = c_upper_bound(f)
    if not math.isfinite(value) or value <= 0.0:
        return {"score": 0.0, "stage": "invalid",
                "error": f"computed C={value!r} is not a positive finite number",
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
        "c_upper_bound": value,
        "n_intervals": len(f),
        "record": RECORD,
        "best_claude": BEST_CLAUDE,
        "evox": EVOX,
        "thetaevolve_8b": THETAEVOLVE_8B,
        "indicator": INDICATOR,
        "beats_thetaevolve": value < THETAEVOLVE_8B,
        "beats_evox": value < EVOX,
        "beats_record": value < RECORD,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
