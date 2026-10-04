"""Evaluator for the max-to-min distance ratio, 14 points in three dimensions.

Called by evaluate.sh with the solution's path as argv[1]; prints one JSON line.

Objective, minimised:

    ratio^2 = max_{i<j} |x_i - x_j|^2 / min_{i<j} |x_i - x_j|^2

Everything is computed in SQUARED distances, which is both exact (no square roots) and the
convention the published numbers use: sqrt(4.16579) = 2.0410 is the true max/min
distance of the best known configuration, just above the value 2 attained by the
13-point cuboctahedron (a centre plus its 12 nearest neighbours). See PROBLEM.md.

The objective is scale- and translation-invariant, so no bounding box is imposed. With
n = 14 there are only 91 pairs, so the evaluation is exhaustive and instant.

Reference values: 4.16579 best published (CodeEvolve), 4.16585 AlphaEvolve, 4.16 as
reported for Claude Opus 4.6, 9.0 for the grid seed.
"""
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

N_POINTS = 14
DIM = 3
# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 120.0))
MIN_SEPARATION_SQ = 1e-18

RECORD = 4.16579             # CodeEvolve
ALPHAEVOLVE = 4.16585
BEST_CLAUDE = 4.16           # CORAL on Claude Opus 4.6
GRID_SEED = 9.0


def _run_solve(solution_dir: Path, timeout: float) -> tuple[list, float]:
    """Run solve() in a fresh subprocess. Return (points, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': [[float(c) for c in p] for p in v], 'time': dt}))\n"
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


def _verify_shape(points) -> str | None:
    if not isinstance(points, list):
        return f"expected a list of points, got {type(points).__name__}"
    if len(points) != N_POINTS:
        return f"expected exactly {N_POINTS} points, got {len(points)}"
    for i, p in enumerate(points):
        if not isinstance(p, list) or len(p) != DIM:
            return f"point {i} is not a list of {DIM} coordinates: {p!r}"
        for c in p:
            if not isinstance(c, (int, float)) or not math.isfinite(c):
                return f"point {i} has a non-finite coordinate: {p!r}"
    return None


def ratio_squared(points: list[list[float]]) -> tuple[float, float, float]:
    """Return (ratio^2, min_sq, max_sq) over all pairs."""
    lo, hi = math.inf, 0.0
    for i in range(len(points)):
        pi = points[i]
        for j in range(i + 1, len(points)):
            pj = points[j]
            d = sum((a - b) ** 2 for a, b in zip(pi, pj))
            lo = min(lo, d)
            hi = max(hi, d)
    return (hi / lo if lo > 0 else math.inf), lo, hi


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        points, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    shape_err = _verify_shape(points)
    if shape_err is not None:
        return {"score": 0.0, "stage": "invalid", "error": shape_err,
                "time": time.perf_counter() - t0}

    value, lo, hi = ratio_squared(points)
    if lo <= MIN_SEPARATION_SQ:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = sum((a - b) ** 2 for a, b in zip(points[i], points[j]))
                if d <= MIN_SEPARATION_SQ:
                    return {"score": 0.0, "stage": "invalid",
                            "error": (f"points {i} and {j} coincide (squared distance "
                                      f"{d:.3e}); all points must be distinct or the ratio "
                                      "is undefined"),
                            "time": time.perf_counter() - t0}
    if not math.isfinite(value) or value < 1.0:
        return {"score": 0.0, "stage": "invalid",
                "error": f"computed ratio^2 {value!r} is impossible (it cannot be below 1)",
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
        "objective_name": "ratio^2",
        "objective_direction": "min",
        "objective_target": RECORD,
        "ratio_squared": value,
        "ratio": math.sqrt(value),
        "min_dist": math.sqrt(lo),
        "max_dist": math.sqrt(hi),
        "record": RECORD,
        "alphaevolve": ALPHAEVOLVE,
        "best_claude": BEST_CLAUDE,
        "grid_seed": GRID_SEED,
        "beats_alphaevolve": value < ALPHAEVOLVE,
        "beats_record": value < RECORD,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
