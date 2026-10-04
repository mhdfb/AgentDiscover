"""Evaluator for circle packing: 26 disjoint disks in the unit square, maximise sum of radii.

Called by evaluate.sh with the solution's path as argv[1]; prints one JSON line.

Validity is checked exactly, in plain arithmetic, with a single absolute tolerance: disks
must sit inside the unit square and must not overlap. The tolerance is 1e-9 — tight enough
that the published nine-decimal records remain meaningful, loose enough that a legitimate
numerical optimiser is not punished for floating-point dust.

The score is normalised by the PROVEN upper bound rather than by the current record, so a
new record is visible as a score above the record's own normalised value rather than being
clipped at 1.0. That bound: the disks are disjoint and inside a unit square, so the total
area sum(pi r_i^2) <= 1, and Cauchy-Schwarz gives sum(r_i) <= sqrt(26 * sum(r_i^2))
<= sqrt(26/pi) = 2.87704...

Reference values (see PROBLEM.md): 2.634 Friedman 2012, 2.63586276 AlphaEvolve,
2.6359 OpenEvolve on Claude Opus 4.6, 2.6359857 best published.
"""
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

N_CIRCLES = 26
TOL = 1e-9
# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 120.0))

UPPER_BOUND = math.sqrt(N_CIRCLES / math.pi)   # 2.87704... proven, loose
RECORD = 2.6359857          # best published (ThetaEvolve)
ALPHAEVOLVE = 2.63586276
BEST_CLAUDE = 2.6359        # OpenEvolve run on Claude Opus 4.6, per CORAL's table
HUMAN_2012 = 2.634


def _run_solve(solution_dir: Path, timeout: float) -> tuple[list, float]:
    """Run solve() in a fresh subprocess. Return (circles, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': [[float(c) for c in row] for row in v], 'time': dt}))\n"
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


def verify(circles) -> str | None:
    """Return None if the packing is valid, else a specific error string."""
    if not isinstance(circles, list):
        return f"expected a list of circles, got {type(circles).__name__}"
    if len(circles) != N_CIRCLES:
        return f"expected exactly {N_CIRCLES} circles, got {len(circles)}"
    for i, row in enumerate(circles):
        if not isinstance(row, list) or len(row) != 3:
            return f"circle {i} is not a [x, y, r] triple: {row!r}"
        x, y, r = row
        if not all(math.isfinite(v) for v in (x, y, r)):
            return f"circle {i} has a non-finite value: {row!r}"
        if r < -TOL:
            return f"circle {i} has negative radius {r!r}"
        if x - r < -TOL or x + r > 1.0 + TOL or y - r < -TOL or y + r > 1.0 + TOL:
            return (f"circle {i} at ({x:.12g}, {y:.12g}) with r={r:.12g} is not inside the "
                    "unit square")
    for i in range(N_CIRCLES):
        xi, yi, ri = circles[i]
        for j in range(i + 1, N_CIRCLES):
            xj, yj, rj = circles[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < ri + rj - TOL:
                return (f"circles {i} and {j} overlap by {ri + rj - d:.3e} "
                        f"(centres {d:.12g} apart, radii sum {ri + rj:.12g})")
    return None


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        circles, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    err = verify(circles)
    if err is not None:
        return {"score": 0.0, "stage": "invalid", "error": err,
                "time": time.perf_counter() - t0}

    total = sum(max(0.0, row[2]) for row in circles)
    if total > UPPER_BOUND + 1e-6:
        return {"score": 0.0, "stage": "invalid",
                "error": (f"sum of radii {total!r} exceeds the proven upper bound "
                          f"{UPPER_BOUND} — the packing cannot be valid"),
                "time": time.perf_counter() - t0}

    return {
        "score": max(0.0, min(total / UPPER_BOUND, 1.0)),
        "stage": "full",
        # The three keys the briefing and the steering messages read. fitness is a
        # normalised ratio and compresses near the record; these carry the number the
        # agent actually reasons about, at full precision.
        "objective": total,
        "objective_name": "sum r",
        "objective_direction": "max",
        "objective_target": RECORD,
        "sum_radii": total,
        "upper_bound": UPPER_BOUND,
        "record": RECORD,
        "alphaevolve": ALPHAEVOLVE,
        "best_claude": BEST_CLAUDE,
        "human_2012": HUMAN_2012,
        "beats_human_2012": total > HUMAN_2012,
        "beats_alphaevolve": total > ALPHAEVOLVE,
        "beats_record": total > RECORD,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
