"""Third autocorrelation inequality — CORAL's metric, unchanged; scoring hardened.

Ported from CORAL's `examples/math/third_autocorr_ineq` grader (identical to
SkyDiscover's `benchmarks/math/third_autocorr_ineq` evaluator): the candidate's `run()`
returns `(f_values, c3_achieved, loss, n_points)`, and

  * `f_values.shape` must equal `(n_points,)`;
  * `dx = 0.5 / n_points`, `integral = (sum(f) * dx)**2`, rejected below 1e-9;
  * `C3 = max(|convolve(f, f, "full")| * dx) / integral`;
  * `score = BENCHMARK / C3` with `BENCHMARK = 1.4556427953745406`, so a score above 1 is
    under the benchmark. C3 minimised.

Two things differ from CORAL/SkyDiscover, both to close scoring holes they share (they
run the candidate and do the scoring math in ONE process, and they score the value the
candidate REPORTS after only a 1e-3 cross-check):

  1. `run()` executes in a fresh subprocess (`_run_solution`) that returns only the raw
     `f_values` (plus the candidate's reported c3, for the record). ALL checks and the C3
     recomputation happen HERE, in the evaluator, from `f_values` — so a candidate cannot
     override numpy or print a fabricated result to reach the scoring path.
  2. **The score is the RECOMPUTED C3, not the reported one.** CORAL scores
     `BENCHMARK / c3_achieved` where `c3_achieved` is what the candidate claims, checked
     only to `atol=1e-3` — and at C3 ~ 1.4556 that 1e-3 is ~0.07% of the score, enough on
     its own to cross the "new record" line. The consistency check is kept exactly as
     theirs (a reported bound off by more than 1e-3 is still an invalid candidate); only
     the number scored is the one the submitted `f` actually attains. `reported_c3`
     carries what the candidate claimed. Same reasoning as erdos_over / txn_scheduling:
     score what the task means, not what the weaker check allows.

Honest candidates are unaffected (their reported and recomputed C3 agree). `main()` prints
the one JSON line the platform's contract asks for.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# The constant CORAL and SkyDiscover both score against, kept exactly so our
# `combined_score` is directly comparable to their published one.
BENCHMARK = 1.4556427953745406

# The bar to get clearly under, reported to the agent as the target, to the four digits
# the published bound is quoted to rather than to CORAL's full precision.
TARGET = 1.4557

# The whole evaluation's budget; the orchestrator passes the per-problem figure so the
# number the agent is told and the number enforced here cannot drift apart.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 600.0))


def _run_solution(solution_dir: Path, timeout: float) -> dict:
    """Run the candidate's run() in a fresh subprocess and return only the raw outputs.

    The subprocess does NO scoring: it hands back f_values (and what the candidate
    reported), and the evaluator checks and scores them. So the candidate's process can
    neither reach the scoring code nor fabricate the number that is scored."""
    code = (
        "import json, sys, os, time\n"
        "import numpy as np\n"
        "sys.path.insert(0, '.')\n"
        "import solution as program\n"
        "start = time.time()\n"
        "try:\n"
        "    f_values, c3_achieved, loss, n_points = program.run()\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': 'run() failed: {}: {}'.format(type(e).__name__, e)}))\n"
        "    sys.exit(0)\n"
        "eval_time = time.time() - start\n"
        "try:\n"
        "    f = np.asarray(f_values, dtype=float)\n"
        "    n_points = int(n_points)\n"
        "    c3_achieved = float(c3_achieved)\n"
        "    loss = float(loss)\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': 'run() returned unreadable values: {}: {}'.format(type(e).__name__, e)}))\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'f': f.tolist(), 'c3_reported': c3_achieved, 'loss': loss, "
        "'n_points': n_points, 'eval_time': eval_time}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"run() ran past its {timeout:.0f}s budget and was killed. Keep its internal "
            "time budgets safely below the evaluator's limit."
        ) from None
    if out.returncode != 0:
        detail = (out.stderr or "").strip() or (out.stdout or "").strip()[-500:]
        reason = f"exit code {out.returncode}"
        if out.returncode < 0 or out.returncode in (134, 137, 139):
            reason += " (killed — likely the memory cap or another resource limit)"
        raise RuntimeError(f"run() failed, {reason}: {detail[:500] or 'no output produced'}")
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("run() produced no JSON line")


def _check_and_score(status: dict) -> dict:
    """CORAL's validation and C3 recomputation, verbatim, on what run() returned — but
    run HERE, and the score taken from the recomputed C3. Returns a dict with either
    "error" (their message) or the scored fields."""
    f = np.asarray(status["f"], dtype=float)
    c3_reported = float(status["c3_reported"])
    n_points = int(status["n_points"])

    if f.shape != (n_points,):
        return {"error": f"Expected shape ({n_points},), got {f.shape}"}

    dx = 0.5 / n_points
    integral_f_sq = (np.sum(f) * dx) ** 2
    if integral_f_sq < 1e-9:
        return {"error": "Function integral is close to zero, ratio is unstable"}

    conv = np.convolve(f, f, mode="full")
    computed_c3 = float(np.max(np.abs(conv * dx)) / integral_f_sq)

    delta = abs(computed_c3 - c3_reported)
    if delta > 1e-3:
        return {"error": (f"C3 mismatch: reported {c3_reported:.6f}, computed "
                          f"{computed_c3:.6f}, delta: {delta:.6f}")}

    return {"c3": computed_c3, "reported_c3": c3_reported, "delta": delta}


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        status = _run_solution(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — a bad candidate scores 0, never crashes us
        return {"score": 0.0, "stage": "error", "error": str(e)[:400],
                "time": time.perf_counter() - t0}
    if "error" in status:
        return {"score": 0.0, "stage": "invalid", "error": str(status["error"])[:400],
                "time": time.perf_counter() - t0}

    try:
        result = _check_and_score(status)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error",
                "error": f"Evaluation failed: {type(e).__name__}: {e}"[:400],
                "time": time.perf_counter() - t0}
    if "error" in result:
        return {"score": 0.0, "stage": "invalid", "error": str(result["error"])[:400],
                "time": time.perf_counter() - t0}

    c3 = result["c3"]
    if not math.isfinite(c3) or c3 <= 0:
        return {"score": 0.0, "stage": "invalid",
                "error": f"C3 must be positive and finite, got {c3}",
                "time": time.perf_counter() - t0}

    eval_time = float(status.get("eval_time", 0.0))
    return {
        "score": BENCHMARK / c3,
        "stage": "full",
        "objective": c3,
        "objective_name": "C3",
        "objective_direction": "min",
        "objective_target": TARGET,
        "combined_score": BENCHMARK / c3,
        "c3": c3,
        "reported_c3": result["reported_c3"],
        "delta": result["delta"],
        "loss": float(status.get("loss", 0.0)),
        "n_points": int(status["n_points"]),
        "target": TARGET,
        "solve_time": eval_time,
        "time": time.perf_counter() - t0,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
