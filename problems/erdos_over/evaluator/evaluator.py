"""Erdős minimum overlap — a step function h: [0,2] -> [0,1] minimising the maximum
overlap integral, an upper bound on C5. CORAL's metric, unchanged.

Ported from CORAL's `examples/math/erdos_min_overlap` (grader.py), the bundle behind
their published figure; it is SkyDiscover's `benchmarks/math/erdos_min_overlap`
evaluator re-expressed as a grader, with the same checks and messages. The task, the
metric and the feedback are theirs:

  * The candidate defines `run()` returning `(h_values, c5_bound, n_points)`.
  * Validation, in their order and with their messages verbatim: `h` must have shape
    `(n_points,)`; every value must lie in [0, 1]; `sum(h) * dx` with `dx = 2/n_points`
    must equal 1 to `atol=1e-3`; the C5 they recompute as
    `max(np.correlate(h, 1 - h, mode="full") * dx)` must agree with the reported
    `c5_bound` to `atol=1e-4`. A `run()` that raises reports "run() failed: …".
  * `combined_score = BENCHMARK / C5` with `BENCHMARK = 0.38092303510845016`, higher is
    better and > 1 is past the benchmark. That score IS the platform's fitness here.
  * The evaluation returns their fields — `c5_bound`, `combined_score`, `n_points`,
    `eval_time`, the benchmark and whether the score passed 1 ("NEW RECORD!" in their
    grader) — and nothing finer. The 600 s budget is their `grader.timeout: 600`.

What differs, to fit this platform and to close one hole:

  * `run()` executes in a fresh subprocess (`_run_solution`) that prints `h`, the
    reported bound and `n_points` and exits; every check and the score are computed
    HERE, afterwards, from this file's own numpy call — so a candidate can never reach
    the scoring code. In CORAL's layout everything happens in the subprocess that
    imported the candidate.
  * **The score is taken from the RECOMPUTED C5, not the reported one.** CORAL divides
    the benchmark by the candidate's own `c5_bound`, checked only to `atol=1e-4` against
    the true value of the submitted `h` — and this problem is decided in the fifth
    decimal (0.38088 vs 0.38125), so under-reporting by 1e-4 would be worth more than
    any real improvement. The consistency check stays exactly as theirs (a reported
    bound off by more than 1e-4 is still an invalid candidate); only the number scored
    is the one the submitted function actually attains. `reported_c5_bound` carries what
    the candidate claimed, for the record. Same reasoning as txn_scheduling's
    completeness rule: enforce what the task means, not what the weaker check allows.
  * The interpretable objective published to the briefing is C5 itself, minimised — the
    form the paper reports (0.38088) — alongside their score.
  * `main()` prints the one JSON line the platform's contract asks for.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# CORAL's normaliser: AlphaEvolve's C5 upper bound, so that a score of 1.0 is matching
# it. Their task text states it to the agent, so it is not hidden.
BENCHMARK = 0.38092303510845016

# The objective the briefing tracks is C5, minimised. TARGET is a bar to BEAT by a clear
# margin, not to meet: 0.38088 is the figure the user set for this problem, the "SOTA"
# column of CORAL's Table 1 (CORAL's own single-agent Opus 4.6 result there is 0.38089).
# It is rendered to the agent as "target" and never attributed.
TARGET = 0.38088

# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone. 600 s matches CORAL's grader timeout.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 600.0))


def _run_solution(solution_dir: Path, timeout: float) -> dict:
    """Call run() in a fresh subprocess. Returns the parsed status dict.

    The script is CORAL's `_run_evaluation` up to the point where run() has returned,
    then it emits what it got instead of checking and scoring it."""
    code = (
        "import json, sys, os, time\n"
        "import numpy as np\n"
        "sys.path.insert(0, '.')\n"
        "import solution as program\n"
        "start = time.time()\n"
        "try:\n"
        "    h_values, c5_bound, n_points = program.run()\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': f'run() failed: {e}'}))\n"
        "    sys.exit(0)\n"
        "eval_time = time.time() - start\n"
        "h = np.array(h_values, dtype=float)\n"
        "if isinstance(n_points, np.generic):\n"
        "    n_points = n_points.item()\n"
        "print(json.dumps({'h': h.tolist(), 'c5_bound': float(c5_bound), 'n_points': n_points, "
        "'eval_time': eval_time}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        # CORAL's wording for the same event.
        raise RuntimeError(f"Evaluation timed out after {timeout:.0f}s") from None
    if out.returncode != 0:
        # CORAL: a non-zero exit surfaces the stderr tail as "Evaluation failed: …".
        detail = (out.stderr or "").strip()[-2000:] or (out.stdout or "").strip()[-500:]
        if out.returncode < 0 or out.returncode in (134, 137, 139):
            detail = (f"exit code {out.returncode} (killed — likely the memory cap or "
                      f"another resource limit) {detail}")
        raise RuntimeError(f"Evaluation failed: {detail or 'no output produced'}")
    stdout = out.stdout.strip()
    if not stdout:
        raise RuntimeError("Evaluation failed: Script produced no output.\n"
                           f"stderr: {out.stderr.strip()[-1000:]}")
    # Tolerant parse, as theirs: the last line that is a JSON object wins.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"Evaluation failed: No valid JSON in output.\nstdout: {stdout[-500:]}")


def _check_and_score(status: dict) -> dict:
    """CORAL's validation and scoring, verbatim, on what run() returned.

    Returns a dict with either "error" (their message) or the scored fields."""
    h = np.array(status["h"], dtype=float)
    c5_bound = status["c5_bound"]
    n_points = status["n_points"]

    # Validate shape
    if h.shape != (n_points,):
        return {"error": f"Expected h shape ({n_points},), got {h.shape}"}

    # Validate h(x) in [0, 1]
    if np.any(h < 0) or np.any(h > 1):
        return {"error": f"h(x) not in [0,1]. Range: [{h.min()}, {h.max()}]"}

    # Validate integral of h = 1
    dx = 2.0 / n_points
    integral_h = np.sum(h) * dx
    if not np.isclose(integral_h, 1.0, atol=1e-3):
        return {"error": f"Integral of h is not close to 1. Got: {integral_h:.6f}"}

    # Recompute C5 via cross-correlation
    j = 1.0 - h
    correlation = np.correlate(h, j, mode="full") * dx
    computed_c5 = float(np.max(correlation))

    # Verify consistency
    if not np.isclose(computed_c5, float(c5_bound), atol=1e-4):
        return {"error": f"C5 mismatch: reported {c5_bound:.6f}, computed {computed_c5:.6f}"}

    # Scored on the value the submitted h actually attains (see the module docstring).
    return {
        "c5_bound": computed_c5,
        "reported_c5_bound": float(c5_bound),
        "combined_score": BENCHMARK / computed_c5,
        "n_points": int(n_points),
        "integral_h": float(integral_h),
    }


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        status = _run_solution(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — a bad candidate scores 0, never crashes us
        return {"score": 0.0, "stage": "error", "error": str(e)[:400],
                "time": time.perf_counter() - t0}
    if "error" in status:
        # CORAL prefixes these with "Error: " in the feedback it returns.
        return {"score": 0.0, "stage": "invalid", "error": f"Error: {status['error']}"[:400],
                "time": time.perf_counter() - t0}

    try:
        result = _check_and_score(status)
    except Exception as e:  # noqa: BLE001 — in CORAL an exception here is the subprocess
        # dying, reported as "Evaluation failed: <traceback>".
        return {"score": 0.0, "stage": "error",
                "error": f"Evaluation failed: {type(e).__name__}: {e}"[:400],
                "time": time.perf_counter() - t0}
    if "error" in result:
        return {"score": 0.0, "stage": "invalid", "error": f"Error: {result['error']}"[:400],
                "time": time.perf_counter() - t0}

    score = float(result["combined_score"])
    if not math.isfinite(score) or score <= 0:
        return {"score": 0.0, "stage": "invalid",
                "error": f"Error: C5 bound {result['c5_bound']!r} gives no finite score",
                "time": time.perf_counter() - t0}

    eval_time = float(status.get("eval_time", 0.0))
    return {
        "score": score,
        "stage": "full",
        # The keys the briefing and the steering messages read: C5 itself, minimised.
        "objective": result["c5_bound"],
        "objective_name": "C5 bound",
        "objective_direction": "min",
        "objective_target": TARGET,
        # CORAL's fields, as their grader reports them.
        "c5_bound": result["c5_bound"],
        "reported_c5_bound": result["reported_c5_bound"],
        "combined_score": score,
        "n_points": result["n_points"],
        "eval_time": eval_time,
        "benchmark": BENCHMARK,
        "new_record": score > 1.0,
        "target": TARGET,
        "solve_time": eval_time,
        "time": time.perf_counter() - t0,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
