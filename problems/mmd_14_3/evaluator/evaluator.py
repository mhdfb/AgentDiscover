"""MMD-14-3 — 14 points in R^3 minimising the max/min pairwise distance ratio. CORAL's
metric, unchanged.

Ported from CORAL's `examples/math/minimizing_max_min_dist_3d` (grader.py), the bundle
behind their published figure. The task, the metric and the feedback are theirs:

  * The candidate defines `run()` returning an array of shape (14, 3).
  * Pairwise distances come from `scipy.spatial.distance.pdist`, exactly as there;
    `min_max_ratio = (min_distance / max_distance) ** 2` and
    `combined_score = min_max_ratio / BENCHMARK`, with `BENCHMARK = 1 / 4.165849767`,
    higher is better and > 1 is past the benchmark. That score IS the platform's
    fitness here, so the number the search ranks on is the number CORAL reports.
  * Validation and its messages are theirs verbatim: shape must be (14, 3), no NaN,
    max distance must be positive; a `run()` that raises reports "run() failed: …";
    coincident points are not an error — they make the ratio 0 and the score 0, as
    in the original.
  * The evaluation returns their fields — `min_max_ratio`, `combined_score`,
    `min_distance`, `max_distance`, `eval_time`, the benchmark and whether the score
    passed 1 ("NEW RECORD!" in their grader) — and nothing finer.
  * The 600 s budget is their `grader.timeout: 600`.

What differs is only the process layout, to fit this platform:

  * `run()` executes in a fresh subprocess (`_run_solution`) that prints the points and
    exits; the distances and the score are computed HERE, afterwards, with the same
    scipy call — so a candidate can never reach the scoring code or report a number
    for itself. In CORAL's layout the score is computed in the same subprocess that
    imported the candidate.
  * The interpretable objective published to the briefing is `ratio^2 =
    (max_distance / min_distance) ** 2`, minimised — the form CORAL's paper reports
    (4.16) — alongside their (min/max)^2 and score. It is the same number inverted.
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
import scipy.spatial.distance

NUM_POINTS = 14
DIMENSION = 3
# CORAL's normaliser: the best known (max/min)^2 for 14 points in R^3, so that a score
# of 1.0 is matching it. Their task text states it to the agent, so it is not hidden.
BENCHMARK = 1 / 4.165849767

# The objective the briefing tracks is ratio^2 = (max/min)^2, minimised. TARGET is a bar
# to BEAT by a clear margin, not to meet: 4.16 is the figure CORAL's paper reports for
# this task (Table 1, single agent, Claude Opus 4.6), which is how we compare. It is
# rendered to the agent as "target" and never attributed.
TARGET = 4.16

# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone. 600 s matches CORAL's grader timeout.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 600.0))


def _run_solution(solution_dir: Path, timeout: float) -> dict:
    """Call run() in a fresh subprocess. Returns the parsed status dict.

    The script is CORAL's `_run_evaluation` up to the point where the points are
    known, then it emits them instead of scoring them."""
    code = (
        "import json, sys, os, time\n"
        "import numpy as np\n"
        f"NUM_POINTS = {NUM_POINTS}\n"
        f"DIMENSION = {DIMENSION}\n"
        "sys.path.insert(0, '.')\n"
        "import solution as program\n"
        "start = time.time()\n"
        "try:\n"
        "    points = program.run()\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': f'run() failed: {e}'}))\n"
        "    sys.exit(0)\n"
        "eval_time = time.time() - start\n"
        "if not isinstance(points, np.ndarray):\n"
        "    points = np.array(points)\n"
        "if points.shape != (NUM_POINTS, DIMENSION):\n"
        "    print(json.dumps({'error': f'Invalid shape: {points.shape}, expected ({NUM_POINTS}, {DIMENSION})'}))\n"
        "    sys.exit(0)\n"
        "if np.isnan(points).any():\n"
        "    print(json.dumps({'error': 'NaN values detected in points'}))\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'points': np.asarray(points, dtype=float).tolist(), 'eval_time': eval_time}))\n"
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

    points = np.asarray(status["points"], dtype=float)
    eval_time = float(status.get("eval_time", 0.0))
    if points.shape != (NUM_POINTS, DIMENSION) or np.isnan(points).any():
        # Cannot happen unless the subprocess was tampered with; say so plainly.
        return {"score": 0.0, "stage": "invalid",
                "error": "the points returned to the evaluator do not match what was validated",
                "time": time.perf_counter() - t0}

    # CORAL's scoring, verbatim, on the returned points.
    pairwise_distances = scipy.spatial.distance.pdist(points)
    min_distance = float(np.min(pairwise_distances))
    max_distance = float(np.max(pairwise_distances))
    if max_distance <= 0:
        return {"score": 0.0, "stage": "invalid", "error": "Error: Max distance is zero or negative",
                "time": time.perf_counter() - t0}
    inv_ratio_squared = (min_distance / max_distance) ** 2
    score = float(inv_ratio_squared / BENCHMARK)
    if not math.isfinite(score):
        # Infinite coordinates get here; CORAL would carry the NaN into its score.
        return {"score": 0.0, "stage": "invalid", "error": "Error: non-finite distances",
                "time": time.perf_counter() - t0}

    result = {
        "score": score,
        "stage": "full",
        "objective_name": "ratio^2 (max/min)",
        "objective_direction": "min",
        "objective_target": TARGET,
        # CORAL's fields, as their grader reports them.
        "min_max_ratio": inv_ratio_squared,
        "combined_score": score,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "eval_time": eval_time,
        "benchmark": BENCHMARK,
        "new_record": score > 1.0,
        "target": TARGET,
        "solve_time": eval_time,
        "time": time.perf_counter() - t0,
    }
    if inv_ratio_squared > 0:
        # The number the agent reasons about, at full precision. Undefined when two
        # points coincide — then the score is 0, as in the original, and no objective.
        result["objective"] = 1.0 / inv_ratio_squared
        result["ratio_squared"] = 1.0 / inv_ratio_squared
    return result


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
