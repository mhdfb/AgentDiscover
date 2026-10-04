"""Source-faithful evaluator for TTT-Discover's first autocorrelation inequality."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 1000.0))
CPU_LIMIT = 2
TARGET = 1.5030


def evaluate_sequence(sequence: list[float]) -> float:
    """The verifier from examples/ac_inequalities/env.py in the source repository."""
    if not isinstance(sequence, list) or not sequence:
        return np.inf
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return np.inf
        if np.isnan(value) or np.isinf(value):
            return np.inf

    sequence = [min(1000.0, max(0.0, float(value))) for value in sequence]
    n = len(sequence)
    convolution = np.convolve(sequence, sequence)
    total = np.sum(sequence)
    if total < 0.01:
        return np.inf
    return float(2 * n * max(convolution) / (total**2))


def _run_solve(solution_dir: Path) -> tuple[list, float]:
    code = (
        "import json, sys, time\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t=time.perf_counter(); value=solve(); elapsed=time.perf_counter()-t\n"
        "print(json.dumps({'value': value, 'time': elapsed}))\n"
    )
    solve_env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        solve_env[variable] = str(CPU_LIMIT)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=solution_dir, capture_output=True,
            text=True, timeout=SOLVE_TIMEOUT, env=solve_env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"solve() exceeded its {SOLVE_TIMEOUT:.0f}s budget") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-500:]
        raise RuntimeError(f"solve() failed with exit code {proc.returncode}: {detail}")
    for line in reversed(proc.stdout.splitlines()):
        if line.lstrip().startswith("{"):
            payload = json.loads(line)
            return payload["value"], payload["time"]
    raise RuntimeError("solve() produced no JSON result")


def evaluate(solution_path: Path) -> dict:
    started = time.perf_counter()
    try:
        sequence, solve_time = _run_solve(solution_path.parent)
        bound = evaluate_sequence(sequence)
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(exc)[:300],
                "time": time.perf_counter() - started}
    if not math.isfinite(bound):
        return {"score": 0.0, "stage": "invalid", "error": "invalid height sequence",
                "time": time.perf_counter() - started}
    return {
        "score": 1.0 / (1e-8 + bound),
        "stage": "full",
        "objective": bound,
        "objective_name": "C1 upper bound",
        "objective_direction": "min",
        "objective_target": TARGET,
        "upper_bound": bound,
        "n_intervals": len(sequence),
        "target": TARGET,
        "solve_time": solve_time,
        "time": time.perf_counter() - started,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))
