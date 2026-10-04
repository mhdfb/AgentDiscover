"""Source-faithful evaluator for TTT-Discover's second autocorrelation inequality."""

import json
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# Match TTT-Discover's execution path: its configured eval_timeout is 1100 s,
# and run_eval_code passes eval_timeout + 5 to the candidate-process runner.
EVAL_TIMEOUT = float(
    os.environ.get("TTT_AC_EVAL_TIMEOUT_SECONDS", 1100.0)
)
CANDIDATE_TIMEOUT = EVAL_TIMEOUT + 5.0
CPU_LIMIT = 2
TARGET = 0.97


def evaluate_sequence(sequence: list[float]) -> float:
    """Verifier from examples/ac_inequalities/env.py at TTT-Discover 6c40e82."""
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")
    if not sequence:
        raise ValueError("Empty sequence")
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Invalid sequence element type")
        if np.isnan(value) or np.isinf(value):
            raise ValueError("Invalid sequence element value")

    sequence = [max(0, float(value)) for value in sequence]
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    sequence = [min(1000.0, value) for value in sequence]

    convolution = np.convolve(sequence, sequence)
    num_points = len(convolution)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution) + 1):
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        l2_norm_squared += (h / 3) * (y1**2 + y1 * y2 + y2**2)

    norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)
    norm_inf = np.max(np.abs(convolution))
    return float(l2_norm_squared / (norm_1 * norm_inf))


def _run_solve(solution_dir: Path) -> tuple[list, float]:
    result_path = solution_dir / "result.pkl"
    code = (
        "import concurrent.futures as cf\n"
        "import concurrent.futures.process as cfp\n"
        "import multiprocessing as mp\n"
        "import os, pickle, sys, time\n"
        "try:\n"
        "    mp.set_start_method('spawn', force=True)\n"
        "except RuntimeError:\n"
        "    pass\n"
        "if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):\n"
        "    allowed = sorted(os.sched_getaffinity(0))[:2]\n"
        "    if allowed:\n"
        "        os.sched_setaffinity(0, set(allowed))\n"
        "_OriginalExecutor = cfp.ProcessPoolExecutor\n"
        "class _CappedExecutor(_OriginalExecutor):\n"
        "    def __init__(self, max_workers=None, *args, **kwargs):\n"
        "        requested = max_workers if max_workers is not None else (os.cpu_count() or 1)\n"
        "        super().__init__(max_workers=max(1, min(int(requested), 2)), *args, **kwargs)\n"
        "cfp.ProcessPoolExecutor = _CappedExecutor\n"
        "cf.ProcessPoolExecutor = _CappedExecutor\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t=time.perf_counter(); value=solve(); elapsed=time.perf_counter()-t\n"
        f"with open({str(result_path)!r}, 'wb') as handle:\n"
        "    pickle.dump({'value': value, 'time': elapsed}, handle)\n"
    )
    solve_env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        solve_env[variable] = str(CPU_LIMIT)
    proc = subprocess.Popen(
        [sys.executable, "-c", code], cwd=solution_dir,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=solve_env, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=CANDIDATE_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=1.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.communicate()
        raise RuntimeError(
            f"solve() exceeded the source-faithful {CANDIDATE_TIMEOUT:.0f}s hard deadline"
        ) from None
    if proc.returncode != 0:
        detail = (stderr or stdout).strip()[-500:]
        raise RuntimeError(f"solve() failed with exit code {proc.returncode}: {detail}")
    if not result_path.is_file():
        raise RuntimeError("solve() produced no result")
    with result_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["value"], payload["time"]


def evaluate(solution_path: Path) -> dict:
    started = time.perf_counter()
    try:
        # Upstream evaluates one generated program file. Do not let a CORAL
        # attempt depend on auxiliary workspace artifacts unavailable to the
        # equivalent AgentDiscover candidate.
        with tempfile.TemporaryDirectory(prefix="ttt-ac2-") as tmp:
            isolated_solution = Path(tmp) / "solution.py"
            isolated_solution.write_bytes(solution_path.read_bytes())
            sequence, solve_time = _run_solve(isolated_solution.parent)
        bound = evaluate_sequence(sequence)
    except Exception as exc:  # noqa: BLE001
        return {
            "score": 0.0,
            "stage": "error",
            "error": str(exc)[:300],
            "time": time.perf_counter() - started,
        }
    if not math.isfinite(bound):
        return {
            "score": 0.0,
            "stage": "invalid",
            "error": "invalid height sequence",
            "time": time.perf_counter() - started,
        }
    return {
        "score": bound,
        "stage": "full",
        "objective": bound,
        "objective_name": "C2 lower bound",
        "objective_direction": "max",
        "objective_target": TARGET,
        "lower_bound": bound,
        "n_intervals": len(sequence),
        "target": TARGET,
        "solve_time": solve_time,
        "time": time.perf_counter() - started,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))
