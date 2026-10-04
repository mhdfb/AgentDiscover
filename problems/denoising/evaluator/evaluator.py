"""Single-cell denoising — how a candidate is scored. This file is readable at
`/resources/evaluator.py` and runnable there (`python3 /resources/evaluator.py
solution.py` prints the same JSON line the platform gets, given the data set at
$AGENTDISCOVER_DATA_DIR); it is the authority on the contract PROBLEM.md describes.

The task is the OpenProblems denoising benchmark as TTT-Discover runs it
(examples/denoising/utils.py `run_denoising_eval`), step for step:

  1. the pancreas data set (figshare 36086813) restricted to the inDrop1 batch, empty
     genes and cells removed — `openproblems.data.pancreas.load_pancreas(test=False,
     keep_techs=["inDrop1"])`, v1.0.0, prepared once by fetch.sh and read from
     `$AGENTDISCOVER_DATA_DIR/pancreas_indrop1.h5ad` as their cached loader reads it;
  2. molecular cross-validation: every count is split binomially, 90 % into a training
     matrix and 10 % into a held-out test matrix, with seed 42 —
     `openproblems.tasks.denoising.datasets.utils.split_data(adata, seed=42)`;
  3. your `magic_denoise(X_train, random_state=42)` runs on the dense training matrix,
     alone in its own process, pinned to ONE CPU core with every BLAS/OpenMP/numba thread
     pool capped at one thread, within the budget PROBLEM.md states — the budget's clock
     starts after the allowed libraries are imported, so loading them costs you nothing;
  4. the result must be finite, non-negative, and no entry may exceed the training
     matrix's total count — otherwise it is invalid, as in theirs;
  5. MSE in log-normalised space and the Poisson negative log-likelihood against the
     held-out matrix, with TTT-Discover's `evaluate_mse` / `evaluate_poisson` verbatim
     (see openproblems_vendored.py);
  6. the Poisson gate: normalised Poisson `(0.257575 - poisson) / (0.257575 - 0.031739)`
     must be at least 0.97 and at most 1 (their `verify_denoising`); otherwise the
     candidate is "Invalid solution." and scores 0.

The objective is the MSE, minimised. Fitness is TARGET_MSE / mse: 1.0 at the target,
above 1 past it. The normalised scores OpenProblems reports are returned alongside.

Your process never sees the held-out matrix: it receives the training matrix as a file,
writes its denoised matrix to a file, and exits; the split and the metrics live in this
parent process. The data set directory is mounted read-only in the evaluation and in your
sandbox alike.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from openproblems_vendored import (BASELINES, evaluate_mse, evaluate_poisson,  # noqa: E402
                                   split_data)

DATASET = "pancreas"
KEEP_TECHS = ["inDrop1"]
SEED = 42
# The bar to BEAT, not to meet: TTT-Discover's published final program on this split, as
# measured by this evaluator on 2026-10-01 (Poisson 0.032741). The starting program, MAGIC
# with reversed normalisation, measures 0.231412 / Poisson 0.036922 here (TTT-Discover's
# initial state reports the same to four digits: 0.2316 / 0.0370).
TARGET_MSE = 0.165521
POISSON_GATE = 0.97

# The candidate's budget. The orchestrator passes the per-problem figure so the number
# the agent is told and the number enforced here cannot drift; the fallback keeps the
# bundle runnable standalone (TTT-Discover's statement: 400 s).
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 400.0))
# One core, one thread per pool: TTT-Discover's sandbox pins each evaluation to a CPU
# group of one core and caps every thread pool to its size.
CANDIDATE_CPUS = 1
# Wall clock allowed for the candidate process's library imports before its budget
# starts (seconds). Measured ~2 min off this cluster's network file system.
IMPORT_ALLOWANCE = 600


def _data_file() -> Path:
    d = os.environ.get("AGENTDISCOVER_DATA_DIR")
    if not d:
        raise SystemExit("AGENTDISCOVER_DATA_DIR is not set: the pancreas data set (fetch.sh) "
                         "must be mounted for this evaluation")
    f = Path(d) / "pancreas_indrop1.h5ad"
    if not f.exists():
        raise SystemExit(f"{f} not found — fetch.sh did not complete")
    return f


def prepare_split():
    """(X_train, X_test) as dense float64, exactly as run_denoising_eval builds them:
    the cached `load_pancreas(test=False, keep_techs=["inDrop1"])` object (written by
    fetch_prepare.py, see there), then `split_data(adata, seed=42)`."""
    import anndata
    import scprep

    adata = anndata.read_h5ad(str(_data_file()))
    adata = split_data(adata, seed=SEED)
    X_train = scprep.utils.toarray(adata.obsm["train"])
    X_test = scprep.utils.toarray(adata.obsm["test"])
    return X_train, X_test


_CANDIDATE = r"""
import os, sys, signal, time, traceback
# One core for the candidate, inherited by anything it spawns.
try:
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:__CPUS__]))
except Exception:
    pass
# The libraries the task statement allows are loaded BEFORE the clock starts: in
# TTT-Discover's sandbox they come off local disk in seconds; here they may come off a
# network file system, which is the platform's cost, not the candidate's.
import numpy as np
import scipy, scipy.sparse, sklearn, sklearn.decomposition, sklearn.neighbors
import graphtools, scprep, scanpy, anndata, magic
_emit = sys.stdout
sys.stdout = sys.stderr          # the candidate's own printing goes to the operator log
src = open(__SOLUTION__).read()
ns = {"__name__": "__candidate__"}
X = np.load(__XTRAIN__)
def _out_of_time(signum, frame):
    raise TimeoutError("budget")
signal.signal(signal.SIGALRM, _out_of_time)
signal.alarm(int(__BUDGET__))     # the budget the task statement names, from here on
t0 = time.perf_counter()
try:
    exec(src, ns)
    if "magic_denoise" not in ns:
        raise RuntimeError("solution.py defines no magic_denoise")
    Y = ns["magic_denoise"](X, random_state=__SEED__)
    Y = np.asarray(Y, dtype=np.float64)
    signal.alarm(0)
    np.save(__OUT__, Y)
    _emit.write("solve_time %.1f\n" % (time.perf_counter() - t0))
except TimeoutError:
    sys.exit(4)
except BaseException:
    traceback.print_exc()
    sys.exit(3)
"""

def _run_candidate(solution: Path, x_train: Path, out: Path, budget: float) -> dict:
    code = (_CANDIDATE.replace("__CPUS__", str(CANDIDATE_CPUS))
            .replace("__SOLUTION__", repr(str(solution)))
            .replace("__XTRAIN__", repr(str(x_train)))
            .replace("__OUT__", repr(str(out)))
            .replace("__BUDGET__", str(int(budget)))
            .replace("__SEED__", str(SEED)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTDISCOVER_")}
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        env[var] = str(CANDIDATE_CPUS)
    # The budget is enforced inside the process, after its library imports; this outer
    # clock only catches a process that never gets that far.
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              timeout=budget + IMPORT_ALLOWANCE, cwd=str(solution.parent),
                              env=env)
    except subprocess.TimeoutExpired:
        return {"error_class": "timeout", "solve_time": budget}
    tail = (proc.stderr or "")[-2000:]
    if proc.stderr:
        sys.stderr.write(proc.stderr[-4000:])
    solve_time = 0.0
    for line in (proc.stdout or "").splitlines():
        if line.startswith("solve_time "):
            solve_time = float(line.split()[1])
    if proc.returncode == 4:
        return {"error_class": "timeout", "solve_time": budget}
    if proc.returncode != 0:
        return {"error_class": "exit", "detail": tail, "solve_time": solve_time}
    if not out.exists():
        return {"error_class": "no_output", "detail": tail, "solve_time": solve_time}
    return {"solve_time": solve_time}


def _fail(stage: str, error: str, t0: float, **extra) -> dict:
    out = {"score": 0.0, "stage": stage, "error": error, "time": time.perf_counter() - t0}
    out.update(extra)
    return out


def evaluate(solution_path: Path, run_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    X_train, X_test = prepare_split()
    t_prep = time.perf_counter() - t0

    made_tmp = run_dir is None
    work = Path(tempfile.mkdtemp(prefix="denoise-")) if made_tmp else run_dir
    work.mkdir(parents=True, exist_ok=True)
    try:
        x_path, y_path = work / "X_train.npy", work / "denoised.npy"
        np.save(x_path, X_train)
        ran = _run_candidate(solution_path.resolve(), x_path, y_path, SOLVE_TIMEOUT)
        solve_time = round(float(ran.get("solve_time", 0.0)), 1)
        if "error_class" in ran:
            if ran["error_class"] == "timeout":
                return _fail("timeout", f"Evaluation timed out after {SOLVE_TIMEOUT:.0f}s",
                             t0, solve_time=solve_time)
            return _fail("error", "Evaluation failed: "
                         + (ran.get("detail") or ran["error_class"]).strip()[-1500:],
                         t0, solve_time=solve_time)
        Y = np.load(y_path)
    finally:
        if made_tmp:
            shutil.rmtree(work, ignore_errors=True)

    # run_denoising_eval's validity checks, then verify_denoising's gate.
    if Y.shape != X_train.shape:
        return _fail("invalid", f"Invalid solution: shape {Y.shape} != {X_train.shape}",
                     t0, solve_time=solve_time)
    if not np.isfinite(Y).all():
        return _fail("invalid", "Invalid solution: non-finite values", t0, solve_time=solve_time)
    if np.any(Y < 0):
        return _fail("invalid", "Invalid solution: negative values", t0, solve_time=solve_time)
    if Y.max() > X_train.sum():
        return _fail("invalid", "Invalid solution: an entry exceeds the total count",
                     t0, solve_time=solve_time)

    mse = float(evaluate_mse(X_test, Y))
    poisson = float(evaluate_poisson(X_train, X_test, Y))
    base = BASELINES[DATASET]
    mse_norm = (base["baseline_mse"] - mse) / (base["baseline_mse"] - base["perfect_mse"])
    poisson_norm = ((base["baseline_poisson"] - poisson)
                    / (base["baseline_poisson"] - base["perfect_poisson"]))
    extra = {"mse": round(mse, 6), "poisson": round(poisson, 6),
             "mse_normalized": round(mse_norm, 4), "poisson_normalized": round(poisson_norm, 4),
             "solve_time": solve_time, "prep_time": round(t_prep, 1)}
    if not (np.isfinite(mse) and np.isfinite(poisson)):
        return _fail("invalid", "Invalid solution: non-finite metrics", t0, **extra)
    if poisson < base["perfect_poisson"] or poisson_norm < POISSON_GATE:
        return _fail("invalid", f"Invalid solution: normalised Poisson {poisson_norm:.4f} is "
                                f"outside [{POISSON_GATE}, 1]", t0, **extra)

    clipped = lambda v: max(0.0, min(1.0, v))  # noqa: E731 — OpenProblems clips to [0, 1]
    return {
        "score": float(TARGET_MSE / mse),
        "stage": "full",
        "objective": round(mse, 6),
        "objective_name": "mse",
        "objective_direction": "min",
        "objective_target": TARGET_MSE,
        "openproblems_score": round((clipped(mse_norm) + clipped(poisson_norm)) / 2, 4),
        "target_mse": TARGET_MSE,
        "dataset": f"{DATASET}/{'+'.join(KEEP_TECHS)}",
        "cells": int(X_train.shape[0]),
        "genes": int(X_train.shape[1]),
        **extra,
        "time": time.perf_counter() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("solution")
    p.add_argument("--run-dir", default=None,
                   help="where the candidate runs (default: a fresh temporary directory)")
    a = p.parse_args()
    print(json.dumps(evaluate(Path(a.solution).resolve(),
                              Path(a.run_dir).resolve() if a.run_dir else None)))


if __name__ == "__main__":
    main()
