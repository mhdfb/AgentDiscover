#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
#
# One thread per pool for the scorer too: its own work (two log-normalisations of a
# 1,937 x 15,000 matrix) is small, and numpy's OpenBLAS would otherwise reserve a
# per-thread arena for every core of the host at import. The candidate's process gets
# the same caps plus a one-core affinity from evaluator.py.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 \
  exec python3 "$(dirname "$0")/evaluator.py" "$1" --run-dir "$(pwd)/run"
