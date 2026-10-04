#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
#
# Pin the BLAS thread pools to one thread: numpy's OpenBLAS reserves a per-thread
# memory arena at import, which can overrun the sandbox's address-space cap on a
# many-core host before a single case is placed, and the task has a 10 s per-case
# budget, so the score must not depend on how many cores happened to be free.
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS

exec python3 "$(dirname "$0")/evaluator.py" "$1"
