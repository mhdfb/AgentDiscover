#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
#
# BLAS threads are pinned to 1: scipy's OpenBLAS reserves a per-thread arena at import,
# which overruns the sandbox's address-space cap on a many-core host before a single
# distance is computed. With 14 points there is nothing for BLAS threads to speed up.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  exec python3 "$(dirname "$0")/evaluator.py" "$1"
