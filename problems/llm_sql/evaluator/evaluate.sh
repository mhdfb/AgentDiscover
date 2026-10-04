#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
#
# BLAS threads are pinned to 1. pandas imports numpy, whose OpenBLAS reserves a per-thread
# arena at import time — on a many-core host that alone overruns the sandbox's
# address-space cap before a single row is read. Nothing in this task uses BLAS, so the
# pin changes no result; the Python thread pools the seed and the metric use are
# unaffected by these variables.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  exec python3 "$(dirname "$0")/evaluator.py" "$1"
