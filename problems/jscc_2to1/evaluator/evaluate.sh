#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
#
# Threads are pinned to 1. numpy's BLAS reserves a per-thread arena at import, which
# overruns the address-space cap on a many-core host before a single sample is drawn;
# pinning also makes the scoring cost independent of how many cores were free, which a
# timed budget needs.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  exec python3 "$(dirname "$0")/evaluator.py" "$1"
