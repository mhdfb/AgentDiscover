#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
#
# Pin the BLAS thread pools to one thread. numpy/scipy's OpenBLAS reserves a per-thread
# memory arena at import, sized from the host's core count, and on a big machine that
# overruns the sandbox's address-space cap — "OpenBLAS error: Memory allocation still
# failed after 10 retries", before any optimisation starts. Seen here on a candidate that
# imported scipy, while the numpy-only seed had passed.
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS

exec python3 "$(dirname "$0")/evaluator.py" "$1"
