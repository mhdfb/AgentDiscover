#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
#
# Pin the thread pools to one thread: torch sizes its intra-op pool from the host's core
# count, and the per-thread arenas overrun the sandbox's address-space cap on a big
# machine. It also keeps the timing half of the score from depending on how many cores
# happened to be free — this metric is half speed.
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS

exec python3 "$(dirname "$0")/evaluator.py" "$1"
