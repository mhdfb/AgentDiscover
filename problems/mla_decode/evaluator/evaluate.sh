#!/bin/sh
# Entry point of the evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits, one GPU.
# The harness runs under the staging directory (the sandbox's own /tmp is tens of MB,
# too small for the Triton compile cache); the platform removes it afterwards.
exec python3 "$(dirname "$0")/evaluator.py" "$1" --run-dir "$(pwd)/run"
