#!/bin/sh
# Entry point of the labs evaluator bundle. Contract (SkyDiscover-compatible):
#   evaluate.sh <solution_path>   →  exactly one JSON line on stdout.
# Runs inside the evaluation sandbox: no network, hard resource limits.
exec python3 "$(dirname "$0")/evaluator.py" "$1"
