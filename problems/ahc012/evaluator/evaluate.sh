#!/bin/sh
# Standalone ALE-Bench adapter; one JSON metrics line on stdout.
exec python3 "$(dirname "$0")/evaluator.py" "$1"
