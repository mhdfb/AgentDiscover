#!/bin/bash
#SBATCH --job-name=agentdiscover
#SBATCH --output=logs/agentdiscover_%j.out
#SBATCH --error=logs/agentdiscover_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00                  # hrs:min:sec
##SBATCH --partition=<partition>         # uncomment and set for your cluster
##SBATCH --qos=<qos>                     # uncomment and set if your cluster needs one

# SLURM wrapper for run.sh — all machine- and run-specific choices live here;
# run.sh itself is never edited.
#
# Copy this file per cluster/run, adjust the SBATCH header and the inputs below,
# then:  sbatch examples/slurm.sh     (or run it directly, no SLURM needed)
#
# No GPU is requested; for a GPU problem see examples/slurm-gpu.sh. The host needs no
# conda, module loads or Node, and the container runtime is detected automatically.

set -euo pipefail

# ---- the run's inputs (see run.sh for every knob and its default) ----
# export AGENTDISCOVER_USER="researcher"   # owns the agents; defaults to your login name
export PROBLEM="circle_packing_26"

# continue = add iterations to this problem's existing run (numbering carries on).
# fresh    = start a new run at iteration 1. The previous graph, worktrees and logs are
#            archived to runs/_archive/ first; nothing is deleted.
export RUN_MODE="continue"

export ITERATIONS=6
export PROPOSALS_PER_ITER=10
export META_EVERY=2
export META_HARNESS="claude"
export META_MODEL="haiku"
export AGENTS="researcher-a:claude:haiku"
# export AGENTS="researcher-a:claude:haiku researcher-c:codex:gpt-5.6-terra"

# Seconds one agent session, and one meta pass, may run. Each candidate blocks for up to
# the problem's solve_seconds (problems/<p>/problem.toml) while it is evaluated, so allow
# more than PROPOSALS_PER_ITER x solve_seconds; the launcher raises a value too small.
export AGENT_TIMEOUT=28800              # 8 h
export META_TIMEOUT=3600                # 1 h

# export SANDBOX_BACKEND=singularity    # auto-detected; uncomment to force

# Change all three to run two problems on the same node at once (defaults shown).
# export NEO4J_PORT=7687
# export NEO4J_HTTP_PORT=7474
# export AGENTDISCOVER_PORT=8900

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "$REPO_ROOT/run.sh"
