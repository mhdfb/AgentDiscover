#!/bin/bash
#SBATCH --job-name=agentdiscover_gpu
#SBATCH --output=logs/agentdiscover_%j.out
#SBATCH --error=logs/agentdiscover_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2                     # e.g. gpu:a100:2 or gpu:h200:2
#SBATCH --time=4-00:00:00
##SBATCH --partition=<gpu-partition>     # uncomment and set for your cluster
##SBATCH --qos=<qos>                     # uncomment and set if your cluster needs one

# SLURM wrapper for run.sh on a GPU problem (problem.toml [limits] gpu = true: trimul,
# mla_decode). Two GPUs are requested: the evaluator times kernels on the first and the
# agent's own test runs use the second. With one GPU set both GPU variables below to "0"
# and accept noisier timings.
#
#   sbatch examples/slurm-gpu.sh                              # TriMul on 2 GPUs
#   PROBLEM=mla_decode sbatch examples/slurm-gpu.sh           # MLA-Decode
#
# Stop it with `touch runs/<problem>/STOP`, or scancel the job.

set -euo pipefail

# ---- the run's inputs (see run.sh for every knob and its default) ----
# export AGENTDISCOVER_USER="researcher"   # owns the agents; defaults to your login name
export PROBLEM="${PROBLEM:-trimul}"
export RUN_MODE="${RUN_MODE:-continue}"

export ITERATIONS="${ITERATIONS:-12}"
export PROPOSALS_PER_ITER="${PROPOSALS_PER_ITER:-8}"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-28800}"      # 8 h
export META_TIMEOUT="${META_TIMEOUT:-7200}"
export META_EVERY="${META_EVERY:-1}"
export META_HARNESS="${META_HARNESS:-claude}"
export META_MODEL="${META_MODEL:-opus}"
export AGENTS="${AGENTS:-researcher-a:claude:opus}"

# Which GPU each container sees (CUDA_VISIBLE_DEVICES, numbered within the allocation).
export AGENTDISCOVER_EVAL_GPUS="${AGENTDISCOVER_EVAL_GPUS:-0}"
export AGENTDISCOVER_AGENT_GPUS="${AGENTDISCOVER_AGENT_GPUS:-1}"

# export SANDBOX_BACKEND=singularity    # auto-detected; uncomment to force

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
echo "node $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
exec "$REPO_ROOT/run.sh"
