#!/usr/bin/env bash
# AgentDiscover launcher — machine agnostic.
#
# Every input is an environment variable with a default. Cluster wrappers set the
# variables and exec this script — see examples/slurm.sh.
#
#   AGENTS               space-separated "name:harness:model" entries
#   PROBLEM              selects problems/<name>/
#   RUN_MODE             continue (default) | fresh. continue carries on from the last
#                        iteration in the database. fresh first archives this problem's
#                        graph, worktrees and logs to runs/_archive/; nothing is deleted.
#   ITERATIONS           iterations THIS invocation adds per agent
#   PROPOSALS_PER_ITER   candidates per session
#   META_EVERY           iterations between meta passes
#   META_HARNESS/MODEL   the meta agent's harness and model
#   AGENT_TIMEOUT        seconds one agent session may run. Raised automatically, and
#                        logged, if it cannot fit PROPOSALS_PER_ITER evaluations.
#   META_TIMEOUT         seconds one meta pass may run
#   AGENTDISCOVER_WEB_ACCESS
#                        1 gives agents and meta passes the harness's web search/fetch
#                        tools (default 0). Bash in the container keeps network access.
#   AGENTDISCOVER_EFFORT reasoning effort passed to the harness (claude: low..max,
#                        codex: minimal..xhigh). Unset: the harness's default.
#   AGENTDISCOVER_BUDGET_USD
#                        stop the run once its spend at list prices reaches this.
#                        Checked after each session; unset: no cap.
#   AGENTDISCOVER_EVAL_GPUS, AGENTDISCOVER_AGENT_GPUS
#                        GPU problems only (problem.toml [limits] gpu = true): the
#                        CUDA_VISIBLE_DEVICES of evaluations and of search sessions. Both
#                        default to "0"; with two GPUs set the second to "1".
#   AGENTDISCOVER_USER   owns the agents (defaults to your login name)
#   SANDBOX_BACKEND      docker|podman|apptainer|singularity|enroot|none (auto-detected)
#
# Host requirements: bash, git, uv, and one container runtime.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export AGENTDISCOVER_ROOT="$REPO_ROOT"
export PATH="$HOME/.local/bin:$PATH"   # where the uv installer puts uv

command -v uv >/dev/null 2>&1 || {
    echo "uv not found — run ./script/setup first (installs it without root)." >&2
    exit 1
}

# .env is loaded here, outside any sandbox; the file is never mounted into a container.
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

: "${AGENTDISCOVER_USER:=$(id -un)}"
: "${PROBLEM:=labs}"
: "${RUN_MODE:=continue}"
: "${ITERATIONS:=1}"
: "${PROPOSALS_PER_ITER:=2}"
: "${META_EVERY:=50}"
: "${META_HARNESS:=claude}"
: "${META_MODEL:=haiku}"
: "${AGENTS:=researcher-a:claude:haiku}"

# Export every setting agentdiscover/runconfig.py reads: `: "${X:=default}"` assigns but
# does not export, so a value defaulted here would otherwise be invisible to it.
export AGENTDISCOVER_PROPOSALS="$PROPOSALS_PER_ITER"
export META_EVERY AGENT_TIMEOUT META_TIMEOUT AGENTDISCOVER_TOOL_TIMEOUT AGENTDISCOVER_IMAGE
export AGENTDISCOVER_WEB_ACCESS AGENTDISCOVER_EVAL_GPUS AGENTDISCOVER_AGENT_GPUS

RUNS_DIR="$REPO_ROOT/runs/$PROBLEM"
mkdir -p "$RUNS_DIR"

uv sync -q                                     # idempotent; creates .venv on first run
BACKEND="${SANDBOX_BACKEND:-$(uv run python -m agentdiscover.sandbox)}"
echo "sandbox backend: $BACKEND"

run() { uv run python -m "$@"; }

# ---------------------------------------------------------------------------
# Services: database + MCP service. A missing service stops the run here.
# ---------------------------------------------------------------------------
# The fresh-run reset must come before `ensure`, which would reuse the existing database.
case "$RUN_MODE" in
    fresh)    run agentdiscover.orchestrator.services reset --problem "$PROBLEM" --backend "$BACKEND" ;;
    continue) ;;
    *)        echo "RUN_MODE must be 'continue' or 'fresh', got '$RUN_MODE'" >&2; exit 1 ;;
esac

run agentdiscover.orchestrator.services ensure --problem "$PROBLEM" --backend "$BACKEND"
run agentdiscover.orchestrator.resources --problem "$PROBLEM" --backend "$BACKEND"

# ---------------------------------------------------------------------------
# Per-agent setup
# ---------------------------------------------------------------------------
read -ra AGENT_LIST <<< "$AGENTS"
declare -A AGENT_HARNESS AGENT_MODEL AGENT_START AGENT_BIAS

run agentdiscover.orchestrator.runlog --problem "$PROBLEM" --header \
    "job ${SLURM_JOB_ID:-local} | agents: $AGENTS"

bias_idx=1
for entry in "${AGENT_LIST[@]}"; do
    name="${entry%%:*}"; rest="${entry#*:}"
    harness="${rest%%:*}"; model="${rest#*:}"

    line="$(run agentdiscover.orchestrator.setup_agent \
        --problem "$PROBLEM" --user "$AGENTDISCOVER_USER" --name "$name")"
    start_iter="$(sed -n 's/.*start_iteration=\([0-9]*\).*/\1/p' <<< "$line")"
    state="$(sed -n 's/.*worktree=\([a-z]*\).*/\1/p' <<< "$line")"

    AGENT_HARNESS["$name"]="$harness"
    AGENT_MODEL["$name"]="$model"
    AGENT_START["$name"]="$start_iter"
    AGENT_BIAS["$name"]="$bias_idx"

    report="[$name] start_iteration=$start_iter  $harness/$model   worktree $state"
    echo "$report"
    run agentdiscover.orchestrator.runlog --problem "$PROBLEM" --line "$report"
    bias_idx=$((bias_idx + 1))
done

# `touch runs/<problem>/STOP` ends a run safely: each agent stops after its current
# iteration. Remove any leftover from an earlier run.
rm -f "$RUNS_DIR/STOP"

# ---------------------------------------------------------------------------
# The iteration loop: parallel across agents, sequential within one, with a
# barrier before each meta pass.
# ---------------------------------------------------------------------------

run_agent_rounds() {
    local name="$1" rel_start="$2" rel_end="$3"
    local harness="${AGENT_HARNESS[$name]}" model="${AGENT_MODEL[$name]}"
    local start="${AGENT_START[$name]}" bias="${AGENT_BIAS[$name]}"

    for rel in $(seq "$rel_start" "$rel_end"); do
        [[ -f "$RUNS_DIR/STOP" ]] && return 0
        local abs=$(( start + rel - 1 ))

        # Quota check: if the 5-hour window is spent, wait for its reset. Exit 2 = STOP
        # appeared during the wait; any other failure of the check is ignored.
        local qrc=0
        run agentdiscover.orchestrator.quota before-session \
            --problem "$PROBLEM" --label "$name iteration $abs" \
            --harness "$harness" --backend "$BACKEND" || qrc=$?
        [[ "$qrc" -eq 2 ]] && return 0

        local session_id="" jsonl=""
        while IFS='=' read -r k v; do
            case "$k" in
                session_id) session_id="$v" ;;
                jsonl)      jsonl="$v" ;;
            esac
        done < <(run agentdiscover.orchestrator.session open \
            --problem "$PROBLEM" --user "$AGENTDISCOVER_USER" --name "$name" \
            --harness "$harness" --model "$model" --iteration "$abs" \
            --bias-idx "$bias" --num-agents "${#AGENT_LIST[@]}" \
            --rel-iteration "$rel" --total-iterations "$ITERATIONS")

        # A failed `open` must stop this agent: launching anyway would reuse the previous
        # iteration's prompt and a dead session token.
        if [[ -z "$session_id" ]]; then
            local err="[$name] iteration $abs: session open FAILED — agent stopped; see the run output for the traceback"
            echo "$err" >&2
            run agentdiscover.orchestrator.runlog --problem "$PROBLEM" --line "$err"
            return 1
        fi

        local status
        status="$(run agentdiscover.orchestrator.launch \
            --problem "$PROBLEM" --backend "$BACKEND" --user "$AGENTDISCOVER_USER" \
            --name "$name" --harness "$harness" --model "$model" \
            --iteration "$abs" | sed -n 's/^status=//p')"

        # A session cut off by the 5-hour window is resumed after the reset (max 5 times).
        # Exit 3 = the window has reset, resume now; 2 = STOP appeared while waiting.
        local resumes=0
        while [[ "$resumes" -lt 5 ]]; do
            qrc=0
            run agentdiscover.orchestrator.quota after-session \
                --problem "$PROBLEM" --jsonl "$jsonl" \
                --label "$name iteration $abs" --harness "$harness" \
                --backend "$BACKEND" || qrc=$?
            [[ "$qrc" -eq 2 ]] && { touch "$RUNS_DIR/STOP"; break; }
            [[ "$qrc" -ne 3 ]] && break
            resumes=$(( resumes + 1 ))
            run agentdiscover.orchestrator.runlog --problem "$PROBLEM" \
                --line "[$name] iteration $abs: resuming after the quota reset (resume $resumes)"
            status="$(run agentdiscover.orchestrator.launch \
                --problem "$PROBLEM" --backend "$BACKEND" --user "$AGENTDISCOVER_USER" \
                --name "$name" --harness "$harness" --model "$model" \
                --iteration "$abs" --resume | sed -n 's/^status=//p')"
        done

        run agentdiscover.orchestrator.session close \
            --problem "$PROBLEM" --session-id "$session_id" \
            --jsonl "$jsonl" --harness "$harness" --status "${status:-unknown}"

        # Budget check: exit 4 = spent; STOP then ends every agent after its session.
        if [[ -n "${AGENTDISCOVER_BUDGET_USD:-}" ]]; then
            local brc=0
            run agentdiscover.orchestrator.spend --problem "$PROBLEM" \
                --budget "$AGENTDISCOVER_BUDGET_USD" || brc=$?
            [[ "$brc" -eq 4 ]] && { touch "$RUNS_DIR/STOP"; return 0; }
        fi
    done
}

rel_start=1
while [[ "$rel_start" -le "$ITERATIONS" ]]; do
    rel_end=$(( rel_start + META_EVERY - 1 ))
    [[ "$rel_end" -gt "$ITERATIONS" ]] && rel_end="$ITERATIONS"

    PIDS=()
    for entry in "${AGENT_LIST[@]}"; do
        run_agent_rounds "${entry%%:*}" "$rel_start" "$rel_end" &
        PIDS+=("$!")
    done
    trap 'echo "stopping..."; touch "$RUNS_DIR/STOP"; kill "${PIDS[@]}" 2>/dev/null || true' INT TERM
    wait

    [[ -f "$RUNS_DIR/STOP" ]] && break

    # The meta pass runs only when another block will read the guidance it writes.
    # A failed pass is skipped and never stops the run.
    if [[ "$rel_end" -lt "$ITERATIONS" ]]; then
        qrc=0
        run agentdiscover.orchestrator.quota before-session \
            --problem "$PROBLEM" --label "meta pass" --harness "$META_HARNESS" \
            --backend "$BACKEND" || qrc=$?
        [[ "$qrc" -eq 2 ]] && break
        run agentdiscover.orchestrator.meta \
            --problem "$PROBLEM" --backend "$BACKEND" --user "$AGENTDISCOVER_USER" \
            --harness "$META_HARNESS" --model "$META_MODEL" \
            --agents "$AGENTS" --meta-every "$META_EVERY" \
            || echo "[meta] pass failed — continuing with existing guidance"
    else
        msg="[meta] skipped after the final iteration — no session would read it"
        echo "$msg"
        run agentdiscover.orchestrator.runlog --problem "$PROBLEM" --line "$msg"
    fi

    rel_start=$(( rel_end + 1 ))
done

echo "run complete."
