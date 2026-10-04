"""EPLB — MoE expert parallelism load balancing. CORAL's metric, unchanged.

Ported from CORAL's `examples/ADRS/eplb/taskdata/evaluator.py`, which is the version that
produced their published figure. CORAL is the reference for this problem, so their
evaluator is the one copied here: `load_workloads`, `simulate_inference` and the scoring
in `evaluate` are theirs line for line, including

  * the fixed configuration NUM_REPLICAS=288, NUM_GROUPS=8, NUM_GPUS=32, NUM_NODES=4;
  * REBALANCE_INTERVAL=100, so the recorded vLLM load history becomes a sequence of
    workloads, and a rearrangement computed on workload i is scored on workload i+1;
  * `speed_score = 0.002 / avg_time_inference` and
    `combined_score = (avg_balancedness_score_expert + speed_score) / 2`.

CORAL's version differs from SkyDiscover's on two points, both listed in CORAL's
Appendix D.3 as corrections they made ("dropped experts and redundant averaging"):

  * an expert that carries load but was given no replicas is PENALISED — its load is
    concentrated on one physical slot, the worst case for balance — where SkyDiscover
    silently skipped it, so dropping an expert used to be free;
  * `balancedness_expert` is computed without SkyDiscover's `if expert_layer_max > 0`
    guard.

Two guards of ours, on top of CORAL's scoring, close holes that CORAL's evaluator (and,
for the second, SkyDiscover's too) leave open. Neither changes an honest candidate's
score — verified against CORAL and SkyDiscover:

  * `check_mapping` refuses a rearrangement that claims more expert copies than there are
    physical slots (`logcnt.sum` per layer > NUM_REPLICAS). Without it a candidate spreads
    every expert over imaginary slots and fakes near-perfect balance. This is the check in
    SkyDiscover's PR #59; CORAL lacks it. We keep `<=` (not their strict `==`) and keep
    CORAL's penalty for the dropped-expert case, so the number stays comparable to CORAL.
  * `_hidden_launch` runs the candidate in a mount namespace with the workload data dir
    blanked out, so its process cannot read the recorded history and hand the future back
    through its output. The candidate already ran in its own process (below); this also
    denies it the data on disk. CORAL and SkyDiscover run the candidate where it can reach
    the data, so both are open to this.

What differs from CORAL is only the output protocol, to fit this platform: they bridge
this module through SkyDiscover's `wrapper.py`, whereas here `main()` prints the one JSON
line the platform's contract asks for. The numbers come from the same code either way.

The workload data is `expert-load.json` beside this file — 224 MB of recorded vLLM load
history, downloaded separately (see PROBLEM.md), not stored in the repository.
"""
import collections
import contextlib
import functools
import io
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TypedDict

import torch

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# The platform fetches the data set once (fetch.sh) and mounts it read-only, naming the
# directory here. Falling back to this file's own directory keeps the bundle runnable
# standalone, the way CORAL's and SkyDiscover's are.
WORKLOAD_PATH = os.path.join(os.environ.get("AGENTDISCOVER_DATA_DIR", _CURRENT_DIR),
                             "expert-load.json")
REBALANCE_INTERVAL = 100

NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4

# The objective is `combined_score`, maximised. It has no analytic ceiling: the balance
# term is at most 1, but the speed term 0.002/avg_time_inference grows without bound as
# the rearrangement gets faster, so the metric is bounded only by how fast the simulation
# itself can run.
#
# TARGET is a bar to BEAT by a clear margin, not to meet.
TARGET = 0.149

# The candidate's budget for the whole sequence of rearrangements. The orchestrator passes
# the per-problem figure so the number the agent is told and the number enforced here
# cannot drift; the fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 300.0))


@functools.cache
def load_workloads(path: str) -> list[torch.Tensor]:
    with open(path, "r") as f:
        data = json.load(f)

    total_len = len(data['load_history'])
    workloads = []
    for i in range(0, total_len, REBALANCE_INTERVAL):
        start = i
        end = min(start + REBALANCE_INTERVAL, total_len)

        load = torch.tensor([x['logical_expert_load']
                             for x in data['load_history'][start:end]]).sum(dim=0)
        workloads.append(load)

    return workloads


class EvaluationResult(TypedDict, total=False):
    balancedness_score_gpu: float
    balancedness_score_expert: float
    times_algorithm: float
    times_inference: float
    speed_score: float
    combined_score: float
    error: str


def simulate_inference(
        log2phy: torch.Tensor,
        logcnt: torch.Tensor,
        workload: torch.Tensor,
    ) -> tuple[float, float]:
    '''
    Simulate a MoE inference with the given expert mapping, and return the balancedness factor.
    '''
    num_layers, num_logical_experts = workload.shape

    num_physical_experts = NUM_REPLICAS
    total_physical_load = torch.zeros(num_layers, num_physical_experts,
                                      dtype=torch.float, device=workload.device)

    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            logical_load = workload[layer_id][logical_id].item()

            if logical_load <= 0:
                continue

            num_replicas = int(logcnt[layer_id][logical_id].item())

            if num_replicas <= 0:
                # Expert has load but no replicas — penalize by concentrating
                # all its load on physical slot 0 (worst-case imbalance).
                total_physical_load[layer_id, 0] += logical_load
                continue

            physical_ids = log2phy[layer_id][logical_id][:num_replicas]

            replica_load = logical_load / num_replicas

            total_physical_load[layer_id, physical_ids] += replica_load

    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0, 0.0

    # Compute expert load
    expert_layer_avg = total_physical_load.mean(dim=1).sum().item()
    expert_layer_max = total_physical_load.max(dim=1).values.sum().item()
    balancedness_expert = expert_layer_avg / expert_layer_max

    gpu_load = total_physical_load.view(num_layers, NUM_GPUS, -1).sum(dim=2)

    layer_avg = gpu_load.mean(dim=1)  # (num_layers,)
    layer_max = gpu_load.max(dim=1).values  # (num_layers,)

    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()

    balancedness_gpu = avg_load / max_load if max_load > 0 else 0.0

    return balancedness_gpu, balancedness_expert


def _zero(error: str) -> dict:
    return {"score": 0.0, "stage": "invalid", "error": error[:400],
            "balancedness_score_gpu": 0.0, "balancedness_score_expert": 0.0,
            "speed_score": 0.0, "combined_score": 0.0}


def check_mapping(logcnt: torch.Tensor, workload: torch.Tensor) -> str | None:
    """None if the returned replica counts are physically possible, else why not.

    There are only NUM_REPLICAS physical slots per layer, so the copies of all experts
    must sum to at most that. Without this, a candidate can claim many more copies than
    exist and spread every expert's load across imaginary slots, faking perfect balance
    (balancedness 1.0 vs ~0.21 for the honest seed). This is SkyDiscover's PR #59 check;
    CORAL's evaluator (which we otherwise follow) lacks it. `<=`, not `==`, so a candidate
    that leaves an expert with fewer copies still runs and is penalised as CORAL does —
    only the impossible over-claim is refused, and the honest seed (sum == NUM_REPLICAS)
    is unaffected."""
    if tuple(logcnt.shape) != tuple(workload.shape):
        return (f"replica-count grid has shape {tuple(logcnt.shape)}, expected "
                f"{tuple(workload.shape)} (layers x experts)")
    if (logcnt < 0).any():
        return "replica counts contain negative values"
    per_layer = logcnt.sum(dim=1)
    if (per_layer > NUM_REPLICAS).any():
        worst = int(per_layer.max().item())
        return (f"claims {worst} expert copies in a layer, but only {NUM_REPLICAS} "
                "physical slots exist — the copies of all experts must sum to at most "
                f"{NUM_REPLICAS} per layer")
    return None


# --------------------------------------------------------------------------------------
# Keeping the future workload away from the candidate
# --------------------------------------------------------------------------------------
#
# The task is to rearrange experts for load you have NOT seen: a mapping computed on
# workload i is scored on workload i+1. A run of this problem produced a candidate that
# simply read the answer, two ways at once — it walked the call stack out of
# `rebalance_experts` into the evaluator's frames (`sys._getframe`, `f_back`, `f_locals`),
# found the list of workloads and took the next one; and it opened the evaluator's own
# source file off disk. Balance jumped from 0.25 to 0.44 in a single iteration.
#
# Two defences, because the prize is reachable by two routes:
#
#   * the candidate now runs in its OWN process (_worker.py), handed one workload at a
#     time. The future workload is not in that process at all, and there is no caller
#     frame above it that holds it. This is structural.
#   * the data set itself is mounted in the same sandbox, so a determined candidate could
#     re-read it and reconstruct the sequence. Denying that structurally needs the data
#     outside the sandbox entirely (a platform change); until then the guard below
#     refuses a genome that reaches for either route.
#
# CORAL keeps its data set out of the agent's worktree but still calls the candidate
# in-process, so the first route is open there too.
_FORBIDDEN = (
    (r"_getframe|f_back|f_locals|f_globals|inspect\.(stack|currentframe|getouterframes)"
     r"|currentframe", "stack introspection"),
    (r"expert-load\.json|AGENTDISCOVER_DATA_DIR", "reading the evaluation data set"),
    (r"evaluator\.py|_worker\.py|/bundle", "reading the evaluator"),
)


def check_genome(source: str) -> str | None:
    """None if the genome is clean, else why it was refused."""
    for pattern, why in _FORBIDDEN:
        m = re.search(pattern, source)
        if m:
            return (f"refused: the program uses {why} ({m.group(0)!r}). The mapping must "
                    "be computed from the workload you are given — the next workload, "
                    "the data set and the evaluator are not yours to read.")
    return None


@functools.cache
def _can_hide() -> bool:
    """Whether we can start the candidate in a mount namespace that hides the data dir.
    True on the platform's singularity sandbox; False on backends without unprivileged
    user namespaces, where we fall back to the source blocklist alone."""
    try:
        r = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "true"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _hidden_launch(worker_argv: list[str]) -> list[str]:
    """Wrap the worker so the candidate's process cannot read the private workload data.

    The future workload is the answer the candidate is asked to predict, and the whole
    recorded history sits in the data dir inside this sandbox. We start the worker in its
    own mount namespace with an empty tmpfs mounted over that dir, so the candidate simply
    cannot open it — a structural block, unlike the source-text blocklist. Only done when
    the data dir is separate from the evaluator bundle (real runs mount it at its own
    path) and the sandbox allows user namespaces; otherwise the worker starts normally and
    the blocklist is the only guard."""
    data_dir = os.path.abspath(os.path.dirname(WORKLOAD_PATH))
    bundle_dir = os.path.abspath(os.path.dirname(__file__))
    if data_dir == bundle_dir or not os.path.isdir(data_dir) or not _can_hide():
        return worker_argv
    bootstrap = (
        "import subprocess, sys, os\n"
        f"subprocess.run(['mount', '-t', 'tmpfs', 'none', {data_dir!r}])\n"
        f"os.execv({worker_argv[0]!r}, {worker_argv!r})\n"
    )
    return ["unshare", "--user", "--map-root-user", "--mount",
            worker_argv[0], "-c", bootstrap]


OUTPUT_TAIL_CHARS = 4000    # of the program's own printing, kept for the OPERATOR's stderr


class _ProgramError(RuntimeError):
    """The candidate's process failed. `cls` is the only part the agent gets to see:
    the exception class the program raised (or 'exit' / 'timeout' for a process that
    died or overran); the message is operator-only."""

    def __init__(self, cls: str, detail: str):
        super().__init__(detail)
        self.cls = cls


def _error_class(e: BaseException) -> str:
    """What the agent is told about a failure: an exception class name, nothing more."""
    return e.cls if isinstance(e, _ProgramError) else type(e).__name__


class _Candidate:
    """The candidate's process: started once, fed one workload per call.

    Everything the program prints (stdout and stderr both end up on the worker's
    stderr) is drained continuously into a bounded buffer — an undrained pipe fills at
    64KB and would deadlock any program that prints — and then DISCARDED: it goes to
    this evaluator's own stderr for the operator and never into the result. Until
    2026-09-17 the tail was returned to the agent as `program_output`, as "the honest
    instrument for learning the workload structure". It was also a memorisation channel:
    the same workload sequence is replayed on every evaluation, and call k's input is
    the very load call k-1 is scored on, so a candidate that printed its inputs handed
    the agent the answer key for the next candidate. Our own 0.1696 run did exactly
    that — its top candidates carried ~2300 hardcoded expert ids copied from earlier
    printouts; the best candidate without them scored 0.1473. Nothing the candidate
    writes, prints or raises reaches the agent now; only the scores do.
    """

    def __init__(self, solution_path: Path, timeout: float):
        self.timeout = timeout
        env = {k: v for k, v in os.environ.items()
               if k not in ("AGENTDISCOVER_DATA_DIR", "AGENTDISCOVER_SOLVE_SECONDS")}
        env["EPLB_CHILD"] = "1"
        # -u: unbuffered. The program's prints otherwise sit in an 8KB block buffer
        # and the tail collected below misses everything since the last full block.
        # The protocol is unaffected — the worker writes it through .buffer with
        # explicit flushes.
        worker_argv = [sys.executable, "-u",
                       str(Path(__file__).resolve().parent / "_worker.py"),
                       str(solution_path), str(NUM_REPLICAS), str(NUM_GROUPS),
                       str(NUM_NODES), str(NUM_GPUS)]
        self.proc = subprocess.Popen(
            _hidden_launch(worker_argv),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(solution_path.parent), env=env)
        self._output = collections.deque(maxlen=64 * 1024)   # bytes, bounded
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain.start()

    def _drain_stderr(self):
        with contextlib.suppress(Exception):
            while True:
                chunk = self.proc.stderr.read(4096)
                if not chunk:
                    break
                self._output.extend(chunk)

    def output_tail(self) -> str:
        """The program's last printed characters — for the operator's log only."""
        return bytes(self._output).decode(errors="replace")[-OUTPUT_TAIL_CHARS:]

    def rearrange(self, weight: torch.Tensor):
        """Send one workload, get (log2phy, logcnt) back. Raises on any failure."""
        buf = io.BytesIO()
        torch.save({"weight": weight}, buf)
        data = buf.getvalue()
        try:
            self.proc.stdin.write(struct.pack("!I", len(data)))
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
            header = self.proc.stdout.read(4)
            if not header or len(header) < 4:
                time.sleep(0.2)      # let the drain thread catch the last words
                raise _ProgramError("exit", "the program exited without answering: "
                                    + self.output_tail()[-600:])
            (size,) = struct.unpack("!I", header)
            reply = torch.load(io.BytesIO(self.proc.stdout.read(size)), weights_only=False)
        except (BrokenPipeError, OSError) as e:
            raise _ProgramError("exit", f"lost the program's process: {e}") from None
        if "error" in reply:
            # The worker sends only the exception's class name (see _worker.py).
            raise _ProgramError(str(reply["error"])[:80], "the program raised " + str(reply["error"]))
        # int64 again for indexing; the narrow form is only for the wire (see _worker).
        return reply["log2phy"].to(torch.int64), reply["logcnt"].to(torch.int64)

    def close(self):
        """Idempotent; joins the drain so output_tail() is complete afterwards."""
        with contextlib.suppress(Exception):
            self.proc.stdin.write(struct.pack("!I", 0))
            self.proc.stdin.flush()
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            self.proc.kill()
        self._drain.join(timeout=2)


def evaluate(program_path: str) -> dict:
    t0 = time.perf_counter()
    # Missing data is an operator problem, not the candidate's: say so plainly rather
    # than letting a FileNotFoundError traceback reach the agent as its own failure.
    if not os.path.exists(WORKLOAD_PATH):
        return {"score": 0.0, "stage": "error", "time": time.perf_counter() - t0,
                "error": f"workload data not found at {WORKLOAD_PATH}. The platform "
                         "fetches it via evaluator/fetch.sh; see PROBLEM.md."}

    candidate = None
    call_index = None       # which rearrangement is in flight, for the failure message
    try:
        refusal = check_genome(Path(program_path).read_text())
        if refusal is not None:
            return _zero(refusal)

        workloads = load_workloads(WORKLOAD_PATH)
        candidate = _Candidate(Path(program_path).resolve(), SOLVE_TIMEOUT)

        balancedness_scores_gpu = []
        balancedness_scores_expert = []
        times_algorithm = []
        times_inference = []
        for i in range(len(workloads) - 1):
            call_index = i
            start_time = time.perf_counter()
            log2phy, logcnt = candidate.rearrange(workloads[i])
            end_time_algorithm = time.perf_counter()
            bad = check_mapping(logcnt, workloads[i])
            if bad is not None:
                candidate.close()
                return _zero(f"invalid rearrangement on workload {i}: {bad}")
            balancedness_score_gpu, balancedness_score_expert = simulate_inference(
                log2phy, logcnt, workloads[i + 1])
            end_time = time.perf_counter()
            balancedness_scores_gpu.append(balancedness_score_gpu)
            balancedness_scores_expert.append(balancedness_score_expert)
            times_algorithm.append(end_time_algorithm - start_time)
            times_inference.append(end_time - start_time)

        candidate.close()   # flushes the program's last words into the drain
        avg_balancedness_score_gpu = sum(balancedness_scores_gpu) / len(balancedness_scores_gpu)
        avg_balancedness_score_expert = sum(balancedness_scores_expert) / len(balancedness_scores_expert)
        avg_time_algorithm = sum(times_algorithm) / len(times_algorithm)
        avg_time_inference = sum(times_inference) / len(times_inference)
        speed_score = 0.002 / avg_time_inference
        combined_score = (avg_balancedness_score_expert + speed_score) / 2

        return {
            "score": float(combined_score),
            "stage": "full",
            "objective": float(combined_score),
            "objective_name": "combined_score",
            "objective_direction": "max",
            "objective_target": TARGET,
            "combined_score": float(combined_score),
            "balancedness_score_expert": float(avg_balancedness_score_expert),
            "balancedness_score_gpu": float(avg_balancedness_score_gpu),
            "speed_score": float(speed_score),
            "times_algorithm": float(avg_time_algorithm),
            "times_inference": float(avg_time_inference),
            "rearrangements_scored": len(workloads) - 1,
            "target": TARGET,
            "solve_time": float(sum(times_algorithm)),
            "time": time.perf_counter() - t0,
        }

    except Exception as e:  # noqa: BLE001 — a bad candidate scores 0, never crashes us
        # The agent learns only that the candidate failed, on which call, and the
        # exception's CLASS. The exception's message, the traceback and the program's
        # printout are candidate-controlled text — a `raise ValueError(weight.tolist())`
        # would carry the workload out just as a print did — so they go to the
        # operator's stderr and nowhere else.
        detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
        if candidate is not None:
            candidate.close()   # flush, so the tail below shows the dying words
            detail += "\n--- program output (tail) ---\n" + candidate.output_tail()
        sys.stderr.write(detail + "\n")
        which = f" on rearrangement {call_index + 1}" if call_index is not None else ""
        return {"score": 0.0, "stage": "error",
                "error": f"the candidate failed{which} with {_error_class(e)}; its "
                         "message and printout are not returned. Reproduce it locally.",
                "time": time.perf_counter() - t0}
    finally:
        if candidate is not None:
            candidate.close()


def main():
    path = os.path.abspath(sys.argv[1])
    sys.path.insert(0, os.path.dirname(path))
    # The candidate and the reference algorithm both print progress; keep stdout clean so
    # the JSON line below is the only thing on it (CORAL's wrapper does the same).
    with contextlib.redirect_stdout(sys.stderr):
        metrics = evaluate(path)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
