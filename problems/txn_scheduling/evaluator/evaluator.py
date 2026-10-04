"""Transaction scheduling — minimise the makespan of three database workloads.

Ported from SkyDiscover's `benchmarks/ADRS/txn_scheduling`, which is the bundle CORAL
runs. The task, the data and the metric are theirs, unchanged:

  * `txn_simulator.py` and `workloads.py` are byte-identical copies of SkyDiscover's.
  * The three workloads are WORKLOAD_1/2/3, 100 transactions each, exactly as shipped.
  * The objective is `combined_score = 1000 / (1 + makespan) * 1000`, higher is better,
    over the SUM of the three workloads' makespans — SkyDiscover's `get_random_costs`
    returns `cost1 + cost2 + cost3` and its evaluator scores that sum.
  * The solve budget is 600 s, matching their `evaluator.timeout: 600`.

What differs is only the interface, to fit this platform's `solve()` contract: their
`initial_program.py` evolves `get_best_schedule` and computes its own makespan, whereas
here `solve()` returns the three orderings and the makespan is computed HERE, by this
evaluator, from its own copy of the simulator. A candidate therefore cannot report a
score for itself — the number always comes from the same code for every candidate.

The evaluator directory is placed on the subprocess's `sys.path`, exactly as SkyDiscover
does, so `solve()` can `from txn_simulator import Workload` and
`from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3` while it searches. Copies of
both modules are also mounted read-only at /resources so a session can develop against
them outside the evaluator.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from txn_simulator import Workload                       # noqa: E402
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3  # noqa: E402

WORKLOAD_JSON = (WORKLOAD_1, WORKLOAD_2, WORKLOAD_3)

# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone. 600 s matches SkyDiscover's config.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 600.0))

# The objective is the MAKESPAN — the real scheduling quantity, in time units, summed
# over the three workloads, minimised. SkyDiscover's combined_score = 1000/(1+makespan)
# *1000 is a strictly decreasing function of it, so the two orderings are identical; we
# score on the makespan because it is the interpretable one, and report both.
#
# A bar to BEAT by a clear margin, not to meet. Competitor numbers live in
# paper/benchmark.md and are deliberately kept out of everything the agent reads —
# naming one anchors the search on it.
TARGET = 218.0


def _run_solve(solution_dir: Path, timeout: float) -> tuple[list, float]:
    """Run solve() in a fresh subprocess. Return (schedules, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        f"sys.path.insert(0, {str(_HERE)!r})\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': [[int(i) for i in s] for s in v], 'time': dt}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"solve() ran past its {timeout:.0f}s budget and was killed. Keep its "
            "internal time budgets safely below the evaluator's limit."
        ) from None
    if out.returncode != 0:
        detail = (out.stderr or "").strip() or (out.stdout or "").strip()[-500:]
        reason = f"exit code {out.returncode}"
        if out.returncode < 0 or out.returncode in (134, 137, 139):
            reason += " (killed — likely the memory cap or another resource limit)"
        raise RuntimeError(f"solve failed, {reason}: {detail[:500] or 'no output produced'}")
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            return data["value"], data["time"]
    raise RuntimeError("solve produced no JSON line")


def _verify(schedules, workloads) -> str | None:
    """None if the schedules are valid, else an error string.

    This is SkyDiscover's `validate_schedule`, deliberately reproduced as-is so our
    number is measured under the same rule CORAL's is:

        for i in range(len(txn_seq)):
            if not i in txn_seq: return False

    It checks only that indices 0..len-1 all appear — it does NOT require the schedule
    to contain every transaction. A shorter schedule is therefore accepted and costs
    less, which is why their metric and a completeness-enforcing one are not comparable.
    We ran this problem with completeness enforced first: the best complete schedule
    found was makespan 230, against CORAL's reported 218. Rather than compare across two
    different rules, we adopt theirs.

    We keep two guards that are not relaxations of their rule: indices must be in range
    (the simulator would raise otherwise) and must not repeat (running one transaction
    twice is not a schedule).
    """
    if not isinstance(schedules, list):
        return f"expected a list of 3 schedules, got {type(schedules).__name__}"
    if len(schedules) != len(workloads):
        return f"expected {len(workloads)} schedules, got {len(schedules)}"
    for i, (seq, wl) in enumerate(zip(schedules, workloads)):
        n = wl.num_txns
        if not isinstance(seq, list):
            return f"schedule {i} is not a list"
        if not seq:
            return f"schedule {i} is empty"
        if any(not isinstance(t, int) or not (0 <= t < n) for t in seq):
            return f"schedule {i} has an index outside 0..{n - 1}"
        if len(set(seq)) != len(seq):
            return f"schedule {i} repeats a transaction"
        # Every transaction must run. SkyDiscover's validate_schedule only asks that a
        # schedule of length k contain 0..k-1, which a schedule of length 1 satisfies —
        # and the makespan of running one transaction out of a hundred is tiny. A run
        # here found that in its first session and scored a "makespan" of 46 against a
        # target of 218 by scheduling exactly one transaction per workload. CORAL's task
        # statement says the schedule is "a permutation of every transaction index in the
        # workload", so full coverage is the intended rule; this enforces what both
        # statements mean rather than what the weaker check happens to allow.
        if sorted(seq) != list(range(n)):
            return (f"schedule {i} must be a permutation of all {n} transaction "
                    f"indices 0..{n - 1}; it lists {len(seq)}")
    return None


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    workloads = [Workload(j) for j in WORKLOAD_JSON]

    try:
        schedules, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    err = _verify(schedules, workloads)
    if err is not None:
        return {"score": 0.0, "stage": "invalid", "error": err,
                "time": time.perf_counter() - t0}

    # The makespans are recomputed here from the returned orderings; nothing the
    # candidate printed about its own quality is used.
    per_workload = [wl.get_opt_seq_cost(seq) for wl, seq in zip(workloads, schedules)]
    makespan = float(sum(per_workload))
    combined = 1000.0 / (1.0 + makespan) * 1000.0

    return {
        # Minimised objective: TARGET/makespan rises as the makespan falls, passes 1.0 at
        # the target, and is not clamped — the target is a goal, not a proven bound.
        "score": TARGET / makespan if makespan > 0 else 0.0,
        "stage": "full",
        "objective": makespan,
        "objective_name": "makespan",
        "objective_direction": "min",
        "objective_target": TARGET,
        "makespan": makespan,
        "makespan_per_workload": per_workload,
        "combined_score": combined,
        # How many transactions each schedule actually ran. Under SkyDiscover's rule a
        # schedule may be shorter than the workload, so this has to be visible.
        "scheduled_per_workload": [len(s) for s in schedules],
        "workload_sizes": [wl.num_txns for wl in workloads],
        "target": TARGET,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
