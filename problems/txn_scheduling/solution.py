"""Transaction scheduling: order three workloads to minimise total makespan.

See PROBLEM.md. The mutable region is marked by # EVOLVE-BLOCK-START /
# EVOLVE-BLOCK-END. Everything else in this file is fixed scaffolding — do not modify it.

`txn_simulator` and `workloads` are importable both here and inside the evaluator, so
you can measure any ordering yourself with `workload.get_opt_seq_cost(seq)` — that is the
same function the evaluator scores you with. Read-only copies are also at /resources.
"""
import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

WORKLOAD_JSON = (WORKLOAD_1, WORKLOAD_2, WORKLOAD_3)


def solve() -> list[list[int]]:
    """Return one schedule per workload: three permutations of 0..99.

    Scored by 1000 / (1 + total_makespan) * 1000, where total_makespan is the sum of
    the three workloads' makespans. Higher is better.
    """
    workloads = [Workload(j) for j in WORKLOAD_JSON]
    return [get_best_schedule(w) for w in workloads]


# EVOLVE-BLOCK-START
def get_best_schedule(workload, num_samples: int = 10) -> list[int]:
    """Baseline: greedy with a random start, sampling `num_samples` candidates per step.

    At each step it draws a few of the remaining transactions, appends whichever gives
    the cheapest prefix so far, and puts the rest back.
    """
    n = workload.num_txns
    start = random.randint(0, n - 1)
    seq = [start]
    remaining = [t for t in range(n) if t != start]

    while remaining:
        best_cost, best_txn, held = None, None, []
        for _ in range(min(num_samples, len(remaining))):
            idx = random.randint(0, len(remaining) - 1)
            t = remaining.pop(idx)
            held.append(t)
            cost = workload.get_opt_seq_cost(seq + [t])
            if best_cost is None or cost < best_cost:
                best_cost, best_txn = cost, t
        seq.append(best_txn)
        held.remove(best_txn)
        remaining.extend(held)

    return seq
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Local sanity print; the evaluator is authoritative.
    schedules = solve()
    total = sum(Workload(j).get_opt_seq_cost(s)
                for j, s in zip(WORKLOAD_JSON, schedules))
    print(f"total makespan: {total}   combined_score: {1000 / (1 + total) * 1000:.3f}")
