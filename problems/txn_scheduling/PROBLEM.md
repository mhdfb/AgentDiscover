# Transaction scheduling — minimise makespan

You are optimising database transaction scheduling. A workload is a set of transactions,
each a sequence of read (`r-{key}`) and write (`w-{key}`) operations on data items:

    "txn0": "w-17 r-5 w-3 r-4 r-54 r-14 w-6 r-11 w-22 r-7 w-1 w-8 w-9 w-27 r-2 r-25"

Read–write and write–write conflicts on the same key create dependencies between
transactions: a conflicting transaction cannot start until the lock it needs is released,
so the order you execute them in changes how long the whole workload takes.

**Your task**: find, for each workload, an execution ordering that minimises the total
**makespan** — the time to execute the whole schedule.

## Targets

There are three workloads of 100 transactions each. Your objective is the **total
makespan** — the sum of the three workloads' makespans. **Lower is better.**

| | total makespan ↓ |
|---|---|
| the seed below (sampled greedy) | ~359 |
| **target — beat this, clearly** | **218** |

The target is a bar to get **meaningfully under**, not to land on: reaching ~218 is not
success, and there is no proven optimum for this problem — how low the makespan can go is
open. The evaluator also reports `combined_score = 1000/(1+makespan)*1000`, which is
just a higher-is-better restatement of the same number; the makespan is what you are
scored on.

Published results are deliberately not listed here. A number someone else reached is an
anchor, not a bound: it tells you where a previous search stopped, not where this one has
to. The evaluator returns your `combined_score`, the `makespan`, its breakdown per
workload, and the target.

## What is known to work

- **Greedy**: try iteratively picking the transaction that increases the makespan least.
- Avoid relying only on surface heuristics like transaction length or number of writes —
  these do not correspond to the actual makespan of the schedule.

## Interface

`solve()` returns a list of three lists of integers: one schedule per workload. Each
schedule must be a permutation of all 100 transaction indices, `0 .. 99`, with every
index appearing exactly once. The ordering may change; the set of transactions may not.

`txn_simulator` and `workloads` are importable from `solution.py` — construct
`Workload(WORKLOAD_1)` and call `workload.get_opt_seq_cost(seq)` to measure any ordering
yourself. **That is the same function the evaluator scores you with**, so a schedule you
have measured locally will score exactly the same when submitted. Both modules are mounted
read-only at `/support` and are already on your `PYTHONPATH` — `import txn_simulator`
works from your worktree directly, with no path setup and nothing to copy.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that figure
and use it — leaving time unspent is wasted search, and overrunning it means the candidate
scores 0. The evaluation as a whole is capped at 700 s, so leave a margin for interpreter
start-up and imports. Your briefing repeats the number in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- constructive heuristics: build the order one transaction at a time, scoring each choice by its true incremental cost
- local search: start from a complete ordering and improve it with swaps, insertions and segment reversals
- conflict structure: analyse which transactions contend on which keys, and use that structure to decide what to separate
- portfolio and restarts: run several randomised searches with different seeds or parameters and keep the best
