# kernel_builder — provenance (operator only)

This file is **not** mounted anywhere an agent can read. It holds the notes that used to
sit in `evaluator.py`'s docstring; they moved here when `evaluator.py` became readable by
the search agent at `/resources/evaluator.py`, because they name published results that
`PROBLEM.md` deliberately withholds.

## Where the task comes from

Ported from CORAL's `examples/kernel_builder` — the task behind Figure 1 of their paper,
originally Anthropic's VLIW SIMD kernel-engineering take-home.

- `frozen_problem.py` is their frozen simulator, byte for byte.
- `solution.py` is their `seed/kernel_builder.py`, verbatim. It scores **11,910 cycles**;
  the 147,734 its own docstring quotes is the original naive scalar baseline, not what the
  seed does (the shipped seed is already vectorised).
- The workload and correctness protocol are theirs: `REAL_PARAMS` = forest_height 10,
  rounds 16, batch_size 256; eight trials, each on a freshly generated random tree and
  batch (`Tree.generate` / `Input.generate`, unseeded); the output must match
  `reference_kernel2` on every trial; the reported cycle count is the last trial's
  (deterministic for a given instruction list).

## Published numbers (withheld from the agent)

| | cycles |
|---|---|
| naive scalar baseline | 147,734 |
| shipped seed | 11,910 |
| Anthropic's own record on the take-home | 1,363 |
| CORAL paper, 1 agent (56 evals) | 1,350 |
| CORAL paper, 4 agents (596 evals) | **1,103** |
| CORAL paper, OpenEvolve (363 evals) | 2,740 |

`TARGET = 1,103` — the best published figure. CORAL's `task.yaml` and grader were edited
to state the same target (2026-09-20), so an agent on either platform is told the same
bar. Their seed docstring still says "best known ~1,363"; left untouched, and identical on
both sides.

## What differs from CORAL's grader, and why

**Process layout.** Their grader `exec`s the candidate's file in the *same* process as the
simulator and then calls `Machine(...).run()` there, so a candidate can patch `Machine`,
`reference_kernel2` or the cycle counter before the grader reads them. Here the candidate's
process only BUILDS: it is handed the four sizes, returns the instruction list and the
scratch map as JSON, and exits. The simulator, the random inputs, the reference and the
cycle counter all live in the parent, which the candidate never runs in. Verified: a
candidate that replaces `frozen_problem.Machine` in its own process still scores its honest
11,910. Knowing the wire buys a candidate nothing — a forged JSON line is still simulated
honestly by the parent, so the worst it can do is submit a different program.

**One build, not eight.** Their `do_kernel_test` constructs a fresh `KernelBuilder` inside
each of the 8 trials; we build once and run that one instruction list against 8 random
inputs. Ours is the stricter correctness test (one kernel must handle all 8) and closes the
hole where their reported cycles come from the last build only. The consequence for
fairness is the build clock — see below.

**Fitness.** `1103 / cycles`: 1.0 at the target, above 1 past it, monotone in the objective.
Their leaderboard ranks the raw cycle count, minimised — same ordering.

## Comparability audit (2026-09-20)

Verified equal: workload constants, simulator file, correctness comparison and its failure
string, objective and target, seed, prompt content. `DebugInfo` carries only `scratch_map`
and is read solely in trace/debug paths with `enable_debug=False`, so reconstructing it
from JSON cannot shift cycles or correctness.

Four differences were found and ours was moved to match theirs (the user's call: change our
side, not CORAL's):

1. **Build clock.** CORAL rebuilds inside each of 8 trials under one 120 s grader timeout →
   ~14 s of builder time for the kernel it is scored on. Ours allowed 60 s. `solve_seconds`
   is now **14**.
2. **Scoring code visibility.** CORAL symlinks the whole `grader/` package read-only into
   every agent worktree by design; only `taskdata/` (the simulator) is private. Ours showed
   the agent nothing. `evaluator.py` is now copied to `resources/`, mounted read-only at
   `/resources`; `frozen_problem.py` stays hidden, matching their private `taskdata`.
   **`resources/evaluator.py` is a copy — re-copy it after every edit to `evaluator.py`.**
3. **Compute.** On the singularity backend `cpus` only sets a CPU-seconds ulimit
   (`wall × cpus`), so 1.0 killed any multiprocess builder after 120 CPU-seconds; CORAL's
   grader is uncapped on a 96-core box. `cpus` is now **96.0**; `wall_seconds` and the
   14 s build timeout still bound the evaluation.
4. **`coral eval --tune`.** Their grader implements a cheap eval mode (rounds 4, batch 64,
   2 trials, ~10x faster) that is hidden from the leaderboard and does not tick their
   plateau counter, and CORAL.md teaches agents to sweep with it. We have no equivalent;
   matching it would be a platform feature (submit_candidate, the briefing, quota
   accounting), not a problem change. **Decision: left in place and reported** — the one
   acknowledged asymmetry on this task, in CORAL's favour. It still costs tokens, so it
   shows on the cost axis, but it buys them information per token our agent cannot buy.
   Say so in the paper rather than quietly removing it.
