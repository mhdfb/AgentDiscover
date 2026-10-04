# trimul — provenance (operator only)

This file is **not** mounted anywhere an agent can read. It records where the task comes
from, what was copied unchanged, and what differs from the setting TTT-Discover reports.

## Where the task comes from

GPUMode's TriMul competition (leaderboard 496), as used by "Learning to Discover at Test
Time" (TTT-Discover, arXiv:2601.16175, §Kernel Engineering). The harness files come from
TTT-Discover's repository, `examples/gpu_mode/lib/bioml/trimul/`, which carries GPUMode's
`reference-kernels` task definition:

| file | copied to | md5 |
|---|---|---|
| eval.py | evaluator/kernelbot/ | 8df13c37209033457c6f04ddc965b592 |
| reference.py | evaluator/kernelbot/ | cfa3f5fc30f26829007c726de87a1131 |
| utils.py | evaluator/kernelbot/ | d9e6cb8126d3d8397c783ff9ac8c8ca1 |
| task.py | evaluator/kernelbot/ | cbc065d3afbee88c5780c65923146335 |
| task.yml | evaluator/kernelbot/ | 195376a9bbc80c05b7bc6bb641774081 |

Byte for byte; do not edit them. `evaluator.py` replaces only their runner
(`libkernelbot/run_eval.py` + `submission.compute_score`): same phases, same timeouts,
same scoring formula. **`resources/` holds a copy of `evaluator.py` and `kernelbot/` —
re-copy after every edit here.**

PROBLEM.md is TTT-Discover's prompt for this task (`examples/gpu_mode/prompt.py`,
`TRIMUL_PROMPT`, plus the rules appended in `env.py`), with "H100" replaced by the GPU
we run on and the platform's interface section added. The seed is their initial state:
no previous attempt.

## Published numbers (withheld from the agent except the target)

TriMul, µs, lower is better (TTT-Discover Table "TriMul"; leaderboard numbers for A100
and H100, local re-timing for B200/MI300X):

| | A100 | H100 | B200 | MI300X |
|---|---|---|---|---|
| 1st human | 4,531.5 | 1,371.1 | 1,038.9 | 2,515.8 |
| Best-of-25600 (gpt-oss-120b) | 9,219.7 | 5,390.3 | 3,254.9 | 4,902.0 |
| TTT-Discover (gpt-oss-120b) | **2,198.2** | **1,161.2** | **914.2** | **1,555.7** |

TTT-Discover searched with H100 timings as the reward and submitted its best kernels to
the leaderboard for the A100/H100 numbers; the A100 figure is a kernel never tuned on an
A100. Their prompt stated a target of 1000 µs (an H100 figure). Ours states
TTT-Discover's own published figure for the GPU in use (`GPU_TARGET_US` in
evaluator.py), so fitness 1.0 means matching their result. (Until 2026-10-01, before
any run, the target was the best human entry; the user chose their number.)

## What differs from TTT-Discover's setting

1. **Hardware.** They timed on Modal H100s (80 GB HBM3) during search. We time on an
   A100 80GB PCIe — the GPU with a leaderboard
   column to compare against. The torch/triton versions are theirs (torch 2.7.1+cu128,
   triton 3.3.1, the Modal image's).
2. **Tool use.** Their policy only emitted code; ours can run the harness locally on its
   own GPU (`/resources/evaluator.py`), as it can run the AHC tester on `ahc006_prov`.
3. **Seed.** No secret seed (`task.seed` is null in their config too), so the cases run
   with the seeds in task.yml. GPUMode's live leaderboard uses a secret seed per
   leaderboard; TTT-Discover's verifier did not.
4. **Harness quirk, kept.** `eval.py` parses case files with a regex that keeps `True`
   and `False` as strings, so `generate_input(nomask="False")` is truthy and every
   "masked" case runs with an all-ones mask. This is the competition's and
   TTT-Discover's behaviour alike; reproduced, not fixed.
5. **Memory.** No address-space cap (CUDA needs it off); `mem_gb` only binds under
   docker/podman. TTT-Discover ran inside Modal's container with no explicit cap either.

## Verification

Run `python3 evaluator.py <TTT-Discover's results/kernel-engineering/trimul.py>` on an
A100: their published best kernel should land near its leaderboard figure, 2,198.2 µs
(expect some spread: their number is Modal's A100 PCIe, ours a different machine).
Verified 2026-10-01 through the sandboxed path on an A100 80GB PCIe:
18/18 tests pass, 2,298.1 µs (fitness 1.97), whole evaluation 58 s.
