# mla_decode — provenance (operator only)

This file is **not** mounted anywhere an agent can read. It records where the task comes
from, what was copied unchanged, and what differs from the setting TTT-Discover reports.

## Where the task comes from

GPUMode's AMD MLA-Decode competition (leaderboard 463), as used by "Learning to Discover
at Test Time" (TTT-Discover, arXiv:2601.16175, §Kernel Engineering) in its NVIDIA
variant (`mla_decode_nvidia`). The harness files come from TTT-Discover's repository,
`examples/gpu_mode/lib/mla-decode/`, which carries GPUMode's task definition:

| file | copied to | md5 |
|---|---|---|
| eval.py | evaluator/kernelbot/ | 25df92613dedb2c5fc56fd9e33b44bd9 |
| reference.py | evaluator/kernelbot/ | 59c8ed330aeaf8fb6a015a493c529828 |
| utils.py | evaluator/kernelbot/ | 14b02ea52894cbf9f15cba5862a0e07c |
| task.py | evaluator/kernelbot/ | afc40e2622428fb4ce13d8b5fb41902f |
| task.yml | evaluator/kernelbot/ | 2e466b2ca3e6ec5b142f2ad90ecc7f36 |

Byte for byte; do not edit them. `evaluator.py` replaces only their runner
(`libkernelbot/run_eval.py` + `submission.compute_score`): same phases, same timeouts,
same scoring. **`resources/` holds a copy of `evaluator.py` and `kernelbot/` — re-copy
after every edit here.**

PROBLEM.md is TTT-Discover's prompt for this task (`examples/gpu_mode/prompt.py`,
`MLA_DECODE_PROMPT` + `MLA_DECODE_PROMPT_END`) with the platform's interface section
added. The seed, `solution.py`, is their `MLA_DECODE_INITIAL_STATE` verbatim: the
correct, unoptimized kernel they had the base model write first (3,846 µs on their
H200, `MLA_DECODE_INITIAL_VALUE`).

## Published numbers (withheld from the agent except the target)

The competition ran on AMD MI300X. TTT-Discover searched on H200s (Modal), then re-timed
its selected kernels on three MI300X instances. MLA-Decode, µs, lower is better
(TTT-Discover Table "AMD MI300X - MLA Decode", instance 1 / 2 / 3):

| | instance 1 | instance 2 | instance 3 |
|---|---|---|---|
| 1st human | 1,653.8 | 1,688.6 | 1,668.7 |
| Best-of-25600 (gpt-oss-120b) | 2,286.0 | 2,324.1 | 2,275.2 |
| TTT-Discover (gpt-oss-120b) | 1,669.1 | 1,706.1 | 1,671.3 |
| TTT-Discover, Triton-only kernels | 1,740.6 | 1,754.4 | 1,707.1 |

They did not beat the best human with statistical significance. Their prompt's target
was 1700 µs, which `TARGET_US` keeps; there is no NVIDIA leaderboard for this task, so
numbers measured here are comparable to their H200 search-time numbers (initial state
3,846 µs), not to the MI300X table.

## What differs from TTT-Discover's setting

1. **Hardware.** They timed on Modal H200s during search. A node with H200s
   (ours had 2 x H200) matches that; an A100 does not (less memory
   bandwidth, which is what a decode kernel is bound by). The torch/triton versions are
   theirs (torch 2.8.0+cu129, triton 3.4.0, the Modal image's).
2. **Tool use.** Their policy only emitted code; ours can run the harness locally on its
   own GPU (`/resources/evaluator.py`).
3. **Seed.** No secret seed, as in their config; the cases run with task.yml's seeds.
4. **Memory.** No address-space cap (CUDA needs it off); `mem_gb` only binds under
   docker/podman.

## Verification

`python3 evaluator.py solution.py` (the seed) on an H200 should land near 3,846 µs
(verified 2026-10-01 through the sandboxed path on an H200 NVL: 4/4 tests
pass, 4,010.6 µs, whole evaluation 30 s);
their best kernels (`results/kernel-engineering/mla_code_{1,2,3}.py` in their
repository) are selected for MI300X and were not reported on H200.
