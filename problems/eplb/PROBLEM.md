# Expert Parallelism Load Balancer (EPLB)

Improve the Mixture-of-Expert models Expert Parallelism Load Balancer (MoE EPLB) expert
rearrangement algorithm.

This algorithm takes the load metrics recorded by the vLLM server, and rearranges the
experts to balance the load. It can make replicas of some experts to achieve better load
balancing.

Your goal is two-fold:

1. Improve the algorithm to achieve better load balancing; while
2. Improve the algorithm to be more efficient, i.e. reduce the execution time of the
   algorithm itself, since perfect load balancing is NP-hard.

The current algorithm is implemented in the `rebalance_experts` function:

    def rebalance_experts(
        weight: torch.Tensor,   # [num_moe_layers, num_logical_experts] load stats
        num_replicas: int,      # total physical experts (must be multiple of num_gpus)
        num_groups: int,        # number of expert groups
        num_nodes: int,         # number of server nodes
        num_gpus: int,          # number of GPUs (multiple of num_nodes)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: (physical_to_logical_map, logical_to_physical_map, expert_count)

The expert copies you assign (`expert_count`) must sum, per layer, to at most
`num_replicas` — there are only that many physical slots. A mapping that claims more is
rejected and scores 0.

The score is: `0.5 * balancedness_score_expert + 0.5 * speed_score`, where
`speed_score = 0.002 / avg_inference_time` (higher is better).

## Targets

Your objective is the **`combined_score`** above. **Higher is better.** The recorded load
history is split into consecutive workloads; your rearrangement is computed on one and
scored on the *next*, so it has to balance load you have not seen yet.

| | combined_score ↑ |
|---|---|
| the seed below (DeepSeek EPLB, hierarchical) | ~0.126 |
| **target — beat this, clearly** | **0.149** |

The target is a bar to get **meaningfully past**, not to land on: matching ~0.149 is not
success, and nothing marks it as a limit — it is simply where a previous search stopped.
Published results are deliberately not listed here. Alongside `combined_score` the
evaluator returns `balancedness_score_expert`, `balancedness_score_gpu`, `speed_score`,
`times_algorithm` and `times_inference`, so you can see which half is costing you.

Both halves are real. The balance term is at most 1 and the seed reaches only 0.25, so
there is a great deal of room in it; the speed term is small at the seed (~0.002) and
bounded by the simulation's own cost, so speed alone cannot carry a candidate.

## Interface

`solution.py` defines `rebalance_experts` with the signature above; the evaluator imports
it and calls it once per workload. It is scored on what it returns:
`logical_to_physical_map` and `expert_count` decide where load lands, and the wall-clock
time of the call is half the metric. The whole file is inside the EVOLVE-BLOCK — the
helper functions (`balanced_packing`, `replicate_experts`,
`rebalance_experts_hierarchical`) are yours to change or replace, as long as
`rebalance_experts` keeps its name and signature.

The fixed configuration is 288 physical experts, 8 groups, 32 GPUs, 4 nodes.

PyTorch is installed in your sandbox and in the evaluator.

**Compute budget: 300 s** for the full sequence of rearrangements, out of a 360 s cap on
the whole evaluation. The budget is deliberately generous: spending it is what the speed
half of the score punishes, not a timeout. Your briefing repeats the numbers in force
each session.

**Nothing your program prints comes back to you.** The evaluator returns the scores
above and nothing else: not your program's stdout or stderr, not the message of an
exception it raises (only the exception's class and the rearrangement it failed on). The
workloads are private, and a candidate that echoed what it saw would hand the next
candidate the answer key, since the same sequence is scored every time. Learn the
workload's structure from the scores, and debug your program locally on synthetic loads.

The workload data (224 MB of recorded vLLM load history) is too large for the repository,
so the platform downloads it on this problem's first evaluation and caches it — nothing
to install by hand.
