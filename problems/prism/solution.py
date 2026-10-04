"""PRISM: place LLM models on GPUs to minimise the maximum KV-cache pressure.

See PROBLEM.md. The mutable region is marked by # EVOLVE-BLOCK-START /
# EVOLVE-BLOCK-END. Everything else in this file is fixed scaffolding — do not modify it.

The evaluator imports this file and calls `compute_model_placement(gpu_num, models)`
once per test case, with its own `Model` objects (the same five fields as the class
below). It supplies the cases; you never see them in advance.

Run `python3 solution.py` to place a small demo case and print its maximum KVPR.
"""
from dataclasses import dataclass

GPU_MEM_SIZE = 80 # GB


@dataclass
class Model:
    model_name: str
    model_size: int     # GB
    req_rate: int       # requests per second
    slo: int            # latency target, seconds
    cur_gpu_id: int


# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # Greedy KVPR-minimizing placement based on Algorithm 1 (without τ check)
    # 1) Sort models by r_j / s_j in descending order
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes current KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    return placement

# EVOLVE-BLOCK-END


def max_kvpr(placement) -> float:
    """The quantity the evaluator minimises per case: the largest KVPR over the GPUs,
    KVPR = sum(req_rate / slo) / (GPU_MEM_SIZE - sum(model_size)); a full GPU counts
    10^6. Same formula as the evaluator's."""
    worst = float("-inf")
    for models in placement.values():
        free = GPU_MEM_SIZE - sum(m.model_size for m in models)
        load = sum(m.req_rate / m.slo for m in models)
        worst = max(worst, load / free if free > 0 else 1000000)
    return worst


def _demo_case():
    """A small hand-written case for local runs. NOT one of the evaluator's cases."""
    specs = [(24, 7, 5), (18, 3, 8), (29, 9, 6), (12, 2, 9), (21, 6, 7),
             (15, 8, 5), (27, 4, 9), (10, 5, 6)]
    models = [Model(f"model_{j}", size, rate, slo, j)
              for j, (size, rate, slo) in enumerate(specs)]
    return 4, models


if __name__ == "__main__":
    # Local sanity print; the evaluator is authoritative.
    gpu_num, models = _demo_case()
    placement = compute_model_placement(gpu_num, models)
    for gpu_id in sorted(placement):
        names = ", ".join(m.model_name for m in placement[gpu_id])
        used = sum(m.model_size for m in placement[gpu_id])
        print(f"GPU {gpu_id}: {used:2d}/{GPU_MEM_SIZE} GB  [{names}]")
    print(f"max KVPR: {max_kvpr(placement):.4f}")
