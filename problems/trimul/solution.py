"""TriMul (AlphaFold3 outgoing triangle multiplicative update) — Triton kernel candidate.
See PROBLEM.md.

This whole file is the submission: the evaluator copies it next to the competition's
harness as `submission.py` and imports `custom_kernel` from it. The mutable region is
marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END; keep the entry point's name and
signature.
"""

# EVOLVE-BLOCK-START
# No previous attempt has been made. Below is the entry point's skeleton, as given in the
# task statement; it is not a working kernel yet.
import torch
import triton
import triton.language as tl


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]

    # Access the given weights of the model
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # Perform TriMul
    raise NotImplementedError("No previous attempt has been made.")

    return out
# EVOLVE-BLOCK-END
