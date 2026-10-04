"""First autocorrelation inequality Python candidate. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

import os

# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy import optimize
linprog = optimize.linprog


def get_good_direction_to_move_into(sequence):
    """Returns a better direction using LP to find g with larger sum while keeping conv bounded."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_g = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
    t = 0.01
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    return new_sequence


def solve_convolution_lp(f_sequence, rhs):
    """Solves the LP: maximize sum(b) s.t. conv(f, b) <= rhs, b >= 0."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)
    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, options={
        'time_limit': 10.0,   # seconds, make sure we don't get stuck
        'disp': False,
    })
    if result.success:
        return result.x
    return None


def propose_candidate(seed=42, budget_s=1000, **kwargs):
    np.random.seed(seed)
    deadline = time.time() + budget_s - 10
        
    if np.random.rand() < 0.5:
        # Start from the SOTA sequence (already available as height_sequence_1)
        best_sequence = list(height_sequence_1)
    else:
        # Start from random initialization, could help if height_sequence_1 is a local minimum
        best_sequence = [np.random.random()] * np.random.randint(100, 1000)
    curr_sequence = best_sequence.copy()
    best_score = evaluate_sequence(best_sequence)
    
    while time.time() < deadline:
        h_function = get_good_direction_to_move_into(curr_sequence)
        if h_function is None:
            # Random perturbation if LP fails
            idx = np.random.randint(len(curr_sequence))
            curr_sequence[idx] = max(0, curr_sequence[idx] + np.random.randn() * 0.01)
        else:
            curr_sequence = h_function
        
        try:
            curr_score = evaluate_sequence(curr_sequence)
            if curr_score < best_score:
                best_score = curr_score
                best_sequence = curr_sequence.copy()
        except:
            pass
    
    return best_sequence
# EVOLVE-BLOCK-END


def evaluate_sequence(sequence: list[float]) -> float:
    """Source-provided helper, injected by TTT-Discover in the original environment."""
    if not isinstance(sequence, list) or not sequence:
        return np.inf
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return np.inf
        if np.isnan(value) or np.isinf(value):
            return np.inf
    sequence = [min(1000.0, max(0.0, float(value))) for value in sequence]
    total = np.sum(sequence)
    if total < 0.01:
        return np.inf
    return float(2 * len(sequence) * max(np.convolve(sequence, sequence)) / total**2)


_initial_rng = np.random.default_rng(12345)
height_sequence_1 = [_initial_rng.random()] * _initial_rng.integers(1000, 8000)


def solve() -> list[float]:
    budget = int(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", "1000"))
    return propose_candidate(budget_s=budget)


if __name__ == "__main__":
    result = solve()
    print(len(result), evaluate_sequence(result))
