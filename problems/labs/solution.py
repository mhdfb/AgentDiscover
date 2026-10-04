"""LABS: low-autocorrelation binary sequences. See PROBLEM.md for the full statement.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""
import random

# Fixed problem size. Do not change this.
N = 60


def solve() -> list[int]:
    """Return a sequence of length N with entries in {-1, +1}."""
    # EVOLVE-BLOCK-START
    # Baseline: a single random ±1 sequence. Replace with a better method.
    random.seed(0)
    return [random.choice([-1, 1]) for _ in range(N)]
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    seq = solve()
    # Local sanity print; evaluate.py is authoritative.
    n = len(seq)
    total = 0
    for k in range(1, n):
        c = sum(seq[i] * seq[i + k] for i in range(n - k))
        total += c * c
    mf = n * n / (2 * total) if total > 0 else float("inf")
    print(f"N={n} merit_factor={mf:.4f}")
