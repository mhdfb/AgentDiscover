"""Erdős minimum overlap problem: minimise the worst-case overlap. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Limits enforced by the evaluator.
MAX_N = 4000


def solve() -> list[float]:
    """Return the heights of a step function f on [-1, 1].

    Every height must lie in [0, 1], and the heights must sum to exactly n/2
    (that is the constraint int f = 1, since each step has width 2/n).
    Scored by max over shifts of the cross-correlation of f with 1 - f; lower is better.
    """
    # EVOLVE-BLOCK-START
    # Baseline: the flat function f == 1/2, which satisfies the mass constraint
    # (sum = n/2) and scores exactly 0.5.
    n = 200
    return [0.5] * n
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    f = solve()
    n = len(f)
    h = 2.0 / n
    g = [1.0 - x for x in f]
    # Local sanity print; the evaluator is authoritative.
    best = 0.0
    for k in range(-(n - 1), n):
        s = sum(f[i] * g[i + k] for i in range(n) if 0 <= i + k < n)
        best = max(best, s)
    print(f"n={n} sum={sum(f):.6f} (needs {n / 2}) overlap={h * best:.6f}")
