"""Third autocorrelation inequality: minimise max|f*f| / (int f)^2. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Limit enforced by the evaluator.
MAX_N = 4000


def solve() -> list[float]:
    """Return the heights of a step function f on [-1/4, 1/4].

    Heights may be negative — the cancellation that allows is the whole point.
    Only the shape matters; scaling f by a constant leaves the score unchanged.
    Scored by max|f*f| / (int f)^2; lower is better.
    """
    # EVOLVE-BLOCK-START
    # Baseline: the indicator of the interval, which scores exactly 2.0.
    n = 100
    return [1.0] * n
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    f = solve()
    n = len(f)
    # Local sanity print; the evaluator is authoritative.
    c = [0.0] * (2 * n - 1)
    for i, fi in enumerate(f):
        for j, fj in enumerate(f):
            c[i + j] += fi * fj
    print(f"n={n} C={2 * n * max(abs(v) for v in c) / sum(f) ** 2:.6f}")
