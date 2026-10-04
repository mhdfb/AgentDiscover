"""Max-to-min distance ratio: 14 points in three dimensions. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Fixed problem size. Do not change these.
N_POINTS = 14
DIM = 3


def solve() -> list[list[float]]:
    """Return 14 distinct points in R^3.

    Scored by (max pairwise distance / min pairwise distance)^2; lower is better.
    Scale and translation do not matter — only the shape of the configuration.
    """
    # EVOLVE-BLOCK-START
    # Baseline: the first 14 points of a 3x3x2 unit grid. Min distance 1, max
    # distance 3 (from (0,2,0) to (2,0,1)), so the squared ratio is exactly 9.
    grid = [[float(x), float(y), float(z)]
            for z in range(2) for y in range(3) for x in range(3)]
    return grid[:N_POINTS]
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    import math

    pts = solve()
    # Local sanity print; the evaluator is authoritative.
    ds = [sum((a - b) ** 2 for a, b in zip(p, q))
          for i, p in enumerate(pts) for q in pts[i + 1:]]
    print(f"n={len(pts)} ratio^2={max(ds) / min(ds):.9f} ratio={math.sqrt(max(ds) / min(ds)):.6f}")
