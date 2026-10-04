"""Max-to-min distance ratio: 16 points in the plane. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Fixed problem size. Do not change these.
N_POINTS = 16
DIM = 2


def solve() -> list[list[float]]:
    """Return 16 distinct points in the plane.

    Scored by (max pairwise distance / min pairwise distance)^2; lower is better.
    Scale and translation do not matter — only the shape of the configuration.
    """
    # EVOLVE-BLOCK-START
    # Baseline: a 4x4 unit grid. Min distance 1, max distance sqrt(18),
    # so the squared ratio is exactly 18.
    return [[float(col), float(row)] for row in range(4) for col in range(4)]
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    import math

    pts = solve()
    # Local sanity print; the evaluator is authoritative.
    ds = [sum((a - b) ** 2 for a, b in zip(p, q))
          for i, p in enumerate(pts) for q in pts[i + 1:]]
    print(f"n={len(pts)} ratio^2={max(ds) / min(ds):.9f} ratio={math.sqrt(max(ds) / min(ds)):.6f}")
