"""Circle packing: 26 disjoint disks in the unit square, maximise the sum of radii.
See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Fixed problem size. Do not change this.
N_CIRCLES = 26


def solve() -> list[list[float]]:
    """Return 26 triples [x, y, r]: the centre and radius of each disk.

    Disks must lie inside the unit square and must not overlap; touching is fine.
    Scored by the sum of the radii.
    """
    # EVOLVE-BLOCK-START
    # Baseline: a 5x5 grid of radius-0.1 disks, plus one disk dropped into the
    # gap between four of them. Sum of radii = 2.5414...
    circles = []
    for row in range(5):
        for col in range(5):
            circles.append([0.1 + 0.2 * col, 0.1 + 0.2 * row, 0.1])
    # The gap at a four-way junction fits a disk of radius 0.1*(sqrt(2) - 1).
    gap_r = 0.1 * (2.0 ** 0.5 - 1.0)
    circles.append([0.2, 0.2, gap_r])
    return circles
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    import math

    cs = solve()
    # Local sanity print; the evaluator is authoritative.
    ok = all(
        math.hypot(a[0] - b[0], a[1] - b[1]) >= a[2] + b[2] - 1e-9
        for i, a in enumerate(cs) for b in cs[i + 1:]
    )
    print(f"n={len(cs)} sum_radii={sum(c[2] for c in cs):.6f} disjoint={ok}")
