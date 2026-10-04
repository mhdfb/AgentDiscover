"""Sum-difference problem: maximise log|A-A| / log|A+A|. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Limits enforced by the evaluator.
MAX_SIZE = 4000
MAX_ABS = 10 ** 12


def solve() -> list[int]:
    """Return a list of distinct integers A.

    Scored by log|A-A| / log|A+A|, which certifies C >= that value.
    """
    # EVOLVE-BLOCK-START
    # Baseline: the classic small set with more differences than sums,
    # {0, 2, 3, 4, 7, 11, 12, 14}, which has |A+A| = 26 and |A-A| = 25.
    return [0, 2, 3, 4, 7, 11, 12, 14]
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    import math

    a = solve()
    # Local sanity print; the evaluator is authoritative.
    s = len({x + y for x in a for y in a})
    d = len({x - y for x in a for y in a})
    print(f"|A|={len(a)} |A+A|={s} |A-A|={d} ratio={math.log(d) / math.log(s):.6f}")
