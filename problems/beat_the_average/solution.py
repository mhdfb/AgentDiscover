"""Beat the average game: maximise P[X1 + X2 + X3 < 2*X4]. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Upper limit on the support size the evaluator accepts.
MAX_N = 20_000


def solve() -> list[float]:
    """Return non-negative weights w_i, read as P[X = i] proportional to w_i.

    The evaluator normalises them, so they need not sum to 1.
    """
    # EVOLVE-BLOCK-START
    # Baseline: a two-atom distribution. Most mass sits at 0, a little at 1,
    # so a draw of 1 beats three draws that are usually 0.
    w = [0.0] * 8
    w[0] = 0.75
    w[1] = 0.25
    return w
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Local sanity print; the evaluator is authoritative.
    weights = solve()
    print(f"support={len(weights)} nonzero={sum(1 for x in weights if x > 0)}")
