"""Erdős discrepancy problem (C=2). See PROBLEM.md for the full statement.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.
"""

# Longest sequence that can possibly have discrepancy 2 (Konev & Lisitsa, 2014).
MAX_LENGTH = 1160


def solve() -> list[int]:
    """Return a sequence of +1/-1 values. a[0] is the first term a_1.

    Scored by the length of the longest prefix with discrepancy <= 2.
    """
    # EVOLVE-BLOCK-START
    # Baseline: greedy extension. At each position try both signs and keep one
    # that does not push any homogeneous progression sum past 2. Stops at the
    # first position where neither sign works (no backtracking).
    sums: dict[int, int] = {}
    seq: list[int] = []

    def divisors(m: int) -> list[int]:
        ds, d = [], 1
        while d * d <= m:
            if m % d == 0:
                ds.append(d)
                if d != m // d:
                    ds.append(m // d)
            d += 1
        return ds

    for position in range(1, MAX_LENGTH + 1):
        ds = divisors(position)
        for sign in (1, -1):
            if all(abs(sums.get(d, 0) + sign) <= 2 for d in ds):
                for d in ds:
                    sums[d] = sums.get(d, 0) + sign
                seq.append(sign)
                break
        else:
            break  # neither sign keeps the discrepancy at 2

    return seq
    # EVOLVE-BLOCK-END


if __name__ == "__main__":
    s = solve()
    # Local sanity print; the evaluator is authoritative.
    print(f"length={len(s)} first_terms={s[:20]}")
