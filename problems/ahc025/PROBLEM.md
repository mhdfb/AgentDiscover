# AHC025 — Balancing by Balance

There are `N` items with unknown positive weights. Divide them into `D` groups with nearly equal total weight, using exactly `Q` balance queries. The judge first supplies `N D Q`, where `30<=N<=100`, `2<=D<=N/4`, and `2N<=Q<=32N`.

If group totals are `t_0...t_(D-1)` and their variance is `V`, the **absolute** case score is `1 + round(100*sqrt(V))`; lower is better. The contest also calculates a higher-is-better rank score from standings, but this evaluator searches on the absolute score.

## Input

```text
N D Q
```

Inputs are generated with `N` uniform in `[30,100]`, `D` uniform in `[2,floor(N/4)]`, and `Q=round(N*2^U)` for uniform real `U` in `[1,5)`. Hidden weights are independent exponential samples of rate `10^-5`, rounded to at least 1 and regenerated above `10^5*N/D`.

## Output

For each of exactly `Q` queries, print and flush:

```text
n_L n_R l_0 ... l_(n_L-1) r_0 ... r_(n_R-1)
```

Both sets must be nonempty, disjoint, and contain IDs in `0..N-1`. The judge replies `<`, `>`, or `=` for the sums of the left and right sets. After exactly `Q` replies, print:

```text
d_0 ... d_(N-1)
```

Each `d_i` must be a group ID in `0..D-1`.

## Interface

Use C++20 and define the entire program in the `SOURCE` raw string in `solution.py`. The benchmark's original seed program performs balance queries and group assignment. The evaluator uses all 50 ALE-Bench public cases (seeds 0–49) with 13 workers and the reactive tester. It minimizes the mean absolute case score and returns higher-is-better fitness `score = 330,000 / mean`; invalid candidates score zero. Aim to exceed **1.0 fitness** (mean absolute score below 330,000), a contest-winner reference from a different 5,000-case suite. Each case has a 2-second time limit and 1024 MiB memory limit.
