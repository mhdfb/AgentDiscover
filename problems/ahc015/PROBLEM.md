# AHC015 — Halloween Candy

A 10×10 box receives 100 candies of three flavors. The full flavor sequence `f_1 ... f_100` is known at the start. At turn `t`, the judge supplies `p_t`, the new candy's position among empty cells in front-to-back, left-to-right order. Tilt all candies forward, backward, left, or right until each is blocked. The next position is withheld until the direction is printed and flushed. The 100th tilt has no effect and may be omitted. Each flavor is uniform in `{1,2,3}`; each position is uniform among empty cells.

At the end, let `n_j` be the sizes of four-neighbor connected components of same-flavor candies, and let `d_i` be the total count of flavor `i`. The case score is `round(10^6 * sum_j n_j^2 / sum_i d_i^2)`. Higher is better. A non-AC case makes the contest's 200-case submission score zero.

## Input

```text
f_1 ... f_100
```

Then at each turn the judge gives one integer `p_t` in `[1,101-t]` after the previous direction has been printed.

## Output

After each `p_t`, print and flush one of `F`, `B`, `L`, or `R` to choose the tilt. The 100th output may be skipped.

## Interface

Use C++20 and define the entire program in the `SOURCE` raw string in `solution.py`. The benchmark's original seed program performs lookahead. The evaluator uses all 50 ALE-Bench public cases (seeds 0–49) with 13 workers and the reactive tester. Its fitness `score` is the mean absolute case score, with accepted-case partial reward if some cases fail. There is no extra fitness normalization. Aim to exceed **810,000 mean points**, a contest-winner reference from a different 200-case suite. Each case has a 2-second time limit and 1024 MiB memory limit.
