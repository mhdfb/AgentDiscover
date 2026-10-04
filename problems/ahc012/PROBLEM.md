# AHC012 — AtCoder 10th Anniversary

A circular cake of radius `10^4` has `N` strawberries. Make at most `K=100` full straight-line cuts. For `d=1..10`, `a_d` people want a piece containing exactly `d` strawberries. If `b_d` pieces contain `d` strawberries, maximize `sum_d min(a_d,b_d)`. A strawberry whose center lies on a cut belongs to no piece. The case score is `round(10^6 * sum_d min(a_d,b_d) / sum_d a_d)`. Higher is better. A non-AC case makes the contest's 100-case submission score zero.

## Input

```text
N K
a_1 ... a_10
x_1 y_1
...
x_N y_N
```

Here `N=sum_d d*a_d`, `1<=a_d<=100`, and each strawberry center lies strictly within the cake. Centers are sampled uniformly in the circle, rejecting a new point within distance 10 of a previous center.

## Output

```text
k
p_x^1 p_y^1 q_x^1 q_y^1
...
p_x^k p_y^k q_x^k q_y^k
```

`0<=k<=K`. Each pair of different integer points specifies a full cut line, and every point coordinate must lie in `[-10^9,10^9]`. Several solutions may be printed; only the last is scored.

## Interface

Use C++20 and define the entire program in the `SOURCE` raw string in `solution.py`. The starting `SOURCE` is empty; build the first program from the statement. The evaluator uses all 50 ALE-Bench public cases (seeds 0–49) with 13 workers. Its fitness `score` is the mean absolute case score, with accepted-case partial reward if some cases fail. There is no extra fitness normalization. Aim to reach **1,000,000 mean points**, the score ceiling already achieved on a different 100-case suite. Each case has a 3-second time limit and 1024 MiB memory limit.
