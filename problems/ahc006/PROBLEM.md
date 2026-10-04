# AHC006 — Food Delivery

Choose exactly 50 of 1000 orders. Order `i` has pickup `(a_i,b_i)` and delivery `(c_i,d_i)`. Start and finish at `(400,400)` and visit each chosen pickup before its delivery. You may carry any number of orders at once. Minimize the route's total Manhattan length `T`; the case score is `round(10^8/(1000+T))`. Higher is better. A non-AC case makes the contest's 100-case submission score zero.

## Input

```text
a_1 b_1 c_1 d_1
...
a_1000 b_1000 c_1000 d_1000
```

Coordinates are integers in `[0,800]`. Orders are generated uniformly, rejecting pickup/delivery pairs whose Manhattan distance is under 100.

## Output

```text
50 r_1 ... r_50
n x_1 y_1 ... x_n y_n
```

Order IDs are 1-based. The route must start and end at `(400,400)`. If several solutions are printed, only the last is scored.

## Interface

Use C++20 and define the entire program in the `SOURCE` raw string in `solution.py`. The starting `SOURCE` is empty; build the first program from the statement. The evaluator uses all 50 ALE-Bench public cases (seeds 0–49) with 13 workers. Its fitness `score` is the mean absolute case score, with accepted-case partial reward if some cases fail. There is no extra fitness normalization. Aim to exceed **23,000 mean points**, a contest-winner reference from a different 100-case suite. Each case has a 2-second time limit and 1024 MiB memory limit.
