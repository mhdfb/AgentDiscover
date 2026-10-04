# AHC039 — Purse Seine Fishing

There are `N` mackerels and `N` sardines on a two-dimensional plane. Construct a
polygon and maximize the number of mackerels inside it minus the number of sardines
inside it. Points on an edge count as inside.

The polygon must satisfy all of these conditions:

1. It has at most 1000 vertices and total edge length at most `4 * 10^5`.
2. Every vertex has integer coordinates `(x, y)` with `0 <= x, y <= 10^5`.
3. Every edge is parallel to the x-axis or y-axis.
4. It does not self-intersect: non-adjacent edges share no points, and adjacent edges
   meet only at their endpoints.

If `a` mackerels and `b` sardines are inside, the case score is
`max(0, a - b + 1)`. The AtCoder judge evaluates a submission on 150 test cases and
uses their total as the submission score. An illegal output or exceeding the 2 second
limit on any case makes the whole submission WA or TLE. AtCoder did not perform a
separate system test after the contest. Only the highest score among a participant's
in-contest submissions determined their contest rank; equal scores received the same
rank. Higher is better.

## Input

```text
N
x_0 y_0
...
x_(2N-1) y_(2N-1)
```

In every case `N = 5000`. The first `N` points are mackerels and the remaining `N`
points are sardines. All coordinates are distinct.

## Output

```text
m
a_0 b_0
...
a_(m-1) b_(m-1)
```

Here `4 <= m <= 1000`. All output vertices must be distinct, but three consecutive
vertices may be collinear. Vertices may be clockwise or counterclockwise. A program may
print multiple solutions; only the last is scored.

## Interface

Use C++20 and define the entire program in the `SOURCE` raw string in `solution.py`.
`solve()` returns that source string for local inspection. The evaluator compiles it
and reports the mean absolute score over a fixed local benchmark suite of 150 cached
generated instances (seeds 0 through 149). The benchmark's original seed had local mean
performance `3755.4`; the search target is `5000`.

The local evaluator reports the mean to stay on the source search environment's reward
scale. On this fixed local suite, maximizing the mean is exactly equivalent to
maximizing the sum of the 150 case scores.

Try diverse approaches. A strong solution should use the full 2 second time limit
without exceeding it.
