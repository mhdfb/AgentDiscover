# AHC058 — Apple Incremental Game

There are `N * L` machine types: IDs `j = 0, ..., N - 1` at levels
`i = 0, ..., L - 1`. Machine `j` at level 0 produces `A_j` apples. Every
machine type has an initial strengthening cost `C[i][j]`.

Initially there is one machine of every type, every machine has power zero, and
there are `K` apples. On each of `T` turns, take exactly one action:

- Strengthen machine `(i, j)`, spending `C[i][j] * (P[i][j] + 1)` apples and
  increasing its power by one. An action that makes the apple count negative is
  illegal.
- Do nothing.

After the action, production happens in level order `0, 1, 2, 3`:

- A level-0 machine adds `A[j] * B[0][j] * P[0][j]` apples.
- A machine at level `i >= 1` adds `B[i][j] * P[i][j]` machines to level
  `i - 1` with the same ID.

Maximize the number `S` of apples after all turns. A case scores
`round(100000 * log2(S))`; higher is better. The AtCoder judge evaluates a submission
on 150 test cases and uses their total as the submission score. An illegal output or a
time-limit violation on any case makes the whole submission WA or TLE. AtCoder did not
perform a separate system test after the contest.

## Input

```text
N L T K
A_0 A_1 ... A_(N-1)
C_(0,0) C_(0,1) ... C_(0,N-1)
...
C_(L-1,0) C_(L-1,1) ... C_(L-1,N-1)
```

For every case, `N = 10`, `L = 4`, `T = 500`, and `K = 1`. The values `A_j`
are sorted, with `1 <= A_j <= 100`; all costs satisfy
`1 <= C[i][j] <= 1.25 * 10^12`.

The instances are generated as follows. Let `rand_double(a, b)` be uniform over
the real interval `[a, b]`.

- `A_0 = 1`; for `j != 0`, `A_j = round(10^rand_double(0, 2))`; then `A` is
  sorted.
- `C[0][0] = 1`; otherwise,
  `C[i][j] = round(A_j * 500^i * 10^rand_double(0, 2))`.

## Output

Output exactly `T` actions, one per line. To strengthen machine `(i, j)`, print
`i j`. To do nothing, print `-1`. Comment lines beginning with `#` are allowed.
Taking fewer than `T` actions, choosing a nonexistent machine, or buying an
unaffordable strengthening is invalid.

## Interface

Use C++20 and define the entire program in the `SOURCE` raw string in
`solution.py`. The evaluator compiles it and reports the mean absolute score
over a fixed local benchmark suite of 150 cached generated instances (seeds 0
through 149). The search target is local mean performance `6500000`, and the
fitness is `mean / 3000000`.

The starting
`SOURCE` is deliberately empty. Build the first program from the statement.
Try diverse approaches. A strong solution should use the full 2 second time
limit without exceeding it.
