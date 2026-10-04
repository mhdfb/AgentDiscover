# First autocorrelation inequality

For a nonnegative function `f` supported on `[-1/4, 1/4]`, define `C1` as the largest
constant such that

    max_{|t| <= 1/2} (f * f)(t) >= C1 (integral f)^2

holds for every such `f`. The goal is to certify the tightest upper bound on `C1`; any
valid construction certifies

    C1 <= ||f * f||_infinity / ||f||_1^2.

Until early 2025, the best known upper bound was `C1 <= 1.50973`. AlphaEvolve improved
this to `C1 <= 1.5053`, AlphaEvolve V2 further improved it to `C1 <= 1.50317`, and
ThetaEvolve refined AlphaEvolve's construction to obtain `C1 <= 1.50314`.

## Evaluation

Return a list containing the nonnegative heights of an equal-width step function. For a
sequence `(a_0, ..., a_(n-1))`, the evaluator computes

    2 n max(a * a) / (sum a)^2,

where `a * a` is the discrete autoconvolution. Lower is better. Values below zero are
clipped to zero, values above `1000` are clipped to `1000`, and a sequence whose sum is
below `0.01` is invalid.

The source target is an upper bound of `1.5030`. Your search function has 1000 seconds
and may use up to 2 CPUs. As in the source environment, you may use NumPy, SciPy,
`math`, and CVXPY with the CBC, CVXOPT, GLOP, GLPK, PDLP, SCIP, and ECOS solvers.
It may call `evaluate_sequence()` as many times as needed.
Longer sequences can offer more room for improvement, but sequences with hundreds of
thousands of entries may be too slow to search within the budget. Explore substantially
different algorithms rather than only tuning the supplied local search. The upstream
TTT-Discover result is `1.50287`; a lower value here beats that published result.

## Previous approach

A previous state of the art began with a nonnegative step function normalized so its
heights sum to `sqrt(2n)`. It set `M = ||f*f||_infinity`, then found `g0` by linear
programming: maximize the sum of `g0` subject to `g0 >= 0` and
`||f*g0||_infinity <= M`. After normalizing `g0` the same way, it updated
`f <- (1-t)f + t g` for a small positive `t`, repeating until stable.

## Interface

`solution.py` must define `solve()`, returning the list of heights. The benchmark's
original seed performs a random-initialization/linear-program search, with its
`propose_candidate()` implementation retained inside the evolve block. Its
`height_sequence_1` is only the source's initial constant-height construction (objective
approximately `2.0`), not a dynamically updated incumbent.
