# Second autocorrelation inequality

For a nonnegative function `f`, define the second autocorrelation ratio

    C2(f) = ||f * f||_2^2 / (||f * f||_1 ||f * f||_infinity).

The goal is to construct a function giving the largest certified lower bound on `C2`.
Young's inequality implies that the ratio is at most `1`.

## Evaluation

Return a list containing the nonnegative heights of an equal-width step function. The
evaluator clips negative values to zero and values above `1000` to `1000`, forms the
discrete autoconvolution, and interprets it as a piecewise-linear function with zero at
both endpoints. It computes `||f*f||_2^2` exactly on those linear pieces, uses the
corresponding grid-scaled `L1` norm, and divides by the maximum convolution value.
Higher is better.

The source target is a lower bound of `0.97`. Your search function has 1000 seconds and
may use up to 2 CPUs. As in the source environment, you may use NumPy, SciPy, `math`,
and CVXPY with the CBC, CVXOPT, GLOP, GLPK, PDLP, SCIP, and ECOS solvers. It may call
`evaluate_sequence()` as many times as needed. Longer sequences can offer more room for
improvement, but sequences with hundreds of thousands of entries may be too slow to
search within the budget. The upstream TTT-Discover result is `0.9591`; a higher value
here beats that published result.

## Previous approach

A previous state of the art used a coarse-to-fine optimization. It began with stochastic
global search, repeatedly perturbing the incumbent and retaining improvements while
gradually reducing the perturbation scale. It then used projected gradient ascent for
local refinement. A good low-resolution solution was lifted by repeating its entries and
refined again; repeating this explore-refine-upscale cycle produced the final bound.

## Interface

`solution.py` must define `solve()`, returning the list of heights. The supplied seed is
the source's ThetaEvolve-derived four-phase gradient search, retained inside the evolve
block. Its `height_sequence_1` is only the source's initial constant-height construction,
not a dynamically updated incumbent.
