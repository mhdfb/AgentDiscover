# Erdős Minimum Overlap

Find a step function h: [0,2] -> [0,1] that minimizes the maximum overlap integral,
providing an upper bound for the Erdős–Selfridge C5 constant.

The function h is discretized into n_points intervals over [0,2], with each h(x) in
[0,1]. The integral of h must equal 1 (i.e., `sum(h) * dx = 1` where `dx = 2/n_points`).

The objective is to minimize `C5 = max_k integral h(x)(1 - h(x+k)) dx`, computed via
cross-correlation of h with (1-h).

Your program file `solution.py` must define a `run()` function that returns
`(h_values, c5_bound, n_points)` where:

- `h_values`: numpy array of shape `(n_points,)` with values in [0,1]
- `c5_bound`: float, the achieved upper bound for C5
- `n_points`: int, number of discretization points

Score = BENCHMARK / c5_bound (higher is better; lower C5 yields higher score). The
benchmark (AlphaEvolve best) is 0.38092303510845016.

Consider:

- Gradient-based optimization (Adam, L-BFGS) with sigmoid parameterization
- FFT-based correlation computation for speed
- Penalty methods for the integral constraint
- Multi-start optimization with different initializations

Tips:

- Eval timeout is 1000s for `run()` (1100s for the whole evaluation). If your solution takes longer, it scores as a timeout.
- The evaluator checks: h in [0,1], integral of h = 1 (atol=1e-3), recomputes C5 via
  `np.correlate` and verifies consistency (atol=1e-4).
- numpy and scipy are always available, and so is a convex-optimisation toolbox:
  `cvxpy` with the solvers CLARABEL, CVXOPT, GLPK, GLPK_MI, HIGHS, OSQP, SCIP, SCIPY and
  SCS, plus `scipy.optimize.linprog(method="highs")`. These are worth knowing about
  because the inner step here is a linear program: with the binding shifts held fixed,
  minimising the maximum correlation subject to `h in [0,1]` and `sum(h) = n_points/2`
  is an LP. The same packages are installed in your own sandbox, so you can develop and
  test against them before submitting.
- Best known C5 upper bound is ~0.3809 (AlphaEvolve result).
- Score = 0.38092303510845016 / your_c5_bound; 1.0 means matching the benchmark.
- Use FFT-based convolution for efficient correlation computation.
- The integral constraint `sum(h)*dx = 1` must be satisfied precisely.

## Targets

The evaluator scores you on the **score** above — `BENCHMARK / C5`, higher is better —
and tracks **C5 itself, lower is better**, as your objective; that is the form results
for this problem are quoted in.

| | C5 ↓ | score ↑ |
|---|---|---|
| the seed below (random restarts + coordinate perturbation, n = 200) | 0.4252 | 0.896 |
| the benchmark, score 1.0 | 0.380923 | 1.0 |
| **target — beat this, clearly** | **0.38088** | **1.00011** |

The target is a bar to get **meaningfully under**, not to land on: reaching ~0.38088 is
not success. Note the scale of this problem — the whole spread of serious results lives
in the fourth and fifth decimal, so improvements are small in absolute terms and a
candidate's C5 is reported to ten decimals. Published results are deliberately not
listed here; a number someone else reached is an anchor, not a bound.

One thing the evaluator does that the original's description leaves implicit: the C5 it
scores is the one it **recomputes** from your `h` with `np.correlate`, not the one you
report. Your reported `c5_bound` still has to agree with it to 1e-4 or the candidate is
invalid, but it cannot move your score.

Alongside `score` (`combined_score`) the evaluator returns `c5_bound` (recomputed),
`reported_c5_bound`, `n_points`, `eval_time`, the benchmark and whether the score passed
1. A candidate that fails is told why in the original benchmark's words — a shape that
is not `(n_points,)`, a value outside [0,1], an integral off by more than 1e-3, a C5
mismatch beyond 1e-4, a `run()` that raised, or a timeout.

## Interface

`solution.py` defines `run()` returning `(h_values, c5_bound, n_points)` as above. The
evaluator imports it, calls it once in a fresh process, and checks and scores what it
returns with its own numpy call. `dx = 2 / n_points` throughout, so the integral
constraint is `sum(h) == n_points / 2` to within `1e-3 * n_points / 2`. There is no cap
on `n_points`; the check is one `np.correlate` and costs about 2 s at n = 100 000. The
whole file, imports included, is inside the EVOLVE-BLOCK, exactly as in the original
benchmark.

`python3 solution.py` prints the current function's integral, reported and recomputed C5
and score with the evaluator's own formula, so anything you measure locally scores the
same when submitted.

**Compute budget: `run()` may run for 1000 s.** Budget your search against that figure
and use it — leaving time unspent is wasted search, and overrunning it means the
candidate scores 0. The evaluation as a whole is capped at 1100 s, so leave a margin for
interpreter start-up and imports. Your briefing repeats the number in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- bang-bang structure: search over where `h` switches between 0 and 1, treating the problem as a choice of switching points rather than of heights
- continuous optimisation: keep the heights soft in [0,1] and push them with projected gradient or quasi-Newton steps on the exact objective
- minimax reformulation: attack the max over shifts directly — find which shifts are active at the optimum and equalise them
- resolution laddering: solve at coarse n, interpolate to finer n, and re-optimise from that warm start
