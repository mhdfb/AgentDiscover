# Third autocorrelation inequality

Minimize the C3 constant in the third autocorrelation inequality.

Given a function f on [0, 0.5] (can be positive or negative), the C3 constant is:

    C3 = max(|autoconvolution(f)|) / (integral(f))^2

where autoconvolution is computed as `np.convolve(f, f, mode="full") * dx`,
`dx = 0.5 / n_points`, and the integral is `sum(f) * dx`.

Note: unlike C1, f is NOT required to be non-negative for C3, and we take the maximum of
the absolute value of the autoconvolution.

Your program file `solution.py` must define a `run()` function that returns
`(f_values, c3_achieved, loss, n_points)` where:

- `f_values`: numpy array of shape `(n_points,)`, function values (may be negative)
- `c3_achieved`: float, the C3 ratio achieved
- `loss`: float, the optimization loss
- `n_points`: int, number of discretization points

The goal is to minimize C3 (find a tighter upper bound).
Score = BENCHMARK / C3 where BENCHMARK = 1.4556427953745406.
A score of 1.0 means matching the benchmark; higher is better.

Consider:

- Gradient-based optimization (Adam, L-BFGS)
- Allowing f to take negative values for better C3 bounds
- Multi-start optimization to escape local minima
- Learning rate schedules with warmup

## Tips

- Eval timeout is 600s. If your solution takes longer, it scores as a timeout.
- The evaluator verifies: shape matches `n_points`, integral is non-zero, recomputes C3
  independently.
- numpy and scipy are always available.
- **The number to beat is C3 = 1.4557**, and to beat it clearly: matching it is not the
  result this problem exists to produce. Getting under it puts your score above 1.
- Score = BENCHMARK / your_C3. Lower C3 values yield higher scores.
- This is a non-convex problem with multiple local optima.
- The seed below reaches C3 ≈ 2.758 (score ≈ 0.528) with a plain random-perturbation
  search, so there is a great deal of room before the benchmark is even in sight.

## Interface

`run()` is called once, in a fresh subprocess, with the whole 600 s budget. Everything in
`solution.py` between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` is yours to change
or replace, as long as `run()` keeps its name and returns that 4-tuple. Alongside your
score the evaluator returns `c3`, the value it recomputed independently, their difference,
your `loss` and `n_points`. Your briefing repeats the budget in force each session.
