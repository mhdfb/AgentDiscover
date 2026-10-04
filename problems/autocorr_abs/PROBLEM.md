# Third autocorrelation inequality — functions that may go negative

Let `C` be the best constant for which

    max_{-1/2 ≤ t ≤ 1/2}  | ∫ f(t−x) f(x) dx |  ≥  C · ( ∫_{-1/4}^{1/4} f(x) dx )²

holds for every `f : [-1/4, 1/4] → ℝ`. Note that **`f` may take negative values** — that is
what separates this from its better-known siblings, and it is what makes the problem hard:
cancellation in the autoconvolution is allowed, so the peak of `|f ∗ f|` can be pushed down
much further than for a non-negative `f`.

**Your task**: find an `f` making that peak as small as possible relative to `(∫f)²`. Every
`f` you produce certifies `C ≤ ratio(f)`, so your score is an upper bound and lower is better.

## Where the record stands

| Construction | Upper bound on C |
|---|---|
| indicator of the interval (the seed below) | 2.0 |
| EvoX (Gemini-3-Pro) | 1.459 |
| EvoX / OpenEvolve (GPT-5) | 1.461 |
| ThetaEvolve (8B open model) | 1.4930 |
| **AlphaEvolve, and matched by CORAL on Claude Opus 4.6** | **1.4557** |

No lower bound has ever been published for this constant, so unlike the minimum-overlap
problem **nobody knows how much room is left below 1.4557**. That is the reason to work on
it: the record has been matched but never beaten, and there is no proof it cannot fall.

CORAL matching 1.4557 on Opus 4.6 is the best published Claude result. Beating 1.4557 sets
a record outright.

> **A naming hazard.** DeepMind's own problem page warns that this family of autocorrelation
> problems is routinely conflated in the literature — *"some results for one problem
> incorrectly attributed to another."* This problem is the one where `f` may be **negative**
> and the objective involves `|f ∗ f|`. Its sibling in this repository, `autocorr_l2`, is a
> different constant over non-negative `f`. Do not carry numbers between them.

## The search space

You return a **step function**: `n` heights on `[-1/4, 1/4]`, each held over an interval of
width `1/(2n)`. Heights may be any sign.

Scoring is **exact, not numerical**: the autoconvolution of a step function is piecewise
linear, so `max |f ∗ f|` is attained at a grid point and is computed in closed form from your
heights. There is no integrator to exploit — a fact worth stating, because DeepMind's first
attempt at a sibling problem let the model propose arbitrary analytic functions and it
learned to game the quadrature instead of solving the problem.

Only the shape matters: scaling `f` by any non-zero constant leaves the ratio unchanged.

## Interface

`solve()` returns a list of `n` floats (`2 ≤ n ≤ 4000`), the step heights, with `∫f ≠ 0`.
`numpy` is available.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that figure
and use it — leaving time unspent is wasted search, and overrunning it means the whole
candidate scores 0. The evaluation as a whole is capped at 700 s, so keep a margin for
interpreter start-up and imports. Your briefing repeats the number in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- exploit cancellation: place negative lobes deliberately so the autoconvolution cancels near its peak, rather than treating sign as a free parameter
- minimax equalisation: find which points of `|f ∗ f|` are active at the optimum and flatten the peak across all of them
- continuous optimisation: parametrise the heights and push them with gradient or quasi-Newton steps on the exact objective
- resolution laddering: solve at coarse n, interpolate to finer n, and re-optimise from that warm start
