# Minimizing Max-Min Distance 2D

Place 16 points in 2 dimensions to maximize the ratio `(min_distance / max_distance)^2`
over all pairwise distances.

Your program file `solution.py` must define a `run()` function that returns a numpy
array of shape (16, 2) with the (x, y) coordinates of the 16 points.

The score is computed as:

    (min_dist / max_dist)^2 / BENCHMARK

where BENCHMARK = 1 / 12.889266112.

A score of 1.0 means matching the best known result; above 1.0 is a new record.

Tips:

- Eval timeout is 600s. If your solution takes longer, it scores as a timeout.
- The evaluator computes all pairwise distances using `scipy.spatial.distance.pdist`.
- numpy and scipy are always available.
- Score = (min_dist/max_dist)^2 / (1/12.889266112); 1.0 means matching the best known
  result.
- Key reformulation: maximize (d_min/d_max)^2, equivalent to finding points where
  pairwise distances are as uniform as possible.
- Consider optimization approaches: simulated annealing, gradient descent,
  basin-hopping.

## Targets

The evaluator scores you on the **score** above — `(min/max)^2 / BENCHMARK`, higher is
better — and tracks the same quantity the other way up as your objective:
`ratio^2 = (max_distance / min_distance)^2`, **lower is better**, which is the form
results for this problem are usually quoted in (BENCHMARK is 1/12.889266112 because
12.889266112 is the best known `ratio^2`).

| | ratio² (max/min) ↓ | score ↑ |
|---|---|---|
| the seed below (16 Gaussian random points, fixed seed) | ~651 | ~0.020 |
| **target — beat this, clearly** (the benchmark, score 1.0) | **12.889266112** | **1.0** |

The target is a bar to get **meaningfully under**, not to land on: reaching
~12.889266 is not success — it is the benchmark itself, and nothing marks it as a limit.
No lower bound is published for 16 points in the plane, so how far below the answer
lies is open. Published results are deliberately not listed here; a number
someone else reached is an anchor, not a bound.

Alongside `score` (`combined_score`) the evaluator returns `min_max_ratio`
(`(min/max)^2`), `ratio_squared`, `eval_time`, the benchmark and whether the score
passed 1. A candidate that fails is told why in the original benchmark's words — an
invalid shape, a `run()` that raised, or a timeout. Two coincident points make the
minimum distance 0 and the score 0; that is not an error, just a score of zero.

## Interface

`solution.py` defines `run()` returning a numpy array (or anything `np.array` turns into
one) of shape exactly `(16, 2)`. The evaluator imports it, calls it once in a fresh
process, and scores the points it returns: `scipy.spatial.distance.pdist`, then
`(min/max)^2 / BENCHMARK`. Scale, translation and rotation do not matter — only the
shape of the configuration. The whole file, imports included, is inside the
EVOLVE-BLOCK, exactly as in the original benchmark.

`python3 solution.py` prints the current configuration's `(min/max)^2`, `ratio^2` and
score with the evaluator's own formula, so anything you measure locally scores the same
when submitted.

**Compute budget: `run()` may run for 600 s.** Budget your search against that figure
and use it — leaving time unspent is wasted search, and overrunning it means the
candidate scores 0. The evaluation as a whole is capped at 700 s, so leave a margin for
interpreter start-up and imports. Your briefing repeats the number in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- structured configurations: start from concentric regular polygons, hexagonal-lattice fragments and symmetric arrangements, then perturb and polish
- minimax optimisation: identify which pairs realise the max and the min at the optimum and equalise them, treating it as a contact-graph problem
- continuous optimisation: smooth the max and min with a soft-max surrogate, run gradient descent from many random starts, then polish exactly
- global search: simulated annealing or basin-hopping over point coordinates with many random restarts, keeping a diverse pool of local optima
