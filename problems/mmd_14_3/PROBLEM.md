# Minimizing Max-Min Distance (3D, 14 points)

Place 14 points in 3D space to maximize the ratio of minimum to maximum pairwise
distance.

Given N=14 points in R^3, compute all pairwise distances and find:

    ratio = (min_distance / max_distance)^2

The goal is to maximize this ratio by placing points as "uniformly" as possible.

Your program file `solution.py` must define a `run()` function that returns a numpy
array of shape (14, 3) containing the point coordinates.

Score = ratio / BENCHMARK where BENCHMARK = 1/4.165849767. A score of 1.0 means matching
the benchmark; higher is better.

Consider:

- Optimization on the sphere (normalize to unit max distance)
- Simulated annealing with pairwise distance objectives
- Force-directed methods (repulsion between close points)
- Known polyhedra and geometric constructions for 14 points
- Gradient descent on smooth objective formulations

Tips:

- Eval timeout is 600s. If your solution takes longer, it scores as a timeout.
- The evaluator checks: shape is (14, 3), computes pairwise distances via
  `scipy.spatial.distance.pdist`.
- numpy and scipy are always available.
- Best known ratio for 14 points in 3D has BENCHMARK = 1/4.165849767.
- Score = (min_dist/max_dist)^2 / BENCHMARK; 1.0 means matching the best known result.
- Higher ratio values yield higher scores.

## Targets

The evaluator scores you on the **score** above — `(min/max)^2 / BENCHMARK`, higher is
better — and tracks the same quantity the other way up as your objective:
`ratio^2 = (max_distance / min_distance)^2`, **lower is better**, which is the form
results for this problem are usually quoted in (BENCHMARK is 1/4.165849767 because
4.165849767 is the best known `ratio^2`).

| | ratio² (max/min) ↓ | score ↑ |
|---|---|---|
| the seed below (14 Gaussian random points, fixed seed) | ~231.5 | ~0.018 |
| the benchmark, score 1.0 | 4.165849767 | 1.0 |
| **target — beat this, clearly** | **4.16** | **1.0014** |

The target is a bar to get **meaningfully under**, not to land on: reaching ~4.16 is not
success, and no lower bound is published for 14 points in three dimensions, so how far
below it the answer lies is open. A landmark sits just below: the 13-point
cuboctahedron — a centre with its 12 nearest neighbours — has max distance exactly 2 and
min exactly 1, so `ratio^2 = 4`; everything above 4 is the price of the fourteenth point.
Published results are deliberately not listed here; a number someone else reached is an
anchor, not a bound.

Alongside `score` (`combined_score`) the evaluator returns `min_max_ratio`
(`(min/max)^2`), `ratio_squared`, `min_distance`, `max_distance`, `eval_time`, the
benchmark and whether the score passed 1. A candidate that fails is told why in the
original benchmark's words — an invalid shape, NaN coordinates, a `run()` that raised,
or a timeout.

## Interface

`solution.py` defines `run()` returning a numpy array (or anything `np.array` turns into
one) of shape exactly `(14, 3)`, with no NaN. The evaluator imports it, calls it once in
a fresh process, and scores the points it returns: `scipy.spatial.distance.pdist`, then
`(min/max)^2 / BENCHMARK`. Scale and translation do not matter — only the shape of the
configuration. Two coincident points make the minimum distance 0 and the score 0; that is
not an error, just a score of zero. The whole file, imports included, is inside the
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

- structured clusters: start from the cuboctahedron, the icosahedron plus centre, and fcc/hcp fragments, then add or move the extra point
- minimax optimisation: identify which pairs realise the max and the min at the optimum and equalise them, treating it as a contact-graph problem
- continuous optimisation: smooth the max and min with a soft-max surrogate, run gradient descent from many random starts, then polish exactly
- shell decomposition: fix a centre point and optimise the remaining thirteen on one or two spherical shells, tuning the radii
