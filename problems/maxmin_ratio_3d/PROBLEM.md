# Max-to-min distance ratio — 14 points in three dimensions

For `n` distinct points `x₁, …, xₙ` in `ℝᵈ`, let

    ratio² = ( max_{i<j} ‖xᵢ − xⱼ‖ )²  /  ( min_{i<j} ‖xᵢ − xⱼ‖ )²

**Your task**: place **14 points in ℝ³** so that this squared ratio is as small as possible.
Lower is better, and every configuration is an upper bound on `C(3,14)` — the largest number
such that *every* 14-point configuration in space has `max distance ≥ C · min distance`.

The objective is **scale- and translation-invariant**, so there is no box to fit inside and
no normalisation to worry about. Only the shape matters.

> **Note on the metric.** The published numbers are the **squared** ratio, not the ratio
> itself: √4.16579 = 2.0410, the true max/min distance of the best known configuration.
> Implementations work in squared distances to avoid square roots, and the benchmark
> inherited that convention. Reporting the un-squared value would silently look like a huge
> improvement.

## Where the record stands

| Construction | ratio² |
|---|---|
| 3×3×2 grid, first 14 points (the seed below) | 9.0 |
| ShinkaEvolve / EvoX on Claude Opus 4.6 | 4.46 |
| OpenEvolve on Claude Opus 4.6 | 4.21 |
| EvoX (GPT-5) | 4.21 |
| CORAL on Claude Opus 4.6, EvoX (Gemini-3-Pro) | 4.16 |
| AlphaEvolve | 4.16585 |
| **best published** | **4.16579** (CodeEvolve) |
| proven lower bound | none published |

There is a natural landmark just below: the **cuboctahedron** — one point at the centre with
its 12 nearest neighbours around it — has 13 points with max distance exactly 2 and min
exactly 1, so `ratio² = 4`. The record of 4.16579 is what adding a fourteenth point costs.
Whether 4.16579 is optimal is unknown; no lower bound has ever been published.

Note that this task **separates the systems** more than most: on Claude Opus 4.6 the spread
runs from 4.46 down to 4.16, so it still discriminates rather than having collapsed to a
single value.

## Interface

`solve()` returns a list of 14 points, each a list of 3 finite floats. All points must be
distinct — a repeated point makes the minimum distance zero and the ratio undefined, and is
rejected with a message naming the offending pair.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that
figure and use it — leaving time unspent is wasted search, and overrunning it means
the candidate scores 0. The evaluation as a whole is capped at 700 s, so leave a
margin for interpreter start-up and imports. Your briefing repeats the number in
force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- structured clusters: start from the cuboctahedron, the icosahedron plus centre, and fcc/hcp fragments, then add or move the extra point
- minimax optimisation: identify which pairs realise the max and the min at the optimum and equalise them, treating it as a contact-graph problem
- continuous optimisation: smooth the max and min with a soft-max surrogate, run gradient descent from many random starts, then polish exactly
- shell decomposition: fix a centre point and optimise the remaining thirteen on one or two spherical shells, tuning the radii
