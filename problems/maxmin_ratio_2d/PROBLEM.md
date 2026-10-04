# Max-to-min distance ratio — 16 points in the plane

For `n` distinct points `x₁, …, xₙ` in `ℝᵈ`, let

    ratio² = ( max_{i<j} ‖xᵢ − xⱼ‖ )²  /  ( min_{i<j} ‖xᵢ − xⱼ‖ )²

**Your task**: place **16 points in the plane** so that this squared ratio is as small as
possible. Lower is better, and every configuration you produce is an upper bound on the
constant `C(2,16)` — the largest number such that *every* 16-point planar configuration has
`max distance ≥ C · min distance`.

Intuitively you are packing 16 points as evenly as possible: spreading them out raises the
maximum distance, crowding them lowers the minimum, and the optimum balances the two.

The objective is **scale- and translation-invariant** — multiplying every coordinate by a
constant or shifting the whole configuration leaves the ratio unchanged — so there is no box
to fit inside and no normalisation to worry about. Only the shape matters.

> **Note on the metric.** The published numbers for this task are the **squared** ratio, not
> the ratio itself: √12.889230201 = 3.5901, which is the actual max/min distance for the best
> known 16-point configuration. Implementations work in squared distances to avoid square
> roots, and the benchmark inherited that convention. Reporting the un-squared value would
> silently look like a huge improvement.

## Where the record stands

| Construction | ratio² |
|---|---|
| 4×4 unit grid (the seed below) | 18.0 |
| ShinkaEvolve / EvoX / CORAL on Claude Opus 4.6 | 12.89 |
| AlphaEvolve | 12.889266112 |
| **best published** | **12.889230201** (FM Agent) |
| proven lower bound | none published |

No lower bound has ever been published for `C(2,16)`, so how much room remains below
12.889230201 is unknown. The systems that have attempted it agree to four significant
figures, which suggests a strong local optimum — but the gap between AlphaEvolve's
12.889266112 and FM Agent's 12.889230201 shows the last digits are still moving.

Beating 12.889230201 sets a record. Matching 12.889 matches every published system.

## Interface

`solve()` returns a list of 16 points, each a list of 2 finite floats. All points must be
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

- structured lattices: start from hexagonal, square and centred-hexagonal clusters and trim them to exactly 16 points
- minimax optimisation: identify which pairs realise the max and the min at the optimum and equalise them, treating it as a contact-graph problem
- continuous optimisation: smooth the max and min with a soft-max surrogate, run gradient descent from many random starts, then polish exactly
- boundary shaping: fix the convex hull to a near-circular polygon and optimise only the interior points
