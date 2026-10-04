# Beat the average game

Let `X_1, X_2, X_3, X_4` be independent draws from the same probability distribution `μ`
on the non-negative integers. Let

    C  =  sup_μ  P[ X_1 + X_2 + X_3  <  2·X_4 ]

In words: four players draw independently from a distribution of your choosing, and you
want to maximise the chance that the fourth player's draw, doubled, beats the sum of the
other three. Nothing forces this to be small — the distribution can be as lopsided as you
like — but it is far from obvious how large it can get.

**Your task**: choose the distribution. Every `μ` you produce certifies `C ≥ P[...]`, and
your score is that lower bound.

## Where the record stands

| Construction | Lower bound on C |
|---|---|
| uniform / naive distributions | ≈ 0.30 |
| **AlphaEvolve, no expert hints** | **0.389** |
| best known (Bellec & Fritz) | **0.400695** |

DeepMind's write-up: *"With the most straightforward setup and no expert hints, AlphaEvolve
obtained the lower bound C ≥ 0.389 within only a few hours. While this does not outperform
the final results of Bellec and Fritz (C ≥ 0.400695), it was competitive with the
contemporaneous state of the art."*

They also record a structural observation worth having: the good measures are **sparse** —
most weights are zero, and the non-zero ones fall into a particular pattern. Their published
construction puts almost all of its mass on a handful of small indices out of 20 000.

## Interface

`solve()` returns a list of non-negative weights `w_0, ..., w_{n-1}` (`2 ≤ n ≤ 20000`),
interpreted as the distribution `P[X = i] ∝ w_i`. The evaluator normalises for you, so the
weights need not sum to 1; at least one must be positive. `numpy` is available.

The probability is computed exactly by convolution over the integer support — no sampling,
no Monte Carlo noise — so the score is a deterministic, certified lower bound.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that
figure and use it — leaving time unspent is wasted search, and overrunning it means
the candidate scores 0. The evaluation as a whole is capped at 700 s, so leave a
margin for interpreter start-up and imports. Your briefing repeats the number in
force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- sparse support: search over which few indices carry mass, then optimise the weights on that support
- continuous optimisation: parametrise the weights and push them with gradient or quasi-Newton steps on the exact objective
- support geometry: study how the optimal spacing of the atoms scales, and extrapolate the pattern to finer supports
- two-scale mixtures: combine a heavy atom near zero with a light tail, and tune the mass split between the scales
