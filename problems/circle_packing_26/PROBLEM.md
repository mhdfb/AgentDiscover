# Circle packing — 26 circles, maximise the sum of radii

Place 26 disjoint open disks inside the unit square `[0,1]²` so as to maximise

    Σ rᵢ

The disks may have **different radii**, must lie entirely inside the square, and must not
overlap each other (touching is allowed). This is the variant AlphaEvolve made famous — it
is not the classical equal-circles packing problem.

## Where the record stands

| Construction | Σ radii |
|---|---|
| 5×5 grid plus one in a gap (the seed below) | 2.5414 |
| Friedman, 2012 — human best before 2025 | 2.634 |
| AlphaEvolve | 2.63586276 |
| CORAL, Claude Opus 4.6 | 2.6320 |
| **OpenEvolve on Claude Opus 4.6** | **2.6359** |
| ShinkaEvolve (mixed ensemble incl. Sonnet 4) | 2.635983099 |
| **best published** | **2.6359857** (ThetaEvolve, 8B open model) |
| proven upper bound | **2.87704** |

The upper bound is rigorous but loose: the disks are disjoint and inside a unit square, so
`Σπrᵢ² ≤ 1`, and Cauchy–Schwarz gives `Σrᵢ ≤ √(26 Σrᵢ²) ≤ √(26/π) = 2.87704`. No packing can
exceed it; the true optimum is unknown but is believed to be very close to 2.63599.

**Be warned: this problem is close to saturated.** Six independent systems have landed
between 2.6359 and 2.63599 — they disagree only in the fifth decimal. It is included as a
calibration task, not as a frontier one: if a search cannot reach ≈2.635 here, something is
wrong with it, and if it beats 2.6359857 that is a genuine record.

One instructive detail: in CORAL's own table, plain **OpenEvolve on Opus 4.6 scored 2.6359,
beating CORAL's own 2.6320**. On a strong enough model the simplest baseline wins this task,
which is exactly what saturation looks like.

## Interface

`solve()` returns a list of 26 triples `[x, y, r]` — centre and radius of each disk.
Validity is checked to a tolerance of 1e-9, tight enough that the published nine-decimal
records are meaningful:

- `r ≥ 0`, and all values finite;
- each disk inside the square: `r ≤ x ≤ 1−r` and `r ≤ y ≤ 1−r`;
- pairwise disjoint: `dist(i,j) ≥ rᵢ + rⱼ`.

A violation is reported with the offending pair and the size of the overlap, so a near-miss
costs one call rather than the iteration.

NumPy and SciPy are installed in your sandbox and in the evaluator. Maximising the sum of
radii subject to the two constraints above is a constrained nonlinear program, so
`scipy.optimize` (`minimize` with SLSQP, `NonlinearConstraint`) applies directly to all
78 variables at once.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that
figure and use it — leaving time unspent is wasted search, and overrunning it means
the candidate scores 0. The evaluation as a whole is capped at 700 s, so leave a
margin for interpreter start-up and imports. Your briefing repeats the number in
force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- constrained optimisation: treat it as a smooth programme in the 78 coordinates and push it with SLSQP or an augmented-Lagrangian method from many random starts
- structured seeds: start from hexagonal, square, or ring-based layouts with a few distinct radius classes, then relax
- pattern search: fix a contact graph — which disks touch which — solve the radii exactly for that graph, then search over graphs
- physical relaxation: grow all radii simultaneously and resolve overlaps by pushing disks apart, in the spirit of a billiard or inflation dynamic
