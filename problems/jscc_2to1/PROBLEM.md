# Space-filling-curve JSCC, 2:1 — send a Gaussian pair over one channel use

Design a curve that *is* the whole transmitter and receiver of an analog joint
source-channel coding system with bandwidth compression. A source sample is a point
`s ~ N(0, I₂)`; you get **one** real AWGN channel use per pair, so the bandwidth ratio is
ρ = 1/2 and two numbers have to survive as one. A one-dimensional curve threaded through
the plane does the compression: the transmitter projects the source point onto the curve
and sends its scalar coordinate; the receiver reads that coordinate back as a point.

The system is fixed end to end and you control only the curve:

1. **Encoder (fixed).** Project `s` onto the nearest point of your table, searched on a
   uniform grid of `min(4P, 40000)` transmit values spanning `±max|z|`, and transmit that
   point's `z`.
2. **Channel (fixed).** `y = z + n`, noise variance `E[z²]/CSNR` — the power is
   normalised to what your curve actually transmits, so rescaling every `z` is a no-op.
   Only the *relative* spacing of `z` matters.
3. **Decoder (fixed).** Map `y` back through the curve by monotone linear interpolation,
   end-clamped to the curve's `z`-range. There is no other receiver processing: no MMSE
   estimator, no shrinkage, no denoiser. Any receiver behaviour you want has to be baked
   into the geometry and the `z`-spacing.

Per SNR, `SDR = 10·log10(1 / per-component MSE)` in dB. The sweep is 0–30 dB in 5 dB
steps.

## Targets

Your objective is the **`combined_score`** the evaluator returns:

    combined_score = 1 / (1 + 0.5·mean_gap + 0.5·worst_gap)

where each gap is `OPTA − SDR` in dB at one SNR (clamped at 0), and the mean and max run
over the seven swept SNRs. **Higher is better.**

| | combined_score ↑ |
|---|---|
| the seed below (untuned Archimedes spiral, same shape at every SNR) | 0.1969 |
| a per-SNR tuned spiral — what parameter search alone buys | ≈0.26 |
| **target — beat this, clearly** | **0.362** |
| ceiling | 1.0 |

The ceiling of 1 is analytic, not a guess. `OPTA = ρ·10·log10(1 + CSNR)` is the
information-theoretic optimum for this bandwidth ratio, a gap to it can never be
negative, and 1 is what you would score by sitting exactly on OPTA at every SNR. **An SDR
above OPTA is impossible**: the evaluator rejects such a candidate outright rather than
crediting it, because it can only mean the table or the measurement is broken. OPTA over
the sweep is 1.51 / 3.10 / 5.21 / 7.57 / 10.02 / 12.51 / 15.00 dB at 0/5/10/15/20/25/30 dB.

Unlike the ceiling, the target is not a property of the problem: 0.362 is simply the best
curve anyone has built for it so far — the strongest design we have, at mean gap 1.66 dB
and worst gap 1.87 dB. It is a bar to get **meaningfully past**, not to land on. Matching
~0.362 is not success; it is where a previous search stopped, and there is nothing
special about that point — 0.362 and the ceiling of 1.0 are more than half the problem
apart, and the whole low-SNR end of the sweep is still open. Alongside the score the
evaluator returns `mean_gap_db`, `worst_gap_db`, and the per-SNR SDR, gap and winning
variant, so you can see which end of the sweep is costing you.

The middle row of the table is the one worth reading twice. Tuning the parameters of a
spiral, per SNR, gets you to ≈0.26 and no further — the remaining 0.1 to the target came
from changing the *shape*, not the numbers in it.

Note the shape of the objective: `worst_gap` carries as much weight as the mean, so one
bad SNR point costs as much as the whole rest of the sweep. A design that is excellent at
high SNR and falls off a cliff at 0 dB scores worse than a mediocre design that is flat.

## The central tradeoff

Fold spacing controls three competing error sources:

- **approximation error** — the sample lies off the curve; falls with denser folds and
  longer reach;
- **noise amplification** — more arc length inside a fixed transmit power budget means a
  given channel perturbation slides the decoded point further along the curve;
- **anomalous errors** — noise jumps the decoder onto a *neighbouring* fold, a large
  non-local error, and this is what produces the low-SNR cliff.

The three trade against each other differently at every SNR, which is why per-SNR
adaptation matters: below the threshold SNR a barely-winding, almost linear curve beats a
space-filling one, and above it the reverse. Any geometry is legal — any curve family,
non-uniform winding density, piecewise or asymmetric shapes. The `z`-labelling is a
second, independent lever (companding / power allocation): uniform-in-arc-length is the
neutral default, and deviating from it shapes where channel noise hurts least.

## What the evaluator grants you for free

At every SNR the evaluator scores a small **canonical variant set** of your table and
keeps the best one, so a good geometry is never sunk by a labelling convention you never
tuned:

- **z-labelling:** as submitted, and re-derived as signed arc length;
- **arm pairing:** as submitted, and with the negative-`z` arm rebuilt as `R · (positive
  arm)` for each sign reflection `R` in `(−1,−1)`, `(+1,−1)`, `(−1,+1)`.

Selection among the variants runs at 8 000 samples; the winner is then re-measured at
200 000 samples, and *that* is the number scored. The winning variant is reported per SNR.

## Interface

`solution.py` defines

    get_curve(snr_db: float) -> {"z": (P,) float, "points": (P, 2) float}

- `points` is a `(P, 2)` array of curve points in source space; `z` is the `(P,)`
  transmit value assigned to each point. `z` must be strictly increasing and finite,
  `64 ≤ P ≤ 20000`, and `|points| ≤ 50`. Anything else scores 0.
- `get_curve` receives the true operating SNR — **per-SNR adaptation is allowed and
  expected.** It is called once per swept SNR.
- It must be **deterministic**: it is called a second time at 15 dB and the two tables
  have to match bit for bit, so seed any randomness you use from `snr_db`.
- `P` above ~10000 buys decoder resolution but no further encoder resolution, since the
  encoder's search grid saturates at 40000 points. Large tables are not penalised in
  score — the cap is affordable, it just costs wall clock.

`import jscc_channel` to measure any curve yourself: it is mounted read-only at
`/support`, is already on your `PYTHONPATH`, and is a **byte-identical copy of the module
the evaluator scores with** — so a curve you have measured locally scores exactly the
same when submitted. `ch.validate`, `ch.sdr_db`, `ch.opta_db`, `ch.variants`,
`ch.best_sdr_db` and `ch.score_curve` are all there, along with the sweep, the sample
counts and the fixed seeds. `python3 solution.py` prints a quick sweep of the current
curve. `sdr_db` is a Monte-Carlo estimate: ±0.08 dB at the 8 000-sample probe, ±0.02 dB
at the 200 000-sample scoring pass, and the seeds are fixed, so the same table always
gives the same number. The same module is importable *during evaluation* too, so
`get_curve` may measure and optimise its own candidates inside the call — the scoring
itself happens afterwards, in a separate process, from the evaluator's own copy, so a
curve can never report a number for itself.

**Compute budget: `get_curve` gets 300 s for the whole sweep** — all seven calls plus the
repeat, not 300 s per call. That is enough to run a real numerical search inside the call
if you want one, and leaving it unspent is wasted search; overrunning it scores 0. The
evaluation as a whole is capped at 660 s, because scoring your curve costs 74–193 s
depending on table size. Your briefing repeats the numbers in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- curve families: search parametric shapes — spirals with varying pitch laws, sinusoidal and zig-zag windings, piecewise families — and tune their parameters per SNR
- companding: hold the geometry and search the z-labelling, allocating transmit power along the curve where channel noise costs least
- direct optimization: treat the point coordinates themselves as free variables and improve them numerically against the measured SDR
- threshold behaviour: attack the low-SNR end specifically, where anomalous fold-jumping dominates and the worst-gap term is decided
