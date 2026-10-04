# Erdős minimum overlap problem

Let `C` be the largest constant for which

    sup_{x ∈ [-2,2]}  ∫_{-1}^{1} f(t) · g(x+t) dt  ≥  C

holds for **every** pair of non-negative `f, g : [-1,1] → [0,1]` with `f + g = 1` on `[-1,1]`
and `∫ f = 1` (both extended by zero outside `[-1,1]`).

You choose `f`; then `g = 1 − f` is forced, and so is the mass constraint. Whatever `f` you
produce, the value

    overlap(f) = max_x ∫ f(t)·(1−f)(x+t) dt

is an **upper bound** on `C` — you are trying to make the two halves overlap as little as
possible, no matter how they are shifted against each other. Your score is that bound, and
lower is better.

This is the continuous form of Erdős' original question: split `{1, …, 2n}` into two equal
halves and shift one against the other; some shift always forces at least `C·n` matches.

## Where the record stands

This is the most tightly bracketed problem in the set — it has a **proven bound on both
sides**, so the remaining room is known exactly.

| Bound | Value | Source |
|---|---|---|
| lower bound on `C` (proven) | **0.379005** | White, 2022 |
| ← *only 0.49% of room lies between these* → | | |
| best construction | **0.380876** | Yuksekgonul et al., Jan 2026 |
| AlphaEvolve | 0.380924 | Georgiev–Gómez-Serrano–Tao–Wagner, May 2025 |
| earlier record | 0.380927 | Haugland, 2016 |
| flat `f ≡ ½` (the seed below) | 0.5 | trivial |

Anything **below 0.379005 is impossible** — if you compute such a value, the construction is
invalid or the arithmetic is wrong, not a breakthrough. The evaluator rejects it.

Two useful facts: CORAL reached 0.38089 on Claude Opus 4.6, and every system that has
attempted this problem lands between 0.38088 and 0.38188. Beating 0.380876 sets a record;
beating 0.38089 beats the best published Claude result.

## The search space

You return a **step function**: `n` heights on `[-1,1]`, each held over an interval of width
`2/n`, every height in `[0,1]`. The mass constraint `∫f = 1` becomes `sum(f) = n/2` exactly,
since each step has width `2/n`. The evaluator enforces this to a tolerance of 1e-6 and tells
you the exact sum it wants if you miss.

Scoring is **exact, not numerical**: the cross-correlation of two step functions on a common
grid is piecewise linear, so its supremum is attained at a grid point and is computed in
closed form from your heights. There is no integrator to exploit.

## Interface

`solve()` returns a list of `n` floats in `[0,1]` (`2 ≤ n ≤ 4000`), summing to `n/2`.
`numpy` is available.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that
figure and use it — leaving time unspent is wasted search, and overrunning it means
the candidate scores 0. The evaluation as a whole is capped at 700 s, so leave a
margin for interpreter start-up and imports. Your briefing repeats the number in
force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- bang-bang structure: search over where `f` switches between 0 and 1, treating the problem as a choice of switching points rather than of heights
- continuous optimisation: keep the heights soft in [0,1] and push them with projected gradient or quasi-Newton steps on the exact objective
- minimax reformulation: attack the max over shifts directly — find which shifts are active at the optimum and equalise them
- resolution laddering: solve at coarse n, interpolate to finer n, and re-optimise from that warm start
