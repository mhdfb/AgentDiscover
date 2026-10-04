# LABS — Low Autocorrelation Binary Sequences

Construct a sequence b ∈ {-1, +1}^60 that maximizes the merit factor

    F(b) = N^2 / (2 * sum_{k=1}^{N-1} c_k(b)^2),
        c_k(b) = sum_{i=0}^{N-1-k} b_i * b_{i+k}

c_k(b) is the aperiodic autocorrelation at lag k. Higher F is better. `evaluate.py` reports
`score = min(F / 10, 1)`.

Reference points at N=60:
- random ±1 sequences: F ≈ 1 (score ≈ 0.1)
- structured constructions (Legendre/Jacobi, rotated): F up to ~6 (score ≈ 0.6)
- best known via search (tabu, branch-and-bound): F ≈ 8-9 (score ≈ 0.85)

There is no closed form for the optimum; this is a genuinely open combinatorial-search
problem in signal design.

> **Note on saturation.** At N=60 the optimum is *known exactly* — LABS has been solved by
> branch-and-bound for all N ≤ 66 — so this instance measures search quality against a
> known ceiling and cannot yield a record. To use LABS as a frontier benchmark instead,
> raise N past 66 (e.g. N=101 or N=201), where only heuristic best-known merit factors
> exist. See `problems/erdos_discrepancy/` for a benchmark with published AI baselines
> that are far from the proven optimum.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- algebraic construction: Legendre / Jacobi / Rudin-Shapiro / m-sequences and rotations
- local search: single-flip and segment-flip hill climbing, simulated annealing
- tabu / metaheuristic: tabu search with restart, large-neighborhood search, ILS with kick moves
- hybrid construction + repair: start from a structured construction, then run targeted bit-flips to suppress the worst autocorrelation lags