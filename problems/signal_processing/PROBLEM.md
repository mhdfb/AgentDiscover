# Real-time adaptive signal processing

Improve a signal processing algorithm that filters volatile, non-stationary time series
data using a sliding window approach. The algorithm must minimise noise while preserving
signal dynamics with minimal computational latency and phase delay.

Focus on the multi-objective optimisation of: (1) slope change minimisation — reducing
spurious directional reversals; (2) lag error minimisation — maintaining responsiveness;
(3) tracking accuracy — preserving genuine signal trends; and (4) false reversal penalty
— avoiding noise-induced trend changes.

Consider advanced techniques like adaptive filtering (Kalman filters, particle filters),
multi-scale processing (wavelets, EMD), predictive enhancement (polynomial fitting,
neural networks), and trend detection methods.

## Targets

You are scored on several synthetic test signals of varying length and noise level,
which you never see in advance: the filter is given the noisy samples and judged against
the clean signal underneath them. Your objective is the **`combined_score`** the
evaluator returns. **Higher is better.**

| | combined_score ↑ |
|---|---|
| the seed below (exponentially weighted moving average) | ~0.499 |
| **target — beat this, clearly** | **0.718** |

The target of 0.718 is a bar to get **past**, and it is known to be reachable. Alongside
your `combined_score` the evaluator returns the quantities that went into it, so you can
see which one is costing you.

## Interface

`solution.py` defines

    run_signal_processing(noisy_signal=None, window_size=20, ...) -> dict

The evaluator calls it once per test signal and reads `"filtered_signal"` from the dict
you return: a one-dimensional filtered sequence whose expected length is
`len(noisy_signal) - window_size + 1`. The function must handle arbitrary input signals,
not only the demo signal in the seed program. A returned series that is empty, not
one-dimensional, or contains NaN or infinity is not scored.

`run_signal_processing` is fixed scaffolding; you evolve `process_signal` and anything
else you add inside the EVOLVE-BLOCK. NumPy, SciPy and PyWavelets are installed in your
sandbox and in the evaluator, so `python3 solution.py` runs as it stands and anything you
can test locally will import when it is scored.

**Compute budget: each test signal gets 10 s**, and a signal your filter fails or
overruns on is dropped and counts against you. The evaluation as a whole is capped at
360 s, so the per-signal limit is the one that binds: this is a *real-time* filter, and
an expensive method has to earn its cost. Your briefing repeats the numbers in force
each session.
