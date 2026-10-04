"""Real-time adaptive filtering of noisy, non-stationary time series.

See PROBLEM.md. The mutable region is marked by # EVOLVE-BLOCK-START /
# EVOLVE-BLOCK-END. Everything else in this file is fixed scaffolding — do not modify it.

The evaluator calls `run_signal_processing(noisy_signal=..., window_size=20)` once per
test signal and reads `filtered_signal` from the dict you return. It supplies the
signals; you never see the clean series they were built from.

Run `python3 solution.py` to filter a demo signal and print a few sanity numbers.
"""
import numpy as np


def run_signal_processing(noisy_signal=None, window_size=20, **kwargs) -> dict:
    """Fixed scaffolding: filter `noisy_signal` and return it under 'filtered_signal'.

    The output must be a 1-D sequence of length len(noisy_signal) - window_size + 1:
    one sample per full window, so entry i is the filter's estimate at input index
    i + window_size - 1. All work happens in `process_signal`, which you evolve.
    """
    if noisy_signal is None:                       # `python3 solution.py` path
        noisy_signal, _ = _demo_signal()
    x = np.asarray(noisy_signal, dtype=float)
    filtered = np.asarray(process_signal(x, window_size), dtype=float)
    return {"filtered_signal": filtered}


# EVOLVE-BLOCK-START
def process_signal(x, window_size=20):
    """Baseline: an exponentially weighted moving average over each window.

    Weights rise towards the newest sample in the window, which tracks a moving signal
    a little better than a flat mean while still suppressing noise.

    Args:
        x: 1-D array of noisy samples.
        window_size: number of samples in the sliding window.

    Returns:
        A 1-D array of length len(x) - window_size + 1.
    """
    if len(x) < window_size:
        raise ValueError(f"signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)

    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.sum(window * weights)

    return y


# EVOLVE-BLOCK-END


def _demo_signal(length=600, noise_level=0.3, seed=7):
    """A noisy signal for local runs. NOT one of the evaluator's five test signals."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, length)
    clean = np.sin(2 * np.pi * 0.5 * t) + 0.4 * np.sin(2 * np.pi * 2.0 * t) + 0.1 * t
    return clean + rng.normal(0, noise_level, length), clean


if __name__ == "__main__":
    noisy, clean = _demo_signal()
    window = 20
    out = run_signal_processing(noisy_signal=noisy, window_size=window)["filtered_signal"]
    aligned = clean[window - 1 : window - 1 + len(out)]
    print(f"filtered {len(out)} samples (expected {len(noisy) - window + 1})")
    print(f"correlation with the clean signal: {np.corrcoef(out, aligned)[0, 1]:.4f}")
    print(f"mean |error| vs clean:             {np.mean(np.abs(out - aligned)):.4f}")
