"""Real-time adaptive signal processing — the SkyDiscover metric, unchanged.

Ported from SkyDiscover's `benchmarks/math/signal_processing`, which is the bundle CORAL
runs. The task, the test signals and the metric are theirs:

  * `generate_test_signals` produces the same five signals from the same seeds
    (np.random.seed(42 + i), lengths 500..900, noise 0.2..0.6);
  * the penalty terms S / L_recent / L_avg / R, their normalisation, and
    J = 1/(1 + 0.3*S + 0.2*L_recent + 0.2*L_avg + 0.3*R) are copied verbatim;
  * `combined_score` = 0.4*composite + 0.2*smoothness + 0.2*accuracy
    + 0.1*noise_reduction + 0.1*success_rate, zeroed when accuracy < 0.1;
  * each signal keeps SkyDiscover's own 10 s per-call timeout, and the whole evaluation
    keeps their 360 s cap (`evaluator.timeout` in their config.yaml).

Verified against the upstream file (skydiscover-ai/skydiscover,
benchmarks/math/signal_processing/evaluator/evaluator.py): running their `evaluate()` and
this one over the same candidates returns identical `combined_score` and identical values
for every component metric. CORAL's Appendix D.3 lists the four SkyDiscover evaluators it
corrected — PRISM, transaction scheduling, EPLB and LLM-SQL — and this task is not among
them, so the code below is what produced their published figure too.

The scoring math is unchanged, but TWO guards close holes that CORAL's and SkyDiscover's
evaluators share (both verified 2026-09-12: a candidate returning only its first few
filtered samples scores ~0.92 on all three, against an honest ceiling of ~0.727):

  1. Length check. Every returned `filtered_signal` must have length
     `len(noisy_signal) - window_size + 1`, the value the task documents. Their
     evaluators never check it, so a short output has almost no slope changes and a
     near-perfect correlation over its handful of points; here a wrong length drops that
     signal (it is not scored), exactly as an empty or non-finite output already did. An
     honest full-length filter is unaffected, so the number stays comparable to theirs.
  2. Process isolation. `_run_filter` runs the candidate in a FRESH subprocess that hands
     back only the filtered arrays; scoring happens here, in a process the candidate's
     code never entered. Their evaluators import and run the candidate in the SAME process
     that scores, so a candidate could patch `pearsonr`/the metric functions or print its
     own score line — reaching any number. Ours cannot be reached that way. Arrays are
     passed as plain JSON (not pickle), so no unpickling code path is introduced either.

What else differs is only the output protocol: SkyDiscover bridges this module through a
`wrapper.py`, whereas here `main()` prints the one JSON line the platform's contract asks
for. The candidate supplies `run_signal_processing(noisy_signal=..., window_size=20)`; it
never sees the clean signals its output is measured against.
"""
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

_HERE = Path(__file__).resolve().parent   # so the candidate's subprocess can import this bundle

WINDOW_SIZE = 20
NUM_SIGNALS = 5
PER_SIGNAL_TIMEOUT = 10.0      # SkyDiscover's own per-call limit

# The objective is `combined_score`, maximised. There is no useful ceiling to quote: the
# metric rewards smoothness, so a filter can score ABOVE perfect denoising by being
# smoother than the clean signal itself (a real candidate reached 0.734, vs 0.727 for
# returning the exact clean signal). TARGET is simply a bar known to be reachable. The
# previous 0.8229 was only ever reached by the short-output exploit this evaluator now
# rejects (see _run_filter / the length check).
TARGET = 0.718


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely"""
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def calculate_slope_changes(signal_data):
    """Slope change penalty S(theta) — counts directional reversals."""
    if len(signal_data) < 3:
        return 0

    diffs = np.diff(signal_data)

    sign_changes = 0
    for i in range(1, len(diffs)):
        if np.sign(diffs[i]) != np.sign(diffs[i - 1]) and diffs[i - 1] != 0:
            sign_changes += 1

    return sign_changes


def calculate_lag_error(filtered_signal, original_signal, window_size):
    """Instantaneous lag error L_recent(theta) = |y[n] - x[n]|."""
    if len(filtered_signal) == 0:
        return 1.0  # Maximum penalty

    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0

    recent_filtered = filtered_signal[-1]
    recent_original = original_signal[delay + len(filtered_signal) - 1]

    return abs(recent_filtered - recent_original)


def calculate_average_tracking_error(filtered_signal, original_signal, window_size):
    """Average tracking error L_avg(theta) over the processed samples."""
    if len(filtered_signal) == 0:
        return 1.0  # Maximum penalty

    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0

    aligned_original = original_signal[delay : delay + len(filtered_signal)]

    min_length = min(len(filtered_signal), len(aligned_original))
    if min_length == 0:
        return 1.0

    filtered_aligned = filtered_signal[:min_length]
    original_aligned = aligned_original[:min_length]

    return np.mean(np.abs(filtered_aligned - original_aligned))


def calculate_false_reversal_penalty(filtered_signal, clean_signal, window_size):
    """False reversal penalty R(theta) — trend changes the clean signal does not have."""
    if len(filtered_signal) < 3 or len(clean_signal) < 3:
        return 0

    delay = window_size - 1
    if len(clean_signal) <= delay:
        return 1.0

    aligned_clean = clean_signal[delay : delay + len(filtered_signal)]
    min_length = min(len(filtered_signal), len(aligned_clean))

    if min_length < 3:
        return 0

    filtered_aligned = filtered_signal[:min_length]
    clean_aligned = aligned_clean[:min_length]

    filtered_diffs = np.diff(filtered_aligned)
    clean_diffs = np.diff(clean_aligned)

    false_reversals = 0
    for i in range(1, len(filtered_diffs)):
        filtered_change = (
            np.sign(filtered_diffs[i]) != np.sign(filtered_diffs[i - 1])
            and filtered_diffs[i - 1] != 0
        )

        clean_change = (
            np.sign(clean_diffs[i]) != np.sign(clean_diffs[i - 1]) and clean_diffs[i - 1] != 0
        )

        if filtered_change and not clean_change:
            false_reversals += 1

    return false_reversals


def calculate_composite_score(S, L_recent, L_avg, R, alpha=[0.3, 0.2, 0.2, 0.3]):
    """J(theta) = a1*S + a2*L_recent + a3*L_avg + a4*R, as a maximisation score."""
    S_norm = min(S / 50.0, 2.0)

    L_recent_norm = min(L_recent, 2.0)
    L_avg_norm = min(L_avg, 2.0)

    R_norm = min(R / 25.0, 2.0)

    penalty = (
        alpha[0] * S_norm + alpha[1] * L_recent_norm + alpha[2] * L_avg_norm + alpha[3] * R_norm
    )

    score = 1.0 / (1.0 + penalty)

    return score


def generate_test_signals(num_signals=NUM_SIGNALS):
    """The five fixed test signals: sinusoid+trend, multi-frequency, non-stationary,
    step changes, random walk. Seeded, so every candidate meets identical inputs."""
    test_signals = []

    for i in range(num_signals):
        np.random.seed(42 + i)  # Different seed for each signal
        length = 500 + i * 100  # Varying lengths
        noise_level = 0.2 + i * 0.1  # Varying noise levels

        t = np.linspace(0, 10, length)

        if i == 0:
            # Smooth sinusoidal with trend
            clean = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * t
        elif i == 1:
            # Multiple frequency components
            clean = (
                np.sin(2 * np.pi * 0.5 * t)
                + 0.5 * np.sin(2 * np.pi * 2 * t)
                + 0.2 * np.sin(2 * np.pi * 5 * t)
            )
        elif i == 2:
            # Non-stationary with changing frequency
            clean = np.sin(2 * np.pi * (0.5 + 0.2 * t) * t)
        elif i == 3:
            # Step changes
            clean = np.concatenate(
                [
                    np.ones(length // 3),
                    2 * np.ones(length // 3),
                    0.5 * np.ones(length - 2 * (length // 3)),
                ]
            )
        else:
            # Random walk with trend
            clean = np.cumsum(np.random.randn(length) * 0.1) + 0.05 * t

        noise = np.random.normal(0, noise_level, length)
        noisy = clean + noise

        test_signals.append((noisy, clean))

    return test_signals


def _run_filter(solution_dir, timeout):
    """Run the candidate on every test signal in a FRESH subprocess and return, per
    signal, only the filtered array (or a failure string). The candidate never runs in
    THIS scoring process, so it cannot patch the metric functions or print its own score;
    the subprocess regenerates the signals with the bundle's own generator, gives the
    candidate the noisy signal only, and hands back just the numbers, which are scored
    here. Returns (status dict, wall seconds)."""
    code = (
        "import sys, json\n"
        "import time as _t\n"
        "sys.path.insert(0, '.')\n"
        f"sys.path.insert(0, {str(_HERE)!r})\n"
        "import numpy as np\n"
        "from evaluator import (generate_test_signals, run_with_timeout,\n"
        "                       WINDOW_SIZE, NUM_SIGNALS, PER_SIGNAL_TIMEOUT)\n"
        "try:\n"
        "    import solution as program\n"
        "except Exception as e:\n"
        "    print(json.dumps({'import_error': str(e)})); sys.exit(0)\n"
        "if not hasattr(program, 'run_signal_processing'):\n"
        "    print(json.dumps({'missing': True})); sys.exit(0)\n"
        "out = []\n"
        "for i, (noisy, clean) in enumerate(generate_test_signals(NUM_SIGNALS)):\n"
        "    rec = {}\n"
        "    try:\n"
        "        st = _t.perf_counter()\n"
        "        result = run_with_timeout(program.run_signal_processing,\n"
        "                 kwargs={'noisy_signal': noisy, 'window_size': WINDOW_SIZE},\n"
        "                 timeout_seconds=PER_SIGNAL_TIMEOUT)\n"
        "        rec['time'] = _t.perf_counter() - st\n"
        "        if not isinstance(result, dict):\n"
        "            rec['fail'] = f\"returned {type(result).__name__}, not a dict\"\n"
        "        elif 'filtered_signal' not in result:\n"
        "            rec['fail'] = \"no 'filtered_signal' key in the returned dict\"\n"
        "        else:\n"
        "            arr = np.asarray(result['filtered_signal'], dtype=float)\n"
        "            rec['ndim'] = int(arr.ndim)\n"
        "            rec['filtered'] = arr.ravel().tolist()\n"
        "    except TimeoutError:\n"
        "        rec['fail'] = f'ran past its {PER_SIGNAL_TIMEOUT:.0f}s budget'\n"
        "    except Exception as e:\n"
        "        rec['fail'] = f'{type(e).__name__}: {e}'\n"
        "    out.append(rec)\n"
        "print(json.dumps({'signals': out}))\n"
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=timeout, cwd=solution_dir)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"run_signal_processing did not finish the five signals within {timeout:.0f}s "
            "and was killed.") from None
    wall = time.perf_counter() - t0
    if proc.stderr:                       # the candidate's own prints stay with the operator
        sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(f"the filter subprocess exited {proc.returncode}: {detail or 'no output'}")
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line), wall
    raise RuntimeError("the filter subprocess produced no status line")


def evaluate(solution_path):
    """Score one candidate on all five signals. Returns the platform's metrics dict."""
    t0 = time.perf_counter()
    try:
        status, _child_wall = _run_filter(Path(solution_path).parent,
                                          NUM_SIGNALS * PER_SIGNAL_TIMEOUT + 30.0)
        if status.get("import_error"):
            return {"score": 0.0, "stage": "error", "error": status["import_error"][:300],
                    "time": time.perf_counter() - t0}
        if status.get("missing"):
            return {"score": 0.0, "stage": "invalid",
                    "error": "solution.py defines no run_signal_processing function",
                    "time": time.perf_counter() - t0}

        test_signals = generate_test_signals(NUM_SIGNALS)
        records = status.get("signals", [])

        all_scores = []
        all_metrics = []
        successful_runs = 0
        failures = []

        for i, (noisy_signal, clean_signal) in enumerate(test_signals):
            try:
                rec = records[i] if i < len(records) else {"fail": "no result returned"}
                if "fail" in rec:
                    failures.append(f"signal {i}: {rec['fail']}")
                    continue

                execution_time = float(rec.get("time", 0.0))
                filtered_signal = np.asarray(rec.get("filtered", []), dtype=float)

                if filtered_signal.size == 0:
                    failures.append(f"signal {i}: filtered_signal is empty")
                    continue
                if int(rec.get("ndim", filtered_signal.ndim)) != 1:
                    failures.append(f"signal {i}: filtered_signal is {rec.get('ndim')}-D, "
                                    "expected a 1-D sequence")
                    continue
                if not np.all(np.isfinite(filtered_signal)):
                    failures.append(f"signal {i}: filtered_signal contains NaN or inf")
                    continue
                expected_len = len(noisy_signal) - WINDOW_SIZE + 1
                if filtered_signal.size != expected_len:
                    failures.append(f"signal {i}: filtered_signal has length "
                                    f"{filtered_signal.size}, expected {expected_len} "
                                    "(len(signal) - window_size + 1)")
                    continue

                window_size = WINDOW_SIZE

                S = calculate_slope_changes(filtered_signal)
                L_recent = calculate_lag_error(filtered_signal, noisy_signal, window_size)
                L_avg = calculate_average_tracking_error(filtered_signal, noisy_signal, window_size)
                R = calculate_false_reversal_penalty(filtered_signal, clean_signal, window_size)

                composite_score = calculate_composite_score(S, L_recent, L_avg, R)

                correlation = 0.0
                noise_reduction = 0.0

                try:
                    delay = window_size - 1
                    aligned_clean = clean_signal[delay : delay + len(filtered_signal)]
                    min_length = min(len(filtered_signal), len(aligned_clean))

                    if min_length > 1:
                        corr_result = pearsonr(
                            filtered_signal[:min_length], aligned_clean[:min_length]
                        )
                        correlation = corr_result[0] if not np.isnan(corr_result[0]) else 0.0

                    aligned_noisy = noisy_signal[delay : delay + len(filtered_signal)]
                    aligned_noisy = aligned_noisy[:min_length]
                    aligned_clean = aligned_clean[:min_length]

                    if min_length > 0:
                        noise_before = np.var(aligned_noisy - aligned_clean)
                        noise_after = np.var(filtered_signal[:min_length] - aligned_clean)
                        noise_reduction = (
                            (noise_before - noise_after) / noise_before if noise_before > 0 else 0
                        )
                        noise_reduction = max(0, noise_reduction)

                except Exception as e:  # noqa: BLE001 — one signal's extras, never the run
                    failures.append(f"signal {i}: extra metrics failed ({e})")

                all_metrics.append({
                    "slope_changes": safe_float(S),
                    "lag_error": safe_float(L_recent),
                    "avg_error": safe_float(L_avg),
                    "false_reversals": safe_float(R),
                    "composite_score": safe_float(composite_score),
                    "correlation": safe_float(correlation),
                    "noise_reduction": safe_float(noise_reduction),
                    "execution_time": safe_float(execution_time),
                    "signal_length": len(filtered_signal),
                })
                all_scores.append(composite_score)
                successful_runs += 1

            except TimeoutError:
                failures.append(f"signal {i}: ran past its {PER_SIGNAL_TIMEOUT:.0f}s budget")
                continue
            except Exception as e:  # noqa: BLE001 — a bad candidate scores 0, never crashes us
                failures.append(f"signal {i}: {type(e).__name__}: {e}")
                continue

        if successful_runs == 0:
            return {"score": 0.0, "stage": "invalid",
                    "error": "no test signal was processed successfully. "
                             + "; ".join(failures)[:400],
                    "time": time.perf_counter() - t0}

        avg_composite_score = np.mean(all_scores)
        avg_slope_changes = np.mean([m["slope_changes"] for m in all_metrics])
        avg_lag_error = np.mean([m["lag_error"] for m in all_metrics])
        avg_avg_error = np.mean([m["avg_error"] for m in all_metrics])
        avg_false_reversals = np.mean([m["false_reversals"] for m in all_metrics])
        avg_correlation = np.mean([m["correlation"] for m in all_metrics])
        avg_noise_reduction = np.mean([m["noise_reduction"] for m in all_metrics])
        avg_execution_time = np.mean([m["execution_time"] for m in all_metrics])
        success_rate = successful_runs / len(test_signals)

        smoothness_score = 1.0 / (1.0 + avg_slope_changes / 20.0)
        responsiveness_score = 1.0 / (1.0 + avg_lag_error)
        accuracy_score = max(0, avg_correlation)

        combined = (
            0.4 * avg_composite_score
            + 0.2 * smoothness_score
            + 0.2 * accuracy_score
            + 0.1 * avg_noise_reduction
            + 0.1 * success_rate
        )

        # SkyDiscover's gate: a filter uncorrelated with the clean signal scores nothing,
        # however smooth it is.
        gated = bool(accuracy_score < 0.1)   # numpy bool_ is not JSON serialisable
        if gated:
            combined = 0.0

        return {
            "score": safe_float(combined),
            "stage": "full",
            "objective": safe_float(combined),
            "objective_name": "combined_score",
            "objective_direction": "max",
            "objective_target": TARGET,
            "combined_score": safe_float(combined),
            "composite_score": safe_float(avg_composite_score),
            "smoothness_score": safe_float(smoothness_score),
            "accuracy_score": safe_float(accuracy_score),
            "responsiveness_score": safe_float(responsiveness_score),
            "correlation": safe_float(avg_correlation),
            "noise_reduction": safe_float(avg_noise_reduction),
            "slope_changes": safe_float(avg_slope_changes),
            "lag_error": safe_float(avg_lag_error),
            "avg_error": safe_float(avg_avg_error),
            "false_reversals": safe_float(avg_false_reversals),
            "success_rate": safe_float(success_rate),
            "signals_scored": successful_runs,
            "correlation_gate_applied": gated,
            "target": TARGET,
            "solve_time": safe_float(avg_execution_time),
            "time": time.perf_counter() - t0,
            **({"error": "; ".join(failures)[:300]} if failures else {}),
        }

    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error",
                "error": f"{type(e).__name__}: {e}"[:300],
                "traceback": traceback.format_exc()[-400:],
                "time": time.perf_counter() - t0}


def main():
    path = Path(sys.argv[1]).resolve()
    # The candidate is imported, so its own directory must be importable too.
    sys.path.insert(0, str(path.parent))
    print(json.dumps(evaluate(str(path))))


if __name__ == "__main__":
    main()
