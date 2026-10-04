"""Evaluator for the beat-the-average game (lower bound on C).

Called by evaluate.sh with the solution's path as argv[1]; prints one JSON line.

Scoring reproduces DeepMind's published verification code for this problem. With
p = the normalised weights over support {0, ..., n-1}:

    pmf_Y   = p * p * p              (distribution of X1 + X2 + X3, by convolution)
    cdf_Y   = cumulative sum of pmf_Y
    P[...]  = sum_m p[m] * P[Y < 2m] = sum_m p[m] * cdf_Y[2m - 1]

since Y is integer-valued, P[Y < 2m] = P[Y <= 2m-1]; the m = 0 term is zero. The whole
computation is exact convolution over the integer support — no sampling — so the result is
a deterministic certified lower bound on C.

Large supports use FFT convolution (O(n log n)); small ones use the direct method, which is
exact. Both agree to well past the precision that separates the published records.

Reference values (see PROBLEM.md): AlphaEvolve reached 0.389 with no expert hints; the best
known bound is 0.400695 (Bellec & Fritz).
"""
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MAX_N = 20_000
# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 120.0))
FFT_THRESHOLD = 1500          # above this, direct convolution gets slow

ALPHAEVOLVE = 0.389
BEST_KNOWN = 0.400695


def _run_solve(solution_dir: Path, timeout: float) -> tuple[list, float]:
    """Run solve() in a fresh subprocess. Return (weights, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': [float(x) for x in v], 'time': dt}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"solve() ran past its {timeout:.0f}s budget and was killed. Keep its "
            "internal time budgets safely below the evaluator's limit."
        ) from None
    if out.returncode != 0:
        # A killed process (negative exit code, or 134/137/139) usually means the
        # memory cap or another resource limit — and it leaves NO stderr, so say
        # what happened instead of returning an empty message.
        detail = (out.stderr or "").strip() or (out.stdout or "").strip()[-500:]
        reason = f"exit code {out.returncode}"
        if out.returncode < 0 or out.returncode in (134, 137, 139):
            reason += " (killed — likely the memory cap or another resource limit)"
        raise RuntimeError(f"solve failed, {reason}: {detail[:500] or 'no output produced'}")
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            return data["value"], data["time"]
    raise RuntimeError("solve produced no JSON line")


def _verify_shape(w) -> str | None:
    """Return None if w is a valid weight vector, else an error string."""
    if not isinstance(w, list):
        return f"expected list, got {type(w).__name__}"
    if len(w) < 2:
        return "need at least 2 weights"
    if len(w) > MAX_N:
        return f"at most {MAX_N} weights allowed, got {len(w)}"
    total = 0.0
    for i, x in enumerate(w):
        if not isinstance(x, (int, float)) or not math.isfinite(x):
            return f"non-finite weight at index {i}: {x!r}"
        if x < 0:
            return f"negative weight at index {i}: {x!r} (weights must be non-negative)"
        total += x
    if total <= 0:
        return "all weights are zero"
    return None


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear convolution, FFT-based for large inputs. Negatives are FFT round-off."""
    if max(len(a), len(b)) <= FFT_THRESHOLD:
        return np.convolve(a, b)
    size = len(a) + len(b) - 1
    fsize = 1 << (size - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(a, fsize) * np.fft.rfft(b, fsize), fsize)[:size]
    return np.maximum(out, 0.0)


def win_probability(weights: list[float]) -> float:
    """P[X1 + X2 + X3 < 2*X4] for iid draws from the normalised weights."""
    p = np.asarray(weights, dtype=np.float64)
    p = np.maximum(p, 0.0)
    p /= p.sum()

    pmf_y = _convolve(_convolve(p, p), p)          # support 0 .. 3(n-1)
    cdf_y = np.cumsum(pmf_y)

    n = len(p)
    probs = np.zeros(n)
    # For X4 = m >= 1: P[Y < 2m] = cdf_Y[2m - 1]. The m = 0 term contributes nothing.
    idx = 2 * np.arange(1, n) - 1
    probs[1:] = cdf_y[idx]
    return float(np.dot(p, probs))


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        w, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    shape_err = _verify_shape(w)
    if shape_err is not None:
        return {"score": 0.0, "stage": "invalid", "error": shape_err,
                "time": time.perf_counter() - t0}

    prob = win_probability(w)
    if not math.isfinite(prob) or prob > 1.0 + 1e-9:
        return {"score": 0.0, "stage": "invalid",
                "error": f"computed probability {prob!r} is not a valid probability",
                "time": time.perf_counter() - t0}

    prob = max(0.0, min(prob, 1.0))
    return {
        # No upper clamp: the divisor is a RECORD, not a proven bound, so a
        # candidate that beats it must score above 1.0. Clamping here made a
        # new record indistinguishable from merely matching one, and destroyed
        # the ordering among record-beating candidates.
        "score": prob / BEST_KNOWN,
        "stage": "full",
        # The three keys the briefing and the steering messages read. fitness is a
        # normalised ratio and compresses near the record; these carry the number the
        # agent actually reasons about, at full precision.
        "objective": prob,
        "objective_name": "P(win)",
        "objective_direction": "max",
        "objective_target": BEST_KNOWN,
        "win_probability": prob,
        "support_size": len(w),
        "nonzero_atoms": int(sum(1 for x in w if x > 0)),
        "alphaevolve": ALPHAEVOLVE,
        "best_known": BEST_KNOWN,
        "beats_alphaevolve": prob > ALPHAEVOLVE,
        "beats_best_known": prob > BEST_KNOWN,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
