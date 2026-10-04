"""The 2:1 analog JSCC channel — the exact code the evaluator scores you with.

This module is mounted read-only at /support and is already on your PYTHONPATH, so
`import jscc_channel` works from your worktree with no path setup. A byte-identical
copy lives inside the evaluator bundle, so a curve you measure here scores the same
when submitted.

Typical use while searching:

    import jscc_channel as ch
    from solution import get_curve

    z, pts = ch.validate(get_curve(15.0))            # raises on a malformed table
    sdr = ch.best_sdr_db(z, pts, 15.0, ch.PROBE_SAMPLES, ch.snr_seed(3))[1]
    print(sdr, ch.opta_db(15.0))

    result = ch.score_curve({s: ch.validate(get_curve(float(s)))
                             for s in ch.SWEEP_DB})   # the full scored sweep
    print(result["combined_score"], result["mean_gap_db"], result["per_snr"])

`sdr_db` is a Monte-Carlo estimate: +-0.08 dB at PROBE_SAMPLES, +-0.02 dB at
SCORE_SAMPLES. Seeds are fixed, so the same table always yields the same number.
"""
import math

import numpy as np

# ---------------------------------------------------------------------------
# The problem instance
# ---------------------------------------------------------------------------
M = 2                                       # source dimension: s ~ N(0, I_2)
RHO = 1.0 / M                               # bandwidth ratio: 1 channel use per pair
SWEEP_DB = (0, 5, 10, 15, 20, 25, 30)       # the scored SNR points

# Canonical sign reflections used to re-pair the negative arm (see `variants`).
ROTATIONS = ((-1, -1), (1, -1), (-1, 1))

# Table shape limits.
MIN_POINTS, MAX_POINTS = 64, 20_000
MAX_COORD = 50.0

# Encoder search resolution: the nearest-point projection runs on a uniform grid of
# min(4P, MAX_SEARCH_POINTS) transmit values, so a table beyond ~10000 points buys
# decoder resolution but no further encoder resolution.
MAX_SEARCH_POINTS = 40_000

# Monte-Carlo protocol. The variant probe only has to RANK the canonical variants, so
# it runs cheap; the winner is then re-measured at full precision and that is the
# number scored.
PROBE_SAMPLES = 8_000
SCORE_SAMPLES = 200_000
SEED = 20260904

# An SDR this far above OPTA cannot happen — it means the table or the measurement is
# broken, so the evaluator rejects the candidate rather than crediting it.
OPTA_MARGIN_DB = 0.05


def opta_db(snr_db):
    """OPTA = rho * 10*log10(1 + CSNR) — the information-theoretic ceiling, in dB."""
    return RHO * 10.0 * np.log10(1.0 + 10.0 ** (snr_db / 10.0))


def snr_seed(index):
    """The fixed seed used at sweep position `index`."""
    return SEED + 977 * index


def combined_score(mean_gap_db, worst_gap_db):
    """The objective: 1 / (1 + 0.5*mean_gap + 0.5*worst_gap), in (0, 1]."""
    return 1.0 / (1.0 + 0.5 * mean_gap_db + 0.5 * worst_gap_db)


# ---------------------------------------------------------------------------
# Table validation
# ---------------------------------------------------------------------------
def validate(curve):
    """Check one get_curve() return value. Returns (z, points); raises ValueError."""
    if not isinstance(curve, dict) or "z" not in curve or "points" not in curve:
        raise ValueError(f"get_curve must return {{'z': (P,), 'points': (P, {M})}}")
    z = np.asarray(curve["z"], dtype=np.float64).reshape(-1)
    pts = np.asarray(curve["points"], dtype=np.float64)
    if pts.ndim != 2 or pts.shape != (z.size, M):
        raise ValueError(f"points shape {pts.shape} != ({z.size}, {M})")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(pts))):
        raise ValueError("non-finite values in z or points")
    if not (MIN_POINTS <= z.size <= MAX_POINTS):
        raise ValueError(f"P={z.size} outside [{MIN_POINTS}, {MAX_POINTS}]")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing")
    if np.max(np.abs(pts)) > MAX_COORD:
        raise ValueError(f"|points| exceeds {MAX_COORD}")
    return z, pts


# ---------------------------------------------------------------------------
# The channel simulation — the ground truth of this problem
# ---------------------------------------------------------------------------
def sdr_db(z, pts, snr_db, num_samples, seed):
    """End-to-end SDR of one table at one SNR, in dB.

    Encoder: nearest point on the table, searched on a uniform grid of
    min(4P, 40000) transmit values spanning +-max|z|; the winning point's z is sent.
    Channel: y = z + n with noise variance mean(z_tx^2)/CSNR, so the power is
    normalised to what the curve actually transmits and rescaling all z is a no-op.
    Decoder: monotone linear interpolation of y back through the table, end-clamped.
    SDR = 10*log10(1 / per-component MSE) against unit source variance.
    """
    m = pts.shape[1]
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((num_samples, m))

    half = max(abs(z[0]), abs(z[-1]))
    grid = np.linspace(-half, half, min(4 * z.size, MAX_SEARCH_POINTS))
    curve = np.stack([np.interp(grid, z, pts[:, j]) for j in range(m)], axis=1)

    s32, c32 = s.astype(np.float32), curve.astype(np.float32)
    c_norm = (c32 ** 2).sum(axis=1)
    best_d = np.full(num_samples, np.inf, dtype=np.float32)
    best_i = np.zeros(num_samples, dtype=np.int64)
    for a in range(0, num_samples, 8192):
        sa = s32[a:a + 8192]
        bd, bi = best_d[a:a + 8192], best_i[a:a + 8192]
        for b in range(0, grid.size, 8192):
            d = c_norm[b:b + 8192] - 2.0 * (sa @ c32[b:b + 8192].T)
            i = d.argmin(axis=1)
            v = d[np.arange(sa.shape[0]), i]
            better = v < bd
            bd[better] = v[better]
            bi[better] = i[better] + b
        best_d[a:a + 8192], best_i[a:a + 8192] = bd, bi

    z_tx = grid[best_i]
    p_tx = float(np.mean(z_tx ** 2))
    sigma = math.sqrt(p_tx / 10.0 ** (snr_db / 10.0))
    y = z_tx + rng.standard_normal(num_samples) * sigma

    s_hat = np.stack([np.interp(y, z, pts[:, j]) for j in range(m)], axis=1)
    mse = float(np.mean((s - s_hat) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


# ---------------------------------------------------------------------------
# Canonical variants: granted free, so a good geometry is never sunk by a
# labelling convention its author never tuned.
# ---------------------------------------------------------------------------
def arclength_relabel(z, pts):
    """Relabel z as signed arc length, anchored at the point nearest z = 0."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 0])
    z, pts = z[keep], pts[keep]
    if z.size < MIN_POINTS:
        return None
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return arc - arc[np.argmin(np.abs(z))], pts


def mirrored(z, pts, rotation):
    """Rebuild the negative arm as `rotation * positive arm`."""
    i0 = np.searchsorted(z, 0.0)
    z_pos, p_pos = z[i0:], pts[i0:]
    if z_pos.size < MIN_POINTS // 2 + 1:
        return None
    if abs(z_pos[0]) > 1e-9:
        z_pos = np.concatenate([[0.0], z_pos])
        p_pos = np.concatenate([[p_pos[0]], p_pos], axis=0)
    r = np.asarray(rotation, dtype=np.float64)
    return (np.concatenate([-z_pos[:0:-1], z_pos]),
            np.concatenate([(r * p_pos)[:0:-1], p_pos], axis=0))


def variants(z, pts, rotations=ROTATIONS):
    """The canonical variant set of one table: [(tag, z, points), ...].

    Two free levers, crossed: the z-labelling (as submitted, or re-derived as signed
    arc length) and the arm pairing (as submitted, or the negative arm rebuilt as a
    sign reflection of the positive one).
    """
    out = [("as-is", z, pts)]
    arc = arclength_relabel(z, pts)
    labelings = [("", z, pts)] + ([("arc", arc[0], arc[1])] if arc else [])
    for ltag, lz, lp in labelings:
        if ltag:
            out.append(("arc", lz, lp))
        for r in rotations:
            m = mirrored(lz, lp, r)
            if m is None:
                continue
            mz, mp = m
            if mz.size == z.size and np.allclose(mz, z) and np.allclose(mp, pts):
                continue                      # identical to the submission
            rtag = "rot(" + ",".join("+" if v > 0 else "-" for v in r) + ")"
            out.append(((ltag + "+" if ltag else "") + rtag, mz, mp))
    return out


def best_variant(z, pts, snr, num_samples, seed, rotations=ROTATIONS):
    """(tag, vz, vpts, sdr) of the best canonical variant at one SNR, every variant
    measured at `num_samples`. Use it to compare designs cheaply; `score_curve` is
    what actually scores."""
    return max(((tag, vz, vp, sdr_db(vz, vp, snr, num_samples, seed))
                for tag, vz, vp in variants(z, pts, rotations)),
               key=lambda t: t[3])


def best_sdr_db(z, pts, snr, num_samples, seed, rotations=ROTATIONS):
    """(tag, sdr) of the best canonical variant at one SNR."""
    tag, _, _, sdr = best_variant(z, pts, snr, num_samples, seed, rotations)
    return tag, sdr


# ---------------------------------------------------------------------------
# The scored sweep
# ---------------------------------------------------------------------------
def score_curve(tables, probe_samples=PROBE_SAMPLES, score_samples=SCORE_SAMPLES):
    """Score a whole sweep. `tables` maps each SNR in SWEEP_DB to a (z, points) pair.

    Two stages per SNR: every canonical variant is probed at `probe_samples` to pick
    the winner, and only the winner is re-measured at `score_samples`. The scored
    number is the second one.

    Returns a dict with combined_score, mean/worst gap, and the per-SNR breakdown.
    """
    per_snr, gaps, above_opta = [], [], []
    for i, snr in enumerate(SWEEP_DB):
        z, pts = tables[snr]
        seed = snr_seed(i)
        tag, vz, vp, _ = best_variant(z, pts, float(snr), probe_samples, seed)
        sdr = sdr_db(vz, vp, float(snr), score_samples, seed)
        opta = float(opta_db(snr))
        gap = opta - sdr
        gaps.append(max(gap, 0.0))
        if gap < -OPTA_MARGIN_DB:
            above_opta.append(snr)
        per_snr.append({"snr_db": int(snr), "sdr_db": round(float(sdr), 4),
                        "opta_db": round(opta, 4), "gap_db": round(float(gap), 4),
                        "variant": tag})

    mean_gap, worst_gap = float(np.mean(gaps)), float(np.max(gaps))
    return {"combined_score": combined_score(mean_gap, worst_gap),
            "mean_gap_db": mean_gap, "worst_gap_db": worst_gap,
            "per_snr": per_snr, "above_opta": above_opta}
