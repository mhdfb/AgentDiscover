"""A space-filling curve for 2:1 analog joint source-channel coding.

See PROBLEM.md. The mutable region is marked by # EVOLVE-BLOCK-START /
# EVOLVE-BLOCK-END. Everything outside it is fixed scaffolding — do not modify it.

CONTRACT (the evaluator relies on exactly this):

    get_curve(snr_db: float) -> {"z": (P,) float, "points": (P, 2) float}

  * points[i] is a point of the curve in SOURCE space (2-D iid Gaussian, sigma^2 = 1).
  * z[i] is the TRANSMITTED channel symbol assigned to that point; z must be strictly
    increasing and finite; 64 <= P <= 20000; |points| <= 50.
  * Encoder (fixed): nearest-point projection of each source sample onto the table;
    transmits that point's z. Channel: AWGN with noise variance mean(z_tx^2)/CSNR —
    power-normalised, so rescaling all z changes nothing and only the RELATIVE
    spacing of z matters.
  * Decoder (fixed): receives the noisy z, maps it back through the table by linear
    interpolation, end-clamped. No other receiver processing exists.

Design freedoms: the geometry (any shape), how densely points cover each region, the
z-spacing along the curve (companding / power allocation), and full per-SNR adaptation
through the snr_db argument. get_curve must be deterministic — it is called twice at
one SNR and the two tables have to match exactly.

`import jscc_channel` (mounted at /support, already on PYTHONPATH) to measure any
curve with the evaluator's own code. Run `python3 solution.py` for a quick sweep.
"""
import numpy as np

M = 2  # source dimension (do not change)

# EVOLVE-BLOCK-START

PARAMS = {
    "delta": 1.6,      # spiral pitch: radius grows delta/pi per radian
    "r_max": 4.5,      # curve reach in source-space sigmas
    "n_points": 4001,  # table size (odd -> symmetric around the origin)
}


def get_curve(snr_db):
    """Archimedes spiral, arc-length parametrised, same shape at every SNR.

    The seed ignores snr_db entirely, which is the first thing worth fixing: the
    fold spacing that wins at 0 dB is not the one that wins at 30 dB.

    Args:
        snr_db: the true operating channel SNR in dB, one of the swept values.

    Returns:
        {"z": (P,) strictly increasing, "points": (P, 2)}.
    """
    p = PARAMS
    a = p["delta"] / np.pi
    phi_max = p["r_max"] / a

    # closed-form arc length of r = a*phi, inverted numerically
    phi_grid = np.linspace(0.0, phi_max, 20001)
    arc_grid = 0.5 * a * (phi_grid * np.sqrt(1.0 + phi_grid ** 2)
                          + np.arcsinh(phi_grid))

    half = (int(p["n_points"]) + 1) // 2
    z_pos = np.linspace(0.0, arc_grid[-1], half)
    phi = np.interp(z_pos, arc_grid, phi_grid)
    pts_pos = np.stack([a * phi * np.cos(phi), a * phi * np.sin(phi)], axis=1)

    # negative arm: point reflection through the origin
    z = np.concatenate([-z_pos[::-1], z_pos[1:]])
    pts = np.concatenate([-pts_pos[::-1], pts_pos[1:]], axis=0)
    return {"z": z, "points": pts}

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import jscc_channel as ch

    print(f"{'SNR':>4} {'SDR':>8} {'OPTA':>8} {'gap':>7}   variant")
    gaps = []
    for i, snr in enumerate(ch.SWEEP_DB):
        z, pts = ch.validate(get_curve(float(snr)))
        tag, sdr = ch.best_sdr_db(z, pts, float(snr), ch.PROBE_SAMPLES, ch.snr_seed(i))
        opta = float(ch.opta_db(snr))
        gaps.append(max(opta - sdr, 0.0))
        print(f"{snr:>4} {sdr:8.2f} {opta:8.2f} {opta - sdr:7.2f}   {tag}")
    mean_gap, worst_gap = float(np.mean(gaps)), float(np.max(gaps))
    print(f"\nmean gap {mean_gap:.2f} dB | worst gap {worst_gap:.2f} dB | "
          f"combined_score {ch.combined_score(mean_gap, worst_gap):.4f}  "
          f"(probe precision — the evaluator re-measures the winner at "
          f"{ch.SCORE_SAMPLES} samples)")
