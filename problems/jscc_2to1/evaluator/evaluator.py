"""2:1 analog JSCC — score a space-filling curve end to end over the SNR sweep.

Called by evaluate.sh with the solution's path as argv[1]; prints a single JSON line
to stdout. Runs inside the evaluation sandbox — no network, hard resource limits — so
it assumes nothing about the machine beyond python3 and numpy.

Design notes:
- The candidate only ever produces TABLES. `_run_get_curve` executes get_curve() in a
  fresh subprocess which writes the tables to an .npz and exits; every byte of the
  channel simulation, the variant search and the scoring then happens here, in this
  process, from this bundle's own copy of `jscc_channel`. A candidate therefore cannot
  reach the scoring code, the RNG, or its own number — the anti-gaming hook that
  matters for a Monte-Carlo metric.
- `jscc_channel.py` here is byte-identical to the copy mounted at /support, so a curve
  the agent measured while searching scores exactly the same when submitted.
- SDR above OPTA is information-theoretically impossible. It is treated as an invalid
  candidate, never as a discovery: it can only mean the table or the measurement is
  broken, and crediting it would reward exactly the wrong thing.
- get_curve is called once per swept SNR plus one repeat, and the repeat must match
  bit for bit. A table that changes between calls cannot be re-verified later.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np                      # noqa: E402
import jscc_channel as ch               # noqa: E402

# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone. It covers the whole set of get_curve
# calls (one per swept SNR plus the determinism repeat), not one call.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 300.0))

# The objective is `combined_score` = 1/(1 + 0.5*mean_gap + 0.5*worst_gap) over the
# sweep, maximised, in (0, 1]. Its ceiling of 1 is analytic, not a guess: both gaps are
# measured against OPTA = rho*10*log10(1 + CSNR), the information-theoretic optimum
# for this bandwidth ratio, and a gap can never go below 0. Reaching 1 would mean
# hitting OPTA at every SNR with a one-dimensional curve, which no known construction
# does.
#
# TARGET is a bar to BEAT by a clear margin, not to meet. Unlike most problems here, this
# one's target IS the best known design rather than a round number set above it — 0.362,
# the strongest curve supplied with the task (mean gap 1.66 dB, worst 1.87 dB) — because
# beating it by a clear margin is the result this problem exists to produce. It is
# rendered to the agent as "target" and never as a record: a session told it has passed
# the record reads itself as finished, while a target it is still short of keeps the gap
# live. The provenance and the full comparison table live in paper/benchmark.md.
CEILING = 1.0
TARGET = 0.362

# The determinism repeat is run at this sweep position.
REPEAT_INDEX = 3


def _run_get_curve(solution_dir: Path, timeout: float) -> tuple[Path, float]:
    """Call get_curve() for every swept SNR in a fresh subprocess.

    Returns (path to the .npz of tables, wall seconds spent inside get_curve)."""
    out_path = solution_dir / "_curves.npz"
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        f"sys.path.insert(0, {str(_HERE)!r})\n"
        "import numpy as np\n"
        "from solution import get_curve\n"
        f"sweep = {list(ch.SWEEP_DB)!r}\n"
        f"calls = sweep + [sweep[{REPEAT_INDEX}]]\n"
        "tables, t = {}, time.perf_counter()\n"
        "for k, snr in enumerate(calls):\n"
        "    c = get_curve(float(snr))\n"
        "    if not isinstance(c, dict) or 'z' not in c or 'points' not in c:\n"
        "        raise TypeError(f'get_curve({snr}) did not return a dict with z and points')\n"
        "    tables[f'z{k}'] = np.asarray(c['z'], dtype=np.float64).reshape(-1)\n"
        "    tables[f'p{k}'] = np.asarray(c['points'], dtype=np.float64)\n"
        "dt = time.perf_counter() - t\n"
        f"np.savez({str(out_path)!r}, **tables)\n"
        "print(json.dumps({'time': dt}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"get_curve() ran past the {timeout:.0f}s budget for the whole sweep and "
            "was killed. Keep any per-SNR search you do inside get_curve safely below "
            "that total."
        ) from None
    if out.returncode != 0:
        # A killed process (negative exit code, or 134/137/139) usually means the
        # memory cap or another resource limit — and it leaves NO stderr, so say what
        # happened instead of returning an empty message.
        detail = (out.stderr or "").strip() or (out.stdout or "").strip()[-500:]
        reason = f"exit code {out.returncode}"
        if out.returncode < 0 or out.returncode in (134, 137, 139):
            reason += " (killed — likely the memory cap or another resource limit)"
        raise RuntimeError(f"get_curve failed, {reason}: {detail[:500] or 'no output produced'}")
    if not out_path.exists():
        raise RuntimeError("get_curve produced no table file")
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return out_path, float(json.loads(line)["time"])
    raise RuntimeError("get_curve produced no JSON line")


def _load_tables(npz_path: Path) -> tuple[dict, str | None]:
    """Validate every table. Returns ({snr: (z, points)}, error or None)."""
    data = np.load(npz_path)
    tables = {}
    for k, snr in enumerate(ch.SWEEP_DB):
        try:
            tables[snr] = ch.validate({"z": data[f"z{k}"], "points": data[f"p{k}"]})
        except (ValueError, KeyError) as e:
            return {}, f"the table returned at {snr} dB is invalid: {e}"

    # Determinism: the repeat call has to reproduce its table bit for bit.
    n = len(ch.SWEEP_DB)
    z0, p0 = tables[ch.SWEEP_DB[REPEAT_INDEX]]
    if f"z{n}" not in data:
        return {}, "the determinism repeat produced no table"
    if not (np.array_equal(z0, data[f"z{n}"]) and np.array_equal(p0, data[f"p{n}"])):
        return {}, (f"get_curve({ch.SWEEP_DB[REPEAT_INDEX]}) returned a different table "
                    "on a second call — get_curve must be deterministic, so seed any "
                    "randomness you use from snr_db")
    return tables, None


def evaluate(solution_path: Path) -> dict:
    """Single-stage evaluation. Returns a dict suitable for JSON dump."""
    t0 = time.perf_counter()
    try:
        npz_path, solve_time = _run_get_curve(solution_path.parent, timeout=SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — a bad candidate scores 0, never crashes us
        return {"score": 0.0, "stage": "error", "error": str(e)[:400],
                "time": time.perf_counter() - t0}

    tables, err = _load_tables(npz_path)
    if err is not None:
        return {"score": 0.0, "stage": "invalid", "error": err[:400],
                "time": time.perf_counter() - t0}

    r = ch.score_curve(tables)

    if r["above_opta"]:
        return {"score": 0.0, "stage": "invalid",
                "error": ("SDR above OPTA at " +
                          ", ".join(f"{s} dB" for s in r["above_opta"]) +
                          " — that is information-theoretically impossible, so the "
                          "table or the way it is built is broken, not record-setting."),
                "per_snr": r["per_snr"], "time": time.perf_counter() - t0}

    score = float(r["combined_score"])
    return {
        "score": score,
        "stage": "full",
        # The keys the briefing and the steering messages read. The objective is the
        # combined score itself: it is already the [0, 1] quantity the sweep defines.
        "objective": score,
        "objective_name": "combined_score",
        "objective_direction": "max",
        "objective_target": TARGET,
        "combined_score": score,
        "mean_gap_db": round(r["mean_gap_db"], 4),
        "worst_gap_db": round(r["worst_gap_db"], 4),
        # Per-SNR feedback: which end of the sweep is costing the score, and which
        # canonical variant won there.
        "per_snr": r["per_snr"],
        "sdr_db": [p["sdr_db"] for p in r["per_snr"]],
        "gap_db": [p["gap_db"] for p in r["per_snr"]],
        "worst_snr_db": max(r["per_snr"], key=lambda p: p["gap_db"])["snr_db"],
        "table_size": int(tables[ch.SWEEP_DB[0]][0].size),
        "ceiling": CEILING,
        "target": TARGET,
        "solve_time": solve_time,
        "time": time.perf_counter() - t0,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
