"""LLM-SQL — column reordering for LLM prefix caching. CORAL's metric, unchanged.

Ported from CORAL's `examples/ADRS/llm_sql` (taskdata/evaluator.py, utils.py, solver.py),
which is byte-identical to SkyDiscover's `benchmarks/ADRS/llm_sql/evaluator` — the
bundle behind CORAL's published figure. The task, the data and the metric are theirs:

  * `utils.py` and `solver.py` are byte-identical copies. The hit count is
    `utils.evaluate_df_prefix_hit_cnt`, their function, called on the returned frame.
  * The five data sets, their order, the `col_merge` argument for each, and the fixed
    call `reorder(df, early_stop=100000, distinct_value_threshold=0.7, row_stop=4,
    col_stop=2, col_merge=...)` are copied from their evaluator, unchanged.
  * The candidate is validated as there — row count preserved, total character count not
    shrunk, with their error messages verbatim — plus TWO guards of ours, because their
    checks let a candidate ignore the data and still score ~0.99 (constant identical rows
    grow the character count and self-overlap to a near-perfect hit rate), or run
    arbitrary code in this process at unpickle time:
      1. `_reordering_error`: every returned row must be one input row's cells, in any
         column order and optionally merged — same characters, each original cell intact.
         A row that is not fails the candidate (score 0). Honest reorderings — including
         the seed's and every scored candidate of the archived Opus 4.6 run — pass
         unchanged, so the number stays identical to CORAL's and SkyDiscover's.
      2. `_FrameUnpickler`: the frame pickled by the candidate's subprocess is loaded
         with a restricted unpickler that admits only pandas/numpy frame internals, so a
         malicious `__reduce__` cannot execute in the scoring process.
  * `combined_score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, runtime)) / 12`,
    averaged over the five data sets, higher is better. CORAL times the `reorder` call
    alone, measured inside the candidate's own process; we score the speed on the wall
    clock of the WHOLE candidate subprocess (see below), so it cannot be gamed by moving
    work to import time or under-reported from inside that process. Honest candidates move
    only in the 4th decimal.
  * A data set that raises is reported the way their evaluator reports it — "1 or more
    files failed to run", or "No files processed successfully" if it was the first —
    with no traceback. That is the feedback CORAL's agents got, so it is the feedback
    ours get; the traceback goes to stderr for the operator only.

What differs is only the process layout, to fit this platform:

  * The candidate's `reorder` runs in a fresh subprocess (`_run_reorders`) that pickles
    each returned frame to the work directory and exits; validation and scoring happen
    HERE, afterwards, from this bundle's own copies of the metric. The candidate can
    therefore never reach the scoring code or report a number for itself — in CORAL's
    layout the candidate module is executed inside the grader process.
  * The speed term is scored on `child_wall`, the wall clock of the whole subprocess as
    measured by this parent process — out of the candidate's reach entirely. The
    per-`reorder` times the subprocess reports are kept only as feedback; the tools that
    measure and emit them are bound before the candidate is imported, so a candidate that
    runs in that process cannot patch `time`/`json`/`stdout` to falsify them either.
  * The data sets are fetched once by `fetch.sh` and mounted read-only; their location
    arrives in AGENTDISCOVER_DATA_DIR. The fallback to this directory keeps the bundle
    runnable standalone, the way SkyDiscover's is (datasets/ beside evaluator.py).
  * `main()` prints the one JSON line the platform's contract asks for, where
    SkyDiscover bridges the same dict through their `wrapper.py`.

The score the platform ranks on IS the combined score: it is already the [0, 1]
quantity the task defines, and 1 is its analytic ceiling (a hit rate is at most 1, the
runtime term at most 1).
"""
import contextlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import pandas as pd                                   # noqa: E402
from utils import evaluate_df_prefix_hit_cnt          # noqa: E402

# The platform fetches the data set once (fetch.sh) and mounts it read-only, naming the
# directory here. Falling back to this file's own directory keeps the bundle runnable
# standalone, the way SkyDiscover's is.
DATA_DIR = Path(os.environ.get("AGENTDISCOVER_DATA_DIR", str(_HERE))) / "datasets"

# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone. It covers the five `reorder` calls
# together — CORAL's grader has one timeout for the whole evaluation — not one call.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 300.0))

# The objective is `combined_score`, maximised, in [0, 1]. The ceiling of 1 is analytic:
# the hit rate is a fraction of characters and the runtime term is clamped to [0, 1].
#
# TARGET is a bar to BEAT by a clear margin, not to meet. Competitor numbers live in
# paper/benchmark.md and are deliberately kept out of everything the agent reads —
# naming one anchors the search on it.
CEILING = 1.0
TARGET = 0.739

# SkyDiscover's test set, verbatim: the five tables, in this order, and the col_merge
# argument passed for each. The order matters for the failure messages (see above).
TEST_FILES = ["movies.csv", "beer.csv", "BIRD.csv", "PDMX.csv", "products.csv"]
COL_MERGES = [
    [['movieinfo', 'movietitle', 'rottentomatoeslink']],
    [['beer/beerId', 'beer/name']],
    [['PostId', 'Body']],
    [['path', 'metadata'], ['hasmetadata', 'isofficial', 'isuserpublisher', 'isdraft',
                            'hasannotations', 'subsetall']],
    [['product_title', 'parent_asin']],
]
# The fixed keyword arguments of every reorder call, theirs.
REORDER_KWARGS = {"early_stop": 100000, "distinct_value_threshold": 0.7,
                  "row_stop": 4, "col_stop": 2}

# CORAL's / SkyDiscover's score formula, theirs.
HIT_WEIGHT, RUNTIME_WEIGHT, RUNTIME_CAP = 0.95, 0.05, 12.0


def _combined(average_hit_rate: float, average_runtime: float) -> float:
    return (HIT_WEIGHT * average_hit_rate
            + RUNTIME_WEIGHT * (RUNTIME_CAP - min(RUNTIME_CAP, average_runtime)) / RUNTIME_CAP)


def _run_reorders(solution_dir: Path, timeout: float) -> tuple[dict, float]:
    """Call the candidate's reorder() on every data set, in a fresh subprocess.

    The subprocess imports `solution` from the work directory, with this bundle first
    on its path so `from solver import Algorithm` resolves to the real base class. Each
    returned frame is pickled to `_reordered_<i>.pkl`; the frames are never inspected
    there. Returns (status dict from the subprocess, its wall seconds)."""
    code = (
        "import sys, json, pickle, traceback\n"
        # Bound before the candidate exists, so a patched `time`/`json`/`stdout` cannot
        # reach them: the candidate runs in this process and could otherwise swap the tools
        # the harness uses to measure and emit its status.
        "from time import perf_counter as _clock\n"
        "_dumps = json.dumps\n"
        "_emit = sys.stdout.write\n"
        "def _say(obj):\n"
        "    _emit(_dumps(obj) + '\\n')\n"
        "sys.path.insert(0, '.')\n"
        f"sys.path.insert(0, {str(_HERE)!r})\n"
        "import pandas as pd\n"
        f"files = {[str(DATA_DIR / f) for f in TEST_FILES]!r}\n"
        f"merges = {COL_MERGES!r}\n"
        f"kwargs = {REORDER_KWARGS!r}\n"
        "status = {'runtimes': [], 'failed_at': None}\n"
        "try:\n"
        "    import solution as program\n"
        "except Exception as e:\n"
        "    status['import_error'] = str(e)\n"
        "    _say(status); sys.exit(0)\n"
        "if not hasattr(program, 'Evolved'):\n"
        "    status['missing'] = True\n"
        "    _say(status); sys.exit(0)\n"
        "for i, (path, col_merge) in enumerate(zip(files, merges)):\n"
        "    try:\n"
        "        master_df = pd.read_csv(path)\n"
        "        st = _clock()\n"
        "        reordered, _ = program.Evolved().reorder(master_df, col_merge=col_merge, **kwargs)\n"
        "        rt = _clock() - st\n"
        "        with open(f'_reordered_{i}.pkl', 'wb') as fh:\n"
        "            pickle.dump(reordered, fh, protocol=pickle.HIGHEST_PROTOCOL)\n"
        "        status['runtimes'].append(rt)\n"
        "    except Exception:\n"
        "        traceback.print_exc()\n"
        "        status['failed_at'] = i\n"
        "        break\n"
        "_say(status)\n"
    )
    t0 = time.perf_counter()
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Evolved.reorder ran past the {timeout:.0f}s budget for the five data sets "
            "together and was killed. Keep the total well below that; runtime above "
            "12 s per data set already scores nothing on the speed term."
        ) from None
    wall = time.perf_counter() - t0
    # The candidate's own printing and any traceback stay with the operator.
    if out.stderr:
        sys.stderr.write(out.stderr[-4000:])
    if out.returncode != 0:
        # A killed process (negative exit code, or 134/137/139) usually means the
        # memory cap or another resource limit — and it leaves NO stderr, so say what
        # happened instead of returning an empty message.
        detail = (out.stderr or "").strip() or (out.stdout or "").strip()[-500:]
        reason = f"exit code {out.returncode}"
        if out.returncode < 0 or out.returncode in (134, 137, 139):
            reason += " (killed — likely the memory cap or another resource limit)"
        raise RuntimeError(f"reorder failed, {reason}: {detail[:500] or 'no output produced'}")
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line), wall
    raise RuntimeError("the reorder subprocess produced no status line")


def _char_count(df: pd.DataFrame) -> float:
    """SkyDiscover's character count, verbatim."""
    return df.astype(str).apply(lambda x: x.str.len().sum(), axis=1).sum()


# What a pickled pandas DataFrame refers to, and nothing else. The frame comes from the
# subprocess that ran the candidate's code, and a plain pickle.load would let whatever
# that code returned run arbitrary code HERE, in the scoring process, via __reduce__.
_SAFE_MODULES = ("pandas.core.", "pandas._libs.", "numpy.core.", "numpy._core.",
                 "numpy.dtypes", "datetime", "zoneinfo", "dateutil.tz", "pytz")
_SAFE_BUILTINS = {"slice", "set", "frozenset", "bool", "int", "float", "str", "bytes",
                  "bytearray", "list", "dict", "tuple", "complex", "range", "object"}


class _FrameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" or module.startswith(_SAFE_MODULES) \
                or (module == "builtins" and name in _SAFE_BUILTINS):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"reorder must return a plain pandas DataFrame; refusing to unpickle {module}.{name}")


def _reordering_error(master_df: pd.DataFrame, reordered: pd.DataFrame) -> str | None:
    """SkyDiscover's checks accept any frame with the same row count and at least as
    many characters — a frame of identical constant rows scores 0.99 while ignoring the
    data. This requires every returned row to be one input row's cells, in any column
    order, possibly concatenated (merged columns): the same characters, with each
    original cell intact. Rows are matched one-to-one. None if valid, else the error."""
    pool: dict[str, list[list[str]]] = {}
    for cells in master_df.astype(str).values.tolist():
        pool.setdefault("".join(sorted("".join(cells))), []).append(cells)
    for i, cells in enumerate(reordered.astype(str).values.tolist()):
        row = "".join(cells)
        matches = pool.get("".join(sorted(row)), [])
        for k, orig in enumerate(matches):
            if all(c in row for c in orig):
                matches.pop(k)
                break
        else:
            return (f"Evaluation failed: returned row {i} is not a reordering of any input "
                    "row. Each row must keep exactly its original cells, reordered or "
                    "merged, with no data changed, added or dropped.")
    return None


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    missing = [f for f in TEST_FILES if not (DATA_DIR / f).exists()]
    if missing:
        # An operator problem, never the candidate's: the data set was not fetched.
        return {"score": 0.0, "stage": "error",
                "error": f"data set files missing from {DATA_DIR}: {', '.join(missing)} "
                         "— evaluator/fetch.sh has not run on this machine",
                "time": time.perf_counter() - t0}

    work = solution_path.parent
    try:
        status, child_wall = _run_reorders(work, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — a bad candidate scores 0, never crashes us
        return {"score": 0.0, "stage": "error", "error": str(e)[:400],
                "time": time.perf_counter() - t0}

    # SkyDiscover's own error strings, so the feedback is the one CORAL's agents read.
    if "import_error" in status:
        return {"score": 0.0, "stage": "error", "error": status["import_error"][:400],
                "time": time.perf_counter() - t0}
    if status.get("missing"):
        return {"score": 0.0, "stage": "invalid", "error": "Missing algorithm function",
                "time": time.perf_counter() - t0}

    runtimes = [float(r) for r in status["runtimes"]]
    if sum(runtimes) > child_wall + 1.0:
        return {"score": 0.0, "stage": "invalid",
                "error": "the reported reorder runtimes exceed the wall clock of the "
                         "process that ran them — the timing has been tampered with",
                "time": time.perf_counter() - t0}

    # Validation and scoring, in data-set order, mirroring SkyDiscover's loop: a
    # row-count or character-count violation returns at once with its message; an
    # exception (theirs or ours) counts the file as failed and stops the loop.
    hit_rates: list[float] = []
    total_runtime = 0.0
    failed_files = 1 if status["failed_at"] is not None else 0
    for i, runtime in enumerate(runtimes):
        try:
            master_df = pd.read_csv(DATA_DIR / TEST_FILES[i])
            total_chars_before = _char_count(master_df)
            original_row_count = len(master_df)

            with open(work / f"_reordered_{i}.pkl", "rb") as fh:
                try:
                    reordered = _FrameUnpickler(fh).load()
                except pickle.UnpicklingError as e:
                    return {"score": 0.0, "stage": "invalid", "error": str(e)[:400],
                            "time": time.perf_counter() - t0}
            if not isinstance(reordered, pd.DataFrame):
                raise TypeError("reorder did not return a DataFrame")

            reordered_row_count = len(reordered)
            if reordered_row_count != original_row_count:
                diff = reordered_row_count - original_row_count
                if diff < 0:
                    msg = (f"Evaluation failed: row count decreases by {abs(diff)} rows. "
                           "Data were lost - you might have dropped some rows or failed "
                           "to preserve all data during reordering.")
                else:
                    msg = (f"Evaluation failed: row count increases by {diff} rows. "
                           "Data were duplicated - you might have duplicated some rows "
                           "during reordering.")
                return {"score": 0.0, "stage": "invalid", "error": msg,
                        "time": time.perf_counter() - t0}

            total_chars_after = _char_count(reordered)
            if total_chars_after < total_chars_before:
                char_diff = total_chars_before - total_chars_after
                char_diff_pct = ((char_diff / total_chars_before * 100)
                                 if total_chars_before > 0 else 0)
                msg = (f"Evaluation failed: character decreases by {char_diff_pct:.2f}%. "
                       "Data were lost - you might have dropped some data or failed to "
                       "preserve all data during reordering.")
                return {"score": 0.0, "stage": "invalid", "error": msg,
                        "time": time.perf_counter() - t0}

            msg = _reordering_error(master_df, reordered)
            if msg is not None:
                return {"score": 0.0, "stage": "invalid", "error": msg,
                        "time": time.perf_counter() - t0}

            # Their metric prints as it goes; keep stdout for the JSON line.
            with contextlib.redirect_stdout(sys.stderr):
                results = evaluate_df_prefix_hit_cnt(reordered)
            hit_rates.append(results[1] / 100)
            total_runtime += runtime
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failed_files += 1
            break

    successful_files = len(hit_rates)
    if successful_files == 0:
        return {"score": 0.0, "stage": "invalid", "error": "No files processed successfully",
                "time": time.perf_counter() - t0}
    if failed_files > 0:
        return {"score": 0.0, "stage": "invalid", "error": "1 or more files failed to run",
                "time": time.perf_counter() - t0}

    average_hit_rate = sum(hit_rates) / successful_files
    average_runtime = total_runtime / successful_files
    # The speed term is scored on child_wall — the wall clock of the WHOLE candidate
    # subprocess, measured HERE by the parent — divided over the data sets, NOT on the
    # per-reorder times the subprocess reports. Two holes close at once: a candidate can
    # no longer move its work to import time to make the timed calls look instant (import
    # is inside child_wall), and it can no longer under-report the timing from inside its
    # own process (the parent's clock is out of its reach). The reported per-reorder
    # `average_runtime` stays as feedback. child_wall carries this bundle's own overhead
    # (reading the CSVs, pickling) — ~0.2 s/data set, well under the 12 s cap, so an honest
    # candidate's score moves in the 4th decimal; see paper/fix_evaluator.md.
    scored_runtime = child_wall / successful_files
    score = float(_combined(average_hit_rate, scored_runtime))

    return {
        "score": score,
        "stage": "full",
        # The keys the briefing and the steering messages read. The objective is the
        # combined score itself: it is already the [0, 1] quantity the task defines.
        "objective": score,
        "objective_name": "combined_score",
        "objective_direction": "max",
        "objective_target": TARGET,
        "combined_score": score,
        # The same three figures CORAL's grader reports back — the average over the
        # data sets and the summed runtime — and nothing finer, so the feedback is theirs.
        "average_hit_rate": round(average_hit_rate, 6),
        "total_runtime": round(total_runtime, 4),
        "average_runtime": round(average_runtime, 4),
        "scored_runtime": round(scored_runtime, 4),
        "ceiling": CEILING,
        "target": TARGET,
        "solve_time": round(child_wall, 3),
        "time": time.perf_counter() - t0,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
