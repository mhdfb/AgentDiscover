"""PRISM — LLM model placement on a GPU cluster, CORAL's metric unchanged.

Ported from CORAL's `examples/ADRS/prism/taskdata/evaluator.py` (SkyDiscover's
`benchmarks/ADRS/prism` evaluator with CORAL's one correction). The task, the test cases
and the metric are theirs:

  * `generate_test_gpu_models` produces the same 50 cases from `np.random.seed(42)`:
    5..9 GPUs, twice as many models, sizes 10..29 GB, request rates 1..9, SLOs 5..9;
  * KVPR of a GPU = sum(req_rate/slo) / (80 - sum(model_size)), 10^6 when the GPU is
    full; the per-case quantity is the maximum KVPR over the GPUs (`calculate_kvcache_pressure`);
  * `combined_score` = 1 / mean(max KVPR over the 50 cases) + success_rate;
  * a case whose call raises or runs past 10 s counts 10^6 towards the mean and 0
    towards the success rate — CORAL's fix (paper, Appendix D.3) for SkyDiscover's
    evaluator, which skipped failed cases and so let a candidate that crashed on the
    hard cases be graded on the easy ones alone;
  * the whole evaluation keeps CORAL's 360 s cap (`grader.timeout` in their task.yaml).

The five functions from their file are copied verbatim (the comments are theirs).

Two holes are closed, both open in CORAL (and the first in SkyDiscover too):

  1. Validity. CORAL never checks that a placement is a placement: a result that leaves
     models out, lists one twice, or invents GPU ids is measured on what it contains and
     scores ABOVE the optimum — every published PRISM figure over 26.256 (SeaEvo's
     78.0040, EvoX's 30.52 / 27.67, AdaEvolve's 26.37) is above the exact optimum of these
     50 cases and can only have come that way. Here `_build_placement` rebuilds the
     placement from the authoritative models and rejects any such case (10^6 towards the
     mean, a failure towards the success rate). SkyDiscover added the same check (PR #59).
  2. Isolation. CORAL and SkyDiscover import and run the candidate in the SAME process
     that scores it, so a candidate can override the metric functions or print its own
     score line and exit — reaching any number. Here `_run_placements` runs the candidate
     in a fresh subprocess that returns only its placement as gpu_id -> [model_name];
     the score is computed HERE, in a process the candidate never entered. Names are
     unique per case, so the parent maps each back to the real Model and the candidate
     cannot smuggle in fake sizes.

Honest placements score identically to CORAL's, so the number stays comparable. `main()`
prints the one JSON line the platform's contract asks for.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GPU_MEM_SIZE = 80 # GB
MIN_INT = float('-inf')  # Define MIN_INT as negative infinity
NUM_CASES = 50
PER_CASE_TIMEOUT = 10.0        # CORAL's own per-call limit
_HERE = Path(__file__).resolve().parent   # so the candidate's subprocess can import this bundle
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 300.0))

# The objective is `combined_score`, maximised. The target is the PROVEN OPTIMUM of the
# evaluator's 50 cases: every case solved exactly (binary search on the KVPR threshold
# with a MILP feasibility check per threshold, 2026-09-12), each optimal placement
# rescored by `calculate_kvcache_pressure` below. 1/mean = 25.255971749535238, plus a
# success rate of 1. CORAL's table prints it as 26.26 and reports that every system it
# ran (OpenEvolve, ShinkaEvolve, EvoX, CORAL itself) reached it; nothing above it is
# achievable by a valid placement. Unlike the other problems' targets, this one is not
# a bar to beat: reaching it is the whole task.
OPTIMUM = 26.255971749535238
TARGET = OPTIMUM


@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int


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


def calculate_kvcache_pressure(placement_data: dict[int, list[Model]]) -> float:
    """
    Calculate the KVCache pressure
    """
    max_kvpr = MIN_INT
    for gpu_id, models in placement_data.items():
        total_model_size = sum(model.model_size for model in models)
        total_weighted_req_rate = sum(model.req_rate / model.slo for model in models)
        if GPU_MEM_SIZE - total_model_size > 0:
            kvpr = total_weighted_req_rate / (GPU_MEM_SIZE - total_model_size)
        else:
            kvpr = 1000000
        max_kvpr = max(max_kvpr, kvpr)

    return max_kvpr


def generate_test_gpu_models(num_tests=50):
    """
    Generate multiple test signals with different characteristics
    """
    test_cases = []
    np.random.seed(42)

    for i in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        gpu_models = []
        for j in range(gpu_num*2):
            model_size = np.random.randint(10, 30)
            req_rate = np.random.randint(1, 10)
            slo = np.random.randint(5, 10)
            gpu_models.append(Model(model_name=f"model_{j}", model_size=model_size, req_rate=req_rate, slo=slo, cur_gpu_id=j))

        test_cases.append((gpu_num, gpu_models))

    return test_cases


def _run_placements(solution_dir, timeout):
    """Run the candidate on every case in a FRESH subprocess and return, per case, only
    the placement it produced as gpu_id -> [model_name] (or a failure string). The
    candidate never runs in THIS scoring process, so it cannot patch the metric or print
    its own score; the subprocess regenerates the cases with the bundle's own generator
    and hands back just the names, which are rebuilt and scored here. Model names are
    unique per case, so the parent maps each name back to the authoritative Model (the
    candidate cannot smuggle in fake sizes). Returns (status dict, wall seconds)."""
    code = (
        "import sys, json\n"
        "import time as _t\n"
        "sys.path.insert(0, '.')\n"
        f"sys.path.insert(0, {str(_HERE)!r})\n"
        "from evaluator import generate_test_gpu_models, run_with_timeout, NUM_CASES, PER_CASE_TIMEOUT\n"
        "try:\n"
        "    import solution as program\n"
        "except Exception as e:\n"
        "    print(json.dumps({'import_error': str(e)})); sys.exit(0)\n"
        "if not hasattr(program, 'compute_model_placement'):\n"
        "    print(json.dumps({'missing': True})); sys.exit(0)\n"
        "out = []\n"
        "for i, (gpu_num, models) in enumerate(generate_test_gpu_models(NUM_CASES)):\n"
        "    rec = {}\n"
        "    try:\n"
        "        st = _t.perf_counter()\n"
        "        result = run_with_timeout(program.compute_model_placement,\n"
        "                 kwargs={'gpu_num': gpu_num, 'models': models},\n"
        "                 timeout_seconds=PER_CASE_TIMEOUT)\n"
        "        rec['time'] = _t.perf_counter() - st\n"
        "        if not isinstance(result, dict):\n"
        "            rec['fail'] = f\"returned {type(result).__name__}, not a dict\"\n"
        "        else:\n"
        "            assign = {}\n"
        "            for gid, placed in result.items():\n"
        "                if not isinstance(placed, (list, tuple)):\n"
        "                    raise ValueError(f'GPU {gid}: not a list of models')\n"
        "                assign[str(gid)] = [str(getattr(m, 'model_name')) for m in placed]\n"
        "            rec['assign'] = assign\n"
        "    except TimeoutError:\n"
        "        rec['fail'] = f'ran past its {PER_CASE_TIMEOUT:.0f}s budget'\n"
        "    except Exception as e:\n"
        "        rec['fail'] = f'{type(e).__name__}: {e}'\n"
        "    out.append(rec)\n"
        "print(json.dumps({'cases': out}))\n"
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=timeout, cwd=solution_dir)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"compute_model_placement did not finish the {NUM_CASES} cases within "
            f"{timeout:.0f}s and was killed.") from None
    wall = time.perf_counter() - t0
    if proc.stderr:
        sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(f"the placement subprocess exited {proc.returncode}: {detail or 'no output'}")
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line), wall
    raise RuntimeError("the placement subprocess produced no status line")


def _build_placement(assign: dict, gpu_num: int, models: list):
    """Rebuild the placement from the subprocess's gpu_id -> [model_name] mapping, using
    THIS process's authoritative Model objects. Returns (placement, None) if every model
    is placed exactly once on a real GPU, else (None, reason). Same guarantees as the old
    identity check, but on names the candidate cannot forge into a different model."""
    if not isinstance(assign, dict):
        return None, f"placement is {type(assign).__name__}, not a dict"
    by_name = {m.model_name: m for m in models}
    placement: dict[int, list] = {}
    seen: dict[str, int] = {}
    for gid_str, names in assign.items():
        try:
            gid = int(gid_str)
        except (TypeError, ValueError):
            return None, f"GPU id {gid_str!r} is not an integer"
        if not (0 <= gid < gpu_num):
            return None, f"GPU id {gid} is not one of 0..{gpu_num - 1}"
        placed = []
        for name in names:
            if name not in by_name:
                return None, f"placed a model {name!r} that is not one of the given models"
            seen[name] = seen.get(name, 0) + 1
            placed.append(by_name[name])
        placement[gid] = placed
    for name, m in by_name.items():
        n = seen.get(name, 0)
        if n == 0:
            return None, f"{name} was not placed on any GPU"
        if n > 1:
            return None, f"{name} was placed {n} times"
    return placement, None


def evaluate(solution_path):
    """Score one candidate on all 50 cases. Returns the platform's metrics dict."""
    t0 = time.perf_counter()
    try:
        status, _child_wall = _run_placements(Path(solution_path).parent, SOLVE_TIMEOUT)
        if status.get("import_error"):
            return {"score": 0.0, "stage": "error", "error": status["import_error"][:300],
                    "time": time.perf_counter() - t0}
        if status.get("missing"):
            return {"score": 0.0, "stage": "invalid",
                    "error": "solution.py defines no compute_model_placement function",
                    "time": time.perf_counter() - t0}

        test_gpu_models = generate_test_gpu_models(NUM_CASES)
        records = status.get("cases", [])

        all_kvpr = []
        all_metrics = []
        successful_runs = 0
        failures = []

        for i, (gpu_num, gpu_models) in enumerate(test_gpu_models):
            try:
                rec = records[i] if i < len(records) else {"fail": "no result returned"}
                if "fail" in rec:
                    raise ValueError(rec["fail"])

                # Rebuild and verify the placement HERE, from the authoritative models.
                placement, err = _build_placement(rec.get("assign", {}), gpu_num, gpu_models)
                if err is not None:
                    raise ValueError(f"invalid placement: {err}")

                max_kvpr = calculate_kvcache_pressure(placement)

                metrics = {
                    'max_kvpr': safe_float(max_kvpr),
                    'execution_time': safe_float(rec.get("time", 0.0)),
                }

                all_kvpr.append(safe_float(max_kvpr))
                all_metrics.append(metrics)
                successful_runs += 1

            except Exception as e:  # noqa: BLE001 — a bad candidate scores low, never crashes us
                failures.append(f"case {i}: {e}")
                all_kvpr.append(1000000.0)
                continue

        if successful_runs == 0:
            return {"score": 0.0, "stage": "invalid",
                    "error": "no case was placed successfully. "
                             + "; ".join(failures)[:400],
                    "time": time.perf_counter() - t0}

        mean_max_kvpr = float(np.mean(all_kvpr))
        avg_inv_kvpr = 1.0 / mean_max_kvpr if mean_max_kvpr != 0 else 0.0
        avg_execution_time = np.mean([m['execution_time'] for m in all_metrics])
        success_rate = successful_runs / len(test_gpu_models)
        combined = safe_float(avg_inv_kvpr) + safe_float(success_rate)

        return {
            # Maximised objective with a proven optimum: the fraction of it reached.
            "score": safe_float(combined) / OPTIMUM,
            "stage": "full",
            "objective": safe_float(combined),
            "objective_name": "combined_score",
            "objective_direction": "max",
            "objective_target": TARGET,
            "combined_score": safe_float(combined),
            "avg_inv_kvpr": safe_float(avg_inv_kvpr),
            "mean_max_kvpr": safe_float(mean_max_kvpr),
            "success_rate": safe_float(success_rate),
            "cases_scored": successful_runs,
            "cases_total": len(test_gpu_models),
            "target": TARGET,
            "solve_time": safe_float(avg_execution_time) * len(test_gpu_models),
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
