"""TriMul — how a candidate is scored. This file is readable at `/resources/evaluator.py`
and runnable there too (`python3 /resources/evaluator.py solution.py` prints the same JSON
line the platform gets); it is the authority on the contract PROBLEM.md describes.

The scoring path is GPUMode's own. The five files in `kernelbot/` are the TriMul
competition's task definition, unchanged — `eval.py`, `reference.py`, `utils.py`,
`task.py`, `task.yml` — and this file stands in for their server-side runner
(libkernelbot/run_eval.py: `run_pytorch_script` + `run_evaluation` in "leaderboard"
mode), step for step:

  1. the submission is written next to the task files as `submission.py` and run once on
     its own (`python3 submission.py`, 120 s), which warms any compile cache;
  2. `python3 eval.py test <cases>` — the 18 correctness cases of task.yml, each checked
     against the fp32 PyTorch reference with rtol = atol = 2e-2. One failure scores 0;
  3. `python3 eval.py leaderboard <cases>` — the 7 benchmark cases of task.yml. Each is
     timed with CUDA events, re-generated and re-checked on every repeat, until the
     standard error of the mean is under 0.1 % or 30 s of kernel time has accumulated
     (at most 100 repeats, 120 s wall clock per case);
  4. the score is the geometric mean of the 7 mean runtimes (task.yml: `ranking_by:
     geom`), in microseconds. Lower is better.

The harness reports over a pipe (POPCORN_FD) exactly as on GPUMode's machines, and the
phase clocks are task.yml's: test_timeout and ranked_timeout, 1200 s each. No secret
seed is set, so the cases run with the seeds task.yml lists.

Two rules the TTT-Discover reward function adds are kept verbatim: the program must
contain `@triton.jit`, and must not contain the word `identity`.

Fitness is TARGET_US / score_us — 1.0 when the kernel is as fast as TTT-Discover's
published kernel on this GPU type, above 1 past it. The objective is the score itself.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
KERNELBOT = _HERE / "kernelbot"
TASK_FILES = ("eval.py", "reference.py", "utils.py", "task.py")

# The bar to BEAT on each GPU type: the best published result for that type —
# TTT-Discover's kernel (their Table "TriMul": leaderboard figures for A100/H100, their
# re-timing for B200/MI300X). Lower is better. The fastest human entries, for reference:
# 4,531.5 / 1,371.1 / 1,038.9 / 2,515.8.
GPU_TARGET_US = {
    "A100": 2198.2,
    "H100": 1161.2,
    "B200": 914.2,
    "MI300": 1555.7,
}
# A type with no leaderboard of its own (H200, say) is held to the H100 bar.
DEFAULT_TARGET_US = 1161.2

# libkernelbot.consts.Timeout.COMPILE: the once-through run of submission.py.
COMPILE_TIMEOUT = 120


def _gpu_name() -> str:
    """The device's name, read in a child process so this one never holds a CUDA
    context (and its memory) on the GPU while the harness is timing on it."""
    try:
        out = subprocess.run([sys.executable, "-c",
                              "import torch; print(torch.cuda.get_device_name(0))"],
                             capture_output=True, text=True, timeout=120)
        name = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        return name or f"unknown (exit {out.returncode}: {out.stderr.strip()[-200:]})"
    except Exception as e:  # noqa: BLE001 — reported, not fatal: the harness decides
        return f"unknown ({type(e).__name__})"


def target_for(gpu: str) -> float:
    for key, us in GPU_TARGET_US.items():
        if key in gpu:
            return us
    return DEFAULT_TARGET_US


def _cases(entries: list[dict]) -> str:
    """libkernelbot.run_eval.build_test_string: one `k: v; k: v` line per case."""
    return "".join("; ".join(f"{k}: {v}" for k, v in e.items()) + "\n" for e in entries)


def _run(argv: list[str], cwd: Path, env: dict, timeout: int) -> dict:
    """libkernelbot.run_eval.run_program: run the harness with a result pipe on
    POPCORN_FD and return what it wrote there, plus the process outcome."""
    pipe_r, pipe_w = os.pipe()
    env = {**env, "POPCORN_FD": str(pipe_w)}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, text=True,
                              pass_fds=[pipe_w], timeout=timeout)
    except subprocess.TimeoutExpired as e:
        os.close(pipe_w)
        os.close(pipe_r)
        def _tail(x):
            return (x.decode(errors="replace") if isinstance(x, bytes) else (x or ""))[-4000:]
        return {"timed_out": True, "returncode": None, "duration": timeout, "result": {},
                "stdout": _tail(e.stdout), "stderr": _tail(e.stderr)}
    os.close(pipe_w)
    with os.fdopen(pipe_r) as fh:
        report = fh.read()
    result = {}
    for line in report.splitlines():
        key, _, value = line.partition(":")
        if key != "" or value != "":
            result[key.strip()] = value.strip()
    return {"timed_out": False, "returncode": proc.returncode,
            "duration": time.perf_counter() - t0, "result": result,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def _fail(stage: str, error: str, t0: float, **extra) -> dict:
    out = {"score": 0.0, "stage": stage, "error": error, "time": time.perf_counter() - t0}
    out.update(extra)
    return out


def _first_failure(r: dict) -> tuple[int, str]:
    """(how many cases failed, the first failure's spec and message) from a harness
    report — test or benchmark entries, whichever the phase wrote."""
    failed = sorted(k for k, v in r.items() if k.endswith(".status") and v == "fail")
    if not failed:
        return 0, "no detail"
    stem = failed[0][: -len(".status")]
    return len(failed), f"{r.get(stem + '.spec', stem)}: {r.get(stem + '.error', '')}"


def evaluate(solution_path: Path, run_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    code = solution_path.read_text()

    # TTT-Discover's two textual rules (examples/gpu_mode/env.py, get_reward), verbatim.
    if "@triton.jit" not in code:
        return _fail("invalid", "Code must contain @triton.jit.", t0)
    if "identity" in code:
        return _fail("invalid", "Identity kernel is not allowed.", t0)

    task = yaml.safe_load((KERNELBOT / "task.yml").read_text())
    gpu = _gpu_name()
    target = target_for(gpu)

    made_tmp = run_dir is None
    work = Path(tempfile.mkdtemp(prefix="trimul-")) if made_tmp else run_dir
    work.mkdir(parents=True, exist_ok=True)
    try:
        for name in TASK_FILES:
            shutil.copy(KERNELBOT / name, work / name)
        (work / "submission.py").write_text(code)
        tests = work / "tests.txt"
        tests.write_text(_cases(task["tests"]))
        benchmarks = work / "benchmarks.txt"
        benchmarks.write_text(_cases(task["benchmarks"]))

        # The harness's processes: the platform's own variables stay out, the compile
        # caches go under the run directory (the sandbox's /tmp is small).
        env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTDISCOVER_")}
        env.update({
            "HOME": str(work),
            "TRITON_CACHE_DIR": str(work / ".triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(work / ".inductor"),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        py = sys.executable

        # 1. run_pytorch_script's "compile" step: submission.py on its own, once.
        comp = _run([py, "submission.py"], work, {**env, "POPCORN_SEED": "1"}, COMPILE_TIMEOUT)
        if comp["timed_out"]:
            return _fail("timeout", f"submission.py did not finish its own import run in "
                                    f"{COMPILE_TIMEOUT} s", t0, gpu=gpu)
        if comp["returncode"] != 0:
            return _fail("error", "submission.py fails on import: "
                         + (comp["stderr"] or comp["stdout"]).strip()[-1500:], t0, gpu=gpu)

        # 2. correctness: every test case against the reference.
        test = _run([py, "eval.py", "test", str(tests)], work, env, int(task["test_timeout"]))
        if test["timed_out"]:
            return _fail("timeout", f"correctness tests exceeded test_timeout "
                                    f"({task['test_timeout']} s)", t0, gpu=gpu)
        r = test["result"]
        if test["returncode"] not in (0, 112) or not r:
            return _fail("error", "the test harness crashed: "
                         + (test["stderr"] or test["stdout"]).strip()[-1500:], t0, gpu=gpu)
        if r.get("check") != "pass":
            n_failed, detail = _first_failure(r)
            return _fail("invalid", f"Failed to pass test cases ({n_failed} of "
                                    f"{r.get('test-count')}); first: {detail}"[:1500],
                         t0, gpu=gpu, tests_failed=n_failed)

        # 3. the ranked run: every benchmark case, re-checked and timed.
        lb = _run([py, "eval.py", "leaderboard", str(benchmarks)], work, env,
                  int(task["ranked_timeout"]))
        if lb["timed_out"]:
            return _fail("timeout", f"benchmarks exceeded ranked_timeout "
                                    f"({task['ranked_timeout']} s)", t0, gpu=gpu)
        r = lb["result"]
        if lb["returncode"] not in (0, 112) or not r:
            return _fail("error", "the benchmark harness crashed: "
                         + (lb["stderr"] or lb["stdout"]).strip()[-1500:], t0, gpu=gpu)
        if r.get("check") != "pass":
            _, detail = _first_failure(r)
            return _fail("invalid", f"benchmark failed its re-check: {detail}"[:1500], t0, gpu=gpu)

        # 4. libkernelbot.submission.compute_score, ranking_by geom.
        n = int(r["benchmark-count"])
        means_ns = [float(r[f"benchmark.{i}.mean"]) for i in range(n)]
        if task.get("ranking_by", "last") == "geom":
            score_s = math.pow(math.prod(m / 1e9 for m in means_ns), 1.0 / n)
        elif task.get("ranking_by") == "mean":
            score_s = sum(means_ns) / n / 1e9
        else:
            score_s = means_ns[-1] / 1e9
        score_us = score_s * 1e6
    finally:
        if made_tmp:
            shutil.rmtree(work, ignore_errors=True)

    return {
        "score": float(target / score_us),
        "stage": "full",
        "objective": round(score_us, 3),
        "objective_name": "runtime_us",
        "objective_direction": "min",
        "objective_target": target,
        "runtime_us": round(score_us, 3),
        "benchmark_means_us": [round(m / 1e3, 3) for m in means_ns],
        "benchmark_runs": [int(r.get(f"benchmark.{i}.runs", 0)) for i in range(n)],
        "tests_passed": int(test["result"].get("test-count", 0)),
        "gpu": gpu,
        "target_us": target,
        "test_time": round(test["duration"], 1),
        "benchmark_time": round(lb["duration"], 1),
        "time": time.perf_counter() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("solution")
    p.add_argument("--run-dir", default=None,
                   help="where the harness runs (default: a fresh temporary directory)")
    a = p.parse_args()
    print(json.dumps(evaluate(Path(a.solution).resolve(),
                              Path(a.run_dir).resolve() if a.run_dir else None)))


if __name__ == "__main__":
    main()
