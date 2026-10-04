"""Standalone ALE-Bench search adapter for ahc006.

Source summary: SkyDiscover benchmarks/ale_bench/ale-bench-lite-problems/
ahc006/evaluator.py selects 50 generated public seeds, 13 workers, C++20,
and returns overall_absolute_score * optim_factor / 50, where optim_factor is
+1 for maximize and -1 for minimize. This adapter keeps those score semantics
and runs the unmodified ALE-Bench Rust generator/tester directly because
AgentDiscover's evaluation sandbox cannot call ALE-Bench's nested Docker session.
The ALE-Bench dataset is pinned by fetch.sh; C++ compilation follows its
202301 judge profile. Reactive tasks use the upstream tester as the process
that launches the candidate. PROBLEM.md describes the agent-facing task.
"""
from __future__ import annotations

import ast
import concurrent.futures
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROBLEM_ID = "ahc006"
PROBLEM_TYPE = "batch"
SCORE_TYPE = "maximize"
TIME_LIMIT = 2.0
# Pinned ALE archive standings_scores.csv: rank 1 scored 2,220,425 on 100
# contest cases. Preconfigured target, rounded above the winner's mean.
OBJECTIVE_TARGET = 23000
NUM_PUBLIC_CASES = 50
NUM_WORKERS = 13
MEMORY_LIMIT_BYTES = 1024**3
# ALE-Bench allows the timeout wrapper a margin but rejects a candidate when
# GNU time reports max(elapsed, CPU) above the exact 2/3-second task limit.
CASE_TIMEOUT = math.ceil(TIME_LIMIT + 0.1) + 0.2
PROFILE = re.compile(
    r"elapsed=(?P<elapsed>[0-9.]+) user=(?P<user>[0-9.]+) "
    r"system=(?P<system>[0-9.]+) max_rss_kb=(?P<rss>\d+) exit=(?P<exit>\d+)"
)
SCORE_PATTERN = re.compile(r"Score\s*=\s*(-?\d+)")


def _source(solution_path: Path) -> str:
    tree = ast.parse(solution_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SOURCE" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("solution.py must assign C++20 text to SOURCE")


def _compile(source: str, work: Path) -> Path:
    main = work / "Main.cpp"
    binary = work / "a.out"
    main.write_text(source)
    # ALE-Bench judge_command_profiles.py, cpp20/202301 profile.
    command = [
        "g++-12", "-std=gnu++20", "-O2", "-DONLINE_JUDGE", "-DATCODER",
        "-Wall", "-Wextra", "-mtune=native", "-march=native",
        "-fconstexpr-depth=2147483647", "-fconstexpr-loop-limit=2147483647",
        "-fconstexpr-ops-limit=2147483647", "-I/opt/ac-library",
        "-I/opt/boost/gcc/include", "-L/opt/boost/gcc/lib", "-o",
        str(binary), str(main), "-lgmpxx", "-lgmp", "-I/usr/include/eigen3",
    ]
    built = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if built.returncode:
        raise ValueError("C++20 compilation failed: " + built.stderr[-1800:])
    return binary


def _case(index: int, input_path: Path, binary: Path, tester: Path, work: Path) -> dict:
    output_path = work / f"{index:04d}.out"
    profile_path = work / f"{index:04d}.profile"
    # Direct Rust tester invocation mirrors ALE-Bench's batch or reactive command.
    candidate = ["/usr/bin/time", "-f",
                 "elapsed=%e user=%U system=%S max_rss_kb=%M exit=%x",
                 "-o", str(profile_path), "prlimit",
                 f"--cpu={math.ceil(TIME_LIMIT + 0.1)}",
                 f"--as={MEMORY_LIMIT_BYTES}", str(binary)]
    command = ([str(tester), *candidate] if PROBLEM_TYPE == "reactive"
               else candidate)
    try:
        with input_path.open("rb") as input_file, output_path.open("wb") as output_file:
            run = subprocess.run(command, stdin=input_file, stdout=output_file,
                                 stderr=subprocess.PIPE, timeout=CASE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"index": index, "judge_result": "TIME_LIMIT_EXCEEDED", "score": 0,
                "message": f"exceeded {CASE_TIMEOUT}s judge wall limit"}
    stderr = run.stderr.decode("utf-8", "replace")[-1800:]
    match = PROFILE.search(profile_path.read_text()) if profile_path.exists() else None
    elapsed = float(match["elapsed"]) if match else 0.0
    cpu = float(match["user"]) + float(match["system"]) if match else 0.0
    memory = int(match["rss"]) * 1024 if match else 0
    execution_time = max(elapsed, cpu)
    result = {"index": index, "score": 0, "elapsed": elapsed,
              "execution_time": execution_time, "cpu": cpu,
              "memory_bytes": memory, "message": stderr.strip()}
    if not match:
        result["judge_result"] = "RUNTIME_ERROR" if run.returncode else "INTERNAL_ERROR"
        return result
    if execution_time > TIME_LIMIT:
        result["judge_result"] = "TIME_LIMIT_EXCEEDED"
        return result
    if memory > MEMORY_LIMIT_BYTES:
        result["judge_result"] = "MEMORY_LIMIT_EXCEEDED"
        return result
    if run.returncode:
        result["judge_result"] = "RUNTIME_ERROR"
        return result
    if PROBLEM_TYPE == "batch":
        judged = subprocess.run([str(tester), str(input_path), str(output_path)],
                                capture_output=True, text=True, timeout=10)
        result["message"] = (judged.stdout + judged.stderr)[-1800:].strip()
        scored = SCORE_PATTERN.search(result["message"])
        if judged.returncode or not scored:
            result["judge_result"] = "WRONG_ANSWER"
            return result
    else:
        scored = SCORE_PATTERN.search(stderr)
        if not scored:
            result["judge_result"] = "WRONG_ANSWER"
            return result
    result["score"] = int(scored.group(1))
    result["judge_result"] = "ACCEPTED"
    return result


def evaluate(solution_path: Path) -> dict:
    started = time.perf_counter()
    data_dir = Path(os.environ.get("AGENTDISCOVER_DATA_DIR", ""))
    tester = data_dir / "bin" / "tester"
    inputs = [data_dir / "in" / f"{index:04d}.txt"
              for index in range(NUM_PUBLIC_CASES)]
    try:
        if not tester.is_file() or any(not path.is_file() for path in inputs):
            raise FileNotFoundError("ALE-Bench tester or public inputs missing; run fetch.sh")
        source = _source(solution_path)
        with tempfile.TemporaryDirectory(prefix=f"{PROBLEM_ID}-eval-") as temp:
            work = Path(temp)
            binary = _compile(source, work)
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
                cases = list(pool.map(
                    lambda item: _case(item[0], item[1], binary, tester, work),
                    enumerate(inputs),
                ))
    except ValueError as exc:
        # ALE-Bench case_eval returns zero aggregate for compilation errors.
        penalty = -sys.maxsize - 1 if SCORE_TYPE == "minimize" else 0.0
        return {"score": penalty, "combined_score": penalty, "overall_score": 0,
                "stage": "invalid", "judge_result": "COMPILATION_ERROR",
                "objective_target": OBJECTIVE_TARGET,
                "error": str(exc)[:1800], "standard_error": str(exc)[:1800],
                "message": str(exc)[:1800], "time": time.perf_counter() - started}
    except Exception as exc:
        return {"score": 0.0, "stage": "error", "objective_target": OBJECTIVE_TARGET,
                "error": str(exc)[:1800],
                "time": time.perf_counter() - started}

    accepted = [case for case in cases if case["judge_result"] == "ACCEPTED"]
    failures = [case for case in cases if case["judge_result"] != "ACCEPTED"]
    statuses: dict[str, int] = {}
    for case in cases:
        status = case["judge_result"]
        statuses[status] = statuses.get(status, 0) + 1
    # Match ALE-Bench Result.overall_judge_result precedence and its zero
    # overall_absolute_score for INTERNAL_ERROR; case_eval permits partial
    # accepted-case sums for other non-AC outcomes.
    precedence = ("INTERNAL_ERROR", "WRONG_ANSWER", "RUNTIME_ERROR",
                  "TIME_LIMIT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED",
                  "OUTPUT_LIMIT_EXCEEDED", "COMPILATION_ERROR")
    overall_status = next((status for status in precedence if status in statuses),
                          "ACCEPTED")
    accepted_total = sum(case["score"] for case in accepted)
    overall_score = 0 if overall_status == "INTERNAL_ERROR" else accepted_total
    mean = overall_score / NUM_PUBLIC_CASES
    signed_mean = mean if SCORE_TYPE == "maximize" else -mean
    # AgentDiscover consumes score; combined_score keeps SkyDiscover's field name.
    # SkyDiscover preserves accepted-case partial credit for maximize tasks. Its
    # ahc025 evaluator sends any invalid minimize candidate below all valid ones.
    if failures and SCORE_TYPE == "minimize":
        signed_mean = -sys.maxsize - 1
    max_elapsed = max(case.get("execution_time", 0.0) for case in cases)
    max_memory = max(case.get("memory_bytes", 0) for case in cases)
    feedback = failures[0] if failures else cases[0]
    metrics = {
        "score": signed_mean, "combined_score": signed_mean,
        "stage": "full" if not failures else "invalid",
        "judge_result": overall_status, "overall_score": overall_score,
        "objective": mean, "objective_name": "mean absolute public-case score",
        "objective_direction": "min" if SCORE_TYPE == "minimize" else "max",
        "objective_target": OBJECTIVE_TARGET,
        "raw_score": mean, "total_score": overall_score,
        "correctness": 1.0 if not failures else 0.0,
        "accepted_cases": len(accepted), "total_cases": NUM_PUBLIC_CASES,
        "judge_results": statuses,
        "max_execution_time": max_elapsed, "max_execution_time_sec": max_elapsed,
        "max_cpu_time": max(case.get("cpu", 0.0) for case in cases),
        "max_memory_bytes": max_memory,
        "max_memory_usage_mib": max_memory // 1024 // 1024,
        "standard_error": feedback.get("message", ""),
        "message": feedback.get("message", ""),
        "time": time.perf_counter() - started,
    }
    if failures:
        metrics["first_failure"] = failures[0]
        metrics["partial_total_score"] = accepted_total
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))
