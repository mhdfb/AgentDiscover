"""AHC039 adapter around the public inputs and tester shipped by TTT-Discover."""
import ast
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Upstream fans all 150 cases out to a Ray/HPC cluster. This evaluator owns only the
# two CPUs declared in problem.toml, so two workers keep one single-threaded candidate
# on each CPU. At the 2.5s tolerated wall limit, 75 waves take at most 187.5s, safely
# inside the source's 530s whole-evaluation timeout after compilation and judging.
PROBLEM_ID = "ahc039"
NUM_WORKERS = 2
CONTEST_TIME_LIMIT = 2.0
TIME_LIMIT_TOLERANCE = 0.5
CASE_TIMEOUT = CONTEST_TIME_LIMIT + TIME_LIMIT_TOLERANCE
CASE_MEMORY_BYTES = 1024**3
SOURCE_TARGET = 5000.0
SCORE_SCALE = 1500.0

PROFILE_PATTERN = re.compile(
    r"elapsed=(?P<elapsed>[0-9.]+) user=(?P<user>[0-9.]+) "
    r"system=(?P<system>[0-9.]+) max_rss_kb=(?P<rss>\d+) exit=(?P<exit>\d+)"
)


def _extract_source(solution_path: Path) -> str:
    tree = ast.parse(solution_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "SOURCE" for target in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str) and "int main" in value.lower():
                    return value
    raise ValueError("solution.py must assign a C++ program containing int main to SOURCE")


def _compile(source: str, work: Path) -> Path:
    source_file = work / "Main.cpp"
    binary = work / "a.out"
    source_file.write_text(source)
    command = [
        "g++-12", "-std=gnu++20", "-O2", "-DONLINE_JUDGE", "-DATCODER",
        "-Wall", "-Wextra", "-mtune=native", "-march=native",
        "-fconstexpr-depth=2147483647", "-fconstexpr-loop-limit=2147483647",
        "-fconstexpr-ops-limit=2147483647", "-I/opt/ac-library",
        "-I/opt/boost/gcc/include", "-L/opt/boost/gcc/lib", "-o", str(binary),
        str(source_file), "-lgmpxx", "-lgmp", "-I/usr/include/eigen3",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError("C++ compilation failed: " + proc.stderr[-1000:])
    return binary


def _case_result(index: int, judge_result: str, *, score: int = 0,
                 message: str = "", elapsed: float = 0.0, cpu: float = 0.0,
                 memory_bytes: int = 0) -> dict:
    return {
        "case": index,
        "score": score,
        "judge_result": judge_result,
        "message": message[-1000:],
        "execution_time": elapsed,
        "cpu_time": cpu,
        "memory_bytes": memory_bytes,
    }


def _score_case(binary: Path, tester: Path, index: int, input_text: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"ahc039-{index:03d}-") as temp:
        case_dir = Path(temp)
        input_file = case_dir / "input.txt"
        output_file = case_dir / "output.txt"
        profile_file = case_dir / "profile.txt"
        input_file.write_text(input_text)
        command = [
            "prlimit", f"--as={CASE_MEMORY_BYTES}", "--",
            "/usr/bin/time", "-o", str(profile_file),
            "-f", "elapsed=%e user=%U system=%S max_rss_kb=%M exit=%x",
            str(binary),
        ]
        try:
            with input_file.open("rb") as stdin, output_file.open("wb") as stdout:
                run = subprocess.run(command, stdin=stdin, stdout=stdout,
                                     stderr=subprocess.PIPE, timeout=CASE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return _case_result(
                index, "TIME_LIMIT_EXCEEDED",
                message=f"elapsed time exceeded {CASE_TIMEOUT:.1f}s "
                        f"({CONTEST_TIME_LIMIT:.1f}s + {TIME_LIMIT_TOLERANCE:.1f}s tolerance)",
                elapsed=CASE_TIMEOUT,
            )

        profile_text = profile_file.read_text(errors="replace") if profile_file.exists() else ""
        profile_match = PROFILE_PATTERN.search(profile_text)
        elapsed = float(profile_match.group("elapsed")) if profile_match else 0.0
        cpu = (float(profile_match.group("user")) + float(profile_match.group("system"))
               if profile_match else 0.0)
        memory_bytes = int(profile_match.group("rss")) * 1024 if profile_match else 0
        stderr = run.stderr.decode(errors="replace").strip()

        if run.returncode != 0:
            memory_markers = ("bad_alloc", "cannot allocate memory", "memoryerror")
            result = ("MEMORY_LIMIT_EXCEEDED"
                      if run.returncode in (-9, 9, 137)
                      or any(marker in stderr.lower() for marker in memory_markers)
                      else "RUNTIME_ERROR")
            return _case_result(index, result, message=stderr or f"exit code {run.returncode}",
                                elapsed=elapsed, cpu=cpu, memory_bytes=memory_bytes)
        if not profile_match:
            return _case_result(index, "INTERNAL_ERROR", message="GNU time produced no profile")
        if cpu > CONTEST_TIME_LIMIT or elapsed > CASE_TIMEOUT:
            return _case_result(
                index, "TIME_LIMIT_EXCEEDED",
                message=f"CPU {cpu:.3f}s, elapsed {elapsed:.3f}s",
                elapsed=elapsed, cpu=cpu, memory_bytes=memory_bytes,
            )
        if memory_bytes > CASE_MEMORY_BYTES:
            return _case_result(
                index, "MEMORY_LIMIT_EXCEEDED",
                message=f"peak RSS {memory_bytes} exceeds {CASE_MEMORY_BYTES}",
                elapsed=elapsed, cpu=cpu, memory_bytes=memory_bytes,
            )
        judged = subprocess.run([str(tester), str(input_file), str(output_file)],
                                capture_output=True, text=True, timeout=10)
        message = (judged.stdout or "") + "\n" + (judged.stderr or "")
        match = re.search(r"Score\s*=\s*(\d+)", message)
        if judged.returncode != 0 or match is None:
            return _case_result(index, "WRONG_ANSWER", message=message.strip(),
                                elapsed=elapsed, cpu=cpu, memory_bytes=memory_bytes)
        return _case_result(index, "ACCEPTED", score=int(match.group(1)),
                            message=message.strip(), elapsed=elapsed, cpu=cpu,
                            memory_bytes=memory_bytes)


def evaluate(solution_path: Path) -> dict:
    started = time.perf_counter()
    data_root = Path(os.environ.get("AGENTDISCOVER_DATA_DIR", "")) / "cache"
    tester = data_root / "tester_binaries" / "ahc039_tester"
    input_cache = data_root / "public_inputs_150" / "ahc039_693bb5c4c2a78b4b.json"
    try:
        source = _extract_source(solution_path)
        payload = json.loads(input_cache.read_text())
        if payload.get("problem_id") != PROBLEM_ID or payload.get("lite_version") is not False:
            raise ValueError("AHC cache metadata does not match the source's full public test set")
        inputs = payload["inputs"]
        if not shutil.which("prlimit") or not Path("/usr/bin/time").is_file():
            raise RuntimeError("evaluator image must provide prlimit and /usr/bin/time")
        with tempfile.TemporaryDirectory(prefix="ahc039-build-") as temp:
            binary = _compile(source, Path(temp))
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
                results = list(pool.map(
                    lambda item: _score_case(binary, tester, item[0], item[1]),
                    enumerate(inputs),
                ))
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(exc)[:500],
                "time": time.perf_counter() - started}

    failures = [result for result in results if result["judge_result"] != "ACCEPTED"]
    total = sum(result["score"] for result in results)
    raw_score = total / len(results)
    judge_results: dict[str, int] = {}
    for result in results:
        status = result["judge_result"]
        judge_results[status] = judge_results.get(status, 0) + 1
    slowest = max(results, key=lambda result: result["execution_time"])
    highest_cpu = max(results, key=lambda result: result["cpu_time"])
    highest_memory = max(results, key=lambda result: result["memory_bytes"])
    metrics = {
        # The upstream search wrapper exposes a partial mean even when correctness is
        # zero, but the embedded contest statement says any WA/TLE makes the submission
        # score zero. Honor the contest rule for selection while retaining the partial
        # aggregate below as diagnostics for improving a failed candidate.
        "score": 0.0 if failures else raw_score / SCORE_SCALE,
        "stage": "full" if not failures else "invalid",
        "objective": raw_score,
        "objective_name": "mean public-case score",
        "objective_direction": "max",
        "objective_target": SOURCE_TARGET,
        "raw_score": raw_score,
        "total_score": 0 if failures else total,
        "correctness": 1.0 if not failures else 0.0,
        "accepted_cases": len(results) - len(failures),
        "total_cases": len(results),
        "judge_results": judge_results,
        "max_execution_time": slowest["execution_time"],
        "max_cpu_time": highest_cpu["cpu_time"],
        "max_memory_bytes": highest_memory["memory_bytes"],
        "time": time.perf_counter() - started,
    }
    if failures:
        metrics["partial_total_score"] = total
        metrics["first_failure"] = failures[0]
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))
