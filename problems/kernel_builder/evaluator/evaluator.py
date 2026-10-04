"""Kernel builder — how a candidate is scored. This file is readable at
`/resources/evaluator.py`; it is the authority on the contract PROBLEM.md describes.

The workload is fixed (`REAL_PARAMS`): forest_height=10, rounds=16, batch_size=256. Your
`build_kernel` is called ONCE, with those four sizes and nothing else, and the single
instruction list it produces is then run against eight freshly generated random trees and
batches (`Tree.generate` / `Input.generate`, unseeded). Its output must match
`reference_kernel2` on every one of the eight; the cycle count reported is the last
trial's, which is deterministic for a given instruction list. The objective is that cycle
count, minimised, and fitness is `TARGET / cycles` — 1.0 at the target, above 1 past it.

Your process only BUILDS. It is handed the four sizes, returns the instruction list and
the scratch map as JSON, and exits. The simulator, the random inputs, the reference and
the cycle counter live in this parent process, which your code never runs in and whose
inputs it never sees — the kernel is built before any input exists, so there is no way to
write a program that depends on the data it will be given, and no way to reach the
counter that scores you. Forging the JSON line buys nothing: whatever instruction list
arrives is simulated honestly here.

`frozen_problem.py`, the simulator, is not provided. Debug your builder locally against
the instruction forms the seed uses; this evaluator is authoritative.

Failures: an incorrect kernel scores 0 with "Kernel produces incorrect output: Iteration
i: ..."; a builder that raises scores 0 and gets its traceback back; a builder that runs
past its budget scores 0 with "Evaluation timed out". Printing from a successful build
goes to the operator's log, not into your result.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from frozen_problem import (DebugInfo, Input, Machine, N_CORES, Tree,  # noqa: E402
                            build_mem_image, reference_kernel2)

# The naive scalar kernel this task started from, kept only to report a speedup.
BASELINE_CYCLES = 147_734
REAL_PARAMS = {"forest_height": 10, "rounds": 16, "batch_size": 256, "iterations": 8}

# The bar to BEAT, not to meet (the objective is minimised): the best number a previous
# search reached on this workload.
TARGET = 1_103

# The candidate's budget for building the instruction list. The orchestrator passes the
# per-problem figure so the number the agent is told and the number enforced here cannot
# drift; the fallback keeps the bundle runnable standalone.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 14.0))

# The instruction list crosses a pipe as JSON. The seed is 11,910 instructions (~0.4 MB);
# a program a hundred times longer would take more cycles than the naive baseline and
# cannot be a contender, so anything past this is refused rather than parsed.
MAX_WIRE_BYTES = 64 * 1024 * 1024


def _build_in_subprocess(solution_path: Path, sizes: dict, timeout: float) -> dict:
    """Run the candidate's `KernelBuilder.build_kernel(...)` in its own process and
    return {"instrs": [...], "scratch_map": {...}} — or {"error_class": ...}."""
    code = (
        "import json, sys\n"
        "_emit = sys.stdout.write\n"        # bound before the candidate can patch anything
        "_dumps = json.dumps\n"
        "sys.stdout = sys.stderr\n"         # the candidate's own printing stays out of the wire
        f"src = open({str(solution_path)!r}).read()\n"
        "ns = {'__name__': '__main__'}\n"
        "out = None\n"
        "try:\n"
        "    exec(src, ns)\n"
        "    if 'KernelBuilder' not in ns:\n"
        "        out = {'error_class': 'missing'}\n"
        "    else:\n"
        "        kb = ns['KernelBuilder']()\n"
        f"        kb.build_kernel({sizes['forest_height']}, {sizes['n_nodes']}, "
        f"{sizes['batch_size']}, {sizes['rounds']})\n"
        "        dbg = kb.debug_info()\n"
        "        smap = getattr(dbg, 'scratch_map', None) or {}\n"
        "        out = {'instrs': list(kb.instrs),\n"
        "               'scratch_map': {str(k): [str(v[0]), int(v[1])] for k, v in dict(smap).items()}}\n"
        "except BaseException as e:\n"                 # SystemExit and KeyboardInterrupt included
        "    import traceback; traceback.print_exc()\n"
        "    out = {'error_class': type(e).__name__}\n"
        "_emit('\\n' + _dumps(out) + '\\n')\n"          # exactly one JSON line, on its own line
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTDISCOVER_")}
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=timeout, cwd=str(solution_path.parent), env=env)
    except subprocess.TimeoutExpired:
        return {"error_class": "timeout"}
    tail = (out.stderr or "")[-2000:]                 # handed back on failure
    if out.stderr:
        sys.stderr.write(out.stderr[-4000:])          # operator sees more
    if len(out.stdout) > MAX_WIRE_BYTES:
        return {"error_class": "program_too_large", "detail": tail}
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                break
            if "error_class" in parsed:
                parsed["detail"] = tail
            return parsed
    return {"error_class": "exit" if out.returncode else "no_output", "detail": tail}


def _simulate(instrs, scratch_map, params) -> tuple[int, bool]:
    """One correctness trial on a fresh random tree and batch: (cycles, correct)."""
    forest = Tree.generate(params["forest_height"])
    inp = Input.generate(forest, params["batch_size"], params["rounds"])
    mem = build_mem_image(forest, inp)
    machine = Machine(mem, instrs, DebugInfo(scratch_map=scratch_map), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    for ref_mem in reference_kernel2(mem):
        pass
    p = ref_mem[6]
    n = len(inp.values)
    return machine.cycle, machine.mem[p:p + n] == ref_mem[p:p + n]


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()
    params = REAL_PARAMS
    # The sizes the builder is told are fixed by the workload, so the kernel is built
    # once, before any input exists; the same list is then replayed on every trial.
    probe_forest = Tree.generate(params["forest_height"])
    probe_inp = Input.generate(probe_forest, params["batch_size"], params["rounds"])
    sizes = {"forest_height": probe_forest.height, "n_nodes": len(probe_forest.values),
             "batch_size": len(probe_inp.indices), "rounds": params["rounds"]}

    built = _build_in_subprocess(solution_path.resolve(), sizes, SOLVE_TIMEOUT)
    t_build = time.perf_counter() - t0
    if "error_class" in built:
        cls, detail = built["error_class"], str(built.get("detail") or "").strip()
        if cls == "missing":
            return {"score": 0.0, "stage": "invalid", "error": "KernelBuilder class not found in program",
                    "time": t_build}
        if cls == "timeout":
            return {"score": 0.0, "stage": "timeout",
                    "error": f"Evaluation timed out after {SOLVE_TIMEOUT:.0f}s", "time": t_build}
        return {"score": 0.0, "stage": "error",
                "error": f"Evaluation failed: {detail or cls}", "time": t_build}

    try:
        instrs = built["instrs"]
        scratch_map = {int(k): (str(v[0]), int(v[1])) for k, v in (built.get("scratch_map") or {}).items()}
        if not isinstance(instrs, list) or not instrs:
            return {"score": 0.0, "stage": "invalid", "error": "build_kernel produced no instructions",
                    "time": time.perf_counter() - t0}
        cycles = None
        for i in range(params["iterations"]):
            cycles, ok = _simulate(instrs, scratch_map, params)
            if not ok:
                return {"score": 0.0, "stage": "invalid",
                        "error": f"Kernel produces incorrect output: Iteration {i}: Incorrect output values",
                        "cycles": int(cycles), "time": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001 — a malformed program scores 0, never crashes us
        sys.stderr.write(traceback.format_exc()[-1500:])
        return {"score": 0.0, "stage": "invalid",
                "error": f"the simulator rejected the program ({type(e).__name__})",
                "time": time.perf_counter() - t0}

    cycles = int(cycles)
    fitness = TARGET / cycles
    return {
        "score": float(fitness),
        "stage": "full",
        "objective": cycles,
        "objective_name": "cycles",
        "objective_direction": "min",
        "objective_target": TARGET,
        "cycles": cycles,
        "speedup": round(BASELINE_CYCLES / cycles, 3),
        "baseline": BASELINE_CYCLES,
        "n_instructions": len(instrs),
        "correctness_trials": params["iterations"],
        "target": TARGET,
        "solve_time": round(t_build, 3),
        "time": time.perf_counter() - t0,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
