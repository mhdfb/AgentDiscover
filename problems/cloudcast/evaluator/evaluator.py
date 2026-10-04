"""CloudCast — minimise total multi-cloud broadcast cost across five configurations.

Ported from SkyDiscover's `benchmarks/ADRS/cloudcast`, the bundle CORAL runs. The task,
data and cost model are theirs, unchanged:

  * `simulator.py`, `broadcast.py`, `utils.py` are byte-identical copies of SkyDiscover's;
    `profiles/*.csv` and `examples/config/*.json` come from their download_dataset.sh.
  * The five configurations, their order, and `num_vms = 2` match their evaluator.
  * The objective is the SUM of the simulated cost over the five configurations
    (their evaluator's `total_cost`), lower is better. Their `combined_score` is the
    monotone transform 1/(1+total_cost) of the same number.
  * The solve budget is 600 s, matching their `evaluator.timeout: 600`.

What differs is the interface, to fit this platform's `solve()` contract, plus one
hardening: SkyDiscover's simulator prices each hop using the edge data EMBEDDED in the
submitted paths, and only validates that the edges exist. Here every hop's cost and
throughput are rewritten from the real graph before simulation, so a candidate cannot
lower its bill by embedding fabricated edge data. For honest submissions (which embed
`G[s][t]` anyway) the number is identical.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cloudcast_data import NUM_VMS, load_configs   # noqa: E402
from simulator import BCSimulator                  # noqa: E402
from utils import make_nx_graph                    # noqa: E402

# The orchestrator passes the per-problem budget from problems/<p>/problem.toml so the
# number the agent is told and the number enforced here can never drift apart. The
# fallback keeps the bundle runnable standalone. 600 s matches SkyDiscover's config.
SOLVE_TIMEOUT = float(os.environ.get("AGENTDISCOVER_SOLVE_SECONDS", 600.0))

# A bar to BEAT by a clear margin, not to meet: the agent is told that landing at the
# target is not success. Competitor numbers live in paper/benchmark.md and are
# deliberately kept out of everything the agent reads — naming one anchors the search.
TARGET = 618.0


def _run_solve(solution_dir: Path, timeout: float) -> tuple[dict, float]:
    """Run solve() in a fresh subprocess. Return (paths_by_config, wall_seconds)."""
    code = (
        "import sys, json, time\n"
        "sys.path.insert(0, '.')\n"
        f"sys.path.insert(0, {str(_HERE)!r})\n"
        "from solution import solve\n"
        "t = time.perf_counter(); v = solve(); dt = time.perf_counter() - t\n"
        "print(json.dumps({'value': v, 'time': dt}))\n"
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


def _validate(paths, source_node, terminal_nodes, num_partitions, G) -> str | None:
    """SkyDiscover's topology validation, on the real graph. None if valid.

    Every destination must have every partition routed by a contiguous chain of real
    edges from the source to that destination.
    """
    if not isinstance(paths, dict):
        return f"paths must be a dict keyed by destination, got {type(paths).__name__}"
    missing = set(terminal_nodes) - set(paths)
    if missing:
        return f"missing destinations: {sorted(missing)}"
    for dst in terminal_nodes:
        for pid in range(num_partitions):
            hops = (paths[dst] or {}).get(str(pid))
            if not hops:
                return f"destination {dst!r}, partition {pid}: no path"
            at = source_node
            for hop in hops:
                if not isinstance(hop, (list, tuple)) or len(hop) < 3:
                    return f"destination {dst!r}, partition {pid}: malformed hop {hop!r}"
                s, t = hop[0], hop[1]
                if not G.has_edge(s, t):
                    return f"destination {dst!r}, partition {pid}: edge {s}->{t} not in the network"
                if s != at:
                    return (f"destination {dst!r}, partition {pid}: path discontinuity — "
                            f"expected hop from {at}, got {s}")
                at = t
            if at != dst:
                return (f"destination {dst!r}, partition {pid}: path ends at {at}, "
                        f"not the destination")
    return None


class _Topology:
    """The duck-typed object BCSimulator.initialization expects."""

    def __init__(self, src, dsts, num_partitions, paths):
        self.src, self.dsts, self.num_partitions, self.paths = src, dsts, num_partitions, paths


def evaluate(solution_path: Path) -> dict:
    t0 = time.perf_counter()

    try:
        value, solve_time = _run_solve(solution_path.parent, SOLVE_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "stage": "error", "error": str(e)[:300],
                "time": time.perf_counter() - t0}

    per_config: dict[str, float] = {}
    quiet = io.StringIO()  # the simulator prints noisily; stdout must stay one JSON line
    with contextlib.redirect_stdout(quiet):
        G = make_nx_graph(num_vms=NUM_VMS)
    for name, cfg in load_configs():
        paths = value.get(name) if isinstance(value, dict) else None
        if paths is None:
            return {"score": 0.0, "stage": "invalid",
                    "error": f"solve() returned no entry for configuration {name!r}",
                    "time": time.perf_counter() - t0}
        err = _validate(paths, cfg["source_node"], cfg["dest_nodes"],
                        cfg["num_partitions"], G)
        if err is not None:
            return {"score": 0.0, "stage": "invalid", "error": f"{name}: {err}",
                    "time": time.perf_counter() - t0}
        # Harden: price every hop from the real graph, never from submitted data.
        for dst in cfg["dest_nodes"]:
            for pid in range(cfg["num_partitions"]):
                for hop in paths[dst][str(pid)]:
                    hop[2] = dict(G[hop[0]][hop[1]])
        topo = _Topology(cfg["source_node"], cfg["dest_nodes"], cfg["num_partitions"], paths)
        try:
            with contextlib.redirect_stdout(quiet):
                sim = BCSimulator(NUM_VMS)
                _, cost = sim.evaluate_path(topo, cfg)
        except Exception as e:  # noqa: BLE001
            return {"score": 0.0, "stage": "error",
                    "error": f"simulation failed on {name}: {str(e)[:200]}",
                    "time": time.perf_counter() - t0}
        per_config[name] = float(cost)

    total = float(sum(per_config.values()))

    return {
        # Minimised objective: TARGET/total rises as the cost falls, passes 1.0 at the
        # target, and is not clamped — the target is a goal, not a proven bound.
        "score": TARGET / total if total > 0 else 0.0,
        "stage": "full",
        "objective": total,
        "objective_name": "total_cost",
        "objective_direction": "min",
        "objective_target": TARGET,
        "total_cost": total,
        "cost_per_config": per_config,
        "target": TARGET,
        "time": time.perf_counter() - t0,
        "solve_time": solve_time,
    }


def main():
    print(json.dumps(evaluate(Path(sys.argv[1]).resolve())))


if __name__ == "__main__":
    main()
