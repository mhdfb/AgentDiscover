"""Run a genome through the problem's evaluator bundle, inside the sandbox.
Bundle: problems/<problem>/evaluator/evaluate.sh <solution_path> prints one JSON line;
requirements.txt, Dockerfile and fetch.sh are optional. The evaluation has no network.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from .. import config, runconfig, sandbox


class EvaluatorSetupError(Exception):
    """The bundle/image is unusable — an operator problem, not the agent's."""


def bundle_dir(problem: str) -> Path:
    d = config.problem_dir(problem) / "evaluator"
    if not (d / "evaluate.sh").exists():
        raise EvaluatorSetupError(
            f"problems/{problem}/evaluator/evaluate.sh not found — every problem needs "
            "an evaluator bundle (see problems/labs/evaluator for the shape)"
        )
    return d


def _images_cache() -> Path:
    return config.repo_root() / "runs" / ".images"


def _bundle_fingerprint(bundle: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in bundle.rglob("*") if p.is_file()):
        h.update(f.relative_to(bundle).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def resolve_image(problem: str, backend: str) -> str:
    """Returns the runnable image reference for this problem's evaluator: the configured
    [evaluator].image, else the bundle's Dockerfile, else the platform image."""
    bundle = bundle_dir(problem)
    cfg = config.load_problem_config(problem).get("evaluator", {})

    try:
        if cfg.get("image"):
            return sandbox.prepare_image(backend, cfg["image"], _images_cache())

        if (bundle / "Dockerfile").exists():
            if backend in ("docker", "podman"):
                tag = f"agentdiscover-eval-{problem}:{_bundle_fingerprint(bundle)}"
                have = subprocess.run([backend, "image", "inspect", tag],
                                      capture_output=True, check=False)
                if have.returncode != 0:
                    subprocess.run([backend, "build", "-t", tag, str(bundle)], check=True)
                return tag
            if backend == "none":
                raise EvaluatorSetupError(
                    f"problems/{problem} has its own Dockerfile; backend `none` cannot "
                    "provide that environment. Use a container backend."
                )
            # Backends that cannot build pull the published image.
            return sandbox.prepare_image(backend, config.eval_image_ref(problem), _images_cache())

        return sandbox.prepare_image(backend, runconfig.get(problem, "platform_image"),
                                    _images_cache())
    except subprocess.CalledProcessError as e:
        raise EvaluatorSetupError(
            f"could not obtain the evaluation image ({' '.join(map(str, e.cmd[:3]))}… "
            f"failed with exit {e.returncode}). The image may not be published or the "
            "registry unreachable — an operator problem, not yours."
        ) from None


def _ensure_deps(problem: str, backend: str, image: str) -> Path | None:
    """Install requirements.txt (platform-image bundles only) into a cached dir,
    with network, once per requirements change. Returns the deps dir or None."""
    bundle = bundle_dir(problem)
    req = bundle / "requirements.txt"
    if not req.exists() or (bundle / "Dockerfile").exists():
        return None
    fp = hashlib.sha256(req.read_bytes()).hexdigest()[:12]
    deps = config.runs_dir(problem) / "eval-deps" / fp
    marker = deps / ".complete"
    if marker.exists():
        return deps
    if deps.exists():
        shutil.rmtree(deps)
    deps.mkdir(parents=True)
    # pip's temp dir sits inside the target dir so its `--target` moves are renames,
    # not cross-mount copies.
    scratch = sandbox.scratch_dir("pip")
    piptmp = deps / ".piptmp"
    piptmp.mkdir()
    # Name paths in the command via sandbox.cpath: under backend `none` the mounts are
    # the host directories themselves, so /deps and /bundle do not exist.
    deps_mount = sandbox.Mount(deps, "/deps", ro=False)
    bundle_mount = sandbox.Mount(bundle, "/bundle", ro=True)
    spec = sandbox.SandboxSpec(
        backend=backend, image=image,
        command=["python3", "-m", "pip", "install", "--quiet", "--no-input",
                 "--no-cache-dir", "--disable-pip-version-check",
                 "--target", sandbox.cpath(backend, deps_mount),
                 "-r", f"{sandbox.cpath(backend, bundle_mount)}/requirements.txt"],
        mounts=[deps_mount, bundle_mount],
        env={"TMPDIR": f"{sandbox.cpath(backend, deps_mount)}/.piptmp"},
        network="host",  # pip needs the network; the genome is not present here
        home_dir=scratch,
    )
    # Large installs onto a network file system are slow; allow an hour.
    try:
        result = sandbox.run(spec, wall_seconds=3600)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(piptmp, ignore_errors=True)
    if result.timed_out or result.returncode != 0:
        what = ("timed out after 3600 s" if result.timed_out
                else f"failed (exit {result.returncode})")
        raise EvaluatorSetupError(
            f"pip install of problems/{problem}/evaluator/requirements.txt {what}:\n"
            + (result.stderr or result.stdout)[-2000:]
        )
    marker.touch()
    return deps


def _ensure_data(problem: str, backend: str, image: str,
                 deps: Path | None = None) -> Path | None:
    """Run the bundle's fetch.sh once, with network and no genome; returns the data dir
    or None. The script gets the destination dir as its only argument and has installed
    requirements on PYTHONPATH. Every bundle file named `fetch*` keys the cache."""
    bundle = bundle_dir(problem)
    fetch = bundle / "fetch.sh"
    if not fetch.exists():
        return None
    # The key is fetch.sh's bytes, extended by each helper's name and bytes.
    h = hashlib.sha256(fetch.read_bytes())
    for f in sorted(bundle.glob("fetch*")):
        if f != fetch:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    fp = h.hexdigest()[:12]
    data = config.runs_dir(problem) / "eval-data" / fp
    marker = data / ".complete"
    if marker.exists():
        return data
    data.mkdir(parents=True, exist_ok=True)
    scratch = sandbox.scratch_dir("fetch")
    data_mount = sandbox.Mount(data, "/data", ro=False)
    bundle_mount = sandbox.Mount(bundle, "/bundle", ro=True)
    mounts = [data_mount, bundle_mount]
    env = {}
    if deps is not None:
        mounts.append(sandbox.Mount(deps, "/deps", ro=True))
        env["PYTHONPATH"] = sandbox.cpath(backend, mounts[-1])
    spec = sandbox.SandboxSpec(
        backend=backend, image=image,
        command=["/bin/sh", f"{sandbox.cpath(backend, bundle_mount)}/fetch.sh",
                 sandbox.cpath(backend, data_mount)],
        mounts=mounts, env=env,
        network="host",   # the download needs the network; the genome is not present
        home_dir=scratch,
        tmp_dir=scratch,
    )
    try:
        result = sandbox.run(spec, wall_seconds=3600)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if result.timed_out or result.returncode != 0:
        raise EvaluatorSetupError(
            f"problems/{problem}/evaluator/fetch.sh failed"
            + (" (timed out after 3600s)" if result.timed_out else "")
            + ":\n" + (result.stderr or result.stdout)[-2000:]
        )
    marker.touch()
    return data


def prepare(problem: str, backend: str) -> str:
    """Build the evaluation environment (deps, data) before the run starts, so a broken
    bundle fails at launch. Cached and idempotent. Returns a one-line summary."""
    image = resolve_image(problem, backend)
    deps = _ensure_deps(problem, backend, image)
    data = _ensure_data(problem, backend, image, deps)
    parts = []
    if deps is not None:
        parts.append(f"deps {deps.name}")
    if data is not None:
        parts.append(f"data {data.name}")
    return "evaluator: " + (", ".join(parts) + " ready" if parts else "nothing to install")


def evaluate(problem: str, backend: str, genome: str, genome_hash: str) -> dict:
    """Score one genome. Always returns a metrics dict with `score` and `stage`;
    a crash, timeout, or unparseable output scores 0.0 and the run continues."""
    limits = config.problem_limits(problem)
    bundle = bundle_dir(problem)
    image = resolve_image(problem, backend)
    deps = _ensure_deps(problem, backend, image)
    data = _ensure_data(problem, backend, image, deps)

    staging = config.runs_dir(problem) / "eval" / f"{genome_hash[:12]}-{int(time.time() * 1000)}"
    staging.mkdir(parents=True)
    (staging / "solution.py").write_text(genome)

    mounts = [
        sandbox.Mount(staging, "/work", ro=False),
        sandbox.Mount(bundle, "/bundle", ro=True),
    ]
    env = {"HOME": "/tmp"}
    if limits.solve_seconds is not None:
        # The bundle reads the genome's time budget from this variable.
        env["AGENTDISCOVER_SOLVE_SECONDS"] = str(limits.solve_seconds)
    if deps is not None:
        mounts.append(sandbox.Mount(deps, "/deps", ro=True))
        env["PYTHONPATH"] = sandbox.cpath(backend, mounts[-1])
    if data is not None:
        mounts.append(sandbox.Mount(data, "/data", ro=True))
        env["AGENTDISCOVER_DATA_DIR"] = sandbox.cpath(backend, mounts[-1])
    if limits.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(runconfig.get(problem, "eval_gpus"))

    b = backend
    work = sandbox.cpath(b, mounts[0])
    bnd = sandbox.cpath(b, mounts[1])
    spec = sandbox.SandboxSpec(
        backend=b, image=image,
        command=["/bin/sh", f"{bnd}/evaluate.sh", f"{work}/solution.py"],
        mounts=mounts, env=env, workdir=work,
        network="none",
        limits=limits,
        name=f"agentdiscover-eval-{genome_hash[:12]}",
        extra_args=(["--tmpfs", "/tmp"] if b in ("docker", "podman") else []),
        gpu=limits.gpu,
    )

    t0 = time.perf_counter()
    result = sandbox.run(spec, wall_seconds=limits.wall_seconds)
    wall = time.perf_counter() - t0
    shutil.rmtree(staging, ignore_errors=True)

    if result.timed_out:
        return {"score": 0.0, "stage": "timeout",
                "error": f"evaluation exceeded {limits.wall_seconds}s wall clock",
                "time": wall}

    # The last stdout line that looks like JSON wins (bundles may print progress first).
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                metrics = json.loads(line)
            except ValueError:
                continue
            if isinstance(metrics, dict) and "score" in metrics:
                metrics.setdefault("stage", "full")
                metrics.setdefault("time", wall)
                return metrics
    tail = (result.stderr or result.stdout)[-400:].strip()
    return {"score": 0.0, "stage": "harness_error",
            "error": f"evaluator exited with code {result.returncode}"
                     + (f"; output tail: {tail}" if tail else " and produced no output"),
            "time": wall}
