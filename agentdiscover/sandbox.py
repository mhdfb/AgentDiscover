"""The sandbox layer: builds every container command line, one case per backend.

Limits: runtime flags where the backend enforces them, plus an `ulimit` preamble
everywhere. Network: "host", or "none" for evaluations (enroot cannot isolate it).
"""
from __future__ import annotations

import math
import os
import pwd
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Limits

BACKENDS = ("docker", "podman", "apptainer", "singularity", "enroot")


def detect_backend() -> str:
    """SANDBOX_BACKEND overrides; otherwise the first runtime on PATH (never `none`)."""
    override = os.environ.get("SANDBOX_BACKEND", "").strip()
    if override:
        if override not in BACKENDS + ("none",):
            raise SystemExit(f"SANDBOX_BACKEND={override!r} is not one of {BACKENDS + ('none',)}")
        return override
    for backend in BACKENDS:
        if shutil.which(backend):
            return backend
    raise SystemExit(
        "No container runtime found (looked for: " + ", ".join(BACKENDS) + ").\n"
        "Install one — Podman or Apptainer install without root — or, for debugging "
        "only, export SANDBOX_BACKEND=none to run without a sandbox."
    )


@dataclass(frozen=True)
class Mount:
    host: Path
    cont: str
    ro: bool = True


def cpath(backend: str, mount: Mount) -> str:
    """A mount's path as the sandboxed command sees it (the host path under `none`)."""
    return str(mount.host) if backend == "none" else mount.cont


# ---------------------------------------------------------------------------
# The ulimit preamble (portable floor)
# ---------------------------------------------------------------------------


def _nproc_ceiling(backend: str, nproc: int) -> int | None:
    """The value for `ulimit -u`, or None to leave it alone. RLIMIT_NPROC is per-UID
    and counts threads, so under `none` (no PID namespace) the ceiling is the user's
    current thread count plus `nproc`."""
    if backend != "none":
        return nproc
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        # Count only this UID's threads; if /proc is unreadable, set no limit.
        uid = os.getuid()
        used = 0
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != uid:
                    continue
                used += int((entry / "status").read_text().split("Threads:")[1].split()[0])
            except (OSError, IndexError, ValueError):
                continue
        ceiling = used + nproc
        if hard != resource.RLIM_INFINITY:
            ceiling = min(ceiling, hard)
        return ceiling
    except Exception:  # noqa: BLE001 - never let limit computation break the run
        return None


def _ulimit_wrap(command: list[str], limits: Limits, backend: str = "none") -> list[str]:
    """Wrap `command` so the OS caps are set before it runs. `exec` keeps the real
    command as the direct child, so signals from outside reach it."""
    mem_kb = int(limits.mem_gb * 1024 * 1024)
    cpu_s = int(limits.wall_seconds * max(1.0, math.ceil(limits.cpus)))
    fsize_kb = limits.fsize_mb * 1024
    parts = []
    # No address-space cap for a GPU evaluation: see Limits.gpu.
    if not limits.gpu:
        parts.append(f"ulimit -v {mem_kb} 2>/dev/null || echo 'agentdiscover: ulimit -v unsupported' >&2; ")
    parts.append(f"ulimit -t {cpu_s} 2>/dev/null || echo 'agentdiscover: ulimit -t unsupported' >&2; ")
    ceiling = _nproc_ceiling(backend, limits.nproc)
    if ceiling is not None:
        parts.append(f"ulimit -u {ceiling} 2>/dev/null || true; ")
    parts.append(f"ulimit -f {fsize_kb} 2>/dev/null || true; ")
    return ["/bin/sh", "-c", "".join(parts) + 'exec "$@"', "sh", *command]


# ---------------------------------------------------------------------------
# Image acquisition
# ---------------------------------------------------------------------------


def _slug(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", ref)


def _enroot_uri(oci_ref: str) -> str:
    """An OCI reference in the form enroot parses: docker://[REGISTRY#]IMAGE[:TAG].
    `docker.io` is not a registry API endpoint, so it becomes registry-1.docker.io."""
    host, sep, rest = oci_ref.partition("/")
    if sep and ("." in host or ":" in host or host == "localhost"):
        if host in ("docker.io", "index.docker.io"):
            host = "registry-1.docker.io"
        return f"docker://{host}#{rest}"
    # No registry component: Docker Hub, where a bare name means library/<name>.
    return f"docker://registry-1.docker.io#{oci_ref if sep else f'library/{oci_ref}'}"


_enroot_checked = False


def _enroot_writable_paths(cache_dir: Path | None = None) -> None:
    """Give enroot writable scratch directories when its site config has none (a
    cluster config often points at /run/enroot, which exists only inside a batch job).
    Probes once; variables the operator set are left alone."""
    global _enroot_checked
    if _enroot_checked:
        return
    _enroot_checked = True
    try:
        probe = subprocess.run(["enroot", "list"], capture_output=True, text=True)
    except FileNotFoundError:
        # No enroot binary: building an argv (a dry run) must still work.
        return
    if probe.returncode == 0 and "Permission denied" not in probe.stderr:
        return
    base = Path(tempfile.gettempdir()) / f"enroot-{os.getuid()}"
    defaults = {
        "ENROOT_RUNTIME_PATH": base / "runtime",
        "ENROOT_DATA_PATH": base / "data",
        "ENROOT_TEMP_PATH": base / "tmp",
    }
    if cache_dir is not None:
        defaults["ENROOT_CACHE_PATH"] = cache_dir / "enroot-cache"
    for var, path in defaults.items():
        if os.environ.get(var):
            continue
        path.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(path.resolve())


def prepare_image(backend: str, oci_ref: str, cache_dir: Path) -> str:
    """Return the image reference in the form this backend runs: the OCI ref for
    docker/podman, a SIF cached under `cache_dir` for apptainer/singularity, a named
    container for enroot."""
    if backend in ("docker", "podman", "none"):
        return oci_ref
    cache_dir.mkdir(parents=True, exist_ok=True)
    if backend in ("apptainer", "singularity"):
        # An existing .sif file is used as-is, so no registry access is needed.
        if oci_ref.endswith(".sif") and Path(oci_ref).is_file():
            return oci_ref
        sif = cache_dir / f"{_slug(oci_ref)}.sif"
        if not sif.exists():
            subprocess.run(
                [backend, "pull", str(sif), f"docker://{oci_ref}"],
                check=True,
            )
        return str(sif)
    if backend == "enroot":
        _enroot_writable_paths(cache_dir)
        name = _slug(oci_ref)
        sqsh = cache_dir / f"{name}.sqsh"
        if not sqsh.exists():
            subprocess.run(["enroot", "import", "-o", str(sqsh), _enroot_uri(oci_ref)], check=True)
        listed = subprocess.run(["enroot", "list"], capture_output=True, text=True)
        if name not in listed.stdout.split():
            subprocess.run(["enroot", "create", "--name", name, str(sqsh)], check=True)
        return name
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


@dataclass
class SandboxSpec:
    backend: str
    image: str                       # from prepare_image()
    command: list[str]
    mounts: list[Mount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None
    network: str = "host"            # "host" | "none"
    limits: Limits | None = None     # None = no cage (trusted service containers)
    name: str | None = None          # docker/podman container name (for kill-by-name)
    home_dir: Path | None = None     # host dir to serve as the container's $HOME
    tmp_dir: Path | None = None      # host dir to serve as the container's /tmp
    extra_args: list[str] = field(default_factory=list)  # backend-specific extras
    # Pass the host's NVIDIA GPUs through; CUDA_VISIBLE_DEVICES in `env` picks which.
    gpu: bool = False


def scratch_dir(prefix: str) -> Path:
    """A private host directory for a sandbox's /tmp, created where this machine keeps
    temporary files (TMPDIR). The caller removes it when the sandbox exits."""
    return Path(tempfile.mkdtemp(prefix=f"agentdiscover-{prefix}-"))


def home_path(backend: str, home_dir: Path | None) -> str:
    """Where a home_dir appears inside the sandbox. Singularity/Apptainer set HOME from
    the /etc/passwd entry and ignore the caller's $HOME, so there home_dir is bound
    over the passwd home; the other backends get a neutral /home/agent."""
    if home_dir is None:
        return os.environ.get("HOME", "/tmp")
    if backend == "none":
        return str(home_dir)
    if backend in ("apptainer", "singularity"):
        try:
            return pwd.getpwuid(os.getuid()).pw_dir
        except KeyError:            # no passwd entry (some container hosts)
            return os.environ.get("HOME", "/home/agent")
    return "/home/agent"


def build_argv(spec: SandboxSpec) -> list[str]:
    b = spec.backend
    command = spec.command
    if spec.limits is not None:
        command = _ulimit_wrap(command, spec.limits, b)

    # The home and tmp binds go first: Apptainer/Singularity apply binds in order and
    # refuse one whose destination does not exist yet (e.g. a bind inside the home).
    mounts: list[Mount] = []
    env = dict(spec.env)
    if spec.tmp_dir is not None:
        # Apptainer's --contain /tmp is a small in-memory dir, so bind a host directory.
        # Trusted installs only: the small /tmp bounds what a bad genome can write.
        if b != "none":
            mounts.append(Mount(spec.tmp_dir, "/tmp", ro=False))
            env.setdefault("TMPDIR", "/tmp")
        else:
            env.setdefault("TMPDIR", str(spec.tmp_dir))
    if spec.home_dir is not None:
        home = home_path(b, spec.home_dir)
        if b != "none":
            mounts.append(Mount(spec.home_dir, home, ro=False))
        if b not in ("apptainer", "singularity"):
            env["HOME"] = home
    if b in ("apptainer", "singularity"):
        # Singularity refuses a HOME override via the environment; drop the caller's.
        env.pop("HOME", None)
    mounts += spec.mounts

    if b == "none":
        # No container: the ulimit wrapper is the whole cage; run() applies env and cwd.
        return command

    if b in ("docker", "podman"):
        argv = [b, "run", "--rm", "--init"]
        if spec.name:
            argv += ["--name", spec.name]
        argv += [f"--network={spec.network}"]
        if b == "docker":
            # Run as the invoking user, never root; files written to mounts stay theirs.
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        else:
            argv += ["--userns=keep-id"]
        if spec.limits is not None:
            argv += [
                "--memory", f"{int(spec.limits.mem_gb * 1024)}m",
                "--cpus", str(spec.limits.cpus),
                "--pids-limit", str(spec.limits.nproc),
            ]
        if spec.gpu:
            # docker: the NVIDIA container toolkit; podman: its CDI device name.
            argv += ["--gpus", "all"] if b == "docker" else ["--device", "nvidia.com/gpu=all"]
        for m in mounts:
            argv += ["-v", f"{m.host}:{m.cont}" + (":ro" if m.ro else "")]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        if spec.workdir:
            argv += ["-w", spec.workdir]
        argv += spec.extra_args
        argv += [spec.image]
        return argv + command

    if b in ("apptainer", "singularity"):
        # --cleanenv: never leak the orchestrator's environment (it holds secrets).
        # --contain: minimal /dev, session-scoped /tmp and an empty $HOME.
        # --pid: own process namespace, so the agent cannot see or kill host processes.
        argv = [b, "exec", "--cleanenv", "--contain", "--pid"]
        if spec.gpu:
            # --nv binds the host's driver libraries and device nodes into the image.
            argv.append("--nv")
        if spec.network == "none":
            # "none" is the only network mode unprivileged users may request.
            argv += ["--net", "--network", "none"]
        for m in mounts:
            argv += ["--bind", f"{m.host}:{m.cont}" + (":ro" if m.ro else "")]
        for k, v in env.items():
            argv += ["--env", f"{k}={v}"]
        if spec.workdir:
            argv += ["--pwd", spec.workdir]
        argv += spec.extra_args
        argv += [spec.image]
        return argv + command

    if b == "enroot":
        _enroot_writable_paths()
        argv = ["enroot", "start", "--rw"]
        if spec.gpu:
            # Enroot's nvidia hook injects the driver for whatever this variable names.
            env.setdefault("NVIDIA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "all"))
        for k, v in env.items():
            argv += ["--env", f"{k}={v}"]
        for m in mounts:
            # x-create=auto: enroot refuses a mount whose destination does not exist.
            flags = "bind,x-create=auto" + (",ro" if m.ro else "")
            argv += ["--mount", f"{m.host}:{m.cont}:none:{flags}"]
        argv += spec.extra_args
        argv += [spec.image]
        # Enroot overwrites the image's PATH with a bare system default; restore it from
        # the environment file enroot wrote into the rootfs.
        restore_path = ('p=$(sed -n "s/^PATH=//p" /etc/environment 2>/dev/null | tail -1); '
                        '[ -n "$p" ] && export PATH="$p"; ')
        if spec.workdir:
            cmd = ["/bin/sh", "-c", restore_path + 'cd "$1" && shift && exec "$@"',
                   "sh", spec.workdir, *command]
        else:
            cmd = ["/bin/sh", "-c", restore_path + 'exec "$@"', "sh", *command]
        return argv + cmd

    raise ValueError(f"unknown backend {b!r}")


# ---------------------------------------------------------------------------
# Execution with a wall-clock deadline
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def run(spec: SandboxSpec, *, wall_seconds: float | None = None,
        capture: bool = True, stdout_file=None, stderr_file=None) -> SandboxResult:
    """Run the sandboxed command, enforcing the wall clock here. On timeout the process
    group is killed; a docker/podman container is also killed by name, because its
    processes live under the runtime daemon, outside our process group."""
    argv = build_argv(spec)
    env = None
    cwd = None
    if spec.backend == "none":
        base = {k: os.environ[k] for k in ("PATH", "LANG", "TERM") if k in os.environ}
        env = {**base, **spec.env, "HOME": home_path("none", spec.home_dir)}
        cwd = spec.workdir

    proc = subprocess.Popen(
        argv,
        stdout=(subprocess.PIPE if capture else stdout_file),
        stderr=(subprocess.PIPE if capture else stderr_file),
        text=capture,
        env=env,
        cwd=cwd,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=wall_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if spec.backend in ("docker", "podman") and spec.name:
            subprocess.run([spec.backend, "kill", spec.name],
                           capture_output=True, check=False)
        out, err = proc.communicate()
    return SandboxResult(
        returncode=proc.returncode,
        stdout=out if capture else "",
        stderr=err if capture else "",
        timed_out=timed_out,
    )


def spawn_detached(spec: SandboxSpec, log_path: Path) -> int:
    """Start a long-lived sandboxed process and return its pid, for stop_detached()."""
    argv = build_argv(spec)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    env = None
    if spec.backend == "none":
        env = {**os.environ, **spec.env}
    proc = subprocess.Popen(
        argv, stdout=log, stderr=log, env=env,
        cwd=spec.workdir if spec.backend == "none" else None,
        start_new_session=True,
    )
    return proc.pid


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_detached(pid: int, grace: float = 15.0) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    # `python -m agentdiscover.sandbox` prints the detected backend (used by run.sh).
    print(detect_backend())
