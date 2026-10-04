"""Load the files in problems/<problem>/resources/ into Resource nodes, one shared node
per file: `name` is the filename, `path` is where a sandboxed agent finds it (the
directory is mounted read-only at /resources)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import config, db

CONTAINER_DIR = "/resources"

# Importable code the candidate needs (problems/<problem>/support/), mounted and put on
# PYTHONPATH, unlike the reading material in resources/.
SUPPORT_CONTAINER_DIR = "/support"


def resources_dir(problem: str) -> Path:
    return config.problem_dir(problem) / "resources"


def support_dir(problem: str) -> Path:
    return config.problem_dir(problem) / "support"


def resource_files(problem: str) -> list[Path]:
    d = resources_dir(problem)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file())


def container_path(backend: str, problem: str, filename: str) -> str:
    if backend == "none":
        return str(resources_dir(problem) / filename)
    return f"{CONTAINER_DIR}/{filename}"


def load(problem: str, backend: str) -> int:
    files = resource_files(problem)
    for f in files:
        db.run_write(
            """
            MERGE (r:Resource {name: $name})
            SET r.path = $path
            """,
            name=f.name, path=container_path(backend, problem, f.name),
        )
    return len(files)


def resource_names(problem: str) -> set[str]:
    return {f.name for f in resource_files(problem)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.resources")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    n = load(args.problem, args.backend)
    print(f"resources: {n} file(s) loaded")


if __name__ == "__main__":
    main()
