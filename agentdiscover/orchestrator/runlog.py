"""Append-only run.log, kept in the problem's runs directory."""
from __future__ import annotations

import datetime

from .. import config


def append(problem: str, line: str) -> None:
    path = config.runs_dir(problem) / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(line.rstrip("\n") + "\n")


def header(problem: str, text: str) -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    append(problem, f"=== {stamp} | {text} ===")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.runlog")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--header", default=None)
    parser.add_argument("--line", default=None)
    args = parser.parse_args()
    if args.header:
        header(args.problem, args.header)
    if args.line:
        append(args.problem, args.line)


if __name__ == "__main__":
    main()
