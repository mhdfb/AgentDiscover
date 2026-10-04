"""Model spend of a problem's run so far, at list prices, and an optional budget stop.

Read from the harness logs under runs/<problem>/: Claude stream-json transcripts and
codex rollouts. With --budget, exit code 4 means the budget is spent; run.sh touches
STOP on that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import config
from . import runlog

# $ per million tokens, list prices. "write5"/"write1" are Anthropic's 5-minute/1-hour
# cache writes; OpenAI bills uncached input at the input price.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku":  {"input": 1.00, "cached": 0.10, "write5": 1.25, "write1": 2.00, "output": 5.00},
    "claude-sonnet": {"input": 2.00, "cached": 0.20, "write5": 2.50, "write1": 4.00, "output": 10.00},
    "claude-opus":   {"input": 5.00, "cached": 0.50, "write5": 6.25, "write1": 10.00, "output": 25.00},
    "claude-opus-5-5": {"input": 4.00, "cached": 0.20, "write5": 5.00, "write1": 8.00, "output": 20.00},
    # Short-context rates; the higher price for inputs over 272K tokens is not modeled.
    "gpt-6-astra":   {"input": 10.00, "cached": 1.00, "output": 50.00},
    "gpt-6.1-sol":   {"input": 2.00, "cached": 0.10, "output": 10.00},
    "gpt-6-sol":     {"input": 2.00, "cached": 0.20, "output": 10.00},
    "gpt-6-luna":    {"input": 0.10, "cached": 0.01, "output": 0.50},
    "gpt-5.6-sol":   {"input": 4.00, "cached": 0.40, "output": 20.00},
    "gpt-5.6-terra": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gpt-5.6-luna":  {"input": 0.20, "cached": 0.02, "output": 1.20},
    "gpt-5.5":       {"input": 5.00, "cached": 0.50, "output": 30.00},
    "gpt-5.4-nano":  {"input": 0.20, "cached": 0.02, "output": 1.25},
    "gpt-5.4-mini":  {"input": 0.75, "cached": 0.075, "output": 4.50},
    "gpt-5.4":       {"input": 2.50, "cached": 0.25, "output": 15.00},
    "gpt-5.3-codex": {"input": 1.75, "cached": 0.175, "output": 14.00},
    "gpt-5.2":       {"input": 1.75, "cached": 0.175, "output": 14.00},
    "gpt-5.1":       {"input": 1.25, "cached": 0.125, "output": 10.00},
    "gpt-5-nano":    {"input": 0.05, "cached": 0.005, "output": 0.40},
    "gpt-5-mini":    {"input": 0.25, "cached": 0.025, "output": 2.00},
    "gpt-5":         {"input": 1.25, "cached": 0.125, "output": 10.00},
}


def price_of(model: str) -> dict[str, float] | None:
    """Longest matching prefix, so gpt-5.4-mini is not priced as gpt-5. The prefix must
    end at a `-` or the end of the name: gpt-5.6-sol is not gpt-5, gpt-5-2025-08-07 is."""
    m = (model or "").lower()
    if m in ("haiku", "sonnet", "opus"):
        m = "claude-" + m
    best = max((k for k in PRICES if m == k or m.startswith(k + "-")), key=len, default=None)
    return PRICES[best] if best else None


def claude_log_cost(path: Path) -> tuple[float, set[str]]:
    """($, unpriced models) for one Claude Code stream-json transcript."""
    msgs: dict[str, dict] = {}
    models: dict[str, str] = {}
    out_total = 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "result":
            out_total += ((d.get("usage") or {}).get("output_tokens") or 0)
        if d.get("type") != "assistant":
            continue
        m = d.get("message") or {}
        msgs.setdefault(m.get("id", ""), m.get("usage") or {})
        models.setdefault(m.get("id", ""), m.get("model") or "")
    cost, unpriced = 0.0, set()
    per_msg_out = (out_total / len(msgs)) if (out_total and msgs) else None
    for mid, u in msgs.items():
        p = price_of(models.get(mid, ""))
        if p is None:
            unpriced.add(models.get(mid, "") or "?")
            continue
        cc = u.get("cache_creation") or {}
        w5 = cc.get("ephemeral_5m_input_tokens", u.get("cache_creation_input_tokens", 0)) if cc \
            else u.get("cache_creation_input_tokens", 0)
        w1 = cc.get("ephemeral_1h_input_tokens", 0) if cc else 0
        out = per_msg_out if per_msg_out is not None else u.get("output_tokens", 0)
        cost += (u.get("input_tokens", 0) * p["input"] + u.get("cache_read_input_tokens", 0) * p["cached"]
                 + w5 * p["write5"] + w1 * p["write1"] + out * p["output"]) / 1e6
    return cost, unpriced


def codex_rollout_cost(path: Path) -> tuple[float, set[str]]:
    """($, unpriced models) for one codex rollout."""
    model, cost, unpriced = "", 0.0, set()
    for line in path.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        p = d.get("payload") or {}
        if d.get("type") == "turn_context" and p.get("model"):
            model = p["model"]
        if p.get("type") != "token_count" or not (p.get("info") or {}).get("last_token_usage"):
            continue
        u = p["info"]["last_token_usage"]
        pr = price_of(model)
        if pr is None:
            unpriced.add(model or "?")
            continue
        cached = u.get("cached_input_tokens", 0)
        cost += ((u.get("input_tokens", 0) - cached) * pr["input"] + cached * pr["cached"]
                 + u.get("output_tokens", 0) * pr["output"]) / 1e6
    return cost, unpriced


def run_spend(problem: str) -> tuple[float, set[str]]:
    rd = config.runs_dir(problem)
    total, unpriced = 0.0, set()
    for path in list(rd.glob("agent-*/agent-logs/iter-*.jsonl")) + list(rd.glob("meta-logs/*.jsonl")):
        c, u = claude_log_cost(path)
        total += c; unpriced |= u
    for path in list(rd.glob("agent-*/.codex/sessions/**/rollout-*.jsonl")) + \
            list(rd.glob("meta-home/.codex/sessions/**/rollout-*.jsonl")):
        c, u = codex_rollout_cost(path)
        total += c; unpriced |= u
    return total, unpriced


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.spend")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--budget", type=float, default=None, help="USD; exit 4 once reached")
    args = parser.parse_args()
    total, unpriced = run_spend(args.problem)
    line = f"[spend] ${total:.2f} at list prices"
    if args.budget is not None:
        line += f" of the ${args.budget:.2f} budget"
    if unpriced:
        line += f" — NOT counted, no price known for: {', '.join(sorted(unpriced))}"
    print(line, flush=True)
    if args.budget is not None and total >= args.budget:
        runlog.append(args.problem, line + " — budget spent, stopping the run")
        sys.exit(4)


if __name__ == "__main__":
    main()
