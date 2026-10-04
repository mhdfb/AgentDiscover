"""The 5-hour quota window: check it before a session, resume a session it cut off.

Exit codes for run.sh: 0 = proceed, 2 = end the run (STOP appeared during a wait),
3 = resume the session. A probe that cannot run never blocks a run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .. import config
from . import runlog

PROBE_URL = "https://api.anthropic.com/v1/messages"
PROBE_MODEL = "claude-haiku-4-5-20251001"   # smallest model; max_tokens=1


def _oauth_token() -> str | None:
    try:
        creds = json.loads((Path.home() / ".claude" / ".credentials.json").read_text())
        return creds.get("claudeAiOauth", {}).get("accessToken") or None
    except (OSError, ValueError):
        return None


def probe() -> dict | None:
    """One ~1-token request. Returns {"spent": bool, "reset_5h": epoch, "util_5h": x},
    or None when nothing can be learned (no credentials, endpoint unreachable).
    A 429 carries the same rate-limit headers as a success, so it is read too."""
    token = _oauth_token()
    if not token:
        return None
    body = json.dumps({
        "model": PROBE_MODEL, "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()
    req = urllib.request.Request(PROBE_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })
    try:
        spent = False
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                h = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return None
            spent = True
            h = {k.lower(): v for k, v in (e.headers or {}).items()}
        reset = h.get("anthropic-ratelimit-unified-5h-reset")
        if reset is None:
            return None
        util = float(h.get("anthropic-ratelimit-unified-5h-utilization", 1.0 if spent else 0.0))
        return {"spent": spent or util >= 1.0, "reset_5h": int(reset), "util_5h": util}
    except Exception:  # noqa: BLE001 — any other failure means "nothing learned"
        return None


def _is_result(line: str) -> bool:
    try:
        return line.lstrip().startswith("{") and json.loads(line).get("type") == "result"
    except ValueError:
        return False


def ended_in_error(jsonl: Path) -> bool:
    """Did this session's last launch end in an API error, rather than finish its work?

    Reads `is_error` from the terminal `result` event; do not match transcript text,
    healthy transcripts contain "rate_limit" too. A resumed session appends to the
    same transcript, so only the last launch is judged."""
    try:
        lines = jsonl.read_text(errors="replace").splitlines()
    except OSError:
        return False
    earlier = [i for i, line in enumerate(lines[:-1]) if _is_result(line)]
    start = earlier[-1] + 1 if earlier else 0

    final = None
    for line in lines[start:]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "result":
            final = event
    # No terminal result at all (killed at the wall clock, say) is not an API error.
    return bool(final is not None and final.get("is_error"))


def _wait_until(problem: str, reset_5h: int, label: str, why: str) -> int:
    """Sleep until the window resets (plus a minute's slack). 0 = done, 2 = STOP."""
    wait = max(0, int(reset_5h - time.time()) + 60)
    if wait == 0:
        return 0
    reset_str = time.strftime("%H:%M", time.localtime(reset_5h))
    msg = f"[quota] {why} — holding {label} {wait // 60} min, until the {reset_str} reset"
    print(msg, flush=True)
    runlog.append(problem, msg)
    stop_file = config.runs_dir(problem) / "STOP"
    deadline = time.time() + wait
    while time.time() < deadline:
        if stop_file.exists():
            print(f"[quota] STOP appeared while holding {label} — ending the run")
            return 2
        time.sleep(min(60, max(1, deadline - time.time())))
    print(f"[quota] window reset — continuing with {label}", flush=True)
    return 0


# Matched only against codex's `turn.failed` / `error` events and stderr ERROR lines,
# never the agent's own messages.
_CODEX_ACCOUNT = re.compile(r"no credits|billing|insufficient|\b401\b|unauthorized|"
                            r"invalid api key", re.IGNORECASE)
_CODEX_LIMIT = re.compile(r"usage limit|rate.?limit|\b429\b|too many requests|"
                          r"try again (at|in|later)", re.IGNORECASE)


def codex_api_failure(jsonl: Path) -> str | None:
    """Why the API refused this codex session: "account" (no credit, key rejected),
    "limit" (plan window used up), or None (it ended for its own reasons)."""
    try:
        lines = jsonl.read_text(errors="replace").splitlines()
    except OSError:
        return None
    kind = None
    for line in lines:
        line = line.strip()
        msg = ""
        if not line.startswith("{"):
            if " ERROR " in line:
                msg = line
        else:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "turn.failed":
                msg = str((event.get("error") or {}).get("message", ""))
            elif event.get("type") == "item.completed" and \
                    (event.get("item") or {}).get("type") == "error":
                msg = str(event["item"].get("message", ""))
        if not msg:
            continue
        if _CODEX_ACCOUNT.search(msg):
            return "account"
        if _CODEX_LIMIT.search(msg):
            kind = "limit"
    return kind


# codex's plan windows cannot be read, so a used-up one is polled at this interval.
CODEX_POLL_SECONDS = 600
_PROBE_PROMPT = "Reply with exactly the word OK."


def codex_probe(problem: str, backend: str) -> bool | None:
    """One tiny codex request in the platform image with the operator's credentials.
    True = answered, False = refused by a plan window, None = could not tell."""
    import os
    import shutil

    from .. import runconfig, sandbox

    try:
        image_ref = runconfig.get(problem, "platform_image")
    except Exception:  # noqa: BLE001 — no database yet: the launcher's default image
        image_ref = runconfig._BY_NAME["platform_image"].default
    image = sandbox.prepare_image(backend, image_ref, config.repo_root() / "runs" / ".images")
    scratch = sandbox.scratch_dir("codex-probe")
    try:
        (scratch / ".codex").mkdir()
        (scratch / ".home").mkdir()
        mount = sandbox.Mount(scratch, "/work", ro=False)
        work = sandbox.cpath(backend, mount)
        env = {"CODEX_HOME": f"{work}/.codex"}
        if os.environ.get("OPENAI_API_KEY"):
            env["CODEX_API_KEY"] = os.environ["OPENAI_API_KEY"]
        else:
            login = Path.home() / ".codex" / "auth.json"
            if not login.is_file():
                return None
            shutil.copy(login, scratch / ".codex" / "auth.json")
        spec = sandbox.SandboxSpec(
            backend=backend, image=image,
            command=["codex", "exec", "-c", 'web_search="disabled"',
                     "--dangerously-bypass-approvals-and-sandbox", "--json", _PROBE_PROMPT],
            mounts=[mount], env=env, workdir=work, network="host", limits=None,
            name="agentdiscover-codex-probe", home_dir=scratch / ".home",
        )
        result = sandbox.run(spec, wall_seconds=300)
    except Exception as e:  # noqa: BLE001 — a broken probe must never block a run
        print(f"[quota] codex probe could not run ({e})")
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    refused = False
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            return True
        msg = str((event.get("error") or {}).get("message", "")) if event.get("type") == "turn.failed" \
            else str(item.get("message", "")) if item.get("type") == "error" else ""
        if msg and _CODEX_LIMIT.search(msg):
            refused = True
    return False if refused else None


def _codex_wait(problem: str, backend: str, label: str, why: str) -> int:
    """Poll until codex answers again. 0 = it does (or the probe cannot tell),
    2 = STOP appeared meanwhile."""
    msg = f"[quota] {why} — polling codex every {CODEX_POLL_SECONDS // 60} min before {label}"
    print(msg, flush=True)
    runlog.append(problem, msg)
    stop_file = config.runs_dir(problem) / "STOP"
    while True:
        if stop_file.exists():
            print(f"[quota] STOP appeared while holding {label} — ending the run")
            return 2
        if codex_probe(problem, backend) is not False:
            print(f"[quota] codex answers again — continuing with {label}", flush=True)
            runlog.append(problem, f"[quota] codex answers again — continuing with {label}")
            return 0
        deadline = time.time() + CODEX_POLL_SECONDS
        while time.time() < deadline:
            if stop_file.exists():
                print(f"[quota] STOP appeared while holding {label} — ending the run")
                return 2
            time.sleep(min(60, max(1, deadline - time.time())))


def before_session(problem: str, label: str, harness: str = "claude",
                   backend: str = "none") -> int:
    if harness == "codex":
        if codex_probe(problem, backend) is False:
            return _codex_wait(problem, backend, label, "the codex plan window is used up")
        print(f"[quota] codex answers — starting {label}")
        return 0
    if harness != "claude":
        # The probe only asks about the Claude window; nothing to wait for elsewhere.
        print(f"[quota] {harness} harness — starting {label} without a window check")
        return 0
    info = probe()
    if info is None:
        print(f"[quota] probe unavailable — starting {label} without checking")
        return 0
    if not info["spent"]:
        print(f"[quota] 5h window at {info['util_5h']:.0%} — starting {label}")
        return 0
    return _wait_until(problem, info["reset_5h"], label, "the 5h window is spent")


def after_session(problem: str, jsonl: Path, label: str, harness: str = "claude",
                  backend: str = "none") -> int:
    """0 = the session ended for its own reasons; 3 = it was cut off by the window,
    which has now reset — resume it; 2 = STOP appeared while waiting.

    claude: waits only when the API says the window is spent AND the session ended in
    an API error. codex: no resume; a used-up plan window is polled, then the next
    session starts (0); an account that cannot pay stops the run (2)."""
    if harness != "claude":
        kind = codex_api_failure(jsonl) if harness == "codex" else None
        if kind == "account":
            msg = (f"[quota] {label}: codex could not get answers from the API "
                   "(no credit, or the key was rejected) — stopping the run; fix the "
                   "account and re-run with RUN_MODE=continue")
            print(msg, flush=True)
            runlog.append(problem, msg)
            return 2
        if kind == "limit":
            return _codex_wait(problem, backend, label,
                               f"{label} was cut off by the codex plan window")
        return 0
    info = probe()
    if info is not None and not info["spent"]:
        return 0        # there is quota, so the window did not end this session
    if not ended_in_error(jsonl):
        return 0        # the window may be spent, but this session finished its work
    if info is None:
        print(f"[quota] {label}: ended in an API error and the window state is unknown "
              "— resuming now")
        return 3
    rc = _wait_until(problem, info["reset_5h"], label,
                     "the 5h window ran out mid-session")
    return rc if rc == 2 else 3


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentdiscover.orchestrator.quota")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("before-session", aliases=["gate"])   # `gate`: the old name
    b.add_argument("--problem", required=True)
    b.add_argument("--label", default="agent")
    b.add_argument("--harness", default="claude")
    b.add_argument("--backend", default="none")     # for the codex probe's sandbox
    a = sub.add_parser("after-session")
    a.add_argument("--problem", required=True)
    a.add_argument("--jsonl", required=True)
    a.add_argument("--label", default="agent")
    a.add_argument("--harness", default="claude")
    a.add_argument("--backend", default="none")
    args = parser.parse_args()
    try:
        if args.cmd in ("before-session", "gate"):
            sys.exit(before_session(args.problem, args.label, args.harness, args.backend))
        sys.exit(after_session(args.problem, Path(args.jsonl), args.label, args.harness,
                               args.backend))
    except Exception as e:  # noqa: BLE001 — a broken check must never block a run
        print(f"[quota] check error ({e}) — proceeding")
        sys.exit(0)


if __name__ == "__main__":
    main()
