"""The candidate's own process. It is handed ONE workload at a time and knows nothing
else — no data set, no evaluator, and no caller frame holding the future.

Protocol on stdin/stdout, length-prefixed torch payloads:

    parent -> child : 4-byte length + torch.save({"weight": tensor})   (or length 0 = quit)
    child  -> parent: 4-byte length + torch.save({"log2phy":…, "logcnt":…})
                      or 4-byte length + torch.save({"error": "..."})

Started once and reused, so the torch import is paid once and the per-call timing the
parent measures is the candidate's work plus a pipe round trip.

Why a separate process at all: with the candidate running inside the evaluator, a genome
walked the call stack (`sys._getframe`, `f_back`, `f_locals`) into the evaluator's frames,
found the list of workloads, and read the NEXT one — the very thing it is asked to predict.
Here the future workload is not in this process, so there is nothing to find.
"""
import importlib.util
import io
import struct
import sys

import torch


def _read(stream) -> dict | None:
    header = stream.read(4)
    if not header or len(header) < 4:
        return None
    (size,) = struct.unpack("!I", header)
    if size == 0:
        return None
    payload = stream.read(size)
    return torch.load(io.BytesIO(payload), weights_only=False)


def _write(stream, obj: dict) -> None:
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    stream.write(struct.pack("!I", len(data)))
    stream.write(data)
    stream.flush()


def _narrow(t: torch.Tensor) -> torch.Tensor:
    """int16 instead of int64 for the maps sent back.

    Every value is a physical-expert index or a replica count, both bounded by
    NUM_REPLICAS = 288, so 16 bits are ample. It matters because this crosses a pipe on
    every call and the timing of that round trip is part of the score: as int64 the
    log2phy map is 3.9 MB per call, which measurably slowed the candidate's own timing.
    """
    return t.to(torch.int16) if t.dtype != torch.int16 else t


def main() -> None:
    path, num_replicas, num_groups, num_nodes, num_gpus = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    # The candidate may print; keep stdout clean for the protocol.
    sys.stdout = sys.stderr

    spec = importlib.util.spec_from_file_location("candidate", path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "rebalance_experts"):
        _write(stdout, {"error": "solution.py defines no rebalance_experts function"})
        return

    while True:
        msg = _read(stdin)
        if msg is None:
            return
        try:
            out = program.rebalance_experts(
                msg["weight"], num_replicas, num_groups, num_nodes, num_gpus)
            _, log2phy, logcnt = out
            _write(stdout, {"log2phy": _narrow(torch.as_tensor(log2phy)),
                            "logcnt": _narrow(torch.as_tensor(logcnt))})
        except Exception as e:  # noqa: BLE001 — reported, and the parent scores it 0
            # Only the exception's class crosses to the parent. Its message is the
            # candidate's own text and must never reach the agent (it could carry the
            # workload out); the full traceback goes to stderr for the operator.
            import traceback
            traceback.print_exc()
            _write(stdout, {"error": type(e).__name__})
            return


if __name__ == "__main__":
    main()
