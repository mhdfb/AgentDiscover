# PRISM — LLM model placement on a GPU cluster

You are an expert in model placement on GPUs. Improve `compute_model_placement`, which
assigns LLM models to the available GPUs of a serving cluster.

The placement must **minimise the maximum KVPR (KV-cache pressure) across all GPUs**
while every model fits within GPU memory (80 GB per GPU):

    KVPR of a GPU = sum(model.req_rate / model.slo  for model in models_on_gpu)
                    / (GPU_MEM_SIZE - sum(model.model_size  for model in models_on_gpu))

Lower KVPR means a less crowded GPU, with more memory headroom left for serving. Each
model has a size in GB (`model_size`), a request rate in requests per second
(`req_rate`), and a latency SLO in seconds (`slo`).

## Targets

You are scored on 50 test cases — clusters of a few GPUs with about twice as many models
to place — which you never see in advance; the evaluator generates them itself. For each
case it computes the maximum KVPR of your placement, and your objective is

    combined_score = 1 / mean(max KVPR over the cases) + success_rate

where the success rate is the fraction of cases your function placed validly. A case
that fails (an exception, an invalid placement, or a call that overruns its budget)
counts a KVPR of 10⁶ towards the mean and 0 towards the success rate, so one failure
costs far more than any placement can win back. **Higher is better.**

| | combined_score ↑ |
|---|---|
| the seed below (greedy by req_rate / slo) | ~21.89 |
| **target — the proven optimum** | **26.256** |

The target is not a stretch goal: it is the score of placing every one of the 50 cases
optimally, and no valid placement can exceed it. Reaching it is the whole task; the only
question is how close you get, and a candidate that leaves a case short of optimal is
short of the target by exactly that case's share. Alongside `combined_score` the
evaluator returns `avg_inv_kvpr` (1 / mean max KVPR), `success_rate`, how many cases
were scored, and the reason for every case that failed.

## Interface

`solution.py` defines

    compute_model_placement(gpu_num: int, models: list[Model]) -> dict[int, list[Model]]

The evaluator calls it once per test case and reads the mapping you return:
`gpu_id -> list of the Model objects placed on that GPU`. A valid placement uses only GPU
ids `0 .. gpu_num - 1`, places **every** model in `models` exactly once, and keeps
`sum(model.model_size) <= 80` on each GPU. A placement that drops or duplicates a model,
or names a GPU that does not exist, is not scored: it is a failed case, penalised as
above. A GPU filled to 80 GB or beyond has KVPR 10⁶. The same rule applies whether the
function raises or returns something else.

`Model` is a dataclass with the fields `model_name`, `model_size` (GB), `req_rate`
(requests/s), `slo` (s), and `cur_gpu_id`; the evaluator passes its own instances, and
your code only reads their attributes. The fixed scaffolding provides `GPU_MEM_SIZE`,
`Model`, and `max_kvpr(placement)` — the evaluator's formula — so `python3 solution.py`
runs a small demo case as it stands. NumPy is installed in your sandbox and in the
evaluator; nothing else is.

**Compute budget: each call gets 10 s, and all 50 calls together must fit in 300 s** —
about 6 s per case on average. A call that overruns its 10 s is a failed case. The
evaluation as a whole is capped at 360 s, and that is the limit that binds a slow
method: an expensive search has to earn its cost across every case. Your briefing repeats
the numbers in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- exact methods: formulate each case as a constraint or integer program over a fixed pressure threshold and solve it outright, tightening the threshold
- local search: start from a complete placement and improve it by moving and swapping models between GPUs
- constructive heuristics: build the placement one model at a time, scoring each choice by the pressure it leaves on the worst GPU
- portfolio and restarts: run several randomised or differently ordered constructions per case and keep the best
