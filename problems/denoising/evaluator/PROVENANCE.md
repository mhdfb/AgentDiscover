# denoising — provenance (operator only)

This file is **not** mounted anywhere an agent can read. It records where the task comes
from, what was copied unchanged, and what differs from the setting TTT-Discover reports.

## Where the task comes from

The OpenProblems denoising benchmark (Luecken et al. 2025, "Defining and benchmarking
open problems in single-cell analysis"), as used by "Learning to Discover at Test Time"
(TTT-Discover, arXiv:2601.16175, §Single Cell Analysis): `examples/denoising/` in their
repository.

- `PROBLEM.md` is their `SYSTEM_PROMPT` (examples/denoising/prompt.py) with the two
  metric functions inlined, as their `get_question` does, plus the platform's Targets,
  Interface and Local tools sections. One addition their model was not given: a note that
  the final evaluation is on PBMC and Tabula, so general improvements are wanted
  (2026-10-01, the user's call, in place of mixing those sets into the search score —
  which would have broken their held-out protocol). One number changed: "CPUs: 2 available" became
  "1 available" — their `discover_denoising()` config sets `num_cpus_per_task=1`, and
  their sandbox pins each evaluation to a one-core CPU group with one-thread pools
  (`sandbox_reward_evaluator.py`); the prompt's "2" is what the released prompt file
  hard-codes, the paper's template has a placeholder. We state and enforce the one core.
- `solution.py` is their initial state, `MAGIC_FUNC` (examples/denoising/utils.py,
  `magic_denoise`), verbatim: MAGIC with reversed normalisation, the benchmark's best
  entry.
- `openproblems_vendored.py` carries, verbatim: openproblems v1.0.0 `load_pancreas`
  (the part after the download) and `filter_genes_cells`; `split_data` with their one
  patch (`np.int` → `int`); molecular-cross-validation's `split_molecules` and
  `poisson_nll_loss`; their `evaluate_mse`, `evaluate_poisson`, `BASELINES`. Vendored
  because `openproblems` v1.0.0 pins numpy < 1.24 / pandas 1.3.5 (they install it with
  `--no-deps`) and `molecular-cross-validation` pulls torch.
- `requirements.txt` is their pin list for the single-cell libraries
  (requirements/denoising/requirements-denoising.txt). They used Python 3.11; the
  platform image is 3.13 — same package versions, see Verification.
- The data set: figshare 36086813, the URL in openproblems/data/pancreas.py, fetched by
  fetch.sh; inDrop1 batch, empty genes/cells removed; split with seed 42
  (`run_denoising_eval(..., seed=42)`).

## Published numbers (withheld from the agent except the target)

OpenProblems denoising, score = mean of normalised MSE and normalised Poisson, higher
is better (TTT-Discover Table "Denoising"), on the two HELD-OUT data sets — they searched
on pancreas and report the found algorithm on pbmc and tabula:

| | PBMC score | PBMC MSE | PBMC Poisson | Tabula score | Tabula MSE | Tabula Poisson |
|---|---|---|---|---|---|---|
| MAGIC (approx., reversed) | 0.64 | 0.19 | 0.05 | 0.64 | 0.18 | 0.03 |
| ALRA (sqrt, reversed) | 0.50 | 0.26 | 0.05 | 0.47 | 0.27 | 0.03 |
| OpenEvolve (gpt-oss-120b) | 0.70 | 0.16 | 0.05 | 0.71 | 0.15 | 0.03 |
| Best-of-25600 (gpt-oss-120b) | 0.62 | 0.20 | 0.05 | 0.65 | 0.18 | 0.03 |
| TTT-Discover (gpt-oss-120b) | **0.71** | **0.15** | 0.05 | **0.73** | **0.14** | 0.03 |

Their search metric on pancreas: initial state MSE 0.2316, Poisson 0.0370
(`create_initial_state`). Their found algorithm is `results/denoising/denoise_ttt.py`
in their repository.

## What differs from TTT-Discover's setting

1. **Process layout.** Theirs concatenates the metric functions, the data loading and
   the generated code into one program and runs it in one sandboxed process (the
   candidate could, in principle, reach the test matrix). Ours runs the candidate's
   process on the training matrix only and scores in the parent — stricter, same
   numbers.
2. **Memory.** Their ray task reserves 1 GB and the paper states a 3 GB limit, as
   resident memory. Our cap is address space (12 GB), so a candidate above 3 GB resident
   is not killed here. A candidate near that limit would be unusual for a 240 MB matrix.
3. **Time.** Same 400 s statement; their whole program had 530 s (data load included),
   ours gives the candidate 400 s on its own clock and the evaluation 600 s.
4. **Tool use.** Their policy only emitted code; ours can run the evaluator locally on
   the same split (`mount_data`). Equivalent to the AHC tasks' local tester.

## Held-out reporting (to do before a paper number)

TTT-Discover reports the found algorithm on PBMC and Tabula, not on pancreas. Their
`results/denoising/benchmark_denoising.ipynb` scores it exactly as the benchmark does:
the openproblems v1.0.0 dataset functions `tasks.denoising.datasets.pbmc.pbmc(test=False)`
(`data.tenx.load_tenx_1k_pbmc`) and
`tasks.denoising.datasets.tabula_muris_senis.tabula_muris_senis_lung_random(test=False)`
(`data.tabula_muris_senis.load_tabula_muris_senis(organ_list=["lung"],
method_list=["droplet"])`, which needs their CZI API patch), each split with
`utils.split_data(adata)` at the benchmark's default **seed 0** (not 42), then the
benchmark's own `metrics.mse.mse` / `metrics.poisson.poisson`, normalised against
`no_denoising` and `perfect_denoising` run on the same split, and averaged. Their
notebook's reference values for MAGIC: PBMC MSE 0.1888 / Poisson 0.0495, Tabula MSE
0.1841 / Poisson 0.0297 (normalised 0.30 / 0.98, mean 0.64 on both). The candidate the
search ends with has to be re-scored this way to fill a comparable row. Not part of the
search evaluator.

## Verification

`python3 evaluator.py solution.py` (the seed) must reproduce TTT-Discover's initial
state on the same split: MSE 0.2316, Poisson 0.0370. Verified 2026-10-01 through the
sandboxed path on a compute node: MSE 0.231412, Poisson 0.036922, 1,937 cells x 15,501
genes, MAGIC itself 14.6 s, the whole evaluation ~5 min (library imports off NFS, ~2 min
per process — the login node is ten times slower still; run only on compute nodes).
`TARGET_MSE` was that measured value until 2026-10-01; it is now **0.165521**, the MSE our
evaluator measures for TTT-Discover's published final program (`results/denoising/
denoise_ttt.py`, same split, Poisson 0.032741, 39 s on one core; log
`runs/.tmp/verify/denoise_ttt.log`) — the user wants their number as the target, as on
the kernel tasks. Fitness 1.0 therefore means matching their program on pancreas. A different number means the data, the split or a
library version differs — do not run the search before this matches.
