# Single-cell RNA-seq denoising — the OpenProblems benchmark

You are an expert in computational biology and single-cell RNA-seq analysis.
Your task is to develop a denoising algorithm for scRNA-seq count data. You are experienced in
compuational biology libraries and tools and are familiar with problems in denoising in the single-cell field.

## Problem

Single-cell RNA-seq data is noisy due to technical dropout and low capture efficiency.
Given noisy count data, predict the true expression levels.

Your prediction is evaluated against held-out molecules using two metrics:
1. **MSE** - Mean Squared Error in log-normalized space
2. **Poisson Loss** - Poisson negative log-likelihood

You need to implement a novel denoising algorithm that outperforms the current state-of-the-art without overfitting.

## Data Format

- Input `X`: numpy array of shape (n_cells, n_genes) - **raw count data**
- Output: numpy array of same shape - your denoised counts

## Evaluation

Your output is evaluated using these exact functions:

```python
def evaluate_mse(test_data, denoised):
    test_X = scprep.utils.toarray(test_data).copy()
    denoised_X = np.asarray(denoised).copy()

    test_adata = anndata.AnnData(X=test_X)
    denoised_adata = anndata.AnnData(X=denoised_X)

    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)

    return sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X)
```

```python
def evaluate_poisson(train_data, test_data, denoised):
    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()

    initial_sum = train_data.sum()
    target_sum = test_X.sum()
    denoised_scaled = denoised_X * target_sum / initial_sum

    return poisson_nll_loss(test_X, denoised_scaled)
```

## Scoring

**Poisson is a HARD CONSTRAINT.** Your solution is REJECTED if `poisson_norm < 0.97`.
- `poisson_norm = (0.257575 - poisson) / (0.257575 - 0.031739)`
- MAGIC baseline achieves ≈0.97

**Reward = MSE score only** (after passing Poisson constraint).

## Budget & Resources

- **Time budget**: 400s for your code to run. You should time your code and make sure it runs within the time budget.
- **CPUs**: 1 available

## Function Signature to return

```python
def magic_denoise(X, **kwargs):
    # kwargs may include: budget_s, random_state, knn, t, n_pca, solver, decay, knn_max, n_jobs
    # You can add your own parameters too
    # Your implementation
    return denoised_X  # same shape as X
```

## Rules

- Implement `magic_denoise(X, ...)` that returns denoised data
- Use numpy, scipy, sklearn, graphtools, scprep, scanpy
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO

## Key Insights from Benchmarks

- NORMALIZATION ORDER MATTERS: Denoise raw/log counts first, then normalize. "Reversed normalization order" achieves Poisson ~0.98 vs ~0.55 for standard order.
- Square root transform is variance-stabilizing for Poisson distributions
- Poisson loss is highly affected by low non-zero values - push values < 1 toward zero
- The original MAGIC with reversed normalization achieves best results

## Targets

Your objective is the **MSE** above, on the benchmark's pancreas data set (inDrop1
batch: 1,937 cells, 15,502 genes before the split), scored on the held-out 10 % of
molecules. **Lower is better.** The Poisson gate must hold or the candidate scores 0.

| | MSE ↓ | Poisson |
|---|---|---|
| the starting program below (MAGIC, reversed normalisation) | 0.231412 | 0.036922 |
| **target — beat this, clearly** | **0.165521** | 0.032741 |

The target is the best published result on this split, a bar to get **meaningfully
past**, not to land on. The fitness the platform ranks on is `0.165521 / mse` (1.0 at the
target, above 1 past it); the evaluator also returns `poisson`, the normalised `mse_normalized` and
`poisson_normalized`, and the benchmark's own `openproblems_score` (their mean).

The score you see is on the pancreas data set only. The final evaluation runs your
program, unchanged, on two other data sets it has never seen — human PBMC (10x v3,
~1,000 cells) and mouse lung (Tabula Muris Senis, ~24,000 cells) — with different
protocol, sparsity and size. Prefer general improvements over anything tuned to this
data set.

## Interface

`solution.py` is the whole submission: the evaluator executes the file and calls
`magic_denoise(X_train, random_state=42)` with the dense float64 training matrix. All
helpers at top level, inside the EVOLVE block. The starting file is the current
implementation, MAGIC with reversed normalisation.

Your function runs alone in its own process, pinned to one CPU core with every thread
pool (BLAS, OpenMP, numba) capped at one thread, for at most **400 s** — the clock starts
after the allowed libraries are imported, so their load time is not yours; the whole
evaluation is capped at 1200 s. The result must be finite, non-negative, and no entry may
exceed the training matrix's total count. The held-out matrix never enters your process:
the split and both metrics are computed by the evaluator afterwards.

## Local tools

The scoring path is readable at `/resources/evaluator.py`, with the data loading, the
split and the two metrics it uses in `/resources/openproblems_vendored.py` (OpenProblems
v1.0.0 and the metrics above, verbatim). The data set (the inDrop1 batch, as the
benchmark's loader caches it) is mounted read-only at
`$AGENTDISCOVER_DATA_DIR/pancreas_indrop1.h5ad`, and your sandbox has the same library versions as
the evaluator, so

    python3 /resources/evaluator.py solution.py

runs the exact evaluation locally — same split, same seed — and prints the JSON line the
platform would return.
