"""The OpenProblems denoising benchmark's data path and TTT-Discover's two metrics,
vendored verbatim so the evaluation needs neither the `openproblems` package (v1.0.0
pins numpy < 1.24 and pandas 1.3.5) nor `molecular-cross-validation` (which pulls torch).
Every function below is a copy; the source is named above each one.

The pancreas file itself is downloaded once by fetch.sh (figshare 36086813, the URL in
openproblems/data/pancreas.py) and read here.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse

# ---------------------------------------------------------------------------
# openproblems v1.0.0 — openproblems/data/pancreas.py `load_pancreas` (the part after
# the download) and openproblems/data/utils.py `filter_genes_cells`.
# ---------------------------------------------------------------------------


def filter_genes_cells(adata):
    """Remove empty cells and genes."""
    import scanpy as sc

    if "var_names_all" not in adata.uns:
        # fill in original var names before filtering
        adata.uns["var_names_all"] = adata.var.index.to_numpy()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_cells(adata, min_counts=2)


def load_pancreas(filepath, keep_techs=None):
    """`load_pancreas(test=False, keep_techs=...)` from the downloaded file onward."""
    import scanpy as sc

    adata = sc.read(filepath)

    if keep_techs is not None:
        adata = adata[adata.obs["tech"].isin(keep_techs)].copy()

    # NOTE: adata.X contains log-normalized data, so we're moving it
    adata.layers["log_normalized"] = adata.X
    adata.X = adata.layers["counts"]

    # Ensure there are no cells or genes with 0 counts
    filter_genes_cells(adata)

    return adata


# ---------------------------------------------------------------------------
# molecular-cross-validation (czbiohub) — molecular_cross_validation/util.py
# `split_molecules` and molecular_cross_validation/mcv_sweep.py `poisson_nll_loss`.
# ---------------------------------------------------------------------------


def split_molecules(
    umis: np.ndarray,
    data_split: float,
    overlap_factor: float = 0.0,
    random_state: np.random.RandomState = None,
):
    """Splits molecules into two (potentially overlapping) groups.

    :param umis: Array of molecules to split
    :param data_split: Proportion of molecules to assign to the first group
    :param overlap_factor: Overlap correction factor, if desired
    :param random_state: For reproducible sampling
    :return: umis_X and umis_Y, representing ``split`` and ``~(1 - split)`` counts
             sampled from the input array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    umis_X_disjoint = random_state.binomial(umis, data_split - overlap_factor)
    umis_Y_disjoint = random_state.binomial(
        umis - umis_X_disjoint, (1 - data_split) / (1 - data_split + overlap_factor)
    )
    overlap_factor = umis - umis_X_disjoint - umis_Y_disjoint
    umis_X = umis_X_disjoint + overlap_factor
    umis_Y = umis_Y_disjoint + overlap_factor

    return umis_X, umis_Y


def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()


# ---------------------------------------------------------------------------
# openproblems v1.0.0 — openproblems/tasks/denoising/datasets/utils.py `split_data`,
# with TTT-Discover's one-line patch (requirements/denoising/openproblems_api_fix.patch):
# `np.int` → `int`, which numpy 2 requires.
# ---------------------------------------------------------------------------


def split_data(adata, train_frac: float = 0.9, seed: int = 0):
    """Split data using molecular cross-validation.

    Stores "train" and "test" dataset using the AnnData.obsm property.
    """
    random_state = np.random.RandomState(seed)

    X = adata.X

    if scipy.sparse.issparse(X):
        X = np.array(X.todense())
    if np.allclose(X, X.astype(int)):
        X = X.astype(int)
    else:
        raise TypeError("Molecular cross-validation requires integer count data.")

    X_train, X_test = split_molecules(X, 0.9, 0.0, random_state)
    # remove zero entries
    is_missing = X_train.sum(axis=0) == 0
    X_train, X_test = X_train[:, ~is_missing], X_test[:, ~is_missing]

    adata = adata[:, ~is_missing].copy()
    adata.obsm["train"] = scipy.sparse.csr_matrix(X_train).astype(float)
    adata.obsm["test"] = scipy.sparse.csr_matrix(X_test).astype(float)

    return adata


# ---------------------------------------------------------------------------
# TTT-Discover — examples/denoising/utils.py: the two metrics the task statement shows
# the agent, and the per-dataset normalisation constants of the benchmark.
# ---------------------------------------------------------------------------

BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    },
    "pbmc": {
        "baseline_mse": 0.270945,
        "baseline_poisson": 0.300447,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.043569,
    },
    "tabula": {
        "baseline_mse": 0.261763,
        "baseline_poisson": 0.206542,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.026961,
    },
}


def evaluate_mse(test_data, denoised):
    import scprep
    import anndata
    import scanpy as sc
    import sklearn.metrics

    test_X = scprep.utils.toarray(test_data).copy()
    denoised_X = np.asarray(denoised).copy()

    test_adata = anndata.AnnData(X=test_X)
    denoised_adata = anndata.AnnData(X=denoised_X)

    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)

    return sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X)


def evaluate_poisson(train_data, test_data, denoised):
    import scprep

    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()

    initial_sum = train_data.sum()
    target_sum = test_X.sum()
    denoised_scaled = denoised_X * target_sum / initial_sum

    return poisson_nll_loss(test_X, denoised_scaled)


def normalize_score(score, worst, best):
    if worst == best:
        return 0.0
    return (worst - score) / (worst - best)
