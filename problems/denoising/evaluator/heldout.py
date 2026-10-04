"""Operator-side held-out scoring: a denoiser on the benchmark's PBMC and Tabula Muris
Senis (lung) data sets, the way TTT-Discover's results/denoising/benchmark_denoising.ipynb
reports its found algorithm. Not part of the search evaluator and not mounted anywhere an
agent can read; run it on the candidate the search ends with.

Reproduces the notebook step for step with the vendored openproblems v1.0.0 code:
  - PBMC: `data.tenx.load_tenx_1k_pbmc` (figshare 36088667), empty genes/cells removed;
  - Tabula: `data.tabula_muris_senis.load_tabula_muris_senis(organ_list=["lung"],
    method_list=["droplet"])` over the CZI cellxgene API, with TTT-Discover's API patch;
  - `tasks.denoising.datasets.utils.split_data(adata)` at the benchmark's default seed 0;
  - the benchmark's `mse` and `poisson` metrics (identical to evaluate_mse /
    evaluate_poisson), normalised against `no_denoising` and `perfect_denoising` on the
    same split, and averaged into the benchmark's score.
The candidate is called as the notebook calls it: `magic_denoise(X_train)` on the dense
training matrix, default parameters, no seed.

Usage, with the bundle's requirements on PYTHONPATH and network access (the files are
cached under --cache, ~1 GB):

    python3 heldout.py solution.py [--cache DIR] [--datasets pbmc,tabula]

Notebook reference values for MAGIC (approximate solver, reversed normalisation), to
check the pipeline: PBMC MSE 0.1888 / Poisson 0.0495 (normalised 0.30 / 0.98, score
0.64); Tabula MSE 0.1841 / Poisson 0.0297 (0.30 / 0.98, 0.64).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openproblems_vendored import (evaluate_mse, evaluate_poisson,  # noqa: E402
                                   filter_genes_cells, normalize_score, split_data)

PBMC_1K_URL = "https://ndownloader.figshare.com/files/36088667"
# openproblems/data/tabula_muris_senis.py
COLLECTION_ID = "0b9d8a04-bb9d-44da-aa27-705bb65b54eb"
API_BASE = "https://api.cellxgene.cziscience.com"
METHOD_ALIASES = {"10x 3' v2": "droplet", "Smart-seq2": "facs"}


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)


def load_pbmc(cache: Path):
    """`load_tenx_1k_pbmc(test=False)`."""
    import scanpy as sc

    f = cache / "pbmc_1k.h5ad"
    _download(PBMC_1K_URL, f)
    adata = sc.read_h5ad(str(f))
    filter_genes_cells(adata)
    return adata


def _get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"Content-Type": "application/json"})) as r:
        return json.load(r)


def load_tabula_lung_droplet(cache: Path):
    """`load_tabula_muris_senis(organ_list=["lung"], method_list=["droplet"])`, with the
    cellxgene API as TTT-Discover's openproblems_api_fix.patch reads it (dataset_id,
    embedded assets, plain url)."""
    import anndata as ad
    import scanpy as sc

    datasets = _get_json(f"{API_BASE}/curation/v1/collections/{COLLECTION_ID}")["datasets"]
    parts = []
    for d in datasets:
        if len(d["assay"]) > 1 or len(d["tissue"]) > 1:
            continue
        method = METHOD_ALIASES[d["assay"][0]["label"]]
        if d["tissue"][0]["label"].lower() != "lung" or method != "droplet":
            continue
        assets = [a for a in d["assets"] if a["filetype"] == "H5AD"]
        assert len(assets) == 1
        f = cache / f"{COLLECTION_ID}_{d['dataset_id']}.h5ad"
        _download(assets[0]["url"], f)
        adata = sc.read_h5ad(str(f))
        filter_genes_cells(adata)
        if getattr(adata, "raw", None) is not None:
            adata = adata.raw.to_adata()
        parts.append(adata)
    assert parts, "no lung/droplet data set in the collection"
    adata = ad.concat(parts, join="outer")
    del adata.obs["is_primary_data"]
    return adata


def score(adata, denoise) -> dict:
    """The notebook's per-dataset table: raw and normalised MSE/Poisson and their mean,
    for the candidate, against the no/perfect-denoising baselines on this split."""
    import scprep

    adata = split_data(adata)                       # seed 0, the benchmark's default
    X_train = scprep.utils.toarray(adata.obsm["train"])
    X_test = scprep.utils.toarray(adata.obsm["test"])

    def metrics(Y):
        return float(evaluate_mse(X_test, Y)), float(evaluate_poisson(X_train, X_test, Y))

    mse_none, poi_none = metrics(X_train)
    mse_perf, poi_perf = metrics(X_test)
    t0 = time.perf_counter()
    Y = np.asarray(denoise(X_train), dtype=np.float64)
    secs = time.perf_counter() - t0
    mse, poi = metrics(Y)
    mse_n = normalize_score(mse, mse_none, mse_perf)
    poi_n = normalize_score(poi, poi_none, poi_perf)
    return {"cells": int(X_train.shape[0]), "genes": int(X_train.shape[1]),
            "mse": round(mse, 6), "poisson": round(poi, 6),
            "mse_normalized": round(mse_n, 4), "poisson_normalized": round(poi_n, 4),
            "score": round((mse_n + poi_n) / 2, 4), "seconds": round(secs, 1),
            "baselines": {"no_denoising": [round(mse_none, 6), round(poi_none, 6)],
                          "perfect": [round(mse_perf, 6), round(poi_perf, 6)]}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("solution")
    p.add_argument("--cache", default=os.environ.get("AGENTDISCOVER_DATA_DIR") or
                   os.path.join(tempfile.gettempdir(), "openproblems_heldout"))
    p.add_argument("--datasets", default="pbmc,tabula")
    a = p.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    ns = {"__name__": "__candidate__"}
    exec(Path(a.solution).read_text(), ns)
    denoise = ns["magic_denoise"]

    loaders = {"pbmc": load_pbmc, "tabula": load_tabula_lung_droplet}
    out = {}
    for name in a.datasets.split(","):
        out[name] = score(loaders[name](cache), denoise)
        print(f"[{name}] {json.dumps(out[name])}", file=sys.stderr, flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
