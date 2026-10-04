"""Second half of fetch.sh: cut the downloaded pancreas file to the object the benchmark
scores on. Runs once, in the fetch step, with the bundle's libraries on PYTHONPATH.

`load_pancreas(test=False, keep_techs=["inDrop1"])` in openproblems v1.0.0 is cached by
its `@loader` decorator: the first call downloads and writes the inDrop1 subset (counts
in X, empty genes and cells removed) to an h5ad, and every later call reads that file.
TTT-Discover's evaluations read the cached file. This script writes the same object to
`pancreas_indrop1.h5ad`, and the evaluator reads it as they did.

Usage: fetch_prepare.py <data dir holding pancreas.h5ad>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openproblems_vendored import load_pancreas  # noqa: E402

data = Path(sys.argv[1])
adata = load_pancreas(str(data / "pancreas.h5ad"), keep_techs=["inDrop1"])
# openproblems.data.utils.write_h5ad → _fix_adata: categoricals and the counts layer are
# already in place; the subset is written as is.
adata.strings_to_categoricals()
out = data / "pancreas_indrop1.h5ad"
adata.write_h5ad(out)
print(f"prepared {out}: {adata.n_obs} cells x {adata.n_vars} genes")
