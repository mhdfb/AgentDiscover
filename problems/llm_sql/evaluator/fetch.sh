#!/bin/sh
# Fetch this problem's data set into the directory given as $1. Run once per machine by
# the platform, with network, before any candidate exists; the result is cached and
# mounted read-only into every evaluation as $AGENTDISCOVER_DATA_DIR.
#
# The five CSV tables SkyDiscover's `benchmarks/ADRS/llm_sql/evaluator/download_dataset.sh`
# fetches, from the same HuggingFace repository, so the numbers here are measured on the
# same data CORAL's are. ~69 MB in total — too large to keep in the repository. The
# checksums are those of the files as downloaded on 2026-09-06; a mismatch means the
# upstream files changed and the comparison would no longer hold, so it fails loudly.
set -eu

DEST="$1/datasets"
BASE="https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/llm_sql/datasets"
mkdir -p "$DEST"

for f in movies.csv beer.csv BIRD.csv PDMX.csv products.csv; do
    # Download to a temporary name first, so an interrupted fetch can never leave a
    # truncated file that looks complete.
    curl -fL --retry 3 --retry-delay 5 -o "$DEST/$f.part" "$BASE/$f"
    mv "$DEST/$f.part" "$DEST/$f"
done

cd "$DEST"
md5sum -c - <<'EOF'
3688e17c1abc02ac53606b8712def347  movies.csv
5a13db468f5ea86044ed0f147ca18b53  beer.csv
07b1b4e707d68e174ef8b6d467daac09  BIRD.csv
38db6bd964e11f1ab9e85b6e98e03bc7  PDMX.csv
dbc34d5fd50d2413d96812a13356cc6b  products.csv
EOF
echo "fetched $(du -sh . | cut -f1) of data sets to $DEST"
