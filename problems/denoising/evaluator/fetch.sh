#!/bin/sh
# Fetch this problem's data set into the directory given as $1. Run once per machine by
# the platform, with network and with the bundle's requirements on PYTHONPATH, before
# any candidate exists; the result is cached and mounted read-only into every evaluation
# (and, for this problem, into the agent's sandbox) as $AGENTDISCOVER_DATA_DIR.
#
# The OpenProblems pancreas data set: the sparsified h5ad openproblems v1.0.0 downloads
# in openproblems/data/pancreas.py (figshare 36086813, from Luecken et al. 2022,
# "Benchmarking atlas-level data integration in single-cell genomics"), then cut to the
# inDrop1 batch by fetch_prepare.py exactly as load_pancreas does — the object their
# loader caches and every evaluation of theirs reads.
set -eu

DEST="$1/pancreas.h5ad"
URL="https://ndownloader.figshare.com/files/36086813"

# Download to a temporary name first, so an interrupted fetch can never leave a truncated
# file that looks complete; then check it is an HDF5 file.
curl -fL --retry 3 --retry-delay 5 -o "$DEST.part" "$URL"
magic="$(head -c 8 "$DEST.part" | od -An -c | tr -d ' \n')"
case "$magic" in
    *HDF*) ;;
    *) echo "fetch.sh: $DEST.part is not an HDF5 file" >&2; exit 1 ;;
esac
mv "$DEST.part" "$DEST"
echo "fetched $(wc -c < "$DEST") bytes to $DEST"

python3 "$(dirname "$0")/fetch_prepare.py" "$1"
# The 900 MB download is not needed after the cut; the fetch is reproducible.
rm -f "$DEST"
