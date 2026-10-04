#!/bin/sh
# Fetch this problem's data set into the directory given as $1. Run once per machine by
# the platform, with network, before any candidate exists; the result is cached and
# mounted read-only into every evaluation as $AGENTDISCOVER_DATA_DIR.
#
# 224 MB of load metrics recorded from a vLLM server, the same file SkyDiscover's and
# CORAL's EPLB benchmarks score against. Too large for the repository, hence this script.
set -eu

DEST="$1/expert-load.json"
URL="https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json"

# Download to a temporary name first, so an interrupted fetch can never leave a truncated
# file that looks complete.
curl -fL --retry 3 --retry-delay 5 -o "$DEST.part" "$URL"
mv "$DEST.part" "$DEST"
echo "fetched $(wc -c < "$DEST") bytes to $DEST"
