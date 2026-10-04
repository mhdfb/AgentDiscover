#!/usr/bin/env bash
# Build the platform image locally with Singularity/Apptainer, for a machine that has no
# Docker and cannot pull the published image. Needs fakeroot (an entry in /etc/subuid).
# The SIF is written to the image cache, so a plain ./run.sh then uses it.
#
#   ./scripts/build-image-singularity.sh                  # for the default image ref
#   AGENTDISCOVER_IMAGE=ghcr.io/me/agentdiscover:v2 ./scripts/build-image-singularity.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUNTIME="${SANDBOX_BACKEND:-}"
if [[ -z "$RUNTIME" || "$RUNTIME" == none ]]; then
    if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer; else RUNTIME=singularity; fi
fi
REF="${AGENTDISCOVER_IMAGE:-ghcr.io/mhdfb/agentdiscover:latest}"
# Must match agentdiscover.sandbox._slug: every run of other characters becomes "_".
SLUG=$(printf '%s' "$REF" | sed 's/[^A-Za-z0-9._-]\{1,\}/_/g')
OUT="runs/.images/$SLUG.sif"

mkdir -p runs/.images
echo "building $OUT from Singularity.def with $RUNTIME (fakeroot) ..."
"$RUNTIME" build --fakeroot --force "$OUT" Singularity.def
echo "built $OUT — runs referring to $REF now use it"
