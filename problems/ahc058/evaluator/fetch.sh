#!/bin/sh
set -eu

destination=$1
archive="$destination/cache_ahc.zip"
wget --no-check-certificate \
  "https://drive.google.com/uc?export=download&id=1bA044QSbhsQWLjgs467ygoCpoxH3NevD" \
  -O "$archive"
unzip -q "$archive" -d "$destination"
rm "$archive"
test -x "$destination/cache/tester_binaries/ahc058_tester"
test -f "$destination/cache/public_inputs_150/ahc058_abeaef4e6038cc60.json"
