#!/bin/sh
# Source: SakanaAI/ALE-Bench dataset revision 5489d98badbe48ec3e632ec145e8cb22f51d86bb,
# ahc006.zip. Keep its Rust generator/tester unchanged and generate seeds 0..49,
# matching SkyDiscover's case_gen(list(range(50))) search evaluation.
set -eu

destination=$1
archive="$destination/ahc006.zip"
curl -L --fail --silent --show-error \
  "https://huggingface.co/datasets/SakanaAI/ALE-Bench/resolve/5489d98badbe48ec3e632ec145e8cb22f51d86bb/ahc006.zip" \
  -o "$archive"
python3 - "$archive" "$destination" "ahc006" <<'PYDATA'
import json
import sys
import zipfile
from pathlib import Path
archive, destination, problem = sys.argv[1:]
root = Path(destination)
with zipfile.ZipFile(archive) as source:
    for name in source.namelist():
        parts = [part for part in name.split("/") if part]
        if len(parts) < 2 or parts[0] != problem:
            continue
        if parts[1] != "data.json" and parts[1] != "tools":
            continue
        target = root.joinpath(*parts[1:])
        if name.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(name))
metadata = json.loads((root / "data.json").read_text())
assert metadata["metadata"]["problem_id"] == problem
assert metadata["seeds"]["public"] == list(range(50))
(root / "seeds.txt").write_text("".join(f"{seed}\n" for seed in range(50)))
PYDATA

# The ALE judge image provides C++20 but no Rust compiler. Build the original tools
# during this network-enabled, genome-free setup phase. The tools and inputs are
# later mounted read-only; candidate evaluation itself has no network.
export RUSTUP_HOME=/tmp/ale-rustup
export CARGO_HOME=/tmp/ale-cargo
export CARGO_TARGET_DIR=/tmp/ale-target
curl -L --fail --silent --show-error \
  https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
  -o /tmp/rustup-init
chmod +x /tmp/rustup-init
/tmp/rustup-init -y --profile minimal --default-toolchain 1.98.1 --no-modify-path
"$CARGO_HOME/bin/cargo" build --manifest-path "$destination/tools/Cargo.toml" \
  --release --locked --bin gen --bin tester
mkdir -p "$destination/bin"
cp "$CARGO_TARGET_DIR/release/gen" "$destination/bin/gen"
cp "$CARGO_TARGET_DIR/release/tester" "$destination/bin/tester"
(cd "$destination" && "$destination/bin/gen" "$destination/seeds.txt")
test -f "$destination/in/0000.txt"
test -f "$destination/in/0049.txt"
rm "$archive" "$destination/data.json"
