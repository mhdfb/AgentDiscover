"""Paths to the CloudCast benchmark data, importable by solution.py and the evaluator.

The data itself (profiles/*.csv, examples/config/*.json) is SkyDiscover's, downloaded by
their own download_dataset.sh from the adrs-data dataset. A copy sits next to this file;
a read-only copy is also mounted for agents at /resources.
"""
import json
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
COST_CSV = os.path.join(DATA_DIR, "profiles", "cost.csv")
THROUGHPUT_CSV = os.path.join(DATA_DIR, "profiles", "throughput.csv")

# The five network configurations SkyDiscover's evaluator runs, in its order.
CONFIG_NAMES = ("intra_aws", "intra_azure", "intra_gcp", "inter_agz", "inter_gaz2")

# Fixed across the whole benchmark, from SkyDiscover's evaluator (`num_vms = 2`).
NUM_VMS = 2


def load_configs():
    """[(name, config dict)] for the five evaluation configurations."""
    out = []
    for name in CONFIG_NAMES:
        with open(os.path.join(DATA_DIR, "examples", "config", f"{name}.json")) as f:
            out.append((name, json.load(f)))
    return out
