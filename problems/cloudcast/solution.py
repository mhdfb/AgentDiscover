"""CloudCast: broadcast data across clouds at minimum total transfer cost. See PROBLEM.md.

The mutable region is marked by # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END. Everything
else in this file is fixed scaffolding — do not modify it.

`cloudcast_data` (data paths, the five configurations, NUM_VMS) is importable both here
and inside the evaluator; read-only copies of the data are also at /resources.
"""
from cloudcast_data import COST_CSV, THROUGHPUT_CSV, NUM_VMS, load_configs


def solve() -> dict:
    """Return {config_name: paths} for the five evaluation configurations.

    `paths` is BroadCastTopology.paths: dict(dst) -> dict(partition str) -> list of
    [src, dst, edge data] hops. Scored by the total simulated transfer cost across all
    five configurations — lower is better.
    """
    out = {}
    for name, cfg in load_configs():
        G = make_nx_graph(COST_CSV, THROUGHPUT_CSV, num_vms=NUM_VMS)
        bc = search_algorithm(cfg["source_node"], cfg["dest_nodes"], G, cfg["num_partitions"])
        out[name] = bc.paths
    return out


# EVOLVE-BLOCK-START
import networkx as nx
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """Set paths for partition = partition to reach dst"""
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """Append path for partition = partition to reach dst"""
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
    import pandas as pd

    cost = pd.read_csv(cost_path if cost_path is not None else COST_CSV)
    throughput = pd.read_csv(throughput_path if throughput_path is not None else THROUGHPUT_CSV)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(row["src_region"], row["dst_region"], cost=None,
                   throughput=num_vms * row["throughput_sent"] / 1e9)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    return G
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Local sanity print; the evaluator is authoritative.
    result = solve()
    for name, paths in result.items():
        print(name, "->", len(paths), "destinations routed")
