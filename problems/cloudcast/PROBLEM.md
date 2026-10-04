# CloudCast — multi-cloud broadcast at minimum cost

You are optimising cloud infrastructure. Your task is to evolve the
`search_algorithm(src, dsts, G, num_partitions)` function to minimise the overall data
transfer cost of broadcasting input data from one cloud region to multiple destination
regions across multiple clouds (AWS, GCP, Azure).

Focus on efficiently broadcasting the data to all destination nodes by leveraging
parallel paths and overlapping transfers across networks. Use the `BroadCastTopology`
class and the `make_nx_graph` function to identify low-cost routes. Prioritise strategies
that reduce redundant transfers, balance load across networks, and exploit multi-network
topologies to minimise cost.

The network is a directed graph: nodes are cloud regions, edges carry a `cost` ($/GB) and
a `throughput` (Gbps) from real measured profiles. Each cloud provider also caps total
ingress and egress per region, so flows through a region share its limits. The data is
split into partitions; every partition must reach every destination along a contiguous
path of real edges, and the simulated cost of the resulting transfer is what you pay.

## Targets

Your schedule is scored on five fixed network configurations (three intra-cloud, two
inter-cloud, defined in `examples/config/*.json`) and the objective is the **sum of the
simulated total cost** over all five. Lower is better.

| | total cost |
|---|---|
| the seed below (per-destination cheapest paths) | ~1035 |
| **target — beat this, clearly** | **618** |

The target is a bar to get **meaningfully under**, not to land on: reaching ~618 is not
success, and there is no proven optimum — every dollar below it is open ground. Published
results are deliberately not listed here: a number someone else reached is an anchor, not
a bound. The evaluator returns your `total_cost`, its breakdown per configuration, and
the target.

## Interface

`solve()` (fixed scaffolding) runs your `search_algorithm` on each of the five
configurations and returns the resulting `BroadCastTopology.paths`. You evolve everything
inside the EVOLVE-BLOCK: `search_algorithm`, and the `BroadCastTopology` /
`make_nx_graph` helpers if you need to change them.

A valid topology routes **every partition of every destination** along a contiguous chain
of edges that exist in the real network, starting at the source and ending at the
destination. Anything else scores 0. Costs are always priced from the real network data,
not from what your paths carry.

`cloudcast_data` is importable from `solution.py` (data file paths, the five configs,
`NUM_VMS = 2`); the profiles and configs are also mounted read-only at `/resources`.

**Compute budget: `solve()` may run for 600 s** across all five configurations. Budget
your search against that figure and use it — leaving time unspent is wasted search, and
overrunning it means the candidate scores 0. The evaluation as a whole is capped at
700 s, so leave a margin for interpreter start-up and imports. Your briefing repeats the
number in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- overlay trees: build shared distribution trees so data crosses expensive links once and fans out from cheap relays
- parallel paths: split partitions across multiple routes to overlap transfers and spread load across networks
- multi-cloud relays: route through other providers' regions when their links are cheaper than the direct path
- cost-structure analysis: study the cost and throughput profiles first, and shape routes around the asymmetries they reveal
