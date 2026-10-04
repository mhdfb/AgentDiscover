# AHC041 — Christmas Tree Cutting

Given a connected planar graph, partition every vertex into rooted trees using only graph edges. For each vertex `v`, output its parent `p_v` or `-1` if it is a root. Every tree must have height at most `H=10`. Vertex `v` has beauty `A_v`; its contribution is `(h_v+1)A_v`, where `h_v` is its depth. The case score is `1 + sum_v (h_v+1)A_v`; higher is better. A non-AC case makes the contest's 150-case submission score zero.

## Input

```text
N M H
A_0 ... A_(N-1)
u_0 v_0
...
u_(M-1) v_(M-1)
x_0 y_0
...
x_(N-1) y_(N-1)
```

Here `N=1000`, `1000<=M<=3000`, `H=10`, and `1<=A_v<=100`. The generator samples lattice vertices in a radius-500 circle centered at `(500,500)`, rejects vertices within distance 15 of previous vertices, and builds edges from their Delaunay triangulation. Beauty values are uniform in `[1,100]`.

## Output

```text
p_0 ... p_(N-1)
```

A root has parent `-1`; every other parent must be adjacent in the graph. The resulting forest must meet the height limit. Several solutions may be printed; only the last is scored.

## Interface

Use C++20 and define the entire program in the `SOURCE` raw string in `solution.py`. The starting `SOURCE` is empty; build the first program from the statement. The evaluator uses all 50 ALE-Bench public cases (seeds 0–49) with 13 workers. Its fitness `score` is the mean absolute case score, with accepted-case partial reward if some cases fail. There is no extra fitness normalization. Aim to exceed **520,000 mean points**, a contest-winner reference from a different 150-case suite. Each case has a 2-second time limit and 1024 MiB memory limit.
