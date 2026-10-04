"""MMD-16-2: place 16 points in the plane so that the ratio of the minimum to the
maximum pairwise distance is as large as possible.

See PROBLEM.md. The mutable region is marked by # EVOLVE-BLOCK-START /
# EVOLVE-BLOCK-END and holds the whole program, as in the original benchmark (whose seed
file carries no markers at all — everything in it was the agent's to change): the
imports and `run()` are yours. Everything outside the block is fixed scaffolding — do
not modify it.

CONTRACT (the evaluator relies on exactly this):

    run() -> np.ndarray of shape (16, 2)

  * The evaluator imports `run` from this file, calls it once, and scores the returned
    points: pairwise distances via scipy.spatial.distance.pdist,
    ratio = (min_distance / max_distance)^2, score = ratio / BENCHMARK with
    BENCHMARK = 1/12.889266112. Higher is better.
  * The shape must be exactly (16, 2). Scale, translation and rotation do not matter.
"""

# EVOLVE-BLOCK-START
import numpy as np


def run() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio
    (min_distance / max_distance)^2 over all pairwise distances.

    Returns:
        points: np.ndarray of shape (16, 2) containing the (x, y) coordinates of the 16 points.
    """
    n = 16
    d = 2

    # Places points randomly
    np.random.seed(42)
    points = np.random.randn(n, d)

    return points
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Local sanity print with the evaluator's own formula; the evaluator is authoritative.
    import scipy.spatial.distance

    pts = np.asarray(run(), dtype=float)
    dists = scipy.spatial.distance.pdist(pts)
    ratio = (dists.min() / dists.max()) ** 2 if dists.max() > 0 else 0
    benchmark = 1 / 12.889266112
    print(f"shape={pts.shape}  (min/max)^2={ratio:.8f}  "
          f"(max/min)^2={(1 / ratio if ratio else float('inf')):.6f}  score={ratio / benchmark:.6f}")
