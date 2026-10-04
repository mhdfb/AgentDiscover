"""MMD-14-3: place 14 points in R^3 so that the ratio of the minimum to the maximum
pairwise distance is as large as possible.

See PROBLEM.md. The mutable region is marked by # EVOLVE-BLOCK-START /
# EVOLVE-BLOCK-END and, exactly as in the original benchmark, it holds the whole
program: the imports and `run()` are yours to change. Everything outside it is fixed
scaffolding — do not modify it.

CONTRACT (the evaluator relies on exactly this):

    run() -> np.ndarray of shape (14, 3)

  * The evaluator imports `run` from this file, calls it once, and scores the returned
    points: pairwise distances via scipy.spatial.distance.pdist,
    ratio = (min_distance / max_distance)^2, score = ratio / BENCHMARK with
    BENCHMARK = 1/4.165849767. Higher is better.
  * The shape must be exactly (14, 3) with no NaN. Scale and translation do not matter.
"""

# EVOLVE-BLOCK-START
import numpy as np


def run() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to
    maximum pairwise distance: (min_dist / max_dist)^2.

    Returns:
        points: np.ndarray of shape (14, 3) containing the coordinates.
    """
    n = 14
    d = 3

    # Initialize with random points
    np.random.seed(42)
    points = np.random.randn(n, d)

    return points


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Local sanity print with the evaluator's own formula; the evaluator is authoritative.
    import scipy.spatial.distance

    pts = np.asarray(run(), dtype=float)
    dists = scipy.spatial.distance.pdist(pts)
    ratio = (dists.min() / dists.max()) ** 2
    benchmark = 1 / 4.165849767
    print(f"shape={pts.shape}  (min/max)^2={ratio:.8f}  (max/min)^2={1 / ratio:.6f}  "
          f"score={ratio / benchmark:.6f}")
