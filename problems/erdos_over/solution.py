"""Erdős minimum overlap: a step function h: [0,2] -> [0,1] with integral 1 whose maximum
overlap with its own complement, over all shifts, is as small as possible — an upper
bound on the constant C5.

See PROBLEM.md. The mutable region is marked by # EVOLVE-BLOCK-START /
# EVOLVE-BLOCK-END and, exactly as in the original benchmark, it holds the whole
program: the imports, the optimiser and `run()` are yours to change. Everything outside
it is fixed scaffolding — do not modify it.

CONTRACT (the evaluator relies on exactly this):

    run() -> (h_values, c5_bound, n_points)

  * h_values: array of shape (n_points,) with every value in [0, 1] and
    sum(h_values) * (2 / n_points) == 1 (to 1e-3).
  * c5_bound: the C5 value you computed for it — the evaluator recomputes
    max(np.correlate(h, 1 - h, mode="full") * dx) and requires agreement to 1e-4.
  * n_points: the number of discretisation points. There is no cap on it.
  * Score = 0.38092303510845016 / C5, higher is better; the evaluator scores the C5 it
    recomputes from your h, so the number you report cannot help or hurt you beyond
    the consistency check.
"""

# EVOLVE-BLOCK-START
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    num_intervals: int = 200
    learning_rate: float = 0.005
    num_steps: int = 20000
    penalty_strength: float = 1000000.0


class ErdosOptimizer:
    """
    Finds a step function h that minimizes the maximum overlap integral.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 2.0
        self.dx = self.domain_width / self.hypers.num_intervals

    def compute_c5(self, h: np.ndarray) -> float:
        """Compute C5 bound via cross-correlation of h with (1-h)."""
        j = 1.0 - h
        correlation = np.correlate(h, j, mode="full") * self.dx
        return float(np.max(correlation))

    def run_optimization(self):
        """Simple optimization using random restarts and local perturbation."""
        N = self.hypers.num_intervals
        best_h = None
        best_c5 = float("inf")

        for trial in range(5):
            np.random.seed(42 + trial)

            # Initialize h so that integral is 1
            h = np.random.uniform(0.3, 0.7, N)
            h = h / (np.sum(h) * self.dx)  # Normalize integral to 1
            h = np.clip(h, 0, 1)

            # Re-normalize after clipping
            current_integral = np.sum(h) * self.dx
            if current_integral > 0:
                h = h * (1.0 / current_integral)
                h = np.clip(h, 0, 1)

            # Simple perturbation-based local search
            for step in range(self.hypers.num_steps):
                c5 = self.compute_c5(h)

                # Random perturbation
                idx = np.random.randint(0, N)
                delta = np.random.uniform(-0.02, 0.02)
                h_new = h.copy()
                h_new[idx] = np.clip(h_new[idx] + delta, 0, 1)

                # Re-normalize integral
                current_integral = np.sum(h_new) * self.dx
                if current_integral > 0:
                    h_new = h_new * (1.0 / current_integral)
                    h_new = np.clip(h_new, 0, 1)

                c5_new = self.compute_c5(h_new)
                if c5_new < c5:
                    h = h_new

            c5 = self.compute_c5(h)
            if c5 < best_c5:
                best_c5 = c5
                best_h = h.copy()

        print(f"Optimization complete. Final C5 upper bound: {best_c5:.8f}")
        return best_h, best_c5


def run():
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()

    return final_h_values, c5_bound, hypers.num_intervals


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Local sanity print with the evaluator's own formula; the evaluator is authoritative.
    import time

    t = time.perf_counter()
    h_values, c5_bound, n_points = run()
    elapsed = time.perf_counter() - t
    h = np.array(h_values, dtype=float)
    dx = 2.0 / n_points
    c5 = float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))
    print(f"n_points={n_points}  integral={np.sum(h) * dx:.6f}  reported C5={c5_bound:.10f}  "
          f"recomputed C5={c5:.10f}  score={0.38092303510845016 / c5:.6f}  ({elapsed:.1f} s)")
