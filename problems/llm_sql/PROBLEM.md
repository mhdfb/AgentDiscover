# LLM-SQL — column reordering for prefix caching

You are an expert in data optimization and LLM prompt caching. Your task is to evolve
the existing `Evolved` class to maximize prefix hit count (PHC) for efficient LLM prompt
caching.

**Problem Context:**

- You are given a pandas DataFrame with text data in rows and columns
- The goal is to reorder columns to maximize prefix reuse when processing rows
  sequentially
- Prefix reuse occurs when consecutive rows have matching values in the same column
  positions
- This reduces LLM computation costs by reusing cached prefixes

**Objective:**

- Dual objective: (1) maximize prefix reuse across consecutive rows and (2) minimize
  runtime
- Combined score: `0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12`

**Formally:**

- For a given column ordering C, `PHC(C) = sum over all rows r of hit(C, r)`
- `hit(C, r) = sum of len(df[r][C[f]])^2 for all f in prefix where df[r][C[f]] == df[r-1][C[f]]`;
  zero if mismatch starts at the first field.
- Runtime is measured as wall-clock seconds to compute the reordered DataFrame from the
  input.
- Combined score: `0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12`

**Required API (DO NOT CHANGE):**

- Keep the `Evolved` class structure and the `reorder` method signature:

      class Evolved(Algorithm):
          def reorder(
              self,
              df: pd.DataFrame,
              early_stop: int = 0,
              row_stop: int = None,
              col_stop: int = None,
              col_merge: List[List[str]] = [],
              one_way_dep: List[Tuple[str, str]] = [],
              distinct_value_threshold: float = 0.8,
              parallel: bool = True,
          ) -> Tuple[pd.DataFrame, List[List[str]]]:

- You can modify internal implementation but must preserve class structure and method
  signatures
- The `reorder` method must return a tuple of `(reordered_dataframe, column_orderings)`

**Algorithm Design Guidelines:**

- For each row, determine the optimal column order based on matches with the previous
  row
- Consider column statistics (unique values, string lengths) for ordering
- Focus on columns with high value frequency and long strings
- Handle missing values and mixed data types appropriately
- Optimize the existing recursive approach or replace it with more efficient vectorized
  methods
- Consider prefix-aware greedy approaches that condition on the current matched prefix

**Constraints:**

- Do not add/remove rows or columns
- You must have different column orderings for different rows to maximize prefix hit
  rate
- Return a DataFrame with the same shape as input
- Use exact string matching for prefix calculations
- Keep memory usage reasonable for large datasets
- Preserve all existing method signatures and class structure
- The algorithm will be called with the same parameters as the original `Evolved`

## Targets

Your objective is the **`combined_score`** above, averaged over the evaluator's data
sets. **Higher is better.**

| | combined_score ↑ |
|---|---|
| the seed below (recursive greedy grouping, slow) | ~0.69 |
| **target — beat this, clearly** | **0.739** |
| ceiling | 1.0 |

The ceiling of 1 is analytic, not a guess: a hit rate is a fraction, so it is at most 1,
and the runtime term is clamped to `[0, 1]`. Nothing marks the target as a limit — it is
simply where a previous search stopped, and it is a bar to get **meaningfully past**, not
to land on: matching ~0.739 is not success. Published results are deliberately not
listed here; a number someone else reached is an anchor, not a bound.

Both terms are real, and the seed shows how they trade. Its hit rate is respectable, but
its five `reorder` calls average well over 12 s, so it collects nothing at all from the
speed term — a fast method with a *lower* hit rate can beat it outright, and the gap
between the seed and the target is smaller than the whole speed term. Getting past the
target takes both: the full speed term *and* a hit rate the seed does not reach.

Alongside `combined_score` the evaluator returns `average_hit_rate`, `total_runtime`
(the sum of the `reorder` wall clocks over the data sets) and `average_runtime`, so you
can see which term is costing you. A candidate that fails is told why in the original
benchmark's words — a changed row count, a shrunken character count, or that one or
more files failed to run — and nothing finer. Reordering only rearranges data: every
returned row must be one original row's cells, in any column order and optionally merged
(same characters, each original cell kept intact); a row that is not scores 0.

## Interface

`solution.py` defines `class Evolved(Algorithm)` with the `reorder` signature above. The
evaluator imports `Evolved` from it and calls `reorder` once per data set — five
real-world tables, each with the same fixed keyword arguments the original benchmark
uses, including a `col_merge` list per table — and scores what comes back: the returned
DataFrame's prefix hit rate, and the wall clock of the call. The whole file, imports
included, is inside the EVOLVE-BLOCK, exactly as in the original benchmark: everything
in it is yours to change or replace, as long as the class name, its base class and the
method signature stay.

**The data sets, the metric code and the `solver` module are private to the evaluator,
exactly as in the original benchmark.** Nothing is mounted at `/support` or
`/resources`; your only measurement is the score a submission returns. `Algorithm` in
`solver` is a plain base class with a few static helpers the seed calls
(`calculate_col_stats`, which ranks columns by how much repeated text they carry, and
`merging_columns`, which concatenates the columns named in `col_merge` into one) — you
may reimplement any of them inside your own class. To run `solution.py` locally, write
your own stand-in `solver.py` in your worktree with an `Algorithm` class; the evaluator
only ever reads `solution.py`, so a stand-in can never reach scoring.
`python3 solution.py` then runs the class on a small synthetic table.

**Compute budget: 240 s for the five `reorder` calls together**, not per call; the
evaluation as a whole is capped at 360 s. Scoring takes roughly 100 s of that on a
decent result and up to twice as much on a poor one — the metric's cost grows the less
consecutive rows share — so a candidate that is both slow and poor can run out of the
360 s. Overrunning either cap scores 0. The budget is a safety net rather than a design
constraint: an average of 12 s per data set already forfeits the whole speed term. Your
briefing repeats the numbers in force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- column statistics: rank columns by how much repeated text they carry (value frequency times string length) and sort rows so that shared leading values become adjacent
- prefix-aware greedy: choose each next column conditioned on the group of rows that already agree on the current prefix, so the ordering can differ from row to row
- row ordering: treat the order of the rows as the main lever — cluster rows by shared values before deciding any column order, so consecutive rows agree on their leading fields
- speed: replace the recursion with vectorised pandas or numpy operations and make every data set finish in well under a second, then spend the remaining effort on the hit rate
