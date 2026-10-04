# Sum-difference problem — how much can differences outnumber sums?

Let `C` be the least constant such that

    |A − A|  ≤  |A + A|^C

holds for every non-empty finite set of integers `A`, where
`A + A = {a + b : a, b ∈ A}` and `A − A = {a − b : a, b ∈ A}`.

Because subtraction is not commutative in the way addition is, a generic set has
*more* differences than sums. The question is how far that can be pushed.

**Your task**: find a finite set of integers `A` maximising

    ratio(A) = log |A − A| / log |A + A|

Every set you produce certifies `C ≥ ratio(A)`. Your score is that lower bound.

## Where the record stands

| Construction | Lower bound on C |
|---|---|
| a random or interval-like set | ≈ 1.00 |
| **AlphaEvolve, no human hints** | **≈ 1.21** |
| best known (high-dimensional simplex construction) | **log(1+√2)/log 2 = 1.2715…** |

DeepMind's write-up is explicit about the failure: *"Without any human hints, AlphaEvolve was
not able to discover this construction within a few hours, and only managed to find
constructions giving a lower bound of around 1.21."* The gap between 1.21 and 1.2715 is the
target, and the known construction is a specific, discoverable object rather than a lucky
numerical accident.

## The idea behind the record

The extremal constructions are not one-dimensional in spirit. They come from taking a set in
`Z^d` — typically a simplex or a corner of a lattice box — where the sumset is provably small
relative to the difference set, and then mapping it injectively into the integers with a
**Freiman isomorphism**: pick a base `B` larger than any coordinate interaction and send
`(x_1, ..., x_d) → x_1 + x_2·B + x_3·B² + ...`. If `B` is large enough, the map preserves
both `|A + A|` and `|A − A|` exactly, so the `d`-dimensional ratio survives in `Z`. Raising
`d` pushes the ratio towards `log(1+√2)/log 2`.

## Interface

`solve()` returns a list of distinct integers (`2 ≤ |A| ≤ 4000`, each `|a| ≤ 10^12`).
Duplicates are rejected rather than silently removed, so deduplicate before returning.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that
figure and use it — leaving time unspent is wasted search, and overrunning it means
the candidate scores 0. The evaluation as a whole is capped at 700 s, so leave a
margin for interpreter start-up and imports. Your briefing repeats the number in
force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

- higher-dimensional lifts: build sets in Z^d and map them to the integers with a base-B Freiman isomorphism, then raise d
- direct combinatorial search: local search and simulated annealing over subsets of a bounded integer window
- algebraic structure: arithmetic progressions, generalised APs, and modular/Sidon-flavoured constructions and their products
- product and recursive constructions: take a small set with a good ratio and compose it with itself to amplify the exponent
