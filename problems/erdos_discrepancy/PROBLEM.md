# Erdős discrepancy problem (C = 2)

The **discrepancy** of a sign pattern `a_1, ..., a_N ∈ {-1, +1}` is the largest value of

    | a_d + a_2d + a_3d + ... + a_kd |

over all homogeneous arithmetic progressions `d, 2d, ..., kd` inside `{1, ..., N}`.

**Your task**: produce the longest possible sign pattern whose discrepancy is at most **2**.

The score is the length of the longest prefix of your sequence that has discrepancy ≤ 2.
So a prefix that stays good for 400 terms scores 400, no matter what follows it.

## Target

The maximum length is *known exactly*: `C(2) = 1160`. It was certified in 2014 by Konev
and Lisitsa with a SAT solver — a sequence of length 1160 with discrepancy 2 exists, and
no sequence of length 1161 does.

| | Length |
|---|---|
| the seed below (greedy, no backtracking) | 8 |
| **target — the proven optimum** | **1160** |

So the target is not a stretch goal: a sequence of exactly this length exists, and 1160 is
both the goal and a hard ceiling. Nothing shorter is optimal and nothing longer is possible.

How anyone else has approached this, and how far they got, is deliberately not stated here.
Someone else's number tells you where a previous search stopped, not where this one has to.

## Interface

`solve()` returns a list of `+1`/`-1` integers. `a[0]` is the first term `a_1`. Returning
more than 1160 terms is allowed but pointless — the score counts the good prefix only.

**Compute budget: `solve()` may run for 600 s.** Budget your search against that
figure and use it — leaving time unspent is wasted search, and overrunning it means
the candidate scores 0. The evaluation as a whole is capped at 700 s, so leave a
margin for interpreter start-up and imports. Your briefing repeats the number in
force each session.

## Strategy biases

Optional; assigned one per branch, in order (branches repeat the list if there are more
branches than biases). This section itself is never shown to the agent — only its own
assigned line is, via `{{STRATEGY_BIAS}}`.

Search-method families only. They must not encode any insight into what the good sequences
look like — AlphaEvolve's published unhinted result had no such insight supplied, and a
bias that hands one over would make our number incomparable to it.

- constructive search: extend a partial sequence position by position, undoing earlier choices when the extension dead-ends
- global optimisation: treat the whole fixed-length sign vector as one objective and improve it with local or population moves
- exact methods: encode prefixes as a constraint or satisfiability problem and solve them outright, then extend
- build on the database: start from the best sequence any session has found, locate where it first fails, and work on that
