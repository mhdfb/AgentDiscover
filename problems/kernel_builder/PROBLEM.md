# Kernel builder — a VLIW SIMD kernel for tree traversal

Optimize a VLIW SIMD kernel for a tree traversal algorithm on a simulated machine.

The machine has the following engines and slot limits per cycle:
- alu: 12 scalar operations
- valu: 6 vector operations (VLEN=8, processes 8 elements at once)
- load: 2 memory reads
- store: 2 memory writes
- flow: 1 control flow operation

Your `solution.py` must define a `KernelBuilder` class with a `build_kernel()` method
that builds an optimized program (a list of VLIW instructions) implementing the tree
traversal algorithm.

The algorithm: for each element in a batch, traverse a binary tree for multiple rounds.
At each node, XOR the value with the node value, hash it, then go left or right based on
parity. The kernel must produce identical output to the reference.

Key optimization opportunities:
1. VLIW: pack multiple operations per cycle (up to 12 ALU, 6 VALU, 2 load, 2 store)
2. SIMD: use vector operations (VLEN=8) to process 8 elements at once
3. Loop unrolling: reduce loop overhead
4. Memory access patterns: use vload/vstore for contiguous access
5. Instruction scheduling: avoid dependencies, maximize ILP

## Targets

Your objective is the **cycle count** of your kernel on the evaluator's workload
(forest height 10, 16 rounds, batch of 256). **Lower is better.**

| | cycles ↓ |
|---|---|
| the original naive scalar kernel | 147,734 |
| the seed below (vectorised, greedy VLIW packing) | 11,910 |
| **target — beat this, clearly** | **1,103** |

The target is a bar to get **meaningfully past**, not to land on: matching 1,103 is not
success, and nothing marks it as a limit — it is simply the best number a previous
search reached. Published results are deliberately not listed here. The fitness the
platform ranks on is `1103 / cycles` (1.0 at the target, above 1 past it); the evaluator
also returns `cycles`, `speedup` over the naive baseline and `n_instructions`.

## Interface

`solution.py` defines `KernelBuilder` with

    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    kb.instrs        # the program: a list of VLIW instructions
    kb.debug_info()  # a DebugInfo whose scratch_map names scratch addresses (optional)

An instruction is a dict from engine name to a list of slots, each slot a tuple whose
first element is the operation, for example
`{"valu": [("*", 4, 0, 0), ("+", 8, 4, 0)], "load": [("load", 16, 17)]}`. Every number
in a slot is a scratch address except in `const` and jumps; except for `store` and some
flow operations the first operand is the destination. The seed shows the instruction
forms it uses; the whole file is yours to change or replace as long as
`KernelBuilder.build_kernel` keeps its name and signature.

The kernel is built **before any input exists**: `build_kernel` is told only the four
sizes, which are fixed by the workload, and the same instruction list is then run on
eight freshly generated random trees and batches.

Correctness is mandatory: the output must match the reference on all eight trials, and an
incorrect kernel scores 0 with `Kernel produces incorrect output: Iteration i: Incorrect
output values`. A builder that raises scores 0 and returns its traceback.

The simulator is frozen: you cannot modify it, only optimize how you build instructions.
A copy of it is readable at `/resources/frozen_problem.py`, for local testing. The
evaluator is authoritative and you can read it too, at `/resources/evaluator.py`. Between
them they are the whole scoring path.

**Compute budget: 14 s** for `build_kernel`, out of a 120 s cap on the whole evaluation.
The seed builds in 0.15 s, so this is generous for a builder that constructs its schedule
directly and tight for one that searches at build time — budget accordingly. Your briefing
repeats the numbers in force each session.

## Strategy biases

- Fill every engine slot every cycle: the seed's greedy packer leaves most slots empty.
- Process 8 batch elements per valu operation; keep the batch in vectors end to end.
- The hash has 6 stages of 3 operations each: look for instruction-level parallelism
  within and across stages, and across independent batch elements.
- Fully unroll loops where scratch space (1536 words) allows, to remove flow control.
- Minimize data dependencies between consecutive instructions; reorder freely where the
  packer's read/write rule permits.
