# MLA-Decode — a Triton kernel for DeepSeek's multi-head latent attention (decode)

You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

Below is a pytorch implementation of the multi-head latent attention (MLA) module. You will want to implement a Triton kernel for the operations in the forward call:

```python
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model//2,dtype=torch.bfloat16) / (d_model//2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin

class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple) -> None:
        super().__init__()
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device='cuda'))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()

    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.seq_len : self.seq_len + c_kv.size(1), :
        ] = c_kv
        self.seq_len += c_kv.size(1)

        return self.data[:, :self.seq_len], self.seq_len

@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor

class MLA(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # Down-projection matrices
        self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False, dtype=torch.bfloat16)

        # Up-projection and rope projection matrices
        self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)
        self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)

        # RoPE on half embeddings
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)

        # Output projection
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        # seq_len = 1 always here
        batch_size, seq_len, model_dim = x.size()

        ## Step 1: Handle down-projection + KV cache ##

        q_lora = self.Q_proj_down(x)
        kv_lora = self.KV_proj_down(x)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1

        ## Step 2: Up-project and prepare NoPE + RoPE ##

        # Handle queries Q first
        q_nope_and_rope = self.Q_proj_up(q_lora).view(
            batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        # Handle keys and values K/V. V does not need RoPE
        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_nope = self.KV_proj_up(kv_nope).view(
            batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)

        ## Step 3: Handle RoPE Stream ##

        # Compute RoPE for queries and combine with no-RoPE part
        q_rope = q_rope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q_rope = self.q_rope(q_rope, start_pos=query_pos)

        q_nope = q_nope.permute(0, 2, 1, 3) # bs x n_heads x seq_len x rope_head_dim
        q = torch.concat([q_nope, q_rope], dim=-1)

        # Compute RoPE for keys and combine with no-RoPE part
        k_rope = k_rope[:, None, :, :]
        k_rope = self.k_rope(k_rope).expand(-1,self.n_heads,-1,-1)
        k_nope = k_nope.permute(0, 2, 1, 3) # bs x kv_len x n_heads x rope_head_dim
        k = torch.concat([k_nope, k_rope], dim=-1)

        ## Step 4: Compute Multi-head Attention ##

        v = v.permute(0, 2, 1, 3) # bs x n_heads x kv_len x v_head_dim
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v).view(batch_size, 1, -1)
        y = self.wo(y)

        return y, kv_cache.get_data()
```

Your function should be defined as `custom_kernel` (skeleton provided below)

```python
### DO NOT CHANGE THIS IMPORT STATEMENTS BLOCK ###
import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
from reference import KVCache, Config  # Definition of KVCache and Config classes are shown above. Must import this way. Do not rewrite yourself.
### END OF IMPORT STATEMENTS BLOCK ###

### Import other packages here if needed

def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, KVCache]:
    config, x, kv_cache = data

    bs = config.batch_size
    sl = config.seq_len
    pl = kv_cache.seq_len
    msl = config.max_seq_len
    nh = config.n_heads
    d =  config.dim
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    dnope = config.qk_nope_head_dim
    drope = config.qk_rope_head_dim
    dv = config.v_head_dim

    wDQ  = config.Q_proj_down_weight
    wDKV = config.KV_proj_down_weight
    wUQ  = config.Q_proj_up_weight
    wUKV = config.KV_proj_up_weight
    wO   = config.wo_weight

    # Perform MLA operations to process data into output and updated kv_cache

    return output, kv_cache.data
```

with the following signature:

Input:
- `data`: Tuple of (config: Config, x: torch.Tensor, kv_cache: KVCache)
    - config: An instance of class `Config` containing model configurations and weights
    - x: Input tensor of shape [batch_size, seq_len, dim]
    - kv_cache: An instance of KVCache class for caching the keys and values

Output:
- output: Output tensor [batch_size, seq_len, dim]
- kv_cache.data: The data field of the updated `KVCache` instance with the new keys and values added

Below are the different configs that your kernel will be tested on:

Common configs:
  - {"batch_size": 128, "seq_len": 1, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "v_head_dim": 128, "n_heads": 128, "dim": 7168, "q_lora_rank": 1536, "max_seq_len": 8192}

For correctness check:
  - {"prefill": 128}
  - {"prefill": 512}
  - {"prefill": 1024}
  - {"prefill": 2048}

For performance benchmark (optimize runtime for these):
  - {"prefill": 6144}

Rules:
- The tensors arguments passed in will be already on your cuda device.
- The weights for all parameters in the MLA will be given as input.
- All weights and data will be in `torch.bfloat16` format.
- Define all of your code in `solution.py` (see Interface).
- The entrypoint to your code must be named 'custom_kernel'.
- You will be using triton 3.4.0 (torch 2.8.0, CUDA 12.9) and your kernels will be run on an Nvidia H200 GPU.
- Consider optimizing multiple operations with triton, not just limited to softmax. E.g., rope, attention, etc.
- You are allowed to use torch.compile().
- A program without `@triton.jit` is rejected.

Important rules in triton 3.4.0:
- `tl.load` does not have an argument called `dtype`. Never use it like `tl.load(..., dtype=...)`.
- Triton dtypes are not callable, so never use them like `tl.float16(1.0)`, `tl.float32(0.0)`.
- `tl.arange(start, end)`:
    - range length (end - start) must be power-of-2
    - start, end must be of type `tl.constexpr`
- `tl.range(start, end, step, num_stages)`:
    - keep loop index type stable, don't reassign it
    - start, end, step do not have to be `tl.constexpr` but must stay scalar integer types
    - num_stages must be `tl.constexpr`
- Do not something like x[0] or offs[0] inside a Triton kernel. Triton tensors are SIMD vectors; scalar indexing like [0] is not generally supported.

Here's an simple example correctly following these rules:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_right(
    x_ptr, y_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,          # constexpr; also power-of-2 for tl.arange
    ROW_STEP: tl.constexpr,
    NUM_STAGES: tl.constexpr,     # constexpr; used by tl.range(num_stages=...)
):
    pid = tl.program_id(axis=0)

    # ------------------------------------------------------------
    # arange: constexpr args + power-of-2 range
    # ------------------------------------------------------------
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # (0, BLOCK) are constexpr
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    # ------------------------------------------------------------
    # Dtypes not callable: typed constants and casting
    # ------------------------------------------------------------
    one_f32 = tl.full([], 1.0, tl.float32)               # typed scalar
    acc = tl.zeros((BLOCK,), dtype=tl.float32)           # typed vector
    acc = tl.cast(x, tl.float32) + tl.cast(y, tl.float32) + one_f32

    # ------------------------------------------------------------
    # Avoid x[0]: scalar address load + broadcast
    # ------------------------------------------------------------
    base = tl.full([], pid * BLOCK, tl.int32)
    x0 = tl.load(x_ptr + base, mask=(base < n_elements), other=0.0)
    x0_vec = tl.full((BLOCK,), x0, tl.float32)

    out_vec = acc + x0_vec

    # ------------------------------------------------------------
    # tl.range: keep loop index type stable, don't reassign it
    #
    # WRONG (causes "Loop-carried variable ... type stays consistent" assertion):
    #   for row in tl.range(row, n_rows, row_step):
    #       row = tl.load(...)  # row (int32) reassigned to tensor/bf16/...
    #
    # RIGHT:
    #   - use a fresh name for loop index (e.g., r)
    #   - compute offsets/tensors into *different* vars
    #   - keep r as an integer index (int32) throughout
    # ------------------------------------------------------------
    # We'll do a tiny staged reduction over "rows" just as a demo.
    n_rows = tl.full([], 4, tl.int32)  # small fixed count for demo (scalar int32)

    extra = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.range(0, n_rows, ROW_STEP, num_stages=NUM_STAGES):
        # r is an int32 loop index. Keep it that way.

        # Use r to build an integer shift; keep shifts as ints too.
        shift = r * tl.full([], 1, tl.int32)

        # Compute new offsets (int) without mutating r:
        offs_r = offs + shift

        # Load something; store into a separate var (tensor), not r:
        xr = tl.load(x_ptr + offs_r, mask=(offs_r < n_elements), other=0.0)
        extra += tl.cast(xr, tl.float32)

    out_vec = out_vec + extra

    tl.store(out_ptr + offs, tl.cast(out_vec, tl.float16), mask=mask)
```

## Targets

Your objective is the **runtime** of your kernel on the performance benchmark
(prefill 6144), in microseconds: its mean runtime over repeated, re-checked calls,
measured by the GPUMode competition's own harness on the GPU named above.
**Lower is better.**

| | runtime (µs) ↓ |
|---|---|
| the starting kernel below, as measured on this hardware | see your briefing |
| **target — beat this, clearly** | **1,700** |

The target is a bar to get **meaningfully past**, not to land on. The fitness the
platform ranks on is `1700 / runtime_us` (1.0 at the target, above 1 past it).

## Interface

`solution.py` is the whole submission. The evaluator places it next to the competition's
harness files as `submission.py` and imports `custom_kernel` from it, so everything the
kernel needs — imports, `@triton.jit` kernels, helpers — lives in that one file, inside
the EVOLVE block. `reference` (the module `KVCache` and `Config` must be imported from)
is the harness's own `reference.py`, importable wherever the harness runs. The starting
file is a correct but unoptimized kernel: PyTorch for the projections and attention, one
Triton kernel each for the query RoPE and the softmax.

Correctness is mandatory: the harness first runs the four correctness cases (prefill
128, 512, 1024, 2048) against the bf16 PyTorch reference, comparing both the output and
the returned KV cache with `rtol = 2e-2, atol = 8e-3`; one failure scores 0 and reports
the first failing case. Then it warms up for 0.2 s and times the prefill-6144 benchmark
with the wall clock around synchronised calls, re-generating the inputs and re-checking
the output on every repeat, until the standard error of the mean is under 1 % or 30 s of
kernel time has accumulated (at most 100 repeats). A kernel that fails a re-check scores
0.

**Compute budget per evaluation:** 120 s for the import run of your file, then 900 s for
the correctness phase and 1200 s for the benchmark phase — the competition's own clocks.
Your briefing repeats the numbers in force each session.

## Local tools

The scoring path is readable at `/resources/evaluator.py` with the harness files beside
it in `/resources/kernelbot/` (`eval.py`, `reference.py`, `utils.py`, `task.py`,
`task.yml` — the competition's, unchanged). Your sandbox has the same torch 2.8.0 and
triton 3.4.0 and a GPU of its own, so

    python3 /resources/evaluator.py solution.py

runs the exact evaluation locally and prints the JSON line the platform would return.
The GPU that scores submissions is a separate, otherwise idle GPU of the same type as
yours, so the runtime you measure locally is what the evaluator will see, up to noise.
