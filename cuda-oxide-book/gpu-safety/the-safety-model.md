# The Safety Model

A GPU kernel runs thousands of threads that all see the same memory at the
same time. On a CPU, Rust prevents data races through ownership and borrowing
-- one mutable reference, no aliases, enforced at compile time. On a GPU,
you have 2048 threads per SM, all launched from the same function, all
pointing at the same output buffer. The borrow checker was not designed for
this.

cuda-oxide solves the problem in layers. The common case -- one thread writes
one element -- is safe by construction, no `unsafe` required. The uncommon
cases -- shared memory, warp shuffles, hardware intrinsics -- require
`unsafe` with documented contracts. And the frontier cases -- TMA, tensor
cores, cluster-level communication -- are fully manual, matching the
complexity of the hardware they control.

This chapter explains the model, walks through each layer, and tells you
exactly when you need `unsafe` and why.

---

## Three tiers

cuda-oxide organizes kernel safety into three tiers based on how much the
compiler can verify:

| Tier       | Description                                             | `unsafe` Required? |
|:-----------|:--------------------------------------------------------|:-------------------|
| **Tier 1** | Safe by construction -- the type system prevents misuse | No                 |
| **Tier 2** | Explicit `unsafe` with clear safety contracts           | Yes, scoped        |
| **Tier 3** | Raw hardware intrinsics -- full user responsibility     | Yes, pervasive     |

Most application kernels live entirely in Tier 1 or straddle Tier 1 and 2.
Tier 3 is for performance engineers building at the level of CUTLASS or
Triton IR. If you are writing a vecadd, a GEMM, or a reduction, you will
rarely leave Tier 2.

---

## Tier 1: safe by default

### The core idea: `DisjointSlice<T>` + `ThreadIndex`

The primary safety abstraction is a pair of types that together guarantee
race-free parallel writes without `unsafe` at the call site:

- **`ThreadIndex`** -- an opaque newtype around `usize` with no public
  constructor. You cannot create one from an arbitrary integer. The only way
  to obtain a `ThreadIndex` is through trusted functions (`index_1d`,
  `index_2d`) that derive it from **hardware built-in variables**
  (`threadIdx`, `blockIdx`, `blockDim`) -- read-only special registers
  assigned by the runtime at kernel launch.
- **`DisjointSlice<T>`** -- a slice-like type whose `get_mut()` method
  accepts only `ThreadIndex`, not raw `usize`. It returns
  `Option<&mut T>` -- `None` for out-of-bounds indices, `Some(&mut T)` for
  valid ones.

Put them together and you get a kernel with zero `unsafe`:

```rust
use cuda_device::{kernel, thread, DisjointSlice};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

Safety follows from three facts:

1. `index_1d()` produces a unique value per thread (hardware guarantee:
   `threadIdx.x < blockDim.x`, so the linear index
   `blockIdx.x * blockDim.x + threadIdx.x` is unique across the grid).
2. `get_mut()` is bounds-checked -- out-of-range threads get `None`.
3. Different threads get different `ThreadIndex` values, so different
   `&mut T` references. No aliasing, no data race.

The borrow checker sees a single `&mut T` per thread. The hardware
guarantees the indices are disjoint. The type system ties the two together.

### Trusted index functions

`ThreadIndex` is only as trustworthy as the functions that create it. Here
are the constructors cuda-oxide provides:

| Function           | Formula                                 | Return Type          | Uniqueness Guarantee                                                        |
|:-------------------|:----------------------------------------|:---------------------|:----------------------------------------------------------------------------|
| `index_1d()`       | `blockIdx.x * blockDim.x + threadIdx.x` | `ThreadIndex`        | Unconditional -- `threadIdx.x < blockDim.x` is hardware-enforced            |
| `index_2d(stride)` | `row * stride + col`                    | `Option<ThreadIndex>`| Unconditional -- returns `None` when `col >= stride`, enforcing injectivity |
| `index_2d_row()`   | `blockIdx.y * blockDim.y + threadIdx.y` | `usize`              | Component accessor, not a `ThreadIndex` constructor                         |
| `index_2d_col()`   | `blockIdx.x * blockDim.x + threadIdx.x` | `usize`              | Component accessor, not a `ThreadIndex` constructor                         |

Notice that `index_2d_row()` and `index_2d_col()` return plain `usize` --
they give you the row and column for arithmetic, but they cannot be used to
index into a `DisjointSlice`. Only the linearized result, after a uniqueness
check, earns the `ThreadIndex` type.

### Why `index_2d` returns `Option`

This one deserves a closer look, because it is the subtlest part of the
safety model.

The formula `row * stride + col` is only *injective* (one-to-one)
when `col < stride`. In a 2D grid, `col` is derived from `blockIdx.x`
and `threadIdx.x` -- it can exceed the matrix column count when the grid
dimensions overshoot (which they almost always do, since block dimensions
must be multiples of the warp size). If the function returned a bare
`ThreadIndex` for those threads, two distinct threads could compute the
same linear index. That would mean two `&mut T` references to the same
element. That is undefined behavior.

`index_2d` eliminates this by checking `col < row_stride` internally:

```rust
pub fn index_2d(row_stride: usize) -> Option<ThreadIndex> {
    let row = (blockIdx_y() * blockDim_y() + threadIdx_y()) as usize;
    let col = (blockIdx_x() * blockDim_x() + threadIdx_x()) as usize;
    if col < row_stride {
        Some(ThreadIndex(row * row_stride + col))
    } else {
        None
    }
}
```

Threads that fail the check get `None`. They simply do not participate in
the write -- no aliasing, no race. Even if you pass a "wrong" stride
(say, `index_2d(1)`), the worst that happens is that most threads get
`None` and only the `col == 0` threads write. You get fewer writers than
you intended, but never undefined behavior.

:::{tip}
The informal proof: suppose two threads with `(row_a, col_a)` and
`(row_b, col_b)`, both satisfying `col < stride`, produce the same index:

```text
row_a * stride + col_a == row_b * stride + col_b
=> (row_a - row_b) * stride == col_b - col_a
```

The right side is in `(-stride, stride)` because both cols are in
`[0, stride)`. The left side is a multiple of `stride`. The only value in
both sets is zero -- so `row_a == row_b` and `col_a == col_b`. But distinct
hardware threads have distinct `(row, col)` pairs.
:::

### The GEMM pattern

For 2D kernels, the typical pattern looks like this:

```rust
let row = thread::index_2d_row();
let col = thread::index_2d_col();

if let Some(c_idx) = thread::index_2d(n as usize) {
    // col < n is guaranteed by Some -- no manual check needed
    if row < m as usize {
        // ... compute dot product ...
        if let Some(c_elem) = c.get_mut(c_idx) {
            *c_elem = alpha * sum + beta * (*c_elem);
        }
    }
}
```

The `if let Some` from `index_2d` replaces the manual `col < n` guard you
would write in CUDA C++. The `row < m` check remains because it guards
against reading garbage from the input matrices (though `get_mut` would also
return `None` for out-of-bounds writes).

### What makes a kernel Tier 1

A kernel is fully safe -- Tier 1 -- when:

1. All mutable output access goes through `DisjointSlice::get_mut(ThreadIndex)`
2. All inputs are shared immutable references (`&[T]`)
3. No shared memory, no raw pointers, no intrinsics beyond thread indexing

Examples in this tier include `vecadd`, `helper_fn`, `generic`, `host_closure`,
and the naive GEMM kernels in the `gemm` and `async_mlp` examples.

---

## Tier 2: scoped `unsafe`

Not every kernel fits the "one thread, one output element" pattern. When
threads need to cooperate -- sharing data through fast on-chip memory,
communicating across lanes in a warp, or performing atomic updates -- you
need `unsafe`. The key property of Tier 2 is that the `unsafe` is *scoped*
and *auditable*: each block has a documented safety contract, and the rest
of the kernel remains safe.

### Shared memory

Shared memory is fast, on-chip, and visible to every thread in a block.
That last property is exactly why it requires `unsafe` -- the borrow checker
cannot reason about 256 threads writing to the same `static mut` array:

```rust
static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

unsafe { TILE[ty * TILE_SIZE + tx] = value; }

thread::sync_threads();

let neighbor = unsafe { TILE[other_idx] };
```

The contract: ensure no conflicting writes from concurrent threads without
synchronization. The `sync_threads()` barrier is the tool that makes this
work -- it guarantees all threads have finished writing before any thread
reads.

| API                       | Safety Obligation                                                                                  |
|:--------------------------|:---------------------------------------------------------------------------------------------------|
| `SharedArray<T, N>`       | Accessed via `static mut`. No conflicting writes without synchronization.                          |
| `DynamicSharedArray<T>`   | Same rules, but size is set at launch time via `LaunchConfig::shared_mem_bytes`.                   |

:::{seealso}
[Shared Memory and Synchronization](../advanced/shared-memory-and-synchronization.md)
for the full treatment: tiling, bank conflicts, dynamic allocation, and
double-buffered pipelines.
:::

### Warp intrinsics

Warp-level primitives let threads within a warp exchange data without
touching memory at all -- register-to-register transfers, coordinated in
hardware. They are `unsafe` because the hardware does not check thread
convergence: if you pass a mask that includes a diverged thread, you get
undefined behavior (typically a silent hang, which is worse than a crash).

| API                                                                | Safety Obligation                                            |
|:-------------------------------------------------------------------|:-------------------------------------------------------------|
| `shfl_sync`, `shfl_up_sync`, `shfl_down_sync`, `shfl_xor_sync`     | Source lane must be active; mask must include calling thread |
| `ballot_sync`, `any_sync`, `all_sync`                              | All threads in mask must be converged                        |
| `activemask`                                                       | Result is only meaningful at the point of call               |

:::{seealso}
[Warp-Level Programming](../advanced/warp-level-programming.md) for shuffle
patterns, reductions, and prefix sums using warp intrinsics.
:::

### Barriers and lifecycle

The `ManagedBarrier` typestate API encodes the barrier lifecycle
(`Uninit` -> `Ready` -> `Invalidated`) in the type system, so you cannot
wait on a barrier that was never initialized or use one that has been
invalidated. The `init()` and `inval()` transitions still require `unsafe`
because they interact with the hardware, but the type states prevent the
most common mistakes at compile time.

| API                                    | Safety Obligation                                                               |
|:---------------------------------------|:--------------------------------------------------------------------------------|
| `mbarrier_init`                        | Must be called by exactly one thread; barrier must be in shared memory          |
| `mbarrier_arrive` / `mbarrier_wait`    | Barrier must be initialized; token must match                                   |
| `ManagedBarrier` (typestate)           | `init()` and `inval()` require `unsafe`; state machine enforced at compile time |

### Atomics

Atomic operations are safe to *call* once you have a valid atomic reference.
The `unsafe` surface is at construction -- creating a `DeviceAtomicU32`
from a raw pointer requires the caller to guarantee that the pointer is
valid and properly aligned:

```rust
let atom = unsafe { DeviceAtomicU32::new(ptr) };
atom.fetch_add(1, Ordering::Relaxed);  // safe call
```

### Unchecked slice access

When the "one thread, one element" model does not fit -- for instance, in a
warp-level reduction where only lane 0 writes the result --
`DisjointSlice::get_unchecked_mut(usize)` provides an escape hatch:

```rust
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    // SAFETY: Only lane 0 of each warp writes; warp indices are unique
    unsafe { *out.get_unchecked_mut(warp_idx) = sum; }
}
```

The safety obligation is the same as the `ThreadIndex` system enforces
automatically: index in bounds, no two threads share the same index. The
difference is that you prove it yourself instead of letting the type system
do it for you.

---

## Tier 3: raw hardware

At the bottom of the stack are the raw hardware intrinsics -- the APIs
that talk directly to specific functional units on specific GPU
architectures. Every call is `unsafe`, the safety contracts are complex
and architecture-dependent, and the documentation lives in the PTX ISA
manual more than in Rust doc comments.

| Feature                              | Key APIs                                                      | Architectures      |
|:-------------------------------------|:--------------------------------------------------------------|:-------------------|
| **TMA** (Tensor Memory Accelerator)  | `tma_load_2d`, `tma_store_2d`, `TmaDescriptor`                | sm_90+ (Hopper)    |
| **tcgen05** (Tensor Core Gen 5)      | `tcgen05_mma`, `tcgen05_commit`, `TensorMemoryHandle`         | sm_120 (Blackwell) |
| **WGMMA** (Warpgroup MMA)            | `wgmma_mma_async`, `wgmma_commit_group`, `wgmma_wait_group`   | sm_90+ (Hopper)    |
| **Cluster**                          | `cluster_rank`, `map_shared_rank`, `cluster_barrier_arrive`   | sm_90+ (Hopper)    |
| **CLC** (Cluster Launch Control)     | `clc_prefetch`, `clc_query_channel`                           | sm_120 (Blackwell) |
| **TMEM** (Tensor Memory)             | `TmemGuard` (typestate), `tmem_alloc`, `tmem_dealloc`         | sm_120 (Blackwell) |

If you are writing application-level kernels, you should not need Tier 3
APIs. They exist for the people building the next CUTLASS -- and for those
people, cuda-oxide provides the same hardware access as inline PTX in
CUDA C++, with Rust's type system available (but not enforced) as a
guardrail.

:::{seealso}
[Tensor Memory Accelerator](../advanced/tensor-memory-accelerator.md),
[Matrix Multiply Accelerators](../advanced/matrix-multiply-accelerators.md),
and [Cluster Programming](../advanced/cluster-programming.md) for detailed
coverage of Tier 3 features.
:::

---

## What the borrow checker gives you

cuda-oxide is not a DSL or a macro system -- it runs the real `rustc`
frontend on your kernel code. That means every safety guarantee Rust
provides on the CPU is also enforced on the GPU:

| Guarantee                            | How It Works                                                                                                                                             |
|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Ownership and borrowing**          | Lifetime errors, use-after-free, and aliasing violations caught at compile time                                                                          |
| **Safe parallel writes**             | `DisjointSlice<T>` + `ThreadIndex` -- type-level proof that writes do not race                                                                           |
| **Explicit `unsafe` scoping**        | Raw pointer access requires `unsafe`, making obligations visible and auditable                                                                           |
| **Convergent attribute enforcement** | Sync primitives (barriers, fences, shuffles) marked `convergent` in the IR, preventing the optimizer from moving or duplicating them across control flow |

The first three are standard Rust. The fourth is GPU-specific: CUDA's
`bar.sync`, fence, and warp shuffle instructions must not be duplicated or
reordered by the compiler. cuda-oxide marks them `convergent` in the IR so
that LLVM's optimization passes leave them alone.

---

## The hard problems

Rust's borrow checker was designed for single-threaded ownership with
`Send`/`Sync` for CPU concurrency. SIMT execution introduces patterns that
the borrow checker was never taught to reason about. Here is an honest
accounting of what cuda-oxide does *not* enforce today -- and why these
problems are solvable.

### Thread-divergent control flow

Rustc's JumpThreading MIR optimization duplicates function calls into both
branches of an if-statement -- a perfectly sound optimization on CPUs, but
it breaks GPU barrier semantics where all threads in a block must converge
at the same `bar.sync` instruction. cuda-oxide currently disables
JumpThreading for device code (`-Z mir-enable-passes=-JumpThreading`). A
proper solution would teach the compiler about convergence requirements so
it can optimize around them instead of disabling the pass entirely.

### Shared memory access patterns

The borrow checker cannot reason about whether thread 0 writing `smem[0]`
and thread 1 writing `smem[1]` is safe -- it sees `&mut smem` and rejects
it. `DisjointSlice` solves the unique-index-write pattern, but not
cooperative patterns like reductions, scans, or producer/consumer pipelines
where multiple threads intentionally access overlapping regions with
synchronization between phases.

### Warp-level convergence

Operations like `shfl_sync` and `ballot_sync` require that all threads
named in the participation mask are actually converged at the call site.
The type system cannot enforce this today. If threads have diverged and you
pass a full mask, you get a silent hang -- the worst kind of bug, because
there is no crash and no error message, just a kernel that never finishes.

### Memory space awareness

GPU memory has distinct address spaces -- global, shared, local, TMEM.
A `&mut` to shared memory is visible to every thread in the block; a
`&mut` to local memory is private to one thread. The borrow checker treats
them identically. This is conservative (it rejects some safe programs) but
never unsound (it does not accept unsafe ones). Still, a memory-space-aware
borrow checker could accept more programs without `unsafe`.

### Why these are solvable

The building blocks already exist in Rust's type system. They need to be
extended, not reinvented:

| Idea                                  | What It Solves                                                                                                                                                    |
|:--------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Execution-resource-aware types**    | Functions annotated with their execution level (grid / block / warp / thread). A barrier call inside a divergent branch becomes a compile-time error.             |
| **Memory views**                      | Generalized parallel access patterns -- like `DisjointSlice` but covering blocked, striped, transposed, and composed layouts. Type-checked race-freedom at scale. |
| **Extended borrow checking for sync** | Statically enforce that barriers cannot be forgotten, placed at divergent control flow, or duplicated by the optimizer. Convergence in the type system.           |

All of this is compile-time analysis. The generated PTX is identical to what
you would write by hand -- the safety net disappears at code generation.
Zero runtime cost.

cuda-oxide is well-positioned to deliver this incrementally. The real `rustc`
borrow checker already runs on device code. The IR infrastructure (pliron
dialects) supports GPU-aware analysis passes. The full compilation pipeline
from MIR to PTX is under our control. And each new safety check is additive
-- existing kernels keep compiling while new ones get stronger guarantees.

---

## Writing safe kernels: a cheat sheet

### The default path

For most kernels, start here:

```rust
#[kernel]
pub fn my_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(out) = output.get_mut(idx) {
        *out = transform(input[idx.get()]);
    }
}
```

The rules:

- Use `DisjointSlice` for all mutable outputs.
- Use `&[T]` for all read-only inputs.
- Use `index_1d()` for 1D grids, `index_2d(stride)` for 2D grids.
- Always bounds-check via `get_mut()` (returns `Option`).

If your kernel compiles without `unsafe`, it is race-free by construction.

### When you need `unsafe`

| Pattern              | Why                                           | Mitigation                                                          |
|:---------------------|:----------------------------------------------|:--------------------------------------------------------------------|
| Shared memory        | Multiple threads access the same `static mut` | Synchronize with `sync_threads()` before cross-thread reads         |
| Warp shuffles        | Thread convergence is not compiler-checked    | Use `FULL_MASK` for full-warp operations; document partial masks    |
| Atomics              | Construction from a raw pointer               | Wrap in a helper; the atomic operations themselves are safe         |
| Non-uniform writes   | Not every thread writes to its own index      | Use `get_unchecked_mut` with a documented uniqueness argument       |
| Hardware intrinsics  | Complex, architecture-specific contracts      | Follow the PTX ISA documentation; test on target hardware           |

### The `SAFETY` comment

For every `unsafe` block, document *why* the invariants hold. Not what the
code does -- the code already says that -- but why this particular usage is
safe:

```rust
// SAFETY: Only lane 0 of each warp executes this branch.
// Warp indices (gid / 32) are unique across warps, so no two
// threads write to the same output element.
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    unsafe { *partial_sums.get_unchecked_mut(warp_idx) = warp_sum; }
}
```

This is not ceremony. When a kernel data-races at 2 AM and you are staring
at a `compute-sanitizer` log, past-you's safety comments are the fastest
path to the bug.

:::{tip}
If you cannot write a convincing `SAFETY` comment for an `unsafe` block,
that is a signal that the invariant is not actually maintained. Restructure
the code until the argument is obvious, or use a safe API instead.
:::

---

## Summary

| Property                                                  | Status                                                          |
|:----------------------------------------------------------|:----------------------------------------------------------------|
| Borrow checker on device code                             | Enforced (real `rustc` frontend)                                |
| Safe parallel writes (`DisjointSlice` + `ThreadIndex`)    | Enforced for both 1D and 2D grids (`index_2d` returns `Option`) |
| Explicit `unsafe` for shared memory, intrinsics           | Enforced (Rust language rules)                                  |
| Convergent attribute on sync primitives                   | Enforced (IR-level `convergent` marking)                        |
| Thread convergence for warp ops                           | NOT enforced (runtime obligation)                               |
| Memory space awareness (shared vs global)                 | NOT enforced (future work)                                      |

The safety model is designed to make the common case safe by default while
providing explicit escape hatches for everything else. Write your kernel,
let the type system catch the races, and save `unsafe` for the parts where
you genuinely know something the compiler does not.
