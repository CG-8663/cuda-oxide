# Glossary

Terms are defined as they are used in the cuda-oxide project and this book.

---

## ABI Scalarization

The process of decomposing composite types (slices, structs) into scalar kernel
parameters at function boundaries. For example, `&[T]` becomes a `(ptr, len)`
pair in PTX. The compiler reconstructs the composite type inside the function
body via `insertvalue`/`extractvalue`.

## Block (Thread Block)

A group of threads that execute on the same Streaming Multiprocessor and share
access to shared memory. Threads within a block can synchronize via
`sync_threads()`. Identified by `blockIdx` and sized by `blockDim`.

## Cluster (Thread Block Cluster)

A Hopper+ (sm_90) feature: a group of up to 16 thread blocks guaranteed to be
co-scheduled on the same GPC. Blocks in a cluster can access each other's shared
memory via Distributed Shared Memory (DSMEM) and synchronize via
`cluster_sync()`. Configured with `#[cluster_launch]`.

## Codegen Backend

The `rustc_codegen_cuda` crate — a custom rustc backend loaded as a dylib. It
intercepts MIR during compilation, lowers it through `dialect-mir` →
`mem2reg` → `dialect-llvm` → LLVM IR → PTX, and emits the PTX alongside the
normal host binary.

## `cuda-async`

The async execution layer. Provides `DeviceOperation` (lazy GPU work
description), `DeviceFuture` (stream-bound execution), and `DeviceBox<T>`
(device-owned memory). Compose work with `zip!`, `and_then`, and `value()`.

## `cuda-device`

The `#![no_std]` device-side crate providing all GPU intrinsics and types:
thread identification, shared memory, warp primitives, barriers, TMA, tensor
cores, atomics, and debug facilities.

## `cuda-core`

Safe RAII wrappers around the CUDA Driver API: `CudaContext`, `CudaStream`,
`DeviceBuffer<T>`, and module loading. Handles GPU context and memory management
on the host side.

## `DeviceOperation`

A lazy, composable description of GPU work (allocation, kernel launch, or data
transfer). Not executed until `.sync()` or `.await` is called. Can be combined
with `zip!` (parallel) and `and_then` (sequential).

## `DisjointSlice<T>`

A safe mutable output abstraction for kernels. Accepts only `ThreadIndex` for
mutable access, providing bounds-checked `Option<&mut T>` returns. Prevents
data races by construction — each thread can only write to its own element.

## Distributed Shared Memory (DSMEM)

A Hopper+ feature allowing blocks within a cluster to read and write each
other's shared memory directly, without going through global memory. Accessed
via `cluster::map_shared_rank()` to translate local shared pointers to remote
addresses.

## Grid

The top-level organization of threads in a kernel launch. A grid is a
3-dimensional array of thread blocks, sized by `gridDim`. The total number of
threads is `gridDim × blockDim`.

## HMM (Heterogeneous Memory Management)

A Linux kernel feature (6.1.24+) that lets the GPU access host memory directly
through page faults, without explicit `cudaMemcpy`. cuda-oxide leverages HMM for
reference captures in closures — the GPU reads host addresses transparently.
Also called Unified Memory Management (UMM).

## Lane

A single thread within a warp. Lanes are numbered 0–31 and identified by
`warp::lane_id()`. Warp shuffle and vote operations communicate between lanes
without shared memory or barriers.

## LTOIR (Link-Time Optimized IR)

An intermediate representation used for device-side link-time optimization.
cuda-oxide can emit LTOIR (via `--dlto`) for linking Rust device code with CUDA
C++ device code using `nvJitLink`.

## `ManagedBarrier<State, Kind, ID>`

A typestate barrier for async operations on Hopper+. Tracks its lifecycle at
compile time (`Uninit → Ready → Invalidated`) and its purpose (`TmaBarrier`,
`MmaBarrier`, `GeneralBarrier`). Invalid state transitions are compile errors.

## Monomorphization

The process by which the Rust compiler generates specialized copies of generic
functions for each concrete type used. cuda-oxide fully supports
monomorphization on device — `scale::<f32>` and `scale::<f64>` each become
separate PTX functions.

## Pliron

An MLIR-inspired IR framework written in Rust, used as the intermediate
representation in the cuda-oxide compilation pipeline. MIR is imported into
Pliron IR, transformed through dialect passes, and exported as LLVM IR.

## PTX (Parallel Thread Execution)

NVIDIA's low-level virtual ISA for GPU kernels. The cuda-oxide compiler emits
PTX as its primary output, which the CUDA driver JIT-compiles to native GPU
machine code (SASS) at load time.

## `SharedArray<T, N, ALIGN>`

A compile-time sized, block-scoped shared memory array. Declared as
`static mut` in kernel functions. Optional alignment parameter (use `ALIGN=128`
for TMA destinations). Access requires `unsafe` because shared memory is `!Sync`.

## SM (Streaming Multiprocessor)

The primary processing unit on an NVIDIA GPU. Each SM has its own registers,
shared memory, warp schedulers, and execution pipelines including Tensor Cores.
Thread blocks are scheduled onto SMs by the hardware scheduler.

## `sync_threads()`

The block-level barrier: all threads in the thread block must reach this point
before any thread proceeds past it. Equivalent to `__syncthreads()` in CUDA C++.
Lowers to `llvm.nvvm.barrier0()`.

## Tensor Cores

Specialized matrix multiply-accumulate hardware units. WGMMA (Hopper, sm_90)
operates at warpgroup granularity from shared memory. tcgen05 (Blackwell,
sm_100+) uses single-thread issue with dedicated Tensor Memory (TMEM).

## `ThreadIndex`

An opaque newtype that can only be constructed by `thread::index_1d()` or
`thread::index_2d(row_stride)`. `index_1d` always returns a `ThreadIndex`;
`index_2d` returns `Option<ThreadIndex>`, yielding `None` when the thread's
column index exceeds the stride — enforcing the uniqueness invariant.
Guarantees that each thread holds a unique value, enabling `DisjointSlice`
to enforce data-race freedom at the type level.

## TMA (Tensor Memory Accelerator)

Hopper+ hardware unit for asynchronous bulk copies between global and shared
memory. Operates via `TmaDescriptor` (128-byte opaque descriptor built on host)
and `cp_async_bulk_tensor_*` intrinsics. Completion tracked by `ManagedBarrier`.

## `TmemGuard<State, N_COLS>`

A typestate wrapper for Blackwell Tensor Memory (TMEM) — dedicated accumulator
storage for tcgen05 MMA operations. Manages TMEM lifetime:
`TmemUninit → TmemReady → TmemDeallocated`. Invalid transitions are compile
errors.

## Warp

A group of 32 threads that execute instructions in lockstep on an SM. Warps are
the smallest scheduling unit. Warp-level operations (shuffle, vote) exchange
data between lanes in ~1 cycle without shared memory or synchronization barriers.

## Warpgroup

Four consecutive warps (128 threads) that operate collectively for WGMMA
operations on Hopper. The warpgroup is the issuing unit for warpgroup-level
matrix multiply-accumulate.

## WGMMA (Warpgroup Matrix Multiply-Accumulate)

The Hopper (sm_90) tensor core instruction set. Four warps collectively issue
MMA from shared memory operands into register accumulators. Asynchronous
execution with commit/wait via `ManagedBarrier`.
