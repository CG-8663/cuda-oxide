# cuda-async

Async execution layer for CUDA device operations, built on top of `cuda-core`.

## Architecture

The crate is organized around a single idea: GPU work is described lazily, scheduled late, and composed freely before any hardware instruction is issued.

```text
  module.kernel_async(...)
         |
         v
  AsyncKernelLaunch          <-- lazy description, no GPU work yet
         |
    .and_then(|()| ...)      <-- compose with other DeviceOperations
         |
    .sync() / .await         <-- SchedulingPolicy picks a stream, executes
         |
         v
  cuLaunchKernel(stream)     <-- actual GPU dispatch
  cuLaunchHostFunc(stream)   <-- host callback wakes the Rust future
```

## Core concepts

### `DeviceOperation` trait

A lazy, composable unit of GPU work. Implements `Send + Sized + IntoFuture`. Key methods:

| Method              | Stream chosen by                      | Blocks thread? |
|---------------------|---------------------------------------|----------------|
| `.await`            | Default device's `SchedulingPolicy`   | No (suspends)  |
| `.sync()`           | Default device's `SchedulingPolicy`   | Yes            |
| `.sync_on(&stream)` | The explicit stream you provide       | Yes            |

Combinators: `.and_then(f)`, `.and_then_with_context(f)`, `.arc()`, `zip!(a, b)`, `unzip!(op)`.

### `DeviceFuture`

Bridges CUDA stream completion to Rust's `Future` trait. When a `DeviceOperation` is scheduled, `DeviceFuture` enqueues the work on a CUDA stream, then registers a host callback via `cuLaunchHostFunc` that wakes the async task when the GPU finishes.

### `SchedulingPolicy`

Determines which CUDA stream a `DeviceOperation` runs on:

- **`StreamPoolRoundRobin`** (default) -- rotates through a pool of N streams, enabling automatic overlap of independent operations.
- **`SingleStream`** -- all operations execute on one stream in strict FIFO order.

### `DeviceBox<T>`

Owning smart pointer for device memory. Frees memory asynchronously via `cuMemFreeAsync` on a dedicated deallocator stream when dropped, avoiding the full device synchronization that `cuMemFree` would cause.

### `AsyncDeviceContext`

Thread-local per-device state: CUDA context, scheduling policy, deallocator stream, and a kernel function cache. Initialized via `init_device_contexts(default_device_id, num_devices)`.

## Usage

Borrow buffers when the launch completes in the current scope:

```rust
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_host::cuda_module;
use cuda_core::LaunchConfig;

// 1. Enable cuda-host's "async" feature, then initialize once per thread.
init_device_contexts(0, 1)?;
let module = kernels::load_async(0)?;

// 2. Build a lazy operation
let op = module.vecadd_async(
    LaunchConfig::for_num_elems(1024),
    &a_dev,
    &b_dev,
    &mut c_dev,
)?;

// 3. Execute
op.sync()?;       // blocking
// or: op.await?  // async
```

Move buffers into the launch when it needs to be spawned or stored as a
`'static` future:

```rust
use std::future::IntoFuture;

let op = module.vecadd_async_owned(
    LaunchConfig::for_num_elems(1024),
    a_dev,
    b_dev,
    c_dev,
)?;

let (a_dev, b_dev, c_dev) = tokio::spawn(op.into_future()).await??;
```

The owned form keeps device buffers alive until the CUDA stream reaches the
kernel completion callback, then returns those buffers as the operation output.
