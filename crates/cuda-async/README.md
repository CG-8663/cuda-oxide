# cuda-async

Async execution layer for CUDA device operations, built on top of `cuda-core`.

## Architecture

The crate is organized around a single idea: GPU work is described lazily, scheduled late, and composed freely before any hardware instruction is issued.

```text
  cuda_launch_async! { ... }
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

```rust
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_host::cuda_launch_async;
use cuda_core::LaunchConfig;

// 1. Initialize (once per thread)
init_device_contexts(0, 1)?;
let module = cuda_async::device_context::load_module_from_file("kernel.ptx", 0)?;

// 2. Build a lazy operation
let op = cuda_launch_async! {
    kernel: vecadd,
    module: module,
    config: LaunchConfig::for_num_elems(1024),
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
};

// 3. Execute
op.sync()?;       // blocking
// or: op.await?  // async
```
