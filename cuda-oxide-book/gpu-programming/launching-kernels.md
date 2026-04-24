# Launching Kernels

Writing a kernel is only half the story. The host must load the compiled PTX,
configure the launch grid, marshal arguments, and dispatch the work to the GPU.
cuda-oxide provides two launch paths: `cuda_launch!` for straightforward
synchronous enqueuing, and `cuda_launch_async!` for composable, lazy execution
graphs.

:::{seealso}
[CUDA Programming Guide -- Execution Configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/#execution-configuration)
for the authoritative reference on `<<<grid, block, smem, stream>>>` semantics.
:::

## The launch lifecycle

Every kernel launch follows the same sequence:

1. **Initialize a CUDA context** -- bind to a GPU device.
2. **Load the PTX module** -- the compiled device code produced by
   `cargo oxide build`.
3. **Look up the kernel function** -- by its PTX entry point name.
4. **Configure the grid** -- block dimensions, grid dimensions, shared memory.
5. **Launch** -- enqueue the kernel on a stream.
6. **Synchronize** -- wait for results (explicit or implicit).

```{figure} images/launch-lifecycle.svg
:align: center
:width: 100%

The kernel launch lifecycle. The host initializes a context, loads the PTX
module, configures the grid, and launches via cuda_launch! (which handles
steps 3-5 in one call). The GPU scheduler dispatches blocks to SMs.
```

In practice, `cuda_launch!` and `cuda_launch_async!` handle steps 3--5 in a
single macro invocation. You typically only interact with context creation and
module loading directly.

## `cuda_launch!` -- synchronous launch

The `cuda_launch!` macro is the standard way to launch a kernel. It is
"synchronous" in the sense that you provide a specific stream and the kernel is
enqueued immediately (though execution on the GPU is asynchronous relative to the
host):

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_launch;

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = ctx.load_module_from_file("vecadd.ptx").unwrap();

    let a = DeviceBuffer::from_host(&stream, &[1.0f32; 1024]).unwrap();
    let b = DeviceBuffer::from_host(&stream, &[2.0f32; 1024]).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    cuda_launch! {
        kernel: vecadd,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(1024),
        args: [slice(a), slice(b), slice_mut(c)]
    }
    .expect("Kernel launch failed");

    let result = c.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 3.0);
}
```

### Field-by-field breakdown

| Field    | Type              | Description                                          |
|:---------|:------------------|:-----------------------------------------------------|
| `kernel` | Path              | Kernel name (`vecadd`) or generic (`scale::<f32>`)   |
| `stream` | `Arc<CudaStream>` | The stream to enqueue on                             |
| `module` | `CudaModule`      | Loaded PTX module containing the kernel              |
| `config` | `LaunchConfig`    | Grid/block dimensions and shared memory              |
| `args`   | `[...]`           | Kernel arguments (see below)                         |

### Argument wrappers

The `args` list uses wrapper functions to handle
[scalarization](memory-and-data-movement.md#argument-scalarization):

| Syntax              | Kernel parameter   | What's passed to the GPU |
|:--------------------|:-------------------|:-------------------------|
| `slice(buf)`        | `&[T]`             | Pointer + length         |
| `slice_mut(buf)`    | `DisjointSlice<T>` | Pointer + length         |
| `Scalar(val)`       | `T`                | Value directly           |
| `move \|x\| expr`   | Closure `F`        | Captures individually    |

### Return value

`cuda_launch!` returns `Result<(), DriverError>`. The `Ok` case means the
kernel was successfully **enqueued** -- not that it finished. To check for
runtime errors (e.g., out-of-bounds trap), synchronize the stream or context
afterward.

## `LaunchConfig`

`LaunchConfig` specifies the grid shape:

```rust
use cuda_core::LaunchConfig;

let config = LaunchConfig {
    grid_dim: (num_blocks, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
};
```

| Field              | Type              | Description                     |
|:-------------------|:------------------|:--------------------------------|
| `grid_dim`         | `(u32, u32, u32)` | Number of blocks in x, y, z     |
| `block_dim`        | `(u32, u32, u32)` | Threads per block in x, y, z    |
| `shared_mem_bytes` | `u32`             | Dynamic shared memory per block |

### `for_num_elems` helper

For 1D data-parallel kernels, the common pattern is one thread per element:

```rust
let config = LaunchConfig::for_num_elems(N as u32);
```

This uses 256 threads per block and computes the grid size via ceiling
division: `grid_x = (N + 255) / 256`. It's the right default for most
element-wise operations.

### 2D and 3D configurations

For matrix operations, use 2D block and grid dimensions:

```rust
let config = LaunchConfig {
    grid_dim: ((cols + 15) / 16, (rows + 15) / 16, 1),
    block_dim: (16, 16, 1),
    shared_mem_bytes: 0,
};
```

Inside the kernel, combine `threadIdx_x()` / `blockIdx_x()` with their `_y()`
counterparts to compute row and column indices.

### Choosing block size

The block size is the single most important tuning parameter (see the
[Execution Model](execution-model.md#choosing-block-size) chapter for details).
Quick guidelines:

- **256** is a safe default for most kernels.
- **Powers of 2** (128, 256, 512) align with warp boundaries.
- Use `#[launch_bounds]` to hint the compiler about your intended block size.

## `cuda_launch_async!` -- composable async launch

The async launch macro returns a `DeviceOperation` instead of enqueuing
immediately. No stream is specified at launch time -- the scheduling policy
chooses one when the operation is executed:

```rust
use cuda_async::{init_device_contexts, DeviceOperation};
use cuda_host::cuda_launch_async;

init_device_contexts(0, 1)?;

let op = cuda_launch_async! {
    kernel: vecadd,
    module: module,
    config: LaunchConfig::for_num_elems(1024),
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
};

// Execute and wait
op.sync()?;
```

### `push_arg` and `push_args`

Under the hood, `cuda_launch_async!` builds an `AsyncKernelLaunch` and calls
`push_arg` for each argument. You can also build launches manually with
`push_args` for tuple-based argument passing:

```rust
let mut launch = AsyncKernelLaunch::new(func, config);
launch.push_args((ptr_a, len_a, ptr_b, len_b, ptr_c, len_c));
```

`push_args` accepts tuples of up to 32 elements, where each element implements
`KernelArgument`. Scalar types (`u32`, `f32`, `u64`, etc.) are boxed
automatically.

### `.sync()` vs `.await`

| Method    | What it does                                                                      |
|:----------|:----------------------------------------------------------------------------------|
| `.sync()` | Execute on the default scheduling policy, block the current thread until complete |
| `.await`  | Execute and yield the current async task (requires a Tokio runtime)               |

## Composing GPU work

`DeviceOperation` supports functional composition. Chain operations with
`and_then` and run independent work in parallel with `zip!`:

```rust
use cuda_async::zip;

let forward_pass = layer1_op
    .and_then(|output1| layer2_op(output1))
    .and_then(|output2| layer3_op(output2));

// Run two independent operations concurrently
let combined = zip!(branch_a, branch_b);
let (result_a, result_b) = combined.sync()?;
```

Each operation in the chain is scheduled onto a stream only when it executes.
The `and_then` combinator passes the output of one operation as input to the
next, forming a lazy computation graph.

:::{seealso}
The [Async GPU Programming](../async-programming/the-device-operation-model.md)
section covers `DeviceOperation`, scheduling policies, and stream management in
depth.
:::

## Cluster launch

Thread Block Clusters (Hopper and newer) allow blocks to cooperate beyond shared
memory via **distributed shared memory** (DSMEM). To launch with clusters, add
`#[cluster_launch]` to the kernel and include `cluster_dim` in the launch:

```rust
use cuda_device::{kernel, cluster, cluster_launch, DisjointSlice};

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(mut out: DisjointSlice<u32>) {
    let rank = cluster::block_rank();
    // Blocks 0-3 can communicate via DSMEM
}
```

On the host, the launch uses `launch_kernel_ex` (the extended launch API) with
cluster dimensions. `cuda_launch!` supports this via the `cluster_dim` field:

```rust
cuda_launch! {
    kernel: cluster_kernel,
    stream: stream,
    module: module,
    config: config,
    cluster_dim: (4, 1, 1),
    args: [slice_mut(out_dev)]
}
.expect("Cluster launch failed");
```

:::{tip}
Cluster launch requires **Hopper (sm_90)** or newer. The maximum cluster size is
typically 16 blocks. Use `cargo oxide build --arch sm_90` to target Hopper.
:::

## Common launch errors

| Error                                  | Likely cause                                           | Fix                                                                  |
|:---------------------------------------|:-------------------------------------------------------|:---------------------------------------------------------------------|
| `CUDA_ERROR_INVALID_VALUE`             | Grid or block dimensions are zero or exceed limits     | Check `LaunchConfig` values; max block is 1024 threads               |
| `CUDA_ERROR_NOT_FOUND`                 | PTX entry point name doesn't match                     | Verify `#[kernel]` name matches the loaded module                    |
| `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`   | Too much shared memory or too many registers per block | Reduce `shared_mem_bytes` or block size; use `#[launch_bounds]`      |
| `CUDA_ERROR_ILLEGAL_INSTRUCTION`       | Kernel hit a trap (panic, assert failure, OOB)         | Debug with `cargo oxide debug` or `gpu_printf!`                      |
| `CUDA_ERROR_NO_BINARY_FOR_GPU`         | PTX compiled for wrong architecture                    | Rebuild with `--arch` matching your GPU                              |

:::{seealso}
The [Error Handling and Debugging](error-handling-and-debugging.md) chapter
covers how to diagnose and fix kernel failures in detail.
:::
