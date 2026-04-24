# Writing Your First Kernel

This section walks through installing cuda-oxide, creating a project, writing a GPU kernel, and running it -- all in pure Rust.

---

## Install cargo-oxide

If you haven't already, install the build tool:

```bash
cargo install --git https://gitlab-master.nvidia.com/nihalp/cuda-oxide.git cargo-oxide
```

Verify that your environment is set up correctly:

```bash
cargo oxide doctor
```

This checks for a compatible GPU, CUDA toolkit, LLVM, and the codegen backend. Fix any issues it reports before continuing (see [Installation](installation.md) for details).

---

## Create a project

Scaffold a new project with `cargo oxide new`:

```bash
cargo oxide new my_first_kernel
cd my_first_kernel
```

This generates a ready-to-run project:

```text
my_first_kernel/
├── Cargo.toml          # dependencies on cuda-device, cuda-host, cuda-core
├── rust-toolchain.toml # pins the required nightly toolchain
└── src/
    └── main.rs          # kernel + host code in one file
```

Build and run it:

```bash
cargo oxide run
```

You should see `PASSED: all 1024 elements correct`. The generated template is a vector addition kernel -- a good starting point, but let's look at something more interesting.

---

## Anatomy of a kernel

Here's a vector addition with a twist: the element-wise addition is factored out into a plain helper function. Both the kernel and the helper live in the same file alongside host code:

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_launch;

/// Plain helper function -- no annotation needed.
/// The compiler discovers it automatically because `vecadd` calls it.
fn add(a: f32, b: f32) -> f32 {
    a + b
}

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = add(a[idx.get()], b[idx.get()]);
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = ctx
        .load_module_from_file("my_first_kernel.ptx")
        .expect("Failed to load PTX module");

    cuda_launch! {
        kernel: vecadd,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }
    .unwrap();

    let result = c_dev.to_host_vec(&stream).unwrap();
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 3.0).abs() < 1e-5);
    println!("PASSED");
}
```

There's a lot happening here. Let's unpack the key pieces.

### Single-source compilation

The kernel and host code live in **the same file** and are compiled with a single `cargo` command invocation. The codegen backend intercepts compilation, routes `#[kernel]` functions through the MIR-to-PTX pipeline, and delegates everything else to standard LLVM. The output is a native binary plus a `.ptx` file.

### `#[kernel]`

Marks a function as a **launchable kernel entry point** -- the GPU equivalent of `main`. The function is compiled to PTX via the pipeline:

```text
Rust source → MIR → Pliron IR → LLVM IR → PTX
```

The same function is also visible to the host compiler for type-checking, but its body is never called on the CPU.

### Device functions (auto-discovery)

The `add` helper above has **no annotation**. When the compiler processes a `#[kernel]`, it walks the call graph and automatically discovers every function the kernel calls. Those functions are compiled to PTX as device functions and inlined by the backend -- you don't need to mark them.

:::{note}
The `#[device]` attribute exists but serves a different purpose: it marks a function as a standalone device compilation root (for building Rust device libraries consumed by C++) or is used in `#[device] extern "C" { ... }` blocks to declare external device functions for FFI with CUDA C++ LTOIR. You do **not** need `#[device]` for private helper functions called from a kernel.
:::

### `cuda_launch!`

The launch macro ties everything together at the call site:

```rust
cuda_launch! {
    kernel: vecadd,            // which kernel (by name, with generics)
    stream: stream,            // CUDA stream to launch on
    module: module,            // loaded PTX module
    config: LaunchConfig::for_num_elems(N as u32),  // grid/block dims
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
}
```

The macro looks up the PTX entry point name from the `CudaKernel` trait, loads the function from the module, and marshals arguments for the driver call.

### Argument scalarization

Aggregate types (slices, structs, closures) are **scalarized** at the host/device boundary. A `&[f32]` is decomposed into its `(ptr, len)` components and passed as two separate kernel parameters. On the device side, the compiler reassembles them back into a slice. This avoids ABI mismatches between host and device compilers:

```text
Host:   cuda_launch! { args: [slice(data)] }
          → extracts ptr + len, passes as 2 args

PTX:    .entry kernel(.param .u64 ptr, .param .u64 len, ...)
          → receives flat parameters

Device: kernel body sees unified &[T] slice
          → compiler reconstructs at entry
```

This is fully transparent -- you never see it.

### Dynamic struct layout

When you pass structs to the GPU, cuda-oxide queries rustc for the **exact byte offsets** of each field and rebuilds the layout with explicit padding on the device side. This means `#[repr(C)]` is **not required** -- regular Rust structs work as-is, even across HMM (GPU direct access to host memory).

---

## Going async

For multi-kernel pipelines or concurrent workloads, cuda-oxide provides an async execution model built on Tokio. Let's scaffold an async project and walk through the differences.

### Create an async project

```bash
cargo oxide new my_async_kernel --async
cd my_async_kernel
cargo oxide run
```

The `--async` flag generates a project with `tokio` and `cuda-async` dependencies pre-configured.

### Full example

Here's the generated async vecadd template (with minor formatting edits for readability):

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_launch_async;
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_core::LaunchConfig;

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::device_box::DeviceBox;
    use cuda_core::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    // 1. Initialize the device context map (default device 0, 1 device).
    //    The round-robin stream pool is created lazily on first use.
    init_device_contexts(0, 1)?;

    // 2. Load the PTX module.
    let module = cuda_async::device_context::load_module_from_file(
        "my_async_kernel.ptx", 0,
    )?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    // 3. Allocate device memory and copy host data.
    let (a_dev, b_dev, mut c_dev) =
        cuda_async::device_context::with_cuda_context(0, |ctx| {
            let stream = ctx.default_stream();
            let bytes = N * mem::size_of::<f32>();
            unsafe {
                let a = malloc_async(stream.cu_stream(), bytes).unwrap();
                let b = malloc_async(stream.cu_stream(), bytes).unwrap();
                let c = malloc_async(stream.cu_stream(), bytes).unwrap();
                memcpy_htod_async(a, a_host.as_ptr(), bytes, stream.cu_stream()).unwrap();
                memcpy_htod_async(b, b_host.as_ptr(), bytes, stream.cu_stream()).unwrap();
                stream.synchronize().unwrap();
                (
                    DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
                )
            }
        })?;

    // 4. Launch -- returns a lazy DeviceOperation, no GPU work yet.
    cuda_launch_async! {
        kernel: vecadd,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }
    .sync()?;  // Block until the GPU finishes.

    // 5. Copy results back to host.
    let mut c_host = vec![0.0f32; N];
    cuda_async::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    // 6. Verify.
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}
```

### What changed from sync

The kernel itself is **identical** -- async only changes how you launch and manage GPU work on the host side.

`cuda_launch_async!` instead of `cuda_launch!`
: Returns a lazy `DeviceOperation` rather than launching immediately. No GPU work happens until you explicitly schedule it. This lets you build a computation graph before committing resources.

`init_device_contexts(default_device, num_devices)`
: Initializes the thread-local device context map, setting the default GPU ordinal and capacity for multi-device use. The round-robin stream pool is created lazily on first use. Operations are then assigned to streams in round-robin order, maximizing GPU occupancy without manual stream management.

`DeviceBox` instead of `DeviceBuffer`
: Async-safe wrapper for device memory. Works with the stream pool and supports async allocation via `malloc_async`.

`.sync()` vs `.await`
: `.sync()` blocks the calling thread until the GPU finishes -- use it when you have nothing else to do on the host. `.await` suspends the current Tokio task and lets other tasks progress while waiting -- use it when you have concurrent host work or multiple GPU pipelines in flight.

`and_then` / `zip!`
: Chain dependent operations with `.and_then(|result| next_op)`. Run independent operations concurrently with `zip!(op_a, op_b)` -- both are submitted to the stream pool, and the combined result is available when both complete. These combinators let you express complex multi-kernel pipelines declaratively.

:::{tip}
For a more complete async example, see `async_mlp` -- a multi-kernel forward pass (GEMM, MatVec, ReLU) with `and_then` chaining, `zip!` for parallel allocation, and `Arc`-shared weights across concurrent batches. Run it with `cargo oxide run async_mlp` from the cuda-oxide workspace.
:::
