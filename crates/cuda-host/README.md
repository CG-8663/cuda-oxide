# cuda-host

Host-side infrastructure for launching CUDA kernels compiled by cuda-oxide. Provides the `cuda_launch!` and `cuda_launch_async!` macros, kernel marker traits, argument wrappers, and tcgen05 tiling utilities.

```text
  #[kernel]
  pub fn vecadd(...)           cuda-macros generates:
       │                        • cuda_oxide_kernel_vecadd (entry point)
       │                        • __vecadd_CudaKernel (marker struct)
       ▼
  cuda_launch! {               cuda-host provides:
      kernel: vecadd,           • CudaKernel / GenericCudaKernel traits
      stream: ...,              • Argument wrappers (Scalar, ReadOnly, WriteOnly)
      module: ...,              • cuda_launch! / cuda_launch_async! (re-exported from cuda-macros)
      config: ...,              • HasLength trait for DeviceBuffer
      args: [...]
  }
```

## Kernel Launch

### Synchronous (`cuda_launch!`)

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
    let module = ctx.load_module_from_file("vecadd.ptx").unwrap();
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_dev = DeviceBuffer::from_host(&stream, &vec![1.0f32; N]).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &vec![2.0f32; N]).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    cuda_launch! {
        kernel: vecadd,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }
    .unwrap();
}
```

### Asynchronous (`cuda_launch_async!`)

Returns an `AsyncKernelLaunch` implementing `DeviceOperation` for use with `cuda-async` scheduling. No `stream:` field -- the stream is chosen by the async scheduler.

```rust
use cuda_host::cuda_launch_async;

let op = cuda_launch_async! {
    kernel: vecadd,
    module: module,
    config: LaunchConfig::for_num_elems(N as u32),
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
};
op.sync()?;  // or .await in async context
```

### Macro Fields

| Field          | Required | Description                                              |
|----------------|----------|----------------------------------------------------------|
| `kernel`       | Yes      | Kernel name or `name::<T>` for generics                  |
| `stream`       | Sync only| CUDA stream for execution                                |
| `module`       | Yes      | Loaded PTX module                                        |
| `config`       | Yes      | `LaunchConfig` (grid/block dims)                         |
| `cluster_dim`  | No       | `(x, y, z)` cluster dimensions (uses `launch_kernel_ex`) |
| `args`         | Yes      | `[...]` argument list                                    |

### Argument Syntax

| Syntax          | Kernel Parameter        | What's Passed                  |
|-----------------|-------------------------|--------------------------------|
| `expr`          | `T` (scalar)            | `&value` as raw pointer        |
| `slice(buf)`    | `&[T]`                  | device ptr + len (two args)    |
| `slice_mut(buf)`| `DisjointSlice<T>`      | device ptr + len (two args)    |
| `move \|..\| ..`| Closure `F`             | Each capture as separate arg   |
| `\|..\| ..`     | Closure `F` (by-ref)    | Pointers to captures (HMM)     |

## Traits

| Trait                | Purpose                                              |
|----------------------|------------------------------------------------------|
| `CudaKernel`         | Non-generic kernels; `const PTX_NAME: &str`          |
| `GenericCudaKernel`  | Generic kernels; `fn ptx_name() -> &'static str`     |
| `HasLength`          | Types with `.len()` (implemented for `DeviceBuffer`) |

## Argument Wrappers

| Type             | Description                         |
|------------------|-------------------------------------|
| `Scalar<T>`      | Pass-by-value scalar                |
| `ReadOnly<'a,T>` | Read-only device buffer reference   |
| `WriteOnly<'a,T>`| Write-only device buffer reference  |

## Tiling Utilities (tcgen05)

Host-side layout transformations for Blackwell tensor cores. tcgen05 requires specific 8x8 tile arrangements:

| Function               | Description                               |
|------------------------|-------------------------------------------|
| `to_k_major_f16`       | Row-major → tcgen05 K-major (matrix A)    |
| `to_mn_major_f16`      | Row-major → tcgen05 MN-major (matrix B)   |
| `k_major_index`        | Compute linear index in K-major layout    |
| `mn_major_index`       | Compute linear index in MN-major layout   |
| `print_layout_indices` | Debug print layout as 2D table            |
| `TILE_SIZE`            | Constant `8` (8x8 tile for f16/bf16)      |

## Source Layout

```text
src/
├── lib.rs       # Re-exports from launch, tiling, and cuda-macros
├── launch.rs    # CudaKernel, GenericCudaKernel, argument wrappers, HasLength
└── tiling.rs    # tcgen05 tile layout transformations
```

## Further Reading

- [cuda-device](../cuda-device/) -- device-side intrinsics
- [cuda-macros](../cuda-macros/) -- proc-macro implementations
- [cuda-core](../cuda-core/) -- CUDA driver API, `DeviceBuffer`, `LaunchConfig`
- [cuda-async](../cuda-async/) -- async scheduling for `cuda_launch_async!`
