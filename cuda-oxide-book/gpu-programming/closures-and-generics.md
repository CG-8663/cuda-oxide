# Closures and Generics

Rust's zero-cost abstractions -- generics, closures, and trait bounds -- work on
the GPU. This is one of cuda-oxide's most distinctive capabilities: you can write
a single generic kernel that operates on any numeric type, or pass a closure from
the host to customize GPU behavior, all without runtime overhead.

## Generic kernels

A kernel can be generic over types and trait bounds, just like any Rust function.
The compiler monomorphizes each instantiation into a separate PTX entry point:

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use core::ops::Mul;

#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(
    factor: T,
    input: &[T],
    mut out: DisjointSlice<T>,
) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[idx.get()] * factor;
    }
}
```

### PTX naming

Each monomorphization produces a distinct PTX entry point. The name is derived
from the function name and the concrete type parameters:

| Instantiation      | PTX entry point name |
|:-------------------|:---------------------|
| `scale::<f32>`     | `scale__f32`         |
| `scale::<i32>`     | `scale__i32`         |
| `scale::<MyType>`  | `scale__MyType`      |

### Launching generic kernels

When launching, specify the type parameter explicitly so the macro can look up
the correct PTX entry point via `GenericCudaKernel::ptx_name()`:

```rust
use cuda_host::{cuda_launch, Scalar};
use cuda_core::LaunchConfig;

cuda_launch! {
    kernel: scale::<f32>,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems(N as u32),
    args: [Scalar(2.0f32), slice(input_dev), slice_mut(output_dev)]
}
.expect("Launch failed");
```

The `cuda_launch!` macro forces monomorphization of `scale::<f32>` using a
volatile pointer trick that prevents dead-code elimination, ensuring the
instantiation appears in the compiled PTX even though it is never "called"
directly on the host.

## Host closures as kernel arguments

cuda-oxide supports passing closures from the host to the GPU. This enables
powerful `map`-style patterns where the kernel's behavior is parameterized by
a function:

```rust
#[kernel]
pub fn map<F: Fn(i32) -> i32>(f: F, input: &[i32], mut out: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[idx.get()]);
    }
}
```

Launch with a closure:

```rust
let factor = 3i32;
cuda_launch! {
    kernel: map::<_>,
    stream: stream,
    module: module,
    config: config,
    args: [
        move |x| x * factor,
        slice(input_dev),
        slice_mut(output_dev)
    ]
}
.expect("Launch failed");
```

### How capture extraction works

The `cuda_launch!` macro analyzes the closure at compile time:

1. **Identify captures** -- walk the closure body's AST, collecting identifiers
   that are not parameters or local bindings. In `move |x| x * factor`, `x` is
   a parameter and `factor` is a capture.
2. **Scalarize captures** -- each captured variable becomes a separate kernel
   parameter (just like [argument scalarization](memory-and-data-movement.md#argument-scalarization)).
3. **Reconstruct on device** -- inside the kernel, the compiler reassembles the
   closure from its individual scalar fields.

```{figure} images/closure-capture-flow.svg
:align: center
:width: 100%

Closure capture extraction: the cuda_launch! macro analyzes the closure AST,
extracts captured variables (factor, offset), and passes each as a separate
scalarized kernel parameter. The device kernel reconstructs the closure from
its individual scalar fields.
```

### PTX naming for closures

Closure kernels get unique PTX names based on source location to avoid
collisions when multiple closures instantiate the same generic kernel:

| Closure                                      | PTX entry point |
|:---------------------------------------------|:----------------|
| `move \|x\| x * factor` at line 42, col 8    | `map_L42C8`     |
| `move \|x\| x + offset` at line 50, col 8    | `map_L50C8`     |

## Move vs reference closures

The `move` keyword determines how captures are transferred to the GPU:

### Move closures (recommended default)

```rust
let factor = 3i32;
move |x| x * factor   // `factor` is copied to the GPU
```

- Each capture is **copied by value** to the device through scalarized kernel
  parameters.
- The host value can be dropped after launch.
- Works on all systems -- no special hardware support needed.

### Reference closures (HMM)

```rust
let factor = 3i32;
|x| x * factor   // `factor` stays on host; GPU accesses via pointer
```

- Captures are passed as **pointers to host memory**.
- The GPU reads them through **Hardware-Managed Memory (HMM)** -- automatic
  page migration from host to device on access.
- The host variable **must remain alive** until the kernel completes.
- Requires HMM support (Turing+ GPU, Linux 6.1.24+, CUDA 12.2+).

### When to use which

| Scenario                                       | Use                                                               |
|:-----------------------------------------------|:------------------------------------------------------------------|
| Small scalar captures (numbers, booleans)      | `move` (zero-copy overhead)                                       |
| Large struct captures                          | `move` if the kernel reads it many times; HMM if rarely accessed  |
| Prototyping                                    | Either works; `move` is more portable                             |
| Shared mutable state between host and device   | Reference (HMM) -- but beware synchronization                     |

:::{tip}
When in doubt, use `move` closures. They are simpler to reason about, work
everywhere, and avoid the synchronization hazards of shared host/device memory.
:::

## In-kernel closures

Closures defined and called entirely within device code work with normal Rust
semantics -- no capture extraction or scalarization is involved because
everything is already on the GPU:

```rust
#[kernel]
pub fn apply_transform(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();

    let transform = |x: f32| -> f32 {
        let clamped = if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x };
        clamped * clamped
    };

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = transform(input[idx.get()]);
    }
}
```

In-kernel closures are inlined by the compiler and have zero overhead. They are
useful for factoring logic within a kernel without introducing a separate device
function.

## Cross-crate kernels

Kernels can be defined in a library crate and launched from a binary crate:

```rust
// In lib crate `my_kernels`:
use cuda_device::{kernel, thread, DisjointSlice};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

```rust
// In binary crate:
use my_kernels::vecadd;

cuda_launch! {
    kernel: vecadd,
    stream: stream,
    module: module,
    config: config,
    args: [slice(a), slice(b), slice_mut(c)]
}
.expect("Launch failed");
```

The compiler handles cross-crate kernel discovery through the `CudaKernel` trait
generated by `#[kernel]`. The PTX name is resolved at compile time via the
marker struct, so there is no runtime lookup overhead.

:::{tip}
For generic cross-crate kernels, the monomorphization happens in the **calling**
crate (where the concrete type is known), so the PTX is generated as part of
the binary's compilation.
:::
