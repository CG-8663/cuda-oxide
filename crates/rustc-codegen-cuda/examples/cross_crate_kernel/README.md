# Cross-Crate Kernel Example

This example demonstrates **defining kernels in a library crate** and using them from a binary crate.

## Structure

```text
cross_crate_kernel/
├── Cargo.toml           # Binary crate
├── src/main.rs          # Uses kernels from kernel-lib
├── README.md            # This file
└── kernel-lib/          # Library crate with #[kernel] functions
    ├── Cargo.toml
    └── src/lib.rs       # Generic kernels: scale<T>, add<T>, etc.
```

## What This Tests

1. **Generic kernels in library crates** - `kernel_lib::scale<T>` is defined externally
2. **Monomorphization at use site** - `scale::<f32>`, `scale::<i32>` instantiated in binary
3. **Cross-crate PTX generation** - All kernel monomorphizations compiled to PTX
4. **Device helper functions** - `kernel_lib::device_scale_helper` called from kernel

## Run

```bash
cargo oxide run cross_crate_kernel
```

## Expected Output

```text
=== Cross-Crate Kernel Test ===

Testing kernels defined in kernel-lib crate.

Test 1: kernel_lib::scale::<f32>
  ✓ PASSED: scale::<f32> from library works!

Test 2: kernel_lib::scale::<i32>
  ✓ PASSED: scale::<i32> from library works!

Test 3: kernel_lib::add::<f32>
  ✓ PASSED: add::<f32> from library works!

Test 4: kernel_lib::scale_with_helper::<f32> (uses device helper)
  ✓ PASSED: scale_with_helper uses device function from library!

=== All Cross-Crate Tests Passed! ===
```

## How It Works

### 1. Kernel Definition (kernel-lib/src/lib.rs)

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use core::ops::Mul;

#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[idx.get()] * factor;
    }
}
```

### 2. Kernel Usage (src/main.rs)

```rust
use kernel_lib::*;

fn main() {
    // scale::<f32> is monomorphized HERE, not in kernel-lib
    cuda_launch! {
        kernel: scale::<f32>,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [factor, slice(input_dev), slice_mut(output_dev)]
    }
    .expect("Kernel launch failed");
}
```

### 3. PTX Generation

The codegen backend:

1. Finds `cuda_oxide_kernel_scale` marked with `#[kernel]` attribute
2. Discovers all monomorphizations: `scale::<f32>`, `scale::<i32>`, etc.
3. Generates unique PTX entry points: `scale__f32`, `scale__i32`
4. Collects device helper functions transitively

## Key Implementation Details

### Generic Kernel Naming

Each monomorphization gets a unique PTX name:

| Rust Code      | PTX Entry Point |
|----------------|-----------------|
| `scale::<f32>` | `scale__f32`    |
| `scale::<i32>` | `scale__i32`    |
| `add::<f32>`   | `add__f32`      |

The naming scheme:
- Base name extracted from `cuda_oxide_kernel_<name>`
- Type suffix appended: `__<type_name>`

### Cross-Crate Intrinsic Handling

When `kernel-lib` calls `cuda_device::thread::index_1d()`:
- The collector finds the call in the library's MIR
- The MIR importer recognizes it as an intrinsic
- It's expanded inline to NVVM operations (no function call)

### Panic/Unwind Handling

Library code may use standard library types like `core::ops::Mul`. These have unwind paths
in their MIR (compiled without `panic=abort`). The codegen backend handles this by:

1. **Accepting** `UnwindAction::Continue` from external crates
2. **Treating** unwind paths as unreachable (CUDA toolchain doesn't support unwinding today)
3. **No special flags needed** - works with vanilla Rust code

This enables seamless cross-crate kernel development without custom sysroots or special
build configurations.

## See Also

- [generic](../generic/) - Generic kernels in same crate
- [rustc-codegen-cuda README](../../README.md) - Backend documentation
