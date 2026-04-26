# device_global

Tests ordinary Rust `static mut` values in CUDA global memory.

Run with:

```bash
cargo oxide run device_global
```

The kernel updates two ordinary device statics:

```rust
static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;
```

Expected behavior:

| Static kind                 | Memory space       |
|----------------------------|--------------------|
| Ordinary `static mut`      | Global `addrspace(1)` |
| `SharedArray` / `Barrier`  | Shared `addrspace(3)` |
| `DynamicSharedArray::get()`| Shared `addrspace(3)` |

The example launches the kernel twice. `DEVICE_COUNTER` should persist across
launches, proving it is global device storage and not per-block shared memory.

Current limitation: ordinary device statics must start at zero. Non-zero initial
values are rejected until the LLVM dialect exporter supports global initializer
data.
