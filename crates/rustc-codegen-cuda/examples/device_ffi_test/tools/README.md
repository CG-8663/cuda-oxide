# Device FFI Tools

C tools for the LTOIR linking pipeline. These will be replaced with Rust bindings
in the future (NVIDIA internal - LTOIR format is not publicly documented).

---

## Tools

| Tool            | Purpose                        | Library       |
|-----------------|--------------------------------|---------------|
| `compile_ltoir` | LLVM IR (`.ll`) → LTOIR        | libNVVM       |
| `link_ltoir`    | Multiple LTOIR → cubin         | nvJitLink     |
| `launch_cubin`  | Load and run cubin (legacy)    | CUDA Driver   |

**Note**: `launch_cubin` is legacy. Use the Rust harness instead:

```bash
cargo run --release
```

---

## Build

```bash
./build_tools.sh
```

Requires CUDA Toolkit with libNVVM and nvJitLink.

---

## Usage

### compile_ltoir

Compiles LLVM IR to LTOIR using libNVVM with `-gen-lto`:

```bash
./compile_ltoir <input.ll> <arch> [output.ltoir]

# Examples:
./compile_ltoir ../device_ffi_test.ll sm_120 ../device_ffi_test.ltoir
./compile_ltoir kernel.ll sm_90 kernel.ltoir
```

### link_ltoir

Links multiple LTOIR files into a single cubin:

```bash
./link_ltoir -arch=<arch> -o <output.cubin> <input1.ltoir> [input2.ltoir ...]

# Example:
./link_ltoir -arch=sm_120 -o ../merged.cubin \
    ../device_ffi_test.ltoir \
    ../external_device_funcs.ltoir \
    ../cccl_wrappers.ltoir
```

### launch_cubin (legacy)

Load cubin and run tests (use Rust harness instead):

```bash
./launch_cubin <cubin_file>
./launch_cubin ../merged.cubin
```

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LTOIR LINKING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  cuda-oxide                           CUDA C++ (CCCL)                       │
│  ──────────                           ───────────────                       │
│  kernel.rs                            external_funcs.cu                     │
│      │                                      │                               │
│      ▼ cargo oxide run --emit-nvvm-ir           ▼ nvcc -dc -dlto            │
│  kernel.ll                            external_funcs.ltoir                  │
│      │                                      │                               │
│      ▼ compile_ltoir (libNVVM)              │                               │
│  kernel.ltoir                               │                               │
│      │                                      │                               │
│      └──────────────┬───────────────────────┘                               │
│                     ▼                                                       │
│              link_ltoir (nvJitLink)                                         │
│                     │                                                       │
│                     ▼                                                       │
│              merged.cubin                                                   │
│                     │                                                       │
│                     ▼                                                       │
│              Rust main.rs (cuda-core)                                       │
│                     │                                                       │
│                     ▼                                                       │
│                 Run on GPU                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Future: Rust Bindings

These C tools are temporary. The plan is to create:

- `libnvvm-sys` - Rust FFI bindings to libNVVM
- `nvjitlink-sys` - Rust FFI bindings to nvJitLink

This will enable seamless LTOIR linking from Rust without external tools.
