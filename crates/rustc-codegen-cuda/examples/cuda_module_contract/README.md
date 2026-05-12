# cuda_module_contract

Focused regression coverage for the typed `#[cuda_module]` host ABI.

The kernel mixes common argument shapes that the generated launch method must
marshal correctly:

- scalar `f32` arguments
- `&[f32]` input through `&DeviceBuffer<f32>`
- raw device pointer argument
- `DisjointSlice<f32>` output through `&mut DeviceBuffer<f32>`

Regular Rust struct layout is tested separately by `abi_hmm`; cuda-oxide does
not require `#[repr(C)]` for Rust-only shared structs.

```bash
cargo oxide run cuda_module_contract
```
