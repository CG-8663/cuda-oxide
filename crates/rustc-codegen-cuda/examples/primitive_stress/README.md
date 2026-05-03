# primitive_stress

Small stress test for primitive scalar support in cuda-oxide.

It covers cases that are easy for a MIR importer or lowering pass to miss:

- `char` constants and casts.
- `u128` / `i128` constants, arithmetic, shifts, and ABI passing.
- `usize` / `isize` arithmetic.
- Rust integer bit methods that call `core::intrinsics`, including `rotate_left`, `rotate_right`, `count_ones`, `leading_zeros`, `trailing_zeros`, `swap_bytes`, and `reverse_bits`.

Run it with:

```bash
cargo oxide run primitive_stress
```

The bit-method checks are especially useful because `u128::rotate_left` used to compile into an unresolved `core::intrinsics::rotate_left` symbol. The expected path now is:

```text
Rust method -> core::intrinsics::* -> cuda-oxide marker call -> LLVM intrinsic
```
