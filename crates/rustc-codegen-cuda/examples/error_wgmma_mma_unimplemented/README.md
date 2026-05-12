# error_wgmma_mma_unimplemented

Negative test: confirms that the codegen backend rejects calls to
`cuda_device::wgmma::wgmma_mma_*` with a clear "not yet implemented"
diagnostic, rather than silently emitting a comment placeholder and
producing PTX that multiplies-accumulates to zero.

## What this tests

Until full WGMMA MMA lowering lands (it requires register allocation for
16+ output registers), `convert_mma` must fail loud. This crate calls
`wgmma_mma_m64n64k16_f32_bf16` from a `#[kernel]`; the build is expected
to fail.

## Usage

```bash
cargo oxide run error_wgmma_mma_unimplemented
```

## Expected output

The build **must fail** with a diagnostic similar to:

```
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
       Lowering failed: Compilation error: invalid input program.
       wgmma.mma_async lowering is not yet implemented; calls to
       `cuda_device::wgmma::wgmma_mma_*` from a kernel are currently
       unsupported. Tracking issue: full lowering requires register
       allocation for 16+ output registers.
```

If the build succeeds, the silent-miscompile regression has returned —
`convert_mma` is once again emitting `// wgmma.mma placeholder` and the
multiply-accumulate is being erased.

## Categorisation

`scripts/smoketest.sh` classifies this example as the `error` category,
so its expected verdict is "compilation must fail with a recognised
diagnostic" — the same convention as the existing `error/` example.
