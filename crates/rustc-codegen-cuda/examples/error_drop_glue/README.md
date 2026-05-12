# error_drop_glue

Negative test: confirms that the mir-importer rejects MIR `Drop`
terminators with a clear "drop of `<Type>` is not supported on the
device" diagnostic, rather than silently lowering to a goto and
elision the destructor.

## What this tests

rustc emits `TerminatorKind::Drop` for places whose type has drop glue
(non-`Copy` types with an `impl Drop`, recursively through fields and
parameters). cuda-oxide does not yet emit device-side `drop_in_place`
calls, so any drop-glued type reaching `translate_drop` would have its
destructor silently skipped.

The previous lowering accepted every Drop terminator and emitted only a
goto; this example owns a `DropMarker` whose `Drop::drop` writes
`0xDEADBEEF` through a captured pointer. With the old behaviour the
write disappeared from PTX; with the new behaviour the build fails.

## Usage

```bash
cargo oxide run error_drop_glue
```

## Expected output

The build **must fail** with a diagnostic similar to:

```
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
       Translation failed: drop_glue_kernel: ... Compilation error:
       invalid input program.
       Unsupported construct: drop of `RigidTy(Adt(AdtDef(...
       error_drop_glue::DropMarker), GenericArgs([])))` is not supported
       on the device; cuda-oxide does not yet emit device-side
       `drop_in_place` calls. Restructure the kernel to use only `Copy`
       types, or wrap the value in `core::mem::ManuallyDrop` to suppress
       drop glue.
```

If the build succeeds, the silent-miscompile regression has returned —
`translate_drop` is once again accepting Drop terminators and emitting
only a goto.

## Categorisation

`scripts/smoketest.sh` classifies this example as the `error` category,
so its expected verdict is "compilation must fail with a recognised
diagnostic" — the same convention as the existing `error/` example.
