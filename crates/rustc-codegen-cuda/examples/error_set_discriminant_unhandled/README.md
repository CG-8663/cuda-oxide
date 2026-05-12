# error_set_discriminant_unhandled

Negative test: confirms that the mir-importer rejects MIR
`StatementKind::SetDiscriminant` with a clear "not yet supported"
diagnostic, rather than silently dropping an enum discriminant write from
the lowered PTX.

## What this tests

The example calls a `custom_mir` helper from a `#[kernel]` to emit
`StatementKind::SetDiscriminant` directly. Until the importer implements
that lowering, enum discriminant writes in device-reachable MIR must
produce a hard build error; the previous catch-all in
`crates/mir-importer/src/translator/statement.rs` silently returned
`Ok(prev_op)`, so the enum state update disappeared from the PTX.

## Usage

```bash
cargo oxide run error_set_discriminant_unhandled
```

## Expected output

The build **must fail** with a diagnostic similar to:

```
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
       Translation failed: set_discriminant_kernel: ... Compilation
       error: invalid input program.
       Unsupported construct: SetDiscriminant statements are not yet
       supported on the device; until they are lowered, enum discriminant
       writes would be silently dropped
```

If the build succeeds, the silent-miscompile regression has returned --
the importer is once again routing `SetDiscriminant` through the
catch-all `Ok(prev_op)` arm.

## Categorisation

`scripts/smoketest.sh` classifies this example as the `error` category,
so its expected verdict is "compilation must fail with a recognised
diagnostic" -- the same convention as the existing `error/` example.
