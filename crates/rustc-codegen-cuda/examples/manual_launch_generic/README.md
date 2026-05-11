# manual_launch_generic

Regression coverage for the lower-level launch API.

This example intentionally uses `load_kernel_module` and `cuda_launch!` instead
of `#[cuda_module]`. It launches the same generic `affine<T>` kernel as both
`affine::<f32>` and `affine::<i32>`, then verifies both outputs.

Use this when checking that the explicit sidecar-artifact path still works for
generic kernels.

```bash
cargo oxide run manual_launch_generic
```
