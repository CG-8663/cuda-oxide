# cuda-oxide Development Container

This image provides the toolchain expected by the cuda-oxide docs:

- Ubuntu 24.04
- CUDA Toolkit 13.0
- LLVM 21 with NVPTX support
- Clang 21 resource headers for `bindgen`
- Rust `nightly-2026-04-03` with `rust-src` and `rustc-dev`

Build it from the repository root:

```bash
docker build -f docker/Dockerfile -t cuda-oxide-dev .
```

Run it with GPU access:

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace/cuda-oxide \
  -w /workspace/cuda-oxide \
  cuda-oxide-dev
```

Inside the container:

```bash
cargo oxide doctor
cargo oxide run vecadd
```

GPU execution requires a compatible host NVIDIA driver and the NVIDIA Container
Toolkit. Without `--gpus all`, the image can still build host-side crates, but
CUDA driver checks and kernel execution will fail.
