# cuda-oxide Development Container

This image provides the toolchain expected by the cuda-oxide docs:

- Ubuntu 24.04
- CUDA Toolkit 13.0
- LLVM 21 with NVPTX support
- Clang 21 resource headers for `bindgen`
- Rust `nightly-2026-04-03` with `rust-src` and `rustc-dev`

Build it from the repository root:

```bash
docker build -f docker/Dockerfile \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -t cuda-oxide-dev .
```

The bind-mounted workflow writes both Cargo build artifacts under `target/`
and generated kernel artifacts such as `vecadd.ptx` back into the checkout, so
the image must be built with the caller's UID/GID when you plan to mount the
repository from the host. Reusing the default `1000:1000` image against a
checkout owned by another user will fail with permission errors. Changing only
`CARGO_TARGET_DIR` is not enough because PTX export still writes into the
example directory.

Run it with GPU access:

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace/cuda-oxide \
  -w /workspace/cuda-oxide \
  cuda-oxide-dev
```

Or run the verification command directly:

```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace/cuda-oxide \
  -w /workspace/cuda-oxide \
  cuda-oxide-dev \
  bash -lc 'cargo oxide doctor && cargo oxide run vecadd'
```

Inside the container:

```bash
cargo oxide doctor
cargo oxide run vecadd
```

GPU execution requires a compatible host NVIDIA driver and the NVIDIA Container
Toolkit. Without `--gpus all`, the image can still build host-side crates, but
CUDA driver checks and kernel execution will fail.
