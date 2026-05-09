# Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_TOOLCHAIN=nightly-2026-04-03
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDA_OXIDE_LLC=/usr/bin/llc-21 \
    LIBCLANG_PATH=/usr/lib/llvm-21/lib \
    LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 \
    PATH=/usr/lib/llvm-21/bin:/usr/local/cuda/bin:/home/${USERNAME}/.cargo/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        gnupg \
        libc6-dev \
        make \
        pkg-config \
        python3 \
        python3-pip \
        xz-utils \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" \
        > /etc/apt/sources.list.d/llvm-toolchain-noble-21.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-21 \
        libclang-common-21-dev \
        lld-21 \
        llvm-21 \
        llvm-21-dev \
    && if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
        if ! getent group "${USER_GID}" >/dev/null; then \
            groupadd --gid "${USER_GID}" "${USERNAME}"; \
        fi; \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"; \
    fi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
WORKDIR /workspace/cuda-oxide

RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && rustup component add rust-src rustc-dev rust-analyzer rustfmt clippy --toolchain "${RUST_TOOLCHAIN}"

CMD ["/bin/bash"]
