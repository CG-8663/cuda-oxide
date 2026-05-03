/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side utilities for CUDA kernel development.
//!
//! This crate provides CPU-side utilities for preparing data and setting up
//! GPU kernel execution. Unlike `cuda-device` which contains device-side primitives,
//! this crate runs entirely on the host.
//!
//! ## Modules
//!
//! - [`tiling`]: Layout transformations for tensor core operations (tcgen05)
//! - [`launch`]: Kernel launch traits (`CudaKernel`, `GenericCudaKernel`)
//!
//! ## Macros
//!
//! - [`cuda_launch!`]: Launch a kernel synchronously on a given stream
//!   (via `cuda_core::launch_kernel`)
//! - [`cuda_launch_async!`]: Build a lazy `DeviceOperation` for async kernel launch
//!   (via `cuda_async`)
//!
//! ## Usage
//!
//! ```ignore
//! use cuda_device::{kernel, thread, DisjointSlice};
//! use cuda_host::cuda_launch;
//! use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
//!
//! #[kernel]
//! pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { ... }
//!
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.default_stream();
//! let module = ctx.load_module_from_file("vecadd.ptx")?;
//!
//! let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
//! let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
//! let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
//!
//! cuda_launch! {
//!     kernel: vecadd,
//!     stream: stream,
//!     module: module,
//!     config: LaunchConfig::for_num_elems(N as u32),
//!     args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
//! }?;
//!
//! let c_host = c_dev.to_host_vec(&stream)?;
//! ```

pub mod launch;
pub mod ltoir;
pub mod tiling;

pub use launch::{CudaKernel, GenericCudaKernel, HasLength, ReadOnly, Scalar, WriteOnly};

/// Loads a compiled kernel module by name. Tries `<name>.cubin`, then
/// `<name>.ptx`, and finally falls through to the LTOIR build path
/// (`<name>.ll` plus libdevice → cubin) when cuda-oxide auto-detected
/// CUDA libdevice math intrinsics during the build. Most beginner code
/// never sees the LTOIR path because `vecadd`-style kernels emit `.ptx`
/// directly. See [`ltoir`] for the underlying pipeline and discovery rules.
pub use ltoir::{LtoirError, load_kernel_module};

// Re-export launch macros from cuda-macros for convenience.
pub use cuda_macros::cuda_launch;

/// Re-export of [`cuda_macros::cuda_launch_async`].
///
/// Returns a lazy [`cuda_async::launch::AsyncKernelLaunch`] that implements
/// [`DeviceOperation`]. Stream assignment is deferred to the scheduling policy --
/// call `.sync()` to block or `.await` to suspend.
pub use cuda_macros::cuda_launch_async;
pub use tiling::{
    TILE_SIZE, k_major_index, mn_major_index, print_layout_indices, to_k_major_f16, to_mn_major_f16,
};
