/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(non_snake_case)]
//! CUDA thread intrinsics and thread-safe index types.
//!
//! This module provides:
//! - `ThreadIndex`: A newtype derived from hardware built-in variables, ensuring uniqueness
//! - Thread intrinsics: `threadIdx_x`, `blockIdx_x`, etc.
//! - Index helpers: `index_1d`, `index_2d` that return `ThreadIndex`
//!
//! # Safety Model
//!
//! The safety of parallel writes to `DisjointSlice` relies on each thread accessing
//! a unique memory location. This is guaranteed by:
//!
//! 1. **ThreadIndex** can only be constructed by trusted functions (`index_1d`, `index_2d`)
//! 2. These functions derive the index from hardware built-in variables (`threadIdx`,
//!    `blockIdx`, `blockDim`) -- read-only special registers assigned by the runtime
//!    at kernel launch. The formula `outer * stride + inner` combines these into a
//!    unique scalar index per thread.
//! 3. `index_1d`: `inner < stride` is hardware-guaranteed (`threadIdx.x < blockDim.x`)
//! 4. `index_2d`: `inner < stride` is enforced by returning `Option<ThreadIndex>` —
//!    threads where `col >= row_stride` get `None` and cannot access the slice
//!
//! See: "A Note on the Algebra of CuTe Layouts" for formal treatment of layout composition.

// =============================================================================
// ThreadIndex - Type-Safe Thread-Unique Index
// =============================================================================

/// A thread-unique index derived from hardware built-in variables (special registers).
///
/// This type guarantees that the contained index is unique per thread, making it safe
/// to use for parallel writes to `DisjointSlice`.
///
/// # Construction
///
/// `ThreadIndex` cannot be constructed directly. Use one of the trusted functions:
/// - [`index_1d()`] - For 1D grids
/// - [`index_2d()`] - For 2D grids (e.g., GEMM)
///
/// # Example
///
/// ```rust
/// use cuda_device::thread::{index_1d, ThreadIndex};
/// use cuda_device::DisjointSlice;
///
/// #[kernel]
/// fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///     let idx = index_1d();
///     if let Some(c_elem) = c.get_mut(idx) {
///         *c_elem = a[idx.get()] + b[idx.get()];
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ThreadIndex(usize);

impl ThreadIndex {
    /// Get the raw index value.
    ///
    /// Use this when you need the index for array indexing on regular slices.
    #[inline(always)]
    pub fn get(self) -> usize {
        self.0
    }

    /// Check if this index is less than a bound.
    ///
    /// Convenience method for bounds checking.
    #[inline(always)]
    pub fn in_bounds(self, len: usize) -> bool {
        self.0 < len
    }
}

// =============================================================================
// 1D Index Helper
// =============================================================================

/// Get the global 1D thread index.
///
/// Computes: `blockIdx.x * blockDim.x + threadIdx.x`
///
/// Designed for **1D grid launches** (grids where only the X dimension is used).
/// For 2D grids, use [`index_2d`] instead.
///
/// # Uniqueness Guarantee
///
/// This produces a unique index per thread because:
/// - `threadIdx.x ∈ [0, blockDim.x)` (hardware-guaranteed)
/// - These are hardware built-in variables (read-only special registers); the formula
///   `bid * bdim + tid` combines them into non-overlapping ranges per block
///
/// # Example
///
/// ```rust
/// let idx = index_1d();
/// if let Some(c_elem) = c.get_mut(idx) {
///     *c_elem = a[idx.get()] + b[idx.get()];
/// }
/// ```
#[inline(always)]
pub fn index_1d() -> ThreadIndex {
    let tid = threadIdx_x();
    let bid = blockIdx_x();
    let bdim = blockDim_x();
    // SAFETY: bid, bdim, tid are hardware built-in variables (read-only special registers).
    // The formula `bid * bdim + tid` produces unique indices because tid < bdim is
    // guaranteed by the hardware.
    ThreadIndex((bid * bdim + tid) as usize)
}

// =============================================================================
// 2D Index Helper
// =============================================================================

/// Get the global 2D thread index, linearized to 1D.
///
/// Returns `Some(ThreadIndex)` when `col < row_stride`, `None` otherwise.
///
/// Computes: `row * row_stride + col`
///
/// Where:
/// - `row = blockIdx.y * blockDim.y + threadIdx.y`
/// - `col = blockIdx.x * blockDim.x + threadIdx.x`
///
/// # Uniqueness Guarantee
///
/// The index is derived from hardware built-in variables (`threadIdx`, `blockIdx`,
/// `blockDim`) -- read-only special registers assigned by the runtime at launch.
/// The formula `row * stride + col` is injective (unique per thread) when
/// `col < stride`. This function enforces that invariant by returning `None` for
/// threads where `col >= row_stride`, so every `ThreadIndex` it produces is
/// guaranteed unique -- no caller obligation required.
///
/// **Proof sketch:** Two threads with distinct `(row_a, col_a)` and `(row_b, col_b)`
/// where both `col_a < stride` and `col_b < stride`:
///
/// ```text
///   row_a * stride + col_a == row_b * stride + col_b
///   => (row_a - row_b) * stride == col_b - col_a
/// ```
///
/// Since `col_a, col_b ∈ [0, stride)`, the RHS is in `(-stride, stride)`.
/// The LHS is a multiple of `stride`, so the only solution is `row_a == row_b`
/// AND `col_a == col_b`. But distinct hardware threads have distinct `(row, col)`.
///
/// # Parameters
///
/// - `row_stride`: The stride for row-major layout (typically the number of columns N)
///
/// # Example
///
/// ```rust
/// // GEMM: C[row, col] = ...
/// let row = index_2d_row();
/// let col = index_2d_col();
/// if let Some(c_idx) = index_2d(n as usize) {
///     // col < n is guaranteed by Some
///     if row < m {
///         if let Some(c_elem) = c.get_mut(c_idx) {
///             *c_elem = ...;
///         }
///     }
/// }
/// ```
#[inline(always)]
pub fn index_2d(row_stride: usize) -> Option<ThreadIndex> {
    let row = (blockIdx_y() * blockDim_y() + threadIdx_y()) as usize;
    let col = (blockIdx_x() * blockDim_x() + threadIdx_x()) as usize;
    if col < row_stride {
        Some(ThreadIndex(row * row_stride + col))
    } else {
        None
    }
}

/// Get the row component of a 2D thread index.
///
/// Computes: `blockIdx.y * blockDim.y + threadIdx.y`
#[inline(always)]
pub fn index_2d_row() -> usize {
    (blockIdx_y() * blockDim_y() + threadIdx_y()) as usize
}

/// Get the column component of a 2D thread index.
///
/// Computes: `blockIdx.x * blockDim.x + threadIdx.x`
#[inline(always)]
pub fn index_2d_col() -> usize {
    (blockIdx_x() * blockDim_x() + threadIdx_x()) as usize
}

// =============================================================================
// X-Dimension Intrinsics
// =============================================================================

/// Get threadIdx.x (thread index within block, X dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn threadIdx_x() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    unreachable!("threadIdx_x called outside CUDA kernel context")
}

/// Get blockIdx.x (block index within grid, X dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockIdx_x() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
    unreachable!("blockIdx_x called outside CUDA kernel context")
}

/// Get blockDim.x (block dimension, X dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockDim_x() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    unreachable!("blockDim_x called outside CUDA kernel context")
}

// =============================================================================
// Y-Dimension Intrinsics
// =============================================================================

/// Get threadIdx.y (thread index within block, Y dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn threadIdx_y() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
    unreachable!("threadIdx_y called outside CUDA kernel context")
}

/// Get blockIdx.y (block index within grid, Y dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockIdx_y() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
    unreachable!("blockIdx_y called outside CUDA kernel context")
}

/// Get blockDim.y (block dimension, Y dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockDim_y() -> u32 {
    unreachable!("blockDim_y called outside CUDA kernel context")
}

// =============================================================================
// Z-Dimension Intrinsics
// =============================================================================

/// Get threadIdx.z (thread index within block, Z dimension).
#[inline(never)]
pub fn threadIdx_z() -> u32 {
    unreachable!("threadIdx_z called outside CUDA kernel context")
}

/// Get blockIdx.z (block index within grid, Z dimension).
#[inline(never)]
pub fn blockIdx_z() -> u32 {
    unreachable!("blockIdx_z called outside CUDA kernel context")
}

/// Get blockDim.z (block dimension, Z dimension).
#[inline(never)]
pub fn blockDim_z() -> u32 {
    unreachable!("blockDim_z called outside CUDA kernel context")
}

// =============================================================================
// Grid Dimensions (gridDim)
// =============================================================================

/// Get gridDim.x — number of blocks along the X axis of the grid.
#[inline(never)]
pub fn gridDim_x() -> u32 {
    unreachable!("gridDim_x called outside CUDA kernel context")
}

/// Get gridDim.y — number of blocks along the Y axis of the grid.
#[inline(never)]
pub fn gridDim_y() -> u32 {
    unreachable!("gridDim_y called outside CUDA kernel context")
}

/// Get gridDim.z — number of blocks along the Z axis of the grid.
#[inline(never)]
pub fn gridDim_z() -> u32 {
    unreachable!("gridDim_z called outside CUDA kernel context")
}

// =============================================================================
// Synchronization Intrinsics
// =============================================================================

/// Block-level thread synchronization barrier.
///
/// All threads in a block must reach this barrier before any thread can proceed.
/// This is equivalent to `__syncthreads()` in CUDA C/C++.
///
/// # Usage
///
/// ```rust
/// use cuda_device::thread;
///
/// // Write to shared memory
/// shared_tile[tid] = value;
///
/// // Ensure all threads have written before any thread reads
/// thread::sync_threads();
///
/// // Now safe to read values written by other threads
/// let neighbor = shared_tile[other_tid];
/// ```
///
/// # Safety
///
/// - All threads in the block must reach the same barrier (no divergent barriers)
/// - Placing `sync_threads()` inside a conditional where not all threads enter
///   will cause deadlock
#[inline(never)]
pub fn sync_threads() {
    // Lowered to: call void @llvm.nvvm.barrier0()
    unreachable!("sync_threads called outside CUDA kernel context")
}

// =============================================================================
// Compile-Time Launch Bounds Configuration
// =============================================================================

/// Marker function for compile-time launch bounds configuration.
///
/// This is a compile-time configuration marker that tells the compiler to emit
/// `.maxntid` and `.minnctapersm` PTX directives for this kernel. It does NOT
/// generate any runtime code.
///
/// # Usage
///
/// This function should NOT be called directly. Use the `#[launch_bounds(max, min)]`
/// attribute macro instead, which injects this marker:
///
/// ```rust,ignore
/// #[kernel]
/// #[launch_bounds(256)]           // max 256 threads per block
/// pub fn my_kernel(output: DisjointSlice<f32>) { ... }
///
/// #[kernel]
/// #[launch_bounds(256, 2)]        // max 256 threads, min 2 blocks per SM
/// pub fn optimized_kernel(output: DisjointSlice<f32>) { ... }
/// ```
///
/// # How It Works
///
/// 1. The `#[launch_bounds]` macro injects `__launch_bounds_config::<MAX, MIN>()` at kernel start
/// 2. MIR importer detects this call and extracts the const generic parameters
/// 3. The marker call is NOT compiled - it's removed during compilation
/// 4. LLVM export emits `!nvvm.annotations` with `maxntid` and `minctasm` metadata
/// 5. LLVM NVPTX backend emits `.maxntid` and `.minnctapersm` in PTX
///
/// # PTX Output
///
/// ```ptx
/// .entry my_kernel .maxntid 256 .minnctapersm 2 { ... }
/// ```
///
/// # Parameters
///
/// - `MAX_THREADS` - Maximum threads per block (required). Maps to `.maxntid`.
/// - `MIN_BLOCKS` - Minimum blocks per SM for occupancy (optional, default 0 = unspecified).
///   Maps to `.minnctapersm`.
///
/// # Performance Impact
///
/// Launch bounds help the compiler:
/// - Allocate registers more efficiently
/// - Optimize occupancy (threads per SM)
/// - Make better scheduling decisions
///
/// Using appropriate launch bounds can significantly improve performance for
/// register-heavy kernels or kernels with specific occupancy requirements.
#[inline(never)]
pub fn __launch_bounds_config<const MAX_THREADS: u32, const MIN_BLOCKS: u32>() {
    // This function is detected at compile time and removed.
    // The const generics are extracted to set launch bounds.
    // No runtime code is generated.
}
