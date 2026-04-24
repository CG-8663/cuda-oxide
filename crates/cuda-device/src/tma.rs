/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor Memory Accelerator (TMA) for async bulk tensor copies.
//!
//! TMA is a hardware unit on Hopper+ (sm_90+) that performs asynchronous
//! bulk memory copies without using thread resources. Unlike manual memory
//! copies, TMA operates as a DMA engine that frees threads for computation.
//!
//! # Architecture
//!
//! ```text
//! Traditional Copy (threads do work):
//! ┌─────────────┐    threads     ┌──────────────┐
//! │   Global    │ ──────────────►│    Shared    │
//! │   Memory    │   (expensive)  │    Memory    │
//! └─────────────┘                └──────────────┘
//!
//! TMA Copy (hardware DMA):
//! ┌─────────────┐      TMA       ┌──────────────┐
//! │   Global    │ ══════════════►│    Shared    │
//! │   Memory    │   (async DMA)  │    Memory    │
//! └─────────────┘                └──────────────┘
//!      │                              │
//!      └── Threads free to compute! ──┘
//! ```
//!
//! # Key Concepts
//!
//! 1. **TmaDescriptor**: A 128-byte descriptor created on the host that describes
//!    the tensor layout in global memory. Passed to kernels as a parameter.
//!
//! 2. **Async Copy**: `cp.async.bulk.tensor.*` instructions copy tiles from
//!    global memory to shared memory without blocking threads.
//!
//! 3. **Barrier Integration**: TMA completion is tracked via `mbarrier` - the
//!    hardware automatically signals the barrier when transfer completes.
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, SharedArray};
//! use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
//! use cuda_device::barrier::{Barrier, mbarrier_init, mbarrier_arrive, mbarrier_wait};
//!
//! #[kernel]
//! pub fn tma_copy_kernel(
//!     desc: *const TmaDescriptor,  // Host-created descriptor
//!     // ...
//! ) {
//!     static mut TILE: SharedArray<f32, 4096> = SharedArray::UNINIT;
//!     static mut BAR: Barrier = Barrier::UNINIT;
//!
//!     // Initialize barrier (thread 0 only)
//!     if thread::threadIdx_x() == 0 {
//!         unsafe { mbarrier_init(&raw mut BAR, 1); }
//!     }
//!     thread::sync_threads();
//!
//!     // Thread 0 initiates TMA copy
//!     if thread::threadIdx_x() == 0 {
//!         unsafe {
//!             cp_async_bulk_tensor_2d_g2s(
//!                 &raw mut TILE as *mut u8,  // Shared memory destination
//!                 desc,                       // TMA descriptor
//!                 tile_x, tile_y,            // Tile coordinates
//!                 &raw mut BAR,              // Barrier for completion
//!             );
//!         }
//!     }
//!
//!     // All threads wait for TMA completion
//!     let token = unsafe { mbarrier_arrive(&raw const BAR) };
//!     unsafe { mbarrier_wait(&raw const BAR, token); }
//!
//!     // Now shared memory contains the tile data
//! }
//! ```
//!
//! # Host-Side Descriptor Creation
//!
//! TMA descriptors are created on the host using the CUDA driver API:
//!
//! ```rust,ignore
//! use cuda_core::sys::*;
//!
//! // Create descriptor for 2D tensor
//! let mut desc = std::mem::MaybeUninit::<CUtensorMap>::uninit();
//! unsafe {
//!     cuTensorMapEncodeTiled(
//!         desc.as_mut_ptr(),
//!         CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
//!         2,  // 2D tensor
//!         device_ptr as *mut _,
//!         dims.as_ptr(),
//!         strides.as_ptr(),
//!         box_dims.as_ptr(),
//!         element_strides.as_ptr(),
//!         CU_TENSOR_MAP_INTERLEAVE_NONE,
//!         CU_TENSOR_MAP_SWIZZLE_NONE,
//!         CU_TENSOR_MAP_L2_PROMOTION_NONE,
//!         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
//!     );
//! }
//! ```
//!
//! # Hardware Support
//!
//! - **sm_90+ (Hopper)**: Full TMA support with 1D-5D tensors
//! - **sm_100+ (Blackwell)**: Enhanced TMA with additional features
//! - **sm_120 (Blackwell)**: Latest TMA capabilities

use crate::barrier::Barrier;

// =============================================================================
// TMA Descriptor Type
// =============================================================================

/// Opaque TMA descriptor (created on host, passed to kernel).
///
/// This is a 128-byte structure that describes the tensor layout in global
/// memory. The descriptor is created on the host using `cuTensorMapEncodeTiled`
/// and passed to the kernel as a parameter.
///
/// # Size
///
/// - CUDA 12.0-12.x: 128 bytes, 64-byte aligned
/// - CUDA 13.0+: 128 bytes, 128-byte aligned
///
/// # Safety
///
/// - Must be created on host via CUDA driver API
/// - Must remain valid for the duration of the kernel execution
/// - Contents are opaque - do not modify
#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct TmaDescriptor {
    /// Opaque 128-byte descriptor data (16 x u64)
    _opaque: [u64; 16],
}

impl TmaDescriptor {
    /// Create an uninitialized descriptor.
    ///
    /// # Safety
    ///
    /// This creates invalid descriptor data. Only use for memory allocation;
    /// the descriptor must be properly initialized via `cuTensorMapEncodeTiled`
    /// on the host before use.
    pub const UNINIT: Self = Self { _opaque: [0; 16] };
}

// =============================================================================
// TMA Copy Operations - Global to Shared (G2S)
// =============================================================================

/// Async 1D tensor copy from global to shared memory via TMA.
///
/// Initiates an asynchronous copy of a 1D tile from global memory to shared
/// memory. The barrier is automatically signaled when the transfer completes.
///
/// # Parameters
///
/// - `dst`: Destination pointer in shared memory (address space 3)
/// - `tensor_map`: TMA descriptor created on host
/// - `coord0`: Coordinate along dimension 0
/// - `barrier`: Barrier to signal on completion
///
/// # Safety
///
/// - `dst` must be valid shared memory with sufficient size
/// - `tensor_map` must be a valid TMA descriptor
/// - `barrier` must be initialized
/// - Only ONE thread should call this per tile
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
///     [%dst], [%tensor_map, {%coord0}], [%barrier];
/// ```
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_1d_g2s(
    dst: *mut u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    barrier: *mut Barrier,
) {
    let _ = (dst, tensor_map, coord0, barrier);
    // Lowered to: @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d(...)
    unreachable!("cp_async_bulk_tensor_1d_g2s called outside CUDA kernel context")
}

/// Async 2D tensor copy from global to shared memory via TMA.
///
/// Initiates an asynchronous copy of a 2D tile from global memory to shared
/// memory. This is the most common TMA operation for matrix tiles.
///
/// # Parameters
///
/// - `dst`: Destination pointer in shared memory
/// - `tensor_map`: TMA descriptor created on host
/// - `coord0`: Coordinate along dimension 0 (typically X/column)
/// - `coord1`: Coordinate along dimension 1 (typically Y/row)
/// - `barrier`: Barrier to signal on completion
///
/// # Safety
///
/// - `dst` must be valid shared memory with sufficient size for the tile
/// - `tensor_map` must be a valid 2D TMA descriptor
/// - `barrier` must be initialized
/// - Only ONE thread should initiate the copy (typically thread 0)
///
/// # Example
///
/// ```rust,ignore
/// // Copy a 64x64 tile at position (tile_x * 64, tile_y * 64)
/// if thread::threadIdx_x() == 0 {
///     unsafe {
///         cp_async_bulk_tensor_2d_g2s(
///             &raw mut TILE as *mut u8,
///             tensor_map,
///             tile_x as i32 * 64,  // Column offset
///             tile_y as i32 * 64,  // Row offset
///             &raw mut BAR,
///         );
///     }
/// }
/// ```
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
///     [%dst], [%tensor_map, {%coord0, %coord1}], [%barrier];
/// ```
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_2d_g2s(
    dst: *mut u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    barrier: *mut Barrier,
) {
    let _ = (dst, tensor_map, coord0, coord1, barrier);
    // Lowered to: @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(...)
    unreachable!("cp_async_bulk_tensor_2d_g2s called outside CUDA kernel context")
}

/// Async 2D tensor copy from global to shared memory via TMA with **multicast**.
///
/// Same as [`cp_async_bulk_tensor_2d_g2s`] but delivers the tile to multiple
/// CTAs in a thread block cluster simultaneously. The `cta_mask` is a bitmask
/// where bit *i* means "deliver to CTA rank *i*". For example, `0b1111` sends
/// the tile to ranks 0-3 in a 4-CTA cluster.
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.tensor.2d.shared::cluster.global.tile.multicast::cluster
///     [%dst], [%tensor_map, {%coord0, %coord1}], [%barrier], %cta_mask;
/// ```
///
/// # Safety
///
/// - Must be called from within a cluster-launched kernel
/// - Only ONE thread per cluster should call this (typically CTA-0, thread 0)
/// - `cta_mask` must only set bits for valid CTA ranks in the cluster
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_2d_g2s_multicast(
    dst: *mut u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    barrier: *mut Barrier,
    cta_mask: u16,
) {
    let _ = (dst, tensor_map, coord0, coord1, barrier, cta_mask);
    unreachable!("cp_async_bulk_tensor_2d_g2s_multicast called outside CUDA kernel context")
}

/// Async 2D tensor copy from global to shared memory via TMA with
/// **multicast** and **cta_group::2** (TPC pair awareness).
///
/// Like [`cp_async_bulk_tensor_2d_g2s_multicast`] but emits the
/// `cta_group::2` qualifier. The TMA hardware automatically coordinates
/// barrier completion across both CTAs in the TPC pair, eliminating
/// the need for manual cross-CTA barrier relay.
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global
///     .mbarrier::complete_tx::bytes.multicast::cluster
///     [%dst], [%tensor_map, {%coord0, %coord1}], [%barrier], %cta_mask;
/// ```
///
/// # Safety
///
/// - Must be called from within a cluster-launched kernel with `cta_group::2`
/// - Only ONE thread per TPC pair should call this (typically warp 0, lane 0)
/// - `cta_mask` must only set bits for valid CTA ranks in the cluster
/// - All `tcgen05` operations must also use `cta_group::2` variants
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_2d_g2s_multicast_cg2(
    dst: *mut u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    barrier: *mut Barrier,
    cta_mask: u16,
) {
    let _ = (dst, tensor_map, coord0, coord1, barrier, cta_mask);
    unreachable!("cp_async_bulk_tensor_2d_g2s_multicast_cg2 called outside CUDA kernel context")
}

/// Async 3D tensor copy from global to shared memory via TMA.
///
/// # Parameters
///
/// - `dst`: Destination pointer in shared memory
/// - `tensor_map`: TMA descriptor created on host
/// - `coord0`, `coord1`, `coord2`: Coordinates along each dimension
/// - `barrier`: Barrier to signal on completion
///
/// # PTX
///
/// # Safety
///
/// - All pointers must be valid
/// - `tensor_map` must be a valid TMA descriptor
/// - Must be called from within a CUDA kernel context
///
/// ```ptx
/// cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
///     [%dst], [%tensor_map, {%c0, %c1, %c2}], [%barrier];
/// ```
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_3d_g2s(
    dst: *mut u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    coord2: i32,
    barrier: *mut Barrier,
) {
    let _ = (dst, tensor_map, coord0, coord1, coord2, barrier);
    unreachable!("cp_async_bulk_tensor_3d_g2s called outside CUDA kernel context")
}

/// Async 4D tensor copy from global to shared memory via TMA.
///
/// # Safety
///
/// - All pointers must be valid
/// - `tensor_map` must be a valid TMA descriptor
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_4d_g2s(
    dst: *mut u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    coord2: i32,
    coord3: i32,
    barrier: *mut Barrier,
) {
    let _ = (dst, tensor_map, coord0, coord1, coord2, coord3, barrier);
    unreachable!("cp_async_bulk_tensor_4d_g2s called outside CUDA kernel context")
}

/// Async 5D tensor copy from global to shared memory via TMA.
///
/// # Safety
///
/// - All pointers must be valid
/// - `tensor_map` must be a valid TMA descriptor
/// - Must be called from within a CUDA kernel context
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_5d_g2s(
    dst: *mut u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    coord2: i32,
    coord3: i32,
    coord4: i32,
    barrier: *mut Barrier,
) {
    let _ = (
        dst, tensor_map, coord0, coord1, coord2, coord3, coord4, barrier,
    );
    unreachable!("cp_async_bulk_tensor_5d_g2s called outside CUDA kernel context")
}

// =============================================================================
// TMA Copy Operations - Shared to Global (S2G)
// =============================================================================

/// Async 1D tensor copy from shared to global memory via TMA.
///
/// Initiates an asynchronous copy of a 1D tile from shared memory back to
/// global memory. Unlike G2S, this does NOT use a barrier - completion is
/// tracked via `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group`.
///
/// # Parameters
///
/// - `src`: Source pointer in shared memory
/// - `tensor_map`: TMA descriptor created on host
/// - `coord0`: Coordinate along dimension 0
///
/// # Safety
///
/// - `src` must be valid shared memory containing the data to copy
/// - `tensor_map` must be a valid TMA descriptor
/// - Must call `cp_async_bulk_commit_group()` after initiating copies
/// - Must call `cp_async_bulk_wait_group()` before reading from global memory
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.tensor.1d.global.shared::cta.tile
///     [%tensor_map, {%coord0}], [%src];
/// ```
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_1d_s2g(
    src: *const u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
) {
    let _ = (src, tensor_map, coord0);
    unreachable!("cp_async_bulk_tensor_1d_s2g called outside CUDA kernel context")
}

/// Async 2D tensor copy from shared to global memory via TMA.
///
/// # Safety
///
/// - `src` must be valid shared memory
/// - `tensor_map` must be a valid TMA descriptor
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_2d_s2g(
    src: *const u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
) {
    let _ = (src, tensor_map, coord0, coord1);
    unreachable!("cp_async_bulk_tensor_2d_s2g called outside CUDA kernel context")
}

/// Async 3D tensor copy from shared to global memory via TMA.
///
/// # Safety
///
/// - `src` must be valid shared memory
/// - `tensor_map` must be a valid TMA descriptor
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_3d_s2g(
    src: *const u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    coord2: i32,
) {
    let _ = (src, tensor_map, coord0, coord1, coord2);
    unreachable!("cp_async_bulk_tensor_3d_s2g called outside CUDA kernel context")
}

/// Async 4D tensor copy from shared to global memory via TMA.
///
/// # Safety
///
/// - `src` must be valid shared memory
/// - `tensor_map` must be a valid TMA descriptor
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_4d_s2g(
    src: *const u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    coord2: i32,
    coord3: i32,
) {
    let _ = (src, tensor_map, coord0, coord1, coord2, coord3);
    unreachable!("cp_async_bulk_tensor_4d_s2g called outside CUDA kernel context")
}

/// Async 5D tensor copy from shared to global memory via TMA.
///
/// # Safety
///
/// - `src` must be valid shared memory
/// - `tensor_map` must be a valid TMA descriptor
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn cp_async_bulk_tensor_5d_s2g(
    src: *const u8,
    tensor_map: *const TmaDescriptor,
    coord0: i32,
    coord1: i32,
    coord2: i32,
    coord3: i32,
    coord4: i32,
) {
    let _ = (src, tensor_map, coord0, coord1, coord2, coord3, coord4);
    unreachable!("cp_async_bulk_tensor_5d_s2g called outside CUDA kernel context")
}

// =============================================================================
// Async Copy Group Operations (for S2G completion tracking)
// =============================================================================

/// Commit pending async bulk operations to a group.
///
/// Creates a completion group for all preceding `cp.async.bulk` operations.
/// Must be called after initiating S2G copies, before waiting.
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.commit_group;
/// ```
#[inline(never)]
pub fn cp_async_bulk_commit_group() {
    // Lowered to: @llvm.nvvm.cp.async.bulk.commit.group()
    unreachable!("cp_async_bulk_commit_group called outside CUDA kernel context")
}

/// Wait for async bulk operation groups to complete.
///
/// Waits until at most `n` groups of async bulk operations are pending.
/// Use `n = 0` to wait for ALL pending operations.
///
/// # Parameters
///
/// - `n`: Maximum number of pending groups to allow (0 = wait for all)
///
/// # Example
///
/// ```rust,ignore
/// // Initiate S2G copy
/// unsafe { cp_async_bulk_tensor_2d_s2g(src, tensor_map, x, y); }
///
/// // Commit to a group
/// cp_async_bulk_commit_group();
///
/// // Wait for all groups to complete
/// cp_async_bulk_wait_group(0);
///
/// // Now safe to read from global memory
/// ```
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.wait_group N;
/// ```
#[inline(never)]
pub fn cp_async_bulk_wait_group(n: u32) {
    let _ = n;
    // Lowered to: @llvm.nvvm.cp.async.bulk.wait.group(i32 %n)
    unreachable!("cp_async_bulk_wait_group called outside CUDA kernel context")
}

/// Wait for async bulk operation groups with read completion.
///
/// Like `cp_async_bulk_wait_group` but ensures data is visible for reading.
///
/// # PTX
///
/// ```ptx
/// cp.async.bulk.wait_group.read N;
/// ```
#[inline(never)]
pub fn cp_async_bulk_wait_group_read(n: u32) {
    let _ = n;
    unreachable!("cp_async_bulk_wait_group_read called outside CUDA kernel context")
}
