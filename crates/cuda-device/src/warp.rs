/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level primitives.
//!
//! These operations enable fast data exchange within a warp (32 threads)
//! without explicit synchronization. Unlike shared memory operations, warp
//! shuffles use registers and require no barriers.
//!
//! # Performance
//!
//! | Operation | Shared Memory | Warp Shuffle |
//! |-----------|---------------|--------------|
//! | Latency | ~20 cycles | ~2 cycles |
//! | Synchronization | Requires `sync_threads()` | Implicit within warp |
//! | Scope | Block (up to 1024 threads) | Warp (32 threads) |
//!
//! # Example: Warp Reduction
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, warp};
//!
//! #[kernel]
//! pub fn warp_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
//!     let gid = thread::index_1d();
//!     let lane = warp::lane_id();
//!
//!     let mut val = data[gid.get()];
//!
//!     // Butterfly reduction using shuffle_xor
//!     val = val + warp::shuffle_xor_f32(val, 16);
//!     val = val + warp::shuffle_xor_f32(val, 8);
//!     val = val + warp::shuffle_xor_f32(val, 4);
//!     val = val + warp::shuffle_xor_f32(val, 2);
//!     val = val + warp::shuffle_xor_f32(val, 1);
//!
//!     // Lane 0 has the sum
//!     if lane == 0 {
//!         let warp_idx = gid.get() / 32;
//!         *out.get_unchecked_mut(warp_idx) = val;
//!     }
//! }
//! ```

// =============================================================================
// Lane Identification
// =============================================================================

/// Get the lane ID within the current warp (0-31).
///
/// Each thread in a warp has a unique lane ID. This is useful for:
/// - Determining which thread should perform special actions (e.g., lane 0 writes output)
/// - Computing shuffle source lanes
/// - Implementing lane-specific logic
///
/// # Example
///
/// ```rust,ignore
/// let lane = warp::lane_id();
/// if lane == 0 {
///     // Only lane 0 writes the result
///     *output = result;
/// }
/// ```
#[inline(never)]
pub fn lane_id() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
    unreachable!("lane_id called outside CUDA kernel context")
}

/// Get the warp ID within the current block.
///
/// Computes: `threadIdx.x / 32`
///
/// This is a derived value, not a hardware register.
/// Only valid for 1D thread blocks; for multi-dimensional blocks,
/// compute your own warp ID from the linearized thread index.
#[inline(always)]
pub fn warp_id() -> u32 {
    crate::thread::threadIdx_x() / 32
}

// =============================================================================
// Warp Shuffle - Integer (u32)
// =============================================================================

/// Shuffle: get value from any lane in the warp.
///
/// Returns the value of `var` from the thread at `src_lane`.
/// All threads in the warp execute this simultaneously.
///
/// # Parameters
///
/// - `var`: The value to share (each thread provides its own)
/// - `src_lane`: The lane ID (0-31) to read from
///
/// # Example
///
/// ```rust,ignore
/// // Broadcast lane 0's value to all lanes
/// let broadcasted = warp::shuffle(my_value, 0);
/// ```
#[inline(never)]
pub fn shuffle(var: u32, src_lane: u32) -> u32 {
    let _ = (var, src_lane);
    // Lowered to: call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %var, i32 %src_lane, i32 31)
    unreachable!("shuffle called outside CUDA kernel context")
}

/// Shuffle XOR: exchange values with lane at `(lane_id ^ lane_mask)`.
///
/// This is commonly used for butterfly reductions. Each lane exchanges
/// with a partner determined by XOR-ing its lane ID with the mask.
///
/// # Parameters
///
/// - `var`: The value to exchange
/// - `lane_mask`: XOR mask (typically powers of 2: 1, 2, 4, 8, 16)
///
/// # Example: Butterfly Reduction
///
/// ```rust,ignore
/// // Sum all 32 values in warp
/// let mut sum = my_value;
/// sum = sum + warp::shuffle_xor(sum, 16);  // Exchange with lane +/- 16
/// sum = sum + warp::shuffle_xor(sum, 8);   // Exchange with lane +/- 8
/// sum = sum + warp::shuffle_xor(sum, 4);   // etc.
/// sum = sum + warp::shuffle_xor(sum, 2);
/// sum = sum + warp::shuffle_xor(sum, 1);
/// // Now all lanes have the sum
/// ```
#[inline(never)]
pub fn shuffle_xor(var: u32, lane_mask: u32) -> u32 {
    let _ = (var, lane_mask);
    // Lowered to: call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %var, i32 %lane_mask, i32 31)
    unreachable!("shuffle_xor called outside CUDA kernel context")
}

/// Shuffle down: get value from lane at `(lane_id + delta)`.
///
/// Commonly used for sequential reductions where each lane reads from
/// a higher-numbered lane.
///
/// # Parameters
///
/// - `var`: The value to share
/// - `delta`: Offset to add to lane ID (positive direction)
///
/// # Example
///
/// ```rust,ignore
/// // Sequential reduction
/// let mut sum = my_value;
/// sum = sum + warp::shuffle_down(sum, 16);
/// sum = sum + warp::shuffle_down(sum, 8);
/// sum = sum + warp::shuffle_down(sum, 4);
/// sum = sum + warp::shuffle_down(sum, 2);
/// sum = sum + warp::shuffle_down(sum, 1);
/// // Lane 0 has the sum
/// ```
#[inline(never)]
pub fn shuffle_down(var: u32, delta: u32) -> u32 {
    let _ = (var, delta);
    // Lowered to: call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %var, i32 %delta, i32 31)
    unreachable!("shuffle_down called outside CUDA kernel context")
}

/// Shuffle up: get value from lane at `(lane_id - delta)`.
///
/// Commonly used for prefix sums (scan) where each lane reads from
/// a lower-numbered lane.
///
/// # Parameters
///
/// - `var`: The value to share
/// - `delta`: Offset to subtract from lane ID (negative direction)
///
/// # Example
///
/// ```rust,ignore
/// // Prefix sum (inclusive scan)
/// let mut sum = my_value;
/// let tmp = warp::shuffle_up(sum, 1);
/// if warp::lane_id() >= 1 { sum = sum + tmp; }
/// let tmp = warp::shuffle_up(sum, 2);
/// if warp::lane_id() >= 2 { sum = sum + tmp; }
/// // ... continue for 4, 8, 16
/// ```
#[inline(never)]
pub fn shuffle_up(var: u32, delta: u32) -> u32 {
    let _ = (var, delta);
    // Lowered to: call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %var, i32 %delta, i32 0)
    unreachable!("shuffle_up called outside CUDA kernel context")
}

// =============================================================================
// Warp Shuffle - Float (f32)
// =============================================================================

/// Shuffle f32: get value from any lane in the warp.
///
/// Float version of [`shuffle`]. Returns the value of `var` from the thread at `src_lane`.
#[inline(never)]
pub fn shuffle_f32(var: f32, src_lane: u32) -> f32 {
    let _ = (var, src_lane);
    // Lowered to: call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %var, i32 %src_lane, i32 31)
    unreachable!("shuffle_f32 called outside CUDA kernel context")
}

/// Shuffle XOR f32: exchange values with lane at `(lane_id ^ lane_mask)`.
///
/// Float version of [`shuffle_xor`]. Commonly used for floating-point reductions.
#[inline(never)]
pub fn shuffle_xor_f32(var: f32, lane_mask: u32) -> f32 {
    let _ = (var, lane_mask);
    // Lowered to: call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %var, i32 %lane_mask, i32 31)
    unreachable!("shuffle_xor_f32 called outside CUDA kernel context")
}

/// Shuffle down f32: get value from lane at `(lane_id + delta)`.
///
/// Float version of [`shuffle_down`].
#[inline(never)]
pub fn shuffle_down_f32(var: f32, delta: u32) -> f32 {
    let _ = (var, delta);
    // Lowered to: call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %var, i32 %delta, i32 31)
    unreachable!("shuffle_down_f32 called outside CUDA kernel context")
}

/// Shuffle up f32: get value from lane at `(lane_id - delta)`.
///
/// Float version of [`shuffle_up`].
#[inline(never)]
pub fn shuffle_up_f32(var: f32, delta: u32) -> f32 {
    let _ = (var, delta);
    // Lowered to: call float @llvm.nvvm.shfl.sync.up.f32(i32 -1, float %var, i32 %delta, i32 0)
    unreachable!("shuffle_up_f32 called outside CUDA kernel context")
}

// =============================================================================
// Warp Vote Operations
// =============================================================================

/// Warp vote: returns true if ALL active threads have predicate true.
///
/// This is a collective operation - all threads in the warp participate.
///
/// # Example
///
/// ```rust,ignore
/// let all_valid = warp::all(my_value > 0.0);
/// if !all_valid {
///     // At least one thread has invalid data
/// }
/// ```
#[inline(never)]
pub fn all(predicate: bool) -> bool {
    let _ = predicate;
    // Lowered to: call i1 @llvm.nvvm.vote.sync.all(i32 -1, i1 %predicate)
    unreachable!("all called outside CUDA kernel context")
}

/// Warp vote: returns true if ANY active thread has predicate true.
///
/// This is a collective operation - all threads in the warp participate.
///
/// # Example
///
/// ```rust,ignore
/// let any_overflow = warp::any(result > MAX_VALUE);
/// if any_overflow {
///     // At least one thread detected overflow
/// }
/// ```
#[inline(never)]
pub fn any(predicate: bool) -> bool {
    let _ = predicate;
    // Lowered to: call i1 @llvm.nvvm.vote.sync.any(i32 -1, i1 %predicate)
    unreachable!("any called outside CUDA kernel context")
}

/// Warp ballot: returns a 32-bit mask where bit i is set if thread i has predicate true.
///
/// This is useful for counting threads, finding active lanes, and
/// implementing warp-level control flow.
///
/// # Example
///
/// ```rust,ignore
/// let mask = warp::ballot(my_value > 0.0);
/// let count = mask.count_ones();  // How many threads have positive values
///
/// // Find the first lane with positive value
/// let first_positive_lane = mask.trailing_zeros();
/// ```
#[inline(never)]
pub fn ballot(predicate: bool) -> u32 {
    let _ = predicate;
    // Lowered to: call i32 @llvm.nvvm.vote.sync.ballot(i32 -1, i1 %predicate)
    unreachable!("ballot called outside CUDA kernel context")
}

/// Count threads with predicate true (population count of ballot).
///
/// Convenience function equivalent to `ballot(predicate).count_ones()`.
#[inline(always)]
pub fn popc(predicate: bool) -> u32 {
    ballot(predicate).count_ones()
}
