/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed cooperative-groups handles.
//!
//! This module wraps the existing low-level intrinsics (`thread::*`,
//! `warp::*`, `cluster::*`, `grid::*`) in typed handles so the
//! participating-lane mask becomes part of the type rather than a silent
//! integer the caller has to remember.
//!
//! Each emitted PTX collective instruction (`vote.sync`, `shfl.sync`,
//! `match.sync`, ...) is byte-identical to the equivalent raw
//! `warp::*_sync` call — the participation mask folds at compile time.
//! Whether the wrapper *function* itself folds away is a separate
//! question, decided by rustc's MIR `Inline` cost threshold: for the
//! smallest wrappers it does and the call site is indistinguishable
//! from the raw form, but for larger wrappers the wrapper survives as
//! a `.visible .func` and the call site emits a `call.uni`. The
//! `hashmap_v2` example ships `find_kernel_warp` (raw) and
//! `find_kernel_warp_typed` (this API) side by side as an in-tree
//! reference for the current cost of that trade-off.
//!
//! # The universal trio
//!
//! Every group type implements [`ThreadGroup`] and exposes the same
//! three operations:
//!
//! - [`size`](ThreadGroup::size)         — how many threads are in this group
//! - [`thread_rank`](ThreadGroup::thread_rank) — my index within this group, `0..size`
//! - [`sync`](ThreadGroup::sync)         — barrier scoped to this group
//!
//! # The hierarchy
//!
//! ```text
//! Grid             this_grid()                grid::sync()        cooperative launch only
//! Cluster          this_cluster()             cluster::cluster_sync()  sm_90+
//! ThreadBlock      this_thread_block()        thread::sync_threads()
//!   WarpTile<32>     block.tiled_partition()  ballot/shfl/match    full warp
//!   WarpTile<N<32>>  block.tiled_partition()  warp::sync_mask     sub-warp tile
//!   CoalescedThreads coalesced_threads()      warp::sync_mask     runtime active mask
//! ```
//!
//! # Example: warp-wide ballot from `find_kernel_warp`
//!
//! ```rust,ignore
//! use cuda_device::cooperative_groups::{this_thread_block, ThreadGroup, WarpCollective};
//!
//! let block = this_thread_block();
//! let warp = block.tiled_partition::<32>();
//!
//! let m_h2 = warp.ballot(tag == h2);
//! if m_h2 != 0 {
//!     let leader = m_h2.trailing_zeros();
//!     let payload = warp.shfl(my_payload, leader);
//!     // ...
//! }
//! ```
//!
//! Switching to a 16-lane sub-warp tile is one line:
//!
//! ```rust,ignore
//! let tile = block.tiled_partition::<16>();
//! let m_h2 = tile.ballot(tag == h2);  // mask is 0xFFFF or 0xFFFF0000
//! ```

use crate::{cluster, grid, thread, warp};

// =============================================================================
// Traits
// =============================================================================

/// Universal "what every cooperative group exposes" trait.
///
/// Every group type — [`Grid`], [`Cluster`], [`ThreadBlock`],
/// [`WarpTile<N>`], [`CoalescedThreads`] — implements this. The three
/// methods are the entire reason cooperative groups exist as an
/// abstraction: at every level of the CUDA hierarchy you can ask "how
/// many of us are there", "which one am I", and "wait for everyone".
pub trait ThreadGroup: Copy {
    /// Number of threads in this group.
    fn size(&self) -> u32;

    /// This thread's index inside the group, `0..size()`.
    fn thread_rank(&self) -> u32;

    /// Barrier scoped to this group. Every thread in the group must reach
    /// this call before any of them proceeds.
    fn sync(&self);
}

/// Operations that only warp-scoped groups support.
///
/// Implemented by [`WarpTile<N>`] and [`CoalescedThreads`]. These are the
/// register-to-register collectives backed by PTX `shfl.sync` / `vote.sync`
/// / `match.sync` — they do not work at block, cluster, or grid scope.
///
/// All `*_rank` arguments and return values are **tile-relative**: lane 0
/// of a [`WarpTile<16>`] is the lane at warp position 0 *or* 16,
/// whichever bucket this thread is in. The implementation translates
/// to/from absolute warp lanes internally.
pub trait WarpCollective: ThreadGroup {
    /// Bitmask: bit `k` (within this group) set iff lane `k`'s `predicate` is true.
    ///
    /// PTX `vote.sync.ballot.b32` with the group's participation mask.
    fn ballot(&self, predicate: bool) -> u32;

    /// True iff every lane in this group has `predicate == true`.
    ///
    /// PTX `vote.sync.all.pred`.
    fn all(&self, predicate: bool) -> bool;

    /// True iff at least one lane in this group has `predicate == true`.
    ///
    /// PTX `vote.sync.any.pred`.
    fn any(&self, predicate: bool) -> bool;

    /// Read `var` from the lane at position `src_rank` in this group.
    ///
    /// PTX `shfl.sync.idx.b32`. `src_rank` is tile-relative (`0..size()`).
    fn shfl(&self, var: u32, src_rank: u32) -> u32;

    /// Butterfly exchange under this group's mask.
    ///
    /// PTX `shfl.sync.bfly.b32`. `lane_mask` is the XOR mask; for
    /// pairwise swaps of size `2^k` use `lane_mask = 1 << k`.
    fn shfl_xor(&self, var: u32, lane_mask: u32) -> u32;

    /// Read from `(my_rank + delta)` within this group.
    ///
    /// PTX `shfl.sync.down.b32`. Lanes near the high end of the group
    /// receive their own value (no wraparound).
    fn shfl_down(&self, var: u32, delta: u32) -> u32;

    /// Read from `(my_rank - delta)` within this group.
    ///
    /// PTX `shfl.sync.up.b32`. Lanes near the low end of the group
    /// receive their own value (no wraparound).
    fn shfl_up(&self, var: u32, delta: u32) -> u32;

    /// `f32` variant of [`shfl`](Self::shfl).
    fn shfl_f32(&self, var: f32, src_rank: u32) -> f32;

    /// `f32` variant of [`shfl_xor`](Self::shfl_xor).
    fn shfl_xor_f32(&self, var: f32, lane_mask: u32) -> f32;

    /// `f32` variant of [`shfl_down`](Self::shfl_down).
    fn shfl_down_f32(&self, var: f32, delta: u32) -> f32;

    /// `f32` variant of [`shfl_up`](Self::shfl_up).
    fn shfl_up_f32(&self, var: f32, delta: u32) -> f32;

    /// Bitmask of lanes in this group whose `value` equals mine.
    ///
    /// PTX `match.any.sync.b32` (sm_70+). Bits are absolute warp-lane
    /// positions (so the result is meaningful when AND-ed with the
    /// group's participation mask).
    fn match_any(&self, value: u32) -> u32;

    /// 64-bit value variant of [`match_any`](Self::match_any).
    ///
    /// PTX `match.any.sync.b64` (sm_70+).
    fn match_any_i64(&self, value: u64) -> u32;

    /// Group's participation mask if every lane in the group has the same
    /// `value`, else 0.
    ///
    /// PTX `match.all.sync.b32` (sm_70+). Recover the all-match predicate
    /// as `result != 0`.
    fn match_all(&self, value: u32) -> u32;

    /// 64-bit value variant of [`match_all`](Self::match_all).
    fn match_all_i64(&self, value: u64) -> u32;
}

// =============================================================================
// Helpers (not part of the public surface)
// =============================================================================

/// Linearize the (x, y, z) thread index inside the current block.
#[inline(always)]
fn thread_in_block_linear() -> u32 {
    let tx = thread::threadIdx_x();
    let ty = thread::threadIdx_y();
    let tz = thread::threadIdx_z();
    let dx = thread::blockDim_x();
    let dy = thread::blockDim_y();
    (tz * dy + ty) * dx + tx
}

/// Total threads per block.
#[inline(always)]
fn threads_per_block() -> u32 {
    thread::blockDim_x() * thread::blockDim_y() * thread::blockDim_z()
}

/// Total blocks per grid.
#[inline(always)]
fn blocks_per_grid() -> u32 {
    thread::gridDim_x() * thread::gridDim_y() * thread::gridDim_z()
}

/// Linear block index inside the grid.
#[inline(always)]
fn block_in_grid_linear() -> u32 {
    let bx = thread::blockIdx_x();
    let by = thread::blockIdx_y();
    let bz = thread::blockIdx_z();
    let gx = thread::gridDim_x();
    let gy = thread::gridDim_y();
    (bz * gy + by) * gx + bx
}

// =============================================================================
// Grid
// =============================================================================

/// Handle for the entire grid: every thread in every block of the launch.
///
/// Construct with [`this_grid`]. Cooperative launches required for
/// [`sync`](ThreadGroup::sync) — see [`grid::sync`] for the contract.
#[derive(Copy, Clone, Debug)]
pub struct Grid {
    _priv: (),
}

/// Get the grid handle for this thread.
#[inline(always)]
pub fn this_grid() -> Grid {
    Grid { _priv: () }
}

impl Grid {
    /// Number of blocks in this grid.
    #[inline(always)]
    pub fn num_blocks(&self) -> u32 {
        blocks_per_grid()
    }

    /// This block's linear rank within the grid, `0 .. num_blocks()`.
    #[inline(always)]
    pub fn block_rank(&self) -> u32 {
        block_in_grid_linear()
    }
}

impl ThreadGroup for Grid {
    #[inline(always)]
    fn size(&self) -> u32 {
        blocks_per_grid() * threads_per_block()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        block_in_grid_linear() * threads_per_block() + thread_in_block_linear()
    }

    #[inline(always)]
    fn sync(&self) {
        grid::sync();
    }
}

// =============================================================================
// Cluster
// =============================================================================

/// Handle for the current thread block cluster (sm_90+ only).
///
/// Construct with [`this_cluster`]. Outside a clustered launch every
/// query degenerates: `num_blocks() == 1`, `block_rank() == 0`, and
/// `sync()` is equivalent to a block barrier.
#[derive(Copy, Clone, Debug)]
pub struct Cluster {
    _priv: (),
}

/// Get the cluster handle for this thread.
#[inline(always)]
pub fn this_cluster() -> Cluster {
    Cluster { _priv: () }
}

impl Cluster {
    /// Number of blocks in this cluster.
    #[inline(always)]
    pub fn num_blocks(&self) -> u32 {
        cluster::cluster_size()
    }

    /// This block's linear rank within the cluster, `0 .. num_blocks()`.
    #[inline(always)]
    pub fn block_rank(&self) -> u32 {
        cluster::block_rank()
    }
}

impl ThreadGroup for Cluster {
    #[inline(always)]
    fn size(&self) -> u32 {
        cluster::cluster_size() * threads_per_block()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        cluster::block_rank() * threads_per_block() + thread_in_block_linear()
    }

    #[inline(always)]
    fn sync(&self) {
        cluster::cluster_sync();
    }
}

// =============================================================================
// Thread block
// =============================================================================

/// Handle for the current thread block (CTA).
///
/// Construct with [`this_thread_block`]. This is the natural starting
/// point for partitioning: call [`tiled_partition`](ThreadBlock::tiled_partition)
/// to get a [`WarpTile<N>`] for the warp or sub-warp this thread belongs to.
#[derive(Copy, Clone, Debug)]
pub struct ThreadBlock {
    _priv: (),
}

/// Get the thread-block handle for this thread.
#[inline(always)]
pub fn this_thread_block() -> ThreadBlock {
    ThreadBlock { _priv: () }
}

impl ThreadBlock {
    /// Partition this block into tiles of `N` lanes.
    ///
    /// `N` must be a power of two in `1..=32`. For `N == 32` the result
    /// is a full-warp tile; for smaller `N` it's a sub-warp tile whose
    /// participation mask is computed at runtime from this lane's
    /// `lane_id()`.
    ///
    /// Compile-time `N` validation: a `const { assert!(...) }` block
    /// rejects illegal sizes at compile time.
    #[inline(always)]
    pub fn tiled_partition<const N: u32>(self) -> WarpTile<N> {
        const {
            assert!(
                N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                "WarpTile size must be 1, 2, 4, 8, 16, or 32",
            );
        }
        WarpTile { _priv: () }
    }
}

impl ThreadGroup for ThreadBlock {
    #[inline(always)]
    fn size(&self) -> u32 {
        threads_per_block()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        thread_in_block_linear()
    }

    #[inline(always)]
    fn sync(&self) {
        thread::sync_threads();
    }
}

// =============================================================================
// WarpTile<N>
// =============================================================================

/// A tile of `N` lanes inside the current warp.
///
/// `N` is a compile-time constant, so the participation mask folds away
/// at codegen and the resulting PTX is byte-identical to a hand-written
/// `*_sync(mask, ...)` call.
///
/// Get one from [`ThreadBlock::tiled_partition`]:
///
/// ```rust,ignore
/// let warp = this_thread_block().tiled_partition::<32>();
/// let half = this_thread_block().tiled_partition::<16>();
/// ```
///
/// All [`WarpCollective`] methods take and return values relative to
/// **this tile**: `shfl(var, 0)` reads from the lowest lane of *this*
/// tile, not from absolute warp lane 0.
#[derive(Copy, Clone, Debug)]
pub struct WarpTile<const N: u32> {
    _priv: (),
}

impl<const N: u32> WarpTile<N> {
    /// Participation mask for this thread's tile.
    #[inline(always)]
    fn mask(&self) -> u32 {
        if N == 32 {
            u32::MAX
        } else {
            let lane = warp::lane_id();
            let tile_idx = lane / N;
            ((1u32 << N) - 1) << (tile_idx * N)
        }
    }

    /// Lane mask aligned to the start of this tile within the warp.
    #[inline(always)]
    fn tile_base_lane(&self) -> u32 {
        warp::lane_id() & !(N - 1)
    }
}

impl<const N: u32> ThreadGroup for WarpTile<N> {
    #[inline(always)]
    fn size(&self) -> u32 {
        N
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        warp::lane_id() & (N - 1)
    }

    #[inline(always)]
    fn sync(&self) {
        if N == 32 {
            warp::sync_mask(u32::MAX);
        } else {
            warp::sync_mask(self.mask());
        }
    }
}

impl<const N: u32> WarpCollective for WarpTile<N> {
    #[inline(always)]
    fn ballot(&self, predicate: bool) -> u32 {
        let raw = warp::ballot_sync(self.mask(), predicate);
        if N == 32 {
            raw
        } else {
            (raw >> self.tile_base_lane()) & ((1u32 << N) - 1)
        }
    }

    #[inline(always)]
    fn all(&self, predicate: bool) -> bool {
        warp::all_sync(self.mask(), predicate)
    }

    #[inline(always)]
    fn any(&self, predicate: bool) -> bool {
        warp::any_sync(self.mask(), predicate)
    }

    #[inline(always)]
    fn shfl(&self, var: u32, src_rank: u32) -> u32 {
        let abs_src = self.tile_base_lane() | (src_rank & (N - 1));
        warp::shuffle_sync(self.mask(), var, abs_src)
    }

    #[inline(always)]
    fn shfl_xor(&self, var: u32, lane_mask: u32) -> u32 {
        warp::shuffle_xor_sync(self.mask(), var, lane_mask)
    }

    #[inline(always)]
    fn shfl_down(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_down_sync(self.mask(), var, delta)
    }

    #[inline(always)]
    fn shfl_up(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_up_sync(self.mask(), var, delta)
    }

    #[inline(always)]
    fn shfl_f32(&self, var: f32, src_rank: u32) -> f32 {
        let abs_src = self.tile_base_lane() | (src_rank & (N - 1));
        warp::shuffle_f32_sync(self.mask(), var, abs_src)
    }

    #[inline(always)]
    fn shfl_xor_f32(&self, var: f32, lane_mask: u32) -> f32 {
        warp::shuffle_xor_f32_sync(self.mask(), var, lane_mask)
    }

    #[inline(always)]
    fn shfl_down_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_down_f32_sync(self.mask(), var, delta)
    }

    #[inline(always)]
    fn shfl_up_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_up_f32_sync(self.mask(), var, delta)
    }

    #[inline(always)]
    fn match_any(&self, value: u32) -> u32 {
        warp::match_any_sync(self.mask(), value) & self.mask()
    }

    #[inline(always)]
    fn match_any_i64(&self, value: u64) -> u32 {
        warp::match_any_i64_sync(self.mask(), value) & self.mask()
    }

    #[inline(always)]
    fn match_all(&self, value: u32) -> u32 {
        warp::match_all_sync(self.mask(), value)
    }

    #[inline(always)]
    fn match_all_i64(&self, value: u64) -> u32 {
        warp::match_all_i64_sync(self.mask(), value)
    }
}

// =============================================================================
// CoalescedThreads
// =============================================================================

/// The set of warp lanes that are currently converged with this thread.
///
/// Construct with [`coalesced_threads`]. The participation mask is
/// captured at construction time from PTX `activemask.b32`, so the
/// group is stable across subsequent collectives even if control flow
/// later diverges further.
///
/// Use this when you're already inside a divergent branch and need a
/// type-safe handle for the lanes that took it. For straight-line warp
/// code use [`WarpTile<32>`] instead.
#[derive(Copy, Clone, Debug)]
pub struct CoalescedThreads {
    mask: u32,
}

/// Capture the currently-converged lanes as a [`CoalescedThreads`] group.
#[inline(always)]
pub fn coalesced_threads() -> CoalescedThreads {
    CoalescedThreads {
        mask: warp::active_mask(),
    }
}

impl CoalescedThreads {
    /// The captured participation mask.
    #[inline(always)]
    pub fn raw_mask(&self) -> u32 {
        self.mask
    }
}

impl ThreadGroup for CoalescedThreads {
    #[inline(always)]
    fn size(&self) -> u32 {
        self.mask.count_ones()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        let lane = warp::lane_id();
        let lower = self.mask & ((1u32 << lane) - 1);
        lower.count_ones()
    }

    #[inline(always)]
    fn sync(&self) {
        warp::sync_mask(self.mask);
    }
}

impl WarpCollective for CoalescedThreads {
    #[inline(always)]
    fn ballot(&self, predicate: bool) -> u32 {
        warp::ballot_sync(self.mask, predicate) & self.mask
    }

    #[inline(always)]
    fn all(&self, predicate: bool) -> bool {
        warp::all_sync(self.mask, predicate)
    }

    #[inline(always)]
    fn any(&self, predicate: bool) -> bool {
        warp::any_sync(self.mask, predicate)
    }

    #[inline(always)]
    fn shfl(&self, var: u32, src_rank: u32) -> u32 {
        // Map this group's tile-relative `src_rank` to an absolute warp
        // lane: pick the lane whose own thread_rank within this group
        // equals src_rank. Implementation detail: scan by repeatedly
        // popping the lowest set bit `src_rank` times. Compiles to a
        // small straight-line sequence of `popc`/`bfind`-like ops via
        // LLVM's NVPTX folding.
        let mut m = self.mask;
        let mut k = src_rank;
        while k > 0 {
            m &= m - 1;
            k -= 1;
        }
        let abs_lane = m.trailing_zeros();
        warp::shuffle_sync(self.mask, var, abs_lane)
    }

    #[inline(always)]
    fn shfl_xor(&self, var: u32, lane_mask: u32) -> u32 {
        warp::shuffle_xor_sync(self.mask, var, lane_mask)
    }

    #[inline(always)]
    fn shfl_down(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_down_sync(self.mask, var, delta)
    }

    #[inline(always)]
    fn shfl_up(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_up_sync(self.mask, var, delta)
    }

    #[inline(always)]
    fn shfl_f32(&self, var: f32, src_rank: u32) -> f32 {
        let mut m = self.mask;
        let mut k = src_rank;
        while k > 0 {
            m &= m - 1;
            k -= 1;
        }
        let abs_lane = m.trailing_zeros();
        warp::shuffle_f32_sync(self.mask, var, abs_lane)
    }

    #[inline(always)]
    fn shfl_xor_f32(&self, var: f32, lane_mask: u32) -> f32 {
        warp::shuffle_xor_f32_sync(self.mask, var, lane_mask)
    }

    #[inline(always)]
    fn shfl_down_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_down_f32_sync(self.mask, var, delta)
    }

    #[inline(always)]
    fn shfl_up_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_up_f32_sync(self.mask, var, delta)
    }

    #[inline(always)]
    fn match_any(&self, value: u32) -> u32 {
        warp::match_any_sync(self.mask, value) & self.mask
    }

    #[inline(always)]
    fn match_any_i64(&self, value: u64) -> u32 {
        warp::match_any_i64_sync(self.mask, value) & self.mask
    }

    #[inline(always)]
    fn match_all(&self, value: u32) -> u32 {
        warp::match_all_sync(self.mask, value)
    }

    #[inline(always)]
    fn match_all_i64(&self, value: u64) -> u32 {
        warp::match_all_i64_sync(self.mask, value)
    }
}
