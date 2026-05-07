/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cooperative-groups primitives smoke test.
//!
//! Each kernel exercises one new compiler intrinsic so the generated PTX
//! can be inspected in `coop_groups_demo.ptx`. The host harness verifies
//! end-to-end correctness on a real GPU.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, this_grid, this_thread_block};
use cuda_device::{DisjointSlice, grid, kernel, thread, warp};
use cuda_host::cuda_launch;

// =============================================================================
// KERNELS
// =============================================================================

/// Each thread writes the warp's `active_mask()` (full warp → 0xFFFFFFFF).
#[kernel]
pub fn test_active_mask(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let mask = warp::active_mask();
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// Each thread reports the lane mask of every other thread sharing its
/// `value`. With `value = lane / 4` the warp is partitioned into 8 buckets
/// of 4 contiguous lanes each, so every thread should see `0xF << (group*4)`.
#[kernel]
pub fn test_match_any(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let lane = warp::lane_id();
    let value: u32 = lane / 4;
    let mask = warp::match_any_sync(u32::MAX, value);
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// All lanes use the same value, so `match_all_sync` returns the full mask.
#[kernel]
pub fn test_match_all(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let mask = warp::match_all_sync(u32::MAX, 42u32);
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// Smoke test for `grid::sync()`. Each block's thread 0 writes a marker
/// (`blockIdx.x + 1`), the grid synchronises, then thread 0 reads every
/// other block's marker via the raw base pointer and writes the sum into
/// `out[blockIdx.x]`. Expected value: `gridDim.x * (gridDim.x + 1) / 2`.
#[kernel]
pub fn test_grid_sync(mut markers: DisjointSlice<u32>, mut out: DisjointSlice<u32>) {
    let block_id = thread::blockIdx_x();
    let n = thread::gridDim_x();

    if thread::threadIdx_x() == 0 {
        unsafe {
            *markers.get_unchecked_mut(block_id as usize) = block_id + 1;
        }
    }

    grid::sync();

    if thread::threadIdx_x() == 0 {
        let base = markers.as_mut_ptr() as *const u32;
        let mut sum: u32 = 0;
        let mut i: u32 = 0;
        while i < n {
            unsafe {
                sum = sum.wrapping_add(*base.add(i as usize));
            }
            i += 1;
        }
        unsafe {
            *out.get_unchecked_mut(block_id as usize) = sum;
        }
    }
}

// =============================================================================
// Typed cooperative-groups kernels
// =============================================================================

/// `WarpTile<32>::ballot(predicate)` should be byte-identical to
/// `warp::ballot(predicate)`. With `predicate = lane_id() & 1`, every
/// lane should report `0xAAAAAAAA`.
#[kernel]
pub fn test_typed_warp32_ballot(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let warp_tile = this_thread_block().tiled_partition::<32>();
    let mask = warp_tile.ballot((warp::lane_id() & 1) != 0);
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// `WarpTile<16>::ballot(predicate)` returns a *tile-relative* mask: bit
/// `k` is set iff the lane at tile-rank `k` had `predicate == true`.
/// With `predicate = lane_id() & 1` every tile should see `0xAAAA`.
#[kernel]
pub fn test_typed_warp16_ballot(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let tile = this_thread_block().tiled_partition::<16>();
    let mask = tile.ballot((warp::lane_id() & 1) != 0);
    if let Some(slot) = out.get_mut(gid) {
        *slot = mask;
    }
}

/// `WarpTile<16>::shfl(my_lane_id, 0)` broadcasts each tile's lane-0
/// value to every lane in that tile. Tile 0 (lanes 0..16) should all
/// see `0`; tile 1 (lanes 16..32) should all see `16`.
#[kernel]
pub fn test_typed_warp16_shfl(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let tile = this_thread_block().tiled_partition::<16>();
    let lane = warp::lane_id();
    let broadcast = tile.shfl(lane, 0);
    if let Some(slot) = out.get_mut(gid) {
        *slot = broadcast;
    }
}

/// `this_grid().sync()` must produce the same observable result as the
/// raw `grid::sync()` test above: every block sees every other block's
/// pre-barrier marker write.
#[kernel]
pub fn test_typed_grid_sync(mut markers: DisjointSlice<u32>, mut out: DisjointSlice<u32>) {
    let grid_handle = this_grid();
    let block_id = thread::blockIdx_x();
    let n = thread::gridDim_x();

    if thread::threadIdx_x() == 0 {
        unsafe {
            *markers.get_unchecked_mut(block_id as usize) = block_id + 1;
        }
    }

    grid_handle.sync();

    if thread::threadIdx_x() == 0 {
        let base = markers.as_mut_ptr() as *const u32;
        let mut sum: u32 = 0;
        let mut i: u32 = 0;
        while i < n {
            unsafe {
                sum = sum.wrapping_add(*base.add(i as usize));
            }
            i += 1;
        }
        unsafe {
            *out.get_unchecked_mut(block_id as usize) = sum;
        }
    }
}

/// Probe `Grid::size()` / `Grid::thread_rank()` from every thread.
/// `out[i]` records `thread_rank()` for the thread whose `index_1d == i`;
/// for a 1D launch the recorded value should equal `i`.
#[kernel]
pub fn test_typed_grid_rank(mut out: DisjointSlice<u32>) {
    let gid = thread::index_1d();
    let g = this_grid();
    let rank = g.thread_rank();
    if let Some(slot) = out.get_mut(gid) {
        *slot = rank;
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Cooperative Groups Demo ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("coop_groups_demo.ptx")
        .expect("Failed to load PTX module");

    const N: usize = 256;
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: ((N / 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // --- active_mask ---
    println!("--- active_mask() ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    cuda_launch! {
        kernel: test_active_mask,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(out)]
    }
    .expect("test_active_mask launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == u32::MAX);
    println!(
        "  every lane saw 0xFFFFFFFF: {}",
        if ok { "yes" } else { "NO" }
    );

    // --- match_any_sync ---
    println!("\n--- match_any_sync(value = lane / 4) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    cuda_launch! {
        kernel: test_match_any,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(out)]
    }
    .expect("test_match_any launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().enumerate().all(|(i, &m)| {
        let group = (i % 32) / 4;
        m == 0xF << (group * 4)
    });
    println!(
        "  every lane saw its 4-bucket mask: {}",
        if ok { "yes" } else { "NO" }
    );

    // --- match_all_sync ---
    println!("\n--- match_all_sync(constant) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    cuda_launch! {
        kernel: test_match_all,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(out)]
    }
    .expect("test_match_all launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == u32::MAX);
    println!(
        "  every lane saw 0xFFFFFFFF: {}",
        if ok { "yes" } else { "NO" }
    );

    // --- grid::sync ---
    println!("\n--- grid::sync() (cooperative launch) ---");
    const BLOCKS: u32 = 32;
    let block_threads = 128u32;
    let coop_cfg = LaunchConfig {
        block_dim: (block_threads, 1, 1),
        grid_dim: (BLOCKS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut markers = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    let mut sums = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    cuda_launch! {
        kernel: test_grid_sync,
        stream: stream,
        module: module,
        config: coop_cfg,
        cooperative: true,
        args: [slice_mut(markers), slice_mut(sums)]
    }
    .expect("test_grid_sync cooperative launch failed");
    let host = sums.to_host_vec(&stream).unwrap();
    let expected: u32 = (1..=BLOCKS).sum();
    let ok = host.iter().all(|&s| s == expected);
    println!(
        "  every block saw the full barrier-flushed marker sum {} : {}",
        expected,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed sums: {:?}", host);
        std::process::exit(1);
    }

    // =========================================================================
    // TYPED COOPERATIVE-GROUPS API
    // =========================================================================

    println!("\n=== Typed cooperative_groups API ===");

    // --- WarpTile<32>::ballot ---
    println!("\n--- WarpTile<32>::ballot(lane_id & 1) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    cuda_launch! {
        kernel: test_typed_warp32_ballot,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(out)]
    }
    .expect("test_typed_warp32_ballot launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == 0xAAAAAAAA);
    println!(
        "  every lane saw 0xAAAAAAAA: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed: {:?}", &host[..32]);
        std::process::exit(1);
    }

    // --- WarpTile<16>::ballot ---
    println!("\n--- WarpTile<16>::ballot(lane_id & 1) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    cuda_launch! {
        kernel: test_typed_warp16_ballot,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(out)]
    }
    .expect("test_typed_warp16_ballot launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().all(|&m| m == 0xAAAA);
    println!(
        "  every lane in every 16-lane tile saw 0xAAAA: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed: {:?}", &host[..32]);
        std::process::exit(1);
    }

    // --- WarpTile<16>::shfl ---
    println!("\n--- WarpTile<16>::shfl(lane_id, 0) ---");
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    cuda_launch! {
        kernel: test_typed_warp16_shfl,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(out)]
    }
    .expect("test_typed_warp16_shfl launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().enumerate().all(|(i, &v)| {
        let lane = (i as u32) % 32;
        let expected = if lane < 16 { 0 } else { 16 };
        v == expected
    });
    println!(
        "  tile 0 broadcasts 0, tile 1 broadcasts 16: {}",
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed (first 32): {:?}", &host[..32]);
        std::process::exit(1);
    }

    // --- this_grid().sync() ---
    println!("\n--- this_grid().sync() (cooperative launch) ---");
    let mut markers = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    let mut sums = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize).unwrap();
    cuda_launch! {
        kernel: test_typed_grid_sync,
        stream: stream,
        module: module,
        config: coop_cfg,
        cooperative: true,
        args: [slice_mut(markers), slice_mut(sums)]
    }
    .expect("test_typed_grid_sync cooperative launch failed");
    let host = sums.to_host_vec(&stream).unwrap();
    let expected: u32 = (1..=BLOCKS).sum();
    let ok = host.iter().all(|&s| s == expected);
    println!(
        "  typed grid sync produces the same sum {} : {}",
        expected,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        println!("  observed sums: {:?}", host);
        std::process::exit(1);
    }

    // --- this_grid().thread_rank() ---
    println!("\n--- this_grid().thread_rank() ---");
    let total = (BLOCKS * block_threads) as usize;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, total).unwrap();
    cuda_launch! {
        kernel: test_typed_grid_rank,
        stream: stream,
        module: module,
        config: coop_cfg,
        args: [slice_mut(out)]
    }
    .expect("test_typed_grid_rank launch failed");
    let host = out.to_host_vec(&stream).unwrap();
    let ok = host.iter().enumerate().all(|(i, r)| (*r as usize) == i);
    println!(
        "  thread_rank() forms the identity permutation 0..{}: {}",
        total,
        if ok { "yes" } else { "NO" }
    );
    if !ok {
        let mismatches: Vec<_> = host
            .iter()
            .enumerate()
            .filter(|(i, r)| (**r as usize) != *i)
            .take(8)
            .collect();
        println!("  first mismatches (idx, observed): {:?}", mismatches);
        std::process::exit(1);
    }

    println!("\n=== All cooperative-groups checks passed ===");
}
