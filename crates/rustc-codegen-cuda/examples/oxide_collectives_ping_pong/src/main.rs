/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Two-GPU runtime validation for `oxide-collectives` Phase 3.
//!
//! This example launches one kernel per PE and performs this ping-pong:
//!
//! 1. GPU 1 runs the consumer kernel and keeps polling its local `flag` slot.
//! 2. GPU 0 runs the producer kernel and writes `PAYLOAD_VALUE` into GPU 1's
//!    remote `payload` slot.
//! 3. GPU 0 publishes `READY_VALUE` into GPU 1's remote `flag` slot with
//!    `threadfence_system()` plus a release store.
//! 4. GPU 1 sees `READY_VALUE` with acquire loads, then reads the payload.
//! 5. GPU 1 writes the value it observed into its local `observed` slot so the
//!    host can verify that the payload arrived only after the ready flag.
//!
//! Build and run with:
//!
//! ```bash
//! cargo oxide run oxide_collectives_ping_pong
//! ```

use cuda_core::{CudaContext, IntoResult, LaunchConfig, memory, peer, sys};
use cuda_device::atomic::SystemAtomicU32;
use cuda_device::{kernel, thread};
use cuda_host::cuda_launch;
use oxide_collectives::SymmetricHeap;
use oxide_collectives::device::{Team, signal};
use std::mem::MaybeUninit;
use std::sync::Arc;

const ONE_THREAD_LAUNCH: LaunchConfig = LaunchConfig {
    grid_dim: (1, 1, 1),
    block_dim: (1, 1, 1),
    shared_mem_bytes: 0,
};
const READY_VALUE: u32 = 1;
const PAYLOAD_VALUE: u32 = 0x1234_5678;
const TIMEOUT_SENTINEL: u32 = 0xBAD0_BAD0;
const SPIN_LIMIT: u32 = 100_000_000;

/// GPU 0 kernel that writes the payload into GPU 1's symmetric slot and then
/// publishes the ready flag with system scope.
///
/// The first four arguments are the scalarized pieces of the device-side
/// [`Team`] view. We pass them individually instead of launching with `Team` as
/// one by-value kernel argument so the host/device launch ABI is explicit.
#[kernel]
pub fn producer(
    pe: u32,
    pe_count: u32,
    heap_base: *mut u8,
    chunk_size: usize,
    payload_offset: usize,
    flag_offset: usize,
    payload_value: u32,
) {
    if thread::index_1d().get() != 0 {
        return;
    }

    // Pass team metadata as scalar kernel params and reconstruct the device-side
    // `Team` view inside the kernel.
    let team = unsafe { make_team(pe, pe_count, heap_base, chunk_size) };
    let target_pe = team.right_pe();
    let payload = team
        .symmetric_ref::<u32>(payload_offset, 1)
        .expect("payload slot must fit inside one PE chunk");
    let flags = team
        .symmetric_ref::<u32>(flag_offset, 1)
        .expect("flag slot must fit inside one PE chunk");

    unsafe {
        payload.put(&payload_value, target_pe);
        let remote_flag = signal::atomic_ref(flags.remote_ptr(target_pe) as *const SystemAtomicU32);
        signal::publish_system(remote_flag, READY_VALUE);
    }
}

/// GPU 1 kernel that waits for the ready flag in its local symmetric slot,
/// reads the payload, and stores the observed value for host verification.
///
/// Like [`producer`], the team metadata is passed as scalar launch parameters
/// and reconstructed into a [`Team`] inside the kernel body.
#[kernel]
pub fn consumer(
    pe: u32,
    pe_count: u32,
    heap_base: *mut u8,
    chunk_size: usize,
    payload_offset: usize,
    flag_offset: usize,
    observed_offset: usize,
) {
    if thread::index_1d().get() != 0 {
        return;
    }

    let team = unsafe { make_team(pe, pe_count, heap_base, chunk_size) };
    let payload = team
        .symmetric_ref::<u32>(payload_offset, 1)
        .expect("payload slot must fit inside one PE chunk");
    let flags = team
        .symmetric_ref::<u32>(flag_offset, 1)
        .expect("flag slot must fit inside one PE chunk");
    let observed = team
        .symmetric_ref::<u32>(observed_offset, 1)
        .expect("observed slot must fit inside one PE chunk");

    let local_flag = unsafe { signal::atomic_ref(flags.local_ptr() as *const SystemAtomicU32) };
    let mut spins = 0;
    while spins < SPIN_LIMIT {
        if signal::is_published_system(local_flag, READY_VALUE) {
            let value = unsafe { payload.get(team.pe()) };
            unsafe {
                observed.put(&value, team.pe());
            }
            return;
        }
        spins += 1;
    }

    unsafe {
        observed.put(&TIMEOUT_SENTINEL, team.pe());
    }
}

/// Reconstructs the device-side [`Team`] view from its scalar launch
/// parameters.
///
/// # Safety
///
/// The arguments must satisfy the invariants required by
/// [`Team::new_unchecked`]:
/// - `pe < pe_count`
/// - `heap_base` must be the base of this PE's local symmetric-heap alias
/// - `chunk_size` must match the host-side symmetric-heap layout
#[inline(always)]
unsafe fn make_team<'a>(pe: u32, pe_count: u32, heap_base: *mut u8, chunk_size: usize) -> Team<'a> {
    unsafe { Team::new_unchecked(pe, pe_count, heap_base, chunk_size) }
}

/// Initializes the CUDA driver and returns the number of visible CUDA devices.
fn gpu_count() -> Result<usize, cuda_core::DriverError> {
    unsafe { cuda_core::init(0)? };
    let mut count = MaybeUninit::uninit();
    unsafe {
        sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

/// Writes one `u32` into device memory using the context's default stream and
/// synchronizes before returning.
fn write_u32(ctx: &Arc<CudaContext>, dst: sys::CUdeviceptr, value: u32) {
    let stream = ctx.default_stream();
    ctx.bind_to_thread()
        .expect("bind context before HtoD copy into symmetric slot");
    unsafe {
        memory::memcpy_htod_async(
            dst,
            std::ptr::addr_of!(value),
            std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
        .expect("HtoD copy into symmetric slot");
    }
    ctx.synchronize()
        .expect("synchronize after HtoD copy into symmetric slot");
}

/// Reads one `u32` from device memory using the context's default stream and
/// synchronizes before returning the host value.
fn read_u32(ctx: &Arc<CudaContext>, src: sys::CUdeviceptr) -> u32 {
    let stream = ctx.default_stream();
    let mut value = 0u32;
    ctx.bind_to_thread()
        .expect("bind context before DtoH copy from symmetric slot");
    unsafe {
        memory::memcpy_dtoh_async(
            std::ptr::addr_of_mut!(value),
            src,
            std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
        .expect("DtoH copy from symmetric slot");
    }
    ctx.synchronize()
        .expect("synchronize after DtoH copy from symmetric slot");
    value
}

fn main() {
    println!("=== oxide-collectives ping-pong ===\n");

    let count = gpu_count().expect("query visible CUDA device count");
    if count < 2 {
        println!("skipping: need at least two GPUs, found {count}");
        return;
    }

    let ctx0 = CudaContext::new(0).expect("create GPU 0 context");
    let ctx1 = CudaContext::new(1).expect("create GPU 1 context");

    let can_01 = peer::can_access_peer(&ctx0, &ctx1).expect("query peer access 0->1");
    let can_10 = peer::can_access_peer(&ctx1, &ctx0).expect("query peer access 1->0");
    if !(can_01 && can_10) {
        println!("skipping: GPUs 0 and 1 are not mutually peer accessible");
        return;
    }

    let heap = SymmetricHeap::new(&[ctx0.clone(), ctx1.clone()], 4096)
        .expect("create symmetric heap across GPUs 0 and 1");
    let payload = heap
        .alloc::<u32>(1)
        .expect("allocate symmetric payload slot");
    let flag = heap.alloc::<u32>(1).expect("allocate symmetric flag slot");
    let observed = heap
        .alloc::<u32>(1)
        .expect("allocate symmetric observed slot");

    for (pe, ctx) in [(0usize, &ctx0), (1usize, &ctx1)] {
        write_u32(
            ctx,
            payload.local_ptr(pe).expect("payload local pointer"),
            0,
        );
        write_u32(ctx, flag.local_ptr(pe).expect("flag local pointer"), 0);
        write_u32(
            ctx,
            observed.local_ptr(pe).expect("observed local pointer"),
            0,
        );
    }

    let team0 = heap.team(0).expect("build GPU 0 team view");
    let team1 = heap.team(1).expect("build GPU 1 team view");
    let ptx_path = format!(
        "{}/oxide_collectives_ping_pong.ptx",
        env!("CARGO_MANIFEST_DIR")
    );

    let stream0 = ctx0.default_stream();
    let stream1 = ctx1.default_stream();
    let module0 = ctx0
        .load_module_from_file(&ptx_path)
        .expect("load PTX module into GPU 0 context");
    let module1 = ctx1
        .load_module_from_file(&ptx_path)
        .expect("load PTX module into GPU 1 context");

    // Pass the device-side `Team` fields explicitly so the host launch packet
    // matches the kernel signature one scalar parameter at a time. The kernels
    // rebuild `Team` internally from these four metadata values.
    cuda_launch! {
        kernel: consumer,
        stream: stream1,
        module: module1,
        config: ONE_THREAD_LAUNCH,
        args: [
            team1.pe().get(),
            team1.pe_count(),
            team1.heap_base(),
            team1.chunk_size_bytes(),
            payload.offset_bytes(),
            flag.offset_bytes(),
            observed.offset_bytes()
        ]
    }
    .expect("launch consumer on GPU 1");

    cuda_launch! {
        kernel: producer,
        stream: stream0,
        module: module0,
        config: ONE_THREAD_LAUNCH,
        args: [
            team0.pe().get(),
            team0.pe_count(),
            team0.heap_base(),
            team0.chunk_size_bytes(),
            payload.offset_bytes(),
            flag.offset_bytes(),
            PAYLOAD_VALUE
        ]
    }
    .expect("launch producer on GPU 0");

    ctx0.synchronize()
        .expect("synchronize GPU 0 after producer");
    ctx1.synchronize()
        .expect("synchronize GPU 1 after consumer");

    let gpu1_flag = read_u32(&ctx1, flag.local_ptr(1).expect("GPU 1 flag pointer"));
    let gpu1_payload = read_u32(&ctx1, payload.local_ptr(1).expect("GPU 1 payload pointer"));
    let gpu1_observed = read_u32(
        &ctx1,
        observed.local_ptr(1).expect("GPU 1 observed pointer"),
    );

    println!("GPU 1 flag     = {gpu1_flag:#010x}");
    println!("GPU 1 payload  = {gpu1_payload:#010x}");
    println!("GPU 1 observed = {gpu1_observed:#010x}");

    if gpu1_observed != PAYLOAD_VALUE {
        eprintln!(
            "\nping-pong failed: expected observed {PAYLOAD_VALUE:#010x}, got {gpu1_observed:#010x}"
        );
        if gpu1_observed == TIMEOUT_SENTINEL {
            eprintln!("consumer timed out waiting for the system-scope ready flag");
        }
        std::process::exit(1);
    }

    println!("\nPASS: GPU 1 observed the payload after the ready flag publication");
}
