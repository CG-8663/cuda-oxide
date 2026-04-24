/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration test for the Phase 2 symmetric heap substrate.
//!
//! The test builds a two-PE heap, writes one source buffer per PE, and then
//! uses peer-visible DtoD copies into separate staging buffers to prove that
//! each PE can reach the other PE's symmetric allocation.

use cuda_bindings::CUdeviceptr;
use cuda_core::{CudaContext, IntoResult, memory, peer};
use oxide_collectives::SymmetricHeap;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// Returns the number of CUDA devices visible to the current process.
fn gpu_count() -> Result<usize, cuda_core::DriverError> {
    unsafe { cuda_core::init(0)? };
    let mut count = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

/// Uploads `values` into `dst` using `ctx`'s default stream.
fn write_u32s(ctx: &Arc<CudaContext>, dst: CUdeviceptr, values: &[u32]) {
    let stream = ctx.default_stream();
    let num_bytes = std::mem::size_of_val(values);
    ctx.bind_to_thread().expect("bind context before HtoD copy");
    unsafe {
        memory::memcpy_htod_async(dst, values.as_ptr(), num_bytes, stream.cu_stream())
            .expect("HtoD copy into symmetric heap");
    }
    ctx.synchronize().expect("synchronize after HtoD copy");
}

/// Reads `len` `u32`s from `src` back to host memory.
fn read_u32s(ctx: &Arc<CudaContext>, src: CUdeviceptr, len: usize) -> Vec<u32> {
    let stream = ctx.default_stream();
    let mut values = vec![0u32; len];
    let num_bytes = std::mem::size_of_val(values.as_slice());
    ctx.bind_to_thread().expect("bind context before DtoH copy");
    unsafe {
        memory::memcpy_dtoh_async(values.as_mut_ptr(), src, num_bytes, stream.cu_stream())
            .expect("DtoH copy from symmetric heap");
    }
    ctx.synchronize().expect("synchronize after DtoH copy");
    values
}

/// Copies `num_bytes` from `src` to `dst` within the symmetric heap.
fn copy_dtod(ctx: &Arc<CudaContext>, dst: CUdeviceptr, src: CUdeviceptr, num_bytes: usize) {
    let stream = ctx.default_stream();
    ctx.bind_to_thread().expect("bind context before DtoD copy");
    unsafe {
        memory::memcpy_dtod_async(dst, src, num_bytes, stream.cu_stream())
            .expect("DtoD copy inside symmetric heap");
    }
    ctx.synchronize().expect("synchronize after DtoD copy");
}

#[test]
fn symmetric_heap_cross_gpu_roundtrip() {
    let count = gpu_count().expect("GPU count query");
    if count < 2 {
        eprintln!("skipping symmetric heap test: need at least two GPUs, found {count}");
        return;
    }

    let ctx0 = CudaContext::new(0).expect("GPU 0 context");
    let ctx1 = CudaContext::new(1).expect("GPU 1 context");

    let can_01 = peer::can_access_peer(&ctx0, &ctx1).expect("query peer access 0->1");
    let can_10 = peer::can_access_peer(&ctx1, &ctx0).expect("query peer access 1->0");
    if !(can_01 && can_10) {
        eprintln!("skipping symmetric heap test: GPUs 0 and 1 are not mutually peer accessible");
        return;
    }

    let heap = SymmetricHeap::new(&[ctx0.clone(), ctx1.clone()], 4096)
        .expect("create symmetric heap across two GPUs");
    // Keep source and staging separate so each direction can be checked
    // independently without destroying the original patterns.
    let source = heap
        .alloc::<u32>(256)
        .expect("allocate symmetric u32 buffer");
    let staging = heap
        .alloc::<u32>(256)
        .expect("allocate symmetric staging buffer");

    let pattern0: Vec<u32> = (0..source.len()).map(|i| i as u32 * 3 + 1).collect();
    let pattern1: Vec<u32> = (0..source.len()).map(|i| 10_000 + i as u32 * 7).collect();
    let num_bytes = std::mem::size_of_val(pattern0.as_slice());

    write_u32s(
        &ctx0,
        source.local_ptr(0).expect("local ptr for PE 0"),
        &pattern0,
    );
    write_u32s(
        &ctx1,
        source.local_ptr(1).expect("local ptr for PE 1"),
        &pattern1,
    );

    copy_dtod(
        &ctx0,
        staging.local_ptr(0).expect("local staging dst on PE 0"),
        source.remote_ptr(0, 1).expect("remote src from PE 1"),
        num_bytes,
    );
    let readback_from_1 = read_u32s(
        &ctx0,
        staging.local_ptr(0).expect("readback staging ptr on PE 0"),
        256,
    );
    assert_eq!(readback_from_1, pattern1);

    copy_dtod(
        &ctx1,
        staging.local_ptr(1).expect("local staging dst on PE 1"),
        source.remote_ptr(1, 0).expect("remote src from PE 0"),
        num_bytes,
    );
    let readback_from_0 = read_u32s(
        &ctx1,
        staging.local_ptr(1).expect("readback staging ptr on PE 1"),
        256,
    );
    assert_eq!(readback_from_0, pattern0);
}
