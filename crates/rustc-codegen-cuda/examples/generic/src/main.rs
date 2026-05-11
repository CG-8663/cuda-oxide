/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Generic Kernel Example
//!
//! Tests whether the collector correctly handles monomorphized generic kernels.
//!
//! Build and run with:
//!   cargo oxide pipeline generic
//!   cargo oxide run generic
//!
//! ## What This Tests
//!
//! 1. Generic kernel definition: `fn scale<T>(factor: T, ...)`
//! 2. Monomorphization: When called with `scale::<f32>`, rustc creates a specific version
//! 3. Collection: Does our collector find the monomorphized instance?
//! 4. PTX generation: Does the backend generate valid PTX?
//!
//! ## Expected PTX
//!
//! We should see a PTX entry point for `scale` (or `scale_f32` if we add type info).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use std::ops::{Add, Mul};

// =============================================================================
// GENERIC KERNELS
// =============================================================================

/// Generic scale kernel - multiplies each element by a factor.
///
/// This kernel is generic over T. When called with `scale::<f32>`, rustc
/// monomorphizes it to a concrete f32 version.
#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[idx_raw] * factor;
    }
}

/// Generic add kernel - adds two arrays element-wise.
#[kernel]
pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx_raw] + b[idx_raw];
    }
}

// =============================================================================
// HOST CODE
// =============================================================================
//
// Using cuda_launch! with type parameters to trigger monomorphization.
// The macro expansion references cuda_oxide_kernel_<hash>_scale::<f32>, which forces
// rustc to generate the monomorphized version.

use cuda_host::cuda_launch;
fn main() {
    println!("=== Unified Generic Kernel Test ===\n");

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Test data
    const N: usize = 1024;
    let factor: f32 = 2.5;
    let input_data: Vec<f32> = (0..N).map(|i| i as f32).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_data).expect("Failed to copy input");
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N).expect("Failed to alloc output");

    let module = ctx
        .load_module_from_file("generic.ptx")
        .expect("Failed to load PTX module");

    // =========================================================================
    // THE KEY: cuda_launch! with type parameter forces monomorphization!
    // =========================================================================
    //
    // cuda_launch! {
    //     kernel: scale::<f32>,  // <-- Type parameter here!
    //     stream: stream,
    //     module: module,
    //     config: LaunchConfig::for_num_elems(N as u32),
    //     args: [factor, slice(input_dev), slice_mut(output_dev)]
    // }
    //
    // This expands to code that references cuda_oxide_kernel_<hash>_scale::<f32>,
    // which forces rustc to monomorphize it, which makes it visible to the
    // backend's collector.

    println!("\nLaunching scale::<f32> kernel...");
    println!("  factor = {}", factor);
    println!("  N = {}", N);

    cuda_launch! {
        kernel: scale::<f32>,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [factor, slice(input_dev), slice_mut(output_dev)]
    }
    .expect("Kernel launch failed");

    let output_host = output_dev
        .to_host_vec(&stream)
        .expect("Failed to copy output back");

    let errors = (0..N)
        .filter(|&i| (output_host[i] - input_data[i] * factor).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("\n✓ SUCCESS: All {} elements correct!", N);
    } else {
        println!("\n✗ FAILED: {} errors", errors);
    }
}
