/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused `#[cuda_module]` host ABI contract test.
//!
//! The kernel intentionally mixes common host-side argument shapes the typed
//! module macro must lower correctly: scalars, slice, raw device pointer, and
//! `DisjointSlice` output.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn mixed_abi(
        scale: f32,
        bias: f32,
        extra: f32,
        input: &[f32],
        raw_offsets: *const f32,
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = output.get_mut(idx) {
            let offset = unsafe { *raw_offsets.add(idx_raw) };
            *out_elem = input[idx_raw] * scale + bias + extra + offset;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuda_module ABI Contract Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    const N: usize = 1024;
    let scale = 1.5f32;
    let bias = 2.0f32;
    let extra = 7.0f32;
    let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let offset_host: Vec<f32> = (0..N).map(|i| (i % 5) as f32).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
    let offset_dev = DeviceBuffer::from_host(&stream, &offset_host)?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    module.mixed_abi(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        scale,
        bias,
        extra,
        &input_dev,
        offset_dev.cu_deviceptr() as *const f32,
        &mut output_dev,
    )?;

    let output = output_dev.to_host_vec(&stream)?;
    let errors = (0..N)
        .filter(|&i| {
            let expected = input_host[i] * scale + bias + extra + offset_host[i];
            (output[i] - expected).abs() > 1e-5
        })
        .count();

    assert_eq!(errors, 0, "mixed ABI kernel produced {errors} errors");
    println!("SUCCESS: mixed ABI typed launch passed");
    Ok(())
}
