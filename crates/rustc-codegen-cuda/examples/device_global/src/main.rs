/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ordinary device global static example.
//!
//! Build and run with:
//!   cargo oxide run device_global

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::kernel;
use cuda_host::cuda_launch;

static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;

#[kernel]
pub unsafe fn device_global(out: *mut u64) {
    unsafe {
        DEVICE_COUNTER = DEVICE_COUNTER + 1;
        DEVICE_MARKER = 0x00C0_FFEE;
        *out = DEVICE_COUNTER ^ (DEVICE_MARKER as u64);
    }
}

fn main() {
    println!("=== Device Global Static Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("Failed to allocate output");

    let module = ctx
        .load_module_from_file("device_global.ptx")
        .expect("Failed to load PTX module");

    for launch_idx in 1..=2 {
        cuda_launch! {
            kernel: device_global,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(1),
            args: [out_dev.cu_deviceptr() as *mut u64]
        }
        .expect("Kernel launch failed");

        let result = out_dev.to_host_vec(&stream).expect("Failed to copy result")[0];
        let expected = launch_idx ^ 0x00C0_FFEEu64;

        println!("Launch {launch_idx}: result = {result:#x}");
        if result != expected {
            eprintln!("FAILED: expected {expected:#x}, got {result:#x}");
            std::process::exit(1);
        }
    }

    println!("\nSUCCESS: device global static persisted across launches.");
}
