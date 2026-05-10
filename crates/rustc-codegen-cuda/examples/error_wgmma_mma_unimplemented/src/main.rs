/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: `wgmma_mma_*` is not yet implemented.
//!
//! `cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16` and friends call into
//! a placeholder lowering whose full implementation requires register
//! allocation for 16+ output registers. Until that lands, the codegen
//! backend is expected to reject these calls at build time with a clear
//! diagnostic — not silently emit a comment and produce PTX that
//! multiplies-accumulates to zero.
//!
//! Usage:
//!   cargo oxide run error_wgmma_mma_unimplemented
//!
//! Expected: build FAILS with
//!   "wgmma.mma_async lowering is not yet implemented; ..."

use cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16;
use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub unsafe fn unsupported_wgmma_mma_kernel(mut out: DisjointSlice<u32>) {
    let mut acc: [[f32; 8]; 4] = [[0.0f32; 8]; 4];
    unsafe {
        wgmma_mma_m64n64k16_f32_bf16(&mut acc, 0u64, 0u64);
    }
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc[0][0].to_bits();
    }
}

fn main() {
    println!("=== error_wgmma_mma_unimplemented ===");
    println!("This example is intentionally broken to test the diagnostic for");
    println!("the not-yet-implemented `wgmma.mma_async` lowering.");
    println!();
    println!("If you see this message, the build did NOT fail and the test");
    println!("would have detected the previous silent-miscompile regression.");
}
