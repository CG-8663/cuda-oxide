/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error Test Example - Tests compiler error handling
//!
//! This example contains both valid and intentionally broken kernels.
//! The broken kernel should cause compilation to FAIL with helpful error messages.
//!
//! Usage:
//!   cargo oxide run error
//!
//! Expected: Compilation should FAIL with error messages

use cuda_device::{DisjointSlice, kernel, thread};

/// VALID: f64 → f32 cast example (this compiles correctly)
///
/// This kernel demonstrates a VALID f64 to f32 conversion using `as f32`.
/// The compiler correctly generates: load f64 → add f64 → cvt.f32.f64 → store f32
#[kernel]
pub fn valid_f64_to_f32_kernel(a: &[f64], b: &[f64], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();

    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = (a[idx_raw] + b[idx_raw]) as f32;
    }
}

/// ERROR: Uses format_args! which isn't supported on GPU
///
/// The compiler should fail with an error about unsupported operations.
#[kernel]
pub fn unsupported_format_kernel(a: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();

    if let Some(c_elem) = c.get_mut(idx) {
        let _formatted = core::format_args!("{}", a[idx_raw]);
        *c_elem = a[idx_raw];
    }
}
fn main() {
    println!("=== Error Test Example (Unified) ===");
    println!();
    println!("This example is intentionally broken to test error handling.");
    println!("It should NOT compile successfully.");
    println!();
    println!("If you see this message, something went wrong!");
    println!("The kernel compilation should have failed.");
}
