/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: `core::ptr::copy_nonoverlapping` is not yet supported.
//!
//! `copy_nonoverlapping` lowers to a MIR `StatementKind::Intrinsic(
//! NonDivergingIntrinsic::CopyNonOverlapping(_))`. The mir-importer's
//! statement translator must reject this with a clear diagnostic until the
//! lowering is implemented — the previous catch-all silently dropped the
//! statement, producing PTX where the memcpy was completely absent.
//!
//! Usage:
//!   cargo oxide run error_copy_nonoverlapping_unhandled
//!
//! Expected: build FAILS with
//!   "core::ptr::copy_nonoverlapping is not yet supported on the device; ..."

use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn copy_nonoverlapping_kernel(input: &[u32], mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        unsafe {
            let src = input.as_ptr().add(idx.get());
            let dst = slot as *mut u32;
            core::ptr::copy_nonoverlapping(src, dst, 1);
        }
    }
}

fn main() {
    println!("=== error_copy_nonoverlapping_unhandled ===");
    println!("This example is intentionally broken to test the diagnostic for");
    println!("the not-yet-implemented `core::ptr::copy_nonoverlapping` lowering.");
    println!();
    println!("If you see this message, the build did NOT fail and the test");
    println!("would have detected the previous silent-miscompile regression.");
}
