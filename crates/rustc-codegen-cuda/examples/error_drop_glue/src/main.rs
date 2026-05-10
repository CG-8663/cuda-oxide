/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: drop glue is not supported on the device.
//!
//! cuda-oxide does not yet emit device-side `drop_in_place` calls. The
//! mir-importer's `translate_drop` must reject `TerminatorKind::Drop` for
//! types with drop glue and surface a clear diagnostic, rather than
//! lowering to a goto and silently skipping the destructor.
//!
//! Usage:
//!   cargo oxide run error_drop_glue
//!
//! Expected: build FAILS with
//!   "drop of `<TypeName>` is not supported on the device; ..."

use cuda_device::{DisjointSlice, kernel, thread};

pub struct DropMarker {
    target: *mut u32,
}

impl Drop for DropMarker {
    fn drop(&mut self) {
        // The body is irrelevant for the test; the bug is that the entire
        // drop call is silently elided. Anything observable here would be
        // gone from the PTX without the importer-side rejection.
        unsafe {
            self.target.write(0xDEADBEEFu32);
        }
    }
}

#[kernel]
pub fn drop_glue_kernel(mut out: DisjointSlice<u32>) {
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = 0;
        let _m = DropMarker {
            target: slot as *mut u32,
        };
        // _m drops at end of scope; expected to write 0xDEADBEEF.
    }
}

fn main() {
    println!("=== error_drop_glue ===");
    println!("This example is intentionally broken to test the diagnostic for");
    println!("drop glue on the device. The build must FAIL at codegen time.");
    println!();
    println!("If you see this message, the build did NOT fail and the test");
    println!("would have detected the previous silent-miscompile regression.");
}
