/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

//! Negative test: enum discriminant writes are not yet supported.
//!
//! The custom MIR helper below emits `StatementKind::SetDiscriminant`.
//! The mir-importer's statement translator must reject this with a clear
//! diagnostic until the lowering is implemented.
//!
//! Usage:
//!   cargo oxide run error_set_discriminant_unhandled
//!
//! Expected: build FAILS with
//!   "SetDiscriminant statements are not yet supported on the device; ..."

use core::intrinsics::mir::*;
use cuda_device::{DisjointSlice, kernel};

enum DeviceState {
    Empty,
    Full(u32),
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_set_discriminant(state: &mut DeviceState) {
    mir!({
        SetDiscriminant(*state, 0);
        Return()
    })
}

#[kernel]
pub fn set_discriminant_kernel(mut out: DisjointSlice<u32>) {
    if let Some((slot, idx)) = out.get_mut_indexed() {
        let raw_idx = idx.get();
        let mut state = if raw_idx & 1 == 0 {
            DeviceState::Full(raw_idx as u32)
        } else {
            DeviceState::Empty
        };

        let before = match state {
            DeviceState::Empty => 0,
            DeviceState::Full(value) => value,
        };

        // This helper emits the exact MIR statement that must be rejected until
        // SetDiscriminant lowering is implemented.
        force_set_discriminant(&mut state);

        let after = match state {
            DeviceState::Empty => 0,
            DeviceState::Full(value) => value,
        };

        *slot = before + after;
    }
}

fn main() {
    println!("=== error_set_discriminant_unhandled ===");
    println!("This example is intentionally broken to test the diagnostic for");
    println!("not-yet-implemented MIR SetDiscriminant lowering.");
    println!();
    println!("If you see this message, the build did NOT fail and the test");
    println!("would have detected the previous silent-miscompile regression.");
}
