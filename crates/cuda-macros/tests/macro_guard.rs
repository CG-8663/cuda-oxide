// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail tests for the `#[kernel]` / `#[device]` macro guard
//! against the reserved `cuda_oxide_*` namespace.
//!
//! These tests verify that the proc macros reject user-defined functions
//! whose name starts with the reserved cuda-oxide prefix, before the
//! compiler ever sees the (potentially-confusing) renamed form.
//!
//! See `crates/reserved-oxide-symbols/` for the source of truth on the
//! reserved namespace.

#[test]
fn reserved_name_macro_guard() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/kernel_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_extern_reserved_name.rs");
}
