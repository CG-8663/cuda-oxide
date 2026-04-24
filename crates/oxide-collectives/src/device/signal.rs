/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-GPU signaling helpers built on system-scope atomics.

use cuda_device::atomic::{AtomicOrdering, SystemAtomicU32};
use cuda_device::threadfence_system;

/// Interprets `ptr` as a `SystemAtomicU32`.
///
/// # Safety
///
/// `ptr` must point at memory whose layout is compatible with
/// [`SystemAtomicU32`] and remain valid for the returned reference.
#[inline(always)]
pub unsafe fn atomic_ref<'a>(ptr: *const SystemAtomicU32) -> &'a SystemAtomicU32 {
    unsafe { &*ptr }
}

/// Publishes `value` through `flag` after a system-scoped fence.
///
/// The intended pairing is:
///
/// - writer: ordinary remote-memory stores
/// - writer: `publish_system(flag, ready_value)`
/// - reader: `wait_until_system(flag, ready_value)`
/// - reader: ordinary loads of the payload
///
/// # Safety
///
/// Must be called from CUDA device code. On the host, the intrinsic stubs in
/// `cuda-device` are `unreachable!()`.
#[inline(always)]
pub unsafe fn publish_system(flag: &SystemAtomicU32, value: u32) {
    threadfence_system();
    flag.store(value, AtomicOrdering::Release);
}

/// Returns `true` when `flag` has reached `expected`.
#[inline(always)]
pub fn is_published_system(flag: &SystemAtomicU32, expected: u32) -> bool {
    flag.load(AtomicOrdering::Acquire) == expected
}

/// Spins until `flag` reaches `expected`.
#[inline(always)]
pub fn wait_until_system(flag: &SystemAtomicU32, expected: u32) {
    while !is_published_system(flag, expected) {
        core::hint::spin_loop();
    }
}
