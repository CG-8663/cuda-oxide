/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side collective setup primitives.
//!
//! - [`Topology`] captures which participating PEs can reach one another.
//! - [`SymmetricHeap`] owns the cross-PE VMM layout.
//! - [`SymmetricAlloc`] names a typed offset inside that symmetric heap.

pub mod symmetric_heap;
pub mod topology;

pub use symmetric_heap::{SymmetricAlloc, SymmetricHeap};
pub use topology::Topology;
