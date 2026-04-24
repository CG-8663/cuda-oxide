/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side building blocks for symmetric multi-GPU collectives.
//!
//! Phase 2 introduced the host substrate; the current Phase 3 slice starts to
//! expose the device-side view that kernels will use on top of it:
//!
//! - [`device::PeIndex`] names one PE inside a launched team.
//! - [`device::Team`] carries the current PE plus its local symmetric-heap
//!   alias.
//! - [`device::SymmetricRef`] performs typed local/remote pointer arithmetic on
//!   top of that team view.
//! - [`Topology`] discovers which PEs can directly reach one another.
//! - [`SymmetricHeap`] creates one per-PE chunk of VMM-backed memory and maps
//!   every chunk into every PE's address space at a deterministic offset.
//! - [`SymmetricAlloc`] is a typed handle into that symmetric heap.
//!
//! In this prototype, "symmetric" means:
//!
//! - every PE owns one physical chunk,
//! - all chunks are visible from every PE,
//! - each typed allocation lives at the same byte offset inside every chunk.
//!
//! The current implementation reserves one virtual-address window per PE and
//! maps the same physical chunks into each window. That means the logical
//! layout is identical everywhere, but a given allocation may have multiple
//! valid virtual-address aliases rather than one globally identical pointer.

pub mod device;
pub mod error;
pub mod host;

pub use device::{PeIndex, SymmetricRef, Team};
pub use error::{CollectiveError, Result};
pub use host::{SymmetricAlloc, SymmetricHeap, Topology};
