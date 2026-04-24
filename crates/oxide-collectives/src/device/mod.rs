/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side collective building blocks.
//!
//! These types are the first Phase 3 layer on top of the Phase 2 symmetric
//! heap:
//!
//! - [`PeIndex`] identifies one participant PE.
//! - [`Team`] carries the local heap alias and basic PE topology helpers.
//! - [`SymmetricRef`] turns a typed offset inside the symmetric heap into local
//!   and remote pointers.
//! - [`signal`] provides the release/acquire helpers used for cross-GPU
//!   publication with system-scope atomics.

pub mod pe;
pub mod put_get;
pub mod signal;
pub mod team;

pub use pe::PeIndex;
pub use put_get::SymmetricRef;
pub use team::Team;
