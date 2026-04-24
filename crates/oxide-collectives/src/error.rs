/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types for collective host-side setup.

use cuda_core::DriverError;
use thiserror::Error;

/// Crate-local result type.
pub type Result<T> = std::result::Result<T, CollectiveError>;

/// Errors that can occur while building or using the Phase 2 topology and
/// symmetric heap substrate.
#[derive(Debug, Error)]
pub enum CollectiveError {
    /// Propagated CUDA driver failure.
    #[error(transparent)]
    Driver(#[from] DriverError),

    /// No contexts were supplied to a collective constructor.
    #[error("at least one CUDA context is required")]
    NoContexts,

    /// The same physical device ordinal was provided more than once.
    #[error(
        "CUDA contexts must refer to distinct device ordinals; device {0} was provided more than once"
    )]
    DuplicateDevice(usize),

    /// The requested PE pair does not have direct peer connectivity.
    #[error(
        "PE {from} (device {from_ordinal}) cannot directly access PE {to} (device {to_ordinal})"
    )]
    PeerAccessUnavailable {
        /// Source PE index inside the current topology.
        from: usize,
        /// CUDA device ordinal for the source PE.
        from_ordinal: usize,
        /// Destination PE index inside the current topology.
        to: usize,
        /// CUDA device ordinal for the destination PE.
        to_ordinal: usize,
    },

    /// A PE index was outside the known topology bounds.
    #[error("invalid PE index {pe}; expected 0..{pe_count}")]
    InvalidPe {
        /// The offending PE index.
        pe: usize,
        /// Number of known PEs.
        pe_count: usize,
    },

    /// The current number of PEs does not fit in the device-side `u32` ABI.
    #[error("PE count {pe_count} exceeds the device-side u32 ABI")]
    PeCountTooLarge {
        /// Number of known PEs on the host.
        pe_count: usize,
    },

    /// A topology was used with a different number of contexts than it was
    /// built from.
    #[error(
        "topology/context mismatch: expected {expected_pes} contexts, received {actual_contexts}"
    )]
    TopologyMismatch {
        /// Number of PEs recorded in the topology.
        expected_pes: usize,
        /// Number of contexts supplied by the caller.
        actual_contexts: usize,
    },

    /// The requested Rust type has zero size and therefore no meaningful device
    /// footprint inside the symmetric heap.
    #[error("zero-sized types are not supported in the symmetric heap: {type_name}")]
    ZeroSizedType {
        /// Rust type name produced by `core::any::type_name`.
        type_name: &'static str,
    },

    /// `Layout::array::<T>(len)` overflowed.
    #[error("allocation layout overflow for type {type_name} and length {len}")]
    LayoutOverflow {
        /// Rust type name produced by `core::any::type_name`.
        type_name: &'static str,
        /// Number of requested elements.
        len: usize,
    },

    /// The aligned reservation size does not fit in `usize`.
    #[error("total heap size overflow: chunk size {chunk_size} * pe count {pe_count}")]
    SizeOverflow {
        /// Per-PE chunk size after alignment.
        chunk_size: usize,
        /// Number of participating PEs.
        pe_count: usize,
    },

    /// A zero-byte per-PE heap chunk is not meaningful.
    #[error("symmetric heap chunk size must be greater than zero")]
    InvalidChunkSize,

    /// A computed device pointer overflowed `CUdeviceptr`.
    #[error("device pointer arithmetic overflow: base {base:#x} + offset {offset}")]
    AddressOverflow {
        /// Base pointer before offsetting.
        base: u64,
        /// Requested byte offset.
        offset: usize,
    },

    /// The requested offset does not fit inside a per-PE chunk.
    #[error("byte offset {offset} plus size {size} exceeds chunk size {chunk_size}")]
    OutOfBounds {
        /// Starting byte offset within the per-PE chunk.
        offset: usize,
        /// Requested byte size.
        size: usize,
        /// Total bytes available in one PE chunk.
        chunk_size: usize,
    },

    /// The symmetric heap's per-PE chunk does not have enough room for another
    /// allocation.
    #[error(
        "symmetric heap exhausted: requested {requested} bytes, only {remaining} bytes remaining in each PE chunk"
    )]
    HeapExhausted {
        /// Number of bytes requested by the new allocation.
        requested: usize,
        /// Remaining bytes left in the per-PE chunk.
        remaining: usize,
    },
}
