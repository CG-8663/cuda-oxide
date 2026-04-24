/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side team metadata.

use super::{PeIndex, SymmetricRef};
use core::marker::PhantomData;

/// Team of participating PEs inside one running kernel.
///
/// This is the device-side view of the Phase 2 symmetric heap. `heap_base`
/// points at the local PE's VA alias, and every remote owner's chunk is reached
/// by `heap_base + owner_pe * chunk_size`.
#[derive(Clone, Copy, Debug)]
pub struct Team<'a> {
    pe: PeIndex,
    pe_count: u32,
    heap_base: *mut u8,
    chunk_size: usize,
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> Team<'a> {
    /// Creates a validated team view.
    pub fn new(pe: u32, pe_count: u32, heap_base: *mut u8, chunk_size: usize) -> Option<Self> {
        let pe = PeIndex::try_new(pe, pe_count)?;
        if heap_base.is_null() || chunk_size == 0 {
            return None;
        }

        Some(Self {
            pe,
            pe_count,
            heap_base,
            chunk_size,
            _marker: PhantomData,
        })
    }

    /// Creates a team view without re-checking its invariants.
    ///
    /// # Safety
    ///
    /// - `pe` must be strictly less than `pe_count`
    /// - `heap_base` must be the base of the local PE's symmetric-heap alias
    /// - `chunk_size` must be the per-PE chunk size used by the host heap
    pub const unsafe fn new_unchecked(
        pe: u32,
        pe_count: u32,
        heap_base: *mut u8,
        chunk_size: usize,
    ) -> Self {
        Self {
            pe: unsafe { PeIndex::new_unchecked(pe) },
            pe_count,
            heap_base,
            chunk_size,
            _marker: PhantomData,
        }
    }

    /// Returns the current PE.
    pub const fn pe(self) -> PeIndex {
        self.pe
    }

    /// Returns the number of participating PEs.
    pub const fn pe_count(self) -> u32 {
        self.pe_count
    }

    /// Returns the local PE's symmetric-heap base alias.
    pub const fn heap_base(self) -> *mut u8 {
        self.heap_base
    }

    /// Returns the per-PE chunk size in bytes.
    pub const fn chunk_size_bytes(self) -> usize {
        self.chunk_size
    }

    /// Returns the PE to the left in a logical ring.
    pub fn left_pe(self) -> PeIndex {
        let current = self.pe.get();
        let prev = if current == 0 {
            self.pe_count - 1
        } else {
            current - 1
        };
        // SAFETY: `prev` is constructed modulo `pe_count`.
        unsafe { PeIndex::new_unchecked(prev) }
    }

    /// Returns the PE to the right in a logical ring.
    pub fn right_pe(self) -> PeIndex {
        let next = (self.pe.get() + 1) % self.pe_count;
        // SAFETY: `next` is constructed modulo `pe_count`.
        unsafe { PeIndex::new_unchecked(next) }
    }

    /// Creates a typed view into one symmetric allocation at `offset_bytes`.
    pub fn symmetric_ref<T>(self, offset_bytes: usize, len: usize) -> Option<SymmetricRef<'a, T>> {
        SymmetricRef::new(self, offset_bytes, len)
    }

    pub(crate) fn chunk_base(self, owner_pe: PeIndex) -> Option<*mut u8> {
        let slot_offset = owner_pe.as_usize().checked_mul(self.chunk_size)?;
        let addr = (self.heap_base as usize).checked_add(slot_offset)?;
        Some(addr as *mut u8)
    }
}

#[cfg(test)]
mod tests {
    use super::Team;

    #[test]
    fn ring_neighbors_wrap_correctly() {
        let team = Team::new(0, 4, 0x1000usize as *mut u8, 0x80).unwrap();
        assert_eq!(team.left_pe().get(), 3);
        assert_eq!(team.right_pe().get(), 1);

        let team = Team::new(3, 4, 0x1000usize as *mut u8, 0x80).unwrap();
        assert_eq!(team.left_pe().get(), 2);
        assert_eq!(team.right_pe().get(), 0);
    }
}
