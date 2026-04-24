/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed pointer arithmetic and remote put/get helpers on the symmetric heap.

use super::{PeIndex, Team, signal};
use core::marker::PhantomData;
use core::mem::size_of;
use cuda_device::atomic::SystemAtomicU32;

/// Typed view into one symmetric allocation at a fixed offset in every PE's
/// chunk.
///
/// `SymmetricRef<T>` stores only:
///
/// - the local [`Team`] view,
/// - the shared byte offset inside every chunk,
/// - and the logical element count.
///
/// Local and remote pointers are derived on demand from those three pieces.
#[derive(Debug)]
pub struct SymmetricRef<'a, T> {
    team: Team<'a>,
    offset_bytes: usize,
    len: usize,
    _marker: PhantomData<*mut T>,
}

impl<T> Copy for SymmetricRef<'_, T> {}

impl<T> Clone for SymmetricRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> SymmetricRef<'a, T> {
    /// Creates a typed symmetric reference if the allocation fits inside one
    /// owner chunk.
    pub fn new(team: Team<'a>, offset_bytes: usize, len: usize) -> Option<Self> {
        let elem_size = size_of::<T>();
        if elem_size == 0 {
            return None;
        }

        let byte_len = len.checked_mul(elem_size)?;
        let end = offset_bytes.checked_add(byte_len)?;
        if end > team.chunk_size_bytes() {
            return None;
        }

        Some(Self {
            team,
            offset_bytes,
            len,
            _marker: PhantomData,
        })
    }

    /// Returns the team this view is attached to.
    pub const fn team(self) -> Team<'a> {
        self.team
    }

    /// Returns the number of logical `T` elements.
    pub const fn len(self) -> usize {
        self.len
    }

    /// Returns `true` when there are no logical elements.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Returns the shared byte offset inside every owner chunk.
    pub const fn offset_bytes(self) -> usize {
        self.offset_bytes
    }

    /// Returns a pointer to element `index` in `owner_pe`'s chunk.
    pub fn ptr_at(self, owner_pe: PeIndex, index: usize) -> Option<*mut T> {
        if index >= self.len {
            return None;
        }

        let elem_offset = index.checked_mul(size_of::<T>())?;
        let byte_offset = self.offset_bytes.checked_add(elem_offset)?;
        let base = self.team.chunk_base(owner_pe)?;
        let addr = (base as usize).checked_add(byte_offset)?;
        Some(addr as *mut T)
    }

    /// Returns a pointer to the first element in `owner_pe`'s chunk.
    pub fn ptr_for(self, owner_pe: PeIndex) -> Option<*mut T> {
        self.ptr_at(owner_pe, 0)
    }

    /// Returns a pointer to the first local element.
    pub fn local_ptr(self) -> *mut T {
        self.ptr_for(self.team.pe())
            .expect("validated symmetric ref should have an in-bounds local pointer")
    }

    /// Returns a pointer to the first element in `target_pe`'s chunk.
    pub fn remote_ptr(self, target_pe: PeIndex) -> *mut T {
        self.ptr_for(target_pe)
            .expect("validated symmetric ref should have an in-bounds remote pointer")
    }
}

impl<'a, T: Copy> SymmetricRef<'a, T> {
    /// Stores `*src` into the first element of `target_pe`'s chunk.
    ///
    /// # Safety
    ///
    /// The pointed-to remote slot must be exclusively owned by the caller's
    /// higher-level protocol logic. This method performs plain global-memory
    /// stores; it does not add synchronization on its own.
    pub unsafe fn put(self, src: &T, target_pe: PeIndex) {
        unsafe {
            self.put_at(src, target_pe, 0);
        }
    }

    /// Stores `*src` into element `index` of `target_pe`'s chunk.
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`put`](Self::put), plus `index` must be in
    /// bounds.
    pub unsafe fn put_at(self, src: &T, target_pe: PeIndex, index: usize) {
        let dst = self
            .ptr_at(target_pe, index)
            .expect("put_at requires an in-bounds symmetric element");
        unsafe {
            core::ptr::write(dst, *src);
        }
    }

    /// Loads the first element from `target_pe`'s chunk.
    ///
    /// # Safety
    ///
    /// The caller must ensure the remote value has been fully published before
    /// loading it, typically via [`signal::wait_until_system`].
    pub unsafe fn get(self, target_pe: PeIndex) -> T {
        unsafe { self.get_at(target_pe, 0) }
    }

    /// Loads element `index` from `target_pe`'s chunk.
    ///
    /// # Safety
    ///
    /// Same safety requirements as [`get`](Self::get), plus `index` must be in
    /// bounds.
    pub unsafe fn get_at(self, target_pe: PeIndex, index: usize) -> T {
        let src = self
            .ptr_at(target_pe, index)
            .expect("get_at requires an in-bounds symmetric element");
        unsafe { core::ptr::read(src) }
    }

    /// Stores `*src` to `target_pe` and then publishes `signal_value` through
    /// `signal_flag` with system scope.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `signal_flag` points at a valid
    /// system-scope atomic slot associated with the same publication protocol as
    /// this payload.
    pub unsafe fn put_signal(
        self,
        src: &T,
        target_pe: PeIndex,
        signal_flag: &SystemAtomicU32,
        signal_value: u32,
    ) {
        unsafe {
            self.put(src, target_pe);
            signal::publish_system(signal_flag, signal_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SymmetricRef;
    use crate::device::Team;

    #[test]
    fn symmetric_ref_pointer_arithmetic_matches_chunk_layout() {
        let team = Team::new(1, 3, 0x1000usize as *mut u8, 0x100).unwrap();
        let reference = SymmetricRef::<u32>::new(team, 0x20, 4).unwrap();

        assert_eq!(reference.local_ptr() as usize, 0x1120);
        assert_eq!(reference.ptr_for(team.left_pe()).unwrap() as usize, 0x1020);
        assert_eq!(reference.ptr_for(team.right_pe()).unwrap() as usize, 0x1220);
        assert_eq!(
            reference.ptr_at(team.right_pe(), 2).unwrap() as usize,
            0x1228
        );
    }

    #[test]
    fn symmetric_ref_rejects_out_of_bounds_layouts() {
        let team = Team::new(0, 2, 0x2000usize as *mut u8, 0x40).unwrap();
        assert!(SymmetricRef::<u32>::new(team, 0x3c, 2).is_none());
    }
}
