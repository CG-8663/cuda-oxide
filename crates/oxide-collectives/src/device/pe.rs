/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side PE indices.

/// Index of one participating PE inside a launched team.
///
/// Like `cuda_device::thread::ThreadIndex`, this is a lightweight newtype used
/// to distinguish validated PE identities from arbitrary integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PeIndex(u32);

impl PeIndex {
    /// Creates a validated PE index if `raw < pe_count`.
    pub const fn try_new(raw: u32, pe_count: u32) -> Option<Self> {
        if raw < pe_count {
            Some(Self(raw))
        } else {
            None
        }
    }

    /// Creates a PE index without re-checking bounds.
    ///
    /// # Safety
    ///
    /// `raw` must be strictly less than the team's `pe_count`.
    pub const unsafe fn new_unchecked(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw PE number.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Returns the PE number as `usize`.
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::PeIndex;

    #[test]
    fn pe_index_validates_bounds() {
        assert_eq!(PeIndex::try_new(0, 2).unwrap().get(), 0);
        assert_eq!(PeIndex::try_new(1, 2).unwrap().get(), 1);
        assert!(PeIndex::try_new(2, 2).is_none());
    }
}
