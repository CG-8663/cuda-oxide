/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DisjointSlice - a type-safe abstraction for parallel GPU writes.
//!
//! This module provides `DisjointSlice<T>`, which guarantees that each thread
//! accesses a unique element, preventing data races.
//!
//! # Safety Model
//!
//! Safety is enforced through the type system and bounds checking:
//!
//! 1. **ThreadIndex**: Can only be constructed by `index_1d` / `index_2d`, which
//!    derive the index from hardware built-in variables (`threadIdx`, `blockIdx`,
//!    `blockDim`) -- read-only special registers assigned by the runtime at kernel
//!    launch. The formula combines these into a unique scalar index per thread.
//!    `index_2d` returns `Option<ThreadIndex>`, enforcing `col < stride`.
//!
//! 2. **`get_mut()`**: Bounds-checked access via `ThreadIndex`. Returns
//!    `Option<&mut T>` — `None` for out-of-bounds threads. Sound by default.
//!
//! 3. **`get_unchecked_mut()`**: Unsafe escape hatch for performance-critical
//!    paths where bounds have been validated by other means.
//!
//! The unsafe boundary is pushed to the construction of `DisjointSlice` from raw
//! memory, not to the per-access level.

use crate::thread::ThreadIndex;
use core::marker::PhantomData;

/// A slice-like type that can only be accessed with thread-local indices.
///
/// # Safety Invariants
///
/// The type system enforces these invariants:
/// 1. Default access via `get_mut(ThreadIndex)` is bounds-checked and sound
/// 2. `ThreadIndex` can only be created by `index_1d` / `index_2d`, which derive
///    the index from hardware built-in variables -- read-only special registers
///    assigned by the runtime at launch
/// 3. Each thread's `ThreadIndex` is unique (the hardware guarantees distinct
///    coordinates per thread; the formula preserves that uniqueness)
///
/// These guarantees ensure that each thread accesses a unique element,
/// making parallel writes safe without synchronization.
///
/// # Memory Layout
///
/// Internally, this is identical to a slice: `{ ptr: *mut T, len: usize }`
/// The safety comes from type-level enforcement and bounds checking.
///
/// # Soundness
///
/// `get_mut()` returns `Option<&mut T>`, making out-of-bounds access
/// impossible in safe code. The previous API (`get() -> &mut T`) relied on
/// the caller to check bounds externally; in release builds this was UB for
/// out-of-bounds indices — a soundness hole. The current design follows
/// `slice::get_mut()` / `slice::get_unchecked_mut()` from std: the safe
/// path is always sound, and the unsafe escape hatch (`get_unchecked_mut`)
/// is explicitly opted into.
///
/// The type is `Send` but NOT `Sync`: each GPU thread gets its own copy of
/// the struct (with the same backing pointer), then uses its unique
/// `ThreadIndex` to access a different element. Sharing `&DisjointSlice`
/// across threads is not meaningful.
///
/// # Example
///
/// ```rust
/// use cuda_device::{thread, DisjointSlice};
///
/// #[kernel]
/// pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///     let idx = thread::index_1d();
///     if let Some(c_elem) = c.get_mut(idx) {
///         *c_elem = a[idx.get()] + b[idx.get()];
///     }
/// }
/// ```
#[repr(C)]
pub struct DisjointSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> DisjointSlice<'a, T> {
    /// Create a DisjointSlice from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to valid, aligned memory for `len` elements of type `T`
    /// - The memory will remain valid and not be deallocated for lifetime `'a`
    /// - No other code will access this memory during `'a` (exclusive access)
    /// - The kernel launch configuration ensures threads access disjoint elements
    ///   (i.e., grid dimensions match the data dimensions)
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        DisjointSlice {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Create a DisjointSlice from a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The kernel launch configuration ensures threads access disjoint elements
    /// - No other code accesses the slice during kernel execution
    #[inline]
    pub unsafe fn from_mut_slice(slice: &'a mut [T]) -> Self {
        unsafe { Self::from_raw_parts(slice.as_mut_ptr(), slice.len()) }
    }

    /// Get the length of the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a mutable reference to an element at a thread-local index,
    /// returning `None` if the index is out of bounds.
    ///
    /// This is the default, sound access method. Mirrors `slice::get_mut()`.
    ///
    /// # Safety Argument
    ///
    /// This method is safe (not marked `unsafe`) because:
    ///
    /// 1. **Bounds checked**: Returns `None` for out-of-bounds indices.
    ///
    /// 2. **Unique access**: `ThreadIndex` can only be constructed by `index_1d()`
    ///    or `index_2d()`, which derive the index from hardware built-in variables
    ///    (`threadIdx`, `blockIdx`, `blockDim`) -- read-only special registers
    ///    assigned by the runtime at kernel launch. The formula combines these
    ///    into a single scalar index while preserving the per-thread uniqueness
    ///    that the hardware guarantees.
    ///
    /// 3. **No data races**: Since each thread has a unique `ThreadIndex`, and
    ///    threads cannot share `ThreadIndex` values, each thread accesses a
    ///    different memory location.
    ///
    /// # Example
    ///
    /// ```rust
    /// let idx = thread::index_1d();
    /// if let Some(elem) = c.get_mut(idx) {
    ///     *elem = a[idx.get()] + b[idx.get()];
    /// }
    /// ```
    #[inline]
    pub fn get_mut(&mut self, idx: ThreadIndex) -> Option<&mut T> {
        let i = idx.get();
        if i < self.len {
            // SAFETY:
            // - Bounds check passed above.
            // - `idx` is a ThreadIndex derived from hardware built-in variables,
            //   guaranteeing a unique index per thread (no data races).
            // - The DisjointSlice was constructed with valid memory (from_raw_parts safety).
            Some(unsafe { &mut *self.ptr.add(i) })
        } else {
            None
        }
    }

    /// Get a raw pointer to the underlying data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get a mutable reference to an element at a raw index, without
    /// bounds checking.
    ///
    /// This is an escape hatch for performance-critical paths where bounds
    /// have been validated by other means, such as:
    /// - Warp reductions where only lane 0 writes to a unique warp index
    /// - Histogram updates with atomic operations
    /// - Scatter operations with known-unique destinations
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `idx < self.len()` (bounds are valid)
    /// - No two threads write to the same index simultaneously
    /// - The uniqueness guarantee comes from the algorithm (document it!)
    ///
    /// # Example: Warp Reduction
    ///
    /// ```rust
    /// // SAFETY: Only lane 0 of each warp writes, and warp indices are unique
    /// if warp::lane_id() == 0 {
    ///     let warp_idx = gid.get() / 32;
    ///     unsafe { *out.get_unchecked_mut(warp_idx) = sum; }
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut T {
        debug_assert!(
            idx < self.len,
            "Index out of bounds: {} >= {}",
            idx,
            self.len
        );
        unsafe { &mut *self.ptr.add(idx) }
    }
}

// SAFETY: DisjointSlice can be sent between threads because:
// - Each thread will access unique elements (guaranteed by ThreadIndex)
// - The pointer and length are just data, no thread affinity
// - T: Send means the elements themselves can be sent between threads
unsafe impl<'a, T: Send> Send for DisjointSlice<'a, T> {}

// DisjointSlice auto-trait summary:
//   Send: yes (explicit impl above, when T: Send)
//   Sync: NO (not implemented) — each GPU thread gets its own copy of the
//         struct, then uses its unique ThreadIndex to access a different
//         element. Sharing &DisjointSlice across threads would allow
//         multiple threads to call get_mut() on the same struct, which
//         would produce aliasing &mut T references — unsound.
