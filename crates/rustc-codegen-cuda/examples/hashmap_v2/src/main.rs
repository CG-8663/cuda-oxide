/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Hashmap v2 — SwissTable-Inspired
//!
//! Layered on top of the v1 baseline. v2 introduces:
//!   - A separate control-byte array (`ctrl: DeviceBuffer<u32>` packing 4
//!     1-byte tags per word) so probe walks examine fingerprints, not the
//!     full `(key, value)` payload.
//!   - hashbrown's h1/h2 hash split — h1 picks the probe position, h2 is a
//!     7-bit per-slot fingerprint stored in the tag.
//!   - Triangular probing in 32-tag-byte tiles (`PROBE_TILE`). Insert,
//!     find, and delete all walk the same triangular sequence so any
//!     key insert places is always reachable by find.
//!   - Tombstone delete (`FULL(h2)` -> `DELETED (0x80)` via `u32` CAS).
//!   - Warp-cooperative find (`find_kernel_warp`) — one warp per key,
//!     32 tag bytes inspected in parallel via `warp::ballot`.
//!
//! Insert protocol shipped: **Protocol B (payload-first)** — one
//! `DeviceAtomicU64::compare_exchange` on the slot followed by one
//! `DeviceAtomicU32::compare_exchange` on the ctrl word. The slot CAS
//! itself acts as the serialization point: concurrent inserts of the
//! same key always see each other via `Err(actual)` and degenerate into
//! the duplicate-handling path. Protocol A (RESERVED-tag handshake)
//! lands in cut 2 alongside the bench harness.
//!
//! Build and run with:
//!   cargo oxide run hashmap_v2

use std::sync::Arc;

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_launch;

// =============================================================================
// SHARED CONSTANTS AND HELPERS (compiled both host- and device-side)
// =============================================================================

/// Number of tag bytes packed into one `ctrl` word. Fixed at 4 because
/// cuda-oxide has no 8-bit atomics: a single 32-bit CAS is the smallest
/// instruction that can transition one tag while leaving the other three
/// untouched. This is a *storage* constant — it does NOT determine the
/// probe-step width.
const GROUP: usize = 4;

/// Probe-step width in tag bytes. Both single-thread and warp-cooperative
/// kernels examine `PROBE_TILE` consecutive tag bytes per probe step and
/// advance triangularly in `PROBE_TILE` units. Fixed at 32 because:
///   - The warp-cooperative kernel uses 32 lanes (one tag byte per lane).
///   - cuda-oxide's `warp::ballot` and `warp::shuffle` operate on the
///     full 32-lane warp; no sub-warp mask is exposed today.
///   - Insert and find MUST use the same `PROBE_TILE` so they walk the
///     same triangular sequence — otherwise find can terminate early
///     on an `EMPTY` slot that insert had skipped, missing valid keys.
///
/// `PROBE_TILE` must be a multiple of `GROUP` (one ctrl word covers
/// `GROUP` tag bytes; we read `PROBE_TILE / GROUP` ctrl words per step).
const PROBE_TILE: usize = 32;

/// Tag byte = "this slot is free". All slots start as `EMPTY_TAG`. The
/// initial all-`0xFF` ctrl array gives us this for free via
/// `memset_d8_async(0xFF, ...)`.
const EMPTY_TAG: u8 = 0xFF;

/// Tag byte = "this slot was once occupied; do not stop probing here, but
/// also do not treat it as live". Insert in v2 cut 1 does **not** reclaim
/// these (no `DELETED -> FULL` CAS path); they linger until v3's rehash.
const DELETED_TAG: u8 = 0x80;

/// Tag byte for an occupied slot is `FULL(h2)` — the top bit is clear and
/// the low seven bits are `h2`, the high-byte fingerprint of the key's
/// hash. h2 thus lives in `[0x00, 0x7F]` and can never collide with
/// `EMPTY_TAG (0xFF)` or `DELETED_TAG (0x80)`.
///
/// Format:
/// ```text
///   bit:   7   6   5   4   3   2   1   0
///        +---+---+---+---+---+---+---+---+
/// EMPTY  | 1   1   1   1   1   1   1   1 |   0xFF
/// DELETED| 1   0   0   0   0   0   0   0 |   0x80
/// FULL   | 0 |       h2 (7 bits)         |   0x00..0x7F
///        +---+---+---+---+---+---+---+---+
/// ```
///
/// Helper to build a `FULL(h2)` byte; the input is already 7-bit so this is
/// a no-op, but we name it to make intent obvious in the kernel.
#[inline(always)]
fn full_tag(h2: u8) -> u8 {
    h2
}

/// Slot sentinel for "this slot is unclaimed". Same value as v1's `EMPTY`
/// — a `u64::MAX` packed pair `(u32::MAX, u32::MAX)`. Initial `slots`
/// buffer is `memset` to all-`0xFF` bytes so every slot reads as this.
const EMPTY_SLOT: u64 = u64::MAX;

/// Sentinel returned by `find_bulk` for missing keys. Same as v1.
const MISS: u32 = u32::MAX;

/// Per-key flag value for "this key was already present" (`try_insert_bulk`)
/// or "this key was not in the table" (`delete_bulk`). The host narrows this
/// to a `bool` before returning.
const FLAG_PRESENT: u32 = 1;
/// Per-key flag value for "fresh insert" / "successful delete".
const FLAG_FRESH_OR_OK: u32 = 0;

/// FxHash multiplier — same constant as v1.
const FX_K: u64 = 0x517c_c1b7_2722_0a95;

/// FxHash-style single-multiply hash. Returns 64 bits so we can split into
/// h1 (low bits, probe position) and h2 (top 7 bits, fingerprint) without
/// a second hash call.
#[inline(always)]
fn hash_u32(key: u32) -> u64 {
    (key as u64).wrapping_mul(FX_K)
}

/// Extract the 7-bit fingerprint stored in the FULL tag. We pull from the
/// **top** of the hash so it's statistically independent of the low-bit
/// position — the same split hashbrown uses (`raw.rs:60`).
#[inline(always)]
fn h2_from_hash(hash: u64) -> u8 {
    ((hash >> 57) & 0x7F) as u8
}

/// Extract the tag byte at index `i` (0..4) from a packed ctrl word.
#[inline(always)]
fn get_tag(word: u32, i: usize) -> u8 {
    ((word >> (8 * i)) & 0xFF) as u8
}

/// Replace the tag byte at index `i` (0..4) inside a packed ctrl word.
/// Returns the new word; the other three bytes are preserved.
#[inline(always)]
fn set_tag(word: u32, i: usize, tag: u8) -> u32 {
    let shift = 8 * i;
    (word & !(0xFFu32 << shift)) | ((tag as u32) << shift)
}

/// Pack `(key, value)` into a single `u64` slot. Same layout as v1.
#[inline(always)]
fn pack(key: u32, value: u32) -> u64 {
    ((key as u64) << 32) | (value as u64)
}

/// Recover the key from a packed slot (upper 32 bits).
#[inline(always)]
fn unpack_key(slot: u64) -> u32 {
    (slot >> 32) as u32
}

/// Recover the value from a packed slot (lower 32 bits).
#[inline(always)]
fn unpack_value(slot: u64) -> u32 {
    (slot & 0xFFFF_FFFF) as u32
}

// =============================================================================
// KERNELS
// =============================================================================

/// `insert_kernel` — last-writer-wins, one thread per input key.
///
/// Storage:
///   - `ctrl[g]` is the `u32` packing tags for slots `g*GROUP .. g*GROUP+GROUP`.
///   - `slots[s]` is the packed `(key, value)` for slot `s`.
///
/// Probe shape: triangular in `PROBE_TILE`-byte tiles. Each step
/// examines `PROBE_TILE` consecutive tag bytes (= `PROBE_TILE / GROUP`
/// ctrl words) starting at `probe_base`, and advances by `stride *
/// PROBE_TILE` (with `stride += 1` per step). All four kernels — and
/// the warp-cooperative find — share this exact shape so they walk
/// the same sequence and EMPTY-termination remains correct.
///
/// Insert protocol (Protocol B, payload-first):
///   1. Phase 1 — walk the tile ctrl word by ctrl word looking for any
///      `FULL(h2)` tag whose slot holds our key. On match, slot-CAS
///      overwrite the value (last-writer-wins) and return.
///   2. Phase 2 — re-walk the tile looking for any `EMPTY_TAG` byte.
///      For each one, try the slot CAS `EMPTY_SLOT -> pack(k, v)`.
///      The slot CAS is the serialization point: concurrent inserts of
///      the same key see `Err(actual)` with a matching key and
///      degenerate to the overwrite path; concurrent inserts of
///      *different* keys see `Err(actual)` with a mismatched key and
///      skip past.
///   3. After a successful slot CAS, publish via a ctrl-word CAS retry
///      loop: `set_tag(current_word, j, FULL(h2))` on the specific
///      ctrl word containing byte `j`. Other bytes in that word may
///      have changed concurrently, so the loop re-reads on failure;
///      byte `j` itself cannot change under us because no other thread
///      can claim a slot we already own.
///   4. No FULL(h2) match anywhere in the tile and no EMPTY_TAG to
///      claim → triangular advance and repeat.
#[kernel]
pub fn insert_kernel(ctrl: &[u32], slots: &[u64], keys: &[u32], values: &[u32]) {
    let tid = thread::index_1d().get();
    if tid >= keys.len() {
        return;
    }

    let key = keys[tid];
    let value = values[tid];
    let hash = hash_u32(key);
    let h2 = h2_from_hash(hash);
    let mask = slots.len() - 1;
    let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
    let mut stride = 0usize;

    loop {
        // Phase 1: walk the entire 32-byte tile, ctrl word by ctrl word,
        // checking for an already-published FULL(h2) entry holding our key.
        let mut g = 0usize;
        while g < PROBE_TILE {
            let ctrl_word_idx = (probe_base + g) / GROUP;
            let ctrl_atomic =
                unsafe { DeviceAtomicU32::from_ptr(ctrl.as_ptr().add(ctrl_word_idx).cast_mut()) };
            let word = ctrl_atomic.load(AtomicOrdering::Acquire);
            let mut j = 0;
            while j < GROUP {
                if get_tag(word, j) == h2 {
                    let slot_idx = probe_base + g + j;
                    let slot_atomic = unsafe {
                        DeviceAtomicU64::from_ptr(slots.as_ptr().add(slot_idx).cast_mut())
                    };
                    let observed = slot_atomic.load(AtomicOrdering::Acquire);
                    if unpack_key(observed) == key {
                        insert_overwrite(slot_atomic, observed, key, value);
                        return;
                    }
                }
                j += 1;
            }
            g += GROUP;
        }

        // Phase 2: try to claim an EMPTY tag anywhere in this tile.
        let mut g = 0usize;
        while g < PROBE_TILE {
            let ctrl_word_idx = (probe_base + g) / GROUP;
            let ctrl_atomic =
                unsafe { DeviceAtomicU32::from_ptr(ctrl.as_ptr().add(ctrl_word_idx).cast_mut()) };
            // Re-read this word: another thread may have mutated it
            // between Phase 1 and now.
            let word = ctrl_atomic.load(AtomicOrdering::Acquire);
            let mut j = 0;
            while j < GROUP {
                if get_tag(word, j) == EMPTY_TAG {
                    let slot_idx = probe_base + g + j;
                    let slot_atomic = unsafe {
                        DeviceAtomicU64::from_ptr(slots.as_ptr().add(slot_idx).cast_mut())
                    };
                    match slot_atomic.compare_exchange(
                        EMPTY_SLOT,
                        pack(key, value),
                        AtomicOrdering::AcqRel,
                        AtomicOrdering::Relaxed,
                    ) {
                        Ok(_) => {
                            publish_full_tag(ctrl_atomic, word, j, h2);
                            return;
                        }
                        Err(actual) => {
                            if unpack_key(actual) == key {
                                insert_overwrite(slot_atomic, actual, key, value);
                                return;
                            }
                            // Different key already in this slot; skip.
                        }
                    }
                }
                j += 1;
            }
            g += GROUP;
        }

        // No FULL(h2) match and no claimable EMPTY in this tile: advance.
        stride += 1;
        probe_base = (probe_base + stride * PROBE_TILE) & mask;
    }
}

/// `try_insert_kernel` — first-writer-wins variant.
///
/// Same probe / claim shape as `insert_kernel`, but on duplicate it leaves
/// the existing slot untouched and writes per-thread output:
///   `out[tid] = FLAG_FRESH_OR_OK (0)`  -> we claimed a fresh slot
///   `out[tid] = FLAG_PRESENT (1)`      -> key was already in the table
#[kernel]
pub fn try_insert_kernel(
    ctrl: &[u32],
    slots: &[u64],
    keys: &[u32],
    values: &[u32],
    mut out: DisjointSlice<u32>,
) {
    let tid = thread::index_1d();
    let i_thread = tid.get();
    if i_thread >= keys.len() {
        return;
    }

    let key = keys[i_thread];
    let value = values[i_thread];
    let hash = hash_u32(key);
    let h2 = h2_from_hash(hash);
    let mask = slots.len() - 1;
    let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
    let mut stride = 0usize;

    loop {
        // Phase 1: published-FULL duplicate detection across the tile.
        let mut g = 0usize;
        while g < PROBE_TILE {
            let ctrl_word_idx = (probe_base + g) / GROUP;
            let ctrl_atomic =
                unsafe { DeviceAtomicU32::from_ptr(ctrl.as_ptr().add(ctrl_word_idx).cast_mut()) };
            let word = ctrl_atomic.load(AtomicOrdering::Acquire);
            let mut j = 0;
            while j < GROUP {
                if get_tag(word, j) == h2 {
                    let slot_idx = probe_base + g + j;
                    let slot_atomic = unsafe {
                        DeviceAtomicU64::from_ptr(slots.as_ptr().add(slot_idx).cast_mut())
                    };
                    let observed = slot_atomic.load(AtomicOrdering::Acquire);
                    if unpack_key(observed) == key {
                        if let Some(o) = out.get_mut(tid) {
                            *o = FLAG_PRESENT;
                        }
                        return;
                    }
                }
                j += 1;
            }
            g += GROUP;
        }

        // Phase 2: claim an EMPTY slot.
        let mut g = 0usize;
        while g < PROBE_TILE {
            let ctrl_word_idx = (probe_base + g) / GROUP;
            let ctrl_atomic =
                unsafe { DeviceAtomicU32::from_ptr(ctrl.as_ptr().add(ctrl_word_idx).cast_mut()) };
            let word = ctrl_atomic.load(AtomicOrdering::Acquire);
            let mut j = 0;
            while j < GROUP {
                if get_tag(word, j) == EMPTY_TAG {
                    let slot_idx = probe_base + g + j;
                    let slot_atomic = unsafe {
                        DeviceAtomicU64::from_ptr(slots.as_ptr().add(slot_idx).cast_mut())
                    };
                    match slot_atomic.compare_exchange(
                        EMPTY_SLOT,
                        pack(key, value),
                        AtomicOrdering::AcqRel,
                        AtomicOrdering::Relaxed,
                    ) {
                        Ok(_) => {
                            publish_full_tag(ctrl_atomic, word, j, h2);
                            if let Some(o) = out.get_mut(tid) {
                                *o = FLAG_FRESH_OR_OK;
                            }
                            return;
                        }
                        Err(actual) => {
                            if unpack_key(actual) == key {
                                // Same-key race; some other thread is the
                                // first-writer. Report PRESENT, leave slot.
                                if let Some(o) = out.get_mut(tid) {
                                    *o = FLAG_PRESENT;
                                }
                                return;
                            }
                        }
                    }
                }
                j += 1;
            }
            g += GROUP;
        }

        stride += 1;
        probe_base = (probe_base + stride * PROBE_TILE) & mask;
    }
}

/// `find_kernel` — single-thread find, one thread per key.
///
/// Walks the same triangular probe sequence as the insert kernels (same
/// `PROBE_TILE = 32` width), so EMPTY-termination is sound — see
/// `find_kernel_warp` below for why probe-width coherence matters.
///
/// At each tile (32 consecutive tag bytes):
///   - For every byte tagged `FULL(h2)` matching our key's fingerprint,
///     load the slot and key-compare; on match return the value.
///   - If any byte in the tile is `EMPTY_TAG`, the key cannot live past
///     this point in its triangular chain (insert would have stopped
///     at this same EMPTY), so return `MISS`.
///   - Otherwise (tile holds only FULL-mismatch + DELETED), triangular
///     advance and repeat. DELETED never terminates find.
#[kernel]
pub fn find_kernel(ctrl: &[u32], slots: &[u64], keys: &[u32], mut out: DisjointSlice<u32>) {
    let tid = thread::index_1d();
    let i_thread = tid.get();
    if i_thread >= keys.len() {
        return;
    }

    let key = keys[i_thread];
    let hash = hash_u32(key);
    let h2 = h2_from_hash(hash);
    let mask = slots.len() - 1;
    let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
    let mut stride = 0usize;

    loop {
        let mut g = 0usize;
        let mut has_empty = false;
        while g < PROBE_TILE {
            let ctrl_word_idx = (probe_base + g) / GROUP;
            let ctrl_atomic =
                unsafe { DeviceAtomicU32::from_ptr(ctrl.as_ptr().add(ctrl_word_idx).cast_mut()) };
            let word = ctrl_atomic.load(AtomicOrdering::Acquire);
            let mut j = 0;
            while j < GROUP {
                let tag = get_tag(word, j);
                if tag == h2 {
                    let slot_idx = probe_base + g + j;
                    let slot_atomic = unsafe {
                        DeviceAtomicU64::from_ptr(slots.as_ptr().add(slot_idx).cast_mut())
                    };
                    let observed = slot_atomic.load(AtomicOrdering::Acquire);
                    if unpack_key(observed) == key {
                        if let Some(o) = out.get_mut(tid) {
                            *o = unpack_value(observed);
                        }
                        return;
                    }
                } else if tag == EMPTY_TAG {
                    has_empty = true;
                }
                j += 1;
            }
            g += GROUP;
        }

        if has_empty {
            if let Some(o) = out.get_mut(tid) {
                *o = MISS;
            }
            return;
        }

        stride += 1;
        probe_base = (probe_base + stride * PROBE_TILE) & mask;
    }
}

/// `find_kernel_warp` — warp-cooperative find, one warp (32 lanes) per key.
///
/// Each probe step pulls 32 tag bytes (= `PROBE_TILE`) into the warp via a
/// single coalesced ctrl load: lanes 0..3 share `ctrl[probe_base/4 + 0]`,
/// lanes 4..7 share `+1`, ..., lanes 28..31 share `+7`. Each lane then
/// extracts its byte and the warp uses ballot/shuffle to:
///   1. `m_h2 = ballot(tag == h2)` — a 32-bit fingerprint match mask.
///   2. For each set bit in `m_h2` (lowest first), the matching lane
///      loads its slot, broadcasts the packed `(key, value)` via two
///      `shuffle`s (one per `u32` half), and all lanes key-compare.
///      On hit, lane 0 writes `out[warp_idx] = value` and the warp returns.
///   3. `m_empty = ballot(tag == EMPTY_TAG)` — if non-zero, the key
///      cannot live past an EMPTY in this hash chain; lane 0 writes MISS.
///   4. Otherwise, triangular advance `probe_base` by `stride * PROBE_TILE`
///      and repeat.
///
/// Launch with `blockDim.x` a multiple of 32; the host driver does
/// `LaunchConfig::for_num_elems(keys.len() * 32)` so each warp receives
/// exactly one key. DELETED tags are skipped (they don't terminate the
/// probe), same as the single-thread `find_kernel`.
#[kernel]
pub fn find_kernel_warp(ctrl: &[u32], slots: &[u64], keys: &[u32], mut out: DisjointSlice<u32>) {
    let lane = warp::lane_id();
    let global_tid = thread::index_1d().get();
    let warp_idx = global_tid / PROBE_TILE;
    if warp_idx >= keys.len() {
        return;
    }

    let key = keys[warp_idx];
    let hash = hash_u32(key);
    let h2 = h2_from_hash(hash);
    let mask = slots.len() - 1;
    let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
    let mut stride = 0usize;

    loop {
        // Each lane owns exactly one tag byte at (probe_base + lane).
        let tag_pos = probe_base + (lane as usize);
        let ctrl_word_idx = tag_pos / GROUP;
        let byte_in_word = tag_pos % GROUP;
        let word = unsafe {
            DeviceAtomicU32::from_ptr(ctrl.as_ptr().add(ctrl_word_idx).cast_mut())
                .load(AtomicOrdering::Acquire)
        };
        let tag: u8 = ((word >> (8 * byte_in_word)) & 0xFF) as u8;

        let mut m_h2 = warp::ballot(tag == h2);
        let m_empty = warp::ballot(tag == EMPTY_TAG);

        // Walk h2 candidates lowest-bit-first.
        while m_h2 != 0 {
            let cand = m_h2.trailing_zeros();
            // Only the candidate lane materializes the slot; others feed
            // a dummy value into shuffle. The shuffle's source is `cand`
            // so all lanes converge on the candidate's slot value.
            let local_slot: u64 = if lane == cand {
                let slot_idx = probe_base + (cand as usize);
                unsafe {
                    DeviceAtomicU64::from_ptr(slots.as_ptr().add(slot_idx).cast_mut())
                        .load(AtomicOrdering::Acquire)
                }
            } else {
                0
            };
            // cuda-oxide's warp::shuffle is u32-only, so broadcast the
            // packed slot in two halves and reassemble.
            let lo = warp::shuffle(local_slot as u32, cand);
            let hi = warp::shuffle((local_slot >> 32) as u32, cand);
            let observed: u64 = ((hi as u64) << 32) | (lo as u64);

            if unpack_key(observed) == key {
                if lane == 0 {
                    // SAFETY: warp_idx < keys.len() == out.len(), and
                    // each warp has a unique warp_idx so writes by lane
                    // 0 across warps are disjoint.
                    unsafe {
                        *out.get_unchecked_mut(warp_idx) = unpack_value(observed);
                    }
                }
                return;
            }
            m_h2 &= m_h2 - 1;
        }

        if m_empty != 0 {
            if lane == 0 {
                // SAFETY: same uniqueness argument as above.
                unsafe {
                    *out.get_unchecked_mut(warp_idx) = MISS;
                }
            }
            return;
        }

        // Triangular advance in tile units. Both operands are
        // PROBE_TILE-aligned, so the sum stays aligned and `& mask`
        // keeps it in [0, N).
        stride += 1;
        probe_base = (probe_base + stride * PROBE_TILE) & mask;
    }
}

/// `delete_kernel` — tombstone the slot for each input key.
///
/// One thread per key. Probes the same `PROBE_TILE`-wide triangular
/// sequence as insert and find. When it locates the key it CAS-flips
/// the byte in the containing ctrl word from `FULL(h2)` to
/// `DELETED_TAG`. The `(key, value)` payload is **not** cleared:
/// readers only ever materialize slots whose tag is `FULL(h2)`, so a
/// stale slot under a `DELETED` tag is unreachable.
///
/// The CAS targets the specific ctrl word containing the matching tag
/// byte (not "the group's word" — there are now `PROBE_TILE / GROUP`
/// ctrl words per tile). On CAS failure (some other thread mutated
/// this word), the inner loop re-reads and re-scans this word before
/// moving on to the next word in the tile.
///
/// Output:
///   `out[tid] = FLAG_FRESH_OR_OK (0)` -> deleted successfully
///   `out[tid] = FLAG_PRESENT (1)`     -> key was not in the table
#[kernel]
pub fn delete_kernel(ctrl: &[u32], slots: &[u64], keys: &[u32], mut out: DisjointSlice<u32>) {
    let tid = thread::index_1d();
    let i_thread = tid.get();
    if i_thread >= keys.len() {
        return;
    }

    let key = keys[i_thread];
    let hash = hash_u32(key);
    let h2 = h2_from_hash(hash);
    let mask = slots.len() - 1;
    let mut probe_base = (hash as usize) & mask & !(PROBE_TILE - 1);
    let mut stride = 0usize;

    'tile: loop {
        let mut has_empty = false;
        let mut g = 0usize;
        while g < PROBE_TILE {
            let ctrl_word_idx = (probe_base + g) / GROUP;
            let ctrl_atomic =
                unsafe { DeviceAtomicU32::from_ptr(ctrl.as_ptr().add(ctrl_word_idx).cast_mut()) };

            // Per-word retry loop: if our CAS to flip a tag to DELETED
            // fails because someone else mutated this same word, re-read
            // and re-scan this word.
            loop {
                let word = ctrl_atomic.load(AtomicOrdering::Acquire);
                let mut j = 0;
                let mut cas_collided = false;
                while j < GROUP {
                    let tag = get_tag(word, j);
                    if tag == h2 {
                        let slot_idx = probe_base + g + j;
                        let slot_atomic = unsafe {
                            DeviceAtomicU64::from_ptr(slots.as_ptr().add(slot_idx).cast_mut())
                        };
                        let observed = slot_atomic.load(AtomicOrdering::Acquire);
                        if unpack_key(observed) == key {
                            let new_word = set_tag(word, j, DELETED_TAG);
                            match ctrl_atomic.compare_exchange(
                                word,
                                new_word,
                                AtomicOrdering::AcqRel,
                                AtomicOrdering::Relaxed,
                            ) {
                                Ok(_) => {
                                    if let Some(o) = out.get_mut(tid) {
                                        *o = FLAG_FRESH_OR_OK;
                                    }
                                    return;
                                }
                                Err(_) => {
                                    cas_collided = true;
                                    break;
                                }
                            }
                        }
                    } else if tag == EMPTY_TAG {
                        has_empty = true;
                    }
                    j += 1;
                }
                if !cas_collided {
                    break; // word fully scanned; move to next word in the tile
                }
                // else: retry this same word with a freshly-loaded view.
            }
            g += GROUP;
        }

        if has_empty {
            if let Some(o) = out.get_mut(tid) {
                *o = FLAG_PRESENT;
            }
            return;
        }

        stride += 1;
        probe_base = (probe_base + stride * PROBE_TILE) & mask;
        continue 'tile;
    }
}

// =============================================================================
// DEVICE-SIDE HELPERS
// =============================================================================

/// Inner CAS loop that overwrites a slot's value while preserving its key.
/// Used by both `insert_kernel` and `try_insert_kernel`'s last-writer-wins
/// branches, and only ever entered when the slot already holds our key.
#[inline(always)]
fn insert_overwrite(slot_atomic: &DeviceAtomicU64, mut expected: u64, key: u32, value: u32) {
    let desired = pack(key, value);
    loop {
        match slot_atomic.compare_exchange(
            expected,
            desired,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => return,
            // Someone else's overwrite landed; re-read and retry so the
            // last-writer-wins guarantee survives concurrent duplicates.
            Err(actual) => expected = actual,
        }
    }
}

/// Publish a tag byte from `EMPTY_TAG` to `FULL(h2)` via a ctrl-word CAS
/// retry loop. The slot at byte `i` is already ours (we won its slot CAS),
/// so byte `i` can't change under us — the only reason this CAS fails is
/// concurrent mutation of a different byte in the same word, in which case
/// we re-read and rebuild the new word.
#[inline(always)]
fn publish_full_tag(ctrl_atomic: &DeviceAtomicU32, mut current_word: u32, i: usize, h2: u8) {
    loop {
        let new_word = set_tag(current_word, i, full_tag(h2));
        match ctrl_atomic.compare_exchange(
            current_word,
            new_word,
            AtomicOrdering::Release,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual) => current_word = actual,
        }
    }
}

// =============================================================================
// HOST DRIVER
// =============================================================================

/// Forbidden user key. `(FORBIDDEN_KEY, _)` and `(_, _)` for any value
/// produce an `EMPTY_SLOT` collision when the value also happens to be
/// `u32::MAX`; the simplest invariant is to forbid `u32::MAX` as a key
/// outright (matches v1).
const FORBIDDEN_KEY: u32 = u32::MAX;

/// Host-side handle to a v2 SwissTable-style GPU hashmap.
///
/// Owns two device-resident buffers:
///   - `ctrl`: `DeviceBuffer<u32>` of length `capacity / GROUP`. Each `u32`
///     holds 4 tag bytes packed little-endian.
///   - `slots`: `DeviceBuffer<u64>` of length `capacity`. Each `u64` packs
///     a `(key, value)` pair, same layout as v1.
///
/// Both buffers are `memset_d8_async(0xFF)` at construction so every tag
/// reads as `EMPTY_TAG` and every slot reads as `EMPTY_SLOT`.
struct GpuSwissMap {
    /// Packed tag bytes; 4 tags per `u32` word.
    ctrl: DeviceBuffer<u32>,
    /// Packed `(key, value)` slots, same layout as v1.
    slots: DeviceBuffer<u64>,
    /// Number of slots. Power of two, multiple of `GROUP`.
    capacity: usize,
}

impl GpuSwissMap {
    /// Allocate a fresh, empty table of `capacity` slots. `capacity` must
    /// be a non-zero power of two and at least `GROUP` (so the ctrl array
    /// has at least one word).
    fn new(capacity: usize, stream: &Arc<CudaStream>) -> Result<Self, cuda_core::DriverError> {
        assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two"
        );
        assert!(
            capacity >= PROBE_TILE,
            "capacity must be >= PROBE_TILE ({PROBE_TILE})"
        );

        let ctrl = DeviceBuffer::<u32>::zeroed(stream, capacity / GROUP)?;
        let slots = DeviceBuffer::<u64>::zeroed(stream, capacity)?;
        unsafe {
            cuda_core::memory::memset_d8_async(
                ctrl.cu_deviceptr(),
                0xFF,
                ctrl.num_bytes(),
                stream.cu_stream(),
            )?;
            cuda_core::memory::memset_d8_async(
                slots.cu_deviceptr(),
                0xFF,
                slots.num_bytes(),
                stream.cu_stream(),
            )?;
        }

        Ok(Self {
            ctrl,
            slots,
            capacity,
        })
    }

    /// Number of slots in the table. Fixed at construction time.
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Last-writer-wins bulk insert. Overwrites existing values.
    fn insert_bulk(
        &self,
        keys: &[u32],
        values: &[u32],
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(keys.len(), values.len());
        if keys.is_empty() {
            return Ok(());
        }
        debug_assert!(
            keys.iter().all(|&k| k != FORBIDDEN_KEY),
            "u32::MAX is reserved and may not be used as a key"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let values_dev = DeviceBuffer::from_host(stream, values)?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        cuda_launch! {
            kernel: insert_kernel,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(self.ctrl), slice(self.slots), slice(keys_dev), slice(values_dev)]
        }?;

        Ok(())
    }

    /// First-writer-wins bulk insert. Returns a `Vec<bool>` of length
    /// `keys.len()`; `true` means the key was fresh (and the table now
    /// contains the new value), `false` means the key was already present
    /// (and the table is unchanged for that key).
    fn try_insert_bulk(
        &self,
        keys: &[u32],
        values: &[u32],
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        assert_eq!(keys.len(), values.len());
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        debug_assert!(
            keys.iter().all(|&k| k != FORBIDDEN_KEY),
            "u32::MAX is reserved and may not be used as a key"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let values_dev = DeviceBuffer::from_host(stream, values)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        cuda_launch! {
            kernel: try_insert_kernel,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(self.ctrl), slice(self.slots), slice(keys_dev), slice(values_dev), slice_mut(out_dev)]
        }?;

        let raw = out_dev.to_host_vec(stream)?;
        Ok(raw.into_iter().map(|x| x == FLAG_FRESH_OR_OK).collect())
    }

    /// Bulk find. Returns `Vec<u32>` of length `keys.len()`; entries equal
    /// to `MISS = u32::MAX` mean "key not present".
    fn find_bulk(
        &self,
        keys: &[u32],
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        cuda_launch! {
            kernel: find_kernel,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(self.ctrl), slice(self.slots), slice(keys_dev), slice_mut(out_dev)]
        }?;

        Ok(out_dev.to_host_vec(stream)?)
    }

    /// Bulk find using the warp-cooperative kernel — one warp (32 lanes)
    /// per query key, 32 tag bytes inspected in parallel per probe step
    /// via `warp::ballot`. Same return contract as `find_bulk`.
    ///
    /// Requires `capacity >= PROBE_TILE = 32`, which holds for any
    /// power-of-two capacity at or above 32.
    fn find_bulk_warp(
        &self,
        keys: &[u32],
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        debug_assert!(
            self.capacity >= PROBE_TILE,
            "warp-cooperative find requires capacity >= {PROBE_TILE} slots"
        );

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        // One warp per key: launch (keys.len() * 32) threads, block size
        // 256 means 8 warps per block, 8 keys per block.
        let total_threads = (keys.len() as u32).saturating_mul(PROBE_TILE as u32);
        let cfg = LaunchConfig::for_num_elems(total_threads);
        cuda_launch! {
            kernel: find_kernel_warp,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(self.ctrl), slice(self.slots), slice(keys_dev), slice_mut(out_dev)]
        }?;

        Ok(out_dev.to_host_vec(stream)?)
    }

    /// Bulk delete (tombstone). Returns a `Vec<bool>` of length `keys.len()`;
    /// `true` means the key was present and is now tombstoned, `false` means
    /// the key was not in the table.
    fn delete_bulk(
        &self,
        keys: &[u32],
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let keys_dev = DeviceBuffer::from_host(stream, keys)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(stream, keys.len())?;

        let cfg = LaunchConfig::for_num_elems(keys.len() as u32);
        cuda_launch! {
            kernel: delete_kernel,
            stream: stream,
            module: module,
            config: cfg,
            args: [slice(self.ctrl), slice(self.slots), slice(keys_dev), slice_mut(out_dev)]
        }?;

        let raw = out_dev.to_host_vec(stream)?;
        Ok(raw.into_iter().map(|x| x == FLAG_FRESH_OR_OK).collect())
    }
}

// =============================================================================
// CORRECTNESS TESTS
// =============================================================================

/// Tiny xorshift32 — same as v1; avoids pulling in a crate for randomness.
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Sample `n` distinct keys, all `< u32::MAX`, deterministically seeded.
fn distinct_keys(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let k = xorshift32(&mut state);
        if k != FORBIDDEN_KEY && seen.insert(k) {
            out.push(k);
        }
    }
    out
}

fn main() {
    println!("=== GPU Hashmap v2 — SwissTable-Inspired ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("hashmap_v2.ptx")
        .expect("Failed to load PTX module");

    const CAPACITY: usize = 1 << 14;
    const M: usize = CAPACITY / 2;

    let keys_v0 = distinct_keys(M, 0xDEAD_BEEF);
    let values_v0: Vec<u32> = (0..M as u32).collect();

    // -------------------------------------------------------------------------
    // Test 1: insert_bulk roundtrip (parity with v1).
    // -------------------------------------------------------------------------
    println!("--- Test 1: insert_bulk roundtrip ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");
        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");

        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v0[i] {
                bad += 1;
            }
        }
        if bad == 0 {
            println!("  {M} / {M} keys round-tripped, capacity = {CAPACITY}");
        } else {
            println!("  FAIL: {bad} mismatches");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: miss on absent keys (parity with v1).
    // -------------------------------------------------------------------------
    println!("\n--- Test 2: miss on absent keys ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0xC0FF_EEEEu32;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }

        let found = map.find_bulk(&absent, &module, &stream).expect("find");
        let bad: usize = found.iter().filter(|&&v| v != MISS).count();
        if bad == 0 {
            println!("  {M} / {M} absent keys correctly missed");
        } else {
            println!("  FAIL: {bad} absent keys returned non-MISS");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: last-writer-wins on re-insert (parity with v1).
    // -------------------------------------------------------------------------
    println!("\n--- Test 3: insert_bulk last-writer-wins on re-insert ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert v0");

        let values_v1: Vec<u32> = values_v0.iter().map(|v| v ^ 0xA5A5_A5A5).collect();
        map.insert_bulk(&keys_v0, &values_v1, &module, &stream)
            .expect("insert v1");

        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");
        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v1[i] {
                bad += 1;
            }
        }
        if bad == 0 {
            println!("  all {M} keys reflect last-writer values");
        } else {
            println!("  FAIL: {bad} stale values");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: try_insert_bulk first-writer-wins (parity with v1).
    // -------------------------------------------------------------------------
    println!("\n--- Test 4: try_insert_bulk first-writer-wins ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let first_flags = map
            .try_insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("try_insert pass 1");
        let all_fresh = first_flags.iter().all(|&b| b);

        let values_v1: Vec<u32> = values_v0.iter().map(|v| v ^ 0xA5A5_A5A5).collect();
        let second_flags = map
            .try_insert_bulk(&keys_v0, &values_v1, &module, &stream)
            .expect("try_insert pass 2");
        let none_fresh = second_flags.iter().all(|&b| !b);

        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");
        let bad: usize = found
            .iter()
            .zip(&values_v0)
            .filter(|(f, v)| *f != *v)
            .count();

        if all_fresh && none_fresh && bad == 0 {
            println!("  pass 1: all {M} keys fresh");
            println!("  pass 2: no keys fresh, table preserves v0 values");
        } else {
            println!("  FAIL: all_fresh={all_fresh} none_fresh={none_fresh} bad_values={bad}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 5: load-factor stress at 75% (parity with v1 test 6).
    // -------------------------------------------------------------------------
    println!("\n--- Test 5: load-factor stress (~75%) ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0x1234_5678);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk(&dense_keys, &module, &stream)
            .expect("dense find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped at load factor {:.1}%, capacity = {}",
                100.0 * m_dense as f64 / map.capacity() as f64,
                map.capacity()
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: delete-then-find — deleted keys must miss; survivors must hit.
    // -------------------------------------------------------------------------
    println!("\n--- Test 6: delete-then-find ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        // Delete the first third.
        let third = M / 3;
        let to_delete = &keys_v0[..third];
        let survivors = &keys_v0[third..];

        let del_flags = map
            .delete_bulk(to_delete, &module, &stream)
            .expect("delete");
        let all_deleted = del_flags.iter().all(|&b| b);

        // Survivors must still hit with their original values.
        let surv_found = map
            .find_bulk(survivors, &module, &stream)
            .expect("find survivors");
        let mut bad_surv = 0usize;
        for (i, k_idx) in (third..M).enumerate() {
            if surv_found[i] != values_v0[k_idx] {
                bad_surv += 1;
            }
        }

        // Deleted keys must miss.
        let deleted_found = map
            .find_bulk(to_delete, &module, &stream)
            .expect("find deleted");
        let bad_del: usize = deleted_found.iter().filter(|&&v| v != MISS).count();

        if all_deleted && bad_surv == 0 && bad_del == 0 {
            println!(
                "  deleted {third} keys; {} survivors all hit, {third} deleted all miss",
                M - third
            );
        } else {
            println!("  FAIL: all_deleted={all_deleted} bad_surv={bad_surv} bad_del={bad_del}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 7: delete-then-reinsert — re-inserted key sees new value.
    //         (v2 cut 1 does not reclaim DELETED slots, so the new entry
    //          lands at a fresh slot; find still returns the new value
    //          because it walks past the tombstone.)
    // -------------------------------------------------------------------------
    println!("\n--- Test 7: delete-then-reinsert ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert v0");

        let third = M / 3;
        let to_recycle = &keys_v0[..third];
        let _ = map
            .delete_bulk(to_recycle, &module, &stream)
            .expect("delete");

        let new_values: Vec<u32> = values_v0[..third]
            .iter()
            .map(|v| v.wrapping_add(0x10_0000))
            .collect();
        map.insert_bulk(to_recycle, &new_values, &module, &stream)
            .expect("re-insert");

        let found = map
            .find_bulk(to_recycle, &module, &stream)
            .expect("find re-inserted");
        let bad: usize = found
            .iter()
            .zip(&new_values)
            .filter(|(f, v)| *f != *v)
            .count();

        if bad == 0 {
            println!("  {third} keys re-inserted with new values, all observable");
        } else {
            println!("  FAIL: {bad} re-inserted keys returned stale or missing values");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 8: warp-cooperative find parity with single-thread find on a
    //         mixed hit/miss query batch at moderate load (50%).
    // -------------------------------------------------------------------------
    println!("\n--- Test 8: warp-cooperative find parity ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        // Build a query batch that's half hits (the inserted keys) and
        // half misses (fresh random keys). Total = 2 * M queries.
        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0x0BAD_F00D;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }
        let mut query_keys = Vec::with_capacity(2 * M);
        query_keys.extend_from_slice(&keys_v0);
        query_keys.extend_from_slice(&absent);

        let single = map
            .find_bulk(&query_keys, &module, &stream)
            .expect("find single");
        let warp = map
            .find_bulk_warp(&query_keys, &module, &stream)
            .expect("find warp");

        let bad: usize = single.iter().zip(&warp).filter(|(s, w)| *s != *w).count();
        let hits: usize = warp.iter().filter(|&&v| v != MISS).count();

        if bad == 0 && hits == M {
            println!(
                "  warp-coop find matches single-thread find on {} queries ({hits} hits, {} misses)",
                query_keys.len(),
                query_keys.len() - hits
            );
        } else {
            println!("  FAIL: bad={bad} hits={hits} (expected {M})");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 9: warp-cooperative find at 75% load — the regime where it
    //         is supposed to win against single-thread find.
    // -------------------------------------------------------------------------
    println!("\n--- Test 9: warp-cooperative find at ~75% load ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0xFEED_FACE);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk_warp(&dense_keys, &module, &stream)
            .expect("warp find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped via warp-coop find at load factor {:.1}%",
                100.0 * m_dense as f64 / map.capacity() as f64,
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    println!("\n=== SUCCESS: All 9 hashmap v2 tests passed! ===");
}
