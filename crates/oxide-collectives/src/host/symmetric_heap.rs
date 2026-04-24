/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Symmetric multi-GPU virtual memory setup.

use super::Topology;
use crate::device::Team;
use crate::error::{CollectiveError, Result};
use cuda_bindings::CUdeviceptr;
use cuda_core::CudaContext;
use cuda_core::vmm::{self, Mapping, PhysicalAllocation, VirtualReservation};
use std::alloc::Layout;
use std::any::type_name;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

/// One PE-local virtual-address view of the symmetric heap.
struct ContextView {
    /// Context whose current virtual-address alias owns this view.
    ctx: Arc<CudaContext>,
    /// One VA reservation local to `ctx`.
    ///
    /// The current Phase 2 design reserves one window per PE, so the same
    /// logical symmetric slot can have one virtual-address alias per PE.
    reservation: Option<VirtualReservation>,
    /// One mapping per owner PE, laid out contiguously inside `reservation`.
    mappings: Vec<Mapping>,
}

impl ContextView {
    /// Returns the base address of this PE-local VA alias.
    fn base(&self) -> CUdeviceptr {
        self.reservation
            .as_ref()
            .expect("symmetric heap reservation is only removed during drop")
            .base()
    }
}

/// One VMM-backed symmetric heap chunk per PE, mapped into every PE's address
/// space at the same logical layout.
///
/// If the heap has `N` PEs and per-PE chunk size `S`, then each PE reserves a
/// local virtual address range of `N * S` bytes. The range is laid out as:
///
/// - `[0 * S, 1 * S)` = PE 0's physical allocation
/// - `[1 * S, 2 * S)` = PE 1's physical allocation
/// - ...
/// - `[(N - 1) * S, N * S)` = PE `N - 1`'s physical allocation
///
/// This lets later device-side code derive a peer pointer from just:
///
/// - the current PE's reservation base,
/// - the owner PE index,
/// - and a byte offset within that owner's chunk.
///
/// In this prototype, each PE reserves its own VA window. The windows all have
/// the same layout, but they do not have to share the same numeric base
/// address. As a result, a logical symmetric slot may have multiple valid
/// virtual-address aliases, one per PE-local reservation.
pub struct SymmetricHeap {
    /// Peer ordering and reachability used to construct the heap.
    topology: Topology,
    /// Per-PE chunk capacity after rounding up to a granularity supported by
    /// every participating device.
    chunk_size: usize,
    /// Total bytes reserved in each PE-local VA window.
    total_size: usize,
    /// One PE-local VA view of the symmetric layout per participant.
    views: Vec<ContextView>,
    /// Physical backing allocation for each owner PE's chunk.
    allocations: Vec<PhysicalAllocation>,
    /// Host-side bump cursor shared by all typed allocations.
    next_offset: Mutex<usize>,
}

impl SymmetricHeap {
    /// Creates a symmetric heap with `chunk_size` bytes per PE.
    ///
    /// The constructor:
    ///
    /// 1. Discovers peer connectivity and requires a full peer mesh.
    /// 2. Enables peer access for every directed PE pair.
    /// 3. Allocates one physical VMM allocation per PE.
    /// 4. Reserves one VA window per PE and maps all per-PE chunks into it.
    /// 5. Grants every participating device read/write access to every mapping.
    ///
    /// Because Step 4 creates one reservation per PE rather than one shared
    /// reservation for the whole process, the same logical slot may be
    /// reachable through multiple virtual-address aliases. The layout is still
    /// symmetric because the owner-PE ordering and per-allocation offsets are
    /// identical in every reservation.
    pub fn new(contexts: &[Arc<CudaContext>], chunk_size: usize) -> Result<Arc<Self>> {
        if chunk_size == 0 {
            return Err(CollectiveError::InvalidChunkSize);
        }

        let topology = Topology::discover(contexts)?;
        topology.require_full_mesh()?;
        topology.enable_peer_access(contexts)?;

        let chunk_size = aligned_chunk_size(contexts, chunk_size)?;
        let pe_count = contexts.len();
        let total_size = chunk_size
            .checked_mul(pe_count)
            .ok_or(CollectiveError::SizeOverflow {
                chunk_size,
                pe_count,
            })?;

        let mut heap = SymmetricHeap {
            topology,
            chunk_size,
            total_size,
            views: Vec::with_capacity(pe_count),
            allocations: Vec::with_capacity(pe_count),
            next_offset: Mutex::new(0),
        };

        for ctx in contexts {
            ctx.bind_to_thread()?;
            heap.allocations
                .push(PhysicalAllocation::new(ctx.cu_device(), chunk_size)?);
        }

        let devices: Vec<_> = contexts.iter().map(|ctx| ctx.cu_device()).collect();
        for ctx in contexts {
            ctx.bind_to_thread()?;
            let reservation = VirtualReservation::new(total_size, 0)?;
            let base = reservation.base();
            let mut mappings = Vec::with_capacity(heap.allocations.len());

            for (owner_pe, allocation) in heap.allocations.iter().enumerate() {
                let slot_offset =
                    owner_pe
                        .checked_mul(chunk_size)
                        .ok_or(CollectiveError::SizeOverflow {
                            chunk_size,
                            pe_count,
                        })?;
                let mapping_va = add_offset(base, slot_offset)?;
                mappings.push(Mapping::new(mapping_va, chunk_size, allocation, 0)?);
            }

            vmm::set_access(base, total_size, &devices)?;
            heap.views.push(ContextView {
                ctx: ctx.clone(),
                reservation: Some(reservation),
                mappings,
            });
        }

        Ok(Arc::new(heap))
    }

    /// Number of participating PEs.
    pub fn pe_count(&self) -> usize {
        self.topology.pe_count()
    }

    /// Per-PE heap capacity in bytes after granularity alignment.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Total reserved VA size in each PE's address space.
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Discovered peer topology used to build this heap.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Base pointer of `pe`'s local reservation.
    ///
    /// This is the base of that PE's local alias of the symmetric heap, not a
    /// globally unique address for the heap.
    pub fn base_ptr(&self, pe: usize) -> Result<CUdeviceptr> {
        self.validate_pe(pe)?;
        Ok(self.views[pe].base())
    }

    /// Creates the Phase 3 device-side team view for `pe`.
    ///
    /// The returned [`Team`] contains the local PE's heap alias plus the
    /// metadata needed to derive remote owner slots inside that alias.
    pub fn team(&self, pe: usize) -> Result<Team<'_>> {
        self.validate_pe(pe)?;
        let pe_count =
            u32::try_from(self.pe_count()).map_err(|_| CollectiveError::PeCountTooLarge {
                pe_count: self.pe_count(),
            })?;

        Ok(
            // SAFETY: `validate_pe` proved `pe < pe_count`, `pe_count` fits in
            // `u32`, and the heap stores a non-null reservation base.
            unsafe {
                Team::new_unchecked(
                    pe as u32,
                    pe_count,
                    self.views[pe].base() as *mut u8,
                    self.chunk_size,
                )
            },
        )
    }

    /// Returns the pointer, as seen from `observer_pe`, to the start of
    /// `owner_pe`'s chunk plus `byte_offset`.
    ///
    /// `observer_pe` selects which PE-local reservation alias to use. Two
    /// different observers can therefore receive different numeric pointers for
    /// the same underlying logical slot.
    pub fn ptr_for(
        &self,
        observer_pe: usize,
        owner_pe: usize,
        byte_offset: usize,
    ) -> Result<CUdeviceptr> {
        self.ptr_for_range(observer_pe, owner_pe, byte_offset, 0)
    }

    /// Reserves `len` elements of type `T` at the next aligned offset in every
    /// PE's chunk and returns a typed handle.
    ///
    /// This is a host-side bump allocation: each call chooses one byte offset,
    /// and that same offset names one same-shaped slot in every owner's chunk.
    /// The returned [`SymmetricAlloc`] can then compute local and remote
    /// pointers for any observer/owner PE pair.
    pub fn alloc<T>(self: &Arc<Self>, len: usize) -> Result<SymmetricAlloc<T>> {
        if size_of::<T>() == 0 {
            return Err(CollectiveError::ZeroSizedType {
                type_name: type_name::<T>(),
            });
        }

        let layout = Layout::array::<T>(len).map_err(|_| CollectiveError::LayoutOverflow {
            type_name: type_name::<T>(),
            len,
        })?;

        let mut next_offset = self
            .next_offset
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let start = round_up_to_multiple(*next_offset, layout.align()).ok_or(
            CollectiveError::HeapExhausted {
                requested: layout.size(),
                remaining: 0,
            },
        )?;
        let end = start
            .checked_add(layout.size())
            .ok_or(CollectiveError::HeapExhausted {
                requested: layout.size(),
                remaining: self.chunk_size.saturating_sub(start.min(self.chunk_size)),
            })?;

        if end > self.chunk_size {
            return Err(CollectiveError::HeapExhausted {
                requested: layout.size(),
                remaining: self.chunk_size.saturating_sub(start.min(self.chunk_size)),
            });
        }

        *next_offset = end;
        drop(next_offset);

        Ok(SymmetricAlloc {
            heap: self.clone(),
            offset_bytes: start,
            len,
            byte_len: layout.size(),
            _marker: PhantomData,
        })
    }

    /// Internal helper used by [`ptr_for`](Self::ptr_for) and
    /// [`SymmetricAlloc`] to validate bounds and compute the requested alias.
    fn ptr_for_range(
        &self,
        observer_pe: usize,
        owner_pe: usize,
        byte_offset: usize,
        size: usize,
    ) -> Result<CUdeviceptr> {
        self.validate_pe(observer_pe)?;
        self.validate_pe(owner_pe)?;
        self.validate_range(byte_offset, size)?;

        let slot_offset =
            owner_pe
                .checked_mul(self.chunk_size)
                .ok_or(CollectiveError::SizeOverflow {
                    chunk_size: self.chunk_size,
                    pe_count: self.pe_count(),
                })?;
        let absolute_offset =
            slot_offset
                .checked_add(byte_offset)
                .ok_or(CollectiveError::AddressOverflow {
                    base: self.views[observer_pe].base(),
                    offset: byte_offset,
                })?;

        add_offset(self.views[observer_pe].base(), absolute_offset)
    }

    /// Validates that `pe` names a participating PE in this heap.
    fn validate_pe(&self, pe: usize) -> Result<()> {
        if pe < self.pe_count() {
            Ok(())
        } else {
            Err(CollectiveError::InvalidPe {
                pe,
                pe_count: self.pe_count(),
            })
        }
    }

    /// Validates that `[offset, offset + size)` fits inside one owner's chunk.
    fn validate_range(&self, offset: usize, size: usize) -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or(CollectiveError::OutOfBounds {
                offset,
                size,
                chunk_size: self.chunk_size,
            })?;

        if end <= self.chunk_size {
            Ok(())
        } else {
            Err(CollectiveError::OutOfBounds {
                offset,
                size,
                chunk_size: self.chunk_size,
            })
        }
    }
}

impl Drop for SymmetricHeap {
    fn drop(&mut self) {
        // `cuMemUnmap` / `cuMemAddressFree` are context-scoped. We therefore
        // tear down each view under its owning PE's current context before
        // dropping the underlying physical allocations.
        for view in &mut self.views {
            let _ = view.ctx.bind_to_thread();
            view.mappings.clear();
            drop(view.reservation.take());
        }
        self.allocations.clear();
    }
}

/// A typed allocation carved out of every PE's heap chunk at the same offset.
///
/// A `SymmetricAlloc<T>` names one logical allocation, but physically there is
/// still one backing slot per owner PE. The handle stores only the shared byte
/// offset and length; concrete device pointers are derived on demand.
pub struct SymmetricAlloc<T> {
    /// Heap that owns the underlying symmetric layout.
    heap: Arc<SymmetricHeap>,
    /// Shared byte offset of this allocation inside every owner's chunk.
    offset_bytes: usize,
    /// Logical element count requested by the caller.
    len: usize,
    /// Total byte footprint of the allocation in each owner's chunk.
    byte_len: usize,
    /// Carries the logical element type without storing any host-side values.
    _marker: PhantomData<T>,
}

impl<T> SymmetricAlloc<T> {
    /// Number of logical `T` elements in this allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when `len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of bytes occupied by the allocation inside each PE's chunk.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Byte offset shared by this allocation in every PE's chunk.
    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    /// Returns the pointer, as seen from `observer_pe`, to this allocation
    /// inside `owner_pe`'s chunk.
    ///
    /// Different `observer_pe` values may produce different numeric pointers
    /// for the same logical allocation because the current heap uses one VA
    /// reservation per PE.
    pub fn ptr_for(&self, observer_pe: usize, owner_pe: usize) -> Result<CUdeviceptr> {
        self.heap
            .ptr_for_range(observer_pe, owner_pe, self.offset_bytes, self.byte_len)
    }

    /// Pointer to the allocation in `pe`'s own chunk, using `pe`'s local VA
    /// alias of the heap.
    pub fn local_ptr(&self, pe: usize) -> Result<CUdeviceptr> {
        self.ptr_for(pe, pe)
    }

    /// Pointer to the allocation in `target_pe`'s chunk, as seen from
    /// `observer_pe`.
    pub fn remote_ptr(&self, observer_pe: usize, target_pe: usize) -> Result<CUdeviceptr> {
        self.ptr_for(observer_pe, target_pe)
    }

    /// The heap that owns this typed allocation.
    pub fn heap(&self) -> &Arc<SymmetricHeap> {
        &self.heap
    }
}

/// Rounds the requested per-PE chunk size up to a granularity accepted by
/// every participating device.
fn aligned_chunk_size(contexts: &[Arc<CudaContext>], requested: usize) -> Result<usize> {
    let mut granularity_lcm = 1usize;

    for ctx in contexts {
        ctx.bind_to_thread()?;
        let granularity = vmm::allocation_granularity(ctx.cu_device())?;
        granularity_lcm =
            lcm(granularity_lcm, granularity).ok_or(CollectiveError::SizeOverflow {
                chunk_size: requested,
                pe_count: contexts.len(),
            })?;
    }

    round_up_to_multiple(requested, granularity_lcm).ok_or(CollectiveError::SizeOverflow {
        chunk_size: requested,
        pe_count: contexts.len(),
    })
}

/// Adds a byte offset to a device pointer while checking for overflow.
fn add_offset(base: CUdeviceptr, offset: usize) -> Result<CUdeviceptr> {
    base.checked_add(offset as u64)
        .ok_or(CollectiveError::AddressOverflow { base, offset })
}

/// Rounds `value` up to the next multiple of `multiple`.
fn round_up_to_multiple(value: usize, multiple: usize) -> Option<usize> {
    if multiple == 0 {
        return Some(value);
    }
    let remainder = value % multiple;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(multiple - remainder)
    }
}

/// Least common multiple with checked arithmetic.
fn lcm(a: usize, b: usize) -> Option<usize> {
    if a == 0 || b == 0 {
        return Some(0);
    }
    a.checked_div(gcd(a, b))?.checked_mul(b)
}

/// Euclidean greatest common divisor.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}
