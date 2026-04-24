/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level operations: shuffle, vote, and lane identification.
//!
//! A warp is a group of 32 threads that execute in lockstep. These operations
//! enable efficient intra-warp communication without shared memory.
//!
//! # Shuffle Operations
//!
//! Shuffle operations allow threads to exchange register values directly:
//!
//! ```text
//! ┌──────┬──────────────────────┬───────────────────────────────────┐
//! │ Mode │ PTX                  │ Description                       │
//! ├──────┼──────────────────────┼───────────────────────────────────┤
//! │ idx  │ shfl.sync.idx.b32    │ Read from specific lane           │
//! │ bfly │ shfl.sync.bfly.b32   │ XOR lane ID with mask (butterfly) │
//! │ down │ shfl.sync.down.b32   │ Read from lane + delta            │
//! │ up   │ shfl.sync.up.b32     │ Read from lane - delta            │
//! └──────┴──────────────────────┴───────────────────────────────────┘
//! ```
//!
//! # Vote Operations
//!
//! Vote operations perform warp-wide predicate evaluation:
//!
//! ```text
//! ┌─────────────┬──────────────────────────────────────────────────────┐
//! │ Operation   │ Returns                                              │
//! ├─────────────┼──────────────────────────────────────────────────────┤
//! │ vote.all    │ true if ALL active threads have predicate true       │
//! │ vote.any    │ true if ANY active thread has predicate true         │
//! │ vote.ballot │ 32-bit mask where bit[i] = thread i's predicate      │
//! └─────────────┴──────────────────────────────────────────────────────┘
//! ```

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::IntegerType,
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

// =============================================================================
// Lane Identification
// =============================================================================

/// Read the lane ID within the warp (0-31).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.laneid` / PTX `%laneid`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_laneid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregLaneIdOp;

impl ReadPtxSregLaneIdOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregLaneIdOp { op }
    }
}

impl Verify for ReadPtxSregLaneIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_laneid result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_laneid result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Warp Shuffle - Integer (i32)
// =============================================================================

/// Warp shuffle: read from a specific lane (idx mode) for i32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.idx.i32`.
///
/// # Operands
///
/// - `value` (i32): the value to share
/// - `src_lane` (i32): the lane index to read from (0-31)
///
/// # Results
///
/// - `result` (i32): the value from the source lane
#[pliron_op(
    name = "nvvm.shfl_sync_idx_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncIdxI32Op;

impl ShflSyncIdxI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncIdxI32Op { op }
    }
}

/// Warp shuffle: butterfly (XOR) pattern for i32.
///
/// Reads from lane `(lane_id XOR lane_mask)`. This pattern is commonly used
/// for parallel reductions (e.g., XOR with 16, 8, 4, 2, 1 for warp-wide sum).
///
/// Corresponds to `llvm.nvvm.shfl.sync.bfly.i32`.
///
/// # Operands
///
/// - `value` (i32): the value to exchange
/// - `lane_mask` (i32): XOR mask for lane calculation
///
/// # Results
///
/// - `result` (i32): the value from lane `(self XOR mask)`
#[pliron_op(
    name = "nvvm.shfl_sync_bfly_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncBflyI32Op;

impl ShflSyncBflyI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncBflyI32Op { op }
    }
}

/// Warp shuffle: read from higher lane (down mode) for i32.
///
/// Reads from lane `(lane_id + delta)`. Values from out-of-range lanes are undefined.
///
/// Corresponds to `llvm.nvvm.shfl.sync.down.i32`.
///
/// # Operands
///
/// - `value` (i32): the value to share
/// - `delta` (i32): offset to add to lane ID
///
/// # Results
///
/// - `result` (i32): the value from lane `(self + delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_down_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncDownI32Op;

impl ShflSyncDownI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncDownI32Op { op }
    }
}

/// Warp shuffle: read from lower lane (up mode) for i32.
///
/// Reads from lane `(lane_id - delta)`. Values from negative lanes are undefined.
///
/// Corresponds to `llvm.nvvm.shfl.sync.up.i32`.
///
/// # Operands
///
/// - `value` (i32): the value to share
/// - `delta` (i32): offset to subtract from lane ID
///
/// # Results
///
/// - `result` (i32): the value from lane `(self - delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_up_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncUpI32Op;

impl ShflSyncUpI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncUpI32Op { op }
    }
}

// =============================================================================
// Warp Shuffle - Float (f32)
// =============================================================================

/// Warp shuffle: read from a specific lane (idx mode) for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.idx.f32`.
///
/// # Operands
///
/// - `value` (f32): the value to share
/// - `src_lane` (i32): the lane index to read from (0-31)
///
/// # Results
///
/// - `result` (f32): the value from the source lane
#[pliron_op(
    name = "nvvm.shfl_sync_idx_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncIdxF32Op;

impl ShflSyncIdxF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncIdxF32Op { op }
    }
}

/// Warp shuffle: butterfly (XOR) pattern for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.bfly.f32`.
///
/// # Operands
///
/// - `value` (f32): the value to exchange
/// - `lane_mask` (i32): XOR mask for lane calculation
///
/// # Results
///
/// - `result` (f32): the value from lane `(self XOR mask)`
#[pliron_op(
    name = "nvvm.shfl_sync_bfly_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncBflyF32Op;

impl ShflSyncBflyF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncBflyF32Op { op }
    }
}

/// Warp shuffle: read from higher lane (down mode) for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.down.f32`.
///
/// # Operands
///
/// - `value` (f32): the value to share
/// - `delta` (i32): offset to add to lane ID
///
/// # Results
///
/// - `result` (f32): the value from lane `(self + delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_down_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncDownF32Op;

impl ShflSyncDownF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncDownF32Op { op }
    }
}

/// Warp shuffle: read from lower lane (up mode) for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.up.f32`.
///
/// # Operands
///
/// - `value` (f32): the value to share
/// - `delta` (i32): offset to subtract from lane ID
///
/// # Results
///
/// - `result` (f32): the value from lane `(self - delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_up_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ShflSyncUpF32Op;

impl ShflSyncUpF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncUpF32Op { op }
    }
}

// =============================================================================
// Warp Vote Operations
// =============================================================================

/// Warp vote: returns true if ALL active threads have predicate true.
///
/// Corresponds to `llvm.nvvm.vote.sync.all`.
///
/// # Operands
///
/// - `predicate` (i1): the condition to check
///
/// # Results
///
/// - `result` (i1): true if all active threads have predicate true
#[pliron_op(
    name = "nvvm.vote_sync_all",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct VoteSyncAllOp;

impl VoteSyncAllOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VoteSyncAllOp { op }
    }
}

/// Warp vote: returns true if ANY active thread has predicate true.
///
/// Corresponds to `llvm.nvvm.vote.sync.any`.
///
/// # Operands
///
/// - `predicate` (i1): the condition to check
///
/// # Results
///
/// - `result` (i1): true if any active thread has predicate true
#[pliron_op(
    name = "nvvm.vote_sync_any",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct VoteSyncAnyOp;

impl VoteSyncAnyOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VoteSyncAnyOp { op }
    }
}

/// Warp ballot: returns a 32-bit mask where bit[i] indicates thread i's predicate.
///
/// Corresponds to `llvm.nvvm.vote.sync.ballot`.
///
/// # Operands
///
/// - `predicate` (i1): the condition to check
///
/// # Results
///
/// - `result` (i32): bitmask where bit `i` is set if thread `i` has predicate true
#[pliron_op(
    name = "nvvm.vote_sync_ballot",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct VoteSyncBallotOp;

impl VoteSyncBallotOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VoteSyncBallotOp { op }
    }
}

/// Register warp operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // Lane identification
    ReadPtxSregLaneIdOp::register(ctx);
    // Shuffle - i32
    ShflSyncIdxI32Op::register(ctx);
    ShflSyncBflyI32Op::register(ctx);
    ShflSyncDownI32Op::register(ctx);
    ShflSyncUpI32Op::register(ctx);
    // Shuffle - f32
    ShflSyncIdxF32Op::register(ctx);
    ShflSyncBflyF32Op::register(ctx);
    ShflSyncDownF32Op::register(ctx);
    ShflSyncUpF32Op::register(ctx);
    // Vote
    VoteSyncAllOp::register(ctx);
    VoteSyncAnyOp::register(ctx);
    VoteSyncBallotOp::register(ctx);
}
