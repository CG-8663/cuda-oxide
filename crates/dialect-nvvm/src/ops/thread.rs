/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread, block, and grid indexing operations.
//!
//! This module provides operations for reading GPU thread hierarchy registers:
//!
//! ```text
//! ┌──────────────────────┬──────────────┬────────────────────────────┐
//! │ Operation            │ PTX Register │ Description                │
//! ├──────────────────────┼──────────────┼────────────────────────────┤
//! │ ReadPtxSregTidXOp    │ %tid.x       │ Thread ID within block (X) │
//! │ ReadPtxSregTidYOp    │ %tid.y       │ Thread ID within block (Y) │
//! │ ReadPtxSregCtaidXOp  │ %ctaid.x     │ Block ID within grid (X)   │
//! │ ReadPtxSregCtaidYOp  │ %ctaid.y     │ Block ID within grid (Y)   │
//! │ ReadPtxSregNtidXOp   │ %ntid.x      │ Block dimension (X)        │
//! │ ReadPtxSregNtidYOp   │ %ntid.y      │ Block dimension (Y)        │
//! │ Barrier0Op           │ bar.sync 0   │ Block-wide barrier         │
//! │ ThreadfenceBlockOp   │ membar.cta   │ Block-scoped memory fence  │
//! │ ThreadfenceOp        │ membar.gl    │ Device-scoped memory fence │
//! │ ThreadfenceSystemOp  │ membar.sys   │ System-scoped memory fence │
//! └──────────────────────┴──────────────┴────────────────────────────┘
//! ```
//!
//! # Thread Hierarchy
//!
//! ```text
//! Grid (gridDim.x × gridDim.y blocks)
//! └── Block (blockDim.x × blockDim.y threads)
//!     └── Thread (identified by threadIdx)
//! ```
//!
//! Each operation returns a 32-bit integer representing the index or dimension.

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
// X-Dimension Indexing
// =============================================================================

/// Read the X component of the thread ID within the block.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.tid.x` / PTX `%tid.x`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_tid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregTidXOp;

impl ReadPtxSregTidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregTidXOp { op }
    }
}

impl Verify for ReadPtxSregTidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_tid_x result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_tid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the X component of the block ID within the grid.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ctaid.x` / PTX `%ctaid.x`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ctaid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregCtaidXOp;

impl ReadPtxSregCtaidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregCtaidXOp { op }
    }
}

impl Verify for ReadPtxSregCtaidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_ctaid_x result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ctaid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the X component of the block dimension (threads per block).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ntid.x` / PTX `%ntid.x`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ntid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNtidXOp;

impl ReadPtxSregNtidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNtidXOp { op }
    }
}

impl Verify for ReadPtxSregNtidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_ntid_x result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ntid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Y-Dimension Indexing
// =============================================================================

/// Read the Y component of the thread ID within the block.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.tid.y` / PTX `%tid.y`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_tid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregTidYOp;

impl ReadPtxSregTidYOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregTidYOp { op }
    }
}

impl Verify for ReadPtxSregTidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_tid_y result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_tid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Y component of the block ID within the grid.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ctaid.y` / PTX `%ctaid.y`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ctaid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregCtaidYOp;

impl ReadPtxSregCtaidYOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregCtaidYOp { op }
    }
}

impl Verify for ReadPtxSregCtaidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_ctaid_y result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ctaid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Y component of the block dimension (threads per block).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ntid.y` / PTX `%ntid.y`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ntid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNtidYOp;

impl ReadPtxSregNtidYOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNtidYOp { op }
    }
}

impl Verify for ReadPtxSregNtidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_ntid_y result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ntid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Block Synchronization
// =============================================================================

/// Block-wide barrier synchronization.
///
/// All threads in the block must reach this barrier before any can proceed.
/// Corresponds to `llvm.nvvm.barrier0` / PTX `bar.sync 0`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 0 results
#[pliron_op(
    name = "nvvm.barrier0",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Barrier0Op;

impl Barrier0Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Barrier0Op { op }
    }
}

/// Block-scoped memory fence.
///
/// Orders the calling thread's prior memory operations before later memory
/// operations as observed by threads in the same CTA. Corresponds to PTX
/// `membar.cta`.
#[pliron_op(
    name = "nvvm.threadfence_block",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceBlockOp;

impl ThreadfenceBlockOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceBlockOp { op }
    }
}

/// Device-scoped memory fence.
///
/// Orders the calling thread's prior global-memory operations before later
/// memory operations as observed by threads on the same GPU. Corresponds to
/// PTX `membar.gl`.
#[pliron_op(
    name = "nvvm.threadfence",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceOp;

impl ThreadfenceOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceOp { op }
    }
}

/// System-scoped memory fence.
///
/// Orders the calling thread's prior global-memory operations before later
/// memory operations as observed by other GPUs or the CPU. Corresponds to PTX
/// `membar.sys`.
#[pliron_op(
    name = "nvvm.threadfence_system",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceSystemOp;

impl ThreadfenceSystemOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceSystemOp { op }
    }
}

/// Register thread indexing operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // X-dimension
    ReadPtxSregTidXOp::register(ctx);
    ReadPtxSregCtaidXOp::register(ctx);
    ReadPtxSregNtidXOp::register(ctx);
    // Y-dimension
    ReadPtxSregTidYOp::register(ctx);
    ReadPtxSregCtaidYOp::register(ctx);
    ReadPtxSregNtidYOp::register(ctx);
    // Synchronization
    Barrier0Op::register(ctx);
    ThreadfenceBlockOp::register(ctx);
    ThreadfenceOp::register(ctx);
    ThreadfenceSystemOp::register(ctx);
}
