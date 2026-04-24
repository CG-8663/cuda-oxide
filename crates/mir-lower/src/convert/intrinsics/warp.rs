/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level intrinsic conversion: shuffle and vote operations.
//!
//! # Shuffle Operations
//!
//! | Operation          | LLVM Intrinsic                 | Description       |
//! |--------------------|--------------------------------|-------------------|
//! | `ShflSyncIdxI32`   | `llvm.nvvm.shfl.sync.idx.i32`  | Indexed shuffle   |
//! | `ShflSyncBflyI32`  | `llvm.nvvm.shfl.sync.bfly.i32` | Butterfly shuffle |
//! | `ShflSyncDownI32`  | `llvm.nvvm.shfl.sync.down.i32` | Down shuffle      |
//! | `ShflSyncUpI32`    | `llvm.nvvm.shfl.sync.up.i32`   | Up shuffle        |
//!
//! # Vote Operations
//!
//! | Operation        | LLVM Intrinsic               | Description           |
//! |------------------|------------------------------|-----------------------|
//! | `VoteSyncAll`    | `llvm.nvvm.vote.all.sync`    | All lanes true        |
//! | `VoteSyncAny`    | `llvm.nvvm.vote.any.sync`    | Any lane true         |
//! | `VoteSyncBallot` | `llvm.nvvm.vote.ballot.sync` | Bitmask of predicates |

use crate::convert::intrinsics::common::*;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert i32 shuffle operation to LLVM intrinsic call.
pub(crate) fn convert_shuffle_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("Warp shuffle i32 requires 2 operands");
    }
    let (val, lane_or_delta) = (operands[0], operands[1]);

    let mask_val = create_i32_const(ctx, rewriter, -1);
    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask_val, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert f32 shuffle operation to LLVM intrinsic call.
pub(crate) fn convert_shuffle_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let f32_ty = FP32Type::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("Warp shuffle f32 requires 2 operands");
    }
    let (val, lane_or_delta) = (operands[0], operands[1]);

    let mask_val = create_i32_const(ctx, rewriter, -1);
    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        f32_ty.into(),
        vec![i32_ty.into(), f32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask_val, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert vote operation to LLVM intrinsic call.
pub(crate) fn convert_vote(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("Warp vote requires 1 operand");
    }
    let predicate = operands[0];

    let mask_val = create_i32_const(ctx, rewriter, -1);

    let result_ty: Ptr<pliron::r#type::TypeObj> = if intrinsic_name.contains("ballot") {
        i32_ty.into()
    } else {
        i1_ty.into()
    };

    let func_ty =
        llvm_types::FuncType::get(ctx, result_ty, vec![i32_ty.into(), i1_ty.into()], false);
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask_val, predicate],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}
