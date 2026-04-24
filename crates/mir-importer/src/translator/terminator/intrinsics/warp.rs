/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level primitives.
//!
//! Handles translation of warp shuffle and vote intrinsics.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::ReadPtxSregLaneIdOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emits `lane_id()`: Get the lane index within the warp.
///
/// Returns the thread's position within its 32-thread warp (0-31).
///
/// # Generated Operation
///
/// `nvvm.read.ptx.sreg.laneid` - Maps to PTX `mov.u32 %r, %laneid`
///
/// # Returns
///
/// `u32` - Lane index (0-31)
pub fn emit_lane_id(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    // Create lane_id operation
    let lane_id_op = Operation::new(
        ctx,
        ReadPtxSregLaneIdOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    lane_id_op.deref_mut(ctx).set_loc(loc.clone());

    let lane_id_op = if let Some(prev) = prev_op {
        lane_id_op.insert_after(ctx, prev);
        lane_id_op
    } else {
        lane_id_op.insert_at_front(block_ptr, ctx);
        lane_id_op
    };

    let result_value = lane_id_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        lane_id_op,
        value_map,
        block_map,
        loc,
        "lane_id call without target block",
    )
}

/// Emit a warp shuffle operation for i32.
///
/// # Parameters
/// - `shuffle_opid`: The NVVM opid for the specific shuffle variant
/// - `args`: [value, lane/mask/delta]
pub fn emit_warp_shuffle_i32(
    ctx: &mut Context,
    body: &mir::Body,
    shuffle_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use pliron::value::Value;

    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp shuffle expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    // Get the value operand (arg 0)
    let (val, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the lane/mask/delta operand (arg 1) - often a constant like 16, 8, 4, 2, 1
    let (lane_or_mask, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Create the shuffle operation
    let shuffle_op = Operation::new(
        ctx,
        shuffle_opid,
        vec![u32_type.to_ptr()],
        vec![val, lane_or_mask],
        vec![],
        0,
    );
    shuffle_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        shuffle_op.insert_after(ctx, prev);
    } else {
        shuffle_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = Value::OpResult {
        op: shuffle_op,
        res_idx: 0,
    };
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        shuffle_op,
        value_map,
        block_map,
        loc,
        "warp shuffle call without target block",
    )
}

/// Emit a warp shuffle operation for f32.
pub fn emit_warp_shuffle_f32(
    ctx: &mut Context,
    body: &mir::Body,
    shuffle_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use pliron::builtin::types::FP32Type;
    use pliron::value::Value;

    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp shuffle f32 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let f32_type = FP32Type::get(ctx);

    // Get the value operand (arg 0)
    let (val, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the lane/mask/delta operand (arg 1) - often a constant like 16, 8, 4, 2, 1
    let (lane_or_mask, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Create the shuffle operation
    let shuffle_op = Operation::new(
        ctx,
        shuffle_opid,
        vec![f32_type.into()],
        vec![val, lane_or_mask],
        vec![],
        0,
    );
    shuffle_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        shuffle_op.insert_after(ctx, prev);
    } else {
        shuffle_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = Value::OpResult {
        op: shuffle_op,
        res_idx: 0,
    };
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        shuffle_op,
        value_map,
        block_map,
        loc,
        "warp shuffle f32 call without target block",
    )
}

/// Emit a warp vote operation (all, any, ballot).
pub fn emit_warp_vote(
    ctx: &mut Context,
    body: &mir::Body,
    vote_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    result_is_i32: bool, // true for ballot (i32), false for all/any (i1)
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use pliron::value::Value;

    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp vote expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    // Get the predicate operand (arg 0)
    let (predicate, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Result type: i32 for ballot, i1 (bool, signless to match Rust `bool`)
    // for all/any.
    let result_type = if result_is_i32 {
        IntegerType::get(ctx, 32, Signedness::Unsigned).to_ptr()
    } else {
        types::get_bool_type(ctx).to_ptr()
    };

    // Create the vote operation
    let vote_op = Operation::new(
        ctx,
        vote_opid,
        vec![result_type],
        vec![predicate],
        vec![],
        0,
    );
    vote_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        vote_op.insert_after(ctx, prev);
    } else {
        vote_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = Value::OpResult {
        op: vote_op,
        res_idx: 0,
    };
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        vote_op,
        value_map,
        block_map,
        loc,
        "warp vote call without target block",
    )
}
