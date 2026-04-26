/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_nvvm::ops::{
    Barrier0Op, ReadPtxSregLaneIdOp, ReadPtxSregTidXOp, ThreadfenceBlockOp, ThreadfenceOp,
    ThreadfenceSystemOp,
};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn test_thread_register_ops_verify_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);

    let tid_x = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregTidXOp::new(tid_x).verify(&ctx).is_ok());

    let lane_id = Operation::new(
        &mut ctx,
        ReadPtxSregLaneIdOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLaneIdOp::new(lane_id).verify(&ctx).is_ok());
}

#[test]
fn test_thread_register_ops_reject_non_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
    let op = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );

    assert!(ReadPtxSregTidXOp::new(op).verify(&ctx).is_err());
}

#[test]
fn test_sync_ops_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let barrier = Operation::new(
        &mut ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(Barrier0Op::new(barrier).verify(&ctx).is_ok());

    let block_fence = Operation::new(
        &mut ctx,
        ThreadfenceBlockOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceBlockOp::new(block_fence).verify(&ctx).is_ok());

    let device_fence = Operation::new(
        &mut ctx,
        ThreadfenceOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceOp::new(device_fence).verify(&ctx).is_ok());

    let system_fence = Operation::new(
        &mut ctx,
        ThreadfenceSystemOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceSystemOp::new(system_fence).verify(&ctx).is_ok());
}
