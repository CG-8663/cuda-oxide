/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler bit-manipulation intrinsics.
//!
//! These are `core::intrinsics::*` calls emitted by libcore for primitive
//! integer methods such as `rotate_left`, `count_ones`, and `swap_bytes`.
//! They lower to target-independent LLVM intrinsics during MIR -> LLVM lowering.

use super::super::helpers;
use crate::error::TranslationResult;
use crate::translator::types;
use crate::translator::values::ValueMap;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::operation::Operation;
use rustc_public::mir;

pub const CALLEE_ROTATE_LEFT: &str = "__cuda_oxide_rust_intrinsic_rotate_left";
pub const CALLEE_ROTATE_RIGHT: &str = "__cuda_oxide_rust_intrinsic_rotate_right";
pub const CALLEE_CTPOP: &str = "__cuda_oxide_rust_intrinsic_ctpop";
pub const CALLEE_CTLZ: &str = "__cuda_oxide_rust_intrinsic_ctlz";
pub const CALLEE_CTLZ_NONZERO: &str = "__cuda_oxide_rust_intrinsic_ctlz_nonzero";
pub const CALLEE_CTTZ: &str = "__cuda_oxide_rust_intrinsic_cttz";
pub const CALLEE_CTTZ_NONZERO: &str = "__cuda_oxide_rust_intrinsic_cttz_nonzero";
pub const CALLEE_BSWAP: &str = "__cuda_oxide_rust_intrinsic_bswap";
pub const CALLEE_BITREVERSE: &str = "__cuda_oxide_rust_intrinsic_bitreverse";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustBitIntrinsic {
    RotateLeft,
    RotateRight,
    Ctpop,
    Ctlz { zero_undef: bool },
    Cttz { zero_undef: bool },
    Bswap,
    Bitreverse,
}

impl RustBitIntrinsic {
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::rotate_left" | "std::intrinsics::rotate_left" => {
                Some(Self::RotateLeft)
            }
            "core::intrinsics::rotate_right" | "std::intrinsics::rotate_right" => {
                Some(Self::RotateRight)
            }
            "core::intrinsics::ctpop" | "std::intrinsics::ctpop" => Some(Self::Ctpop),
            "core::intrinsics::ctlz" | "std::intrinsics::ctlz" => {
                Some(Self::Ctlz { zero_undef: false })
            }
            "core::intrinsics::ctlz_nonzero" | "std::intrinsics::ctlz_nonzero" => {
                Some(Self::Ctlz { zero_undef: true })
            }
            "core::intrinsics::cttz" | "std::intrinsics::cttz" => {
                Some(Self::Cttz { zero_undef: false })
            }
            "core::intrinsics::cttz_nonzero" | "std::intrinsics::cttz_nonzero" => {
                Some(Self::Cttz { zero_undef: true })
            }
            "core::intrinsics::bswap" | "std::intrinsics::bswap" => Some(Self::Bswap),
            "core::intrinsics::bitreverse" | "std::intrinsics::bitreverse" => {
                Some(Self::Bitreverse)
            }
            _ => None,
        }
    }

    pub fn marker_callee(self) -> &'static str {
        match self {
            Self::RotateLeft => CALLEE_ROTATE_LEFT,
            Self::RotateRight => CALLEE_ROTATE_RIGHT,
            Self::Ctpop => CALLEE_CTPOP,
            Self::Ctlz { zero_undef: false } => CALLEE_CTLZ,
            Self::Ctlz { zero_undef: true } => CALLEE_CTLZ_NONZERO,
            Self::Cttz { zero_undef: false } => CALLEE_CTTZ,
            Self::Cttz { zero_undef: true } => CALLEE_CTTZ_NONZERO,
            Self::Bswap => CALLEE_BSWAP,
            Self::Bitreverse => CALLEE_BITREVERSE,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn emit_rust_bit_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustBitIntrinsic,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let return_type = types::translate_type(ctx, &body.locals()[destination.local].ty)?;
    helpers::emit_function_call(
        ctx,
        body,
        intrinsic.marker_callee(),
        args,
        destination,
        return_type,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}
