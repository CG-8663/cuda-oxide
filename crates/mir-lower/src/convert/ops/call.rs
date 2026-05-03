/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Call operation conversion: `dialect-mir` → `dialect-llvm`.
//!
//! Handles function call lowering with ABI-level transformations:
//! - Slice arguments flattened to (ptr, len) pairs
//! - Struct arguments flattened to individual fields
//! - Unit return type becomes void
//! - Pointer arguments cast to generic address space (for ABI compatibility)
//!
//! These transformations match the function signature flattening done in
//! `convert_function_type`.
//!
//! # Device Extern Symbol Resolution
//!
//! When calling device extern functions (declared with `#[device] extern "C"`),
//! the MIR contains calls to prefixed symbols like `cuda_oxide_device_extern_foo`.
//! This prefix is added by the proc-macro for internal detection. However, the
//! external LTOIR (e.g., CCCL libraries) exports the original symbol name `foo`.
//!
//! We strip the prefix during lowering so the LLVM IR emits:
//! ```llvm
//! call @foo(...)  ; NOT @cuda_oxide_device_extern_foo
//! ```
//!
//! This allows nvJitLink to resolve the symbol against the external LTOIR.
//!
//! # Address Space Handling
//!
//! Function signatures use generic pointers (addrspace 0) for ABI compatibility.
//! When passing pointers from specific address spaces (e.g., shared memory = addrspace 3),
//! we insert `addrspacecast` to convert them to generic pointers at the call site.

use crate::convert::types::{convert_type, is_zero_sized_type};
use crate::helpers;
use dialect_llvm::op_interfaces::CastOpInterface;
use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use dialect_mir::ops::MirCallOp;
use dialect_mir::rust_intrinsics;
use dialect_mir::types::{MirDisjointSliceType, MirSliceType, MirStructType, MirTupleType};
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeObj, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Generic address space (can alias any memory).
const ADDRSPACE_GENERIC: u32 = 0;

/// Prefix added by `#[device] extern "C"` proc-macro for internal detection.
/// Stripped during lowering so LLVM IR uses the original symbol name.
const DEVICE_EXTERN_PREFIX: &str = "cuda_oxide_device_extern_";

/// Internal placeholder for rustc bit intrinsics that need LLVM intrinsic calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustBitIntrinsic {
    RotateLeft,
    RotateRight,
    Ctpop,
    Ctlz { zero_undef: bool },
    Cttz { zero_undef: bool },
    Bswap,
    Bitreverse,
}

impl RustBitIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_ROTATE_LEFT => Some(Self::RotateLeft),
            rust_intrinsics::CALLEE_ROTATE_RIGHT => Some(Self::RotateRight),
            rust_intrinsics::CALLEE_CTPOP => Some(Self::Ctpop),
            rust_intrinsics::CALLEE_CTLZ => Some(Self::Ctlz { zero_undef: false }),
            rust_intrinsics::CALLEE_CTLZ_NONZERO => Some(Self::Ctlz { zero_undef: true }),
            rust_intrinsics::CALLEE_CTTZ => Some(Self::Cttz { zero_undef: false }),
            rust_intrinsics::CALLEE_CTTZ_NONZERO => Some(Self::Cttz { zero_undef: true }),
            rust_intrinsics::CALLEE_BSWAP => Some(Self::Bswap),
            rust_intrinsics::CALLEE_BITREVERSE => Some(Self::Bitreverse),
            _ => None,
        }
    }
}

/// Internal placeholder for rustc saturating arithmetic intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustSaturatingIntrinsic {
    Add,
    Sub,
}

impl RustSaturatingIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_SATURATING_ADD => Some(Self::Add),
            rust_intrinsics::CALLEE_SATURATING_SUB => Some(Self::Sub),
            _ => None,
        }
    }
}

/// Internal placeholder for rustc float math intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustFloatMathIntrinsic {
    SqrtF32,
    SqrtF64,
    PowiF32,
    PowiF64,
    SinF32,
    SinF64,
    CosF32,
    CosF64,
    TanF32,
    TanF64,
    PowfF32,
    PowfF64,
    ExpF32,
    ExpF64,
    Exp2F32,
    Exp2F64,
    LogF32,
    LogF64,
    Log2F32,
    Log2F64,
    Log10F32,
    Log10F64,
    FmaF32,
    FmaF64,
    FmuladdF32,
    FmuladdF64,
    FloorF32,
    FloorF64,
    CeilF32,
    CeilF64,
    TruncF32,
    TruncF64,
    RoundF32,
    RoundF64,
    RoundevenF32,
    RoundevenF64,
    Fabs,
    CopysignF32,
    CopysignF64,
}

impl RustFloatMathIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_SQRT_F32 => Some(Self::SqrtF32),
            rust_intrinsics::CALLEE_SQRT_F64 => Some(Self::SqrtF64),
            rust_intrinsics::CALLEE_POWI_F32 => Some(Self::PowiF32),
            rust_intrinsics::CALLEE_POWI_F64 => Some(Self::PowiF64),
            rust_intrinsics::CALLEE_SIN_F32 => Some(Self::SinF32),
            rust_intrinsics::CALLEE_SIN_F64 => Some(Self::SinF64),
            rust_intrinsics::CALLEE_COS_F32 => Some(Self::CosF32),
            rust_intrinsics::CALLEE_COS_F64 => Some(Self::CosF64),
            rust_intrinsics::CALLEE_TAN_F32 => Some(Self::TanF32),
            rust_intrinsics::CALLEE_TAN_F64 => Some(Self::TanF64),
            rust_intrinsics::CALLEE_POWF_F32 => Some(Self::PowfF32),
            rust_intrinsics::CALLEE_POWF_F64 => Some(Self::PowfF64),
            rust_intrinsics::CALLEE_EXP_F32 => Some(Self::ExpF32),
            rust_intrinsics::CALLEE_EXP_F64 => Some(Self::ExpF64),
            rust_intrinsics::CALLEE_EXP2_F32 => Some(Self::Exp2F32),
            rust_intrinsics::CALLEE_EXP2_F64 => Some(Self::Exp2F64),
            rust_intrinsics::CALLEE_LOG_F32 => Some(Self::LogF32),
            rust_intrinsics::CALLEE_LOG_F64 => Some(Self::LogF64),
            rust_intrinsics::CALLEE_LOG2_F32 => Some(Self::Log2F32),
            rust_intrinsics::CALLEE_LOG2_F64 => Some(Self::Log2F64),
            rust_intrinsics::CALLEE_LOG10_F32 => Some(Self::Log10F32),
            rust_intrinsics::CALLEE_LOG10_F64 => Some(Self::Log10F64),
            rust_intrinsics::CALLEE_FMA_F32 => Some(Self::FmaF32),
            rust_intrinsics::CALLEE_FMA_F64 => Some(Self::FmaF64),
            rust_intrinsics::CALLEE_FMULADD_F32 => Some(Self::FmuladdF32),
            rust_intrinsics::CALLEE_FMULADD_F64 => Some(Self::FmuladdF64),
            rust_intrinsics::CALLEE_FLOOR_F32 => Some(Self::FloorF32),
            rust_intrinsics::CALLEE_FLOOR_F64 => Some(Self::FloorF64),
            rust_intrinsics::CALLEE_CEIL_F32 => Some(Self::CeilF32),
            rust_intrinsics::CALLEE_CEIL_F64 => Some(Self::CeilF64),
            rust_intrinsics::CALLEE_TRUNC_F32 => Some(Self::TruncF32),
            rust_intrinsics::CALLEE_TRUNC_F64 => Some(Self::TruncF64),
            rust_intrinsics::CALLEE_ROUND_F32 => Some(Self::RoundF32),
            rust_intrinsics::CALLEE_ROUND_F64 => Some(Self::RoundF64),
            rust_intrinsics::CALLEE_ROUNDEVEN_F32 => Some(Self::RoundevenF32),
            rust_intrinsics::CALLEE_ROUNDEVEN_F64 => Some(Self::RoundevenF64),
            rust_intrinsics::CALLEE_FABS => Some(Self::Fabs),
            rust_intrinsics::CALLEE_COPYSIGN_F32 => Some(Self::CopysignF32),
            rust_intrinsics::CALLEE_COPYSIGN_F64 => Some(Self::CopysignF64),
            _ => None,
        }
    }

    /// CUDA libdevice function name for this Rust math intrinsic.
    fn libdevice_name(
        self,
        ctx: &Context,
        result_ty: Ptr<TypeObj>,
        loc: pliron::location::Location,
    ) -> Result<&'static str> {
        match self {
            Self::SqrtF32 => Ok("__nv_sqrtf"),
            Self::SqrtF64 => Ok("__nv_sqrt"),
            Self::PowiF32 => Ok("__nv_powif"),
            Self::PowiF64 => Ok("__nv_powi"),
            Self::SinF32 => Ok("__nv_sinf"),
            Self::SinF64 => Ok("__nv_sin"),
            Self::CosF32 => Ok("__nv_cosf"),
            Self::CosF64 => Ok("__nv_cos"),
            Self::TanF32 => Ok("__nv_tanf"),
            Self::TanF64 => Ok("__nv_tan"),
            Self::PowfF32 => Ok("__nv_powf"),
            Self::PowfF64 => Ok("__nv_pow"),
            Self::ExpF32 => Ok("__nv_expf"),
            Self::ExpF64 => Ok("__nv_exp"),
            Self::Exp2F32 => Ok("__nv_exp2f"),
            Self::Exp2F64 => Ok("__nv_exp2"),
            Self::LogF32 => Ok("__nv_logf"),
            Self::LogF64 => Ok("__nv_log"),
            Self::Log2F32 => Ok("__nv_log2f"),
            Self::Log2F64 => Ok("__nv_log2"),
            Self::Log10F32 => Ok("__nv_log10f"),
            Self::Log10F64 => Ok("__nv_log10"),
            Self::FmaF32 | Self::FmuladdF32 => Ok("__nv_fmaf"),
            Self::FmaF64 | Self::FmuladdF64 => Ok("__nv_fma"),
            Self::FloorF32 => Ok("__nv_floorf"),
            Self::FloorF64 => Ok("__nv_floor"),
            Self::CeilF32 => Ok("__nv_ceilf"),
            Self::CeilF64 => Ok("__nv_ceil"),
            Self::TruncF32 => Ok("__nv_truncf"),
            Self::TruncF64 => Ok("__nv_trunc"),
            Self::RoundF32 => Ok("__nv_roundf"),
            Self::RoundF64 => Ok("__nv_round"),
            Self::RoundevenF32 => Ok("__nv_rintf"),
            Self::RoundevenF64 => Ok("__nv_rint"),
            Self::Fabs => fabs_libdevice_name(ctx, result_ty, loc),
            Self::CopysignF32 => Ok("__nv_copysignf"),
            Self::CopysignF64 => Ok("__nv_copysign"),
        }
    }

    /// Number of operands expected by the libdevice function.
    fn arg_count(self) -> usize {
        match self {
            Self::PowiF32
            | Self::PowiF64
            | Self::PowfF32
            | Self::PowfF64
            | Self::CopysignF32
            | Self::CopysignF64 => 2,
            Self::FmaF32 | Self::FmaF64 | Self::FmuladdF32 | Self::FmuladdF64 => 3,
            _ => 1,
        }
    }
}

fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

/// Convert `mir.call` to `llvm.call` with argument flattening.
///
/// Performs ABI-level transformations to match CUDA calling conventions:
/// - Slice arguments: flattened to `(ptr, len)` pairs
/// - Struct arguments: flattened to individual fields
/// - Unit return type: converted to void
/// - Callee name: `::` mangled to `__` for LLVM identifier validity
/// - Device extern calls: prefix stripped to use original symbol name
pub fn convert(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let callee_name: String = {
        let mir_call = MirCallOp::new(op);
        let callee_attr = match mir_call.get_attr_callee(ctx) {
            Some(a) => a,
            None => {
                return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "MirCallOp missing callee attribute"
                );
            }
        };
        (*callee_attr).clone().into()
    };

    if let Some(intrinsic) = RustBitIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_bit_intrinsic(ctx, rewriter, op, intrinsic);
    }

    if let Some(intrinsic) = RustSaturatingIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_saturating_intrinsic(ctx, rewriter, op, operands_info, intrinsic);
    }

    if let Some(intrinsic) = RustFloatMathIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_float_math_intrinsic(ctx, rewriter, op, intrinsic);
    }

    let callee_ident: pliron::identifier::Identifier = {
        let resolved_name = resolve_device_extern_symbol(&callee_name);

        resolved_name
            .try_into()
            .expect("callee name should have been legalized during MIR import")
    };

    let args: Vec<Value> = op.deref(ctx).operands().collect();

    let has_result = op.deref(ctx).get_num_results() > 0;
    let mir_result_ty_ptr = if has_result {
        Some(op.deref(ctx).get_result(0).get_type(ctx))
    } else {
        None
    };

    let result_type = if let Some(mir_ty) = mir_result_ty_ptr {
        let is_unit = mir_ty.deref(ctx).is::<MirTupleType>();
        if is_unit {
            llvm_types::VoidType::get(ctx).into()
        } else {
            convert_type(ctx, mir_ty).map_err(anyhow_to_pliron)?
        }
    } else {
        llvm_types::VoidType::get(ctx).into()
    };

    let (flattened_args, flattened_arg_types) =
        flatten_arguments(ctx, rewriter, &args, operands_info)?;

    let func_type = llvm_types::FuncType::get(ctx, result_type, flattened_arg_types, false);
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(callee_ident),
        func_type,
        flattened_args,
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let is_void = result_type.deref(ctx).is::<llvm_types::VoidType>();
    if has_result && !is_void && llvm_call.get_operation().deref(ctx).get_num_results() > 0 {
        rewriter.replace_operation(ctx, op, llvm_call.get_operation());
    } else {
        rewriter.erase_operation(ctx, op);
    }

    Ok(())
}

/// Lower placeholder calls for rustc's integer bit intrinsics to LLVM intrinsics.
///
/// Rust methods like `u128::rotate_left` call `core::intrinsics::rotate_left`
/// in libcore. The importer preserves that as a placeholder `mir.call`; here we
/// recover the concrete integer width and emit the corresponding overloaded
/// LLVM intrinsic.
fn convert_rust_bit_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustBitIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust bit intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let Some(&value) = args.first() else {
        return pliron::input_err!(loc, "Rust bit intrinsic call missing integer operand");
    };

    let value_ty = value.get_type(ctx);
    let value_width = integer_bit_width(ctx, value_ty, loc.clone())?;
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_type = convert_type(ctx, mir_result_ty).map_err(anyhow_to_pliron)?;

    if matches!(intrinsic, RustBitIntrinsic::Bswap) && value_width == 8 {
        // LLVM has no useful byte swap for a single byte; Rust's semantics are identity.
        let bitcast = llvm::BitcastOp::new(ctx, value, result_type);
        rewriter.insert_operation(ctx, bitcast.get_operation());
        rewriter.replace_operation(ctx, op, bitcast.get_operation());
        return Ok(());
    }

    let (intrinsic_name, intrinsic_args, intrinsic_result_ty) = match intrinsic {
        RustBitIntrinsic::RotateLeft | RustBitIntrinsic::RotateRight => {
            if args.len() != 2 {
                return pliron::input_err!(
                    loc,
                    "rotate intrinsic requires value and shift operands"
                );
            }
            let (shift, _) =
                cast_integer_value_to_type(ctx, rewriter, args[1], value_ty, loc.clone())?;
            let suffix = match intrinsic {
                RustBitIntrinsic::RotateLeft => "fshl",
                RustBitIntrinsic::RotateRight => "fshr",
                _ => unreachable!(),
            };
            (
                format!("llvm_{suffix}_i{value_width}"),
                vec![value, value, shift],
                value_ty,
            )
        }
        RustBitIntrinsic::Ctpop => (format!("llvm_ctpop_i{value_width}"), vec![value], value_ty),
        RustBitIntrinsic::Ctlz { zero_undef } => {
            let zero_undef = create_i1_constant(ctx, rewriter, zero_undef);
            (
                format!("llvm_ctlz_i{value_width}"),
                vec![value, zero_undef],
                value_ty,
            )
        }
        RustBitIntrinsic::Cttz { zero_undef } => {
            let zero_undef = create_i1_constant(ctx, rewriter, zero_undef);
            (
                format!("llvm_cttz_i{value_width}"),
                vec![value, zero_undef],
                value_ty,
            )
        }
        RustBitIntrinsic::Bswap => (format!("llvm_bswap_i{value_width}"), vec![value], value_ty),
        RustBitIntrinsic::Bitreverse => (
            format!("llvm_bitreverse_i{value_width}"),
            vec![value],
            value_ty,
        ),
    };

    let arg_types = intrinsic_args
        .iter()
        .map(|arg| arg.get_type(ctx))
        .collect::<Vec<_>>();
    let func_ty = llvm_types::FuncType::get(ctx, intrinsic_result_ty, arg_types, false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(loc.clone(), "Rust bit intrinsic call has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .as_str()
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(sym_name),
        func_ty,
        intrinsic_args,
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let call_result = llvm_call.get_operation().deref(ctx).get_result(0);
    let (_, final_op) =
        cast_integer_value_to_type(ctx, rewriter, call_result, result_type, loc.clone())?;
    let replacement = final_op.unwrap_or_else(|| llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, replacement);

    Ok(())
}

/// Lower placeholder calls for rustc's saturating integer intrinsics.
///
/// Rust preserves signedness in the original MIR type. The converted LLVM value
/// is signless, so this uses `operands_info` to choose `sadd/ssub` versus
/// `uadd/usub`.
fn convert_rust_saturating_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
    intrinsic: RustSaturatingIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust saturating intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    if args.len() != 2 {
        return pliron::input_err!(
            loc,
            "Rust saturating intrinsic requires left and right operands"
        );
    }

    let lhs = args[0];
    let rhs = args[1];
    let lhs_ty = lhs.get_type(ctx);
    let width = integer_bit_width(ctx, lhs_ty, loc.clone())?;
    let is_signed =
        if let Some(int_ty) = operands_info.lookup_most_recent_of_type::<IntegerType>(ctx, lhs) {
            int_ty.signedness() == Signedness::Signed
        } else {
            return pliron::input_err!(loc, "expected integer type for Rust saturating intrinsic");
        };

    let (rhs, _) = cast_integer_value_to_type(ctx, rewriter, rhs, lhs_ty, loc.clone())?;
    let op_stem = match (is_signed, intrinsic) {
        (true, RustSaturatingIntrinsic::Add) => "sadd",
        (false, RustSaturatingIntrinsic::Add) => "uadd",
        (true, RustSaturatingIntrinsic::Sub) => "ssub",
        (false, RustSaturatingIntrinsic::Sub) => "usub",
    };
    let intrinsic_name = format!("llvm_{op_stem}_sat_i{width}");
    let func_ty = llvm_types::FuncType::get(ctx, lhs_ty, vec![lhs_ty, lhs_ty], false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "Rust saturating intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .as_str()
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(sym_name),
        func_ty,
        vec![lhs, rhs],
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let call_result = llvm_call.get_operation().deref(ctx).get_result(0);
    let (_, final_op) =
        cast_integer_value_to_type(ctx, rewriter, call_result, result_ty, loc.clone())?;
    let replacement = final_op.unwrap_or_else(|| llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, replacement);

    Ok(())
}

/// Lower placeholder calls for rustc's `f32` / `f64` math intrinsics to libdevice.
fn convert_rust_float_math_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust float math intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let expected_args = intrinsic.arg_count();
    if args.len() != expected_args {
        return pliron::input_err!(
            loc,
            "Rust float math intrinsic requires {expected_args} operand(s)"
        );
    }

    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let intrinsic_name = intrinsic.libdevice_name(ctx, result_ty, loc.clone())?;
    let arg_types = args.iter().map(|arg| arg.get_type(ctx)).collect::<Vec<_>>();
    let func_ty = llvm_types::FuncType::get(ctx, result_ty, arg_types, false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "Rust float math intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(ctx, CallOpCallable::Direct(sym_name), func_ty, args);
    rewriter.insert_operation(ctx, llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, llvm_call.get_operation());

    Ok(())
}

/// Read the width from an integer type, or report a useful lowering error.
fn integer_bit_width(
    ctx: &Context,
    ty: Ptr<TypeObj>,
    loc: pliron::location::Location,
) -> Result<u32> {
    let ty_ref = ty.deref(ctx);
    let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() else {
        return pliron::input_err!(loc, "expected integer type for Rust bit intrinsic");
    };
    Ok(int_ty.width())
}

/// Return the libdevice `fabs` entry point for the concrete float type.
fn fabs_libdevice_name(
    ctx: &Context,
    ty: Ptr<TypeObj>,
    loc: pliron::location::Location,
) -> Result<&'static str> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<FP32Type>() {
        Ok("__nv_fabsf")
    } else if ty_ref.is::<FP64Type>() {
        Ok("__nv_fabs")
    } else {
        pliron::input_err!(
            loc,
            "expected f32 or f64 type for Rust float math intrinsic"
        )
    }
}

/// Insert the `i1` flag operand used by `llvm.ctlz` and `llvm.cttz`.
fn create_i1_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: bool,
) -> Value {
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let width = NonZeroUsize::new(1).expect("1 is non-zero");
    let apint = APInt::from_u64(u64::from(value), width);
    let attr = IntegerAttr::new(i1_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Cast an integer value to the target width when Rust and LLVM disagree.
///
/// This is needed for count/zero intrinsics: LLVM returns `iN`, while Rust's
/// public methods return `u32`.
fn cast_integer_value_to_type(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    target_ty: Ptr<TypeObj>,
    loc: pliron::location::Location,
) -> Result<(Value, Option<Ptr<Operation>>)> {
    let source_width = integer_bit_width(ctx, value.get_type(ctx), loc.clone())?;
    let target_width = integer_bit_width(ctx, target_ty, loc)?;

    if source_width == target_width {
        return Ok((value, None));
    }

    let cast_op = if source_width < target_width {
        let zext = llvm::ZExtOp::new(ctx, value, target_ty);
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        zext.get_operation().deref_mut(ctx).attributes.0.insert(
            nneg_key,
            pliron::builtin::attributes::BoolAttr::new(false).into(),
        );
        zext.get_operation()
    } else {
        llvm::TruncOp::new(ctx, value, target_ty).get_operation()
    };
    rewriter.insert_operation(ctx, cast_op);
    Ok((cast_op.deref(ctx).get_result(0), Some(cast_op)))
}

/// Flatten arguments according to ABI rules.
///
/// - Slice types → (ptr, len) pair
/// - Struct types → individual field values (in MEMORY ORDER)
/// - Other types → pass through
fn flatten_arguments(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    args: &[Value],
    operands_info: &OperandsInfo,
) -> Result<(Vec<Value>, Vec<Ptr<TypeObj>>)> {
    let mut flattened_args = Vec::new();
    let mut flattened_arg_types = Vec::new();

    for arg in args.iter() {
        let arg_ty = arg.get_type(ctx);

        enum FlattenKind {
            Slice,
            Struct {
                field_types: Vec<Ptr<TypeObj>>,
                mem_to_decl: Vec<usize>,
            },
            None,
        }

        let flatten_kind = if let Some(mir_ty) = operands_info.lookup_most_recent_type(*arg) {
            let ty_ref = mir_ty.deref(ctx);
            if ty_ref.is::<MirSliceType>() || ty_ref.is::<MirDisjointSliceType>() {
                FlattenKind::Slice
            } else if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
                FlattenKind::Struct {
                    field_types: struct_ty.field_types.clone(),
                    mem_to_decl: struct_ty.memory_order(),
                }
            } else {
                FlattenKind::None
            }
        } else {
            FlattenKind::None
        };

        match flatten_kind {
            FlattenKind::Slice => {
                let ptr_ty = llvm_types::PointerType::get_generic(ctx);
                let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);

                let extract_ptr = llvm::ExtractValueOp::new(ctx, *arg, vec![0])?;
                rewriter.insert_operation(ctx, extract_ptr.get_operation());
                let ptr_val = extract_ptr.get_operation().deref(ctx).get_result(0);

                let extract_len = llvm::ExtractValueOp::new(ctx, *arg, vec![1])?;
                rewriter.insert_operation(ctx, extract_len.get_operation());
                let len_val = extract_len.get_operation().deref(ctx).get_result(0);

                flattened_args.push(ptr_val);
                flattened_args.push(len_val);
                flattened_arg_types.push(ptr_ty.into());
                flattened_arg_types.push(len_ty.into());
            }
            FlattenKind::Struct {
                field_types,
                mem_to_decl,
            } => {
                let mut llvm_idx = 0u32;
                for mem_idx in 0..field_types.len() {
                    let decl_idx = mem_to_decl[mem_idx];
                    let llvm_field_ty =
                        convert_type(ctx, field_types[decl_idx]).map_err(anyhow_to_pliron)?;
                    if is_zero_sized_type(ctx, llvm_field_ty) {
                        continue;
                    }
                    let extract_op = llvm::ExtractValueOp::new(ctx, *arg, vec![llvm_idx])?;
                    rewriter.insert_operation(ctx, extract_op.get_operation());
                    let field_val = extract_op.get_operation().deref(ctx).get_result(0);
                    flattened_args.push(field_val);
                    flattened_arg_types.push(llvm_field_ty);
                    llvm_idx += 1;
                }
            }
            FlattenKind::None => {
                let (final_arg, final_ty) =
                    cast_pointer_to_generic_if_needed(ctx, rewriter, *arg, arg_ty)?;
                flattened_args.push(final_arg);
                flattened_arg_types.push(final_ty);
            }
        }
    }

    Ok((flattened_args, flattened_arg_types))
}

/// Cast a pointer to generic address space if it's in a specific address space.
///
/// Function signatures use generic pointers (addrspace 0) for ABI compatibility.
/// When the argument is a pointer in a non-generic address space (e.g., shared=3),
/// we insert an `addrspacecast` instruction to convert it.
///
/// Returns the (possibly casted) value and its type.
fn cast_pointer_to_generic_if_needed(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    arg: Value,
    arg_ty: Ptr<TypeObj>,
) -> Result<(Value, Ptr<TypeObj>)> {
    let maybe_addrspace = {
        let arg_ty_ref = arg_ty.deref(ctx);
        arg_ty_ref
            .downcast_ref::<llvm_types::PointerType>()
            .map(|ptr_ty| ptr_ty.address_space())
    };

    if let Some(addrspace) = maybe_addrspace
        && addrspace != ADDRSPACE_GENERIC
    {
        let cast_op = llvm::AddrSpaceCastOp::new(ctx, arg, ADDRSPACE_GENERIC);
        rewriter.insert_operation(ctx, cast_op.get_operation());
        let casted_val = cast_op.get_operation().deref(ctx).get_result(0);
        let generic_ptr_ty = llvm_types::PointerType::get_generic(ctx);
        return Ok((casted_val, generic_ptr_ty.into()));
    }

    Ok((arg, arg_ty))
}

/// Resolve device extern symbol name by stripping internal prefix.
///
/// When calling `#[device] extern "C"` functions, the Rust code (after macro expansion)
/// calls symbols like `cuda_oxide_device_extern_foo`. We strip this prefix to emit
/// calls to the original symbol `foo` that external LTOIR provides.
///
/// # Limitations
///
/// This approach derives the original name by stripping the prefix, which means:
/// - Custom `#[link_name = "..."]` attributes are NOT honored (edge case)
/// - If the original function is `bar` but user specifies `#[link_name = "custom"]`,
///   we'd emit `bar` not `custom`. This is acceptable for most use cases.
fn resolve_device_extern_symbol(callee_name: &str) -> String {
    if let Some(pos) = callee_name.find(DEVICE_EXTERN_PREFIX) {
        let after_prefix = &callee_name[pos + DEVICE_EXTERN_PREFIX.len()..];
        return after_prefix.to_string();
    }

    callee_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_device_extern_symbol() {
        assert_eq!(
            resolve_device_extern_symbol("cuda_oxide_device_extern_dot_product"),
            "dot_product"
        );

        assert_eq!(
            resolve_device_extern_symbol("device_ffi_test::cuda_oxide_device_extern_foo"),
            "foo"
        );

        assert_eq!(
            resolve_device_extern_symbol("my_module::regular_function"),
            "my_module::regular_function"
        );

        assert_eq!(
            resolve_device_extern_symbol("cuda_oxide_kernel_my_kernel"),
            "cuda_oxide_kernel_my_kernel"
        );
    }
}
