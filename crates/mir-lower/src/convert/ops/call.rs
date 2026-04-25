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
use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use dialect_mir::ops::MirCallOp;
use dialect_mir::types::{MirDisjointSliceType, MirSliceType, MirStructType, MirTupleType};
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeObj, Typed};
use pliron::value::Value;

/// Generic address space (can alias any memory).
const ADDRSPACE_GENERIC: u32 = 0;

/// Prefix added by `#[device] extern "C"` proc-macro for internal detection.
/// Stripped during lowering so LLVM IR uses the original symbol name.
const DEVICE_EXTERN_PREFIX: &str = "cuda_oxide_device_extern_";

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
    let callee_ident: pliron::identifier::Identifier = {
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
        let callee_name: String = (*callee_attr).clone().into();

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
