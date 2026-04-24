/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attributes belonging to the MIR dialect.

use pliron::attribute::Attribute;
use pliron::context::Context;
use pliron_derive::pliron_attr;

/// MIR cast kind — preserves the semantic intent of the cast from Rust MIR.
///
/// The lowering dispatches on this to pick the correct LLVM instruction,
/// rather than guessing from source/destination types.
#[pliron_attr(name = "mir.cast_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum MirCastKindAttr {
    IntToInt,
    IntToFloat,
    FloatToInt,
    FloatToFloat,
    PtrToPtr,
    FnPtrToPtr,
    PointerExposeAddress,
    PointerWithExposedProvenance,
    Transmute,
    PointerCoercionUnsize,
    PointerCoercionMutToConst,
    PointerCoercionArrayToPointer,
    PointerCoercionReifyFnPointer,
    PointerCoercionUnsafeFnPointer,
    PointerCoercionClosureFnPointer,
    Subtype,
}

/// Boolean attribute for reference mutability.
///
/// Replaces the overloaded `IntegerAttr` pattern with a self-documenting
/// domain-specific attribute.
#[pliron_attr(name = "mir.mutability", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct MutabilityAttr(pub bool);

/// Structural field index for aggregate access ops
/// (`mir.extract_field`, `mir.insert_field`, `mir.field_addr`, `mir.enum_payload`).
#[pliron_attr(name = "mir.field_index", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct FieldIndexAttr(pub u32);

/// Enum variant index for variant-level ops
/// (`mir.construct_enum`, `mir.enum_payload`).
#[pliron_attr(name = "mir.variant_index", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct VariantIndexAttr(pub u32);

pub fn register(ctx: &mut Context) {
    MirCastKindAttr::register(ctx);
    MutabilityAttr::register(ctx);
    FieldIndexAttr::register(ctx);
    VariantIndexAttr::register(ctx);
}
