/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM Dialect for [pliron]

use pliron::{
    context::Context,
    dialect::{Dialect, DialectName},
};

pub mod attributes;
pub mod export;
pub mod op_interfaces;
pub mod ops;
pub mod types;

/// Register LLVM dialect, its ops, types and attributes into context.
pub fn register(ctx: &mut Context) {
    Dialect::register(ctx, &DialectName::new("llvm"));
    ops::register(ctx);
    types::register(ctx);
    attributes::register(ctx);
}
