/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM dialect operations.
//!
//! This module defines and registers all LLVM dialect operations by re-exporting them
//! from their respective sub-modules. Operations are organized by semantic category.
//!
//! # Module Organization
//!
//! ```text
//! ┌──────────────┬───────────────────────────────────────┬─────┐
//! │ Module       │ Description                           │ Ops │
//! ├──────────────┼───────────────────────────────────────┼─────┤
//! │ arithmetic   │ Integer and floating-point binary ops │ 19  │
//! │ atomic       │ Atomic load/store/RMW/CAS/fence       │ 5   │
//! │ comparison   │ Integer and floating-point comparisons│ 2   │
//! │ cast         │ Type conversion operations            │ 13  │
//! │ memory       │ Memory allocation and access          │ 4   │
//! │ control_flow │ Branches, switches, terminators       │ 5   │
//! │ aggregate    │ Struct/array value manipulation       │ 3   │
//! │ constants    │ Constant, zero, and undefined values  │ 3   │
//! │ symbol       │ Functions, globals, addressof         │ 3   │
//! │ call         │ Function and intrinsic calls          │ 2   │
//! │ select       │ Conditional selection                 │ 1   │
//! │ asm          │ Inline assembly operations            │ 2   │
//! │ va           │ Variadic argument handling            │ 1   │
//! └──────────────┴───────────────────────────────────────┴─────┘
//! ```
//!
//! # Verification Strategy
//!
//! LLVM dialect operations use **comprehensive type and structural verification**.
//! This rigorous approach is appropriate because LLVM IR has strict typing rules
//! and early error detection significantly improves the developer experience.
//!
//! ## Verification by Category
//!
//! ```text
//! ┌──────────────┬─────┬─────────────────────────────────────────────────────────┐
//! │ Category     │ Ops │ Verification                                            │
//! ├──────────────┼─────┼─────────────────────────────────────────────────────────┤
//! │ Arithmetic   │ 19  │ ✅ Full: operand types match, result matches,           │
//! │              │     │    integer/float type check, fastmath flags             │
//! │ Comparison   │  2  │ ✅ Full: predicate attr, result is i1, operand types    │
//! │ Cast         │ 13  │ ✅ Full: width relationships (sext/zext/trunc/fpext),   │
//! │              │     │    type verification (int↔float, ptr↔int)               │
//! │ Memory       │  4  │ ✅ Full: pointer types, element types, GEP indices      │
//! │ Control Flow │  5  │ ✅ Full: successors valid, block arg matching,          │
//! │              │     │    condition is i1, switch case types                   │
//! │ Aggregate    │  3  │ ✅ Full: index bounds, element type consistency         │
//! │ Constants    │  3  │ ✅ Good: type attributes present                        │
//! │ Symbol       │  3  │ ✅ Full: function type validity, entry block args       │
//! │ Call         │  2  │ ✅ Full: callee type, argument types match, varargs     │
//! │ Select       │  1  │ ✅ Full: condition is i1, true/false types match        │
//! │ Asm          │  2  │ ✅ Good: structural verification                        │
//! │ VA           │  1  │ ✅ Good: structural verification                        │
//! └──────────────┴─────┴─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## What IS Verified
//!
//! Every LLVM operation verifies:
//!
//! - **Operand count**: Correct number of operands for the operation.
//! - **Result count**: Correct number of results produced.
//! - **Type compatibility**: Operand types match expected patterns.
//! - **Attribute presence**: Required attributes (predicates, flags) exist.
//! - **Special constraints**: Width relationships for casts, i1 for conditions, etc.
//!
//! ## Key Verification Examples
//!
//! - **Integer arithmetic** (`add`, `sub`, `mul`, etc.): Both operands must be
//!   the same integer type; result type must match operand type.
//! - **Floating arithmetic** (`fadd`, `fsub`, etc.): Same as integer, but for
//!   float types; optional fastmath flags checked.
//! - **Cast operations** (`sext`, `zext`, `trunc`): Source and destination widths
//!   verified (e.g., trunc requires destination < source).
//! - **Comparisons** (`icmp`, `fcmp`): Predicate attribute required; result must
//!   be `i1`; operands must match.
//! - **Control flow** (`cond_br`): Condition must be `i1`; successor block
//!   arguments must type-match.
//! - **GEP**: Base must be pointer; indices verified against struct/array types.
//! - **Calls**: Argument types match callee signature; result type matches return.
//!
//! ## Why Comprehensive Verification?
//!
//! 1. **LLVM IR is strictly typed**: Type mismatches cause cryptic LLVM errors
//!    or silent miscompilation. Catching errors early is critical.
//!
//! 2. **User-constructible**: Unlike NVVM, LLVM ops may be constructed
//!    programmatically, so validation at construction time is valuable.
//!
//! 3. **Foundation for lowering**: MIR → LLVM lowering must produce valid IR.
//!    Verification catches bugs in the lowering passes.
//!
//! 4. **Clear error messages**: Verification failures identify the exact
//!    operation and constraint violated.
//!
//! # Usage
//!
//! All operations are re-exported at this module level for convenient access:
//!
//! ```ignore
//! use dialect_llvm::ops::{AddOp, LoadOp, FuncOp};
//! ```

use pliron::context::Context;

pub mod aggregate;
pub mod arithmetic;
pub mod asm;
pub mod atomic;
pub mod call;
pub mod cast;
pub mod comparison;
pub mod constants;
pub mod control_flow;
pub mod memory;
pub mod select;
pub mod symbol;
pub mod va;

// Re-export all operations for convenient access
pub use aggregate::*;
pub use arithmetic::*;
pub use asm::*;
pub use atomic::*;
pub use call::*;
pub use cast::*;
pub use comparison::*;
pub use constants::*;
pub use control_flow::*;
pub use memory::*;
pub use select::*;
pub use symbol::*;
pub use va::*;

/// Attribute key for LLVM function type.
pub mod func_op_attr_names {
    use pliron::identifier::Identifier;
    use std::sync::LazyLock;

    /// Attribute key for the LLVM function type.
    pub static ATTR_KEY_LLVM_FUNC_TYPE: LazyLock<Identifier> =
        LazyLock::new(|| "llvm_func_type".try_into().unwrap());
}

/// Attribute keys for LLVM global variables.
pub mod global_op_attr_names {
    use pliron::identifier::Identifier;
    use std::sync::LazyLock;

    /// Attribute key for the global variable's initializer value.
    pub static ATTR_KEY_GLOBAL_INITIALIZER: LazyLock<Identifier> =
        LazyLock::new(|| "global_initializer".try_into().unwrap());

    /// Attribute key for the global variable's LLVM type.
    pub static ATTR_KEY_LLVM_GLOBAL_TYPE: LazyLock<Identifier> =
        LazyLock::new(|| "llvm_global_type".try_into().unwrap());
}

/// Register all LLVM dialect operations into the given context.
pub fn register(ctx: &mut Context) {
    arithmetic::register(ctx);
    atomic::register(ctx);
    comparison::register(ctx);
    cast::register(ctx);
    memory::register(ctx);
    control_flow::register(ctx);
    aggregate::register(ctx);
    constants::register(ctx);
    symbol::register(ctx);
    call::register(ctx);
    select::register(ctx);
    asm::register(ctx);
    va::register(ctx);
}
