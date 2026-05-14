/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stable 128-bit type identifiers for kernel PTX naming.
//!
//! The host needs to compute the same per-type hash that the backend computes
//! via `tcx.type_id_hash(ty).as_u128()`. The stable [`core::any::TypeId::of`]
//! API would force a `T: 'static` bound on the kernel marker, which would in
//! turn reject perfectly valid non-`'static` borrowing closures (e.g. a kernel
//! launcher capturing `&[f32]` from a stack frame the caller keeps alive
//! across the launch). The `core::intrinsics::type_id` form has bound
//! `T: ?Sized` — i.e. no `'static` requirement — and produces the exact same
//! 128-bit value that `tcx.type_id_hash` does for that type, because both go
//! through the same `erase_and_anonymize_regions` + stable-hash pipeline.
//!
//! Framing note for future contributors: `core::intrinsics::type_id` is an
//! internal API and requires `#![feature(core_intrinsics)]` on the owning
//! crate. cuda-oxide already ships against `rustc_private` and pins a
//! nightly toolchain, so this is inside our existing risk surface — but the
//! helper cannot be lifted into a stable-feeling utility crate without
//! re-introducing the feature gate there.

use core::any::TypeId;

/// Returns the same 128-bit hash that the cuda-oxide backend uses for
/// kernel export names.
///
/// At runtime the value is just the 16 raw hash bytes (see the layout
/// comment in `core::any::TypeId`). The intrinsic is const-evaluated by
/// rustc using its internal `Ty<'tcx>` representation, so the call site
/// only ever sees a constant `u128`.
///
/// Bound is intentionally `T: ?Sized` (not `T: 'static`). The typed launch
/// path must keep accepting non-`'static` borrowing closures, the same way
/// the legacy `type_name`-based path did. Adding `'static` here would
/// silently tighten the typed API without enforcing the actual launch-
/// outlives-borrow invariant — that responsibility still sits with the
/// caller (the borrow must outlive `stream.synchronize()`).
#[inline]
pub fn type_id_u128<T: ?Sized>() -> u128 {
    let id = const { core::intrinsics::type_id::<T>() };
    unsafe { core::mem::transmute::<TypeId, u128>(id) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_types_hash_distinctly() {
        assert_ne!(type_id_u128::<f32>(), type_id_u128::<i32>());
        assert_ne!(type_id_u128::<u32>(), type_id_u128::<i32>());
    }

    #[test]
    fn same_type_hashes_stably() {
        let a = type_id_u128::<f32>();
        let b = type_id_u128::<f32>();
        assert_eq!(a, b);
    }

    #[test]
    fn static_borrow_collides_with_free_borrow() {
        // Confirms erase_and_anonymize_regions: free lifetimes (including
        // 'static) all hash to the same value.
        fn free<'a>() -> u128 {
            type_id_u128::<&'a i32>()
        }
        assert_eq!(type_id_u128::<&'static i32>(), free());
    }

    #[test]
    fn distinct_closure_literals_hash_distinctly() {
        let factor = 2.5f32;
        let cl1 = move |x: f32| x * factor;
        let cl2 = move |x: f32| x * factor;
        fn id<T>(_: &T) -> u128 {
            type_id_u128::<T>()
        }
        assert_ne!(id(&cl1), id(&cl2));
    }
}
