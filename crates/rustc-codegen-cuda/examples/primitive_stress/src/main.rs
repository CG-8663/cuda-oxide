/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Primitive scalar stress test.
//!
//! Exercises primitive types and methods that are easy to miss in MIR import
//! and lowering: `char`, `u128`/`i128`, pointer-sized integers, and rustc
//! bit-manipulation intrinsics used by primitive integer methods.
//!
//! Run: cargo oxide run primitive_stress

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_launch;

/// Checks that `char` imports as a 32-bit scalar and casts cleanly to `u32`.
#[kernel]
pub fn test_char(mut out: DisjointSlice<u32>) {
    if thread::index_1d().get() == 0 {
        let ascii: char = 'A';
        let wide: char = '\u{1f980}';
        let value = (ascii as u32)
            .wrapping_add(wide as u32)
            .wrapping_add(core::mem::size_of::<char>() as u32);

        unsafe {
            *out.get_unchecked_mut(0) = value;
        }
    }
}

/// Checks wide integer constants, arithmetic, shifts, and argument passing.
#[kernel]
pub fn test_u128_i128(a: u128, b: u128, c: i128, mut out: DisjointSlice<u64>) {
    if thread::index_1d().get() == 0 {
        let unsigned = a
            .wrapping_mul(3)
            .wrapping_add(b)
            .wrapping_mul(0x1_0001_u128)
            .wrapping_add(b >> 111)
            ^ 0xfeed_face_cafe_beef_0123_4567_89ab_cdef_u128;
        let signed = c.wrapping_mul(-7).wrapping_add(0x1234_5678_9abc_def0_i128);
        let signed_bits = signed as u128;

        unsafe {
            *out.get_unchecked_mut(0) = unsigned as u64;
            *out.get_unchecked_mut(1) = (unsigned >> 64) as u64;
            *out.get_unchecked_mut(2) = signed_bits as u64;
            *out.get_unchecked_mut(3) = (signed_bits >> 64) as u64;
        }
    }
}

/// Checks target pointer-sized integer import and arithmetic.
#[kernel]
pub fn test_pointer_sized(a: usize, b: isize, mut out: DisjointSlice<u64>) {
    if thread::index_1d().get() == 0 {
        let unsigned = a
            .wrapping_mul(5)
            .wrapping_add(core::mem::size_of::<usize>());
        let signed = b
            .wrapping_mul(-3)
            .wrapping_sub(core::mem::size_of::<isize>() as isize);

        unsafe {
            *out.get_unchecked_mut(0) = unsigned as u64;
            *out.get_unchecked_mut(1) = signed as u64;
        }
    }
}

/// Checks primitive integer methods that call rustc bit intrinsics in libcore.
#[kernel]
pub fn test_bit_intrinsics(a: u128, b: u64, c: u32, mut out: DisjointSlice<u64>) {
    if thread::index_1d().get() == 0 {
        let wide = a.rotate_left(17) ^ a.rotate_right(29) ^ a.swap_bytes() ^ a.reverse_bits();
        let wide_counts = (a.count_ones() as u64) | ((a.leading_zeros() as u64) << 32);
        let wide_trailing = a.trailing_zeros() as u64;

        let mid = b.rotate_left(7) ^ b.rotate_right(13) ^ b.swap_bytes() ^ b.reverse_bits();
        let narrow = c.rotate_left(5) ^ c.rotate_right(11) ^ c.swap_bytes() ^ c.reverse_bits();
        let narrow_counts = (c.count_ones() as u64)
            | ((c.leading_zeros() as u64) << 32)
            | ((c.trailing_zeros() as u64) << 48);

        unsafe {
            *out.get_unchecked_mut(0) = wide as u64;
            *out.get_unchecked_mut(1) = (wide >> 64) as u64;
            *out.get_unchecked_mut(2) = wide_counts;
            *out.get_unchecked_mut(3) = wide_trailing;
            *out.get_unchecked_mut(4) = mid;
            *out.get_unchecked_mut(5) = narrow as u64;
            *out.get_unchecked_mut(6) = narrow_counts;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Primitive Stress Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module_from_file("primitive_stress.ptx")?;
    let cfg = LaunchConfig::for_num_elems(1);

    let mut passed = 0u32;
    let mut failed = 0u32;

    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        cuda_launch! {
            kernel: test_char,
            stream: stream, module: module, config: cfg,
            args: [slice_mut(out)]
        }?;
        let result = out.to_host_vec(&stream)?[0];
        let expected = ('A' as u32)
            .wrapping_add('\u{1f980}' as u32)
            .wrapping_add(core::mem::size_of::<char>() as u32);
        check(
            "char constants and casts",
            result,
            expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let a = 0x8000_0000_0000_0000_0000_0000_0000_0011_u128;
        let b = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210_u128;
        let c = -0x1234_5678_9abc_def_i128;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 4)?;
        cuda_launch! {
            kernel: test_u128_i128,
            stream: stream, module: module, config: cfg,
            args: [a, b, c, slice_mut(out)]
        }?;
        let result = out.to_host_vec(&stream)?;
        let unsigned = a
            .wrapping_mul(3)
            .wrapping_add(b)
            .wrapping_mul(0x1_0001_u128)
            .wrapping_add(b >> 111)
            ^ 0xfeed_face_cafe_beef_0123_4567_89ab_cdef_u128;
        let signed_bits = c.wrapping_mul(-7).wrapping_add(0x1234_5678_9abc_def0_i128) as u128;
        let expected = [
            unsigned as u64,
            (unsigned >> 64) as u64,
            signed_bits as u64,
            (signed_bits >> 64) as u64,
        ];
        check_slice(
            "u128/i128 arithmetic",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let a = 0x8000_1234_usize;
        let b = -12345_isize;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 2)?;
        cuda_launch! {
            kernel: test_pointer_sized,
            stream: stream, module: module, config: cfg,
            args: [a, b, slice_mut(out)]
        }?;
        let result = out.to_host_vec(&stream)?;
        let expected = [
            a.wrapping_mul(5)
                .wrapping_add(core::mem::size_of::<usize>()) as u64,
            b.wrapping_mul(-3)
                .wrapping_sub(core::mem::size_of::<isize>() as isize) as u64,
        ];
        check_slice(
            "usize/isize arithmetic",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    {
        let a = 0x8000_0000_0000_0000_0123_4567_89ab_cdef_u128;
        let b = 0x0123_4567_89ab_cdef_u64;
        let c = 0x8020_0401_u32;
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, 7)?;
        cuda_launch! {
            kernel: test_bit_intrinsics,
            stream: stream, module: module, config: cfg,
            args: [a, b, c, slice_mut(out)]
        }?;
        let result = out.to_host_vec(&stream)?;
        let wide = a.rotate_left(17) ^ a.rotate_right(29) ^ a.swap_bytes() ^ a.reverse_bits();
        let wide_counts = (a.count_ones() as u64) | ((a.leading_zeros() as u64) << 32);
        let mid = b.rotate_left(7) ^ b.rotate_right(13) ^ b.swap_bytes() ^ b.reverse_bits();
        let narrow = c.rotate_left(5) ^ c.rotate_right(11) ^ c.swap_bytes() ^ c.reverse_bits();
        let narrow_counts = (c.count_ones() as u64)
            | ((c.leading_zeros() as u64) << 32)
            | ((c.trailing_zeros() as u64) << 48);
        let expected = [
            wide as u64,
            (wide >> 64) as u64,
            wide_counts,
            a.trailing_zeros() as u64,
            mid,
            narrow as u64,
            narrow_counts,
        ];
        check_slice(
            "bit intrinsic methods",
            &result,
            &expected,
            &mut passed,
            &mut failed,
        );
    }

    println!("\n=== Results ===");
    println!("Passed: {passed}");
    println!("Failed: {failed}");

    if failed == 0 {
        println!("\nPASS: primitive scalar checks matched");
        Ok(())
    } else {
        eprintln!("\nFAIL: {failed} primitive scalar checks failed");
        std::process::exit(1);
    }
}

fn check<T: Eq + std::fmt::Debug>(
    name: &str,
    got: T,
    expected: T,
    passed: &mut u32,
    failed: &mut u32,
) {
    if got == expected {
        println!("PASS {name}: {got:?}");
        *passed += 1;
    } else {
        println!("FAIL {name}: got {got:?}, expected {expected:?}");
        *failed += 1;
    }
}

fn check_slice<T: Eq + std::fmt::Debug>(
    name: &str,
    got: &[T],
    expected: &[T],
    passed: &mut u32,
    failed: &mut u32,
) {
    if got == expected {
        println!("PASS {name}: {got:?}");
        *passed += 1;
    } else {
        println!("FAIL {name}: got {got:?}, expected {expected:?}");
        *failed += 1;
    }
}
