/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA kernel launch builder with argument marshalling.
//!
//! [`AsyncKernelLaunch`] accumulates a kernel function reference, launch
//! configuration, and type-erased argument pointers, then submits the launch
//! through the CUDA driver when executed as a [`DeviceOperation`].
//!
//! Arguments are heap-allocated via [`KernelArgument::push_arg`] and freed when
//! the launcher is dropped after submission. This keeps the pointed-to values
//! alive until `cuLaunchKernel` / `cuLaunchKernelEx` has copied the launch
//! parameter values out of the host-side argument array.
//!
//! [`DeviceOperation`]: crate::device_operation::DeviceOperation

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOperation, ExecutionContext};
use crate::error::DeviceError;
use crate::scheduling_policies::SchedulingPolicy;
use cuda_core::{CudaFunction, CudaStream, LaunchConfig};
use std::ffi::c_void;
use std::future::IntoFuture;
use std::sync::Arc;

/// Builder that accumulates kernel arguments and submits a CUDA kernel launch.
///
/// Implements [`DeviceOperation`] so it can be composed with other operations
/// and scheduled onto any stream. Also implements [`IntoFuture`] for `.await`
/// syntax.
#[derive(Debug)]
pub struct AsyncKernelLaunch {
    /// Handle to the compiled device function.
    pub func: Arc<CudaFunction>,
    /// Heap-allocated, type-erased argument pointers passed to the CUDA driver.
    ///
    /// Storage for these values is kept alive until launch submission returns,
    /// after which the driver has already copied the parameter payload.
    pub args: Vec<*mut c_void>,
    /// Grid/block dimensions and shared memory size. Must be set before launch.
    cfg: Option<LaunchConfig>,
}

/// # Safety
///
/// The `*mut c_void` pointers in `args` are heap-allocated boxes that do not
/// alias mutable state. The `Arc<CudaFunction>` is `Send + Sync`.
unsafe impl Send for AsyncKernelLaunch {}

/// Reclaims heap-allocated argument storage.
///
/// Each pointer in `args` was produced by `Box::into_raw`, so reconstructing
/// and dropping the boxes here is safe. By the time `Drop` runs, launch
/// submission has either not happened yet or has already copied the parameter
/// values into the driver's internal launch packet.
impl Drop for AsyncKernelLaunch {
    fn drop(&mut self) {
        let _ = self
            .args
            .iter()
            .map(|&arg| unsafe { Box::from_raw(arg as *mut usize) })
            .collect::<Vec<_>>();
    }
}

impl AsyncKernelLaunch {
    /// Creates a launcher for `func` with no arguments and no launch config.
    pub fn new(func: Arc<CudaFunction>) -> Self {
        Self {
            func,
            args: Vec::new(),
            cfg: None,
        }
    }

    /// Appends a kernel argument. The value is heap-allocated and its pointer
    /// stored for the driver call.
    ///
    /// Scalars like `u32`, `f32`, `u64` etc. are auto-boxed -- no need to
    /// wrap them in `Box::new`.
    ///
    /// The allocated storage remains alive until launch submission finishes or
    /// the builder is dropped without launching.
    #[inline(always)]
    pub fn push_arg<T: KernelArgument>(&mut self, arg: T) -> &mut Self {
        arg.push_arg(self);
        self
    }

    /// Appends multiple kernel arguments at once from a tuple.
    ///
    /// Equivalent to chained [`push_arg`](Self::push_arg) calls but allows
    /// grouping all arguments in a single expression:
    ///
    /// ```ignore
    /// launch.push_args((m, n, k, alpha, a_ptr, a_len, b_ptr, b_len, beta, c_ptr, c_len));
    /// ```
    ///
    /// Supports tuples up to 32 elements.
    #[inline(always)]
    pub fn push_args<T: KernelArguments>(&mut self, args: T) -> &mut Self {
        args.push_args(self);
        self
    }

    /// Sets the grid/block dimensions and shared memory size for the launch.
    pub fn set_launch_config(&mut self, cfg: LaunchConfig) -> &mut Self {
        self.cfg = Some(cfg);
        self
    }

    /// Submits the kernel to `stream` via `cuLaunchKernel`.
    ///
    /// # Safety
    ///
    /// - `self.func` must refer to a kernel loaded from the same CUDA context
    ///   that owns `stream`.
    /// - All argument pointers in `self.args` must point to correctly typed and
    ///   aligned host-side values for the kernel's formal parameters.
    /// - The pointed-to argument values must remain valid until launch
    ///   submission returns. The stream-aware launch helper binds the correct
    ///   context before calling into the CUDA driver.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::Launch`] if no launch config was set, or
    /// [`DeviceError::Driver`] if context binding or launch submission fails.
    unsafe fn launch(mut self, stream: &Arc<CudaStream>) -> Result<(), DeviceError> {
        let cfg = self
            .cfg
            .ok_or_else(|| DeviceError::Launch("Launch config not set.".to_string()))?;
        unsafe {
            cuda_core::launch_kernel_on_stream(
                self.func.as_ref(),
                cfg.grid_dim,
                cfg.block_dim,
                cfg.shared_mem_bytes,
                stream.as_ref(),
                &mut self.args,
            )
        }
        .map_err(DeviceError::Driver)?;
        Ok(())
    }
}

/// Trait for types that can be marshalled into a CUDA kernel argument list.
///
/// Implementors heap-allocate the value and push a `*mut c_void` into the
/// launcher's argument vector.
///
/// Implemented for all common scalar primitives (`u8`–`u64`, `i8`–`i64`,
/// `f32`, `f64`, `usize`, `isize`, `bool`) so callers can write
/// `.push_arg(42u32)` without manual `Box::new`.
pub trait KernelArgument {
    /// Heap-allocates `self` and appends the pointer to `launcher.args`.
    fn push_arg(self, launcher: &mut AsyncKernelLaunch);
}

/// Passes the box's raw pointer directly as a kernel argument.
///
/// This is the low-level escape hatch. Prefer passing scalars directly -- they
/// are auto-boxed via the blanket scalar impls.
impl<T> KernelArgument for Box<T> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        let r = Box::into_raw(self);
        launcher.args.push(r as *mut _);
    }
}

macro_rules! impl_scalar_kernel_arg {
    ($($t:ty),*) => {
        $(
            impl KernelArgument for $t {
                #[inline(always)]
                fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
                    Box::new(self).push_arg(launcher);
                }
            }
        )*
    };
}

impl_scalar_kernel_arg!(
    u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64, bool
);

// ---------------------------------------------------------------------------
// KernelArguments — push multiple heterogeneous args in a single call
// ---------------------------------------------------------------------------

/// Trait for a group of kernel arguments that can be pushed together.
///
/// Implemented for tuples of [`KernelArgument`] types up to arity 32, enabling
/// `launch.push_args((dim_m, dim_n, alpha, ptr, len))` as an alternative to
/// chained `.push_arg()` calls.
#[diagnostic::on_unimplemented(
    message = "cannot push `{Self}` as kernel arguments",
    note = "KernelArguments is implemented for tuples of KernelArgument types up to 32 elements"
)]
pub trait KernelArguments {
    /// Pushes every element into `launcher` in order.
    fn push_args(self, launcher: &mut AsyncKernelLaunch);
}

macro_rules! impl_kernel_args_tuple {
    // Base case: empty tuple
    () => {
        impl KernelArguments for () {
            #[inline(always)]
            fn push_args(self, _launcher: &mut AsyncKernelLaunch) {}
        }
    };
    // Recursive case: (A, B, C, ...) where each element is a KernelArgument
    ($($idx:tt : $T:ident),+) => {
        impl<$($T: KernelArgument),+> KernelArguments for ($($T,)+) {
            #[inline(always)]
            fn push_args(self, launcher: &mut AsyncKernelLaunch) {
                $(launcher.push_arg(self.$idx);)+
            }
        }
    };
}

impl_kernel_args_tuple!();
impl_kernel_args_tuple!(0: A);
impl_kernel_args_tuple!(0: A, 1: B);
impl_kernel_args_tuple!(0: A, 1: B, 2: C);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD, 30: AE);
impl_kernel_args_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P, 16: Q, 17: R, 18: S, 19: T, 20: U, 21: V, 22: W, 23: X, 24: Y, 25: Z, 26: AA, 27: AB, 28: AC, 29: AD, 30: AE, 31: AF);

/// Launches the kernel on the stream bound to the execution context.
impl DeviceOperation for AsyncKernelLaunch {
    type Output = ();

    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        unsafe { self.launch(ctx.get_cuda_stream()) }
    }
}

/// Schedules the kernel launch via the thread-local default scheduling policy.
impl IntoFuture for AsyncKernelLaunch {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), AsyncKernelLaunch>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}
