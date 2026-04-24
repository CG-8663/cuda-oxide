/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA module and function management (RAII, PTX/cubin loading).
//!
//! A [`CudaModule`] wraps a `CUmodule` loaded from PTX source or a cubin file.
//! [`CudaFunction`] extracts a kernel entry point from a loaded module by
//! symbol name. Both types are reference-counted and tie their lifetime to the
//! parent [`CudaContext`] / [`CudaModule`] respectively.
//!
//! # Typical workflow
//!
//! ```ignore
//! let ctx = CudaContext::new(0)?;
//! let module = ctx.load_module_from_ptx_src(ptx)?;
//! let kernel = module.load_function("my_kernel")?;
//! ```

use crate::context::CudaContext;
use crate::error::{DriverError, IntoResult};
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// An RAII wrapper around a `CUmodule` handle.
///
/// Holds an `Arc<CudaContext>` to ensure the context outlives the module.
/// Unloaded automatically via `cuModuleUnload` on [`Drop`].
#[derive(Debug)]
pub struct CudaModule {
    /// Raw CUDA module handle.
    pub(crate) cu_module: cuda_bindings::CUmodule,
    /// Owning context. Kept alive for the lifetime of this module.
    pub(crate) ctx: Arc<CudaContext>,
}

/// # Safety
///
/// `CUmodule` handles are not thread-local. The CUDA driver permits querying
/// functions from a module on any thread, provided the owning context is bound.
unsafe impl Send for CudaModule {}
/// See [`Send`] impl.
unsafe impl Sync for CudaModule {}

/// Unloads the module on drop.
///
/// Binds the context to the current thread first (required by
/// `cuModuleUnload`). Errors are recorded on the context rather than
/// panicking.
impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { cuda_bindings::cuModuleUnload(self.cu_module).result() });
    }
}

impl CudaContext {
    /// JIT-compiles PTX source and loads the resulting module into this
    /// context.
    ///
    /// `ptx_src` must be a valid, null-terminator-free PTX string. The driver
    /// performs JIT compilation targeting the current device architecture.
    ///
    /// # Panics
    ///
    /// Panics if `ptx_src` contains interior null bytes.
    pub fn load_module_from_ptx_src(
        self: &Arc<Self>,
        ptx_src: &str,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let c_src = CString::new(ptx_src).unwrap();
        let cu_module = unsafe {
            let mut cu_module = MaybeUninit::uninit();
            cuda_bindings::cuModuleLoadData(cu_module.as_mut_ptr(), c_src.as_ptr() as *const _)
                .result()?;
            cu_module.assume_init()
        };
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }

    /// Loads a module from a cubin or PTX file on disk.
    ///
    /// `filename` is the filesystem path. The driver selects the loader based
    /// on file contents (PTX text or cubin ELF).
    ///
    /// # Panics
    ///
    /// Panics if `filename` contains interior null bytes.
    pub fn load_module_from_file(
        self: &Arc<Self>,
        filename: &str,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let c_str = CString::new(filename).unwrap();
        let mut cu_module = MaybeUninit::uninit();
        let cu_module = unsafe {
            cuda_bindings::cuModuleLoad(cu_module.as_mut_ptr(), c_str.as_ptr()).result()?;
            cu_module.assume_init()
        };
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }
}

/// A handle to a device kernel entry point within a loaded [`CudaModule`].
///
/// Holds an `Arc<CudaModule>` so the module (and transitively the context)
/// remains loaded for the lifetime of this handle. Cloning is cheap (just an
/// `Arc` bump).
#[derive(Debug, Clone)]
pub struct CudaFunction {
    /// Raw CUDA function handle.
    pub(crate) cu_function: cuda_bindings::CUfunction,
    /// Owning module. Prevents unloading while this function handle exists.
    #[allow(unused)]
    pub(crate) module: Arc<CudaModule>,
}

/// # Safety
///
/// `CUfunction` handles are derived from a `CUmodule` and valid in any thread
/// that has the owning context bound.
unsafe impl Send for CudaFunction {}
/// See [`Send`] impl.
unsafe impl Sync for CudaFunction {}

impl CudaModule {
    /// Looks up a kernel entry point by `fn_name` in this module.
    ///
    /// The returned [`CudaFunction`] holds an `Arc` back to this module,
    /// preventing unloading while the handle is live.
    ///
    /// This method first binds the module's owning context to the calling
    /// thread, then performs `cuModuleGetFunction`. That makes it safe to look
    /// up functions from any host thread, provided the module and its context
    /// are still alive.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the module's context fails or if
    /// `cuModuleGetFunction` cannot resolve `fn_name` in this module.
    ///
    /// # Panics
    ///
    /// Panics if `fn_name` contains interior null bytes.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        self.ctx.bind_to_thread()?;
        let c_name = CString::new(fn_name).unwrap();
        let cu_function = unsafe {
            let mut cu_function = MaybeUninit::uninit();
            cuda_bindings::cuModuleGetFunction(
                cu_function.as_mut_ptr(),
                self.cu_module,
                c_name.as_ptr(),
            )
            .result()?;
            cu_function.assume_init()
        };
        Ok(CudaFunction {
            cu_function,
            module: self.clone(),
        })
    }
}

impl CudaFunction {
    /// Returns the raw `CUfunction` handle.
    ///
    /// # Safety
    ///
    /// The returned handle is invalidated if the parent [`CudaModule`] is
    /// dropped. Because `CudaFunction` holds an `Arc<CudaModule>`, this
    /// cannot happen while `self` is alive -- but the raw handle must not
    /// be stashed beyond the lifetime of this `CudaFunction`.
    pub unsafe fn cu_function(&self) -> cuda_bindings::CUfunction {
        self.cu_function
    }
}
