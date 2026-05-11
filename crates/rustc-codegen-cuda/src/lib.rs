/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # rustc_codegen_cuda: Unified Host/Device Compilation Backend
//!
//! A custom rustc codegen backend that enables single-source CUDA compilation for Rust,
//! similar to NVIDIA's nvc++ compiler for C++. This backend intercepts rustc's code
//! generation phase to extract device code and compile it to PTX while delegating
//! host code compilation to the standard LLVM backend.
//!
//! ## Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                              RUSTC COMPILATION                                  │
//! │                                                                                 │
//! │   Source Code (.rs)                                                             │
//! │         │                                                                       │
//! │         ▼                                                                       │
//! │   ┌───────────────────────────────────────────────────────────────────────┐     │
//! │   │                         RUSTC FRONTEND                                │     │
//! │   │                                                                       │     │
//! │   │   Parsing ──▶ HIR ──▶ Type Check ──▶ MIR Generation ──▶ MIR Passes    │     │
//! │   │                                                                       │     │
//! │   │   Outputs: Fully monomorphized, OPTIMIZED MIR                         │     │
//! │   │            (affected by -C opt-level, -Z mir-enable-passes)           │     │
//! │   └───────────────────────────────────────────────────────────────────────┘     │
//! │         │                                                                       │
//! │         │  MIR passes have ALREADY run by this point                            │
//! │         │  (including JumpThreading unless disabled)                            │
//! │         ▼                                                                       │
//! │   ┌───────────────────────────────────────────────────────────────────────┐     │
//! │   │                    rustc_codegen_cuda (THIS BACKEND)                  │     │
//! │   │                                                                       │     │
//! │   │   Entry: codegen_crate(TyCtxt) called by rustc                        │     │
//! │   │                                                                       │     │
//! │   │   ┌─────────────────────────────────────────────────────────────┐     │     │
//! │   │   │  1. KERNEL DETECTION                                        │     │     │
//! │   │   │     - Scan CGUs for functions in the reserved namespace     │     │     │
//! │   │   │       `cuda_oxide_kernel_<hash>_*` (set by #[kernel] macro) │     │     │
//! │   │   └─────────────────────────────────────────────────────────────┘     │     │
//! │   │                          │                                            │     │
//! │   │                          ▼                                            │     │
//! │   │   ┌─────────────────────────────────────────────────────────────┐     │     │
//! │   │   │  2. DEVICE FUNCTION COLLECTION (collector.rs)               │     │     │
//! │   │   │     - Start from kernel entry points                        │     │     │
//! │   │   │     - Walk MIR call graph transitively                      │     │     │
//! │   │   │     - Collect all reachable functions from:                 │     │     │
//! │   │   │       • Local crate                                         │     │     │
//! │   │   │       • cuda_device (intrinsics)                            │     │     │
//! │   │   │       • core (iterators, Option, etc.)                      │     │     │
//! │   │   │     - Filter out: fmt::*, panicking::*, intrinsic stubs     │     │     │
//! │   │   └─────────────────────────────────────────────────────────────┘     │     │
//! │   │                          │                                            │     │
//! │   │          ┌───────────────┴───────────────┐                            │     │
//! │   │          ▼                               ▼                            │     │
//! │   │   ┌─────────────────┐           ┌─────────────────────────┐           │     │
//! │   │   │  DEVICE PATH    │           │      HOST PATH          │           │     │
//! │   │   │                 │           │                         │           │     │
//! │   │   │  3. Bridge to   │           │  4. Delegate to         │           │     │
//! │   │   │     stable_mir  │           │     rustc_codegen_llvm  │           │     │
//! │   │   │                 │           │                         │           │     │
//! │   │   │  device_codegen │           │  Standard LLVM backend  │           │     │
//! │   │   │  .rs handles    │           │  handles all host code  │           │     │
//! │   │   └────────┬────────┘           └────────────┬────────────┘           │     │
//! │   │            │                                 │                        │     │
//! │   │            ▼                                 ▼                        │     │
//! │   │   ┌─────────────────┐           ┌─────────────────────────┐           │     │
//! │   │   │ cuda-oxide      │           │  Host Object Files      │           │     │
//! │   │   │ Pipeline:       │           │  (.o / .rlib)           │           │     │
//! │   │   │                 │           │                         │           │     │
//! │   │   │ dialect-mir     │           │  Standard x86_64 code   │           │     │
//! │   │   │     ▼ (mem2reg) │           │                         │           │     │
//! │   │   │ dialect-llvm    │           │                         │           │     │
//! │   │   │     ▼           │           │                         │           │     │
//! │   │   │ LLVM IR (.ll)   │           │                         │           │     │
//! │   │   │     ▼ (llc)     │           │                         │           │     │
//! │   │   │ PTX (.ptx)      │           │                         │           │     │
//! │   │   └─────────────────┘           └─────────────────────────┘           │     │
//! │   │                                                                       │     │
//! │   └───────────────────────────────────────────────────────────────────────┘     │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## How MIR is Obtained
//!
//! When `codegen_crate()` is called, rustc has ALREADY:
//!
//! 1. **Parsed** the source code
//! 2. **Type checked** everything
//! 3. **Generated MIR** for all functions
//! 4. **Run MIR optimization passes** based on `-C opt-level` and `-Z mir-enable-passes`
//!
//! We receive a `TyCtxt` containing **optimized MIR**. The MIR we get depends entirely
//! on what flags were passed to rustc:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                           MIR OPTIMIZATION PASSES                               │
//! │                                                                                 │
//! │   User runs:  rustc -C opt-level=3 -Z mir-enable-passes=-JumpThreading ...      │
//! │                         │                        │                              │
//! │                         ▼                        ▼                              │
//! │              ┌──────────────────┐    ┌──────────────────────────┐               │
//! │              │ Enable passes:   │    │ Disable passes:          │               │
//! │              │ - Inlining       │    │ - JumpThreading (MUST!)  │               │
//! │              │ - ConstProp      │    │                          │               │
//! │              │ - GVN            │    │                          │               │
//! │              │ - DeadCode       │    │                          │               │
//! │              │ - etc.           │    │                          │               │
//! │              └──────────────────┘    └──────────────────────────┘               │
//! │                                                                                 │
//! │   Result: We get MIR that has been through these passes                         │
//! │           This affects BOTH host and device code (same MIR for both)            │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Required Compiler Flags
//!
//! The following flags MUST be passed to rustc when using this backend:
//!
//! | Flag                                  | Purpose                | Why Required                                                                                               |
//! |---------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------|
//! | `-Z mir-enable-passes=-JumpThreading` | Disable JumpThreading  | **CRITICAL**: JumpThreading duplicates barrier calls into branches, breaking GPU synchronization semantics |
//!
//! Recommended for production:
//!
//! | Flag                       | Purpose                  | Why Recommended                                              |
//! |----------------------------|--------------------------|--------------------------------------------------------------|
//! | `-C opt-level=3`           | Maximum MIR optimization | Better inlining, smaller device code                         |
//! | `-C debug-assertions=off`  | Remove debug checks      | `debug_assert!` pulls in fmt code that can't compile for GPU |
//!
//! **Note:** `panic=abort` is **NOT required**. The codegen backend treats all unwind
//! paths as unreachable since the CUDA toolchain does not support unwinding today. This means standard library code
//! compiled without `panic=abort` works fine -- unwind edges are simply ignored.
//!
//! ### Why JumpThreading Must Be Disabled
//!
//! JumpThreading is a MIR optimization that duplicates code to eliminate jumps.
//! This is problematic for GPU code because it can duplicate barrier calls:
//!
//! ```text
//! BEFORE JumpThreading:              AFTER JumpThreading (BROKEN!):
//! ┌─────────────────────────┐        ┌─────────────────────────────────────┐
//! │ bb0:                    │        │ bb0:                                │
//! │   if cond -> bb1, bb2   │        │   if cond -> bb1, bb2               │
//! │                         │        │                                     │
//! │ bb1:                    │        │ bb1:                                │
//! │   a()                   │        │   a()                               │
//! │   goto bb3              │        │   __syncthreads()  ◄─── Thread 0-15 │
//! │                         │        │   c()                               │
//! │ bb2:                    │        │   return                            │
//! │   goto bb3              │        │                                     │
//! │                         │        │ bb2:                                │
//! │ bb3:                    │        │   __syncthreads()  ◄─── Thread 16-31│
//! │   __syncthreads()       │        │   c()                               │
//! │   c()                   │        │   return                            │
//! │   return                │        │                                     │
//! └─────────────────────────┘        └─────────────────────────────────────┘
//!
//! Different threads execute DIFFERENT barrier instances = DEADLOCK!
//! ```
//!
//! ## `no_std` Requirement
//!
//! Kernel crates MUST use `#![no_std]`. The collector enforces this with a
//! single hard rule: **the `std` crate itself is forbidden, every other
//! crate is allowed** (provided it's reachable from a kernel and itself
//! avoids `std`). The check is on the *originating crate*
//! (`tcx.crate_name(def_id.krate)`), not on display paths -- which matters,
//! because rustc's MIR pretty-printer routinely emits `std::*` for items
//! that are merely re-exported from `core`.
//!
//! See [`collector::should_collect_from_crate`] for the exact policy.
//!
//! ### Why `std::*` shows up in MIR dumps (and isn't a problem)
//!
//! Run `cargo oxide pipeline vecadd` (or `atomics`, or most other examples)
//! and the rustc MIR section will be peppered with paths like:
//!
//! ```text
//! _4 = std::option::Option::<&mut f32>::Some(copy _21)
//! _4 = const std::option::Option::<&mut f32>::None
//! _3 = copy _14 as *const std::sync::atomic::Atomic<u32> (PtrToPtr)
//! _4 = std::intrinsics::atomic_xadd::<u32, u32, ...>(move _15, ...) -> ...
//! ```
//!
//! These are **`core` items shown under their `std::` re-export path**.
//! `def_path_str` chooses the most user-visible path, which is usually the
//! `std::*` form. The actual `DefId` lives in `core` (or `core::sync::atomic`,
//! `core::intrinsics`, ...), so the collector's
//! `crate_name(def_id.krate) == "std"` check is `false` and they're collected
//! normally. Treat `std::*` in MIR output as cosmetic; only a hard collector
//! error means actual `std` was reached.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                        CRATE FILTERING FOR DEVICE CODE                          │
//! │                                                                                 │
//! │   Allowed (originating crate, i.e. DefId.krate):                                │
//! │   ┌──────────────────────────────────────────────────────────────────────┐      │
//! │   │ local crate (your kernel code)                                       │      │
//! │   │ cuda_device  (GPU intrinsics)                                        │      │
//! │   │ core         (Option, Result, UnsafeCell, sync::atomic, intrinsics)  │      │
//! │   │ alloc        (Vec / Box, only if you wired up a GPU allocator)       │      │
//! │   │ any other no_std crate, if transitively reachable from a kernel      │      │
//! │   │   (libm, num-traits, your own helper crates, ...)                    │      │
//! │   └──────────────────────────────────────────────────────────────────────┘      │
//! │                                                                                 │
//! │   Forbidden (hard error at collection time):                                    │
//! │   ┌──────────────────────────────────────────────────────────────────────┐      │
//! │   │ std -- only when the *originating* crate is std (not just a display  │      │
//! │   │        re-export). Example: an actual call into std::thread,         │      │
//! │   │        std::fs, std::io, std::sync::Mutex, etc.                      │      │
//! │   └──────────────────────────────────────────────────────────────────────┘      │
//! │                                                                                 │
//! │   When that genuine std call is reached, the collector emits a                  │
//! │   CollectDecision::Forbidden, and process_call_operand aborts compilation       │
//! │   with a formatted error box naming the function -- no silent skip, no          │
//! │   cryptic PTX "undefined symbol" later in the pipeline.                         │
//! │                                                                                 │
//! │   Intentionally skipped (no error, just dropped): `core::fmt::*`,               │
//! │   `core::panicking::*`, and `*::precondition_check`. These are reached          │
//! │   by panic/UB-check paths that can't actually fire at runtime under             │
//! │   panic=abort + `-C debug-assertions=off`.                                      │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Compilation Model
//!
//! **Unified single-source** compilation is fully supported. Device code is marked
//! with `#[kernel]` and the backend automatically splits based on kernel reachability
//! -- no `#[cfg(cuda_device)]` needed.
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, DisjointSlice};
//! use cuda_host::cuda_launch;
//!
//! #[kernel]
//! pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
//!     let idx = thread::index_1d();
//!     if let Some(c_elem) = c.get_mut(idx) {
//!         *c_elem = a[idx.get()] + b[idx.get()];
//!     }
//! }
//!
//! fn main() {
//!     // Host code -- compiled to native x86_64 by LLVM
//!     // Kernel is compiled to PTX by cuda-oxide
//! }
//! ```
//!
//! See `examples/` for working examples.
//!
//! ## Example Usage
//!
//! ```bash
//! # Build the backend
//! cd crates/rustc-codegen-cuda
//! cargo build
//!
//! # Compile a kernel crate with the backend
//! CUDA_OXIDE_VERBOSE=1 rustc \
//!     --edition 2021 \
//!     -C opt-level=3 \
//!     -C debug-assertions=off \
//!     -Z mir-enable-passes=-JumpThreading \
//!     -Z codegen-backend=./target/debug/librustc_codegen_cuda.so \
//!     my_kernel.rs
//! ```
//!
//! ## Environment Variables
//!
//! | Variable               | Effect                               |
//! |------------------------|--------------------------------------|
//! | `CUDA_OXIDE_VERBOSE`   | Print detailed compilation progress  |
//! | `CUDA_OXIDE_DUMP_MIR`  | Dump the `dialect-mir` module        |
//! | `CUDA_OXIDE_DUMP_LLVM` | Dump the `dialect-llvm` module       |
//! | `CUDA_OXIDE_PTX_DIR`   | Override PTX output directory        |
//! | `CUDA_OXIDE_TARGET`    | Override GPU target (e.g., `sm_90a`) |
//!
//! ## Module Structure
//!
//! - [`collector`]: Device function collection via MIR call graph traversal
//! - [`device_codegen`]: Bridge to the cuda-oxide pipeline (MIR → PTX)
//! - [`layout`]: Unified type layouts for host/device ABI compatibility

#![feature(rustc_private)]
#![allow(unused_imports)]
#![allow(dead_code)]

// Import rustc internal crates
extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

// rustc_public (stable MIR) and its bridge - for calling mir-importer
extern crate rustc_public;
extern crate rustc_public_bridge;

// The standard LLVM backend - we delegate host codegen to this
extern crate rustc_codegen_llvm;

mod collector;
mod device_codegen;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, ModuleKind};
use rustc_data_structures::fx::FxIndexMap;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::Session;
use rustc_session::config::OutputFilenames;
use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The CUDA codegen backend.
///
/// This backend wraps `rustc_codegen_llvm` for host code while adding
/// device code compilation via cuda-oxide. It implements the [`CodegenBackend`]
/// trait which rustc uses to delegate code generation.
///
/// ## Delegation Strategy
///
/// Rather than reimplementing all of LLVM codegen, we:
/// 1. Intercept `codegen_crate()` to extract and compile device code
/// 2. Delegate ALL other methods to `rustc_codegen_llvm`
///
/// This means host code gets the full, battle-tested LLVM backend while
/// device code goes through our specialized cuda-oxide pipeline.
pub struct CudaCodegenBackend {
    config: CudaCodegenConfig,
    /// The underlying LLVM backend for host code generation
    llvm_backend: Box<dyn CodegenBackend>,
}

struct CudaOngoingCodegen {
    host: Box<dyn Any>,
    artifact_objects: Vec<PathBuf>,
}

static ARTIFACT_OBJECT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Configuration for the CUDA codegen backend.
///
/// All configuration is read from environment variables at backend load time.
/// This avoids the need to thread configuration through rustc's argument parsing.
#[derive(Clone, Default)]
pub struct CudaCodegenConfig {
    /// Print detailed compilation progress to stderr.
    pub verbose: bool,
    /// Dump raw rustc MIR before translation (requires --verbose flag).
    pub dump_rustc_mir: bool,
    /// Dump the `dialect-mir` module during device compilation.
    pub dump_mir_dialect: bool,
    /// Dump the `dialect-llvm` module during device compilation.
    pub dump_llvm_dialect: bool,
    /// Override PTX output directory (defaults to current directory).
    pub ptx_output_dir: Option<std::path::PathBuf>,
}

impl CudaCodegenConfig {
    /// Load configuration from environment variables.
    ///
    /// | Variable                    | Config Field        |
    /// |-----------------------------|---------------------|
    /// | `CUDA_OXIDE_VERBOSE`        | `verbose`           |
    /// | `CUDA_OXIDE_SHOW_RUSTC_MIR` | `dump_rustc_mir`    |
    /// | `CUDA_OXIDE_DUMP_MIR`       | `dump_mir_dialect`  |
    /// | `CUDA_OXIDE_DUMP_LLVM`      | `dump_llvm_dialect` |
    /// | `CUDA_OXIDE_PTX_DIR`        | `ptx_output_dir`    |
    pub fn from_env() -> Self {
        Self {
            verbose: std::env::var("CUDA_OXIDE_VERBOSE").is_ok(),
            dump_rustc_mir: std::env::var("CUDA_OXIDE_SHOW_RUSTC_MIR").is_ok(),
            dump_mir_dialect: std::env::var("CUDA_OXIDE_DUMP_MIR").is_ok(),
            dump_llvm_dialect: std::env::var("CUDA_OXIDE_DUMP_LLVM").is_ok(),
            ptx_output_dir: std::env::var("CUDA_OXIDE_PTX_DIR")
                .ok()
                .map(std::path::PathBuf::from),
        }
    }
}

impl CodegenBackend for CudaCodegenBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn init(&self, sess: &Session) {
        // Note: Don't log here - init() is called for ALL crates including dependencies.
        // We log in codegen_crate() only when there are kernels to compile.

        // Initialize the underlying LLVM backend
        self.llvm_backend.init(sess);
    }

    fn print_version(&self) {
        println!(
            "rustc_codegen_cuda version {} (wrapping rustc_codegen_llvm)",
            env!("CARGO_PKG_VERSION")
        );
        self.llvm_backend.print_version();
    }

    fn target_cpu(&self, sess: &Session) -> String {
        self.llvm_backend.target_cpu(sess)
    }

    fn target_config(&self, sess: &Session) -> rustc_codegen_ssa::TargetConfig {
        self.llvm_backend.target_config(sess)
    }

    fn provide(&self, providers: &mut rustc_middle::util::Providers) {
        // Delegate to LLVM backend
        self.llvm_backend.provide(providers);
    }

    /// Main codegen entry point - this is where device/host splitting happens.
    ///
    /// ## Execution Flow
    ///
    /// ```text
    /// codegen_crate(TyCtxt)
    ///       │
    ///       ├──▶ 1. Get monomorphized items from rustc
    ///       │       tcx.collect_and_partition_mono_items()
    ///       │
    ///       ├──▶ 2. Count kernels (functions in the reserved cuda_oxide_kernel_ namespace)
    ///       │
    ///       ├──▶ 3. If kernels found:
    ///       │       │
    ///       │       ├──▶ collector::collect_device_functions()
    ///       │       │       Walk call graph from kernels
    ///       │       │       Return Vec<CollectedFunction>
    ///       │       │
    ///       │       └──▶ device_codegen::generate_device_code()
    ///       │               Enter stable_mir context
    ///       │               Convert instances
    ///       │               Call mir_importer::run_pipeline()
    ///       │               Output: .ll and .ptx files
    ///       │
    ///       └──▶ 4. llvm_backend.codegen_crate(tcx)
    ///               Let LLVM handle ALL host code
    /// ```
    fn codegen_crate(&self, tcx: TyCtxt<'_>, crate_info: &CrateInfo) -> Box<dyn Any> {
        // Wrap entire function in with_no_trimmed_paths! to prevent diagnostic state issues.
        // This is necessary because we use tcx.def_path_str() and other functions that
        // trigger trimmed_def_paths. rust-gpu uses the same pattern.
        with_no_trimmed_paths!({
            // Step 1: Analyze for device code
            let mono_partitions = tcx.collect_and_partition_mono_items(());
            let kernel_count = collector::count_kernels_in_cgus(tcx, mono_partitions.codegen_units);
            let device_fn_count =
                collector::count_device_fns_in_cgus(tcx, mono_partitions.codegen_units);
            let has_device_code = kernel_count > 0 || device_fn_count > 0;
            let mut artifact_objects = Vec::new();

            // Only log for crates that have device code (reduces noise from dependency crates)
            if self.config.verbose && has_device_code {
                let crate_name = tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE);
                eprintln!(
                    "[rustc_codegen_cuda] Compiling crate '{}': {} CGUs, {} kernel(s), {} device fn(s)",
                    crate_name,
                    mono_partitions.codegen_units.len(),
                    kernel_count,
                    device_fn_count
                );
            }

            // Step 2: If device code exists, compile via cuda-oxide
            let _device_result = if has_device_code {
                if self.config.verbose {
                    eprintln!("[rustc_codegen_cuda] Compiling device code via cuda-oxide...");
                }

                // Collect all device-reachable functions (kernels + their callees)
                let collection_result = collector::collect_device_functions(
                    tcx,
                    mono_partitions.codegen_units,
                    self.config.verbose,
                );

                if self.config.verbose {
                    eprintln!(
                        "[rustc_codegen_cuda] Collected {} device functions, {} device externs for PTX compilation",
                        collection_result.functions.len(),
                        collection_result.device_externs.len()
                    );

                    // Dump MIR info for verification
                    collector::dump_device_mir_info(tcx, &collection_result.functions);
                }

                // Extract references for the pipeline
                let device_functions = &collection_result.functions;

                // Create device codegen config from our config
                let device_config =
                    device_codegen::DeviceCodegenConfig {
                        output_dir: self.config.ptx_output_dir.clone().unwrap_or_else(|| {
                            std::env::current_dir().unwrap_or_else(|_| ".".into())
                        }),
                        output_name: tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).to_string(),
                        verbose: self.config.verbose,
                        dump_rustc_mir: self.config.dump_rustc_mir,
                        dump_mir_dialect: self.config.dump_mir_dialect,
                        dump_llvm_dialect: self.config.dump_llvm_dialect,
                    };

                // Run the cuda-oxide pipeline!
                match device_codegen::generate_device_code(
                    tcx,
                    device_functions,
                    &collection_result.device_externs,
                    &device_config,
                ) {
                    Ok(result) => {
                        if self.config.verbose {
                            eprintln!(
                                "[rustc_codegen_cuda] Device codegen complete: {} (target: {})",
                                result.ptx_path.display(),
                                result.target
                            );
                        }
                        if let Some(ptx_content) = result.ptx_content.as_deref() {
                            match write_ptx_artifact_object(
                                &device_config.output_dir,
                                &device_config.output_name,
                                tcx.sess.target.llvm_target.as_ref(),
                                &result,
                                ptx_content,
                                device_functions,
                            ) {
                                Ok(path) => {
                                    if self.config.verbose {
                                        eprintln!(
                                            "[rustc_codegen_cuda] Embedded artifact object complete: {}",
                                            path.display()
                                        );
                                    }
                                    artifact_objects.push(path);
                                }
                                Err(e) => {
                                    tcx.dcx().fatal(format!(
                                        "[rustc_codegen_cuda] Failed to embed PTX artifact: {e}"
                                    ));
                                }
                            }
                        } else if self.config.verbose {
                            eprintln!(
                                "[rustc_codegen_cuda] Skipping embedded PTX artifact: no PTX output"
                            );
                        }
                        Some(result)
                    }
                    Err(e) => {
                        // Hard-fail: a swallowed device codegen error produces
                        // a host binary with stale or missing PTX, which then
                        // silently mis-runs on the GPU. The wrapper script
                        // (cargo-oxide) reports "✓ Build succeeded" in that
                        // case because the host LLVM backend below succeeds.
                        // Surface the failure as a rustc fatal so cargo exits
                        // non-zero and the wrapper's success print never fires.
                        // See `.cursor/rules/compiler-gaps-are-bugs.mdc`.
                        tcx.dcx()
                            .fatal(format!("[rustc_codegen_cuda] Device codegen failed: {}", e));
                    }
                }
            } else {
                None
            };

            // Step 3: Delegate ALL host codegen to LLVM backend
            // (No logging here - it fires for every crate including dependencies)
            let host_result = self.llvm_backend.codegen_crate(tcx, crate_info);

            // Return the LLVM backend's result
            Box::new(CudaOngoingCodegen {
                host: host_result,
                artifact_objects,
            })
        })
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        let ongoing = *ongoing_codegen
            .downcast::<CudaOngoingCodegen>()
            .expect("rustc_codegen_cuda received unexpected ongoing codegen state");
        let (mut compiled_modules, work_products) =
            self.llvm_backend.join_codegen(ongoing.host, sess, outputs);
        for (index, object) in ongoing.artifact_objects.into_iter().enumerate() {
            compiled_modules.modules.push(CompiledModule {
                name: format!("oxide_artifact_embed_{index}"),
                kind: ModuleKind::Regular,
                object: Some(object),
                dwarf_object: None,
                bytecode: None,
                assembly: None,
                llvm_ir: None,
                links_from_incr_cache: Vec::new(),
            });
        }
        (compiled_modules, work_products)
    }

    fn link(
        &self,
        sess: &Session,
        compiled_modules: CompiledModules,
        crate_info: CrateInfo,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        self.llvm_backend
            .link(sess, compiled_modules, crate_info, metadata, outputs);
    }
}

fn write_ptx_artifact_object(
    output_dir: &Path,
    output_name: &str,
    host_target: &str,
    result: &device_codegen::DeviceCodegenResult,
    ptx_content: &str,
    functions: &[collector::CollectedFunction<'_>],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let ptx_name = format!("{output_name}.ptx");
    let bundle_name = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| output_name.to_string());
    let mut spec = oxide_artifacts::ArtifactBundleSpec::new(&bundle_name, &result.target)
        .with_payload(oxide_artifacts::ArtifactPayloadSpec::new(
            oxide_artifacts::ArtifactPayloadKind::Ptx,
            &ptx_name,
            ptx_content.as_bytes(),
        ));
    for function in functions {
        let kind = if function.is_kernel {
            oxide_artifacts::ArtifactEntryKind::Kernel
        } else {
            oxide_artifacts::ArtifactEntryKind::DeviceFunction
        };
        spec = spec.with_entry(oxide_artifacts::ArtifactEntrySpec::new(
            &function.export_name,
            kind,
        ));
    }

    let blob = oxide_artifacts::build_artifact_blob(&spec)?;
    let object = oxide_artifacts::build_host_object_for_target(&blob, host_target)?;
    let safe_output_name = sanitize_path_component(output_name);
    let artifact_id = ARTIFACT_OBJECT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let object_dir = output_dir
        .join(".oxide-artifacts")
        .join(&safe_output_name)
        .join(sanitize_path_component(host_target));
    std::fs::create_dir_all(&object_dir)?;
    let object_path = object_dir.join(format!(
        "{safe_output_name}.{}.{artifact_id}.embed.o",
        std::process::id(),
    ));
    std::fs::write(&object_path, object)?;
    Ok(object_path)
}

fn sanitize_path_component(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => ch,
            _ => '_',
        })
        .collect()
}

/// Entry point called by rustc to instantiate the backend.
///
/// This function is discovered by rustc via the `#[no_mangle]` attribute and the
/// specific name `__rustc_codegen_backend`. When a user specifies
/// `-Z codegen-backend=path/to/librustc_codegen_cuda.so`, rustc loads the shared
/// library and calls this function to get a `Box<dyn CodegenBackend>`.
///
/// ## Initialization Sequence
///
/// ```text
/// rustc -Z codegen-backend=librustc_codegen_cuda.so ...
///       │
///       ├──▶ dlopen("librustc_codegen_cuda.so")
///       │
///       ├──▶ dlsym("__rustc_codegen_backend")
///       │
///       └──▶ __rustc_codegen_backend()
///               │
///               ├──▶ CudaCodegenConfig::from_env()
///               │       Read CUDA_OXIDE_* env vars
///               │
///               ├──▶ rustc_codegen_llvm::LlvmCodegenBackend::new()
///               │       Create the wrapped LLVM backend
///               │
///               └──▶ Return Box<CudaCodegenBackend>
/// ```
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    let config = CudaCodegenConfig::from_env();

    // Note: Don't log here - this function is called for EVERY crate in the dependency tree.
    // We log in codegen_crate() only when there are kernels to compile.

    // Get the LLVM backend - this is the same function rustc calls normally
    let llvm_backend = rustc_codegen_llvm::LlvmCodegenBackend::new();

    Box::new(CudaCodegenBackend {
        config,
        llvm_backend,
    })
}
