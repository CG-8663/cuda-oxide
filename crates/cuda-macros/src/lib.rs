/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod printf;

use proc_macro::TokenStream;

/// GPU printf macro for formatted output from GPU kernels.
///
/// This macro translates Rust-style format strings to C-style and calls
/// CUDA's `vprintf` function.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::gpu_printf;
///
/// #[kernel]
/// fn my_kernel() {
///     let tid = thread::index_1d().get();
///     gpu_printf!("Thread {}: Hello from GPU!\n", tid);
/// }
/// ```
///
/// # Format Specifiers
///
/// | Specifier | Description     | Example                                        |
/// |-----------|-----------------|------------------------------------------------|
/// | `{}`      | Default format  | `gpu_printf!("{}", 42)`                        |
/// | `{:x}`    | Hex (lower)     | `gpu_printf!("{:x}", 255)` → "ff"              |
/// | `{:X}`    | Hex (upper)     | `gpu_printf!("{:X}", 255)` → "FF"              |
/// | `{:#x}`   | Hex with prefix | `gpu_printf!("{:#x}", 255)` → "0xff"           |
/// | `{:o}`    | Octal           | `gpu_printf!("{:o}", 8)` → "10"                |
/// | `{:e}`    | Scientific      | `gpu_printf!("{:e}", 1000.0)` → "1.000000e+03" |
/// | `{:.N}`   | Precision       | `gpu_printf!("{:.2}", 3.14159)` → "3.14"       |
/// | `{:N}`    | Width           | `gpu_printf!("{:8}", 42)` → "      42"         |
/// | `{:0N}`   | Zero-pad        | `gpu_printf!("{:08}", 42)` → "00000042"        |
///
/// # Returns
///
/// The number of arguments (i32), or negative on error.
/// Note: CUDA vprintf returns arg count, not character count.
#[proc_macro]
pub fn gpu_printf(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as printf::GpuPrintfInput);
    printf::gpu_printf_impl(input).into()
}
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::HashSet;
use syn::{
    FnArg, ForeignItem, GenericParam, Ident, ItemFn, ItemForeignMod, Pat, Token, Type, bracketed,
    parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
    visit::Visit,
};

/// Attribute arguments for #[kernel(...)]
/// Supports: #[kernel] or #[kernel(Type1, Type2, Type3)]
struct KernelArgs {
    /// Types to instantiate generic kernels for
    instantiate_types: Vec<Type>,
}

impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(KernelArgs {
                instantiate_types: vec![],
            });
        }

        let types: Punctuated<Type, Token![,]> = Punctuated::parse_terminated(input)?;
        Ok(KernelArgs {
            instantiate_types: types.into_iter().collect(),
        })
    }
}

/// Marks a function as a CUDA kernel.
///
/// This attribute:
/// 1. Adds `#[no_mangle]` to preserve the function name in the binary
/// 2. Marks the function for detection by the `rustc-codegen-cuda` backend
///
/// # Generic Kernels
///
/// For generic kernels (like `template<class F> __global__` in CUDA C++),
/// specify the types to instantiate:
///
/// ```ignore
/// #[kernel(Scale, Fma, Square)]
/// pub fn map<F: GpuFn>(f: F, input: &[i32], output: DisjointSlice<i32>) {
///     // ...
/// }
/// ```
///
/// This generates three PTX entry points: `map_Scale`, `map_Fma`, `map_Square`.
/// Each is a monomorphized version of the generic kernel.
///
/// # Example (non-generic)
///
/// ```ignore
/// #[kernel]
/// pub fn simple_kernel(data: &mut [i32]) {
///     // ...
/// }
/// ```
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as KernelArgs);
    let input = parse_macro_input!(item as ItemFn);

    // Check if function has type parameters
    let has_generics = input
        .sig
        .generics
        .params
        .iter()
        .any(|p| matches!(p, GenericParam::Type(_)));

    if has_generics && args.instantiate_types.is_empty() {
        // Generic kernel without explicit types - allow it!
        // Instantiation will happen from call sites (nvcc-style)
        return generate_generic_kernel_no_instantiation(input);
    }

    if !has_generics && !args.instantiate_types.is_empty() {
        // Non-generic kernel with instantiation types - error
        return syn::Error::new_spanned(
            &input.sig.ident,
            "Instantiation types only apply to generic kernels",
        )
        .to_compile_error()
        .into();
    }

    if has_generics {
        // Generate wrapper kernels for each instantiation type
        generate_generic_kernel(input, args.instantiate_types)
    } else {
        // Simple non-generic kernel
        generate_simple_kernel(input)
    }
}

/// Find the generic type parameter that has a Fn/FnMut/FnOnce bound (the closure type).
/// Returns the type parameter name if found.
fn find_closure_generic(generics: &syn::Generics) -> Option<syn::Ident> {
    for param in &generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            for bound in &type_param.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    if let Some(segment) = trait_bound.path.segments.last() {
                        let name = segment.ident.to_string();
                        if name == "Fn" || name == "FnMut" || name == "FnOnce" {
                            return Some(type_param.ident.clone());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Find which function parameter uses the closure type.
/// Returns the index and info of the closure parameter.
fn find_closure_param<'a>(
    args_info: &'a [(&'a Ident, &'a Type)],
    closure_type_name: &syn::Ident,
) -> Option<(usize, &'a (&'a Ident, &'a Type))> {
    for (idx, (_name, ty)) in args_info.iter().enumerate() {
        // Check if the type is a simple path matching our closure generic
        if let Type::Path(type_path) = *ty {
            if type_path.qself.is_none() {
                if let Some(segment) = type_path.path.segments.first() {
                    if type_path.path.segments.len() == 1 && segment.ident == *closure_type_name {
                        return Some((idx, &args_info[idx]));
                    }
                }
            }
        }
    }
    None
}

/// Strip `mut` from function argument patterns.
///
/// The wrapper function just forwards arguments, so it doesn't need `mut`.
/// Keeping `mut` causes "variable does not need to be mutable" warnings.
fn strip_mut_from_inputs(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
) -> Vec<FnArg> {
    inputs
        .iter()
        .map(|arg| {
            match arg {
                FnArg::Typed(pat_type) => {
                    let mut new_pat_type = pat_type.clone();
                    if let Pat::Ident(pat_ident) = &*pat_type.pat {
                        if pat_ident.mutability.is_some() {
                            // Create new PatIdent without mut
                            let new_pat_ident = syn::PatIdent {
                                attrs: pat_ident.attrs.clone(),
                                by_ref: pat_ident.by_ref,
                                mutability: None, // Strip mut
                                ident: pat_ident.ident.clone(),
                                subpat: pat_ident.subpat.clone(),
                            };
                            new_pat_type.pat = Box::new(Pat::Ident(new_pat_ident));
                        }
                    }
                    FnArg::Typed(new_pat_type)
                }
                other => other.clone(),
            }
        })
        .collect()
}

/// Generate a generic kernel that will be instantiated from call sites (nvcc-style)
fn generate_generic_kernel_no_instantiation(input: ItemFn) -> TokenStream {
    let fn_name = &input.sig.ident;
    let vis = &input.vis;
    let generics = &input.sig.generics;
    let where_clause = &input.sig.generics.where_clause;
    let inputs = &input.sig.inputs;
    let output = &input.sig.output;
    let block = &input.block;

    // Create the renamed version for rustc-codegen-cuda to find
    let kernel_name_str = format!("cuda_oxide_kernel_{}", fn_name);
    let kernel_name = syn::Ident::new(&kernel_name_str, fn_name.span());

    // Create the instantiate helper name (matches cuda_oxide_instantiate_* pattern)
    let instantiate_name = format_ident!("cuda_oxide_instantiate_{}", fn_name);

    // For the wrapper function, strip `mut` from parameters since it just forwards them
    let wrapper_inputs = strip_mut_from_inputs(inputs);

    // Extract argument names and info for forwarding
    let args_info: Vec<_> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    return Some((&pat_ident.ident, &*pat_type.ty));
                }
            }
            None
        })
        .collect();

    let arg_names: Vec<_> = args_info.iter().map(|(name, _)| *name).collect();

    // Find the closure generic type (looks for Fn/FnMut/FnOnce bounds)
    let closure_generic = find_closure_generic(generics);

    // Extract generic type parameter names (T, F, etc.) for use in function pointer cast
    let generic_param_names: Vec<&syn::Ident> = generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Type(type_param) = p {
                Some(&type_param.ident)
            } else {
                None
            }
        })
        .collect();

    // Generate the instantiate helper only if we found a closure parameter
    let instantiate_helper = if let Some(closure_type_name) = closure_generic {
        // Find which parameter uses the closure type
        if let Some((_closure_idx, (_closure_name, closure_type))) =
            find_closure_param(&args_info, &closure_type_name)
        {
            // Build the function type for the kernel (for the function pointer)
            let arg_types: Vec<TokenStream2> =
                args_info.iter().map(|(_, ty)| quote! { #ty }).collect();

            quote! {
                /// Auto-generated helper to force kernel monomorphization.
                /// Takes the closure and its source location, returns the PTX export name.
                /// This forces rustc to monomorphize the kernel with the closure type
                /// WITHOUT actually calling the kernel (which would panic on host).
                ///
                /// The line/col parameters come from the proc-macro's span, ensuring the
                /// export name matches what the backend generates.
                #[doc(hidden)]
                #[inline(never)]
                #vis fn #instantiate_name #generics (_f: #closure_type, line: u32, col: u32) -> &'static str #where_clause {
                    // Force monomorphization by referencing the kernel with explicit type params
                    // CRITICAL: Use volatile write/read to prevent optimization from eliminating
                    // the function pointer reference. Without this, the `let _ = ...` gets DCE'd
                    // and rustc doesn't generate the CGU entry.
                    let __kernel_ptr = #kernel_name::<#(#generic_param_names),*> as fn(#(#arg_types),*) as *const ();
                    unsafe {
                        let mut __force_mono: *const () = core::ptr::null();
                        core::ptr::write_volatile(&mut __force_mono, __kernel_ptr);
                        let _ = core::ptr::read_volatile(&__force_mono);
                    }
                    // Return the PTX export name - based on source location
                    // The backend uses the same naming scheme: "{kernel}_L{line}C{col}"
                    // Leak a formatted string (only happens once per monomorphization)
                    let name = std::boxed::Box::leak(
                        format!("{}_L{}C{}", stringify!(#fn_name), line, col).into_boxed_str()
                    );
                    name
                }
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    // Generate the GenericCudaKernel trait implementation for unified compilation
    let generic_cuda_kernel_impl =
        generate_generic_cuda_kernel_impl(fn_name, generics, where_clause);

    let expanded = quote! {
        // Original generic kernel implementation
        #[inline(always)]
        #vis fn #fn_name #generics (#inputs) #output #where_clause
        #block

        // Entry point for collector - NOT inlined so we can detect it
        // When called with concrete types, this instantiates the kernel
        // Note: wrapper_inputs has `mut` stripped since we just forward args
        #[inline(never)]
        #vis fn #kernel_name #generics (#(#wrapper_inputs),*) #output #where_clause {
            #fn_name(#(#arg_names),*)
        }

        #instantiate_helper

        #generic_cuda_kernel_impl
    };

    TokenStream::from(expanded)
}

/// Generate a dummy binding for a given type.
/// Used by instantiate! helper to create zero-valued arguments.
///
/// The generated values are never actually executed - they exist only to force
/// rustc to monomorphize the kernel with the correct types.
fn _generate_dummy_binding(name: &Ident, ty: &Type) -> TokenStream2 {
    match ty {
        // Special case: &[T] or &mut [T] → empty slice literal
        // (slices don't implement Default and can't be safely zeroed)
        Type::Reference(type_ref) if matches!(&*type_ref.elem, Type::Slice(_)) => {
            if let Type::Slice(slice) = &*type_ref.elem {
                let elem_ty = &slice.elem;
                if type_ref.mutability.is_some() {
                    quote! { let #name: &mut [#elem_ty] = &mut []; }
                } else {
                    quote! { let #name: &[#elem_ty] = &[]; }
                }
            } else {
                unreachable!()
            }
        }

        // Everything else: use mem::zeroed()
        // Safe because this code never actually runs - it only exists to
        // force monomorphization of the kernel with the correct types.
        _ => {
            quote! { let #name: #ty = unsafe { core::mem::zeroed() }; }
        }
    }
}

/// Generate a simple non-generic kernel
fn generate_simple_kernel(mut input: ItemFn) -> TokenStream {
    let fn_name = input.sig.ident.clone();
    let mangled_name_str = format!("cuda_oxide_kernel_{}", fn_name);
    let new_name = syn::Ident::new(&mangled_name_str, fn_name.span());

    // Clone the original function for the CudaKernel impl
    let original_fn = input.clone();
    input.sig.ident = new_name;

    // The PTX entry name is the ORIGINAL function name (e.g., "vecadd")
    // The collector strips the "cuda_oxide_kernel_" prefix when generating PTX
    let ptx_entry_name = fn_name.to_string();

    // Generate the CudaKernel trait implementation (host-side only)
    // This provides the PTX name for cuda_launch! to look up
    let cuda_kernel_impl = generate_cuda_kernel_impl(&fn_name, &ptx_entry_name, &original_fn);

    let expanded = quote! {
        #[unsafe(no_mangle)]
        #input

        #cuda_kernel_impl
    };

    TokenStream::from(expanded)
}

/// Generate the GenericCudaKernel trait implementation for a generic kernel.
///
/// For generic kernels like `fn scale<T>()`, we generate:
/// ```ignore
/// pub struct __scale_CudaKernel<T>(PhantomData<T>);
/// impl<T> GenericCudaKernel for __scale_CudaKernel<T> {
///     fn ptx_name() -> &'static str { ... }
/// }
/// ```
///
/// The PTX name is computed at runtime based on `std::any::type_name::<Self>()`.
/// The backend uses the same naming scheme when generating PTX.
fn generate_generic_cuda_kernel_impl(
    fn_name: &Ident,
    generics: &syn::Generics,
    where_clause: &Option<syn::WhereClause>,
) -> TokenStream2 {
    let marker_name = format_ident!("__{}_CudaKernel", fn_name);
    let base_name = fn_name.to_string();

    // Extract just the type parameters (for PhantomData and impl)
    let type_params: Vec<_> = generics.params.iter().collect();
    let type_param_names: Vec<_> = generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Type(tp) = p {
                Some(&tp.ident)
            } else {
                None
            }
        })
        .collect();

    quote! {
        /// Marker type for generic kernel, implements GenericCudaKernel trait.
        /// The type parameters match the kernel's generic parameters.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        pub struct #marker_name<#(#type_params),*>(
            core::marker::PhantomData<(#(#type_param_names),*)>
        ) #where_clause;

        impl<#(#type_params),*> cuda_host::GenericCudaKernel for #marker_name<#(#type_param_names),*>
        #where_clause
        {
            fn ptx_name() -> &'static str {
                // Generate a unique PTX name for this specific monomorphization.
                // Uses type_name to distinguish e.g. scale::<f32> from scale::<i32>.
                //
                // The naming scheme must match the collector's compute_generic_kernel_name()
                // which uses the same approach: base_name + sanitized type params.
                //
                // We use a const fn to compute this at compile time, leaking a String
                // to get a &'static str. This is acceptable as kernel names are few.
                const BASE: &str = #base_name;

                // Get the full type name which includes generic params
                // e.g., "cross_crate_kernel::__scale_CudaKernel<f32>"
                let full_name = core::any::type_name::<Self>();

                // Extract just the type params from the angle brackets
                // e.g., "<f32>" or "<i32, SomeType>"
                if let Some(start) = full_name.find('<') {
                    if let Some(end) = full_name.rfind('>') {
                        let type_params = &full_name[start + 1..end];
                        // Sanitize: replace invalid PTX chars with underscores
                        let sanitized: String = type_params
                            .chars()
                            .map(|c| match c {
                                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => c,
                                _ => '_',
                            })
                            .collect();
                        // Combine: "scale" + "__" + "f32" = "scale__f32"
                        let name = format!("{}__{}", BASE, sanitized);
                        return Box::leak(name.into_boxed_str());
                    }
                }
                // Fallback: no type params, use base name
                BASE
            }
        }
    }
}

/// Generate the CudaKernel trait implementation for a kernel function.
///
/// This generates a marker struct that implements `CudaKernel`, allowing
/// `cuda_launch!` to look up the PTX entry point name at compile time.
fn generate_cuda_kernel_impl(fn_name: &Ident, ptx_name: &str, _func: &ItemFn) -> TokenStream2 {
    // Create a marker struct for this kernel
    // We use a struct because Rust doesn't allow trait impls on function pointers easily
    let marker_name = format_ident!("__{}_CudaKernel", fn_name);

    quote! {
        /// Marker type for the kernel, implements CudaKernel trait.
        /// This enables cuda_launch! to look up the PTX entry point name.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        pub struct #marker_name;

        impl cuda_host::CudaKernel for #marker_name {
            const PTX_NAME: &'static str = #ptx_name;
        }
    }
}

/// Generate wrapper kernels for a generic kernel
fn generate_generic_kernel(input: ItemFn, instantiate_types: Vec<Type>) -> TokenStream {
    let fn_name = &input.sig.ident;
    let vis = &input.vis;
    let generics = &input.sig.generics;

    // Extract the type parameter name (assume single type param for now)
    let type_param = generics
        .params
        .iter()
        .find_map(|p| {
            if let GenericParam::Type(tp) = p {
                Some(&tp.ident)
            } else {
                None
            }
        })
        .expect("Expected type parameter");

    // Extract function arguments (excluding self)
    let args: Vec<_> = input.sig.inputs.iter().collect();

    // Build the argument pattern and types for wrappers
    let arg_names: Vec<TokenStream2> = args
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    return Some(quote! { #pat_ident });
                }
            }
            None
        })
        .collect();

    // For each instantiation type, generate a wrapper that substitutes the type
    let wrappers: Vec<TokenStream2> = instantiate_types
        .iter()
        .map(|inst_type| {
            // Get a clean name for the type (for the kernel name suffix)
            let type_name = get_type_name(inst_type);
            let wrapper_name_str = format!("cuda_oxide_kernel_{}_{}", fn_name, type_name);
            let wrapper_name = syn::Ident::new(&wrapper_name_str, fn_name.span());

            // Export name (what appears in PTX)
            let export_name_str = format!("{}_{}", fn_name, type_name);

            // Generate wrapper function args with substituted types
            let wrapper_args: Vec<TokenStream2> = args
                .iter()
                .map(|arg| {
                    if let FnArg::Typed(pat_type) = arg {
                        let pat = &pat_type.pat;
                        let ty = &pat_type.ty;
                        // Substitute type parameter with concrete type
                        let subst_ty = substitute_type(ty, type_param, inst_type);
                        quote! { #pat: #subst_ty }
                    } else {
                        quote! { #arg }
                    }
                })
                .collect();

            quote! {
                #[unsafe(no_mangle)]
                #[unsafe(export_name = #export_name_str)]
                #vis fn #wrapper_name(#(#wrapper_args),*) {
                    #fn_name::<#inst_type>(#(#arg_names),*);
                }
            }
        })
        .collect();

    // Keep the original generic function (without #[no_mangle] - it's not an entry point)
    // and add all the wrapper kernels
    let expanded = quote! {
        #[inline(always)]
        #input

        #(#wrappers)*
    };

    TokenStream::from(expanded)
}

/// Get a clean name from a type for use in function names
fn get_type_name(ty: &Type) -> String {
    match ty {
        Type::Path(type_path) => {
            // Get the last segment (e.g., "Scale" from "crate::Scale")
            type_path
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_else(|| "Unknown".to_string())
        }
        _ => "Unknown".to_string(),
    }
}

/// Substitute a type parameter with a concrete type in a type expression
fn substitute_type(ty: &Type, param: &syn::Ident, replacement: &Type) -> TokenStream2 {
    match ty {
        Type::Path(type_path) => {
            // Check if this is just the type parameter
            if type_path.path.is_ident(param) {
                return quote! { #replacement };
            }
            quote! { #ty }
        }
        Type::Reference(type_ref) => {
            let elem = substitute_type(&type_ref.elem, param, replacement);
            let lifetime = &type_ref.lifetime;
            let mutability = &type_ref.mutability;
            quote! { &#lifetime #mutability #elem }
        }
        _ => quote! { #ty },
    }
}

/// Specifies launch bounds for a kernel (max threads per block, min blocks per SM).
///
/// This attribute sets kernel launch bounds at compile time by emitting `.maxntid`
/// and `.minnctapersm` PTX directives. This helps the CUDA compiler optimize
/// register allocation and occupancy.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{kernel, launch_bounds, DisjointSlice};
///
/// #[kernel]
/// #[launch_bounds(256)]              // Max 256 threads per block
/// pub fn simple_kernel(output: DisjointSlice<f32>) { ... }
///
/// #[kernel]
/// #[launch_bounds(256, 2)]           // Max 256 threads, min 2 blocks per SM
/// pub fn optimized_kernel(output: DisjointSlice<f32>) { ... }
/// ```
///
/// # Parameters
///
/// - First parameter (required): Maximum threads per block
/// - Second parameter (optional): Minimum blocks per SM for occupancy hints
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone)
/// - The `#[launch_bounds]` attribute must come AFTER `#[kernel]`
///
/// # Performance Impact
///
/// Launch bounds help the compiler:
/// - Allocate registers more efficiently
/// - Optimize occupancy (threads per SM)
/// - Make better scheduling decisions
///
/// # PTX Output
///
/// ```ptx
/// .entry my_kernel .maxntid 256 .minnctapersm 2 { ... }
/// ```
#[proc_macro_attribute]
pub fn launch_bounds(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args: LaunchBoundsArgs = parse_macro_input!(attr as LaunchBoundsArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let max_threads = args.max_threads;
    let min_blocks = args.min_blocks;

    // Inject the launch bounds config marker at the start of the function body
    let marker_call: syn::Stmt = syn::parse_quote! {
        cuda_device::thread::__launch_bounds_config::<#max_threads, #min_blocks>();
    };

    // Prepend the marker to the function body
    input.block.stmts.insert(0, marker_call);

    quote! {
        #input
    }
    .into()
}

/// Arguments for `#[launch_bounds(...)]` attribute.
struct LaunchBoundsArgs {
    max_threads: u32,
    min_blocks: u32,
}

impl Parse for LaunchBoundsArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(input)?;
        let values: Vec<u32> = args
            .iter()
            .map(|lit| lit.base10_parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;

        match values.len() {
            1 => Ok(LaunchBoundsArgs {
                max_threads: values[0],
                min_blocks: 0, // Unspecified
            }),
            2 => Ok(LaunchBoundsArgs {
                max_threads: values[0],
                min_blocks: values[1],
            }),
            _ => Err(syn::Error::new_spanned(
                args.first().unwrap(),
                "launch_bounds expects 1 or 2 parameters: #[launch_bounds(max_threads)] or #[launch_bounds(max_threads, min_blocks)]",
            )),
        }
    }
}

/// Specifies compile-time cluster dimensions for a kernel.
///
/// This attribute sets the thread block cluster size at compile time by emitting
/// the `.reqnctapercluster` PTX directive. When used, the kernel will automatically
/// launch with the specified cluster configuration.
///
/// Note: Named `cluster_launch` (not `cluster`) to avoid conflict with `cuda_device::cluster` module.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{kernel, cluster, cluster_launch, DisjointSlice};
///
/// #[kernel]
/// #[cluster_launch(4, 1, 1)]  // 4 blocks per cluster in X dimension
/// pub fn my_cluster_kernel(output: DisjointSlice<u32>) {
///     let rank = cluster::block_rank();
///     // ...
/// }
/// ```
///
/// # Cluster Dimensions
///
/// - `#[cluster_launch(n)]` - 1D cluster with n blocks
/// - `#[cluster_launch(x, y)]` - 2D cluster with x*y blocks
/// - `#[cluster_launch(x, y, z)]` - 3D cluster with x*y*z blocks
///
/// Maximum cluster size is typically 16 blocks (hardware dependent).
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone)
/// - Requires sm_90+ (Hopper) or newer GPU
/// - The `#[cluster_launch]` attribute must come AFTER `#[kernel]`
///
/// # How It Works
///
/// The macro injects `cuda_device::cluster::__cluster_config::<X, Y, Z>()` at the
/// start of the kernel. The compiler:
/// 1. Detects this marker during MIR translation
/// 2. Extracts the const generic parameters (X, Y, Z)
/// 3. Emits `!nvvm.annotations` metadata with cluster dimensions
/// 4. LLVM NVPTX backend generates `.reqnctapercluster X, Y, Z` in PTX
///
/// # PTX Output
///
/// ```ptx
/// .entry my_cluster_kernel .reqnctapercluster 4, 1, 1 { ... }
/// ```
///
/// # Compile-Time vs Runtime Clusters
///
/// | Method | Pros | Cons |
/// |--------|------|------|
/// | `#[cluster_launch(x,y,z)]` (compile-time) | Simple, no special launch API | Fixed at compile time |
/// | `cuLaunchKernelEx` (runtime) | Dynamic cluster sizes | Requires FFI, complex setup |
#[proc_macro_attribute]
pub fn cluster_launch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args: ClusterArgs = parse_macro_input!(attr as ClusterArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let x = args.x;
    let y = args.y;
    let z = args.z;

    // Inject the cluster config marker at the start of the function body
    let marker_call: syn::Stmt = syn::parse_quote! {
        cuda_device::cluster::__cluster_config::<#x, #y, #z>();
    };

    // Prepend the marker to the function body
    input.block.stmts.insert(0, marker_call);

    quote! {
        #input
    }
    .into()
}

/// Arguments for `#[cluster_launch(...)]` attribute.
struct ClusterArgs {
    x: u32,
    y: u32,
    z: u32,
}

impl Parse for ClusterArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(input)?;
        let values: Vec<u32> = args
            .iter()
            .map(|lit| lit.base10_parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;

        match values.len() {
            1 => Ok(ClusterArgs {
                x: values[0],
                y: 1,
                z: 1,
            }),
            2 => Ok(ClusterArgs {
                x: values[0],
                y: values[1],
                z: 1,
            }),
            3 => Ok(ClusterArgs {
                x: values[0],
                y: values[1],
                z: values[2],
            }),
            _ => Err(syn::Error::new_spanned(
                args.first().unwrap(),
                "cluster expects 1, 2, or 3 dimensions: #[cluster(x)], #[cluster(x, y)], or #[cluster(x, y, z)]",
            )),
        }
    }
}

/// Marks a function as a CUDA device function.
///
/// Device functions run on the GPU and can be called from kernels or other device functions,
/// but cannot be called from host code.
///
/// This attribute:
/// 1. Adds `#[no_mangle]` to preserve the function name in the binary
/// 2. Renames the function with a `cuda_oxide_device_` prefix for detection
/// 3. Marks the function for extraction by the `rustc-codegen-cuda` backend
///
/// Device functions can:
/// - Return values (unlike kernels which must return `()`)
/// - Be called from kernels and other device functions
/// - Use generics (each monomorphization becomes a separate device function)
///
/// # Example: Device Function Definition
///
/// ```ignore
/// use cuda_device::device;
///
/// #[device]
/// pub fn helper(x: f32, y: f32) -> f32 {
///     x * x + y * y
/// }
///
/// #[kernel]
/// pub fn my_kernel(data: *mut f32) {
///     let result = helper(1.0, 2.0);
///     unsafe { *data = result; }
/// }
/// ```
///
/// # Example: External Device Function Declaration (FFI)
///
/// ```ignore
/// use cuda_device::{device, convergent};
///
/// // Declare external device functions from LTOIR (e.g., CCCL)
/// #[device]
/// extern "C" {
///     #[convergent]
///     fn cub_block_reduce_sum_f32(input: f32, temp: *mut u8) -> f32;
///
///     fn fast_math_helper(x: f32) -> f32;
/// }
///
/// #[kernel]
/// pub fn my_kernel(data: *mut f32) {
///     let result = unsafe { cub_block_reduce_sum_f32(*data, temp_ptr) };
/// }
/// ```
#[proc_macro_attribute]
pub fn device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Try parsing as a function definition first
    if let Ok(input) = syn::parse::<ItemFn>(item.clone()) {
        return generate_device_function(input);
    }

    // Try parsing as an extern block
    if let Ok(input) = syn::parse::<ItemForeignMod>(item.clone()) {
        return generate_device_extern_block(input);
    }

    // Neither worked - emit error
    syn::Error::new_spanned(
        proc_macro2::TokenStream::from(item),
        "#[device] can only be applied to functions or extern blocks",
    )
    .to_compile_error()
    .into()
}

/// Generate a device function definition.
///
/// Renames the function with `cuda_oxide_device_` prefix for collector detection,
/// and generates a thin wrapper with the original name so user code can call
/// `my_func()` instead of `cuda_oxide_device_my_func()`.
///
/// Handles both non-generic and generic device functions:
/// - **Non-generic**: `#[no_mangle]` on the prefixed function, `#[inline(always)]` wrapper.
/// - **Generic**: No `#[no_mangle]` (generics use mangled symbols), `#[inline(never)]` on
///   the prefixed function (so monomorphizations appear in CGUs for the collector),
///   `#[inline(always)]` wrapper with generics + turbofish forwarding.
///
/// This mirrors the pattern used by `#[kernel]` for generic kernels
/// (see `generate_generic_kernel_no_instantiation`).
fn generate_device_function(mut input: ItemFn) -> TokenStream {
    let fn_name = input.sig.ident.clone();
    let vis = input.vis.clone();
    let new_name_str = format!("cuda_oxide_device_{}", fn_name);
    let new_name = syn::Ident::new(&new_name_str, fn_name.span());

    // Check if the function has type parameters
    let has_generics = input
        .sig
        .generics
        .params
        .iter()
        .any(|p| matches!(p, GenericParam::Type(_)));

    // Extract parameter names for forwarding
    let params: Vec<_> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    return Some(pat_ident.ident.clone());
                }
            }
            None
        })
        .collect();

    let return_type = &input.sig.output;
    let generics = &input.sig.generics;
    let where_clause = &input.sig.generics.where_clause;

    // Strip `mut` from wrapper parameters since the wrapper just forwards args.
    // In Rust, `mut` on a by-value parameter is purely local binding mutability —
    // it's not part of the function's type signature and callers don't need `mut`
    // to pass a value. The original (renamed) function keeps `mut` for its body,
    // but the wrapper only forwards the value and never mutates it locally.
    let wrapper_inputs = strip_mut_from_inputs(&input.sig.inputs);

    // Rename the original function with the prefix
    input.sig.ident = new_name.clone();

    if has_generics {
        // Generic device function: mirrors the generic kernel pattern.
        //
        // - No #[no_mangle] — generic functions use mangled symbol names per
        //   monomorphization (e.g., `cuda_oxide_device_add::<f32>` gets a unique
        //   mangled name). #[no_mangle] requires a single concrete symbol.
        //
        // - #[inline(never)] on the prefixed function — ensures each monomorphization
        //   appears as a distinct CGU item so the collector can find it. If it were
        //   inlined, the function would disappear from the CGU.
        //
        // - The wrapper forwards type parameters via turbofish:
        //   `cuda_oxide_device_add::<T>(a, b)`.

        // Extract type parameter names for turbofish forwarding (T, U, etc.)
        let type_param_names: Vec<&syn::Ident> = generics
            .params
            .iter()
            .filter_map(|p| {
                if let GenericParam::Type(type_param) = p {
                    Some(&type_param.ident)
                } else {
                    None
                }
            })
            .collect();

        let expanded = quote! {
            #[inline(never)]
            #input

            /// Wrapper for the generic device function with the original name.
            #[inline(always)]
            #vis fn #fn_name #generics (#(#wrapper_inputs),*) #return_type #where_clause {
                #new_name::<#(#type_param_names),*>(#(#params),*)
            }
        };

        TokenStream::from(expanded)
    } else {
        // Non-generic device function: simple case.
        let expanded = quote! {
            #[unsafe(no_mangle)]
            #input

            /// Wrapper for the device function with the original name.
            #[inline(always)]
            #vis fn #fn_name(#(#wrapper_inputs),*) #return_type {
                #new_name(#(#params),*)
            }
        };

        TokenStream::from(expanded)
    }
}

/// Generate device extern block declarations (for FFI with external LTOIR)
///
/// For each function in the extern block:
/// 1. Rename it with `cuda_oxide_device_extern_` prefix (for collector detection)
/// 2. Generate a wrapper function with the original name (for user code)
///
/// This allows user code to call `foo()` while the collector sees `cuda_oxide_device_extern_foo`.
fn generate_device_extern_block(mut input: ItemForeignMod) -> TokenStream {
    let mut wrappers = Vec::new();

    // Process each item in the extern block
    for item in &mut input.items {
        if let ForeignItem::Fn(foreign_fn) = item {
            // Save original info for wrapper generation
            let original_name = foreign_fn.sig.ident.clone();
            let original_attrs = foreign_fn.attrs.clone();
            let original_sig = foreign_fn.sig.clone();

            // Rename with cuda_oxide_device_extern_ prefix
            let new_name_str = format!("cuda_oxide_device_extern_{}", original_name);
            let new_name = syn::Ident::new(&new_name_str, original_name.span());
            foreign_fn.sig.ident = new_name.clone();

            // Store original name as link_name for the linker
            let original_name_str = original_name.to_string();
            foreign_fn.attrs.push(syn::parse_quote! {
                #[doc(hidden)]
            });
            foreign_fn.attrs.push(syn::parse_quote! {
                #[link_name = #original_name_str]
            });

            // Generate wrapper function with original name
            // This allows user code to call foo() instead of cuda_oxide_device_extern_foo()
            let params: Vec<_> = original_sig
                .inputs
                .iter()
                .filter_map(|arg| {
                    if let syn::FnArg::Typed(pat_type) = arg {
                        if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                            return Some(pat_ident.ident.clone());
                        }
                    }
                    None
                })
                .collect();

            let return_type = &original_sig.output;
            let inputs = &original_sig.inputs;

            // Keep user's attributes (like #[convergent]) on the wrapper
            let wrapper = quote! {
                #(#original_attrs)*
                #[inline(always)]
                #[allow(non_snake_case)]
                pub unsafe fn #original_name(#inputs) #return_type {
                    #new_name(#(#params),*)
                }
            };
            wrappers.push(wrapper);
        }
    }

    let expanded = quote! {
        #input

        #(#wrappers)*
    };

    TokenStream::from(expanded)
}

// ============================================================================
// NVVM Attributes for Device FFI
// ============================================================================

/// Marks a device function as convergent.
///
/// Convergent functions must be called by all threads in a warp/block together.
/// This prevents the optimizer from moving calls across control flow boundaries.
///
/// # When to Use
///
/// - Synchronization primitives (`__syncthreads`, barriers)
/// - Warp-collective operations (`__shfl_*`, warp vote, warp reduce)
/// - Block-collective operations (CUB block reduce/scan)
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[convergent]
///     fn cub_block_reduce_sum(input: f32, temp: *mut u8) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @cub_block_reduce_sum(float, ptr) #0
/// attributes #0 = { convergent nounwind }
/// ```
#[proc_macro_attribute]
pub fn convergent(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This is a marker attribute - just pass through the item unchanged.
    // The collector will read this attribute and apply the LLVM convergent attribute.
    item
}

/// Marks a device function as pure (no side effects).
///
/// Pure functions only depend on their inputs and have no side effects.
/// This enables aggressive optimizations like CSE and dead code elimination.
///
/// # When to Use
///
/// - Math functions that don't access memory
/// - Functions that compute results purely from input arguments
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[pure]
///     fn fast_rsqrt(x: f32) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @fast_rsqrt(float) #0
/// attributes #0 = { nounwind readnone }
/// ```
#[proc_macro_attribute]
pub fn pure(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Marker attribute - collector will read and apply LLVM readnone attribute
    item
}

/// Marks a device function as read-only.
///
/// Read-only functions may read memory but never write to it.
/// This enables optimizations like load hoisting and caching.
///
/// # When to Use
///
/// - Lookup table functions
/// - Functions that only read from input arrays
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[readonly]
///     fn lookup_table(table: *const f32, idx: i32) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @lookup_table(ptr, i32) #0
/// attributes #0 = { nounwind readonly }
/// ```
#[proc_macro_attribute]
pub fn readonly(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Marker attribute - collector will read and apply LLVM readonly attribute
    item
}

// ============================================================================
// cuda_launch! Macro (unified compilation)
// ============================================================================

// ============================================================================
// Closure Capture Extraction
// ============================================================================

/// Collects identifiers from an expression AST.
/// Used to find potential captured variables in closures.
struct IdentCollector {
    /// Collected identifiers (simple names, not paths)
    idents: Vec<syn::Ident>,
    /// Variables that are bound locally (shadow outer scope)
    local_bindings: HashSet<String>,
}

impl IdentCollector {
    fn new() -> Self {
        Self {
            idents: Vec::new(),
            local_bindings: HashSet::new(),
        }
    }
}

impl<'ast> Visit<'ast> for IdentCollector {
    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        // Only collect simple identifiers, not qualified paths like std::mem::drop
        if node.path.segments.len() == 1 && node.qself.is_none() {
            let ident = &node.path.segments[0].ident;
            let name = ident.to_string();
            // Skip if it's a local binding (shadowed variable)
            if !self.local_bindings.contains(&name) {
                self.idents.push(ident.clone());
            }
        }
        syn::visit::visit_expr_path(self, node);
    }

    fn visit_local(&mut self, node: &'ast syn::Local) {
        // Track local `let` bindings - they shadow outer variables
        if let syn::Pat::Ident(pat_ident) = &node.pat {
            self.local_bindings.insert(pat_ident.ident.to_string());
        } else if let syn::Pat::Type(pat_type) = &node.pat {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                self.local_bindings.insert(pat_ident.ident.to_string());
            }
        }
        // Still visit the initializer and body
        syn::visit::visit_local(self, node);
    }

    fn visit_expr_closure(&mut self, _node: &'ast syn::ExprClosure) {
        // Don't recurse into nested closures - their captures are their own
        // We only care about captures at the current closure level
    }
}

/// Extract captured variables from a closure expression.
///
/// This function parses the closure's parameters and body to determine which
/// variables are captured from the surrounding scope.
///
/// # Algorithm
/// 1. Collect parameter names (they're not captures)
/// 2. Walk the body AST, collect all simple identifiers
/// 3. Captures = body identifiers - parameters - local bindings
///
/// # Example
/// ```ignore
/// move |x: u32| x * factor + offset
/// // params = ["x"]
/// // body_idents = ["x", "factor", "offset"]
/// // captures = ["factor", "offset"]
/// ```
fn extract_closure_captures(closure: &syn::ExprClosure) -> Vec<syn::Ident> {
    // Step 1: Get parameter names
    let params: HashSet<String> = closure
        .inputs
        .iter()
        .filter_map(|pat| {
            // Handle both `|x|` and `|x: Type|` patterns
            match pat {
                syn::Pat::Ident(pat_ident) => Some(pat_ident.ident.to_string()),
                syn::Pat::Type(pat_type) => {
                    if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                        Some(pat_ident.ident.to_string())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
        .collect();

    // Step 2: Walk body and collect identifiers
    let mut visitor = IdentCollector::new();
    syn::visit::visit_expr(&mut visitor, &closure.body);

    // Step 3: Filter to get captures (identifiers that aren't parameters)
    let mut seen = HashSet::new();
    visitor
        .idents
        .into_iter()
        .filter(|id| {
            let name = id.to_string();
            // Keep if: not a parameter, not a placeholder (_), not already seen
            !params.contains(&name) && !name.starts_with('_') && seen.insert(name)
            // dedup
        })
        .collect()
}

/// Try to extract closure from an expression
fn as_closure_expr(expr: &syn::Expr) -> Option<&syn::ExprClosure> {
    match expr {
        syn::Expr::Closure(closure) => Some(closure),
        _ => None,
    }
}

/// Argument type for cuda_launch! - same as LaunchArg but renamed for clarity
enum CudaLaunchArg {
    /// Direct expression - passed via .arg()
    Direct(syn::Expr),
    /// Slice with explicit length - passed as ptr + len
    SliceWithLen(syn::Expr),
    /// Mutable slice with explicit length - passed as ptr + len
    SliceMutWithLen(syn::Expr),
    /// Closure expression - captures extracted and passed as individual args
    Closure {
        /// The original closure expression (for type inference and monomorphization)
        closure_expr: syn::ExprClosure,
        /// Captured variables extracted from the closure body
        captures: Vec<syn::Ident>,
        /// Whether this is a `move` closure (captures by value) or non-move (captures by reference)
        ///
        /// - `move` closure: captures are values, pass `&cap`
        /// - non-move closure: captures are references, pass `&(&cap as *const _)` (the address)
        is_move: bool,
    },
}

impl Parse for CudaLaunchArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Check for tagged arguments
        if input.peek(Ident) {
            let ident: Ident = input.fork().parse()?;
            match ident.to_string().as_str() {
                "slice" => {
                    input.parse::<Ident>()?;
                    let content;
                    parenthesized!(content in input);
                    let expr: syn::Expr = content.parse()?;
                    return Ok(CudaLaunchArg::SliceWithLen(expr));
                }
                "slice_mut" => {
                    input.parse::<Ident>()?;
                    let content;
                    parenthesized!(content in input);
                    let expr: syn::Expr = content.parse()?;
                    return Ok(CudaLaunchArg::SliceMutWithLen(expr));
                }
                // "move" keyword starts a move closure
                "move" => {
                    // Parse the full closure expression (move |args| body)
                    let expr: syn::Expr = input.parse()?;
                    if let Some(closure) = as_closure_expr(&expr) {
                        let captures = extract_closure_captures(closure);
                        let is_move = closure.capture.is_some(); // `move` keyword present
                        return Ok(CudaLaunchArg::Closure {
                            closure_expr: closure.clone(),
                            captures,
                            is_move,
                        });
                    }
                    // Not a closure, treat as direct expression
                    return Ok(CudaLaunchArg::Direct(expr));
                }
                _ => {}
            }
        }

        // Check for closure starting with `|` (non-move closure)
        if input.peek(Token![|]) {
            let expr: syn::Expr = input.parse()?;
            if let Some(closure) = as_closure_expr(&expr) {
                let captures = extract_closure_captures(closure);
                let is_move = closure.capture.is_some(); // `move` keyword present (false here)
                return Ok(CudaLaunchArg::Closure {
                    closure_expr: closure.clone(),
                    captures,
                    is_move,
                });
            }
            // Shouldn't happen, but fallback to direct
            return Ok(CudaLaunchArg::Direct(expr));
        }

        // Default: direct expression
        let expr: syn::Expr = input.parse()?;

        // Check if the parsed expression happens to be a closure
        if let Some(closure) = as_closure_expr(&expr) {
            let captures = extract_closure_captures(closure);
            let is_move = closure.capture.is_some(); // `move` keyword present
            return Ok(CudaLaunchArg::Closure {
                closure_expr: closure.clone(),
                captures,
                is_move,
            });
        }

        Ok(CudaLaunchArg::Direct(expr))
    }
}

/// Input for cuda_launch! macro
struct CudaLaunchInput {
    /// Kernel path - can be simple name or path with generics: `scale` or `scale::<f32>`
    kernel: syn::Path,
    stream: syn::Expr,
    module: syn::Expr,
    config: syn::Expr,
    args: Vec<CudaLaunchArg>,
    /// Optional cluster dimensions (x, y, z) for thread block cluster launches.
    /// When present, uses `cuLaunchKernelEx` via `launch_cluster()` instead of `cuLaunchKernel`.
    cluster_dim: Option<syn::Expr>,
}

impl CudaLaunchInput {
    /// Extract the base kernel name (without generics) and generic arguments
    fn kernel_parts(&self) -> (Ident, Option<&syn::PathArguments>) {
        let last_segment = self
            .kernel
            .segments
            .last()
            .expect("kernel path must have segments");
        let base_name = last_segment.ident.clone();
        let generics = match &last_segment.arguments {
            syn::PathArguments::None => None,
            args => Some(args),
        };
        (base_name, generics)
    }

    /// Check if this is a generic kernel (has type parameters)
    fn is_generic(&self) -> bool {
        self.kernel
            .segments
            .last()
            .map(|seg| !matches!(seg.arguments, syn::PathArguments::None))
            .unwrap_or(false)
    }
}

impl Parse for CudaLaunchInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut kernel = None;
        let mut stream = None;
        let mut module = None;
        let mut config = None;
        let mut args = Vec::new();
        let mut cluster_dim = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "kernel" => kernel = Some(input.parse()?),
                "stream" => stream = Some(input.parse()?),
                "module" => module = Some(input.parse()?),
                "config" => config = Some(input.parse()?),
                "cluster_dim" => cluster_dim = Some(input.parse()?),
                "args" => {
                    let content;
                    bracketed!(content in input);
                    if !content.is_empty() {
                        let parsed: Punctuated<CudaLaunchArg, Token![,]> =
                            Punctuated::parse_terminated(&content)?;
                        args = parsed.into_iter().collect();
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown field: {}. Expected: kernel, stream, module, config, cluster_dim, args",
                            key
                        ),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        Ok(CudaLaunchInput {
            kernel: kernel.ok_or_else(|| syn::Error::new(input.span(), "missing 'kernel'"))?,
            stream: stream.ok_or_else(|| syn::Error::new(input.span(), "missing 'stream'"))?,
            module: module.ok_or_else(|| syn::Error::new(input.span(), "missing 'module'"))?,
            config: config.ok_or_else(|| syn::Error::new(input.span(), "missing 'config'"))?,
            args,
            cluster_dim,
        })
    }
}

/// Launch a CUDA kernel synchronously on a given stream.
///
/// Uses the `CudaKernel` trait (generated by `#[kernel]`) to look up the PTX
/// entry point name. Arguments are marshaled into a `Vec<*mut c_void>` and
/// passed directly to `cuda_core::launch_kernel` (`cuLaunchKernel`).
///
/// # Usage
///
/// ```ignore
/// cuda_launch! {
///     kernel: vecadd,
///     stream: stream,
///     module: module,
///     config: LaunchConfig::for_num_elems(n as u32),
///     args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
/// }
/// ```
///
/// # Fields
///
/// | Field         | Type              | Description                                   |
/// |---------------|-------------------|-----------------------------------------------|
/// | `kernel`      | path              | `#[kernel]` function name (may be generic)    |
/// | `stream`      | `Arc<CudaStream>` | Stream to launch on                           |
/// | `module`      | `Arc<CudaModule>` | Loaded PTX module containing the kernel       |
/// | `config`      | `LaunchConfig`    | Grid/block dimensions, shared memory          |
/// | `cluster_dim` | `(u32,u32,u32)`   | *(optional)* Cluster dims for `cuLaunchKernelEx` |
/// | `args`        | `[arg, ...]`      | Kernel arguments (see below)                  |
///
/// # Argument forms
///
/// - `expr` -- scalar or pointer passed directly
/// - `slice(buf)` -- immutable device buffer; pushes `(cu_deviceptr, len)` as two args
/// - `slice_mut(buf)` -- mutable device buffer; same as `slice` but borrows `&mut`
/// - `move |captures| body` -- closure whose captures are marshaled individually
/// - `|captures| body` -- non-move closure; captures passed as raw pointers (HMM)
///
/// # Returns
///
/// `Result<(), cuda_core::DriverError>` -- the launch is asynchronous, so
/// a successful return only means the launch was enqueued.  Call
/// `stream.synchronize()` to wait for completion.
#[proc_macro]
pub fn cuda_launch(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as CudaLaunchInput);

    let _kernel_path = &input.kernel;
    let stream = &input.stream;
    let module = &input.module;
    let config = &input.config;
    let cluster_dim = &input.cluster_dim;

    // Get base kernel name and generic arguments
    let (kernel_base, generics) = input.kernel_parts();

    // Build the kernel entry point name (cuda_oxide_kernel_<name>)
    let kernel_entry = format_ident!("cuda_oxide_kernel_{}", kernel_base);

    // Build the marker type name for CudaKernel lookup
    let marker_name = format_ident!("__{}_CudaKernel", kernel_base);

    // Check if any argument is a closure (for special handling)
    let has_closure = input
        .args
        .iter()
        .any(|arg| matches!(arg, CudaLaunchArg::Closure { .. }));

    // Extract closure info if present (for monomorphization)
    let closure_info: Option<(&syn::ExprClosure, &Vec<syn::Ident>)> =
        input.args.iter().find_map(|arg| {
            if let CudaLaunchArg::Closure {
                closure_expr,
                captures,
                is_move: _,
            } = arg
            {
                Some((closure_expr, captures))
            } else {
                None
            }
        });

    // Generate argument marshaling code.
    //
    // Each argument becomes a stack-local variable whose address is pushed
    // into a `Vec<*mut c_void>`. This directly matches what cuLaunchKernel
    // expects: an array of pointers-to-argument-values. No trait dispatch
    // (PushKernelArg) or heap allocation per arg.
    let arg_code: Vec<TokenStream2> = input
        .args
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let val_name = format_ident!("__arg_{}", i);
            match arg {
                CudaLaunchArg::Direct(expr) => {
                    quote! {
                        let mut #val_name = #expr;
                        __args.push(&mut #val_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::SliceWithLen(expr) => {
                    let ptr_name = format_ident!("__arg_{}_ptr", i);
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #val_name = &#expr;
                        let mut #ptr_name = #val_name.cu_deviceptr();
                        let mut #len_name = #val_name.len() as u64;
                        __args.push(&mut #ptr_name as *mut _ as *mut std::ffi::c_void);
                        __args.push(&mut #len_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::SliceMutWithLen(expr) => {
                    let ptr_name = format_ident!("__arg_{}_ptr", i);
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #val_name = &mut #expr;
                        let mut #ptr_name = #val_name.cu_deviceptr();
                        let mut #len_name = #val_name.len() as u64;
                        __args.push(&mut #ptr_name as *mut _ as *mut std::ffi::c_void);
                        __args.push(&mut #len_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::Closure {
                    closure_expr: _,
                    captures,
                    is_move,
                } => {
                    // Each captured variable becomes an individual kernel argument.
                    //
                    // Move closures capture BY VALUE → pass &mut cap.
                    // Non-move closures capture BY REFERENCE → pass &mut (&cap as *const _),
                    // the GPU accesses the host address via HMM.
                    let capture_args: Vec<TokenStream2> = captures
                        .iter()
                        .enumerate()
                        .map(|(ci, cap)| {
                            let cap_name = format_ident!("__cap_{}_{}", i, ci);
                            if *is_move {
                                quote! {
                                    let mut #cap_name = #cap;
                                    __args.push(&mut #cap_name as *mut _ as *mut std::ffi::c_void);
                                }
                            } else {
                                quote! {
                                    let mut #cap_name = &(#cap) as *const _;
                                    __args.push(&mut #cap_name as *mut _ as *mut std::ffi::c_void);
                                }
                            }
                        })
                        .collect();

                    if captures.is_empty() {
                        quote! {}
                    } else {
                        quote! { #(#capture_args)* }
                    }
                }
            }
        })
        .collect();

    // Build the instantiate helper name (for closures)
    let instantiate_name = format_ident!("cuda_oxide_instantiate_{}", kernel_base);

    // Generate the launch call — either regular or cluster.
    //
    // Both paths use the stream-aware cuda_core helpers. Those helpers bind the
    // stream's owning CUDA context to the calling thread and then delegate to
    // the raw cuLaunchKernel/cuLaunchKernelEx wrappers.
    let launch_call = if let Some(cdim) = cluster_dim {
        quote! {
            {
                let __cfg = #config;
                cuda_core::launch_kernel_ex_on_stream(
                    &__func,
                    __cfg.grid_dim,
                    __cfg.block_dim,
                    __cfg.shared_mem_bytes,
                    #cdim,
                    (#stream).as_ref(),
                    &mut __args,
                )
            }
        }
    } else {
        quote! {
            {
                let __cfg = #config;
                cuda_core::launch_kernel_on_stream(
                    &__func,
                    __cfg.grid_dim,
                    __cfg.block_dim,
                    __cfg.shared_mem_bytes,
                    (#stream).as_ref(),
                    &mut __args,
                )
            }
        }
    };

    let expanded = if has_closure {
        let (closure_expr, _captures) = closure_info.expect("has_closure but no closure_info");

        let closure_span = closure_expr.span();
        let start = closure_span.start();
        let line = start.line as u32;
        let col = start.column as u32;

        quote! {
            {
                let __closure = #closure_expr;
                let __ptx_name: &'static str = #instantiate_name(__closure, #line, #col);
                let __func = #module.load_function(__ptx_name).expect("Failed to load kernel function");

                let mut __args: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                unsafe {
                    #launch_call
                }
            }
        }
    } else if input.is_generic() {
        quote! {
            {
                let __kernel_ptr = #kernel_entry #generics as *const ();
                unsafe {
                    let mut __force_mono: *const () = core::ptr::null();
                    core::ptr::write_volatile(&mut __force_mono, __kernel_ptr);
                    let _ = core::ptr::read_volatile(&__force_mono);
                }

                let __ptx_name = <#marker_name #generics as cuda_host::GenericCudaKernel>::ptx_name();
                let __func = #module.load_function(__ptx_name).expect("Failed to load kernel function");

                let mut __args: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                unsafe {
                    #launch_call
                }
            }
        }
    } else {
        quote! {
            {
                const __PTX_NAME: &str = <#marker_name as cuda_host::CudaKernel>::PTX_NAME;
                let __func = #module.load_function(__PTX_NAME).expect("Failed to load kernel function");

                let mut __args: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                unsafe {
                    #launch_call
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// ============================================================================
// cuda_launch_async! Macro (async path via cuda-async)
// ============================================================================

/// Parsed input for the [`cuda_launch_async!`] macro.
///
/// Unlike [`CudaLaunchInput`], this struct has no `stream` field. The stream
/// is assigned later by the [`SchedulingPolicy`] when the returned
/// [`AsyncKernelLaunch`] is `.sync()`'d or `.await`'d.
struct CudaLaunchAsyncInput {
    /// Path to the `#[kernel]` function, possibly with generic arguments.
    kernel: syn::Path,
    /// Expression resolving to an `Arc<CudaModule>` that contains the compiled PTX.
    module: syn::Expr,
    /// Expression resolving to a [`LaunchConfig`] (grid/block dims, shared mem).
    config: syn::Expr,
    /// Kernel arguments: `slice(x)`, `slice_mut(x)`, direct values, or closures.
    args: Vec<CudaLaunchArg>,
}

impl CudaLaunchAsyncInput {
    /// Splits the kernel path into its base identifier and optional generic arguments.
    /// For `vecadd::<f32>` returns `("vecadd", Some(<f32>))`.
    fn kernel_parts(&self) -> (Ident, Option<&syn::PathArguments>) {
        let last_segment = self
            .kernel
            .segments
            .last()
            .expect("kernel path must have segments");
        let base_name = last_segment.ident.clone();
        let generics = match &last_segment.arguments {
            syn::PathArguments::None => None,
            args => Some(args),
        };
        (base_name, generics)
    }

    /// Returns `true` if the kernel path has explicit generic type arguments.
    fn is_generic(&self) -> bool {
        self.kernel
            .segments
            .last()
            .map(|seg| !matches!(seg.arguments, syn::PathArguments::None))
            .unwrap_or(false)
    }
}

/// Parses the `cuda_launch_async! { kernel: ..., module: ..., config: ..., args: [...] }` syntax.
///
/// Fields can appear in any order. The `args` field uses bracket syntax with the same
/// argument forms as `cuda_launch!`: `slice(x)`, `slice_mut(x)`, direct values, or closures.
impl Parse for CudaLaunchAsyncInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut kernel = None;
        let mut module = None;
        let mut config = None;
        let mut args = Vec::new();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "kernel" => kernel = Some(input.parse()?),
                "module" => module = Some(input.parse()?),
                "config" => config = Some(input.parse()?),
                "args" => {
                    let content;
                    bracketed!(content in input);
                    if !content.is_empty() {
                        let parsed: Punctuated<CudaLaunchArg, Token![,]> =
                            Punctuated::parse_terminated(&content)?;
                        args = parsed.into_iter().collect();
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown field: {}. Expected: kernel, module, config, args",
                            key
                        ),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        Ok(CudaLaunchAsyncInput {
            kernel: kernel.ok_or_else(|| syn::Error::new(input.span(), "missing 'kernel'"))?,
            module: module.ok_or_else(|| syn::Error::new(input.span(), "missing 'module'"))?,
            config: config.ok_or_else(|| syn::Error::new(input.span(), "missing 'config'"))?,
            args,
        })
    }
}

/// Launch a CUDA kernel asynchronously, returning a lazy [`AsyncKernelLaunch`].
///
/// Unlike [`cuda_launch!`], this macro does **not** take a `stream:` parameter. The
/// CUDA stream is assigned later by the active [`SchedulingPolicy`] when the returned
/// operation is `.sync()`'d or `.await`'d. This enables lazy composition: multiple
/// launches can be chained with `.and_then()`, run in parallel with `zip!()`, or
/// awaited individually.
///
/// # Fields
///
/// | Field    | Type                | Description                                |
/// |----------|---------------------|--------------------------------------------|
/// | `kernel` | path                | `#[kernel]` function name (may be generic) |
/// | `module` | `Arc<CudaModule>`   | Loaded PTX module containing the kernel    |
/// | `config` | `LaunchConfig`      | Grid/block dimensions, shared memory       |
/// | `args`   | `[arg, ...]`        | Kernel arguments (see below)               |
///
/// # Argument forms
///
/// - `slice(x)` -- immutable device slice; pushes `(ptr, len)` as two kernel args
/// - `slice_mut(x)` -- mutable device slice; same as `slice` but takes `&mut`
/// - `expr` -- scalar or device pointer passed directly
/// - `|captures| body` -- closure whose captures are marshalled as individual args
///
/// # Returns
///
/// An [`AsyncKernelLaunch`] implementing [`DeviceOperation`]. No GPU work is enqueued
/// until the caller schedules it.
///
/// # Usage
///
/// ```ignore
/// use cuda_host::cuda_launch_async;
/// use cuda_core::LaunchConfig;
///
/// let op = cuda_launch_async! {
///     kernel: vecadd,
///     module: module,
///     config: LaunchConfig::for_num_elems(N as u32),
///     args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
/// };
///
/// // Synchronous (blocks calling thread):
/// op.sync()?;
///
/// // Or asynchronous (suspends the async task):
/// // op.await?;
///
/// // Or compose before executing:
/// // let chained = op.and_then(|()| another_op);
/// // chained.sync()?;
/// ```
#[proc_macro]
pub fn cuda_launch_async(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as CudaLaunchAsyncInput);

    let module = &input.module;
    let config = &input.config;
    let (kernel_base, generics) = input.kernel_parts();
    let marker_name = format_ident!("__{}_CudaKernel", kernel_base);

    let arg_code: Vec<TokenStream2> = input
        .args
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let tmp_name = format_ident!("__arg_{}", i);
            match arg {
                CudaLaunchArg::Direct(expr) => {
                    quote! {
                        __launch.push_arg(Box::new(#expr));
                    }
                }
                CudaLaunchArg::SliceWithLen(expr) => {
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #tmp_name = &#expr;
                        __launch.push_arg(Box::new(#tmp_name.cu_deviceptr()));
                        let #len_name = #tmp_name.len() as u64;
                        __launch.push_arg(Box::new(#len_name));
                    }
                }
                CudaLaunchArg::SliceMutWithLen(expr) => {
                    let len_name = format_ident!("__arg_{}_len", i);
                    quote! {
                        let #tmp_name = &mut #expr;
                        __launch.push_arg(Box::new(#tmp_name.cu_deviceptr()));
                        let #len_name = #tmp_name.len() as u64;
                        __launch.push_arg(Box::new(#len_name));
                    }
                }
                CudaLaunchArg::Closure {
                    captures, is_move, ..
                } => {
                    let capture_args: Vec<TokenStream2> = captures
                        .iter()
                        .map(|cap| {
                            if *is_move {
                                quote! { __launch.push_arg(Box::new(#cap)); }
                            } else {
                                quote! {
                                    let __ref_capture = &(#cap) as *const _ as usize;
                                    __launch.push_arg(Box::new(__ref_capture));
                                }
                            }
                        })
                        .collect();
                    quote! { #(#capture_args)* }
                }
            }
        })
        .collect();

    let expanded = if input.is_generic() {
        let kernel_entry = format_ident!("cuda_oxide_kernel_{}", kernel_base);
        quote! {
            {
                let __kernel_ptr = #kernel_entry #generics as *const ();
                unsafe {
                    let mut __force_mono: *const () = core::ptr::null();
                    core::ptr::write_volatile(&mut __force_mono, __kernel_ptr);
                    let _ = core::ptr::read_volatile(&__force_mono);
                }
                let __ptx_name = <#marker_name #generics as cuda_host::GenericCudaKernel>::ptx_name();
                let __func = #module.load_function(__ptx_name)
                    .expect("Failed to load kernel function");
                let mut __launch = cuda_async::launch::AsyncKernelLaunch::new(
                    std::sync::Arc::new(__func),
                );
                #(#arg_code)*
                __launch.set_launch_config(#config);
                __launch
            }
        }
    } else {
        quote! {
            {
                const __PTX_NAME: &str =
                    <#marker_name as cuda_host::CudaKernel>::PTX_NAME;
                let __func = #module.load_function(__PTX_NAME)
                    .expect("Failed to load kernel function");
                let mut __launch = cuda_async::launch::AsyncKernelLaunch::new(
                    std::sync::Arc::new(__func),
                );
                #(#arg_code)*
                __launch.set_launch_config(#config);
                __launch
            }
        }
    };

    TokenStream::from(expanded)
}
