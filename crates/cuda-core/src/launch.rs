/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel launch configuration.
//!
//! [`LaunchConfig`] bundles the grid dimensions, block dimensions, and dynamic
//! shared memory size required by [`launch_kernel`](crate::launch_kernel). Use
//! [`LaunchConfig::for_num_elems`] as a quick 1-D launch helper.

/// Grid and block dimensions plus dynamic shared memory size for a kernel
/// launch.
///
/// Each dimension tuple is `(x, y, z)`. Pass this to
/// [`launch_kernel`](crate::launch_kernel) or destructure it manually for
/// custom launch wrappers.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// Grid dimensions `(x, y, z)` in blocks.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)` in threads.
    pub block_dim: (u32, u32, u32),
    /// Bytes of dynamic shared memory allocated per block.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a 1-D launch configuration for `n` elements.
    ///
    /// Uses a block size of 256 threads and computes the grid size via
    /// ceiling division. No dynamic shared memory is requested.
    ///
    /// Suitable for simple element-wise kernels where thread index maps
    /// directly to element index.
    pub fn for_num_elems(n: u32) -> Self {
        const DEFAULT_BLOCK_SIZE: u32 = 256;
        let grid_x = (n + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
        LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}
