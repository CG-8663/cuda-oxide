/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Peer topology discovery for collective participants.

use crate::error::{CollectiveError, Result};
use cuda_core::{CudaContext, peer};
use std::collections::HashSet;
use std::sync::Arc;

/// Pairwise P2P reachability between the participating PEs.
#[derive(Clone, Debug)]
pub struct Topology {
    /// CUDA device ordinals in PE order.
    ordinals: Vec<usize>,
    /// Directed reachability matrix: `adjacency[from][to]` is `true` when `from`
    /// can directly access memory owned by `to`.
    adjacency: Vec<Vec<bool>>,
}

impl Topology {
    /// Discovers pairwise peer connectivity for `contexts` in PE order.
    ///
    /// The resulting topology is directional because CUDA peer reachability is
    /// queried direction-by-direction with `cuDeviceCanAccessPeer`.
    pub fn discover(contexts: &[Arc<CudaContext>]) -> Result<Self> {
        if contexts.is_empty() {
            return Err(CollectiveError::NoContexts);
        }

        let mut seen = HashSet::with_capacity(contexts.len());
        let ordinals: Vec<usize> = contexts.iter().map(|ctx| ctx.ordinal()).collect();
        for &ordinal in &ordinals {
            if !seen.insert(ordinal) {
                return Err(CollectiveError::DuplicateDevice(ordinal));
            }
        }

        let pe_count = contexts.len();
        let mut adjacency = vec![vec![false; pe_count]; pe_count];
        for from in 0..pe_count {
            adjacency[from][from] = true;
            for to in 0..pe_count {
                if from == to {
                    continue;
                }
                adjacency[from][to] = peer::can_access_peer(&contexts[from], &contexts[to])?;
            }
        }

        Ok(Self {
            ordinals,
            adjacency,
        })
    }

    /// Number of participating PEs.
    pub fn pe_count(&self) -> usize {
        self.ordinals.len()
    }

    /// CUDA device ordinals in PE order.
    pub fn ordinals(&self) -> &[usize] {
        &self.ordinals
    }

    /// Returns whether `from` can directly access `to`.
    ///
    /// `from` and `to` are PE indices, not CUDA device ordinals.
    pub fn can_access(&self, from: usize, to: usize) -> Result<bool> {
        self.validate_pe(from)?;
        self.validate_pe(to)?;
        Ok(self.adjacency[from][to])
    }

    /// Returns `true` when every PE can directly reach every other PE.
    pub fn is_fully_connected(&self) -> bool {
        for from in 0..self.pe_count() {
            for to in 0..self.pe_count() {
                if from != to && !self.adjacency[from][to] {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the first missing peer edge as an error.
    ///
    /// Phase 2's [`crate::SymmetricHeap`] currently requires a full peer mesh
    /// because every PE maps every other PE's chunk into its local VA window.
    pub fn require_full_mesh(&self) -> Result<()> {
        for from in 0..self.pe_count() {
            for to in 0..self.pe_count() {
                if from != to && !self.adjacency[from][to] {
                    return Err(CollectiveError::PeerAccessUnavailable {
                        from,
                        from_ordinal: self.ordinals[from],
                        to,
                        to_ordinal: self.ordinals[to],
                    });
                }
            }
        }
        Ok(())
    }

    /// Enables peer access for each reachable directed edge.
    ///
    /// This is idempotent: `cuda-core` treats CUDA's "already enabled" result
    /// as success.
    pub fn enable_peer_access(&self, contexts: &[Arc<CudaContext>]) -> Result<()> {
        if contexts.len() != self.pe_count() {
            return Err(CollectiveError::TopologyMismatch {
                expected_pes: self.pe_count(),
                actual_contexts: contexts.len(),
            });
        }

        for from in 0..self.pe_count() {
            for to in 0..self.pe_count() {
                if from == to || !self.adjacency[from][to] {
                    continue;
                }
                peer::enable_peer_access(&contexts[from], &contexts[to])?;
            }
        }
        Ok(())
    }

    /// Validates that `pe` names one of the known participants.
    fn validate_pe(&self, pe: usize) -> Result<()> {
        if pe < self.pe_count() {
            Ok(())
        } else {
            Err(CollectiveError::InvalidPe {
                pe,
                pe_count: self.pe_count(),
            })
        }
    }
}
