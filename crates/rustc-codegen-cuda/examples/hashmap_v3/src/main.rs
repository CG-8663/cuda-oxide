/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `hashmap_v3` correctness tests — 11 GPU hardware checks for the
//! cooperative-groups SwissTable hashmap.
//!
//! Kernels and the `GpuSwissMap` host driver live in `lib.rs` (shared
//! with the bench binary). This `main` only drives the test scenarios
//! and prints PASS / FAIL per test.
//!
//! Build and run with:
//!   cargo oxide run hashmap_v3

use cuda_core::CudaContext;
use hashmap_v3::*;

fn main() {
    println!("=== GPU Hashmap v3 — Cooperative-Groups SwissTable ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("hashmap_v3.ptx")
        .expect("Failed to load PTX module");

    const CAPACITY: usize = 1 << 14;
    const M: usize = CAPACITY / 2;

    let keys_v0 = distinct_keys(M, 0xDEAD_BEEF);
    let values_v0: Vec<u32> = (0..M as u32).collect();

    // -------------------------------------------------------------------------
    // Test 1: insert_bulk roundtrip (parity with v1).
    // -------------------------------------------------------------------------
    println!("--- Test 1: insert_bulk roundtrip ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");
        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");

        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v0[i] {
                bad += 1;
            }
        }
        if bad == 0 {
            println!("  {M} / {M} keys round-tripped, capacity = {CAPACITY}");
        } else {
            println!("  FAIL: {bad} mismatches");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: miss on absent keys (parity with v1).
    // -------------------------------------------------------------------------
    println!("\n--- Test 2: miss on absent keys ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0xC0FF_EEEEu32;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }

        let found = map.find_bulk(&absent, &module, &stream).expect("find");
        let bad: usize = found.iter().filter(|&&v| v != MISS).count();
        if bad == 0 {
            println!("  {M} / {M} absent keys correctly missed");
        } else {
            println!("  FAIL: {bad} absent keys returned non-MISS");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: last-writer-wins on re-insert (parity with v1).
    // -------------------------------------------------------------------------
    println!("\n--- Test 3: insert_bulk last-writer-wins on re-insert ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert v0");

        let values_v1: Vec<u32> = values_v0.iter().map(|v| v ^ 0xA5A5_A5A5).collect();
        map.insert_bulk(&keys_v0, &values_v1, &module, &stream)
            .expect("insert v1");

        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");
        let mut bad = 0usize;
        for i in 0..M {
            if found[i] != values_v1[i] {
                bad += 1;
            }
        }
        if bad == 0 {
            println!("  all {M} keys reflect last-writer values");
        } else {
            println!("  FAIL: {bad} stale values");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: try_insert_bulk first-writer-wins (parity with v1).
    // -------------------------------------------------------------------------
    println!("\n--- Test 4: try_insert_bulk first-writer-wins ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let first_flags = map
            .try_insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("try_insert pass 1");
        let all_fresh = first_flags.iter().all(|&b| b);

        let values_v1: Vec<u32> = values_v0.iter().map(|v| v ^ 0xA5A5_A5A5).collect();
        let second_flags = map
            .try_insert_bulk(&keys_v0, &values_v1, &module, &stream)
            .expect("try_insert pass 2");
        let none_fresh = second_flags.iter().all(|&b| !b);

        let found = map.find_bulk(&keys_v0, &module, &stream).expect("find");
        let bad: usize = found
            .iter()
            .zip(&values_v0)
            .filter(|(f, v)| *f != *v)
            .count();

        if all_fresh && none_fresh && bad == 0 {
            println!("  pass 1: all {M} keys fresh");
            println!("  pass 2: no keys fresh, table preserves v0 values");
        } else {
            println!("  FAIL: all_fresh={all_fresh} none_fresh={none_fresh} bad_values={bad}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 5: load-factor stress at 75% (parity with v1 test 6).
    // -------------------------------------------------------------------------
    println!("\n--- Test 5: load-factor stress (~75%) ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0x1234_5678);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk(&dense_keys, &module, &stream)
            .expect("dense find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped at load factor {:.1}%, capacity = {}",
                100.0 * m_dense as f64 / map.capacity() as f64,
                map.capacity()
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: delete-then-find — deleted keys must miss; survivors must hit.
    // -------------------------------------------------------------------------
    println!("\n--- Test 6: delete-then-find ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        // Delete the first third.
        let third = M / 3;
        let to_delete = &keys_v0[..third];
        let survivors = &keys_v0[third..];

        let del_flags = map
            .delete_bulk(to_delete, &module, &stream)
            .expect("delete");
        let all_deleted = del_flags.iter().all(|&b| b);

        // Survivors must still hit with their original values.
        let surv_found = map
            .find_bulk(survivors, &module, &stream)
            .expect("find survivors");
        let mut bad_surv = 0usize;
        for (i, k_idx) in (third..M).enumerate() {
            if surv_found[i] != values_v0[k_idx] {
                bad_surv += 1;
            }
        }

        // Deleted keys must miss.
        let deleted_found = map
            .find_bulk(to_delete, &module, &stream)
            .expect("find deleted");
        let bad_del: usize = deleted_found.iter().filter(|&&v| v != MISS).count();

        if all_deleted && bad_surv == 0 && bad_del == 0 {
            println!(
                "  deleted {third} keys; {} survivors all hit, {third} deleted all miss",
                M - third
            );
        } else {
            println!("  FAIL: all_deleted={all_deleted} bad_surv={bad_surv} bad_del={bad_del}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 7: delete-then-reinsert — re-inserted key sees new value.
    //         (v3 does not yet reclaim DELETED slots on insert, so
    //          the new entry lands at a fresh slot; find still
    //          returns the new value because it walks past the
    //          tombstone.)
    // -------------------------------------------------------------------------
    println!("\n--- Test 7: delete-then-reinsert ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert v0");

        let third = M / 3;
        let to_recycle = &keys_v0[..third];
        let _ = map
            .delete_bulk(to_recycle, &module, &stream)
            .expect("delete");

        let new_values: Vec<u32> = values_v0[..third]
            .iter()
            .map(|v| v.wrapping_add(0x10_0000))
            .collect();
        map.insert_bulk(to_recycle, &new_values, &module, &stream)
            .expect("re-insert");

        let found = map
            .find_bulk(to_recycle, &module, &stream)
            .expect("find re-inserted");
        let bad: usize = found
            .iter()
            .zip(&new_values)
            .filter(|(f, v)| *f != *v)
            .count();

        if bad == 0 {
            println!("  {third} keys re-inserted with new values, all observable");
        } else {
            println!("  FAIL: {bad} re-inserted keys returned stale or missing values");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 8: warp-cooperative find parity with single-thread find on a
    //         mixed hit/miss query batch at moderate load (50%).
    // -------------------------------------------------------------------------
    println!("\n--- Test 8: warp-cooperative find parity ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        // Build a query batch that's half hits (the inserted keys) and
        // half misses (fresh random keys). Total = 2 * M queries.
        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0x0BAD_F00D;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }
        let mut query_keys = Vec::with_capacity(2 * M);
        query_keys.extend_from_slice(&keys_v0);
        query_keys.extend_from_slice(&absent);

        let single = map
            .find_bulk(&query_keys, &module, &stream)
            .expect("find single");
        let tile_32 = map
            .find_bulk_tile_32(&query_keys, &module, &stream)
            .expect("find tile_32");

        let bad: usize = single
            .iter()
            .zip(&tile_32)
            .filter(|(s, w)| *s != *w)
            .count();
        let hits: usize = tile_32.iter().filter(|&&v| v != MISS).count();

        if bad == 0 && hits == M {
            println!(
                "  tile_32 find matches single-thread find on {} queries ({hits} hits, {} misses)",
                query_keys.len(),
                query_keys.len() - hits
            );
        } else {
            println!("  FAIL: bad={bad} hits={hits} (expected {M})");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 9: tile_32 find at 75% load — the regime where warp-cooperative
    //         is supposed to win against single-thread find.
    // -------------------------------------------------------------------------
    println!("\n--- Test 9: tile_32 find at ~75% load ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0xFEED_FACE);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk_tile_32(&dense_keys, &module, &stream)
            .expect("tile_32 find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped via tile_32 find at load factor {:.1}%",
                100.0 * m_dense as f64 / map.capacity() as f64,
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 10: tile_16 parity vs tile_32 on 16384 mixed (hit + miss)
    //          queries. Same table populated via insert_bulk; both
    //          find variants must agree on every result.
    // -------------------------------------------------------------------------
    println!("\n--- Test 10: tile_16 vs tile_32 parity on 16384 mixed queries ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        map.insert_bulk(&keys_v0, &values_v0, &module, &stream)
            .expect("insert");

        let inserted: std::collections::HashSet<u32> = keys_v0.iter().copied().collect();
        let mut absent = Vec::with_capacity(M);
        let mut state: u32 = 0x1234_5678u32;
        while absent.len() < M {
            let k = xorshift32(&mut state);
            if k != FORBIDDEN_KEY && !inserted.contains(&k) {
                absent.push(k);
            }
        }
        let mut query_keys = Vec::with_capacity(2 * M);
        query_keys.extend_from_slice(&keys_v0);
        query_keys.extend_from_slice(&absent);

        let tile_32 = map
            .find_bulk_tile_32(&query_keys, &module, &stream)
            .expect("tile_32");
        let tile_16 = map
            .find_bulk_tile_16(&query_keys, &module, &stream)
            .expect("tile_16");

        let bad: usize = tile_32
            .iter()
            .zip(&tile_16)
            .filter(|(a, b)| *a != *b)
            .count();
        let hits: usize = tile_16.iter().filter(|&&v| v != MISS).count();

        if bad == 0 && hits == M {
            println!(
                "  tile_16 matches tile_32 on {} queries ({hits} hits, {} misses)",
                query_keys.len(),
                query_keys.len() - hits
            );
        } else {
            println!("  FAIL: bad={bad} hits={hits} (expected {M})");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Test 11: tile_16 find at ~75% load — sub-warp variant under load.
    // -------------------------------------------------------------------------
    println!("\n--- Test 11: tile_16 find at ~75% load ---");
    {
        let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
        let m_dense = (CAPACITY * 3) / 4;
        let dense_keys = distinct_keys(m_dense, 0xFEED_FACE);
        let dense_values: Vec<u32> = (0..m_dense as u32).collect();
        map.insert_bulk(&dense_keys, &dense_values, &module, &stream)
            .expect("dense insert");

        let found = map
            .find_bulk_tile_16(&dense_keys, &module, &stream)
            .expect("tile_16 find");
        let bad: usize = found
            .iter()
            .zip(&dense_values)
            .filter(|(f, v)| *f != *v)
            .count();
        if bad == 0 {
            println!(
                "  {m_dense} keys round-tripped via tile_16 find at load factor {:.1}%",
                100.0 * m_dense as f64 / map.capacity() as f64,
            );
        } else {
            println!("  FAIL: {bad} mismatches under load");
            std::process::exit(1);
        }
    }

    println!("\n=== SUCCESS: All 11 hashmap v3 tests passed! ===");
}
