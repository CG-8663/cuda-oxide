# hashmap_v2

## GPU Hashmap v2 — SwissTable-Inspired

A `u32 -> u32` hashmap that adds the three structural ideas behind
hashbrown's SwissTable to the v1 baseline:

1. **Control-byte array.** Probing reads a packed `ctrl: DeviceBuffer<u32>`
   of 1-byte tags (4 tags per word) instead of the full `(key, value)`
   payload. Most probe steps now touch one cache-line of metadata.
2. **h1 / h2 hash split.** The same hash drives two roles — `h1` (low
   bits, probe position) and `h2` (top 7 bits, in-tag fingerprint).
3. **Triangular probing.** Probe step `i` advances by `i` groups, which
   visits every group exactly once and avoids primary clustering.

Plus the operation v1 didn't have:

4. **Tombstone delete.** A successful delete CAS-flips the slot's tag
   from `FULL(h2)` to `DELETED`. Find skips past tombstones; insert in
   v2 cut 1 does **not** reclaim them.

v2 cut 1 is single-thread-per-key — same kernel shape as v1, just over
the new storage. The warp-cooperative probe (the actual SwissTable
lookup with `ballot` as the GPU analog of SSE2 `_mm_movemask_epi8`)
lands in cut 2.

## Build and Run

```bash
cargo oxide run hashmap_v2
```

## Storage Layout

Two device-resident buffers, both `memset_d8_async(0xFF, ...)` at
construction so every tag reads `EMPTY` and every slot reads
`EMPTY_SLOT`:

```text
ctrl: DeviceBuffer<u32>   length N / GROUP   (GROUP = 4)

   one u32 = 4 packed tag bytes:

      31              24 23              16 15               8 7                0
      +------------------+------------------+------------------+------------------+
      |   tag for slot 3 |   tag for slot 2 |   tag for slot 1 |   tag for slot 0 |
      +------------------+------------------+------------------+------------------+
      \_______________________________ ctrl[group_idx] ___________________________/

   tag byte encoding:

      bit:   7   6   5   4   3   2   1   0
            +---+---+---+---+---+---+---+---+
   EMPTY    | 1   1   1   1   1   1   1   1 |   0xFF
   DELETED  | 1   0   0   0   0   0   0   0 |   0x80
   FULL(h2) | 0 |       h2 (7 bits)         |   0x00..0x7F
            +---+---+---+---+---+---+---+---+

slots: DeviceBuffer<u64>   length N (power of two, multiple of GROUP)

   each u64 packs (key, value) — same layout as v1:

      63                 32 31                  0
      +--------------------+--------------------+
      |       key (u32)    |     value (u32)    |
      +--------------------+--------------------+

   sentinel:                0xFFFF_FFFF_FFFF_FFFF      (= u64::MAX)
   forbidden user pair:     key = u32::MAX            (would collide with EMPTY_SLOT)
```

The slot layout is unchanged from v1. The new piece is the parallel
`ctrl` array — probing reads tags first and only touches a slot when a
tag's fingerprint matches.

## Hash Split

```rust
let hash = (key as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
let h1   = hash as usize;                    // probe position
let h2   = ((hash >> 57) & 0x7F) as u8;      // 7-bit fingerprint, top bit clear
```

`h2` is taken from the top of the hash so it's statistically independent
of the low-bit position used for probing — same split hashbrown uses.
The 7-bit format guarantees `h2` can never collide with `EMPTY (0xFF)`
or `DELETED (0x80)`.

## Insert Protocol — Protocol B (Payload-First)

v2 cut 1 ships exactly one of the two protocols described in the design
docs: **Protocol B**, the cuCollections-style payload-first insert. Two
atomics per insert, no RESERVED state, no reader spin.

```text
1. group_idx = (h1 / GROUP) & ((N / GROUP) - 1)
2. word = atomic_load(ctrl[group_idx], Acquire)

   Phase 1 -- already-present check:
   for each byte i with tag(word, i) == h2:
     observed = atomic_load(slots[group_idx*GROUP + i], Acquire)
     if unpack_key(observed) == key:
       last-writer-wins:    slot-CAS overwrite loop, return
       first-writer-wins:   report PRESENT, return

   Phase 2 -- claim an EMPTY slot:
   for each byte i with tag(word, i) == EMPTY:
     match slots[group_idx*GROUP + i].cas(EMPTY_SLOT -> pack(k, v)):
       Ok          => publish: ctrl-word CAS sets byte i to FULL(h2). return
       Err(actual) where unpack_key(actual) == k => same-key race:
                       last-writer-wins:  slot-CAS overwrite loop, return
                       first-writer-wins: report PRESENT, return
       Err(actual) => different key in flight; skip byte, try next

3. No FULL(h2) match and no EMPTY in this group: stride += 1;
   group_idx = (group_idx + stride) & ((N / GROUP) - 1); loop.
```

The slot CAS itself is the serialization point. Two threads racing on
the same key always converge on the same slot (deterministic probe
order), and exactly one wins the CAS; the other observes `Err(actual)`
with a matching key and falls into the duplicate-handling path. Two
threads racing on different keys at the same slot likewise have one
winner, and the loser sees the mismatched key and probes past.

The publish step is a small CAS retry loop because **other** threads
may concurrently mutate **other** bytes in the same ctrl word for their
own inserts. Byte `i` itself can never change under us — no other
thread can claim a slot we already own — so the loop terminates as soon
as we observe a stable view of the other three bytes.

## Find

```text
1. group_idx = (h1 / GROUP) & ((N / GROUP) - 1)
2. word = atomic_load(ctrl[group_idx], Acquire)

   for each byte i in 0..GROUP:
     if tag(word, i) == h2:
       observed = atomic_load(slots[group_idx*GROUP + i], Acquire)
       if unpack_key(observed) == key:
         return unpack_value(observed)

   if any byte in word is EMPTY: return MISS  (key cannot be in this chain)

3. stride += 1; group_idx = (group_idx + stride) & mask; loop.
```

Find skips past `DELETED` tags (they don't terminate the probe) and
stops on `EMPTY` (which does, because no later insert could legally
land beyond an empty slot in this hash chain).

## Delete

Tombstone-only. Probe like find; on key match, CAS the ctrl word from
the byte's current value to `DELETED`. The `(key, value)` payload is
left as-is — readers only ever load slots whose tag is `FULL(h2)`, so a
stale slot with a `DELETED` tag is unreachable.

v2 cut 1 deliberately does **not** reclaim deleted slots on insert. A
delete-then-reinsert sequence still works (the new entry lands at a
fresh slot, find walks past the tombstone), at the cost of effective
capacity erosion under churn. Reclaim arrives with the rehash path.

## Probe-Step Width

All four kernels use the **same** triangular probe sequence with a
32-tag-byte tile per step (`PROBE_TILE`). Single-thread kernels iterate
the 32 bytes serially via a `GROUP`-sized inner loop (one ctrl word at
a time, four bytes each). The warp-cooperative find kernel inspects all
32 bytes in parallel — one lane per byte, decision via `ballot`.

Insert and find **must** share the probe shape: if one walked
4-byte tiles and the other walked 32-byte tiles, find could terminate
on an `EMPTY` slot in the wider window that insert had skipped over,
and miss perfectly valid keys.

## Warp-Cooperative Find

`find_bulk_warp` launches one warp (32 lanes) per query key. Each lane
owns one tag byte at `(probe_base + lane)`. Per probe step:

```text
1. Coalesced load of 8 ctrl words (= 32 tag bytes) into the warp.
2. m_h2    = ballot(tag == h2(K))   -> 32-bit fingerprint match mask
3. m_empty = ballot(tag == EMPTY)   -> 32-bit empty mask
4. while m_h2 != 0:
       cand = trailing_zeros(m_h2)
       lane `cand` loads slots[probe_base + cand]
       broadcast (key, value) via two `shuffle`s, all lanes key-compare
       hit  -> lane 0 writes out[warp_idx], return
       miss -> m_h2 &= m_h2 - 1, try next candidate
5. if m_empty != 0: lane 0 writes MISS, return
6. else: triangular advance, repeat
```

The single-thread `find_kernel` stays in the binary as the comparison
baseline. Both produce identical results on every input (verified by
Test 8); the bench harness in cut 2 measures the throughput crossover.

## Correctness Tests (nine)

| # | Name                           | What it verifies                                                  |
|:--|:-------------------------------|:------------------------------------------------------------------|
| 1 | `insert_bulk` roundtrip        | every inserted key is findable with the inserted value            |
| 2 | miss on absent keys            | disjoint key set must miss                                        |
| 3 | last-writer-wins on re-insert  | second `insert_bulk` overwrites every value                       |
| 4 | `try_insert_bulk` first-writer | pass 2 reports all-present, table preserves pass-1 values         |
| 5 | load-factor stress (~75%)      | 12288 keys at 75% load all round-trip                             |
| 6 | delete-then-find               | survivors hit with original values, deleted keys all return MISS  |
| 7 | delete-then-reinsert           | re-inserted keys observable with new values, even past tombstones |
| 8 | warp-coop find parity          | warp-coop and single-thread find agree on 16384 mixed queries     |
| 9 | warp-coop find at ~75% load    | 12288 keys round-trip via the warp-coop kernel under load         |

Default capacity is `1 << 14` slots.

## Intentionally Out of Scope (still)

- **Protocol A (RESERVED-tag handshake)** — the hashbrown-faithful
  three-CAS protocol lands alongside the bench harness so we can
  measure both protocols head-to-head.
- **16-lane sub-warp tiles** — would need `ballot_sync` /
  `shuffle_sync` with sub-warp masks, which arrives with the
  cooperative-groups design pass.
- **DELETED slot reclaim on insert** — left for v3 with rehash. With
  enough delete-insert churn the table fragments and effective capacity
  shrinks; rehash compacts both at once.
- **Resize / rehash** — left for v3, blocked on cooperative-groups
  (grid-wide barrier).
- **Generic `<K, V>`** — deferred for API design reasons.
- **Float keys** — PTX has no `compare_exchange` for floats.

## Next Steps

| Cut   | What lands                                                                                   |
|:------|:---------------------------------------------------------------------------------------------|
|   2.1 | done — single-thread + warp-coop find on a unified `PROBE_TILE` triangular probe             |
|   2.2 | Protocol A insert (RESERVED-tag handshake) as a parallel kernel; correctness tests           |
|   2.3 | `--bench` harness: insert (B vs A) × find (single vs warp) at 50 / 75 / 90% load             |
|   3   | Cooperative-groups in cuda-oxide; then 16-lane subwarp find, rehash, DELETED reclaim, resize |
