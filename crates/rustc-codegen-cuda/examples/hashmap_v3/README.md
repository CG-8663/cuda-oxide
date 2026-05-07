# hashmap_v3

> v3 of the GPU SwissTable hashmap; under active development.

## Status

This crate is being built up incrementally. Today it ships:

- Payload-first insert (slot-CAS-first), tombstone delete, and
  single-thread find — all carried over from v2.
- A typed cooperative-groups warp-cooperative find, parameterised
  over a const-generic lane-tile size: `find_kernel_tile_32` (full
  warp, one query per warp) and `find_kernel_tile_16` (sub-warp,
  two queries per warp). The same v3 table is queryable by either.

The full v3 README — algorithm walk-through, decision rationale,
bench tables, version-progression — ships once the rest of the
roadmap (DELETED-slot reclaim on insert, single-kernel `grid.sync()`
rehash, host-side `resize`) lands.

For the v2 documentation it is layered on, see the `hashmap_v2/`
example crate alongside this one.

## Build and run

```bash
# Default binary: 11 GPU correctness tests.
cargo oxide run hashmap_v3

# Perf bench (GPU vs CPU `hashbrown`).
./crates/rustc-codegen-cuda/examples/hashmap_v3/run-bench.sh
```

## Bench snapshot (RTX 5090, sm_120)

```text
Find — lookup (every query hits)
                          load=50%   load=75%   load=90%
GPU single-thread          23095.6   19498.6   17726.3
GPU tile_32 (1 key/warp)    7117.0    7115.0    7081.6
GPU tile_16 (2 keys/warp)   9451.9    8574.1    8378.2
tile_16 / tile_32             1.3x      1.2x      1.2x
```

`tile_16` beats `tile_32` by 1.2-1.3x on hits at every load level.
The single-thread kernel still wins overall (per-key work is small
relative to warp-coordination overhead), as in v2 — that's a
separate optimisation axis from the warp-vs-sub-warp tile choice.
