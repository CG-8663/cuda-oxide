# warp_reduce

## Warp-Level Primitives - Shuffle Operations

Demonstrates warp-level operations: `shuffle_xor`, `shuffle_down`, and `shuffle` (broadcast). These are the fastest way to communicate between threads without shared memory.

## What This Example Does

1. **warp_reduce_sum**: Butterfly reduction using `shuffle_xor` - all lanes get the sum
2. **warp_reduce_sum_down**: Sequential reduction using `shuffle_down` - only lane 0 gets sum
3. **warp_broadcast**: Broadcast lane 0's value to all lanes using `shuffle`
4. **test_lane_id**: Verify `lane_id()` intrinsic

## Key Concepts Demonstrated

### Butterfly Reduction (shuffle_xor)

```rust
// All lanes end up with the complete sum
val = val + warp::shuffle_xor_f32(val, 16);  // Exchange with lane ± 16
val = val + warp::shuffle_xor_f32(val, 8);   // Exchange with lane ± 8
val = val + warp::shuffle_xor_f32(val, 4);   // Exchange with lane ± 4
val = val + warp::shuffle_xor_f32(val, 2);   // Exchange with lane ± 2
val = val + warp::shuffle_xor_f32(val, 1);   // Exchange with lane ± 1
// val now contains sum of all 32 lanes
```

### Sequential Reduction (shuffle_down)

```rust
// Only lane 0 gets the final sum
val = val + warp::shuffle_down_f32(val, 16);
val = val + warp::shuffle_down_f32(val, 8);
val = val + warp::shuffle_down_f32(val, 4);
val = val + warp::shuffle_down_f32(val, 2);
val = val + warp::shuffle_down_f32(val, 1);
// Lane 0 has the sum; other lanes have partial sums
```

### Broadcast

```rust
// Get lane 0's value to all lanes
let broadcast_val = warp::shuffle_f32(my_val, 0);
// All lanes now have lane 0's original value
```

### Lane ID

```rust
let lane = warp::lane_id();  // 0-31 within the warp
let warp_id = warp::warp_id();  // Which warp in the block
```

## Build and Run

```bash
cargo oxide run warp_reduce
```

## Expected Output

```text
=== Unified Warp Reduction Example ===

Input data: 256 elements, 8 warps
  First warp values: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
  Expected warp sum: 496

--- Test 1: Butterfly Reduction (shuffle_xor) ---
Warp sums: [496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0]
✓ All 8 warp sums correct (each = 496)

--- Test 2: Sequential Reduction (shuffle_down) ---
Warp sums: [496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0]
✓ All 8 warp sums correct (each = 496)

--- Test 3: Broadcast (shuffle to lane 0) ---
✓ Broadcast correct: all lanes have lane 0's value

--- Test 4: Lane ID ---
✓ Lane IDs correct: 0-31 pattern for each warp

✓ SUCCESS: All warp tests passed!
```

## Hardware Requirements

- **Minimum GPU**: Kepler (sm_30) or newer
- **CUDA Driver**: 11.0+

## Shuffle Functions

| Function                       | Description                      | Result Location |
|--------------------------------|----------------------------------|-----------------|
| `shuffle_f32(val, src_lane)`   | Read from specific lane          | All lanes       |
| `shuffle_xor_f32(val, mask)`   | Exchange with lane XOR'd by mask | All lanes       |
| `shuffle_down_f32(val, delta)` | Read from lane + delta           | Lower lanes     |
| `shuffle_up_f32(val, delta)`   | Read from lane - delta           | Upper lanes     |

## Why Warp Operations?

| Communication   | Latency     | Synchronization              |
|-----------------|-------------|------------------------------|
| Shuffle         | ~1 cycle    | Implicit (warp-synchronous)  |
| Shared Memory   | ~20 cycles  | Requires sync_threads()      |
| Global Memory   | ~400 cycles | Requires barriers            |

Shuffles are:
- **Lock-step**: All 32 lanes execute together
- **No synchronization needed**: Warps are SIMD units
- **Register-to-register**: No memory accesses

## Common Patterns

### Warp-Wide Max

```rust
val = max(val, warp::shuffle_xor_f32(val, 16));
val = max(val, warp::shuffle_xor_f32(val, 8));
val = max(val, warp::shuffle_xor_f32(val, 4));
val = max(val, warp::shuffle_xor_f32(val, 2));
val = max(val, warp::shuffle_xor_f32(val, 1));
```

### Prefix Sum (Scan)

```rust
let mut offset = 1;
while offset < 32 {
    let n = warp::shuffle_up_f32(val, offset);
    if lane >= offset { val += n; }
    offset *= 2;
}
```

### Warp-Level Output

```rust
if lane == 0 {
    let warp_idx = gid / 32;
    output[warp_idx] = val;  // One write per warp
}
```

## Generated PTX

```ptx
// Shuffle XOR
shfl.sync.bfly.b32 %f_result, %f_val, %r_mask, 0x1f, 0xffffffff;

// Shuffle down
shfl.sync.down.b32 %f_result, %f_val, %r_delta, 0x1f, 0xffffffff;

// Broadcast (shuffle to lane 0)
shfl.sync.idx.b32 %f_result, %f_val, 0, 0x1f, 0xffffffff;

// Lane ID
mov.u32 %r_lane, %laneid;
```
