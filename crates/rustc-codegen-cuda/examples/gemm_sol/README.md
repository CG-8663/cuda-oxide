# gemm_sol — Speed-of-Light GEMM Kernels

Speed-of-light GEMM kernels for Blackwell (sm_100) using tcgen05 tensor cores, TMA, and no host-side pre-tiling.

## Kernels

### Phase 1: `gemm_sol_tiled` (K-Loop + Grid Tiling, SWIZZLE_NONE)

1. **Host allocates flat row-major buffers** — A is M×K, B is N×K (transposed, K contiguous). No pre-tiling, no 8×8 rearrangement. Raw `memcpy` to device.

2. **TMA descriptors use small boxes** — `cuTensorMapEncodeTiled` with `box_dim=[8, 128]`. Each TMA copy fetches 8 K-elements for 128 M/N rows. The kernel issues 8 copies per K-tile to build a tiled SMEM layout where each 128×8 block has 16-byte row strides, matching the core matrix format the MMA hardware requires.

3. **Grid launch puts one CTA per 128×128 output tile** — `grid_dim=(M/128, N/128)`, `block_dim=128`. Each CTA uses `blockIdx` to pick its tile of C.

4. **K-loop iterates over the K dimension** — `K/64` outer iterations. Each iteration: 8 TMA copies load a 128×64 tile of A and B into SMEM (tiled layout), then 4 sequential tcgen05 MMAs (128×128×16 each) accumulate into TMEM. Phase-based mbarrier synchronization between TMA and MMA.

5. **SMEM descriptors use compile-time SBO/LBO** — after TMA copies, each K-group block in SMEM is 128 rows × 8 elements with 16-byte row stride. SBO=128 bytes (stride between 8-row groups within a K-group block), LBO=2048 bytes (stride between K-group blocks). These are constants baked into PTX, not launch parameters.

6. **Epilogue writes TMEM → registers → SMEM → global** — `tcgen05_ld` reads f32 accumulators from TMEM, `cvt_f32x2_bf16x2` converts to bf16, `stmatrix` writes to SMEM, then threads scatter-copy to the correct global C location.

### Phase 1.5: `gemm_sol_swizzled` (SWIZZLE_128B, single TMA copy)

Same structure as Phase 1 with one key change: TMA uses `SWIZZLE_128B` to copy an entire 128×64 tile in a **single instruction** per matrix, replacing the 8-copy approach.

Changes from Phase 1:
- **TMA**: `box_dim=[64, 128]` with `CU_TENSOR_MAP_SWIZZLE_128B` — 1 copy per matrix per K-tile (was 8)
- **SMEM descriptor**: SBO=1024, LBO=16, swizzle mode=2 (was SBO=128, LBO=2048, swizzle=0)
- **MMA byte offsets**: `j * 32` (was `j * 4096`)
- **Grid tiling, epilogue, K-loop structure**: identical to Phase 1

The TMA hardware applies a byte-level XOR swizzle during the GMEM→SMEM transfer. The MMA hardware reads with the matching swizzle mode in the descriptor, so the rearrangement is transparent.

### Phase 2: `gemm_sol_pipelined` (Double-Buffered SMEM)

Double-buffered SMEM with per-stage mbarriers for TMA/MMA overlap. Same `block_dim=128` as earlier phases.

Changes from Phase 1.5:
- **SMEM**: 2× buffers (SMEM_A0/A1, SMEM_B0/B1) — ping-pong between them
- **Barriers**: TMA_BAR0/TMA_BAR1 (one per stage) + MMA_BAR (commit signal)
- **K-loop**: Prologue loads k=0 into buffer 0; steady state prefetches k+1 into the other buffer while computing on the current one
- **Parity**: `(k_idx >> 1) & 1` for barrier phase tracking

**Result**: No speedup over Phase 1.5. Single-threaded dispatch on thread 0 serializes MMA and TMA instruction issue, so the overlap window is negligible. The data structures (double-buffered SMEM, per-stage barriers) are correct infrastructure but require warp specialization to deliver actual overlap.

### Phase 3: `gemm_sol_warp_spec` (Warp-Specialized Pipeline)

Warp-specialized producer/consumer architecture. `block_dim=192` (6 warps).

Changes from Phase 2:
- **6 warps**: Warp 4 = TMA producer, Warp 5 = MMA consumer, Warps 0-3 = epilogue
- **5 barriers**: TMA_BAR0/1 (producer→consumer), MMA_BAR0/1 (consumer→producer), COMPUTE_BAR (consumer→epilogue)
- **Pre-signal trick**: `mbarrier_arrive` on MMA_BAR0/1 after init so the producer can start immediately
- **Independent K-loops**: Producer and consumer run their own loops, coordinated only through mbarrier waits — no `sync_threads` in the K-loop
- **Last-iteration commit**: MMA warp commits to COMPUTE_BAR on the final iteration; epilogue warps wait on it
- **GMEM copy**: All 192 threads with stride 192

See `docs/gemm/warp-specialization.md` for the full architecture diagram and barrier protocol trace.

### Phase 4A: `gemm_sol_persistent` (Persistent + TMEM Accumulator Pipeline)

Persistent kernel with 2-stage TMEM accumulator ping-pong. `block_dim=192` (6 warps), `cluster_dim=(4,1,1)`.

Changes from Phase 3:
- **Persistent**: Fixed CTA count (148 = one per SM), each loops over many output tiles via global atomic counter
- **TMEM accumulator pipeline**: 2-stage TMEM — MMA fills stage N while epilogue drains stage N-1, hiding epilogue latency behind tensor core work
- **Barriers**: Adds `ACCUM_FULL0/1` (MMA→Epilogue) and `ACCUM_EMPTY0/1` (Epilogue→MMA) for TMEM pipeline, plus `TILE_READY` (TMA→MMA+Epilogue) for tile coordination

See `docs/gemm/phase4a-persistent.md` for the full walkthrough and bug log.

### Phase 4B: `gemm_sol_clc` (CLC Tile Scheduling, No Multicast)

CLC hardware work-stealing replaces the global atomic counter. Per-CTA unicast TMA for both A and B. `block_dim=192`, `cluster_dim=(4,1,1)`.

Changes from Phase 4A:
- **CLC replaces atomic counter**: Full grid launch (1 CTA per tile), running CTAs steal pending work via `clc_try_cancel` instead of `atomicAdd` on a global counter
- **Column-major rasterization**: Linear ctaid maps to `(row, col)` for cluster-aware tile locality
- **No `tile_counter` parameter**: CLC eliminates the need for a global atomic counter

See `docs/gemm/phase4b-clc.md` for CLC protocol details.

### Phase 4C: `gemm_sol_clc_multicast` (CLC + TMA Multicast for B)

Builds on Phase 4B by multicasting B tiles from rank 0 to all cluster CTAs. `block_dim=192`, `cluster_dim=(4,1,1)`.

Changes from Phase 4B:
- **TMA multicast for B**: Rank 0 broadcasts B to all 4 CTAs via `cp_async_bulk_tensor_2d_g2s_multicast`
- **MCAST_BAR protocol**: Cluster-wide `mbarrier_arrive_cluster` + rank 0 `try_wait_parity` to gate multicast until all CTAs have consumed the previous B buffer
- **`arrive_expect_tx` before TMA loads**: Critical fix — arm barrier before any bytes can arrive from multicast
- **`~{memory}` clobbers**: All barrier/cluster intrinsics emit memory clobbers to prevent LLVM reordering
- **CLC multicast cancel**: `clc_try_cancel_multicast` (rank 0 steals for entire cluster)

See `docs/gemm/phase4c-multicast.md` for the multicast protocol, fixes, and performance analysis.

### Phase 4D: `gemm_sol_clc_multicast_4_stage_pipeline` (cta_group::2 + 4-Stage Pipeline)

Combines `cta_group::2` pair-UMMA with 4-stage SMEM pipelining and CLC work-stealing. `block_dim=192`, `cluster_dim=(2,1,1)`.

Changes from Phase 4C:
- **cta_group::2**: Cluster size 2 (CTA pairs). pair-UMMA (`tcgen05_mma_f16_cg2`) reads both CTAs' SMEM simultaneously — leader-issued only
- **4-stage SMEM**: TMA_BAR0..3 / MMA_BAR0..3 replace the 2-stage pipeline. `stage = gk & 3`, `parity = (gk >> 2) & 1`
- **Barrier aliasing**: TMA barrier address has bit 24 cleared via `0xFEFFFFF8` mask in PTX codegen, redirecting both CTAs' completions to leader's barrier
- **Leader-only expect_tx**: Doubled byte count (49152 = 2 × 24576) since both CTAs' TMA completions converge
- **Leader-only TMA wait**: Follower's MMA warp is idle — pair-UMMA handles both CTAs' data
- **No MCAST_BAR**: Barrier aliasing eliminates the per-K cluster sync overhead that caused Phase 4C's regression
- **Cross-CTA epilogue**: Follower signals leader's ACCUM_EMPTY via `mbarrier_arrive_cluster`
- **cluster_sync() at exit**: Prevents "Cluster target block not present" when fast CTA exits before partner

See `docs/gemm/phase4d-implementation-log.md` for the full architecture, barrier aliasing protocol, and bug chronicle.

## Results (B200)

**cuBLAS baseline**: `cublasLtMatmul` (FP16 input, FP32 compute, TN format, heuristic algorithm).
Per-size baselines: 4K=1502, 8K=1402, 16K=1526 TFLOPS.
The previous 743 TFLOPS baseline was from `cublasGemmEx` (legacy API).
See `bench/cublaslt_bench.c` and `bench/README.md` for details.

### Phase 1 (`gemm_sol_tiled`)

| Size              | TFLOPS | % of cublasLt SoL  |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 182    | 12.1%              |
| 8192×8192×8192    | 187    | 13.3%              |
| 16384×16384×16384 | 191    | 12.5%              |

### Phase 1.5 (`gemm_sol_swizzled`)

| Size              | TFLOPS | % of cublasLt SoL  |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 270    | 18.0%              |
| 8192×8192×8192    | 276    | 19.7%              |
| 16384×16384×16384 | 280    | 18.3%              |

### Phase 2 (`gemm_sol_pipelined`)

| Size              | TFLOPS | % of cublasLt SoL  |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 270    | 18.0%              |
| 8192×8192×8192    | 276    | 19.7%              |
| 16384×16384×16384 | 280    | 18.3%              |

No speedup — pipelining alone does not help without warp specialization (see `docs/gemm/software-pipelining.md`).

### Phase 3 (`gemm_sol_warp_spec`)

| Size              | TFLOPS | % of cublasLt SoL  |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 479    | 31.9%              |
| 8192×8192×8192    | 496    | 35.4%              |
| 16384×16384×16384 | 465    | 30.5%              |

**~1.8x speedup** over Phase 2. Warp specialization delivers real TMA/MMA overlap, nearly doubling throughput. Peak of 496 TFLOPS at 8192³.

### Phase 4 Kernel A (`gemm_sol_persistent`)

| Size              | TFLOPS | % of cublasLt SoL |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 542    | 36.1%              |
| 8192×8192×8192    | 476    | 34.0%              |
| 16384×16384×16384 | 424    | 27.8%              |

Persistent kernel with 2-stage TMEM accumulator pipeline. Gains at 4K (+13% vs Phase 3) but regresses at larger sizes due to atomic counter contention and per-tile overhead. See `docs/gemm/phase4a-persistent.md` for analysis.

### Phase 4B (`gemm_sol_clc`)

| Size              | TFLOPS | % of cublasLt SoL  |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 469    | 31.23%             |
| 8192×8192×8192    | 479    | 34.17%             |
| 16384×16384×16384 | 477    | 31.26%             |

CLC tile scheduling replaces the atomic counter. Eliminates the large-size regression from Phase 4A — consistent ~475 TFLOPS across all sizes. See `docs/gemm/phase4b-clc.md`.

### Phase 4C (`gemm_sol_clc_multicast`)

| Size              | TFLOPS | % of cublasLt SoL  |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 287    | 19.1%              |
| 8192×8192×8192    | 278    | 19.8%              |
| 16384×16384×16384 | 271    | 17.8%              |

CLC + TMA multicast for B tiles. Passes correctness. Performance regression from MCAST_BAR per-K-iteration cluster synchronization overhead with only 2 pipeline stages. Root causes of initial deadlock: missing `~{memory}` clobbers and `arrive_expect_tx` ordering. See `docs/gemm/phase4c-multicast.md`.

### Phase 4D (`gemm_sol_clc_multicast_4_stage_pipeline`)

| Size              | TFLOPS | % of cublasLt SoL  |
|-------------------|--------|--------------------|
| 4096×4096×4096    | 868    | 57.8%              |
| 8192×8192×8192    | 737    | 52.5%              |
| 16384×16384×16384 | 534    | 35.0%              |

CLC + cta_group::2 pair-UMMA + 4-stage SMEM pipeline. Cluster size 2 (CTA pairs). TMA barrier aliasing via bit-24 mask (`0xFEFFFFF8`) converges both CTAs' completions on leader. 2-3x improvement over Phase 4C. See `docs/gemm/phase4d-implementation-log.md`.

## Build and run

```bash
cargo oxide run gemm_sol
```

Requires sm_100+ (Blackwell). On older GPUs, only PTX generation is verified. Runs all eight kernels: correctness tests and benchmarks for each phase.

## Benchmarking

10 warmup + 100 timed iterations with `cudaEventRecord`/`cudaEventElapsedTime`, reporting TFLOPS and % of cublasLtMatmul SoL (FP16 in, FP32 compute, TN format).

Per-size baselines on B200: 4K=1502, 8K=1402, 16K=1526 TFLOPS.

## Roadmap

See `docs/gemm/sol-roadmap.md` for the full phased plan:
- **Phase 1** (`gemm_sol_tiled`): K-loop + grid tiling, tiled TMA → ~190 TFLOPS ✅
- **Phase 1.5** (`gemm_sol_swizzled`): SWIZZLE_128B, single TMA copy → ~280 TFLOPS ✅
- **Phase 2** (`gemm_sol_pipelined`): Double-buffered SMEM (no speedup without warp spec) ✅
- **Phase 3** (`gemm_sol_warp_spec`): Warp-specialized producer/consumer → ~496 TFLOPS ✅
- **Phase 4A** (`gemm_sol_persistent`): Persistent + TMEM accum pipeline → 542 TFLOPS (4K) ✅
- **Phase 4B** (`gemm_sol_clc`): CLC tile scheduling (no multicast) → ~475 TFLOPS (consistent) ✅
- **Phase 4C** (`gemm_sol_clc_multicast`): CLC + TMA multicast for B → ~287 TFLOPS (correct, perf regression) ✅
- **Phase 4D** (`gemm_sol_clc_multicast_4_stage_pipeline`): CLC + cta_group::2 + 4-stage pipeline → 868 TFLOPS (4K, 57.8% SoL) ✅
- **Next**: Performance optimization — tile rasterization, epilogue TMA store, L2 hints
