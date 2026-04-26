# cuda-oxide fuzzer support

`crates/fuzzer` contains the reusable pieces for rustlantis-based differential
codegen testing:

- `src/trace.rs`: the `no_std` trace API used by both CPU and GPU runs.
- `rustlantis/`: vendored upstream rustlantis, used as a MIR program generator.
- `tools/mir_generator.py`: adapts one rustlantis seed into a cuda-oxide smoke case.
- `tools/run_seed.py`: generates a seed, injects it into `rustlantis-smoke`, and runs it.

The execution harness is still the example at
`crates/rustc-codegen-cuda/examples/rustlantis-smoke`. The fuzzer tools rewrite
only `src/generated_case.rs`; `src/main.rs` remains the stable CPU/GPU harness.

## Basic usage

Run one seed:

```bash
python3 crates/fuzzer/tools/run_seed.py --seed 192
```

Run a range:

```bash
python3 crates/fuzzer/tools/run_seed.py --start 0 --count 20 --keep-going
```

The seed controls rustlantis' pseudo-random generator. Same seed plus same
rustlantis config produces the same custom-MIR program, which makes failures
reproducible.

## What gets compared

For each accepted seed:

1. rustlantis generates a Rust/custom-MIR function.
2. `mir_generator.py` rewrites rustlantis' `dump_var(...)` calls into the
   generic `fuzzer::dump_var(...)` trace API.
3. `rustlantis-smoke` runs the same generated case on the CPU and GPU.
4. The CPU and GPU traces are compared as `u64` hashes.

`dump_var` hashes intermediate values, not just the final return value. A seed
can have one dump site or several dump sites. Seed `192` is the current checked
in example because it has two dump sites:

```rust
__rl_dump0 = (Move(_1), Move(_2), Move(_3), Move(_4));
Call(_9 = dump_var(Move(__rl_dump0)), ReturnTo(bb4), UnwindUnreachable())

__rl_dump1 = (Move(_6),);
Call(_9 = dump_var(Move(__rl_dump1)), ReturnTo(bb5), UnwindUnreachable())
```

## Result statuses

- `PASS`: The adapter produced a case, both CPU and GPU runs completed, and the
  trace hashes matched.
- `MISMATCH`: Both CPU and GPU runs completed, but the trace hashes differed.
  This is the highest-priority result because it can indicate a backend
  correctness bug.
- `COMPILE_FAIL [backend]`: The adapter produced a case, but cuda-oxide failed
  while compiling or running it. The log records the backend reason and includes
  the generated `generated_case.rs` snapshot.
- `UNSUPPORTED [adapter]`: rustlantis generated a MIR program, but our Python
  adapter refused to turn it into a cuda-oxide smoke case.

For example, seed `0` currently reports:

```text
UNSUPPORTED [adapter] unsupported dumped type for Stage 2 adapter: u128
```

That means rustlantis successfully generated a program, but a generated
`dump_var(...)` call included a `u128`. Our current trace API only hashes:

```text
bool, i8, i16, i32, i64, u8, u16, u32, u64
```

It does not yet hash `u128`, `i128`, `usize`, `isize`, or `char`. In many
`UNSUPPORTED [adapter]` cases, the MIR can probably be patched by widening the
adapter and trace API. The adapter stops because it does not yet know how to
rewrite/hash that dumped type safely.

## Artifacts

`run_seed.py` writes artifacts under `crates/fuzzer/artifacts/`, which is
ignored by git.

Per-seed logs:

```text
crates/fuzzer/artifacts/seed-<N>-<status>.log
```

Failure logs include:

- seed
- status
- stage (`adapter`, `backend`, or `run`)
- reason
- return code
- command
- full command output
- generated case snapshot, when the adapter produced one

The run summary is also written as:

```text
crates/fuzzer/artifacts/summary.jsonl
```

By default, `summary.jsonl` is replaced at the start of each `run_seed.py`
invocation, so it represents the latest run. Use `--append-summary` to append
new records instead.

The terminal also prints a full per-seed summary, for example:

```text
results:
  seed 0: UNSUPPORTED [adapter] unsupported dumped type for Stage 2 adapter: u128 (...)
  seed 1: COMPILE_FAIL [backend] Unsupported construct: Type translation not yet implemented for: RigidTy(Char) (...)
summary: COMPILE_FAIL=1, UNSUPPORTED=1
```
