# Pliron -- Pliron IR (MLIR-like)

cuda-oxide does not compile Rust directly to PTX in one step. It lowers code
through a series of intermediate representations, each capturing a different
level of abstraction. The framework that makes this possible is **pliron** -- an
extensible compiler IR framework written in pure Rust, inspired by LLVM's MLIR.
We refer to the IR built on this framework as **Pliron IR**.

This chapter explains what upstream MLIR is, why pliron exists as an alternative,
and how its core data structures work. If you have never built a compiler, don't
worry -- nothing here requires prior compiler experience, just a working
knowledge of Rust.

## What is MLIR (and why should you care)?

LLVM IR is a fixed instruction set. It has roughly 70 opcodes (`add`, `load`,
`br`, `getelementptr`, and so on), and if your domain doesn't map cleanly to
those opcodes, tough luck -- you flatten your high-level concepts into low-level
instructions and hope the optimizer can reconstruct what you meant.

**MLIR** (Multi-Level Intermediate Representation) takes a different approach.
Instead of one fixed instruction set, MLIR gives you a framework for defining
*many* instruction sets -- called **dialects** -- each tailored to a specific
domain. A dialect is a collection of operations, types, and attributes that model
the concepts you actually care about.

The key idea is straightforward:

1. Define a dialect with operations that match your domain.
2. Write **passes** that transform one dialect into another (lowering).
3. Chain passes together until you reach something a backend can consume.

Consider how Triton (the GPU compiler behind PyTorch) uses MLIR. Python GPU code
first lowers to **TTIR**, a tensor-level IR where operations like "broadcast this
scalar across a tensor" are first-class. At that level, Triton can apply
domain-specific optimizations -- for instance, replacing a `splat + mul`
(broadcast a scalar, then multiply element-wise) with a single native
vector-scalar multiply. That optimization is trivial to express when you have
tensor operations in your IR and nearly impossible to recover after flattening
to LLVM IR.

cuda-oxide faces a similar challenge. We need to represent three very different
levels of abstraction:

| Level              | What it models                                                | Example operations                           |
| :----------------- | :------------------------------------------------------------ | :------------------------------------------- |
| **`dialect-mir`**  | Rust semantics -- tuples, enums, slices, checked arithmetic   | `mir.extract_field`, `mir.get_discriminant`  |
| **`dialect-llvm`** | Machine-near operations -- integer math, memory, control flow | `llvm.add`, `llvm.load`, `llvm.br`           |
| **`dialect-nvvm`** | GPU intrinsics -- thread indexing, warp shuffles, TMA, WGMMA  | `nvvm.read_ptx_sreg_tid_x`, `nvvm.shfl_sync` |

Without an extensible IR, we would have to either jam Rust enums into LLVM IR
(losing semantic information) or build three separate IR frameworks (losing our
sanity). MLIR lets us define all three as dialects in a single system and lower
between them with well-typed passes.

## Enter pliron

MLIR is a great idea. MLIR's implementation, however, is C++ with a side of
TableGen, a build system that requires you to compile all of LLVM, and debugging
sessions that make you question your career choices.

[Pliron](https://github.com/vaivaswatha/pliron) is an MLIR-inspired extensible
compiler IR framework written in **pure Rust**. It follows the same conceptual
model -- operations, regions, basic blocks, types, attributes, dialects, passes
-- but trades C++ and TableGen for `cargo build` and standard Rust tooling.

| Aspect            | pliron                                   | Upstream MLIR                      |
| :---------------- | :--------------------------------------- | :--------------------------------- |
| Language          | Pure Rust                                | C++ with Python bindings           |
| Build system      | `cargo build`                            | CMake + full LLVM build            |
| DSLs required     | None -- just Rust macros                 | TableGen, ODS                      |
| Debugging         | Standard Rust tooling (`dbg!`, rust-gdb) | gdb on C++ templates               |
| Extensibility     | Add a dialect as a Rust crate            | C++ extension against LLVM headers |
| Dependency weight | One crate (git dependency)               | Gigabytes of LLVM build artifacts  |

For cuda-oxide, this means the entire compiler -- from MIR import to LLVM IR
export -- is a single `cargo build` invocation. No CMake. No C++ linker errors.
No staring at TableGen error messages and wondering what you did to deserve this.

## Core data structures

Pliron's IR is built from a handful of core types. Understanding these is the
key to reading (and writing) dialect code.

### Context

The `Context` is the central data structure that owns *all* IR data --
operations, basic blocks, regions, types, attributes, and dialect registrations.
Think of it as the arena allocator for your entire compilation unit.

```rust
// Simplified -- the real Context has more fields
pub struct Context {
    operations: SlotMap<Ptr<Operation>, Operation>,
    basic_blocks: SlotMap<Ptr<BasicBlock>, BasicBlock>,
    regions: SlotMap<Ptr<Region>, Region>,
    dialects: HashMap<DialectName, Dialect>,
    type_store: TypeStore,     // uniqued (deduplicated) types
    attr_store: AttrStore,     // uniqued attributes
}
```

Pliron uses **generational arenas** (from the `slotmap` crate) instead of
`Box`-allocated heap nodes. This gives you:

- **O(1) insert and remove** -- no tree rebalancing, no linked-list traversal.
- **Stable indices** -- inserting or removing an element does not invalidate
  other indices.
- **Generational versioning** -- each slot carries a generation counter. If you
  delete an operation and the slot gets reused, any old reference will have a
  stale generation and fail at runtime instead of silently reading garbage.

Every piece of IR is stored inside the Context. You never hold a direct `&mut
Operation` across function boundaries -- you hold a `Ptr<Operation>` and deref
it through the Context when needed.

### Operations

An **operation** is a single node in the IR graph. It has operands (inputs),
results (outputs), attributes (compile-time metadata), and optionally regions
(nested structure for things like function bodies and loop nests).

Every operation belongs to a dialect. The naming convention is
`dialect_name.op_name`:

```rust
#[pliron_op(name = "mir.func", dialect = "mir")]
pub struct MirFuncOp;

#[pliron_op(name = "nvvm.read_ptx_sreg_tid_x", dialect = "nvvm")]
pub struct ReadPtxSregTidXOp;

#[pliron_op(name = "llvm.add", dialect = "llvm")]
pub struct AddOp;
```

The `#[pliron_op(...)]` proc macro (from `pliron-derive`) generates the
boilerplate for registering the operation with its dialect, assigning it an
opcode, and wiring up the `Op` trait. You define the operation's semantics by
implementing `Verify`, `Printable`, and `Parsable`.

### Types and attributes

**Types** represent data types in the IR. Pliron's built-in dialect provides
integers (`i1`, `i32`, `i64`) and floats (`f32`, `f64`). cuda-oxide's MIR
dialect extends these with Rust-specific types:

```rust
#[pliron_type(name = "mir.tuple", dialect = "mir")]
pub struct MirTupleType {
    pub types: Vec<Ptr<TypeObj>>,
}

#[pliron_type(name = "mir.slice", dialect = "mir")]
pub struct MirSliceType {
    pub element_ty: Ptr<TypeObj>,
}

#[pliron_type(name = "mir.enum", dialect = "mir")]
pub struct MirEnumType {
    pub name: String,
    pub discriminant_ty: Ptr<TypeObj>,
    pub variant_names: Vec<String>,
    // ...
}
```

**Attributes** attach compile-time metadata to operations -- constant values,
flags, predicate kinds, cast kinds, and so on.

Both types and attributes are **uniqued** (deduplicated) in the Context. If you
create two `MirTupleType { types: vec![i32, f32] }` instances, pliron stores
only one copy and hands back the same pointer. This makes type equality a pointer
comparison instead of a deep structural compare.

### Ptr\<T\> -- safe arena references

`Ptr<T>` is pliron's equivalent of a pointer into the arena. Under the hood, it
is composed of two fields:

```rust
pub struct Ptr<T> {
    index: u32,           // slot index in the arena
    version: NonZeroU32,  // generational version
    _phantom: PhantomData<T>,
}
```

The **version** field is what makes this safe. When you delete an operation,
pliron bumps the generation counter on that slot. If someone later tries to
dereference an old `Ptr<Operation>` whose version no longer matches the slot's
current generation, the access fails instead of reading the new (unrelated)
occupant.

The `PhantomData<T>` ensures type safety at compile time -- you cannot
accidentally use a `Ptr<Operation>` to index into the `BasicBlock` arena. The
compiler won't let you.

```{note}
You interact with arena contents through the Context:

- `ptr.deref(ctx)` returns a shared reference (`&T`).
- `ptr.deref_mut(ctx)` returns an exclusive reference (`&mut T`).

This follows Rust's borrow model -- the Context is the owner, and you borrow
through it.
```

## Def-use chains (memory-safe)

In any compiler IR, you need to answer two questions constantly:

- **Use-def**: "Where is this value defined?" (Given a use, find the definition.)
- **Def-use**: "Where is this value used?" (Given a definition, find all uses.)

Traditional compilers implement these chains with raw pointers and manual
bookkeeping. Forget to update a use-list when you delete an operation? Dangling
pointer. Replace a value but miss one use? Stale reference. Welcome to your
afternoon of debugging a segfault in `opt -O2`.

Pliron implements def-use chains using Rust's type system. A `Value` in pliron
is an enum:

```rust
pub enum Value {
    OpResult { op: Ptr<Operation>, index: usize },
    BlockArgument { block: Ptr<BasicBlock>, index: usize },
}
```

Every value is either the result of an operation or an argument to a basic block.
Each definition tracks the set of all its uses, and each use stores a pointer
back to its definition. When you call `replace_all_uses_with`, both sides update
automatically.

This means:

- **No dangling references** -- removing an operation updates all use-lists.
- **No stale uses** -- replacing a value propagates to every consumer.
- **No segfaults during IR transforms** -- the borrow checker and generational
  arenas prevent the entire class of bugs that plague C++ compiler frameworks.

```{note}
If you are familiar with LLVM's `Value` / `Use` / `User` system, pliron's
design serves the same purpose. The difference is that LLVM implements it with
intrusive linked lists and raw `Value*` pointers, while pliron implements it
with arena indices and `HashSet<UseNode>`. Same semantics, fewer late-night
debugging sessions.
```

## Dynamic type casting

Pliron stores operations as `dyn Any` internally. This is necessary because each
dialect defines its own operation types -- the framework cannot know at compile
time what types will exist.

Rust's `dyn Any` supports downcasting to concrete types via `downcast_ref::<T>()`.
But there is a gap: you cannot downcast `dyn Any` to a *trait object*
(`dyn SomeTrait`). Rust's type system simply does not support that -- there is
no way to go from `TypeId` to a vtable pointer without compiler support.

Pliron solves this with a registration-based casting system:

```rust
// At dialect registration time, register the cast:
type_to_trait!(MirFuncOp, dyn Verify);
type_to_trait!(MirFuncOp, dyn SymbolOpInterface);

// At runtime, look up the cast:
let verifiable: &dyn Verify = any_to_trait::<dyn Verify>(&*op)?;
verifiable.verify(ctx)?;
```

Behind the scenes, a global map (`TRAIT_CASTERS_MAP`) stores pairs of
`(TypeId, TraitId) -> fn(&dyn Any) -> &dyn Trait`. The `type_to_trait!` macro
inserts an entry; `any_to_trait()` performs the lookup.

This powers pliron's verification and interface system. When you run
`verify_op(ctx, op_ptr)`, pliron:

1. Reads the operation's concrete `TypeId`.
2. Looks up whether it implements `dyn Verify`.
3. If so, calls `verify()` through the retrieved trait object.

The alternative would be making the entire framework generic over all possible
operation types, which would either require an enum with hundreds of variants or
propagate type parameters through every function signature in the codebase. The
dynamic casting approach keeps the framework extensible without generics
pollution -- adding a new dialect is just adding a new crate, not modifying a
central enum.

```{note}
This is a runtime cost (one `HashMap` lookup per cast), but it only happens
during verification and pass dispatch -- not in the hot path of IR construction.
The ergonomic benefit of fully decoupled dialects is well worth one hash lookup
per operation.
```

## How cuda-oxide uses pliron

cuda-oxide defines three dialects, each as its own crate:

```rust
pub fn register_dialects(ctx: &mut Context) {
    dialect_mir::register(ctx);
    dialect_nvvm::register(ctx);
    pliron::builtin::register(ctx);
}

// Later, when lowering to LLVM:
dialect_llvm::register(ctx);
```

### dialect-mir -- Rust semantics

`dialect-mir` captures Rust's mid-level IR as pliron operations, preserving
semantic information that would be lost if we lowered directly to LLVM.

- **Function definition**: `MirFuncOp` -- entry point for each device function,
  with typed block arguments matching the Rust function signature.
- **Arithmetic and comparison**: checked and unchecked binary ops (`mir.add`,
  `mir.sub`, `mir.eq`, `mir.lt`, ...) that preserve Rust's overflow semantics.
- **Aggregate types**: `MirTupleType`, `MirStructType`, `MirEnumType`,
  `MirSliceType`, `MirArrayType` -- first-class Rust compound types with
  operations like `mir.extract_field` and `mir.get_discriminant`.
- **Memory and control flow**: `mir.load`, `mir.store`, `mir.ref`,
  `mir.goto`, `mir.cond_br`, `mir.return` -- with GPU address-space
  tracking (`global`, `shared`, `local`, `tmem`).

### dialect-llvm -- machine-near IR

`dialect-llvm` models LLVM IR as pliron operations, providing a 1:1 mapping
to textual `.ll` files.

- **Arithmetic and casts**: all 19 LLVM binary ops (`llvm.add` through
  `llvm.frem`), plus 13 cast ops (`llvm.sext`, `llvm.trunc`, `llvm.bitcast`, ...).
- **Control flow**: `llvm.br`, `llvm.cond_br`, `llvm.switch`, `llvm.return`,
  `llvm.unreachable` -- with block arguments translated to PHI nodes on export.
- **Textual export**: the `dialect_llvm::export` module emits valid LLVM IR text,
  including `@llvm.used` arrays and `!nvvm.annotations` metadata for GPU kernels.

### dialect-nvvm -- GPU intrinsics

`dialect-nvvm` wraps LLVM's NVPTX backend intrinsics as typed pliron
operations.

- **Thread indexing**: `nvvm.read_ptx_sreg_tid_x`, `nvvm.read_ptx_sreg_ctaid_x`,
  `nvvm.barrier0` -- the building blocks of `thread::index_1d()`.
- **Warp-level primitives**: shuffle (`nvvm.shfl_sync`), vote, and reduce
  operations for warp-cooperative algorithms.
- **Accelerator ops**: TMA bulk copies, WGMMA matrix multiply-accumulate
  (Hopper), and tcgen05 tensor core operations (Blackwell) -- the hardware
  instructions behind the [Advanced GPU Features](../advanced/tensor-memory-accelerator.md)
  chapters.

Each dialect is covered in detail in [Pliron Dialects](mlir-dialects.md).
