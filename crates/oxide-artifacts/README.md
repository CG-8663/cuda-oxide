# oxide-artifacts

`oxide-artifacts` defines the embedded device-artifact container used by
`cuda-oxide`.

The crate is deliberately accelerator-neutral. It knows how to describe bundles
of generated PTX payloads, but it does not know how to compile or launch them.
Runtime crates can parse bundles and decide whether they can consume them.

## Design

An artifact bundle is a small binary blob stored in a retained host-object
section named `.oxart`.

Each bundle contains:

- a bundle name, normally the producing Rust crate name
- a device target string, such as `sm_90`
- zero or more entry records, such as CUDA kernels
- one or more payload records, such as generated PTX bytes

Multiple bundle blobs may be concatenated in the same section. Parsers walk the
section by reading each blob's `total_len` field.

The section-object writer is behind the `object-write` feature and uses the
Rust `object` crate rather than hand-writing ELF. That keeps the common
infrastructure portable across the CUDA host platforms this crate supports
today: Linux on AMD64 and ARM64.

## Wire Format

All integer fields are little-endian. Offsets are relative to the start of the
bundle blob, not the host object file or section.

```text
Artifact Blob

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Magic "OXIDEART"                      |
 |                                                               |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |          Version              |         Header Length         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Total Length                          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |        Name Length            |        Target Length          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |       Payload Count           |        Entry Count            |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Reserved (zero)                       |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Bundle Name ...                      /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Target String ...                    /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Payload Records ...                  /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Entry Records ...                    /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Payload Names/Data and Entry Symbols  /
 |                         (each variable item is 8-byte aligned)/
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Header size is currently 32 bytes.

```text
Payload Record (24 bytes)

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |       Payload Kind            |          Flags (zero)         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Data Offset                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Data Length                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Name Offset                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |        Name Length            |          Reserved             |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Reserved                              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Payload kind values:

- `0x0100`: PTX

```text
Entry Record (24 bytes)

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |         Entry Kind            |             Flags             |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Metadata                              |
 |                                                               |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Symbol Offset                         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |       Symbol Length           |          Reserved             |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Reserved                              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Entry kind values:

- `1`: kernel
- `2`: device function

If bit 0 of `Flags` is set, `Metadata` is present and interpreted by the
consumer for that entry kind. No metadata interpretation is currently required
for PTX kernel loading.

## Object Storage

When `object-write` is enabled, `build_host_object_for_target` creates a
relocatable object with a single `.oxart` data section.

The writer marks the ELF section as retained with
`SHF_ALLOC | SHF_GNU_RETAIN`.

The rustc CUDA backend writes generated PTX into one bundle blob, emits a host
object for the current host target, and appends that object to rustc's compiled
module list before linking. At runtime, `cuda-core` can read the executable's
object sections and load the PTX payload with `cuModuleLoadData`.

## Constraints

- The format version is currently `1`.
- All numeric fields are little-endian.
- The blob header is fixed at 32 bytes.
- Payload and entry records are fixed at 24 bytes each.
- String and payload offsets are 32-bit, so a single blob must fit in `u32::MAX`
  bytes.
- String lengths and record counts are 16-bit.
- Bundle names, target strings, payload names, and entry symbols must be UTF-8.
- Payloads must be non-empty.
- Unknown payload or entry kind values are rejected by the current parser.
- Compression is not part of the current wire format.

## Feature Flags

- `object-read`: parse artifact bundles out of host object/executable bytes.
- `object-write`: emit host relocatable objects with an `.oxart`
  section.
- `object`: enables both read and write support.

## TODO

- Investigate whether compression is useful or necessary for embedded payloads,
  especially for large PTX bundles, and whether it belongs in this crate
  or in a higher-level packaging layer.
- Consider Windows support later.
