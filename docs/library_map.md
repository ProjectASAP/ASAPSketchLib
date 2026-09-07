# Library Map

## Core Modules

- **`src/common/`** - Foundation for all sketches ([api_common.md](./api/api_common.md))
  - `input.rs` - `DataInput` enum, `HeapItem`, `HHItem`, framework enums (`HydraCounter`, `L2HH`, `HydraQuery`)
  - `structures/` - High-performance data structures (`Vector1D`, `Vector2D`, `Vector3D`, `BitMatrix`, `CommonHeap`, `MatrixStorage`, `FixedMatrix`)
  - `heap.rs` - `HHHeap` convenience wrapper for heavy hitter tracking
  - `hash.rs` - Hashing utilities (`hash_for_matrix`, `hash64_seeded`, `SEEDLIST`, `BOTTOM_LAYER_FINDER`) plus `SketchHasher` for custom hasher injection and `DigestHasher` / `DigestBuildHasher` for maps keyed by an already-hashed digest
  - `hashspec.rs` - `HashSpec`/`derive_index`/`derive_sign` byte parity with `sketchlib-go`; `numerical.rs` - `NumericalValue`; `structure_utils.rs` - `Nitro`, `NitroContext`, `ToF64`

- **`src/sketches/`** - Sketch implementations (status source: [apis.md](./apis.md))
  - `Ready` in API index: `countminsketch.rs`, `countsketch.rs`, `hll.rs`, `kll.rs`, `ddsketch.rs`, `countminsketch_topk.rs`, `countsketch_topk.rs`, `space_saving.rs`, `bloom.rs`, `elastic.rs`, `coco.rs`
  - `Unstable` in API index: `uniform.rs`, `kmv.rs`
  - Not in the API index: `fold_cms.rs`, `fold_cs.rs`, `kll_dynamic.rs`, `octo_delta.rs`, `mode.rs` (`RegularPath` / `FastPath`)
  - A sketch on the ASAPv1 wire format uses a `<sketch>.rs` (algorithm) + `<sketch>/wire.rs` (serialization) split: `hll.rs`, `countminsketch.rs`, `countminsketch_topk.rs`, `countsketch.rs`, `countsketch_topk.rs`, `kll.rs`, `kll_dynamic.rs`, `ddsketch.rs`, `elastic.rs`, `coco.rs`, `uniform.rs`, `kmv.rs`, `bloom.rs` and `space_saving.rs` (see [asapv1_wire_format.md](./asapv1_wire_format.md))
  - Two of those directories carry a second, shared wire module: `countminsketch_topk/heap_wire.rs` holds the top-k heap encoding CMSHeap and CSHeap share, and `countsketch_topk/l2hh_wire.rs` holds CountL2HH's own encoding plus the layer sub-payload the UnivMon family inlines

- **`src/sketch_framework/`** - Orchestration and serving layers (status source: [apis.md](./apis.md))
  - `Ready` in API index: `hydra.rs`, `hashlayer.rs`, `univmon.rs`, `univmon_optimized.rs`, `nitro.rs`, `eh.rs`, `eh_sketch_list.rs`, `octo.rs`
  - `Experimental` in API index: `univmon_q.rs`
  - `Unstable` in API index: `eh_univ_optimized.rs`
  - Not in the API index: `tumbling.rs`, `sketch_catalog.rs`
  - The same `<framework>.rs` + `<framework>/wire.rs` split carries the ASAPv1 wire format for `hydra.rs`, `univmon.rs`, `univmon_optimized.rs`, `univmon_q.rs`, `eh.rs` and `eh_sketch_list.rs`

- **`src/message_pack_format/`** - Serialization plumbing ([message_pack_format.md](./message_pack_format.md)). The current format is **ASAPv1**, specified in [asapv1_wire_format.md](./asapv1_wire_format.md)
  - `envelope.rs` — the shared, sketch-agnostic ASAPv1 framing (magic/version/`kind_id` + length prefixes, `encode`/`split`); every `wire.rs` under `src/sketches/` and `src/sketch_framework/` calls into it
  - `codec.rs` — the `MessagePackCodec` trait; `error.rs` — the unified `Error`
  - `portable/` — **deprecated**, being retired. The older per-sketch wire types (`CountMinSketch`, `HllSketch`, …); ASAPv1 (per-sketch `wire.rs`) is now what `sketchlib-go` mirrors, not these
  - `native/` — **deprecated**, being retired. Older `MessagePackCodec` shims over `src/sketches/` byte serialization

## Documentation

- **`docs/`** - API and feature documentation
  - [apis.md](./apis.md) - Canonical API index with one page per API surface
  - [api_common.md](./api/api_common.md) - Common module canonical reference
  - [features.md](./features.md) - Feature status and roadmap

## Utilities

- The large precomputed hash/sample tables are built lazily at runtime via
  `std::sync::LazyLock` in
  `src/common/precompute_hash.rs`, `src/common/precompute_sample.rs`, and
  `src/common/precompute_sample2.rs`.

## Proto code generation

- `proto/**/*.proto` is the cross-language wire-format source of truth shared
  with `sketchlib-go`.
- The corresponding Rust types are **vendored** under
  `src/proto/generated/sketchlib.v1.rs` and re-exported by `src/proto.rs` as
  `crate::proto::sketchlib`. Downstream users therefore build the crate as
  pure Rust without needing `protoc` or any build script.
- To regenerate after editing any `.proto` file, run from the repository root:

  ```bash
  cargo run --manifest-path tools/gen-proto/Cargo.toml
  ```

  CI enforces that the committed file matches the result of regeneration.
