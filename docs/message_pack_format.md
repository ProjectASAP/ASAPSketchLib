# Message Pack Format

`src/message_pack_format/` holds the Rust-side serialization plumbing. The
current, self-describing wire format is **ASAPv1**; its byte-level layout is the
single source of truth in
[**`asapv1_wire_format.md`**](./asapv1_wire_format.md) — read that for the
authoritative spec. This page describes how the code is organized.

## The ASAPv1 model

Every serialized sketch is one self-delimiting envelope:

```md
[ magic:6 | version:u8 | kind_id_len:u8 | kind_id:bytes
          | metadata_len:u32_be | payload_len:u32_be
          | metadata:msgpack-map | payload:msgpack-array ]
```

- **Envelope** — the sketch-agnostic frame (magic `b"ASAPv1"`, version,
  `kind_id`, and the two length prefixes). It answers *is this ours?*, *how do I
  parse the frame?*, and *what algorithm?* with zero knowledge of any sketch.
- **Metadata** — a msgpack **map** (self-describing) carrying the hash spec plus
  the structural params needed to interpret the payload (HLL `precision`;
  Count-Min `rows`/`cols`/`counter_type`/`mode`). The hash-spec values are
  **derived from the hasher's [`HashProfile`](../src/common/hash.rs)** trait, so
  the bytes truthfully describe how the sketch was hashed and custom hash
  profiles serialize self-describingly. Each sketch has its own fixed metadata
  schema with `#[serde(deny_unknown_fields)]` (fail-closed on unknown/missing
  keys). A sketch that never hashes its inputs omits the hash-spec group
  entirely and carries structural params alone — KLL, DDSketch and
  UniformSampling, plus the two wrappers `ExponentialHistogram` and
  `EHSketchList`, whose hash specs live inside the sketch each one carries.
- **Payload** — a positional msgpack **array** of the raw sketch state only
  (registers, counter matrix). No field names, and nothing the `kind_id` or
  metadata already determines.
- **A nested sketch is inlined, not wrapped in an envelope of its own.** A
  CMSHeap's base matrix, an Elastic's light Count-Min, a Hydra cell and a
  UnivMon layer all write their raw state straight into the enclosing positional
  array. Where the nested algorithm is data rather than a type, the nested block
  is that variant's own envelope with the framing stripped: the
  `[kind_id, descriptor, state]` triple `EHSketchList` carries and
  `ExponentialHistogram` inlines per bucket.

`kind_id` + metadata together fix the payload structure completely. See
[`asapv1_wire_format.md`](./asapv1_wire_format.md) for the `kind_id` registry,
the metadata field tables, and the byte-level encoding rules.

## Code organization

### Shared framing — `envelope.rs`

`src/message_pack_format/envelope.rs` is the **one shared, sketch-agnostic**
module every sketch calls into. It owns the magic sentinel, the layout version,
and the byte framing (`encode` / `split`). It validates only the magic, version,
and framing — it does **not** know the `kind_id` registry or any sketch. Rule
"which `kind_id` do I own?" and all metadata validation happen in each sketch's
decoder.

### Per-sketch serialization — `<sketch>/wire.rs`

Serialization lives **with each sketch**, split from the algorithm, under
`src/sketches/` and `src/sketch_framework/` alike:

- `<sketch>.rs` — the **algorithm** (struct, marker types, aliases,
  insert/estimate/merge). It declares `mod wire;`.
- `<sketch>/wire.rs` — the **serialization** (metadata/payload DTOs,
  `kind_id` consts, wire-variant/counter/mode marker traits, and the
  `serialize_to_bytes` / `deserialize_from_bytes` impls). Because `wire` is a
  child submodule of the sketch, it reads the struct's **private** fields
  (`self.registers`, `self.counts`) directly — no field is widened for
  serialization.

Two encodings are shared by more than one sketch and live in a second module
beside `wire.rs`, declared `pub(crate)` so the other user can import it by path:

- `src/sketches/countminsketch_topk/heap_wire.rs` — the top-k heap metadata
  schema, payload, `key_type` mapping, emitted order and heap rebuild that
  CMSHeap (`0x03 0x00`) and CSHeap (`0x0a 0x00`) share.
- `src/sketches/countsketch_topk/l2hh_wire.rs` — CountL2HH (`0x19 0x00`), plus
  the `[counts, l2]` layer sub-payload the whole UnivMon family inlines.

Converted:

- **HLL** (`src/sketches/hll/wire.rs`) — all variants (Classic → `0x01 0x01`,
  Ertl-MLE → `0x01 0x02`, HIP → `0x01 0x03`) × precisions (P12/P14/P16) ×
  `H: HashProfile`. HLL is **fully wire-covered**. (HIP is a non-generic struct
  hashed through the default functions, so it is wire-eligible under the standard
  profile only.)
- **Count-Min** (`src/sketches/countminsketch/wire.rs`, kind `0x02 0x00`) —
  restricted to wire-eligible configs:
  `CountMin<Vector2D<T>, Mode, H>` where `T` is `i32`, `i64` or `f64`
  (`CmsWireCounter`), `Mode` is `FastPath` or `RegularPath` (`CmsWireMode`), and
  `H: HashProfile`. `i32` is carried at its own width rather than widened.
  `i128` and non-`Vector2D` storage must be converted first. `rows`
  and `cols` live in the metadata; the payload is just `[counts]`.
- **Count Sketch** (`src/sketches/countsketch/wire.rs`, kind `0x04 0x00`) — the
  same shape as Count-Min: `Count<Vector2D<T>, Mode, H>` where `T` is `i32` or
  `i64` (`CsWireCounter`), `Mode` is `FastPath` or `RegularPath` (`CsWireMode`),
  and `H: HashProfile`. Counters must be signed and negatable, so there is no
  `f64` counterpart; `i128` and non-`Vector2D` storage must be converted first.
  `i32` is carried at its own width rather than widened, so a nested
  `Vector2D<i32>` sketch decodes back into its own type. `rows`, `cols`,
  `counter_type` and `mode` live in the metadata; the payload is just
  `[counts]`, with signed cells.
- **KLL** (`src/sketches/kll/wire.rs` compact → `0x06 0x00`,
  `src/sketches/kll_dynamic/wire.rs` dynamic → `0x06 0x01`) — both variants share
  one payload `[levels, items, coin]` and differ only by `kind_id`. KLL never
  hashes, so its metadata carries no hash-spec group.
- **Bloom** (`src/sketches/bloom/wire.rs`, kind `0x17 0x00`) — the payload is the
  bit grid packed into `u64`s plus the insert count; the wire covers the
  geometries `Bloom::with_capacity` produces.
- **Space-Saving** (`src/sketches/space_saving/wire.rs`, kind `0x18 0x00`) — the
  payload is the `(key, count, error)` triples plus `total` and `discarded_max`; the
  Stream-Summary's links and arenas are rebuilt on decode.
- **CMSHeap** (`src/sketches/countminsketch_topk/wire.rs`, `0x03 0x00`) and
  **CSHeap** (`src/sketches/countsketch_topk/wire.rs`, `0x0a 0x00`) — a base
  matrix inlined ahead of the shared top-k heap entries.
- **DDSketch** (`src/sketches/ddsketch/wire.rs`, `0x05 0x00`) — the bucket store
  verbatim plus `offset`, `sum`, `min` and `max`; no hash-spec group.
- **Elastic** (`src/sketches/elastic/wire.rs`, `0x0b 0x00`) — the heavy table's
  four parallel arrays, the `stale_copies` flag, and the inlined light Count-Min.
- **Coco** (`src/sketches/coco/wire.rs`, `0x0c 0x00`) — the dense bucket table as
  parallel nullable-key and mass arrays.
- **UniformSampling** (`src/sketches/uniform/wire.rs`, `0x0d 0x00`) — the
  priority-ordered sample pairs plus `total_seen` and the SplitMix64 state; no
  hash-spec group.
- **KMV** (`src/sketches/kmv/wire.rs`, `0x0e 0x00`) — the retained digests,
  strictly ascending.
- **CountL2HH** (`src/sketches/countsketch_topk/l2hh_wire.rs`, `0x19 0x00`) —
  `[counts, l2]`.
- **Hydra** (`src/sketch_framework/hydra/wire.rs`, `0x07 0x00`-`0x07 0x04`) — one
  `kind_id` per counter variant; the cells' state tiles or nests one array.
- **UnivMon** (`src/sketch_framework/univmon/wire.rs`, `0x10 0x00`) and
  **UnivMon Optimized** (`src/sketch_framework/univmon_optimized/wire.rs`,
  `0x11 0x00`) — one payload shape, two metadata schemas.
- **UnivMon-Q** (`src/sketch_framework/univmon_q/wire.rs`, `0x1a 0x00`) — the
  levels' counters, candidates, extrema and coordinated occurrence sample.
- **ExponentialHistogram** (`src/sketch_framework/eh/wire.rs`, `0x13 0x00`) and
  **EHSketchList** (`src/sketch_framework/eh_sketch_list/wire.rs`, `0x14 0x00`) —
  one `[kind_id, descriptor, state]` triple, standalone and per bucket.

NitroBatch, FoldCMS, FoldCS, HashSketchEnsemble, EHUnivOptimized and OctoSketch
have **no `wire.rs`**; their `kind_id`s are reserved with payload TBD.

## `portable/` and `native/` are being retired

The `portable/` and `native/` sub-modules (and the `MessagePackCodec` trait /
unified `Error` at the module root) are the **older** serialization path and are
being **phased out** in favor of the per-sketch `wire.rs` + the shared
`envelope.rs`.

- `portable/` holds a set of per-sketch wire types that predate the envelope.
  **`sketchlib-go` mirrors ASAPv1, not these**: the per-sketch payload under
  `wire.rs` is the cross-language format, and the `portable` types are internal
  to Rust.
- `native/` is a set of thin `MessagePackCodec` shims over the sketches'
  `serialize_to_bytes` / `deserialize_from_bytes`. Every type it wraps emits the
  ASAPv1 envelope, so each shim is a pass-through.

Sequencing note (from the spec): `portable` is not deleted until the golden
byte-vector fixtures are the drift guard on both sides.

## Cross-language parity

Cross-language parity with `sketchlib-go` is proven by **golden byte-vectors** in
[`asapv1_golden/`](../asapv1_golden) (exercised by
[`tests/asapv1_golden.rs`](../tests/asapv1_golden.rs)): both languages must
decode → re-encode them byte-identically. The `kind_id` registry is mirrored
verbatim with Go's `wire/asapmsgpack/magic_ids.go`, never independently
allocated.

That proof covers the six `kind_id`s a fixture exists for — HLL's three
estimators, Count-Min, Count Sketch and compact KLL. Every other implemented
kind has **no golden and therefore no cross-language drift guard**;
[`asapv1_golden/README.md`](../asapv1_golden/README.md) lists what is covered.
`portable` is not deleted until the goldens are the drift guard on both sides,
so it stays until that gap closes.

## Cross-Reference

- [`asapv1_wire_format.md`](./asapv1_wire_format.md) — the authoritative
  byte-level spec (envelope, metadata schema, per-sketch payloads, encoding
  rules, wire coverage).
- Generated rustdoc: `cargo doc --no-deps --all-features --open`.
