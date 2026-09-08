# API: CSHeap

Status: `Ready`

## Purpose

Count Sketch with integrated heavy-hitter heap (`HHHeap`) for top-k tracking.

## Type/Struct

- `CSHeap<S = Vector2D<i64>, M = RegularPath, H = DefaultXxHasher>`

## Constructors

```rust
fn new(rows: usize, cols: usize, top_k: usize) -> Self
fn from_storage(storage: S, top_k: usize) -> Self
fn default() -> Self
```

## Insert/Update

```rust
fn insert(&mut self, key: &DataInput)
fn insert_many(&mut self, key: &DataInput, many: S::Counter)
fn bulk_insert(&mut self, values: &[DataInput])
fn clear_heap(&mut self)
```

## Query

```rust
fn estimate(&self, key: &DataInput) -> f64
fn rows(&self) -> usize
fn cols(&self) -> usize
fn cs(&self) -> &Count<S, M, H>
fn heap(&self) -> &HHHeap
```

## Merge

```rust
fn merge(&mut self, other: &Self)
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope (kind `0x0a 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). They are **not** available
on every `CSHeap`: the impl exists only for wire-eligible configs
`CSHeap<Vector2D<T>, Mode, H>` where `T` is `i32` or `i64` (`CsWireCounter`),
`Mode` is `FastPath` or `RegularPath` (`CsWireMode`), and `H: HashProfile`. Count Sketch
counters must be signed and negatable, so there is no `f64` counterpart to
Count-Min's; an `i128` or non-`Vector2D` sketch must be converted first (only
you know if the mapping is lossless).

A sketch travels as the base matrix plus the heap's entries: the metadata
carries `rows`, `cols`, `counter_type`, `mode`, the heap capacity `k` and the
heap's `key_type`, and the payload is `[counts, keys, heap_counts]`. The heap's
digest index is rebuilt on load, so no index reaches the wire, and `k` never
sizes an allocation on decode. The base matrix is bound by the same
`1 <= rows <= 20` (`MATRIX_MAX_ROWS`, the seed list length) the stand-alone
sketch is: past that, the regular path gives row `r` and row `r + 20` the same
seed and identical counters, so a wider matrix is refused on both sides in
either mode.

Heap keys are `HeapItem`s, so the key type is a runtime property: `key_type`
names the **exact** variant (`"i32"` stays `"i32"`, never widened to `"i64"`)
and the `keys` array is homogeneous in it. A heap whose keys mix variants, or
holds an `I128` / `U128` key, does not serialize. An empty heap emits
`key_type = "u64"`.

Entries are emitted in descending count, ties broken by a total order over the
key, so a decoded sketch re-serializes byte-identically.

`CountL2HH` is a different algorithm with its own ASAPv1 kind (`0x19 0x00`);
see [the Count Sketch API doc](./api_count_sketch.md).

## Examples

```rust
use asap_sketchlib::{CSHeap, DataInput, Vector2D, RegularPath};

let mut sk = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 256, 8);
sk.insert(&DataInput::Str("flow"));
assert!(sk.estimate(&DataInput::Str("flow")) >= 1.0);
```

## Caveats

- Estimate semantics follow Count Sketch and may be non-integer.
- Merge requires matching dimensions and compatible type parameters.
- **Heap counts are a saturating, truncating view of the estimate.** The insert
  path stores `cs_heap_count(estimate)`, where the estimate is the row median as
  `f64` and the heap holds `i64`. The conversion saturates at `i64::MAX` /
  `i64::MIN` (and maps `NaN` to `0`) rather than wrapping, because a wrapped
  negative count would corrupt the heap's ordering. An `i128`-backed sketch can
  genuinely hold counts past `i64::MAX` — `insert_many(key, i128)` accepts them,
  and the sketch keeps them — but its heap entry clamps. Above `2^53` the
  estimate has already lost precision inside `Count::estimate`.
- Every storage backend is insertable, `i128` included: the bound is
  `S::Counter: CountSketchCounter`, which `i32`, `i64` and `i128` all satisfy.
  This is *not* true of `CMSHeap`; see its page.

## Status

Useful helper wrapper; tested but less central than base sketches.
