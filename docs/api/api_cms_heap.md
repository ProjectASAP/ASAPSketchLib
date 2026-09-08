# API: CMSHeap

Status: `Ready`

## Purpose

Count-Min with integrated heavy-hitter heap (`HHHeap`) for top-k tracking.

## Type/Struct

- `CMSHeap<S = Vector2D<i64>, M = RegularPath, H = DefaultXxHasher>`

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
fn estimate(&self, key: &DataInput) -> S::Counter
fn rows(&self) -> usize
fn cols(&self) -> usize
fn cms(&self) -> &CountMin<S, M, H>
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

These produce/consume the **ASAPv1** wire envelope (kind `0x03 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). They are **not** available
on every `CMSHeap`: the impl exists only for wire-eligible configs
`CMSHeap<Vector2D<T>, Mode, H>` where `T` is `i32`, `i64` or `f64`
(`CmsWireCounter`), `Mode` is `FastPath` or `RegularPath` (`CmsWireMode`), and
`H: HashProfile`. An `i128` or non-`Vector2D` sketch must be converted first
(only you know if the mapping is lossless).

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

## Examples

```rust
use asap_sketchlib::{CMSHeap, DataInput, Vector2D, RegularPath};

let mut sk = CMSHeap::<Vector2D<i64>, RegularPath>::new(3, 256, 8);
sk.insert(&DataInput::Str("flow"));
assert!(sk.estimate(&DataInput::Str("flow")) >= 1);
```

## Caveats

- Heap can be cleared independently of sketch counters.
- Merge requires matching dimensions and compatible type parameters.
- **Eight built-in instances are constructible but inert.** `insert`,
  `insert_many`, `estimate` and `merge` live in an impl bounded on
  `S::Counter: Copy + Ord + From<i32> + Into<i64> + AddAssign`. `i128` has no
  `Into<i64>` and `f64` has neither `Ord` nor `Into<i64>`, so
  `CMSHeap<Vector2D<i128>>`, `CMSHeap<Vector2D<f64>>`,
  `CMSHeap<QuickMatrixI128>` and `CMSHeap<DefaultMatrixI128>` — on both hashing
  paths — allocate a sketch and a heap and can then do nothing but report their
  dimensions. Use `CSHeap` when you need a 128-bit counter beside a top-k heap;
  it supports every backend.

## Status

Useful helper wrapper; tested but less central than base sketches.
