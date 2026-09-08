# API: CountMin

Status: `Ready`

## Purpose

Approximate frequency estimation with sub-linear memory.

## Type/Struct

- `CountMin<S = Vector2D<i32>, Mode = RegularPath, H = DefaultXxHasher>`

## Constructors

```rust
fn default() -> Self
fn with_dimensions(rows: usize, cols: usize) -> Self
fn from_storage(counts: S) -> Self
```

## Insert/Update

```rust
fn insert(&mut self, value: &DataInput)
fn insert_many(&mut self, value: &DataInput, many: S::Counter)
fn bulk_insert(&mut self, values: &[DataInput])
fn bulk_insert_many(&mut self, values: &[(DataInput, S::Counter)])
fn fast_insert_with_hash_value(&mut self, hashed_val: &H::HashType)
fn fast_insert_many_with_hash_value(&mut self, hashed_val: &H::HashType, many: S::Counter)
```

## Query

```rust
fn estimate(&self, value: &DataInput) -> S::Counter
fn fast_estimate_with_hash(&self, hashed_val: &H::HashType) -> S::Counter
fn rows(&self) -> usize
fn cols(&self) -> usize
fn as_storage(&self) -> &S
fn as_storage_mut(&mut self) -> &mut S
```

## Merge

```rust
fn merge(&mut self, other: &Self)
fn merge_max(&mut self, other: &Self)
```

`merge` adds each counter pair. `merge_max` keeps the larger of each pair,
which is tighter but correct only when the two sketches observed disjoint key
sets — a key both sides counted reads back as the larger side, not the sum.
Both assert matching dimensions.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope (kind `0x02 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). They are **not** available
on every `CountMin`: the impl exists only for wire-eligible configs
`CountMin<Vector2D<T>, Mode, H>` where `T` is `i32`, `i64` or `f64`
(`CmsWireCounter`), `Mode` is `FastPath` or `RegularPath` (`CmsWireMode`), and
`H: HashProfile`. `i32` is carried at its own width and pinned on decode. An
`i128` / other exotic-counter or non-`Vector2D` sketch must be converted to a
wire-eligible storage first (only you know if the mapping is lossless).
`rows`/`cols` are carried in the envelope metadata; the payload is just
`[counts]`. The wire covers `1 <= rows <= 20` (`MATRIX_MAX_ROWS`, the seed
list length): past that, the regular path gives row `r` and row `r + 20` the
same seed and identical counters, so a wider matrix is refused on both sides in
either mode.

## Examples

```rust
use asap_sketchlib::{CountMin, DataInput};

let mut cm = CountMin::with_dimensions(3, 1024);
cm.insert(&DataInput::U64(42));
let est = cm.estimate(&DataInput::U64(42));
assert!(est >= 1);
```

## Caveats

- `merge` requires matching dimensions.
- `FastPath` and regular modes must be paired consistently.

## Status

Used as a primary building block in frameworks and orchestrated paths.
