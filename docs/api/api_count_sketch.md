# API: Count

Status: `Ready`

## Purpose

`Count` is this crate's Count Sketch implementation for approximate frequency estimation with signed counters and median aggregation.

## Type/Struct

- `Count<S = Vector2D<i32>, Mode = RegularPath, H = DefaultXxHasher>`
- `CountL2HH<H = DefaultXxHasher>`

## Constructors

```rust
fn default() -> Self
fn with_dimensions(rows: usize, cols: usize) -> Self
fn from_storage(counts: S) -> Self

// CountL2HH
fn default() -> Self
fn with_dimensions(rows: usize, cols: usize) -> Self
fn with_dimensions_and_seed(rows: usize, cols: usize, seed_idx: usize) -> Self
```

## Insert/Update

```rust
fn insert(&mut self, value: &DataInput)
fn insert_many(&mut self, value: &DataInput, many: S::Counter)
fn fast_insert_with_hash_value(&mut self, hashed_val: &H::HashType)

// CountL2HH
fn fast_insert_with_count(&mut self, val: &DataInput, c: i64)
fn fast_insert_with_count_and_hash(&mut self, hashed_val: u128, c: i64)
fn fast_insert_with_count_without_l2_and_hash(&mut self, hashed_val: u128, c: i64)
```

## Query

```rust
fn estimate(&self, value: &DataInput) -> f64
fn fast_estimate_with_hash(&self, hashed_val: &H::HashType) -> f64

// CountL2HH
fn fast_get_est(&self, val: &DataInput) -> f64
fn fast_get_est_with_hash(&self, hashed_val: u128) -> f64
fn fast_update_and_est(&mut self, val: &DataInput, c: i64) -> f64
fn fast_update_and_est_without_l2(&mut self, val: &DataInput, c: i64) -> f64
fn get_l2(&self) -> f64
fn get_l2_sqr(&self) -> f64
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

These produce/consume the **ASAPv1** wire envelope (kind `0x04 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). They are **not** available
on every `Count`: the impl exists only for wire-eligible configs
`Count<Vector2D<T>, Mode, H>` where `T` is `i32` or `i64` (`CsWireCounter`),
`Mode` is `FastPath` or `RegularPath` (`CsWireMode`), and `H: HashProfile`.
Count Sketch counters must be signed and negatable, so there is no `f64`
counterpart to Count-Min's; an `i128` or non-`Vector2D` sketch must be converted
first (only you know if the mapping is lossless).

`i32` is **not** widened to `i64` on the wire. The counter type is carried in
the metadata and pinned on decode, so `i32` bytes do not decode into an `i64`
sketch, or the reverse — which is what lets a nested `Vector2D<i32>` Count
Sketch (the variant `HydraCounter` and `EHSketchList` hold) round-trip back into
its own type. `rows`/`cols` and the `mode` are carried in the metadata too; the
payload is just `[counts]`, packed row-major with signed cells. The wire covers `1 <= rows <= 20` (`MATRIX_MAX_ROWS`, the seed
list length): past that, the regular path gives row `r` and row `r + 20` the
same seed and identical counters, so a wider matrix is refused on both sides in
either mode.

`CountL2HH` has its own ASAPv1 kind (`0x19 0x00`) and the same two methods,
available for any `H: HashProfile`. Its counters are always `i64` and its
column derivation is fixed by the algorithm, so the metadata carries no
`counter_type` and no `mode` — only `seed_index`, `rows` and `cols`. Its rows
are seeded per row too, so it carries the same `1 <= rows <= 20` bound. The
payload is `[counts, l2]`: the matrix packed row-major, then one L2
accumulator per row. `l2` is carried rather than recomputed, because
`fast_insert_with_count_without_l2_and_hash` moves counters without it and the
accumulator saturates one way. The same `(counts, l2)` pair is what a `UnivMon`
layer inlines into its own payload.

## Examples

```rust
use asap_sketchlib::{Count, DataInput};

let mut cs = Count::with_dimensions(5, 2048);
cs.insert(&DataInput::Str("alpha"));
let est = cs.estimate(&DataInput::Str("alpha"));
assert!(est >= 1.0);
```

## Caveats

- `merge` requires matching dimensions.
- `CountL2HH` is the L2-heavy-hitter variant used by `UnivMon` internals.

## Status

Core frequency primitive; this crate exposes it as `Count` and uses it widely in framework layers.
