# API: CountMinHll

Status: `Experimental`

> Warning: Behind the `experimental` cargo feature. The API and the wire form
> may change without a major version bump. For a per-key distinct count on a
> stable API today, use [`Hydra`](./api_hydra.md) with `HydraCounter::HLL`.

## Purpose

Grouped distinct counting: given a stream of `(key, distinct_value)` pairs,
answers "how many distinct `distinct_value`s have been seen for this `key`?"

A Count-Min-shaped `rows x cols` grid whose every cell is a small
HyperLogLog, stored as one contiguous `Vector3D<u8>` of shape
`rows x cols x 2^precision`. There is no Count-Sketch sign; the estimator is
the **minimum** across rows, because a cell accumulates the distinct values of
every key that lands in it and so can only over-report.

## Type/Struct

- `CountMinHll<H = DefaultXxHasher>`

## Constructors

```rust
fn default() -> Self                                            // 4 x 64, p = 8
fn with_dimensions(rows: usize, cols: usize, precision: u32) -> Self
```

`cols` must be a power of two and at least 2, `precision` must be in
`1..=18`, and `rows * log2(cols)` must not exceed 128 — the packed column
hash is one `u128`. All three are asserted at construction, and every one of
them is re-checked on deserialization against a forged payload in the tests.

A single column is refused because it would route every row to the same
bucket, collapsing the minimum across rows to one HyperLogLog while the
`rows * log2(cols)` guard, whose mask width is then zero, silently admitted
any row count.

## Insert/Update

```rust
fn insert(&mut self, key: &DataInput, distinct_value: &DataInput)
fn insert_many(&mut self, pairs: &[(&DataInput, &DataInput)])
```

Two hash calls per insert regardless of row count: one `hash128_seeded(0, ..)`
over the key, which `Vector3D` slices into one column index per row, and one
`hash64_seeded` over the value for the register index and rank. The value
hash uses a fixed non-zero seed index — not one derived from `rows`, since
`SEEDLIST` indices wrap and a row count equal to its length would have
selected the column hash's own seed.

## Query

```rust
fn estimate(&self, key: &DataInput) -> f64
```

## Accessors

```rust
fn rows(&self) -> usize
fn cols(&self) -> usize
fn precision(&self) -> u32
fn registers_per_bucket(&self) -> usize      // 2^precision
fn as_storage(&self) -> &Vector3D<u8>
```

## Merge

```rust
fn merge(&mut self, other: &Self) -> Result<(), String>
```

Element-wise register maximum. Requires equal `(rows, cols, precision)` and
returns an error otherwise, leaving `self` untouched.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

Plain MessagePack, **not** an ASAPv1 envelope: no `kind_id` is allocated for
this sketch while it is experimental.

## Accuracy

A value's register index and rank depend only on the value, never on the
bucket, so a key writes the same register array into every bucket it touches.
A bucket no other key shares therefore holds exactly that array, and a key
with at least one collision-free row is answered by HyperLogLog alone — the
relative standard error is `1.04 / sqrt(2^precision)` and nothing is added by
the grid. Collisions can only raise an estimate, never lower it.

## Caveats

- Non-power-of-two `cols` is rejected rather than folded. Folding a hash into
  such a width gives some columns up to twice the load, and the point of the
  minimum across rows is that the rows are comparable.
- The distinct count is per key, not over the whole stream; a value paired
  with several keys is recorded once per key.

## See Also

- [HyperLogLog](./api_hyperloglog.md) — one sketch for the whole stream.
- [Hydra](./api_hydra.md) — the same query on a stable API, with one
  heap-allocated HLL per cell instead of a contiguous register plane.
- [Common Structures](./api_common_structures.md) — `Vector3D`.

## Examples

```rust
use asap_sketchlib::{CountMinHll, DataInput};

// Distinct pages visited, per user.
let mut sk = CountMinHll::default();
for page in 0..500u64 {
    sk.insert(&DataInput::Str("alice"), &DataInput::U64(page));
    sk.insert(&DataInput::Str("bob"), &DataInput::U64(page % 20));
}

let alice = sk.estimate(&DataInput::Str("alice"));
let bob = sk.estimate(&DataInput::Str("bob"));
assert!(alice > bob);
```

## Status

`Experimental`. Behind the `experimental` cargo feature; the API and the wire
form may change without a major version bump, and no ASAPv1 `kind_id` is
allocated for it.
