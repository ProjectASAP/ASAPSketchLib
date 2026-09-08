# API: UnivMon

Status: `Ready`

## Purpose

Universal stream-monitoring sketch for L1/L2/cardinality/entropy.

## Type/Struct

- `UnivMon`

## Constructors

```rust
fn default() -> Self
fn init_univmon(heap_size: usize, sketch_row: usize, sketch_col: usize, layer_size: usize) -> Self
```

## Insert/Update

```rust
fn insert(&mut self, key: &DataInput, value: i64)
fn fast_insert(&mut self, key: &DataInput, value: i64)
fn free(&mut self)
```

`insert` maintains the cumulative sampled substream at every selected layer.
`fast_insert` uses the Joltik update-last-layer layout and reconstructs logical
layers during queries. Choose one update method for the lifetime of a sketch;
mixing the two physical layouts is rejected.

## Query

```rust
fn calc_l1(&self) -> f64
fn calc_l2(&self) -> f64
fn calc_entropy(&self) -> f64
fn calc_card(&self) -> f64
fn calc_g_sum<F>(&self, g: F, is_card: bool) -> f64
```

Because UnivMon accepts only non-negative updates, `calc_l1()` returns the
exact tracked total weight. The generic recurrence remains available through
`calc_g_sum` for compatible frequency functions.

## Merge

```rust
fn merge(&mut self, other: &UnivMon)
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope (kind `0x10 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). UnivMon hashes through the
crate's default hasher, so its metadata carries that profile and no seed index:
the bottom-layer finder's seed is fixed by the algorithm, and layer `i`'s
counter hashes at seed index `i`.

The pyramid shape (`layer_size`, `sketch_row`, `sketch_col`, `heap_size`) and
the heaps' `key_type` live in the metadata; the payload is
`[counts, l2, heap_lens, keys, heap_counts, candidate_complete, bucket_size,
update_mode]`. Each layer's `CountL2HH` state is inlined rather than nested in
an envelope of its own, and the heap indexes are rebuilt on load, so no index
reaches the wire. Heap entries are emitted layer by layer, each layer in
descending count with ties broken by a total order over the key, so a decoded
pyramid re-serializes byte-identically. `update_mode` and `candidate_complete`
are carried: they are acquired from the stream, and a decoder that guessed them
would pick the wrong query recurrence or report a widened threshold as zero.

Every layer is a `CountL2HH` matrix hashed per row, so `sketch_row` carries that
sketch's bound: `1 <= sketch_row <= 20` (`MATRIX_MAX_ROWS`, the seed list
length), refused on both sides past it.

## Examples

```rust
use asap_sketchlib::{DataInput, UnivMon};

let mut um = UnivMon::init_univmon(32, 3, 1024, 4);
um.insert(&DataInput::Str("flow"), 1);
assert!(um.calc_l1() >= 1.0);
```

## Caveats

- Updates must have non-negative weights.
- Structure parameters and update strategies must match before merge.
- Merge combines total weight, recomputes cached L2 state, and re-estimates the
  union of bounded heavy-hitter candidates from merged counters.

## Status

Primary multi-metric framework.
