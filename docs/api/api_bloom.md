# API: Bloom

Status: `Ready`

## Purpose

Approximate set membership. `contains` never says no about a key that was
inserted, and says yes about a bounded fraction of keys that were not.

## Type/Struct

- `Bloom<Mode = RegularPath, H = DefaultXxHasher>`
- `BitMatrix` — the packed bit grid, a `MatrixStorage` in its own right

This is the *partitioned* variant: the filter is `rows` slices of `cols` bits,
one slice per hash function, which is the `rows x cols` shape `CountMin` probes.
A membership query is the minimum across rows, which over single bits is their
AND.

## Constructors

```rust
fn with_capacity(expected_items: usize, target_fpp: f64) -> Self
fn with_dimensions(rows: usize, cols: usize) -> Self
fn dimensions_for(expected_items: usize, target_fpp: f64) -> (usize, usize)

const BLOOM_MAX_SLICES: usize   // MATRIX_MAX_ROWS (20), the hasher's seed count
const BLOOM_MAX_BITS: usize     // 2^31, the sizing ceiling
```

`with_capacity` takes `k = round(log2(1/p))` slices, **capped at
`BLOOM_MAX_SLICES`**, then solves for the slice width that hits `p` with
exactly that many slices — `cols = -n / ln(1 - p^(1/k))` — and rounds it up to a
power of two. The rounding keeps the column fold free of modulo bias, so the
delivered rate lands under the target rather than over it. `dimensions_for`
reports the choice without allocating.

The cap is the reason the width is solved for `k` rather than taken as the
`k`-optimal split `m/k`. Row `r` hashes with seed index `r % 20`, so a filter
asking for 23 slices would get 20 distinct hash functions and 3 copies. Capping
`k` and widening the slices to compensate costs bits — about 20% more at
`p = 1e-12` — and delivers the target, where the `k`-optimal split would have
delivered `2^-20` no matter what was asked for.

`with_capacity` and `dimensions_for` **panic on a NaN or infinite
`target_fpp`**; a finite value outside `(0, 1)` is clamped into it. A target
needing more than `BLOOM_MAX_BITS` gets the widest slices that fit, and
`predicted_fpp` then reports the rate those slices deliver.

`Default` is 7 x 65536 — 56 KiB of packed bits — which holds 20k distinct keys
at about 1 false positive in 11,500.

`with_dimensions` is the escape hatch when memory is fixed. A non-power-of-two
`cols` folds with a modulo and carries the bias the sized path avoids. More than
`BLOOM_MAX_SLICES` rows is legal but pointless: rows past the seed list repeat
an earlier row bit for bit.

## Insert/Update

```rust
fn insert(&mut self, value: &DataInput)
fn bulk_insert(&mut self, values: &[DataInput])
```

One bit per slice. Re-inserting a key sets nothing new, so the fill is a
function of the distinct keys alone while `inserted()` counts every call.

## Query

```rust
fn contains(&self, value: &DataInput) -> bool
```

False is exact: the key was never inserted. True is probabilistic.

```rust
fn predicted_fpp(&self, distinct_items: usize) -> f64
fn estimated_fpp(&self) -> f64
fn fill_ratio(&self) -> f64
fn inserted(&self) -> u64
fn rows(&self) -> usize
fn effective_rows(&self) -> usize
fn cols(&self) -> usize
fn bit_capacity(&self) -> usize
fn size_in_bytes(&self) -> usize
fn is_empty(&self) -> bool
```

`predicted_fpp` is the model, `(1 - e^(-n/cols))^effective_rows`.
`estimated_fpp` reads the bits actually set, so it is the one to trust on a
filter whose distinct count is unknown. Both raise the per-slice rate to
`effective_rows`, not `rows`: duplicate slices agree by construction and add no
selectivity, so counting them would report a rate the filter cannot deliver.

## Merge

```rust
fn merge(&mut self, other: &Self)
```

Bitwise union. Both filters must have the same dimensions and hasher; the
result is exactly the filter the concatenated streams would have built, which
makes the filter shardable without loss. Mismatched dimensions assert.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope (kind `0x17 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). The impl is bounded on
`Mode: BloomMode` and `H: SketchHasher + HashProfile`, so a hasher that declares
no profile cannot serialize at all. `rows`, `cols` and `mode` are carried in the
envelope metadata; the payload is just `[words, inserted]`, the packed bit words
row-major plus the insert counter, with the word stride derived from `cols`.

The wire covers the geometries `with_capacity` produces: at most
`BLOOM_MAX_SLICES` slices, a power-of-two `cols`, and at most `BLOOM_MAX_BITS`
bits. `with_dimensions` can build a filter outside that subset, and both
`serialize_to_bytes` and `deserialize_from_bytes` reject it, so the format never
emits bytes it would refuse to read back. Decode also rejects a word count that
disagrees with the declared dimensions, and any bit set in a row's trailing
padding past `cols` — unreachable by `contains`, but counted by `count_ones`, so
it would skew `fill_ratio` and `estimated_fpp` alone.

Independently of ASAPv1, `Bloom` is plain `Serialize`/`Deserialize`; that form is
`{ bits, inserted, mode }`, where `mode` is `"regular"` or `"fast"`. Decoding
into the other hash path fails rather than producing a filter that reports its
own members absent. `BitMatrix` carries `{ words, rows, cols }` only — the word
stride and the fold masks are recomputed on decode, and a payload whose word
count disagrees with its dimensions is rejected there rather than panicking
later.

## Examples

```rust
use asap_sketchlib::{Bloom, DataInput, RegularPath};

let mut seen = Bloom::<RegularPath>::with_capacity(1_000_000, 0.001);
seen.insert(&DataInput::Str("10.0.0.1:443"));

if !seen.contains(&DataInput::Str("10.0.0.2:443")) {
    // definitely new
}
```

## Caveats

- No deletion. Clearing a bit would erase it for every key that hashes there.
  A counting variant would need a counter matrix rather than a bit one.
- No count and no cardinality. The filter answers membership only.
- Whether the two hash paths set the same bits depends on the geometry. When
  `rows * (log2(cols) + 1)` fits in 128 bits the fast path slices one packed
  hash into per-row windows and lands on different columns than the regular
  path's per-row seeded hashes; past 128 bits it falls back to those same
  per-row hashes and the two agree exactly. The default 7 x 65536 needs 119
  bits and so **differs**; 8 x 65536 needs 136 and so **agrees**. Treat the two
  as non-interchangeable regardless — a filter validated on an agreeing
  geometry will lose almost every member on a packed one. The `mode` tag makes
  a cross-path decode fail rather than fail silently.
- Selectivity stops at `BLOOM_MAX_SLICES` (20) slices, the hasher's seed count.
  `with_capacity` never asks for more; `with_dimensions` will, and the extra
  rows cost memory and a hash each for nothing.
- `with_dimensions(1, 1)` is legal and answers yes to everything after the
  first insert.

## Relation to the paper

Bloom's 1970 construction with the per-slice partitioning of Kirsch and
Mitzenmacher: `k` disjoint slices rather than one shared array. The false-positive
constant is marginally worse than the shared-array form and the row loop is the
same one every matrix-backed sketch in this crate runs.

## Status

Ready.
