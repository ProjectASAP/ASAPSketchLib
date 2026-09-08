# API: DDSketch

Status: `Ready`

## Purpose

Approximate quantile estimation with relative error guarantees.

## Type/Struct

- `DDSketch`

## Constructors

```rust
fn new(alpha: f64) -> Self
```

## Insert/Update

```rust
fn add<T: NumericalValue>(&mut self, val: &T)
fn add_input(&mut self, v: &DataInput) -> Result<(), &'static str>
```

`add` accepts any type implementing `NumericalValue` (all standard integer and
float primitives). Values are internally converted to `f64`; non-positive or
non-finite values are ignored. `add_input` is the `DataInput` adapter used by
the type-erased dispatch path.

## Query

```rust
fn get_value_at_quantile(&self, q: f64) -> Option<f64>
fn get_count(&self) -> u64
fn min(&self) -> Option<f64>
fn max(&self) -> Option<f64>
```

## Merge

```rust
fn merge(&mut self, other: &DDSketch) -> Result<(), String>
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope (kind `0x05 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). DDSketch never hashes its
inputs, so the metadata carries **no hash-spec group**: it is `metadata_version`
plus `alpha`, the sketch's one construction parameter. `gamma` and its logs are
derived from `alpha` and never reach the wire.

The payload is `[counts, offset, sum, min, max]`: the dense bucket store carried
verbatim (the bucket index of `counts[i]` is `offset + i`), plus the three
scalars the buckets do not determine. `count` is **not** carried — it is the sum
of the bucket counts and is recovered exactly on decode. `sum`, `min` and `max`
come back exactly rather than as α-bounded bucket estimates, so a decoded sketch
re-serializes byte-identically.

There is one positive-range store and no zero-count bucket, because `add` drops
non-positive, non-finite and non-indexable values. Bucket indices are still
signed, so `offset` may be negative. Decode rejects an `alpha` outside `(0, 1)`,
a store span past `i32`, a non-zero offset on an empty store, bucket counts that
overflow the total, and scalars inconsistent with that total; `serialize_to_bytes`
enforces the same rules, so the format never emits bytes it would refuse to read
back.

Independently of ASAPv1, `DDSketch` is plain `Serialize`/`Deserialize` for use
where it is nested in another type (`EHSketchList`); that form skips the four
running scalars.

## Examples

```rust
use asap_sketchlib::DDSketch;

let mut dds = DDSketch::new(0.01);
dds.add(&1.0_f64);
dds.add(&2_i32);
let p50 = dds.get_value_at_quantile(0.5).unwrap();
assert!(p50 >= 1.0);
```

## Quantile convention

`get_value_at_quantile(q)` uses the **nearest-rank** convention
`rank = ceil(q * n)` (1-based), so it answers the order statistic
`sorted[ceil(q*n) - 1]`. `q <= 0` and `q >= 1` short-circuit to the exactly
retained minimum and maximum rather than to a bucket representative, so the
endpoints carry zero error.

This differs from the portable wire twin
`message_pack_format::portable::ddsketch::DdSketch::quantile`, which uses the
lower-quantile convention `floor(q * (n - 1))` of the DDSketch paper and of the
Go reference implementation, and which retains no min/max scalars — its
endpoints are ordinary bucket representatives, accurate to `alpha` like any
other rank. The two agree at most `q` and diverge at small `n` or ragged `q`
(at `n = 3, q = 0.4` this type answers `sorted[1]` and the portable one
`sorted[0]`). Both conventions are deliberate and are pinned by
`ddsketch_core_and_portable_answer_different_order_statistics`.

## Caveats

- Inputs must be positive values.
- Merge returns `Err` unless both sketches share the same `alpha`.

## Status

Supported and tested for multiple distributions.
