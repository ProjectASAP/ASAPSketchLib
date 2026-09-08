# API: HyperLogLog

Status: `Ready`

## Purpose

Approximate cardinality (distinct-count) estimation.

## Types

The library provides three HLL algorithm variants, each available at
multiple precision levels.

### Algorithm Variants

- `HyperLogLog<Classic, H>` — classic HyperLogLog (Flajolet et al.).
- `HyperLogLog<ErtlMLE, H>` — improved estimator (Ertl, arXiv:1702.01284).
- `HyperLogLogHIP` — Historic Inverse Probability estimator (Lang, arXiv:1708.06839). Not mergeable.

### Precision Aliases

Each variant is backed by a register storage with a configurable precision
parameter `p` (number of address bits). Higher precision uses more memory
but improves accuracy.

| Alias | Precision | Registers | Underlying Type |
| --- | --- | --- | --- |
| `HyperLogLogP12<Variant, H>` | p=12 | 4,096 | `HyperLogLogImpl<Variant, HllBucketListP12, H>` |
| `HyperLogLogP14<Variant, H>` | p=14 | 16,384 | `HyperLogLogImpl<Variant, HllBucketListP14, H>` |
| `HyperLogLogP16<Variant, H>` | p=16 | 65,536 | `HyperLogLogImpl<Variant, HllBucketListP16, H>` |
| `HyperLogLog<Variant, H>` | p=14 (default) | 16,384 | = `HyperLogLogP14<Variant, H>` |
| `HyperLogLogHIPP12` | p=12 | 4,096 | `HyperLogLogHIPImpl<HllBucketListP12>` |
| `HyperLogLogHIPP14` | p=14 | 16,384 | `HyperLogLogHIPImpl<HllBucketListP14>` |
| `HyperLogLogHIP` | p=14 (default) | 16,384 | = `HyperLogLogHIPP14` |

`H` defaults to `DefaultXxHasher` and can be omitted in most usage.

## Constructors

```rust
fn new() -> Self
fn default() -> Self
```

## Insert / Update

```rust
fn insert(&mut self, obj: &DataInput)
fn insert_many(&mut self, items: &[DataInput])
fn insert_with_hash(&mut self, hashed: u64)
fn insert_many_with_hashes(&mut self, hashes: &[u64])
```

## Query

```rust
fn estimate(&self) -> usize
```

`Classic` also exposes:

```rust
fn indicator(&self) -> f64
```

## Merge

```rust
fn merge(&mut self, other: &Self)
```

Available on `Classic` and `ErtlMLE` variants. **Not available on HIP.**

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). On the generic
`HyperLogLogImpl<Variant, Registers, H>`, the methods are bounded on
`Variant: HllWireVariant` and `H: HashProfile`, so every
(variant × precision × hasher) combination serializes — **HLL is fully
wire-covered**. Each variant maps to its own `kind_id` (Classic → `0x01 0x01`,
Ertl-MLE → `0x01 0x02`, HIP → `0x01 0x03`); `precision` and the
`HashProfile`-derived hash spec live in the envelope metadata, and the payload is
the register bytes (plus three HIP scalars for the HIP variant). `HyperLogLogHIP`
is a non-generic struct that hashes through the default functions, so it is
wire-eligible under the **standard** hash profile only.

## Examples

```rust
use asap_sketchlib::{ErtlMLE, HyperLogLog, DataInput};

let mut hll = HyperLogLog::<ErtlMLE>::default();
for i in 0..1000u64 {
    hll.insert(&DataInput::U64(i));
}
let card = hll.estimate();
assert!(card > 900);
```

## Caveats

- `HyperLogLogHIP` is not mergeable.
- P12 variants trade accuracy for lower memory; expect slightly higher
  error than the default P14.

## Status

Canonical cardinality implementation in this library.
