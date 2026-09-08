# API: EHSketchList

Status: `Ready`

## Purpose

Unified enum wrapper for sketch payloads used by EH-style frameworks.

## Type/Struct

```rust
enum EHSketchList {
    CM(CountMin<Vector2D<i32>, FastPath>),
    COCO(Coco),
    COUNTL2HH(CountL2HH),
    CS(Count<Vector2D<i32>, FastPath>),
    DDS(DDSketch),
    ELASTIC(Elastic),
    HLL(HyperLogLog<ErtlMLE>),
    KLL(KLL),
    #[cfg(feature = "experimental")]
    UNIFORM(UniformSampling),
    UNIVMON(UnivMon),
}
```

## Constructors

Enum-based; construct by variant.

## Insert/Update

```rust
fn insert(&mut self, val: &DataInput)
```

## Query

```rust
fn query(&self, key: &DataInput) -> Result<f64, &'static str>
fn supports_norm(&self, norm: SketchNorm) -> bool
fn sketch_type(&self) -> &'static str
```

## Merge

```rust
fn merge(&mut self, other: &EHSketchList) -> Result<(), &'static str>
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error>
```

ASAPv1 MessagePack, kind_id `0x14 0x00`. The metadata carries only
`metadata_version`; the payload is the triple `[kind_id, descriptor, state]`,
all three msgpack `bin`. A nested variant carries its **own** kind_id, metadata
block and payload block, with the envelope framing (magic, version, lengths)
stripped, so each variant's own validation applies unchanged. An
`ExponentialHistogram` bucket inlines the same triple, so there is one
EHSketchList encoding.

The ten nested kind_ids:

| Variant | Nested kind_id | Registry name | Feature |
| ------- | -------------- | ------------- | ------- |
| `CM` | `0x02 0x00` | Count-Min | default |
| `COCO` | `0x0c 0x00` | Coco | default |
| `COUNTL2HH` | `0x19 0x00` | CountL2HH | default |
| `CS` | `0x04 0x00` | Count Sketch | default |
| `DDS` | `0x05 0x00` | DDSketch | default |
| `ELASTIC` | `0x0b 0x00` | Elastic | default |
| `HLL` | `0x01 0x02` | HLL Ertl-MLE | default |
| `KLL` | `0x06 0x00` | KLL compact | default |
| `UNIFORM` | `0x0d 0x00` | UniformSampling | `experimental` |
| `UNIVMON` | `0x10 0x00` | UnivMon | default |

Each id is pinned to one algorithm: `HLL` is Ertl-MLE, so `0x01 0x01` (Classic)
and `0x01 0x03` (HIP) are rejected, and `KLL` is compact, so `0x06 0x01`
(dynamic) is rejected. The dispatch is the same in every build: a decoder built
without `experimental` rejects `0x0d 0x00` with an error naming the variant and
the feature, and its encoder can never emit it.
An unrecognized kind_id is rejected. `EHSketchList` also derives serde.

## Examples

```rust
use asap_sketchlib::{CountMin, EHSketchList, FastPath, DataInput, Vector2D};

let mut sk = EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::default());
sk.insert(&DataInput::U64(1));
let _ = sk.query(&DataInput::U64(1));
```

## Caveats

- Some merge/query variant combinations are intentionally unsupported.

## Status

Core wrapper in EH and optimized window frameworks.
