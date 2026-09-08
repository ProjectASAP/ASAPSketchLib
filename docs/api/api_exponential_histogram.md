# API: ExponentialHistogram

Status: `Ready`

## Purpose

Sliding-window coordinator over `EHSketchList` payload buckets.

## Type/Struct

- `EHBucket`
- `ExponentialHistogram`

## Constructors

```rust
fn new(k: usize, window: u64, eh_type: EHSketchList) -> Self
```

## Insert/Update

```rust
fn update(&mut self, time: u64, val: &DataInput)
fn update_with<F>(&mut self, time: u64, update_fn: F) where F: FnOnce(&mut EHSketchList)
fn update_window(&mut self, window: u64)
```

## Query

```rust
fn query_interval_merge(&self, t1: u64, t2: u64) -> Option<EHSketchList>
fn cover(&self, mint: u64, maxt: u64) -> bool
fn get_min_time(&self) -> Option<u64>
fn get_max_time(&self) -> Option<u64>
fn bucket_count(&self) -> usize
fn get_memory_info(&self) -> (usize, Vec<usize>)
```

## Merge

Managed internally during bucket compaction.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error>
```

ASAPv1 MessagePack, kind_id `0x13 0x00`. The metadata carries `window` and `k`;
the payload is `[buckets, sizes, min_times, max_times, prototype]`. `buckets`
holds one inlined `EHSketchList` triple `[kind_id, descriptor, state]` per
bucket, oldest to newest, and the next three arrays are parallel to it;
`prototype` is the `type_to_clone` triple. The bucket count is `len(buckets)`.

There is no hash-spec group: the histogram never hashes, and each bucket's own
`descriptor` carries whatever hash spec its sketch has. `l2_mass` and
`merge_norm` are recomputed on decode rather than carried, so a state whose
cached values disagree with the sketches they derive from does not serialize.

A `k` of zero, a bucket of size zero, an inverted bucket time range, parallel
arrays of unequal length, an unknown or feature-gated nested kind_id, and a
bucket whose descriptor names a different hash profile are all rejected.

## Examples

```rust
use asap_sketchlib::{CountMin, EHSketchList, ExponentialHistogram, FastPath, DataInput, Vector2D};

let template = EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::default());
let mut eh = ExponentialHistogram::new(3, 120, template);
eh.update(10, &DataInput::Str("flow"));
let _ = eh.query_interval_merge(0, 120);
```

## Caveats

- Payload behavior depends on selected `EHSketchList` variant.
- `payload`, `l2_mass` and `merge_norm` are public fields; a histogram edited
  into a state `update_with` cannot reach does not serialize.

## Status

Ready sliding-window coordinator.
