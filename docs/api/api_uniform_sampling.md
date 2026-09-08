# API: UniformSampling

Status: `Unstable`

> Warning: Useful and tested, but currently documented as Unstable until broader API alignment work completes.

## Purpose

Reservoir-like uniform sampler with merge support.

## Type/Struct

- `UniformSampling`

## Constructors

```rust
fn new(sample_rate: f64) -> Self
fn with_seed(sample_rate: f64, seed: u64) -> Self
```

## Insert/Update

```rust
fn update(&mut self, value: f64)
fn update_input(&mut self, value: &DataInput) -> Result<(), &'static str>
```

## Query

```rust
fn sample_rate(&self) -> f64
fn len(&self) -> usize
fn is_empty(&self) -> bool
fn total_seen(&self) -> u64
fn samples(&self) -> Vec<f64>
fn sample_at(&self, idx: usize) -> Option<f64>
```

## Merge

```rust
fn merge(&mut self, other: &UniformSampling) -> Result<(), &'static str>
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error>
```

ASAPv1 wire format, kind_id `0x0d 0x00`. Metadata carries `metadata_version`,
`sample_rate` and `item_type`; the sampler does not hash, so there is no
hash-spec group. The payload is `[priorities, values, total_seen, rng_state]`,
emitted in ascending priority. `rng_state` is the SplitMix64 word the next
priority is drawn from, so a decoded sampler continues the same draw sequence.

Also derives serde for the internal Rust-only codec.

## Examples

```rust
use asap_sketchlib::UniformSampling;

let mut sk = UniformSampling::new(0.2);
sk.update(1.0);
let _ = sk.samples();
```

## Accuracy

This is **priority (bottom-k) sampling**: each update draws an independent
uniform 64-bit priority and the list is truncated to the smallest
`ceil(total_seen * sample_rate)` priorities. Two consequences:

- the retained size is `ceil(n * rate)` **exactly** — it is computed, never
  sampled, so it has no band around it;
- because the priorities are independent of the values, the retained set is a
  uniform random sample **without replacement** of that size. A sample
  statistic therefore carries the finite-population variance
  `Var[mean] = (sigma_N^2 / m) * (N - m) / (N - 1)`.

## Caveats

- Supports numeric inputs only in `update_input`.
- Merge requires matching sampling rates.

## Status

Unstable.
