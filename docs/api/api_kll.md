# API: KLL

Status: `Ready`

## Purpose

Approximate quantile estimation with rank-error guarantees.

## Type/Struct

- `KLL<T = f64>`
- `Cdf`

## Constructors

```rust
fn default() -> Self
fn init_kll(k: i32) -> Self
fn init(k: usize, m: usize) -> Self
fn init_kll_with_seed(k: i32, seed: u64) -> Self
fn init_with_seed(k: usize, m: usize, seed: u64) -> Self
```

The unseeded constructors draw the compaction coin from the wall clock, so two
runs over the same input produce different sketches. The `*_with_seed` forms
fix the coin and store the seed, so `clear()` re-seeds from it too — use them
for reproducible replay, cross-process parity, and any test asserting an
accuracy bound. `KLLDynamic` exposes the same pair
(`init_with_seed` / `init_kll_with_seed`).

## Insert/Update

```rust
fn update(&mut self, val: &T)
fn bulk_update(&mut self, values: &[T])
fn update_data_input(&mut self, val: &DataInput) -> Result<(), &'static str> // KLL<f64> only
fn bulk_update_data_input(&mut self, values: &[DataInput]) -> Result<(), &'static str> // KLL<f64> only
fn clear(&mut self)
```

`bulk_update` / `bulk_update_data_input` are exactly equivalent to looping `update` / `update_data_input` (including coin-state and byte-identical `serialize_to_bytes` for same seed), with empty slices as a no-op. `KLLDynamic` and `HydraCounter::KLL` expose the same `bulk_update` / `bulk_insert` via delegation.

## Query

```rust
fn quantile(&self, q: f64) -> f64
fn rank(&self, x: f64) -> usize
fn count(&self) -> usize
fn cdf(&self) -> Cdf

// Cdf
fn quantile(&self, x: f64) -> f64
fn query(&self, p: f64) -> f64
fn quantile_li(&self, x: f64) -> f64
fn query_li(&self, p: f64) -> f64
```

## Merge

```rust
fn merge(&mut self, other: &KLL<T>)
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

## Examples

```rust
use asap_sketchlib::{KLL, DataInput};

let mut kll = KLL::<i64>::init_kll(200);
kll.update(&10);
kll.update(&20);
let q50 = kll.quantile(0.5);
assert!(q50 >= 10.0);
```

```rust
use asap_sketchlib::{KLL, DataInput};

let mut kll = KLL::<f64>::init_kll(200);
kll.update_data_input(&DataInput::F64(10.0)).unwrap();
kll.update_data_input(&DataInput::F64(20.0)).unwrap();
let q50 = kll.quantile(0.5);
assert!(q50 >= 10.0);
```

## Caveats

- `KLL<T>` is generic over numeric types implementing `NumericalValue`.
- `update_data_input` exists only on `KLL<f64>` for type-erased `DataInput` call paths.
- Query-side APIs still return `f64`.

## Status

Production-usable quantile sketch with comprehensive tests.
