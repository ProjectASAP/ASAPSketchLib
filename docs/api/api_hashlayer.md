# API: HashSketchEnsemble

Status: `Ready`

## Purpose

Group multiple compatible sketches that share a hash layout so the hash is
computed once per insert and fanned out to every sketch in the ensemble.
Supports both frequency sketches (CountMin, Count) and cardinality sketches
(HyperLogLog variants).

### Quick Example

```rust
use asap_sketchlib::*;
use asap_sketchlib::HashSketchEnsemble;

// Two CMS + one HLL sharing one hash per insert
let mut ensemble = HashSketchEnsemble::<DefaultXxHasher>::new(vec![
    CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 4096).into(),
    CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 4096).into(),
    HyperLogLog::<ErtlMLE>::default().into(),
]).unwrap();

// Insert — hashes once, updates all 3 sketches
ensemble.insert(&DataInput::U64(42));

// Query frequency (CMS at index 0)
let freq = ensemble.estimate(0, &DataInput::U64(42)).unwrap();

// Query cardinality (HLL at index 2)
let card = ensemble.cardinality(2).unwrap();

// Pre-computed hash path for hot loops
let hash = ensemble.hash_input(&DataInput::U64(42));
ensemble.insert_with_hash(&hash);
let freq = ensemble.estimate_with_hash(0, &hash).unwrap();
```

### Motivation

Without an ensemble, sharing a hash across sketches requires manual
coordination:

```rust
let hash = hash_for_matrix_seeded_generic::<MyHasher>(0, rows, cols, &input);
cms_a.fast_insert_with_hash_value(&hash);
cms_b.fast_insert_with_hash_value(&hash);
hll.insert_with_hash(hash.lower_64());
```

`HashSketchEnsemble` wraps this pattern into a single structure that manages
the hash configuration, validates dimensional compatibility, and exposes a
uniform insert/query API.

## Types

- `HashSketchEnsemble<H = DefaultXxHasher>` — the ensemble container, generic over the hasher.
- `EnsembleSketch` — enum wrapping the sketch variants that can live inside an ensemble: `CountMinFast`, `CountFast`, `HllErtl`, `HllClassic`, `HllHip`.

## Compatible Sketches

Only sketches with a prehashed insertion path are accepted:

- `CountMin<_, FastPath, _>` — Count-Min Sketch (fast path)
- `Count<_, FastPath, _>` — Count (Count Sketch, fast path)
- `HyperLogLog<ErtlMLE>` / `HyperLogLog<Classic>` / `HyperLogLogHIP`

All matrix-backed sketches (CMS / `Count`) in one ensemble must share the same hash layout (rows × cols dimensions). HLL sketches can coexist with them because they only consume the lower 64 bits of the shared hash.

## Construction

```rust
fn new(sketches: Vec<EnsembleSketch>) -> Result<Self, &'static str>
fn push(&mut self, sketch: EnsembleSketch) -> Result<(), &'static str>
```

Sketches are converted into `EnsembleSketch` via `From` impls, so you can use `.into()`:

```rust
use asap_sketchlib::{
    CountMin, Count, FastPath, Vector2D, DefaultXxHasher,
    HyperLogLog, ErtlMLE,
};
use asap_sketchlib::sketch_framework::hashlayer::{EnsembleSketch, HashSketchEnsemble};

let ensemble = HashSketchEnsemble::<DefaultXxHasher>::new(vec![
    CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 4096).into(),
    Count::<Vector2D<i32>, FastPath>::with_dimensions(3, 4096).into(),
    HyperLogLog::<ErtlMLE>::default().into(),
]).expect("compatible sketches");
```

## Hashing

```rust
fn hash_input(&self, input: &DataInput) -> H::HashType
```

Computes the shared hash for an input using the ensemble's hash configuration and hasher `H`.

## Insertion

```rust
fn insert(&mut self, val: &DataInput)
fn insert_with_hash(&mut self, hash: &H::HashType)
fn insert_at(&mut self, indices: &[usize], val: &DataInput)
fn insert_at_with_hash(&mut self, indices: &[usize], hash: &H::HashType)
fn bulk_insert(&mut self, values: &[DataInput])
fn bulk_insert_with_hashes(&mut self, hashes: &[H::HashType])
fn bulk_insert_at(&mut self, indices: &[usize], values: &[DataInput])
fn bulk_insert_at_with_hashes(&mut self, indices: &[usize], hashes: &[H::HashType])
```

`insert` hashes once and fans out to all sketches. The `_at` variants target specific sketch indices. The `bulk_` variants accept slices for batch processing.

## Query

Frequency and cardinality queries are split because CMS/`Count` and HLL answer fundamentally different questions.

### Frequency (CMS / Count only)

```rust
fn estimate(&self, index: usize, val: &DataInput) -> Result<f64, &'static str>
fn estimate_with_hash(&self, index: usize, hash: &H::HashType) -> Result<f64, &'static str>
```

Returns an error if the index is out of bounds or the sketch is not a frequency sketch.

### Cardinality (HLL only)

```rust
fn cardinality(&self, index: usize) -> Result<f64, &'static str>
```

Returns an error if the index is out of bounds or the sketch is not an HLL variant.

### Accessors

```rust
fn len(&self) -> usize
fn is_empty(&self) -> bool
fn get(&self, index: usize) -> Option<&EnsembleSketch>
fn get_mut(&mut self, index: usize) -> Option<&mut EnsembleSketch>
```

## Merge

No ensemble-level merge API.

## Serialization

No dedicated serialization API.

## Caveats

- All matrix-backed sketches in one ensemble must have identical dimensions.
- Calling `estimate` on an HLL sketch returns an error; use `cardinality` instead (and vice versa).

## Status

Core optimization layer; actively used and tested.
