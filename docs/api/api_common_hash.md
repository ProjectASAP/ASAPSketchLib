# API: Common Hash Utilities

Status: `Shared`

## Purpose

Shared hashing traits, constants, and helpers used by sketches/frameworks.

## Type/Struct

- `SketchHasher`
- `DefaultXxHasher`
- `DigestHasher` — `std::hash::Hasher` for `u64` keys that are already a digest; one avalanche step, no full hash
- `DigestBuildHasher` — `BuildHasherDefault<DigestHasher>`, the `BuildHasher` to hand a `HashMap`
- `MatrixHashMode`
- `MatrixHashType` (see matrix storage)

## Constructors

Not applicable.

## Insert/Update

Not applicable.

## Query

```rust
pub const SEEDLIST: [u64; 20]
pub const MATRIX_MAX_ROWS: usize
pub const CANONICAL_HASH_SEED: usize
pub const BOTTOM_LAYER_FINDER: usize
pub const HYDRA_SEED: usize

fn hash64_seeded(d: usize, key: &DataInput) -> u64
fn hash128_seeded(d: usize, key: &DataInput) -> u128
fn hash_item64_seeded(d: usize, key: &HeapItem) -> u64
fn hash_item128_seeded(d: usize, key: &HeapItem) -> u128
fn hash_mode_for_matrix(rows: usize, cols: usize) -> MatrixHashMode
fn hash_for_matrix(rows: usize, cols: usize, key: &DataInput) -> MatrixHashType
fn hash_for_matrix_seeded(seed_idx: usize, rows: usize, cols: usize, key: &DataInput) -> MatrixHashType
fn hash_for_matrix_seeded_with_mode(seed_idx: usize, mode: MatrixHashMode, rows: usize, key: &DataInput) -> MatrixHashType
fn hash_for_matrix_seeded_with_mode_generic<H: SketchHasher>(
    seed_idx: usize,
    mode: MatrixHashMode,
    rows: usize,
    key: &DataInput,
) -> MatrixHashType
fn hash_for_matrix_generic<H: SketchHasher>(rows: usize, cols: usize, key: &DataInput) -> MatrixHashType
fn hash_for_matrix_seeded_generic<H: SketchHasher>(
    seed_idx: usize,
    rows: usize,
    cols: usize,
    key: &DataInput,
) -> MatrixHashType
```

## Merge

Not applicable.

## Serialization

Not applicable.

## Examples

```rust
use asap_sketchlib::{hash64_seeded, DataInput, CANONICAL_HASH_SEED};

let h = hash64_seeded(CANONICAL_HASH_SEED, &DataInput::U64(42));
assert!(h > 0 || h == 0);
```

## Caveats

- Matrix hash helper selection depends on row/column shape.
- Seed indices wrap at `SEEDLIST.len()`, so a matrix row `r` and row
  `r + MATRIX_MAX_ROWS` draw the same seed and place every key identically.

## See Also

- [Common Module (Canonical)](./api_common.md)
- [Common Input Types](./api_common_input.md)

## Status

Canonical shared hash utility layer.
