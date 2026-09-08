# API: Common Structures

Status: `Shared`

## Purpose

Shared matrix, vector, and bit storage plus utility structures used by sketch implementations.

## Type/Struct

- `Vector2D<T>`
- `BitMatrix`
- `MatrixStorage` / `FastPathHasher`
- `MatrixHashType`
- `Nitro`

## Constructors

```rust
// Vector2D
fn init(rows: usize, cols: usize) -> Self
fn from_fn<F>(rows: usize, cols: usize, f: F) -> Self

// BitMatrix
fn new(rows: usize, cols: usize) -> Self

// Nitro
fn init_nitro(rate: f64) -> Self
```

## Insert/Update

```rust
// Vector2D
fn update_one_counter<F, V>(&mut self, row: usize, col: usize, op: F, value: V)
fn fast_insert<F, V>(&mut self, op: F, value: V, hashed_val: &MatrixHashType)
fn update_by_row<F, V>(&mut self, row: usize, hashed: u128, op: F, value: V)

// BitMatrix
fn set(&mut self, row: usize, col: usize)
fn put(&mut self, row: usize, col: usize, value: bool)
fn union_from(&mut self, other: &Self)
fn clear(&mut self)

// Nitro utility
fn draw_geometric(&mut self)
fn reduce_to_skip(&mut self)
fn reduce_to_skip_by_count(&mut self, c: usize)
```

## Query

```rust
// Vector2D
fn rows(&self) -> usize
fn cols(&self) -> usize
fn get(&self, row: usize, col: usize) -> Option<&T>
fn row_slice(&self, row: usize) -> &[T]
fn fast_query_min<F, R>(&self, hashed_val: &MatrixHashType, op: F) -> R
fn fast_query_median<F>(&self, hashed_val: &MatrixHashType, op: F) -> f64
fn fast_query_max<F, R>(&self, hashed_val: &MatrixHashType, op: F) -> R

// BitMatrix
fn rows(&self) -> usize
fn cols(&self) -> usize
fn get(&self, row: usize, col: usize) -> bool
fn count_ones(&self) -> usize
fn fill_ratio(&self) -> f64
fn size_in_bytes(&self) -> usize

// Utility
fn compute_median_inline_f64(values: &mut [f64]) -> f64
```

## Merge

Not applicable at this utility-layer boundary.

## Serialization

Not applicable at this utility-layer boundary.

## Examples

```rust
use asap_sketchlib::Vector2D;

let matrix = Vector2D::<i32>::init(3, 16);
assert_eq!(matrix.rows(), 3);
assert_eq!(matrix.cols(), 16);
```

## Caveats

- This page summarizes commonly used entry points; full module context remains in [Common Module (Canonical)](./api_common.md).
- `BitMatrix` packs one bit per cell into `u64` words, one row after another, and satisfies the same `MatrixStorage` interface as the counter matrices.
- `BitMatrix::get`, `set` and `put` panic on a `row` or `col` outside the grid. Rows are padded to a whole number of words, so an unchecked column past `cols` would land on a padding bit or in the next row rather than out of the allocation.
- A serialized `BitMatrix` carries `words`, `rows` and `cols` only; the word stride and column mask are recomputed on load, and a payload whose word count disagrees with its dimensions is rejected.

## See Also

- [Common Module (Canonical)](./api_common.md)
- [Common Input Types](./api_common_input.md)

## Status

Canonical shared structures layer.
