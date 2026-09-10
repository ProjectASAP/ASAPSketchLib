# API: Common Structures

Status: `Shared`

## Purpose

Shared matrix, vector, and bit storage plus utility structures used by sketch implementations.

## Type/Struct

- `Vector2D<T>`
- `Vector3D<T>`
- `BitMatrix`
- `MatrixStorage` / `FastPathHasher`
- `MatrixHashType`
- `Nitro`

## Constructors

```rust
// Vector2D
fn init(rows: usize, cols: usize) -> Self
fn from_fn<F>(rows: usize, cols: usize, f: F) -> Self

// Vector3D
fn init(rows: usize, cols: usize, depth: usize) -> Self
fn fill(&mut self, value: T)

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

// Vector3D — the closure receives a whole `depth`-length bucket
fn fast_insert<Hash, F, V>(&mut self, op: F, value: V, hashed_val: &Hash)
fn as_mut_slice(&mut self) -> &mut [T]

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

// Vector3D
fn rows(&self) -> usize
fn cols(&self) -> usize
fn depth(&self) -> usize
fn bucket_slice(&self, row: usize, col: usize) -> &[T]
fn as_slice(&self) -> &[T]
fn get_mask_bits(&self) -> u32
fn fast_query_min<Hash, F, R>(&self, hashed_val: &Hash, op: F) -> R  // op: Fn(&[T], usize)

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

```rust
use asap_sketchlib::Vector3D;

// A 3 x 16 grid whose every cell is a 20-byte record.
let mut cells = Vector3D::<u8>::init(3, 16, 20);
cells.fill(0);
assert_eq!(cells.bucket_slice(0, 0).len(), 20);
// A whole-table sweep walks the storage in order.
assert_eq!(cells.as_slice().chunks_exact(cells.depth()).count(), 3 * 16);
```

## When to reach for `Vector3D` instead of `Vector2D`

`Vector3D<T>` is a `rows x cols` grid whose every cell is a contiguous
`depth`-length **record**. Reach for it when, and only when, a cell is made of
several **mutually coupled** fields — when the estimator or the update has to
read or write those fields as a unit.

| Structure | Fields in a cell | Coupled how | Vector3D? |
| --- | --- | --- | --- |
| `MicroCM` | `T+2` pixels, zoom `Z`, shutter `S` | every pixel is scaled by the one shared `Z`, and a zoom rescales all of them at once | Yes |
| `CountMinHll` | `2^p` HyperLogLog registers | the estimator reads the whole register array | Yes |
| A hypothetical multi-metric Count-Min (bytes and packets side by side) | `k` independent counters | not coupled at all | No — `k` separate `Vector2D`s are the same thing |

`Vector2D<[T; N]>` has exactly the same memory layout as `Vector3D<T>` at
`depth = N`. The one thing `Vector3D` adds is a **`depth` chosen at run time**,
which is what both callers need: a HyperLogLog precision and a sub-window
count are configuration values, also read back from deserialized bytes, and a
const generic cannot express either.

The layout is array-of-structs: a cell's elements are adjacent and
consecutive cells follow one another, so a per-item path touches one
contiguous run and a whole-table pass over
`as_mut_slice().chunks_exact_mut(depth())` walks the storage in order. How
much of a cell a given sketch reads per item is its own business — `MicroCM`
rewrites the record, `CountMinHll` touches one byte of a 256-byte one. A
sketch whose *hot* path instead touches one field across all cells is not a
fit either way.

## Caveats

- This page summarizes commonly used entry points; full module context remains in [Common Module (Canonical)](./api_common.md).
- `BitMatrix` packs one bit per cell into `u64` words, one row after another, and satisfies the same `MatrixStorage` interface as the counter matrices.
- `BitMatrix::get`, `set` and `put` panic on a `row` or `col` outside the grid. Rows are padded to a whole number of words, so an unchecked column past `cols` would land on a padding bit or in the next row rather than out of the allocation.
- A serialized `Vector3D` carries `data`, `rows`, `cols` and `depth` only; the column mask is recomputed on load, and a payload whose data length disagrees with `rows * cols * depth`, or that declares a zero dimension, is rejected.
- `Vector3D::init` reserves capacity but leaves the storage empty; `fill` must run before any bucket accessor.
- A serialized `BitMatrix` carries `words`, `rows` and `cols` only; the word stride and column mask are recomputed on load, and a payload whose word count disagrees with its dimensions is rejected.

## See Also

- [Common Module (Canonical)](./api_common.md)
- [Common Input Types](./api_common_input.md)

## Status

Canonical shared structures layer.
