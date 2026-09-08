# API: Hydra

Status: `Ready`

## Purpose

Subpopulation sketching over a fixed schema of labelled key columns; an update
fans its row out into all `2^D - 1` non-empty subpopulations.

## Type/Struct

- `Hydra`

## Constructors

```rust
fn with_schema<S, I>(r: usize, c: usize, schema: I, sketch_type: HydraCounter) -> Result<Self, String>
```

## Insert/Update

```rust
fn update(&mut self, key: &[&str], value: &DataInput, count: Option<i32>) -> Result<(), String>
```

## Query

```rust
fn query_key(&self, key: &[Option<&str>], query: &HydraQuery) -> Result<f64, String>
fn query_frequency(&self, key: &[Option<&str>], value: &DataInput) -> Result<f64, String>
fn query_quantile(&self, key: &[Option<&str>], threshold: f64) -> Result<f64, String>
fn schema(&self) -> &[String]
```

A key is positional and full width: one value per column on `update`, and on a
query `Some(v)` to constrain a column or `None` to leave it free, with at least
one constrained. `query_quantile` dispatches `HydraQuery::Cdf`.

## Merge

```rust
fn merge(&mut self, other: &Hydra) -> Result<(), String>
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error>
```

ASAPv1 MessagePack, one kind_id per counter variant: `0x07 0x00` KLL,
`0x07 0x01` Count-Min, `0x07 0x02` Count Sketch, `0x07 0x03` HyperLogLog,
`0x07 0x04` UnivMon. `deserialize_from_bytes` routes on the kind_id and rejects
any other one.

The metadata carries the hash spec (Hydra hashes its subkeys, so it is present
even for the KLL counter), the grid `rows` / `cols`, the key-column `schema`,
and the counter's own structural params — `counter_rows`, `counter_cols`,
`counter_type` (`"i32"`) and `counter_mode` (`"fast"`) for the two matrix
counters; `counter_precision` for HyperLogLog; `counter_k`, `counter_m` and
`counter_item_type` (`"f64"`) for KLL; `counter_layer_size`,
`counter_sketch_row`, `counter_sketch_col`, `counter_heap_size` and
`counter_key_type` for UnivMon.

The payload is the counters' raw state, row-major over the grid: `[counts]` for
the matrix counters, `[registers]` for HyperLogLog, `[cells]` for KLL and
UnivMon. Counters are inlined, not nested in their own envelopes, and every
cell shares the prototype's geometry, so it is carried once.

The grid and the matrix counters are both hashed per row, so each carries the
same bound: `1 <= rows <= 20` and `1 <= counter_rows <= 20` (`MATRIX_MAX_ROWS`,
the seed list length).

A grid mixing counter variants, UnivMon cells mixing key variants, a cell whose
geometry differs from the prototype's, a `type_to_clone` holding data, a
declared grid the storage does not match, a row count past `MATRIX_MAX_ROWS`,
and a payload length that disagrees with the declared geometry are all rejected
on both sides. `Hydra` also derives
serde, which is the nested codec used when a Hydra is embedded in a larger
serde value.

## Examples

```rust
use asap_sketchlib::{DataInput, Hydra, input::HydraCounter};

let counter = HydraCounter::CM(Default::default());
let mut hydra = Hydra::with_schema(4, 4096, ["region", "service"], counter).unwrap();
hydra.update(&["us", "api"], &DataInput::Str("err"), None).unwrap();
let est = hydra.query_frequency(&[Some("us"), None], &DataInput::Str("err")).unwrap();
assert!(est >= 1.0);
```

## Caveats

- Canonical enum/query/input definitions are in [Common Input Types](./api_common_input.md).
- Query compatibility depends on `HydraCounter` variant.

## Status

Primary subpopulation framework with broad test coverage.
