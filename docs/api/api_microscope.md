# API: MicroCM (MicroscopeSketch)

Status: `Experimental`

> Warning: Behind the `experimental` cargo feature. The accuracy this
> implementation reaches has not been measured against the figures the paper
> reports; what is established is the set of properties its tests assert.

## Purpose

Sliding-window frequency estimation: "how often has this key appeared inside
the last `W` items (or the last `W` time units)?"

A sliding-window counter usually costs one counter per sub-window per cell,
each wide enough for the largest count it might ever hold. MicroscopeSketch
gives each cell `T + 2` byte-wide **pixels** and one shared **zoom** exponent
`Z`, so a pixel is worth `c^Z` items and the cell rescales itself when a pixel
is about to overflow. The counters stay narrow; the exponent buys the range.

## Type/Struct

- `MicroCM<H = DefaultXxHasher>` — a Count-Min-shaped grid of cells.
- `MicroParams { t, c }` — sub-windows per window, and the zoom base.
- `SubWindowClock` — `CountBased` or `TimeBased`.
- `DeltaStrategy` — `Over`, `Under`, `Linear(f64)`.
- `MicroLayout` — the byte offsets a cell's fields sit at.

## Constructors

```rust
fn with_dimensions(
    rows: usize,
    cols: usize,
    params: MicroParams,
    clock: SubWindowClock,
    rounding_seed: u64,
) -> Self

fn MicroParams::new(t: usize, c: u32) -> MicroParams   // Default: t = 12, c = 2
fn SubWindowClock::count_based(items_per_sub_window: u64) -> SubWindowClock
fn SubWindowClock::time_based(sub_window_len: u64, epoch: u64) -> SubWindowClock
```

`cols` must be a power of two and at least 2 — a single column would route
every row to the same cell — and `rows * log2(cols)` must not exceed 128.
`T` must be in `1..=4096` and `c` in `2..=256`; above 256 a single zoom-out
would collapse every byte-wide pixel to 0 or 1. `rounding_seed` seeds the probabilistic rounding a
zoom-out uses; a fixed value makes a run reproducible.

`MicroCM::default()` is a 4 x 1024 grid at `T = 12`, `c = 2`, over
count-based sub-windows of 4096 items.

## Insert/Update

```rust
fn insert(&mut self, key: &DataInput)                      // count-based clock
fn insert_at(&mut self, key: &DataInput, timestamp: u64)   // time-based clock
```

Each panics if given the other's clock, rather than silently not advancing it.

## Query

```rust
fn estimate(&self, key: &DataInput) -> f64                                  // Linear
fn estimate_with(&self, key: &DataInput, strategy: DeltaStrategy) -> f64
```

The window ends part-way through a sub-window, so the oldest sub-window is
partially expired and `DeltaStrategy` says what to charge for it: `Over` all
of it, `Under` none of it, `Linear(p)` the fraction `p` still inside.
`estimate` uses the fraction the clock reports.

The minimum across rows is taken **per sub-window** and those minima summed,
not the other way round: `sum(min) <= min(sum)`, so letting a different row
supply each sub-window is the tighter of the two reductions.

`Over` and `Under` bracket the **span**, and while no cell has zoomed
(`max_zoom() == 0`) the count is exact up to Count-Min inflation, which is
one-sided upward. Past a zoom, a pixel has been through probabilistic
rounding and the error is two-sided, bounded by `c^Z` per sub-window read.

## Accessors

```rust
fn rows(&self) -> usize
fn cols(&self) -> usize
fn params(&self) -> MicroParams
fn cell_bytes(&self) -> usize
fn sub_window(&self) -> u64
fn residual_fraction(&self) -> f64
fn max_zoom(&self) -> u8
fn advance_to(&mut self, timestamp: u64)     // time-based clocks only
fn as_storage(&self) -> &Vector3D<u8>
```

`advance_to` ages a time-based sketch without recording anything. Without it
an idle stream would answer for the window that ended at the last insert.

## Merge

```rust
fn merge(&mut self, other: &Self) -> Result<(), String>
```

The paper does not define a merge. This one brings both cells to
`max(Z_a, Z_b)`, adds pixel-wise, and re-zooms until the sums fit bytes again.
It requires equal grids, equal parameters **and equal sub-window numbers** —
the pixel ring is indexed by sub-window number, so merging across different
clocks would add unrelated sub-windows. `self` is untouched on error.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

Plain MessagePack, **not** an ASAPv1 envelope: no `kind_id` is allocated while
this sketch is experimental. The layout is re-derived from the parameters on
load and a payload whose stored cell depth disagrees with it is refused.

## Cost

One record per `(row, key)` per insert; no per-cell timestamp, because the
sub-window number is sketch-wide. The one whole-table pass is the sub-window
boundary, which clears the slots the new sub-windows land on and offers each
cell a zoom-in — once per sub-window, walking the storage sequentially.

A cell is `T + 2` pixel bytes, one zoom byte and four shutter bytes, rounded
up to a multiple of four: 20 bytes at `T = 12`, 8 bytes at `T = 1`.

## Caveats

- `l = 8`: one pixel is one byte. Packing two 4-bit pixels per byte would
  halve the record and is not implemented.
- Deletion (the decay a HeavyGuardian-style variant needs) is not implemented.
- Expiry happens at sub-window boundaries, for every cell — including the
  shutter, which is rounded into the sub-window that just ended and reset.
- A count-based sketch only ages when something is inserted. A time-based
  one can be aged with `advance_to`.
- A zoom-out rounds probabilistically, as does the partial shutter a
  boundary closes out: each is carried up with probability equal to its
  fractional part, so neither introduces a directional bias.

## See Also

- [ExponentialHistogram](./api_exponential_histogram.md) — sliding windows by
  wrapping whole sketches in buckets, rather than inside the cells.
- [Common Structures](./api_common_structures.md) — `Vector3D`.

## Examples

```rust
use asap_sketchlib::sketches::microscope::{DeltaStrategy, MicroCM, MicroParams, SubWindowClock};
use asap_sketchlib::DataInput;

// A window of 8 sub-windows of 1000 items: the last ~8000 items.
let mut sk: MicroCM = MicroCM::with_dimensions(
    4,
    1024,
    MicroParams::new(8, 2),
    SubWindowClock::count_based(1_000),
    /* rounding seed */ 42,
);

for i in 0..5_000u64 {
    sk.insert(&DataInput::U64(i % 100));
}

// Nothing has expired yet and nothing has zoomed, so the only error left
// is Count-Min's, which is one-sided upward.
assert_eq!(sk.max_zoom(), 0);
assert!(sk.estimate_with(&DataInput::U64(7), DeltaStrategy::Over) >= 50.0);
```

## Status

`Experimental`. Behind the `experimental` cargo feature; the API and the wire
form may change without a major version bump, and no ASAPv1 `kind_id` is
allocated for it. The accuracy it reaches has not been measured against the
figures the paper reports.
