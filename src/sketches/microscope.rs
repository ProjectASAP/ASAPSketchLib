//! MicroscopeSketch: sliding-window frequency estimation with adaptive zoom.
//!
//! A sliding-window counter normally pays for its window in memory: keep one
//! counter per sub-window, and a window of `T` sub-windows costs `T` counters
//! per cell, each wide enough for the largest count it might ever hold. Most
//! of those bits are idle most of the time.
//!
//! MicroscopeSketch instead gives each cell `T + 2` narrow **pixels** and one
//! shared **zoom** exponent `Z`, so a pixel is worth `c^Z` items and the cell
//! rescales itself when a pixel is about to overflow. The counters stay
//! narrow; the exponent buys the range.
//!
//! Where this implementation departs from the authors' released code: a
//! zoom-out rounds the other pixels **probabilistically**, which is the
//! method the paper describes, whereas the released code always rounds up
//! (its probabilistic branch is commented out). The same holds for the
//! partial shutter a sub-window boundary closes out.
//!
//! The `cell` module holds the record layout and the per-cell algorithm,
//! on `&[u8]` and nothing else. [`MicroCM`] puts a Count-Min-shaped grid of those cells
//! behind a hash: `rows` independent rows, one cell per row per key, and the
//! minimum across rows as the answer.
//!
//! # Why the fields share a record
//!
//! `Z` is shared by every pixel of a cell: a pixel value means nothing
//! without it, and a zoom rescales all of them together. An insert reads the
//! shutter, may roll it into a pixel, and may rescale every pixel — one
//! cell's worth of coupled state per item. That is what
//! [`Vector3D`](crate::Vector3D) is for, and why `T + 2` separate
//! [`Vector2D`](crate::Vector2D)s would not be the same structure: no
//! arrangement of independent matrices lets one exponent rescale a cell.
//!
//! The sub-window number is sketch-wide, not per cell, so an insert needs no
//! per-cell timestamp and touches only the one record it hashes to. The one
//! whole-table pass is the zoom-in sweep at a sub-window boundary, which runs
//! once per sub-window and walks the storage sequentially.
//!
//! # Status
//!
//! Experimental: behind the `experimental` cargo feature. The algorithm
//! follows Zhao et al., but this implementation has not been checked against
//! the paper's published measurements, so the accuracy it achieves is
//! corroborated only by the property tests in this repository.
//!
//! # References
//!
//! - Zhao, Wang, Li, Dong, Yang, Chen, Zhang, Uhlig, "MicroscopeSketch:
//!   Accurate Sliding Estimation Using Adaptive Zooming," KDD 2023.

/// The per-cell record layout and algorithm.
///
/// Crate-private, like every other sketch's sub-module here: the record
/// operations are an implementation detail, and the types callers need are
/// re-exported below.
pub(crate) mod cell;

pub use cell::{DeltaStrategy, MAX_SUB_WINDOWS, MicroLayout, MicroParams, Rounding};

use crate::{DataInput, DefaultXxHasher, MatrixFastHash, SketchHasher, Vector3D};
use rmp_serde::{
    decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice, to_vec_named,
};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::marker::PhantomData;

/// What advances the sub-window number, and how far into the current
/// sub-window the stream currently sits.
///
/// The paper's count-based and time-based windows differ in exactly two
/// places — when `n` advances, and how the partially expired oldest
/// sub-window is weighted — so they are one type here rather than two
/// sketches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubWindowClock {
    /// A sub-window is a fixed number of items.
    CountBased {
        /// Items per sub-window. The window spans `T` of these.
        items_per_sub_window: u64,
        /// Items recorded so far.
        seen: u64,
    },
    /// A sub-window is a fixed span of time, in whatever unit the caller's
    /// timestamps use.
    TimeBased {
        /// Length of one sub-window.
        sub_window_len: u64,
        /// Timestamp the first sub-window began at.
        epoch: u64,
        /// Most recent timestamp observed.
        now: u64,
    },
}

impl SubWindowClock {
    /// A count-based clock of `items_per_sub_window` items, at the start.
    pub fn count_based(items_per_sub_window: u64) -> Self {
        assert!(
            items_per_sub_window > 0,
            "items_per_sub_window must be non-zero"
        );
        Self::CountBased {
            items_per_sub_window,
            seen: 0,
        }
    }

    /// A time-based clock of `sub_window_len` units, starting at `epoch`.
    pub fn time_based(sub_window_len: u64, epoch: u64) -> Self {
        assert!(sub_window_len > 0, "sub_window_len must be non-zero");
        Self::TimeBased {
            sub_window_len,
            epoch,
            now: epoch,
        }
    }

    /// The current sub-window number.
    #[inline(always)]
    pub fn n(&self) -> u64 {
        match *self {
            Self::CountBased {
                items_per_sub_window,
                seen,
            } => seen / items_per_sub_window,
            Self::TimeBased {
                sub_window_len,
                epoch,
                now,
            } => now.saturating_sub(epoch) / sub_window_len,
        }
    }

    /// How far into the current sub-window the stream sits, in `[0, 1)`.
    #[inline(always)]
    fn progress(&self) -> f64 {
        match *self {
            Self::CountBased {
                items_per_sub_window,
                seen,
            } => (seen % items_per_sub_window) as f64 / items_per_sub_window as f64,
            Self::TimeBased {
                sub_window_len,
                epoch,
                now,
            } => (now.saturating_sub(epoch) % sub_window_len) as f64 / sub_window_len as f64,
        }
    }

    /// The fraction of the oldest sub-window still inside the window, which
    /// is what [`DeltaStrategy::Linear`] charges for it.
    ///
    /// Just after a boundary the whole of sub-window `n-T` is still in the
    /// window, so this is 1; it falls to 0 as the current sub-window fills.
    #[inline(always)]
    pub fn residual_fraction(&self) -> f64 {
        1.0 - self.progress()
    }

    /// Records one item against a count-based clock. Returns `true` when
    /// that crossed into a new sub-window.
    fn tick(&mut self) -> bool {
        match self {
            Self::CountBased {
                items_per_sub_window,
                seen,
            } => {
                let before = *seen / *items_per_sub_window;
                *seen += 1;
                *seen / *items_per_sub_window != before
            }
            Self::TimeBased { .. } => {
                panic!("insert() drives a count-based clock; use insert_at() for a time-based one")
            }
        }
    }

    /// Moves a time-based clock to `timestamp`. Returns how many sub-window
    /// boundaries that crossed.
    fn advance_to(&mut self, timestamp: u64) -> u64 {
        match self {
            Self::TimeBased {
                sub_window_len,
                epoch,
                now,
            } => {
                let before = now.saturating_sub(*epoch) / *sub_window_len;
                // Time does not run backwards; a stale timestamp is pinned to
                // the present rather than rewinding the window.
                *now = (*now).max(timestamp);
                now.saturating_sub(*epoch) / *sub_window_len - before
            }
            Self::CountBased { .. } => {
                panic!("insert_at() drives a time-based clock; use insert() for a count-based one")
            }
        }
    }
}

/// A Count-Min-shaped grid of MicroscopeSketch cells.
///
/// Answers "how often has this key appeared inside the sliding window?" Each
/// of `rows` rows hashes a key to one cell; the estimate is the minimum
/// across rows, because a cell can only be inflated by the other keys that
/// share it.
///
/// See the [module docs](crate::sketches::microscope) for the layout rationale and the current
/// status of this implementation.
#[derive(Clone, Debug, Serialize)]
#[serde(bound = "")]
pub struct MicroCM<H: SketchHasher = DefaultXxHasher> {
    cells: Vector3D<u8>,
    params: MicroParams,
    clock: SubWindowClock,
    rounding: Rounding,
    #[serde(skip)]
    layout: MicroLayout,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

// Only the authoritative fields are read from the wire; the layout is
// re-derived from `params` and then checked against the stored geometry, so
// a payload cannot describe a record shape the cells do not have.
#[derive(Deserialize)]
struct MicroCMSeed {
    cells: Vector3D<u8>,
    params: MicroParams,
    clock: SubWindowClock,
    rounding: Rounding,
}

impl<'de, H: SketchHasher> Deserialize<'de> for MicroCM<H> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let MicroCMSeed {
            cells,
            params,
            clock,
            rounding,
        } = MicroCMSeed::deserialize(deserializer)?;
        // Validate before deriving the layout: `MicroLayout::new` computes
        // `t + 2`, which a crafted `t` could overflow.
        MicroParams::checked(params.t, params.c).map_err(serde::de::Error::custom)?;
        let layout = MicroLayout::new(params);
        if cells.depth() != layout.depth() {
            return Err(serde::de::Error::custom(format!(
                "cell depth {} does not match the {} bytes the layout for t={}, c={} needs",
                cells.depth(),
                layout.depth(),
                params.t,
                params.c
            )));
        }
        let cols = cells.cols();
        if cols < 2 || !cols.is_power_of_two() {
            return Err(serde::de::Error::custom(format!(
                "cols ({cols}) must be a power of two and at least 2"
            )));
        }
        let rows = cells.rows();
        let col_bits = cells.get_mask_bits() as usize;
        let required_bits = rows.saturating_mul(col_bits);
        if required_bits > 128 {
            return Err(serde::de::Error::custom(format!(
                "rows ({rows}) x column bits ({col_bits}) = {required_bits} exceeds the \
                 128-bit packed column hash; reduce rows or cols"
            )));
        }
        match clock {
            SubWindowClock::CountBased {
                items_per_sub_window: 0,
                ..
            } => {
                return Err(serde::de::Error::custom(
                    "items_per_sub_window must be non-zero",
                ));
            }
            SubWindowClock::TimeBased {
                sub_window_len: 0, ..
            } => {
                return Err(serde::de::Error::custom("sub_window_len must be non-zero"));
            }
            _ => {}
        }
        // Each record carries a zoom exponent and a shutter that the layout
        // cannot constrain; a crafted pair panics or wraps on first use.
        for (index, record) in cells.as_slice().chunks_exact(layout.depth()).enumerate() {
            cell::validate(record, &layout)
                .map_err(|detail| serde::de::Error::custom(format!("cell {index}: {detail}")))?;
        }
        Ok(Self {
            cells,
            params,
            clock,
            rounding,
            layout,
            _hasher: PhantomData,
        })
    }
}

impl Default for MicroCM<DefaultXxHasher> {
    /// A 4 x 1024 grid at `T = 12`, `c = 2`, over count-based sub-windows of
    /// 4096 items — a sliding window of roughly 49k items.
    fn default() -> Self {
        Self::with_dimensions(
            4,
            1024,
            MicroParams::default(),
            SubWindowClock::count_based(4096),
            0,
        )
    }
}

impl<H: SketchHasher> MicroCM<H> {
    /// Builds a `rows x cols` grid of cells shaped by `params`, driven by
    /// `clock`.
    ///
    /// `rounding_seed` seeds the probabilistic rounding a zoom-out uses;
    /// passing a fixed value makes a run reproducible.
    ///
    /// Panics if a dimension is zero, if `cols` is not a power of two, or if
    /// the per-row column bits do not fit the 128-bit packed hash.
    pub fn with_dimensions(
        rows: usize,
        cols: usize,
        params: MicroParams,
        clock: SubWindowClock,
        rounding_seed: u64,
    ) -> Self {
        assert!(rows > 0, "rows must be non-zero");
        assert!(
            cols >= 2 && cols.is_power_of_two(),
            "cols must be a power of two and at least 2, got {cols}; a single \
             column would route every row to the same cell and collapse the \
             minimum across rows to one counter"
        );
        let layout = MicroLayout::new(params);
        let mut cells = Vector3D::init(rows, cols, layout.depth());
        cells.fill(0);
        let col_bits = cells.get_mask_bits() as usize;
        assert!(
            rows.saturating_mul(col_bits) <= 128,
            "rows ({rows}) x column bits ({col_bits}) = {} exceeds the 128-bit packed \
             column hash; reduce rows or cols",
            rows * col_bits
        );
        Self {
            cells,
            params,
            clock,
            rounding: Rounding::new(rounding_seed),
            layout,
            _hasher: PhantomData,
        }
    }

    /// Number of hash rows.
    pub fn rows(&self) -> usize {
        self.cells.rows()
    }

    /// Number of columns per row.
    pub fn cols(&self) -> usize {
        self.cells.cols()
    }

    /// The cell parameters.
    pub fn params(&self) -> MicroParams {
        self.params
    }

    /// Bytes per cell.
    pub fn cell_bytes(&self) -> usize {
        self.layout.depth()
    }

    /// The highest zoom exponent any cell has reached.
    ///
    /// A sketch still at zero has counted every item individually and its
    /// answers carry no rounding; past zero, each pixel has been through a
    /// probabilistic division and the estimate is accurate to within `c^Z`
    /// per sub-window read. Useful for deciding whether a configuration has
    /// enough pixels for its load.
    pub fn max_zoom(&self) -> u8 {
        let layout = self.layout;
        self.cells
            .as_slice()
            .chunks_exact(layout.depth())
            .map(|record| cell::zoom(record, &layout))
            .max()
            .unwrap_or(0)
    }

    /// The current sub-window number.
    pub fn sub_window(&self) -> u64 {
        self.clock.n()
    }

    /// The fraction of the oldest sub-window still inside the window, which
    /// is the weight [`Self::estimate`] gives it.
    pub fn residual_fraction(&self) -> f64 {
        self.clock.residual_fraction()
    }

    /// Moves a time-based clock to `timestamp` without recording anything.
    ///
    /// Without this a time-based sketch only ages when something is inserted,
    /// so a query made long after the last insert would answer for the window
    /// that ended then. Call it before querying an idle stream.
    ///
    /// Panics if this sketch's clock is count-based, where the window is
    /// defined by item count and there is nothing to advance.
    pub fn advance_to(&mut self, timestamp: u64) {
        let crossed = self.clock.advance_to(timestamp);
        if crossed > 0 {
            self.enter_sub_window(crossed);
        }
    }

    /// Exposes the backing storage for inspection and testing.
    pub fn as_storage(&self) -> &Vector3D<u8> {
        &self.cells
    }

    /// Moves every cell into the sub-window the clock has reached.
    ///
    /// This is the sketch's only whole-table pass: once per sub-window
    /// boundary, never per item, walking the storage in order. It does the
    /// two things a boundary owes every cell — clear the slots the new
    /// sub-windows land on, and offer a zoom-in — including for the cells
    /// whose keys have gone quiet, which are precisely the ones that would
    /// otherwise keep reporting a stale count.
    fn enter_sub_window(&mut self, crossed: u64) {
        let layout = self.layout;
        let n = self.clock.n();
        // Borrowed as a shared reference so the rounding state advances in
        // place; cloning it here would replay the same draws every boundary.
        let rounding = &self.rounding;
        for record in self.cells.as_mut_slice().chunks_exact_mut(layout.depth()) {
            cell::enter_sub_window(record, &layout, n, crossed, rounding);
        }
    }

    /// Records one occurrence of `key` against a count-based clock.
    ///
    /// Panics if this sketch's clock is time-based; use [`Self::insert_at`].
    pub fn insert(&mut self, key: &DataInput) {
        if self.clock.tick() {
            self.enter_sub_window(1);
        }
        self.record(key);
    }

    /// Records one occurrence of `key` at `timestamp`, against a time-based
    /// clock.
    ///
    /// Panics if this sketch's clock is count-based; use [`Self::insert`].
    pub fn insert_at(&mut self, key: &DataInput, timestamp: u64) {
        self.advance_to(timestamp);
        self.record(key);
    }

    /// Writes one item into the cell each row selects for `key`.
    fn record(&mut self, key: &DataInput) {
        let packed = H::hash128_seeded(0, key);
        let n = self.clock.n();
        let layout = self.layout;
        // Borrowed as a shared reference so the closure stays `Fn`, which is
        // what `fast_insert` takes; the rounding state advances through a
        // `Cell` rather than through `&mut`.
        let rounding = &self.rounding;
        self.cells.fast_insert(
            |record, _: &(), _row| cell::insert(record, &layout, n, rounding),
            (),
            &packed,
        );
    }

    /// Estimates how often `key` occurred inside the sliding window,
    /// weighting the partially expired oldest sub-window by how much of it is
    /// still inside.
    pub fn estimate(&self, key: &DataInput) -> f64 {
        self.estimate_with(key, DeltaStrategy::Linear(self.clock.residual_fraction()))
    }

    /// Estimates with an explicit policy for the oldest sub-window.
    ///
    /// The minimum across rows is taken **per sub-window**, and those minima
    /// are then summed — not the other way round. Both are valid Count-Min
    /// reductions, but `sum(min) <= min(sum)`: taking the minimum first lets
    /// a different row supply each sub-window, so a collision that inflates
    /// one row's sub-window is discarded even when that row is the cleanest
    /// overall. Reducing whole-cell estimates instead would keep it.
    ///
    /// While the cell has not zoomed the answer is exact up to that
    /// Count-Min inflation, which is one-sided upward. Once `Z > 0` the
    /// probabilistic rounding of a zoom makes the per-pixel error two-sided,
    /// bounded by `c^Z` per pixel — so `Over` and `Under` bracket the window
    /// only up to that rounding, not absolutely.
    pub fn estimate_with(&self, key: &DataInput, strategy: DeltaStrategy) -> f64 {
        let packed = H::hash128_seeded(0, key);
        let n = self.clock.n();
        let layout = self.layout;
        let t = self.params.t as u64;
        let cols = self.cells.cols();
        let columns: SmallVec<[usize; 8]> = (0..self.cells.rows())
            .map(|row| MatrixFastHash::col_for_row(&packed, row, cols))
            .collect();
        let record = |row: usize| self.cells.bucket_slice(row, columns[row]);
        let across_rows = |f: &dyn Fn(&[u8]) -> f64| {
            (0..columns.len())
                .map(|row| f(record(row)))
                .fold(f64::INFINITY, f64::min)
        };

        // The current sub-window carries the shutter, so the two are reduced
        // together: they are one row's view of the same span.
        let mut total = across_rows(&|rec| {
            cell::shutter(rec, &layout) as f64 + cell::scaled_pixel(rec, &layout, n)
        });
        // The sub-windows wholly inside the window.
        for back in 1..t {
            let Some(sub_window) = n.checked_sub(back) else {
                break;
            };
            total += across_rows(&|rec| cell::scaled_pixel(rec, &layout, sub_window));
        }
        // The partially expired oldest one, weighted by the policy.
        if let Some(sub_window) = n.checked_sub(t) {
            let oldest = across_rows(&|rec| cell::scaled_pixel(rec, &layout, sub_window));
            total += match strategy {
                DeltaStrategy::Over => oldest,
                DeltaStrategy::Under => 0.0,
                DeltaStrategy::Linear(fraction) => fraction.clamp(0.0, 1.0) * oldest,
            };
        }
        total
    }

    /// Merges `other` into `self`, cell by cell.
    ///
    /// Both sketches must have the same grid, the same cell parameters, and
    /// the same sub-window number. The last is not a formality: the pixel
    /// ring is indexed by sub-window number, so merging two sketches whose
    /// clocks disagree would add unrelated sub-windows together and produce
    /// a sketch whose estimates mean nothing. The paper does not define a
    /// merge; the cell-level `merge_cells` documents what this one preserves.
    ///
    /// `self` is left untouched when the two do not line up.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        let mine = (self.cells.rows(), self.cells.cols(), self.params);
        let theirs = (other.cells.rows(), other.cells.cols(), other.params);
        if mine != theirs {
            return Err(format!(
                "cannot merge sketches of different shape: \
                 (rows, cols, params) is {mine:?} against {theirs:?}"
            ));
        }
        if self.clock.n() != other.clock.n() {
            return Err(format!(
                "cannot merge sketches at different sub-windows: {} against {}",
                self.clock.n(),
                other.clock.n()
            ));
        }
        let layout = self.layout;
        let n = self.clock.n();
        // A shared borrow, not a clone: `Rounding` keeps its state in a
        // `Cell`, so a clone would take every draw on a copy and leave the
        // sketch's own stream untouched — successive merges would then
        // replay identical rounding instead of being independent.
        let rounding = &self.rounding;
        for (dst, src) in self
            .cells
            .as_mut_slice()
            .chunks_exact_mut(layout.depth())
            .zip(other.cells.as_slice().chunks_exact(layout.depth()))
        {
            cell::merge_cells(dst, src, &layout, n, rounding);
        }
        Ok(())
    }

    /// Serializes the sketch into MessagePack bytes.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        to_vec_named(self)
    }

    /// Deserializes a sketch from MessagePack bytes.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> DataInput<'static> {
        DataInput::U64(n)
    }

    fn sketch(rows: usize, cols: usize, t: usize, per_sub_window: u64) -> MicroCM {
        MicroCM::with_dimensions(
            rows,
            cols,
            MicroParams::new(t, 2),
            SubWindowClock::count_based(per_sub_window),
            0xABCD,
        )
    }

    #[test]
    fn cell_depth_follows_the_parameters() {
        assert_eq!(sketch(2, 16, 12, 100).cell_bytes(), 20);
        assert_eq!(sketch(2, 16, 1, 100).cell_bytes(), 8);
    }

    #[test]
    #[should_panic(expected = "cols must be a power of two")]
    fn rejects_non_power_of_two_cols() {
        sketch(2, 17, 4, 100);
    }

    /// The clock is what makes `n` advance, and it advances exactly on the
    /// item that fills a sub-window.
    #[test]
    fn the_sub_window_advances_on_the_item_that_fills_it() {
        let mut sk = sketch(2, 16, 4, 10);
        assert_eq!(sk.sub_window(), 0);
        for _ in 0..9 {
            sk.insert(&key(1));
        }
        assert_eq!(sk.sub_window(), 0, "nine of ten items is still window 0");
        sk.insert(&key(1));
        assert_eq!(sk.sub_window(), 1);
    }

    /// A count-based and a time-based clock fed equivalent inputs walk the
    /// same sequence of sub-window numbers.
    #[test]
    fn count_based_and_time_based_clocks_agree_on_the_sub_window_sequence() {
        let mut counted = SubWindowClock::count_based(10);
        let mut timed = SubWindowClock::time_based(10, 0);
        for i in 0..95u64 {
            counted.tick();
            timed.advance_to(i + 1);
            assert_eq!(
                counted.n(),
                timed.n(),
                "clocks diverged after {} items",
                i + 1
            );
        }
    }

    /// A single key, with rows = 1, is just the cell layer, so the grid must
    /// reproduce the exact count while nothing has zoomed.
    #[test]
    fn a_single_key_in_one_row_is_counted_exactly() {
        let mut sk = sketch(1, 16, 4, 50);
        for _ in 0..200 {
            sk.insert(&key(7));
        }
        let est = sk.estimate_with(&key(7), DeltaStrategy::Over);
        assert_eq!(est, 200.0);
    }

    /// Count-Min inflation is one-sided, so no key is ever under-reported
    /// relative to what its own cell holds.
    #[test]
    fn no_key_is_under_reported_on_a_contended_grid() {
        let mut sk = sketch(3, 16, 4, 500);
        let mut truth = [0u64; 40];
        for round in 0..2_000u64 {
            let k = round % 40;
            sk.insert(&key(k));
            truth[k as usize] += 1;
        }
        // Everything inserted is still inside a window of 4 x 500 = 2000.
        for (k, &count) in truth.iter().enumerate() {
            let est = sk.estimate_with(&key(k as u64), DeltaStrategy::Over);
            assert!(
                est >= count as f64,
                "key {k}: estimate {est} under-reported a true count of {count}"
            );
        }
    }

    /// Keys that stop appearing fall out of the window.
    #[test]
    fn a_key_that_stops_appearing_decays_to_nothing() {
        let mut sk = sketch(3, 64, 4, 100);
        for _ in 0..400 {
            sk.insert(&key(1));
        }
        assert!(sk.estimate_with(&key(1), DeltaStrategy::Under) > 0.0);
        // Push the window past the burst with a different key.
        for _ in 0..600 {
            sk.insert(&key(2));
        }
        let est = sk.estimate_with(&key(1), DeltaStrategy::Over);
        assert!(
            est < 40.0,
            "key 1 should have aged out of the window, got {est}"
        );
    }

    /// A time-based clock that jumps several sub-windows expires everything
    /// it skipped over.
    #[test]
    fn a_time_jump_expires_the_sub_windows_it_skipped() {
        let mut sk: MicroCM = MicroCM::with_dimensions(
            2,
            32,
            MicroParams::new(3, 2),
            SubWindowClock::time_based(100, 0),
            1,
        );
        for i in 0..50u64 {
            sk.insert_at(&key(9), i);
        }
        assert!(sk.estimate_with(&key(9), DeltaStrategy::Over) > 0.0);
        // Jump well past T + 1 sub-windows without inserting key 9 again.
        sk.insert_at(&key(10), 100_000);
        assert_eq!(sk.estimate_with(&key(9), DeltaStrategy::Over), 0.0);
    }

    #[test]
    #[should_panic(expected = "use insert_at()")]
    fn insert_refuses_a_time_based_clock() {
        let mut sk: MicroCM = MicroCM::with_dimensions(
            1,
            16,
            MicroParams::default(),
            SubWindowClock::time_based(10, 0),
            1,
        );
        sk.insert(&key(1));
    }

    #[test]
    #[should_panic(expected = "use insert()")]
    fn insert_at_refuses_a_count_based_clock() {
        let mut sk = sketch(1, 16, 4, 10);
        sk.insert_at(&key(1), 5);
    }

    /// A merge of two halves recovers exactly what a single pass over both
    /// would have counted.
    ///
    /// The load is sized so no cell ever zooms — on either side or after the
    /// merge — which makes the arithmetic exact and the assertion an
    /// equality rather than a band. `merge_advances_the_rounding_stream`
    /// covers the zoomed path.
    #[test]
    fn merge_recovers_the_combined_count_exactly_below_a_zoom() {
        const PER_SUB_WINDOW: u64 = 200;
        let mut left = sketch(2, 32, 4, PER_SUB_WINDOW);
        let mut right = sketch(2, 32, 4, PER_SUB_WINDOW);
        // Key 3 takes every other item on both sides: 100 per sub-window
        // each, so the merged pixel is 200 and still fits a byte.
        for i in 0..4 * PER_SUB_WINDOW {
            left.insert(&key(if i % 2 == 0 { 3 } else { 4 }));
            right.insert(&key(if i % 2 == 0 { 3 } else { 5 }));
        }
        assert_eq!(left.sub_window(), right.sub_window());
        assert_eq!(left.max_zoom(), 0, "the load must stay below a zoom");
        assert_eq!(right.max_zoom(), 0, "the load must stay below a zoom");

        left.merge(&right).expect("same shape and clock");
        assert_eq!(left.max_zoom(), 0, "the merge must not have zoomed");
        assert_eq!(
            left.estimate_with(&key(3), DeltaStrategy::Over),
            800.0,
            "400 inserts on each side, none expired, no rounding anywhere"
        );
    }

    /// A merge takes its rounding draws from the sketch's own stream, so two
    /// successive merges do not replay identical rounding.

    #[test]
    fn merge_advances_the_rounding_stream() {
        let mut a = sketch(2, 32, 4, 4_000);
        let mut b = sketch(2, 32, 4, 4_000);
        // The two sides must disagree about the zoom level of a cell that is
        // non-empty on both, since bringing them to a common Z is what makes
        // the merge round and rounding is what takes the draws — an all-zero
        // pixel divides exactly and never draws. Equal item counts keep the
        // two clocks in step.
        for i in 0..40_000u64 {
            a.insert(&key(1));
            b.insert(&key(if i % 4 == 0 { 9 } else { 1 }));
        }
        let before = a.rounding.state();
        a.merge(&b).expect("same shape and clock");
        assert_ne!(
            a.rounding.state(),
            before,
            "the merge consumed rounding draws but left the sketch's stream \
             untouched, so every merge would replay the same rounding"
        );
    }

    #[test]
    fn merge_refuses_a_clock_mismatch_and_leaves_the_target_alone() {
        let mut a = sketch(2, 32, 4, 10);
        let mut b = sketch(2, 32, 4, 10);
        for _ in 0..25 {
            a.insert(&key(1));
        }
        for _ in 0..5 {
            b.insert(&key(1));
        }
        assert_ne!(a.sub_window(), b.sub_window());
        let before = a.as_storage().as_slice().to_vec();
        let err = a.merge(&b).expect_err("clocks disagree");
        assert!(err.contains("different sub-windows"), "unexpected: {err}");
        assert_eq!(a.as_storage().as_slice(), before.as_slice());
    }

    #[test]
    fn serde_round_trip_preserves_the_answers() {
        let mut sk = sketch(3, 32, 4, 100);
        for i in 0..1_000u64 {
            sk.insert(&key(i % 17));
        }
        let bytes = sk.serialize_to_bytes().expect("serialize");
        let back = MicroCM::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(back.sub_window(), sk.sub_window());
        assert_eq!(back.cell_bytes(), sk.cell_bytes());
        assert_eq!(back.as_storage().as_slice(), sk.as_storage().as_slice());
        for k in 0..17u64 {
            assert_eq!(
                back.estimate_with(&key(k), DeltaStrategy::Over),
                sk.estimate_with(&key(k), DeltaStrategy::Over)
            );
        }
    }

    /// A payload whose parameters imply a different record size than the
    /// stored cells have must be refused, not decoded into a sketch that
    /// reads fields out of the wrong bytes.
    #[test]
    fn deserialize_rejects_params_that_contradict_the_cell_depth() {
        #[derive(Serialize)]
        struct Forged {
            cells: Vector3D<u8>,
            params: MicroParams,
            clock: SubWindowClock,
            rounding: u64,
        }
        let sk = sketch(2, 16, 4, 100);
        // t=4 gives 6 pixels + 1 + 4 = 11 bytes, padded to 12; t=9 wants 16.
        let forged = Forged {
            cells: sk.as_storage().clone(),
            params: MicroParams { t: 9, c: 2 },
            clock: SubWindowClock::count_based(100),
            rounding: 1,
        };
        let bytes = rmp_serde::to_vec_named(&forged).expect("serialize");
        let err = MicroCM::<DefaultXxHasher>::deserialize_from_bytes(&bytes)
            .expect_err("a depth the layout disagrees with must be refused");
        assert!(
            err.to_string().contains("does not match"),
            "unexpected error: {err}"
        );
    }

    /// A payload whose column count is not a power of two is refused, the
    /// same way the constructor refuses one.
    #[test]
    fn deserialize_rejects_non_power_of_two_cols() {
        #[derive(Serialize)]
        struct Forged {
            cells: Vector3D<u8>,
            params: MicroParams,
            clock: SubWindowClock,
            rounding: u64,
        }
        let params = MicroParams::new(4, 2);
        let depth = MicroLayout::new(params).depth();
        let mut cells: Vector3D<u8> = Vector3D::init(2, 17, depth);
        cells.fill(0);
        let forged = Forged {
            cells,
            params,
            clock: SubWindowClock::count_based(100),
            rounding: 1,
        };
        let bytes = rmp_serde::to_vec_named(&forged).expect("serialize");
        let err = MicroCM::<DefaultXxHasher>::deserialize_from_bytes(&bytes)
            .expect_err("a non-power-of-two width must be refused");
        assert!(
            err.to_string().contains("power of two"),
            "unexpected error: {err}"
        );
    }
}
