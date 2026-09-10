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
//! [`cell`](crate::sketches::microscope::cell) holds the record layout and the per-cell algorithm, on `&[u8]`
//! and nothing else. [`MicroCM`] puts a Count-Min-shaped grid of those cells
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
pub mod cell;

pub use cell::{DeltaStrategy, MicroLayout, MicroParams, Rounding};

use crate::{DataInput, DefaultXxHasher, SketchHasher, Vector3D};
use rmp_serde::{
    decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice, to_vec_named,
};
use serde::{Deserialize, Serialize};
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
        if params.t == 0 {
            return Err(serde::de::Error::custom("t must be non-zero"));
        }
        if params.c < 2 {
            return Err(serde::de::Error::custom(format!(
                "zoom base c ({}) must be at least 2",
                params.c
            )));
        }
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
        if !cols.is_power_of_two() {
            return Err(serde::de::Error::custom(format!(
                "cols ({cols}) must be a power of two"
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
        assert!(rows > 0 && cols > 0, "rows and cols must be non-zero");
        assert!(
            cols.is_power_of_two(),
            "cols must be a power of two, got {cols}"
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

    /// The current sub-window number.
    pub fn sub_window(&self) -> u64 {
        self.clock.n()
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
        for record in self.cells.as_mut_slice().chunks_exact_mut(layout.depth()) {
            cell::enter_sub_window(record, &layout, n, crossed);
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
        let crossed = self.clock.advance_to(timestamp);
        if crossed > 0 {
            self.enter_sub_window(crossed);
        }
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
    /// [`DeltaStrategy::Over`] never under-reports and
    /// [`DeltaStrategy::Under`] never over-reports *the window*; both are
    /// still subject to the Count-Min inflation that other keys sharing a
    /// cell cause, which the minimum across rows reduces but cannot remove.
    pub fn estimate_with(&self, key: &DataInput, strategy: DeltaStrategy) -> f64 {
        let packed = H::hash128_seeded(0, key);
        let n = self.clock.n();
        let layout = self.layout;
        self.cells.fast_query_min(&packed, |record, _row, _hash| {
            cell::estimate(record, &layout, n, strategy)
        })
    }

    /// Merges `other` into `self`, cell by cell.
    ///
    /// Both sketches must have the same grid, the same cell parameters, and
    /// the same sub-window number. The last is not a formality: the pixel
    /// ring is indexed by sub-window number, so merging two sketches whose
    /// clocks disagree would add unrelated sub-windows together and produce
    /// a sketch whose estimates mean nothing. The paper does not define a
    /// merge; see [`cell::merge_cells`] for what this one preserves.
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
        let rounding = self.rounding.clone();
        for (dst, src) in self
            .cells
            .as_mut_slice()
            .chunks_exact_mut(layout.depth())
            .zip(other.cells.as_slice().chunks_exact(layout.depth()))
        {
            cell::merge_cells(dst, src, &layout, n, &rounding);
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

    /// A merge of two halves recovers what a single pass over both would
    /// have counted, up to the rounding a zoom introduces.
    #[test]
    fn merge_recovers_the_combined_count() {
        let mut left = sketch(2, 32, 4, 400);
        let mut right = sketch(2, 32, 4, 400);
        let mut single = sketch(2, 32, 4, 400);
        // Both halves must end at the same sub-window, so feed each the same
        // number of items and pad the single sketch to match by feeding it
        // both halves' items.
        for i in 0..800u64 {
            left.insert(&key(if i % 2 == 0 { 3 } else { 4 }));
            right.insert(&key(if i % 2 == 0 { 3 } else { 5 }));
        }
        for i in 0..800u64 {
            single.insert(&key(if i % 2 == 0 { 3 } else { 4 }));
            single.insert(&key(if i % 2 == 0 { 3 } else { 5 }));
        }
        assert_eq!(left.sub_window(), right.sub_window());
        left.merge(&right).expect("same shape and clock");
        // Key 3 was inserted 400 times into each half.
        let merged = left.estimate_with(&key(3), DeltaStrategy::Over);
        assert!(
            (merged - 800.0).abs() <= 800.0 * 0.05,
            "merged estimate {merged} is not within 5% of 800"
        );
        // And the merge is one-sided in the same direction as the sketch.
        assert!(merged >= 700.0, "merge lost too much: {merged}");
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
