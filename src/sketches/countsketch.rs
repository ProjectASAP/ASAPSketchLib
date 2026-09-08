//! Count Sketch implementation.
//!
//! A frequency-estimation sketch that uses random sign projections to estimate
//! item counts with bounded error in the L2 norm.
//!
//! Reference:
//! - Charikar, Chen & Farach-Colton, "Finding Frequent Items in Data Streams,"
//!   ICALP 2002. <https://www.cs.rutgers.edu/~farach/pubs/FrequentStream.pdf>

use crate::common::structure_utils::AdmittedRows;
use crate::{
    DataInput, DefaultMatrixI32, DefaultMatrixI64, DefaultMatrixI128, DefaultXxHasher, FastPath,
    FastPathHasher, FixedMatrix, MatrixFastHash, MatrixStorage, NitroTarget, QuickMatrixI64,
    QuickMatrixI128, RegularPath, SketchHasher, Vector2D, hash64_seeded, nitro_delta_saturated_i32,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Neg;

mod wire;
pub(crate) use wire::{CsWireCounter, CsWireMode};

const DEFAULT_ROW_NUM: usize = 3;
const DEFAULT_COL_NUM: usize = 4096;
const LOWER_32_MASK: u64 = (1u64 << 32) - 1;

/// A frequency-estimation sketch using random sign projections (Count Sketch).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: Deserialize<'de>"))]
pub struct Count<
    S: MatrixStorage = Vector2D<i32>,
    Mode = RegularPath,
    H: SketchHasher = DefaultXxHasher,
> {
    counts: S,
    row: usize,
    col: usize,
    #[serde(skip)]
    _mode: PhantomData<Mode>,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

/// Counter trait required by Count Sketch backends.
pub trait CountSketchCounter: Copy + std::ops::AddAssign + Neg<Output = Self> + From<i32> {
    /// Converts the counter into `f64`.
    fn to_f64(self) -> f64;
}

// Implements CountSketchCounter for i32.
impl CountSketchCounter for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// Implements CountSketchCounter for i64.
impl CountSketchCounter for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// Implements CountSketchCounter for i128.
impl CountSketchCounter for i128 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// Default Count sketch for Vector2D<i32> (RegularPath).
impl Default for Count<Vector2D<i32>, RegularPath> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM)
    }
}

// Default Count sketch for Vector2D<i32> (FastPath).
impl Default for Count<Vector2D<i32>, FastPath> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM)
    }
}

// Default Count sketch for Vector2D<i64> (RegularPath).
impl Default for Count<Vector2D<i64>, RegularPath> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM)
    }
}

// Default Count sketch for Vector2D<i64> (FastPath).
impl Default for Count<Vector2D<i64>, FastPath> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM)
    }
}

// Default Count sketch for Vector2D<i128> (RegularPath).
impl Default for Count<Vector2D<i128>, RegularPath> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM)
    }
}

// Default Count sketch for Vector2D<i128> (FastPath).
impl Default for Count<Vector2D<i128>, FastPath> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM)
    }
}

// Default Count sketch for FixedMatrix (RegularPath).
impl Default for Count<FixedMatrix, RegularPath> {
    fn default() -> Self {
        Count::from_storage(FixedMatrix::default())
    }
}

// Default Count sketch for FixedMatrix (FastPath).
impl Default for Count<FixedMatrix, FastPath> {
    fn default() -> Self {
        Count::from_storage(FixedMatrix::default())
    }
}

// Default Count sketch for DefaultMatrixI32 (RegularPath).
impl Default for Count<DefaultMatrixI32, RegularPath> {
    fn default() -> Self {
        Count::from_storage(DefaultMatrixI32::default())
    }
}

// Default Count sketch for DefaultMatrixI32 (FastPath).
impl Default for Count<DefaultMatrixI32, FastPath> {
    fn default() -> Self {
        Count::from_storage(DefaultMatrixI32::default())
    }
}

// Default Count sketch for DefaultMatrixI64 (RegularPath).
impl Default for Count<DefaultMatrixI64, RegularPath> {
    fn default() -> Self {
        Count::from_storage(DefaultMatrixI64::default())
    }
}

// Default Count sketch for DefaultMatrixI64 (FastPath).
impl Default for Count<DefaultMatrixI64, FastPath> {
    fn default() -> Self {
        Count::from_storage(DefaultMatrixI64::default())
    }
}

// Default Count sketch for DefaultMatrixI128 (RegularPath).
impl Default for Count<DefaultMatrixI128, RegularPath> {
    fn default() -> Self {
        Count::from_storage(DefaultMatrixI128::default())
    }
}

// Default Count sketch for DefaultMatrixI128 (FastPath).
impl Default for Count<DefaultMatrixI128, FastPath> {
    fn default() -> Self {
        Count::from_storage(DefaultMatrixI128::default())
    }
}

// Default Count sketch for QuickMatrixI64 (RegularPath).
impl Default for Count<QuickMatrixI64, RegularPath> {
    fn default() -> Self {
        Count::from_storage(QuickMatrixI64::default())
    }
}

// Default Count sketch for QuickMatrixI64 (FastPath).
impl Default for Count<QuickMatrixI64, FastPath> {
    fn default() -> Self {
        Count::from_storage(QuickMatrixI64::default())
    }
}

// Default Count sketch for QuickMatrixI128 (RegularPath).
impl Default for Count<QuickMatrixI128, RegularPath> {
    fn default() -> Self {
        Count::from_storage(QuickMatrixI128::default())
    }
}

// Default Count sketch for QuickMatrixI128 (FastPath).
impl Default for Count<QuickMatrixI128, FastPath> {
    fn default() -> Self {
        Count::from_storage(QuickMatrixI128::default())
    }
}

// Count constructors for Vector2D-backed storage.
impl<T, M, H: SketchHasher> Count<Vector2D<T>, M, H>
where
    T: CountSketchCounter,
{
    /// Creates a sketch with the requested number of rows and columns.
    pub fn with_dimensions(rows: usize, cols: usize) -> Self {
        let mut sk = Count {
            counts: Vector2D::init(rows, cols),
            row: rows,
            col: cols,
            _mode: PhantomData,
            _hasher: PhantomData,
        };
        sk.counts.fill(T::from(0));
        sk
    }
}

// Core Count API for any storage/counter.
impl<S, C, Mode, H: SketchHasher> Count<S, Mode, H>
where
    S: MatrixStorage<Counter = C>,
    C: CountSketchCounter,
{
    /// Wraps an existing matrix storage as a Count Sketch.
    pub fn from_storage(counts: S) -> Self {
        let row = counts.rows();
        let col = counts.cols();
        Self {
            counts,
            row,
            col,
            _mode: PhantomData,
            _hasher: PhantomData,
        }
    }

    /// Number of rows in the sketch.
    pub fn rows(&self) -> usize {
        self.counts.rows()
    }

    /// Number of columns in the sketch.
    pub fn cols(&self) -> usize {
        self.counts.cols()
    }

    /// Merges another sketch while asserting compatible dimensions.
    pub fn merge(&mut self, other: &Self) {
        let self_rows = self.counts.rows();
        let self_cols = self.counts.cols();
        assert_eq!(
            (self_rows, self_cols),
            (other.counts.rows(), other.counts.cols()),
            "dimension mismatch while merging CountMin sketches"
        );

        for i in 0..self_rows {
            for j in 0..self_cols {
                self.counts.update_one_counter(
                    i,
                    j,
                    |a, b| *a += b,
                    other.counts.query_one_counter(i, j),
                );
            }
        }
    }

    /// Exposes the backing matrix for inspection/testing.
    pub fn as_storage(&self) -> &S {
        &self.counts
    }

    /// Mutable access used internally for testing scenarios.
    pub fn as_storage_mut(&mut self) -> &mut S {
        &mut self.counts
    }
}

// Regular-path Count operations.
impl<S, C, H: SketchHasher> Count<S, RegularPath, H>
where
    S: MatrixStorage<Counter = C>,
    C: CountSketchCounter,
{
    /// Inserts an observation with standard Count Sketch updating algorithm.
    pub fn insert(&mut self, value: &DataInput) {
        let rows = self.counts.rows();
        let cols = self.counts.cols();
        for r in 0..rows {
            let hashed = H::hash64_seeded(r, value);
            let col = ((hashed & LOWER_32_MASK) as usize) % cols;
            let bit = ((hashed >> 63) & 1) as i32;
            let sign_bit = if bit == 1 { 1 } else { -1 };
            let delta = if sign_bit > 0 {
                C::from(1)
            } else {
                -C::from(1)
            };
            self.counts
                .update_one_counter(r, col, |a, b| *a += b, delta);
        }
    }

    /// Inserts an observation with the given count (weight).
    pub fn insert_many(&mut self, value: &DataInput, many: C) {
        let rows = self.counts.rows();
        let cols = self.counts.cols();
        for r in 0..rows {
            let hashed = H::hash64_seeded(r, value);
            let col = ((hashed & LOWER_32_MASK) as usize) % cols;
            let bit = ((hashed >> 63) & 1) as i32;
            let sign_bit = if bit == 1 { 1 } else { -1 };
            let delta = if sign_bit > 0 { many } else { -many };
            self.counts
                .update_one_counter(r, col, |a, b| *a += b, delta);
        }
    }

    /// Returns the frequency estimate for the provided value.
    pub fn estimate(&self, value: &DataInput) -> f64 {
        let rows = self.counts.rows();
        let cols = self.counts.cols();
        let mut estimates = Vec::with_capacity(rows);
        for r in 0..rows {
            let hashed = H::hash64_seeded(r, value);
            let col = ((hashed & LOWER_32_MASK) as usize) % cols;
            let bit = ((hashed >> 63) & 1) as i32;
            let sign_bit = if bit == 1 { 1 } else { -1 };
            let counter = self.counts.query_one_counter(r, col);
            if sign_bit > 0 {
                estimates.push(counter.to_f64());
            } else {
                estimates.push(-counter.to_f64());
            }
        }
        if estimates.is_empty() {
            return 0.0;
        }
        estimates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = estimates.len() / 2;
        if estimates.len() % 2 == 1 {
            estimates[mid]
        } else {
            (estimates[mid - 1] + estimates[mid]) / 2.0
        }
    }
}

// Fast-path Count operations using precomputed hashes.
impl<S, H: SketchHasher> Count<S, FastPath, H>
where
    S: MatrixStorage + crate::FastPathHasher<H>,
    S::Counter: CountSketchCounter,
{
    /// Inserts an observation using the combined hash optimization.
    #[inline(always)]
    pub fn insert(&mut self, value: &DataInput) {
        let hashed_val = <S as FastPathHasher<H>>::hash_for_matrix(&self.counts, value);
        self.counts.fast_insert(
            |counter, value, row| {
                let sign = hashed_val.sign_for_row(row);
                let delta = if sign > 0 { *value } else { -*value };
                *counter += delta;
            },
            S::Counter::from(1),
            &hashed_val,
        );
    }

    /// Inserts an observation with the given count using the combined hash optimization.
    #[inline(always)]
    pub fn insert_many(&mut self, value: &DataInput, many: S::Counter) {
        let hashed_val = <S as FastPathHasher<H>>::hash_for_matrix(&self.counts, value);
        self.counts.fast_insert(
            |counter, value, row| {
                let sign = hashed_val.sign_for_row(row);
                let delta = if sign > 0 { *value } else { -*value };
                *counter += delta;
            },
            many,
            &hashed_val,
        );
    }

    /// Returns the frequency estimate for the provided value.
    #[inline(always)]
    pub fn estimate(&self, value: &DataInput) -> f64 {
        let hashed_val = <S as FastPathHasher<H>>::hash_for_matrix(&self.counts, value);
        self.counts
            .fast_query_median(&hashed_val, |val, row, hash| {
                let sign = hash.sign_for_row(row);
                if sign > 0 {
                    (*val).to_f64()
                } else {
                    -(*val).to_f64()
                }
            })
    }

    /// Inserts an observation using a pre-computed hash value.
    /// Hash value can be reused with other sketches.
    #[inline(always)]
    pub fn fast_insert_with_hash_value(&mut self, hashed_val: &H::HashType) {
        self.counts.fast_insert(
            |counter, value, row| {
                let sign = hashed_val.sign_for_row(row);
                let delta = if sign > 0 { *value } else { -*value };
                *counter += delta;
            },
            S::Counter::from(1),
            hashed_val,
        );
    }

    /// Inserts an observation with the given count using a pre-computed hash value.
    #[inline(always)]
    pub fn fast_insert_many_with_hash_value(&mut self, hashed_val: &H::HashType, many: S::Counter) {
        self.counts.fast_insert(
            |counter, value, row| {
                let sign = hashed_val.sign_for_row(row);
                let delta = if sign > 0 { *value } else { -*value };
                *counter += delta;
            },
            many,
            hashed_val,
        );
    }

    /// Returns the frequency estimate using a pre-computed hash value.
    #[inline(always)]
    pub fn fast_estimate_with_hash(&self, hashed_val: &H::HashType) -> f64 {
        self.counts.fast_query_median(hashed_val, |val, row, hash| {
            let sign = hash.sign_for_row(row);
            if sign > 0 {
                (*val).to_f64()
            } else {
                -(*val).to_f64()
            }
        })
    }
}

// Debug helpers for i32 Vector2D Count.
impl<M, H: SketchHasher> Count<Vector2D<i32>, M, H> {
    /// Human-friendly helper used by the serializer demo binaries.
    pub fn debug(&self) {
        for row in 0..self.counts.rows() {
            println!("row {row}: {:?}", self.counts.row_slice(row));
        }
    }
}

// Nitro sampling helpers for fast-path Count.
impl<H: SketchHasher> Count<Vector2D<i32>, FastPath, H> {
    /// Enables Nitro sampling with the provided rate.
    pub fn enable_nitro(&mut self, sampling_rate: f64) {
        self.counts.enable_nitro(sampling_rate);
    }

    /// Enables Nitro sampling with a reproducible, seed-selected schedule.
    ///
    /// See [`crate::Nitro::init_nitro_seeded`]: the unseeded path starts every
    /// sketch at the same point in the skip table, so two sketches at the same
    /// rate admit the same subset and are not independent trials.
    pub fn enable_nitro_with_seed(&mut self, sampling_rate: f64, seed: u64) {
        self.counts.enable_nitro_with_seed(sampling_rate, seed);
    }

    /// Inserts an observation through Nitro's per-row sampling schedule.
    ///
    /// Cells and signs are derived exactly as a plain `insert` derives them —
    /// `FastPathHasher::hash_for_matrix`, then `col_for_row` and
    /// `sign_for_row` — so `estimate` reads back what this wrote. The hash is
    /// computed once per observation, and the admitted rows are collected into
    /// an inline buffer, so the hot path does not allocate.
    #[inline(always)]
    pub fn fast_insert_nitro(&mut self, value: &DataInput) {
        let rows = self.counts.rows();
        let mut admitted = AdmittedRows::new();
        self.counts.nitro_mut().admit_rows(rows, &mut admitted);
        if admitted.is_empty() {
            return;
        }
        let cols = self.counts.cols();
        let hashed = <Vector2D<i32> as FastPathHasher<H>>::hash_for_matrix(&self.counts, value);
        for (row, weight) in admitted {
            let col = MatrixFastHash::col_for_row(&hashed, row, cols);
            let signed = if hashed.sign_for_row(row) > 0 {
                nitro_delta_saturated_i32(weight)
            } else {
                -nitro_delta_saturated_i32(weight)
            };
            self.counts
                .update_one_counter(row, col, |a: &mut i32, b: i32| *a += b, signed);
        }
    }
}

// NitroTarget integration for fast-path Count.
impl<H: SketchHasher> NitroTarget for Count<Vector2D<i32>, FastPath, H> {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.counts.rows()
    }

    #[inline(always)]
    fn update_row(&mut self, row: usize, hashed: u128, delta: u64) {
        let bit = (hashed >> (127 - row)) & 1;
        let sign = (bit << 1) as i32 - 1;
        self.counts.update_by_row(
            row,
            hashed,
            |a, b| *a += b,
            sign * nitro_delta_saturated_i32(delta),
        );
    }

    #[inline(always)]
    fn update_sample(&mut self, value: &DataInput, delta: u64) {
        // Hash with the SAME derivation the estimator uses so both share one
        // hash domain (raw hash128_seeded bit-slicing diverges in Packed64 mode).
        let hashed = <Vector2D<i32> as FastPathHasher<H>>::hash_for_matrix(&self.counts, value);
        let cols = self.counts.cols();
        for row in 0..self.counts.rows() {
            let col = <H::HashType as MatrixFastHash>::col_for_row(&hashed, row, cols);
            let sign = <H::HashType as MatrixFastHash>::sign_for_row(&hashed, row);
            self.counts.update_one_counter(
                row,
                col,
                |a, b| *a += b,
                sign * nitro_delta_saturated_i32(delta),
            );
        }
    }
}

use crate::input_to_owned;
use crate::octo_delta::{COUNT_PROMASK, CountDelta, KeyedCountDelta, MAX_PROMASK};

impl<S: MatrixStorage<Counter = i32>, H: SketchHasher> Count<S, RegularPath, H> {
    /// Inserts a value, emitting a delta at every promotion of the default
    /// threshold `COUNT_PROMASK`.
    #[inline(always)]
    pub fn insert_emit_delta(&mut self, value: &DataInput, emit: &mut impl FnMut(CountDelta)) {
        self.insert_emit_delta_with_threshold(value, COUNT_PROMASK, emit);
    }

    /// Inserts a value, emitting a delta and clearing the counter each time
    /// `|counter|` reaches `threshold`. Count sketch counters are signed, so
    /// the magnitude is what the threshold is applied to (§4.4).
    #[inline(always)]
    pub fn insert_emit_delta_with_threshold(
        &mut self,
        value: &DataInput,
        threshold: u32,
        emit: &mut impl FnMut(CountDelta),
    ) {
        let threshold = threshold.clamp(1, MAX_PROMASK);
        let rows = self.counts.rows();
        let cols = self.counts.cols();
        for r in 0..rows {
            let hashed = hash64_seeded(r, value);
            let col = ((hashed & LOWER_32_MASK) as usize) % cols;
            let sign: i32 = if ((hashed >> 63) & 1) == 1 { 1 } else { -1 };
            self.counts.increment_by_row(r, col, sign);
            let current = self.counts.query_one_counter(r, col);
            if current.unsigned_abs() >= threshold {
                emit(CountDelta {
                    row: r as u32,
                    col: col as u32,
                    value: current,
                });
                self.counts.update_one_counter(r, col, |c, _| *c = 0, ());
            }
        }
    }

    /// As `insert_emit_delta_with_threshold`, but every delta carries the flow
    /// key so an aggregator can maintain a heavy-hitter heap.
    pub fn insert_emit_keyed_delta_with_threshold(
        &mut self,
        value: &DataInput,
        threshold: u32,
        emit: &mut impl FnMut(KeyedCountDelta),
    ) {
        self.insert_emit_delta_with_threshold(value, threshold, &mut |delta| {
            emit(KeyedCountDelta {
                key: input_to_owned(value),
                delta,
            })
        });
    }
}

impl<S, H: SketchHasher> Count<S, FastPath, H>
where
    S: MatrixStorage<Counter = i32> + FastPathHasher<H>,
{
    /// Inserts a value using the fast path, emitting a delta at every
    /// promotion of the default threshold `COUNT_PROMASK`.
    #[inline(always)]
    pub fn insert_emit_delta(&mut self, value: &DataInput, emit: &mut impl FnMut(CountDelta)) {
        self.insert_emit_delta_with_threshold(value, COUNT_PROMASK, emit);
    }

    /// Fast-path counterpart of the regular-path threshold API.
    #[inline(always)]
    pub fn insert_emit_delta_with_threshold(
        &mut self,
        value: &DataInput,
        threshold: u32,
        emit: &mut impl FnMut(CountDelta),
    ) {
        let threshold = threshold.clamp(1, MAX_PROMASK);
        let hashed_val = <S as FastPathHasher<H>>::hash_for_matrix(&self.counts, value);
        let rows = self.counts.rows();
        let cols = self.counts.cols();
        for r in 0..rows {
            let col = hashed_val.col_for_row(r, cols);
            let sign = hashed_val.sign_for_row(r);
            self.counts.increment_by_row(r, col, sign);
            let current = self.counts.query_one_counter(r, col);
            if current.unsigned_abs() >= threshold {
                emit(CountDelta {
                    row: r as u32,
                    col: col as u32,
                    value: current,
                });
                self.counts.update_one_counter(r, col, |c, _| *c = 0, ());
            }
        }
    }

    /// Fast-path counterpart of `insert_emit_keyed_delta_with_threshold`.
    pub fn insert_emit_keyed_delta_with_threshold(
        &mut self,
        value: &DataInput,
        threshold: u32,
        emit: &mut impl FnMut(KeyedCountDelta),
    ) {
        self.insert_emit_delta_with_threshold(value, threshold, &mut |delta| {
            emit(KeyedCountDelta {
                key: input_to_owned(value),
                delta,
            })
        });
    }
}

impl<S: MatrixStorage, Mode, H: SketchHasher> Count<S, Mode, H>
where
    S::Counter: Copy + std::ops::AddAssign + From<i32>,
{
    /// Applies a previously emitted delta to this sketch.
    pub fn apply_delta(&mut self, delta: CountDelta) {
        self.counts.increment_by_row(
            delta.row as usize,
            delta.col as usize,
            S::Counter::from(delta.value),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{all_counter_zero_i32, counter_index, sample_zipf_u64};
    use crate::{DataInput, hash64_seeded};
    use std::collections::HashMap;

    #[test]
    fn count_child_insert_emits_at_threshold() {
        let mut child = Count::<Vector2D<i32>, RegularPath>::with_dimensions(3, 64);
        let key = DataInput::U64(99);
        let mut deltas: Vec<CountDelta> = Vec::new();

        for _ in 0..200 {
            child.insert_emit_delta(&key, &mut |d| deltas.push(d));
        }
        assert!(
            deltas.len() >= 3,
            "expected at least one promoted delta per row"
        );
    }

    fn counter_sign(row: usize, key: &DataInput) -> i32 {
        let hash = hash64_seeded(row, key);
        if (hash >> 63) & 1 == 1 { 1 } else { -1 }
    }

    fn run_zipf_stream(
        rows: usize,
        cols: usize,
        domain: usize,
        exponent: f64,
        samples: usize,
        seed: u64,
    ) -> (Count, HashMap<u64, i32>) {
        let mut truth = HashMap::<u64, i32>::new();
        let mut sketch = Count::<Vector2D<i32>, RegularPath>::with_dimensions(rows, cols);

        for value in sample_zipf_u64(domain, exponent, samples, seed) {
            let key = DataInput::U64(value);
            sketch.insert(&key);
            *truth.entry(value).or_insert(0) += 1;
        }

        (sketch, truth)
    }

    #[test]
    fn default_initializes_expected_dimensions() {
        let cs = Count::<Vector2D<i32>, RegularPath>::default();
        assert_eq!(cs.rows(), 3);
        assert_eq!(cs.cols(), 4096);
        all_counter_zero_i32(cs.as_storage());
    }

    #[test]
    fn with_dimensions_uses_custom_sizes() {
        let cs = Count::<Vector2D<i32>, RegularPath>::with_dimensions(3, 17);
        assert_eq!(cs.rows(), 3);
        assert_eq!(cs.cols(), 17);

        let storage = cs.as_storage();
        for row in 0..cs.rows() {
            assert!(
                storage.row_slice(row).iter().all(|&value| value == 0),
                "expected row {} to be zero-initialized, got {:?}",
                row,
                storage.row_slice(row)
            );
        }
    }

    #[test]
    fn insert_updates_signed_counters_per_row() {
        let mut sketch = Count::<Vector2D<i32>, RegularPath>::with_dimensions(3, 64);
        let key = DataInput::Str("alpha");

        sketch.insert(&key);

        for row in 0..sketch.rows() {
            let idx = counter_index(row, &key, sketch.cols());
            let expected = counter_sign(row, &key);
            assert_eq!(
                sketch.counts.query_one_counter(row, idx),
                expected,
                "row {row} counter mismatch"
            );
        }
    }

    #[test]
    fn fast_insert_produces_consistent_estimates() {
        let mut fast = Count::<Vector2D<i32>, FastPath>::with_dimensions(4, 128);

        let keys = vec![
            DataInput::Str("alpha"),
            DataInput::Str("beta"),
            DataInput::Str("gamma"),
            DataInput::Str("delta"),
            DataInput::Str("epsilon"),
        ];

        for key in &keys {
            fast.insert(key);
        }

        for key in &keys {
            let estimate = fast.estimate(key);
            assert!(
                (estimate - 1.0).abs() < f64::EPSILON,
                "fast estimate for key {key:?} should be 1.0, got {estimate}"
            );
        }
    }

    #[test]
    fn insert_produces_consistent_estimates() {
        let mut sketch = Count::<Vector2D<i32>, RegularPath>::with_dimensions(3, 64);

        let keys = vec![
            DataInput::Str("alpha"),
            DataInput::Str("beta"),
            DataInput::Str("gamma"),
            DataInput::Str("delta"),
            DataInput::Str("epsilon"),
        ];

        for key in &keys {
            sketch.insert(key);
        }

        for key in &keys {
            let estimate = sketch.estimate(key);
            assert!(
                (estimate - 1.0).abs() < f64::EPSILON,
                "estimate for key {key:?} should be 1.0, got {estimate}"
            );
        }
    }

    #[test]
    fn estimate_recovers_frequency_for_repeated_key() {
        let mut sketch = Count::<Vector2D<i32>, RegularPath>::with_dimensions(3, 64);
        let key = DataInput::Str("theta");

        let repeats = 37;
        for _ in 0..repeats {
            sketch.insert(&key);
        }

        let estimate = sketch.estimate(&key);
        assert!(
            (estimate - repeats as f64).abs() < f64::EPSILON,
            "expected estimate {repeats}, got {estimate}"
        );
    }

    #[test]
    fn fast_path_recovers_repeated_insertions() {
        let mut sketch = Count::<Vector2D<i32>, FastPath>::with_dimensions(4, 256);
        let keys = vec![
            DataInput::Str("alpha"),
            DataInput::Str("beta"),
            DataInput::Str("gamma"),
            DataInput::Str("delta"),
            DataInput::Str("epsilon"),
        ];

        for _ in 0..5 {
            for key in &keys {
                sketch.insert(key);
            }
        }

        for key in &keys {
            let estimate = sketch.estimate(key);
            assert!(
                (estimate - 5.0).abs() < f64::EPSILON,
                "fast estimate for key {key:?} should be 5.0, got {estimate}"
            );
        }
    }

    #[test]
    fn merge_adds_counters_element_wise() {
        let mut left = Count::<Vector2D<i32>, RegularPath>::with_dimensions(2, 32);
        let mut right = Count::<Vector2D<i32>, RegularPath>::with_dimensions(2, 32);
        let key = DataInput::Str("delta");

        left.insert(&key);
        right.insert(&key);
        right.insert(&key);

        let left_indices: Vec<_> = (0..left.rows())
            .map(|row| counter_index(row, &key, left.cols()))
            .collect();

        left.merge(&right);

        for (row, idx) in left_indices.into_iter().enumerate() {
            let expected = counter_sign(row, &key) * 3;
            assert_eq!(left.as_storage().query_one_counter(row, idx), expected);
        }
    }

    #[test]
    #[should_panic(expected = "dimension mismatch while merging CountMin sketches")]
    fn merge_requires_matching_dimensions() {
        let mut left = Count::<Vector2D<i32>, RegularPath>::with_dimensions(2, 32);
        let right = Count::<Vector2D<i32>, RegularPath>::with_dimensions(3, 32);
        left.merge(&right);
    }

    #[test]
    fn zipf_stream_stays_within_twenty_percent_for_most_keys() {
        let (sketch, truth) = run_zipf_stream(5, 8192, 8192, 1.1, 200_000, 0x5eed_c0de);
        let mut within_tolerance = 0usize;
        for (&value, &count) in &truth {
            let estimate = sketch.estimate(&DataInput::U64(value));
            let rel_error = ((estimate - count as f64).abs()) / (count as f64);
            if rel_error < 0.20 {
                within_tolerance += 1;
            }
        }

        let total = truth.len();
        let accuracy = within_tolerance as f64 / total as f64;
        assert!(
            accuracy >= 0.70,
            "Only {:.2}% of keys within tolerance ({} of {}); expected at least 70%",
            accuracy * 100.0,
            within_tolerance,
            total
        );
    }

    #[test]
    fn cs_regular_path_correctness() {
        let mut sk = Count::<Vector2D<i32>, RegularPath>::default();
        // Insert values 0..9 once using the regular path.
        for i in 0..10 {
            sk.insert(&DataInput::I32(i));
        }

        // Build the expected counter array by mirroring the regular-path hashing logic.
        let storage = sk.as_storage();
        let rows = storage.rows();
        let cols = storage.cols();
        let mut expected_once = vec![0_i32; rows * cols];
        for i in 0..10 {
            let value = DataInput::I32(i);
            for r in 0..rows {
                let hashed = hash64_seeded(r, &value);
                let col = ((hashed & LOWER_32_MASK) as usize) % cols;
                let bit = ((hashed >> 63) & 1) as i32;
                let sign_bit = -(1 - 2 * bit);
                let idx = r * cols + col;
                expected_once[idx] += sign_bit;
            }
        }
        // All counters should match the expected single-pass values.
        assert_eq!(storage.as_slice(), expected_once.as_slice());

        // Insert the same values again; counters should double.
        for i in 0..10 {
            sk.insert(&DataInput::I32(i));
        }
        let expected_twice: Vec<i32> = expected_once.iter().map(|v| v * 2).collect();
        assert_eq!(sk.as_storage().as_slice(), expected_twice.as_slice());

        // Estimates for inserted keys should be exactly 2.
        for i in 0..10 {
            let estimate = sk.estimate(&DataInput::I32(i));
            assert!(
                (estimate - 2.0).abs() < f64::EPSILON,
                "estimate for {i} should be 2.0, but get {estimate}"
            );
        }
    }

    #[test]
    fn cs_fast_path_correctness() {
        let mut sk = Count::<Vector2D<i32>, FastPath>::default();
        // Insert values 0..9 once using the fast path.
        for i in 0..10 {
            sk.insert(&DataInput::I32(i));
        }

        // Build the expected counter array by mirroring the fast-path hashing logic.
        let storage = sk.as_storage();
        let rows = storage.rows();
        let cols = storage.cols();
        let mask_bits = storage.get_mask_bits();
        let mask = (1u128 << mask_bits) - 1;
        let mut expected_once = vec![0_i32; rows * cols];

        for i in 0..10 {
            let value = DataInput::I32(i);
            let hash = <Vector2D<i32> as FastPathHasher<DefaultXxHasher>>::hash_for_matrix(
                storage, &value,
            );
            for row in 0..rows {
                let hashed = hash.row_hash(row, mask_bits, mask);
                let col = (hashed % cols as u128) as usize;
                let idx = row * cols + col;
                expected_once[idx] += hash.sign_for_row(row);
            }
        }

        assert_eq!(storage.as_slice(), expected_once.as_slice());
    }

    #[test]
    fn countsketch_error_stays_rank_independent_across_frequency_deciles() {
        const ROWS: usize = 5;
        const COLS: usize = 4096;
        const DOMAIN: usize = 8192;
        const SAMPLES: usize = 200_000;
        const STREAM_SEED: u64 = 1007;
        const DECILES: usize = 10;

        let mut truth = HashMap::<u64, i64>::new();
        let mut cs = Count::<Vector2D<i64>, RegularPath>::with_dimensions(ROWS, COLS);
        for value in sample_zipf_u64(DOMAIN, 1.1, SAMPLES, STREAM_SEED) {
            cs.insert(&DataInput::U64(value));
            *truth.entry(value).or_insert(0) += 1;
        }

        let mut pairs: Vec<(u64, i64)> = truth.into_iter().collect();
        pairs.sort_by_key(|(_, c)| *c);
        let per = pairs.len() / DECILES;
        let means: Vec<f64> = (0..DECILES)
            .map(|d| {
                let slice = &pairs[d * per..(d + 1) * per];
                slice
                    .iter()
                    .map(|(k, c)| (cs.estimate(&DataInput::U64(*k)) - *c as f64).abs())
                    .sum::<f64>()
                    / slice.len() as f64
            })
            .collect();
        let lo = means.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = means.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            hi <= lo * 3.0,
            "Count Sketch mean |error| per frequency decile spans {lo:.1}..{hi:.1} ({:.2}x), \
             beyond the documented 3x empirical band — the error has started tracking key \
             frequency, which Count Sketch's L2 guarantee says it must not. \
             stream_seed={STREAM_SEED} rows={ROWS} cols={COLS} deciles={means:?}",
            hi / lo
        );
    }
}
