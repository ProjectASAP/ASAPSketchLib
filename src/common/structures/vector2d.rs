use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

use crate::{
    DataInput, FastPathHasher, MatrixFastHash, MatrixHashType, MatrixStorage, Nitro, SketchHasher,
    compute_median_inline_f64,
};
/// Shared thin wrapper over `Vec<T>` tailored for sketches.
#[derive(Clone, Debug, Serialize)]
pub struct Vector2D<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    mask_bits: u32,
    mask: u128,
    nitro: Nitro,
}

// Helper type for deserialization: we only read stored fields and recompute
// derived ones (mask_bits, mask) from rows/cols.
#[derive(Deserialize)]
struct Vector2DDeserialize<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
    #[serde(default)]
    nitro: Nitro,
}

impl<'de, T> Deserialize<'de> for Vector2D<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let input = Vector2DDeserialize::deserialize(deserializer)?;
        let mask_bits = super::matrix_storage::cols_mask_bits(input.cols);
        let mask = super::matrix_storage::cols_mask(input.cols);
        Ok(Self {
            data: input.data,
            rows: input.rows,
            cols: input.cols,
            mask_bits,
            mask,
            nitro: input.nitro,
        })
    }
}

impl<T> Vector2D<T> {
    /// Creates an empty matrix with reserved capacity for `rows * cols` elements.
    /// The underlying storage is left uninitialized until `fill` or similar methods are called,
    /// allowing callers to decide when and how counters are populated.
    pub fn init(rows: usize, cols: usize) -> Self {
        let mask_bits = super::matrix_storage::cols_mask_bits(cols);
        let mask = super::matrix_storage::cols_mask(cols);
        Self {
            data: Vec::with_capacity(rows * cols),
            rows,
            cols,
            mask_bits,
            mask,
            nitro: Nitro::default(),
        }
    }

    /// Builds a matrix by invoking a generator for every `(row, col)` position.
    /// Useful for types that require per-cell construction logic (e.g., heaps or buckets)
    /// instead of cloning a single value across all cells.
    pub fn from_fn<F>(rows: usize, cols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mask_bits = super::matrix_storage::cols_mask_bits(cols);
        let mask = super::matrix_storage::cols_mask(cols);
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self {
            data,
            rows,
            cols,
            mask_bits,
            mask,
            nitro: Nitro::default(),
        }
    }

    /// Enables Nitro sampling with the provided rate.
    pub fn enable_nitro(&mut self, sampling_rate: f64) {
        self.nitro = Nitro::init_nitro(sampling_rate);
    }

    /// Enables Nitro sampling with a reproducible, seed-selected schedule.
    ///
    /// [`Self::enable_nitro`] always starts the skip cursor at entry 0, so
    /// every sketch at a given rate admits the same subset. Seeding it is what
    /// lets an accuracy battery run over genuinely independent trials while
    /// still reproducing exactly. See [`Nitro::init_nitro_seeded`].
    pub fn enable_nitro_with_seed(&mut self, sampling_rate: f64, seed: u64) {
        self.nitro = Nitro::init_nitro_seeded(sampling_rate, seed);
    }

    /// Disables Nitro sampling and resets the internal state.
    pub fn disable_nitro(&mut self) {
        self.nitro = Nitro::default();
    }

    #[inline(always)]
    /// Decrements the Nitro skip counter by one.
    pub fn reduce_to_skip(&mut self) {
        self.nitro.reduce_to_skip();
    }

    /// Returns the Nitro configuration.
    #[inline(always)]
    pub fn nitro(&self) -> &Nitro {
        &self.nitro
    }

    #[inline(always)]
    /// Returns the current Nitro delta weight.
    pub fn get_delta(&self) -> u64 {
        self.nitro.delta
    }

    /// Returns a mutable Nitro configuration reference.
    #[inline(always)]
    pub fn nitro_mut(&mut self) -> &mut Nitro {
        &mut self.nitro
    }

    /// Replaces the entire matrix with `rows * cols` clones of `value`, reusing the existing allocation.
    /// This is the most efficient way to reset counters to a baseline without reallocating.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.clear();
        self.data.resize(self.rows * self.cols, value);
    }

    #[inline(always)]
    fn col_for_row<Hash: MatrixFastHash>(&self, hashed_val: &Hash, row: usize) -> usize {
        // Decode with the (mask_bits, mask) pair cached at construction —
        // identical result to recomputing them from `self.cols`, without the
        // per-row ilog2/shift. `fold_to_col` owns the pow2-skip invariant.
        let raw = hashed_val.row_hash(row, self.mask_bits, self.mask);
        super::matrix_storage::fold_to_col(raw, self.cols)
    }

    /// Returns the number of rows.
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Allocates one extra row initialized with `value`.
    pub fn allocate_extra_row(&mut self, value: T)
    where
        T: Clone,
    {
        self.rows += 1;
        self.data.resize(self.rows * self.cols, value);
    }

    /// Returns the total number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    /// Returns `true` when the matrix stores no elements.
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    /// Provides immutable access to the flattened storage.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Provides mutable access to the flattened storage.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a reference to a cell when it exists.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Returns a mutable reference to a cell when it exists.
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            Some(&mut self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Applies an update to a single cell via the supplied operator.
    #[inline(always)]
    pub fn update_one_counter<F, V>(&mut self, row: usize, col: usize, op: F, value: V)
    where
        F: Fn(&mut T, V),
        T: Clone,
    {
        let idx = row * self.cols + col;
        op(&mut self.data[idx], value);
    }

    /// get the number of bits required to cover the col size
    #[inline(always)]
    /// Returns the bit width needed to represent a column index.
    pub fn get_mask_bits(&self) -> u32 {
        super::matrix_storage::cols_mask_bits(self.cols)
    }

    /// get the number of bits required for hashed value
    /// only three case possible: 32, 64, 128
    #[inline]
    /// Returns the packed hash width needed for all rows.
    pub fn get_required_bits(&self) -> usize {
        let mut bits_required = self.get_mask_bits() as usize;
        bits_required *= self.rows;
        bits_required = 32 << ((bits_required > 32) as u32 + (bits_required > 64) as u32);
        bits_required = bits_required.min(128);
        bits_required
    }

    /// Inserts a value along every row using a hashed column selection.
    ///
    /// The closure receives three parameters: mutable counter reference, the value,
    /// and the current row index. For simple operations that don't need the row index,
    /// use `_` to ignore it (zero performance cost due to compiler optimization).
    ///
    /// # Examples
    ///
    /// Simple increment (row-independent):
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let mut sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// sketch.fast_insert(|counter, value, _| *counter += *value, 1i64, &hash);
    /// ```
    ///
    /// Row-dependent operation (e.g., Count sketch with sign bits):
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let mut sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// sketch.fast_insert(|counter, value, row| {
    ///     let sign = hash.sign_for_row(row) as i64;
    ///     *counter += sign * *value;
    /// }, 1i64, &hash);
    /// ```
    #[inline(always)]
    pub fn fast_insert<Hash, F, V>(&mut self, op: F, value: V, hashed_val: &Hash)
    where
        Hash: MatrixFastHash,
        F: Fn(&mut T, &V, usize),
        V: Clone,
    {
        for row in 0..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            op(&mut self.data[idx], &value, row);
        }
    }

    #[inline(always)]
    /// Updates one row using a packed hash value.
    pub fn update_by_row<F, V>(&mut self, row: usize, hashed: u128, op: F, value: V)
    where
        F: Fn(&mut T, V),
        T: Clone,
    {
        let col = (hashed >> (self.mask_bits as usize * row)) as usize & (self.mask as usize);
        self.update_one_counter(row, col, op, value);
    }

    #[inline(always)]
    /// Decrements the Nitro skip counter by `c`.
    pub fn reduce_nitro_skip(&mut self, c: usize) {
        self.nitro.reduce_to_skip_by_count(c)
    }

    #[inline(always)]
    /// Sets the Nitro skip counter to `c`.
    pub fn update_nitro_skip(&mut self, c: usize) {
        self.nitro.to_skip = c
    }

    #[inline(always)]
    /// Returns the current Nitro skip counter.
    pub fn get_nitro_skip(&mut self) -> usize {
        self.nitro.to_skip
    }

    /// Reads a single counter by `(row, col)`.
    /// seems to be faster than [][] operation
    #[inline(always)]
    pub fn query_one_counter(&self, row: usize, col: usize) -> T
    where
        T: Clone,
    {
        self.data[row * self.cols + col].clone()
    }

    /// Queries all rows using precomputed hashed values to find the minimum.
    ///
    /// The closure receives: counter reference, row index, and hash value.
    /// Use `_` to ignore unused parameters (zero performance cost).
    ///
    /// # Examples
    ///
    /// Simple min (row-independent):
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let min = sketch.fast_query_min(&hash, |val, _, _| *val);
    /// # let _ = min;
    /// ```
    ///
    /// Row-dependent with transformation:
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let min = sketch.fast_query_min(&hash, |val, row, _| *val - row as i64);
    /// # let _ = min;
    /// ```
    #[inline(always)]
    pub fn fast_query_min<Hash, F, R>(&self, hashed_val: &Hash, op: F) -> R
    where
        Hash: MatrixFastHash,
        F: Fn(&T, usize, &Hash) -> R,
        R: PartialOrd,
    {
        let c0 = self.col_for_row(hashed_val, 0);
        let mut min = op(&self.data[c0], 0, hashed_val);
        for row in 1..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            let candidate = op(&self.data[idx], row, hashed_val);
            if candidate < min {
                min = candidate;
            }
        }
        min
    }

    /// Queries all rows using precomputed hashed values to find the median.
    ///
    /// The closure receives: counter reference, row index, and hash value.
    /// Returns f64 values which are collected and sorted to compute median.
    /// Use `_` to ignore unused parameters (zero performance cost).
    ///
    /// # Examples
    ///
    /// Simple median (row-independent):
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let median = sketch.fast_query_median(&hash, |val, _, _| *val as f64);
    /// # let _ = median;
    /// ```
    ///
    /// Row-dependent (e.g., Count sketch with sign bits):
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let median = sketch.fast_query_median(&hash, |val, row, hash| {
    ///     let sign = hash.sign_for_row(row) as f64;
    ///     *val as f64 * sign * (row as f64 + 1.0)
    /// });
    /// # let _ = median;
    /// ```
    #[inline(always)]
    pub fn fast_query_median<Hash, F>(&self, hashed_val: &Hash, op: F) -> f64
    where
        Hash: MatrixFastHash,
        F: Fn(&T, usize, &Hash) -> f64,
    {
        let mut estimates = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            estimates.push(op(&self.data[idx], row, hashed_val));
        }

        // Inline median computation
        compute_median_inline_f64(&mut estimates)
    }

    /// Queries all rows using precomputed hashed values to find the maximum.
    ///
    /// The closure receives: counter reference, row index, and hash value.
    /// Use `_` to ignore unused parameters (zero performance cost).
    ///
    /// # Examples
    ///
    /// Simple max (row-independent):
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let max = sketch.fast_query_max(&hash, |val, _, _| *val);
    /// # let _ = max;
    /// ```
    ///
    /// Row-dependent with transformation:
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let max = sketch.fast_query_max(&hash, |val, row, _| *val + row as i64);
    /// # let _ = max;
    /// ```
    #[inline(always)]
    pub fn fast_query_max<F, R>(&self, hashed_val: &MatrixHashType, op: F) -> R
    where
        F: Fn(&T, usize, &MatrixHashType) -> R,
        R: PartialOrd,
    {
        let c0 = self.col_for_row(hashed_val, 0);
        let mut max = op(&self.data[c0], 0, hashed_val);
        for row in 1..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            let candidate = op(&self.data[idx], row, hashed_val);
            if candidate > max {
                max = candidate;
            }
        }
        max
    }

    /// Queries all rows to find the minimum with a query key.
    ///
    /// The closure receives: counter reference, query key, row index, and hash value.
    /// Use `_` to ignore unused parameters (zero performance cost).
    ///
    /// # Examples
    ///
    /// With complex counter type:
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// # let query_key = ();
    /// let min = sketch.fast_query_min_with_key(&hash, &query_key, |val, _, _, _| *val);
    /// # let _ = min;
    /// ```
    #[inline(always)]
    pub fn fast_query_min_with_key<F, Q, R>(
        &self,
        hashed_val: &MatrixHashType,
        query_key: &Q,
        op: F,
    ) -> R
    where
        F: Fn(&T, &Q, usize, &MatrixHashType) -> R,
        R: PartialOrd,
    {
        let c0 = self.col_for_row(hashed_val, 0);
        let mut min = op(&self.data[c0], query_key, 0, hashed_val);
        for row in 1..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            let candidate = op(&self.data[idx], query_key, row, hashed_val);
            if candidate < min {
                min = candidate;
            }
        }
        min
    }

    /// Queries all rows to find the maximum with a query key.
    ///
    /// The closure receives: counter reference, query key, row index, and hash value.
    /// Use `_` to ignore unused parameters (zero performance cost).
    ///
    /// # Examples
    ///
    /// With complex counter type:
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// # let query_key = ();
    /// let max = sketch.fast_query_max_with_key(&hash, &query_key, |val, _, _, _| *val);
    /// # let _ = max;
    /// ```
    #[inline(always)]
    pub fn fast_query_max_with_key<F, Q, R>(
        &self,
        hashed_val: &MatrixHashType,
        query_key: &Q,
        op: F,
    ) -> R
    where
        F: Fn(&T, &Q, usize, &MatrixHashType) -> R,
        R: PartialOrd,
    {
        let c0 = self.col_for_row(hashed_val, 0);
        let mut max = op(&self.data[c0], query_key, 0, hashed_val);
        for row in 1..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            let candidate = op(&self.data[idx], query_key, row, hashed_val);
            if candidate > max {
                max = candidate;
            }
        }
        max
    }

    /// Queries all rows to find the median with a query key.
    ///
    /// The closure receives: counter reference, query key, row index, and hash value.
    /// Returns f64 values which are collected and sorted to compute median.
    /// Use `_` to ignore unused parameters (zero performance cost).
    ///
    /// # Examples
    ///
    /// With complex counter type:
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// # let query_key = ();
    /// let median =
    ///     sketch.fast_query_median_with_key(&hash, &query_key, |val, _, _, _| *val as f64);
    /// # let _ = median;
    /// ```
    #[inline(always)]
    pub fn fast_query_median_with_key<F, Q>(
        &self,
        hashed_val: &MatrixHashType,
        query_key: &Q,
        op: F,
    ) -> f64
    where
        F: Fn(&T, &Q, usize, &MatrixHashType) -> f64,
    {
        let mut estimates = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            estimates.push(op(&self.data[idx], query_key, row, hashed_val));
        }

        compute_median_inline_f64(&mut estimates)
    }

    /// Queries all rows with custom aggregation logic (fold/reduce pattern).
    ///
    /// This is the most flexible query method, allowing custom aggregation beyond
    /// min/max/median. Uses a fold pattern where the closure receives an accumulator
    /// and updates it for each row.
    ///
    /// The closure receives: accumulator, counter reference, query key, row index, and hash value.
    /// Use `_` to ignore unused parameters (zero performance cost).
    ///
    /// # Examples
    ///
    /// Custom sum with row-dependent weights:
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let sum = sketch.fast_query_aggregate(&hash, &(), 0.0, |acc, val, _, row, _| {
    ///     acc + (*val as f64 * (row as f64 + 1.0))
    /// });
    /// # let _ = sum;
    /// ```
    ///
    /// Count sketch estimation (sign-based sum then median):
    /// ```no_run
    /// use asap_sketchlib::{MatrixHashType, Vector2D};
    /// # let rows = 2;
    /// # let cols = 8;
    /// # let sketch = Vector2D::from_fn(rows, cols, |_, _| 0i64);
    /// # let hash = MatrixHashType::Packed128(0x1234);
    /// let mut estimates = Vec::new();
    /// sketch.fast_query_aggregate(&hash, &(), &mut estimates, |acc, val, _, row, hash| {
    ///     let sign = hash.sign_for_row(row) as f64;
    ///     acc.push(*val as f64 * sign);
    ///     acc
    /// });
    /// ```
    #[inline(always)]
    pub fn fast_query_aggregate<F, Q, R>(
        &self,
        hashed_val: &MatrixHashType,
        query_key: &Q,
        init: R,
        fold_fn: F,
    ) -> R
    where
        F: Fn(R, &T, &Q, usize, &MatrixHashType) -> R,
    {
        let mut acc = init;
        for row in 0..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let idx = row * self.cols + col;
            acc = fold_fn(acc, &self.data[idx], query_key, row, hashed_val);
        }
        acc
    }

    /// Returns an immutable slice corresponding to a full row.
    #[inline(always)]
    pub fn row_slice(&self, row: usize) -> &[T] {
        debug_assert!(row < self.rows, "row index out of bounds");
        let start = row * self.cols;
        let end = start + self.cols;
        &self.data[start..end]
    }

    /// Returns a mutable slice corresponding to a full row.
    #[inline(always)]
    pub fn row_slice_mut(&mut self, row: usize) -> &mut [T] {
        debug_assert!(row < self.rows, "row index out of bounds");
        let start = row * self.cols;
        let end = start + self.cols;
        &mut self.data[start..end]
    }

    /// Returns the number of rows (legacy helper).
    #[inline(always)]
    pub fn get_row(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns (legacy helper).
    #[inline(always)]
    pub fn get_col(&self) -> usize {
        self.cols
    }
}

impl<T> MatrixStorage for Vector2D<T>
where
    T: Copy + std::ops::AddAssign,
{
    type Counter = T;
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows()
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols()
    }

    #[inline(always)]
    fn update_one_counter<F, V>(&mut self, row: usize, col: usize, op: F, value: V)
    where
        F: Fn(&mut Self::Counter, V),
    {
        self.update_one_counter(row, col, op, value);
    }

    #[inline(always)]
    fn increment_by_row(&mut self, row: usize, col: usize, value: Self::Counter) {
        let idx = row * self.cols + col;
        self.data[idx] += value;
    }

    #[inline(always)]
    fn fast_insert<Hash, F, V>(&mut self, op: F, value: V, hashed_val: &Hash)
    where
        Hash: MatrixFastHash,
        F: Fn(&mut Self::Counter, &V, usize),
        V: Clone,
    {
        self.fast_insert(op, value, hashed_val);
    }

    #[inline(always)]
    fn fast_query_min<Hash, F, R>(&self, hashed_val: &Hash, op: F) -> R
    where
        Hash: MatrixFastHash,
        F: Fn(&Self::Counter, usize, &Hash) -> R,
        R: PartialOrd,
    {
        self.fast_query_min(hashed_val, op)
    }

    #[inline(always)]
    fn fast_query_median<Hash, F>(&self, hashed_val: &Hash, op: F) -> f64
    where
        Hash: MatrixFastHash,
        F: Fn(&Self::Counter, usize, &Hash) -> f64,
    {
        self.fast_query_median(hashed_val, op)
    }

    #[inline(always)]
    fn query_one_counter(&self, row: usize, col: usize) -> Self::Counter {
        self.query_one_counter(row, col)
    }
}

impl<T, H> FastPathHasher<H> for Vector2D<T>
where
    T: Copy + std::ops::AddAssign,
    H: SketchHasher,
{
    #[inline(always)]
    fn hash_for_matrix(&self, value: &DataInput) -> H::HashType {
        H::HashType::assert_compatible(self.rows, self.cols);
        H::hash_for_matrix_seeded(0, self.rows, self.cols, value)
    }
}

impl<T> Index<usize> for Vector2D<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        self.row_slice(index)
    }
}

impl<T> IndexMut<usize> for Vector2D<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.row_slice_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // placeholder, potentially useful
    #[test]
    fn required_bits_match_expected_thresholds() {
        let default_dims: Vector2D<u64> = Vector2D::init(3, 4096);
        assert_eq!(default_dims.get_required_bits(), 64);

        let smaller_cols: Vector2D<u64> = Vector2D::init(3, 64);
        assert_eq!(smaller_cols.get_required_bits(), 32);

        let larger_shape: Vector2D<u64> = Vector2D::init(5, 1_048_576);
        assert_eq!(larger_shape.get_required_bits(), 128);
    }

    /// End-to-end coverage of `Vector2D::col_for_row` — the cached-field
    /// decode path — with NON-power-of-two column counts. Every in-tree
    /// FastPath caller uses pow2 cols, so without this test the non-pow2
    /// branch of the decode is never executed at all.
    ///
    /// Each insert is mirrored into a reference matrix whose placements are
    /// computed independently (`shift → mask → % cols`), then both full
    /// matrices are compared.
    #[test]
    fn fast_insert_non_pow2_and_degenerate_cols_match_reference() {
        let mut lcg = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next_hash = move || {
            lcg ^= lcg << 13;
            lcg ^= lcg >> 7;
            lcg ^= lcg << 17;
            lcg
        };
        let reference_col = |hash: u64, row: usize, cols: usize| -> usize {
            let mask_bits = if cols.is_power_of_two() {
                cols.ilog2()
            } else {
                cols.ilog2() + 1
            } as usize;
            (((hash >> (mask_bits * row)) & ((1u64 << mask_bits) - 1)) % cols as u64) as usize
        };

        for &cols in &[1usize, 17, 100, 1000] {
            const ROWS: usize = 3;
            let mut m: Vector2D<i64> = Vector2D::from_fn(ROWS, cols, |_, _| 0);
            let mut reference = vec![0i64; ROWS * cols];

            for _ in 0..500 {
                let h = next_hash();
                m.fast_insert(
                    |c: &mut i64, _: &i64, _| *c += 1,
                    1i64,
                    &MatrixHashType::Packed64(h),
                );
                for row in 0..ROWS {
                    reference[row * cols + reference_col(h, row, cols)] += 1;
                }
            }

            assert_eq!(
                m.as_slice(),
                &reference[..],
                "cols={cols}: cached-field decode diverged from reference"
            );
        }
    }
}
