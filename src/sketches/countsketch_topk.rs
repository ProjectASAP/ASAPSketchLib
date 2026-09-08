//! Top-k heavy-hitter variants of the Count Sketch family.
//!
//! - [`CSHeap`]: pairs a [`Count`] sketch with an [`HHHeap`] for automatic top-k tracking.
//!   Every insertion updates both the frequency sketch and the heap, mirroring
//!   the pattern used by `CMSHeap` but with Count Sketch (median estimator).
//! - [`CountL2HH`]: Count Sketch augmented with per-row L2 norm tracking for
//!   heavy-hitter detection.

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::octo_delta::CountDelta;
use crate::sketches::countsketch::CountSketchCounter;
use crate::{
    Count, DataInput, DefaultMatrixI32, DefaultMatrixI64, DefaultMatrixI128, DefaultXxHasher,
    FastPath, FixedMatrix, HHHeap, MatrixStorage, QuickMatrixI64, QuickMatrixI128, RegularPath,
    SketchHasher, Vector1D, Vector2D, compute_median_inline_f64, heap_item_to_sketch_input,
};

pub(crate) mod l2hh_wire;
mod wire;

/// The count a `CSHeap` entry carries for a Count Sketch estimate.
///
/// `Count::estimate` returns the row **median** as `f64`, because a Count
/// Sketch estimate is signed and is a median rather than a counter read. The
/// heap stores `i64`, so a conversion is unavoidable and its semantics are part
/// of the API rather than an accident of a cast:
///
/// - **Saturating**, not wrapping. Rust's `f64 as i64` saturates at
///   `i64::MIN`/`i64::MAX` and maps `NaN` to `0`; this function names that
///   choice so it cannot be silently changed. An `i128`-backed sketch really
///   can hold counts past `i64::MAX` — `CSHeap<QuickMatrixI128>` and friends
///   accept `insert_many(key, i128)` — and those clamp here. A wrapped negative
///   count would corrupt the heap's ordering, which is exactly what saturation
///   prevents.
/// - **Truncating toward zero** on the fractional part a median of an even row
///   count can produce.
///
/// Above `2^53` the estimate has already lost precision in `f64` inside
/// `Count::estimate`, so an `i128` sketch's heap entries are exact only up to
/// that magnitude. Both limits are asserted in
/// `tests/e2e_matrix_instances.rs`.
#[inline]
pub fn cs_heap_count(estimate: f64) -> i64 {
    // `as` on a float is a saturating cast in Rust (since 1.45); this is the
    // documented behaviour, not a fallback.
    estimate as i64
}

const DEFAULT_TOP_K: usize = 32;
const DEFAULT_ROW_NUM: usize = 3;
const DEFAULT_COL_NUM: usize = 4096;

/// A Count Sketch paired with a top-k heavy-hitter heap.
///
/// Generic over the same type parameters as [`Count`].
#[derive(Clone)]
pub struct CSHeap<
    S: MatrixStorage = Vector2D<i64>,
    Mode = RegularPath,
    H: SketchHasher = DefaultXxHasher,
> {
    cs: Count<S, Mode, H>,
    heap: HHHeap,
}

// -- Construction for Vector2D-backed storage --------------------------------

impl<T, M, H: SketchHasher> CSHeap<Vector2D<T>, M, H>
where
    T: CountSketchCounter,
{
    /// Creates a new `CSHeap` with the given CS dimensions and heap capacity.
    pub fn new(rows: usize, cols: usize, top_k: usize) -> Self {
        CSHeap {
            cs: Count::with_dimensions(rows, cols),
            heap: HHHeap::new(top_k),
        }
    }
}

// -- Construction from any MatrixStorage -------------------------------------

impl<S: MatrixStorage, M, H: SketchHasher> CSHeap<S, M, H>
where
    S::Counter: CountSketchCounter,
{
    /// Creates a `CSHeap` from a pre-built storage backend.
    pub fn from_storage(storage: S, top_k: usize) -> Self {
        CSHeap {
            cs: Count::from_storage(storage),
            heap: HHHeap::new(top_k),
        }
    }
}

// -- Default impls -----------------------------------------------------------
//
// Default is available for Vector2D-backed sketches and fixed-size matrix
// backends (Quick/Default/Fixed families). Use `from_storage(...)` when you
// want explicit backend control with a custom `top_k`.

impl Default for CSHeap<Vector2D<i64>, RegularPath> {
    fn default() -> Self {
        Self::new(3, 4096, DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<Vector2D<i64>, FastPath> {
    fn default() -> Self {
        Self::new(3, 4096, DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<Vector2D<i32>, RegularPath> {
    fn default() -> Self {
        Self::new(3, 4096, DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<Vector2D<i32>, FastPath> {
    fn default() -> Self {
        Self::new(3, 4096, DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<FixedMatrix, RegularPath> {
    fn default() -> Self {
        Self::from_storage(FixedMatrix::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<FixedMatrix, FastPath> {
    fn default() -> Self {
        Self::from_storage(FixedMatrix::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<DefaultMatrixI32, RegularPath> {
    fn default() -> Self {
        Self::from_storage(DefaultMatrixI32::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<DefaultMatrixI32, FastPath> {
    fn default() -> Self {
        Self::from_storage(DefaultMatrixI32::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<QuickMatrixI64, RegularPath> {
    fn default() -> Self {
        Self::from_storage(QuickMatrixI64::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<QuickMatrixI64, FastPath> {
    fn default() -> Self {
        Self::from_storage(QuickMatrixI64::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<QuickMatrixI128, RegularPath> {
    fn default() -> Self {
        Self::from_storage(QuickMatrixI128::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<QuickMatrixI128, FastPath> {
    fn default() -> Self {
        Self::from_storage(QuickMatrixI128::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<DefaultMatrixI64, RegularPath> {
    fn default() -> Self {
        Self::from_storage(DefaultMatrixI64::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<DefaultMatrixI64, FastPath> {
    fn default() -> Self {
        Self::from_storage(DefaultMatrixI64::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<DefaultMatrixI128, RegularPath> {
    fn default() -> Self {
        Self::from_storage(DefaultMatrixI128::default(), DEFAULT_TOP_K)
    }
}

impl Default for CSHeap<DefaultMatrixI128, FastPath> {
    fn default() -> Self {
        Self::from_storage(DefaultMatrixI128::default(), DEFAULT_TOP_K)
    }
}

// -- Shared accessors (all storage types) ------------------------------------

impl<S: MatrixStorage, M, H: SketchHasher> CSHeap<S, M, H>
where
    S::Counter: CountSketchCounter,
{
    /// Returns a reference to the internal Count Sketch.
    pub fn cs(&self) -> &Count<S, M, H> {
        &self.cs
    }

    /// Returns a mutable reference to the internal Count Sketch.
    pub fn cs_mut(&mut self) -> &mut Count<S, M, H> {
        &mut self.cs
    }

    /// Returns a reference to the heavy-hitter heap.
    pub fn heap(&self) -> &HHHeap {
        &self.heap
    }

    /// Returns a mutable reference to the heavy-hitter heap.
    pub fn heap_mut(&mut self) -> &mut HHHeap {
        &mut self.heap
    }

    /// Number of rows in the underlying CS.
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.cs.rows()
    }

    /// Number of columns in the underlying CS.
    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.cs.cols()
    }

    /// Clears the heap.
    pub fn clear_heap(&mut self) {
        self.heap.clear();
    }
}

// -- RegularPath insert / estimate / merge -----------------------------------

impl<S: MatrixStorage, H: SketchHasher> CSHeap<S, RegularPath, H>
where
    S::Counter: CountSketchCounter,
{
    /// Inserts a single observation and updates the top-k heap.
    #[inline]
    pub fn insert(&mut self, key: &DataInput) {
        self.cs.insert(key);
        let est = self.cs.estimate(key);
        self.heap.update(key, cs_heap_count(est));
    }

    /// Inserts an observation with the given count and updates the top-k heap.
    #[inline]
    pub fn insert_many(&mut self, key: &DataInput, many: S::Counter) {
        self.cs.insert_many(key, many);
        let est = self.cs.estimate(key);
        self.heap.update(key, cs_heap_count(est));
    }

    /// Inserts a batch of observations, updating the heap after each.
    pub fn bulk_insert(&mut self, values: &[DataInput]) {
        for value in values {
            self.insert(value);
        }
    }

    /// Returns the CS frequency estimate (median) for the given key.
    #[inline]
    pub fn estimate(&self, key: &DataInput) -> f64 {
        self.cs.estimate(key)
    }

    /// Merges another `CSHeap` into `self`.
    ///
    /// After merging the CS counters, all heap items from both sources are
    /// re-queried against the merged sketch to reconcile the top-k heap.
    pub fn merge(&mut self, other: &Self) {
        self.cs.merge(&other.cs);
        let mut candidate_keys = Vec::with_capacity(self.heap.len() + other.heap.len());
        for item in self.heap.heap() {
            candidate_keys.push(item.key.clone());
        }
        for item in other.heap.heap() {
            candidate_keys.push(item.key.clone());
        }
        self.heap.clear();
        for key in candidate_keys {
            let key_ref = heap_item_to_sketch_input(&key);
            let est = self.cs.estimate(&key_ref);
            self.heap.update(&key_ref, cs_heap_count(est));
        }
    }
}

// -- FastPath insert / estimate / merge --------------------------------------

impl<S, H: SketchHasher> CSHeap<S, FastPath, H>
where
    S: MatrixStorage + crate::FastPathHasher<H>,
    S::Counter: CountSketchCounter,
{
    /// Inserts a single observation using fast-path hashing and updates the heap.
    #[inline]
    pub fn insert(&mut self, key: &DataInput) {
        self.cs.insert(key);
        let est = self.cs.estimate(key);
        self.heap.update(key, cs_heap_count(est));
    }

    /// Inserts an observation with the given count using fast-path hashing.
    #[inline]
    pub fn insert_many(&mut self, key: &DataInput, many: S::Counter) {
        self.cs.insert_many(key, many);
        let est = self.cs.estimate(key);
        self.heap.update(key, cs_heap_count(est));
    }

    /// Inserts a batch of observations using fast-path hashing.
    pub fn bulk_insert(&mut self, values: &[DataInput]) {
        for value in values {
            self.insert(value);
        }
    }

    /// Returns the CS frequency estimate (median) using fast-path hashing.
    #[inline]
    pub fn estimate(&self, key: &DataInput) -> f64 {
        self.cs.estimate(key)
    }

    /// Merges another `CSHeap` into `self`.
    pub fn merge(&mut self, other: &Self) {
        self.cs.merge(&other.cs);
        let mut candidate_keys = Vec::with_capacity(self.heap.len() + other.heap.len());
        for item in self.heap.heap() {
            candidate_keys.push(item.key.clone());
        }
        for item in other.heap.heap() {
            candidate_keys.push(item.key.clone());
        }
        self.heap.clear();
        for key in candidate_keys {
            let key_ref = heap_item_to_sketch_input(&key);
            let est = self.cs.estimate(&key_ref);
            self.heap.update(&key_ref, cs_heap_count(est));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataInput;
    use crate::test_utils::sample_zipf_u64;
    use std::collections::{HashMap, HashSet};

    fn heap_count_for_key(heap: &HHHeap, key: &DataInput) -> Option<i64> {
        heap.heap()
            .iter()
            .find(|item| heap_item_to_sketch_input(&item.key) == *key)
            .map(|item| item.count)
    }

    fn run_zipf_stream_regular(
        rows: usize,
        cols: usize,
        top_k: usize,
        domain: usize,
        exponent: f64,
        samples: usize,
        seed: u64,
    ) -> (CSHeap<Vector2D<i64>, RegularPath>, HashMap<u64, i64>) {
        let mut truth = HashMap::<u64, i64>::new();
        let mut sketch = CSHeap::<Vector2D<i64>, RegularPath>::new(rows, cols, top_k);
        for value in sample_zipf_u64(domain, exponent, samples, seed) {
            let key = DataInput::U64(value);
            sketch.insert(&key);
            *truth.entry(value).or_insert(0) += 1;
        }
        (sketch, truth)
    }

    fn run_zipf_stream_fast(
        rows: usize,
        cols: usize,
        top_k: usize,
        domain: usize,
        exponent: f64,
        samples: usize,
        seed: u64,
    ) -> (CSHeap<Vector2D<i64>, FastPath>, HashMap<u64, i64>) {
        let mut truth = HashMap::<u64, i64>::new();
        let mut sketch = CSHeap::<Vector2D<i64>, FastPath>::new(rows, cols, top_k);
        for value in sample_zipf_u64(domain, exponent, samples, seed) {
            let key = DataInput::U64(value);
            sketch.insert(&key);
            *truth.entry(value).or_insert(0) += 1;
        }
        (sketch, truth)
    }

    fn top_k_truth_keys(truth: &HashMap<u64, i64>, k: usize) -> HashSet<u64> {
        let mut entries: Vec<(u64, i64)> =
            truth.iter().map(|(key, count)| (*key, *count)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries.into_iter().take(k).map(|(key, _)| key).collect()
    }

    fn top_k_heap_keys(heap: &HHHeap) -> HashSet<u64> {
        heap.heap()
            .iter()
            .map(|item| match heap_item_to_sketch_input(&item.key) {
                DataInput::U64(v) => v,
                other => panic!("expected U64 key in zipf tests, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn insert_and_estimate() {
        // Verifies single-key inserts update both CS estimate and wrapper behavior.
        let mut sh = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 256, 10);
        let key = DataInput::Str("hello");
        for _ in 0..5 {
            sh.insert(&key);
        }
        assert!((sh.estimate(&key) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn heap_tracks_top_k() {
        // Verifies heap retains only the highest-frequency keys under top-k limit.
        let mut sh = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 1024, 3);

        // Insert 5 distinct keys with different frequencies.
        for i in 1..=5u64 {
            let key = DataInput::U64(i);
            for _ in 0..(i * 100) {
                sh.insert(&key);
            }
        }

        // Heap should contain at most 3 items (top-3).
        assert!(sh.heap().len() <= 3);

        // The top-3 counts should be 300, 400, 500.
        let mut counts: Vec<i64> = sh.heap().heap().iter().map(|item| item.count).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![300, 400, 500]);
    }

    #[test]
    fn merge_reconciles_heaps() {
        // Verifies merge combines sketches and refreshes heap counts from merged state.
        let mut a = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 256, 5);
        let mut b = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 256, 5);

        let key = DataInput::Str("merge_key");
        for _ in 0..10 {
            a.insert(&key);
        }
        for _ in 0..20 {
            b.insert(&key);
        }

        a.merge(&b);

        // After merge the estimate should be the sum.
        assert!((a.estimate(&key) - 30.0).abs() < 1e-9);

        // The heap should reflect the merged estimate.
        let heap_item = a
            .heap()
            .heap()
            .iter()
            .find(|item| {
                let k = heap_item_to_sketch_input(&item.key);
                k == key
            })
            .expect("key should be in heap");
        assert_eq!(heap_item.count, 30);
    }

    #[test]
    fn insert_many_updates_estimate_and_heap() {
        // Verifies insert_many updates estimate and heap count consistently.
        let mut sh = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 2048, 4);
        let key = DataInput::Str("many");
        sh.insert_many(&key, 17);

        let estimate = sh.estimate(&key);
        assert!((estimate - 17.0).abs() < 1e-9);
        assert_eq!(heap_count_for_key(sh.heap(), &key), Some(estimate as i64));
    }

    #[test]
    fn bulk_insert_updates_multiple_keys() {
        // Verifies bulk stream ingestion updates multiple keys and heap tracking.
        let mut sh = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 2048, 4);
        let values = vec![
            DataInput::U64(7),
            DataInput::U64(8),
            DataInput::U64(7),
            DataInput::U64(9),
            DataInput::U64(7),
        ];
        sh.bulk_insert(&values);

        let key = DataInput::U64(7);
        assert!((sh.estimate(&key) - 3.0).abs() < 1e-9);
        assert_eq!(
            heap_count_for_key(sh.heap(), &key),
            Some(sh.estimate(&key) as i64)
        );
    }

    #[test]
    fn clear_heap_keeps_cs_counters() {
        // Verifies clearing heap preserves sketch counters for future updates.
        let mut sh = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 2048, 2);
        let key = DataInput::Str("persist");
        sh.insert_many(&key, 5);

        sh.clear_heap();
        assert!(sh.heap().is_empty());
        assert!((sh.estimate(&key) - 5.0).abs() < 1e-9);

        sh.insert(&key);
        assert_eq!(
            heap_count_for_key(sh.heap(), &key),
            Some(sh.estimate(&key) as i64)
        );
    }

    #[test]
    fn from_storage_uses_storage_dimensions() {
        // Verifies from_storage keeps backend dimensions and heap capacity.
        let storage = Vector2D::<i64>::init(4, 128);
        let sh = CSHeap::<Vector2D<i64>, RegularPath>::from_storage(storage, 9);

        assert_eq!(sh.rows(), 4);
        assert_eq!(sh.cols(), 128);
        assert_eq!(sh.heap().capacity(), 9);
    }

    #[test]
    fn merge_refreshes_existing_self_heap_entries() {
        // Verifies merge refreshes pre-existing self heap keys to merged estimates.
        let mut a = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 2);
        let mut b = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 1);
        let a_key = DataInput::Str("a-key");
        let c_key = DataInput::Str("c-key");
        let b_key = DataInput::Str("b-key");

        a.insert_many(&a_key, 120);
        a.insert_many(&c_key, 10);

        b.insert_many(&a_key, 40);
        b.insert_many(&b_key, 400);

        a.merge(&b);

        let merged_estimate = a.estimate(&a_key) as i64;
        assert_eq!(heap_count_for_key(a.heap(), &a_key), Some(merged_estimate));
    }

    #[test]
    fn fast_path_insert_and_estimate() {
        // Verifies FastPath insert and estimate stay coherent for repeated keys.
        let mut sh = CSHeap::<Vector2D<i64>, FastPath>::new(5, 256, 10);
        let key = DataInput::Str("fast");
        for _ in 0..7 {
            sh.insert(&key);
        }
        assert!((sh.estimate(&key) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn fast_path_insert_many_and_bulk_insert() {
        // Verifies FastPath batched APIs maintain estimate/heap consistency.
        let mut sh = CSHeap::<Vector2D<i64>, FastPath>::new(5, 2048, 4);
        let key = DataInput::Str("fast-many");
        sh.insert_many(&key, 6);
        sh.bulk_insert(&[
            DataInput::Str("fast-many"),
            DataInput::Str("another"),
            DataInput::Str("fast-many"),
        ]);

        let estimate = sh.estimate(&key);
        assert!((estimate - 8.0).abs() < 1e-9);
        assert_eq!(heap_count_for_key(sh.heap(), &key), Some(estimate as i64));
    }

    #[test]
    fn fast_path_heap_tracks_top_k() {
        // Verifies FastPath top-k maintenance under weighted updates.
        let mut sh = CSHeap::<Vector2D<i64>, FastPath>::new(5, 4096, 3);

        for i in 1..=5u64 {
            let key = DataInput::U64(i);
            sh.insert_many(&key, (i as i64) * 100);
        }

        let mut counts: Vec<i64> = sh.heap().heap().iter().map(|item| item.count).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![300, 400, 500]);
    }

    #[test]
    fn fast_path_merge_refreshes_existing_self_heap_entries() {
        // Verifies FastPath merge refreshes self heap counts from merged sketch.
        let mut a = CSHeap::<Vector2D<i64>, FastPath>::new(5, 4096, 2);
        let mut b = CSHeap::<Vector2D<i64>, FastPath>::new(5, 4096, 1);
        let a_key = DataInput::Str("a-fast");
        let c_key = DataInput::Str("c-fast");
        let b_key = DataInput::Str("b-fast");

        a.insert_many(&a_key, 120);
        a.insert_many(&c_key, 10);

        b.insert_many(&a_key, 40);
        b.insert_many(&b_key, 400);

        a.merge(&b);

        let merged_estimate = a.estimate(&a_key) as i64;
        assert_eq!(heap_count_for_key(a.heap(), &a_key), Some(merged_estimate));
    }

    #[test]
    fn default_construction() {
        // Verifies default CSHeap dimensions and default top-k capacity.
        let sh = CSHeap::<Vector2D<i64>, RegularPath>::default();
        assert_eq!(sh.rows(), 3);
        assert_eq!(sh.cols(), 4096);
        assert_eq!(sh.heap().capacity(), DEFAULT_TOP_K);
    }

    #[test]
    fn default_construction_fixed_backends_parity() {
        // Verifies default construction parity across all supported backends.
        let fixed_regular = CSHeap::<FixedMatrix, RegularPath>::default();
        assert_eq!(fixed_regular.rows(), 5);
        assert_eq!(fixed_regular.cols(), 2048);
        assert_eq!(fixed_regular.heap().capacity(), DEFAULT_TOP_K);

        let fixed_fast = CSHeap::<FixedMatrix, FastPath>::default();
        assert_eq!(fixed_fast.rows(), 5);
        assert_eq!(fixed_fast.cols(), 2048);
        assert_eq!(fixed_fast.heap().capacity(), DEFAULT_TOP_K);

        let dm_i32_regular = CSHeap::<DefaultMatrixI32, RegularPath>::default();
        assert_eq!(dm_i32_regular.rows(), 3);
        assert_eq!(dm_i32_regular.cols(), 4096);
        assert_eq!(dm_i32_regular.heap().capacity(), DEFAULT_TOP_K);

        let dm_i32_fast = CSHeap::<DefaultMatrixI32, FastPath>::default();
        assert_eq!(dm_i32_fast.rows(), 3);
        assert_eq!(dm_i32_fast.cols(), 4096);
        assert_eq!(dm_i32_fast.heap().capacity(), DEFAULT_TOP_K);

        let qm_i64_regular = CSHeap::<QuickMatrixI64, RegularPath>::default();
        assert_eq!(qm_i64_regular.rows(), 5);
        assert_eq!(qm_i64_regular.cols(), 2048);
        assert_eq!(qm_i64_regular.heap().capacity(), DEFAULT_TOP_K);

        let qm_i64_fast = CSHeap::<QuickMatrixI64, FastPath>::default();
        assert_eq!(qm_i64_fast.rows(), 5);
        assert_eq!(qm_i64_fast.cols(), 2048);
        assert_eq!(qm_i64_fast.heap().capacity(), DEFAULT_TOP_K);

        let qm_i128_regular = CSHeap::<QuickMatrixI128, RegularPath>::default();
        assert_eq!(qm_i128_regular.rows(), 5);
        assert_eq!(qm_i128_regular.cols(), 2048);
        assert_eq!(qm_i128_regular.heap().capacity(), DEFAULT_TOP_K);

        let qm_i128_fast = CSHeap::<QuickMatrixI128, FastPath>::default();
        assert_eq!(qm_i128_fast.rows(), 5);
        assert_eq!(qm_i128_fast.cols(), 2048);
        assert_eq!(qm_i128_fast.heap().capacity(), DEFAULT_TOP_K);

        let dm_i64_regular = CSHeap::<DefaultMatrixI64, RegularPath>::default();
        assert_eq!(dm_i64_regular.rows(), 3);
        assert_eq!(dm_i64_regular.cols(), 4096);
        assert_eq!(dm_i64_regular.heap().capacity(), DEFAULT_TOP_K);

        let dm_i64_fast = CSHeap::<DefaultMatrixI64, FastPath>::default();
        assert_eq!(dm_i64_fast.rows(), 3);
        assert_eq!(dm_i64_fast.cols(), 4096);
        assert_eq!(dm_i64_fast.heap().capacity(), DEFAULT_TOP_K);

        let dm_i128_regular = CSHeap::<DefaultMatrixI128, RegularPath>::default();
        assert_eq!(dm_i128_regular.rows(), 3);
        assert_eq!(dm_i128_regular.cols(), 4096);
        assert_eq!(dm_i128_regular.heap().capacity(), DEFAULT_TOP_K);

        let dm_i128_fast = CSHeap::<DefaultMatrixI128, FastPath>::default();
        assert_eq!(dm_i128_fast.rows(), 3);
        assert_eq!(dm_i128_fast.cols(), 4096);
        assert_eq!(dm_i128_fast.heap().capacity(), DEFAULT_TOP_K);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch while merging CountMin sketches")]
    fn merge_requires_matching_dimensions_panics() {
        // Verifies merge panics when dimensions differ.
        let mut left = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 256, 4);
        let right = CSHeap::<Vector2D<i64>, RegularPath>::new(6, 256, 4);
        left.merge(&right);
    }

    #[test]
    fn heap_entries_match_cs_estimates_after_mutations() {
        // Verifies heap entries always match current sketch estimates after mutations.
        let mut sh = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 4);
        sh.insert_many(&DataInput::Str("a"), 100);
        sh.insert_many(&DataInput::Str("b"), 70);
        sh.bulk_insert(&[
            DataInput::Str("a"),
            DataInput::Str("c"),
            DataInput::Str("a"),
            DataInput::Str("d"),
        ]);

        for item in sh.heap().heap() {
            let key = heap_item_to_sketch_input(&item.key);
            assert_eq!(item.count, sh.estimate(&key) as i64);
        }

        let mut other = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 4);
        other.insert_many(&DataInput::Str("b"), 90);
        other.insert_many(&DataInput::Str("e"), 200);
        sh.merge(&other);

        for item in sh.heap().heap() {
            let key = heap_item_to_sketch_input(&item.key);
            assert_eq!(item.count, sh.estimate(&key) as i64);
        }
    }

    #[test]
    fn bulk_insert_equivalent_to_repeated_insert() {
        // Verifies bulk_insert behavior matches repeated insert behavior.
        let values = vec![
            DataInput::U64(1),
            DataInput::U64(2),
            DataInput::U64(1),
            DataInput::U64(3),
            DataInput::U64(2),
            DataInput::U64(1),
            DataInput::U64(4),
            DataInput::U64(2),
            DataInput::U64(5),
        ];

        let mut via_bulk = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 3);
        via_bulk.bulk_insert(&values);

        let mut via_repeat = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 3);
        for value in &values {
            via_repeat.insert(value);
        }

        for key in [1_u64, 2, 3, 4, 5] {
            let k = DataInput::U64(key);
            assert!((via_bulk.estimate(&k) - via_repeat.estimate(&k)).abs() < 1e-9);
            assert_eq!(
                heap_count_for_key(via_bulk.heap(), &k),
                heap_count_for_key(via_repeat.heap(), &k)
            );
        }
    }

    #[test]
    fn regular_vs_fast_equivalence_on_same_stream() {
        // Verifies regular and fast wrapper paths match on a short deterministic stream.
        let values = vec![
            DataInput::Str("alpha"),
            DataInput::Str("beta"),
            DataInput::Str("alpha"),
            DataInput::Str("gamma"),
            DataInput::Str("beta"),
            DataInput::Str("alpha"),
            DataInput::Str("delta"),
            DataInput::Str("gamma"),
            DataInput::Str("epsilon"),
            DataInput::Str("alpha"),
        ];

        let mut regular = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 3);
        let mut fast = CSHeap::<Vector2D<i64>, FastPath>::new(5, 4096, 3);
        for value in &values {
            regular.insert(value);
            fast.insert(value);
        }

        for key in ["alpha", "beta", "gamma", "delta", "epsilon"] {
            let k = DataInput::Str(key);
            assert!((regular.estimate(&k) - fast.estimate(&k)).abs() < 1e-9);
            assert_eq!(
                heap_count_for_key(regular.heap(), &k),
                heap_count_for_key(fast.heap(), &k)
            );
        }
    }

    #[test]
    fn merge_with_empty_other_and_empty_self() {
        // Verifies merge behavior is stable when one side is empty.
        let mut non_empty = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 2048, 3);
        non_empty.insert_many(&DataInput::Str("x"), 110);
        non_empty.insert_many(&DataInput::Str("y"), 50);

        let empty = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 2048, 3);
        let before_len = non_empty.heap().len();
        let before_x = non_empty.estimate(&DataInput::Str("x"));
        non_empty.merge(&empty);
        assert_eq!(non_empty.heap().len(), before_len);
        assert!((non_empty.estimate(&DataInput::Str("x")) - before_x).abs() < 1e-9);

        let mut empty_self = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 2048, 3);
        empty_self.merge(&non_empty);
        assert!((empty_self.estimate(&DataInput::Str("x")) - before_x).abs() < 1e-9);
        assert!(heap_count_for_key(empty_self.heap(), &DataInput::Str("x")).is_some());
    }

    #[test]
    fn duplicate_candidate_keys_during_merge_do_not_corrupt_heap() {
        // Verifies duplicate merge candidates do not duplicate heap entries.
        let mut left = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 4);
        let mut right = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4096, 4);

        left.insert_many(&DataInput::Str("dup"), 100);
        left.insert_many(&DataInput::Str("left-only"), 70);

        right.insert_many(&DataInput::Str("dup"), 90);
        right.insert_many(&DataInput::Str("right-only"), 60);

        left.merge(&right);

        let merged_estimate = left.estimate(&DataInput::Str("dup")) as i64;
        let dup_count = heap_count_for_key(left.heap(), &DataInput::Str("dup"));
        assert_eq!(dup_count, Some(merged_estimate));
        assert!(left.heap().len() <= left.heap().capacity());

        let dup_entries = left
            .heap()
            .heap()
            .iter()
            .filter(|item| heap_item_to_sketch_input(&item.key) == DataInput::Str("dup"))
            .count();
        assert_eq!(dup_entries, 1);
    }

    #[test]
    fn zipf_stream_top_k_recall_regular_fast_budget() {
        // Verifies regular-path heap captures most true heavy hitters under a Zipf stream.
        let rows = 5;
        let cols = 4096;
        let top_k = 16;
        let (sketch, truth) =
            run_zipf_stream_regular(rows, cols, top_k, 1024, 1.1, 20_000, 0x5eed_c0de);

        assert!(sketch.heap().len() <= top_k);
        for item in sketch.heap().heap() {
            let key = heap_item_to_sketch_input(&item.key);
            assert_eq!(item.count, sketch.estimate(&key) as i64);
        }

        let truth_top = top_k_truth_keys(&truth, top_k);
        let heap_top = top_k_heap_keys(sketch.heap());
        let recall_hits = truth_top.intersection(&heap_top).count();
        assert!(
            recall_hits >= 15,
            "top-k recall too low: hits={recall_hits}, truth_top={truth_top:?}, heap_top={heap_top:?}"
        );
    }

    #[test]
    fn zipf_stream_top_k_recall_fast_path_fast_budget() {
        // Verifies fast-path heap captures most true heavy hitters under a Zipf stream.
        let rows = 5;
        let cols = 4096;
        let top_k = 16;
        let (sketch, truth) =
            run_zipf_stream_fast(rows, cols, top_k, 1024, 1.1, 20_000, 0x5eed_c0de);

        assert!(sketch.heap().len() <= top_k);
        for item in sketch.heap().heap() {
            let key = heap_item_to_sketch_input(&item.key);
            assert_eq!(item.count, sketch.estimate(&key) as i64);
        }

        let truth_top = top_k_truth_keys(&truth, top_k);
        let heap_top = top_k_heap_keys(sketch.heap());
        let recall_hits = truth_top.intersection(&heap_top).count();
        assert!(
            recall_hits >= 15,
            "top-k recall too low: hits={recall_hits}, truth_top={truth_top:?}, heap_top={heap_top:?}"
        );
    }

    #[test]
    fn zipf_stream_regular_fast_heap_overlap() {
        // Verifies regular and fast paths produce highly overlapping top-k heaps on Zipf input.
        let rows = 5;
        let cols = 4096;
        let top_k = 16;
        let stream = sample_zipf_u64(1024, 1.1, 20_000, 0xABCD_1234);

        let mut regular = CSHeap::<Vector2D<i64>, RegularPath>::new(rows, cols, top_k);
        let mut fast = CSHeap::<Vector2D<i64>, FastPath>::new(rows, cols, top_k);
        for value in &stream {
            let key = DataInput::U64(*value);
            regular.insert(&key);
            fast.insert(&key);
        }

        let regular_heap_keys = top_k_heap_keys(regular.heap());
        let fast_heap_keys = top_k_heap_keys(fast.heap());
        let overlap = regular_heap_keys.intersection(&fast_heap_keys).count();
        assert!(
            (overlap as f64) / (top_k as f64) >= 0.8,
            "heap overlap too low: overlap={overlap}, regular={regular_heap_keys:?}, fast={fast_heap_keys:?}"
        );
    }
}

// =====================================================================
// CountL2HH — Count Sketch with per-row L2 norm tracking for
// heavy-hitter detection. Moved here from `count.rs` so the algorithm
// file stays focused on plain Count Sketch.
// =====================================================================

/// Column and sign `CountL2HH` uses for `row`, given its 128-bit key hash.
///
/// Shared with the OctoSketch worker sketches so a worker and its parent can
/// never disagree about which cell a key lands in.
#[inline(always)]
pub fn l2hh_cell_for_row(
    hashed_val: u128,
    row: usize,
    cols: usize,
    mask_bits: u32,
) -> (usize, i64) {
    let mask = (1u128 << mask_bits) - 1;
    let hashed = (hashed_val >> (row * mask_bits as usize)) & mask;
    let col = (hashed as usize) % cols;
    let bit = ((hashed_val >> (127 - row)) & 1) as i64;
    (col, -(1 - 2 * bit))
}

/// Count Sketch augmented with per-row L2 norm tracking for heavy-hitter detection.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "")]
pub struct CountL2HH<H: SketchHasher = DefaultXxHasher> {
    counts: Vector2D<i64>,
    l2: Vector1D<i64>,
    row: usize,
    col: usize,
    seed_idx: usize,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

// Default CountL2HH configuration.
impl Default for CountL2HH {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM)
    }
}

// CountL2HH constructors and operations.
impl<H: SketchHasher> CountL2HH<H> {
    /// Creates a sketch with the requested number of rows and columns.
    pub fn with_dimensions(rows: usize, cols: usize) -> Self {
        Self::with_dimensions_and_seed(rows, cols, 0)
    }

    /// Creates a sketch with the requested dimensions and a custom seed offset.
    pub fn with_dimensions_and_seed(rows: usize, cols: usize, seed_idx: usize) -> Self {
        let mut sk = CountL2HH {
            counts: Vector2D::init(rows, cols),
            l2: Vector1D::filled(rows, 0),
            row: rows,
            col: cols,
            seed_idx,
            _hasher: PhantomData,
        };
        sk.counts.fill(0);
        sk
    }

    /// Number of rows in the sketch.
    pub fn rows(&self) -> usize {
        self.row
    }

    /// Number of columns in the sketch.
    pub fn cols(&self) -> usize {
        self.col
    }

    /// Exposes the backing matrix for inspection/testing.
    pub fn as_storage(&self) -> &Vector2D<i64> {
        &self.counts
    }

    /// Mutable access used internally for testing scenarios.
    pub fn as_storage_mut(&mut self) -> &mut Vector2D<i64> {
        &mut self.counts
    }

    /// Merges another sketch while asserting compatible dimensions.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(
            (self.row, self.col),
            (other.row, other.col),
            "dimension mismatch while merging CountL2HH sketches"
        );
        assert_eq!(
            self.seed_idx, other.seed_idx,
            "seed mismatch while merging CountL2HH sketches"
        );

        for i in 0..self.row {
            let mut row_l2 = 0_i128;
            for j in 0..self.col {
                self.counts[i][j] += other.counts[i][j];
                let counter = self.counts[i][j] as i128;
                row_l2 = row_l2.saturating_add(counter.saturating_mul(counter));
            }
            self.l2[i] = row_l2.min(i64::MAX as i128) as i64;
        }
    }

    /// Resets all counters and L2 accumulators to zero without reallocating.
    pub fn clear(&mut self) {
        self.counts.fill(0);
        self.l2.fill(0);
    }

    /// Inserts with hash optimization - computes hash once and reuses it.
    /// due to the limitation of seeds, use fast_insert only
    pub fn fast_insert_with_count(&mut self, val: &DataInput, c: i64) {
        let hashed_val = H::hash128_seeded(self.seed_idx, val);
        self.fast_insert_with_count_and_hash(hashed_val, c);
    }

    /// Inserts with hash optimization using precomputed hash value.
    pub fn fast_insert_with_count_and_hash(&mut self, hashed_val: u128, c: i64) {
        let mask_bits = self.counts.get_mask_bits();
        for i in 0..self.row {
            let (idx, sign_bit) = l2hh_cell_for_row(hashed_val, i, self.col, mask_bits);
            let old_value = self.counts.query_one_counter(i, idx);
            self.write_counter(i, idx, old_value + sign_bit * c);
        }
    }

    /// Writes one counter and carries the row's L2 accumulator with it.
    ///
    /// Saturating i128 intermediate so extreme counts degrade gracefully
    /// instead of wrapping (matches merge-path semantics, which also clamps to
    /// [0, i64::MAX]: L2^2 can never go negative, even when a decrement follows
    /// prior saturation).
    #[inline(always)]
    fn write_counter(&mut self, row: usize, col: usize, new_value: i64) {
        let old_value = self.counts.query_one_counter(row, col);
        self.counts[row][col] = new_value;
        let old_l2 = self.l2.as_slice()[row] as i128;
        let new_l2 = old_l2 + (new_value as i128) * (new_value as i128)
            - (old_value as i128) * (old_value as i128);
        self.l2[row] = new_l2.clamp(0, i64::MAX as i128) as i64;
    }

    /// Seed offset this sketch hashes with; UnivMon gives each layer its own.
    pub fn seed_idx(&self) -> usize {
        self.seed_idx
    }

    /// Column-mask width used to slice the 128-bit hash into per-row indices.
    pub fn mask_bits(&self) -> u32 {
        self.counts.get_mask_bits()
    }

    /// Applies a counter delta promoted by an OctoSketch worker.
    ///
    /// The row's L2 accumulator is carried along exactly, so a parent fed only
    /// deltas reports the same `get_l2` as one fed the raw stream.
    pub fn apply_delta(&mut self, delta: CountDelta) {
        let (row, col) = (delta.row as usize, delta.col as usize);
        let new_value = self.counts.query_one_counter(row, col) + delta.value as i64;
        self.write_counter(row, col, new_value);
    }

    /// Inserts without L2 update using precomputed hash value.
    pub fn fast_insert_with_count_without_l2_and_hash(&mut self, hashed_val: u128, c: i64) {
        let mask_bits = self.counts.get_mask_bits() as usize;
        let mask = (1u128 << mask_bits) - 1;
        let mut shift_amount = 0;
        let mut sign_bit_pos = 127;

        for i in 0..self.row {
            let hashed = (hashed_val >> shift_amount) & mask;
            let idx = (hashed as usize) % self.col;
            let bit = ((hashed_val >> sign_bit_pos) & 1) as i64;
            let sign_bit = -(1 - 2 * bit);

            self.counts[i][idx] += sign_bit * c;

            shift_amount += mask_bits;
            sign_bit_pos -= 1;
        }
    }

    /// Update and estimate with hash optimization.
    /// due to the limitation of seeds, use fast_insert only
    pub fn fast_update_and_est(&mut self, val: &DataInput, c: i64) -> f64 {
        let hashed_val = H::hash128_seeded(self.seed_idx, val);
        self.fast_insert_with_count_and_hash(hashed_val, c);
        self.fast_get_est_with_hash(hashed_val)
    }

    /// Update and estimate without L2 with hash optimization.
    /// due to the limitation of seeds, use fast_insert only
    pub fn fast_update_and_est_without_l2(&mut self, val: &DataInput, c: i64) -> f64 {
        let hashed_val = H::hash128_seeded(self.seed_idx, val);
        self.fast_insert_with_count_without_l2_and_hash(hashed_val, c);
        self.fast_get_est_with_hash(hashed_val)
    }

    /// Returns the median estimate of the squared L2 norm across rows.
    pub fn get_l2_sqr(&self) -> f64 {
        let mut values: Vec<f64> = self.l2.as_slice()[..self.row]
            .iter()
            .map(|&v| v as f64)
            .collect();
        compute_median_inline_f64(&mut values)
    }

    /// Returns the estimated L2 norm (square root of `get_l2_sqr`).
    pub fn get_l2(&self) -> f64 {
        let l2 = self.get_l2_sqr();
        l2.sqrt()
    }

    /// Returns the frequency estimate with hash optimization.
    /// due to the limitation of seeds, use fast_insert only
    pub fn fast_get_est(&self, val: &DataInput) -> f64 {
        let hashed_val = H::hash128_seeded(self.seed_idx, val);
        self.fast_get_est_with_hash(hashed_val)
    }

    /// Returns the frequency estimate using precomputed hash value.
    /// due to the limitation of seeds, use fast_insert only
    pub fn fast_get_est_with_hash(&self, hashed_val: u128) -> f64 {
        let mask_bits = self.counts.get_mask_bits() as usize;
        let mask = (1u128 << mask_bits) - 1;
        let mut lst = Vec::with_capacity(self.row);
        let mut shift_amount = 0;
        let mut sign_bit_pos = 127;

        for i in 0..self.row {
            let hashed = (hashed_val >> shift_amount) & mask;
            let idx = (hashed as usize) % self.col;
            let bit = ((hashed_val >> sign_bit_pos) & 1) as i64;
            let sign_bit = -(1 - 2 * bit);
            let counter = self.counts.query_one_counter(i, idx);
            lst.push((sign_bit * counter) as f64);

            shift_amount += mask_bits;
            sign_bit_pos -= 1;
        }
        compute_median_inline_f64(&mut lst[..])
    }
}

#[cfg(test)]
mod tests_count_l2_hh {
    use super::*;

    #[test]
    fn countl2hh_estimates_and_l2_are_consistent() {
        let mut sketch: CountL2HH = CountL2HH::with_dimensions(3, 32);
        let key = DataInput::Str("gamma");

        let est_after_first = sketch.fast_update_and_est(&key, 5);
        assert_eq!(est_after_first, 5.0);

        let est_after_second = sketch.fast_update_and_est(&key, -2);
        assert_eq!(est_after_second, 3.0);

        let l2 = sketch.get_l2();
        assert!(l2 >= 3.0, "expected non-trivial l2, got {l2}");
    }

    #[test]
    fn countl2hh_merge_combines_frequency_vectors() {
        let mut left: CountL2HH = CountL2HH::with_dimensions(3, 32);
        let mut right: CountL2HH = CountL2HH::with_dimensions(3, 32);
        let key = DataInput::U32(42);

        left.fast_insert_with_count(&key, 4);
        assert_eq!(left.fast_get_est(&key), 4.0);
        right.fast_insert_with_count(&key, 9);
        assert_eq!(right.fast_get_est(&key), 9.0);

        left.merge(&right);
        assert_eq!(left.fast_get_est(&key), 13.0);
        assert_eq!(left.get_l2(), 13.0);
    }

    #[test]
    fn countl2hh_round_trip_serialization() {
        let mut sketch: CountL2HH = CountL2HH::with_dimensions_and_seed(3, 32, 7);
        let key = DataInput::Str("serialize");

        sketch.fast_insert_with_count(&key, 11);
        sketch.fast_insert_with_count(&key, -3);
        let base_est = sketch.fast_get_est(&key);
        let base_l2 = sketch.get_l2();

        let encoded = sketch
            .serialize_to_bytes()
            .expect("serialize CountL2HH into MessagePack");
        assert!(!encoded.is_empty(), "serialized bytes should not be empty");
        let data = encoded.clone();

        let decoded: CountL2HH = CountL2HH::deserialize_from_bytes(&data)
            .expect("deserialize CountL2HH from MessagePack");

        assert_eq!(sketch.rows(), decoded.rows());
        assert_eq!(sketch.cols(), decoded.cols());
        assert!(
            (decoded.fast_get_est(&key) - base_est).abs() < f64::EPSILON,
            "estimate changed after round trip"
        );
        assert!(
            (decoded.get_l2() - base_l2).abs() < f64::EPSILON,
            "L2 changed after round trip"
        );
    }
}
