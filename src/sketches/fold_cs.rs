//! Folding Count Sketch (FoldCS)
//!
//! A memory-efficient Count Sketch variant for sub-window aggregation. Instead
//! of allocating the full W columns required by the final merged query, each
//! sub-window uses only W/2^k physical columns (where k is the fold level).
//!
//! Cells lazily expand using the same [`FoldCell`] / `FoldEntry` types as
//! `FoldCMS`. The key difference from FoldCMS is that FoldCS uses **signed
//! counters** (each row has a random ±1 sign per key) and the point-query
//! aggregation is the **median** across rows, not the minimum.
//!
//! When sub-window sketches are merged, columns are progressively "unfolded"
//! until reaching the full Count Sketch resolution. Folding introduces **zero**
//! additional approximation error — the accuracy is identical to a full-width
//! Count Sketch with W columns.

use serde::{Deserialize, Serialize};

use crate::fold_cms::FoldCell;
use crate::{
    DataInput, DefaultXxHasher, HHHeap, SketchHasher, compute_median_inline_f64,
    heap_item_to_sketch_input,
};
use std::marker::PhantomData;

const LOWER_32_MASK: u64 = (1u64 << 32) - 1;

// ---------------------------------------------------------------------------
// FoldCS
// ---------------------------------------------------------------------------

/// Folding Count Sketch.
///
/// A sub-window Count Sketch that uses `full_cols / 2^fold_level` physical
/// columns. Each physical cell lazily tracks which full-CS column(s) it holds,
/// expanding only on real collisions. When sub-windows are merged the columns
/// are "unfolded" back towards the full-width CS.
///
/// Unlike `FoldCMS`, insert applies a random ±1 sign per (row, key), and
/// query returns the **median** of sign-corrected row estimates.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldCS<H: SketchHasher = DefaultXxHasher> {
    /// Number of hash functions (rows). Same across all fold levels.
    rows: usize,
    /// Number of physical columns = `full_cols >> fold_level`.
    fold_cols: usize,
    /// Target full-width CS column count (invariant across merges).
    full_cols: usize,
    /// Folding level: 0 = full-width CS, k = folded by 2^k.
    fold_level: u32,
    /// Flat storage: `cells[row * fold_cols + col]`.
    cells: Vec<FoldCell>,
    /// Top-K heavy-hitter tracking heap.
    heap: HHHeap,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

impl<H: SketchHasher> FoldCS<H> {
    // -- Construction -------------------------------------------------------

    /// Creates a new FoldCS.
    ///
    /// * `rows`       — number of hash functions (typically 3–5).
    /// * `full_cols`  — target full-width CS column count (must be power of 2).
    /// * `fold_level` — folding depth; physical columns = `full_cols / 2^fold_level`.
    /// * `top_k`      — capacity of the heavy-hitter heap.
    ///
    /// # Panics
    ///
    /// Panics if `full_cols` is not a power of two or `fold_level` is too large.
    pub fn new(rows: usize, full_cols: usize, fold_level: u32, top_k: usize) -> Self {
        assert!(
            full_cols.is_power_of_two(),
            "full_cols must be a power of two, got {full_cols}"
        );
        assert!(
            fold_level <= full_cols.trailing_zeros(),
            "fold_level {fold_level} too large for full_cols {full_cols}"
        );

        let fold_cols = full_cols >> fold_level;
        let total_cells = rows * fold_cols;
        let cells = vec![FoldCell::Empty; total_cells];

        FoldCS {
            rows,
            fold_cols,
            full_cols,
            fold_level,
            cells,
            heap: HHHeap::new(top_k),
            _hasher: PhantomData,
        }
    }

    /// Creates a FoldCS equivalent to a full-width CS (fold_level = 0).
    pub fn new_full(rows: usize, full_cols: usize, top_k: usize) -> Self {
        Self::new(rows, full_cols, 0, top_k)
    }

    // -- Accessors ----------------------------------------------------------

    #[inline(always)]
    /// Returns the number of sketch rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    /// Returns the number of physical folded columns.
    pub fn fold_cols(&self) -> usize {
        self.fold_cols
    }

    #[inline(always)]
    /// Returns the target full-width column count.
    pub fn full_cols(&self) -> usize {
        self.full_cols
    }

    #[inline(always)]
    /// Returns the current folding depth.
    pub fn fold_level(&self) -> u32 {
        self.fold_level
    }

    /// Returns a reference to the internal cell grid.
    pub fn cells(&self) -> &[FoldCell] {
        &self.cells
    }

    /// Returns a reference to the heavy-hitter heap.
    pub fn heap(&self) -> &HHHeap {
        &self.heap
    }

    /// Returns a mutable reference to the heavy-hitter heap.
    pub fn heap_mut(&mut self) -> &mut HHHeap {
        &mut self.heap
    }

    /// Returns the cell at `(row, fold_col)`.
    #[inline(always)]
    pub fn cell(&self, row: usize, fold_col: usize) -> &FoldCell {
        &self.cells[row * self.fold_cols + fold_col]
    }

    /// Total number of `(full_col, count)` entries across all cells.
    pub fn total_entries(&self) -> usize {
        self.cells.iter().map(|c| c.entry_count()).sum()
    }

    /// Number of cells that contain more than one entry (real collisions).
    pub fn collided_cells(&self) -> usize {
        self.cells.iter().filter(|c| c.entry_count() > 1).count()
    }

    // -- Hashing helpers ----------------------------------------------------

    /// Compute the full-width column and ±1 sign for `(row, key)`.
    ///
    /// Uses a single `hash64_seeded` call: lower 32 bits → column, bit 63 → sign.
    /// Sign convention matches `count.rs`: bit63==1 → +1, bit63==0 → -1.
    #[inline(always)]
    fn hash_for(&self, row: usize, key: &DataInput) -> (u16, i64) {
        let hashed = H::hash64_seeded(row, key);
        let full_col = ((hashed & LOWER_32_MASK) as usize % self.full_cols) as u16;
        let sign = if (hashed >> 63) & 1 == 1 { 1i64 } else { -1i64 };
        (full_col, sign)
    }

    /// Compute the physical (folded) column from a full column.
    #[inline(always)]
    fn fold_col_of(&self, full_col: u16) -> usize {
        (full_col as usize) & (self.fold_cols - 1)
    }

    // -- Insert -------------------------------------------------------------

    /// Insert `key` with count `delta`.
    ///
    /// For each row, the stored value is `sign * delta` where `sign` is ±1
    /// determined by the hash function.
    pub fn insert(&mut self, key: &DataInput, delta: i64) {
        for r in 0..self.rows {
            let (full_col, sign) = self.hash_for(r, key);
            let fc = self.fold_col_of(full_col);
            self.cells[r * self.fold_cols + fc].insert(full_col, sign * delta);
        }
        // Update top-k heap with current estimate.
        let est = self.query(key);
        self.heap.update(key, est);
    }

    /// Insert `key` once (delta = 1).
    #[inline]
    pub fn insert_one(&mut self, key: &DataInput) {
        self.insert(key, 1);
    }

    // -- Point Query --------------------------------------------------------

    /// Returns the Count Sketch frequency estimate for `key` (median of
    /// sign-corrected row estimates).
    pub fn query(&self, key: &DataInput) -> i64 {
        let mut estimates = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let (full_col, sign) = self.hash_for(r, key);
            let fc = self.fold_col_of(full_col);
            let cell_value = self.cells[r * self.fold_cols + fc].query(full_col);
            estimates.push((sign * cell_value) as f64);
        }
        compute_median_inline_f64(&mut estimates) as i64
    }

    // -- Same-level merge ---------------------------------------------------

    /// Merge `other` into `self` without unfolding. Both must share the same
    /// `full_cols`, `rows`, and `fold_level`.
    pub fn merge_same_level(&mut self, other: &FoldCS<H>) {
        assert_eq!(self.rows, other.rows, "row count mismatch");
        assert_eq!(self.full_cols, other.full_cols, "full_cols mismatch");
        assert_eq!(self.fold_level, other.fold_level, "fold_level mismatch");
        assert_eq!(self.fold_cols, other.fold_cols, "fold_cols mismatch");

        for idx in 0..self.cells.len() {
            self.cells[idx].merge_from(&other.cells[idx]);
        }

        self.reconcile_heap_from(other);
    }

    // -- Scatter helper -----------------------------------------------------

    /// Scatter all entries from `self` into a pre-allocated `target` sketch at
    /// any lower (or equal) fold level. Zero cloning — borrows only.
    ///
    /// The scatter formula `new_fc = full_col & (target.fold_cols - 1)` works
    /// for any source-to-target level jump in a single pass because every
    /// `FoldCell` entry carries its permanent `full_col` address.
    fn scatter_into(&self, target: &mut FoldCS<H>) {
        debug_assert_eq!(self.rows, target.rows);
        debug_assert_eq!(self.full_cols, target.full_cols);
        debug_assert!(target.fold_level <= self.fold_level);

        let target_fold_cols = target.fold_cols;
        for r in 0..self.rows {
            let src_row_off = r * self.fold_cols;
            let dst_row_off = r * target_fold_cols;
            for c in 0..self.fold_cols {
                let cell = &self.cells[src_row_off + c];
                for (full_col, count) in cell.iter() {
                    let new_fc = (full_col as usize) & (target_fold_cols - 1);
                    target.cells[dst_row_off + new_fc].insert(full_col, count);
                }
            }
        }
    }

    // -- Unfold merge -------------------------------------------------------

    /// Merge two **same-level** FoldCS sketches into a new sketch one fold
    /// level lower (doubled physical columns).
    pub fn unfold_merge(a: &FoldCS<H>, b: &FoldCS<H>) -> FoldCS<H> {
        assert_eq!(a.rows, b.rows, "row count mismatch");
        assert_eq!(a.full_cols, b.full_cols, "full_cols mismatch");
        assert_eq!(a.fold_level, b.fold_level, "fold_level mismatch");
        assert!(a.fold_level > 0, "cannot unfold from fold_level 0");

        let new_level = a.fold_level - 1;
        let new_fold_cols = a.full_cols >> new_level;
        let heap_k = a.heap.capacity().max(b.heap.capacity());

        let mut result = FoldCS {
            rows: a.rows,
            fold_cols: new_fold_cols,
            full_cols: a.full_cols,
            fold_level: new_level,
            cells: vec![FoldCell::Empty; a.rows * new_fold_cols],
            heap: HHHeap::new(heap_k),
            _hasher: PhantomData,
        };

        // Single-pass scatter from both sources into the wider grid.
        a.scatter_into(&mut result);
        b.scatter_into(&mut result);

        // Reconcile top-k heaps from both sources.
        for source in [a, b] {
            for item in source.heap.heap() {
                let key_ref = heap_item_to_sketch_input(&item.key);
                let est = result.query(&key_ref);
                result.heap.update(&key_ref, est);
            }
        }

        result
    }

    /// Fully unfold a FoldCS to fold_level 0 (equivalent to a standard CS).
    /// If already at level 0 this returns a clone.
    pub fn unfold_full(&self) -> FoldCS<H> {
        self.unfold_to(0)
    }

    // -- Hierarchical merge -------------------------------------------------

    /// Unfold `self` down to the target fold level (must be <= current level).
    /// If already at the target level, returns a clone.
    ///
    /// Single-pass scatter: 1 allocation, 1 pass — regardless of how many
    /// levels are skipped.
    pub fn unfold_to(&self, target_level: u32) -> FoldCS<H> {
        assert!(
            target_level <= self.fold_level,
            "target_level {target_level} > current fold_level {}",
            self.fold_level
        );
        if target_level == self.fold_level {
            return self.clone();
        }

        let new_fold_cols = self.full_cols >> target_level;
        let mut result = FoldCS {
            rows: self.rows,
            fold_cols: new_fold_cols,
            full_cols: self.full_cols,
            fold_level: target_level,
            cells: vec![FoldCell::Empty; self.rows * new_fold_cols],
            heap: HHHeap::new(self.heap.capacity()),
            _hasher: PhantomData,
        };

        self.scatter_into(&mut result);

        // Reconcile heap.
        for item in self.heap.heap() {
            let key_ref = heap_item_to_sketch_input(&item.key);
            let est = result.query(&key_ref);
            result.heap.update(&key_ref, est);
        }

        result
    }

    // -- N-way hierarchical merge -------------------------------------------

    /// Merge a sequence of FoldCS sketches into a single level-0 sketch.
    ///
    /// Allocates one level-0 result and scatters all N inputs directly into
    /// it. **0 clones, 1 allocation, N scatter passes.** Handles mixed fold
    /// levels — each source is scattered from whatever level it is at.
    pub fn hierarchical_merge(sketches: &[FoldCS<H>]) -> FoldCS<H> {
        assert!(!sketches.is_empty(), "need at least one sketch to merge");
        if sketches.len() == 1 {
            return sketches[0].unfold_to(0);
        }

        let rows = sketches[0].rows;
        let full_cols = sketches[0].full_cols;
        let heap_k = sketches.iter().map(|s| s.heap.capacity()).max().unwrap();

        let mut result = FoldCS {
            rows,
            fold_cols: full_cols,
            full_cols,
            fold_level: 0,
            cells: vec![FoldCell::Empty; rows * full_cols],
            heap: HHHeap::new(heap_k),
            _hasher: PhantomData,
        };

        for sk in sketches {
            assert_eq!(sk.rows, rows, "row count mismatch");
            assert_eq!(sk.full_cols, full_cols, "full_cols mismatch");
            sk.scatter_into(&mut result);
        }

        // Reconcile heaps from all sources.
        for sk in sketches {
            for item in sk.heap.heap() {
                let key_ref = heap_item_to_sketch_input(&item.key);
                let est = result.query(&key_ref);
                result.heap.update(&key_ref, est);
            }
        }

        result
    }

    // -- Conversion ---------------------------------------------------------

    /// Extract the flat i64 counter array equivalent to a standard CS.
    ///
    /// Returns a `rows x full_cols` row-major vector where each element is
    /// the accumulated (signed) count for that `(row, full_col)` cell.
    pub fn to_flat_counters(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.rows * self.full_cols];
        for r in 0..self.rows {
            for c in 0..self.fold_cols {
                let cell = &self.cells[r * self.fold_cols + c];
                for (full_col, count) in cell.iter() {
                    out[r * self.full_cols + full_col as usize] += count;
                }
            }
        }
        out
    }

    // -- Clear --------------------------------------------------------------

    /// Reset all cells to [`FoldCell::Empty`] and clear the heap.
    ///
    /// The outer `Vec<FoldCell>` allocation is preserved; inner `Collided(Vec)`
    /// data is dropped.
    pub fn clear(&mut self) {
        self.cells.fill(FoldCell::Empty);
        self.heap.clear();
    }

    // -- Heap helpers -------------------------------------------------------

    /// Re-query all heap items from `other` against `self` and update our heap.
    fn reconcile_heap_from(&mut self, other: &FoldCS<H>) {
        for item in other.heap.heap() {
            let key_ref = heap_item_to_sketch_input(&item.key);
            let est = self.query(&key_ref);
            self.heap.update(&key_ref, est);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::sample_zipf_u64;
    use crate::{Count, HeapItem, RegularPath, Vector2D};
    use std::collections::HashMap;

    // -- FoldCS basic tests -------------------------------------------------

    #[test]
    fn fold_cs_dimensions() {
        let sketch: FoldCS = FoldCS::new(3, 4096, 4, 10);
        assert_eq!(sketch.rows(), 3);
        assert_eq!(sketch.full_cols(), 4096);
        assert_eq!(sketch.fold_cols(), 256); // 4096 / 2^4
        assert_eq!(sketch.fold_level(), 4);
    }

    #[test]
    fn fold_cs_level_zero_is_full() {
        let sketch: FoldCS = FoldCS::new_full(3, 1024, 10);
        assert_eq!(sketch.fold_cols(), 1024);
        assert_eq!(sketch.fold_level(), 0);
    }

    #[test]
    #[should_panic(expected = "full_cols must be a power of two")]
    fn fold_cs_rejects_non_power_of_two() {
        let _: FoldCS = FoldCS::new(3, 1000, 0, 10);
    }

    #[test]
    #[should_panic(expected = "fold_level")]
    fn fold_cs_rejects_excessive_fold_level() {
        let _: FoldCS = FoldCS::new(3, 256, 9, 10); // 256 = 2^8, fold_level 9 is too big
    }

    #[test]
    fn fold_cs_insert_query_single_key() {
        let mut sketch: FoldCS = FoldCS::new(3, 1024, 4, 10);
        let key = DataInput::Str("hello");
        sketch.insert(&key, 7);
        assert_eq!(sketch.query(&key), 7);
    }

    #[test]
    fn fold_cs_insert_accumulates() {
        let mut sketch: FoldCS = FoldCS::new(3, 1024, 4, 10);
        let key = DataInput::Str("hello");
        sketch.insert(&key, 3);
        sketch.insert(&key, 4);
        assert_eq!(sketch.query(&key), 7);
    }

    #[test]
    fn fold_cs_absent_key_returns_zero() {
        let mut sketch: FoldCS = FoldCS::new(3, 1024, 4, 10);
        sketch.insert(&DataInput::Str("present"), 10);
        assert_eq!(sketch.query(&DataInput::Str("absent")), 0);
    }

    #[test]
    fn fold_cs_multiple_keys() {
        let mut sketch: FoldCS = FoldCS::new(3, 4096, 4, 10);
        for i in 0..100u64 {
            sketch.insert(&DataInput::U64(i), i as i64);
        }
        // Count Sketch estimates can be negative; check they are roughly correct.
        for i in 0..100u64 {
            let est = sketch.query(&DataInput::U64(i));
            // With a wide enough sketch, error should be small.
            let err = (est - i as i64).abs();
            assert!(
                err <= 10,
                "estimate {est} too far from true count {i} (error {err})"
            );
        }
    }

    // -- Sign application check ---------------------------------------------

    #[test]
    fn fold_cs_sign_application() {
        // Verify that raw cell values include both positive and negative entries,
        // confirming sign is being applied on insert.
        let mut sketch: FoldCS = FoldCS::new(5, 1024, 4, 10);
        for i in 0..50u64 {
            sketch.insert(&DataInput::U64(i), 1);
        }

        let mut has_positive = false;
        let mut has_negative = false;
        for cell in sketch.cells() {
            for (_full_col, count) in cell.iter() {
                if count > 0 {
                    has_positive = true;
                }
                if count < 0 {
                    has_negative = true;
                }
            }
        }
        assert!(
            has_positive && has_negative,
            "expected both positive and negative cell values (sign application)"
        );
    }

    // -- Exact match with standard Count Sketch -----------------------------

    #[test]
    fn fold_cs_matches_standard_cs_exact() {
        let rows = 3;
        let cols = 256; // small for deterministic testing
        let fold_level = 3; // 256/8 = 32 physical columns

        let mut fold: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut standard = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

        let keys: Vec<DataInput> = (0..50).map(DataInput::I32).collect();
        for key in &keys {
            fold.insert(key, 1);
            standard.insert(key);
        }

        // Every single query must match exactly.
        for key in &keys {
            let fold_est = fold.query(key);
            let std_est = standard.estimate(key) as i64;
            assert_eq!(
                fold_est, std_est,
                "mismatch for {key:?}: fold={fold_est}, std={std_est}"
            );
        }
    }

    #[test]
    fn fold_cs_matches_standard_cs_flat_counters() {
        let rows = 3;
        let cols = 256;
        let fold_level = 3;

        let mut fold: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut standard = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

        let keys: Vec<DataInput> = (0..50).map(DataInput::I32).collect();
        for key in &keys {
            fold.insert(key, 1);
            standard.insert(key);
        }

        // Verify via flat counter extraction.
        let flat = fold.to_flat_counters();
        let std_flat = standard.as_storage().as_slice();
        assert_eq!(flat.len(), std_flat.len());
        for (i, (f, s)) in flat.iter().zip(std_flat.iter()).enumerate() {
            assert_eq!(
                *f, *s,
                "flat counter mismatch at index {i}: fold={f}, std={s}"
            );
        }
    }

    #[test]
    fn fold_cs_matches_standard_cs_insert_many() {
        let rows = 3;
        let cols = 512;
        let fold_level = 4;

        let mut fold: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut standard = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

        for i in 0..30 {
            let key = DataInput::U64(i);
            let count = (i + 1) as i64;
            fold.insert(&key, count);
            standard.insert_many(&key, count);
        }

        for i in 0..30 {
            let key = DataInput::U64(i);
            assert_eq!(fold.query(&key), standard.estimate(&key) as i64);
        }
    }

    // -- Same-level merge ---------------------------------------------------

    #[test]
    fn same_level_merge_adds_counts() {
        let rows = 3;
        let cols = 1024;
        let fold_level = 3;

        let mut a: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut b: FoldCS = FoldCS::new(rows, cols, fold_level, 10);

        let key = DataInput::Str("user_001");
        a.insert(&key, 100);
        b.insert(&key, 200);

        a.merge_same_level(&b);
        assert_eq!(a.query(&key), 300);
    }

    #[test]
    fn same_level_merge_matches_standard_cs_merge() {
        let rows = 3;
        let cols = 512;
        let fold_level = 4;

        let mut fa: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut fb: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut sa = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);
        let mut sb = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

        for i in 0..20 {
            let key = DataInput::U64(i);
            fa.insert(&key, 1);
            sa.insert(&key);
        }
        for i in 10..30 {
            let key = DataInput::U64(i);
            fb.insert(&key, 1);
            sb.insert(&key);
        }

        fa.merge_same_level(&fb);
        sa.merge(&sb);

        for i in 0..30 {
            let key = DataInput::U64(i);
            assert_eq!(
                fa.query(&key),
                sa.estimate(&key) as i64,
                "mismatch after same-level merge for key {i}"
            );
        }
    }

    // -- Unfold merge -------------------------------------------------------

    #[test]
    fn unfold_merge_reduces_level() {
        let rows = 3;
        let cols = 1024;
        let fold_level = 3;

        let a: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let b: FoldCS = FoldCS::new(rows, cols, fold_level, 10);

        let result = FoldCS::unfold_merge(&a, &b);
        assert_eq!(result.fold_level(), 2);
        assert_eq!(result.fold_cols(), cols >> 2);
    }

    #[test]
    fn unfold_merge_preserves_counts() {
        let rows = 3;
        let cols = 256;
        let fold_level = 2;

        let mut a: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut b: FoldCS = FoldCS::new(rows, cols, fold_level, 10);

        let key_a = DataInput::Str("alpha");
        let key_b = DataInput::Str("beta");
        a.insert(&key_a, 10);
        b.insert(&key_b, 20);

        let merged = FoldCS::unfold_merge(&a, &b);
        assert_eq!(merged.fold_level(), 1);
        assert_eq!(merged.query(&key_a), 10);
        assert_eq!(merged.query(&key_b), 20);
    }

    #[test]
    fn unfold_merge_matches_standard_cs_merge() {
        let rows = 3;
        let cols = 512;
        let fold_level = 2;

        let mut fa: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut fb: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut sa = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);
        let mut sb = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

        for i in 0..40 {
            let key = DataInput::U64(i);
            fa.insert(&key, (i + 1) as i64);
            sa.insert_many(&key, (i + 1) as i64);
        }
        for i in 20..60 {
            let key = DataInput::U64(i);
            fb.insert(&key, (i + 1) as i64);
            sb.insert_many(&key, (i + 1) as i64);
        }

        let merged_fold = FoldCS::unfold_merge(&fa, &fb);
        sa.merge(&sb);

        for i in 0..60 {
            let key = DataInput::U64(i);
            assert_eq!(
                merged_fold.query(&key),
                sa.estimate(&key) as i64,
                "unfold merge mismatch for key {i}"
            );
        }
    }

    // -- Hierarchical merge -------------------------------------------------

    #[test]
    fn hierarchical_merge_four_sketches() {
        let rows = 3;
        let cols = 1024;
        let fold_level = 2; // 1024/4 = 256 physical cols

        let mut sketches = Vec::new();
        let mut standard = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

        for epoch in 0..4u64 {
            let mut sk: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
            for i in (epoch * 10)..((epoch + 1) * 10) {
                let key = DataInput::U64(i);
                sk.insert(&key, 1);
                standard.insert(&key);
            }
            sketches.push(sk);
        }

        let merged = FoldCS::hierarchical_merge(&sketches);
        assert_eq!(merged.fold_level(), 0);

        for i in 0..40u64 {
            let key = DataInput::U64(i);
            assert_eq!(
                merged.query(&key),
                standard.estimate(&key) as i64,
                "hierarchical merge mismatch for key {i}"
            );
        }
    }

    // -- unfold_full --------------------------------------------------------

    #[test]
    fn unfold_full_matches_flat_counters() {
        let rows = 3;
        let cols = 256;
        let fold_level = 4;

        let mut sk: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        for i in 0..30 {
            sk.insert(&DataInput::I32(i), 1);
        }

        let flat_before = sk.to_flat_counters();
        let full = sk.unfold_full();
        assert_eq!(full.fold_level(), 0);
        assert_eq!(full.fold_cols(), cols);

        let flat_after = full.to_flat_counters();
        assert_eq!(flat_before, flat_after);
    }

    // -- to_flat_counters ---------------------------------------------------

    #[test]
    fn to_flat_counters_matches_standard_cs() {
        let rows = 3;
        let cols = 128;
        let fold_level = 3;

        let mut fold: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        let mut standard = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

        for i in 0..20 {
            let key = DataInput::I32(i);
            fold.insert(&key, 1);
            standard.insert(&key);
        }

        let flat = fold.to_flat_counters();
        let std_flat = standard.as_storage().as_slice();
        assert_eq!(
            flat.len(),
            std_flat.len(),
            "the flattened counters must cover the whole rows x cols matrix"
        );
        for (i, (f, s)) in flat.iter().zip(std_flat.iter()).enumerate() {
            assert_eq!(*f, *s, "flat counter mismatch at [{i}]: fold={f}, std={s}");
        }
    }

    // -- Serde round trip ---------------------------------------------------

    /// The derived serde form is the only serialization FoldCS has, so the
    /// round trip has to carry the whole sketch: the folded cells with their
    /// permanent `full_col` tags, the geometry the unfold path reads, and the
    /// heavy-hitter heap (whose index is rebuilt on load rather than carried).
    #[test]
    fn serde_round_trip_keeps_the_cells_the_geometry_and_the_heap() {
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 16;

        let mut sk: FoldCS = FoldCS::new(rows, full_cols, fold_level, top_k);
        let stream = sample_zipf_u64(2_000, 1.2, 20_000, 0x5EED_C55C);
        for &value in &stream {
            sk.insert(&DataInput::U64(value), 1);
        }
        assert!(
            sk.collided_cells() > 0,
            "the fixture should exercise collided cells, not just Single ones"
        );

        let bytes = rmp_serde::to_vec(&sk).expect("encode");
        let restored: FoldCS = rmp_serde::from_slice(&bytes).expect("decode");

        assert_eq!(restored.rows(), sk.rows());
        assert_eq!(restored.fold_cols(), sk.fold_cols());
        assert_eq!(restored.full_cols(), sk.full_cols());
        assert_eq!(restored.fold_level(), sk.fold_level());
        assert_eq!(restored.total_entries(), sk.total_entries());
        assert_eq!(restored.collided_cells(), sk.collided_cells());
        assert_eq!(restored.to_flat_counters(), sk.to_flat_counters());

        for value in 0..2_000u64 {
            let key = DataInput::U64(value);
            assert_eq!(
                restored.query(&key),
                sk.query(&key),
                "estimate for {value} moved across the round trip"
            );
        }

        let before = sk.heap().heap();
        let after = restored.heap().heap();
        assert_eq!(after.len(), before.len(), "heap size moved");
        assert_eq!(after.len(), top_k, "the fixture should fill the heap");
        for (a, b) in after.iter().zip(before.iter()) {
            assert_eq!(a.key, b.key, "heap order moved across the round trip");
            assert_eq!(a.count, b.count, "heap count moved across the round trip");
        }
        // The heap's key index is `#[serde(skip)]` and rebuilt on load, so a
        // lookup has to work on the decoded side too.
        let resident = before[0].key.clone();
        assert!(
            restored.heap().find_heap_item(&resident).is_some(),
            "the rebuilt heap index cannot find its own resident"
        );
    }

    // -- Memory efficiency --------------------------------------------------

    #[test]
    fn sparse_subwindow_has_few_collisions() {
        let rows = 3;
        let cols = 4096;
        let fold_level = 4; // 256 physical cols

        let mut sk: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        for i in 0..50u64 {
            sk.insert(&DataInput::U64(i), 1);
        }

        let total_entries = sk.total_entries();
        let collided = sk.collided_cells();

        assert!(
            total_entries <= rows * 50,
            "total_entries={total_entries} should be <= {} (rows*distinct_keys)",
            rows * 50
        );
        assert!(
            total_entries >= rows * 45,
            "total_entries={total_entries} unexpectedly low"
        );
        assert!(collided < 30, "expected few collided cells, got {collided}");
    }

    // -- Top-K heap integration ---------------------------------------------

    #[test]
    fn heap_tracks_heavy_hitters() {
        let mut sk: FoldCS = FoldCS::new(3, 1024, 3, 5);

        for _ in 0..100 {
            sk.insert(&DataInput::Str("heavy"), 1);
        }
        for _ in 0..10 {
            sk.insert(&DataInput::Str("medium"), 1);
        }
        sk.insert(&DataInput::Str("light"), 1);

        let heap_items = sk.heap().heap();
        assert!(!heap_items.is_empty());

        let heavy = heap_items
            .iter()
            .find(|item| item.key == HeapItem::String("heavy".to_owned()));
        assert!(heavy.is_some(), "heavy hitter should be in heap");
        assert_eq!(heavy.unwrap().count, 100);
    }

    #[test]
    fn heap_survives_same_level_merge() {
        let mut a: FoldCS = FoldCS::new(3, 1024, 3, 5);
        let mut b: FoldCS = FoldCS::new(3, 1024, 3, 5);

        for _ in 0..50 {
            a.insert(&DataInput::Str("user_x"), 1);
        }
        for _ in 0..70 {
            b.insert(&DataInput::Str("user_x"), 1);
        }

        a.merge_same_level(&b);

        let found = a
            .heap()
            .heap()
            .iter()
            .find(|item| item.key == HeapItem::String("user_x".to_owned()));
        assert!(found.is_some());
        assert_eq!(found.unwrap().count, 120);
    }

    #[test]
    fn heap_survives_unfold_merge() {
        let mut a: FoldCS = FoldCS::new(3, 512, 2, 5);
        let mut b: FoldCS = FoldCS::new(3, 512, 2, 5);

        for _ in 0..40 {
            a.insert(&DataInput::Str("endpoint_a"), 1);
        }
        for _ in 0..60 {
            b.insert(&DataInput::Str("endpoint_a"), 1);
        }

        let merged = FoldCS::unfold_merge(&a, &b);
        let found = merged
            .heap()
            .heap()
            .iter()
            .find(|item| item.key == HeapItem::String("endpoint_a".to_owned()));
        assert!(found.is_some());
        assert_eq!(found.unwrap().count, 100);
    }

    // -- Error bound (statistical) ------------------------------------------

    #[test]
    fn fold_cs_error_bound_zipf() {
        let rows = 3;
        let cols = 4096;
        let fold_level = 4;
        let domain = 8192;
        let exponent = 1.1;
        let samples = 200_000;

        let mut fold: FoldCS = FoldCS::new(rows, cols, fold_level, 20);
        let mut truth = HashMap::<u64, i64>::new();

        for value in sample_zipf_u64(domain, exponent, samples, 0x5eed_c0de) {
            fold.insert(&DataInput::U64(value), 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        // CS error bound: |est - truth| < epsilon * ||f||_2
        // with probability >= 1 - delta
        let epsilon = (std::f64::consts::E / cols as f64).sqrt();
        let l2_norm = truth
            .values()
            .map(|&c| (c as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let error_bound = epsilon * l2_norm;
        let delta = 1.0 / std::f64::consts::E.powi(rows as i32);
        let correct_lower_bound = truth.len() as f64 * (1.0 - delta);

        let mut within_count = 0;
        for (key, true_count) in &truth {
            let est = fold.query(&DataInput::U64(*key));
            if ((est - true_count).abs() as f64) < error_bound {
                within_count += 1;
            }
        }

        assert!(
            within_count as f64 > correct_lower_bound,
            "in-bound items {within_count} not > expected {correct_lower_bound}"
        );
    }

    // -- Scatter-based optimization tests -----------------------------------

    #[test]
    fn scatter_merge_matches_standard_cs_n1_to_n8() {
        // Verify N-way scatter merge produces identical results to a standard
        // CS that saw all the same inserts, for N = 1..8.
        let rows = 3;
        let cols = 1024;
        let fold_level = 3;

        for n in 1..=8u64 {
            let mut sketches = Vec::new();
            let mut standard = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);

            for epoch in 0..n {
                let mut sk: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
                for i in (epoch * 10)..((epoch + 1) * 10) {
                    let key = DataInput::U64(i);
                    sk.insert(&key, 1);
                    standard.insert(&key);
                }
                sketches.push(sk);
            }

            let merged = FoldCS::hierarchical_merge(&sketches);
            assert_eq!(merged.fold_level(), 0, "N={n}: should reach level 0");

            for i in 0..(n * 10) {
                let key = DataInput::U64(i);
                assert_eq!(
                    merged.query(&key),
                    standard.estimate(&key) as i64,
                    "N={n}: mismatch for key {i}"
                );
            }
        }
    }

    #[test]
    fn unfold_to_single_pass_preserves_flat_counters() {
        // Verify that unfold_to (single-pass scatter) preserves exact flat
        // counters for every possible target level.
        let rows = 3;
        let cols = 256;
        let fold_level = 4;

        let mut sk: FoldCS = FoldCS::new(rows, cols, fold_level, 10);
        for i in 0..40 {
            sk.insert(&DataInput::U64(i), (i + 1) as i64);
        }

        let expected = sk.to_flat_counters();

        for target in (0..fold_level).rev() {
            let unfolded = sk.unfold_to(target);
            assert_eq!(unfolded.fold_level(), target);
            assert_eq!(unfolded.fold_cols(), cols >> target);
            assert_eq!(
                unfolded.to_flat_counters(),
                expected,
                "flat counters mismatch at target_level={target}"
            );
        }
    }

    #[test]
    fn unfold_to_same_level_returns_clone() {
        let mut sk: FoldCS = FoldCS::new(3, 256, 3, 10);
        sk.insert(&DataInput::Str("x"), 42);

        let result = sk.unfold_to(3);
        assert_eq!(result.fold_level(), 3);
        assert_eq!(result.query(&DataInput::Str("x")), 42);
    }

    #[test]
    fn hierarchical_merge_mixed_fold_levels() {
        // Sketches at different fold levels should merge correctly.
        let rows = 3;
        let cols = 1024;

        let mut sk_high: FoldCS = FoldCS::new(rows, cols, 4, 10);
        let mut sk_low: FoldCS = FoldCS::new(rows, cols, 2, 10);

        for i in 0..20u64 {
            sk_high.insert(&DataInput::U64(i), 1);
        }
        for i in 10..30u64 {
            sk_low.insert(&DataInput::U64(i), 1);
        }

        let merged = FoldCS::hierarchical_merge(&[sk_high.clone(), sk_low.clone()]);
        assert_eq!(merged.fold_level(), 0);

        // Build reference by scattering each to level 0 and merging.
        let a = sk_high.unfold_to(0);
        let b = sk_low.unfold_to(0);
        let mut reference = a;
        reference.merge_same_level(&b);

        for i in 0..30u64 {
            let key = DataInput::U64(i);
            assert_eq!(
                merged.query(&key),
                reference.query(&key),
                "mixed-level merge mismatch for key {i}"
            );
        }
    }
}
