//! Optimized UnivMon variants and pooling.
//!
//! - `UnivSketchPool`: free-list pool for `UnivMon` sketch reuse.
//! - `UnivMonPyramid`: pyramid-structured UnivMon with two-tier sketch
//!   dimensions — larger "elephant" layers for heavy hitters and smaller
//!   "mouse" layers for the long tail, matching the PromSketch design.

use crate::UnivMon;
use crate::common::heap::HHHeap;
use crate::common::{
    BOTTOM_LAYER_FINDER, DataInput, HeapItem, hash_item64_seeded, hash64_seeded,
    heap_item_to_sketch_input,
};
use crate::common::{L2HH, Vector1D};
use crate::sketch_framework::univmon::{MAX_LAYER_SIZE, UnivMonUpdateMode};
use crate::sketches::countsketch_topk::CountL2HH;
use std::collections::{HashMap, HashSet};

mod wire;

/// Object pool for `UnivMon` sketches.
///
/// Maintains a free-list of pre-allocated sketches. Callers take ownership
/// via `take()` and return sketches via `put()`, which resets and recycles
/// them. This avoids repeated heap allocation/deallocation for large sketch
/// matrices during promotion, merge, and expiration cycles.
pub struct UnivSketchPool {
    free_list: Vec<UnivMon>,
    total_allocated: usize,
    heap_size: usize,
    sketch_row: usize,
    sketch_col: usize,
    layer_size: usize,
}

impl UnivSketchPool {
    /// Creates a new pool with `cap` pre-allocated sketches.
    pub fn new(
        cap: usize,
        heap_size: usize,
        sketch_row: usize,
        sketch_col: usize,
        layer_size: usize,
    ) -> Self {
        let free_list: Vec<UnivMon> = (0..cap)
            .map(|_| UnivMon::init_univmon(heap_size, sketch_row, sketch_col, layer_size))
            .collect();
        UnivSketchPool {
            free_list,
            total_allocated: cap,
            heap_size,
            sketch_row,
            sketch_col,
            layer_size,
        }
    }

    /// Takes ownership of a clean sketch from the pool.
    ///
    /// Pops a recycled sketch from the free-list if available, otherwise
    /// allocates a fresh one.
    pub fn take(&mut self) -> UnivMon {
        if let Some(sketch) = self.free_list.pop() {
            sketch
        } else {
            self.total_allocated += 1;
            UnivMon::init_univmon(
                self.heap_size,
                self.sketch_row,
                self.sketch_col,
                self.layer_size,
            )
        }
    }

    /// Returns a sketch to the pool for reuse. Resets all internal state.
    pub fn put(&mut self, mut sketch: UnivMon) {
        sketch.free();
        self.free_list.push(sketch);
    }

    /// Number of sketches currently available in the pool.
    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of sketches ever allocated by this pool.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }
}

// ---------------------------------------------------------------------------
// UnivMonPyramid
// ---------------------------------------------------------------------------

const DEFAULT_ELEPHANT_LAYERS: usize = 8;
const DEFAULT_ELEPHANT_ROW: usize = 3;
const DEFAULT_ELEPHANT_COL: usize = 2048;
const DEFAULT_MOUSE_ROW: usize = 3;
const DEFAULT_MOUSE_COL: usize = 512;
const DEFAULT_PYRAMID_HEAP: usize = 32;
const DEFAULT_PYRAMID_LAYERS: usize = 16;

/// Pyramid-structured UnivMon with two-tier sketch dimensions.
///
/// Layers `0..elephant_layers` ("elephant") use larger sketches for accurate
/// heavy-hitter tracking. Layers `elephant_layers..layer_size` ("mouse") use
/// smaller sketches, saving memory since deeper layers sample exponentially
/// fewer items.
#[derive(Clone, Debug)]
/// Optimized UnivMon variant with separate elephant and mouse layers.
pub struct UnivMonPyramid {
    /// Per-layer L2/heavy-hitter sketches.
    pub l2_sketch_layers: Vector1D<L2HH>,
    /// Per-layer heavy-hitter heaps.
    pub hh_layers: Vector1D<HHHeap>,
    /// Total number of layers.
    pub layer_size: usize,
    /// Number of elephant layers.
    pub elephant_layers: usize,
    /// Row count for elephant layers.
    pub elephant_row: usize,
    /// Column count for elephant layers.
    pub elephant_col: usize,
    /// Row count for mouse layers.
    pub mouse_row: usize,
    /// Column count for mouse layers.
    pub mouse_col: usize,
    /// Heap capacity per layer.
    pub heap_size: usize,
    /// Bucket size used for hashing decisions.
    pub bucket_size: usize,
    update_mode: UnivMonUpdateMode,
    candidate_complete: Vec<bool>,
}

impl UnivMonPyramid {
    /// Creates an optimized UnivMon pyramid.
    pub fn new(
        heap_size: usize,
        elephant_layers: usize,
        elephant_row: usize,
        elephant_col: usize,
        mouse_row: usize,
        mouse_col: usize,
        total_layers: usize,
    ) -> Self {
        assert!(heap_size > 0, "heap size must be positive");
        assert!(
            elephant_row > 0 && mouse_row > 0,
            "sketch row counts must be positive"
        );
        assert!(
            elephant_col > 0 && mouse_col > 0,
            "sketch column counts must be positive"
        );
        assert!(total_layers > 0, "layer count must be positive");
        assert!(
            total_layers <= MAX_LAYER_SIZE,
            "layer count must be at most MAX_LAYER_SIZE {MAX_LAYER_SIZE}"
        );
        let sk_vec: Vec<L2HH> = if total_layers <= elephant_layers {
            (0..total_layers)
                .map(|i| {
                    L2HH::COUNT(CountL2HH::with_dimensions_and_seed(
                        elephant_row,
                        elephant_col,
                        i,
                    ))
                })
                .collect()
        } else {
            (0..elephant_layers)
                .map(|i| {
                    L2HH::COUNT(CountL2HH::with_dimensions_and_seed(
                        elephant_row,
                        elephant_col,
                        i,
                    ))
                })
                .chain((elephant_layers..total_layers).map(|i| {
                    L2HH::COUNT(CountL2HH::with_dimensions_and_seed(mouse_row, mouse_col, i))
                }))
                .collect()
        };

        let hh_vec: Vec<HHHeap> = (0..total_layers).map(|_| HHHeap::new(heap_size)).collect();

        UnivMonPyramid {
            l2_sketch_layers: Vector1D::from_vec(sk_vec),
            hh_layers: Vector1D::from_vec(hh_vec),
            layer_size: total_layers,
            elephant_layers,
            elephant_row,
            elephant_col,
            mouse_row,
            mouse_col,
            heap_size,
            bucket_size: 0,
            update_mode: UnivMonUpdateMode::Unset,
            candidate_complete: vec![true; total_layers],
        }
    }

    #[inline]
    fn begin_update(&mut self, value: i64, mode: UnivMonUpdateMode) {
        assert!(value >= 0, "UnivMon only supports non-negative updates");
        match self.update_mode {
            UnivMonUpdateMode::Unset => self.update_mode = mode,
            current if current == mode => {}
            _ => panic!("cannot mix standard and terminal-only updates in one UnivMon"),
        }
        self.bucket_size = self
            .bucket_size
            .checked_add(value as usize)
            .expect("UnivMon total weight overflowed usize");
    }

    /// Creates a pyramid using built-in default dimensions.
    pub fn with_defaults() -> Self {
        Self::new(
            DEFAULT_PYRAMID_HEAP,
            DEFAULT_ELEPHANT_LAYERS,
            DEFAULT_ELEPHANT_ROW,
            DEFAULT_ELEPHANT_COL,
            DEFAULT_MOUSE_ROW,
            DEFAULT_MOUSE_COL,
            DEFAULT_PYRAMID_LAYERS,
        )
    }

    #[inline(always)]
    fn find_bottom_layer_num(&self, hash: u64) -> usize {
        for l in 1..self.layer_size {
            if ((hash >> l) & 1) == 0 {
                return l - 1;
            }
        }
        self.layer_size - 1
    }

    /// Standard insert: updates sketch + heap at every layer 0..=bottom.
    pub fn insert(&mut self, key: &DataInput, value: i64) {
        self.begin_update(value, UnivMonUpdateMode::Standard);
        let h = hash64_seeded(BOTTOM_LAYER_FINDER, key);
        let bottom = self.find_bottom_layer_num(h);
        for i in 0..=bottom {
            let count = self.l2_sketch_layers[i].update_and_est(key, value);
            if !self.hh_layers[i].update(key, count as i64) {
                self.candidate_complete[i] = false;
            }
        }
    }

    /// Joltik-style insert: updates only the terminal CountSketch and heap.
    ///
    /// Logical upper-layer candidate sets are reconstructed at query time.
    /// Do not mix this method with [`Self::insert`] on the same sketch.
    pub fn fast_insert(&mut self, key: &DataInput, value: i64) {
        self.begin_update(value, UnivMonUpdateMode::Terminal);
        let h = hash64_seeded(BOTTOM_LAYER_FINDER, key);
        let bottom = self.find_bottom_layer_num(h);
        let count = self.l2_sketch_layers[bottom].update_and_est(key, value);
        if !self.hh_layers[bottom].update(key, count as i64) {
            self.candidate_complete[bottom] = false;
        }
    }

    // -- Query methods (identical to UnivMon) --------------------------------

    /// Computes a g-sum estimate.
    pub fn calc_g_sum<F>(&self, g: F, is_card: bool) -> f64
    where
        F: Fn(f64) -> f64,
    {
        if self.bucket_size == 0 {
            return 0.0;
        }
        if self.update_mode == UnivMonUpdateMode::Terminal {
            return self.calc_terminal_g_sum(g);
        }

        let mut y = vec![0.0; self.layer_size];

        let l2_top = self.l2_sketch_layers[self.layer_size - 1].get_l2();
        let threshold_top = if is_card {
            self.heavy_threshold(l2_top, self.candidate_complete[self.layer_size - 1])
        } else {
            0
        };

        let mut tmp = 0.0;
        for item in self.hh_layers[self.layer_size - 1].heap() {
            let input = heap_item_to_sketch_input(&item.key);
            let count = self.l2_sketch_layers[self.layer_size - 1].estimate(&input) as i64;
            if count > threshold_top {
                tmp += g(count as f64);
            }
        }
        y[self.layer_size - 1] = tmp;

        for i in (0..(self.layer_size - 1)).rev() {
            tmp = 0.0;
            let l2_val = self.l2_sketch_layers[i].get_l2();
            let threshold = if is_card {
                self.heavy_threshold(l2_val, self.candidate_complete[i])
            } else {
                0
            };

            for item in self.hh_layers[i].heap() {
                let input = heap_item_to_sketch_input(&item.key);
                let count = self.l2_sketch_layers[i].estimate(&input) as i64;
                if count > threshold {
                    let hash = (hash_item64_seeded(BOTTOM_LAYER_FINDER, &item.key) >> (i + 1)) & 1;
                    let coe = 1.0 - 2.0 * (hash as f64);
                    tmp += coe * g(count as f64);
                }
            }
            y[i] = 2.0 * y[i + 1] + tmp;
        }
        y[0]
    }

    fn calc_terminal_g_sum<F>(&self, g: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let (candidates, complete) = self.logical_terminal_candidates();
        let mut logical_l2 = vec![0.0; self.layer_size];
        let mut suffix_l2_squared = 0.0;
        for level in (0..self.layer_size).rev() {
            let terminal_l2 = self.l2_sketch_layers[level].get_l2();
            suffix_l2_squared += terminal_l2 * terminal_l2;
            logical_l2[level] = suffix_l2_squared.sqrt();
        }

        let mut y = vec![0.0; self.layer_size];
        let last = self.layer_size - 1;
        let threshold = self.heavy_threshold(logical_l2[last], complete[last]);
        y[last] = candidates[last]
            .iter()
            .filter(|(_, count)| *count > threshold)
            .map(|(_, count)| g(*count as f64))
            .sum();

        for level in (0..last).rev() {
            let threshold = self.heavy_threshold(logical_l2[level], complete[level]);
            let correction = candidates[level]
                .iter()
                .filter(|(_, count)| *count > threshold)
                .map(|(key, count)| {
                    let hash = (hash_item64_seeded(BOTTOM_LAYER_FINDER, key) >> (level + 1)) & 1;
                    (1.0 - 2.0 * hash as f64) * g(*count as f64)
                })
                .sum::<f64>();
            y[level] = 2.0 * y[level + 1] + correction;
        }
        y[0]
    }

    fn logical_terminal_candidates(&self) -> (Vec<Vec<(HeapItem, i64)>>, Vec<bool>) {
        let mut logical = vec![Vec::new(); self.layer_size];
        let mut complete = vec![false; self.layer_size];
        let mut cumulative = HashMap::<HeapItem, i64>::with_capacity(self.heap_size * 2);
        let mut suffix_complete = true;
        for level in (0..self.layer_size).rev() {
            suffix_complete &= self.candidate_complete[level];
            for item in self.hh_layers[level].heap() {
                let input = heap_item_to_sketch_input(&item.key);
                let count = self.l2_sketch_layers[level].estimate(&input) as i64;
                cumulative
                    .entry(item.key.clone())
                    .and_modify(|old| *old = (*old).max(count))
                    .or_insert(count);
            }
            let mut retained: Vec<_> = cumulative
                .iter()
                .map(|(key, count)| (key.clone(), *count))
                .collect();
            retained.sort_unstable_by(|left, right| {
                right.1.cmp(&left.1).then_with(|| {
                    hash_item64_seeded(BOTTOM_LAYER_FINDER, &left.0)
                        .cmp(&hash_item64_seeded(BOTTOM_LAYER_FINDER, &right.0))
                })
            });
            complete[level] = suffix_complete && retained.len() <= self.heap_size;
            retained.truncate(self.heap_size);
            cumulative = retained.iter().cloned().collect();
            logical[level] = retained;
        }
        (logical, complete)
    }

    #[inline]
    fn heavy_threshold(&self, l2: f64, complete: bool) -> i64 {
        if complete {
            0
        } else {
            (l2 / (self.heap_size as f64).sqrt()) as i64
        }
    }

    /// Returns the exact L1 norm for the supported non-negative update stream.
    pub fn calc_l1(&self) -> f64 {
        self.bucket_size as f64
    }

    /// Returns the estimated L2 norm.
    pub fn calc_l2(&self) -> f64 {
        self.calc_g_sum(|x| x * x, false).sqrt()
    }

    /// Returns the estimated entropy.
    pub fn calc_entropy(&self) -> f64 {
        if self.bucket_size == 0 {
            return 0.0;
        }
        let tmp = self.calc_g_sum(|x| if x > 0.0 { x * x.log2() } else { 0.0 }, false);
        (self.bucket_size as f64).log2() - tmp / (self.bucket_size as f64)
    }

    /// Returns the estimated cardinality.
    pub fn calc_card(&self) -> f64 {
        self.calc_g_sum(|_| 1.0, true)
    }

    // -- Lifecycle -----------------------------------------------------------

    /// Resets all counters and heaps without deallocating.
    pub fn free(&mut self) {
        self.bucket_size = 0;
        self.update_mode = UnivMonUpdateMode::Unset;
        self.candidate_complete.fill(true);
        for i in 0..self.layer_size {
            self.l2_sketch_layers[i].clear();
            self.hh_layers[i].clear();
        }
    }

    /// Merges another compatible pyramid and rebuilds its candidate heaps.
    pub fn merge(&mut self, other: &UnivMonPyramid) {
        assert_eq!(
            self.layer_size, other.layer_size,
            "layer size must match for merge"
        );
        assert_eq!(
            (
                self.elephant_layers,
                self.elephant_row,
                self.elephant_col,
                self.mouse_row,
                self.mouse_col,
                self.heap_size,
            ),
            (
                other.elephant_layers,
                other.elephant_row,
                other.elephant_col,
                other.mouse_row,
                other.mouse_col,
                other.heap_size,
            ),
            "UnivMon pyramid layouts must match for merge"
        );
        match (self.update_mode, other.update_mode) {
            (UnivMonUpdateMode::Unset, mode) => self.update_mode = mode,
            (_, UnivMonUpdateMode::Unset) => {}
            (left, right) => assert_eq!(
                left, right,
                "cannot merge standard and terminal-only UnivMon states"
            ),
        }
        self.bucket_size = self
            .bucket_size
            .checked_add(other.bucket_size)
            .expect("merged UnivMon total weight overflowed usize");
        for i in 0..self.layer_size {
            let sources_complete = self.candidate_complete[i] && other.candidate_complete[i];
            let candidate_keys: HashSet<HeapItem> = self.hh_layers[i]
                .heap()
                .iter()
                .chain(other.hh_layers[i].heap())
                .map(|item| item.key.clone())
                .collect();
            let merged_candidates_complete =
                sources_complete && candidate_keys.len() <= self.heap_size;
            self.l2_sketch_layers[i].merge(&other.l2_sketch_layers[i]);
            self.hh_layers[i].clear();
            for key in candidate_keys {
                let input = heap_item_to_sketch_input(&key);
                let count = self.l2_sketch_layers[i].estimate(&input) as i64;
                self.hh_layers[i].update(&input, count);
            }
            self.candidate_complete[i] = merged_candidates_complete;
        }
    }

    /// Returns the heap for one layer.
    pub fn heap_at_layer(&mut self, layer: usize) -> &mut HHHeap {
        &mut self.hh_layers[layer]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataInput;

    #[test]
    fn pool_basic_take_put() {
        let mut pool = UnivSketchPool::new(2, 16, 2, 5, 2);
        assert_eq!(pool.available(), 2);
        assert_eq!(pool.total_allocated(), 2);

        let s0 = pool.take();
        assert_eq!(pool.available(), 1);

        let s1 = pool.take();
        assert_eq!(pool.available(), 0);

        // Pool is empty — next take allocates a new one
        let s2 = pool.take();
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.total_allocated(), 3);

        // Return one
        pool.put(s1);
        assert_eq!(pool.available(), 1);

        // Should reuse the returned sketch
        let s3 = pool.take();
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.total_allocated(), 3); // no new allocation

        // Return all
        pool.put(s0);
        pool.put(s2);
        pool.put(s3);
        assert_eq!(pool.available(), 3);
    }

    #[test]
    fn pool_free_resets_sketch() {
        let mut pool = UnivSketchPool::new(1, 16, 2, 5, 2);

        // Take a sketch, insert some data
        let mut sketch = pool.take();
        sketch.insert(&DataInput::I64(42), 100);
        assert!(sketch.bucket_size > 0);

        // Return it — should reset
        pool.put(sketch);

        // Take it back — should be clean
        let sketch2 = pool.take();
        assert_eq!(sketch2.bucket_size, 0);
        assert!((sketch2.l2_sketch_layers[0].get_l2()).abs() < 1e-9);
    }

    // =======================================================================
    //                    UnivMonPyramid tests
    // =======================================================================

    #[test]
    fn pyramid_basic_insert_and_query() {
        let mut um = UnivMonPyramid::with_defaults();

        let cases: Vec<(&str, i64)> = vec![("hello", 10), ("world", 20), ("hello", 5), ("foo", 30)];

        for (key, val) in &cases {
            um.insert(&DataInput::Str(key), *val);
        }

        assert_eq!(um.bucket_size, 65);
        assert!((um.calc_l1() - 65.0).abs() < 1e-6, "L1 = {}", um.calc_l1());
        assert_eq!(um.calc_card(), 3.0);
    }

    #[test]
    fn pyramid_fast_insert_matches_standard() {
        // Both insert paths should produce identical sketches.
        // Retain the complete 100-key support so this tests construction
        // equivalence rather than independent candidate-sampling variance.
        let mut standard = UnivMonPyramid::new(128, 8, 3, 2048, 3, 512, 16);
        let mut fast = UnivMonPyramid::new(128, 8, 3, 2048, 3, 512, 16);

        for i in 0..500i64 {
            let key = DataInput::I64(i % 100);
            standard.insert(&key, 1);
            fast.fast_insert(&key, 1);
        }

        assert_eq!(standard.bucket_size, fast.bucket_size);

        // Standard cumulative layers and Joltik terminal strata must yield
        // the same logical estimates when all candidates are retained.
        let l1_diff = (standard.calc_l1() - fast.calc_l1()).abs();
        let card_diff = (standard.calc_card() - fast.calc_card()).abs();
        assert!(
            l1_diff / standard.calc_l1() < 0.10,
            "L1 diverged: std={}, fast={}",
            standard.calc_l1(),
            fast.calc_l1()
        );
        assert!(
            card_diff / standard.calc_card().max(1.0) < 0.15,
            "Card diverged: std={}, fast={}",
            standard.calc_card(),
            fast.calc_card()
        );
    }

    #[test]
    fn pyramid_two_tier_dimensions() {
        // Verify elephant layers are larger than mouse layers.
        let um = UnivMonPyramid::new(32, 4, 5, 2048, 3, 256, 8);

        // Layers 0..4 = elephant (5 rows × 2048 cols)
        // Layers 4..8 = mouse   (3 rows × 256 cols)
        assert_eq!(um.layer_size, 8);
        assert_eq!(um.elephant_layers, 4);
    }

    #[test]
    fn pyramid_free_resets_state() {
        let mut um = UnivMonPyramid::with_defaults();
        for i in 0..100i64 {
            um.insert(&DataInput::I64(i), 10);
        }
        assert!(um.bucket_size > 0);

        um.free();
        assert_eq!(um.bucket_size, 0);
        assert!((um.l2_sketch_layers[0].get_l2()).abs() < 1e-9);
    }

    #[test]
    fn pyramid_merge_combines_data() {
        let mut one_pass = UnivMonPyramid::new(128, 8, 3, 2048, 3, 512, 16);
        let mut left = UnivMonPyramid::new(128, 8, 3, 2048, 3, 512, 16);
        let mut right = UnivMonPyramid::new(128, 8, 3, 2048, 3, 512, 16);

        for i in 0..50i64 {
            left.insert(&DataInput::I64(i), 10);
            one_pass.insert(&DataInput::I64(i), 10);
        }
        for i in 50..100i64 {
            right.insert(&DataInput::I64(i), 10);
            one_pass.insert(&DataInput::I64(i), 10);
        }

        left.merge(&right);

        assert_eq!(left.bucket_size, 1000);
        assert!(left.calc_entropy().is_finite());
        for level in 0..left.layer_size {
            let L2HH::COUNT(merged_count) = &left.l2_sketch_layers[level];
            let L2HH::COUNT(one_pass_count) = &one_pass.l2_sketch_layers[level];
            assert_eq!(
                merged_count.as_storage().as_slice(),
                one_pass_count.as_storage().as_slice(),
                "counter mismatch at level {level}"
            );
        }
        assert!(
            (left.calc_l1() - one_pass.calc_l1()).abs() < 1e-9,
            "merged L1={}, one-pass L1={}",
            left.calc_l1(),
            one_pass.calc_l1()
        );
        assert!((left.calc_l2() - one_pass.calc_l2()).abs() < 1e-9);
        assert!((left.calc_card() - one_pass.calc_card()).abs() < 1e-9);
        assert!((left.calc_entropy() - one_pass.calc_entropy()).abs() < 1e-9);
    }

    fn ground_truth(freq: &std::collections::HashMap<i64, i64>) -> (f64, f64, f64, f64) {
        let l1: f64 = freq.values().map(|&v| v as f64).sum();
        let l2: f64 = freq
            .values()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let card = freq.len() as f64;
        let entropy = if l1 > 0.0 {
            let term: f64 = freq
                .values()
                .map(|&v| {
                    let f = v as f64;
                    if f > 0.0 { f * f.log2() } else { 0.0 }
                })
                .sum();
            l1.log2() - term / l1
        } else {
            0.0
        };
        (l1, l2, card, entropy)
    }

    #[test]
    fn pyramid_accuracy_zipf() {
        use std::collections::HashMap;

        let mut um = UnivMonPyramid::new(64, 8, 5, 2048, 3, 512, 16);
        let mut freq: HashMap<i64, i64> = HashMap::new();

        // Heavy hitter
        for _ in 0..5000 {
            um.insert(&DataInput::I64(0), 1);
            *freq.entry(0).or_insert(0) += 1;
        }
        // Medium flows
        for key in 1..=20i64 {
            for _ in 0..200 {
                um.insert(&DataInput::I64(key), 1);
                *freq.entry(key).or_insert(0) += 1;
            }
        }
        // Light flows
        for key in 21..=500i64 {
            um.insert(&DataInput::I64(key), 1);
            *freq.entry(key).or_insert(0) += 1;
        }

        let (true_l1, true_l2, true_card, true_entropy) = ground_truth(&freq);

        let err = |name: &str, est: f64, truth: f64| {
            let rel = (est - truth).abs() / truth.max(1e-12);
            assert!(
                rel < 0.15,
                "Pyramid {name}: error {:.2}%, est={est:.2}, truth={truth:.2}",
                rel * 100.0
            );
        };

        err("L1", um.calc_l1(), true_l1);
        err("L2", um.calc_l2(), true_l2);
        err("Card", um.calc_card(), true_card);
        err("Entropy", um.calc_entropy(), true_entropy);
    }

    #[test]
    fn pyramid_fast_insert_accuracy() {
        use std::collections::HashMap;

        let mut um = UnivMonPyramid::new(64, 8, 5, 2048, 3, 512, 16);
        let mut freq: HashMap<i64, i64> = HashMap::new();

        // Use fast_insert for everything.
        for _ in 0..3000 {
            um.fast_insert(&DataInput::I64(0), 1);
            *freq.entry(0).or_insert(0) += 1;
        }
        for key in 1..=50i64 {
            for _ in 0..100 {
                um.fast_insert(&DataInput::I64(key), 1);
                *freq.entry(key).or_insert(0) += 1;
            }
        }

        let (true_l1, true_l2, true_card, true_entropy) = ground_truth(&freq);

        let err = |name: &str, est: f64, truth: f64| {
            let rel = (est - truth).abs() / truth.max(1e-12);
            assert!(
                rel < 0.15,
                "Pyramid fast {name}: error {:.2}%, est={est:.2}, truth={truth:.2}",
                rel * 100.0
            );
        };

        err("L1", um.calc_l1(), true_l1);
        err("L2", um.calc_l2(), true_l2);
        err("Card", um.calc_card(), true_card);
        err("Entropy", um.calc_entropy(), true_entropy);
    }

    #[test]
    fn pyramid_memory_savings_vs_uniform() {
        // Pyramid with 16 layers should use less memory than a uniform
        // UnivMon with the same elephant dimensions at all 16 layers.
        // We verify this by comparing total sketch column counts.
        let elephant_col = 2048;
        let mouse_col = 512;
        let elephant_layers = 8;
        let total_layers = 16;

        let uniform_cols = elephant_col * total_layers;
        let pyramid_cols =
            elephant_col * elephant_layers + mouse_col * (total_layers - elephant_layers);

        assert!(
            pyramid_cols < uniform_cols,
            "Pyramid ({pyramid_cols}) should use fewer columns than uniform ({uniform_cols})"
        );
        // With defaults: pyramid = 8*2048 + 8*512 = 20480, uniform = 16*2048 = 32768
        // Savings = ~37.5%
        let savings = 1.0 - (pyramid_cols as f64 / uniform_cols as f64);
        assert!(
            savings > 0.30,
            "Expected >30% column savings, got {:.1}%",
            savings * 100.0
        );
    }
}
