//! KLL quantile sketch (dynamic / insert-optimized variant).
//!
//! This variant uses a dynamically-growing `Vector1D` buffer instead of a
//! pre-allocated fixed-size array. Insertion and compaction follow the compact
//! KLL layout from:
//! "Insert-optimized implementation of streaming data sketches" (Pfeil et al., 2025).
//! CDF construction follows the pattern described in dgryski/go-kll, based on the
//! weighted CDF approach from the original KLL paper (Karnin, Lang & Liberty, FOCS 2016).
//!
//! References:
//! - Karnin, Lang & Liberty, "Optimal Quantile Approximation in Streams," FOCS 2016.
//!   <https://arxiv.org/abs/1603.05346>
//! - <https://www.amazon.science/publications/insert-optimized-implementation-of-streaming-data-sketches>

use serde::{Deserialize, Serialize};

use crate::common::input::data_input_to_f64;
use crate::common::numerical::NumericalValue;
use crate::{DataInput, Vector1D};

use super::kll::{Coin, merge_sorted_runs, randomly_halve_up};

mod wire;

const CAPACITY_CACHE_LEN: usize = 20;
const MAX_CACHEABLE_K: usize = 26_602;
const CAPACITY_DECAY: f64 = 2.0 / 3.0;
const DEFAULT_K: i32 = 200;

/// One entry in the cumulative distribution, storing a value and its mass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicCdfEntry {
    value: f64,
    quantile: f64,
}

/// KLL quantile sketch using a dynamic, insert-optimized layout.
///
/// Unlike [`KLL`](super::kll::KLL), which pre-allocates a fixed buffer and
/// grows downward, `KLLDynamic` appends items to a growable `Vector1D` and
/// shifts elements during compaction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KLLDynamic<T: NumericalValue = f64> {
    items: Vector1D<T>, // compactors, packed
    /// Stores the START index of each level in `items`.
    levels: Vector1D<usize>,
    k: usize,
    m: usize, // Minimum buffer size (usually 8)
    num_levels: usize,
    co: Coin,
    /// Explicit seed for the compaction coin, when present. Set by
    /// [`KLLDynamic::init_with_seed`] / [`KLLDynamic::init_kll_with_seed`] so
    /// a sketch replays identically across runs and `clear()` re-seeds from
    /// the same value. Not part of the wire format: it describes how the
    /// sketch was built, not what it currently holds.
    #[serde(skip)]
    seed: Option<u64>,
    /// Cached level capacities by height (from top to bottom)
    #[serde(skip)]
    capacity_cache: [u32; CAPACITY_CACHE_LEN],
    /// Tracks current top height so we can index into the cache quickly.
    #[serde(skip)]
    top_height: usize,
    /// Cached capacity for level 0 to speed up hot-path updates
    #[serde(skip)]
    level0_capacity: usize,
}

impl<T: NumericalValue> Default for KLLDynamic<T> {
    fn default() -> Self {
        Self::init_kll(DEFAULT_K)
    }
}

impl<T: NumericalValue> KLLDynamic<T> {
    /// Creates a KLLDynamic sketch with the given `k` and `m` parameters.
    pub fn init(k: usize, m: usize) -> Self {
        Self::init_internal(k, m, Coin::new(), None)
    }

    /// Creates a KLLDynamic sketch with an explicit RNG seed for the
    /// compaction coin. Two sketches built with the same seed and fed the same
    /// input produce identical state, which is what makes reproducible-replay
    /// scenarios and deterministic accuracy tests possible. The seed is also
    /// stored so `clear()` re-seeds deterministically.
    ///
    /// The unseeded [`KLLDynamic::init`] draws from the wall clock, so a
    /// sketch built with it cannot be used to assert an accuracy bound
    /// reproducibly. Mirrors [`crate::KLL::init_with_seed`].
    pub fn init_with_seed(k: usize, m: usize, seed: u64) -> Self {
        Self::init_internal(k, m, Coin::from_seed(seed), Some(seed))
    }

    fn init_internal(k: usize, m: usize, coin: Coin, seed: Option<u64>) -> Self {
        let mut norm_m = m.min(MAX_CACHEABLE_K);
        norm_m = norm_m.max(2);
        let mut norm_k = k.max(norm_m);
        if norm_k > MAX_CACHEABLE_K {
            norm_k = MAX_CACHEABLE_K;
        }
        let mut s = Self {
            items: Vector1D::init(norm_k * 3),
            levels: Vector1D::filled(2, 0),
            k: norm_k,
            m: norm_m,
            num_levels: 1,
            co: coin,
            seed,
            capacity_cache: [0; CAPACITY_CACHE_LEN],
            top_height: 0,
            level0_capacity: 0,
        };
        s.rebuild_capacity_cache();
        s
    }

    /// Creates a KLLDynamic sketch with default `m` and the provided `k`.
    pub fn init_kll(k: i32) -> Self {
        Self::init(k as usize, 8)
    }

    /// `init_kll` with an explicit RNG seed. See [`KLLDynamic::init_with_seed`].
    pub fn init_kll_with_seed(k: i32, seed: u64) -> Self {
        Self::init_with_seed(k as usize, 8, seed)
    }

    fn push_value(&mut self, value: T) {
        self.items.push(value);

        if let Some(last) = self.levels.last_mut() {
            *last = self.items.len();
        }

        let levels_slice = self.levels.as_slice();
        let l0_start = levels_slice[self.num_levels - 1];
        let l0_count = self.items.len() - l0_start;

        if l0_count > self.level0_capacity {
            self.compress_while_needed();
        }
    }

    /// The hot path: O(1) insertion at the end of the vector.
    pub fn update(&mut self, val: &T) {
        self.push_value(*val);
    }

    /// Batch variant of `update`. Exactly equivalent to looping `update`.
    #[inline]
    pub fn bulk_update(&mut self, values: &[T]) {
        for v in values {
            self.push_value(*v);
        }
    }

    /// Loops to maintain the KLL invariant.
    fn compress_while_needed(&mut self) {
        let mut h = 0;
        loop {
            let level_idx = self.num_levels - 1 - h;
            let cap = self.capacity_for_level(h);

            let size = self.level_size(h);

            if size <= cap {
                break;
            }

            if level_idx == 0 {
                self.add_new_top_level();
                continue;
            }

            self.compact(h);
            h += 1;
        }
    }

    fn capacity_for_level(&self, level: usize) -> usize {
        if self.num_levels == 0 {
            return self.m;
        }
        let height_from_top = self.top_height.saturating_sub(level);
        let idx = height_from_top.min(CAPACITY_CACHE_LEN - 1);
        self.capacity_cache[idx] as usize
    }

    fn rebuild_capacity_cache(&mut self) {
        self.top_height = self.num_levels.saturating_sub(1);
        let mut scale = 1.0_f64;
        for idx in 0..CAPACITY_CACHE_LEN {
            let scaled = ((self.k as f64) * scale).ceil() as usize;
            let cap = scaled.max(self.m);
            self.capacity_cache[idx] = cap as u32;
            scale *= CAPACITY_DECAY;
        }
        self.level0_capacity = self.capacity_for_level(0);
    }

    #[inline]
    fn level_size(&self, h: usize) -> usize {
        let idx = self.num_levels - 1 - h;
        let slice = self.levels.as_slice();
        slice[idx + 1] - slice[idx]
    }

    fn add_new_top_level(&mut self) {
        self.levels.insert(0, 0);
        if let Some(last) = self.levels.last_mut() {
            *last = self.items.len();
        }
        self.num_levels += 1;
        self.top_height = self.num_levels - 1;
        self.level0_capacity = self.capacity_for_level(0);
    }

    fn compact(&mut self, h: usize) {
        let cur_lvl_idx = self.num_levels - 1 - h;

        // Get raw indices first
        let levels_slice = self.levels.as_mut_slice();
        let start = levels_slice[cur_lvl_idx];
        let end = levels_slice[cur_lvl_idx + 1];
        let count = end - start;

        let items = self.items.as_mut_slice();

        items[start..end].sort_unstable_by(T::total_cmp);

        let offset = usize::from(self.co.toss());
        let mut survivors = 0;
        let mut i = offset;

        while i < count {
            items[start + survivors] = items[start + i];
            survivors += 1;
            i += 2;
        }

        let garbage_len = count - survivors;
        let start_garbage = start + survivors;
        let end_garbage = end;
        let tail_len = items.len() - end_garbage;

        if tail_len > 0 {
            // Safety: source and destination ranges may overlap, but `ptr::copy` handles overlap.
            // The ranges are within `items` and `tail_len` ensures we stay in-bounds.
            unsafe {
                let ptr = items.as_mut_ptr();
                std::ptr::copy(ptr.add(end_garbage), ptr.add(start_garbage), tail_len);
            }
        }

        let new_len = items.len() - garbage_len;
        self.items.truncate(new_len);

        // Update level pointers after shift
        let levels_slice = self.levels.as_mut_slice();
        levels_slice[cur_lvl_idx] = start + survivors;

        for pos in levels_slice
            .iter_mut()
            .take(self.num_levels + 1)
            .skip(cur_lvl_idx + 1)
        {
            *pos -= garbage_len;
        }

        // Sync last pointer just in case (should be covered by loop, but ensures safety)
        levels_slice[self.num_levels] = self.items.len();
    }

    /// Reset the sketch to its initial state, preserving `k`, `m`, and the
    /// backing `items` allocation. After clearing, the sketch behaves as if
    /// freshly constructed.
    pub fn clear(&mut self) {
        self.items.clear();
        self.levels = Vector1D::filled(2, 0);
        self.num_levels = 1;
        // Re-seed from the construction seed when there is one, so a cleared
        // sketch replays exactly like a fresh one; fall back to the wall clock
        // for the unseeded constructor.
        self.co = match self.seed {
            Some(s) => Coin::from_seed(s),
            None => Coin::new(),
        };
        self.rebuild_capacity_cache();
    }

    /// Prints the compactors for debugging.
    pub fn print_compactors(&self)
    where
        T: std::fmt::Debug,
    {
        println!(
            "KLLDynamic Packed (k={}, levels={}, items={})",
            self.k,
            self.num_levels,
            self.items.len()
        );
        let levels = self.levels.as_slice();
        let items = self.items.as_slice();
        for h in (0..self.num_levels).rev() {
            let idx = self.num_levels - 1 - h;
            let start = levels[idx];
            let end = levels[idx + 1];
            println!("  L{}: {:?}", h, &items[start..end]);
        }
    }

    /// Builds a CDF representation of the sketch.
    pub fn cdf(&self) -> DynamicCdf {
        let mut cdf = DynamicCdf {
            entries: Vector1D::init(self.buffer_size()),
        };
        let mut total_w = 0usize;

        let levels = self.levels.as_slice();
        let items = self.items.as_slice();

        for h in 0..self.num_levels {
            let idx = self.num_levels - 1 - h;
            let start = levels[idx];
            let end = levels[idx + 1];
            let weight = 1 << h;
            for &value in &items[start..end] {
                cdf.entries.push(DynamicCdfEntry {
                    value: value.to_f64(),
                    quantile: weight as f64,
                });
            }
            total_w += (end - start) * weight;
        }

        if total_w == 0 {
            return cdf;
        }

        cdf.entries
            .as_mut_slice()
            .sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

        let mut cur_w = 0.0;
        for entry in cdf.entries.as_mut_slice() {
            cur_w += entry.quantile;
            entry.quantile = cur_w / total_w as f64;
        }

        cdf
    }

    /// Merges another sketch's retained items into this one, preserving
    /// each retained item's level weight (`2^level`) instead of discarding
    /// it. See [`KLL::merge`](super::kll::KLL::merge) for the full
    /// rationale — this is the same weight-preserving, level-by-level
    /// interleave-and-recompact merge (concatenate each level's retained
    /// items, then re-run the same randomized halve-and-promote compaction
    /// ordinary inserts use), adapted to `KLLDynamic`'s growable (rather
    /// than fixed-capacity) backing storage.
    pub fn merge(&mut self, other: &KLLDynamic<T>) {
        if other.items.is_empty() {
            return; // `other` is empty: nothing to merge.
        }

        let target_num_levels = self.num_levels.max(other.num_levels);
        // work[h] holds level h's combined (not-yet-compacted) retained
        // items. +1 slack so `work[h + 1]` is always valid while cascading.
        let mut work: Vec<Vec<T>> = vec![Vec::new(); target_num_levels + 1];

        // Unlike `KLL` (fixed-capacity), whose `compact` only ever sorts
        // level 0 and otherwise maintains sortedness of levels >= 1 as a
        // standing invariant via merge-promotion, `KLLDynamic::compact`
        // re-sorts a level's *entire* current contents from scratch on
        // every call. So a level here is only guaranteed sorted right
        // after its own last compaction — one that has since received a
        // promotion from below is a concatenation of separately-sorted
        // runs, not one sorted run as a whole. Neither operand's levels
        // (including >= 1) can be assumed pre-sorted, so sort each
        // operand's own contribution to a level before combining. That's
        // also cheaper than concatenating first and sorting the combined
        // (up to 2x larger) run: O(a log a + b log b) beats O((a+b)
        // log(a+b)), and the two now-genuinely-sorted runs merge in
        // linear time via the existing `merge_sorted_runs`.
        //
        // Absolute level `h` lives at array index `num_levels - 1 - h`.
        #[allow(clippy::needless_range_loop)] // `h` also indexes self.levels/self.items
        for h in 0..self.num_levels {
            let idx = self.num_levels - 1 - h;
            let levels = self.levels.as_slice();
            let (s, e) = (levels[idx], levels[idx + 1]);
            work[h].extend_from_slice(&self.items.as_slice()[s..e]);
            work[h].sort_unstable_by(T::total_cmp);
        }
        let mut merge_buf: Vec<T> = Vec::new();
        #[allow(clippy::needless_range_loop)] // `h` also indexes other.levels/other.items
        for h in 0..other.num_levels {
            let idx = other.num_levels - 1 - h;
            let levels = other.levels.as_slice();
            let (s, e) = (levels[idx], levels[idx + 1]);
            let self_len = work[h].len();
            work[h].extend_from_slice(&other.items.as_slice()[s..e]);
            work[h][self_len..].sort_unstable_by(T::total_cmp);
            merge_sorted_runs(work[h].as_mut_slice(), self_len, &mut merge_buf);
        }

        // Grow self's level bookkeeping to cover the merged height.
        self.num_levels = target_num_levels;
        self.rebuild_capacity_cache();

        // Cascade-compact exactly like `compress_while_needed`/`compact`,
        // except a level may need more than one halving pass here (a merge
        // can leave a level far over capacity, not just one element over).
        // Every level was sorted up front (above), and each promotion
        // below keeps its target level sorted via `merge_sorted_runs`, so
        // no level needs re-sorting once the cascade begins.
        let mut h = 0;
        while h < self.num_levels {
            while work[h].len() > self.capacity_for_level(h) {
                if h + 1 == self.num_levels {
                    self.num_levels += 1;
                    self.rebuild_capacity_cache();
                    work.resize(self.num_levels + 1, Vec::new());
                }
                let pop = work[h].len();
                let offset = usize::from(self.co.toss());
                let num_survivors = randomly_halve_up(work[h].as_mut_slice(), 0, pop, offset);
                let discard = pop - num_survivors;
                work[h].drain(0..discard);

                // Promote the (sorted) survivors into level h+1, merging
                // with its existing (already-sorted) content.
                let mut promoted = std::mem::take(&mut work[h]);
                let left_len = promoted.len();
                promoted.append(&mut work[h + 1]);
                merge_sorted_runs(promoted.as_mut_slice(), left_len, &mut merge_buf);
                work[h + 1] = promoted;
            }
            h += 1;
        }

        // Rebuild `items`/`levels` from the compacted per-level vectors,
        // top level first, matching this type's storage order (array
        // index 0 == top level, growing toward level 0 at the tail).
        let total: usize = work[..self.num_levels].iter().map(Vec::len).sum();
        let mut items = Vec::with_capacity(total);
        let mut levels = Vec::with_capacity(self.num_levels + 1);
        levels.push(0usize);
        for h in (0..self.num_levels).rev() {
            items.extend_from_slice(&work[h]);
            levels.push(items.len());
        }
        self.items = Vector1D::from_vec(items);
        self.levels = Vector1D::from_vec(levels);
    }

    /// Returns the estimated value at quantile `q`.
    pub fn quantile(&self, q: f64) -> f64 {
        let cdf = self.cdf();
        cdf.query(q)
    }

    /// Returns the estimated rank of value `x`.
    pub fn rank(&self, x: f64) -> usize {
        let mut r = 0;
        let levels = self.levels.as_slice();
        let items = self.items.as_slice();

        for h in 0..self.num_levels {
            let idx = self.num_levels - 1 - h;
            let start = levels[idx];
            let end = levels[idx + 1];
            let weight = 1 << h;

            for &val in &items[start..end] {
                if val.to_f64() <= x {
                    r += weight;
                }
            }
        }
        r
    }

    /// Returns the total count of observations seen by the sketch.
    pub fn count(&self) -> usize {
        let mut total = 0;
        for h in 0..self.num_levels {
            total += self.level_size(h) * (1 << h);
        }
        total
    }

    /// Number of stored samples across all levels.
    fn buffer_size(&self) -> usize {
        self.items.len()
    }
}

impl KLLDynamic<f64> {
    /// Inserts a value from a [`DataInput`] into a `KLLDynamic<f64>` sketch.
    pub fn update_data_input(&mut self, val: &DataInput) -> Result<(), &'static str> {
        let value = data_input_to_f64(val)?;
        self.push_value(value);
        Ok(())
    }

    /// Batch variant of `update_data_input`.
    pub fn bulk_update_data_input(&mut self, values: &[DataInput]) -> Result<(), &'static str> {
        for v in values {
            let value = data_input_to_f64(v)?;
            self.push_value(value);
        }
        Ok(())
    }
}

/// The CDF for quantile queries.
pub struct DynamicCdf {
    entries: Vector1D<DynamicCdfEntry>,
}

impl DynamicCdf {
    /// Returns the quantile for value `x` using the CDF table.
    pub fn quantile(&self, x: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let slice = self.entries.as_slice();
        match slice
            .binary_search_by(|e| e.value.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(idx) => slice[idx].quantile,
            Err(0) => 0.0,
            Err(idx) => slice[idx - 1].quantile,
        }
    }

    /// Prints the CDF entries for debugging.
    pub fn print_entries(&self) {
        println!("entries: {:?}", self.entries);
    }

    /// Returns the estimated value corresponding to quantile `p`.
    pub fn query(&self, p: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let slice = self.entries.as_slice();
        match slice.binary_search_by(|e| {
            e.quantile
                .partial_cmp(&p)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(idx) => slice[idx].value,
            Err(idx) if idx == slice.len() => slice[slice.len() - 1].value,
            Err(idx) => slice[idx].value,
        }
    }

    /// Quantile estimation of value `x` using linear interpolation.
    pub fn quantile_li(&self, x: f64) -> f64 {
        let slice = self.entries.as_slice();
        if slice.is_empty() {
            return 0.0;
        }
        let idx = slice.partition_point(|e| e.value < x);
        if idx == slice.len() {
            return 1.0;
        }
        if idx == 0 {
            return 0.0;
        }
        let a = slice[idx - 1].value;
        let aq = slice[idx - 1].quantile;
        let b = slice[idx].value;
        let bq = slice[idx].quantile;
        ((a - x) * bq + (x - b) * aq) / (a - b)
    }

    /// Value estimation given quantile `p`, using linear interpolation.
    pub fn query_li(&self, p: f64) -> f64 {
        let slice = self.entries.as_slice();
        if slice.is_empty() {
            return 0.0;
        }
        let idx = slice.partition_point(|e| e.quantile < p);
        if idx == slice.len() {
            return slice[slice.len() - 1].value;
        }
        if idx == 0 {
            return slice[0].value;
        }
        let a = slice[idx - 1].value;
        let aq = slice[idx - 1].quantile;
        let b = slice[idx].value;
        let bq = slice[idx].quantile;
        ((aq - p) * b + (p - bq) * a) / (aq - bq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_uniform_f64, sample_zipf_f64};

    const SKETCH_K: i32 = 200;

    // return element from input with given quantile
    fn quantile_from_sorted(data: &[f64], quantile: f64) -> f64 {
        assert!(!data.is_empty(), "data set must not be empty");
        if quantile <= 0.0 {
            return data[0];
        }
        if quantile >= 1.0 {
            return data[data.len() - 1];
        }
        let n = data.len();
        let idx = ((quantile * n as f64).ceil() as isize - 1).clamp(0, (n - 1) as isize) as usize;
        data[idx]
    }

    fn assert_quantiles_within_error(
        sketch: &KLLDynamic,
        sorted_truth: &[f64],
        quantiles: &[(f64, &str)],
        tolerance: f64,
        context: &str,
        sample_size: usize,
        seed: u64,
    ) {
        let cdf = sketch.cdf();
        for &(quantile, label) in quantiles {
            let lower_q = (quantile - tolerance).max(0.0);
            let upper_q = (quantile + tolerance).min(1.0);
            let truth_min = quantile_from_sorted(sorted_truth, lower_q);
            let truth_max = quantile_from_sorted(sorted_truth, upper_q);
            let estimate = cdf.query(quantile);
            assert!(
                (truth_min..=truth_max).contains(&estimate),
                "{label} exceeded tolerance: context={context}, sample_size={sample_size}, seed=0x{seed:08x}, \
                quantile={quantile:.4}, truth_min={truth_min:.4}, truth_max={truth_max:.4}, \
                estimate={estimate:.4}, tolerance={tolerance:.4}, total_length={}",
                sorted_truth.len()
            );
        }
    }

    #[test]
    fn test_data_input_api() {
        let mut kll = KLLDynamic::init_kll(128);

        // Test with different numeric types
        kll.update_data_input(&DataInput::I32(10)).unwrap();
        kll.update_data_input(&DataInput::I64(20)).unwrap();
        kll.update_data_input(&DataInput::F64(30.5)).unwrap();
        kll.update_data_input(&DataInput::F32(40.2)).unwrap();
        kll.update_data_input(&DataInput::U32(50)).unwrap();

        // Query quantiles
        let cdf = kll.cdf();
        let median = cdf.query(0.5);

        // Median should be 30.5
        assert!(median > 20.0 && median < 40.2, "Median = {median}");

        // Test error handling for non-numeric input
        let result = kll.update_data_input(&DataInput::String("not a number".to_string()));
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "KLL sketch only accepts numeric inputs"
        );
    }

    #[test]
    fn test_forced_compact() {
        // force compaction to happen with small k/m
        let mut kll = KLLDynamic::init(3, 3);
        kll.update(&10.0);
        kll.update(&20.0);
        kll.update(&30.0);
        kll.update(&40.0);
        kll.update(&50.0);
        let cdf = kll.cdf();
        let median = cdf.query(0.5);
        // only 30 and 40 is possible
        assert!(median == 30.0 || median == 40.0, "Median = {median}");
    }

    #[test]
    fn test_no_compact() {
        // no compaction should happen
        let mut kll = KLLDynamic::init_kll(8);
        kll.update(&10.0);
        kll.update(&20.0);
        kll.update(&30.0);
        kll.update(&40.0);
        kll.update(&50.0);

        // Query quantiles
        let cdf = kll.cdf();
        let median = cdf.query(0.5);
        // Median should be 30
        assert!(median == 30.0, "Median = {median}");
    }

    #[test]
    fn merge_preserves_quantiles_within_tolerance() {
        const TOLERANCE: f64 = 0.02;
        const QUANTILES: &[(f64, &str)] = &[
            (0.0, "min"),
            (0.10, "p10"),
            (0.25, "p25"),
            (0.50, "p50"),
            (0.75, "p75"),
            (0.90, "p90"),
            (1.0, "max"),
        ];

        let values = sample_uniform_f64(1_000_000.0, 10_000_000.0, 10_000, 0xC0FFEE);
        let mut sketch_a = KLLDynamic::init_kll(SKETCH_K);
        let mut sketch_b = KLLDynamic::init_kll(SKETCH_K);

        for (idx, value) in values.iter().copied().enumerate() {
            if idx % 2 == 0 {
                sketch_a.update(&value);
            } else {
                sketch_b.update(&value);
            }
        }

        sketch_a.merge(&sketch_b);

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_quantiles_within_error(
            &sketch_a,
            &sorted,
            QUANTILES,
            TOLERANCE,
            "merge",
            values.len(),
            0x00C0_FFEE,
        );
    }

    #[test]
    fn merge_into_empty_target_preserves_weight_issue_68_repro() {
        let mut src = KLLDynamic::<f64>::init_kll(SKETCH_K);
        for i in 1..=1000u32 {
            src.update(&(i as f64));
        }
        let src_count = src.count();
        assert!(
            (980..=1020).contains(&src_count),
            "source sketch count before merge should track N=1000 closely, got {src_count}"
        );

        let mut dst = KLLDynamic::<f64>::init_kll(SKETCH_K);
        assert_eq!(dst.count(), 0, "target must start empty for this repro");

        dst.merge(&src);

        assert_eq!(
            dst.count(),
            src_count,
            "merge into an empty target must preserve total weight EXACTLY \
             (the old item-replay merge rescaled this down to src's \
             retained-item count)"
        );

        let median = dst.quantile(0.5);
        assert!(
            (475.0..=525.0).contains(&median),
            "merged median drifted outside tolerance: median={median}"
        );
    }

    // General case: merging two NON-empty KLLDynamic sketches must still
    // preserve total count (within the same inherent rounding budget as
    // ordinary inserts) and produce quantile estimates consistent with a
    // reference built from the union of both inputs' raw data. Guards
    // against a fix that only special-cases the empty-target repro above.
    #[test]
    fn merge_two_nonempty_sketches_preserves_weight_and_quantiles() {
        const TOLERANCE: f64 = 0.03;
        const QUANTILES: &[(f64, &str)] = &[
            (0.0, "min"),
            (0.10, "p10"),
            (0.25, "p25"),
            (0.50, "p50"),
            (0.75, "p75"),
            (0.90, "p90"),
            (1.0, "max"),
        ];

        let values_a = sample_uniform_f64(0.0, 1_000_000.0, 50_000, 0xA11CE);
        let values_b = sample_zipf_f64(0.0, 1_000_000.0, 8_192, 1.1, 50_000, 0xB0B);

        let mut a = KLLDynamic::<f64>::init_kll(SKETCH_K);
        for v in &values_a {
            a.update(v);
        }
        let mut b = KLLDynamic::<f64>::init_kll(SKETCH_K);
        for v in &values_b {
            b.update(v);
        }

        let count_a = a.count();
        let count_b = b.count();
        assert!(
            (count_a as f64 - values_a.len() as f64).abs() / (values_a.len() as f64) < 0.03,
            "sketch a count before merge diverged from N: count={count_a}, n={}",
            values_a.len()
        );
        assert!(
            (count_b as f64 - values_b.len() as f64).abs() / (values_b.len() as f64) < 0.03,
            "sketch b count before merge diverged from N: count={count_b}, n={}",
            values_b.len()
        );

        a.merge(&b);

        let merged_count = a.count() as f64;
        let expected_count = (count_a + count_b) as f64;
        assert!(
            (merged_count - expected_count).abs() / expected_count < 0.03,
            "merging two non-empty sketches must preserve total weight (within the \
             same rounding budget as ordinary inserts): merged={merged_count}, \
             expected~={expected_count} (count_a={count_a}, count_b={count_b})"
        );

        let mut union: Vec<f64> = values_a.iter().chain(values_b.iter()).copied().collect();
        union.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_quantiles_within_error(
            &a,
            &union,
            QUANTILES,
            TOLERANCE,
            "merge_two_nonempty",
            union.len(),
            0xA11C_E0B0,
        );
    }

    // Deterministic regression: level h (h >= 1) is not always a single
    // sorted run in both operands. It is for `KLL` (fixed-capacity), whose
    // `compact` maintains it as a standing invariant via merge-promotion — but
    // `KLLDynamic::compact` re-sorts a level's *entire* contents from
    // scratch on every call instead, so a level that received a
    // promotion since its last compaction can be a concatenation of
    // separately-sorted chunks, not one sorted run. `randomly_halve_up`'s
    // alternating-position subsampling is only value-order-correct on
    // truly sorted input, and doesn't sort its input itself — so feeding
    // it an unsorted level silently produces an unsorted (and thus not
    // rank-error-bounded) survivor set.
    //
    // Hand-build `other`'s level 1 in exactly that legitimate-but-unsorted
    // shape (two independently-ascending chunks concatenated out of
    // order), small enough to stay well under capacity so level 1 is
    // never itself compacted during the merge — the output then reflects
    // the construction phase's ordering assumption directly, with no
    // randomized (`Coin`) compaction able to mask the failure, keeping this
    // test deterministic.
    #[test]
    fn merge_handles_operand_level_that_is_not_a_single_sorted_run() {
        let mut other = KLLDynamic::<f64>::init(50, 4);
        other.items = Vector1D::from_vec(vec![30.0, 40.0, 10.0, 20.0, 5.0]);
        other.levels = Vector1D::from_vec(vec![0, 4, 5]);
        other.num_levels = 2;
        other.rebuild_capacity_cache();
        assert!(
            other.capacity_for_level(1) >= 4,
            "test setup needs level 1 to stay under capacity, uncompacted"
        );

        let mut dst = KLLDynamic::<f64>::init(50, 4);
        dst.merge(&other);

        let levels = dst.levels.as_slice();
        let items = dst.items.as_slice();
        for h in 1..dst.num_levels {
            let level_idx = dst.num_levels - 1 - h;
            let (s, e) = (levels[level_idx], levels[level_idx + 1]);
            assert!(
                items[s..e].windows(2).all(|w| w[0] <= w[1]),
                "level {h} is not a single ascending sorted run after merge: {:?}",
                &items[s..e]
            );
        }
    }

    #[test]
    fn cdf_handles_empty_sketch() {
        let sketch = KLLDynamic::<f64>::init_kll(64);
        let cdf = sketch.cdf();
        assert_eq!(cdf.quantile(123.0), 0.0);
        assert_eq!(cdf.query(0.5), 0.0);
        assert_eq!(cdf.query_li(0.5), 0.0);
    }

    #[test]
    fn kll_dynamic_round_trip_rmp() {
        let mut sketch = KLLDynamic::init_kll(256);
        let samples = sample_uniform_f64(0.0, 1_000_000.0, 5_000, 0xDEAD_BEEF);
        for value in &samples {
            sketch.update(value);
        }

        let bytes = sketch
            .serialize_to_bytes()
            .expect("serialize KLLDynamic with rmp");
        assert!(!bytes.is_empty(), "serialized bytes should not be empty");

        let restored =
            KLLDynamic::deserialize_from_bytes(&bytes).expect("deserialize KLLDynamic with rmp");
        assert_eq!(sketch.k, restored.k);
        assert_eq!(sketch.m, restored.m);
        assert_eq!(sketch.num_levels, restored.num_levels);
        assert_eq!(sketch.top_height, restored.top_height);
        assert_eq!(sketch.level0_capacity, restored.level0_capacity);
        assert_eq!(
            sketch.levels.as_slice(),
            restored.levels.as_slice(),
            "level boundaries changed after round-trip"
        );
        assert_eq!(
            sketch.items.as_slice(),
            restored.items.as_slice(),
            "packed items changed after round-trip"
        );

        let quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
        let original_cdf = sketch.cdf();
        let restored_cdf = restored.cdf();
        for &q in &quantiles {
            assert!(
                (original_cdf.query(q) - restored_cdf.query(q)).abs() < f64::EPSILON,
                "quantile mismatch at p={q}: original={}, restored={}",
                original_cdf.query(q),
                restored_cdf.query(q)
            );
        }
    }

    #[test]
    fn generic_kll_dynamic_i64_sanity() {
        let mut sketch = KLLDynamic::<i64>::init_kll(200);
        let n: i64 = 20_000;
        for v in 1..=n {
            sketch.update(&v);
        }

        let count = sketch.count() as f64;
        assert!(
            (count - n as f64).abs() / (n as f64) < 0.05,
            "count={count} diverged from n={n}"
        );

        let cdf = sketch.cdf();
        let p50 = cdf.query(0.5);
        let p90 = cdf.query(0.9);
        let tol = n as f64 * 0.02;
        assert!(
            (p50 - (n as f64 * 0.5)).abs() < tol,
            "p50={p50} out of range for n={n}"
        );
        assert!(
            (p90 - (n as f64 * 0.9)).abs() < tol,
            "p90={p90} out of range for n={n}"
        );

        let bytes = sketch
            .serialize_to_bytes()
            .expect("serialize KLLDynamic<i64>");
        let restored =
            KLLDynamic::<i64>::deserialize_from_bytes(&bytes).expect("deserialize KLLDynamic<i64>");
        assert_eq!(sketch.count(), restored.count());
    }

    #[test]
    fn bulk_update_equivalent_to_loop() {
        let vals = sample_uniform_f64(0.0, 1000.0, 10_000, 0xCAFE_1234);
        let mut a = KLLDynamic::<f64>::init_kll(200);
        let mut b = KLLDynamic::<f64>::init_kll(200);
        for v in &vals {
            a.update(v);
        }
        b.bulk_update(&vals);
        // KLLDynamic is non-deterministic (Coin::new wall-clock), so counts
        // and quantiles are only approximately equal — same tolerance as
        // merge tests.
        let ca = a.count() as f64;
        let cb = b.count() as f64;
        assert!(
            (ca - cb).abs() / ca < 0.05,
            "count diverged: loop={ca}, bulk={cb}"
        );
        // Quantiles are non-deterministic for KLLDynamic (wall-clock coin),
        // so large-batch quantile comparison is omitted here; small-batch
        // exact check below covers determinism.

        // DataInput batch equivalence — small batch (100) avoids compaction
        // non-determinism, so exact.
        let di: Vec<DataInput> = vals[..100].iter().map(|v| DataInput::F64(*v)).collect();
        let mut c = KLLDynamic::<f64>::init_kll(200);
        let mut d = KLLDynamic::<f64>::init_kll(200);
        for v in &di {
            c.update_data_input(v).unwrap();
        }
        d.bulk_update_data_input(&di).unwrap();
        assert_eq!(c.count(), d.count());
    }
}
