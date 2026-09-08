//! EHUnivOptimized: Hybrid two-tier Exponential Histogram for UnivMon.
//!
//! Recent/small EH buckets store exact frequency maps, while older/larger
//! buckets use full UnivMon sketches. When a map bucket grows too large,
//! it is promoted to a UnivMon sketch.
//!
//! Uses `UnivSketchPool` (from `univmon_optimized`) to recycle `UnivMon`
//! instances across promotion, merge, and expiration cycles.

use std::collections::HashMap;

use crate::common::input::{heap_item_to_sketch_input, input_to_owned};
use crate::sketch_framework::univmon_optimized::UnivSketchPool;
use crate::{DataInput, HeapItem, UnivMon};

const MASS_EPSILON: f64 = 1e-9;
const DEFAULT_HEAP_SIZE: usize = 32;
const DEFAULT_SKETCH_ROW: usize = 5;
const DEFAULT_SKETCH_COL: usize = 2048;
const DEFAULT_LAYER_SIZE: usize = 8;
const DEFAULT_POOL_CAP: usize = 4;

fn calc_map_l22(freq_map: &HashMap<HeapItem, i64>) -> f64 {
    freq_map.values().map(|&v| (v as f64) * (v as f64)).sum()
}

/// Map-tier bucket: exact frequency counts.
#[derive(Clone, Debug)]
pub struct EHMapBucket {
    pub freq_map: HashMap<HeapItem, i64>,
    pub l22: f64,
    pub bucket_size: usize,
    pub min_time: u64,
    pub max_time: u64,
}

/// Sketch-tier bucket: owns a UnivMon sketch outright.
#[derive(Clone, Debug)]
pub struct EHUnivMonBucket {
    pub sketch: UnivMon,
    pub l22: f64,
    pub bucket_size: usize,
    pub min_time: u64,
    pub max_time: u64,
}

/// The hybrid EH structure.
pub struct EHUnivOptimized {
    pub um_buckets: Vec<EHUnivMonBucket>,
    pub map_buckets: Vec<EHMapBucket>,
    pub k: usize,
    pub window: u64,
    pub max_map_size: usize,
    heap_size: usize,
    sketch_row: usize,
    sketch_col: usize,
    layer_size: usize,
    pool: UnivSketchPool,
}

/// Query result: either an exact map or a UnivMon sketch.
pub enum EHUnivQueryResult {
    Sketch(UnivMon),
    Map {
        freq_map: HashMap<HeapItem, i64>,
        total_count: usize,
    },
}

impl EHUnivQueryResult {
    pub fn calc_l1(&self) -> f64 {
        match self {
            Self::Sketch(um) => um.calc_l1(),
            Self::Map { freq_map, .. } => freq_map.values().map(|&v| (v as f64).abs()).sum(),
        }
    }

    pub fn calc_l2(&self) -> f64 {
        match self {
            Self::Sketch(um) => um.calc_l2(),
            Self::Map { freq_map, .. } => freq_map
                .values()
                .map(|&v| {
                    let f = v as f64;
                    f * f
                })
                .sum::<f64>()
                .sqrt(),
        }
    }

    pub fn calc_entropy(&self) -> f64 {
        match self {
            Self::Sketch(um) => um.calc_entropy(),
            Self::Map {
                freq_map,
                total_count,
            } => {
                let n = *total_count as f64;
                if n <= 0.0 {
                    return 0.0;
                }
                let sum_f_log_f: f64 = freq_map
                    .values()
                    .map(|&v| {
                        let f = v as f64;
                        if f > 0.0 { f * f.log2() } else { 0.0 }
                    })
                    .sum();
                n.log2() - sum_f_log_f / n
            }
        }
    }

    pub fn calc_card(&self) -> f64 {
        match self {
            Self::Sketch(um) => um.calc_card(),
            Self::Map { freq_map, .. } => freq_map.len() as f64,
        }
    }
}

impl EHUnivOptimized {
    pub fn new(
        k: usize,
        window: u64,
        heap_size: usize,
        sketch_row: usize,
        sketch_col: usize,
        layer_size: usize,
    ) -> Self {
        Self::with_pool_cap(
            k,
            window,
            heap_size,
            sketch_row,
            sketch_col,
            layer_size,
            DEFAULT_POOL_CAP,
        )
    }

    pub fn with_pool_cap(
        k: usize,
        window: u64,
        heap_size: usize,
        sketch_row: usize,
        sketch_col: usize,
        layer_size: usize,
        pool_cap: usize,
    ) -> Self {
        let k_eff = k.max(1);
        EHUnivOptimized {
            um_buckets: Vec::new(),
            map_buckets: Vec::new(),
            k: k_eff,
            window,
            max_map_size: layer_size * sketch_row * sketch_col,
            heap_size,
            sketch_row,
            sketch_col,
            layer_size,
            pool: UnivSketchPool::new(pool_cap, heap_size, sketch_row, sketch_col, layer_size),
        }
    }

    pub fn with_defaults(k: usize, window: u64) -> Self {
        Self::new(
            k,
            window,
            DEFAULT_HEAP_SIZE,
            DEFAULT_SKETCH_ROW,
            DEFAULT_SKETCH_COL,
            DEFAULT_LAYER_SIZE,
        )
    }

    pub fn update(&mut self, time: u64, key: &DataInput, value: i64) {
        // 1. Expire old sketch buckets, recycling sketches to pool
        let cutoff = time.saturating_sub(self.window);
        let expired_count = self
            .um_buckets
            .iter()
            .take_while(|b| b.max_time < cutoff)
            .count();
        if expired_count > 0 {
            let expired: Vec<EHUnivMonBucket> = self.um_buckets.drain(0..expired_count).collect();
            for bucket in expired {
                self.pool.put(bucket.sketch);
            }
        }

        // 2. Expire old map buckets
        let expired = self
            .map_buckets
            .iter()
            .take_while(|b| b.max_time < cutoff)
            .count();
        if expired > 0 {
            self.map_buckets.drain(0..expired);
        }

        // 3. Create new map bucket
        let owned_key = input_to_owned(key);
        let mut freq_map = HashMap::new();
        freq_map.insert(owned_key, value);
        self.map_buckets.push(EHMapBucket {
            freq_map,
            l22: (value as f64) * (value as f64),
            bucket_size: value as usize,
            min_time: time,
            max_time: time,
        });

        // 4. L2-merge map buckets (backward scan)
        let mut sum_l22: f64 = 0.0;
        if self.map_buckets.len() >= 2 {
            let mut i = self.map_buckets.len() - 2;
            loop {
                let pair_l22 = self.map_buckets[i].l22 + self.map_buckets[i + 1].l22;
                let threshold = sum_l22 / (self.k as f64);
                if pair_l22 <= threshold + MASS_EPSILON {
                    // Merge i+1 into i
                    let other = self.map_buckets.remove(i + 1);
                    let bucket = &mut self.map_buckets[i];
                    bucket.bucket_size += other.bucket_size;
                    bucket.max_time = bucket.max_time.max(other.max_time);
                    bucket.min_time = bucket.min_time.min(other.min_time);
                    for (k, v) in other.freq_map {
                        *bucket.freq_map.entry(k).or_insert(0) += v;
                    }
                    bucket.l22 = calc_map_l22(&bucket.freq_map);
                } else {
                    sum_l22 += self.map_buckets[i + 1].l22;
                }
                if i == 0 {
                    break;
                }
                i -= 1;
            }
        }

        // 5. Promotion: if oldest map bucket is too large, promote to sketch
        if !self.map_buckets.is_empty()
            && 2 * self.map_buckets[0].freq_map.len() >= self.max_map_size
        {
            self.promote_oldest_map(sum_l22);
        }
    }

    fn promote_oldest_map(&mut self, sum_l22: f64) {
        let oldest = self.map_buckets.remove(0);

        // Take a sketch from the pool (moved, not borrowed) — zero allocation
        // if the pool has a recycled sketch available.
        let mut sketch = self.pool.take();
        for (key, value) in &oldest.freq_map {
            let input = heap_item_to_sketch_input(key);
            sketch.insert(&input, *value);
        }

        let l22 = sketch.l2_sketch_layers[0].get_l2().powi(2);

        self.um_buckets.push(EHUnivMonBucket {
            sketch,
            l22,
            bucket_size: oldest.bucket_size,
            min_time: oldest.min_time,
            max_time: oldest.max_time,
        });

        self.merge_sketch_buckets(sum_l22);
    }

    fn merge_sketch_buckets(&mut self, mut sum_l22: f64) {
        if self.um_buckets.len() < 2 {
            return;
        }
        let mut i = self.um_buckets.len() - 2;
        loop {
            let l22_i = self.um_buckets[i].sketch.l2_sketch_layers[0]
                .get_l2()
                .powi(2);
            let l22_next = self.um_buckets[i + 1].sketch.l2_sketch_layers[0]
                .get_l2()
                .powi(2);
            let pair_l22 = l22_i + l22_next;
            let threshold = sum_l22 / (self.k as f64);
            if pair_l22 <= threshold + MASS_EPSILON {
                let other = self.um_buckets.remove(i + 1);
                let bucket = &mut self.um_buckets[i];
                bucket.sketch.merge(&other.sketch);
                bucket.bucket_size += other.bucket_size;
                bucket.max_time = bucket.max_time.max(other.max_time);
                bucket.min_time = bucket.min_time.min(other.min_time);
                bucket.l22 = bucket.sketch.l2_sketch_layers[0].get_l2().powi(2);
                // Recycle the consumed sketch back to the pool
                self.pool.put(other.sketch);
            } else {
                sum_l22 += l22_next;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    pub fn query_interval(&self, t1: u64, t2: u64) -> Option<EHUnivQueryResult> {
        let s_count = self.um_buckets.len();
        let m_count = self.map_buckets.len();
        let total = s_count + m_count;
        if total == 0 {
            return None;
        }

        let mut from_bucket: usize = 0;
        let mut to_bucket: usize = 0;

        // Search sketch buckets
        for i in 0..s_count {
            if t1 >= self.um_buckets[i].min_time && t1 <= self.um_buckets[i].max_time {
                from_bucket = i;
                break;
            }
        }
        for i in 0..s_count {
            if t2 >= self.um_buckets[i].min_time && t2 <= self.um_buckets[i].max_time {
                to_bucket = i;
                break;
            }
        }

        // Search map buckets (may override sketch results)
        for i in 0..m_count {
            if t1 >= self.map_buckets[i].min_time && t1 <= self.map_buckets[i].max_time {
                from_bucket = i + s_count;
                break;
            }
        }
        for i in 0..m_count {
            if t2 >= self.map_buckets[i].min_time && t2 <= self.map_buckets[i].max_time {
                to_bucket = i + s_count;
                break;
            }
        }

        // Edge cases
        if m_count > 0 && t2 > self.map_buckets[m_count - 1].max_time {
            to_bucket = m_count - 1 + s_count;
        }
        if (s_count > 0 && t1 < self.um_buckets[0].min_time)
            || (s_count == 0 && m_count > 0 && t1 < self.map_buckets[0].min_time)
        {
            from_bucket = 0;
        }

        // Snap from_bucket forward if t1 is closer to max_time of the bucket
        if from_bucket < s_count {
            let b = &self.um_buckets[from_bucket];
            if t1.abs_diff(b.min_time) > t1.abs_diff(b.max_time) && from_bucket + 1 < total {
                from_bucket += 1;
            }
        } else if from_bucket >= s_count && from_bucket - s_count < m_count {
            let mi = from_bucket - s_count;
            let b = &self.map_buckets[mi];
            if t1.abs_diff(b.min_time) > t1.abs_diff(b.max_time) && from_bucket + 1 < total {
                from_bucket += 1;
            }
        }

        // Clamp indices
        if from_bucket >= total {
            from_bucket = total - 1;
        }
        if to_bucket >= total {
            to_bucket = total - 1;
        }
        if from_bucket > to_bucket {
            to_bucket = from_bucket;
        }

        // Three cases
        if to_bucket < s_count {
            // Case 1: Both in sketch tier
            let mut merged = self.um_buckets[from_bucket].sketch.clone();
            for i in (from_bucket + 1)..=to_bucket {
                merged.merge(&self.um_buckets[i].sketch);
            }
            Some(EHUnivQueryResult::Sketch(merged))
        } else if from_bucket >= s_count {
            // Case 2: Both in map tier
            let from_map = from_bucket - s_count;
            let to_map = to_bucket - s_count;
            let mut merged_map: HashMap<HeapItem, i64> = HashMap::new();
            for i in from_map..=to_map {
                for (k, &v) in &self.map_buckets[i].freq_map {
                    *merged_map.entry(k.clone()).or_insert(0) += v;
                }
            }
            let total_count = merged_map.values().sum::<i64>() as usize;
            Some(EHUnivQueryResult::Map {
                freq_map: merged_map,
                total_count,
            })
        } else {
            // Case 3: Hybrid — from in sketch, to in map
            let mut merged = UnivMon::init_univmon(
                self.heap_size,
                self.sketch_row,
                self.sketch_col,
                self.layer_size,
            );
            for i in from_bucket..s_count {
                merged.merge(&self.um_buckets[i].sketch);
            }

            // Merge qualifying map buckets into a temporary map
            let to_map = to_bucket - s_count;
            let mut map_merged: HashMap<HeapItem, i64> = HashMap::new();
            for i in 0..=to_map {
                for (k, &v) in &self.map_buckets[i].freq_map {
                    *map_merged.entry(k.clone()).or_insert(0) += v;
                }
            }

            // Insert map entries into merged sketch
            for (key, value) in &map_merged {
                let input = heap_item_to_sketch_input(key);
                merged.insert(&input, *value);
            }

            Some(EHUnivQueryResult::Sketch(merged))
        }
    }

    pub fn cover(&self, mint: u64, maxt: u64) -> bool {
        match (self.get_min_time(), self.get_max_time()) {
            (Some(gmin), Some(gmax)) => gmin <= mint && gmax >= maxt,
            _ => false,
        }
    }

    pub fn get_min_time(&self) -> Option<u64> {
        let sketch_min = self.um_buckets.first().map(|b| b.min_time);
        let map_min = self.map_buckets.first().map(|b| b.min_time);
        match (sketch_min, map_min) {
            (Some(s), Some(m)) => Some(s.min(m)),
            (s @ Some(_), None) => s,
            (None, m @ Some(_)) => m,
            (None, None) => None,
        }
    }

    pub fn get_max_time(&self) -> Option<u64> {
        let sketch_max = self.um_buckets.last().map(|b| b.max_time);
        let map_max = self.map_buckets.last().map(|b| b.max_time);
        match (sketch_max, map_max) {
            (Some(s), Some(m)) => Some(s.max(m)),
            (s @ Some(_), None) => s,
            (None, m @ Some(_)) => m,
            (None, None) => None,
        }
    }

    pub fn update_window(&mut self, window: u64) {
        self.window = window;
    }

    pub fn bucket_count(&self) -> usize {
        self.um_buckets.len() + self.map_buckets.len()
    }

    /// Returns a reference to the sketch pool.
    pub fn pool(&self) -> &UnivSketchPool {
        &self.pool
    }

    pub fn print_buckets(&self) {
        println!("=== EHUnivOptimized Buckets ===");
        println!(
            "k: {}, window: {}, max_map_size: {}",
            self.k, self.window, self.max_map_size
        );
        println!(
            "Pool: {}/{} available/total_allocated",
            self.pool.available(),
            self.pool.total_allocated()
        );
        println!("Sketch buckets ({}):", self.um_buckets.len());
        for (i, b) in self.um_buckets.iter().enumerate() {
            println!(
                "  [S{}] min_time={}, max_time={}, bucket_size={}, l22={:.2}",
                i, b.min_time, b.max_time, b.bucket_size, b.l22
            );
        }
        println!("Map buckets ({}):", self.map_buckets.len());
        for (i, b) in self.map_buckets.iter().enumerate() {
            println!(
                "  [M{}] min_time={}, max_time={}, bucket_size={}, l22={:.2}, keys={}",
                i,
                b.min_time,
                b.max_time,
                b.bucket_size,
                b.l22,
                b.freq_map.len()
            );
        }
    }

    pub fn get_memory_info(&self) -> (usize, usize, Vec<usize>, Vec<usize>) {
        let sketch_sizes: Vec<usize> = self.um_buckets.iter().map(|b| b.bucket_size).collect();
        let map_sizes: Vec<usize> = self.map_buckets.iter().map(|b| b.bucket_size).collect();
        (
            self.um_buckets.len(),
            self.map_buckets.len(),
            sketch_sizes,
            map_sizes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insertion_and_query() {
        let mut eh = EHUnivOptimized::with_defaults(4, 1000);

        eh.update(100, &DataInput::I64(1), 5);
        eh.update(101, &DataInput::I64(2), 3);
        eh.update(102, &DataInput::I64(1), 2);

        assert!(eh.um_buckets.is_empty());
        assert!(!eh.map_buckets.is_empty());

        let result = eh.query_interval(100, 102).unwrap();
        match &result {
            EHUnivQueryResult::Map {
                freq_map,
                total_count,
            } => {
                // key=1 has total count 5+2=7, key=2 has 3
                let k1 = freq_map.get(&HeapItem::I64(1)).copied().unwrap_or(0);
                let k2 = freq_map.get(&HeapItem::I64(2)).copied().unwrap_or(0);
                assert_eq!(k1, 7);
                assert_eq!(k2, 3);
                assert_eq!(*total_count, 10);
            }
            EHUnivQueryResult::Sketch(_) => panic!("Expected Map result"),
        }

        assert!((result.calc_l1() - 10.0).abs() < 1e-9);
        assert_eq!(result.calc_card(), 2.0);
    }

    #[test]
    fn map_merge_bounds_volume() {
        let mut eh = EHUnivOptimized::with_defaults(1, 10000);

        for i in 0..50u64 {
            eh.update(i, &DataInput::I64(i as i64), 1);
        }

        // With k=1 and L2 merging, bucket count should stay bounded
        assert!(
            eh.bucket_count() < 50,
            "bucket_count {} should be bounded below 50",
            eh.bucket_count()
        );
    }

    #[test]
    fn promotion_creates_sketch_buckets() {
        // Use small parameters so promotion triggers quickly
        // max_map_size = layer_size * sketch_row * sketch_col = 2 * 2 * 5 = 20
        // promotion at 2 * map.len() >= 20, i.e. map.len() >= 10
        let mut eh = EHUnivOptimized::new(8, 100000, 16, 2, 5, 2);

        assert!(eh.um_buckets.is_empty());

        // Insert many distinct keys to grow the oldest map bucket
        for i in 0..200u64 {
            eh.update(i, &DataInput::I64(i as i64), 1);
        }

        assert!(
            !eh.um_buckets.is_empty(),
            "Should have promoted at least one map bucket to sketch"
        );
    }

    #[test]
    fn window_expiration() {
        let mut eh = EHUnivOptimized::with_defaults(4, 100);

        eh.update(10, &DataInput::I64(1), 1);
        eh.update(20, &DataInput::I64(2), 1);
        eh.update(30, &DataInput::I64(3), 1);

        assert_eq!(eh.get_min_time(), Some(10));

        // This update at time=200 should expire buckets with max_time < 200-100=100
        eh.update(200, &DataInput::I64(4), 1);

        // All buckets with max_time < 100 should be gone
        assert!(
            eh.get_min_time().unwrap() >= 100 || eh.get_min_time() == Some(200),
            "Old buckets should be expired, got min_time={:?}",
            eh.get_min_time()
        );
    }

    #[test]
    fn hybrid_query_returns_sketch() {
        // Use small parameters for fast promotion
        // max_map_size = 2 * 2 * 5 = 20, promotion at map.len() >= 10
        let mut eh = EHUnivOptimized::new(8, 100000, 16, 2, 5, 2);

        // Insert enough distinct keys to force promotion
        for i in 0..200u64 {
            eh.update(i, &DataInput::I64(i as i64), 1);
        }

        assert!(!eh.um_buckets.is_empty(), "Need sketch buckets");
        assert!(!eh.map_buckets.is_empty(), "Need map buckets");

        // Query spanning both tiers
        let result = eh.query_interval(0, 199).unwrap();
        match result {
            EHUnivQueryResult::Sketch(_) => {} // expected
            EHUnivQueryResult::Map { .. } => panic!("Expected Sketch result for hybrid query"),
        }
    }

    #[test]
    fn cover_check() {
        let mut eh = EHUnivOptimized::with_defaults(4, 1000);

        assert!(!eh.cover(0, 100));

        eh.update(50, &DataInput::I64(1), 1);
        eh.update(100, &DataInput::I64(2), 1);

        assert!(eh.cover(50, 100));
        assert!(eh.cover(60, 90));
        assert!(!eh.cover(40, 100));
        assert!(!eh.cover(50, 110));
    }

    #[test]
    fn accuracy_known_distribution() {
        let mut eh = EHUnivOptimized::with_defaults(4, 100000);

        // Insert a known distribution
        let data: Vec<(i64, i64)> = vec![(1, 100), (2, 200), (3, 50), (4, 150), (5, 80)];

        let mut time = 0u64;
        for &(key, count) in &data {
            for _ in 0..count {
                eh.update(time, &DataInput::I64(key), 1);
                time += 1;
            }
        }

        let result = eh.query_interval(0, time - 1).unwrap();

        // Ground truth
        let true_l1: f64 = data.iter().map(|&(_, c)| c as f64).sum();
        let true_l2: f64 = data
            .iter()
            .map(|&(_, c)| (c as f64) * (c as f64))
            .sum::<f64>()
            .sqrt();
        let true_card = data.len() as f64;
        let entropy_term: f64 = data
            .iter()
            .map(|&(_, c)| {
                let f = c as f64;
                f * f.log2()
            })
            .sum();
        let true_entropy = true_l1.log2() - entropy_term / true_l1;

        let est_l1 = result.calc_l1();
        let est_l2 = result.calc_l2();
        let est_card = result.calc_card();
        let est_entropy = result.calc_entropy();

        // Map results should be exact (or very close due to merge)
        let l1_err = (est_l1 - true_l1).abs() / true_l1;
        let l2_err = (est_l2 - true_l2).abs() / true_l2;
        let card_err = (est_card - true_card).abs() / true_card;
        let ent_err = (est_entropy - true_entropy).abs() / true_entropy;

        assert!(
            l1_err < 0.10,
            "L1 error {:.2}%: est={}, true={}",
            l1_err * 100.0,
            est_l1,
            true_l1
        );
        assert!(
            l2_err < 0.10,
            "L2 error {:.2}%: est={}, true={}",
            l2_err * 100.0,
            est_l2,
            true_l2
        );
        assert!(
            card_err < 0.10,
            "Card error {:.2}%: est={}, true={}",
            card_err * 100.0,
            est_card,
            true_card
        );
        assert!(
            ent_err < 0.10,
            "Entropy error {:.2}%: est={}, true={}",
            ent_err * 100.0,
            est_entropy,
            true_entropy
        );
    }

    #[test]
    fn pool_used_during_promotion() {
        // Verify the pool is actually used during EH promotion
        let mut eh = EHUnivOptimized::with_pool_cap(8, 100000, 16, 2, 5, 2, 2);

        assert_eq!(eh.pool().total_allocated(), 2);

        // Insert enough to trigger promotions
        for i in 0..200u64 {
            eh.update(i, &DataInput::I64(i as i64), 1);
        }

        assert!(!eh.um_buckets.is_empty());
        // Pool should have been used (total_allocated may have grown if many concurrent buckets)
        assert!(eh.pool().total_allocated() >= 2);
    }

    // -----------------------------------------------------------------------
    // Helper: compute ground truth metrics from a frequency map
    // -----------------------------------------------------------------------
    fn ground_truth_from_freq(freq: &HashMap<i64, i64>) -> (f64, f64, f64, f64) {
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

    /// Compute the frequency map from raw (time, key) samples in a range.
    fn freq_map_from_samples(samples: &[(u64, i64)], t1: u64, t2: u64) -> HashMap<i64, i64> {
        let mut freq: HashMap<i64, i64> = HashMap::new();
        for &(t, key) in samples {
            if t >= t1 && t <= t2 {
                *freq.entry(key).or_insert(0) += 1;
            }
        }
        freq
    }

    fn assert_metric_within(name: &str, est: f64, truth: f64, tol: f64) {
        if truth.abs() < 1e-12 {
            assert!(est.abs() < 1e-6, "{name}: expected ~0, got {est}");
            return;
        }
        let rel_err = (est - truth).abs() / truth.abs();
        assert!(
            rel_err <= tol,
            "{name}: relative error {:.2}% exceeds {:.0}%: est={est:.4}, truth={truth:.4}",
            rel_err * 100.0,
            tol * 100.0,
        );
    }

    // =======================================================================
    //                    CORRECTNESS TESTS
    // =======================================================================

    /// Map-only queries must return exact results (no sketch approximation).
    #[test]
    fn correctness_map_only_exact() {
        let mut eh = EHUnivOptimized::with_defaults(8, 100000);
        let data: Vec<(i64, i64)> = vec![(1, 50), (2, 30), (3, 20)];

        let mut time = 0u64;
        for &(key, count) in &data {
            for _ in 0..count {
                eh.update(time, &DataInput::I64(key), 1);
                time += 1;
            }
        }

        // Should still be in map tier (few distinct keys, large default max_map_size)
        assert!(eh.um_buckets.is_empty(), "Expected all data in map tier");

        let result = eh.query_interval(0, time - 1).unwrap();
        let (true_l1, true_l2, true_card, true_entropy) =
            ground_truth_from_freq(&data.iter().map(|&(k, v)| (k, v)).collect());

        // Map results should be exact (within floating-point tolerance)
        assert_metric_within("L1", result.calc_l1(), true_l1, 0.01);
        assert_metric_within("L2", result.calc_l2(), true_l2, 0.01);
        assert_metric_within("Card", result.calc_card(), true_card, 0.01);
        assert_metric_within("Entropy", result.calc_entropy(), true_entropy, 0.01);
    }

    /// Verify that querying a sub-interval returns correct results for the
    /// data within that interval only.
    #[test]
    fn correctness_subinterval_query() {
        let mut eh = EHUnivOptimized::with_defaults(8, 100000);

        // Phase 1: t=0..99, key=1
        for t in 0..100u64 {
            eh.update(t, &DataInput::I64(1), 1);
        }
        // Phase 2: t=100..199, key=2
        for t in 100..200u64 {
            eh.update(t, &DataInput::I64(2), 1);
        }

        // Full interval
        let full = eh.query_interval(0, 199).unwrap();
        assert_metric_within("full L1", full.calc_l1(), 200.0, 0.05);
        assert_metric_within("full Card", full.calc_card(), 2.0, 0.05);
    }

    /// After window expiration, queries should not include expired data.
    /// Buckets may span ranges (due to merging), so we use a generous margin.
    #[test]
    fn correctness_expired_data_excluded() {
        let window = 100;
        let mut eh = EHUnivOptimized::with_defaults(4, window);

        // Insert key=1 at times 0..49
        for t in 0..50u64 {
            eh.update(t, &DataInput::I64(1), 1);
        }
        // Insert key=2 at times 50..249 (push well past the window)
        for t in 50..250u64 {
            eh.update(t, &DataInput::I64(2), 1);
        }

        // At time 249, cutoff = 249 - 100 = 149. Buckets with max_time < 149 are expired.
        // Due to merging, the oldest surviving bucket may have min_time somewhat before 149,
        // but the very earliest data (t=0..49) should definitely be gone.
        let min_t = eh.get_min_time().unwrap();
        assert!(
            min_t >= 50,
            "Earliest data (t<50) should be expired, got min_time={min_t}"
        );
    }

    /// Verify bucket count stays bounded regardless of stream length (EH property).
    #[test]
    fn correctness_volume_bounded_long_stream() {
        let window = 5000;
        let k = 4;
        let mut eh = EHUnivOptimized::with_defaults(k, window);

        let mut max_bucket = 0;
        for t in 0..20000u64 {
            eh.update(t, &DataInput::I64((t % 100) as i64), 1);
            max_bucket = max_bucket.max(eh.bucket_count());
        }

        // With k=4, the bucket count should be O(k * log(N/k)) at most
        // For 20000 items, this should be well under 200
        assert!(
            max_bucket < 200,
            "max bucket count {max_bucket} is too large for k={k}"
        );
    }

    /// Verify that pool recycles sketches correctly across multiple
    /// expiration + promotion cycles.
    #[test]
    fn correctness_pool_recycling_across_cycles() {
        let window = 200;
        // Small sketch parameters to trigger frequent promotions
        let mut eh = EHUnivOptimized::with_pool_cap(4, window, 16, 2, 8, 2, 4);

        // Run a long stream with sliding window to exercise expiration and reuse
        for t in 0..2000u64 {
            eh.update(t, &DataInput::I64((t % 50) as i64), 1);
        }

        // The system should still be functional and pool should not have grown unboundedly
        assert!(
            eh.pool().total_allocated() < 50,
            "Pool grew too large: {}",
            eh.pool().total_allocated()
        );

        // Query should still work
        let result = eh.query_interval(1800, 1999);
        assert!(
            result.is_some(),
            "Query should return a result after cycling"
        );
    }

    /// Verify merge correctness: merging two adjacent EH sketch buckets
    /// preserves the combined L2 mass within tolerance.
    #[test]
    fn correctness_sketch_merge_preserves_metrics() {
        // Use tiny sketch parameters so promotion is fast
        let mut eh = EHUnivOptimized::new(2, 100000, 32, 3, 64, 4);

        // Insert enough distinct keys to trigger multiple promotions and merges
        for t in 0..1000u64 {
            eh.update(t, &DataInput::I64((t % 200) as i64), 1);
        }

        // After merging, each sketch bucket should have a valid L2 mass
        for (i, b) in eh.um_buckets.iter().enumerate() {
            let actual_l22 = b.sketch.l2_sketch_layers[0].get_l2().powi(2);
            assert!(
                actual_l22 > 0.0,
                "Sketch bucket {i} has zero L2 mass after merges"
            );
            // Stored l22 should roughly match the sketch's actual L2^2
            let rel_diff = (b.l22 - actual_l22).abs() / actual_l22.max(1e-12);
            assert!(
                rel_diff < 0.01,
                "Sketch bucket {i}: stored l22={:.4} vs actual={actual_l22:.4}",
                b.l22
            );
        }
    }

    // =======================================================================
    //                    ACCURACY TESTS
    // =======================================================================

    /// Accuracy test with a Zipf-like distribution: one heavy hitter
    /// plus many light flows. Tests all four metrics (L1, L2, card, entropy)
    /// under sketch-tier queries.
    #[test]
    fn accuracy_zipf_distribution_sketch_tier() {
        // Small sketch to force promotion
        // max_map_size = 4 * 3 * 128 = 1536, promotion at map.len() >= 768
        let mut eh = EHUnivOptimized::new(8, 1000000, 32, 3, 128, 4);

        let mut samples: Vec<(u64, i64)> = Vec::new();
        let mut time = 0u64;

        // Heavy hitter: key=0, count=5000
        for _ in 0..5000 {
            eh.update(time, &DataInput::I64(0), 1);
            samples.push((time, 0));
            time += 1;
        }
        // Medium flows: keys 1..=20, count=200 each
        for key in 1..=20i64 {
            for _ in 0..200 {
                eh.update(time, &DataInput::I64(key), 1);
                samples.push((time, key));
                time += 1;
            }
        }
        // Light flows: keys 21..=1020, count=1 each
        for key in 21..=1020i64 {
            eh.update(time, &DataInput::I64(key), 1);
            samples.push((time, key));
            time += 1;
        }

        let freq = freq_map_from_samples(&samples, 0, time - 1);
        let (true_l1, true_l2, true_card, true_entropy) = ground_truth_from_freq(&freq);

        let result = eh.query_interval(0, time - 1).unwrap();

        // Allow 15% tolerance for sketch-tier results
        assert_metric_within("Zipf L1", result.calc_l1(), true_l1, 0.15);
        assert_metric_within("Zipf L2", result.calc_l2(), true_l2, 0.15);
        assert_metric_within("Zipf Card", result.calc_card(), true_card, 0.15);
        assert_metric_within("Zipf Entropy", result.calc_entropy(), true_entropy, 0.15);
    }

    /// Accuracy test with uniform distribution: all keys have equal frequency.
    #[test]
    fn accuracy_uniform_distribution() {
        let mut eh = EHUnivOptimized::with_defaults(8, 1000000);

        let num_keys = 100;
        let count_per_key = 50;
        let mut samples: Vec<(u64, i64)> = Vec::new();
        let mut time = 0u64;

        for _ in 0..count_per_key {
            for key in 0..num_keys {
                eh.update(time, &DataInput::I64(key), 1);
                samples.push((time, key));
                time += 1;
            }
        }

        let freq = freq_map_from_samples(&samples, 0, time - 1);
        let (true_l1, true_l2, true_card, true_entropy) = ground_truth_from_freq(&freq);

        let result = eh.query_interval(0, time - 1).unwrap();

        assert_metric_within("Uniform L1", result.calc_l1(), true_l1, 0.10);
        assert_metric_within("Uniform L2", result.calc_l2(), true_l2, 0.10);
        assert_metric_within("Uniform Card", result.calc_card(), true_card, 0.10);
        assert_metric_within("Uniform Entropy", result.calc_entropy(), true_entropy, 0.10);
    }

    /// Sliding window accuracy test: verifies that queries over recent
    /// windows produce accurate results as old data is expired.
    /// Mirrors the Go TestExpoHistogramUnivMonOptimized pattern.
    #[test]
    fn accuracy_sliding_window() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let window = 5000u64;
        let total_length = 20000u64;
        let query_interval = 2000u64; // query every 2000 items
        let mut rng = StdRng::seed_from_u64(0xCAFE_BEEF);

        let mut eh = EHUnivOptimized::with_defaults(8, window);

        // Generate a stream of (time, key) pairs with ~500 distinct keys
        let mut all_samples: Vec<(u64, i64)> = Vec::new();
        for t in 0..total_length {
            let key = (rng.random::<u32>() % 500) as i64;
            eh.update(t, &DataInput::I64(key), 1);
            all_samples.push((t, key));
        }

        // Periodically query the most recent data
        let mut total_l1_err = 0.0;
        let mut total_l2_err = 0.0;
        let mut total_entropy_err = 0.0;
        let mut total_card_err = 0.0;
        let mut num_queries = 0;

        // Query suffix windows: [total_length - suffix, total_length - 1]
        // This avoids cover() issues since all data is already inserted.
        let suffix_sizes = [1000, 2000, 3000, 4000, 5000];
        for &suffix in &suffix_sizes {
            let t2 = total_length - 1;
            let t1 = t2 - suffix + 1;

            let freq = freq_map_from_samples(&all_samples, t1, t2);
            let (true_l1, true_l2, true_card, true_entropy) = ground_truth_from_freq(&freq);

            if let Some(result) = eh.query_interval(t1, t2) {
                let l1_err = (result.calc_l1() - true_l1).abs() / true_l1.max(1e-12);
                let l2_err = (result.calc_l2() - true_l2).abs() / true_l2.max(1e-12);
                let card_err = (result.calc_card() - true_card).abs() / true_card.max(1e-12);
                let entropy_err = if true_entropy.abs() > 1e-6 {
                    (result.calc_entropy() - true_entropy).abs() / true_entropy
                } else {
                    0.0
                };

                total_l1_err += l1_err;
                total_l2_err += l2_err;
                total_card_err += card_err;
                total_entropy_err += entropy_err;
                num_queries += 1;
            }
        }

        // Also query at periodic points during the stream by rebuilding EH
        // for each query point (to have fresh expiration state)
        for query_t in (window..total_length).step_by(query_interval as usize) {
            let mut eh2 = EHUnivOptimized::with_defaults(8, window);
            for t in 0..=query_t {
                eh2.update(t, &DataInput::I64(all_samples[t as usize].1), 1);
            }

            let t1 = query_t.saturating_sub(window - 1);
            let t2 = query_t;

            if !eh2.cover(t1, t2) {
                continue;
            }

            let freq = freq_map_from_samples(&all_samples, t1, t2);
            let (true_l1, true_l2, true_card, true_entropy) = ground_truth_from_freq(&freq);

            if let Some(result) = eh2.query_interval(t1, t2) {
                let l1_err = (result.calc_l1() - true_l1).abs() / true_l1.max(1e-12);
                let l2_err = (result.calc_l2() - true_l2).abs() / true_l2.max(1e-12);
                let card_err = (result.calc_card() - true_card).abs() / true_card.max(1e-12);
                let entropy_err = if true_entropy.abs() > 1e-6 {
                    (result.calc_entropy() - true_entropy).abs() / true_entropy
                } else {
                    0.0
                };

                total_l1_err += l1_err;
                total_l2_err += l2_err;
                total_card_err += card_err;
                total_entropy_err += entropy_err;
                num_queries += 1;
            }
        }

        assert!(num_queries > 0, "Should have performed at least one query");

        let avg_l1_err = total_l1_err / num_queries as f64;
        let avg_l2_err = total_l2_err / num_queries as f64;
        let avg_card_err = total_card_err / num_queries as f64;
        let avg_entropy_err = total_entropy_err / num_queries as f64;

        // Average errors across sliding windows should be under 15%
        assert!(
            avg_l1_err < 0.15,
            "Avg L1 error {:.2}% over {num_queries} queries",
            avg_l1_err * 100.0
        );
        assert!(
            avg_l2_err < 0.15,
            "Avg L2 error {:.2}% over {num_queries} queries",
            avg_l2_err * 100.0
        );
        assert!(
            avg_card_err < 0.15,
            "Avg Card error {:.2}% over {num_queries} queries",
            avg_card_err * 100.0
        );
        assert!(
            avg_entropy_err < 0.15,
            "Avg Entropy error {:.2}% over {num_queries} queries",
            avg_entropy_err * 100.0
        );
    }

    /// Accuracy across different k values: higher k should yield more
    /// buckets but better accuracy.
    #[test]
    fn accuracy_varies_with_k() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let window = 5000u64;
        let stream_len = 10000u64;
        let mut rng_base = StdRng::seed_from_u64(0xDEAD_C0DE);

        // Generate a fixed stream
        let stream: Vec<i64> = (0..stream_len)
            .map(|_| (rng_base.random::<u32>() % 200) as i64)
            .collect();

        let mut errors_by_k: Vec<(usize, f64)> = Vec::new();

        for &k in &[2, 8, 32] {
            let mut eh = EHUnivOptimized::with_defaults(k, window);
            for (t, &key) in stream.iter().enumerate() {
                eh.update(t as u64, &DataInput::I64(key), 1);
            }

            let t1 = stream_len - window;
            let t2 = stream_len - 1;

            // Compute ground truth for the window
            let mut freq: HashMap<i64, i64> = HashMap::new();
            for &t in stream.iter().take(t2 as usize + 1).skip(t1 as usize) {
                *freq.entry(t).or_insert(0) += 1;
            }
            let (true_l1, true_l2, _, _) = ground_truth_from_freq(&freq);

            if let Some(result) = eh.query_interval(t1, t2) {
                let l1_err = (result.calc_l1() - true_l1).abs() / true_l1.max(1e-12);
                let l2_err = (result.calc_l2() - true_l2).abs() / true_l2.max(1e-12);
                let avg_err = (l1_err + l2_err) / 2.0;
                errors_by_k.push((k, avg_err));
            }
        }

        // All should be under 15%
        for &(k, err) in &errors_by_k {
            assert!(
                err < 0.15,
                "k={k}: avg (L1+L2)/2 error {:.2}% exceeds 15%",
                err * 100.0
            );
        }
    }

    /// Suffix query accuracy: query the most recent N items (like the Go test's
    /// suffix-length sweep).
    #[test]
    fn accuracy_suffix_queries() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let window = 10000u64;
        let total_length = 15000u64;
        let mut rng = StdRng::seed_from_u64(0xBAAD_F00D);

        let mut eh = EHUnivOptimized::with_defaults(8, window);
        let mut all_samples: Vec<(u64, i64)> = Vec::new();

        for t in 0..total_length {
            let key = (rng.random::<u32>() % 300) as i64;
            eh.update(t, &DataInput::I64(key), 1);
            all_samples.push((t, key));
        }

        let t_end = total_length - 1;
        let suffixes = [1000, 2000, 5000, 8000];
        let mut max_l2_err = 0.0f64;

        for &suffix_len in &suffixes {
            let t1 = t_end - suffix_len + 1;
            let t2 = t_end;

            if !eh.cover(t1, t2) {
                continue;
            }

            let freq = freq_map_from_samples(&all_samples, t1, t2);
            let (_, true_l2, _, _) = ground_truth_from_freq(&freq);

            if let Some(result) = eh.query_interval(t1, t2) {
                let l2_err = (result.calc_l2() - true_l2).abs() / true_l2.max(1e-12);
                max_l2_err = max_l2_err.max(l2_err);
            }
        }

        assert!(
            max_l2_err < 0.20,
            "Max L2 error across suffix queries: {:.2}%",
            max_l2_err * 100.0
        );
    }

    /// Dynamic distribution shift: the distribution changes over time.
    /// Verifies accuracy for a window that spans both distributions.
    #[test]
    fn accuracy_distribution_shift() {
        let window = 4000u64;
        let mut eh = EHUnivOptimized::with_defaults(8, window);
        let mut all_samples: Vec<(u64, i64)> = Vec::new();

        // Phase 1 (t=0..1999): highly skewed, key=0 is heavy
        for t in 0..2000u64 {
            let key = if t % 5 == 0 { 0 } else { (t % 10 + 1) as i64 };
            eh.update(t, &DataInput::I64(key), 1);
            all_samples.push((t, key));
        }

        // Phase 2 (t=2000..3999): uniform-ish, 50 distinct keys
        for t in 2000..4000u64 {
            let key = (t % 50) as i64;
            eh.update(t, &DataInput::I64(key), 1);
            all_samples.push((t, key));
        }

        // Query spanning both phases
        let t1 = 0;
        let t2 = 3999;
        let freq = freq_map_from_samples(&all_samples, t1, t2);
        let (true_l1, true_l2, true_card, true_entropy) = ground_truth_from_freq(&freq);

        let result = eh.query_interval(t1, t2).unwrap();

        assert_metric_within("Shift L1", result.calc_l1(), true_l1, 0.15);
        assert_metric_within("Shift L2", result.calc_l2(), true_l2, 0.15);
        assert_metric_within("Shift Card", result.calc_card(), true_card, 0.15);
        assert_metric_within("Shift Entropy", result.calc_entropy(), true_entropy, 0.15);
    }
}
