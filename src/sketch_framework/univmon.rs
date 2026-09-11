//! UnivMon (Universal Monitoring) Implementation
//!
//! This module provides an implementation of the UnivMon algorithm as described in:
//! "Universal Sketches for the Next Generation of Real-time Network Data Analytics"
//! by Liu et al. (ACM SIGCOMM 2016).
//! <https://dl.acm.org/doi/10.1145/2934872.2934906>
//!
//! UnivMon is a universal sketch framework that enables the estimation of multiple
//! network flow metrics—such as L1/L2 norms, Shannon entropy, and cardinality—using
//! a single, hierarchical sampling structure.
//!
//! # Architecture
//! The implementation consists of a "Sketch Pyramid" where:
//! * Each layer $i$ samples a subset of the stream from layer $i-1$ with probability 1/2.
//! * Each layer maintains a functional sketch (e.g., Count-Min Sketch) for frequency estimation.
//! * Each layer maintains a Heavy Hitter heap to track the most frequent elements at that sampling level.
//!
//! # Capabilities
//! * **L1/L2 Norm**: Estimation of total flow volume and second moments.
//! * **Entropy**: Calculation of flow distribution complexity.
//! * **Cardinality**: Estimation of the number of distinct elements.
//! * **Heavy Hitters**: Tracking top flows across different sampling granularities.
//!
//! This implementation is part of the `asap_sketchlib` library.

use crate::common::heap::HHHeap;
use crate::common::{
    BOTTOM_LAYER_FINDER, DataInput, HeapItem, hash_item64_seeded, hash64_seeded,
    heap_item_to_sketch_input,
};
use crate::common::{L2HH, Vector1D};
use crate::octo_delta::LayeredCountDelta;
use crate::sketches::countsketch_topk::CountL2HH;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub(crate) mod wire;

const DEFAULT_SKETCH_ROW: usize = 5;
const DEFAULT_SKETCH_COL: usize = 2048;
const DEFAULT_HEAP_SIZE: usize = 32;
const DEFAULT_LAYER_SIZE: usize = 8;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum UnivMonUpdateMode {
    #[default]
    Unset,
    Standard,
    Terminal,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
/// UnivMon sketch pyramid for multi-metric stream estimation.
pub struct UnivMon {
    /// Per-layer L2/heavy-hitter sketches.
    pub l2_sketch_layers: Vector1D<L2HH>,
    /// Per-layer heavy-hitter heaps.
    pub hh_layers: Vector1D<HHHeap>,
    /// Number of pyramid layers.
    pub layer_size: usize,
    /// Row count of each underlying sketch.
    pub sketch_row: usize,
    /// Column count of each underlying sketch.
    pub sketch_col: usize,
    /// Heap capacity per layer.
    pub heap_size: usize,
    /// Bucket size used for hashing decisions.
    pub bucket_size: usize,
    #[serde(default)]
    update_mode: UnivMonUpdateMode,
    #[serde(default)]
    candidate_complete: Vec<bool>,
}

impl Default for UnivMon {
    fn default() -> Self {
        UnivMon::init_univmon(
            DEFAULT_HEAP_SIZE,
            DEFAULT_SKETCH_ROW,
            DEFAULT_SKETCH_COL,
            DEFAULT_LAYER_SIZE,
        )
    }
}

/// Layers one pyramid may hold. [`bottom_layer_for_hash`] and the query
/// recurrences shift a 64-bit key hash right by up to `layer_size - 1`, so a
/// deeper pyramid would shift past the hash's width.
pub const MAX_LAYER_SIZE: usize = 64;

/// Deepest pyramid layer the given key hash reaches.
#[inline(always)]
pub fn bottom_layer_for_hash(hash: u64, layer_size: usize) -> usize {
    for l in 1..layer_size {
        if ((hash >> l) & 1) == 0 {
            return l - 1;
        }
    }
    layer_size - 1
}

/// How much of the stream reached an aggregator being fed `LayeredCountDelta`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnivMonDeltaFidelity {
    /// Every insert was promoted, so candidate sets can still be complete.
    /// Only true at a promotion threshold of 1.
    EveryInsert,
    /// Only counters that crossed the threshold arrived; candidate sets are
    /// partial by construction.
    PromotedOnly,
}

impl UnivMon {
    /// Whether this state accepts standard updates and standard-state merges.
    /// Empty states have not selected an update strategy yet.
    pub fn accepts_standard_updates(&self) -> bool {
        matches!(
            self.update_mode,
            UnivMonUpdateMode::Unset | UnivMonUpdateMode::Standard
        )
    }
    /// Creates a UnivMon instance with explicit dimensions.
    pub fn init_univmon(
        heap_size: usize,
        sketch_row: usize,
        sketch_col: usize,
        layer_size: usize,
    ) -> Self {
        assert!(heap_size > 0, "heap size must be positive");
        assert!(sketch_row > 0, "sketch row count must be positive");
        assert!(sketch_col > 0, "sketch column count must be positive");
        assert!(layer_size > 0, "layer count must be positive");
        assert!(
            layer_size <= MAX_LAYER_SIZE,
            "layer count must be at most MAX_LAYER_SIZE {MAX_LAYER_SIZE}"
        );
        let sk_vec: Vec<L2HH> = (0..layer_size)
            .map(|i| {
                L2HH::COUNT(CountL2HH::with_dimensions_and_seed(
                    sketch_row, sketch_col, i,
                ))
            })
            .collect();

        let hh_vec: Vec<HHHeap> = (0..layer_size).map(|_| HHHeap::new(heap_size)).collect();

        UnivMon {
            l2_sketch_layers: Vector1D::from_vec(sk_vec),
            hh_layers: Vector1D::from_vec(hh_vec),
            layer_size,
            sketch_row,
            sketch_col,
            heap_size,
            bucket_size: 0,
            update_mode: UnivMonUpdateMode::Unset,
            candidate_complete: vec![true; layer_size],
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

    #[inline(always)]
    fn find_bottom_layer_num(&self, hash: u64, layer: usize) -> usize {
        bottom_layer_for_hash(hash, layer)
    }

    /// Deepest pyramid layer an insert of `key` reaches; it touches `0..=this`.
    ///
    /// A pure function of the key's hash, so an OctoSketch worker picks exactly
    /// the same layers as a single-threaded insert would.
    pub fn bottom_layer_for(&self, key: &DataInput) -> usize {
        bottom_layer_for_hash(hash64_seeded(BOTTOM_LAYER_FINDER, key), self.layer_size)
    }

    /// Whether each layer's heavy-hitter heap still holds every key that layer
    /// received. Queries widen their threshold on the layers where it does not.
    pub fn candidates_complete(&self) -> &[bool] {
        &self.candidate_complete
    }

    /// Marks every layer's candidate set as partial.
    ///
    /// An OctoSketch aggregator calls this up front whenever its workers
    /// promote above a threshold of 1: a layer that received traffic but never
    /// promoted any of it sends the aggregator nothing at all, so waiting for a
    /// delta to arrive before lowering the flag would leave exactly those
    /// layers claiming a completeness they cannot have.
    pub fn mark_candidates_incomplete(&mut self) {
        self.candidate_complete.fill(false);
    }

    /// Marks one layer's candidate set as partial.
    pub fn mark_layer_candidates_incomplete(&mut self, layer: usize) {
        self.candidate_complete[layer] = false;
    }

    /// Overwrites the total weight the sketch believes it has seen.
    ///
    /// An OctoSketch aggregator never observes the raw stream, so it restores
    /// this from the running totals its workers report. This *assigns* where
    /// `insert` and `merge` accumulate, so a sketch fed by deltas must not also
    /// be inserted into or merged with: the next delta would erase whatever
    /// those added. Feeding one exclusively through `apply_layered_delta` and
    /// this method is the supported arrangement.
    pub fn set_total_weight(&mut self, weight: usize) {
        debug_assert_ne!(
            self.update_mode,
            UnivMonUpdateMode::Terminal,
            "a terminal-mode UnivMon cannot be fed by the delta path"
        );
        self.bucket_size = weight;
        if self.update_mode == UnivMonUpdateMode::Unset {
            self.update_mode = UnivMonUpdateMode::Standard;
        }
    }

    /// Applies one delta promoted by an OctoSketch worker.
    ///
    /// Mirrors `update` for a single layer: the counter lands, the layer's
    /// estimate is re-read, and the layer's heavy-hitter heap follows - which
    /// is Algorithm 2 of the OctoSketch paper.
    pub fn apply_layered_delta(
        &mut self,
        delta: &LayeredCountDelta,
        fidelity: UnivMonDeltaFidelity,
    ) {
        if self.update_mode == UnivMonUpdateMode::Unset {
            self.update_mode = UnivMonUpdateMode::Standard;
        }
        let layer = delta.layer as usize;
        assert!(
            layer < self.layer_size,
            "delta names layer {layer} but this pyramid has {} layers",
            self.layer_size
        );

        self.l2_sketch_layers[layer].apply_delta(delta.delta);
        let key = heap_item_to_sketch_input(&delta.key);
        let count = self.l2_sketch_layers[layer].estimate(&key);
        let heap_kept_everything = self.hh_layers[layer].update(&key, count as i64);

        // A worker holds back every counter that has not reached the promotion
        // threshold, so unless the threshold is 1 the aggregator has provably
        // not seen every key this layer received and its candidate set cannot
        // be called complete. Claiming otherwise would send `heavy_threshold`
        // down the permissive branch and overcount.
        let complete = heap_kept_everything && fidelity == UnivMonDeltaFidelity::EveryInsert;
        if !complete {
            self.candidate_complete[layer] = false;
        }
    }

    #[inline(always)]
    fn update(&mut self, key: &DataInput, value: i64, bottom_layer_num: usize) {
        for i in 0..=bottom_layer_num {
            let count = self.l2_sketch_layers[i].update_and_est(key, value);
            if !self.hh_layers[i].update(key, count as i64) {
                self.candidate_complete[i] = false;
            }
        }
    }

    #[inline(always)]
    fn process_univmon(&mut self, key: &DataInput, value: i64, bottom_layer_num: usize) {
        self.update(key, value, bottom_layer_num);
    }

    /// Inserts one weighted update.
    pub fn insert(&mut self, key: &DataInput, value: i64) {
        self.begin_update(value, UnivMonUpdateMode::Standard);
        let h = hash64_seeded(BOTTOM_LAYER_FINDER, key);
        let bottom_layer_num = self.find_bottom_layer_num(h, self.layer_size);
        self.process_univmon(key, value, bottom_layer_num)
    }

    /// Inserts one weighted update into its terminal stratum only.
    ///
    /// This is the Joltik update-last-layer construction. Logical sampled
    /// streams and their candidate sets are reconstructed during queries.
    /// Do not mix this method with [`Self::insert`] on the same sketch.
    pub fn fast_insert(&mut self, key: &DataInput, value: i64) {
        self.begin_update(value, UnivMonUpdateMode::Terminal);
        let h = hash64_seeded(BOTTOM_LAYER_FINDER, key);
        let bottom_layer_num = self.find_bottom_layer_num(h, self.layer_size);
        let count = self.l2_sketch_layers[bottom_layer_num].update_and_est(key, value);
        if !self.hh_layers[bottom_layer_num].update(key, count as i64) {
            self.candidate_complete[bottom_layer_num] = false;
        }
    }

    /// Prints all heavy-hitter layers for debugging.
    pub fn print_hh_layer(&self) {
        print!("Print HH_Layer: ");
        for i in 0..self.layer_size {
            println!("layer {i}: ");
            self.hh_layers[i].print_heap();
        }
    }

    /// Computes a g-sum estimate using the heuristic recurrence.
    pub fn calc_g_sum_heuristic<F>(&self, g: F, is_card: bool) -> f64
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
        let mut tmp: f64;

        let l2_value = self.l2_sketch_layers[self.layer_size - 1].get_l2();
        let threshold = if is_card {
            self.heavy_threshold(l2_value, self.candidate_complete[self.layer_size - 1])
        } else {
            0
        };

        tmp = 0.0;
        for item in self.hh_layers[self.layer_size - 1].heap() {
            let input = heap_item_to_sketch_input(&item.key);
            let count = self.l2_sketch_layers[self.layer_size - 1].estimate(&input) as i64;
            if count > threshold {
                tmp += g(count as f64);
            }
        }
        y[self.layer_size - 1] = tmp;

        for i in (0..(self.layer_size - 1)).rev() {
            tmp = 0.0;
            let l2_value = self.l2_sketch_layers[i].get_l2();
            let threshold = if is_card {
                self.heavy_threshold(l2_value, self.candidate_complete[i])
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

    /// Reconstructs the logical sampled-stream candidate sets from disjoint
    /// terminal strata, as required by the Joltik update-last-layer scheme.
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

    /// Computes a g-sum estimate.
    pub fn calc_g_sum<F>(&self, g: F, is_card: bool) -> f64
    where
        F: Fn(f64) -> f64,
    {
        self.calc_g_sum_heuristic(g, is_card)
    }

    /// Returns the exact L1 norm for the supported non-negative update stream.
    pub fn calc_l1(&self) -> f64 {
        self.bucket_size as f64
    }

    /// Returns the estimated L2 norm.
    pub fn calc_l2(&self) -> f64 {
        let tmp = self.calc_g_sum(|x| x * x, false);
        tmp.sqrt()
    }

    /// Returns the estimated entropy.
    pub fn calc_entropy(&self) -> f64 {
        if self.bucket_size == 0 {
            return 0.0;
        }
        let tmp = self.calc_g_sum(
            |x| {
                if x > 0.0 { x * x.log2() } else { 0.0 }
            },
            false,
        );
        (self.bucket_size as f64).log2() - tmp / (self.bucket_size as f64)
    }

    /// Returns the estimated cardinality.
    pub fn calc_card(&self) -> f64 {
        self.calc_g_sum(|_| 1.0, true)
    }

    /// Resets the sketch to its initial state without reallocating.
    /// Zeroes all counters and clears all heaps, matching the Go `Free()` method.
    pub fn free(&mut self) {
        self.bucket_size = 0;
        self.update_mode = UnivMonUpdateMode::Unset;
        self.candidate_complete.fill(true);
        for i in 0..self.layer_size {
            self.l2_sketch_layers[i].clear();
            self.hh_layers[i].clear();
        }
    }

    /// Merges another compatible UnivMon into this one.
    ///
    /// Both sketches must use the same dimensions and update strategy.
    /// Counters are merged first, then candidate frequencies and cached L2
    /// values are rebuilt from the combined counter state.
    pub fn merge(&mut self, other: &UnivMon) {
        assert_eq!(
            self.layer_size, other.layer_size,
            "layer size should be equal to merge"
        );
        assert_eq!(
            (self.sketch_row, self.sketch_col, self.heap_size),
            (other.sketch_row, other.sketch_col, other.heap_size),
            "UnivMon dimensions and heap size must match for merge"
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
    use crate::{DataInput, HeapItem};
    use core::f64;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::collections::HashMap;

    /// Persisted terminal states must be distinguishable before standard ingestion.
    #[test]
    fn standard_update_compatibility_survives_roundtrip_and_reset() {
        let mut sketch = UnivMon::init_univmon(4, 3, 16, 2);
        assert!(sketch.accepts_standard_updates());
        sketch.insert(&DataInput::U64(1), 1);
        assert!(sketch.accepts_standard_updates());
        let standard =
            UnivMon::deserialize_from_bytes(&sketch.serialize_to_bytes().unwrap()).unwrap();
        assert!(standard.accepts_standard_updates());
        sketch.free();
        sketch.fast_insert(&DataInput::U64(1), 1);
        let mut terminal =
            UnivMon::deserialize_from_bytes(&sketch.serialize_to_bytes().unwrap()).unwrap();
        assert!(!terminal.accepts_standard_updates());
        terminal.free();
        assert!(terminal.accepts_standard_updates());
    }

    #[test]
    fn univmon_round_trip_serialization() {
        let mut um = UnivMon::init_univmon(12, 3, 64, 4);
        let flows = [
            ("alpha", 5),
            ("beta", 7),
            ("gamma", 9),
            ("alpha", 3),
            ("delta", 11),
        ];

        for (key, count) in flows {
            um.insert(&DataInput::String(key.to_string()), count);
        }

        let bucket_size_before = um.bucket_size;
        let l1_before = um.calc_l1();
        let l2_before = um.calc_l2();
        let entropy_before = um.calc_entropy();
        let card_before = um.calc_card();

        let encoded = um
            .serialize_to_bytes()
            .expect("serialize UnivMon into MessagePack");
        assert!(!encoded.is_empty(), "serialized bytes should not be empty");
        let data = encoded.clone();

        let decoded =
            UnivMon::deserialize_from_bytes(&data).expect("deserialize UnivMon from MessagePack");

        assert_eq!(um.layer_size, decoded.layer_size);
        assert_eq!(um.sketch_row, decoded.sketch_row);
        assert_eq!(um.sketch_col, decoded.sketch_col);
        assert_eq!(um.heap_size, decoded.heap_size);
        assert_eq!(bucket_size_before, decoded.bucket_size);
        assert!(
            (decoded.calc_l1() - l1_before).abs() < 1e-6,
            "L1 changed after round trip"
        );
        assert!(
            (decoded.calc_l2() - l2_before).abs() < 1e-6,
            "L2 changed after round trip"
        );
        assert!(
            (decoded.calc_entropy() - entropy_before).abs() < 1e-6,
            "entropy changed after round trip"
        );
        assert!(
            (decoded.calc_card() - card_before).abs() < f64::EPSILON,
            "cardinality changed after round trip"
        );
    }

    #[test]
    fn update_populates_bucket_size_and_heavy_hitters() {
        // processing a single hot key should record its weight in the heavy hitter layers
        let mut um = UnivMon::init_univmon(16, 3, 32, 4);
        let key = "alpha";

        for _ in 0..40 {
            um.insert(&DataInput::Str(key), 1);
        }

        assert_eq!(um.bucket_size, 40);

        let idx = um.hh_layers[0]
            .find_heap_item(&HeapItem::String(key.to_owned()))
            .expect("heavy hitter should track key");
        assert!(
            um.hh_layers[0].heap()[idx].count >= 20,
            "expected significant count for heavy hitter, got {}",
            um.hh_layers[0].heap()[idx].count
        );
        assert!(
            um.calc_l1() == 40.0,
            "L1 Norm: get {}, expecting 1",
            um.calc_l1()
        );
        assert!(
            um.calc_card() == 1.0,
            "Cardinality: get {}, expecting 1",
            um.calc_card()
        );
    }

    #[test]
    fn merge_with_combines_heavy_hitters() {
        // merging two sketches should keep contributions from both sides
        let mut left = UnivMon::init_univmon(16, 3, 32, 4);
        let mut right = UnivMon::init_univmon(16, 3, 32, 4);

        let key_left = "left";
        let key_right = "right";

        for _ in 0..25 {
            left.insert(&DataInput::Str(key_left), 1);
        }
        for _ in 0..30 {
            right.insert(&DataInput::Str(key_right), 1);
        }

        left.merge(&right);

        let left_heap = left.heap_at_layer(00);
        let right_heap = right.heap_at_layer(0);
        let idx_left = left_heap
            .find_heap_item(&HeapItem::String(key_left.to_owned()))
            .expect("left key present");
        let idx_right_in_left = left_heap
            .find_heap_item(&HeapItem::String(key_right.to_owned()))
            .expect("left key present");
        let idx_right = right_heap
            .find_heap_item(&HeapItem::String(key_right.to_owned()))
            .expect("right key present");
        assert!(
            left_heap.heap()[idx_left].count == 25,
            "left in left is: {}",
            left_heap.heap()[idx_left].count
        );
        assert!(
            right_heap.heap()[idx_right].count == 30,
            "right in right is: {}",
            right_heap.heap()[idx_right].count
        );
        assert!(
            left_heap.heap()[idx_right_in_left].count == 30,
            "right in left is: {}",
            left_heap.heap()[idx_right_in_left].count
        );
    }

    #[test]
    fn merge_combines_weight_l2_and_evicted_candidate_counts() {
        let mut left = UnivMon::init_univmon(1, 3, 1024, 1);
        let mut right = UnivMon::init_univmon(1, 3, 1024, 1);
        left.insert(&DataInput::Str("x"), 100);
        right.insert(&DataInput::Str("x"), 5);
        right.insert(&DataInput::Str("y"), 10);

        left.merge(&right);

        assert_eq!(left.bucket_size, 115);
        let expected_l2 = (105.0_f64.powi(2) + 10.0_f64.powi(2)).sqrt();
        assert!((left.l2_sketch_layers[0].get_l2() - expected_l2).abs() < 1e-9);
        let x = HeapItem::String("x".to_owned());
        let index = left.hh_layers[0]
            .find_heap_item(&x)
            .expect("merged top candidate");
        assert_eq!(left.hh_layers[0].heap()[index].count, 105);
        assert!(left.calc_entropy().is_finite());
        assert!(left.calc_entropy() >= 0.0);
    }

    #[test]
    fn standard_and_terminal_merges_match_one_pass_with_complete_candidates() {
        for terminal_only in [false, true] {
            let mut one_pass = UnivMon::init_univmon(128, 5, 2048, 10);
            let mut left = UnivMon::init_univmon(128, 5, 2048, 10);
            let mut right = UnivMon::init_univmon(128, 5, 2048, 10);
            for observation in 0..2000_u64 {
                let key = DataInput::U64(observation % 64);
                if terminal_only {
                    one_pass.fast_insert(&key, 1);
                    if observation % 2 == 0 {
                        left.fast_insert(&key, 1);
                    } else {
                        right.fast_insert(&key, 1);
                    }
                } else {
                    one_pass.insert(&key, 1);
                    if observation % 2 == 0 {
                        left.insert(&key, 1);
                    } else {
                        right.insert(&key, 1);
                    }
                }
            }

            left.merge(&right);
            assert_eq!(left.bucket_size, one_pass.bucket_size);
            assert!((left.calc_l1() - one_pass.calc_l1()).abs() < 1e-9);
            assert!((left.calc_l2() - one_pass.calc_l2()).abs() < 1e-9);
            assert!((left.calc_card() - one_pass.calc_card()).abs() < 1e-9);
            assert!((left.calc_entropy() - one_pass.calc_entropy()).abs() < 1e-9);
        }
    }

    #[test]
    fn standard_updates_l2_for_every_sampled_layer() {
        let mut sketch = UnivMon::init_univmon(16, 3, 128, 6);
        let key = (0_u64..)
            .map(DataInput::U64)
            .find(|key| {
                let hash = hash64_seeded(BOTTOM_LAYER_FINDER, key);
                sketch.find_bottom_layer_num(hash, sketch.layer_size) >= 2
            })
            .expect("find a key sampled into multiple layers");
        let hash = hash64_seeded(BOTTOM_LAYER_FINDER, &key);
        let bottom = sketch.find_bottom_layer_num(hash, sketch.layer_size);
        sketch.insert(&key, 3);
        for level in 0..=bottom {
            assert_eq!(sketch.l2_sketch_layers[level].get_l2(), 3.0);
        }
    }

    #[test]
    fn terminal_update_touches_one_physical_layer_and_reconstructs_queries() {
        let mut sketch = UnivMon::init_univmon(16, 3, 128, 6);
        let key = (0_u64..)
            .map(DataInput::U64)
            .find(|key| {
                let hash = hash64_seeded(BOTTOM_LAYER_FINDER, key);
                sketch.find_bottom_layer_num(hash, sketch.layer_size) >= 2
            })
            .expect("find a key sampled into multiple layers");
        let hash = hash64_seeded(BOTTOM_LAYER_FINDER, &key);
        let bottom = sketch.find_bottom_layer_num(hash, sketch.layer_size);
        sketch.fast_insert(&key, 3);

        for level in 0..sketch.layer_size {
            let expected = if level == bottom { 3.0 } else { 0.0 };
            assert_eq!(sketch.l2_sketch_layers[level].get_l2(), expected);
        }
        assert_eq!(sketch.calc_l1(), 3.0);
        assert_eq!(sketch.calc_l2(), 3.0);
        assert_eq!(sketch.calc_card(), 1.0);
        assert_eq!(sketch.calc_entropy(), 0.0);
    }

    #[test]
    fn univmon_layers_use_different_seeds() {
        // Verify that different layers in UnivMon use different seeds
        // by checking they produce different hash values
        use crate::common::hash128_seeded;

        let _um = UnivMon::init_univmon(20, 3, 1024, 4);

        // Hash the same key with different seed indices (as used by different layers)
        let test_key = DataInput::Str("test_flow");

        // Hash the same key with different seed indices (as used by different layers)
        let hash_0 = hash128_seeded(0, &test_key);
        let hash_1 = hash128_seeded(1, &test_key);
        let hash_2 = hash128_seeded(2, &test_key);
        let hash_3 = hash128_seeded(3, &test_key);

        // All should be different
        assert_ne!(hash_0, hash_1, "Layers 0 and 1 should use different seeds");
        assert_ne!(hash_0, hash_2, "Layers 0 and 2 should use different seeds");
        assert_ne!(hash_0, hash_3, "Layers 0 and 3 should use different seeds");
        assert_ne!(hash_1, hash_2, "Layers 1 and 2 should use different seeds");
        assert_ne!(hash_1, hash_3, "Layers 1 and 3 should use different seeds");
        assert_ne!(hash_2, hash_3, "Layers 2 and 3 should use different seeds");
    }

    #[test]
    fn univmon_cardinality_is_positive() {
        // Basic sanity test: cardinality should be positive after insertions
        let mut um = UnivMon::init_univmon(20, 3, 2048, 8);

        for i in 0..20 {
            let key = format!("flow_{i}");
            um.insert(&DataInput::String(key), 1);
        }

        let card = um.calc_card();
        assert!(
            card == 20.0,
            "Cardinality should be positive after insertions, got {card}"
        );
    }

    #[test]
    fn univmon_bucket_size_tracked_correctly() {
        // Verify that bucket_size is correctly tracked with seed configuration
        let mut um = UnivMon::init_univmon(20, 3, 1024, 6);

        let flows = [("flow_a", 100), ("flow_b", 200), ("flow_c", 150)];
        let expected_total = 450;

        for (key, count) in &flows {
            um.insert(&DataInput::Str(key), *count);
        }

        assert_eq!(
            um.bucket_size, expected_total,
            "Bucket size should equal sum of all counts"
        );
    }

    #[test]
    fn univmon_basic_operation() {
        let cases: Vec<(String, i64)> = vec![
            ("notfound", 1),
            ("hello", 1),
            ("count", 3),
            ("min", 4),
            ("world", 10),
            ("cheatcheat", 3),
            ("cheatcheat", 7),
            ("min", 2),
            ("hello", 2),
            ("tigger", 34),
            ("flow", 9),
            ("miss", 4),
            ("hello", 30),
            ("world", 10),
            ("hello", 10),
            ("mom", 1),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        let mut um = UnivMon::init_univmon(100, 3, 2048, 16);
        for case in cases {
            um.insert(&DataInput::String(case.0), case.1);
        }

        assert_eq!(um.calc_card(), 10.0, "Cardinality estimation incorrect");
        assert_eq!(um.calc_l1(), 131.0, "L1 estimation incorrect");
    }

    #[test]
    fn test_statistical_accuracy() {
        // 1. Setup: Larger sketch for statistical significance
        // k=50 (top-k size), r=5 (rows), c=1024 (cols), l=10 (layers)
        let mut um = UnivMon::init_univmon(50, 5, 1024, 10);

        // 2. Generate Data: A simple skewed distribution
        // 1 heavy hitter (count 1000), 10 medium (count 100), 100 noise (count 1)
        let mut true_l2_sq = 0.0;
        let mut true_entropy_term = 0.0;
        let mut total_count = 0.0;

        let scenarios = vec![("heavy", 1000, 1), ("medium", 100, 10), ("noise", 1, 100)];

        for (prefix, count, repeat) in scenarios {
            for i in 0..repeat {
                let key = format!("{prefix}_{i}");
                let val = count as i64;
                let val_f = val as f64;

                // Ground Truth Calculation
                true_l2_sq += val_f * val_f;
                true_entropy_term += val_f * val_f.log2();
                total_count += val_f;

                // Update Sketch
                um.insert(&DataInput::String(key), val);
            }
        }

        // 3. Calculate True Metrics
        let true_l2 = true_l2_sq.sqrt();
        let true_entropy = total_count.log2() - (true_entropy_term / total_count);

        // 4. Get Estimates
        let est_l2 = um.calc_l2();
        let est_entropy = um.calc_entropy();

        // 5. Assertions (Allowing ~10% error for test-sized sketches)
        let l2_err = (est_l2 - true_l2).abs() / true_l2;
        let ent_err = (est_entropy - true_entropy).abs() / true_entropy;

        println!(
            "True L2: {:.2}, Est L2: {:.2}, Error: {:.2}%",
            true_l2,
            est_l2,
            l2_err * 100.0
        );
        println!(
            "True Ent: {:.2}, Est Ent: {:.2}, Error: {:.2}%",
            true_entropy,
            est_entropy,
            ent_err * 100.0
        );

        // UnivMon is generally very accurate for L2
        assert!(l2_err < 0.15, "L2 Error too high: {:.2}%", l2_err * 100.0);

        // Entropy is harder, usually requires higher k, allowing slightly looser bound
        assert!(
            ent_err < 0.15,
            "Entropy Error too high: {:.2}%",
            ent_err * 100.0
        );
    }

    #[test]
    fn univmon_random_data_matches_ground_truth_within_configured_tolerance() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut um = UnivMon::init_univmon(256, 6, 8192, 16);
        let mut truth: HashMap<String, i64> = HashMap::new();

        for _ in 0..10_000 {
            let key_id = rng.random::<u32>() % 5000;
            let key = format!("key_{key_id}");
            let value = (rng.random::<u32>() % 100 + 1) as i64;
            *truth.entry(key.clone()).or_insert(0) += value;
            um.insert(&DataInput::String(key), value);
        }

        let total_mass: f64 = truth.values().map(|&v| v as f64).sum();
        let true_l1 = total_mass;
        let true_l2 = truth
            .values()
            .map(|&v| {
                let val = v as f64;
                val * val
            })
            .sum::<f64>()
            .sqrt();
        let true_card = truth.len() as f64;
        let entropy_term = truth
            .values()
            .map(|&v| {
                let val = v as f64;
                if val > 0.0 { val * val.log2() } else { 0.0 }
            })
            .sum::<f64>();
        let true_entropy = total_mass.log2() - entropy_term / total_mass;

        let to_check = [
            ("cardinality", um.calc_card(), true_card, 0.07),
            ("l1", um.calc_l1(), true_l1, 0.05),
            ("l2", um.calc_l2(), true_l2, 0.05),
            ("entropy", um.calc_entropy(), true_entropy, 0.05),
        ];

        for (name, estimate, expected, tolerance) in to_check {
            let rel_err = (estimate - expected).abs() / expected;
            assert!(
                rel_err <= tolerance,
                "{name} relative error {:.2}% exceeds {:.2}%: est={estimate}, expected={expected}",
                rel_err * 100.0,
                tolerance * 100.0,
            );
        }
    }
}
