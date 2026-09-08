//! Experimental UnivMon-Q: universal frequency measurements with ordered quantile queries.
//!
//! Each numeric value is encoded into an order-preserving `u64` key. One
//! 128-bit key hash is split into independent fields for CountSketch rows and
//! the Joltik terminal stratum. Updates touch one physical CountSketch layer.
//! A separate coordinated bottom-k sample of stream occurrences is keyed by
//! `(source_id, local_sequence)`. Query-time recursion reconstructs the
//! logical UnivMon hierarchy; reliable heavy frequencies are combined with
//! the residual occurrence sample for rank and quantile estimates. The same
//! sample assists entropy when candidate recovery is incomplete.
//!
//! Provenance: adapted from the experimental `zaoxing/univmon-quantile`
//! implementation and integrated with Sketchlib's numeric input, pluggable
//! hashing, merge, and native MessagePack APIs.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use rmp_serde::decode::Error as RmpDecodeError;
use serde::{Deserialize, Serialize};

mod wire;

use crate::common::input::data_input_to_f64;
use crate::common::numerical::NumericalValue;
use crate::{DataInput, DefaultXxHasher, SketchHasher};

const OCCURRENCE_HASH_SEED_DOMAIN: usize = usize::MAX / 3;
static NEXT_SOURCE_ID: AtomicU64 = AtomicU64::new(1);

/// Memory, accuracy, and hashing controls for [`UnivMonQ`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnivMonQConfig {
    /// Number of geometrically sampled logical substreams.
    pub levels: usize,
    /// Counters in each row of the upper-level CountSketch.
    pub width: usize,
    /// Levels between width halvings. Zero keeps the same width at every level.
    pub width_halving_period: u8,
    /// CountSketch rows. This must be odd so the median is unambiguous.
    pub depth: usize,
    /// Counter representation: compact saturating 32-bit or wide 64-bit.
    pub counter_bits: u8,
    /// Maximum candidate identities retained at each terminal stratum.
    pub candidates: usize,
    /// Coordinated stream-occurrence samples retained for ordered queries.
    /// Set to zero when rank/CDF/quantile queries are not required.
    pub ordered_samples: usize,
    /// Seed index passed to [`SketchHasher::hash128_seeded`].
    pub hash_seed: usize,
}

impl Default for UnivMonQConfig {
    fn default() -> Self {
        Self {
            levels: 10,
            width: 4096,
            width_halving_period: 0,
            depth: 5,
            counter_bits: 32,
            candidates: 1024,
            ordered_samples: 1024,
            hash_seed: 5,
        }
    }
}

impl UnivMonQConfig {
    /// Chooses the smallest hierarchy whose deepest sample fits in the
    /// candidate table with probability at least `1 - failure_probability`.
    /// `max_updates` is the aggregate window bound across all merged shards.
    pub fn with_window_bound(
        mut self,
        max_updates: u64,
        failure_probability: f64,
    ) -> Result<Self, UnivMonQError> {
        if self.candidates == 0 {
            return Err(UnivMonQError::new("candidates must be positive"));
        }
        if !failure_probability.is_finite() || !(0.0..1.0).contains(&failure_probability) {
            return Err(UnivMonQError::new("failure probability must be in (0, 1)"));
        }
        let log_inverse_delta = (1.0 / failure_probability).ln();
        for levels in 2..=63 {
            let mean = max_updates as f64 / 2.0_f64.powi((levels - 1) as i32);
            let upper =
                mean + (2.0 * mean * log_inverse_delta).sqrt() + (2.0 / 3.0) * log_inverse_delta;
            if upper < self.candidates as f64 {
                self.levels = levels;
                return Ok(self);
            }
        }
        Err(UnivMonQError::new(
            "window bound does not fit the candidate table within 63 levels",
        ))
    }
}

/// Configuration, merge, or decoded-state error for [`UnivMonQ`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnivMonQError(String);

impl UnivMonQError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for UnivMonQError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for UnivMonQError {}

/// A recovered value and its normalized cumulative rank.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnivMonQPoint {
    /// Recovered numeric value.
    pub value: f64,
    /// Monotone CDF estimate in `[0, 1]`.
    pub rank: f64,
}

#[derive(Clone, Copy, Debug)]
struct RecoveredCandidate {
    key: u64,
    frequency: f64,
    terminal: usize,
}

#[derive(Debug)]
struct CandidateRecovery {
    physical_f2: Vec<f64>,
    by_terminal: Vec<Vec<RecoveredCandidate>>,
}

/// One retained stream occurrence. Splitting the 128-bit priority into two
/// words keeps the record at 24 bytes instead of introducing `u128` alignment
/// padding on common 64-bit targets.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
struct OrderedOccurrence {
    priority_high: u64,
    priority_low: u64,
    key: u64,
}

/// Reusable, immutable query state prepared from a [`UnivMonQ`] sketch.
///
/// Preparing the view reconstructs the logical UnivMon hierarchy and ordered
/// CDF once. Use it when answering more than one aggregate or ordered query
/// from the same sketch snapshot. Point-frequency queries continue to read the
/// underlying CountSketch directly.
#[derive(Debug)]
pub struct UnivMonQQuery<'a, H: SketchHasher = DefaultXxHasher> {
    sketch: &'a UnivMonQ<H>,
    logical_heavy: Vec<Vec<RecoveredCandidate>>,
    heavy_hitters: Vec<RecoveredCandidate>,
    occurrence_entropy: Option<f64>,
    candidate_recovery_complete: bool,
    cdf: Vec<UnivMonQPoint>,
    cdf_composition: CdfComposition,
}

/// How the ordered CDF was composed, captured while it was built.
#[derive(Clone, Debug, Default)]
struct CdfComposition {
    heavy: Vec<(u64, f64)>,
    heavy_mass: f64,
    residual_mass: f64,
    residual_samples: usize,
}

/// Read-only view of the two quantities the ordered-query error bound
/// `sup_x |F_hat(x) - F(x)| <= 2 E_H + P_hat_R * epsilon_R` is stated over.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrderedQueryDiagnostics {
    /// The values the CDF treated as heavy, with the frequency it credited to
    /// each. `E_H` is the error of these frequencies against the truth,
    /// normalized by the observation count.
    pub heavy: Vec<(f64, f64)>,
    /// Total mass attributed to the heavy set.
    pub heavy_mass: f64,
    /// Mass left to the residual occurrence sample: `N - heavy_mass`.
    pub residual_mass: f64,
    /// `m_R` — retained occurrence samples backing the residual, the `m` in
    /// `epsilon_R = sqrt(ln(2/delta) / (2 m))`.
    pub residual_samples: usize,
}

impl OrderedQueryDiagnostics {
    /// `P_hat_R` — the residual's share of the total mass.
    pub fn residual_mass_fraction(&self, count: u64) -> f64 {
        if count == 0 {
            0.0
        } else {
            (self.residual_mass / count as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct HashLayout {
    bucket_bits: u32,
    terminal_offset: u32,
    terminal_bits: u32,
}

impl HashLayout {
    fn new(config: UnivMonQConfig) -> Result<Self, UnivMonQError> {
        let bucket_bits = usize::BITS - (config.width - 1).leading_zeros();
        let row_bits = bucket_bits + 1;
        let terminal_offset = row_bits * config.depth as u32;
        let terminal_bits = config.levels as u32 - 1;
        let used_bits = terminal_offset + terminal_bits;
        if config.depth > 64 || used_bits > 128 {
            return Err(UnivMonQError::new(
                "hash layout needs more than the available 128 hash bits",
            ));
        }
        Ok(Self {
            bucket_bits,
            terminal_offset,
            terminal_bits,
        })
    }

    #[inline(always)]
    fn terminal_level(self, hash: u128) -> usize {
        let mask = (1_u128 << self.terminal_bits) - 1;
        let field = (hash >> self.terminal_offset) & mask;
        (field.trailing_zeros() as usize).min(self.terminal_bits as usize)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Counters {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PackedCountSketch {
    width: usize,
    depth: usize,
    counters: Counters,
}

impl PackedCountSketch {
    fn new(width: usize, depth: usize, counter_bits: u8) -> Self {
        Self {
            width,
            depth,
            counters: if counter_bits == 32 {
                Counters::I32(vec![0; width * depth])
            } else {
                Counters::I64(vec![0; width * depth])
            },
        }
    }

    #[inline(always)]
    fn update(&mut self, hash: u128, bucket_bits: u32) {
        match &mut self.counters {
            Counters::I32(counters) => {
                update_rows(counters, self.width, self.depth, hash, bucket_bits)
            }
            Counters::I64(counters) => {
                update_rows(counters, self.width, self.depth, hash, bucket_bits)
            }
        }
    }

    fn estimate(&self, hash: u128, bucket_bits: u32) -> i64 {
        let mut estimates = [0_i64; 64];
        match &self.counters {
            Counters::I32(counters) => estimate_rows(
                counters,
                self.width,
                self.depth,
                hash,
                bucket_bits,
                &mut estimates,
            ),
            Counters::I64(counters) => estimate_rows(
                counters,
                self.width,
                self.depth,
                hash,
                bucket_bits,
                &mut estimates,
            ),
        }
        estimates[..self.depth].sort_unstable();
        estimates[self.depth / 2].max(0)
    }

    fn estimated_f2(&self) -> f64 {
        let mut rows = Vec::with_capacity(self.depth);
        match &self.counters {
            Counters::I32(counters) => f2_rows(counters, self.width, self.depth, &mut rows),
            Counters::I64(counters) => f2_rows(counters, self.width, self.depth, &mut rows),
        }
        rows.sort_unstable_by(f64::total_cmp);
        rows[self.depth / 2]
    }

    fn merge(&mut self, other: &Self) {
        match (&mut self.counters, &other.counters) {
            (Counters::I32(left), Counters::I32(right)) => {
                for (left, right) in left.iter_mut().zip(right) {
                    *left = left.saturating_add(*right);
                }
            }
            (Counters::I64(left), Counters::I64(right)) => {
                for (left, right) in left.iter_mut().zip(right) {
                    *left = left.saturating_add(*right);
                }
            }
            _ => unreachable!("matching configurations use matching counter types"),
        }
    }

    fn clear(&mut self) {
        match &mut self.counters {
            Counters::I32(counters) => counters.fill(0),
            Counters::I64(counters) => counters.fill(0),
        }
    }

    fn bytes(&self) -> usize {
        match &self.counters {
            Counters::I32(counters) => counters.len() * size_of::<i32>(),
            Counters::I64(counters) => counters.len() * size_of::<i64>(),
        }
    }

    fn matches(&self, width: usize, depth: usize, counter_bits: u8) -> bool {
        if self.width != width || self.depth != depth {
            return false;
        }
        match &self.counters {
            Counters::I32(values) => counter_bits == 32 && values.len() == width * depth,
            Counters::I64(values) => counter_bits == 64 && values.len() == width * depth,
        }
    }
}

trait Counter: Copy {
    fn saturating_add(self, other: Self) -> Self;
    fn saturating_neg(self) -> Self;
    fn to_i64(self) -> i64;
    fn one() -> Self;
}

impl Counter for i32 {
    fn saturating_add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
    fn saturating_neg(self) -> Self {
        self.saturating_neg()
    }
    fn to_i64(self) -> i64 {
        i64::from(self)
    }
    fn one() -> Self {
        1
    }
}

impl Counter for i64 {
    fn saturating_add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
    fn saturating_neg(self) -> Self {
        self.saturating_neg()
    }
    fn to_i64(self) -> i64 {
        self
    }
    fn one() -> Self {
        1
    }
}

fn update_rows<T: Counter>(
    counters: &mut [T],
    width: usize,
    depth: usize,
    hash: u128,
    bucket_bits: u32,
) {
    let row_bits = bucket_bits + 1;
    let bucket_mask = (1_u128 << bucket_bits) - 1;
    let width_mask = width.is_power_of_two().then_some(width - 1);
    for row in 0..depth {
        let field = hash >> (row as u32 * row_bits);
        let bucket_code = (field & bucket_mask) as usize;
        let bucket = width_mask.map_or_else(|| bucket_code % width, |mask| bucket_code & mask);
        let delta = if (field >> bucket_bits) & 1 == 0 {
            T::one()
        } else {
            T::one().saturating_neg()
        };
        let index = row * width + bucket;
        counters[index] = counters[index].saturating_add(delta);
    }
}

fn estimate_rows<T: Counter>(
    counters: &[T],
    width: usize,
    depth: usize,
    hash: u128,
    bucket_bits: u32,
    estimates: &mut [i64; 64],
) {
    let row_bits = bucket_bits + 1;
    let bucket_mask = (1_u128 << bucket_bits) - 1;
    let width_mask = width.is_power_of_two().then_some(width - 1);
    for (row, estimate) in estimates.iter_mut().enumerate().take(depth) {
        let field = hash >> (row as u32 * row_bits);
        let bucket_code = (field & bucket_mask) as usize;
        let bucket = width_mask.map_or_else(|| bucket_code % width, |mask| bucket_code & mask);
        let counter = counters[row * width + bucket].to_i64();
        *estimate = if (field >> bucket_bits) & 1 == 0 {
            counter
        } else {
            counter.saturating_neg()
        };
    }
}

fn f2_rows<T: Counter>(counters: &[T], width: usize, depth: usize, rows: &mut Vec<f64>) {
    for row in 0..depth {
        let start = row * width;
        rows.push(
            counters[start..start + width]
                .iter()
                .map(|counter| (counter.to_i64() as f64).powi(2))
                .sum(),
        );
    }
}

#[derive(Clone, Debug)]
struct Level {
    sketch: PackedCountSketch,
    candidate_scores: HashMap<u64, u64>,
    candidate_heap: BinaryHeap<Reverse<(u64, u64)>>,
    candidate_capacity: usize,
    ever_evicted: bool,
}

impl Level {
    fn new(width: usize, depth: usize, counter_bits: u8, candidates: usize) -> Self {
        Self {
            sketch: PackedCountSketch::new(width, depth, counter_bits),
            candidate_scores: HashMap::with_capacity(candidates),
            candidate_heap: BinaryHeap::with_capacity(candidates),
            candidate_capacity: candidates,
            ever_evicted: false,
        }
    }

    fn update(&mut self, key: u64, hash: u128, bucket_bits: u32) {
        self.sketch.update(hash, bucket_bits);
        let score = self.sketch.estimate(hash, bucket_bits).max(0) as u64;
        if let Some(stored_score) = self.candidate_scores.get_mut(&key) {
            *stored_score = score;
            return;
        }
        if self.candidate_scores.len() < self.candidate_capacity {
            self.candidate_scores.insert(key, score);
            self.candidate_heap.push(Reverse((score, key)));
            return;
        }
        if let Some((minimum, evicted)) = self.lightest_candidate() {
            self.ever_evicted = true;
            if score > minimum {
                self.candidate_scores.remove(&evicted);
                self.candidate_scores.insert(key, score);
                self.candidate_heap.push(Reverse((score, key)));
            } else {
                self.candidate_heap.push(Reverse((minimum, evicted)));
            }
        }
    }

    fn lightest_candidate(&mut self) -> Option<(u64, u64)> {
        loop {
            let Reverse((stored_score, key)) = *self.candidate_heap.peek()?;
            let Some(&current_score) = self.candidate_scores.get(&key) else {
                self.candidate_heap.pop();
                continue;
            };
            if current_score != stored_score {
                self.candidate_heap.pop();
                self.candidate_heap.push(Reverse((current_score, key)));
                continue;
            }
            self.candidate_heap.pop();
            return Some((stored_score, key));
        }
    }

    fn merge(&mut self, other: &Self, bucket_bits: u32, hash_key: impl Fn(u64) -> u128) {
        self.sketch.merge(&other.sketch);
        let mut combined: HashSet<u64> = self.candidate_scores.keys().copied().collect();
        combined.extend(other.candidate_scores.keys().copied());
        let combined_len = combined.len();
        let mut retained: Vec<(u64, u64)> = combined
            .into_iter()
            .map(|key| {
                let hash = hash_key(key);
                let score = self.sketch.estimate(hash, bucket_bits).max(0) as u64;
                (key, score)
            })
            .collect();
        retained.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        retained.truncate(self.candidate_capacity);
        self.ever_evicted =
            self.ever_evicted || other.ever_evicted || combined_len > self.candidate_capacity;
        self.candidate_scores.clear();
        self.candidate_heap.clear();
        for (key, count) in retained {
            self.candidate_scores.insert(key, count);
            self.candidate_heap.push(Reverse((count, key)));
        }
    }

    fn clear(&mut self) {
        self.sketch.clear();
        self.candidate_scores.clear();
        self.candidate_heap.clear();
        self.ever_evicted = false;
    }
}

/// Experimental mergeable universal sketch extended with rank, CDF, and quantile queries.
///
/// `H` is the same pluggable hash abstraction used throughout Sketchlib. The
/// default uses [`DefaultXxHasher`]. Values use `f64::total_cmp` ordering,
/// including distinct `-0.0`/`0.0` and deterministic placement of NaNs.
#[derive(Clone, Debug)]
pub struct UnivMonQ<H: SketchHasher = DefaultXxHasher> {
    config: UnivMonQConfig,
    hash_layout: HashLayout,
    levels: Vec<Level>,
    count: u64,
    min: Option<u64>,
    max: Option<u64>,
    source_id: u64,
    next_sequence: u64,
    ordered_heap: BinaryHeap<OrderedOccurrence>,
    hasher: PhantomData<H>,
}

impl<'a, H: SketchHasher> UnivMonQQuery<'a, H> {
    /// Number of observations in the prepared sketch snapshot.
    pub fn count(&self) -> u64 {
        self.sketch.count
    }

    /// Exact minimum under `f64::total_cmp` ordering.
    pub fn min(&self) -> Option<f64> {
        self.sketch.min()
    }

    /// Exact maximum under `f64::total_cmp` ordering.
    pub fn max(&self) -> Option<f64> {
        self.sketch.max()
    }

    /// CountSketch point-frequency estimate for a numeric value.
    pub fn estimate_frequency(&self, value: f64) -> u64 {
        self.sketch.estimate_frequency(value)
    }

    /// Experimental fixed-threshold rank estimate from the UnivMon recurrence.
    ///
    /// Unlike [`Self::rank`], this does not use or require the ordered sample.
    /// It evaluates the separable subset function
    /// `g_x(key, frequency) = frequency * I[key <= x]` over the recovered
    /// hierarchy. Individual thresholds inherit UnivMon's candidate-recovery
    /// assumptions; estimates at different thresholds are not jointly forced
    /// to be monotone.
    pub fn estimate_rank_universal(&self, value: f64) -> Option<u64> {
        if self.count() == 0 {
            return None;
        }
        let key = float_to_ordered(value);
        if key < self.sketch.min? {
            return Some(0);
        }
        if key >= self.sketch.max? {
            return Some(self.count());
        }
        Some(
            estimate_keyed_sum_from(&self.logical_heavy, |candidate_key, frequency| {
                if candidate_key <= key { frequency } else { 0.0 }
            })
            .round()
            .clamp(0.0, self.count() as f64) as u64,
        )
    }

    /// Estimates `sum_x g(f_x)` using the prepared logical hierarchy.
    ///
    /// As with UnivMon's generic estimator, useful accuracy requires a
    /// frequency function compatible with the retained-heavy-item recurrence;
    /// callers should normally use the provided F0/F2/entropy methods.
    pub fn estimate_g_sum<F>(&self, g: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        estimate_g_sum_from(&self.logical_heavy, g)
    }

    /// UnivMon estimate of the number of distinct numeric values.
    pub fn estimate_distinct(&self) -> f64 {
        self.estimate_g_sum(|_| 1.0).clamp(0.0, self.count() as f64)
    }

    /// UnivMon estimate of the frequency vector's second moment.
    pub fn estimate_f2(&self) -> f64 {
        self.estimate_g_sum(|frequency| frequency * frequency)
            .max(0.0)
    }

    /// Exact L1 norm for the insertion-only frequency vector.
    pub fn estimate_l1(&self) -> f64 {
        self.count() as f64
    }

    /// Shannon entropy estimate from the original UnivMon recurrence, in nats.
    pub fn estimate_entropy_universal(&self) -> f64 {
        if self.count() == 0 {
            return 0.0;
        }
        let frequency_log_frequency = self.estimate_g_sum(|frequency| {
            if frequency > 0.0 {
                frequency * frequency.ln()
            } else {
                0.0
            }
        });
        ((self.count() as f64).ln() - frequency_log_frequency / self.count() as f64).max(0.0)
    }

    /// Assisted Shannon entropy estimate in nats.
    ///
    /// The universal recurrence is retained for diffuse streams and complete
    /// candidate recovery. Concentrated streams with incomplete recovery use
    /// the coordinated occurrence sample instead.
    pub fn estimate_entropy(&self) -> f64 {
        let universal = self.estimate_entropy_universal();
        if self.count() == 0
            || self.occurrence_entropy.is_none()
            || self.candidate_recovery_complete
        {
            return universal;
        }
        let concentration = self.estimate_f2() / (self.count() as f64).powi(2);
        if concentration < 1.0 / self.sketch.ordered_heap.len() as f64 {
            universal
        } else {
            self.estimate_entropy_occurrence().unwrap_or(universal)
        }
    }

    /// Experimental occurrence-sample entropy estimate in nats.
    ///
    /// This uses `H = E[ln(N / f_X)]` for an occurrence-uniform value `X`,
    /// with frequencies supplied by the terminal CountSketch. It requires the
    /// coordinated ordered sample and is primarily useful as an assisted
    /// alternative to the universal recurrence.
    pub fn estimate_entropy_occurrence(&self) -> Option<f64> {
        self.occurrence_entropy
    }

    /// Recovered heavy values and estimated frequencies in descending order.
    pub fn heavy_hitters(&self, k: usize) -> Vec<(f64, u64)> {
        if k == 0 || self.count() == 0 {
            return Vec::new();
        }
        self.heavy_hitters
            .iter()
            .take(k)
            .map(|candidate| {
                (
                    ordered_to_float(candidate.key),
                    candidate.frequency.round().max(0.0) as u64,
                )
            })
            .collect()
    }

    /// Estimated number of observations at or below `value`.
    pub fn rank(&self, value: f64) -> Option<u64> {
        if self.count() == 0 || self.sketch.config.ordered_samples == 0 {
            return None;
        }
        let min = self.min()?;
        let max = self.max()?;
        if value.total_cmp(&min).is_lt() {
            return Some(0);
        }
        if !value.total_cmp(&max).is_lt() {
            return Some(self.count());
        }
        let index = self
            .cdf
            .partition_point(|point| !point.value.total_cmp(&value).is_gt());
        Some(if index == 0 {
            0
        } else {
            (self.cdf[index - 1].rank * self.count() as f64)
                .round()
                .clamp(0.0, self.count() as f64) as u64
        })
    }

    /// Estimated value at normalized rank `q`.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.count() == 0 || !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return None;
        }
        if q == 0.0 {
            return self.min();
        }
        if q == 1.0 {
            return self.max();
        }
        if self.sketch.config.ordered_samples == 0 {
            return None;
        }
        let target = (q * self.count() as f64).ceil().max(1.0) / self.count() as f64;
        let index = self.cdf.partition_point(|point| point.rank < target);
        self.cdf
            .get(index.min(self.cdf.len().saturating_sub(1)))
            .map(|point| point.value)
    }

    /// Answers several quantiles while reusing the prepared CDF.
    pub fn quantiles(&self, quantiles: &[f64]) -> Vec<Option<f64>> {
        quantiles.iter().map(|&q| self.quantile(q)).collect()
    }

    /// Returns the prepared monotone CDF without rebuilding it.
    pub fn cdf(&self) -> &[UnivMonQPoint] {
        &self.cdf
    }

    /// Read-only diagnostics for the ordered-query error bound.
    ///
    /// Reports the heavy set the CDF actually used and the size of the residual
    /// occurrence sample, so that
    /// `sup_x |F_hat(x) - F(x)| <= 2 E_H + P_hat_R * epsilon_R` can be
    /// evaluated in full rather than only in the diffuse special case where the
    /// heavy set is empty. Purely observational: it returns state the CDF
    /// construction already produced.
    pub fn ordered_query_diagnostics(&self) -> OrderedQueryDiagnostics {
        OrderedQueryDiagnostics {
            heavy: self
                .cdf_composition
                .heavy
                .iter()
                .map(|(key, frequency)| (ordered_to_float(*key), *frequency))
                .collect(),
            heavy_mass: self.cdf_composition.heavy_mass,
            residual_mass: self.cdf_composition.residual_mass,
            residual_samples: self.cdf_composition.residual_samples,
        }
    }
}

impl<H: SketchHasher> Default for UnivMonQ<H> {
    fn default() -> Self {
        Self::with_hasher(UnivMonQConfig::default()).expect("default UnivMon-Q config is valid")
    }
}

impl<H: SketchHasher> UnivMonQ<H> {
    /// Creates an empty sketch using `H` and the supplied configuration.
    pub fn with_hasher(config: UnivMonQConfig) -> Result<Self, UnivMonQError> {
        Self::with_hasher_and_source_id(config, allocate_source_id())
    }

    /// Creates an empty sketch with an explicit distributed source identity.
    ///
    /// Every concurrently mergeable source must use a different ID. Ordered
    /// occurrence priorities are derived from `(source_id, local_sequence)`.
    pub fn with_hasher_and_source_id(
        config: UnivMonQConfig,
        source_id: u64,
    ) -> Result<Self, UnivMonQError> {
        validate_config(config)?;
        let hash_layout = HashLayout::new(config)?;
        let levels = (0..config.levels)
            .map(|level| {
                Level::new(
                    level_width(config, level),
                    config.depth,
                    config.counter_bits,
                    config.candidates,
                )
            })
            .collect();
        Ok(Self {
            config,
            hash_layout,
            levels,
            count: 0,
            min: None,
            max: None,
            source_id,
            next_sequence: 0,
            ordered_heap: BinaryHeap::with_capacity(config.ordered_samples),
            hasher: PhantomData,
        })
    }

    /// Returns the immutable configuration required by compatible merges.
    pub fn config(&self) -> UnivMonQConfig {
        self.config
    }

    /// Identity used to coordinate occurrence priorities from this source.
    pub fn source_id(&self) -> u64 {
        self.source_id
    }

    /// Adds one numeric value.
    #[inline(always)]
    pub fn update(&mut self, value: &f64) {
        self.update_value(*value);
    }

    /// Adds any Sketchlib numeric primitive after projection to `f64`.
    #[inline(always)]
    pub fn add<T: NumericalValue>(&mut self, value: &T) {
        self.update_value(value.to_f64());
    }

    /// Adds a type-erased numeric input. Strings and byte arrays are rejected.
    pub fn update_data_input(&mut self, value: &DataInput) -> Result<(), &'static str> {
        let value =
            data_input_to_f64(value).map_err(|_| "UnivMon-Q sketch only accepts numeric inputs")?;
        self.update_value(value);
        Ok(())
    }

    #[inline(always)]
    fn update_value(&mut self, value: f64) {
        let key = float_to_ordered(value);
        let hash = self.hash_key(key);
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .expect("UnivMon-Q source sequence exhausted");
        self.count = self.count.saturating_add(1);
        self.min = Some(self.min.map_or(key, |old| old.min(key)));
        self.max = Some(self.max.map_or(key, |old| old.max(key)));
        let terminal = self.hash_layout.terminal_level(hash);
        self.levels[terminal].update(key, hash, self.hash_layout.bucket_bits);
        if self.config.ordered_samples > 0 {
            let occurrence = self.occurrence(key, sequence);
            self.update_ordered_sample(occurrence);
        }
    }

    /// Merges a shard built with the identical configuration and hash profile.
    pub fn merge(&mut self, other: &Self) -> Result<(), UnivMonQError> {
        if self.config != other.config {
            return Err(UnivMonQError::new(
                "cannot merge UnivMon-Q sketches with different configurations",
            ));
        }
        self.count = self.count.saturating_add(other.count);
        self.min = match (self.min, other.min) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        self.max = match (self.max, other.max) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
        let hash_seed = self.config.hash_seed;
        let bucket_bits = self.hash_layout.bucket_bits;
        for (left, right) in self.levels.iter_mut().zip(&other.levels) {
            left.merge(right, bucket_bits, |key| {
                H::hash128_seeded(hash_seed, &DataInput::U64(key))
            });
        }
        self.merge_ordered_samples(other);
        Ok(())
    }

    /// Removes all observations without changing configuration or allocations.
    pub fn clear(&mut self) {
        self.count = 0;
        self.min = None;
        self.max = None;
        self.ordered_heap.clear();
        for level in &mut self.levels {
            level.clear();
        }
    }

    /// Number of observations processed, including duplicates.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Alias for [`Self::count`].
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Whether the sketch has no observations.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Exact minimum under `f64::total_cmp` ordering.
    pub fn min(&self) -> Option<f64> {
        self.min.map(ordered_to_float)
    }

    /// Exact maximum under `f64::total_cmp` ordering.
    pub fn max(&self) -> Option<f64> {
        self.max.map(ordered_to_float)
    }

    /// CountSketch point-frequency estimate for a numeric value.
    pub fn estimate_frequency(&self, value: f64) -> u64 {
        if self.is_empty() {
            return 0;
        }
        let key = float_to_ordered(value);
        self.frequency_key(key).max(0) as u64
    }

    /// Experimental fixed-threshold rank estimate from the UnivMon recurrence.
    ///
    /// This path is independent of `ordered_samples`. Prefer
    /// [`Self::prepare_queries`] when evaluating more than one threshold.
    pub fn estimate_rank_universal(&self, value: f64) -> Option<u64> {
        if self.is_empty() {
            return None;
        }
        self.prepare_queries().estimate_rank_universal(value)
    }

    /// Reconstructs a reusable query snapshot.
    ///
    /// Use this when answering multiple universal metrics or ordered queries
    /// without intervening updates. Candidate recovery, CountSketch F2 scans,
    /// and CDF construction are performed once for the whole batch.
    pub fn prepare_queries(&self) -> UnivMonQQuery<'_, H> {
        let recovery = self.recover_candidates();
        let logical_heavy = self.logical_heavy_sets_from(&recovery);
        let heavy_hitters = self.top_candidates_from(&recovery);
        let candidate_recovery_complete = self.candidate_recovery_complete();
        let ordered_heavy = self.assisted_ordered_heavy(&logical_heavy, &heavy_hitters);
        let entropy_heavy = Self::assisted_entropy_heavy(&logical_heavy);
        let occurrence_entropy = self.entropy_from_occurrences(&entropy_heavy);
        let (points, cdf_composition) = self.cdf_keys_and_composition_from(&ordered_heavy);
        let cdf = points
            .into_iter()
            .map(|(value, rank)| UnivMonQPoint {
                value: ordered_to_float(value),
                rank,
            })
            .collect();
        UnivMonQQuery {
            sketch: self,
            logical_heavy,
            heavy_hitters,
            occurrence_entropy,
            candidate_recovery_complete,
            cdf,
            cdf_composition,
        }
    }

    /// Estimates `sum_x g(f_x)` using the UnivMon recurrence.
    ///
    /// Call [`Self::prepare_queries`] and use [`UnivMonQQuery::estimate_g_sum`]
    /// when evaluating more than one function over the same sketch snapshot.
    pub fn estimate_g_sum<F>(&self, g: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        if self.is_empty() {
            return 0.0;
        }
        let recovery = self.recover_candidates();
        let heavy = self.logical_heavy_sets_from(&recovery);
        estimate_g_sum_from(&heavy, g)
    }

    /// UnivMon estimate of the number of distinct numeric values.
    pub fn estimate_distinct(&self) -> f64 {
        self.estimate_g_sum(|_| 1.0).clamp(0.0, self.count as f64)
    }

    /// UnivMon estimate of the frequency vector's second moment.
    pub fn estimate_f2(&self) -> f64 {
        self.estimate_g_sum(|frequency| frequency * frequency)
            .max(0.0)
    }

    /// Exact L1 norm for the insertion-only frequency vector.
    pub fn estimate_l1(&self) -> f64 {
        self.count as f64
    }

    /// Shannon entropy estimate from the original UnivMon recurrence, in nats.
    pub fn estimate_entropy_universal(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let frequency_log_frequency = self.estimate_g_sum(|frequency| {
            if frequency > 0.0 {
                frequency * frequency.ln()
            } else {
                0.0
            }
        });
        ((self.count as f64).ln() - frequency_log_frequency / self.count as f64).max(0.0)
    }

    /// Assisted Shannon entropy estimate in nats.
    pub fn estimate_entropy(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let recovery = self.recover_candidates();
        let logical_heavy = self.logical_heavy_sets_from(&recovery);
        let frequency_log_frequency = estimate_g_sum_from(&logical_heavy, |frequency| {
            if frequency > 0.0 {
                frequency * frequency.ln()
            } else {
                0.0
            }
        });
        let universal =
            ((self.count as f64).ln() - frequency_log_frequency / self.count as f64).max(0.0);
        if self.ordered_heap.is_empty() || self.candidate_recovery_complete() {
            return universal;
        }
        let f2 = estimate_g_sum_from(&logical_heavy, |frequency| frequency * frequency).max(0.0);
        let concentration = f2 / (self.count as f64).powi(2);
        if concentration < 1.0 / self.ordered_heap.len() as f64 {
            universal
        } else {
            let assisted_heavy = Self::assisted_entropy_heavy(&logical_heavy);
            self.entropy_from_occurrences(&assisted_heavy)
                .unwrap_or(universal)
        }
    }

    /// Experimental occurrence-sample entropy estimate in nats.
    pub fn estimate_entropy_occurrence(&self) -> Option<f64> {
        if self.is_empty() {
            return Some(0.0);
        }
        let recovery = self.recover_candidates();
        let logical_heavy = self.logical_heavy_sets_from(&recovery);
        let assisted_heavy = Self::assisted_entropy_heavy(&logical_heavy);
        self.entropy_from_occurrences(&assisted_heavy)
    }

    fn assisted_entropy_heavy(logical_heavy: &[Vec<RecoveredCandidate>]) -> Vec<(u64, f64)> {
        logical_heavy
            .first()
            .into_iter()
            .flatten()
            .take(64)
            .map(|candidate| (candidate.key, candidate.frequency))
            .collect()
    }

    fn entropy_from_occurrences(&self, assisted_heavy: &[(u64, f64)]) -> Option<f64> {
        if self.is_empty() {
            return Some(0.0);
        }
        if self.ordered_heap.is_empty() {
            return None;
        }
        let total = self.count as f64;
        let mut heavy: BTreeMap<u64, f64> = assisted_heavy.iter().copied().collect();
        let heavy_total = heavy.values().sum::<f64>();
        if heavy_total > total {
            let scale = total / heavy_total;
            for frequency in heavy.values_mut() {
                *frequency *= scale;
            }
        }
        let heavy_entropy = heavy
            .values()
            .map(|frequency| {
                let probability = frequency / total;
                if probability > 0.0 {
                    -probability * probability.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        let residual_mass = (1.0 - heavy.values().sum::<f64>() / total).max(0.0);
        let mut sampled_keys = BTreeMap::<u64, usize>::new();
        for occurrence in &self.ordered_heap {
            if !heavy.contains_key(&occurrence.key) {
                *sampled_keys.entry(occurrence.key).or_default() += 1;
            }
        }
        let residual_samples = sampled_keys.values().sum::<usize>();
        if residual_samples == 0 {
            return (residual_mass == 0.0).then_some(heavy_entropy);
        }
        let residual_entropy = sampled_keys
            .into_iter()
            .map(|(key, multiplicity)| {
                let frequency = (self.frequency_key(key).max(1) as f64).min(total);
                multiplicity as f64 * (total / frequency).ln()
            })
            .sum::<f64>();
        Some((heavy_entropy + residual_mass * residual_entropy / residual_samples as f64).max(0.0))
    }

    /// Recovered heavy values and estimated frequencies in descending order.
    pub fn heavy_hitters(&self, k: usize) -> Vec<(f64, u64)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        let recovery = self.recover_candidates();
        self.top_candidates_from(&recovery)
            .into_iter()
            .take(k)
            .map(|candidate| {
                (
                    ordered_to_float(candidate.key),
                    candidate.frequency.round().max(0.0) as u64,
                )
            })
            .collect()
    }

    /// Estimated number of observations at or below `value`.
    /// Returns `None` when ordered sampling is disabled or the sketch is empty.
    pub fn rank(&self, value: f64) -> Option<u64> {
        if self.is_empty() || self.config.ordered_samples == 0 {
            return None;
        }
        let key = float_to_ordered(value);
        if key < self.min? {
            return Some(0);
        }
        if key >= self.max? {
            return Some(self.count);
        }
        let cdf = self.cdf_keys();
        let index = cdf.partition_point(|point| point.0 <= key);
        Some(if index == 0 {
            0
        } else {
            (cdf[index - 1].1 * self.count as f64)
                .round()
                .clamp(0.0, self.count as f64) as u64
        })
    }

    /// Estimated value at normalized rank `q`.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.is_empty() || !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return None;
        }
        if q == 0.0 {
            return self.min();
        }
        if q == 1.0 {
            return self.max();
        }
        if self.config.ordered_samples == 0 {
            return None;
        }
        let cdf = self.cdf_keys();
        self.quantile_from_cdf(q, &cdf)
    }

    /// Alias matching the DDSketch quantile naming convention.
    pub fn get_value_at_quantile(&self, q: f64) -> Option<f64> {
        self.quantile(q)
    }

    /// Answers several quantiles with one CDF reconstruction.
    pub fn quantiles(&self, quantiles: &[f64]) -> Vec<Option<f64>> {
        if self.is_empty() {
            return vec![None; quantiles.len()];
        }
        let cdf = self.cdf_keys();
        quantiles
            .iter()
            .map(|&q| self.quantile_from_cdf(q, &cdf))
            .collect()
    }

    /// Recovered monotone CDF breakpoints.
    pub fn cdf(&self) -> Vec<UnivMonQPoint> {
        self.cdf_keys()
            .into_iter()
            .map(|(value, rank)| UnivMonQPoint {
                value: ordered_to_float(value),
                rank,
            })
            .collect()
    }

    /// Conservative storage estimate for counters and bounded metadata.
    pub fn estimated_memory_bytes(&self) -> usize {
        let counters: usize = self.levels.iter().map(|level| level.sketch.bytes()).sum();
        let candidates = self.config.levels
            * self.config.candidates
            * (size_of::<(u64, u64)>() + 2 * size_of::<usize>());
        let candidate_heaps = self.config.levels * self.config.candidates * size_of::<(u64, u64)>();
        let eviction_history = self.config.levels * size_of::<bool>();
        let ordered = self.config.ordered_samples * size_of::<OrderedOccurrence>();
        counters + candidates + candidate_heaps + eviction_history + ordered
    }

    fn cdf_keys(&self) -> Vec<(u64, f64)> {
        if self.is_empty() || self.config.ordered_samples == 0 {
            return Vec::new();
        }
        let recovery = self.recover_candidates();
        let logical_heavy = self.logical_heavy_sets_from(&recovery);
        let heavy_hitters = self.top_candidates_from(&recovery);
        let heavy = self.assisted_ordered_heavy(&logical_heavy, &heavy_hitters);
        self.cdf_keys_from(&heavy)
    }

    fn cdf_keys_from(&self, global_heavy: &[(u64, f64)]) -> Vec<(u64, f64)> {
        self.cdf_keys_and_composition_from(global_heavy).0
    }

    /// The CDF breakpoints, plus the composition that produced them.
    ///
    /// The composition is what [`UnivMonQQuery::ordered_query_diagnostics`]
    /// reports: which values the CDF treated as heavy and at what estimated
    /// mass, and how many retained occurrence samples formed the residual. Both
    /// are decided inside this function — the fallback below can discard the
    /// heavy set entirely — so they are captured here rather than recomputed by
    /// a caller that would have to duplicate the same branch.
    fn cdf_keys_and_composition_from(
        &self,
        global_heavy: &[(u64, f64)],
    ) -> (Vec<(u64, f64)>, CdfComposition) {
        if self.is_empty() || self.config.ordered_samples == 0 {
            return (Vec::new(), CdfComposition::default());
        }
        let mut heavy: BTreeMap<u64, f64> = global_heavy.iter().copied().collect();
        let raw_heavy_total: f64 = heavy.values().sum();
        if raw_heavy_total > self.count as f64 {
            let scale = self.count as f64 / raw_heavy_total;
            for frequency in heavy.values_mut() {
                *frequency *= scale;
            }
        }
        let heavy_total: f64 = heavy.values().sum();
        let residual_total = (self.count as f64 - heavy_total).max(0.0);
        let mut residual_keys: Vec<u64> = self
            .ordered_heap
            .iter()
            .filter(|occurrence| !heavy.contains_key(&occurrence.key))
            .map(|occurrence| occurrence.key)
            .collect();
        if residual_keys.is_empty() && residual_total > 0.0 {
            heavy.clear();
            residual_keys = self
                .ordered_heap
                .iter()
                .map(|occurrence| occurrence.key)
                .collect();
        }
        let residual_total = if heavy.is_empty() {
            self.count as f64
        } else {
            (self.count as f64 - heavy.values().sum::<f64>()).max(0.0)
        };
        let residual_weight = if residual_keys.is_empty() {
            0.0
        } else {
            residual_total / residual_keys.len() as f64
        };
        let composition = CdfComposition {
            heavy: heavy.iter().map(|(k, f)| (*k, *f)).collect(),
            heavy_mass: heavy.values().sum(),
            residual_mass: residual_total,
            residual_samples: residual_keys.len(),
        };
        let mut weights = heavy;
        for key in residual_keys {
            *weights.entry(key).or_insert(0.0) += residual_weight;
        }
        weights.entry(self.min.unwrap()).or_insert(0.0);
        weights.entry(self.max.unwrap()).or_insert(0.0);
        let mut running = 0.0;
        let mut points: Vec<(u64, f64)> = weights
            .into_iter()
            .map(|(value, weight)| {
                running += weight;
                (value, (running / self.count as f64).clamp(0.0, 1.0))
            })
            .collect();
        if let Some(last) = points.last_mut() {
            last.1 = 1.0;
        }
        isotonicize(&mut points);
        (points, composition)
    }

    fn recover_candidates(&self) -> CandidateRecovery {
        let physical_f2 = self
            .levels
            .iter()
            .map(|level| level.sketch.estimated_f2())
            .collect();
        let by_terminal = self
            .levels
            .iter()
            .enumerate()
            .map(|(terminal, level)| {
                level
                    .candidate_scores
                    .keys()
                    .filter_map(|&key| {
                        let hash = self.hash_key(key);
                        let frequency =
                            level.sketch.estimate(hash, self.hash_layout.bucket_bits) as f64;
                        (frequency > 0.0).then_some(RecoveredCandidate {
                            key,
                            frequency,
                            terminal,
                        })
                    })
                    .collect()
            })
            .collect();
        CandidateRecovery {
            physical_f2,
            by_terminal,
        }
    }

    fn top_candidates_from(&self, recovery: &CandidateRecovery) -> Vec<RecoveredCandidate> {
        let mut candidates: Vec<_> = recovery.by_terminal.iter().flatten().copied().collect();
        candidates.sort_unstable_by(|left, right| {
            right
                .frequency
                .total_cmp(&left.frequency)
                .then_with(|| left.key.cmp(&right.key))
        });
        candidates.truncate(self.config.candidates);
        candidates
    }

    fn assisted_ordered_heavy(
        &self,
        logical_heavy: &[Vec<RecoveredCandidate>],
        heavy_hitters: &[RecoveredCandidate],
    ) -> Vec<(u64, f64)> {
        if self.count == 0 || self.config.ordered_samples == 0 {
            return Vec::new();
        }
        let f2 = estimate_g_sum_from(logical_heavy, |frequency| frequency * frequency).max(0.0);
        let concentration = f2 / (self.count as f64).powi(2);
        if concentration < 1.0 / self.config.ordered_samples as f64 {
            return Vec::new();
        }
        let threshold = (f2 / self.config.width as f64).sqrt();
        heavy_hitters
            .iter()
            .take(64)
            .filter(|candidate| candidate.frequency >= threshold)
            .map(|candidate| (candidate.key, candidate.frequency))
            .collect()
    }

    fn logical_heavy_sets_from(
        &self,
        recovery: &CandidateRecovery,
    ) -> Vec<Vec<RecoveredCandidate>> {
        let mut logical = vec![Vec::new(); self.levels.len()];
        let mut suffix = Vec::with_capacity(self.config.candidates * 2);
        let mut suffix_f2 = 0.0;
        let mut suffix_candidates = 0_usize;
        let mut suffix_ever_evicted = false;
        for terminal in (0..self.levels.len()).rev() {
            suffix_f2 += recovery.physical_f2[terminal];
            suffix_candidates += recovery.by_terminal[terminal].len();
            suffix_ever_evicted |= self.levels[terminal].ever_evicted;
            suffix.extend(recovery.by_terminal[terminal].iter().copied());
            suffix.sort_unstable_by(|left, right| {
                right
                    .frequency
                    .total_cmp(&left.frequency)
                    .then_with(|| left.key.cmp(&right.key))
            });
            suffix.truncate(self.config.candidates);
            let mut recovered = suffix.clone();
            let complete = !suffix_ever_evicted && suffix_candidates <= self.config.candidates;
            if !complete {
                let threshold = 2.0 * (suffix_f2 / self.config.candidates as f64).sqrt();
                recovered.retain(|candidate| candidate.frequency >= threshold);
            }
            logical[terminal] = recovered;
        }
        logical
    }

    fn quantile_from_cdf(&self, q: f64, cdf: &[(u64, f64)]) -> Option<f64> {
        if self.is_empty() || !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return None;
        }
        if q == 0.0 {
            return self.min();
        }
        if q == 1.0 {
            return self.max();
        }
        if self.config.ordered_samples == 0 {
            return None;
        }
        let target = (q * self.count as f64).ceil().max(1.0) / self.count as f64;
        let index = cdf.partition_point(|point| point.1 < target);
        cdf.get(index.min(cdf.len().saturating_sub(1)))
            .map(|point| ordered_to_float(point.0))
    }

    fn update_ordered_sample(&mut self, occurrence: OrderedOccurrence) {
        let capacity = self.config.ordered_samples;
        if capacity == 0 {
            return;
        }
        if self.ordered_heap.len() < capacity {
            self.ordered_heap.push(occurrence);
            return;
        }
        if self
            .ordered_heap
            .peek()
            .is_some_and(|largest| occurrence < *largest)
        {
            self.ordered_heap.pop();
            self.ordered_heap.push(occurrence);
        }
    }

    fn merge_ordered_samples(&mut self, other: &Self) {
        let capacity = self.config.ordered_samples;
        if capacity == 0 {
            return;
        }
        for &occurrence in &other.ordered_heap {
            self.update_ordered_sample(occurrence);
        }
    }

    #[inline(always)]
    fn occurrence(&self, key: u64, sequence: u64) -> OrderedOccurrence {
        let identity = (u128::from(self.source_id) << 64) | u128::from(sequence);
        let priority = H::hash128_seeded(
            self.config.hash_seed ^ OCCURRENCE_HASH_SEED_DOMAIN,
            &DataInput::U128(identity),
        );
        OrderedOccurrence {
            priority_high: (priority >> 64) as u64,
            priority_low: priority as u64,
            key,
        }
    }

    #[inline(always)]
    fn hash_key(&self, key: u64) -> u128 {
        H::hash128_seeded(self.config.hash_seed, &DataInput::U64(key))
    }

    #[inline(always)]
    fn sample_level(&self, key: u64) -> usize {
        self.hash_layout.terminal_level(self.hash_key(key))
    }

    fn frequency_key(&self, key: u64) -> i64 {
        let hash = self.hash_key(key);
        let terminal = self.hash_layout.terminal_level(hash);
        self.levels[terminal]
            .sketch
            .estimate(hash, self.hash_layout.bucket_bits)
    }

    fn candidate_recovery_complete(&self) -> bool {
        self.levels.iter().all(|level| !level.ever_evicted)
            && self
                .levels
                .iter()
                .map(|level| level.candidate_scores.len())
                .sum::<usize>()
                <= self.config.candidates
    }
}

impl UnivMonQ<DefaultXxHasher> {
    /// Creates an empty sketch using Sketchlib's default XXH3 hasher.
    ///
    /// The generated occurrence-sampling source ID is unique only within the
    /// current process. Use [`Self::new_with_source_id`] with a globally unique
    /// shard ID when sketches may be merged across processes.
    pub fn new(config: UnivMonQConfig) -> Result<Self, UnivMonQError> {
        Self::with_hasher(config)
    }

    /// Creates a sketch with an explicit ID for its update source.
    ///
    /// Use a stable, globally unique partition or shard ID when sketches may
    /// be serialized or merged across processes.
    pub fn new_with_source_id(
        config: UnivMonQConfig,
        source_id: u64,
    ) -> Result<Self, UnivMonQError> {
        Self::with_hasher_and_source_id(config, source_id)
    }
}

fn allocate_source_id() -> u64 {
    NEXT_SOURCE_ID
        .fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed, |value| {
            value.checked_add(1)
        })
        .expect("UnivMon-Q automatic source IDs exhausted")
}

fn validate_config(config: UnivMonQConfig) -> Result<(), UnivMonQError> {
    if !(2..=63).contains(&config.levels) {
        return Err(UnivMonQError::new("levels must be in 2..=63"));
    }
    if config.width == 0 {
        return Err(UnivMonQError::new("width must be positive"));
    }
    if config.depth == 0 || config.depth % 2 == 0 {
        return Err(UnivMonQError::new("depth must be a positive odd number"));
    }
    if !matches!(config.counter_bits, 32 | 64) {
        return Err(UnivMonQError::new("counter_bits must be 32 or 64"));
    }
    if config.candidates == 0 {
        return Err(UnivMonQError::new("candidates must be positive"));
    }
    Ok(())
}

fn level_width(config: UnivMonQConfig, level: usize) -> usize {
    let shift = if config.width_halving_period == 0 {
        0
    } else {
        level as u32 / u32::from(config.width_halving_period)
    };
    let recovery_floor = config.candidates.min(config.width);
    config
        .width
        .checked_shr(shift)
        .unwrap_or(0)
        .max(recovery_floor)
}

#[inline(always)]
fn float_to_ordered(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

#[inline(always)]
fn ordered_to_float(value: u64) -> f64 {
    let bits = if value >> 63 == 1 {
        value ^ (1_u64 << 63)
    } else {
        !value
    };
    f64::from_bits(bits)
}

fn estimate_g_sum_from<F>(heavy: &[Vec<RecoveredCandidate>], g: F) -> f64
where
    F: Fn(f64) -> f64,
{
    estimate_keyed_sum_from(heavy, |_, frequency| g(frequency))
}

fn estimate_keyed_sum_from<F>(heavy: &[Vec<RecoveredCandidate>], g: F) -> f64
where
    F: Fn(u64, f64) -> f64,
{
    let Some(last) = heavy.len().checked_sub(1) else {
        return 0.0;
    };
    let mut estimate: f64 = heavy[last]
        .iter()
        .map(|candidate| g(candidate.key, candidate.frequency))
        .sum();
    for level in (0..last).rev() {
        let correction: f64 = heavy[level]
            .iter()
            .map(|candidate| {
                let sign = if candidate.terminal == level {
                    1.0
                } else {
                    -1.0
                };
                sign * g(candidate.key, candidate.frequency)
            })
            .sum();
        estimate = 2.0 * estimate + correction;
    }
    estimate
}

fn isotonicize(points: &mut [(u64, f64)]) {
    #[derive(Clone, Copy)]
    struct Block {
        start: usize,
        end: usize,
        sum: f64,
        weight: f64,
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        blocks.push(Block {
            start: index,
            end: index + 1,
            sum: point.1,
            weight: 1.0,
        });
        while blocks.len() >= 2 {
            let right = blocks[blocks.len() - 1];
            let left = blocks[blocks.len() - 2];
            if left.sum / left.weight <= right.sum / right.weight {
                break;
            }
            blocks.pop();
            blocks.pop();
            blocks.push(Block {
                start: left.start,
                end: right.end,
                sum: left.sum + right.sum,
                weight: left.weight + right.weight,
            });
        }
    }
    for block in blocks {
        let rank = (block.sum / block.weight).clamp(0.0, 1.0);
        for point in &mut points[block.start..block.end] {
            point.1 = rank;
        }
    }
}

fn decode_error(message: impl Into<String>) -> RmpDecodeError {
    RmpDecodeError::Uncategorized(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> UnivMonQConfig {
        UnivMonQConfig {
            levels: 8,
            width: 128,
            width_halving_period: 0,
            depth: 5,
            counter_bits: 64,
            candidates: 32,
            ordered_samples: 32,
            hash_seed: 9,
        }
    }

    fn quantile_rank_error(sorted: &[f64], quantile: f64, estimate: f64) -> f64 {
        let lower = sorted.partition_point(|value| value.total_cmp(&estimate).is_lt()) as f64
            / sorted.len() as f64;
        let upper = sorted.partition_point(|value| value.total_cmp(&estimate).is_le()) as f64
            / sorted.len() as f64;
        if quantile < lower {
            lower - quantile
        } else if quantile > upper {
            quantile - upper
        } else {
            0.0
        }
    }

    fn worst_quantile_rank_error(sketch: &UnivMonQ, sorted: &[f64]) -> f64 {
        [0.01, 0.1, 0.5, 0.9, 0.99]
            .into_iter()
            .map(|quantile| {
                let estimate = sketch.quantile(quantile).unwrap();
                quantile_rank_error(sorted, quantile, estimate)
            })
            .fold(0.0, f64::max)
    }

    fn maximum_cdf_error(sketch: &UnivMonQ, sorted: &[f64]) -> f64 {
        let cdf = sketch.cdf();
        let mut cdf_index = 0;
        let mut estimated = 0.0;
        let mut maximum: f64 = 0.0;
        let mut truth_index = 0;
        while truth_index < sorted.len() {
            let value = sorted[truth_index];
            let mut truth_end = truth_index + 1;
            while truth_end < sorted.len() && sorted[truth_end].total_cmp(&value).is_eq() {
                truth_end += 1;
            }
            while cdf_index < cdf.len() && !cdf[cdf_index].value.total_cmp(&value).is_gt() {
                estimated = cdf[cdf_index].rank;
                cdf_index += 1;
            }
            let exact = truth_end as f64 / sorted.len() as f64;
            maximum = maximum.max((estimated - exact).abs());
            truth_index = truth_end;
        }
        maximum
    }

    fn raw_occurrence_sample_cdf_error(sketch: &UnivMonQ, sorted: &[f64]) -> f64 {
        let mut sample: Vec<u64> = sketch
            .ordered_heap
            .iter()
            .map(|occurrence| occurrence.key)
            .collect();
        sample.sort_unstable();
        let mut maximum: f64 = 0.0;
        let mut truth_index = 0;
        while truth_index < sorted.len() {
            let value = sorted[truth_index];
            let key = float_to_ordered(value);
            let mut truth_end = truth_index + 1;
            while truth_end < sorted.len() && sorted[truth_end].total_cmp(&value).is_eq() {
                truth_end += 1;
            }
            let sample_end = sample.partition_point(|candidate| *candidate <= key);
            let exact = truth_end as f64 / sorted.len() as f64;
            let estimated = sample_end as f64 / sample.len() as f64;
            maximum = maximum.max((estimated - exact).abs());
            truth_index = truth_end;
        }
        maximum
    }

    fn percentile(values: &mut [f64], probability: f64) -> f64 {
        values.sort_unstable_by(f64::total_cmp);
        let index = ((probability * values.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1);
        values[index]
    }

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    #[test]
    fn exact_l1_and_occurrence_entropy_for_known_distribution() {
        let config = UnivMonQConfig {
            levels: 8,
            width: 4_096,
            width_halving_period: 0,
            depth: 5,
            counter_bits: 64,
            candidates: 1,
            ordered_samples: 100,
            hash_seed: 17,
        };
        let mut sketch = UnivMonQ::new(config).unwrap();
        for _ in 0..80 {
            sketch.update(&0.0);
        }
        for _ in 0..20 {
            sketch.update(&1.0);
        }

        let expected_entropy = -(0.8_f64 * 0.8_f64.ln() + 0.2_f64 * 0.2_f64.ln());
        assert_eq!(sketch.estimate_l1(), 100.0);
        assert!((sketch.estimate_entropy_occurrence().unwrap() - expected_entropy).abs() < 1e-12);
        assert!((sketch.estimate_entropy() - expected_entropy).abs() < 1e-12);

        let query = sketch.prepare_queries();
        assert_eq!(query.estimate_l1(), 100.0);
        assert!((query.estimate_entropy() - expected_entropy).abs() < 1e-12);
    }

    #[test]
    fn candidate_eviction_history_distinguishes_full_from_truncated() {
        let mut level = Level::new(32, 1, 64, 2);
        level.update(1, 1, 5);
        level.update(2, 2, 5);
        assert!(!level.ever_evicted, "filling the table is not an eviction");

        level.update(3, 3, 5);
        assert!(level.ever_evicted);
        level.clear();
        assert!(!level.ever_evicted);
    }

    #[test]
    fn candidate_merge_uses_zero_error_for_full_complete_summary() {
        let mut left = Level::new(32, 1, 64, 2);
        for _ in 0..100 {
            left.update(1, 1, 5);
        }
        left.update(2, 2, 5);
        assert!(!left.ever_evicted);

        let mut right = Level::new(32, 1, 64, 2);
        right.update(3, 3, 5);
        right.update(3, 3, 5);
        left.merge(&right, 5, u128::from);

        assert!(left.ever_evicted, "the merged union exceeded capacity");
        assert_eq!(left.candidate_scores.get(&1), Some(&100));
        assert_eq!(left.candidate_scores.get(&3), Some(&2));
        assert!(!left.candidate_scores.contains_key(&2));
    }

    #[test]
    fn asapv1_envelope_preserves_candidate_eviction_history() {
        let config = UnivMonQConfig {
            levels: 2,
            candidates: 1,
            ..tiny_config()
        };
        let mut sketch = UnivMonQ::new(config).unwrap();
        for value in [1.0, 2.0, 3.0] {
            sketch.update(&value);
        }
        assert!(sketch.levels.iter().any(|level| level.ever_evicted));

        let bytes = sketch.serialize_to_bytes().unwrap();
        let decoded = UnivMonQ::<DefaultXxHasher>::deserialize_from_bytes(&bytes).unwrap();
        let original: Vec<bool> = sketch
            .levels
            .iter()
            .map(|level| level.ever_evicted)
            .collect();
        let restored: Vec<bool> = decoded
            .levels
            .iter()
            .map(|level| level.ever_evicted)
            .collect();
        assert_eq!(restored, original);
    }

    #[test]
    fn l2_heavy_candidate_survives_a_diffuse_tail() {
        let config = UnivMonQConfig {
            levels: 2,
            width: 4096,
            depth: 5,
            counter_bits: 64,
            candidates: 32,
            ordered_samples: 0,
            ..tiny_config()
        };
        let mut sketch = UnivMonQ::new(config).unwrap();
        let terminal = 0;
        let target = (0_u64..)
            .find(|value| sketch.sample_level(float_to_ordered(*value as f64)) == terminal)
            .unwrap();
        for _ in 0..200 {
            sketch.add(&target);
        }

        let mut distinct_tail = 0_usize;
        for value in 1_u64.. {
            if value != target && sketch.sample_level(float_to_ordered(value as f64)) == terminal {
                sketch.add(&value);
                distinct_tail += 1;
                if distinct_tail == 50_000 {
                    break;
                }
            }
        }

        let exact_f2 = 200_f64.powi(2) + distinct_tail as f64;
        let l2_threshold = (exact_f2 / config.candidates as f64).sqrt();
        assert!(200.0 > 2.0 * l2_threshold);
        let target_key = float_to_ordered(target as f64);
        assert!(sketch.levels[terminal].ever_evicted);
        assert!(
            sketch.levels[terminal]
                .candidate_scores
                .contains_key(&target_key),
            "an item above twice the L2 threshold must remain recoverable"
        );
        let recovery = sketch.recover_candidates();
        assert!(
            sketch.logical_heavy_sets_from(&recovery)[terminal]
                .iter()
                .any(|candidate| candidate.key == target_key)
        );
    }

    #[test]
    fn exact_for_tiny_frequency_vector() {
        let mut sketch = UnivMonQ::new(tiny_config()).unwrap();
        for value in [5.0, 1.0, 2.0, 2.0, 9.0, 5.0, 5.0] {
            sketch.update(&value);
        }
        assert_eq!(sketch.rank(2.0), Some(3));
        assert_eq!(sketch.estimate_rank_universal(2.0), Some(3));
        assert_eq!(sketch.rank(5.0), Some(6));
        assert_eq!(sketch.quantile(0.5), Some(5.0));
        assert_eq!(sketch.quantile(0.0), Some(1.0));
        assert_eq!(sketch.quantile(1.0), Some(9.0));
        assert_eq!(sketch.estimate_frequency(5.0), 3);
        assert_eq!(sketch.estimate_distinct(), 4.0);
        assert_eq!(sketch.estimate_f2(), 15.0);
        assert_eq!(sketch.estimate_g_sum(|frequency| frequency), 7.0);
        assert_eq!(sketch.heavy_hitters(1), vec![(5.0, 3)]);

        let query = sketch.prepare_queries();
        assert_eq!(query.count(), sketch.count());
        assert_eq!(query.estimate_frequency(5.0), 3);
        assert_eq!(query.estimate_distinct(), 4.0);
        assert_eq!(query.estimate_f2(), 15.0);
        assert_eq!(query.estimate_g_sum(|frequency| frequency), 7.0);
        assert_eq!(query.heavy_hitters(1), vec![(5.0, 3)]);
        assert_eq!(query.rank(2.0), Some(3));
        assert_eq!(query.estimate_rank_universal(2.0), Some(3));
        assert_eq!(
            query.quantiles(&[0.0, 0.5, 1.0]),
            vec![Some(1.0), Some(5.0), Some(9.0)]
        );
        assert_eq!(query.cdf(), sketch.cdf());
    }

    #[test]
    fn ordered_queries_match_the_exact_oracle_when_every_occurrence_is_retained() {
        let values = [
            f64::NEG_INFINITY,
            -100.0,
            -0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            50.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let config = UnivMonQConfig {
            width: 512,
            candidates: 64,
            ordered_samples: values.len(),
            ..tiny_config()
        };
        let mut sketch = UnivMonQ::new_with_source_id(config, 0xfeed).unwrap();
        for value in values {
            sketch.update(&value);
        }
        let mut truth = values.to_vec();
        truth.sort_unstable_by(f64::total_cmp);

        for index in 0..=100 {
            let quantile = index as f64 / 100.0;
            let expected_index = if index == 0 {
                0
            } else {
                ((quantile * truth.len() as f64).ceil() as usize - 1).min(truth.len() - 1)
            };
            assert_eq!(
                sketch.quantile(quantile).unwrap().to_bits(),
                truth[expected_index].to_bits(),
                "quantile={quantile}"
            );
        }

        let mut truth_index = 0;
        while truth_index < truth.len() {
            let value = truth[truth_index];
            let mut truth_end = truth_index + 1;
            while truth_end < truth.len() && truth[truth_end].total_cmp(&value).is_eq() {
                truth_end += 1;
            }
            assert_eq!(sketch.rank(value), Some(truth_end as u64));
            truth_index = truth_end;
        }
        assert_eq!(maximum_cdf_error(&sketch, &truth), 0.0);
    }

    #[test]
    fn data_input_api_rejects_non_numeric_values() {
        let mut sketch = UnivMonQ::new(tiny_config()).unwrap();
        sketch.update_data_input(&DataInput::I64(-10)).unwrap();
        sketch.update_data_input(&DataInput::F32(2.5)).unwrap();
        assert!(sketch.update_data_input(&DataInput::Str("no")).is_err());
        assert_eq!(sketch.count(), 2);
    }

    #[test]
    fn float_encoding_matches_total_order() {
        let values = [
            f64::NEG_INFINITY,
            -10.0,
            -0.0,
            0.0,
            10.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let encoded: Vec<u64> = values
            .iter()
            .map(|value| float_to_ordered(*value))
            .collect();
        assert!(encoded.windows(2).all(|pair| pair[0] < pair[1]));
        for value in values {
            assert_eq!(
                ordered_to_float(float_to_ordered(value)).to_bits(),
                value.to_bits()
            );
        }
    }

    #[test]
    fn merge_matches_one_pass_for_complete_candidates() {
        let values = [5.0, 1.0, 9.0, 5.0, 2.0, 100.0, 2.0, 5.0, -7.0];
        let mut one_pass = UnivMonQ::new(tiny_config()).unwrap();
        for value in values {
            one_pass.update(&value);
        }
        let mut left = UnivMonQ::new(tiny_config()).unwrap();
        let mut right = UnivMonQ::new(tiny_config()).unwrap();
        for value in &values[..4] {
            left.update(value);
        }
        for value in &values[4..] {
            right.update(value);
        }
        left.merge(&right).unwrap();
        assert_eq!(left.count(), one_pass.count());
        assert_eq!(left.cdf(), one_pass.cdf());
        assert_eq!(left.estimate_f2(), one_pass.estimate_f2());
    }

    #[test]
    fn prepared_queries_match_direct_queries_after_candidate_eviction() {
        let config = UnivMonQConfig {
            candidates: 8,
            ordered_samples: 64,
            ..tiny_config()
        };
        let mut sketch = UnivMonQ::new(config).unwrap();
        for index in 0..5_000_u64 {
            let value = if index % 5 == 0 {
                42.0
            } else {
                (index.wrapping_mul(6364136223846793005) % 997) as f64
            };
            sketch.update(&value);
        }

        let direct_f0 = sketch.estimate_distinct();
        let direct_f2 = sketch.estimate_f2();
        let direct_l1 = sketch.estimate_l1();
        let direct_linear_g_sum = sketch.estimate_g_sum(|frequency| frequency);
        let direct_entropy = sketch.estimate_entropy();
        let direct_universal_entropy = sketch.estimate_entropy_universal();
        let direct_heavy = sketch.heavy_hitters(5);
        let direct_rank = sketch.rank(500.0);
        let direct_universal_rank = sketch.estimate_rank_universal(500.0);
        let direct_quantiles = sketch.quantiles(&[0.1, 0.5, 0.9, 0.99]);
        let direct_cdf = sketch.cdf();

        let query = sketch.prepare_queries();
        assert_eq!(query.estimate_distinct(), direct_f0);
        assert_eq!(query.estimate_f2(), direct_f2);
        assert_eq!(query.estimate_l1(), direct_l1);
        assert_eq!(
            query.estimate_g_sum(|frequency| frequency),
            direct_linear_g_sum
        );
        assert_eq!(query.estimate_entropy(), direct_entropy);
        assert_eq!(query.estimate_entropy_universal(), direct_universal_entropy);
        assert_eq!(query.heavy_hitters(5), direct_heavy);
        assert_eq!(query.rank(500.0), direct_rank);
        assert_eq!(query.estimate_rank_universal(500.0), direct_universal_rank);
        assert_eq!(query.quantiles(&[0.1, 0.5, 0.9, 0.99]), direct_quantiles);
        assert_eq!(query.cdf(), direct_cdf);
    }

    #[test]
    fn approximate_ordered_queries_remain_monotone_and_dual_after_eviction() {
        let config = UnivMonQConfig {
            candidates: 8,
            ordered_samples: 64,
            ..tiny_config()
        };
        let mut sketch = UnivMonQ::new_with_source_id(config, 787).unwrap();
        for index in 0..5_000_u64 {
            let value = if index % 4 == 0 {
                42.0
            } else {
                (index.wrapping_mul(6364136223846793005) % 2_003) as f64
            };
            sketch.update(&value);
        }
        assert!(sketch.levels.iter().any(|level| level.ever_evicted));
        let query = sketch.prepare_queries();
        assert!(query.cdf().windows(2).all(|points| {
            points[0].value.total_cmp(&points[1].value).is_lt() && points[0].rank <= points[1].rank
        }));
        assert_eq!(query.cdf().last().unwrap().rank, 1.0);

        let quantiles: Vec<f64> = (0..=100).map(|index| index as f64 / 100.0).collect();
        let estimates: Vec<f64> = query
            .quantiles(&quantiles)
            .into_iter()
            .map(Option::unwrap)
            .collect();
        assert!(
            estimates
                .windows(2)
                .all(|values| !values[0].total_cmp(&values[1]).is_gt())
        );
        for (quantile, estimate) in quantiles.into_iter().zip(estimates) {
            let estimated_rank = query.rank(estimate).unwrap() as f64 / query.count() as f64;
            assert!(
                estimated_rank + 1.0 / query.count() as f64 >= quantile,
                "q={quantile}, value={estimate}, estimated rank={estimated_rank}"
            );
        }
    }

    #[test]
    fn asapv1_round_trip_preserves_queries() {
        let mut sketch = UnivMonQ::new(tiny_config()).unwrap();
        for value in (0..1000).map(|value| (value % 97) as f64) {
            sketch.update(&value);
        }
        let bytes = sketch.serialize_to_bytes().unwrap();
        let decoded = UnivMonQ::<DefaultXxHasher>::deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(decoded.config(), sketch.config());
        assert_eq!(decoded.source_id(), sketch.source_id());
        assert_eq!(decoded.count(), sketch.count());
        assert_eq!(decoded.cdf(), sketch.cdf());
        assert_eq!(decoded.estimate_f2(), sketch.estimate_f2());
    }

    #[test]
    fn asapv1_checkpoint_resume_preserves_occurrence_priorities() {
        let config = UnivMonQConfig {
            candidates: 16,
            ordered_samples: 64,
            ..tiny_config()
        };
        let values: Vec<f64> = (0..5_000_u64)
            .map(|index| {
                if index % 7 == 0 {
                    42.0
                } else {
                    (index.wrapping_mul(6364136223846793005) % 997) as f64
                }
            })
            .collect();
        let mut uninterrupted = UnivMonQ::new_with_source_id(config, 991).unwrap();
        let mut checkpointed = UnivMonQ::new_with_source_id(config, 991).unwrap();
        for value in &values[..1_337] {
            uninterrupted.update(value);
            checkpointed.update(value);
        }
        let bytes = checkpointed.serialize_to_bytes().unwrap();
        let mut resumed = UnivMonQ::<DefaultXxHasher>::deserialize_from_bytes(&bytes).unwrap();
        for value in &values[1_337..] {
            uninterrupted.update(value);
            resumed.update(value);
        }

        let mut expected_sample = uninterrupted.ordered_heap.clone().into_sorted_vec();
        let mut actual_sample = resumed.ordered_heap.clone().into_sorted_vec();
        expected_sample.sort_unstable();
        actual_sample.sort_unstable();
        assert_eq!(resumed.next_sequence, values.len() as u64);
        assert_eq!(actual_sample, expected_sample);
        assert_eq!(resumed.cdf(), uninterrupted.cdf());
        assert_eq!(resumed.estimate_f2(), uninterrupted.estimate_f2());
    }

    #[test]
    fn clear_does_not_reuse_occurrence_identities() {
        let config = UnivMonQConfig {
            ordered_samples: 128,
            ..tiny_config()
        };
        let mut sketch = UnivMonQ::new_with_source_id(config, 31337).unwrap();
        for value in 0..100_u64 {
            sketch.update(&(value as f64));
        }
        let first_priorities: HashSet<(u64, u64)> = sketch
            .ordered_heap
            .iter()
            .map(|occurrence| (occurrence.priority_high, occurrence.priority_low))
            .collect();
        assert_eq!(sketch.next_sequence, 100);

        sketch.clear();
        assert_eq!(sketch.next_sequence, 100);
        for value in 0..100_u64 {
            sketch.update(&(value as f64));
        }
        assert_eq!(sketch.next_sequence, 200);
        assert!(sketch.ordered_heap.iter().all(|occurrence| {
            !first_priorities.contains(&(occurrence.priority_high, occurrence.priority_low))
        }));
    }

    #[test]
    fn occurrence_sample_merge_is_associative() {
        fn shard(source_id: u64, offset: u64) -> UnivMonQ {
            let mut sketch = UnivMonQ::new_with_source_id(tiny_config(), source_id).unwrap();
            for index in 0..1_000_u64 {
                sketch.add(&((index.wrapping_mul(17) + offset) % 991));
            }
            sketch
        }

        let (a, b, c) = (shard(10, 0), shard(20, 1), shard(30, 2));
        let mut left_associative = a.clone();
        left_associative.merge(&b).unwrap();
        left_associative.merge(&c).unwrap();

        let mut right_branch = b;
        right_branch.merge(&c).unwrap();
        let mut right_associative = a;
        right_associative.merge(&right_branch).unwrap();

        let mut left_sample = left_associative.ordered_heap.clone().into_sorted_vec();
        let mut right_sample = right_associative.ordered_heap.clone().into_sorted_vec();
        left_sample.sort_unstable();
        right_sample.sort_unstable();
        assert_eq!(left_sample, right_sample);
        assert_eq!(left_associative.cdf(), right_associative.cdf());
    }

    #[test]
    fn occurrence_sample_merge_retains_the_exact_global_bottom_k() {
        let config = UnivMonQConfig {
            ordered_samples: 64,
            ..tiny_config()
        };
        let mut shards = Vec::new();
        let mut expected = Vec::new();
        for source_id in [101, 202, 303, 404] {
            let mut shard = UnivMonQ::new_with_source_id(config, source_id).unwrap();
            for sequence in 0..500_u64 {
                let value = ((sequence.wrapping_mul(37) + source_id) % 251) as f64;
                let key = float_to_ordered(value);
                expected.push(shard.occurrence(key, sequence));
                shard.update(&value);
            }
            shards.push(shard);
        }
        expected.sort_unstable();
        expected.truncate(config.ordered_samples);

        let mut merged = shards.remove(0);
        for shard in shards {
            merged.merge(&shard).unwrap();
        }
        let mut actual = merged.ordered_heap.clone().into_vec();
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn bottom_k_sampling_is_uniform_over_stream_positions() {
        let stream_length = 256_usize;
        let sample_size = 32_usize;
        let source_trials = 256_u64;
        let config = UnivMonQConfig {
            ordered_samples: sample_size,
            ..tiny_config()
        };
        let mut inclusion_counts = vec![0_u64; stream_length];
        for source_id in 1..=source_trials {
            let mut sketch = UnivMonQ::new_with_source_id(config, source_id).unwrap();
            for position in 0..stream_length {
                sketch.update(&(position as f64));
            }
            for occurrence in &sketch.ordered_heap {
                let position = ordered_to_float(occurrence.key) as usize;
                inclusion_counts[position] += 1;
            }
        }

        let expected = source_trials as f64 * sample_size as f64 / stream_length as f64;
        let chi_squared: f64 = inclusion_counts
            .iter()
            .map(|observed| (*observed as f64 - expected).powi(2) / expected)
            .sum();
        assert_eq!(
            inclusion_counts.iter().sum::<u64>(),
            source_trials * sample_size as u64
        );
        assert!(
            chi_squared < 400.0,
            "position-inclusion chi-squared={chi_squared}"
        );
    }

    #[test]
    fn synthetic_uniform_quantiles_have_dkw_bounded_rank_error() {
        let config = UnivMonQConfig {
            levels: 10,
            width: 2048,
            width_halving_period: 5,
            depth: 3,
            counter_bits: 32,
            candidates: 256,
            ordered_samples: 512,
            hash_seed: 5,
        };
        // A fixed source identity makes this regression independent of the
        // number and order of tests that allocated automatic source IDs.
        let mut sketch = UnivMonQ::new_with_source_id(config, 1).unwrap();
        let n = 100_000_u64;
        let mut truth = Vec::with_capacity(n as usize);
        for value in 0..n {
            let value = value as f64;
            sketch.update(&value);
            truth.push(value);
        }
        for q in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let estimate = sketch.quantile(q).unwrap();
            let normalized_rank_error = quantile_rank_error(&truth, q, estimate);
            assert!(
                normalized_rank_error < 0.06,
                "q={q}, estimate={estimate}, error={normalized_rank_error}"
            );
        }
        assert!(maximum_cdf_error(&sketch, &truth) < 0.06);
    }

    #[test]
    fn ordered_queries_are_accurate_across_sources_and_distributions() {
        let config = UnivMonQConfig {
            levels: 10,
            width: 2048,
            width_halving_period: 5,
            depth: 3,
            counter_bits: 32,
            candidates: 256,
            ordered_samples: 512,
            hash_seed: 5,
        };
        let n = 50_000_u64;
        let distributions: [Vec<f64>; 3] = [
            // Diffuse, unique values.
            (0..n).map(|value| value as f64).collect(),
            // Two separated modes with repeated values.
            (0..n)
                .map(|index| {
                    let mode = if index % 2 == 0 { 0 } else { 100_000 };
                    (mode + (index.wrapping_mul(17) % 5_000)) as f64
                })
                .collect(),
            // One heavy value plus a large diffuse residual distribution.
            (0..n)
                .map(|index| {
                    if index % 10 < 6 {
                        42.0
                    } else {
                        1_000.0 + (index.wrapping_mul(6364136223846793005) % 20_000) as f64
                    }
                })
                .collect(),
        ];

        for (distribution_index, values) in distributions.into_iter().enumerate() {
            let mut truth = values.clone();
            truth.sort_unstable_by(f64::total_cmp);
            let mut rank_errors = Vec::new();
            let mut cdf_errors = Vec::new();
            for source_id in 1..=32 {
                let mut sketch = UnivMonQ::new_with_source_id(config, source_id).unwrap();
                for value in &values {
                    sketch.update(value);
                }
                rank_errors.push(worst_quantile_rank_error(&sketch, &truth));
                cdf_errors.push(maximum_cdf_error(&sketch, &truth));
            }

            let median_rank = percentile(&mut rank_errors.clone(), 0.5);
            let p95_rank = percentile(&mut rank_errors, 0.95);
            let median_cdf = percentile(&mut cdf_errors.clone(), 0.5);
            let p95_cdf = percentile(&mut cdf_errors, 0.95);
            assert!(
                median_rank < 0.035 && p95_rank < 0.06,
                "distribution={distribution_index}, median rank error={median_rank}, p95 rank error={p95_rank}"
            );
            assert!(
                median_cdf < 0.04 && p95_cdf < 0.07,
                "distribution={distribution_index}, median CDF error={median_cdf}, p95 CDF error={p95_cdf}"
            );
        }
    }

    #[test]
    fn cdf_error_decreases_at_the_expected_rate_with_sample_memory() {
        let n = 20_000_u64;
        let values: Vec<f64> = (0..n).map(|value| value as f64).collect();
        let mut mean_errors = Vec::new();
        for ordered_samples in [128, 512, 2_048] {
            let config = UnivMonQConfig {
                levels: 10,
                width: 1_024,
                width_halving_period: 5,
                depth: 3,
                counter_bits: 32,
                candidates: 128,
                ordered_samples,
                hash_seed: 5,
            };
            let mut errors = Vec::new();
            for source_id in 1..=24 {
                let mut sketch = UnivMonQ::new_with_source_id(config, source_id).unwrap();
                for value in &values {
                    sketch.update(value);
                }
                errors.push(maximum_cdf_error(&sketch, &values));
            }
            let average = mean(&errors);
            // The expected Kolmogorov error of an occurrence sample is
            // Theta(1/sqrt(k)); this also guards against accidentally sampling
            // distinct keys instead of occurrences.
            assert!(
                average * (ordered_samples as f64).sqrt() < 1.5,
                "k={ordered_samples}, mean CDF error={average}"
            );
            mean_errors.push(average);
        }
        assert!(
            mean_errors[1] < 0.75 * mean_errors[0],
            "mean errors={mean_errors:?}"
        );
        assert!(
            mean_errors[2] < 0.75 * mean_errors[1],
            "mean errors={mean_errors:?}"
        );
    }

    #[test]
    fn recovered_heavy_item_improves_residual_cdf_accuracy() {
        let n = 20_000_u64;
        let values: Vec<f64> = (0..n)
            .map(|index| {
                if index % 10 < 9 {
                    42.0
                } else {
                    1_000.0 + (index.wrapping_mul(6364136223846793005) % 10_000) as f64
                }
            })
            .collect();
        let mut truth = values.clone();
        truth.sort_unstable_by(f64::total_cmp);
        let config = UnivMonQConfig {
            levels: 10,
            width: 1_024,
            width_halving_period: 5,
            depth: 3,
            counter_bits: 64,
            candidates: 128,
            ordered_samples: 256,
            hash_seed: 5,
        };
        let mut assisted_errors = Vec::new();
        let mut raw_errors = Vec::new();
        let mut assisted_heavy_boundary_errors = Vec::new();
        let mut raw_heavy_boundary_errors = Vec::new();
        for source_id in 1..=32 {
            let mut sketch = UnivMonQ::new_with_source_id(config, source_id).unwrap();
            for value in &values {
                sketch.update(value);
            }
            assert_eq!(sketch.heavy_hitters(1)[0].0, 42.0);
            assisted_errors.push(maximum_cdf_error(&sketch, &truth));
            raw_errors.push(raw_occurrence_sample_cdf_error(&sketch, &truth));
            let exact_heavy_rank = 0.9;
            assisted_heavy_boundary_errors
                .push((sketch.rank(42.0).unwrap() as f64 / n as f64 - exact_heavy_rank).abs());
            let heavy_key = float_to_ordered(42.0);
            let raw_heavy_rank = sketch
                .ordered_heap
                .iter()
                .filter(|occurrence| occurrence.key <= heavy_key)
                .count() as f64
                / sketch.ordered_heap.len() as f64;
            raw_heavy_boundary_errors.push((raw_heavy_rank - exact_heavy_rank).abs());
        }
        let assisted_mean = mean(&assisted_errors);
        let raw_mean = mean(&raw_errors);
        assert!(
            assisted_mean < 0.85 * raw_mean,
            "assisted mean={assisted_mean}, raw occurrence mean={raw_mean}"
        );
        assert!(
            mean(&assisted_heavy_boundary_errors) < 0.25 * mean(&raw_heavy_boundary_errors),
            "assisted heavy-boundary errors={assisted_heavy_boundary_errors:?}, raw errors={raw_heavy_boundary_errors:?}"
        );
        assert!(
            percentile(&mut assisted_errors, 0.95) < 0.025,
            "assisted errors={assisted_errors:?}"
        );
    }

    #[test]
    fn eight_way_merge_retains_multi_metric_accuracy() {
        let config = UnivMonQConfig {
            levels: 10,
            width: 4096,
            width_halving_period: 7,
            depth: 3,
            counter_bits: 64,
            candidates: 512,
            ordered_samples: 1024,
            hash_seed: 5,
        };
        let mut shards: Vec<UnivMonQ> = (0..8).map(|_| UnivMonQ::new(config).unwrap()).collect();
        let mut truth = Vec::with_capacity(100_000);
        let mut exact_frequencies = HashMap::new();
        for index in 0..100_000_u64 {
            let value = if index % 5 < 2 {
                42.0
            } else {
                ((index.wrapping_mul(6364136223846793005) >> 17) % 20_000) as f64
            };
            shards[index as usize % 8].update(&value);
            truth.push(value);
            *exact_frequencies
                .entry(float_to_ordered(value))
                .or_insert(0_u64) += 1;
        }
        while shards.len() > 1 {
            let mut next = Vec::with_capacity(shards.len().div_ceil(2));
            let mut iter = shards.into_iter();
            while let Some(mut left) = iter.next() {
                if let Some(right) = iter.next() {
                    left.merge(&right).unwrap();
                }
                next.push(left);
            }
            shards = next;
        }
        let merged = shards.pop().unwrap();
        truth.sort_unstable_by(f64::total_cmp);
        for q in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let estimate = merged.quantile(q).unwrap();
            let lower = truth.partition_point(|value| value.total_cmp(&estimate).is_lt()) as f64
                / truth.len() as f64;
            let upper = truth.partition_point(|value| value.total_cmp(&estimate).is_le()) as f64
                / truth.len() as f64;
            let error = if q < lower {
                lower - q
            } else if q > upper {
                q - upper
            } else {
                0.0
            };
            assert!(error < 0.03, "q={q}, estimate={estimate}, error={error}");
        }
        let exact_f0 = exact_frequencies.len() as f64;
        let exact_f2: f64 = exact_frequencies
            .values()
            .map(|frequency| (*frequency as f64).powi(2))
            .sum();
        assert!((merged.estimate_distinct() / exact_f0 - 1.0).abs() < 0.1);
        assert!((merged.estimate_f2() / exact_f2 - 1.0).abs() < 0.1);
        assert_eq!(merged.heavy_hitters(1)[0].0, 42.0);
    }

    #[test]
    fn clear_retains_configuration() {
        let mut sketch = UnivMonQ::new(tiny_config()).unwrap();
        sketch.update(&1.0);
        sketch.clear();
        assert!(sketch.is_empty());
        assert_eq!(sketch.config(), tiny_config());
        assert_eq!(sketch.quantile(0.5), None);
    }
}
