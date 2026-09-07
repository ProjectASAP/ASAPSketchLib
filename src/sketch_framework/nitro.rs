//! Nitro Sketch, with batch processing
//! Assume the Nitro Sketch can get a batch of input
//! For streaming Nitro, please refers to Nitro struct in structure_utils.rs
//!
//! Reference:
//! - NitroSketch paper.
//!   <https://dl.acm.org/doi/10.1145/3341302.3342076>

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
use serde::{Deserialize, Serialize};

use crate::{
    Count, CountMin, DataInput, DefaultXxHasher, FastPath, FastPathHasher, MatrixFastHash,
    PRECOMPUTED_SAMPLE, Vector2D,
};

/// Trait for sketch backends that support Nitro row updates.
pub trait NitroTarget {
    /// Returns the number of rows in the target sketch.
    fn rows(&self) -> usize;
    /// Applies a sampled update to one row.
    fn update_row(&mut self, row: usize, hashed: u128, delta: u64);
    /// Applies a sampled record to EVERY row using the target's own fast-path
    /// hash derivation. Insert and estimation must share a single hash domain,
    /// otherwise estimates read cells the inserts never wrote.
    ///
    /// Updating all rows is what makes estimates unbiased: each sampled item
    /// contributes weight ×(1/rate) to every row, so per-row counters converge
    /// to the true frequency (NitroSketch §4, estimator = min/median ÷ rate).
    fn update_sample(&mut self, value: &DataInput, delta: u64);
}

/// Saturates a Nitro update weight into the `i32` counter domain. Counter
/// storage is `i32`, so a weight beyond `i32::MAX` (reachable via rates below
/// ~4.7e-10, or by writing the public `delta` field directly) must saturate
/// rather than wrap — a wrapped negative weight would silently turn Count-Min
/// counters into decrements.
#[inline]
pub fn nitro_delta_saturated_i32(delta: u64) -> i32 {
    delta.min(i32::MAX as u64) as i32
}

/// [`nitro_delta_saturated_i32`] twin for `u32`-backed bare-storage targets.
#[inline]
pub fn nitro_delta_saturated_u32(delta: u64) -> u32 {
    delta.min(u32::MAX as u64) as u32
}

/// Trait for Nitro targets that can be merged.
pub trait NitroMerge {
    /// Merges another target into this one.
    fn merge(&mut self, other: &Self);
}

/// Trait for Nitro targets that support median-style estimation.
pub trait NitroEstimate {
    /// Returns the target's estimate for `value`.
    fn estimate_median(&self, value: &DataInput) -> f64;
}

impl NitroTarget for Vector2D<u32> {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows()
    }

    #[inline(always)]
    fn update_row(&mut self, row: usize, hashed: u128, delta: u64) {
        self.update_by_row(
            row,
            hashed,
            |a, b| *a += b,
            nitro_delta_saturated_u32(delta),
        );
    }

    #[inline(always)]
    fn update_sample(&mut self, value: &DataInput, delta: u64) {
        let hashed = <Self as FastPathHasher<DefaultXxHasher>>::hash_for_matrix(self, value);
        let cols = self.cols();
        for row in 0..self.rows() {
            let col = MatrixFastHash::col_for_row(&hashed, row, cols);
            self.update_one_counter(
                row,
                col,
                |a: &mut u32, b: u32| *a += b,
                nitro_delta_saturated_u32(delta),
            );
        }
    }
}

impl NitroMerge for CountMin<Vector2D<i32>, FastPath> {
    #[inline(always)]
    fn merge(&mut self, other: &Self) {
        CountMin::merge(self, other);
    }
}

impl NitroEstimate for CountMin<Vector2D<i32>, FastPath> {
    #[inline(always)]
    fn estimate_median(&self, value: &DataInput) -> f64 {
        self.nitro_estimate(value)
    }
}

impl NitroMerge for Count<Vector2D<i32>, FastPath> {
    #[inline(always)]
    fn merge(&mut self, other: &Self) {
        Count::merge(self, other);
    }
}

impl NitroEstimate for Count<Vector2D<i32>, FastPath> {
    #[inline(always)]
    fn estimate_median(&self, value: &DataInput) -> f64 {
        self.estimate(value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Batch-oriented Nitro wrapper around a sketch target.
pub struct NitroBatch<S: NitroTarget> {
    sampling_rate: f64,
    /// Remaining items to skip before the next sampled update.
    pub to_skip: usize,
    inv_ln_one_minus_p: f64,
    /// Weight applied to each sampled update.
    pub delta: u64,
    /// Live sampling RNG for [`NitroBatch::insert`]. Not serialized: a decoded
    /// batch restarts it from the OS, so an `insert` stream does not resume
    /// across a round trip. The cached path's state (`idx`, `to_skip`) is
    /// serialized and does resume.
    #[serde(skip)]
    #[serde(default = "new_small_rng")]
    generator: SmallRng,
    idx: usize,
    /// Unused; retained so the serialized field order is unchanged.
    mask: usize,
    sk: S,
}

/// Length of the shared `ln(1 - u)` skip table.
const SKIP_TABLE_LEN: usize = crate::common::precompute_sample::PRECOMPUTED_SAMPLE_LEN;

fn new_small_rng() -> SmallRng {
    let mut seed_rng = rng();
    SmallRng::from_rng(&mut seed_rng)
}

impl Default for NitroBatch<Vector2D<u32>> {
    fn default() -> Self {
        let mut n = NitroBatch {
            sampling_rate: 0.0,
            to_skip: 0,
            inv_ln_one_minus_p: 0.0,
            delta: 0,
            generator: new_small_rng(),
            idx: 0,
            mask: SKIP_TABLE_LEN - 1,
            sk: Vector2D::init(5, 2048),
        };
        n.sk.fill(0);
        n
    }
}

impl NitroBatch<Vector2D<u32>> {
    /// Creates a Nitro sketch with the given sampling rate.
    pub fn init_nitro(rate: f64) -> Self {
        let mut sk = Vector2D::init(5, 2048);
        sk.fill(0);
        Self::with_target(rate, sk)
    }

    /// [`NitroBatch::init_nitro`] with an explicit sampling-RNG seed. See
    /// [`NitroBatch::with_target_and_seed`].
    pub fn init_nitro_with_seed(rate: f64, seed: u64) -> Self {
        let mut sk = Vector2D::init(5, 2048);
        sk.fill(0);
        Self::with_target_and_seed(rate, sk, seed)
    }
}

impl<S: NitroTarget> NitroBatch<S> {
    /// Returns the wrapped target sketch.
    pub fn target(&self) -> &S {
        &self.sk
    }

    /// Returns the wrapped target sketch mutably.
    pub fn target_mut(&mut self) -> &mut S {
        &mut self.sk
    }

    /// Consumes the wrapper and returns the target sketch.
    pub fn into_target(self) -> S {
        self.sk
    }

    /// Wraps an existing target sketch with Nitro sampling.
    ///
    /// The sampling RNG is seeded from the OS, so two runs over the same input
    /// admit different subsets. Use [`NitroBatch::with_target_and_seed`] when
    /// the result has to be reproducible.
    pub fn with_target(rate: f64, sk: S) -> Self {
        Self::build(rate, sk, new_small_rng())
    }

    /// Wraps an existing target sketch with Nitro sampling driven by an
    /// explicitly seeded RNG.
    ///
    /// Sampling is where all of Nitro's randomness lives: which updates reach
    /// the target sketch is drawn from the geometric skip distribution. With a
    /// fixed seed the admitted subset — and therefore every estimate — is a
    /// deterministic function of the input, which is what lets an accuracy
    /// bound be asserted reproducibly instead of re-rolled on every run.
    pub fn with_target_and_seed(rate: f64, sk: S, seed: u64) -> Self {
        let mut this = Self::build(rate, sk, SmallRng::seed_from_u64(seed));
        // `insert` draws its skips live from `generator`; `insert_cached_step`
        // reads the shared table, so the seed has to move that cursor too. The
        // table is a fixed stream of `ln(1 - u)` draws, so two far-apart
        // offsets read disjoint stretches of it.
        this.idx = (seed % SKIP_TABLE_LEN as u64) as usize;
        this
    }

    fn build(rate: f64, sk: S, generator: SmallRng) -> Self {
        assert!(
            !rate.is_nan() && rate > 0.0 && rate <= 1.0,
            "sample_rate must be within (0.0, 1.0]"
        );
        let inv_ln = if (rate - 1.0).abs() <= f64::EPSILON {
            0.0 // Not used for full sampling
        } else {
            1.0 / (1.0 - rate).ln()
        };
        let mut nitro = Self {
            sampling_rate: rate,
            to_skip: 0,
            inv_ln_one_minus_p: inv_ln,
            generator,
            delta: 0,
            idx: 0,
            mask: SKIP_TABLE_LEN - 1,
            sk,
        };
        // `delta` is the integer part of the per-update weight; the fractional
        // remainder is paid per admitted update by `admitted_weight`.
        nitro.delta = nitro.scaled_increment(1);
        nitro
    }

    // for profiling
    #[inline(always)]
    /// Draws the next geometric skip distance.
    pub fn draw_geometric(&mut self) {
        if self.is_full_sampling() {
            self.to_skip = 0;
            return;
        }
        let k = loop {
            let r = self.generator.random::<f64>();
            if r != 0.0_f64 && r != 1.0_f64 {
                break r;
            }
        };
        // Inverse-CDF draw of Geometric(p) on {0, 1, ...}. `floor`, not
        // `ceil`: the caller's `+1` stride supplies the sampled item itself,
        // and `E[skip] = (1-p)/p` holds only under `floor`.
        self.to_skip = ((1.0 - k).ln() * self.inv_ln_one_minus_p).floor() as usize;
        self.next_cursor();
    }

    /// The cursor actually used to index the skip table, taken modulo its
    /// length so a value decoded from an old payload cannot index out of
    /// bounds.
    #[inline(always)]
    fn cursor(&self) -> usize {
        self.idx % SKIP_TABLE_LEN
    }

    /// Advances the cursor by one, wrapping at the table's real length.
    #[inline(always)]
    fn next_cursor(&mut self) {
        self.idx = (self.cursor() + 1) % SKIP_TABLE_LEN;
    }

    /// The next skip distance read from the precomputed table, **scaled to the
    /// configured rate**.
    ///
    /// The table holds `ln(1 - u)` for a fixed stream of uniforms; multiplying
    /// by `inv_ln_one_minus_p = 1 / ln(1 - p)` makes each entry an inverse-CDF
    /// draw of `Geometric(p)`.
    #[inline(always)]
    fn cached_geometric(&mut self) {
        if self.is_full_sampling() {
            self.to_skip = 0;
            return;
        }
        self.to_skip =
            (PRECOMPUTED_SAMPLE[self.cursor()] * self.inv_ln_one_minus_p).floor() as usize;
        self.next_cursor();
    }

    #[inline(always)]
    /// Decrements the current skip counter by one.
    pub fn reduce_to_skip(&mut self) {
        self.to_skip -= 1;
    }

    #[inline(always)]
    /// Decrements the current skip counter by `c`.
    pub fn reduce_to_skip_by_count(&mut self, c: usize) {
        self.to_skip -= c;
    }

    #[inline(always)]
    /// Returns the configured sampling rate.
    pub fn get_sampling_rate(&self) -> f64 {
        self.sampling_rate
    }

    #[inline(always)]
    /// The integer part of the weight one admitted update carries.
    ///
    /// The exact weight is `weight / p`, which is only an integer when `1/p`
    /// is. See [`NitroBatch::admitted_weight`] for how the remainder is paid.
    pub fn scaled_increment(&self, weight: u64) -> u64 {
        if self.is_full_sampling() {
            weight
        } else {
            ((weight as f64) / self.sampling_rate).floor() as u64
        }
    }

    /// The weight to write for one admitted update, by **stochastic rounding**
    /// of `weight / p`.
    ///
    /// Nitro admits each update with probability `p` and compensates by
    /// writing `weight / p`. Counters are integers, so that value has to be
    /// rounded, and rounding it the same way every time biases the estimator
    /// by the rounding error at every rate whose reciprocal is not an integer
    /// (`ceil` at `p = 0.3` writes 4, so `E[est] = 1.2 f`). With
    /// `q = floor(weight/p)` and `r = weight/p - q`,
    ///
    /// ```text
    ///   W = q + Bernoulli(r)      E[W] = weight / p       Var[W] = r (1 - r)
    /// ```
    ///
    /// so `E[est] = weight * f` for **every** rate, at the cost of
    /// `r(1-r) <= 1/4` extra variance per admitted update. The draw is per
    /// *update*, never per key, which is what keeps the estimator unbiased for
    /// each key separately.
    ///
    /// The rounding draw and the geometric skip draw are consecutive outputs
    /// of the same seeded `SmallRng`. `Var[est] = f((1-p)/p + p r(1-r))` needs
    /// the weights to be independent of the admission indicators, which holds
    /// under the usual model that distinct generator outputs are independent
    /// uniforms — the same assumption the geometric schedule already makes.
    ///
    /// When `frac == 0` **no draw is consumed**, so at a reciprocal-integer
    /// rate the generator's stream — and therefore the admitted subset — is
    /// identical to a build without stochastic rounding.
    #[inline(always)]
    pub fn admitted_weight(&mut self, weight: u64) -> u64 {
        if self.is_full_sampling() {
            return weight;
        }
        let exact = (weight as f64) / self.sampling_rate;
        let floor = exact.floor();
        let frac = exact - floor;
        if frac <= 0.0 {
            return floor as u64;
        }
        let u = self.generator.random::<f64>();
        floor as u64 + u64::from(u < frac)
    }

    #[inline(always)]
    fn is_full_sampling(&self) -> bool {
        (self.sampling_rate - 1.0).abs() <= f64::EPSILON
    }

    #[inline(always)]
    /// Legacy cached-path snapshot: `(cursor, 1/ln(1-p), to_skip, unused)`.
    ///
    /// This covers the **cached** schedule only. It does not carry the live
    /// `SmallRng` that [`NitroBatch::insert`] draws from, nor the stochastic
    /// rounding draws that share it, so it cannot resume an `insert` stream.
    pub fn get_ctx(&self) -> (usize, f64, usize, usize) {
        (self.idx, self.inv_ln_one_minus_p, self.to_skip, self.mask)
    }

    #[inline(always)]
    /// Restores the cached-path state captured by [`NitroBatch::get_ctx`].
    pub fn commit_ctx(&mut self, idx: usize, to_skip: usize) {
        self.idx = idx;
        self.to_skip = to_skip;
    }

    /// Inserts a batch of values using geometric skipping.
    pub fn insert(&mut self, data: &[i64]) {
        self.draw_geometric();
        let mut position = self.to_skip;
        while position < data.len() {
            let key = DataInput::I64(data[position]);
            let weight = self.admitted_weight(1);
            self.sk.update_sample(&key, weight);
            self.draw_geometric();
            position += self.to_skip + 1;
        }
    }

    /// Inserts a batch using the precomputed skip table.
    pub fn insert_cached_step(&mut self, data: &[i64]) {
        self.cached_geometric();
        let mut position = self.to_skip;
        while position < data.len() {
            let key = DataInput::I64(data[position]);
            let weight = self.admitted_weight(1);
            self.sk.update_sample(&key, weight);
            self.cached_geometric();
            position += self.to_skip + 1;
        }
    }
}

impl<S: NitroTarget + NitroMerge> NitroBatch<S> {
    /// Merges another Nitro sketch with the same sampling rate.
    pub fn merge(&mut self, other: &Self) {
        assert!(
            (self.sampling_rate - other.sampling_rate).abs() <= f64::EPSILON,
            "nitro merge requires matching sampling rates"
        );
        self.sk.merge(&other.sk);
    }
}

impl<S: NitroTarget + NitroEstimate> NitroBatch<S> {
    /// Returns the wrapped sketch's estimate for `value`.
    pub fn estimate_median(&self, value: &DataInput) -> f64 {
        self.sk.estimate_median(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataInput;
    use crate::test_utils::sample_zipf_u64;
    use std::collections::HashMap;

    /// Fixed sampling-RNG seed. `with_target` seeds from the OS, so an
    /// accuracy assertion built on it would be re-rolled every run.
    const NITRO_TEST_SEED: u64 = 0x0117_5EED;

    /// The cached path walks the shared skip table and wraps at its length.
    ///
    /// The cursor is private, so this is a unit test rather than an E2E one.
    /// `insert_cached_step` reading one entry per admission is what makes the
    /// seeded offset meaningful: a cursor that never advanced would replay one
    /// skip distance forever, and every seed would admit the same subset.
    #[test]
    fn the_cached_path_advances_the_cursor_once_per_admission_and_wraps() {
        const RATE: f64 = 0.1;
        let data = vec![7i64; 10_000];

        let mut nitro = NitroBatch::with_target_and_seed(
            RATE,
            CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 1024),
            0,
        );
        assert_eq!(nitro.cursor(), 0, "seed 0 starts at entry 0");
        nitro.insert_cached_step(&data);
        // One table entry per drawn skip: the leading draw plus one per
        // admission, so roughly `n * p`.
        let advanced = nitro.cursor();
        let admissions = (data.len() as f64 * RATE) as usize;
        assert!(
            advanced >= admissions / 2 && advanced <= 2 * admissions,
            "the cursor advanced {advanced} entries for about {admissions} admissions"
        );

        // Starting near the end must wrap rather than run out of table.
        let start = SKIP_TABLE_LEN - 8;
        let mut wrapping = NitroBatch::with_target_and_seed(
            RATE,
            CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 1024),
            start as u64,
        );
        assert_eq!(wrapping.cursor(), start);
        wrapping.insert_cached_step(&data);
        assert!(
            wrapping.cursor() < start,
            "the cursor must wrap at the table length"
        );

        // A cursor decoded from an older payload can be past the table.
        let mut hostile = NitroBatch::with_target_and_seed(
            RATE,
            CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 1024),
            1,
        );
        hostile.commit_ctx(usize::MAX, 0);
        assert!(hostile.cursor() < SKIP_TABLE_LEN);
        hostile.insert_cached_step(&data); // must not panic
    }

    #[test]
    fn nitro_batch_countmin_error_bound_zipf() {
        let rows = 3;
        let cols = 4096;
        let domain = 8192;
        let exponent = 1.1;
        let samples = 200_000;
        let seed = 0x5eed_c0de;

        let mut truth = HashMap::<i64, u64>::new();
        let data: Vec<i64> = sample_zipf_u64(domain, exponent, samples, seed)
            .into_iter()
            .map(|v| {
                let key = v as i64;
                *truth.entry(key).or_insert(0) += 1;
                key
            })
            .collect();

        let cm = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(rows, cols);
        let mut batch = NitroBatch::with_target_and_seed(1.0, cm, NITRO_TEST_SEED);
        batch.insert(&data);

        let epsilon = std::f64::consts::E / cols as f64;
        let delta = 1.0 / std::f64::consts::E.powi(rows as i32);
        let error_bound = epsilon * samples as f64;
        let correct_lower_bound = truth.len() as f64 * (1.0 - delta);
        let mut within_count = 0;
        for key in truth.keys() {
            let est = batch.estimate_median(&DataInput::I64(*key));
            if (est - (*truth.get(key).unwrap() as f64)).abs() < error_bound {
                within_count += 1;
            }
        }
        assert!(
            within_count as f64 > correct_lower_bound,
            "in-bound items number {within_count} not greater than expected amount {correct_lower_bound}"
        );
    }

    #[test]
    fn nitro_batch_count_error_bound_zipf() {
        let rows = 3;
        let cols = 4096;
        let domain = 8192;
        let exponent = 1.1;
        let samples = 200_000;
        let seed = 0x5eed_c0de;

        let mut truth = HashMap::<i64, u64>::new();
        let data: Vec<i64> = sample_zipf_u64(domain, exponent, samples, seed)
            .into_iter()
            .map(|v| {
                let key = v as i64;
                *truth.entry(key).or_insert(0) += 1;
                key
            })
            .collect();

        let cs = Count::<Vector2D<i32>, FastPath>::with_dimensions(rows, cols);
        let mut batch = NitroBatch::with_target_and_seed(1.0, cs, NITRO_TEST_SEED);
        batch.insert(&data);

        // Count Sketch's bound, not Count-Min's. The error is driven by the L2
        // norm of the residual frequency vector and is rank-independent:
        //
        //   Var[row estimator] <= ||f_-i||_2^2 / w
        //   Chebyshev at t = sqrt(kappa/w) * ||f_-i||_2 -> per-row failure 1/kappa
        //   the reported value is the median of d rows, so the query fails
        //   only when at least ceil(d/2) rows do.
        //
        // Reusing Count-Min's eps*N here would be checking a bound this sketch
        // never claimed — and on a Zipf stream that bound is far looser, so it
        // would pass almost regardless of what the sketch did.
        const KAPPA: f64 = 3.0;
        let f2: f64 = truth.values().map(|c| (*c as f64) * (*c as f64)).sum();
        // P[Bin(3, 1/3) >= 2] = 7/27.
        let median_failure = 7.0 / 27.0;
        let correct_lower_bound = truth.len() as f64 * (1.0 - median_failure);
        let mut within_count = 0;
        for (key, exact) in &truth {
            let f = *exact as f64;
            let residual_l2 = (f2 - f * f).max(0.0).sqrt();
            let error_bound = (KAPPA / cols as f64).sqrt() * residual_l2;
            let est = batch.estimate_median(&DataInput::I64(*key));
            if (est - f).abs() <= error_bound {
                within_count += 1;
            }
        }
        assert!(
            within_count as f64 > correct_lower_bound,
            "{within_count} of {} keys within sqrt(kappa/w)*||f_-i||_2; the median-of-{rows} \
             bound allows a failure probability of {median_failure:.4}, so at least \
             {correct_lower_bound:.1} must be in bound",
            truth.len()
        );
    }
}
