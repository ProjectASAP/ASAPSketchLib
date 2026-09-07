//! DDSketch quantile sketch implementation.
//!
//! A mergeable, relative-error quantile sketch that maps values into
//! logarithmically-spaced buckets, guaranteeing a relative accuracy of alpha
//! for every quantile query.
//!
//! Provenance:
//! This file was adapted from earlier DDSketch work in the private
//! `approx-telemetry/asap_sketchlib` repository. Original contributor for that
//! implementation: Srinath Ramachandran. It was later migrated and modified in
//! this repository.
//!
//! Reference:
//! - Masson, Rim & Lee, "DDSketch: A Fast and Fully-Mergeable Quantile Sketch
//!   with Relative-Error Guarantees," PVLDB 12(12), 2019.
//!   <https://www.vldb.org/pvldb/vol12/p2195-masson.pdf>

use crate::DataInput;
use crate::common::input::data_input_to_f64;
use crate::common::numerical::NumericalValue;
use crate::common::structures::Vector1D;
use crate::octo_delta::DdDelta;
use serde::{Deserialize, Serialize};

/// ASAPv1 wire serialization (kind_id `0x05 0x00`).
mod wire;

// Number of buckets to grow by when expanding.
const GROW_CHUNK: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Buckets {
    counts: Vector1D<u64>,
    offset: i32,
}

impl Buckets {
    fn new() -> Self {
        Self {
            counts: Vector1D::from_vec(Vec::new()),
            offset: 0,
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    #[inline(always)]
    fn range(&self) -> Option<(i32, i32)> {
        if self.counts.is_empty() {
            None
        } else {
            let left = self.offset;
            let right = self.offset + self.counts.len() as i32 - 1;
            Some((left, right))
        }
    }

    /// Ensure bucket k exists, using growth in chunks.
    #[inline(always)]
    fn ensure(&mut self, k: i32) {
        if self.counts.is_empty() {
            self.counts = Vector1D::from_vec(vec![0u64; GROW_CHUNK]);
            self.offset = k - (GROW_CHUNK as i32 / 2);
            return;
        }

        let (left, right) = self.range().unwrap();

        if k < left {
            let needed = (left - k) as usize;
            let grow = needed.max(GROW_CHUNK);

            let mut v = vec![0u64; grow];
            v.extend_from_slice(self.counts.as_slice());

            self.counts = Vector1D::from_vec(v);
            self.offset -= grow as i32;
        } else if k > right {
            let needed = (k - right) as usize;
            let grow = needed.max(GROW_CHUNK);

            let mut v = self.counts.clone().into_vec();
            v.resize(v.len() + grow, 0);
            self.counts = Vector1D::from_vec(v);
        }
    }

    #[inline(always)]
    fn add_one(&mut self, k: i32) {
        // this is the method that gets called on every sample insertion
        let idx_i32 = k - self.offset;

        if idx_i32 >= 0 {
            let idx = idx_i32 as usize;
            let slice = self.counts.as_mut_slice();
            if idx < slice.len() {
                slice[idx] += 1;
                return;
            }
        }

        // This is the method that gets called only on rare expansions
        self.ensure(k);
        let idx = (k - self.offset) as usize;
        self.counts.as_mut_slice()[idx] += 1;
    }
}

/// Mergeable, relative-error quantile sketch using logarithmically-spaced buckets.
///
/// ASAPv1 serialization lives in the `wire` submodule (kind_id `0x05 0x00`); it
/// carries `alpha`, the bucket store, and `sum` / `min` / `max`, and recovers
/// `count` by summing the buckets.
///
/// The derived `Serialize` / `Deserialize` is a separate, Rust-internal form
/// used where a `DDSketch` is nested in another type. It carries the same state
/// as the ASAPv1 payload — `alpha`, the bucket store, and `sum` / `min` / `max`
/// — and rebuilds the index mapping and `count` on the way in, refusing a state
/// the ASAPv1 decoder would refuse.
#[derive(Debug, Serialize, Deserialize)]
#[serde(try_from = "DDSketchState")]
pub struct DDSketch {
    alpha: f64,
    #[serde(skip)]
    gamma: f64,
    #[serde(skip)]
    log_gamma: f64,
    #[serde(skip)]
    inv_log_gamma: f64,

    store: Buckets,
    #[serde(skip)]
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}

/// The fields a serialized [`DDSketch`] carries, in the order it emits them.
/// `gamma` / `log_gamma` / `inv_log_gamma` follow from `alpha` and `count` is
/// the sum of the buckets, so none of the four reaches the wire.
#[derive(Deserialize)]
struct DDSketchState {
    alpha: f64,
    store: Buckets,
    sum: f64,
    min: f64,
    max: f64,
}

impl TryFrom<DDSketchState> for DDSketch {
    type Error = String;

    fn try_from(state: DDSketchState) -> Result<Self, Self::Error> {
        wire::check_alpha(state.alpha)?;
        let counts = state.store.counts.as_slice();
        wire::check_store_span(state.store.offset, counts.len())?;
        let count = wire::total_count(counts)
            .ok_or_else(|| "DDSketch bucket counts overflow the total sample count".to_string())?;
        wire::check_scalars(count, state.sum, state.min, state.max)?;

        let gamma = (1.0 + state.alpha) / (1.0 - state.alpha);
        let log_gamma = gamma.ln();
        Ok(Self {
            alpha: state.alpha,
            gamma,
            log_gamma,
            inv_log_gamma: 1.0 / log_gamma,
            store: state.store,
            count,
            sum: state.sum,
            min: state.min,
            max: state.max,
        })
    }
}

/// Smallest and largest finite positive values whose bucket index is
/// representable without integer overflow (index within `i32`) or
/// `exp`/`powf` overflow, mirroring DataDog's logarithmic_mapping.go
/// `minIndexableValue`/`maxIndexableValue`. Values outside this range are
/// dropped rather than mapped to an arbitrarily distant bucket index — that
/// guards the dense bucket store against a single finite-but-extreme outlier
/// forcing an allocation spanning the whole index gap.
///
/// Single source of truth shared by core `DDSketch`, the portable wire twin,
/// and tests, so the two implementations cannot drift algebraically again.
pub fn ddsketch_indexable_bounds(alpha: f64) -> (f64, f64) {
    let gamma = (1.0 + alpha) / (1.0 - alpha);
    let inv_log_gamma = 1.0 / gamma.ln();
    // 709.0 is just under ln(f64::MAX) so exp() stays finite.
    const EXP_OVERFLOW: f64 = 709.0;
    let min = ((f64::from(i32::MIN)) / inv_log_gamma + 1.0)
        .exp()
        .max(f64::MIN_POSITIVE * gamma);
    let max = ((f64::from(i32::MAX)) / inv_log_gamma - 1.0)
        .exp()
        .min(EXP_OVERFLOW.exp() / (2.0 * gamma) * (gamma + 1.0));
    (min, max)
}

impl DDSketch {
    /// Creates a new DDSketch with relative accuracy guarantee `alpha` (must be in `(0, 1)`).
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0,1)");
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        let log_gamma = gamma.ln();
        let inv_log_gamma = 1.0 / log_gamma;

        Self {
            alpha,
            gamma,
            log_gamma,
            inv_log_gamma,
            store: Buckets::new(),
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Advances the running `sum` by `delta`, saturating at `f64::MAX` instead
    /// of reaching `+inf`. The wire format refuses a non-finite `sum` on a
    /// populated store, so an unguarded `+=` would leave a legally-ingested
    /// sketch that can never be serialized. `count` and the bucket store stay
    /// exact, so only `sum` degrades, and only past `f64::MAX`.
    #[inline(always)]
    fn add_to_sum(&mut self, delta: f64) {
        let next = self.sum + delta;
        self.sum = if next.is_finite() { next } else { f64::MAX };
    }

    /// Adds a positive finite numeric sample to the sketch; non-positive or
    /// non-finite values are ignored.
    ///
    /// Values outside `[min_indexable_value, max_indexable_value]` are also
    /// dropped rather than mapped to an arbitrarily distant bucket index — that
    /// guards the dense store against a single finite-but-extreme outlier
    /// forcing an allocation spanning the whole index gap. Dropped silently,
    /// like the non-positive case, since `add` has no error channel.
    #[inline(always)]
    pub fn add<T: NumericalValue>(&mut self, val: &T) {
        let v = val.to_f64();
        if !(v.is_finite() && v > 0.0) {
            return;
        }
        let (min_indexable, max_indexable) = ddsketch_indexable_bounds(self.alpha);
        if v < min_indexable || v > max_indexable {
            return; // untrackable extreme: would blow up the dense bucket span
        }

        self.count += 1;
        self.add_to_sum(v);
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }

        let k = self.key_for(v);
        self.store.add_one(k);
    }

    /// Bucket index a value maps to, or `None` if `add` would have dropped it.
    ///
    /// Exposed so an OctoSketch worker can hold one-byte counters over the same
    /// bucket space without duplicating the logarithmic mapping.
    pub fn bucket_index_for(&self, value: f64) -> Option<i32> {
        if !(value.is_finite() && value > 0.0) {
            return None;
        }
        let (min_indexable, max_indexable) = ddsketch_indexable_bounds(self.alpha);
        if value < min_indexable || value > max_indexable {
            return None;
        }
        Some(self.key_for(value))
    }

    /// Adds a promoted bucket count from an OctoSketch worker.
    ///
    /// A delta carries only a bucket and a count, so `sum`, `min` and `max` are
    /// advanced with the bucket's representative value - the same α-bounded
    /// estimate a deserialize-and-recompute produces. Quantiles and `count`
    /// stay exact with respect to the bucket store.
    pub fn apply_delta(&mut self, delta: DdDelta) {
        if delta.value == 0 {
            return;
        }
        // `merge` checks that the two sketches share an alpha; a delta carries
        // no alpha to check, so bound the index by what this sketch's own
        // mapping can produce. A worker built with a much finer alpha would
        // otherwise hand over an index near i32::MAX and grow the dense store
        // across the whole gap. Out-of-range values are dropped, which is what
        // `add` already does with values it cannot index.
        let (min_indexable, max_indexable) = ddsketch_indexable_bounds(self.alpha);
        let (lowest, highest) = (self.key_for(min_indexable), self.key_for(max_indexable));
        if delta.index < lowest || delta.index > highest {
            return;
        }
        self.store.ensure(delta.index);
        let slot = (delta.index - self.store.offset) as usize;
        self.store.counts.as_mut_slice()[slot] += delta.value;

        let representative = self.bin_representative(delta.index);
        self.count += delta.value;
        self.add_to_sum(representative * delta.value as f64);
        if representative < self.min {
            self.min = representative;
        }
        if representative > self.max {
            self.max = representative;
        }
    }

    /// Returns the estimated value at quantile `q` (in `[0, 1]`), or `None` if the sketch is empty.
    pub fn get_value_at_quantile(&self, q: f64) -> Option<f64> {
        if self.count == 0 || q.is_nan() {
            return None;
        }
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }

        let rank = (q * self.count as f64).ceil() as u64;
        let mut seen = 0u64;

        let slice = self.store.counts.as_slice();
        let offset = self.store.offset;

        for (i, &c) in slice.iter().enumerate() {
            if c == 0 {
                continue;
            }
            seen += c;
            if seen >= rank {
                let bin = offset + i as i32;
                let mut v = self.bin_representative(bin);
                if v < self.min {
                    v = self.min;
                }
                if v > self.max {
                    v = self.max;
                }
                return Some(v);
            }
        }

        Some(self.max)
    }

    /// Returns the total number of samples inserted so far.
    pub fn get_count(&self) -> u64 {
        self.count
    }

    /// Returns the minimum sample seen, or `None` if the sketch is empty.
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    /// Returns the maximum sample seen, or `None` if the sketch is empty.
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max)
        }
    }

    /// Returns the relative-accuracy parameter `alpha`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the running sum. Exact over the values passed to [`Self::add`];
    /// a bucket promoted through [`Self::apply_delta`] contributes its bucket
    /// representative instead, and the total saturates at `f64::MAX` rather
    /// than overflowing to `+inf`.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Returns the raw bucket-count slice. Each entry is the number of
    /// samples in the bucket whose absolute index is `store_offset() + i`.
    pub fn store_counts(&self) -> &[u64] {
        self.store.counts.as_slice()
    }

    /// Returns the absolute bucket index corresponding to
    /// `store_counts()[0]`.
    pub fn store_offset(&self) -> i32 {
        self.store.offset
    }

    /// Merges another DDSketch into this one. Returns `Err` if the two sketches
    /// use different index mappings (different `alpha`/`gamma`): merging under a
    /// mismatched mapping would reinterpret one sketch's bucket indices under
    /// the other's γ and silently corrupt every quantile.
    ///
    /// This is a REAL runtime check, not a `debug_assert!`: a `debug_assert!`
    /// compiles out in release builds, leaving a mismatched merge to corrupt
    /// results with no signal at all. DataDog's `MergeWith` and sketchlib-go's
    /// Go `Merge` both return an error here; the portable `DdSketch::merge` in
    /// this same crate does too.
    pub fn merge(&mut self, other: &DDSketch) -> Result<(), String> {
        if (self.alpha - other.alpha).abs() >= 1e-12 || (self.gamma - other.gamma).abs() >= 1e-12 {
            return Err(format!(
                "cannot merge DDSketches with different index mappings: alpha {} vs {}",
                self.alpha, other.alpha
            ));
        }

        if other.count == 0 {
            return Ok(());
        }
        if self.count == 0 {
            *self = other.clone();
            return Ok(());
        }

        self.count += other.count;
        self.add_to_sum(other.sum);
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }

        // Merge bucket vectors
        self.merge_buckets_from(other);
        Ok(())
    }

    #[inline(always)]
    fn key_for(&self, v: f64) -> i32 {
        debug_assert!(v > 0.0);
        (v.ln() * self.inv_log_gamma).floor() as i32
    }

    /// Lower edge γ^k of bucket k.
    #[inline]
    fn lower_bound(&self, k: i32) -> f64 {
        self.gamma.powf(k as f64)
    }

    /// Representative of bucket k: the lower bound γ^k scaled by (1+α), matching
    /// DataDog's logarithmic_mapping.go `Value = LowerBound(index) * (1 +
    /// RelativeAccuracy())`. This makes the relative error EXACTLY α at both
    /// bucket edges, as the advertised α-accuracy guarantee requires.
    #[inline]
    fn bin_representative(&self, k: i32) -> f64 {
        self.lower_bound(k) * (1.0 + self.alpha)
    }

    fn merge_buckets_from(&mut self, other: &DDSketch) {
        if other.store.is_empty() {
            return;
        }
        if self.store.is_empty() {
            self.store = other.store.clone();
            return;
        }

        let (self_l, self_r) = self.store.range().unwrap();
        let (other_l, other_r) = other.store.range().unwrap();

        let new_l = self_l.min(other_l);
        let new_r = self_r.max(other_r);
        let new_len = (new_r - new_l + 1) as usize;

        let mut merged = vec![0u64; new_len];

        // Copy self
        for (i, &c) in self.store.counts.as_slice().iter().enumerate() {
            let k = self_l + i as i32;
            merged[(k - new_l) as usize] += c;
        }

        // Add other
        for (i, &c) in other.store.counts.as_slice().iter().enumerate() {
            let k = other_l + i as i32;
            merged[(k - new_l) as usize] += c;
        }

        self.store.counts = Vector1D::from_vec(merged);
        self.store.offset = new_l;
    }
}

impl Clone for DDSketch {
    fn clone(&self) -> Self {
        Self {
            alpha: self.alpha,
            gamma: self.gamma,
            log_gamma: self.log_gamma,
            inv_log_gamma: self.inv_log_gamma,
            store: self.store.clone(),
            count: self.count,
            sum: self.sum,
            min: self.min,
            max: self.max,
        }
    }
}

impl DDSketch {
    /// Adds a sample converted from a [`DataInput`]; returns an error for non-numeric inputs.
    #[inline(always)]
    pub fn add_input(&mut self, v: &DataInput) -> Result<(), &'static str> {
        let value = data_input_to_f64(v).map_err(|_| "DDSketch only accepts numeric inputs")?;
        self.add(&value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query_basic() {
        let mut s = DDSketch::new(0.01);
        let vals = [0.0, -5.0, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 1000.0];
        for &v in &vals {
            s.add(&v);
        }

        // Non-positives ignored
        assert_eq!(s.get_count(), 7);

        let ps = [0.0, 0.5, 0.9, 0.99, 1.0];
        let mut prev = f64::NEG_INFINITY;
        for &p in &ps {
            let q = s.get_value_at_quantile(p).expect("quantile");
            assert!(q >= prev - 1e-12, "non-monotone at p={p}: {q} < {prev}");
            assert!(q <= s.max().unwrap() + 1e-12);
            assert!(q >= s.min().unwrap() - 1e-12);
            prev = q;
        }
    }

    #[test]
    fn empty_quantile_returns_none() {
        let s = DDSketch::new(0.01);
        assert!(s.get_value_at_quantile(0.5).is_none());
        assert!(s.get_value_at_quantile(0.0).is_none());
        assert!(s.get_value_at_quantile(1.0).is_none());
        assert_eq!(s.get_count(), 0);
    }

    #[test]
    fn merge_two_sketches_combines_counts_and_bounds() {
        const ALPHA: f64 = 0.01;

        let mut s1 = DDSketch::new(ALPHA);
        let mut s2 = DDSketch::new(ALPHA);

        let vals1 = [1.0, 2.0, 3.0, 4.0];
        let vals2 = [5.0, 10.0, 20.0];

        for v in vals1 {
            s1.add(&v);
        }
        for v in vals2 {
            s2.add(&v);
        }

        s1.merge(&s2).unwrap();

        // counts and bounds
        assert_eq!(s1.get_count(), (vals1.len() + vals2.len()) as u64);
        assert_eq!(s1.min().unwrap(), 1.0);
        assert_eq!(s1.max().unwrap(), 20.0);

        // extreme quantiles should match bounds
        assert_eq!(s1.get_value_at_quantile(0.0).unwrap(), 1.0);
        assert_eq!(s1.get_value_at_quantile(1.0).unwrap(), 20.0);

        // sanity: middle quantile is within [min, max]
        let mid = s1.get_value_at_quantile(0.5).unwrap();
        assert!((1.0..=20.0).contains(&mid));
    }

    #[test]
    fn dds_serialization_round_trip() {
        let mut s = DDSketch::new(0.01);
        for v in [1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 1000.0] {
            s.add(&v);
        }

        let encoded = s.serialize_to_bytes().expect("DDSketch serialization fail");
        assert!(
            !encoded.is_empty(),
            "encoded bytes should not be empty for DDSketch"
        );
        let decoded =
            DDSketch::deserialize_from_bytes(&encoded).expect("DDSketch deserialization fail");

        // `count` is summed back from the buckets; `sum`/`min`/`max` are carried
        // on the wire, so every scalar and every quantile comes back exact.
        assert_eq!(decoded.get_count(), s.get_count());
        assert_eq!(decoded.sum(), s.sum());
        assert_eq!(decoded.min(), s.min());
        assert_eq!(decoded.max(), s.max());
        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_eq!(
                decoded.get_value_at_quantile(q),
                s.get_value_at_quantile(q),
                "quantile p={q} diverged after a round trip"
            );
        }
    }

    // DataDog-parity tests.

    #[test]
    fn representative_within_alpha_at_bucket_edges() {
        for &alpha in &[0.001, 0.01, 0.05, 0.1] {
            let d = DDSketch::new(alpha);
            for &k in &[-100i32, -1, 0, 1, 7, 500] {
                let lo = d.lower_bound(k);
                let hi = d.lower_bound(k + 1);
                let rep = d.bin_representative(k);
                assert!(rep >= lo && rep <= hi, "rep {rep} outside [{lo},{hi}]");
                assert!(
                    (rep - lo).abs() / lo <= alpha + 1e-9,
                    "alpha={alpha} k={k}: lower-edge relerr exceeds alpha"
                );
                assert!(
                    (rep - hi).abs() / hi <= alpha + 1e-9,
                    "alpha={alpha} k={k}: upper-edge relerr exceeds alpha"
                );
            }
        }
    }

    #[test]
    fn merge_alpha_mismatch_is_a_real_runtime_error() {
        let mut a = DDSketch::new(0.01);
        let b = DDSketch::new(0.02);
        a.add(&5.0);
        assert!(a.merge(&b).is_err(), "mismatched-mapping merge must Err");

        let mut c = DDSketch::new(0.01);
        let mut d = DDSketch::new(0.01);
        c.add(&3.0);
        d.add(&7.0);
        assert!(c.merge(&d).is_ok(), "matched-mapping merge must succeed");
        assert_eq!(c.get_count(), 2);
    }

    #[test]
    fn untrackable_extreme_is_dropped() {
        // A single finite-but-extreme outlier outside the indexable range must
        // not be recorded, so the dense bucket store never spans the whole gap.
        let mut d = DDSketch::new(0.01);
        for i in 1..=2000 {
            d.add(&(f64::from(i)));
        }
        let count_before = d.get_count();
        let span_before = d.store.counts.as_slice().len();

        let (min_indexable, max_indexable) = ddsketch_indexable_bounds(0.01);
        d.add(&(max_indexable * 10.0));
        d.add(&(min_indexable / 10.0));
        assert_eq!(d.get_count(), count_before, "extreme values were recorded");
        assert_eq!(
            d.store.counts.as_slice().len(),
            span_before,
            "store span grew from an untrackable extreme"
        );

        // A large-but-trackable value is still recorded.
        d.add(&(max_indexable / 2.0));
        assert_eq!(d.get_count(), count_before + 1);
    }

    fn populated_sketch() -> DDSketch {
        let mut sketch = DDSketch::new(0.01);
        for v in [0.25f64, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 1000.0] {
            sketch.add(&v);
        }
        sketch
    }

    /// The derived serde form carries every scalar the buckets do not
    /// determine, so a round trip continues the run rather than resetting it.
    #[test]
    fn serde_round_trip_keeps_the_running_scalars() {
        let sketch = populated_sketch();
        let bytes = rmp_serde::to_vec(&sketch).expect("encode");
        let restored: DDSketch = rmp_serde::from_slice(&bytes).expect("decode");

        assert_eq!(restored.get_count(), sketch.get_count());
        assert_eq!(restored.sum(), sketch.sum());
        assert_eq!(restored.min(), sketch.min());
        assert_eq!(restored.max(), sketch.max());
        assert_eq!(restored.alpha(), sketch.alpha());
        assert_eq!(restored.store_counts(), sketch.store_counts());
        assert_eq!(restored.store_offset(), sketch.store_offset());
        for q in [0.0, 0.25, 0.5, 0.9, 1.0] {
            assert_eq!(
                restored.get_value_at_quantile(q),
                sketch.get_value_at_quantile(q),
                "quantile {q} moved across the round trip"
            );
        }
    }

    /// A decoded sketch keeps ingesting on top of the state it came back with.
    #[test]
    fn serde_round_trip_leaves_the_sketch_usable() {
        let sketch = populated_sketch();
        let bytes = rmp_serde::to_vec(&sketch).expect("encode");
        let mut restored: DDSketch = rmp_serde::from_slice(&bytes).expect("decode");

        restored.add(&5.0f64);
        assert_eq!(restored.get_count(), sketch.get_count() + 1);
        assert_eq!(restored.sum(), sketch.sum() + 5.0);
        assert_eq!(
            restored.store_counts().iter().sum::<u64>(),
            restored.get_count(),
            "the recovered count drifted from the buckets"
        );
    }

    /// The scalars are checked against the store the same way the ASAPv1
    /// decoder checks them: a populated store with an empty sketch's scalars
    /// is refused rather than decoded into an inconsistent sketch.
    #[test]
    fn serde_refuses_scalars_that_disagree_with_the_store() {
        #[derive(Serialize)]
        struct CraftedState {
            alpha: f64,
            store: Buckets,
            sum: f64,
            min: f64,
            max: f64,
        }

        let crafted = CraftedState {
            alpha: 0.01,
            store: Buckets {
                counts: Vector1D::from_vec(vec![1u64, 2]),
                offset: -3,
            },
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        };
        let bytes = rmp_serde::to_vec(&crafted).expect("encode crafted state");
        let err = rmp_serde::from_slice::<DDSketch>(&bytes)
            .expect_err("scalars disagreeing with the store must be refused");
        assert!(
            err.to_string().contains("DDSketch scalars"),
            "unexpected error: {err}"
        );
    }

    /// An alpha outside `(0, 1)` gives a meaningless index mapping, so it is
    /// refused on the way in rather than at the first query.
    #[test]
    fn serde_refuses_an_out_of_range_alpha() {
        #[derive(Serialize)]
        struct CraftedState {
            alpha: f64,
            store: Buckets,
            sum: f64,
            min: f64,
            max: f64,
        }

        let crafted = CraftedState {
            alpha: 1.5,
            store: Buckets {
                counts: Vector1D::from_vec(Vec::new()),
                offset: 0,
            },
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        };
        let bytes = rmp_serde::to_vec(&crafted).expect("encode crafted state");
        let err = rmp_serde::from_slice::<DDSketch>(&bytes)
            .expect_err("an out-of-range alpha must be refused");
        assert!(err.to_string().contains("alpha"), "unexpected error: {err}");
    }
}
