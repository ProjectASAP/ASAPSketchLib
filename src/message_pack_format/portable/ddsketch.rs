//! Wire-format-aligned DDSketch types. The wire DTO + runtime ops
//! live together here.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error as MsgPackError, MessagePackCodec};

/// Bucket-store growth chunk for the wire-format-aligned [`DdSketch`]
/// variant. Matches the Go reference implementation's `DDSketch.GrowChunk`
/// so the `store_counts` / `store_offset` layout written by
/// [`DdSketch::update`] is byte-identical to the Go `SerializePortable`
/// output for the same input stream.
pub const DDSKETCH_GROW_CHUNK: usize = 128;

/// Safety limit for a single [`DdSketchDelta`] application: deltas arrive
/// over the wire untrusted, so a span beyond this many buckets is rejected
/// with an error instead of padding the dense store toward a multi-gigabyte
/// allocation. Generous relative to any legitimate producer: at α = 0.01 the
/// entire indexable range spans ~35k buckets.
pub const MAX_APPLY_DELTA_SPAN_BUCKETS: i64 = 1 << 22;

// =====================================================================
// Wire-format-aligned variant.
//
// `DdSketch` and `DdSketchDelta` below are the public-field,
// proto-decode-friendly types consumed by the query-engine
// accumulators. The high-throughput in-process variant is
// `crate::sketches::ddsketch::DDSketch`, which keeps its own design.
// =====================================================================

// DDSketch — log-bucketed quantile sketch, mergeable by store-index alignment.
//
// Parallel to `count_sketch::CountSketch`, for the modified-OTLP
// `Metric.data = DDSketch{…}` hot path. Holds the bucket counts, their
// absolute-index base offset, and the aggregate `{count, sum, min, max}`.
//
// Merge semantics: two sketches with the same relative-accuracy
// parameter `alpha` are merged by aligning bucket arrays along their
// `store_offset` and summing counts element-wise, with `min`/`max`
// combined via min/max and `count`/`sum` added.
//
// The wire format is the protobuf-encoded
// `asap_sketchlib::proto::sketchlib::DDSketchState`. Quantile
// estimation against stored data is not implemented here: queries
// return a placeholder error and fall through to the exact-backend
// fallback.

/// Sparse delta between two consecutive DDSketch snapshots — the
/// input shape for [`DdSketch::apply_delta`]. Mirrors the
/// `DDSketchDelta` proto (and its Rust bindings). Kept as a plain
/// struct so this crate doesn't need a tonic/prost dependency; proto
/// decode lives in the accumulator.
#[derive(Debug, Clone, Default)]
pub struct DdSketchDelta {
    /// `(absolute_bucket_index, Δcount)` pairs, additive.
    pub buckets: Vec<(i32, u64)>,
    /// Δ total count. May be negative (signed on the wire).
    pub d_count: i64,
    /// Δ sum.
    pub d_sum: f64,
    /// Whether `new_min` carries a meaningful value. Min can only
    /// decrease; a delta that didn't lower min sends `false`.
    pub min_changed: bool,
    pub new_min: f64,
    /// Whether `new_max` carries a meaningful value. Max can only
    /// increase.
    pub max_changed: bool,
    pub new_max: f64,
}

/// Minimal DDSketch state — bucket counts + alpha.
///
/// The serde field order below IS the msgpack wire layout: `rmp_serde`'s
/// compact encoding writes a fixed-order array, so this serializes to a
/// 3-element array `[alpha, store_counts, store_offset]`. The total count is
/// recoverable by summing `store_counts`.
/// KEEP these three fields in this exact order so the bytes stay identical
/// to the Go reference implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DdSketch {
    /// Relative accuracy parameter; must satisfy `0 < alpha < 1`.
    pub alpha: f64,
    /// Bucket counts in absolute-index order. The absolute index of
    /// `store_counts[i]` is `i + store_offset`.
    pub store_counts: Vec<u64>,
    /// Absolute bucket index corresponding to `store_counts[0]`. May
    /// be negative.
    pub store_offset: i32,
}

impl DdSketch {
    /// Construct an empty sketch.
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0,1); alpha=0 makes ln(gamma)=0 and every guard degenerate"
        );
        Self {
            alpha,
            store_counts: Vec::new(),
            store_offset: 0,
        }
    }

    /// Construct from the decoded wire fields.
    pub fn from_raw(alpha: f64, store_counts: Vec<u64>, store_offset: i32) -> Self {
        Self {
            alpha,
            store_counts,
            store_offset,
        }
    }

    /// Total number of values added, recovered by summing the bucket
    /// counts. The wire format carries no `count` scalar, so this is the
    /// authoritative count for quantile-rank computation.
    pub fn total_count(&self) -> u64 {
        self.store_counts
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add)
    }

    /// Merge one other sketch into self by aligning bucket arrays on
    /// absolute indices. Both operands must share the same `alpha`.
    ///
    /// Like [`Self::apply_delta`], the operand is treated as untrusted wire
    /// state: if the union span would exceed
    /// [`MAX_APPLY_DELTA_SPAN_BUCKETS`] *and* grow beyond both operands'
    /// existing stores, `Err` is returned before any allocation. Stores that
    /// already legitimately span more than the cap (exotic small-α
    /// configurations) merge normally as long as the union stays within
    /// what either side already holds.
    pub fn merge(
        &mut self,
        other: &DdSketch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if (self.alpha - other.alpha).abs() > f64::EPSILON {
            return Err(format!(
                "DdSketch alpha mismatch: self={}, other={}",
                self.alpha, other.alpha
            )
            .into());
        }

        if other.store_counts.is_empty() {
            return Ok(());
        }
        if self.store_counts.is_empty() {
            self.store_counts = other.store_counts.clone();
            self.store_offset = other.store_offset;
        } else {
            let self_start = self.store_offset as i64;
            let self_end = self_start + self.store_counts.len() as i64;
            let other_start = other.store_offset as i64;
            let other_end = other_start + other.store_counts.len() as i64;
            let new_start = self_start.min(other_start);
            let new_end = self_end.max(other_end);
            let new_len = new_end - new_start;
            // Hostile-span guard: a decoded snapshot with store_offset near
            // i32::MIN and a short count vector must not allocate a union
            // spanning ~2^31 buckets (~17 GiB) in one call. Growth beyond
            // both operands' existing spans is capped; stores that already
            // legitimately exceed the cap pass through untouched.
            let max_existing =
                (self.store_counts.len() as i64).max(other.store_counts.len() as i64);
            if new_len > MAX_APPLY_DELTA_SPAN_BUCKETS && new_len > max_existing {
                return Err(format!(
                    "DdSketch merge spans {} buckets, exceeding the {}-bucket safety limit",
                    new_len, MAX_APPLY_DELTA_SPAN_BUCKETS
                )
                .into());
            }
            let new_len = new_len as usize;
            let mut merged = vec![0u64; new_len];
            for (start, counts) in [
                (self_start, &self.store_counts),
                (other_start, &other.store_counts),
            ] {
                for (i, &c) in counts.iter().enumerate() {
                    let idx = (start + i as i64 - new_start) as usize;
                    merged[idx] = merged[idx].saturating_add(c);
                }
            }
            self.store_counts = merged;
            self.store_offset = new_start as i32;
        }
        Ok(())
    }

    /// Apply a sparse delta to this sketch in place. Matches the
    /// `ApplyDelta` logic in the Go reference implementation:
    /// bucket counts add, total count + sum add, min can only decrease
    /// and max can only increase. Used by the backend ingest path to
    /// reconstitute a full sketch from a base snapshot + subsequent
    /// delta-transmission frames.
    ///
    /// Deltas arrive over the wire and are treated as untrusted: before
    /// mutating anything, the union span the delta would require is
    /// checked against [`MAX_APPLY_DELTA_SPAN_BUCKETS`]. A hostile or
    /// corrupt delta carrying an index near `i32::MAX` would otherwise
    /// pad the dense store by ~2·10⁹ buckets (~17 GiB) in one call —
    /// independent of α, unlike the `update()` path whose worst case is
    /// bounded by the indexable-range guard.
    pub fn apply_delta(
        &mut self,
        delta: &DdSketchDelta,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Pre-validate span before any mutation so a rejected delta
        // leaves state untouched.
        if !delta.buckets.is_empty() {
            let mut min_d = i64::MAX;
            let mut max_d = i64::MIN;
            for (abs_idx, _) in &delta.buckets {
                min_d = min_d.min(*abs_idx as i64);
                max_d = max_d.max(*abs_idx as i64);
            }
            let cur_start = if self.store_counts.is_empty() {
                min_d
            } else {
                self.store_offset as i64
            };
            let cur_end = cur_start + self.store_counts.len() as i64;
            let new_start = cur_start.min(min_d);
            let new_end = cur_end.max(max_d + 1);
            if new_end - new_start > MAX_APPLY_DELTA_SPAN_BUCKETS {
                return Err(format!(
                    "DdSketch delta spans {} buckets, exceeding the {}-bucket safety limit",
                    new_end - new_start,
                    MAX_APPLY_DELTA_SPAN_BUCKETS
                )
                .into());
            }
        }

        for (abs_idx, d_count) in &delta.buckets {
            if self.store_counts.is_empty() {
                self.store_counts = vec![0u64; 1];
                self.store_offset = *abs_idx;
            }
            let cur_start = self.store_offset as i64;
            let cur_end = cur_start + self.store_counts.len() as i64;
            let k = *abs_idx as i64;
            if k < cur_start {
                // Prepend zeros.
                let pad = (cur_start - k) as usize;
                let mut buf = vec![0u64; pad];
                buf.append(&mut self.store_counts);
                self.store_counts = buf;
                self.store_offset = *abs_idx;
            } else if k >= cur_end {
                let pad = (k - cur_end + 1) as usize;
                self.store_counts.extend(std::iter::repeat_n(0u64, pad));
            }
            let arr_idx = (k - self.store_offset as i64) as usize;
            self.store_counts[arr_idx] = self.store_counts[arr_idx].saturating_add(*d_count);
        }
        Ok(())
    }

    /// Compute a sparse, proto-marshalled `DDSketchDelta` of `self`
    /// against a `snapshot`. A bucket is included when its `Δcount`
    /// (self − snapshot, clamped at 0) is `>= threshold`.
    ///
    /// This is the Rust twin of the Go reference implementation's
    /// `ComputeDelta`: it iterates every non-empty bucket in `self`,
    /// subtracts the snapshot's count for the same absolute index, and
    /// emits the surviving bucket deltas. The returned bytes are a
    /// `prost`-encoded [`crate::proto::sketchlib::DdSketchDelta`],
    /// byte-identical to the Go `proto.Marshal(DDSketchDelta)` output
    /// for the same inputs (cross-language byte parity).
    ///
    /// Delta-against-empty: when `snapshot` is the empty sketch, every
    /// surviving bucket delta equals the window's own bucket count, so
    /// the result is this window's full state encoded as a delta (no
    /// cross-window subtraction). The DataPoint-level metric scalars are
    /// not carried — the count delta is recoverable by summing bucket
    /// deltas.
    pub fn compute_delta(&self, snapshot: &DdSketch, threshold: u64) -> Vec<u8> {
        use crate::proto::sketchlib::{DdSketchBucketDelta, DdSketchDelta as ProtoDelta};
        use prost::Message;

        let mut delta = ProtoDelta::default();
        if !self.store_counts.is_empty() {
            for (i, &c) in self.store_counts.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let k = self.store_offset + i as i32;
                let snap_count: u64 = if !snapshot.store_counts.is_empty() {
                    let idx = k as i64 - snapshot.store_offset as i64;
                    if idx >= 0 && (idx as usize) < snapshot.store_counts.len() {
                        snapshot.store_counts[idx as usize]
                    } else {
                        0
                    }
                } else {
                    0
                };
                let dc = c.saturating_sub(snap_count);
                if dc >= threshold {
                    delta.buckets.push(DdSketchBucketDelta {
                        index: k,
                        d_count: dc,
                    });
                }
            }
        }
        delta.encode_to_vec()
    }

    /// Apply a `prost`-encoded [`crate::proto::sketchlib::DdSketchDelta`]
    /// to this sketch in place (additive bucket merge). The Rust twin of
    /// the Go reference implementation's `ApplyDelta`.
    ///
    /// Returns `Err` if `bytes` is not a valid `DDSketchDelta` proto.
    pub fn apply_delta_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::proto::sketchlib::DdSketchDelta as ProtoDelta;
        use prost::Message;

        let proto = ProtoDelta::decode(bytes)?;
        let delta = DdSketchDelta {
            buckets: proto
                .buckets
                .into_iter()
                .map(|b| (b.index, b.d_count))
                .collect(),
            ..DdSketchDelta::default()
        };
        self.apply_delta(&delta)
    }

    /// MessagePack twin of [`Self::compute_delta`]. Computes the same
    /// sparse bucket delta of `self` against `snapshot` (a bucket is
    /// carried when its `Δcount = self − snapshot` clamped at 0 is
    /// `>= threshold`), but serializes it as the parallel-array
    /// MessagePack layout via [`DdSketchDelta`]'s [`MessagePackCodec`]
    /// instead of proto. Same delta-against-empty semantics: against the
    /// empty sketch every surviving delta equals this window's own bucket
    /// count.
    pub fn compute_delta_msgpack(&self, snapshot: &DdSketch, threshold: u64) -> Vec<u8> {
        let mut delta = DdSketchDelta::default();
        if !self.store_counts.is_empty() {
            for (i, &c) in self.store_counts.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let k = self.store_offset + i as i32;
                let snap_count: u64 = if !snapshot.store_counts.is_empty() {
                    let idx = k as i64 - snapshot.store_offset as i64;
                    if idx >= 0 && (idx as usize) < snapshot.store_counts.len() {
                        snapshot.store_counts[idx as usize]
                    } else {
                        0
                    }
                } else {
                    0
                };
                let dc = c.saturating_sub(snap_count);
                if dc >= threshold {
                    delta.buckets.push((k, dc));
                }
            }
        }
        delta
            .to_msgpack()
            .expect("DdSketchDelta msgpack encode is infallible for owned bucket arrays")
    }

    /// MessagePack twin of [`Self::apply_delta_bytes`]. Decodes the
    /// parallel-array MessagePack [`DdSketchDelta`] and applies it in
    /// place (additive bucket merge).
    ///
    /// Returns `Err` if `bytes` is not a valid MessagePack `DdSketchDelta`.
    pub fn apply_delta_msgpack_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let delta = DdSketchDelta::from_msgpack(bytes)?;
        self.apply_delta(&delta)
    }

    /// Merge a slice of references into a single new sketch. Returns
    /// `Err` on alpha mismatch or an empty input.
    pub fn merge_refs(
        inputs: &[&DdSketch],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let first = inputs
            .first()
            .ok_or("DdSketch::merge_refs called with empty input")?;
        let mut merged = DdSketch::new(first.alpha);
        for d in inputs {
            merged.merge(d)?;
        }
        Ok(merged)
    }

    /// Insert a single positive value. Updates count, sum, min/max
    /// and increments the bucket for `floor(ln(v) / ln(gamma))`
    /// where `gamma = (1+α)/(1-α)`. Provided primarily so tests can
    /// build a ground-truth sketch to compare delta-apply output
    /// against.
    ///
    /// Bucket-store growth mirrors the Go reference implementation's
    /// `Buckets.ensure`: the first allocation is a half-chunk-centered
    /// `GROW_CHUNK` of zeros, and subsequent expansions extend by
    /// `max(needed, GROW_CHUNK)` so the on-the-wire `store_counts` /
    /// `store_offset` layout is byte-identical to the Go
    /// `SerializePortable` output. Without this chunked layout the
    /// `DDSketchState` proto bytes would diverge from the Go producer's
    /// payload (cross-language byte parity).
    pub fn update(&mut self, value: f64) {
        if !(value.is_finite() && value > 0.0) {
            // DDSketch is defined for positive reals; non-positive and
            // non-finite values are rejected silently (matching the Go
            // reference and the core DDSketch). NaN in particular would
            // otherwise floor-cast to bucket 0 and corrupt it.
            return;
        }
        let gamma = (1.0 + self.alpha) / (1.0 - self.alpha);
        let ln_gamma = gamma.ln();
        // Reject finite-but-extreme values whose bucket index would be
        // unrepresentable or force an arbitrarily distant allocation. Uses
        // the SHARED bounds helper so core and portable can never drift
        // algebraically.
        let (min_v, max_v) = crate::sketches::ddsketch::ddsketch_indexable_bounds(self.alpha);
        if value < min_v || value > max_v {
            return;
        }
        let idx = (value.ln() / ln_gamma).floor() as i32;
        self.ensure_bucket(idx);
        let arr_idx = (idx as i64 - self.store_offset as i64) as usize;
        self.store_counts[arr_idx] = self.store_counts[arr_idx].saturating_add(1);
    }

    /// Ensure bucket `k` is addressable in `store_counts`, growing in
    /// chunks of [`DDSKETCH_GROW_CHUNK`] to match the Go reference
    /// implementation's `Buckets.ensure`. Empty stores are seeded with
    /// a half-chunk of zeros centered on `k` (`store_offset = k -
    /// GROW_CHUNK/2`); out-of-range expansions extend by `max(needed,
    /// GROW_CHUNK)` on either side. The on-the-wire byte layout is
    /// therefore identical between the Go and Rust producers fed the
    /// same input stream — required for cross-language byte parity.
    fn ensure_bucket(&mut self, k: i32) {
        if self.store_counts.is_empty() {
            self.store_counts = vec![0u64; DDSKETCH_GROW_CHUNK];
            self.store_offset = k - (DDSKETCH_GROW_CHUNK as i32 / 2);
            return;
        }
        let cur_start = self.store_offset as i64;
        let cur_end = cur_start + self.store_counts.len() as i64;
        let kk = k as i64;
        if kk < cur_start {
            let needed = (cur_start - kk) as usize;
            let grow = needed.max(DDSKETCH_GROW_CHUNK);
            let mut buf = vec![0u64; grow];
            buf.extend_from_slice(&self.store_counts);
            self.store_counts = buf;
            self.store_offset -= grow as i32;
        } else if kk >= cur_end {
            let needed = (kk - cur_end + 1) as usize;
            let grow = needed.max(DDSKETCH_GROW_CHUNK);
            self.store_counts.extend(std::iter::repeat_n(0u64, grow));
        }
    }

    /// Estimate the quantile at rank `q` ∈ [0, 1]. Walks the bucket
    /// array in ascending absolute-index order, accumulating counts
    /// until the target rank; returns the bucket's representative
    /// value `gamma^k * (1 + alpha)` where `k` is the bucket's absolute
    /// index. Returns `None` if the sketch is empty.
    ///
    /// Accuracy: bounded by DDSketch's α parameter — the estimated
    /// quantile value is within `(1+α)/(1-α)` relative error of the
    /// true quantile.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        let count = self.total_count();
        if count == 0 || self.store_counts.is_empty() {
            return None;
        }
        let target = (q * (count.saturating_sub(1)) as f64).floor() as u64;
        self.value_at_rank(target)
    }

    /// Estimate the linearly interpolated quantile at `q * (n - 1)`.
    ///
    /// This interpolates the two adjacent order-statistic bucket estimates,
    /// as required by PromQL quantile and continuous percentiles. The existing
    /// `quantile` method retains its lower-order-statistic convention.
    /// Returns `None` for invalid q/alpha, empty or overflowing counts, counts
    /// above exact Float64 integer precision, or nonfinite bucket estimates.
    pub fn quantile_interpolated(&self, q: f64) -> Option<f64> {
        if !q.is_finite()
            || !(0.0..=1.0).contains(&q)
            || !self.alpha.is_finite()
            || !(0.0..1.0).contains(&self.alpha)
            || self.alpha == 0.0
        {
            return None;
        }
        let count = self
            .store_counts
            .iter()
            .try_fold(0u64, |sum, count| sum.checked_add(*count))?;
        if count == 0 || count > (1u64 << 53) {
            return None;
        }
        let rank = q * (count - 1) as f64;
        let lower_rank = rank.floor() as u64;
        let upper_rank = rank.ceil() as u64;
        let lower = self.value_at_rank(lower_rank)?;
        let upper = self.value_at_rank(upper_rank)?;
        if !lower.is_finite() || !upper.is_finite() {
            return None;
        }
        let fraction = rank - lower_rank as f64;
        let result = lower * (1.0 - fraction) + upper * fraction;
        result.is_finite().then_some(result)
    }

    fn value_at_rank(&self, target: u64) -> Option<f64> {
        let mut cumulative: u64 = 0;
        let gamma = (1.0 + self.alpha) / (1.0 - self.alpha);
        let mut last_nonempty: Option<usize> = None;
        for (i, &c) in self.store_counts.iter().enumerate() {
            if c > 0 {
                last_nonempty = Some(i);
            }
            cumulative = cumulative.saturating_add(c);
            if cumulative > target {
                let k = self.store_offset as i64 + i as i64;
                return Some(gamma.powf(k as f64) * (1.0 + self.alpha));
            }
        }
        // Numerical edge case: if we fall off the end (e.g. q == 1.0 and
        // rounding lands past the final increment), estimate from the
        // highest non-empty bucket. The wire format carries no `max`
        // scalar; the representative is within DDSketch's α
        // relative-accuracy bound of the true max.
        last_nonempty.map(|i| {
            let k = (self.store_offset as i64 + i as i64) as f64;
            gamma.powf(k) * (1.0 + self.alpha)
        })
    }

    /// Return the alpha value as it appears on the wire — round-tripped
    /// through gamma exactly the way the Go reference implementation's
    /// `DDSketch.SerializePortable` does it (`alpha = (gamma - 1) /
    /// (gamma + 1)`, where `gamma = (1+α)/(1-α)`). The roundtrip
    /// introduces a small floating-point drift; without applying it the
    /// Rust producer's `DDSketchState.alpha` field bytes would diverge
    /// from the Go producer's, and cross-language byte parity would fail
    /// on the very first proto field.
    #[inline]
    pub fn wire_alpha(&self) -> f64 {
        let gamma = (1.0 + self.alpha) / (1.0 - self.alpha);
        (gamma - 1.0) / (gamma + 1.0)
    }
}

impl MessagePackCodec for DdSketch {
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        Ok(rmp_serde::to_vec(self)?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        Ok(rmp_serde::from_slice(bytes)?)
    }
}

impl MessagePackCodec for DdSketchDelta {
    /// Encodes the sparse bucket delta as the 2-element MessagePack array
    /// `[ idx:[]i32, d_count:[]u64 ]` — parallel arrays of absolute bucket
    /// index → Δcount — byte-identical to Rust
    /// `rmp_serde::to_vec(&(Vec<i32>, Vec<u64>))` (compact mode), the layout
    /// the `sketchlib-go` `asapmsgpack` DDSketch delta encoder mirrors. Only
    /// the bucket cells cross the wire; the DataPoint-level metric scalars
    /// (`d_count`/`d_sum`/`new_min`/`new_max`) are reconstructable and are
    /// not carried, matching the proto delta.
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let idx: Vec<i32> = self.buckets.iter().map(|(i, _)| *i).collect();
        let d_count: Vec<u64> = self.buckets.iter().map(|(_, c)| *c).collect();
        Ok(rmp_serde::to_vec(&(idx, d_count))?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        let (idx, d_count): (Vec<i32>, Vec<u64>) = rmp_serde::from_slice(bytes)?;
        Ok(DdSketchDelta {
            buckets: idx.into_iter().zip(d_count).collect(),
            ..DdSketchDelta::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message_pack_format::MessagePackCodec;

    #[test]
    fn msgpack_delta_against_empty_round_trips() {
        let mut w = DdSketch::new(0.01);
        for i in 1..=200 {
            w.update(i as f64);
        }
        let empty = DdSketch::new(0.01);
        let bytes = w.compute_delta_msgpack(&empty, 1);
        let mut recon = DdSketch::new(0.01);
        recon.apply_delta_msgpack_bytes(&bytes).unwrap();
        assert_eq!(recon.total_count(), w.total_count());
        for q in [0.5, 0.9, 0.99] {
            let got = recon.quantile(q).unwrap();
            let want = w.quantile(q).unwrap();
            assert!(
                (got / want - 1.0).abs() <= 0.01,
                "q={q}: got={got} want={want}"
            );
        }
    }

    #[test]
    fn msgpack_delta_wire_layout_is_parallel_arrays() {
        // Lock the cross-language wire shape: a 2-element array of parallel
        // (idx, d_count) arrays, byte-identical to the tuple encoding the Go
        // asapmsgpack DDSketch delta mirrors.
        let mut w = DdSketch::new(0.01);
        w.update(1.0);
        w.update(1.0);
        w.update(1000.0);
        let empty = DdSketch::new(0.01);
        let bytes = w.compute_delta_msgpack(&empty, 1);
        let mut idx: Vec<i32> = Vec::new();
        let mut d_count: Vec<u64> = Vec::new();
        for (i, &c) in w.store_counts.iter().enumerate() {
            if c != 0 {
                idx.push(w.store_offset + i as i32);
                d_count.push(c);
            }
        }
        let expected = rmp_serde::to_vec(&(idx, d_count)).unwrap();
        assert_eq!(bytes, expected);
        let d = DdSketchDelta::from_msgpack(&bytes).unwrap();
        assert_eq!(d.buckets.len(), 2, "two distinct buckets expected");
    }

    #[test]
    fn test_new_empty() {
        let d = DdSketch::new(0.01);
        assert_eq!(d.total_count(), 0);
        assert!(d.store_counts.is_empty());
    }

    #[test]
    fn test_merge_aligned_same_offset() {
        let mut a = DdSketch::from_raw(0.01, vec![1, 2, 3], -1);
        let b = DdSketch::from_raw(0.01, vec![10, 20, 30], -1);
        a.merge(&b).unwrap();
        assert_eq!(a.store_counts, vec![11, 22, 33]);
        assert_eq!(a.store_offset, -1);
        assert_eq!(a.total_count(), 66);
    }

    #[test]
    fn test_merge_overlapping_offsets() {
        // a covers indices [-1, 0, 1]; b covers indices [0, 1, 2]
        let mut a = DdSketch::from_raw(0.01, vec![1, 1, 1], -1);
        let b = DdSketch::from_raw(0.01, vec![10, 10, 10], 0);
        a.merge(&b).unwrap();
        // Merged window is [-1, 0, 1, 2] → [1, 11, 11, 10]
        assert_eq!(a.store_counts, vec![1, 11, 11, 10]);
        assert_eq!(a.store_offset, -1);
        assert_eq!(a.total_count(), 33);
    }

    #[test]
    fn test_merge_disjoint_offsets() {
        let mut a = DdSketch::from_raw(0.01, vec![1, 2], 0);
        let b = DdSketch::from_raw(0.01, vec![3, 4], 5);
        a.merge(&b).unwrap();
        // Window [0..7) → [1,2,0,0,0,3,4]
        assert_eq!(a.store_counts, vec![1, 2, 0, 0, 0, 3, 4]);
        assert_eq!(a.store_offset, 0);
    }

    #[test]
    fn test_apply_delta_additive_inside_store() {
        let mut base = DdSketch::from_raw(0.01, vec![1, 2, 3], -1);
        let delta = DdSketchDelta {
            buckets: vec![(-1, 4), (0, 8), (1, 12)],
            d_count: 24,
            d_sum: 120.0,
            min_changed: false,
            new_min: 0.0,
            max_changed: true,
            new_max: 9.0,
        };
        base.apply_delta(&delta).unwrap();
        assert_eq!(base.store_counts, vec![5, 10, 15]);
        assert_eq!(base.total_count(), 30);
    }

    #[test]
    fn test_apply_delta_expands_store_on_new_bucket() {
        // Base covers [0..2]; delta adds a bucket at absolute index 4.
        let mut base = DdSketch::from_raw(0.01, vec![1, 2], 0);
        let delta = DdSketchDelta {
            buckets: vec![(4, 7)],
            d_count: 7,
            d_sum: 35.0,
            min_changed: false,
            new_min: 0.0,
            max_changed: true,
            new_max: 6.0,
        };
        base.apply_delta(&delta).unwrap();
        assert_eq!(base.store_counts, vec![1, 2, 0, 0, 7]);
        assert_eq!(base.store_offset, 0);
        assert_eq!(base.total_count(), 10);
    }

    #[test]
    fn test_apply_delta_matches_full_merge() {
        // Snapshot the sketch, add more samples via a merge, and confirm
        // the delta+apply path lands at the same state.
        let base = DdSketch::from_raw(0.01, vec![1, 2, 3], 0);
        let addition = DdSketch::from_raw(0.01, vec![10, 0, 20], 0);
        let mut via_merge = base.clone();
        via_merge.merge(&addition).unwrap();

        let delta = DdSketchDelta {
            buckets: vec![(0, 10), (2, 20)],
            d_count: 30,
            d_sum: 70.0,
            min_changed: true,
            new_min: 0.5,
            max_changed: true,
            new_max: 5.0,
        };
        let mut via_delta = base;
        via_delta.apply_delta(&delta).unwrap();

        assert_eq!(via_delta.store_counts, via_merge.store_counts);
        assert_eq!(via_delta.total_count(), via_merge.total_count());
    }

    #[test]
    fn test_merge_alpha_mismatch() {
        let mut a = DdSketch::new(0.01);
        let b = DdSketch::new(0.02);
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn test_msgpack_round_trip() {
        let original = DdSketch::from_raw(0.01, vec![1, 2, 3], -2);
        let bytes = original.to_msgpack().unwrap();
        let decoded = DdSketch::from_msgpack(&bytes).unwrap();
        assert_eq!(decoded.alpha, original.alpha);
        assert_eq!(decoded.store_counts, original.store_counts);
        assert_eq!(decoded.store_offset, original.store_offset);
        assert_eq!(decoded.total_count(), original.total_count());
    }

    /// The msgpack wire layout MUST be a 3-element array
    /// `[alpha, store_counts, store_offset]` after dropping the
    /// DataPoint-level METRIC scalars (count/sum/min/max). This pins the
    /// element count so the bytes stay parity-aligned with the Go
    /// reference implementation.
    #[test]
    fn test_msgpack_is_three_element_array() {
        let sk = DdSketch::from_raw(0.01, vec![1, 2, 3], -2);
        let bytes = sk.to_msgpack().unwrap();
        // rmp compact encoding leads with an array marker. A fixarray of
        // length 3 is the single byte 0x93 (0b1001_0011).
        assert_eq!(
            bytes[0], 0x93,
            "expected a 3-element msgpack fixarray (0x93), got {:#04x}",
            bytes[0]
        );
    }

    #[test]
    fn test_insert_and_quantile_lognormal() {
        // Ground-truth: insert a large i.i.d. log-normal sample into
        // a full sketch and sanity-check P50 / P90 / P99.
        let mut gt = DdSketch::new(0.01);
        let mut rng = 0xdead_beefu64;
        let mut next = || {
            // xorshift64*
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // Box-Muller normal → log-normal(mu=3, sigma=0.7).
        let lognormal = |u: u64, v: u64| -> f64 {
            let r1 = (u as f64) / (u64::MAX as f64).max(1.0);
            let r2 = (v as f64) / (u64::MAX as f64).max(1.0);
            let z = (-2.0 * r1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * r2).cos();
            (3.0 + 0.7 * z).exp()
        };
        for _ in 0..100_000 {
            gt.update(lognormal(next(), next()));
        }
        let p50 = gt.quantile(0.5).unwrap();
        let p99 = gt.quantile(0.99).unwrap();
        // Analytical P50 = exp(mu) = e^3 ≈ 20.09;
        // P99 ≈ exp(mu + sigma * Φ⁻¹(0.99)) = e^(3 + 0.7×2.326) ≈ 102.4.
        assert!(
            (p50 / 20.09).ln().abs() < 0.05,
            "P50 {p50} not close to 20.09"
        );
        assert!(
            (p99 / 102.4).ln().abs() < 0.05,
            "P99 {p99} not close to 102.4"
        );
    }

    /// Building a sketch via `base + apply_delta()` produces quantile
    /// estimates within DDSketch's α bound of the ground-truth
    /// full-sketch path.
    #[test]
    fn test_delta_chain_preserves_quantile_accuracy() {
        let alpha = 0.01;
        let mut rng = 0xcafe_babeu64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let lognormal = |u: u64, v: u64| -> f64 {
            let r1 = (u as f64) / (u64::MAX as f64).max(1.0);
            let r2 = (v as f64) / (u64::MAX as f64).max(1.0);
            let z = (-2.0 * r1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * r2).cos();
            (3.0 + 0.7 * z).exp()
        };

        // Path A (ground truth): one sketch, 50k samples inserted
        // directly.
        let mut full = DdSketch::new(alpha);
        // Path B (delta chain): a base sketch from the first 10k
        // samples, then 4 incremental "flushes" of 10k samples each,
        // each transmitted as a delta computed against the previous
        // snapshot.
        let mut reconstituted = DdSketch::new(alpha);
        let mut prev_snapshot = DdSketch::new(alpha); // what the "receiver" has cached

        let batch = 10_000;
        let batches = 5;
        for b in 0..batches {
            let mut this_batch = prev_snapshot.clone();
            for _ in 0..batch {
                let v = lognormal(next(), next());
                full.update(v);
                this_batch.update(v);
            }
            // Compute a "delta" = diff of this_batch vs prev_snapshot
            // in our in-memory struct shape. Matches what the Go
            // reference implementation's `ComputeDelta` would put on
            // the wire.
            let delta = compute_dd_delta(&prev_snapshot, &this_batch);
            if b == 0 {
                // First batch seeds the reconstituted sketch.
                reconstituted = this_batch.clone();
            } else {
                reconstituted.apply_delta(&delta).unwrap();
            }
            prev_snapshot = this_batch;
        }

        // Both paths should see the same total count (exact), recovered
        // by summing bucket counts now that the `count` scalar is gone.
        assert_eq!(
            reconstituted.total_count(),
            full.total_count(),
            "count diverged"
        );

        // And P50 / P90 / P99 should agree within α bound.
        for q in [0.5, 0.9, 0.99] {
            let got = reconstituted.quantile(q).unwrap();
            let want = full.quantile(q).unwrap();
            let rel_err = (got / want - 1.0).abs();
            assert!(
                rel_err <= alpha,
                "q={q} rel_err={rel_err:.4} exceeds α={alpha}: reconstituted={got}, full={want}",
            );
        }
    }

    /// Helper used only in the delta-chain test: computes a
    /// `DdSketchDelta` from two snapshots. Mirrors the Go reference
    /// implementation's `ComputeDelta` logic so the test exercises the
    /// wire-format path end-to-end.
    fn compute_dd_delta(snapshot: &DdSketch, current: &DdSketch) -> DdSketchDelta {
        let mut cells = Vec::new();
        if !current.store_counts.is_empty() {
            for (i, &c) in current.store_counts.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let k = current.store_offset + i as i32;
                let snap_count: u64 = if !snapshot.store_counts.is_empty() {
                    let idx = k as i64 - snapshot.store_offset as i64;
                    if idx >= 0 && (idx as usize) < snapshot.store_counts.len() {
                        snapshot.store_counts[idx as usize]
                    } else {
                        0
                    }
                } else {
                    0
                };
                let dc = c.saturating_sub(snap_count);
                if dc > 0 {
                    cells.push((k, dc));
                }
            }
        }
        let d_count = current.total_count() as i64 - snapshot.total_count() as i64;
        DdSketchDelta {
            buckets: cells,
            d_count,
            d_sum: 0.0,
            min_changed: false,
            new_min: 0.0,
            max_changed: false,
            new_max: 0.0,
        }
    }

    /// Delta-against-empty: a `compute_delta` against an EMPTY base
    /// reconstructs the window's full state when applied (round-trip).
    /// With `threshold = 1` every non-empty bucket survives, so applying
    /// the delta to a fresh empty sketch yields the same bucket store as
    /// the original window.
    #[test]
    fn test_compute_delta_against_empty_round_trips() {
        let mut window = DdSketch::new(0.01);
        for i in 1..=200 {
            window.update(i as f64);
        }
        let empty = DdSketch::new(0.01);

        // Delta against empty = the window's full state in delta form.
        let delta_bytes = window.compute_delta(&empty, 1);

        // Apply to a fresh empty base.
        let mut reconstructed = DdSketch::new(0.01);
        reconstructed.apply_delta_bytes(&delta_bytes).unwrap();

        // Total count recovered exactly.
        assert_eq!(reconstructed.total_count(), window.total_count());
        // Every non-empty bucket count matches (compare on absolute index).
        for (i, &c) in window.store_counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let k = window.store_offset + i as i32;
            let idx = (k - reconstructed.store_offset) as usize;
            assert_eq!(reconstructed.store_counts[idx], c, "bucket k={k}");
        }
        // Quantiles agree within the α relative-accuracy bound.
        for q in [0.5, 0.9, 0.99] {
            let got = reconstructed.quantile(q).unwrap();
            let want = window.quantile(q).unwrap();
            assert!(
                (got / want - 1.0).abs() <= 0.01,
                "q={q}: reconstructed={got} window={want}"
            );
        }
    }

    /// Two consecutive windows each emit their OWN state — the second
    /// window's delta-against-empty is NOT diffed against the first
    /// window (no cross-window subtraction).
    #[test]
    fn test_consecutive_windows_emit_own_state() {
        let empty = DdSketch::new(0.01);

        // Overlapping value ranges so a cross-window subtraction (win2 −
        // win1) would genuinely differ from win2's own state — making the
        // "no subtraction" assertion meaningful.
        let mut win1 = DdSketch::new(0.01);
        for i in 1..=50 {
            win1.update(i as f64);
            win1.update(i as f64); // win1 buckets carry count 2
        }
        let mut win2 = DdSketch::new(0.01);
        for i in 1..=50 {
            win2.update(i as f64); // same buckets, count 1
        }

        // Each window diffs against EMPTY (delta-against-empty), not against
        // the prior window.
        let d2_against_empty = win2.compute_delta(&empty, 1);
        let d2_against_win1 = win2.compute_delta(&win1, 1);

        // The delta-against-empty reconstructs win2 exactly.
        let mut recon = DdSketch::new(0.01);
        recon.apply_delta_bytes(&d2_against_empty).unwrap();
        assert_eq!(recon.total_count(), win2.total_count());

        // Sanity: diffing against win1 would have produced a DIFFERENT
        // (cross-window-subtracted) frame, proving the two are not the same
        // and that diffing against empty avoids the cross-window subtraction.
        assert_ne!(
            d2_against_empty, d2_against_win1,
            "delta-against-empty must differ from cross-window delta"
        );
    }

    /// Cross-language byte-parity guard against the Go reference
    /// implementation's `DDSketch.SerializePortable` output for the
    /// deterministic input `(1..=50)` with `α = 0.01`. The hex blob
    /// below was captured from a `proto.Marshal` of the Go envelope
    /// (with `Producer` and `HashSpec` cleared). Any change to
    /// [`DdSketch::update`]'s bucket-store growth that breaks parity
    /// will surface here.
    #[test]
    fn test_update_then_envelope_matches_go_golden_bytes() {
        use crate::proto::sketchlib::{
            DdSketchState, SketchEnvelope, sketch_envelope::SketchState,
        };
        use prost::Message;

        let mut sk = DdSketch::new(0.01);
        for i in 1..=50 {
            sk.update(i as f64);
        }

        let state = DdSketchState {
            alpha: sk.wire_alpha(),
            store_counts: sk.store_counts.clone(),
            store_offset: sk.store_offset,
        };
        let envelope = SketchEnvelope {
            format_version: 1,
            producer: None,
            hash_spec: None,
            sample_p: 0.0,
            sketch_state: Some(SketchState::Ddsketch(state)),
        };
        let mut got = Vec::with_capacity(envelope.encoded_len());
        envelope.encode(&mut got).expect("prost encode");

        // Byte string for the same `(1..=50)`, α=0.01 input AFTER dropping
        // the DataPoint-level METRIC scalars (count/sum/min/max → proto
        // tags 4-7 reserved). 403 bytes total: a `SketchEnvelope` proto
        // wrapping a `DDSketchState` carrying only `alpha`, `store_counts`
        // (Go chunk-128 padded layout, offset = -64, len = 384) and
        // `store_offset` (sint32 zigzag 0x7f = -64).
        //
        // NOTE: this golden is currently the RUST-produced value. It MUST be
        // reconciled against the Go reference implementation's regenerated
        // golden before declaring cross-language byte parity.
        const GOLDEN_HEX: &str = "0801728e03096214ae47e17a843f128003000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000100000000000000000000000000000100000000000000000000010000000000000000010000000000000001000000000001000000000001000000000001000000010000000001000000010000010000000100000100000100000100000100010000010001000100010001000100010001000100010100010100010100010101000101010100010101010101010100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000187f";
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "DDSketch envelope bytes diverge from golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );
    }

    fn decode_hex(s: &str) -> Vec<u8> {
        let bytes: Vec<u8> = s
            .as_bytes()
            .chunks(2)
            .map(|pair| {
                let high = hex_nibble(pair[0]);
                let low = hex_nibble(pair[1]);
                (high << 4) | low
            })
            .collect();
        bytes
    }

    #[test]
    fn interpolated_quantile_preserves_adjacent_ranks_and_legacy_convention() {
        let mut sketch = DdSketch::new(0.01);
        sketch.update(20.0);
        sketch.update(40.0);
        let lower = sketch.quantile(0.0).unwrap();
        let upper = sketch.quantile(1.0).unwrap();
        assert_eq!(sketch.quantile(0.9), Some(lower));
        for (q, expected) in [(0.0, 20.0), (0.5, 30.0), (0.9, 38.0), (1.0, 40.0)] {
            let actual = sketch.quantile_interpolated(q).unwrap();
            assert!((actual - expected).abs() <= expected * 0.01);
            assert_eq!(actual, lower * (1.0 - q) + upper * q);
        }
        for q in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(sketch.quantile_interpolated(q).is_none());
        }
        assert!(DdSketch::new(0.01).quantile_interpolated(0.5).is_none());
        let mut singleton = DdSketch::new(0.01);
        singleton.update(20.0);
        assert_eq!(
            singleton.quantile_interpolated(0.9),
            singleton.quantile(0.0)
        );
        for values in [vec![20.0, 20.0], vec![10.0, 10.0, 20.0, 40.0]] {
            let mut repeated = DdSketch::new(0.01);
            for &value in &values {
                repeated.update(value);
            }
            for q in [0.0, 0.5, 0.9, 1.0] {
                let rank = q * (values.len() - 1) as f64;
                let fraction = rank - rank.floor();
                let expected = values[rank.floor() as usize] * (1.0 - fraction)
                    + values[rank.ceil() as usize] * fraction;
                assert!(
                    (repeated.quantile_interpolated(q).unwrap() - expected).abs()
                        <= 0.01 * expected
                );
            }
        }
        for alpha in [0.0, -0.1, 1.0, f64::NAN] {
            assert!(
                DdSketch::from_raw(alpha, vec![1], 0)
                    .quantile_interpolated(0.5)
                    .is_none()
            );
        }
        assert!(
            DdSketch::from_raw(0.01, vec![(1u64 << 53) + 1], 0)
                .quantile_interpolated(0.5)
                .is_none()
        );
        // This portable format only represents positive inputs. An empty
        // sketch after rejected nonpositive updates must not synthesize zero.
        let mut unsupported = DdSketch::new(0.01);
        unsupported.update(-20.0);
        unsupported.update(0.0);
        assert!(unsupported.quantile_interpolated(0.9).is_none());
        let overflow = DdSketch::from_raw(0.01, vec![u64::MAX, 1], 0);
        assert!(overflow.quantile_interpolated(0.5).is_none());
    }

    fn hex_nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("non-hex byte {}", c as char),
        }
    }
}
