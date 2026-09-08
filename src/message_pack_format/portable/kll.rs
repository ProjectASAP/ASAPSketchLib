//! Wire-format-aligned KLL sketch type for cross-language interop.
//!
//! `KllSketch` is a thin facade over the pure-Rust [`crate::sketches::kll::KLL`]
//! (parity already locked in by the existing
//! `test_update_then_envelope_matches_sketchlib_go_bytes` in this file
//! against `sketchlib-go::KLLSketch.SerializePortable`). The wire shape
//! also serves as a nested field for
//! [`crate::message_pack_format::portable::hydra_kll`].

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error as MsgPackError, MessagePackCodec};
use crate::sketches::kll::KLL;

/// Concrete KLL type backing the wire-format `KllSketch`.
pub type SketchlibKll = KLL<f64>;

/// Creates a fresh sketchlib KLL sketch with the requested accuracy
/// parameter `k`.
pub fn new_sketchlib_kll(k: u16) -> SketchlibKll {
    KLL::init_kll(k as i32)
}

/// [`new_sketchlib_kll`] with an explicit compaction-coin seed, so the sketch
/// replays identically across runs.
pub fn new_sketchlib_kll_with_seed(k: u16, seed: u64) -> SketchlibKll {
    KLL::init_kll_with_seed(k as i32, seed)
}

/// Updates a sketchlib KLL with one numeric observation.
pub fn sketchlib_kll_update(inner: &mut SketchlibKll, value: f64) {
    inner.update(&value);
}

/// Queries a sketchlib KLL for the value at the requested quantile.
pub fn sketchlib_kll_quantile(inner: &SketchlibKll, q: f64) -> f64 {
    inner.quantile(q)
}

/// Merges `src` into `dst`.
pub fn sketchlib_kll_merge(dst: &mut SketchlibKll, src: &SketchlibKll) {
    dst.merge(src);
}

/// Serializes a sketchlib KLL into MessagePack bytes.
pub fn bytes_from_sketchlib_kll(inner: &SketchlibKll) -> Vec<u8> {
    inner.serialize_to_bytes().unwrap()
}

/// Deserializes a sketchlib KLL from MessagePack bytes.
pub fn sketchlib_kll_from_bytes(bytes: &[u8]) -> Result<SketchlibKll, Box<dyn std::error::Error>> {
    Ok(KLL::deserialize_from_bytes(bytes)?)
}

/// Wire-format KLL sketch — a thin facade over [`SketchlibKll`] with a
/// fixed `k` parameter and msgpack codec via [`KllSketchData`].
pub struct KllSketch {
    pub k: u16,
    pub(crate) backend: SketchlibKll,
}

impl KllSketch {
    pub fn new(k: u16) -> Self {
        Self {
            k,
            backend: new_sketchlib_kll(k),
        }
    }

    /// [`KllSketch::new`] with an explicit compaction-coin seed.
    ///
    /// `new` seeds the coin from the wall clock, so two runs over the same
    /// input produce different sketches. Pass a seed when the result has to be
    /// reproducible — replaying a stream, comparing two processes, or
    /// asserting an accuracy bound in a test.
    pub fn with_seed(k: u16, seed: u64) -> Self {
        Self {
            k,
            backend: new_sketchlib_kll_with_seed(k, seed),
        }
    }

    /// The configured `k` parameter (compactor capacity).
    pub fn k(&self) -> u16 {
        self.k
    }

    /// Returns the raw sketch bytes (for JSON serialization, etc.).
    pub fn sketch_bytes(&self) -> Vec<u8> {
        bytes_from_sketchlib_kll(&self.backend)
    }

    pub fn update(&mut self, value: f64) {
        sketchlib_kll_update(&mut self.backend, value);
    }

    pub fn count(&self) -> u64 {
        self.backend.count() as u64
    }

    /// Estimate the value at the given quantile `q ∈ [0, 1]`.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.count() == 0 {
            return 0.0;
        }
        sketchlib_kll_quantile(&self.backend, q)
    }

    /// Merge another KllSketch into self in place. Both operands must
    /// have identical `k`.
    pub fn merge(
        &mut self,
        other: &KllSketch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.k != other.k {
            return Err(format!("KllSketch k mismatch: self={}, other={}", self.k, other.k).into());
        }
        sketchlib_kll_merge(&mut self.backend, &other.backend);
        Ok(())
    }

    /// Merge from references without cloning.
    pub fn merge_refs(
        sketches: &[&Self],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if sketches.is_empty() {
            return Err("No sketches to merge".into());
        }
        let k = sketches[0].k;
        for s in sketches {
            if s.k != k {
                return Err("Cannot merge KllSketch with different k values".into());
            }
        }
        let mut merged = Self::new(k);
        for s in sketches {
            sketchlib_kll_merge(&mut merged.backend, &s.backend);
        }
        Ok(merged)
    }

    /// One-shot aggregation: build a sketch from a slice of values.
    pub fn aggregate_kll(k: u16, values: &[f64]) -> Option<Vec<u8>> {
        if values.is_empty() {
            return None;
        }
        let mut sketch = Self::new(k);
        for &value in values {
            sketch.update(value);
        }
        sketch.to_msgpack().ok()
    }
}

impl Clone for KllSketch {
    fn clone(&self) -> Self {
        // Structural clone: the backing `KLL<f64>` derives `Clone` (its fields
        // are `Box<[..]>`/`Vec`/scalars), so copy it directly instead of a full
        // msgpack serialize -> deserialize -> rebuild round-trip. Produces an
        // identical sketch far more cheaply (no rmp encode/decode, far fewer
        // allocations).
        Self {
            k: self.k,
            backend: self.backend.clone(),
        }
    }
}

impl KllSketch {
    /// Reconstruct directly from portable wire state (k + level-ordered items +
    /// level boundaries) without replaying items through `update()`. Bit-exact.
    pub fn from_portable_state(
        k: u16,
        items: &[f64],
        levels: &[usize],
        num_levels: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            k,
            backend: SketchlibKll::from_portable_state(k as usize, items, levels, num_levels)?,
        })
    }
}

impl std::fmt::Debug for KllSketch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KllSketch")
            .field("k", &self.k)
            .field("sketch_n", &self.count())
            .finish()
    }
}

// Thread safety: sketchlib's KLL is used in a single-threaded context per
// accumulator instance and only shares read-only operations across threads.
unsafe impl Send for KllSketch {}
unsafe impl Sync for KllSketch {}

// ----- Value-offset fixed-point representation (cross-language) -----
//
// KLL `KLLState` (proto/kll/kll.proto) carries the retained samples in one of
// two forms:
//
//   * RAW F64       — `items` (field 5), 8 bytes/sample. The original v1 form.
//   * VALUE-OFFSET  — `offset` (7) + `value_scale` (8) + `residuals` (9),
//                     where `value = residual * 10^value_scale + offset`.
//
// PR2 makes BOTH languages read both forms (this is the decode-first half of
// the rollout). `sketchlib-go` gates emission of the value-offset form behind
// an exactness guard; this crate currently only reads it but the
// encode/decode helpers below are symmetric so the Rust side can produce it
// too (used by the cross-language golden test and any future Rust producer).

/// Per-sketch scale exponents tried by [`encode_value_offset`], in priority
/// order. Mirrors `kllScaleSweep` in `sketchlib-go`.
pub const KLL_SCALE_SWEEP: [i32; 7] = [0, -1, -2, -3, -4, -5, -6];

/// Reconstructs retained samples from the value-offset fixed-point fields:
/// `value = residual * 10^value_scale + offset`. Caller invokes this only when
/// `residuals` is non-empty.
pub fn decode_value_offset(offset: f64, value_scale: i32, residuals: &[i64]) -> Vec<f64> {
    let mul = 10f64.powi(value_scale);
    residuals.iter().map(|&r| r as f64 * mul + offset).collect()
}

/// Attempts to represent `items` exactly as `residual * 10^scale + offset`
/// using the finite [`KLL_SCALE_SWEEP`]. `offset` is the minimum value. Returns
/// `None` (the exactness-guard fallback) when no candidate scale round-trips
/// every sample bit-exactly, or when the slice is empty / contains a non-finite
/// value. Symmetric with `sketchlib-go::encodeValueOffset`.
pub fn encode_value_offset(items: &[f64]) -> Option<(f64, i32, Vec<i64>)> {
    if items.is_empty() {
        return None;
    }
    let mut offset = items[0];
    for &v in items {
        if !v.is_finite() {
            return None;
        }
        if v < offset {
            offset = v;
        }
    }
    for &sc in &KLL_SCALE_SWEEP {
        let mul = 10f64.powi(-sc);
        let mut out = Vec::with_capacity(items.len());
        let mut good = true;
        for &v in items {
            let r = ((v - offset) * mul).round();
            if !r.is_finite() || r > i64::MAX as f64 || r < i64::MIN as f64 {
                good = false;
                break;
            }
            let ri = r as i64;
            // Exactness guard: reconstruct and require bit-identical recovery.
            let recon = ri as f64 * 10f64.powi(sc) + offset;
            if recon != v {
                good = false;
                break;
            }
            out.push(ri);
        }
        if good {
            return Some((offset, sc, out));
        }
    }
    None
}

/// Materialized retained samples + level layout decoded from a proto
/// `KLLState`, dual-reading the raw-f64 and value-offset forms.
#[derive(Debug, Clone)]
pub struct KllProtoItems {
    /// Retained samples in level order (length == `levels[num_levels]`).
    pub items: Vec<f64>,
    /// Level boundary indices, length == `num_levels + 1`.
    pub levels: Vec<usize>,
    pub num_levels: usize,
    pub k: u32,
    pub m: u32,
}

impl KllProtoItems {
    /// Decodes a proto `KLLState`, dual-reading: when `residuals` is non-empty
    /// it reconstructs from (offset, value_scale, residuals); otherwise it
    /// reads raw f64 `items`. Returns an error if both forms are populated or
    /// the level layout is inconsistent.
    pub fn from_state(
        s: &crate::proto::sketchlib::KllState,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let items = if !s.residuals.is_empty() {
            if !s.items.is_empty() {
                return Err("kll: both residuals and raw items[] set".into());
            }
            decode_value_offset(s.offset, s.value_scale, &s.residuals)
        } else {
            s.items.clone()
        };
        let levels: Vec<usize> = s.levels.iter().map(|&v| v as usize).collect();
        if levels.len() < 2 {
            return Err(format!("kll: invalid levels length {}", levels.len()).into());
        }
        if *levels.last().unwrap() != items.len() {
            return Err("kll: invalid item layout".into());
        }
        Ok(Self {
            items,
            levels,
            num_levels: s.num_levels as usize,
            k: s.k,
            m: s.m,
        })
    }

    /// Weighted (value, weight) pairs across all levels.
    pub fn weighted_samples(&self) -> Vec<(f64, u64)> {
        let mut out = Vec::with_capacity(self.items.len());
        for h in 0..self.num_levels {
            let weight: u64 = 1 << h;
            let idx = self.num_levels - 1 - h;
            if idx + 1 >= self.levels.len() {
                continue;
            }
            let start = self.levels[idx];
            let end = self.levels[idx + 1];
            for &v in &self.items[start..end] {
                out.push((v, weight));
            }
        }
        out
    }

    /// Estimated value at quantile `q ∈ [0, 1]` from the weighted samples.
    pub fn quantile(&self, q: f64) -> f64 {
        let mut samples = self.weighted_samples();
        if samples.is_empty() {
            return 0.0;
        }
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let total: u64 = samples.iter().map(|(_, w)| w).sum();
        let target = (q * total as f64).ceil() as u64;
        let mut acc = 0u64;
        for (v, w) in &samples {
            acc += w;
            if acc >= target {
                return *v;
            }
        }
        samples.last().unwrap().0
    }
}

// ----- Wire format -----

/// Wire DTO for [`KllSketch`]. Also referenced as a nested field by
/// [`crate::message_pack_format::portable::hydra_kll::HydraKllSketchWire`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KllSketchData {
    pub k: u16,
    pub sketch_bytes: Vec<u8>,
}

impl MessagePackCodec for KllSketch {
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let wire = KllSketchData {
            k: self.k,
            sketch_bytes: self.sketch_bytes(),
        };
        Ok(rmp_serde::to_vec(&wire)?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        let wire: KllSketchData = rmp_serde::from_slice(bytes)?;
        let backend = KLL::deserialize_from_bytes(&wire.sketch_bytes)?;
        Ok(Self { k: wire.k, backend })
    }
}

impl MessagePackCodec for KllSketchData {
    /// Encodes the `(k, sketch_bytes)` DTO directly, byte-identical to
    /// [`KllSketch::to_msgpack`] (which serializes this same struct). Lets
    /// producers that hold a raw sketchlib KLL backend (via
    /// `serialize_to_bytes`) emit the portable KLL msgpack form without
    /// constructing a [`KllSketch`] facade.
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        Ok(rmp_serde::to_vec(self)?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        Ok(rmp_serde::from_slice(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kll_data_msgpack_matches_kll_sketch() {
        let mut sk = KllSketch::new(200);
        for i in 0..500 {
            sk.update(i as f64);
        }
        let via_sketch = sk.to_msgpack().unwrap();
        let data = KllSketchData {
            k: sk.k(),
            sketch_bytes: sk.sketch_bytes(),
        };
        let via_data = data.to_msgpack().unwrap();
        assert_eq!(
            via_sketch, via_data,
            "KllSketchData codec must byte-match KllSketch"
        );
        let rt = KllSketchData::from_msgpack(&via_data).unwrap();
        assert_eq!(rt.k, data.k);
        assert_eq!(rt.sketch_bytes, data.sketch_bytes);
    }

    #[test]
    fn test_kll_creation() {
        let kll = KllSketch::new(200);
        assert_eq!(kll.count(), 0);
        assert_eq!(kll.k, 200);
    }

    #[test]
    fn test_kll_update() {
        let mut kll = KllSketch::new(200);
        kll.update(10.0);
        kll.update(20.0);
        kll.update(15.0);
        assert_eq!(kll.count(), 3);
    }

    #[test]
    fn test_kll_quantile() {
        let mut kll = KllSketch::new(200);
        for i in 1..=10 {
            kll.update(i as f64);
        }
        assert_eq!(kll.quantile(0.0), 1.0);
        assert_eq!(kll.quantile(1.0), 10.0);
        let median = kll.quantile(0.5);
        assert!(
            (5.0..=6.0).contains(&median),
            "median should be between 5 and 6; got {median}"
        );
    }

    #[test]
    fn test_kll_merge() {
        let mut kll1 = KllSketch::new(200);
        let mut kll2 = KllSketch::new(200);

        for i in 1..=5 {
            kll1.update(i as f64);
        }
        for i in 6..=10 {
            kll2.update(i as f64);
        }

        kll1.merge(&kll2).unwrap();
        assert_eq!(kll1.count(), 10);
        assert_eq!(kll1.quantile(0.0), 1.0);
        assert_eq!(kll1.quantile(1.0), 10.0);
    }

    #[test]
    fn test_msgpack_round_trip() {
        let mut kll = KllSketch::new(200);
        for i in 1..=5 {
            kll.update(i as f64);
        }

        let bytes = kll.to_msgpack().unwrap();
        let deserialized = KllSketch::from_msgpack(&bytes).unwrap();

        assert_eq!(deserialized.k, 200);
        assert_eq!(deserialized.count(), 5);
        assert_eq!(deserialized.quantile(0.0), 1.0);
        assert_eq!(deserialized.quantile(1.0), 5.0);
    }

    #[test]
    fn test_aggregate_kll() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let bytes = KllSketch::aggregate_kll(200, &values).unwrap();
        let kll = KllSketch::from_msgpack(&bytes).unwrap();
        assert_eq!(kll.count(), 5);
        assert_eq!(kll.quantile(0.0), 1.0);
        assert_eq!(kll.quantile(1.0), 5.0);
    }

    #[test]
    fn test_aggregate_kll_empty() {
        assert!(KllSketch::aggregate_kll(200, &[]).is_none());
    }

    /// Cross-language byte-parity guard against `sketchlib-go`'s
    /// `KLLSketch.SerializePortable` output for the deterministic
    /// input `(1..=50)` with `seed=42` and `k=200`.
    #[test]
    fn test_update_then_envelope_matches_sketchlib_go_bytes() {
        use crate::proto::sketchlib::{
            CoinState, KllState, SketchEnvelope, sketch_envelope::SketchState,
        };
        use prost::Message;

        let mut sk: KLL<f64> = KLL::init_kll_with_seed(200, 42);
        for i in 1..=50 {
            sk.update(&(i as f64));
        }

        let (state, bit_cache, remaining_bits) = sk.wire_coin();
        let kll_state = KllState {
            k: sk.wire_k(),
            m: sk.wire_m(),
            num_levels: sk.wire_num_levels(),
            levels: sk.wire_levels(),
            // This golden pins the RAW-F64 wire form; the value-offset fields
            // stay at their proto defaults so the bytes are unchanged.
            items: sk.wire_items(),
            coin: Some(CoinState {
                state,
                bit_cache,
                remaining_bits,
            }),
            offset: 0.0,
            value_scale: 0,
            residuals: vec![],
        };
        let envelope = SketchEnvelope {
            format_version: 1,
            producer: None,
            hash_spec: None,
            sample_p: 0.0,
            sketch_state: Some(SketchState::Kll(kll_state)),
        };
        let mut got = Vec::with_capacity(envelope.encoded_len());
        envelope.encode(&mut got).expect("prost encode");

        const GOLDEN_HEX: &str = "08016aa20308c80110081801220200322a9003000000000000f03f000000000000004000000000000008400000000000001040000000000000144000000000000018400000000000001c40000000000000204000000000000022400000000000002440000000000000264000000000000028400000000000002a400000000000002c400000000000002e4000000000000030400000000000003140000000000000324000000000000033400000000000003440000000000000354000000000000036400000000000003740000000000000384000000000000039400000000000003a400000000000003b400000000000003c400000000000003d400000000000003e400000000000003f4000000000000040400000000000804040000000000000414000000000008041400000000000004240000000000080424000000000000043400000000000804340000000000000444000000000008044400000000000004540000000000080454000000000000046400000000000804640000000000000474000000000008047400000000000004840000000000080484000000000000049403202082a";
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "KLL envelope bytes diverge from sketchlib-go golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );
    }

    fn decode_hex(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|pair| {
                let high = hex_nibble(pair[0]);
                let low = hex_nibble(pair[1]);
                (high << 4) | low
            })
            .collect()
    }

    fn hex_nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("non-hex byte {}", c as char),
        }
    }

    // ----- Value-offset fixed-point representation -----

    use crate::proto::sketchlib::{KllState, SketchEnvelope, sketch_envelope::SketchState};
    use prost::Message;

    /// Round-trip exactness for the value-offset (fixed-point) form: integer
    /// and fixed-decimal series must encode then decode back bit-exactly.
    #[test]
    fn value_offset_round_trip_exact_fixed_point() {
        // Integer-ish series (scale 0).
        let ints: Vec<f64> = (0..200).map(|i| (1_000_000 + i * 7) as f64).collect();
        let (offset, scale, residuals) =
            encode_value_offset(&ints).expect("integer series must be representable");
        assert_eq!(scale, 0, "integer series should pick scale 0");
        let back = decode_value_offset(offset, scale, &residuals);
        assert_eq!(ints, back, "fixed-point round-trip changed integer items");

        // Fixed-decimal series (milli resolution → scale -3).
        let decis: Vec<f64> = (0..128).map(|i| 100.0 + (i as f64) * 0.001).collect();
        let (o2, s2, r2) =
            encode_value_offset(&decis).expect("milli-decimal series must be representable");
        assert!(
            (-3..=0).contains(&s2),
            "expected a small negative scale, got {s2}"
        );
        let back2 = decode_value_offset(o2, s2, &r2);
        assert_eq!(decis, back2, "fixed-point round-trip changed decimal items");
    }

    /// The exactness guard must reject a value not representable at any
    /// candidate scale, so the caller falls back to raw f64.
    #[test]
    fn value_offset_guard_rejects_irrational() {
        let items = vec![0.0, 1.0, std::f64::consts::PI, 3.0];
        assert!(
            encode_value_offset(&items).is_none(),
            "pi is not exactly representable at any decimal scale; guard must reject"
        );
        // Empty slice falls back too.
        assert!(encode_value_offset(&[]).is_none());
        // Non-finite values are rejected.
        assert!(encode_value_offset(&[1.0, f64::NAN]).is_none());
        assert!(encode_value_offset(&[1.0, f64::INFINITY]).is_none());
    }

    /// `KllProtoItems::from_state` must dual-read both the raw-f64 and the
    /// value-offset forms and yield identical items + quantiles.
    #[test]
    fn proto_dual_read_round_trip_both_forms() {
        let mut sk: KLL<f64> = KLL::init_kll_with_seed(200, 7);
        for i in 1..=5000 {
            sk.update(&(i as f64));
        }
        let raw_items = sk.wire_items();
        let levels = sk.wire_levels();
        let num_levels = sk.wire_num_levels();

        // Raw-f64 form.
        let raw_state = KllState {
            k: sk.wire_k(),
            m: sk.wire_m(),
            num_levels,
            levels: levels.clone(),
            items: raw_items.clone(),
            coin: None,
            offset: 0.0,
            value_scale: 0,
            residuals: vec![],
        };
        let from_raw = KllProtoItems::from_state(&raw_state).expect("decode raw-f64");

        // Value-offset form (items empty, residuals populated).
        let (offset, scale, residuals) =
            encode_value_offset(&raw_items).expect("integer items representable");
        let fp_state = KllState {
            k: sk.wire_k(),
            m: sk.wire_m(),
            num_levels,
            levels,
            items: vec![],
            coin: None,
            offset,
            value_scale: scale,
            residuals,
        };
        let from_fp = KllProtoItems::from_state(&fp_state).expect("decode value-offset");

        assert_eq!(
            from_raw.items, from_fp.items,
            "raw-f64 and value-offset decode to different items"
        );
        for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert_eq!(
                from_raw.quantile(q),
                from_fp.quantile(q),
                "quantile mismatch at p={q} between the two decode paths"
            );
        }
    }

    /// Decoding must error if a malformed state sets BOTH raw items and
    /// residuals.
    #[test]
    fn proto_rejects_both_forms_populated() {
        let state = KllState {
            k: 200,
            m: 8,
            num_levels: 1,
            levels: vec![0, 2],
            items: vec![1.0, 2.0],
            coin: None,
            offset: 0.0,
            value_scale: 0,
            residuals: vec![1, 2],
        };
        assert!(KllProtoItems::from_state(&state).is_err());
    }

    /// Cross-language golden: decode the exact envelope bytes that
    /// `sketchlib-go` emits for the deterministic `(1..=50)` integer input via
    /// its value-offset encoder, and confirm the retained samples + quantiles
    /// match. The Go side produces this with `SerializePortable` (fixed-point,
    /// scale 0, offset = 1.0). See `sketches/KLL/portable_test.go::
    /// TestGoldenValueOffsetEnvelopeForRust` in sketchlib-go, which prints this
    /// hex and asserts the identical bytes.
    #[test]
    fn golden_value_offset_envelope_from_go() {
        // SketchEnvelope { format_version: 1, kll: KLLState { k: 200, m: 8,
        //   num_levels: 1, levels: [0, 50], offset: 1.0, value_scale: 0,
        //   residuals: [0..=49] } }, producer/hash_spec cleared.
        const GOLDEN_HEX: &str = GO_VALUE_OFFSET_GOLDEN_HEX;
        let bytes = decode_hex(GOLDEN_HEX);
        let env = SketchEnvelope::decode(bytes.as_slice()).expect("decode envelope");
        let state = match env.sketch_state {
            Some(SketchState::Kll(s)) => s,
            _ => panic!("envelope did not contain a KLLState"),
        };
        assert!(
            state.items.is_empty(),
            "value-offset form must leave items[] empty"
        );
        assert_eq!(state.value_scale, 0);
        assert_eq!(state.offset, 1.0);
        assert_eq!(state.residuals.len(), 50);

        let decoded = KllProtoItems::from_state(&state).expect("dual-read decode");
        let expected: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        assert_eq!(decoded.items, expected, "value-offset decode != (1..=50)");
        // Quantile sanity: with k=200 and 50 inputs, no compaction occurred so
        // these are exact order statistics.
        assert_eq!(decoded.quantile(0.0), 1.0);
        assert_eq!(decoded.quantile(1.0), 50.0);
    }

    /// Reverse-direction cross-language golden: build the value-offset envelope
    /// for (1..=50) with the Rust encoder and assert exact bytes. The hex
    /// printed here is consumed by sketchlib-go `TestRustGoldenDecodesInGo`.
    /// Run with `-- --nocapture` to (re)print when regenerating.
    #[test]
    fn golden_value_offset_envelope_for_go() {
        let items: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let (offset, value_scale, residuals) =
            encode_value_offset(&items).expect("integers representable");
        assert_eq!(offset, 1.0);
        assert_eq!(value_scale, 0);

        let state = KllState {
            k: 200,
            m: 8,
            num_levels: 1,
            levels: vec![0, 50],
            items: vec![],
            // No coin: the reverse-decode in Go only checks the retained set,
            // and an absent coin keeps the fixture minimal/stable.
            coin: None,
            offset,
            value_scale,
            residuals,
        };
        let envelope = SketchEnvelope {
            format_version: 1,
            producer: None,
            hash_spec: None,
            sample_p: 0.0,
            sketch_state: Some(SketchState::Kll(state)),
        };
        let mut got = Vec::with_capacity(envelope.encoded_len());
        envelope.encode(&mut got).expect("prost encode");
        let got_hex: String = got.iter().map(|b| format!("{b:02x}")).collect();
        println!(
            "RUST_VALUE_OFFSET_GOLDEN_HEX ({} bytes):\n{got_hex}",
            got.len()
        );

        const RUST_GOLDEN_HEX: &str = "08016a4808c801100818012202003239000000000000f03f4a3200020406080a0c0e10121416181a1c1e20222426282a2c2e30323436383a3c3e40424446484a4c4e50525456585a5c5e6062";
        assert_eq!(
            got_hex, RUST_GOLDEN_HEX,
            "Rust value-offset envelope bytes drifted; update both this constant \
             and sketchlib-go's RustValueOffsetGoldenHex"
        );

        // Self-consistency: our own dual-reader recovers (1..=50).
        let decoded = KllProtoItems::from_state(match &envelope.sketch_state {
            Some(SketchState::Kll(s)) => s,
            _ => unreachable!(),
        })
        .expect("decode");
        assert_eq!(decoded.items, items);
    }

    // Captured from sketchlib-go's value-offset `SerializePortable` for input
    // (1..=50), k=200, producer/hash_spec cleared. Generated and asserted by
    // sketchlib-go `TestGoldenValueOffsetEnvelopeForRust` (run with `-v` to
    // re-print). 80 bytes: SketchEnvelope{ format_version:1, kll: KLLState{
    // k:200, m:8, num_levels:1, levels:[0,50], coin:{state:42}, offset:1.0,
    // value_scale:0, residuals:[0..=49] } }.
    const GO_VALUE_OFFSET_GOLDEN_HEX: &str = "08016a4c08c80110081801220200323202082a39000000000000f03f4a3200020406080a0c0e10121416181a1c1e20222426282a2c2e30323436383a3c3e40424446484a4c4e50525456585a5c5e6062";
}
