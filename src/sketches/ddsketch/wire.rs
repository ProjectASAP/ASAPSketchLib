//! ASAPv1 wire serialization for DDSketch.
//!
//! Child submodule of [`crate::sketches::ddsketch`]: it holds ALL of DDSketch's
//! ASAPv1 serialization (the metadata/payload DTOs, the kind_id constant, and
//! the `serialize_to_bytes` / `deserialize_from_bytes` impls) while the
//! algorithm lives in the parent module file. Being a descendant module, it
//! reads the sketch's private `store` / `count` / `sum` / `min` / `max` fields
//! directly without widening any field visibility. See
//! `docs/asapv1_wire_format.md` §3.
//!
//! ## DDSketch metadata has no hash spec
//!
//! DDSketch never hashes its inputs — it maps a value to a bucket index with
//! `floor(ln(v) / ln(gamma))`, and carries no hasher type parameter at all. The
//! hash-spec group has no truthful value here, so the metadata is structural
//! params only (`metadata_version`, `alpha`). This is the KLL precedent (Q-KLL).
//!
//! ## Relative accuracy
//!
//! `alpha` is the one construction parameter, so it lives in the metadata. The
//! `gamma` / `log_gamma` / `inv_log_gamma` triple is derived from it
//! (`gamma = (1 + alpha) / (1 - alpha)`) and never reaches the wire.
//!
//! ## One positive-range store
//!
//! DDSketch is defined for positive reals: `add` drops non-positive,
//! non-finite, and non-indexable values. There is no negative-range store and
//! no zero-count bucket, so the payload has no field for either. Bucket
//! *indices* are still signed — a value below `1.0` maps to a negative index —
//! and `offset` carries that as a msgpack int.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::common::structures::Vector1D;
use crate::message_pack_format::envelope;

use super::{Buckets, DDSketch};

/// DDSketch kind_id: family `0x05`, single algorithm variant `0x00`.
const DD_KIND: &[u8] = &[0x05, 0x00];

/// DDSketch descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it). There is no hash-spec group;
/// `alpha` is the sketch's one construction parameter and so, per the spec's
/// config→metadata rule, its only structural param.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DdMetadata {
    metadata_version: u8,
    alpha: f64,
}

/// Builds the DDSketch descriptor metadata for a relative accuracy `alpha`.
fn dd_metadata(alpha: f64) -> DdMetadata {
    DdMetadata {
        metadata_version: 1,
        alpha,
    }
}

/// DDSketch payload (ASAPv1 §3), a msgpack **array** (`to_vec`, positional):
/// `[counts, offset, sum, min, max]`. `counts` is the dense bucket store,
/// carried verbatim including the growth padding, so a decoded sketch
/// re-serializes byte-identically; the bucket index of `counts[i]` is
/// `offset + i`. `sum` / `min` / `max` are the sketch's running scalars, which
/// the bucket counts do not determine: exact over the values passed to `add`,
/// and the bucket representative for a bucket promoted through `apply_delta`.
/// The total sample count is the sum of `counts` and so is not carried.
#[derive(Debug, Serialize, Deserialize)]
struct DdPayload {
    counts: Vec<u64>,
    offset: i32,
    sum: f64,
    min: f64,
    max: f64,
}

/// Checks a relative accuracy against the domain [`DDSketch::new`] accepts:
/// finite and strictly inside `(0, 1)`. Outside it the `gamma` derivation is
/// non-positive or infinite and every bucket index is meaningless.
pub(super) fn check_alpha(alpha: f64) -> Result<(), String> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(format!("DDSketch alpha {alpha} is not in (0, 1)"));
    }
    Ok(())
}

/// Checks a bucket store's index span: an empty store sits at offset `0`, and
/// a populated one's highest index `offset + len - 1` is representable as an
/// `i32`. Applied before the store is rebuilt from the declared offset.
pub(super) fn check_store_span(offset: i32, len: usize) -> Result<(), String> {
    if len == 0 {
        if offset != 0 {
            return Err(format!(
                "DDSketch empty store must be at offset 0, got {offset}"
            ));
        }
        return Ok(());
    }
    let highest = i64::from(offset) + len as i64 - 1;
    if highest > i64::from(i32::MAX) {
        return Err(format!(
            "DDSketch store span past i32: offset={offset}, len={len}"
        ));
    }
    Ok(())
}

/// Total sample count, the checked sum of every bucket. `None` on overflow.
pub(super) fn total_count(counts: &[u64]) -> Option<u64> {
    counts.iter().try_fold(0u64, |acc, &c| acc.checked_add(c))
}

/// Checks the running scalars against the store's total count. An empty sketch
/// carries `0.0` / `+inf` / `-inf`, the state [`DDSketch::new`] starts in. A
/// populated one carries finite scalars with `0 < min <= max`, and `sum >= min`
/// since every ingested value is at least `min`.
pub(super) fn check_scalars(count: u64, sum: f64, min: f64, max: f64) -> Result<(), String> {
    if count == 0 {
        if sum == 0.0 && min == f64::INFINITY && max == f64::NEG_INFINITY {
            return Ok(());
        }
        return Err(format!(
            "DDSketch empty store must carry sum=0, min=inf, max=-inf, got {sum}, {min}, {max}"
        ));
    }
    if !(sum.is_finite() && min.is_finite() && max.is_finite()) {
        return Err(format!(
            "DDSketch scalars must be finite for a populated store: {sum}, {min}, {max}"
        ));
    }
    if !(min > 0.0 && min <= max && sum >= min) {
        return Err(format!(
            "DDSketch scalars out of order: sum={sum}, min={min}, max={max}"
        ));
    }
    Ok(())
}

// Wire serialization for DDSketch. `wire` is a descendant of the sketch module,
// so these impls read the private fields and construct the struct directly.
impl DDSketch {
    /// Serializes the sketch into an ASAPv1 MessagePack envelope
    /// (kind_id `0x05 0x00`). `alpha` lands in the metadata; the payload is the
    /// bucket store plus the three scalars the buckets do not determine.
    ///
    /// A sketch whose state the decoder would refuse — an out-of-range `alpha`,
    /// a store spanning past `i32`, or bucket counts that overflow `u64` — is an
    /// error rather than bytes that would be refused on decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let fail = |e: String| RmpEncodeError::Syntax(format!("ASAPv1 DDSketch envelope: {e}"));
        check_alpha(self.alpha).map_err(fail)?;
        let counts = self.store.counts.as_slice();
        check_store_span(self.store.offset, counts.len()).map_err(fail)?;
        let count = total_count(counts)
            .ok_or_else(|| fail("bucket counts overflow the total sample count".to_string()))?;
        check_scalars(count, self.sum, self.min, self.max).map_err(fail)?;

        let metadata = rmp_serde::to_vec_named(&dd_metadata(self.alpha))?;
        let payload = rmp_serde::to_vec(&DdPayload {
            counts: counts.to_vec(),
            offset: self.store.offset,
            sum: self.sum,
            min: self.min,
            max: self.max,
        })?;
        Ok(envelope::encode(DD_KIND, &metadata, &payload))
    }

    /// Deserializes a DDSketch from an ASAPv1 MessagePack envelope. The index
    /// mapping is rebuilt from the metadata `alpha` and the total sample count
    /// is summed from the buckets. Every inconsistency fails closed.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != DD_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "DDSketch kind_id mismatch: stored {kind_id:?}, expected {DD_KIND:?}"
            )));
        }
        let meta: DdMetadata = from_slice(metadata)?;
        // `alpha` is a property of the stored sketch rather than of the target
        // type, so it is echoed back into the expected block and bounded by
        // range instead of being pinned.
        if meta != dd_metadata(meta.alpha) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 DDSketch envelope: metadata mismatch".to_string(),
            ));
        }
        check_alpha(meta.alpha).map_err(RmpDecodeError::Uncategorized)?;

        let p: DdPayload = from_slice(payload)?;
        check_store_span(p.offset, p.counts.len()).map_err(RmpDecodeError::Uncategorized)?;
        let count = total_count(&p.counts).ok_or_else(|| {
            RmpDecodeError::Uncategorized(
                "DDSketch bucket counts overflow the total sample count".to_string(),
            )
        })?;
        check_scalars(count, p.sum, p.min, p.max).map_err(RmpDecodeError::Uncategorized)?;

        let alpha = meta.alpha;
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        let log_gamma = gamma.ln();
        Ok(DDSketch {
            alpha,
            gamma,
            log_gamma,
            inv_log_gamma: 1.0 / log_gamma,
            store: Buckets {
                counts: Vector1D::from_vec(p.counts),
                offset: p.offset,
            },
            count,
            sum: p.sum,
            min: p.min,
            max: p.max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALPHA: f64 = 0.01;

    fn populated() -> DDSketch {
        let mut sketch = DDSketch::new(ALPHA);
        for v in [0.25f64, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 1000.0] {
            sketch.add(&v);
        }
        sketch
    }

    /// Builds a real envelope with real metadata around a hand-built payload,
    /// so the crafted-bytes tests exercise the decoder's own rules rather than
    /// the envelope's framing.
    fn crafted(alpha: f64, counts: Vec<u64>, offset: i32, sum: f64, min: f64, max: f64) -> Vec<u8> {
        let metadata = rmp_serde::to_vec_named(&dd_metadata(alpha)).unwrap();
        let payload = rmp_serde::to_vec(&DdPayload {
            counts,
            offset,
            sum,
            min,
            max,
        })
        .unwrap();
        envelope::encode(DD_KIND, &metadata, &payload)
    }

    /// The message a rejected decode fails with, so a test can pin *which*
    /// rule fired rather than settling for any error at all.
    fn decode_error(bytes: &[u8]) -> String {
        DDSketch::deserialize_from_bytes(bytes)
            .expect_err("decode must fail")
            .to_string()
    }

    #[test]
    fn ddsketch_envelope_structure_and_round_trip() {
        let sketch = populated();
        let bytes = sketch.serialize_to_bytes().expect("serialize");

        assert!(bytes.starts_with(b"ASAPv1"));
        assert_eq!(&bytes[7..10], &[2u8, 0x05, 0x00]); // kind_id_len=2, kind_id=[0x05,0x00]
        let (kind_id, metadata, _) = envelope::split(&bytes).expect("split");
        assert_eq!(kind_id, &[0x05, 0x00]);
        let meta: DdMetadata = from_slice(metadata).expect("metadata");
        assert_eq!(meta.metadata_version, 1);
        assert_eq!(meta.alpha, ALPHA);

        let decoded = DDSketch::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.store_counts(), sketch.store_counts());
        assert_eq!(decoded.store_offset(), sketch.store_offset());
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            bytes,
            "DDSketch serialized bytes differed after round trip"
        );
        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_eq!(
                decoded.get_value_at_quantile(q),
                sketch.get_value_at_quantile(q)
            );
        }
    }

    /// `sum` / `min` / `max` are not derivable from the buckets, so they are
    /// carried and come back exactly — not as the α-bounded bucket
    /// representatives a recomputation would produce.
    #[test]
    fn ddsketch_scalars_survive_exactly() {
        let sketch = populated();
        let decoded =
            DDSketch::deserialize_from_bytes(&sketch.serialize_to_bytes().expect("serialize"))
                .expect("decode");
        assert_eq!(decoded.sum(), sketch.sum());
        assert_eq!(decoded.min(), Some(0.25));
        assert_eq!(decoded.max(), Some(1000.0));
        assert_eq!(decoded.alpha(), sketch.alpha());
    }

    /// The total sample count is the sum of the buckets, so the payload carries
    /// no `count` field: it is a 5-element array and nothing more.
    #[test]
    fn ddsketch_count_is_recovered_from_the_buckets() {
        let sketch = populated();
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        let (_, _, payload) = envelope::split(&bytes).expect("split");

        let five: (Vec<u64>, i32, f64, f64, f64) = from_slice(payload).expect("5-element payload");
        assert_eq!(five.0.iter().sum::<u64>(), sketch.get_count());
        assert!(
            from_slice::<(Vec<u64>, i32, f64, f64, f64, u64)>(payload).is_err(),
            "the payload must not carry a sixth element"
        );

        let decoded = DDSketch::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.get_count(), sketch.get_count());
    }

    #[test]
    fn ddsketch_empty_round_trip() {
        let sketch = DDSketch::new(ALPHA);
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        let decoded = DDSketch::deserialize_from_bytes(&bytes).expect("decode");

        assert_eq!(decoded.get_count(), 0);
        assert_eq!(decoded.min(), None);
        assert_eq!(decoded.max(), None);
        assert!(decoded.store_counts().is_empty());
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// A merged sketch — a store grown on both sides — round-trips with its
    /// scalars and quantiles intact.
    #[test]
    fn ddsketch_merged_round_trip() {
        let mut left = DDSketch::new(ALPHA);
        let mut right = DDSketch::new(ALPHA);
        for i in 1..=200u32 {
            left.add(&f64::from(i));
            right.add(&(f64::from(i) * 0.001));
        }
        left.merge(&right).expect("merge");

        let bytes = left.serialize_to_bytes().expect("serialize");
        let decoded = DDSketch::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.get_count(), left.get_count());
        assert_eq!(decoded.sum(), left.sum());
        assert_eq!(decoded.min(), left.min());
        assert_eq!(decoded.max(), left.max());
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// A crafted envelope carrying another sketch's kind_id must be rejected
    /// even though its metadata and payload parse cleanly.
    #[test]
    fn ddsketch_rejects_foreign_kind_id() {
        let cms =
            crate::CountMin::<crate::Vector2D<i64>, crate::RegularPath>::with_dimensions(3, 8);
        let cms_bytes = cms.serialize_to_bytes().expect("serialize CMS");
        assert!(
            DDSketch::deserialize_from_bytes(&cms_bytes).is_err(),
            "Count-Min bytes must not decode as a DDSketch"
        );

        let metadata = rmp_serde::to_vec_named(&dd_metadata(ALPHA)).unwrap();
        let payload = rmp_serde::to_vec(&DdPayload {
            counts: Vec::new(),
            offset: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        })
        .unwrap();
        let bytes = envelope::encode(&[0x02, 0x00], &metadata, &payload);
        assert!(
            DDSketch::deserialize_from_bytes(&bytes).is_err(),
            "a foreign kind_id must be rejected"
        );
    }

    /// Fail closed on an unexpected metadata key.
    #[test]
    fn dd_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            alpha: f64,
            bogus_field: u8, // key not in DdMetadata
        }
        let bytes = rmp_serde::to_vec_named(&WithExtra {
            metadata_version: 1,
            alpha: ALPHA,
            bogus_field: 7,
        })
        .unwrap();
        assert!(
            rmp_serde::from_slice::<DdMetadata>(&bytes).is_err(),
            "an unexpected metadata key must be rejected"
        );
    }

    /// `alpha` is required: a DDSketch metadata map missing it does not decode,
    /// so the key cannot be silently defaulted to zero.
    #[test]
    fn dd_metadata_rejects_a_missing_alpha_key() {
        #[derive(Serialize)]
        struct WithoutAlpha {
            metadata_version: u8,
        }
        let bytes = rmp_serde::to_vec_named(&WithoutAlpha {
            metadata_version: 1,
        })
        .unwrap();
        assert!(
            rmp_serde::from_slice::<DdMetadata>(&bytes).is_err(),
            "a missing alpha key must be rejected"
        );
    }

    /// `alpha` outside `(0, 1)` makes every bucket index meaningless, so it is
    /// rejected rather than used to derive a nonsense mapping.
    #[test]
    fn ddsketch_rejects_alpha_outside_the_unit_interval() {
        for alpha in [0.0, 1.0, -0.5, 2.0, f64::NAN, f64::INFINITY] {
            let bytes = crafted(alpha, vec![3], 7, 6.0, 1.5, 2.5);
            let err = decode_error(&bytes);
            assert!(
                err.contains("alpha") || err.contains("metadata mismatch"),
                "alpha={alpha} must be rejected by the alpha rule, got {err}"
            );
        }
        // The neighbouring in-range value is the control case.
        assert!(
            DDSketch::deserialize_from_bytes(&crafted(ALPHA, vec![3], 7, 6.0, 1.5, 2.5)).is_ok()
        );
    }

    /// A store whose declared offset pushes its highest bucket index past
    /// `i32` is rejected from the offset and the array length alone, before the
    /// store is rebuilt — and never panics.
    #[test]
    fn ddsketch_rejects_a_store_span_past_i32() {
        let err = decode_error(&crafted(ALPHA, vec![1; 10], i32::MAX - 2, 5.0, 1.0, 2.0));
        assert!(
            err.contains("store span past i32"),
            "an overflowing store span must be rejected as such, got {err}"
        );
        // The same payload one bucket shorter than the boundary decodes.
        let ok = crafted(ALPHA, vec![1; 3], i32::MAX - 2, 5.0, 1.0, 2.0);
        assert!(DDSketch::deserialize_from_bytes(&ok).is_ok());
    }

    /// An empty store has exactly one encoding, so a crafted non-zero offset on
    /// one is rejected instead of decoding into a sketch that re-serializes to
    /// different bytes.
    #[test]
    fn ddsketch_rejects_a_nonzero_offset_on_an_empty_store() {
        let err = decode_error(&crafted(
            ALPHA,
            Vec::new(),
            42,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ));
        assert!(
            err.contains("empty store must be at offset 0"),
            "a non-zero empty-store offset must be rejected as such, got {err}"
        );
    }

    /// Bucket counts that overflow the recovered total are rejected rather than
    /// wrapping into a count the quantile walk would disagree with.
    #[test]
    fn ddsketch_rejects_a_total_count_overflow() {
        let err = decode_error(&crafted(ALPHA, vec![u64::MAX, 1], 0, 5.0, 1.0, 2.0));
        assert!(
            err.contains("overflow the total sample count"),
            "an overflowing bucket total must be rejected as such, got {err}"
        );
    }

    /// The scalars the buckets do not determine are still bounded by them: an
    /// empty store carries the sentinels, a populated one finite positive
    /// bounds in order.
    #[test]
    fn ddsketch_rejects_inconsistent_scalars() {
        let cases = [
            // Populated store with the empty-store sentinels.
            (vec![3u64], 7i32, 6.0f64, f64::INFINITY, f64::NEG_INFINITY),
            // min above max.
            (vec![3], 7, 6.0, 9.0, 2.0),
            // Non-positive min.
            (vec![3], 7, 6.0, 0.0, 2.0),
            // Sum below the smallest ingested value.
            (vec![3], 7, 0.5, 1.5, 2.5),
            // Empty store carrying a non-zero sum.
            (Vec::new(), 0, 5.0, f64::INFINITY, f64::NEG_INFINITY),
            // All-zero buckets carrying populated-store scalars.
            (vec![0, 0], 7, 6.0, 1.5, 2.5),
        ];
        for (counts, offset, sum, min, max) in cases {
            let err = decode_error(&crafted(ALPHA, counts, offset, sum, min, max));
            assert!(
                err.contains("DDSketch empty store must carry")
                    || err.contains("scalars out of order")
                    || err.contains("scalars must be finite"),
                "sum={sum}, min={min}, max={max} must be rejected by a scalar rule, got {err}"
            );
        }
    }

    /// A sketch whose store spans past `i32` would be refused on decode, so it
    /// must not serialize either.
    #[test]
    fn ddsketch_rejects_serializing_a_store_span_past_i32() {
        let mut sketch = DDSketch::new(ALPHA);
        sketch.add(&1.0f64);
        sketch.store.counts = Vector1D::from_vec(vec![1u64; 10]);
        sketch.store.offset = i32::MAX - 2;
        assert!(
            sketch.serialize_to_bytes().is_err(),
            "a store spanning past i32 must not serialize"
        );
    }

    /// Bucket counts that overflow the total sample count must not serialize
    /// either — the encode side enforces the decoder's rule.
    #[test]
    fn ddsketch_rejects_serializing_an_overflowing_bucket_total() {
        let mut sketch = DDSketch::new(ALPHA);
        sketch.add(&1.0f64);
        sketch.store.counts = Vector1D::from_vec(vec![u64::MAX, 1]);
        sketch.store.offset = 0;
        assert!(
            sketch.serialize_to_bytes().is_err(),
            "bucket counts overflowing the total must not serialize"
        );
    }
}
