//! ASAPv1 wire serialization for the KLL quantile sketches.
//!
//! Child submodule of [`crate::sketches::kll`]: it holds the shared KLL wire
//! DTOs (metadata + payload + coin), the kind_id constants for both KLL
//! variants, the [`KllWireItem`] marker trait, and the `serialize_to_bytes` /
//! `deserialize_from_bytes` impls for the compact [`KLL`]. Being a descendant
//! module, it reads the sketch's private fields (`self.items`, `self.levels`,
//! `self.co`, …) and constructs the struct directly without widening any field
//! visibility. The sibling `kll_dynamic::wire` reuses these DTOs for the dynamic
//! variant. See `docs/asapv1_wire_format.md` §3.
//!
//! ## Why KLL metadata has no hash spec
//!
//! Unlike HLL and Count-Min, KLL never hashes its inputs — it orders raw numeric
//! values with `total_cmp`. So the hash-spec metadata group (profile id,
//! algorithm, inlined `seed_list`, …) that HLL/CMS carry does not apply, and KLL
//! metadata is **structural params only** (`k`, `m`, `item_type`, and an optional
//! `seed`). See the wire doc's KLL section for the recorded decision.
//!
//! ## Payload item order (cross-language contract)
//!
//! The payload's `levels` / `items` use the **top-most-level-first** layout,
//! byte-for-byte matching `sketchlib-go`'s `KLLState` (index `i` in `levels`
//! maps to compactor level `num_levels - 1 - i`; level 0's run is in input
//! order). For the compact KLL this is exactly what [`KLL::wire_levels`] /
//! [`KLL::wire_items`] emit (they reverse the leftward-grown L0 buffer back to
//! input order); decode inverts that mapping.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::common::numerical::NumericalValue;
use crate::message_pack_format::envelope;

use super::{
    CAPACITY_CACHE_LEN, Coin, KLL, MAX_CACHEABLE_K, MAX_LEVELS, checked_weighted_count,
    compute_max_capacity,
};

const KLL_KIND_FAMILY: u8 = 0x06;
/// kind_id for the compact, fixed-buffer [`KLL`] (`0x06 0x00`).
pub(crate) const KLL_KIND_COMPACT: &[u8] = &[KLL_KIND_FAMILY, 0x00];
/// kind_id for the dynamic-buffer `KLLDynamic` (`0x06 0x01`).
pub(crate) const KLL_KIND_DYNAMIC: &[u8] = &[KLL_KIND_FAMILY, 0x01];

/// Names the wire item type carried in the KLL metadata (`item_type`).
/// Implemented only for the two wire-eligible retained-sample types; an exotic
/// `KLL<T>` (any other `NumericalValue`) is not wire-serializable and must be
/// converted to one of these first.
pub trait KllWireItem: Copy {
    /// Metadata `item_type` string — `"f64"` or `"i64"`.
    const ITEM_TYPE: &'static str;
}
impl KllWireItem for f64 {
    const ITEM_TYPE: &'static str = "f64";
}
impl KllWireItem for i64 {
    const ITEM_TYPE: &'static str = "i64";
}

/// KLL descriptor metadata (ASAPv1 §2), a msgpack **map** (`to_vec_named`) with
/// keys in this declaration order — the canonical order the wire spec fixes (Go
/// must mirror it). KLL does not hash, so there is **no hash-spec group**: the
/// fields are structural params only. `deny_unknown_fields` makes decode fail
/// closed on any unexpected key.
///
/// `seed` is the **only optional** field: it is the compaction RNG's reproducible
/// seed (see [`crate::sketches::kll::KLL::init_with_seed`]). It is present only
/// when the sketch carries one (`Some`), and the key is **omitted** otherwise
/// (`skip_serializing_if` + `default`), so an unseeded sketch's bytes are
/// unchanged. It is echoed on decode, never pinned. The compact `KLL` populates
/// it; `KLLDynamic` has no seed and always omits it.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct KllMetadata {
    pub(crate) metadata_version: u8,
    pub(crate) k: u32,
    pub(crate) m: u32,
    pub(crate) item_type: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) seed: Option<u64>,
}

/// Builds the KLL descriptor metadata for a wire-eligible item type `T`. `seed`
/// is the sketch's reproducible compaction seed when it has one (`Some`),
/// otherwise `None` (the key is then omitted from the wire).
pub(crate) fn kll_metadata<T: KllWireItem>(k: u32, m: u32, seed: Option<u64>) -> KllMetadata {
    KllMetadata {
        metadata_version: 1,
        k,
        m,
        item_type: T::ITEM_TYPE.to_string(),
        seed,
    }
}

/// The compaction coin's raw RNG state, carried in the payload so a decoded
/// sketch continues compacting deterministically. Serialized positionally (as
/// part of [`KllPayload`], `to_vec`) so it lands as a nested 3-element array
/// `[state, bit_cache, remaining_bits]`, mirroring `sketchlib-go`'s `CoinState`.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct KllCoinWire {
    pub(crate) state: u64,
    pub(crate) bit_cache: u64,
    pub(crate) remaining_bits: u32,
}

/// KLL payload (ASAPv1 §3), a msgpack **array** (`to_vec`, positional):
/// `[levels, items, coin]`. `num_levels` is `levels.len() - 1` (derived, so not
/// stored); `k` / `m` / `item_type` live in the metadata. Element type of
/// `items` is fixed by the metadata `item_type`.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct KllPayload<T> {
    pub(crate) levels: Vec<u32>,
    pub(crate) items: Vec<T>,
    pub(crate) coin: KllCoinWire,
}

/// Validates the shared parts of a decoded KLL payload's level layout against
/// the retained items, and the coin's `remaining_bits` bound. Returns
/// `(num_levels, Coin)` on success. Shared by both KLL variants' decoders so the
/// fail-closed rules stay identical. `k` / `m` are echoed structural params (the
/// sketch is sized from them), so they are not cross-checked here.
pub(crate) fn validate_kll_payload<T>(
    levels: &[u32],
    items: &[T],
    coin: &KllCoinWire,
) -> Result<usize, RmpDecodeError> {
    if levels.len() < 2 {
        return Err(RmpDecodeError::Uncategorized(format!(
            "KLL payload: levels too short (len {}, need >= 2)",
            levels.len()
        )));
    }
    let num_levels = levels.len() - 1;
    if num_levels > MAX_LEVELS {
        return Err(RmpDecodeError::Uncategorized(format!(
            "KLL payload: num_levels {num_levels} exceeds MAX_LEVELS {MAX_LEVELS}"
        )));
    }
    if levels[0] != 0 {
        return Err(RmpDecodeError::Uncategorized(format!(
            "KLL payload: levels[0] must be 0, got {}",
            levels[0]
        )));
    }
    if levels.windows(2).any(|w| w[0] > w[1]) {
        return Err(RmpDecodeError::Uncategorized(
            "KLL payload: levels must be non-decreasing".to_string(),
        ));
    }
    if *levels.last().unwrap() as usize != items.len() {
        return Err(RmpDecodeError::Uncategorized(format!(
            "KLL payload: levels[last] {} != items.len() {}",
            levels.last().unwrap(),
            items.len()
        )));
    }
    if coin.remaining_bits > u64::BITS {
        return Err(RmpDecodeError::Uncategorized(format!(
            "KLL payload: coin remaining_bits {} exceeds 64",
            coin.remaining_bits
        )));
    }
    // Reject a structurally-valid but adversarial level distribution (e.g. many
    // items parked at a high compactor level) that would overflow the weighted
    // `count()` / `rank()` / `cdf()` at query time. `levels` is top-most-first,
    // so map slot `i` to compactor level `num_levels - 1 - i` to get sizes
    // bottom-first for the weight check.
    let sizes: Vec<usize> = (0..num_levels)
        .map(|h| {
            let i = num_levels - 1 - h;
            (levels[i + 1] - levels[i]) as usize
        })
        .collect();
    if checked_weighted_count(&sizes).is_none() {
        return Err(RmpDecodeError::Uncategorized(
            "KLL payload: level layout overflows weighted count".to_string(),
        ));
    }
    Ok(num_levels)
}

/// Splits the envelope, checks the `kind_id`, and validates the metadata against
/// the target item type `T` (fail closed on any mismatch). `k` / `m` are
/// echoed back into the expected metadata (they are structural and sizing the
/// sketch from them is the whole point), so only `metadata_version` and
/// `item_type` are effectively pinned. Returns the decoded metadata plus the
/// raw payload bytes. Shared by both KLL variants.
pub(crate) fn split_and_validate_meta<'a, T: KllWireItem>(
    bytes: &'a [u8],
    expected_kind_id: &[u8],
) -> Result<(KllMetadata, &'a [u8]), RmpDecodeError> {
    let (kind_id, metadata, payload) =
        envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
    if kind_id != expected_kind_id {
        return Err(RmpDecodeError::Uncategorized(format!(
            "KLL kind_id mismatch: stored {kind_id:?}, expected {expected_kind_id:?}"
        )));
    }
    let meta: KllMetadata = from_slice(metadata)?;
    if meta != kll_metadata::<T>(meta.k, meta.m, meta.seed) {
        return Err(RmpDecodeError::Uncategorized(
            "ASAPv1 KLL envelope: metadata mismatch".to_string(),
        ));
    }
    // Fail closed on out-of-range `k` / `m`. `k` / `m` are echoed structural
    // params (the sketch is sized from them), so the equality check above cannot
    // pin them — without this bound a crafted `k`/`m` near `u32::MAX` would drive
    // `compute_max_capacity` (compact decode) into a multi-terabyte allocation
    // and abort the process. A legitimately serialized sketch always has
    // `2 <= m <= k <= MAX_CACHEABLE_K` because the constructor clamps to that
    // range (`init_internal`), so this never rejects real bytes.
    if meta.m < 2 || meta.m > meta.k || meta.k > MAX_CACHEABLE_K as u32 {
        return Err(RmpDecodeError::Uncategorized(format!(
            "ASAPv1 KLL envelope: k={}, m={} outside valid range (2 <= m <= k <= {MAX_CACHEABLE_K})",
            meta.k, meta.m
        )));
    }
    Ok((meta, payload))
}

// Wire serialization for the compact KLL. `wire` is a descendant of the sketch
// module, so these impls read the private fields and construct the struct
// directly.
impl<T> KLL<T>
where
    T: NumericalValue + KllWireItem + Serialize + for<'de> Deserialize<'de>,
{
    /// Serializes the sketch into an ASAPv1 MessagePack envelope
    /// (kind_id `0x06 0x00`). The retained samples use the top-most-level-first
    /// layout that matches `sketchlib-go`'s `KLLState`.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let metadata =
            rmp_serde::to_vec_named(&kll_metadata::<T>(self.k as u32, self.m as u32, self.seed))?;
        let (state, bit_cache, remaining_bits) = self.co.to_wire();
        let payload = rmp_serde::to_vec(&KllPayload {
            levels: self.wire_levels(),
            items: self.wire_items(),
            coin: KllCoinWire {
                state,
                bit_cache,
                remaining_bits,
            },
        })?;
        Ok(envelope::encode(KLL_KIND_COMPACT, &metadata, &payload))
    }

    /// Deserializes a compact KLL from an ASAPv1 MessagePack envelope. Bytes
    /// whose metadata does not match this item type are rejected (fail closed),
    /// as are inconsistent level layouts.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (meta, payload_bytes) = split_and_validate_meta::<T>(bytes, KLL_KIND_COMPACT)?;
        let payload: KllPayload<T> = from_slice(payload_bytes)?;
        let num_levels = validate_kll_payload(&payload.levels, &payload.items, &payload.coin)?;
        Self::from_wire_top_first(
            meta.k as usize,
            meta.m as usize,
            meta.seed,
            num_levels,
            payload,
        )
    }

    /// Rebuilds the compact leftward-grown buffer from the top-most-first wire
    /// payload. Inverse of [`KLL::wire_levels`] / [`KLL::wire_items`]: L0's
    /// input-order run is reversed back into the buffer (which stores L0
    /// reverse-input), and higher levels are copied as-is, so a round-trip
    /// re-serializes to byte-identical output.
    fn from_wire_top_first(
        k: usize,
        m: usize,
        seed: Option<u64>,
        num_levels: usize,
        payload: KllPayload<T>,
    ) -> Result<Self, RmpDecodeError> {
        let KllPayload {
            levels,
            items,
            coin,
        } = payload;

        let max_cap = compute_max_capacity(k, m);
        let total = items.len();
        if total > max_cap {
            return Err(RmpDecodeError::Uncategorized(format!(
                "KLL payload: {total} items exceed max_capacity {max_cap} for k={k}, m={m}"
            )));
        }
        let offset = max_cap - total;

        let mut buf = vec![T::default(); max_cap].into_boxed_slice();
        let mut internal_levels = vec![0usize; MAX_LEVELS + 1].into_boxed_slice();

        // Place levels bottom-first into the buffer starting at `offset` (free
        // space stays at the front). Wire slot `top_i = num_levels - 1 - h`
        // holds compactor level `h`.
        let mut cursor = offset;
        for h in 0..num_levels {
            let top_i = num_levels - 1 - h;
            let s = levels[top_i] as usize;
            let e = levels[top_i + 1] as usize;
            internal_levels[h] = cursor;
            if h == 0 {
                // Wire L0 is input order; the compact buffer stores L0
                // reverse-input, so reverse it back on the way in.
                for (j, &v) in items[s..e].iter().rev().enumerate() {
                    buf[cursor + j] = v;
                }
            } else {
                buf[cursor..cursor + (e - s)].copy_from_slice(&items[s..e]);
            }
            cursor += e - s;
        }
        // Sentinel boundary for the top level; slots beyond stay 0, matching a
        // live sketch (which only writes `levels[num_levels]`), so a decoded
        // sketch's level array is identical to the source's.
        internal_levels[num_levels] = cursor;
        debug_assert_eq!(cursor, max_cap);

        let mut sketch = KLL {
            items: buf,
            levels: internal_levels,
            k,
            m,
            num_levels,
            max_capacity: max_cap,
            co: Coin::from_wire(coin.state, coin.bit_cache, coin.remaining_bits as u8),
            // The reproducible compaction seed is restored from the metadata (it
            // is `None` for a sketch that never carried one), so a decoded sketch
            // keeps clear()-determinism when the producer had it.
            seed,
            capacity_cache: [0; CAPACITY_CACHE_LEN],
            top_height: 0,
            level0_capacity: 0,
            merge_buf: Vec::with_capacity(k),
            cdf_cache: None,
        };
        sketch.rebuild_capacity_cache();
        sketch.ensure_levels_sorted();
        Ok(sketch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketches::kll::KLL;

    fn build_kll(k: i32, seed: u64, n: u64) -> KLL<f64> {
        let mut sketch = KLL::<f64>::init_kll_with_seed(k, seed);
        for v in 1..=n {
            sketch.update(&(v as f64));
        }
        sketch
    }

    #[test]
    fn kll_envelope_structure_and_round_trip() {
        // Enough data to force several compaction levels.
        let sketch = build_kll(200, 42, 200_000);
        let bytes = sketch.serialize_to_bytes().expect("serialize");

        assert!(bytes.starts_with(envelope::MAGIC));
        assert_eq!(bytes[6], envelope::VERSION);
        assert_eq!(bytes[7], 2, "kind_id_len");
        assert_eq!(&bytes[8..10], KLL_KIND_COMPACT);

        let decoded = KLL::<f64>::deserialize_from_bytes(&bytes).expect("decode");
        // Byte-stable round trip.
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            bytes,
            "KLL serialized bytes differed after round trip"
        );
        // Quantiles preserved exactly (same retained state).
        for &q in &[0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0] {
            assert_eq!(
                decoded.quantile(q),
                sketch.quantile(q),
                "quantile mismatch at q={q} after round trip"
            );
        }
    }

    #[test]
    fn kll_empty_round_trip() {
        let sketch = KLL::<f64>::init_kll_with_seed(200, 7);
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        let decoded = KLL::<f64>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.count(), 0);
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    #[test]
    fn kll_i64_round_trip() {
        let mut sketch = KLL::<i64>::init_kll_with_seed(200, 5);
        for v in 1..=50_000i64 {
            sketch.update(&v);
        }
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        assert_eq!(&bytes[8..10], KLL_KIND_COMPACT);
        let decoded = KLL::<i64>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
        assert_eq!(decoded.count(), sketch.count());
    }

    #[test]
    fn kll_item_type_cross_rejection() {
        // f64 bytes must not decode into an i64 KLL (metadata item_type mismatch).
        let sketch = build_kll(200, 1, 1000);
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        assert!(
            KLL::<i64>::deserialize_from_bytes(&bytes).is_err(),
            "f64 KLL bytes must be rejected by an i64 decoder"
        );
    }

    #[test]
    fn kll_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            k: u32,
            m: u32,
            item_type: String,
            bogus_field: u8,
        }
        let extra = WithExtra {
            metadata_version: 1,
            k: 200,
            m: 8,
            item_type: "f64".to_string(),
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).expect("encode");
        assert!(
            rmp_serde::from_slice::<KllMetadata>(&bytes).is_err(),
            "an unexpected metadata key must be rejected"
        );
    }

    #[test]
    fn kll_rejects_inconsistent_levels() {
        // Valid envelope + metadata, but levels[last] != items.len().
        let metadata = rmp_serde::to_vec_named(&kll_metadata::<f64>(200, 8, None)).unwrap();
        let payload = rmp_serde::to_vec(&KllPayload::<f64> {
            levels: vec![0, 3],
            items: vec![1.0, 2.0], // len 2, but levels claim 3
            coin: KllCoinWire {
                state: 1,
                bit_cache: 0,
                remaining_bits: 0,
            },
        })
        .unwrap();
        let bytes = envelope::encode(KLL_KIND_COMPACT, &metadata, &payload);
        assert!(
            KLL::<f64>::deserialize_from_bytes(&bytes).is_err(),
            "inconsistent level layout must be rejected, not panic"
        );
    }

    #[test]
    fn kll_rejects_out_of_range_k_m() {
        // A crafted envelope with an enormous `k`/`m` must fail closed, not drive
        // compute_max_capacity into a giant allocation. Empty payload so nothing
        // downstream depends on the dimensions.
        let empty_payload = || {
            rmp_serde::to_vec(&KllPayload::<f64> {
                levels: vec![0, 0],
                items: Vec::new(),
                coin: KllCoinWire {
                    state: 1,
                    bit_cache: 0,
                    remaining_bits: 0,
                },
            })
            .unwrap()
        };
        for (k, m) in [
            (u32::MAX, u32::MAX),
            (MAX_CACHEABLE_K as u32 + 1, 8),
            (200, 1),
        ] {
            let metadata = rmp_serde::to_vec_named(&kll_metadata::<f64>(k, m, None)).unwrap();
            let bytes = envelope::encode(KLL_KIND_COMPACT, &metadata, &empty_payload());
            assert!(
                KLL::<f64>::deserialize_from_bytes(&bytes).is_err(),
                "k={k}, m={m} must be rejected, not allocated"
            );
        }
    }

    #[test]
    fn kll_seed_present_when_seeded_omitted_when_unseeded() {
        // A seeded sketch records its seed in the metadata; an unseeded one omits
        // the key entirely (so its bytes are unaffected by the optional field).
        let seeded = KLL::<f64>::init_kll_with_seed(200, 42);
        let bytes = seeded.serialize_to_bytes().expect("serialize");
        let (_k, meta_bytes, _p) = envelope::split(&bytes).expect("split");
        let meta: KllMetadata = rmp_serde::from_slice(meta_bytes).expect("meta");
        assert_eq!(meta.seed, Some(42), "seeded KLL must record its seed");

        let mut unseeded = KLL::<f64>::init_kll(200); // Coin::new(); seed = None
        unseeded.update(&1.0);
        let bytes = unseeded.serialize_to_bytes().expect("serialize");
        let (_k, meta_bytes, _p) = envelope::split(&bytes).expect("split");
        let meta: KllMetadata = rmp_serde::from_slice(meta_bytes).expect("meta");
        assert_eq!(meta.seed, None, "unseeded KLL must omit the seed key");
    }

    #[test]
    fn kll_unseeded_round_trip_byte_stable() {
        // An unseeded compact KLL (seed = None, so the metadata key is omitted)
        // must decode end-to-end and re-serialize byte-identically — exercises the
        // optional-key-absent path through decode, not just encode.
        let mut s = KLL::<f64>::init_kll(200); // Coin::new(); seed = None
        for v in 1..=2000u64 {
            s.update(&(v as f64));
        }
        let bytes = s.serialize_to_bytes().expect("serialize");
        let decoded = KLL::<f64>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            bytes,
            "unseeded KLL round trip not byte-stable"
        );
        assert_eq!(decoded.count(), s.count());
    }

    #[test]
    fn kll_seed_survives_round_trip_so_clear_stays_deterministic() {
        // The point of carrying seed: a decoded seeded sketch must re-seed clear()
        // from the original seed (not wall-clock). If seed were dropped on decode,
        // `a` below would re-randomize on clear() and diverge from `b`.
        let mut src = KLL::<f64>::init_kll_with_seed(200, 42);
        for v in 1..=5000u64 {
            src.update(&(v as f64));
        }
        let mut a = KLL::<f64>::deserialize_from_bytes(&src.serialize_to_bytes().unwrap()).unwrap();
        let mut b = KLL::<f64>::init_kll_with_seed(200, 42);

        a.clear();
        b.clear();
        for v in 1..=3000u64 {
            a.update(&(v as f64));
            b.update(&(v as f64));
        }
        assert_eq!(
            a.serialize_to_bytes().unwrap(),
            b.serialize_to_bytes().unwrap(),
            "decoded sketch lost its seed: clear() diverged from a fresh seeded sketch"
        );
    }

    #[test]
    fn kll_rejects_weighted_count_overflow() {
        // Structurally valid (monotonic levels, levels[last]==items.len()) but
        // adversarial: 16 items parked at compactor level 60 makes 16 * 2^60
        // overflow usize in count(). Decode must reject, not hand back a sketch
        // that panics on the first query.
        let num_levels = 61usize;
        // Top-most-first cumulative levels: the top level (slot 0) holds all 16
        // items, every lower level is empty.
        let mut levels = vec![16u32; num_levels + 1];
        levels[0] = 0;
        let metadata = rmp_serde::to_vec_named(&kll_metadata::<f64>(200, 8, None)).unwrap();
        let payload = rmp_serde::to_vec(&KllPayload::<f64> {
            levels,
            items: vec![1.0; 16],
            coin: KllCoinWire {
                state: 1,
                bit_cache: 0,
                remaining_bits: 0,
            },
        })
        .unwrap();
        let bytes = envelope::encode(KLL_KIND_COMPACT, &metadata, &payload);
        assert!(
            KLL::<f64>::deserialize_from_bytes(&bytes).is_err(),
            "a level layout that overflows the weighted count must be rejected"
        );
    }

    #[test]
    fn kll_dynamic_kind_id_rejected_by_compact() {
        // A compact decoder must reject the dynamic kind_id.
        let metadata = rmp_serde::to_vec_named(&kll_metadata::<f64>(200, 8, None)).unwrap();
        let payload = rmp_serde::to_vec(&KllPayload::<f64> {
            levels: vec![0, 0],
            items: Vec::<f64>::new(),
            coin: KllCoinWire {
                state: 1,
                bit_cache: 0,
                remaining_bits: 0,
            },
        })
        .unwrap();
        let bytes = envelope::encode(KLL_KIND_DYNAMIC, &metadata, &payload);
        assert!(
            KLL::<f64>::deserialize_from_bytes(&bytes).is_err(),
            "dynamic kind_id must be rejected by the compact decoder"
        );
    }
}
