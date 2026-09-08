//! ASAPv1 wire serialization for the dynamic KLL quantile sketch.
//!
//! Child submodule of [`crate::sketches::kll_dynamic`]: it holds the
//! `serialize_to_bytes` / `deserialize_from_bytes` impls for [`KLLDynamic`]
//! (kind_id `0x06 0x01`). The wire DTOs (metadata, payload, coin) and the
//! fail-closed validation are **shared** with the compact KLL and live in
//! [`crate::sketches::kll::wire`]; only the kind_id differs. See
//! `docs/asapv1_wire_format.md` §3.
//!
//! Unlike the compact KLL, `KLLDynamic` already stores its retained samples in
//! the wire's top-most-level-first order with level 0 in input order (it appends
//! to the end of a growable buffer), so encode/decode are a direct copy — no
//! buffer reversal is needed.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};

use crate::Vector1D;
use crate::message_pack_format::envelope;
use crate::sketches::kll::{
    Coin, KLL_KIND_DYNAMIC, KllCoinWire, KllPayload, KllWireItem, kll_metadata,
    split_and_validate_meta, validate_kll_payload,
};

use super::{CAPACITY_CACHE_LEN, KLLDynamic};

// `wire` is a descendant of the sketch module, so these impls read the private
// fields and construct the struct directly.
impl<T> KLLDynamic<T>
where
    T: crate::common::numerical::NumericalValue
        + KllWireItem
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>,
{
    /// Serializes the sketch into an ASAPv1 MessagePack envelope
    /// (kind_id `0x06 0x01`). `KLLDynamic`'s buffer is already in the wire's
    /// top-most-level-first / input-order-L0 layout, so `levels` and `items` are
    /// copied straight out.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        // `KLLDynamic` has no reproducible-seed concept, so `seed` is always
        // `None` and the metadata key is omitted.
        let metadata =
            rmp_serde::to_vec_named(&kll_metadata::<T>(self.k as u32, self.m as u32, None))?;
        let (state, bit_cache, remaining_bits) = self.co.to_wire();
        let payload = rmp_serde::to_vec(&KllPayload {
            levels: self.levels.as_slice().iter().map(|&l| l as u32).collect(),
            items: self.items.as_slice().to_vec(),
            coin: KllCoinWire {
                state,
                bit_cache,
                remaining_bits,
            },
        })?;
        Ok(envelope::encode(KLL_KIND_DYNAMIC, &metadata, &payload))
    }

    /// Deserializes a `KLLDynamic` from an ASAPv1 MessagePack envelope. Bytes
    /// whose metadata does not match this item type are rejected (fail closed),
    /// as are inconsistent level layouts.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (meta, payload_bytes) = split_and_validate_meta::<T>(bytes, KLL_KIND_DYNAMIC)?;
        let payload: KllPayload<T> = from_slice(payload_bytes)?;
        let num_levels = validate_kll_payload(&payload.levels, &payload.items, &payload.coin)?;

        let KllPayload {
            levels,
            items,
            coin,
        } = payload;
        let mut sketch = KLLDynamic {
            items: Vector1D::from_vec(items),
            levels: Vector1D::from_vec(levels.into_iter().map(|l| l as usize).collect()),
            k: meta.k as usize,
            m: meta.m as usize,
            num_levels,
            co: Coin::from_wire(coin.state, coin.bit_cache, coin.remaining_bits as u8),
            // The wire carries the coin's live state, not the construction
            // seed, so a decoded sketch has no seed to re-derive on `clear()`.
            seed: None,
            capacity_cache: [0; CAPACITY_CACHE_LEN],
            top_height: 0,
            level0_capacity: 0,
        };
        sketch.rebuild_capacity_cache();
        Ok(sketch)
    }
}

#[cfg(test)]
mod tests {
    use crate::message_pack_format::envelope;
    use crate::sketches::kll::KLL_KIND_DYNAMIC;
    use crate::sketches::kll_dynamic::KLLDynamic;

    fn build_dynamic(k: i32, n: u64) -> KLLDynamic<f64> {
        let mut sketch = KLLDynamic::<f64>::init_kll(k);
        for v in 1..=n {
            sketch.update(&(v as f64));
        }
        sketch
    }

    #[test]
    fn kll_dynamic_envelope_structure_and_round_trip() {
        let sketch = build_dynamic(200, 200_000);
        let bytes = sketch.serialize_to_bytes().expect("serialize");

        assert!(bytes.starts_with(envelope::MAGIC));
        assert_eq!(bytes[6], envelope::VERSION);
        assert_eq!(bytes[7], 2, "kind_id_len");
        assert_eq!(&bytes[8..10], KLL_KIND_DYNAMIC);

        let decoded = KLLDynamic::<f64>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            bytes,
            "KLLDynamic serialized bytes differed after round trip"
        );
        for &q in &[0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0] {
            assert_eq!(
                decoded.quantile(q),
                sketch.quantile(q),
                "quantile mismatch at q={q} after round trip"
            );
        }
    }

    #[test]
    fn kll_dynamic_empty_round_trip() {
        let sketch = KLLDynamic::<f64>::init_kll(200);
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        let decoded = KLLDynamic::<f64>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    #[test]
    fn kll_dynamic_i64_round_trip() {
        let mut sketch = KLLDynamic::<i64>::init_kll(200);
        for v in 1..=50_000i64 {
            sketch.update(&v);
        }
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        assert_eq!(&bytes[8..10], KLL_KIND_DYNAMIC);
        let decoded = KLLDynamic::<i64>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    #[test]
    fn kll_dynamic_item_type_cross_rejection() {
        let sketch = build_dynamic(200, 1000);
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        assert!(
            KLLDynamic::<i64>::deserialize_from_bytes(&bytes).is_err(),
            "f64 KLLDynamic bytes must be rejected by an i64 decoder"
        );
    }
}
