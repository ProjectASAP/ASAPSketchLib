//! ASAPv1 wire serialization for [`UniformSampling`].
//!
//! Child submodule of [`crate::sketches::uniform`]: it holds ALL of the
//! sampler's serialization (the metadata/payload DTOs, the kind_id constant,
//! and the `serialize_to_bytes` / `deserialize_from_bytes` impls).
//!
//! Being a descendant module, it reads the private `sample_rate`, `total_seen`,
//! `rng_state` and `entries` fields directly and rebuilds the struct without
//! widening any field visibility.
//!
//! UniformSampling is one algorithm — a single kind_id `0x0d 0x00`.
//!
//! ## No hash spec
//!
//! The sampler never hashes: it draws a SplitMix64 priority per update and
//! orders entries by that priority. So the hash-spec metadata group has no
//! truthful value here and is omitted entirely, as KLL's is.
//!
//! ## RNG state
//!
//! `rng_state` is the SplitMix64 word the next priority is drawn from, carried
//! in the payload so a decoded sampler continues the same draw sequence. It is
//! the counterpart of KLL's `coin`.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::message_pack_format::envelope;

use super::{SampleEntry, UniformSampling};

/// UniformSampling kind_id: family `0x0d`, single algorithm variant `0x00`.
const US_KIND: &[u8] = &[0x0d, 0x00];

/// Metadata `item_type`: the sampler stores every retained sample as `f64`.
const SAMPLE_ITEM_TYPE: &str = "f64";

/// UniformSampling descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it).
///
/// Structural params only; the sampler does not hash, so there is no hash-spec
/// group. `deny_unknown_fields` makes decode fail closed on any unexpected key.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct UsMetadata {
    metadata_version: u8,
    sample_rate: f64,
    item_type: String,
}

/// Builds the UniformSampling descriptor metadata. `sample_rate` is the
/// sampler's construction rate; `item_type` names the element type of the
/// payload's `values` array.
fn us_metadata(sample_rate: f64) -> UsMetadata {
    UsMetadata {
        metadata_version: 1,
        sample_rate,
        item_type: SAMPLE_ITEM_TYPE.to_string(),
    }
}

/// UniformSampling payload (ASAPv1 §3.13), a msgpack **array** (`to_vec`,
/// positional): `[priorities, values, total_seen, rng_state]`.
///
/// `priorities` and `values` are parallel and equal-length; the number of
/// retained samples is `len(values)` (derived, so not stored). `sample_rate`
/// lives in the metadata.
#[derive(Debug, Serialize, Deserialize)]
struct UsPayload {
    priorities: Vec<u64>,
    values: Vec<f64>,
    total_seen: u64,
    rng_state: u64,
}

/// The canonical emitted order: ascending `priority`, ties broken by
/// `f64::total_cmp` on the parallel value. Fixes one encoding per retained set.
fn canonical_order(a: &(u64, f64), b: &(u64, f64)) -> std::cmp::Ordering {
    a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1))
}

/// Rejects a rate outside `(0, 1]`, matching the constructor's own bound.
/// A crafted rate must not reach `target_size`.
fn check_sample_rate(sample_rate: f64) -> Result<(), String> {
    if sample_rate.is_finite() && sample_rate > 0.0 && sample_rate <= 1.0 {
        Ok(())
    } else {
        Err(format!(
            "UniformSampling sample_rate must be within (0, 1], got {sample_rate}"
        ))
    }
}

// Wire serialization for the sampler. `wire` is a descendant of the sketch
// module, so this impl reads the private fields directly.
impl UniformSampling {
    /// Serializes the sampler into an ASAPv1 MessagePack envelope. Entries are
    /// emitted in the canonical order (ascending priority, ties by
    /// `total_cmp`), so a decoded sampler re-serializes byte-identically.
    ///
    /// A sampler holding more entries than its own rate allows, or carrying a
    /// rate outside `(0, 1]`, is an error rather than bytes that would be
    /// refused on decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        check_sample_rate(self.sample_rate).map_err(RmpEncodeError::Syntax)?;
        let target = Self::target_size(self.total_seen, self.sample_rate);
        if self.entries.len() > target {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 UniformSampling envelope: {} entries exceed the target size {target} for total_seen {}",
                self.entries.len(),
                self.total_seen
            )));
        }
        let mut ordered: Vec<(u64, f64)> = self
            .entries
            .iter()
            .map(|entry| (entry.priority, entry.value))
            .collect();
        ordered.sort_by(canonical_order);
        let metadata = rmp_serde::to_vec_named(&us_metadata(self.sample_rate))?;
        let payload = rmp_serde::to_vec(&UsPayload {
            priorities: ordered.iter().map(|e| e.0).collect(),
            values: ordered.iter().map(|e| e.1).collect(),
            total_seen: self.total_seen,
            rng_state: self.rng_state,
        })?;
        Ok(envelope::encode(US_KIND, &metadata, &payload))
    }

    /// Deserializes a sampler from an ASAPv1 MessagePack envelope. The rate is
    /// read from the (validated) metadata; the payload carries the retained
    /// entries plus the two running scalars.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != US_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "UniformSampling kind_id mismatch: stored {kind_id:?}, expected {US_KIND:?}"
            )));
        }
        let meta: UsMetadata = from_slice(metadata)?;
        check_sample_rate(meta.sample_rate).map_err(RmpDecodeError::Uncategorized)?;
        // `sample_rate` is a property of the stored sampler, so it is echoed
        // back into the expected block; `metadata_version` and `item_type` are
        // pinned by the comparison.
        if meta != us_metadata(meta.sample_rate) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 UniformSampling envelope: metadata mismatch".to_string(),
            ));
        }
        let p: UsPayload = from_slice(payload)?;
        if p.priorities.len() != p.values.len() {
            return Err(RmpDecodeError::Uncategorized(format!(
                "UniformSampling priorities length {} != values length {}",
                p.priorities.len(),
                p.values.len()
            )));
        }
        // The declared stream length never sizes an allocation: the entry
        // vector is built from `len(values)`, so a payload declaring
        // `total_seen = u64::MAX` with two samples costs two samples.
        let target = Self::target_size(p.total_seen, meta.sample_rate);
        if p.values.len() > target {
            return Err(RmpDecodeError::Uncategorized(format!(
                "UniformSampling payload holds {} samples, above the target size {target} for total_seen {}",
                p.values.len(),
                p.total_seen
            )));
        }
        let mut entries: Vec<SampleEntry> = Vec::with_capacity(p.values.len());
        for (idx, (&priority, &value)) in p.priorities.iter().zip(p.values.iter()).enumerate() {
            // Entries are held in ascending priority; `insert_entry` binary
            // searches on that invariant, so an unordered payload is rejected
            // rather than decoded into a sampler that misplaces its next entry.
            if idx > 0
                && canonical_order(
                    &(p.priorities[idx - 1], p.values[idx - 1]),
                    &(priority, value),
                ) == std::cmp::Ordering::Greater
            {
                return Err(RmpDecodeError::Uncategorized(format!(
                    "UniformSampling entries must be in ascending priority order, broken at index {idx}"
                )));
            }
            entries.push(SampleEntry::new(priority, value));
        }
        Ok(UniformSampling {
            sample_rate: meta.sample_rate,
            total_seen: p.total_seen,
            rng_state: p.rng_state,
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(rate: f64, seed: u64, count: u64) -> UniformSampling {
        let mut sampler = UniformSampling::with_seed(rate, seed);
        for value in 0..count {
            sampler.update(value as f64);
        }
        sampler
    }

    fn crafted(metadata: &UsMetadata, payload: &UsPayload) -> Vec<u8> {
        envelope::encode(
            US_KIND,
            &rmp_serde::to_vec_named(metadata).unwrap(),
            &rmp_serde::to_vec(payload).unwrap(),
        )
    }

    #[test]
    fn uniform_sampling_round_trip_serialization() {
        let sampler = filled(0.25, 0xBEEF_FACE, 40);

        let encoded = sampler.serialize_to_bytes().expect("serialize sampler");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x0d, 0x00]); // kind_id_len=2, kind_id=[0x0d,0x00]

        let decoded =
            UniformSampling::deserialize_from_bytes(&encoded).expect("deserialize sampler");

        assert_eq!(sampler.sample_rate(), decoded.sample_rate());
        assert_eq!(sampler.total_seen(), decoded.total_seen());
        assert_eq!(sampler.len(), decoded.len());
        assert_eq!(sampler.samples(), decoded.samples());
    }

    /// The priorities are payload state, not derived: a decoded sampler must
    /// re-serialize to the exact bytes it decoded from.
    #[test]
    fn uniform_sampling_re_encodes_byte_identically() {
        let sampler = filled(0.5, 0xFACE_FACE, 33);
        let encoded = sampler.serialize_to_bytes().expect("serialize");
        let decoded = UniformSampling::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(encoded, decoded.serialize_to_bytes().expect("re-serialize"));
    }

    /// The payload carries the RNG position, so a decoded sampler fed the same
    /// later updates reaches the same state as the original fed those updates.
    #[test]
    fn uniform_sampling_rng_state_resumes_the_same_sequence() {
        let mut sampler = filled(0.3, 0x1234_5678, 25);
        let encoded = sampler.serialize_to_bytes().expect("serialize");
        let mut decoded = UniformSampling::deserialize_from_bytes(&encoded).expect("decode");

        for value in 100..140 {
            sampler.update(value as f64);
            decoded.update(value as f64);
        }
        assert_eq!(sampler.samples(), decoded.samples());
        assert_eq!(sampler.total_seen(), decoded.total_seen());
        assert_eq!(
            sampler.serialize_to_bytes().expect("re-serialize source"),
            decoded.serialize_to_bytes().expect("re-serialize decoded")
        );
    }

    /// An empty sampler round-trips and has exactly one encoding for a given
    /// rate and RNG position.
    #[test]
    fn uniform_sampling_empty_round_trip_has_exactly_one_encoding() {
        let empty = UniformSampling::with_seed(0.1, 0xABC1);
        assert!(empty.is_empty());

        let encoded = empty.serialize_to_bytes().expect("serialize empty");
        let decoded = UniformSampling::deserialize_from_bytes(&encoded).expect("decode empty");
        assert!(decoded.is_empty());
        assert_eq!(decoded.total_seen(), 0);
        assert_eq!(decoded.sample_rate(), 0.1);

        let twin = UniformSampling::with_seed(0.1, 0xABC1);
        assert_eq!(encoded, twin.serialize_to_bytes().expect("serialize twin"));
        assert_eq!(encoded, decoded.serialize_to_bytes().expect("re-serialize"));
    }

    /// A Count-Min envelope carries a different kind_id and must not decode as
    /// a sampler.
    #[test]
    fn uniform_sampling_rejects_foreign_kind_id() {
        let cms =
            crate::CountMin::<crate::Vector2D<i64>, crate::RegularPath>::with_dimensions(3, 8);
        let cms_bytes = cms.serialize_to_bytes().expect("serialize CMS");
        assert!(
            UniformSampling::deserialize_from_bytes(&cms_bytes).is_err(),
            "CMS bytes must not decode as a UniformSampling"
        );
    }

    /// Fail closed on an unexpected metadata key.
    #[test]
    fn us_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            sample_rate: f64,
            item_type: String,
            bogus_field: u8, // key not in UsMetadata
        }
        let m = us_metadata(0.25);
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            sample_rate: m.sample_rate,
            item_type: m.item_type.clone(),
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).unwrap();
        assert!(rmp_serde::from_slice::<UsMetadata>(&bytes).is_err());
    }

    /// `item_type` is required: a metadata map missing it does not decode, so
    /// the key cannot be silently defaulted.
    #[test]
    fn us_metadata_rejects_a_missing_item_type_key() {
        #[derive(Serialize)]
        struct WithoutItemType {
            metadata_version: u8,
            sample_rate: f64,
        }
        let m = us_metadata(0.25);
        let without = WithoutItemType {
            metadata_version: m.metadata_version,
            sample_rate: m.sample_rate,
        };
        let bytes = rmp_serde::to_vec_named(&without).unwrap();
        assert!(rmp_serde::from_slice::<UsMetadata>(&bytes).is_err());
    }

    /// Samples are `f64`; any other `item_type` name is rejected.
    #[test]
    fn us_metadata_rejects_a_foreign_item_type_name() {
        let meta = UsMetadata {
            metadata_version: 1,
            sample_rate: 0.5,
            item_type: "i64".to_string(),
        };
        let payload = UsPayload {
            priorities: vec![1],
            values: vec![3.0],
            total_seen: 2,
            rng_state: 9,
        };
        assert!(
            UniformSampling::deserialize_from_bytes(&crafted(&meta, &payload)).is_err(),
            "an i64-labelled envelope must not decode as an f64 sampler"
        );
    }

    /// A rate outside `(0, 1]` is rejected before it reaches `target_size`.
    #[test]
    fn uniform_sampling_rejects_an_out_of_range_sample_rate() {
        let payload = UsPayload {
            priorities: Vec::new(),
            values: Vec::new(),
            total_seen: 0,
            rng_state: 1,
        };
        for rate in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            let meta = us_metadata(rate);
            assert!(
                UniformSampling::deserialize_from_bytes(&crafted(&meta, &payload)).is_err(),
                "sample_rate {rate} must be rejected, not panic"
            );
        }
    }

    /// The declared stream length never sizes an allocation: a payload
    /// declaring `total_seen = u64::MAX` with two samples costs two samples.
    #[test]
    fn uniform_sampling_huge_declared_stream_costs_two_samples() {
        let meta = us_metadata(1.0);
        let payload = UsPayload {
            priorities: vec![7, 9],
            values: vec![1.0, 2.0],
            total_seen: u64::MAX,
            rng_state: 3,
        };
        let decoded = UniformSampling::deserialize_from_bytes(&crafted(&meta, &payload))
            .expect("a huge declared stream length is legal state");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.samples(), vec![1.0, 2.0]);
    }

    /// More retained samples than the rate allows for the declared stream
    /// length is not a state the algorithm reaches.
    #[test]
    fn uniform_sampling_rejects_more_samples_than_the_rate_allows() {
        let meta = us_metadata(0.5);
        let payload = UsPayload {
            priorities: vec![1, 2, 3],
            values: vec![1.0, 2.0, 3.0],
            total_seen: 2,
            rng_state: 5,
        };
        assert!(
            UniformSampling::deserialize_from_bytes(&crafted(&meta, &payload)).is_err(),
            "3 samples must not decode at total_seen 2 and rate 0.5"
        );
    }

    /// The two payload arrays are parallel; unequal lengths are rejected.
    #[test]
    fn uniform_sampling_rejects_parallel_array_length_mismatch() {
        let meta = us_metadata(1.0);
        let payload = UsPayload {
            priorities: vec![1, 2, 3],
            values: vec![1.0, 2.0],
            total_seen: 3,
            rng_state: 5,
        };
        assert!(UniformSampling::deserialize_from_bytes(&crafted(&meta, &payload)).is_err());
    }

    /// Entries are held in ascending priority; an unordered payload is
    /// rejected rather than decoded into a sampler that misplaces its next
    /// entry.
    #[test]
    fn uniform_sampling_rejects_unordered_priorities() {
        let meta = us_metadata(1.0);
        let payload = UsPayload {
            priorities: vec![9, 2, 5],
            values: vec![1.0, 2.0, 3.0],
            total_seen: 3,
            rng_state: 5,
        };
        assert!(UniformSampling::deserialize_from_bytes(&crafted(&meta, &payload)).is_err());

        // Equal priorities must still be ordered by `total_cmp` on the value.
        let tied = UsPayload {
            priorities: vec![4, 4],
            values: vec![8.0, 1.0],
            total_seen: 2,
            rng_state: 5,
        };
        assert!(UniformSampling::deserialize_from_bytes(&crafted(&meta, &tied)).is_err());
    }

    /// Truncated, foreign and garbage bytes are errors, never panics.
    #[test]
    fn uniform_sampling_rejects_crafted_bytes_without_panicking() {
        let encoded = filled(0.5, 0x99, 10)
            .serialize_to_bytes()
            .expect("serialize");
        for cut in [0, 1, 6, 9, 14, encoded.len() - 1] {
            assert!(UniformSampling::deserialize_from_bytes(&encoded[..cut]).is_err());
        }
        assert!(UniformSampling::deserialize_from_bytes(&[0xff; 64]).is_err());

        // Valid envelope + valid metadata, garbage payload.
        let metadata = rmp_serde::to_vec_named(&us_metadata(0.5)).unwrap();
        let bytes = envelope::encode(US_KIND, &metadata, &[0xc1, 0xc1, 0xc1]);
        assert!(UniformSampling::deserialize_from_bytes(&bytes).is_err());
    }

    /// The format never emits bytes it would refuse to read back: a sampler
    /// holding more entries than its own rate allows fails to serialize.
    #[test]
    fn uniform_sampling_rejects_serializing_an_over_full_sampler() {
        let meta = us_metadata(1.0);
        let payload = UsPayload {
            priorities: vec![1, 2],
            values: vec![1.0, 2.0],
            total_seen: 2,
            rng_state: 5,
        };
        let mut sampler = UniformSampling::deserialize_from_bytes(&crafted(&meta, &payload))
            .expect("two samples at total_seen 2");
        sampler.total_seen = 1;
        assert!(
            sampler.serialize_to_bytes().is_err(),
            "an over-full sampler must not serialize"
        );
    }
}
