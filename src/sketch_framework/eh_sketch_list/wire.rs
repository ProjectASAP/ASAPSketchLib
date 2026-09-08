//! ASAPv1 wire serialization for [`EHSketchList`], plus the sub-payload
//! [`ExponentialHistogram`] reuses.
//!
//! Child submodule of [`crate::sketch_framework::eh_sketch_list`]: it holds the
//! metadata/payload DTOs, the kind_id constants and the `serialize_to_bytes` /
//! `deserialize_from_bytes` impls.
//!
//! EHSketchList is one kind_id, `0x14 0x00`. It is a tagged union over ten
//! sketch algorithms, so the encoding is the triple
//! `[kind_id, descriptor, state]`.
//!
//! ## One encoding, used in both places
//!
//! [`sketch_state`] and [`rebuild_sketch`] are the only place an EHSketchList is
//! read or rebuilt. The standalone `0x14 0x00` payload is one triple; an
//! [`ExponentialHistogram`] bucket inlines the same triple.
//!
//! ## A nested variant is its own envelope minus the framing
//!
//! The triple carries the variant's own kind_id, metadata block and payload
//! block. The magic, version and length fields are stripped, so the variant's
//! own decoder validates the blocks it wrote.
//!
//! ## The kind_id namespace is fixed in every build
//!
//! All ten ids dispatch whatever features are on. A decoder built without
//! `experimental` rejects `0x0d 0x00` by name with the feature named, and its
//! encoder can never emit it.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::message_pack_format::envelope;

use super::EHSketchList;

/// EHSketchList kind_id: family `0x14`, single algorithm variant `0x00`.
pub(crate) const EH_SKETCH_LIST_KIND: &[u8] = &[0x14, 0x00];

/// Count-Min, the blocks [`EHSketchList::CM`] carries.
pub(crate) const CM_KIND: &[u8] = &[0x02, 0x00];
/// Coco, the blocks [`EHSketchList::COCO`] carries.
pub(crate) const COCO_KIND: &[u8] = &[0x0c, 0x00];
/// CountL2HH, the blocks [`EHSketchList::COUNTL2HH`] carries.
pub(crate) const COUNTL2HH_KIND: &[u8] = &[0x19, 0x00];
/// Count Sketch, the blocks [`EHSketchList::CS`] carries.
pub(crate) const CS_KIND: &[u8] = &[0x04, 0x00];
/// DDSketch, the blocks [`EHSketchList::DDS`] carries.
pub(crate) const DDS_KIND: &[u8] = &[0x05, 0x00];
/// Elastic, the blocks [`EHSketchList::ELASTIC`] carries.
pub(crate) const ELASTIC_KIND: &[u8] = &[0x0b, 0x00];
/// HLL Ertl-MLE, the blocks [`EHSketchList::HLL`] carries. Classic
/// (`0x01 0x01`) and HIP (`0x01 0x03`) are other algorithms.
pub(crate) const HLL_KIND: &[u8] = &[0x01, 0x02];
/// Compact KLL, the blocks [`EHSketchList::KLL`] carries. Dynamic KLL
/// (`0x06 0x01`) is another algorithm.
pub(crate) const KLL_KIND: &[u8] = &[0x06, 0x00];
/// UniformSampling, the blocks `EHSketchList::UNIFORM` carries.
pub(crate) const UNIFORM_KIND: &[u8] = &[0x0d, 0x00];
/// UnivMon, the blocks [`EHSketchList::UNIVMON`] carries.
pub(crate) const UNIVMON_KIND: &[u8] = &[0x10, 0x00];

/// The registry name of a nested kind_id. Error messages only: encoding and
/// decoding dispatch on the bytes.
pub(crate) fn variant_name(kind_id: &[u8]) -> Option<&'static str> {
    match kind_id {
        CM_KIND => Some("CountMin"),
        COCO_KIND => Some("Coco"),
        COUNTL2HH_KIND => Some("CountL2HH"),
        CS_KIND => Some("CountSketch"),
        DDS_KIND => Some("DDSketch"),
        ELASTIC_KIND => Some("Elastic"),
        HLL_KIND => Some("HLL"),
        KLL_KIND => Some("KLL"),
        UNIFORM_KIND => Some("UniformSampling"),
        UNIVMON_KIND => Some("UnivMon"),
        _ => None,
    }
}

/// EHSketchList descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`). The union has no construction config of its own: the
/// kind_id belongs to the triple, which travels whole.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EhSketchListMetadata {
    pub(crate) metadata_version: u8,
}

/// Builds the EHSketchList descriptor metadata.
pub(crate) fn eh_sketch_list_metadata() -> EhSketchListMetadata {
    EhSketchListMetadata {
        metadata_version: 1,
    }
}

/// One EHSketchList as a msgpack **array** (`to_vec`, positional):
/// `[kind_id, descriptor, state]`. All three are msgpack `bin`: the variant's
/// own kind_id bytes, its ASAPv1 metadata block and its ASAPv1 payload block.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SketchState {
    #[serde(with = "serde_bytes")]
    pub(crate) kind_id: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub(crate) descriptor: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub(crate) state: Vec<u8>,
}

/// Rejects a kind_id outside the ten.
pub(crate) fn unknown_variant(kind_id: &[u8]) -> RmpDecodeError {
    RmpDecodeError::Uncategorized(format!(
        "ASAPv1 EHSketchList: kind_id {kind_id:02x?} is not a wire variant"
    ))
}

/// Rejects the variant a build without `experimental` does not carry.
#[cfg(not(feature = "experimental"))]
fn check_variant_available(kind_id: &[u8]) -> Result<(), RmpDecodeError> {
    match kind_id {
        UNIFORM_KIND => Err(RmpDecodeError::Uncategorized(format!(
            "ASAPv1 EHSketchList: variant {} (kind_id {kind_id:02x?}) needs the experimental feature",
            variant_name(kind_id).unwrap_or("unnamed")
        ))),
        _ => Ok(()),
    }
}

/// Every variant is carried in an `experimental` build.
#[cfg(feature = "experimental")]
fn check_variant_available(_kind_id: &[u8]) -> Result<(), RmpDecodeError> {
    Ok(())
}

/// Splits one variant's own envelope into the triple, pinning its kind_id.
fn triple(kind: &'static [u8], bytes: &[u8]) -> Result<SketchState, RmpEncodeError> {
    let (kind_id, descriptor, state) = envelope::split(bytes).map_err(RmpEncodeError::Syntax)?;
    if kind_id != kind {
        return Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 EHSketchList: {} produced kind_id {kind_id:02x?}, expected {kind:02x?}",
            variant_name(kind).unwrap_or("variant")
        )));
    }
    Ok(SketchState {
        kind_id: kind.to_vec(),
        descriptor: descriptor.to_vec(),
        state: state.to_vec(),
    })
}

/// The triple one EHSketchList contributes, in the order the wire emits it.
/// Every state the variant's own encoder refuses is refused here too.
pub(crate) fn sketch_state(sketch: &EHSketchList) -> Result<SketchState, RmpEncodeError> {
    match sketch {
        EHSketchList::CM(s) => triple(CM_KIND, &s.serialize_to_bytes()?),
        EHSketchList::COCO(s) => triple(COCO_KIND, &s.serialize_to_bytes()?),
        EHSketchList::COUNTL2HH(s) => triple(COUNTL2HH_KIND, &s.serialize_to_bytes()?),
        EHSketchList::CS(s) => triple(CS_KIND, &s.serialize_to_bytes()?),
        EHSketchList::DDS(s) => triple(DDS_KIND, &s.serialize_to_bytes()?),
        EHSketchList::ELASTIC(s) => triple(ELASTIC_KIND, &s.serialize_to_bytes()?),
        EHSketchList::HLL(s) => triple(HLL_KIND, &s.serialize_to_bytes()?),
        EHSketchList::KLL(s) => triple(KLL_KIND, &s.serialize_to_bytes()?),
        #[cfg(feature = "experimental")]
        EHSketchList::UNIFORM(s) => triple(UNIFORM_KIND, &s.serialize_to_bytes()?),
        EHSketchList::UNIVMON(s) => triple(UNIVMON_KIND, &s.serialize_to_bytes()?),
    }
}

/// Rebuilds one EHSketchList from a triple, through the variant's own decoder.
/// An unknown kind_id, and one this build does not carry, are rejected before
/// any block is assembled.
pub(crate) fn rebuild_sketch(triple: &SketchState) -> Result<EHSketchList, RmpDecodeError> {
    let kind_id = triple.kind_id.as_slice();
    check_variant_available(kind_id)?;
    if variant_name(kind_id).is_none() {
        return Err(unknown_variant(kind_id));
    }
    let bytes = envelope::encode(kind_id, &triple.descriptor, &triple.state);
    match kind_id {
        CM_KIND => Ok(EHSketchList::CM(crate::CountMin::deserialize_from_bytes(
            &bytes,
        )?)),
        COCO_KIND => Ok(EHSketchList::COCO(crate::Coco::deserialize_from_bytes(
            &bytes,
        )?)),
        COUNTL2HH_KIND => Ok(EHSketchList::COUNTL2HH(
            crate::CountL2HH::deserialize_from_bytes(&bytes)?,
        )),
        CS_KIND => Ok(EHSketchList::CS(crate::Count::deserialize_from_bytes(
            &bytes,
        )?)),
        DDS_KIND => Ok(EHSketchList::DDS(crate::DDSketch::deserialize_from_bytes(
            &bytes,
        )?)),
        ELASTIC_KIND => Ok(EHSketchList::ELASTIC(
            crate::Elastic::deserialize_from_bytes(&bytes)?,
        )),
        HLL_KIND => Ok(EHSketchList::HLL(
            crate::HyperLogLog::<crate::ErtlMLE>::deserialize_from_bytes(&bytes)?,
        )),
        KLL_KIND => Ok(EHSketchList::KLL(crate::KLL::deserialize_from_bytes(
            &bytes,
        )?)),
        #[cfg(feature = "experimental")]
        UNIFORM_KIND => Ok(EHSketchList::UNIFORM(
            crate::UniformSampling::deserialize_from_bytes(&bytes)?,
        )),
        UNIVMON_KIND => Ok(EHSketchList::UNIVMON(
            crate::UnivMon::deserialize_from_bytes(&bytes)?,
        )),
        other => Err(unknown_variant(other)),
    }
}

// Wire serialization for EHSketchList. `wire` is a descendant of the union's
// module, so this impl matches its variants directly.
impl EHSketchList {
    /// Serializes the sketch into an ASAPv1 MessagePack envelope
    /// (kind_id `0x14 0x00`). The payload is the triple
    /// `[kind_id, descriptor, state]`.
    ///
    /// A state the variant's own encoder refuses is an error rather than bytes
    /// that would be refused on decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let metadata = rmp_serde::to_vec_named(&eh_sketch_list_metadata())?;
        let payload = rmp_serde::to_vec(&sketch_state(self)?)?;
        Ok(envelope::encode(EH_SKETCH_LIST_KIND, &metadata, &payload))
    }

    /// Deserializes a sketch from an ASAPv1 MessagePack envelope. The nested
    /// kind_id selects the decoder, which validates its own blocks.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != EH_SKETCH_LIST_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "EHSketchList kind_id mismatch: stored {kind_id:?}, expected {EH_SKETCH_LIST_KIND:?}"
            )));
        }
        let meta: EhSketchListMetadata = from_slice(metadata)?;
        if meta != eh_sketch_list_metadata() {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 EHSketchList envelope: metadata mismatch".to_string(),
            ));
        }
        let triple: SketchState = from_slice(payload)?;
        rebuild_sketch(&triple)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::sketch_framework::univmon::UnivMon;
    use crate::{
        CANONICAL_HASH_SEED, Count, CountL2HH, CountMin, DDSketch, DataInput, DefaultXxHasher,
        ErtlMLE, FastPath, HashProfile, HyperLogLog, KLL, SketchHasher, Vector2D,
    };

    /// An input the named variant accepts. UnivMon's heaps hold one `HeapItem`
    /// variant at a time, so its keys stay strings.
    pub(crate) fn sample_input(sketch_type: &str) -> DataInput<'static> {
        match sketch_type {
            "Coco" | "Elastic" | "UnivMon" => DataInput::Str("flow::a"),
            _ => DataInput::F64(7.5),
        }
    }

    /// One populated sketch per variant this build carries, in kind_id order.
    pub(crate) fn populated_variants() -> Vec<EHSketchList> {
        let mut out = vec![EHSketchList::CM(
            CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 8),
        )];
        out.push(EHSketchList::COCO(crate::Coco::init_with_size(16, 2)));
        out.push(EHSketchList::COUNTL2HH(CountL2HH::with_dimensions(3, 256)));
        out.push(EHSketchList::CS(
            Count::<Vector2D<i32>, FastPath>::with_dimensions(3, 8),
        ));
        out.push(EHSketchList::DDS(DDSketch::new(0.01)));
        out.push(EHSketchList::ELASTIC(crate::Elastic::init_with_dimensions(
            8, 2, 256,
        )));
        out.push(EHSketchList::HLL(HyperLogLog::<ErtlMLE>::default()));
        out.push(EHSketchList::KLL(KLL::init_kll(200)));
        #[cfg(feature = "experimental")]
        out.push(EHSketchList::UNIFORM(crate::UniformSampling::new(0.5)));
        out.push(EHSketchList::UNIVMON(UnivMon::default()));
        for sketch in &mut out {
            let key = sample_input(sketch.sketch_type());
            for _ in 0..5 {
                sketch.insert(&key);
            }
        }
        out
    }

    /// A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    /// declares a DIFFERENT `HashProfile`.
    #[derive(Clone, Debug)]
    pub(crate) struct AltHasher;

    impl SketchHasher for AltHasher {
        type HashType = <DefaultXxHasher as SketchHasher>::HashType;

        fn hash64_seeded(d: usize, key: &DataInput) -> u64 {
            DefaultXxHasher::hash64_seeded(d, key)
        }
        fn hash128_seeded(d: usize, key: &DataInput) -> u128 {
            DefaultXxHasher::hash128_seeded(d, key)
        }
        fn hash_item64_seeded(d: usize, key: &crate::HeapItem) -> u64 {
            DefaultXxHasher::hash_item64_seeded(d, key)
        }
        fn hash_item128_seeded(d: usize, key: &crate::HeapItem) -> u128 {
            DefaultXxHasher::hash_item128_seeded(d, key)
        }
        fn hash_for_matrix_seeded(
            seed_idx: usize,
            rows: usize,
            cols: usize,
            key: &DataInput,
        ) -> Self::HashType {
            DefaultXxHasher::hash_for_matrix_seeded(seed_idx, rows, cols, key)
        }
    }

    impl HashProfile for AltHasher {
        const PROFILE_ID: &'static str = "test.alt.profile.v1";
        const ALGORITHM: &'static str = "xxh3_64_128";
        const SEED_DERIVATION: &'static str = "seed_list_index_wrap";
        const INPUT_ENCODING: &'static str = "projectasap.input.v1";
        fn seed_list() -> Vec<u64> {
            vec![1, 2, 3, 4, 5]
        }
        const CANONICAL_SEED_INDEX: u32 = CANONICAL_HASH_SEED as u32;
        const MATRIX_SEED_INDEX: u32 = 0;
    }

    /// A Count-Min triple whose descriptor names a custom hash profile.
    pub(crate) fn alt_profile_triple() -> SketchState {
        let sketch = CountMin::<Vector2D<i32>, FastPath, AltHasher>::with_dimensions(3, 8);
        let bytes = sketch.serialize_to_bytes().expect("serialize alt CM");
        let (_, descriptor, state) = envelope::split(&bytes).expect("split alt CM");
        SketchState {
            kind_id: CM_KIND.to_vec(),
            descriptor: descriptor.to_vec(),
            state: state.to_vec(),
        }
    }

    /// The triple a variant contributes, with the nested kind_id replaced.
    pub(crate) fn relabelled(sketch: &EHSketchList, kind_id: &[u8]) -> SketchState {
        let mut triple = sketch_state(sketch).expect("state");
        triple.kind_id = kind_id.to_vec();
        triple
    }

    /// Wraps a triple in a `0x14 0x00` envelope.
    pub(crate) fn envelope_for(triple: &SketchState) -> Vec<u8> {
        let metadata = rmp_serde::to_vec_named(&eh_sketch_list_metadata()).unwrap();
        let payload = rmp_serde::to_vec(triple).unwrap();
        envelope::encode(EH_SKETCH_LIST_KIND, &metadata, &payload)
    }

    #[test]
    fn eh_sketch_list_round_trip_serialization() {
        let mut sketch =
            EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 8));
        sketch.insert(&DataInput::U64(42));

        let encoded = sketch.serialize_to_bytes().expect("serialize EHSketchList");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x14, 0x00]); // kind_id_len=2, kind_id=[0x14,0x00]

        let decoded = EHSketchList::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(decoded.sketch_type(), "CountMin");
        assert_eq!(
            decoded.query(&DataInput::U64(42)),
            sketch.query(&DataInput::U64(42))
        );
    }

    /// Every variant this build carries round-trips, keeps its type, and
    /// re-serializes byte-identically.
    #[test]
    fn eh_sketch_list_every_variant_round_trips() {
        for sketch in populated_variants() {
            let encoded = sketch
                .serialize_to_bytes()
                .unwrap_or_else(|e| panic!("serialize {}: {e}", sketch.sketch_type()));
            let decoded = EHSketchList::deserialize_from_bytes(&encoded)
                .unwrap_or_else(|e| panic!("deserialize {}: {e}", sketch.sketch_type()));
            assert_eq!(decoded.sketch_type(), sketch.sketch_type());
            let again = decoded.serialize_to_bytes().expect("re-serialize");
            assert_eq!(
                encoded,
                again,
                "{} is not byte-stable",
                sketch.sketch_type()
            );
        }
    }

    /// The ten nested kind_ids and their registry names are the same in every
    /// build, and every variant emits the id the table gives it.
    #[test]
    fn eh_sketch_list_kind_ids_are_build_independent() {
        let table: [(&[u8], &str); 10] = [
            (CM_KIND, "CountMin"),
            (COCO_KIND, "Coco"),
            (COUNTL2HH_KIND, "CountL2HH"),
            (CS_KIND, "CountSketch"),
            (DDS_KIND, "DDSketch"),
            (ELASTIC_KIND, "Elastic"),
            (HLL_KIND, "HLL"),
            (KLL_KIND, "KLL"),
            (UNIFORM_KIND, "UniformSampling"),
            (UNIVMON_KIND, "UnivMon"),
        ];
        for (kind_id, name) in table {
            assert_eq!(variant_name(kind_id), Some(name), "{kind_id:02x?}");
        }
        assert_eq!(variant_name(&[0xff, 0xff]), None);

        for sketch in populated_variants() {
            let emitted = sketch_state(&sketch).expect("state").kind_id;
            let expected = table
                .iter()
                .find(|(_, name)| *name == sketch.sketch_type())
                .expect("variant in table")
                .0;
            assert_eq!(emitted, expected, "{}", sketch.sketch_type());
        }
    }

    /// The experimental kind_id is rejected without the feature, with an error
    /// naming the variant. Crafted bytes, so the test runs in both builds.
    #[test]
    fn eh_sketch_list_rejects_an_experimental_kind_id() {
        let triple = SketchState {
            kind_id: UNIFORM_KIND.to_vec(),
            descriptor: Vec::new(),
            state: Vec::new(),
        };
        let message = EHSketchList::deserialize_from_bytes(&envelope_for(&triple))
            .expect_err("an empty variant state must not decode")
            .to_string();
        assert!(!message.is_empty());
        #[cfg(not(feature = "experimental"))]
        {
            assert!(message.contains("UniformSampling"), "{message}");
            assert!(message.contains("experimental"), "{message}");
        }
    }

    #[test]
    fn eh_sketch_list_rejects_an_unknown_kind_id() {
        let sketch = EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 8));
        let triple = relabelled(&sketch, &[0xff, 0xff]);
        let message = EHSketchList::deserialize_from_bytes(&envelope_for(&triple))
            .expect_err("an unknown kind_id must not decode")
            .to_string();
        assert!(message.contains("not a wire variant"), "{message}");
    }

    /// A nested kind_id longer than the envelope's one-byte length field is
    /// rejected as unknown, before any block is assembled.
    #[test]
    fn eh_sketch_list_rejects_an_over_long_kind_id() {
        for len in [256usize, 4096] {
            let triple = SketchState {
                kind_id: vec![0x14; len],
                descriptor: Vec::new(),
                state: Vec::new(),
            };
            let message = EHSketchList::deserialize_from_bytes(&envelope_for(&triple))
                .expect_err("an over-long kind_id must not decode")
                .to_string();
            assert!(message.contains("not a wire variant"), "{message}");
        }
    }

    /// The nested ids are pinned to one algorithm each: HLL is Ertl-MLE and KLL
    /// is compact, so their siblings' ids are unknown here.
    #[test]
    fn eh_sketch_list_rejects_sibling_algorithm_kind_ids() {
        let hll = EHSketchList::HLL(HyperLogLog::<ErtlMLE>::default());
        for sibling in [[0x01u8, 0x01], [0x01, 0x03]] {
            let triple = relabelled(&hll, &sibling);
            assert!(
                EHSketchList::deserialize_from_bytes(&envelope_for(&triple)).is_err(),
                "{sibling:02x?} must not decode as the HLL arm"
            );
        }
        let kll = EHSketchList::KLL(KLL::init_kll(200));
        let triple = relabelled(&kll, &[0x06, 0x01]);
        assert!(EHSketchList::deserialize_from_bytes(&envelope_for(&triple)).is_err());
    }

    /// A kind_id that does not match the blocks it carries is rejected by the
    /// variant's own decoder.
    #[test]
    fn eh_sketch_list_rejects_a_mismatched_kind_id_and_descriptor() {
        let sketch = EHSketchList::HLL(HyperLogLog::<ErtlMLE>::default());
        let triple = relabelled(&sketch, CM_KIND);
        assert!(EHSketchList::deserialize_from_bytes(&envelope_for(&triple)).is_err());
    }

    /// A Count-Min envelope and an ExponentialHistogram envelope are not
    /// EHSketchList envelopes.
    #[test]
    fn eh_sketch_list_rejects_foreign_kind_ids() {
        let cms = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 8);
        let cms_bytes = cms.serialize_to_bytes().expect("serialize CMS");
        assert!(EHSketchList::deserialize_from_bytes(&cms_bytes).is_err());

        let eh = crate::ExponentialHistogram::new(
            2,
            100,
            EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 8)),
        );
        let eh_bytes = eh.serialize_to_bytes().expect("serialize EH");
        assert!(EHSketchList::deserialize_from_bytes(&eh_bytes).is_err());
    }

    /// Fail closed on an unexpected metadata key.
    #[test]
    fn eh_sketch_list_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            bogus_field: u8,
        }
        let bytes = rmp_serde::to_vec_named(&WithExtra {
            metadata_version: 1,
            bogus_field: 7,
        })
        .unwrap();
        assert!(rmp_serde::from_slice::<EhSketchListMetadata>(&bytes).is_err());
    }

    /// `metadata_version` is required: a map missing it does not decode.
    #[test]
    fn eh_sketch_list_metadata_rejects_a_missing_key() {
        #[derive(Serialize)]
        struct Empty {}
        let bytes = rmp_serde::to_vec_named(&Empty {}).unwrap();
        assert!(rmp_serde::from_slice::<EhSketchListMetadata>(&bytes).is_err());
    }

    /// Crafted blocks fail closed with an error, never a panic.
    #[test]
    fn eh_sketch_list_rejects_crafted_blocks() {
        let sketch = EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 8));
        let good = sketch_state(&sketch).expect("state");

        let truncated = SketchState {
            kind_id: CM_KIND.to_vec(),
            descriptor: good.descriptor[..good.descriptor.len() / 2].to_vec(),
            state: good.state.clone(),
        };
        assert!(EHSketchList::deserialize_from_bytes(&envelope_for(&truncated)).is_err());

        let garbage = SketchState {
            kind_id: CM_KIND.to_vec(),
            descriptor: good.descriptor.clone(),
            state: vec![0xc1, 0xc1, 0xc1],
        };
        assert!(EHSketchList::deserialize_from_bytes(&envelope_for(&garbage)).is_err());
    }

    /// A descriptor naming a custom hash profile is rejected: the variant's
    /// decoder pins the profile of the type it rebuilds.
    #[test]
    fn eh_sketch_list_rejects_a_custom_hash_profile_descriptor() {
        let triple = alt_profile_triple();
        assert!(EHSketchList::deserialize_from_bytes(&envelope_for(&triple)).is_err());
    }

    /// A decoded union answers a query the way the original did.
    #[test]
    fn eh_sketch_list_query_agrees_after_decode() {
        for sketch in populated_variants() {
            let key = sample_input(sketch.sketch_type());
            let encoded = sketch.serialize_to_bytes().expect("serialize");
            let decoded = EHSketchList::deserialize_from_bytes(&encoded).expect("deserialize");
            assert_eq!(
                sketch.query(&key).ok(),
                decoded.query(&key).ok(),
                "{} disagrees after decode",
                sketch.sketch_type()
            );
        }
    }
}
