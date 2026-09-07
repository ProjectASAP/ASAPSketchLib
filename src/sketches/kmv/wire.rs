//! ASAPv1 wire serialization for the KMV distinct-count sketch.
//!
//! Child submodule of [`crate::sketches::kmv`]: it holds ALL of KMV's
//! serialization (the metadata/payload DTOs, the kind_id constant, and the
//! `serialize_to_bytes` / `deserialize_from_bytes` impls) while the algorithm
//! lives in the parent module file. Being a descendant module, it reads the
//! sketch's private `_hasher` field and constructs the struct directly, without
//! widening any field visibility. See `docs/asapv1_wire_format.md`.
//!
//! KMV is one algorithm — a single kind_id `0x0e 0x00`. Its one structural
//! parameter is the retention bound `k`, which is construction config and lives
//! in the metadata, so the payload is the retained hashes alone: `[hashes]`, a
//! 1-element array. How many there are is `len(hashes)`, and the estimate, the
//! bound and the heap layout all follow from them, so nothing else is carried.
//!
//! ## Emitted order (byte-stable round trips)
//!
//! The heap's array order follows the sequence the keys arrived in and does not
//! survive a rebuild. The payload is therefore **order-defined**: `hashes`
//! ascends, strictly. Two sketches holding the same retained set emit the same
//! bytes whatever order they were inserted in, and re-serializing a decoded
//! sketch reproduces its bytes exactly.
//!
//! Retained values are the 64-bit digests `hash64_seeded` returns and travel at
//! that width, neither widened nor narrowed.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::message_pack_format::envelope;
use crate::{CommonHeap, HashProfile, KeepLargest, SketchHasher};

use super::KMV;

/// KMV kind_id: family `0x0e`, single algorithm variant `0x00`.
const KMV_KIND: &[u8] = &[0x0e, 0x00];

/// KMV descriptor metadata (ASAPv1 §2), a msgpack **map** (`to_vec_named`) with
/// keys in this declaration order — the canonical order the wire spec fixes (Go
/// must mirror it). Hash-spec fields first, then the one structural param `k`.
///
/// KMV hashes each key once, at the profile's canonical seed index, so it
/// carries `canonical_seed_index` exactly as HLL does.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KmvMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    canonical_seed_index: u32,
    k: u32,
}

/// Builds the KMV descriptor metadata from the hasher's [`HashProfile`], so the
/// wire bytes truthfully describe how the sketch was hashed (rather than
/// hardcoding the standard profile). `k` is the retention bound.
fn kmv_metadata<H: HashProfile>(k: u32) -> KmvMetadata {
    KmvMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        canonical_seed_index: H::CANONICAL_SEED_INDEX,
        k,
    }
}

/// KMV payload (ASAPv1 §3), a msgpack **array** (`to_vec`, positional):
/// `[hashes]` — a 1-element array. `hashes` holds the retained 64-bit digests
/// in strictly ascending order; `k` lives in the metadata.
#[derive(Debug, Serialize, Deserialize)]
struct KmvPayload {
    hashes: Vec<u64>,
}

/// The serde shape of [`CommonHeap`]'s own fields.
#[derive(Serialize)]
struct HeapSeed {
    data: Vec<u64>,
    size: usize,
    order: KeepLargest,
}

/// Rebuilds the retained heap through [`CommonHeap`]'s serde form, so the
/// backing vector is sized from the payload and `k` sizes no allocation. A
/// descending run already satisfies the max-heap invariant, so the hashes seat
/// as they are.
fn rebuild_heap(
    k: usize,
    ascending: Vec<u64>,
) -> Result<CommonHeap<u64, KeepLargest>, RmpDecodeError> {
    let len = ascending.len();
    let mut data = ascending;
    data.reverse();
    let seed = rmp_serde::to_vec_named(&HeapSeed {
        data,
        size: k,
        order: KeepLargest,
    })
    .map_err(|err| RmpDecodeError::Uncategorized(err.to_string()))?;
    let heap: CommonHeap<u64, KeepLargest> = from_slice(&seed)?;
    if heap.len() != len || heap.capacity() != k {
        return Err(RmpDecodeError::Uncategorized(format!(
            "KMV heap rebuild: {} of {len} hashes under a bound of {} against k {k}",
            heap.len(),
            heap.capacity()
        )));
    }
    Ok(heap)
}

// Wire serialization for KMV. `wire` is a descendant of the sketch module, so
// this impl constructs the struct with its private field directly.
impl<H: SketchHasher + HashProfile> KMV<H> {
    /// Serializes the sketch into an ASAPv1 MessagePack envelope
    /// (kind_id `0x0e 0x00`). The metadata is derived from the hasher's
    /// [`HashProfile`], so it truthfully describes how the sketch was hashed.
    ///
    /// Fails on a `k` of zero, on a `k` past the metadata's `u32` field, and on
    /// a retained set the bound does not cover — the states decode refuses. `k`
    /// and the heap's own bound are both public, so a `k` that disagrees with
    /// the bound it is meant to name fails too: emitting it would rebuild the
    /// sketch around a different eviction bound.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let k = u32::try_from(self.k).map_err(|_| {
            RmpEncodeError::Syntax(format!("KMV k {} exceeds the u32 metadata field", self.k))
        })?;
        if k == 0 {
            return Err(RmpEncodeError::Syntax(
                "KMV k must be at least 1".to_string(),
            ));
        }
        let hashes = self.wire_hashes();
        if hashes.len() > self.k {
            return Err(RmpEncodeError::Syntax(format!(
                "KMV holds {} hashes over a k of {}",
                hashes.len(),
                self.k
            )));
        }
        if hashes.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RmpEncodeError::Syntax(
                "KMV holds the same hash twice".to_string(),
            ));
        }
        if self.k_vals.capacity() != self.k {
            return Err(RmpEncodeError::Syntax(format!(
                "KMV k {} disagrees with the retained bound {}",
                self.k,
                self.k_vals.capacity()
            )));
        }
        let metadata = rmp_serde::to_vec_named(&kmv_metadata::<H>(k))?;
        let payload = rmp_serde::to_vec(&KmvPayload { hashes })?;
        Ok(envelope::encode(KMV_KIND, &metadata, &payload))
    }

    /// Deserializes a sketch from an ASAPv1 MessagePack envelope. `k` is
    /// structural (a property of the stored sketch, not of the target), so it is
    /// echoed back into the expected metadata; the hash spec is pinned against
    /// this target's profile.
    ///
    /// Every state the algorithm could not have produced is rejected with an
    /// error rather than a panic, and the declared `k` never sizes an
    /// allocation.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != KMV_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "KMV kind_id mismatch: stored {kind_id:?}, expected {KMV_KIND:?}"
            )));
        }
        let meta: KmvMetadata = from_slice(metadata)?;
        if meta != kmv_metadata::<H>(meta.k) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 KMV envelope: metadata mismatch".to_string(),
            ));
        }
        if meta.k == 0 {
            return Err(RmpDecodeError::Uncategorized(
                "KMV k must be at least 1".to_string(),
            ));
        }
        let k = meta.k as usize;
        let decoded: KmvPayload = from_slice(payload)?;
        if decoded.hashes.len() > k {
            return Err(RmpDecodeError::Uncategorized(format!(
                "KMV payload carries {} hashes over a k of {k}",
                decoded.hashes.len()
            )));
        }
        if decoded.hashes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RmpDecodeError::Uncategorized(
                "KMV hashes are not strictly ascending".to_string(),
            ));
        }
        Ok(KMV {
            k,
            k_vals: rebuild_heap(k, decoded.hashes)?,
            _hasher: PhantomData,
        })
    }

    /// The retained hashes in emitted order, ascending. Independent of the
    /// heap's array order, so the bytes are stable across a round trip.
    fn wire_hashes(&self) -> Vec<u64> {
        let mut hashes: Vec<u64> = self.k_vals.iter().copied().collect();
        hashes.sort_unstable();
        hashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, HeapItem};

    fn populated(k: usize, keys: u64) -> KMV {
        let mut sketch: KMV = KMV::new(k);
        for value in 0..keys {
            sketch.insert(&DataInput::U64(value));
        }
        sketch
    }

    fn metadata_of(bytes: &[u8]) -> KmvMetadata {
        let (_, metadata, _) = envelope::split(bytes).expect("split");
        from_slice(metadata).expect("metadata")
    }

    fn payload_of(bytes: &[u8]) -> KmvPayload {
        let (_, _, payload) = envelope::split(bytes).expect("split");
        from_slice(payload).expect("payload")
    }

    #[test]
    fn kmv_round_trip_serialization() {
        let mut sketch = populated(64, 5_000);
        assert_eq!(sketch.k_vals.len(), 64, "the bound was never reached");

        let encoded = sketch.serialize_to_bytes().expect("serialize KMV");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x0e, 0x00]); // kind_id_len=2, kind_id=[0x0e,0x00]

        let meta = metadata_of(&encoded);
        assert_eq!(meta.metadata_version, 1);
        assert_eq!(meta.k, 64);
        assert_eq!(meta.canonical_seed_index, CANONICAL_HASH_SEED as u32);

        let emitted = payload_of(&encoded);
        assert_eq!(emitted.hashes.len(), 64);
        for pair in emitted.hashes.windows(2) {
            assert!(pair[0] < pair[1], "the emitted hashes are not ascending");
        }

        let mut decoded = KMV::<DefaultXxHasher>::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(decoded.k, sketch.k);
        assert_eq!(decoded.k_vals.capacity(), sketch.k_vals.capacity());
        assert_eq!(decoded.wire_hashes(), sketch.wire_hashes());
        assert_eq!(
            decoded.k_vals.peek(),
            emitted.hashes.last(),
            "the rebuilt heap does not hold the largest hash at its root"
        );
        assert_eq!(decoded.estimate(), sketch.estimate());
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            encoded,
            "KMV bytes differed after a round trip"
        );
    }

    /// The emitted order is fixed by the hash value, not by the order the keys
    /// arrived in, so the same retained set always produces the same bytes.
    #[test]
    fn kmv_emitted_order_is_independent_of_insertion_order() {
        let keys: Vec<u64> = (0..2_000).collect();
        let mut forward: KMV = KMV::new(32);
        let mut backward: KMV = KMV::new(32);
        for value in &keys {
            forward.insert(&DataInput::U64(*value));
        }
        for value in keys.iter().rev() {
            backward.insert(&DataInput::U64(*value));
        }
        assert_eq!(forward.k_vals.len(), 32);
        assert_ne!(
            forward.k_vals.as_slice(),
            backward.k_vals.as_slice(),
            "the fixture cannot tell the emitted order from the heap order"
        );

        let bytes = forward.serialize_to_bytes().expect("serialize");
        assert_eq!(
            bytes,
            backward.serialize_to_bytes().expect("serialize"),
            "the emitted order followed the insertion order"
        );

        let decoded = KMV::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// A sketch that retains nothing has exactly one encoding: an empty
    /// `hashes` array beside its bound.
    #[test]
    fn kmv_empty_round_trip() {
        let sketch: KMV = KMV::new(16);
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        assert_eq!(metadata_of(&bytes).k, 16);
        assert!(payload_of(&bytes).hashes.is_empty());

        let mut decoded = KMV::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.k, 16);
        assert_eq!(decoded.k_vals.len(), 0);
        assert_eq!(decoded.k_vals.capacity(), 16);
        assert_eq!(decoded.estimate(), 0.0);
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// The retained values are 64-bit digests and are carried at that width.
    #[test]
    fn kmv_carries_hashes_at_full_u64_width() {
        let mut sketch: KMV = KMV::new(4);
        for value in [0u64, 1, u64::MAX / 2, u64::MAX] {
            sketch.insert_by_hash(value);
        }
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        assert_eq!(
            payload_of(&bytes).hashes,
            vec![0, 1, u64::MAX / 2, u64::MAX]
        );

        let decoded = KMV::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.wire_hashes(), vec![0, 1, u64::MAX / 2, u64::MAX]);
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// KMV and HLL are both single-hash cardinality sketches; a foreign
    /// envelope must not decode as a KMV.
    #[test]
    fn kmv_rejects_foreign_kind_id() {
        let cms =
            crate::CountMin::<crate::Vector2D<i64>, crate::RegularPath>::with_dimensions(3, 8);
        let cms_bytes = cms.serialize_to_bytes().expect("serialize CMS");
        let problem = KMV::<DefaultXxHasher>::deserialize_from_bytes(&cms_bytes)
            .expect_err("CMS bytes must not decode as a KMV")
            .to_string();
        assert!(problem.contains("kind_id mismatch"), "got {problem}");
    }

    /// Fail closed on an unexpected metadata key.
    #[test]
    fn kmv_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            canonical_seed_index: u32,
            k: u32,
            bogus_field: u8, // key not in KmvMetadata
        }
        let m = kmv_metadata::<DefaultXxHasher>(64);
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            canonical_seed_index: m.canonical_seed_index,
            k: m.k,
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).expect("encode");
        assert!(
            from_slice::<KmvMetadata>(&bytes).is_err(),
            "an unexpected metadata key must be rejected"
        );
    }

    /// `k` is required: a KMV metadata map missing it does not decode, so the
    /// key cannot be silently defaulted.
    #[test]
    fn kmv_metadata_rejects_a_missing_k_key() {
        #[derive(Serialize)]
        struct WithoutK {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            canonical_seed_index: u32,
        }
        let m = kmv_metadata::<DefaultXxHasher>(64);
        let without = WithoutK {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            canonical_seed_index: m.canonical_seed_index,
        };
        let bytes = rmp_serde::to_vec_named(&without).expect("encode");
        assert!(
            from_slice::<KmvMetadata>(&bytes).is_err(),
            "a missing metadata key must be rejected"
        );
    }

    /// Builds a well-framed envelope around a crafted payload.
    fn crafted(k: u32, hashes: Vec<u64>) -> Vec<u8> {
        let metadata = rmp_serde::to_vec_named(&kmv_metadata::<DefaultXxHasher>(k)).expect("meta");
        let payload = rmp_serde::to_vec(&KmvPayload { hashes }).expect("payload");
        envelope::encode(KMV_KIND, &metadata, &payload)
    }

    #[test]
    fn kmv_rejects_a_crafted_envelope() {
        let cases: Vec<(Vec<u8>, &str)> = vec![
            (crafted(0, Vec::new()), "k must be at least 1"),
            (crafted(2, vec![1, 2, 3]), "3 hashes over a k of 2"),
            (crafted(4, vec![3, 1, 2]), "not strictly ascending"),
            (crafted(4, vec![1, 1, 2]), "not strictly ascending"),
        ];
        for (bytes, expected) in cases {
            let problem = KMV::<DefaultXxHasher>::deserialize_from_bytes(&bytes)
                .expect_err("a crafted envelope must be rejected, not decoded")
                .to_string();
            assert!(
                problem.contains(expected),
                "expected a complaint about {expected}, got {problem}"
            );
        }
    }

    /// A payload whose array header declares far more hashes than it carries is
    /// rejected on the read, before anything is sized from the declared count.
    #[test]
    fn kmv_rejects_a_payload_declaring_more_hashes_than_it_carries() {
        let metadata = rmp_serde::to_vec_named(&kmv_metadata::<DefaultXxHasher>(4)).expect("meta");
        // fixarray(1), array32 declaring 2^30 elements, then two of them.
        let payload = vec![0x91, 0xdd, 0x40, 0x00, 0x00, 0x00, 0x01, 0x02];
        let bytes = envelope::encode(KMV_KIND, &metadata, &payload);
        assert!(
            KMV::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
            "an over-declared hash count must be rejected, not allocated"
        );
    }

    /// A declared `k` is metadata, never an allocation size: `u32::MAX` retained
    /// hashes would be tens of gigabytes if it sized the heap. The bound still
    /// governs the decoded sketch, so a further hash is appended, not evicted.
    #[test]
    fn kmv_does_not_allocate_a_declared_k() {
        let bytes = crafted(u32::MAX, vec![7, 9]);
        let mut decoded = KMV::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.k, u32::MAX as usize);
        assert_eq!(decoded.k_vals.len(), 2);
        assert_eq!(decoded.estimate(), 2.0);

        decoded.insert_by_hash(11);
        assert_eq!(decoded.k_vals.len(), 3, "the bound evicted instead");
        assert_eq!(decoded.wire_hashes(), vec![7, 9, 11]);
    }

    /// A `k` past the metadata's `u32` field fails the encode: truncating it
    /// would emit a `k` no decode accepts.
    #[test]
    fn kmv_refuses_a_k_the_metadata_cannot_carry() {
        let sketch: KMV = KMV {
            k: 1 << 40,
            k_vals: CommonHeap::new_max(1),
            _hasher: PhantomData,
        };
        let problem = sketch
            .serialize_to_bytes()
            .expect_err("an oversized k must not serialize")
            .to_string();
        assert!(
            problem.contains("exceeds the u32 metadata field"),
            "got {problem}"
        );
    }

    /// The encode side refuses the states decode refuses, so the format never
    /// emits bytes it would read back as invalid.
    #[test]
    fn kmv_refuses_to_serialize_a_state_decode_would_reject() {
        let empty: KMV = KMV::new(0);
        assert!(
            empty.serialize_to_bytes().is_err(),
            "a k of zero must not serialize"
        );

        let mut over: KMV = KMV {
            k: 1,
            k_vals: CommonHeap::new_max(4),
            _hasher: PhantomData,
        };
        over.insert_by_hash(3);
        over.insert_by_hash(5);
        let problem = over
            .serialize_to_bytes()
            .expect_err("a retained set past k must not serialize")
            .to_string();
        assert!(problem.contains("2 hashes over a k of 1"), "got {problem}");
    }

    /// `k` and the heap's bound are both public fields, so they can be set apart
    /// from outside the module. Emitting the declared `k` would rebuild the
    /// sketch around an eviction bound the source never had, in either
    /// direction, so the encode refuses both.
    #[test]
    fn kmv_refuses_a_k_that_disagrees_with_the_retained_bound() {
        let mut wider: KMV = KMV::new(4);
        wider.insert_by_hash(7);
        wider.k = 100;
        let problem = wider
            .serialize_to_bytes()
            .expect_err("a k over the retained bound must not serialize")
            .to_string();
        assert!(
            problem.contains("KMV k 100 disagrees with the retained bound 4"),
            "got {problem}"
        );

        let narrower: KMV = KMV {
            k: 1,
            k_vals: CommonHeap::new_max(4),
            _hasher: PhantomData,
        };
        let problem = narrower
            .serialize_to_bytes()
            .expect_err("a k under the retained bound must not serialize")
            .to_string();
        assert!(
            problem.contains("KMV k 1 disagrees with the retained bound 4"),
            "got {problem}"
        );
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`. KMV metadata is derived from the
    // profile, so an `AltHasher` sketch serializes truthfully. (An *unprofiled*
    // hasher cannot serialize at all — that is a compile-time guarantee, since
    // the wire methods are bounded on `H: HashProfile`.)
    #[derive(Clone, Debug)]
    struct AltHasher;

    impl SketchHasher for AltHasher {
        type HashType = <DefaultXxHasher as SketchHasher>::HashType;

        fn hash64_seeded(d: usize, key: &DataInput) -> u64 {
            DefaultXxHasher::hash64_seeded(d, key)
        }
        fn hash128_seeded(d: usize, key: &DataInput) -> u128 {
            DefaultXxHasher::hash128_seeded(d, key)
        }
        fn hash_item64_seeded(d: usize, key: &HeapItem) -> u64 {
            DefaultXxHasher::hash_item64_seeded(d, key)
        }
        fn hash_item128_seeded(d: usize, key: &HeapItem) -> u128 {
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

    #[test]
    fn kmv_custom_hasher_profile_round_trips_and_is_self_describing() {
        let mut alt: KMV<AltHasher> = KMV::new(16);
        let mut std: KMV = KMV::new(16);
        for value in 0..500u64 {
            alt.insert(&DataInput::U64(value));
            std.insert(&DataInput::U64(value));
        }

        // (a) A sketch built with a custom-profile hasher round-trips.
        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded = KMV::<AltHasher>::deserialize_from_bytes(&alt_bytes).expect("alt decode");
        assert_eq!(decoded.wire_hashes(), alt.wire_hashes());
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            alt_bytes
        );

        // (b) Bytes differ from the standard-profile sketch (the two hash
        // identically, so only the metadata separates them).
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_eq!(std.wire_hashes(), alt.wire_hashes());
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            KMV::<DefaultXxHasher>::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }
}
