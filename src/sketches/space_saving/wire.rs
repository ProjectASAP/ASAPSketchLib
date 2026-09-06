//! ASAPv1 wire serialization for the Space-Saving summary.
//!
//! Child submodule of [`crate::sketches::space_saving`]: it holds ALL of
//! Space-Saving's serialization (the metadata/payload DTOs, the kind_id
//! constant, the `key_type` mapping, and the `serialize_to_bytes` /
//! `deserialize_from_bytes` impls) while the algorithm lives in the parent
//! module file. Being a descendant module, it reads the summary's private
//! `counters` / `buckets` / `total` / `floor` fields and reuses the private
//! `rebuild` entry point directly, without widening any field visibility. See
//! `docs/asapv1_wire_format.md` §3.5.
//!
//! Space-Saving is one algorithm — a single kind_id `0x18 0x00`. Its structural
//! parameters are the counter `capacity` and the `key_type`, both in the
//! metadata, so the payload is the answer triples split into parallel arrays
//! plus the two running scalars: `[keys, counts, errors, total, floor]`. The
//! bucket list, the counter arena and the key index are all derived and are
//! rebuilt on load, so no arena index reaches the wire.
//!
//! ## `key_type` names the exact `HeapItem` variant
//!
//! Unlike Count-Min's counters, a Space-Saving key's type is a **runtime**
//! property: keys are [`HeapItem`]s, a 16-variant enum. One metadata `key_type`
//! names the exact variant and the payload's `keys` array is homogeneous in it.
//! The variant is never widened (an `i32` key is `"i32"`, not `"i64"`) because
//! `HeapItem`'s `PartialEq<DataInput>` and `Hash` both discriminate on the
//! variant: a widened key would hash to the same digest but stop comparing
//! equal to the caller's `DataInput`, so `estimate` would silently read zero.
//! `HeapItem::I128` / `U128` have no msgpack integer form and are not wire
//! types, so a summary holding one refuses to serialize — as does one whose
//! monitored keys mix variants, `String` and `Bytes` included. A `Bytes` key is
//! written as msgpack `bin` through `WireBytes` and read back from `bin` alone,
//! so any byte string survives whether or not it is UTF-8 and a `str`-keyed
//! payload relabelled `"bytes"` is refused.
//!
//! ## Emitted order (byte-stable round trips)
//!
//! `entries()` order follows the counter arena and `top_k` order follows the
//! bucket walk, neither of which survives a rebuild. The payload is therefore
//! **order-defined**: descending count, ties broken by the crate's `key_order`
//! total order over [`HeapItem`] (the same one `merge` uses). Two summaries
//! holding the same triples emit the same bytes whatever order they were seated
//! in, and re-serializing a decoded summary reproduces its bytes exactly.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::message_pack_format::envelope;
use crate::message_pack_format::wire_key::WireBytes;
use crate::{HashProfile, HeapItem, SketchHasher};

use super::{SpaceSaving, SpaceSavingState, key_order};

/// Space-Saving kind_id: family `0x18`, single algorithm variant `0x00`.
const SPACE_SAVING_KIND: &[u8] = &[0x18, 0x00];

/// Metadata `key_type` of a summary that monitors nothing. A summary with no
/// keys has no variant to report, so the wire pins one rather than leaving the
/// field free: empty summaries have one encoding.
const EMPTY_KEY_TYPE: &str = "u64";

/// The metadata `key_type` naming a key's exact [`HeapItem`] variant, or `None`
/// for the 128-bit variants, which are not wire types.
fn key_type_of(key: &HeapItem) -> Option<&'static str> {
    Some(match key {
        HeapItem::I8(_) => "i8",
        HeapItem::I16(_) => "i16",
        HeapItem::I32(_) => "i32",
        HeapItem::I64(_) => "i64",
        HeapItem::ISIZE(_) => "isize",
        HeapItem::U8(_) => "u8",
        HeapItem::U16(_) => "u16",
        HeapItem::U32(_) => "u32",
        HeapItem::U64(_) => "u64",
        HeapItem::USIZE(_) => "usize",
        HeapItem::F32(_) => "f32",
        HeapItem::F64(_) => "f64",
        HeapItem::String(_) => "string",
        HeapItem::Bytes(_) => "bytes",
        HeapItem::I128(_) | HeapItem::U128(_) => return None,
    })
}

/// Space-Saving descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it). Hash-spec fields first, then the
/// structural params `capacity` / `key_type`.
///
/// There is **no seed-index key**: Space-Saving hashes every key with
/// `hash64_seeded(0, ..)`, a fixed part of the algorithm rather than a profile
/// choice, so no `HashProfile` index field would describe it truthfully.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpaceSavingMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    capacity: u32,
    key_type: String,
}

/// Builds the Space-Saving descriptor metadata from the hasher's
/// [`HashProfile`], so the wire bytes truthfully describe how the summary
/// hashed its keys. `capacity` is the counter budget; `key_type` is the exact
/// [`HeapItem`] variant the payload's `keys` array carries.
fn space_saving_metadata<H: HashProfile>(capacity: u32, key_type: &str) -> SpaceSavingMetadata {
    SpaceSavingMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        capacity,
        key_type: key_type.to_string(),
    }
}

/// Space-Saving payload (ASAPv1 §3.5), a msgpack **array** (`to_vec`,
/// positional): `[keys, counts, errors, total, floor]`. The three arrays are
/// parallel and equal-length; `keys`'s element type is fixed by the metadata
/// `key_type`. `total` is the recorded weight, `floor` the largest count known
/// to have left the summary.
#[derive(Debug, Serialize, Deserialize)]
struct SpaceSavingPayload<K> {
    keys: Vec<K>,
    counts: Vec<u64>,
    errors: Vec<u64>,
    total: u64,
    floor: u64,
}

/// The `key_type` the payload will be written in, taken from the first key in
/// emitted order. Rejects a 128-bit key outright; the homogeneity of the rest
/// is enforced by [`encode_payload`].
fn wire_key_type(entries: &[(&HeapItem, u64, u64)]) -> Result<&'static str, RmpEncodeError> {
    match entries.first() {
        None => Ok(EMPTY_KEY_TYPE),
        Some((key, _, _)) => key_type_of(key).ok_or_else(|| {
            RmpEncodeError::Syntax(
                "Space-Saving: 128-bit keys are not ASAPv1 wire types".to_string(),
            )
        }),
    }
}

fn mixed_variant_error(key_type: &str, key: &HeapItem) -> RmpEncodeError {
    RmpEncodeError::Syntax(format!(
        "Space-Saving: monitored keys mix variants — key_type is {key_type}, but a {} key is held",
        key_type_of(key).unwrap_or("128-bit")
    ))
}

/// Packs the emitted triples into the positional payload, with `keys` typed by
/// `key_type`. Any key that is not of that variant fails the encode.
fn encode_payload(
    key_type: &str,
    entries: &[(&HeapItem, u64, u64)],
    total: u64,
    floor: u64,
) -> Result<Vec<u8>, RmpEncodeError> {
    let counts: Vec<u64> = entries.iter().map(|entry| entry.1).collect();
    let errors: Vec<u64> = entries.iter().map(|entry| entry.2).collect();

    macro_rules! pack {
        ($variant:ident) => {{
            let mut keys = Vec::with_capacity(entries.len());
            for (key, _, _) in entries {
                match key {
                    HeapItem::$variant(value) => keys.push(*value),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&SpaceSavingPayload {
                keys,
                counts,
                errors,
                total,
                floor,
            })
        }};
    }

    match key_type {
        "i8" => pack!(I8),
        "i16" => pack!(I16),
        "i32" => pack!(I32),
        "i64" => pack!(I64),
        "isize" => pack!(ISIZE),
        "u8" => pack!(U8),
        "u16" => pack!(U16),
        "u32" => pack!(U32),
        "u64" => pack!(U64),
        "usize" => pack!(USIZE),
        "f32" => pack!(F32),
        "f64" => pack!(F64),
        "string" => {
            let mut keys = Vec::with_capacity(entries.len());
            for (key, _, _) in entries {
                match key {
                    HeapItem::String(value) => keys.push(value.clone()),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&SpaceSavingPayload {
                keys,
                counts,
                errors,
                total,
                floor,
            })
        }
        "bytes" => {
            let mut keys = Vec::with_capacity(entries.len());
            for (key, _, _) in entries {
                match key {
                    HeapItem::Bytes(value) => keys.push(WireBytes(value.clone())),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&SpaceSavingPayload {
                keys,
                counts,
                errors,
                total,
                floor,
            })
        }
        other => Err(RmpEncodeError::Syntax(format!(
            "Space-Saving key_type {other:?} is not an ASAPv1 wire key type"
        ))),
    }
}

/// Reads the payload with `keys` typed by the metadata `key_type` and assembles
/// the state the in-memory rebuild consumes. Rejects an unknown `key_type` and
/// parallel arrays of unequal length; a `keys` array whose msgpack types do not
/// match `key_type` fails in `from_slice`.
fn decode_payload(
    key_type: &str,
    capacity: usize,
    payload: &[u8],
) -> Result<SpaceSavingState, RmpDecodeError> {
    macro_rules! unpack {
        ($variant:ident, $ty:ty) => {{
            let decoded: SpaceSavingPayload<$ty> = from_slice(payload)?;
            (
                decoded
                    .keys
                    .into_iter()
                    .map(HeapItem::$variant)
                    .collect::<Vec<HeapItem>>(),
                decoded.counts,
                decoded.errors,
                decoded.total,
                decoded.floor,
            )
        }};
    }

    let (keys, counts, errors, total, floor) = match key_type {
        "i8" => unpack!(I8, i8),
        "i16" => unpack!(I16, i16),
        "i32" => unpack!(I32, i32),
        "i64" => unpack!(I64, i64),
        "isize" => unpack!(ISIZE, isize),
        "u8" => unpack!(U8, u8),
        "u16" => unpack!(U16, u16),
        "u32" => unpack!(U32, u32),
        "u64" => unpack!(U64, u64),
        "usize" => unpack!(USIZE, usize),
        "f32" => unpack!(F32, f32),
        "f64" => unpack!(F64, f64),
        "string" => unpack!(String, String),
        "bytes" => {
            let decoded: SpaceSavingPayload<WireBytes> = from_slice(payload)?;
            (
                decoded
                    .keys
                    .into_iter()
                    .map(|key| HeapItem::Bytes(key.into_vec()))
                    .collect::<Vec<HeapItem>>(),
                decoded.counts,
                decoded.errors,
                decoded.total,
                decoded.floor,
            )
        }
        other => {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Space-Saving key_type {other:?} is not an ASAPv1 wire key type"
            )));
        }
    };

    if keys.len() != counts.len() || keys.len() != errors.len() {
        return Err(RmpDecodeError::Uncategorized(format!(
            "Space-Saving payload: {} keys against {} counts and {} errors",
            keys.len(),
            counts.len(),
            errors.len()
        )));
    }

    Ok(SpaceSavingState {
        capacity,
        total,
        floor,
        entries: keys
            .into_iter()
            .zip(counts)
            .zip(errors)
            .map(|((key, count), error)| (key, count, error))
            .collect(),
    })
}

// Wire serialization for Space-Saving. `wire` is a descendant of the sketch
// module, so this impl reads the private fields and calls the private rebuild
// directly.
impl<H: SketchHasher + HashProfile> SpaceSaving<H> {
    /// Serializes the summary into an ASAPv1 MessagePack envelope
    /// (kind_id `0x18 0x00`). The metadata is derived from the hasher's
    /// [`HashProfile`], so it truthfully describes how the keys were hashed.
    ///
    /// Fails when the monitored keys mix [`HeapItem`] variants, when any of
    /// them is 128-bit (no msgpack integer form), or when `capacity` overflows
    /// the metadata's `u32` field.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let capacity = u32::try_from(self.capacity).map_err(|_| {
            RmpEncodeError::Syntax(format!(
                "Space-Saving capacity {} exceeds the u32 metadata field",
                self.capacity
            ))
        })?;
        let entries = self.wire_entries();
        let key_type = wire_key_type(&entries)?;
        let metadata = rmp_serde::to_vec_named(&space_saving_metadata::<H>(capacity, key_type))?;
        let payload = encode_payload(key_type, &entries, self.total, self.floor)?;
        Ok(envelope::encode(SPACE_SAVING_KIND, &metadata, &payload))
    }

    /// Deserializes a summary from an ASAPv1 MessagePack envelope. `capacity`
    /// and `key_type` are structural (they are properties of the stored
    /// summary, not of the target), so they are echoed back into the expected
    /// metadata; the hash spec is pinned against this target's profile.
    ///
    /// Every state the algorithm could not have produced is rejected with an
    /// error rather than a panic, and the declared `capacity` never sizes an
    /// allocation.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != SPACE_SAVING_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Space-Saving kind_id mismatch: stored {kind_id:?}, expected {SPACE_SAVING_KIND:?}"
            )));
        }
        let meta: SpaceSavingMetadata = from_slice(metadata)?;
        if meta != space_saving_metadata::<H>(meta.capacity, &meta.key_type) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 Space-Saving envelope: metadata mismatch".to_string(),
            ));
        }
        let state = decode_payload(&meta.key_type, meta.capacity as usize, payload)?;
        Self::rebuild(state).map_err(RmpDecodeError::Uncategorized)
    }

    /// The monitored triples in emitted order: descending count, ties broken by
    /// `key_order`. Independent of the arena's seat order, so the bytes are
    /// stable across a round trip.
    fn wire_entries(&self) -> Vec<(&HeapItem, u64, u64)> {
        let mut entries: Vec<(&HeapItem, u64, u64)> = self
            .counters
            .iter()
            .map(|counter| {
                (
                    &counter.key,
                    self.buckets[counter.bucket].count,
                    counter.error,
                )
            })
            .collect();
        entries.sort_unstable_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| key_order(a.0).cmp(&key_order(b.0)))
        });
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher};

    fn next_random(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A log-uniform (Zipf-ish) stream over `1..=distinct`, weighted 1..=3.
    fn zipfish(capacity: usize, distinct: i64, steps: usize, seed: u64) -> SpaceSaving {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(capacity);
        let mut state = seed;
        for _ in 0..steps {
            let draw = next_random(&mut state);
            let unit = (draw >> 11) as f64 / (1u64 << 53) as f64;
            let key = (distinct as f64).powf(unit) as i64;
            summary.insert_many(&DataInput::I64(key), 1 + draw % 3);
        }
        summary
    }

    fn metadata_of(bytes: &[u8]) -> SpaceSavingMetadata {
        let (_, metadata, _) = envelope::split(bytes).expect("split");
        from_slice(metadata).expect("metadata")
    }

    #[test]
    fn space_saving_envelope_structure_and_round_trip() {
        let distinct = 400i64;
        let summary = zipfish(48, distinct, 30_000, 0xfeed);
        assert_eq!(summary.len(), summary.capacity(), "eviction was not forced");
        assert!(summary.min_count() > 0, "the ceiling was never raised");

        let bytes = summary.serialize_to_bytes().expect("serialize");
        assert!(bytes.starts_with(envelope::MAGIC));
        assert_eq!(bytes[6], envelope::VERSION);
        assert_eq!(bytes[7], 2, "kind_id_len");
        assert_eq!(&bytes[8..10], &[0x18, 0x00], "Space-Saving kind_id");

        let meta = metadata_of(&bytes);
        assert_eq!(meta.metadata_version, 1);
        assert_eq!(meta.key_type, "i64");
        assert_eq!(meta.capacity, 48);

        let (_, _, payload) = envelope::split(&bytes).expect("split");
        let emitted: SpaceSavingPayload<i64> = from_slice(payload).expect("payload");
        assert_eq!(emitted.counts.len(), summary.len());
        assert_eq!(emitted.total, summary.total());
        for pair in emitted.counts.windows(2) {
            assert!(pair[0] >= pair[1], "the emitted counts are not descending");
        }
        assert!(
            emitted.counts[0] > emitted.counts[summary.len() - 1],
            "the fixture cannot tell descending from ascending"
        );

        let decoded =
            SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            bytes,
            "Space-Saving bytes differed after a round trip"
        );

        assert_eq!(decoded.len(), summary.len());
        assert_eq!(decoded.capacity(), summary.capacity());
        assert_eq!(decoded.total(), summary.total());
        assert_eq!(decoded.min_count(), summary.min_count());

        let mut monitored = 0;
        for key in 1..=distinct {
            let probe = DataInput::I64(key);
            assert_eq!(
                decoded.estimate(&probe),
                summary.estimate(&probe),
                "key {key}"
            );
            assert_eq!(decoded.error(&probe), summary.error(&probe), "key {key}");
            assert_eq!(
                decoded.upper_bound(&probe),
                summary.upper_bound(&probe),
                "key {key}"
            );
            assert_eq!(
                decoded.is_guaranteed(&probe),
                summary.is_guaranteed(&probe),
                "key {key}"
            );
            if summary.estimate(&probe) > 0 {
                monitored += 1;
            }
        }
        assert_eq!(
            monitored,
            summary.len(),
            "not every monitored key was probed"
        );
    }

    /// The ceiling a merge leaves behind lives only in `floor`, which is not
    /// derivable from the triples: a payload carrying only the triples would
    /// decode to `min_count == 0` here.
    #[test]
    fn space_saving_merged_ceiling_survives_the_wire() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(33);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(1);
        for _ in 0..10 {
            right.insert(&DataInput::I64(7));
        }
        for _ in 0..20 {
            right.insert(&DataInput::I64(8));
        }
        left.merge(&right);

        assert!(left.len() < left.capacity(), "the merge left room to spare");
        assert!(left.min_count() >= 10, "the merge did not raise a ceiling");

        let bytes = left.serialize_to_bytes().expect("serialize");
        let decoded =
            SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");

        assert_eq!(decoded.min_count(), left.min_count());
        assert_eq!(
            decoded.upper_bound(&DataInput::I64(7)),
            left.upper_bound(&DataInput::I64(7)),
            "the ceiling on the key the other side dropped was lost"
        );
        assert!(decoded.upper_bound(&DataInput::I64(7)) >= 10);
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    #[test]
    fn space_saving_empty_round_trip() {
        let summary: SpaceSaving = SpaceSaving::with_capacity(16);
        let bytes = summary.serialize_to_bytes().expect("serialize");
        assert_eq!(metadata_of(&bytes).key_type, "u64");
        assert_eq!(metadata_of(&bytes).metadata_version, 1);

        let decoded =
            SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.len(), 0);
        assert_eq!(decoded.capacity(), 16);
        assert_eq!(decoded.min_count(), 0);
        assert_eq!(decoded.total(), 0);
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// Every wire-eligible key family round-trips, and the decoded key still
    /// answers to the caller's original `DataInput`. A widened `key_type`
    /// (i32 to i64, say) keeps the same digest but rebuilds a different
    /// `HeapItem` variant, so `estimate` would read zero here.
    #[test]
    fn space_saving_every_key_type_round_trips_and_keeps_its_variant() {
        fn check(expected: &str, values: &[DataInput]) {
            let mut summary: SpaceSaving = SpaceSaving::with_capacity(8);
            for (i, value) in values.iter().enumerate() {
                summary.insert_many(value, 3 * (i as u64 + 1));
            }
            assert_eq!(summary.len(), values.len(), "{expected}: keys collided");

            let bytes = summary.serialize_to_bytes().expect("serialize");
            assert_eq!(metadata_of(&bytes).key_type, expected);

            let decoded =
                SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
            assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
            for (i, value) in values.iter().enumerate() {
                assert_eq!(
                    decoded.estimate(value),
                    3 * (i as u64 + 1),
                    "{expected}: {value:?} no longer matches its DataInput"
                );
            }
        }

        check(
            "i8",
            &[DataInput::I8(-3), DataInput::I8(7), DataInput::I8(120)],
        );
        check("i16", &[DataInput::I16(-3000), DataInput::I16(9)]);
        check("i32", &[DataInput::I32(-7), DataInput::I32(65_537)]);
        check("i64", &[DataInput::I64(-1), DataInput::I64(1 << 40)]);
        check("isize", &[DataInput::ISIZE(-11), DataInput::ISIZE(11)]);
        check("u8", &[DataInput::U8(0), DataInput::U8(255)]);
        check("u16", &[DataInput::U16(1), DataInput::U16(65_535)]);
        check("u32", &[DataInput::U32(7), DataInput::U32(u32::MAX)]);
        check("u64", &[DataInput::U64(9), DataInput::U64(u64::MAX)]);
        check("usize", &[DataInput::USIZE(4), DataInput::USIZE(1 << 33)]);
        check("f32", &[DataInput::F32(-0.5), DataInput::F32(3.25)]);
        check("f64", &[DataInput::F64(2.5), DataInput::F64(-1e300)]);
        check("string", &[DataInput::Str("alpha"), DataInput::Str("beta")]);
        check(
            "string",
            &[
                DataInput::String("gamma".to_string()),
                DataInput::Str("delta"),
            ],
        );
        check(
            "bytes",
            &[
                DataInput::Bytes(b"delta"),
                DataInput::Bytes(&[0xff, 0x00, 0xfe]),
            ],
        );
    }

    /// A byte key reaches the wire as msgpack `bin`, so any byte string round
    /// trips whether or not it is UTF-8, and the decoded key still answers the
    /// `DataInput::Bytes` it was inserted with.
    #[test]
    fn space_saving_byte_keys_round_trip_arbitrary_bytes() {
        let keys: [&[u8]; 3] = [&[0xff, 0x00, 0xfe], &[0x80; 40], b""];
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(8);
        for (i, key) in keys.iter().enumerate() {
            summary.insert_many(&DataInput::Bytes(key), 3 * (i as u64 + 1));
        }

        let bytes = summary.serialize_to_bytes().expect("serialize");
        assert_eq!(metadata_of(&bytes).key_type, "bytes");

        let (_, _, payload) = envelope::split(&bytes).expect("split");
        let emitted: SpaceSavingPayload<WireBytes> = from_slice(payload).expect("payload");
        let mut emitted_keys: Vec<Vec<u8>> = emitted.keys.iter().map(|key| key.0.clone()).collect();
        emitted_keys.sort_unstable();
        let mut expected: Vec<Vec<u8>> = keys.iter().map(|key| key.to_vec()).collect();
        expected.sort_unstable();
        assert_eq!(emitted_keys, expected, "the raw bytes did not survive");

        let decoded =
            SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(
                decoded.estimate(&DataInput::Bytes(key)),
                3 * (i as u64 + 1),
                "{key:?} no longer matches its DataInput"
            );
        }
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// A byte key and a string key are different key types, so a summary
    /// holding both has no single `key_type` and refuses to serialize rather
    /// than coercing one into the other.
    #[test]
    fn space_saving_refuses_byte_keys_mixed_with_string_keys() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(4);
        summary.insert_many(&DataInput::Bytes(b"abc"), 5);
        summary.insert_many(&DataInput::Str("abc"), 2);
        assert_eq!(summary.len(), 2);

        let problem = summary.serialize_to_bytes().expect_err("mixed variants");
        assert!(
            problem.to_string().contains("mix variants"),
            "got {problem}"
        );
    }

    /// The emitted order is fixed by (count, key_order), not by the arena's
    /// seat order, so the same triples always produce the same bytes. Both
    /// count ties below are broken only by the key.
    #[test]
    fn space_saving_emitted_order_is_independent_of_seat_order() {
        let entries = vec![
            (HeapItem::U64(3), 5u64, 1u64),
            (HeapItem::U64(1), 9, 0),
            (HeapItem::U64(2), 5, 2),
            (HeapItem::U64(4), 9, 3),
        ];
        let mut reversed = entries.clone();
        reversed.reverse();

        let seated = SpaceSaving::<DefaultXxHasher>::rebuild(SpaceSavingState {
            capacity: 4,
            total: 28,
            floor: 0,
            entries,
        })
        .expect("rebuild");
        let reseated = SpaceSaving::<DefaultXxHasher>::rebuild(SpaceSavingState {
            capacity: 4,
            total: 28,
            floor: 0,
            entries: reversed,
        })
        .expect("rebuild");

        let bytes = seated.serialize_to_bytes().expect("serialize");
        assert_eq!(
            bytes,
            reseated.serialize_to_bytes().expect("serialize"),
            "the emitted order followed the seat order"
        );

        let (_, _, payload) = envelope::split(&bytes).expect("split");
        let emitted: SpaceSavingPayload<u64> = from_slice(payload).expect("payload");
        assert_eq!(emitted.counts, vec![9, 9, 5, 5], "not descending by count");
        assert_eq!(emitted.keys, vec![1, 4, 2, 3], "ties not broken by key");
        assert_eq!(emitted.errors, vec![0, 3, 2, 1]);
    }

    #[test]
    fn space_saving_refuses_mixed_and_128_bit_keys() {
        let mut mixed: SpaceSaving = SpaceSaving::with_capacity(4);
        mixed.insert(&DataInput::I32(1));
        mixed.insert(&DataInput::I64(2));
        assert_eq!(mixed.len(), 2);
        assert!(
            mixed.serialize_to_bytes().is_err(),
            "a summary mixing key variants must not serialize"
        );

        for key in [DataInput::I128(1), DataInput::U128(2)] {
            let mut wide: SpaceSaving = SpaceSaving::with_capacity(4);
            wide.insert(&key);
            assert!(
                wide.serialize_to_bytes().is_err(),
                "{key:?} is not a wire key type"
            );
        }

        // A 128-bit key behind a wire-eligible one is caught on the way into
        // the payload rather than by the first-key check.
        let mut trailing: SpaceSaving = SpaceSaving::with_capacity(4);
        trailing.insert_many(&DataInput::U64(1), 9);
        trailing.insert_many(&DataInput::U128(2), 1);
        assert!(trailing.serialize_to_bytes().is_err());
    }

    #[test]
    fn space_saving_rejects_a_key_type_the_payload_does_not_carry() {
        let mut strings: SpaceSaving = SpaceSaving::with_capacity(4);
        strings.insert(&DataInput::Str("alpha"));
        let string_bytes = strings.serialize_to_bytes().expect("serialize");
        let (_, _, string_payload) = envelope::split(&string_bytes).expect("split");

        let mut numbers: SpaceSaving = SpaceSaving::with_capacity(4);
        numbers.insert(&DataInput::U64(11));
        let number_bytes = numbers.serialize_to_bytes().expect("serialize");
        let (_, _, number_payload) = envelope::split(&number_bytes).expect("split");

        let mut raw: SpaceSaving = SpaceSaving::with_capacity(4);
        raw.insert(&DataInput::Bytes(&[0xff, 0x00, 0xfe]));
        let raw_bytes = raw.serialize_to_bytes().expect("serialize");
        let (_, _, raw_payload) = envelope::split(&raw_bytes).expect("split");

        for (claimed, payload) in [
            ("u64", string_payload),
            ("string", number_payload),
            ("bytes", number_payload),
            ("bytes", string_payload),
            ("u64", raw_payload),
            ("string", raw_payload),
        ] {
            let metadata =
                rmp_serde::to_vec_named(&space_saving_metadata::<DefaultXxHasher>(4, claimed))
                    .expect("metadata");
            let forged = envelope::encode(SPACE_SAVING_KIND, &metadata, payload);
            assert!(
                SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&forged).is_err(),
                "a payload relabelled {claimed} must be rejected"
            );
        }
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`. Space-Saving metadata is derived from
    // the profile, so an `AltHasher` summary serializes truthfully. (An
    // *unprofiled* hasher cannot serialize at all — that is a compile-time
    // guarantee, since the wire methods are bounded on `H: HashProfile`.)
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
    fn space_saving_custom_hasher_profile_round_trips_and_is_self_describing() {
        let mut alt: SpaceSaving<AltHasher> = SpaceSaving::with_capacity(4);
        let mut std: SpaceSaving = SpaceSaving::with_capacity(4);
        for (key, weight) in [(1u64, 5u64), (2, 3)] {
            alt.insert_many(&DataInput::U64(key), weight);
            std.insert_many(&DataInput::U64(key), weight);
        }

        // (a) A summary built with a custom-profile hasher round-trips.
        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded = SpaceSaving::<AltHasher>::deserialize_from_bytes(&alt_bytes).expect("decode");
        assert_eq!(decoded.estimate(&DataInput::U64(1)), 5);
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            alt_bytes
        );

        // (b) Bytes differ from the standard-profile summary.
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }

    #[test]
    fn space_saving_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            capacity: u32,
            key_type: String,
            bogus_field: u8, // key not in SpaceSavingMetadata
        }
        let m = space_saving_metadata::<DefaultXxHasher>(4, "u64");
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            capacity: m.capacity,
            key_type: m.key_type.clone(),
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).expect("encode");
        assert!(
            from_slice::<SpaceSavingMetadata>(&bytes).is_err(),
            "an unexpected metadata key must be rejected"
        );
    }

    /// Builds a well-framed envelope around a crafted `u64`-keyed payload.
    fn crafted(
        capacity: u32,
        keys: Vec<u64>,
        counts: Vec<u64>,
        errors: Vec<u64>,
        total: u64,
        floor: u64,
    ) -> Vec<u8> {
        let metadata =
            rmp_serde::to_vec_named(&space_saving_metadata::<DefaultXxHasher>(capacity, "u64"))
                .expect("metadata");
        let payload = rmp_serde::to_vec(&SpaceSavingPayload {
            keys,
            counts,
            errors,
            total,
            floor,
        })
        .expect("payload");
        envelope::encode(SPACE_SAVING_KIND, &metadata, &payload)
    }

    #[test]
    fn space_saving_rejects_a_crafted_envelope() {
        let relabelled = |key_type: &str| {
            let metadata =
                rmp_serde::to_vec_named(&space_saving_metadata::<DefaultXxHasher>(4, key_type))
                    .expect("metadata");
            let payload = rmp_serde::to_vec(&SpaceSavingPayload::<u64> {
                keys: vec![1],
                counts: vec![1],
                errors: vec![0],
                total: 1,
                floor: 0,
            })
            .expect("payload");
            (metadata, payload)
        };
        let (unknown_meta, unknown_payload) = relabelled("u256");
        let (foreign_meta, foreign_payload) = relabelled("u64");

        let cases: Vec<(Vec<u8>, &str)> = vec![
            (
                crafted(0, vec![1], vec![2], vec![0], 2, 0),
                "capacity is zero",
            ),
            (
                crafted(1, vec![1, 2], vec![2, 2], vec![0, 0], 4, 0),
                "over a capacity of 1",
            ),
            (
                crafted(4, vec![1, 2], vec![2], vec![0, 0], 2, 0),
                "2 keys against 1 counts",
            ),
            (
                crafted(4, vec![1, 2], vec![2, 2], vec![0], 4, 0),
                "2 counts and 1 errors",
            ),
            (
                crafted(4, vec![1], vec![0], vec![0], 0, 0),
                "counter at zero",
            ),
            (
                crafted(4, vec![1], vec![3], vec![4], 3, 0),
                "error of 4 against a count of 3",
            ),
            (
                crafted(4, vec![1, 1], vec![2, 1], vec![0, 0], 3, 0),
                "same key twice",
            ),
            (
                envelope::encode(SPACE_SAVING_KIND, &unknown_meta, &unknown_payload),
                "is not an ASAPv1 wire key type",
            ),
            (
                crafted(4, vec![1], vec![3], vec![0], 3, u64::MAX),
                "ceiling of 18446744073709551615 above its lowest count of 3",
            ),
            (
                crafted(4, vec![1], vec![9], vec![0], 0, 0),
                "total of 0 under the 9",
            ),
            (
                envelope::encode(&[0x02, 0x00], &foreign_meta, &foreign_payload),
                "kind_id mismatch",
            ),
        ];
        for (bytes, expected) in cases {
            let problem = SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes)
                .expect_err("a crafted envelope must be rejected, not decoded")
                .to_string();
            assert!(
                problem.contains(expected),
                "expected a complaint about {expected}, got {problem}"
            );
        }
    }

    /// A capacity past the metadata's `u32` field fails the encode: truncating
    /// it would emit a capacity of zero, which no decode accepts.
    #[test]
    fn space_saving_refuses_a_capacity_the_metadata_cannot_carry() {
        let summary = SpaceSaving::<DefaultXxHasher>::rebuild(SpaceSavingState {
            capacity: 1 << 40,
            total: 3,
            floor: 0,
            entries: vec![(HeapItem::U64(1), 3, 0)],
        })
        .expect("a sparse state");
        assert_eq!(summary.capacity(), 1 << 40);

        let problem = summary
            .serialize_to_bytes()
            .expect_err("an oversized capacity must not serialize")
            .to_string();
        assert!(
            problem.contains("exceeds the u32 metadata field"),
            "got {problem}"
        );
    }

    /// A declared capacity is metadata, never an allocation size: `u32::MAX`
    /// counters would be tens of gigabytes if it sized the arena.
    #[test]
    fn space_saving_does_not_allocate_a_declared_capacity() {
        let bytes = crafted(u32::MAX, vec![7, 8], vec![9, 4], vec![0, 1], 13, 0);
        let decoded =
            SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.capacity(), u32::MAX as usize);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.estimate(&DataInput::U64(7)), 9);
        assert_eq!(decoded.min_count(), 0, "a sparse summary has no floor");
        assert_eq!(decoded.top_k(8).len(), 2);
    }
}
