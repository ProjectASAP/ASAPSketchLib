//! ASAPv1 wire pieces shared by the heap-backed top-k sketches.
//!
//! [`CMSHeap`](super::CMSHeap) (kind_id `0x03 0x00`) and
//! [`CSHeap`](crate::sketches::countsketch_topk::CSHeap) (kind_id `0x0a 0x00`)
//! are each a matrix sketch plus an [`HHHeap`], so both carry the same
//! descriptor metadata and the same positional payload
//! `[counts, keys, heap_counts]`. This module is the one place that encoding
//! lives: the metadata DTO, the payload DTO, the `key_type` mapping, the
//! emitted order, and the heap rebuild. See `docs/asapv1_wire_format.md` §3.7
//! and §3.10 for the pieces it follows.
//!
//! ## `key_type` names the exact `HeapItem` variant
//!
//! A heap key's type is a **runtime** property: keys are [`HeapItem`]s, a
//! 16-variant enum, and one heap's keys are whatever the caller inserted. One
//! metadata `key_type` names the exact variant and the payload's `keys` array
//! is homogeneous in it. The variant is never widened (an `i32` key is `"i32"`,
//! not `"i64"`) because `HeapItem`'s equality against a query `DataInput`
//! discriminates on the variant while the digest does not: a widened key lands
//! in the same index slot and then silently fails equality.
//! `HeapItem::I128` / `U128` have no msgpack integer form and are not wire
//! types, so a heap holding one refuses to serialize — as does one whose keys
//! mix variants, `String` and `Bytes` included. A `Bytes` key is written as
//! msgpack `bin` through `WireBytes` and read back from `bin` alone, so any
//! byte string survives whether or not it is UTF-8 and a `str`-keyed payload
//! relabelled `"bytes"` is refused.
//!
//! ## Emitted order (byte-stable round trips)
//!
//! The heap's array order follows the sift path and does not survive a rebuild,
//! so the payload is **order-defined**: descending count, ties broken by
//! [`key_order`], a total order over [`HeapItem`] (variant tag first, then the
//! value). Two heaps holding the same entries emit the same bytes whatever
//! order they were seated in, and re-serializing a decoded sketch reproduces
//! its bytes exactly.
//!
//! ## What the heap does not put on the wire
//!
//! `HHHeap`'s `slots` and `positions` are `#[serde(skip)]` and are rebuilt from
//! the entries, so no index reaches the wire and no crafted payload can point
//! one out of bounds or into a cycle. `k` is construction config, so it lives
//! in the metadata; it never sizes an allocation on decode.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::message_pack_format::wire_key::WireBytes;
use crate::{HHHeap, HashProfile, HeapItem};

/// Metadata `key_type` of a heap that holds nothing. A heap with no entries has
/// no variant to report, so the wire pins one: empty heaps have one encoding.
pub(crate) const EMPTY_KEY_TYPE: &str = "u64";

/// The metadata `key_type` naming a key's exact [`HeapItem`] variant, or `None`
/// for the 128-bit variants, which are not wire types.
pub(crate) fn key_type_of(key: &HeapItem) -> Option<&'static str> {
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

/// A total order over keys: variant tag first, then the value.
fn key_order(key: &HeapItem) -> (u8, u128, &[u8]) {
    match key {
        HeapItem::I8(v) => (0, *v as i128 as u128, b""),
        HeapItem::I16(v) => (1, *v as i128 as u128, b""),
        HeapItem::I32(v) => (2, *v as i128 as u128, b""),
        HeapItem::I64(v) => (3, *v as i128 as u128, b""),
        HeapItem::I128(v) => (4, *v as u128, b""),
        HeapItem::ISIZE(v) => (5, *v as i128 as u128, b""),
        HeapItem::U8(v) => (6, u128::from(*v), b""),
        HeapItem::U16(v) => (7, u128::from(*v), b""),
        HeapItem::U32(v) => (8, u128::from(*v), b""),
        HeapItem::U64(v) => (9, u128::from(*v), b""),
        HeapItem::U128(v) => (10, *v, b""),
        HeapItem::USIZE(v) => (11, *v as u128, b""),
        HeapItem::F32(v) => (12, u128::from(v.to_bits()), b""),
        HeapItem::F64(v) => (13, u128::from(v.to_bits()), b""),
        HeapItem::String(v) => (14, 0, v.as_bytes()),
        HeapItem::Bytes(v) => (15, 0, v.as_slice()),
    }
}

/// The complaint both directions report when one key is on the wire twice.
const DUPLICATE_KEY: &str = "ASAPv1 top-k heap: the same key appears twice";

/// Rejects a key set holding the same key twice, judged on the **wire**
/// representation [`key_order`] gives it: floats by their bits, so two identical
/// `NaN`s are one key twice while `+0.0` and `-0.0` stay two keys. Both the
/// encode and the decode path run this one check, so neither emits nor admits a
/// key set the other refuses. The UnivMon pyramid encoders call it per layer,
/// so the heap-backed sketches and the pyramids share one predicate.
pub(crate) fn check_distinct_keys<'a>(
    keys: impl ExactSizeIterator<Item = &'a HeapItem>,
) -> Result<(), &'static str> {
    let mut seen: HashSet<(u8, u128, &[u8])> = HashSet::with_capacity(keys.len());
    for key in keys {
        if !seen.insert(key_order(key)) {
            return Err(DUPLICATE_KEY);
        }
    }
    Ok(())
}

/// Top-k descriptor metadata (ASAPv1 §2), a msgpack **map** (`to_vec_named`)
/// with keys in this declaration order — the canonical order the wire spec
/// fixes (Go must mirror it). Hash-spec fields first, then the base sketch's
/// structural params `rows` / `cols` / `counter_type` / `mode`, then the heap's
/// `k` / `key_type`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TopKMetadata {
    pub(crate) metadata_version: u8,
    pub(crate) hash_profile_id: String,
    pub(crate) hash_algorithm: String,
    pub(crate) seed_derivation: String,
    pub(crate) input_encoding: String,
    pub(crate) seed_list: Vec<u64>,
    pub(crate) matrix_seed_index: u32,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) counter_type: String,
    pub(crate) mode: String,
    pub(crate) k: u32,
    pub(crate) key_type: String,
}

/// Builds the descriptor metadata from the hasher's [`HashProfile`], so the wire
/// bytes truthfully describe how the sketch was hashed. `rows` / `cols` /
/// `counter_type` / `mode` describe the base matrix; `k` is the heap capacity
/// and `key_type` the exact [`HeapItem`] variant the `keys` array carries.
pub(crate) fn topk_metadata<H: HashProfile>(
    rows: u32,
    cols: u32,
    counter_type: &str,
    mode: &str,
    k: u32,
    key_type: &str,
) -> TopKMetadata {
    TopKMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        matrix_seed_index: H::MATRIX_SEED_INDEX,
        rows,
        cols,
        counter_type: counter_type.to_string(),
        mode: mode.to_string(),
        k,
        key_type: key_type.to_string(),
    }
}

/// Top-k payload, a msgpack **array** (`to_vec`, positional):
/// `[counts, keys, heap_counts]`. `counts` is the base matrix packed row-major
/// with the element type the metadata `counter_type` fixes; `keys` and
/// `heap_counts` are the heap's entries, parallel and equal-length, with `keys`
/// typed by the metadata `key_type`.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TopKPayload<C, K> {
    pub(crate) counts: Vec<C>,
    pub(crate) keys: Vec<K>,
    pub(crate) heap_counts: Vec<i64>,
}

/// The heap's entries in emitted order: descending count, ties broken by
/// [`key_order`]. Independent of the heap array's sift order, so the bytes are
/// stable across a round trip.
pub(crate) fn heap_entries(heap: &HHHeap) -> Vec<(&HeapItem, i64)> {
    let mut entries: Vec<(&HeapItem, i64)> = heap
        .heap()
        .iter()
        .map(|item| (&item.key, item.count))
        .collect();
    entries.sort_unstable_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| key_order(a.0).cmp(&key_order(b.0)))
    });
    entries
}

/// The `key_type` the payload will be written in, taken from the first key in
/// emitted order. Rejects a 128-bit key outright; the homogeneity of the rest is
/// enforced by [`encode_payload`].
pub(crate) fn wire_key_type(entries: &[(&HeapItem, i64)]) -> Result<&'static str, RmpEncodeError> {
    match entries.first() {
        None => Ok(EMPTY_KEY_TYPE),
        Some((key, _)) => key_type_of(key).ok_or_else(|| {
            RmpEncodeError::Syntax("ASAPv1 top-k heap: 128-bit keys are not wire types".to_string())
        }),
    }
}

fn mixed_variant_error(key_type: &str, key: &HeapItem) -> RmpEncodeError {
    RmpEncodeError::Syntax(format!(
        "ASAPv1 top-k heap: keys mix variants — key_type is {key_type}, but a {} key is held",
        key_type_of(key).unwrap_or("128-bit")
    ))
}

/// Packs the base counters and the emitted heap entries into the positional
/// payload, with `keys` typed by `key_type`. Any key that is not of that variant
/// fails the encode, as does a key the entries hold twice on the wire — a live
/// heap can seat one, since `NaN` misses its own lookup.
pub(crate) fn encode_payload<C: Serialize>(
    key_type: &str,
    counts: Vec<C>,
    entries: &[(&HeapItem, i64)],
) -> Result<Vec<u8>, RmpEncodeError> {
    check_distinct_keys(entries.iter().map(|(key, _)| *key))
        .map_err(|problem| RmpEncodeError::Syntax(problem.to_string()))?;
    let heap_counts: Vec<i64> = entries.iter().map(|entry| entry.1).collect();

    macro_rules! pack {
        ($variant:ident) => {{
            let mut keys = Vec::with_capacity(entries.len());
            for (key, _) in entries {
                match key {
                    HeapItem::$variant(value) => keys.push(*value),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&TopKPayload {
                counts,
                keys,
                heap_counts,
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
            for (key, _) in entries {
                match key {
                    HeapItem::String(value) => keys.push(value.clone()),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&TopKPayload {
                counts,
                keys,
                heap_counts,
            })
        }
        "bytes" => {
            let mut keys = Vec::with_capacity(entries.len());
            for (key, _) in entries {
                match key {
                    HeapItem::Bytes(value) => keys.push(WireBytes(value.clone())),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&TopKPayload {
                counts,
                keys,
                heap_counts,
            })
        }
        other => Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 top-k heap: key_type {other:?} is not a wire key type"
        ))),
    }
}

/// The base counters and the heap entries one payload decodes into.
pub(crate) type DecodedTopK<C> = (Vec<C>, Vec<(HeapItem, i64)>);

/// Reads the payload with `keys` typed by the metadata `key_type`, returning the
/// base counters and the heap entries. Rejects an unknown `key_type` and
/// parallel arrays of unequal length; a `keys` array whose msgpack types do not
/// match `key_type` fails in `from_slice`.
pub(crate) fn decode_payload<C>(
    key_type: &str,
    payload: &[u8],
) -> Result<DecodedTopK<C>, RmpDecodeError>
where
    C: for<'de> Deserialize<'de>,
{
    macro_rules! unpack {
        ($variant:ident, $ty:ty) => {{
            let decoded: TopKPayload<C, $ty> = from_slice(payload)?;
            (
                decoded.counts,
                decoded
                    .keys
                    .into_iter()
                    .map(HeapItem::$variant)
                    .collect::<Vec<HeapItem>>(),
                decoded.heap_counts,
            )
        }};
    }

    let (counts, keys, heap_counts) = match key_type {
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
            let decoded: TopKPayload<C, WireBytes> = from_slice(payload)?;
            (
                decoded.counts,
                decoded
                    .keys
                    .into_iter()
                    .map(|key| HeapItem::Bytes(key.into_vec()))
                    .collect::<Vec<HeapItem>>(),
                decoded.heap_counts,
            )
        }
        other => {
            return Err(RmpDecodeError::Uncategorized(format!(
                "ASAPv1 top-k heap: key_type {other:?} is not a wire key type"
            )));
        }
    };

    if keys.len() != heap_counts.len() {
        return Err(RmpDecodeError::Uncategorized(format!(
            "ASAPv1 top-k heap: {} keys against {} counts",
            keys.len(),
            heap_counts.len()
        )));
    }
    Ok((counts, keys.into_iter().zip(heap_counts).collect()))
}

/// Seats the decoded entries into a heap of capacity `k`, rebuilding the index
/// as it goes. `k` is checked against the entry count first and the heap
/// reserves from that count, so a declared capacity never sizes an allocation
/// on its own.
///
/// The keys are checked for duplicates on their **wire** representation before
/// any is seated — floats by their bits, so two identical `NaN`s are one key
/// twice even though they do not compare equal — and the seated count is
/// checked afterwards, which catches keys that are distinct on the wire but
/// equal to the heap.
pub(crate) fn rebuild_heap(
    k: usize,
    entries: Vec<(HeapItem, i64)>,
) -> Result<HHHeap, RmpDecodeError> {
    if entries.len() > k {
        return Err(RmpDecodeError::Uncategorized(format!(
            "ASAPv1 top-k heap: {} entries over a k of {k}",
            entries.len()
        )));
    }
    check_distinct_keys(entries.iter().map(|(key, _)| key))
        .map_err(|problem| RmpDecodeError::Uncategorized(problem.to_string()))?;
    let mut heap = HHHeap::with_expected_len(k, entries.len());
    let seated = entries.len();
    for (key, count) in entries {
        heap.update_heap_item(&key, count);
    }
    if heap.len() != seated {
        return Err(RmpDecodeError::Uncategorized(DUPLICATE_KEY.to_string()));
    }
    Ok(heap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message_pack_format::envelope;
    use crate::{DataInput, DefaultXxHasher, RegularPath, Vector2D};

    use super::super::CMSHeap;

    /// CMSHeap's kind_id, so the shared-heap tests can build real envelopes.
    const KIND: &[u8] = &[0x03, 0x00];

    fn sketch_with(keys: &[(DataInput, i64)]) -> CMSHeap<Vector2D<i64>, RegularPath> {
        let mut sketch = CMSHeap::<Vector2D<i64>, RegularPath>::new(2, 8, 8);
        for (key, count) in keys {
            sketch.heap_mut().update(key, *count);
        }
        sketch
    }

    /// The complaint a crafted envelope draws, with no `Debug` bound on the
    /// sketch.
    fn decode_error(bytes: &[u8]) -> String {
        match CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(bytes) {
            Ok(_) => panic!("a crafted envelope must be rejected, not decoded"),
            Err(err) => err.to_string(),
        }
    }

    fn metadata_of(bytes: &[u8]) -> TopKMetadata {
        let (_, metadata, _) = envelope::split(bytes).expect("split");
        from_slice(metadata).expect("metadata")
    }

    /// Every wire-eligible key family round-trips under its own name, and the
    /// decoded key still answers to the caller's original `DataInput`. A widened
    /// `key_type` would keep the digest but rebuild a different variant.
    #[test]
    fn heap_every_key_type_round_trips_and_keeps_its_variant() {
        fn check(expected: &str, values: &[DataInput]) {
            let entries: Vec<(DataInput, i64)> = values
                .iter()
                .enumerate()
                .map(|(i, value)| (value.clone(), 3 * (i as i64 + 1)))
                .collect();
            let sketch = sketch_with(&entries);
            assert_eq!(
                sketch.heap().len(),
                values.len(),
                "{expected}: keys collided"
            );

            let bytes = sketch.serialize_to_bytes().expect("serialize");
            assert_eq!(metadata_of(&bytes).key_type, expected);

            let decoded = CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes)
                .expect("decode");
            assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
            for (i, value) in values.iter().enumerate() {
                let index = decoded
                    .heap()
                    .find(value)
                    .unwrap_or_else(|| panic!("{expected}: {value:?} lost its variant"));
                assert_eq!(decoded.heap().heap()[index].count, 3 * (i as i64 + 1));
            }
        }

        check("i8", &[DataInput::I8(-3), DataInput::I8(120)]);
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
    }

    #[test]
    fn heap_refuses_mixed_and_128_bit_keys() {
        let mixed = sketch_with(&[(DataInput::I32(1), 5), (DataInput::I64(2), 3)]);
        let problem = mixed
            .serialize_to_bytes()
            .expect_err("a heap mixing key variants must not serialize")
            .to_string();
        assert!(problem.contains("keys mix variants"), "got {problem}");
        assert!(problem.contains("key_type is i32"), "got {problem}");

        for key in [DataInput::I128(1), DataInput::U128(2)] {
            let wide = sketch_with(&[(key.clone(), 5)]);
            let problem = wide
                .serialize_to_bytes()
                .expect_err("a 128-bit key is not a wire type")
                .to_string();
            assert!(problem.contains("128-bit"), "got {problem}");
        }

        // A 128-bit key behind a wire-eligible one is caught on the way into the
        // payload rather than by the first-key check.
        let trailing = sketch_with(&[(DataInput::U64(1), 9), (DataInput::U128(2), 1)]);
        assert!(trailing.serialize_to_bytes().is_err());
    }

    #[test]
    fn heap_empty_emits_the_pinned_key_type_and_round_trips() {
        let sketch = CMSHeap::<Vector2D<i64>, RegularPath>::new(2, 8, 8);
        let bytes = sketch.serialize_to_bytes().expect("serialize");
        let meta = metadata_of(&bytes);
        assert_eq!(meta.key_type, EMPTY_KEY_TYPE);
        assert_eq!(meta.k, 8);

        let decoded =
            CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.heap().len(), 0);
        assert_eq!(decoded.heap().capacity(), 8);
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// The emitted order is fixed by (count, key_order), not by the sift path,
    /// so the same entries always produce the same bytes. Both count ties below
    /// are broken only by the key.
    #[test]
    fn heap_emitted_order_is_independent_of_seat_order() {
        let entries = [
            (DataInput::U64(3), 5i64),
            (DataInput::U64(1), 9),
            (DataInput::U64(2), 5),
            (DataInput::U64(4), 9),
        ];
        let mut reversed = entries.clone();
        reversed.reverse();

        let seated = sketch_with(&entries);
        let reseated = sketch_with(&reversed);
        let bytes = seated.serialize_to_bytes().expect("serialize");
        assert_eq!(
            bytes,
            reseated.serialize_to_bytes().expect("serialize"),
            "the emitted order followed the seat order"
        );

        let (_, _, payload) = envelope::split(&bytes).expect("split");
        let emitted: TopKPayload<i64, u64> = from_slice(payload).expect("payload");
        assert_eq!(emitted.heap_counts, vec![9, 9, 5, 5], "not descending");
        assert_eq!(emitted.keys, vec![1, 4, 2, 3], "ties not broken by key");

        let decoded =
            CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            bytes,
            "a decoded heap re-serialized to different bytes"
        );
    }

    /// The `keys` array is read **as** the declared type, so string-keyed bytes
    /// relabelled `"u64"` do not decode.
    #[test]
    fn heap_rejects_a_key_type_the_payload_does_not_carry() {
        let strings = sketch_with(&[(DataInput::Str("alpha"), 5)]);
        let string_bytes = strings.serialize_to_bytes().expect("serialize");
        let (_, _, string_payload) = envelope::split(&string_bytes).expect("split");

        let numbers = sketch_with(&[(DataInput::U64(11), 5)]);
        let number_bytes = numbers.serialize_to_bytes().expect("serialize");
        let (_, _, number_payload) = envelope::split(&number_bytes).expect("split");

        for (claimed, payload) in [("u64", string_payload), ("string", number_payload)] {
            let metadata = rmp_serde::to_vec_named(&topk_metadata::<DefaultXxHasher>(
                2, 8, "i64", "regular", 8, claimed,
            ))
            .expect("metadata");
            let forged = envelope::encode(KIND, &metadata, payload);
            assert!(
                CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&forged).is_err(),
                "a payload relabelled {claimed} must be rejected"
            );
        }
    }

    /// `slots` and `positions` are rebuilt on decode: an update that moves an
    /// entry lands the same way on the decoded heap as on the original.
    #[test]
    fn heap_index_is_rebuilt_so_updates_still_move_entries() {
        let mut original = sketch_with(&[
            (DataInput::U64(1), 9),
            (DataInput::U64(2), 7),
            (DataInput::U64(3), 5),
            (DataInput::U64(4), 3),
        ]);
        let bytes = original.serialize_to_bytes().expect("serialize");
        let mut decoded =
            CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).expect("decode");

        // Rescore the root up, then seat a new key that displaces nothing.
        original.heap_mut().update(&DataInput::U64(4), 40);
        decoded.heap_mut().update(&DataInput::U64(4), 40);
        original.heap_mut().update(&DataInput::U64(5), 20);
        decoded.heap_mut().update(&DataInput::U64(5), 20);

        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            original.serialize_to_bytes().expect("serialize"),
            "the decoded heap diverged after an update"
        );
        for key in 1..=5u64 {
            let probe = DataInput::U64(key);
            let found = decoded.heap().find(&probe).expect("key");
            assert_eq!(
                decoded.heap().heap()[found].count,
                original.heap().heap()[original.heap().find(&probe).expect("key")].count
            );
        }
    }

    /// A declared `k` is metadata, never an allocation size.
    #[test]
    fn heap_does_not_allocate_a_declared_k() {
        let metadata = rmp_serde::to_vec_named(&topk_metadata::<DefaultXxHasher>(
            2,
            4,
            "i64",
            "regular",
            u32::MAX,
            "u64",
        ))
        .expect("metadata");
        let payload = rmp_serde::to_vec(&TopKPayload {
            counts: vec![0i64; 8],
            keys: vec![7u64, 8],
            heap_counts: vec![9i64, 4],
        })
        .expect("payload");
        let bytes = envelope::encode(KIND, &metadata, &payload);

        let decoded =
            CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.heap().capacity(), u32::MAX as usize);
        assert_eq!(decoded.heap().len(), 2);
    }

    /// More entries than `k`, and the same key twice, are both states the heap
    /// could not have reached.
    #[test]
    fn heap_rejects_crafted_entry_sets() {
        let crafted = |k: u32, keys: Vec<u64>, heap_counts: Vec<i64>| {
            let metadata = rmp_serde::to_vec_named(&topk_metadata::<DefaultXxHasher>(
                2, 4, "i64", "regular", k, "u64",
            ))
            .expect("metadata");
            let payload = rmp_serde::to_vec(&TopKPayload {
                counts: vec![0i64; 8],
                keys,
                heap_counts,
            })
            .expect("payload");
            envelope::encode(KIND, &metadata, &payload)
        };

        let cases = [
            (
                crafted(1, vec![1, 2], vec![5, 4]),
                "2 entries over a k of 1",
            ),
            (
                crafted(4, vec![1, 1], vec![5, 4]),
                "the same key appears twice",
            ),
            (crafted(4, vec![1, 2], vec![5]), "2 keys against 1 counts"),
        ];
        for (bytes, expected) in cases {
            let problem = decode_error(&bytes);
            assert!(
                problem.contains(expected),
                "expected a complaint about {expected}, got {problem}"
            );
        }
    }

    /// Duplicates are judged on the wire bits, not on `HeapItem`'s equality.
    /// Two bit-identical `NaN`s do not compare equal, so both would seat and the
    /// seated count would agree; `+0.0` and `-0.0` do compare equal but hash
    /// apart, so the live heap seats both and that payload must still decode.
    #[test]
    fn heap_judges_duplicate_float_keys_on_their_bits() {
        let crafted = |keys: Vec<f64>| {
            let metadata = rmp_serde::to_vec_named(&topk_metadata::<DefaultXxHasher>(
                2, 4, "i64", "regular", 4, "f64",
            ))
            .expect("metadata");
            let payload = rmp_serde::to_vec(&TopKPayload {
                counts: vec![0i64; 8],
                keys,
                heap_counts: vec![5i64, 4],
            })
            .expect("payload");
            envelope::encode(KIND, &metadata, &payload)
        };

        let problem = decode_error(&crafted(vec![f64::NAN, f64::NAN]));
        assert!(
            problem.contains("the same key appears twice"),
            "got {problem}"
        );

        let signed_zeros = crafted(vec![0.0, -0.0]);
        let decoded = CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&signed_zeros)
            .expect("two zeros of opposite sign are two keys the heap can hold");
        assert_eq!(decoded.heap().len(), 2);
    }

    /// A live heap can seat one key twice: `update` compares residents with
    /// `HeapItem`'s equality, and `NaN != NaN`, so a second `NaN` misses the
    /// lookup and takes its own seat. That heap must fail the encode rather than
    /// emit a payload the decoder refuses. `+0.0` and `-0.0` differ in their
    /// bits, so they are two keys on both sides and still round-trip.
    #[test]
    fn heap_refuses_to_serialize_a_key_it_holds_twice() {
        let nans = sketch_with(&[(DataInput::F64(f64::NAN), 5), (DataInput::F64(f64::NAN), 4)]);
        assert_eq!(nans.heap().len(), 2, "the two NaNs did not both seat");
        let problem = nans
            .serialize_to_bytes()
            .expect_err("a heap holding one key twice must not serialize")
            .to_string();
        assert!(
            problem.contains("the same key appears twice"),
            "got {problem}"
        );

        let zeros = sketch_with(&[(DataInput::F64(0.0), 5), (DataInput::F64(-0.0), 4)]);
        assert_eq!(zeros.heap().len(), 2, "the signed zeros did not both seat");
        let bytes = zeros
            .serialize_to_bytes()
            .expect("zeros of opposite sign are two keys");
        let decoded =
            CMSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.heap().len(), 2);
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }
}
