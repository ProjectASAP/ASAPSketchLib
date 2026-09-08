//! ASAPv1 wire serialization for [`UnivMonPyramid`].
//!
//! Child submodule of [`crate::sketch_framework::univmon_optimized`]: it holds
//! the metadata DTO, the kind_id constant and the `serialize_to_bytes` /
//! `deserialize_from_bytes` impls, while the algorithm lives in the parent
//! module file. Being a descendant module, it reads the private `update_mode`
//! and `candidate_complete` fields directly without widening any field
//! visibility. See `docs/asapv1_wire_format.md`.
//!
//! UnivMon Optimized is one algorithm — a single kind_id `0x11 0x00`. It
//! shares [`UnivMon`](crate::UnivMon)'s payload
//! (`[counts, l2, heap_lens, keys, heap_counts, candidate_complete,
//! bucket_size, update_mode]`) and differs only in its metadata: the two-tier
//! layout gives layer `i` the elephant dimensions while `i < elephant_layers`
//! and the mouse dimensions after, so every per-layer geometry is derived and
//! none is stored.
//!
//! [`UnivSketchPool`](super::UnivSketchPool) is a free-list of scratch
//! `UnivMon`s rather than a sketch, and has no wire kind.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::message_pack_format::envelope;
use crate::sketch_framework::univmon::wire::{
    check_layer_size, decode_pyramid, encode_pyramid, pyramid_key_type, pyramid_state,
    rebuild_layers, update_mode_of, update_mode_tag,
};
use crate::{DefaultXxHasher, HashProfile};

use super::UnivMonPyramid;

/// UnivMon Optimized kind_id: family `0x11`, single algorithm variant `0x00`.
const PYRAMID_KIND: &[u8] = &[0x11, 0x00];

/// UnivMonPyramid descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it). Hash-spec fields first, then the
/// two-tier layout and the heaps' `key_type`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PyramidMetadata {
    pub(crate) metadata_version: u8,
    pub(crate) hash_profile_id: String,
    pub(crate) hash_algorithm: String,
    pub(crate) seed_derivation: String,
    pub(crate) input_encoding: String,
    pub(crate) seed_list: Vec<u64>,
    pub(crate) layer_size: u32,
    pub(crate) elephant_layers: u32,
    pub(crate) elephant_row: u32,
    pub(crate) elephant_col: u32,
    pub(crate) mouse_row: u32,
    pub(crate) mouse_col: u32,
    pub(crate) heap_size: u32,
    pub(crate) key_type: String,
}

/// The five layout fields, in the order the metadata carries them.
pub(crate) struct PyramidLayout {
    pub(crate) elephant_layers: u32,
    pub(crate) elephant_row: u32,
    pub(crate) elephant_col: u32,
    pub(crate) mouse_row: u32,
    pub(crate) mouse_col: u32,
}

/// Builds the UnivMonPyramid descriptor metadata from the hasher's
/// [`HashProfile`], so the wire bytes truthfully describe how the sketch was
/// hashed.
pub(crate) fn pyramid_metadata<H: HashProfile>(
    layer_size: u32,
    layout: &PyramidLayout,
    heap_size: u32,
    key_type: &str,
) -> PyramidMetadata {
    PyramidMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        layer_size,
        elephant_layers: layout.elephant_layers,
        elephant_row: layout.elephant_row,
        elephant_col: layout.elephant_col,
        mouse_row: layout.mouse_row,
        mouse_col: layout.mouse_col,
        heap_size,
        key_type: key_type.to_string(),
    }
}

/// Per-layer `(rows, cols)`: elephant while `layer < elephant_layers`, mouse
/// after.
fn geometry_of(layer_size: usize, layout: &PyramidLayout) -> Vec<(usize, usize)> {
    (0..layer_size)
        .map(|layer| {
            if layer < layout.elephant_layers as usize {
                (layout.elephant_row as usize, layout.elephant_col as usize)
            } else {
                (layout.mouse_row as usize, layout.mouse_col as usize)
            }
        })
        .collect()
}

/// Rejects a layout the algorithm never has: `layer_size`, `heap_size` and
/// both tiers' dimensions are each positive. The encoder and the decoder both
/// call this, so the two doors cannot drift.
fn check_layout_dimensions(
    layer_size: usize,
    heap_size: usize,
    elephant_row: usize,
    elephant_col: usize,
    mouse_row: usize,
    mouse_col: usize,
) -> Result<(), String> {
    if layer_size == 0
        || heap_size == 0
        || elephant_row == 0
        || elephant_col == 0
        || mouse_row == 0
        || mouse_col == 0
    {
        return Err(format!(
            "UnivMonPyramid layer_size, heap_size and both tiers' dimensions must be non-zero: layer_size={layer_size}, elephant_row={elephant_row}, elephant_col={elephant_col}, mouse_row={mouse_row}, mouse_col={mouse_col}, heap_size={heap_size}"
        ));
    }
    Ok(())
}

/// Total accumulators the declared layout implies, one per row per layer.
fn accumulator_count(layer_size: usize, layout: &PyramidLayout) -> Option<usize> {
    let elephants = layer_size.min(layout.elephant_layers as usize);
    let mice = layer_size - elephants;
    elephants
        .checked_mul(layout.elephant_row as usize)?
        .checked_add(mice.checked_mul(layout.mouse_row as usize)?)
}

// Wire serialization for UnivMonPyramid. `wire` is a descendant of the sketch
// module, so this impl reads the private fields directly.
impl UnivMonPyramid {
    /// Serializes the pyramid into an ASAPv1 MessagePack envelope
    /// (kind_id `0x11 0x00`). The metadata is derived from
    /// [`DefaultXxHasher`]'s [`HashProfile`], the hasher the pyramid is built
    /// on.
    ///
    /// Fails when a structural parameter is zero, when a layer's geometry or
    /// seed index disagrees with the declared layout, when a layer's heap holds
    /// one wire key twice, when the heaps' keys mix `HeapItem` variants or hold
    /// a 128-bit key, or when a structural parameter overflows its `u32`
    /// metadata field.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        check_layout_dimensions(
            self.layer_size,
            self.heap_size,
            self.elephant_row,
            self.elephant_col,
            self.mouse_row,
            self.mouse_col,
        )
        .map_err(RmpEncodeError::Syntax)?;
        check_layer_size("UnivMonPyramid", self.layer_size).map_err(RmpEncodeError::Syntax)?;
        let field = |name: &str, value: usize| {
            u32::try_from(value).map_err(|_| {
                RmpEncodeError::Syntax(format!(
                    "ASAPv1 UnivMonPyramid envelope: {name} {value} exceeds the u32 metadata field"
                ))
            })
        };
        let layout = PyramidLayout {
            elephant_layers: field("elephant_layers", self.elephant_layers)?,
            elephant_row: field("elephant_row", self.elephant_row)?,
            elephant_col: field("elephant_col", self.elephant_col)?,
            mouse_row: field("mouse_row", self.mouse_row)?,
            mouse_col: field("mouse_col", self.mouse_col)?,
        };
        let geometry = geometry_of(self.layer_size, &layout);
        let state = pyramid_state(
            &self.l2_sketch_layers,
            &self.hh_layers,
            &geometry,
            self.bucket_size,
            update_mode_tag(self.update_mode),
            &self.candidate_complete,
        )?;
        let key_type = pyramid_key_type(&state.entries)?;
        let metadata = rmp_serde::to_vec_named(&pyramid_metadata::<DefaultXxHasher>(
            field("layer_size", self.layer_size)?,
            &layout,
            field("heap_size", self.heap_size)?,
            key_type,
        ))?;
        let payload = encode_pyramid(key_type, state)?;
        Ok(envelope::encode(PYRAMID_KIND, &metadata, &payload))
    }

    /// Deserializes a pyramid from an ASAPv1 MessagePack envelope. The layout
    /// and `key_type` are structural (they are properties of the stored
    /// sketch), so they are echoed back into the expected metadata; the hash
    /// spec is pinned against [`DefaultXxHasher`].
    ///
    /// Every state the algorithm could not have produced is rejected with an
    /// error rather than a panic, and no declared count sizes an allocation
    /// before the payload is measured against it.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != PYRAMID_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "UnivMonPyramid kind_id mismatch: stored {kind_id:?}, expected {PYRAMID_KIND:?}"
            )));
        }
        let meta: PyramidMetadata = from_slice(metadata)?;
        let layout = PyramidLayout {
            elephant_layers: meta.elephant_layers,
            elephant_row: meta.elephant_row,
            elephant_col: meta.elephant_col,
            mouse_row: meta.mouse_row,
            mouse_col: meta.mouse_col,
        };
        if meta
            != pyramid_metadata::<DefaultXxHasher>(
                meta.layer_size,
                &layout,
                meta.heap_size,
                &meta.key_type,
            )
        {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 UnivMonPyramid envelope: metadata mismatch".to_string(),
            ));
        }
        let (layer_size, heap_size) = (meta.layer_size as usize, meta.heap_size as usize);
        let (elephant_row, elephant_col, mouse_row, mouse_col) = (
            meta.elephant_row as usize,
            meta.elephant_col as usize,
            meta.mouse_row as usize,
            meta.mouse_col as usize,
        );
        check_layout_dimensions(
            layer_size,
            heap_size,
            elephant_row,
            elephant_col,
            mouse_row,
            mouse_col,
        )
        .map_err(RmpDecodeError::Uncategorized)?;
        check_layer_size("UnivMonPyramid", layer_size).map_err(RmpDecodeError::Uncategorized)?;
        let decoded = decode_pyramid(&meta.key_type, payload)?;
        // The declared layout is measured against the accumulators the payload
        // actually carries before the geometry is built from it.
        if accumulator_count(layer_size, &layout) != Some(decoded.l2.len()) {
            return Err(RmpDecodeError::Uncategorized(format!(
                "UnivMonPyramid declares a layout the payload's {} accumulators do not match",
                decoded.l2.len()
            )));
        }
        let geometry = geometry_of(layer_size, &layout);
        let (l2_sketch_layers, hh_layers, candidate_complete, bucket_size, mode_tag) =
            rebuild_layers(&geometry, heap_size, decoded)?;
        Ok(UnivMonPyramid {
            l2_sketch_layers,
            hh_layers,
            layer_size,
            elephant_layers: meta.elephant_layers as usize,
            elephant_row,
            elephant_col,
            mouse_row,
            mouse_col,
            heap_size,
            bucket_size,
            update_mode: update_mode_of(mode_tag)?,
            candidate_complete,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch_framework::univmon::MAX_LAYER_SIZE;
    use crate::sketch_framework::univmon::wire::PyramidPayload;
    use crate::sketches::countsketch_topk::CountL2HH;
    use crate::{
        CANONICAL_HASH_SEED, DataInput, HeapItem, L2HH, MATRIX_MAX_ROWS, RegularPath, SketchHasher,
        Vector2D,
    };

    fn populated() -> UnivMonPyramid {
        let mut pyramid = UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4);
        for (key, weight) in [("alpha", 5i64), ("beta", 7), ("gamma", 9), ("delta", 11)] {
            pyramid.insert(&DataInput::Str(key), weight);
        }
        pyramid
    }

    fn metadata_of(bytes: &[u8]) -> PyramidMetadata {
        let (_, metadata, _) = envelope::split(bytes).expect("split");
        from_slice(metadata).expect("metadata")
    }

    fn payload_of<K: for<'de> Deserialize<'de>>(bytes: &[u8]) -> PyramidPayload<K> {
        let (_, _, payload) = envelope::split(bytes).expect("split");
        from_slice(payload).expect("payload")
    }

    fn crafted<K: Serialize>(meta: &PyramidMetadata, payload: &PyramidPayload<K>) -> Vec<u8> {
        let metadata = rmp_serde::to_vec_named(meta).expect("metadata");
        let payload = rmp_serde::to_vec(payload).expect("payload");
        envelope::encode(PYRAMID_KIND, &metadata, &payload)
    }

    #[test]
    fn pyramid_round_trip_serialization() {
        let pyramid = populated();
        let encoded = pyramid.serialize_to_bytes().expect("serialize");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x11, 0x00]); // kind_id_len=2, kind_id=[0x11,0x00]

        let meta = metadata_of(&encoded);
        assert_eq!(meta.metadata_version, 1);
        assert_eq!(
            (
                meta.layer_size,
                meta.elephant_layers,
                meta.elephant_row,
                meta.elephant_col,
                meta.mouse_row,
                meta.mouse_col,
                meta.heap_size,
            ),
            (4, 2, 3, 16, 2, 8, 4)
        );
        assert_eq!(meta.key_type, "string");

        let decoded = UnivMonPyramid::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(decoded.bucket_size, pyramid.bucket_size);
        assert_eq!(decoded.calc_l1(), pyramid.calc_l1());
        assert_eq!(decoded.calc_l2(), pyramid.calc_l2());
        assert_eq!(decoded.calc_card(), pyramid.calc_card());
        assert_eq!(decoded.calc_entropy(), pyramid.calc_entropy());
        assert_eq!(decoded.candidate_complete, pyramid.candidate_complete);
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            encoded,
            "a decoded pyramid re-serialized to different bytes"
        );
    }

    /// The two tiers are derived from the layer's position: the elephant
    /// layers keep their dimensions and the mouse layers keep theirs.
    #[test]
    fn pyramid_two_tier_geometry_survives() {
        let pyramid = populated();
        let encoded = pyramid.serialize_to_bytes().expect("serialize");
        let decoded = UnivMonPyramid::deserialize_from_bytes(&encoded).expect("decode");
        for layer in 0..pyramid.layer_size {
            let L2HH::COUNT(original) = &pyramid.l2_sketch_layers[layer];
            let L2HH::COUNT(rebuilt) = &decoded.l2_sketch_layers[layer];
            let expected = if layer < 2 { (3, 16) } else { (2, 8) };
            assert_eq!((rebuilt.rows(), rebuilt.cols()), expected);
            assert_eq!(rebuilt.seed_idx(), layer);
            assert_eq!(
                original.as_storage().as_slice(),
                rebuilt.as_storage().as_slice()
            );
        }
    }

    /// Layers hold different numbers of entries: each layer's heap contents
    /// survive the round trip.
    #[test]
    fn pyramid_layers_with_different_heap_loads_round_trip() {
        let mut pyramid = UnivMonPyramid::new(8, 2, 3, 32, 2, 16, 4);
        for key in 0..40u64 {
            pyramid.insert(&DataInput::U64(key), 1 + (key as i64 % 5));
        }
        let loads: Vec<usize> = (0..pyramid.layer_size)
            .map(|i| pyramid.hh_layers[i].len())
            .collect();
        assert!(
            loads.windows(2).any(|pair| pair[0] != pair[1]),
            "expected layers of different heap loads, got {loads:?}"
        );

        let encoded = pyramid.serialize_to_bytes().expect("serialize");
        let decoded = UnivMonPyramid::deserialize_from_bytes(&encoded).expect("decode");
        for layer in 0..pyramid.layer_size {
            let (original, rebuilt) = (&pyramid.hh_layers[layer], &decoded.hh_layers[layer]);
            assert_eq!(original.len(), rebuilt.len(), "layer {layer} lost entries");
            for item in original.heap() {
                let found = rebuilt
                    .find_heap_item(&item.key)
                    .unwrap_or_else(|| panic!("layer {layer} lost {:?}", item.key));
                assert_eq!(rebuilt.heap()[found].count, item.count);
            }
        }
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);
    }

    /// A pyramid with no mouse layers is the one-tier case and still
    /// round-trips.
    #[test]
    fn pyramid_without_mouse_layers_round_trips() {
        let mut pyramid = UnivMonPyramid::new(4, 4, 2, 16, 2, 8, 3);
        pyramid.insert(&DataInput::U64(7), 3);
        let encoded = pyramid.serialize_to_bytes().expect("serialize");
        let decoded = UnivMonPyramid::deserialize_from_bytes(&encoded).expect("decode");
        for layer in 0..3 {
            let L2HH::COUNT(rebuilt) = &decoded.l2_sketch_layers[layer];
            assert_eq!((rebuilt.rows(), rebuilt.cols()), (2, 16));
        }
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);
    }

    /// An empty pyramid has exactly one encoding.
    #[test]
    fn pyramid_empty_has_one_encoding() {
        let left = UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4);
        let mut right = UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4);
        right.insert(&DataInput::Str("alpha"), 5);
        right.free();
        let encoded = left.serialize_to_bytes().expect("serialize");
        assert_eq!(right.serialize_to_bytes().expect("serialize"), encoded);
        assert_eq!(metadata_of(&encoded).key_type, "u64");

        let decoded = UnivMonPyramid::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);
    }

    /// `update_mode` and `candidate_complete` are state, not derived.
    #[test]
    fn pyramid_carries_update_mode_and_candidate_flags() {
        let mut terminal = UnivMonPyramid::new(2, 1, 2, 8, 2, 8, 3);
        for key in 0..20u64 {
            terminal.fast_insert(&DataInput::U64(key), 3);
        }
        assert!(
            terminal.candidate_complete.iter().any(|&flag| !flag),
            "expected an evicting layer"
        );
        let encoded = terminal.serialize_to_bytes().expect("serialize");
        assert_eq!(payload_of::<u64>(&encoded).update_mode, 2);

        let decoded = UnivMonPyramid::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(decoded.candidate_complete, terminal.candidate_complete);
        assert_eq!(decoded.calc_card(), terminal.calc_card());
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`.
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

    /// The pyramid hashes through the crate default, so it has one truthful
    /// profile: (a) it emits that profile, (b) a custom-profile envelope is
    /// different bytes, and (c) decode fails closed on it.
    #[test]
    fn pyramid_pins_its_hash_profile() {
        let encoded = populated().serialize_to_bytes().expect("serialize");
        let meta = metadata_of(&encoded);
        assert_eq!(meta.hash_profile_id, DefaultXxHasher::PROFILE_ID);
        assert_eq!(meta.seed_list, DefaultXxHasher::seed_list());

        let layout = PyramidLayout {
            elephant_layers: meta.elephant_layers,
            elephant_row: meta.elephant_row,
            elephant_col: meta.elephant_col,
            mouse_row: meta.mouse_row,
            mouse_col: meta.mouse_col,
        };
        let alt =
            pyramid_metadata::<AltHasher>(meta.layer_size, &layout, meta.heap_size, &meta.key_type);
        let (_, _, payload) = envelope::split(&encoded).expect("split");
        let forged = envelope::encode(
            PYRAMID_KIND,
            &rmp_serde::to_vec_named(&alt).expect("metadata"),
            payload,
        );
        assert_ne!(forged, encoded);
        assert!(
            UnivMonPyramid::deserialize_from_bytes(&forged).is_err(),
            "a custom-profile envelope must be rejected"
        );
    }

    /// Each family's envelope is rejected by the other three, and by a plain
    /// Count Sketch envelope.
    #[test]
    fn pyramid_rejects_foreign_kind_ids() {
        let count_sketch = crate::Count::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8)
            .serialize_to_bytes()
            .expect("serialize Count Sketch");
        let count_l2hh = CountL2HH::<DefaultXxHasher>::with_dimensions(2, 8)
            .serialize_to_bytes()
            .expect("serialize CountL2HH");
        let univmon = crate::UnivMon::init_univmon(4, 2, 8, 2)
            .serialize_to_bytes()
            .expect("serialize UnivMon");
        let univmon_q = crate::UnivMonQ::new(crate::UnivMonQConfig {
            levels: 2,
            width: 8,
            depth: 3,
            candidates: 4,
            ordered_samples: 4,
            ..Default::default()
        })
        .expect("config")
        .serialize_to_bytes()
        .expect("serialize UnivMonQ");

        for foreign in [count_sketch, count_l2hh, univmon, univmon_q] {
            assert!(
                UnivMonPyramid::deserialize_from_bytes(&foreign).is_err(),
                "a foreign envelope must not decode as a UnivMonPyramid"
            );
        }
    }

    /// Fail closed (not panic) on crafted layouts, layer counts and heap
    /// capacities, including a `layer_size` far larger than the payload
    /// carries.
    #[test]
    fn pyramid_rejects_crafted_shapes() {
        let encoded = populated().serialize_to_bytes().expect("serialize");
        let base = metadata_of(&encoded);
        let payload = payload_of::<String>(&encoded);

        let shaped = |layer_size, layout: PyramidLayout, heap_size| {
            pyramid_metadata::<DefaultXxHasher>(layer_size, &layout, heap_size, &base.key_type)
        };
        let layout = || PyramidLayout {
            elephant_layers: 2,
            elephant_row: 3,
            elephant_col: 16,
            mouse_row: 2,
            mouse_col: 8,
        };
        let cases = [
            shaped(u32::MAX, layout(), 4),
            shaped(0, layout(), 4),
            shaped(4, layout(), 0),
            shaped(
                4,
                PyramidLayout {
                    elephant_col: 0,
                    ..layout()
                },
                4,
            ),
            shaped(
                4,
                PyramidLayout {
                    elephant_row: 0,
                    ..layout()
                },
                4,
            ),
            shaped(
                4,
                PyramidLayout {
                    mouse_row: 0,
                    ..layout()
                },
                4,
            ),
            shaped(
                4,
                PyramidLayout {
                    mouse_col: 0,
                    ..layout()
                },
                4,
            ),
            shaped(
                4,
                PyramidLayout {
                    mouse_row: MATRIX_MAX_ROWS as u32,
                    mouse_col: 4096,
                    ..layout()
                },
                4,
            ),
            shaped(4, layout(), 1),
        ];
        for meta in cases {
            assert!(
                UnivMonPyramid::deserialize_from_bytes(&crafted(&meta, &payload)).is_err(),
                "a crafted layout must be rejected, not decoded"
            );
        }

        // A zero row count against an empty payload: the accumulator check
        // alone passes (0 rows over any layer count names 0 accumulators), so
        // the tiers' dimensions must be refused before the geometry is sized.
        let mut empty = payload_of::<String>(&encoded);
        empty.counts.clear();
        empty.l2.clear();
        empty.keys.clear();
        empty.heap_counts.clear();
        let zero_rows = PyramidLayout {
            elephant_row: 0,
            mouse_row: 0,
            ..layout()
        };
        let problem = UnivMonPyramid::deserialize_from_bytes(&crafted(
            &shaped(u32::MAX, zero_rows, 4),
            &empty,
        ))
        .expect_err("a zero tier row count must be rejected, not sized from")
        .to_string();
        assert!(problem.contains("must be non-zero"), "got {problem}");

        let mut short = payload_of::<String>(&encoded);
        short.heap_lens.pop();
        assert!(UnivMonPyramid::deserialize_from_bytes(&crafted(&base, &short)).is_err());

        let mut mode = payload_of::<String>(&encoded);
        mode.update_mode = 9;
        assert!(UnivMonPyramid::deserialize_from_bytes(&crafted(&base, &mode)).is_err());
    }

    /// The constructor asserts all four tier dimensions are positive, so a
    /// layout whose mouse layers never materialize must still declare theirs:
    /// a pyramid decoded with zero mouse dimensions panics on any merge
    /// against a normally built one.
    #[test]
    fn pyramid_rejects_zero_dimensions_of_a_tier_it_never_materializes() {
        let mut one_tier = UnivMonPyramid::new(4, 4, 2, 16, 2, 8, 3);
        one_tier.insert(&DataInput::U64(7), 3);
        let encoded = one_tier.serialize_to_bytes().expect("serialize");
        let mut meta = metadata_of(&encoded);
        meta.mouse_row = 0;
        meta.mouse_col = 0;
        let payload = payload_of::<u64>(&encoded);
        let problem = UnivMonPyramid::deserialize_from_bytes(&crafted(&meta, &payload))
            .expect_err("a zero mouse dimension must be rejected")
            .to_string();
        assert!(problem.contains("must be non-zero"), "got {problem}");
    }

    /// A layer's heap can seat one key twice — `update` compares residents with
    /// `HeapItem`'s equality and `NaN != NaN`, so a second `NaN` takes its own
    /// seat — and `rebuild_heap` refuses that payload. The shared encode path
    /// runs the same `check_distinct_keys` per layer, so the pyramid never
    /// emits it.
    #[test]
    fn pyramid_refuses_to_serialize_a_layer_holding_a_key_twice() {
        let mut nans = UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4);
        nans.hh_layers[2].update(&DataInput::F64(f64::NAN), 5);
        nans.hh_layers[2].update(&DataInput::F64(f64::NAN), 4);
        assert_eq!(nans.hh_layers[2].len(), 2, "the two NaNs did not both seat");
        let problem = nans
            .serialize_to_bytes()
            .expect_err("a layer holding one key twice must not serialize")
            .to_string();
        assert!(
            problem.contains("the same key appears twice"),
            "got {problem}"
        );
        assert!(problem.contains("layer 2"), "got {problem}");
    }

    /// The layout fields are public, so a caller can zero one after
    /// construction (the constructor asserts them positive). The decoder
    /// refuses a zero dimension, so the encoder must too. Each case below is
    /// built by writing the public field directly.
    #[test]
    fn pyramid_refuses_to_serialize_a_zero_dimension() {
        for zero in [
            "layer_size",
            "heap_size",
            "elephant_row",
            "elephant_col",
            "mouse_row",
            "mouse_col",
        ] {
            let mut pyramid = UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4);
            match zero {
                "layer_size" => pyramid.layer_size = 0,
                "heap_size" => pyramid.heap_size = 0,
                "elephant_row" => pyramid.elephant_row = 0,
                "elephant_col" => pyramid.elephant_col = 0,
                "mouse_row" => pyramid.mouse_row = 0,
                _ => pyramid.mouse_col = 0,
            }
            let problem = match pyramid.serialize_to_bytes() {
                Ok(_) => panic!("a zero {zero} must not serialize"),
                Err(err) => err.to_string(),
            };
            assert!(problem.contains("must be non-zero"), "got {problem}");
        }
    }

    /// The layer finder shifts a 64-bit hash by up to `layer_size - 1`, so
    /// [`MAX_LAYER_SIZE`] layers is the ceiling: the boundary round-trips and a
    /// deeper pyramid is refused at the decode door.
    #[test]
    fn pyramid_rejects_layers_past_the_shift_bound() {
        let deepest = UnivMonPyramid::new(4, 2, 2, 16, 2, 8, MAX_LAYER_SIZE);
        let encoded = deepest.serialize_to_bytes().expect("serialize");
        assert!(UnivMonPyramid::deserialize_from_bytes(&encoded).is_ok());

        let layers = MAX_LAYER_SIZE + 1;
        let mut meta = metadata_of(&encoded);
        meta.layer_size = layers as u32;
        let mut payload: PyramidPayload<u64> = payload_of(&encoded);
        payload.counts = vec![0; 2 * (2 * 16) + (layers - 2) * (2 * 8)];
        payload.l2 = vec![0; 2 * layers];
        payload.heap_lens = vec![0; layers];
        payload.candidate_complete = vec![true; layers];
        let problem = UnivMonPyramid::deserialize_from_bytes(&crafted(&meta, &payload))
            .expect_err("a pyramid past the shift bound must be rejected")
            .to_string();
        assert!(problem.contains("MAX_LAYER_SIZE"), "got {problem}");
    }

    #[test]
    #[should_panic(expected = "at most MAX_LAYER_SIZE")]
    fn pyramid_refuses_constructing_past_the_shift_bound() {
        UnivMonPyramid::new(4, 2, 2, 16, 2, 8, MAX_LAYER_SIZE + 1);
    }

    /// A pyramid whose layers disagree with their declared tier or seed index
    /// must not serialize.
    #[test]
    fn pyramid_rejects_serializing_an_inconsistent_layout() {
        let mut wrong_tier = UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4);
        wrong_tier.l2_sketch_layers[3] = L2HH::COUNT(CountL2HH::with_dimensions_and_seed(3, 16, 3));
        assert!(
            wrong_tier.serialize_to_bytes().is_err(),
            "a mouse layer holding elephant dimensions must not serialize"
        );

        let mut wrong_seed = UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4);
        wrong_seed.l2_sketch_layers[3] = L2HH::COUNT(CountL2HH::with_dimensions_and_seed(2, 8, 0));
        assert!(
            wrong_seed.serialize_to_bytes().is_err(),
            "a layer hashing at another layer's seed index must not serialize"
        );
    }
    /// Fail closed on an unexpected metadata key, and on a missing required
    /// one.
    #[test]
    fn pyramid_metadata_rejects_unknown_and_missing_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            layer_size: u32,
            elephant_layers: u32,
            elephant_row: u32,
            elephant_col: u32,
            mouse_row: u32,
            mouse_col: u32,
            heap_size: u32,
            key_type: String,
            bogus_field: u8, // key not in PyramidMetadata
        }
        #[derive(Serialize)]
        struct WithoutMouseCol {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            layer_size: u32,
            elephant_layers: u32,
            elephant_row: u32,
            elephant_col: u32,
            mouse_row: u32,
            heap_size: u32,
            key_type: String,
        }
        let layout = PyramidLayout {
            elephant_layers: 2,
            elephant_row: 3,
            elephant_col: 16,
            mouse_row: 2,
            mouse_col: 8,
        };
        let m = pyramid_metadata::<DefaultXxHasher>(4, &layout, 4, "u64");
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            layer_size: m.layer_size,
            elephant_layers: m.elephant_layers,
            elephant_row: m.elephant_row,
            elephant_col: m.elephant_col,
            mouse_row: m.mouse_row,
            mouse_col: m.mouse_col,
            heap_size: m.heap_size,
            key_type: m.key_type.clone(),
            bogus_field: 7,
        };
        let without = WithoutMouseCol {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            layer_size: m.layer_size,
            elephant_layers: m.elephant_layers,
            elephant_row: m.elephant_row,
            elephant_col: m.elephant_col,
            mouse_row: m.mouse_row,
            heap_size: m.heap_size,
            key_type: m.key_type.clone(),
        };
        assert!(
            from_slice::<PyramidMetadata>(&rmp_serde::to_vec_named(&extra).unwrap()).is_err(),
            "an unknown metadata key must be rejected"
        );
        assert!(
            from_slice::<PyramidMetadata>(&rmp_serde::to_vec_named(&without).unwrap()).is_err(),
            "a missing required key must be rejected"
        );
    }
}
