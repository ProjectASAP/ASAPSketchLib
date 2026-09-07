//! ASAPv1 wire serialization for [`UnivMon`], plus the pyramid payload
//! [`UnivMonPyramid`] reuses.
//!
//! Child submodule of [`crate::sketch_framework::univmon`]: it holds the
//! metadata/payload DTOs, the kind_id constant and the `serialize_to_bytes` /
//! `deserialize_from_bytes` impls, while the algorithm lives in the parent
//! module file. Being a descendant module, it reads the private `update_mode`
//! and `candidate_complete` fields directly without widening any field
//! visibility. See `docs/asapv1_wire_format.md`.
//!
//! UnivMon is one algorithm — a single kind_id `0x10 0x00`. The pyramid shape
//! (`layer_size`, `sketch_row`, `sketch_col`, `heap_size`) and the heaps'
//! `key_type` are construction config and live in the metadata, so the payload
//! carries the layers' raw state.
//!
//! ## Layers are inlined, not nested
//!
//! Each layer is a `CountL2HH` plus an `HHHeap`. Their state is written
//! straight into UnivMon's own positional array through
//! [`l2hh_wire::layer_state`] and [`heap_entries`]; no layer carries an
//! envelope, a magic or a metadata map of its own.
//!
//! ## The hash profile is the crate default
//!
//! UnivMon hashes through `hash64_seeded` / `hash_item64_seeded` and holds
//! `CountL2HH<DefaultXxHasher>` layers, so it has no hasher type parameter and
//! its metadata is derived from [`DefaultXxHasher`]'s [`HashProfile`].
//!
//! It carries no seed-index key: the bottom-layer finder hashes at
//! `BOTTOM_LAYER_FINDER` unconditionally — a fixed part of the algorithm, not
//! a profile choice — and layer `i`'s counter hashes at seed index `i`, which
//! the layer's position already gives.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::message_pack_format::envelope;
use crate::message_pack_format::wire_key::WireBytes;
use crate::sketches::countminsketch_topk::heap_wire::{
    EMPTY_KEY_TYPE, check_distinct_keys, heap_entries, key_type_of, rebuild_heap,
};
use crate::sketches::countsketch_topk::l2hh_wire;
use crate::{DefaultXxHasher, HHHeap, HashProfile, HeapItem, L2HH, Vector1D};

use super::{MAX_LAYER_SIZE, UnivMon, UnivMonUpdateMode};

/// UnivMon kind_id: family `0x10`, single algorithm variant `0x00`.
const UNIVMON_KIND: &[u8] = &[0x10, 0x00];

/// UnivMon descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it). Hash-spec fields first, then the
/// structural params `layer_size` / `sketch_row` / `sketch_col` / `heap_size`
/// / `key_type`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnivMonMetadata {
    pub(crate) metadata_version: u8,
    pub(crate) hash_profile_id: String,
    pub(crate) hash_algorithm: String,
    pub(crate) seed_derivation: String,
    pub(crate) input_encoding: String,
    pub(crate) seed_list: Vec<u64>,
    pub(crate) layer_size: u32,
    pub(crate) sketch_row: u32,
    pub(crate) sketch_col: u32,
    pub(crate) heap_size: u32,
    pub(crate) key_type: String,
}

/// Builds the UnivMon descriptor metadata from the hasher's [`HashProfile`],
/// so the wire bytes truthfully describe how the sketch was hashed. Every
/// layer shares the same dimensions and heap capacity.
pub(crate) fn univmon_metadata<H: HashProfile>(
    layer_size: u32,
    sketch_row: u32,
    sketch_col: u32,
    heap_size: u32,
    key_type: &str,
) -> UnivMonMetadata {
    UnivMonMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        layer_size,
        sketch_row,
        sketch_col,
        heap_size,
        key_type: key_type.to_string(),
    }
}

/// The pyramid payload both UnivMon kinds share, a msgpack **array**
/// (`to_vec`, positional):
/// `[counts, l2, heap_lens, keys, heap_counts, candidate_complete,
/// bucket_size, update_mode]`.
///
/// `counts` and `l2` concatenate the layers' `CountL2HH` state in layer order;
/// `heap_lens` cuts the parallel `keys` / `heap_counts` arrays into per-layer
/// runs, with `keys` typed by the metadata `key_type`.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PyramidPayload<K> {
    pub(crate) counts: Vec<i64>,
    pub(crate) l2: Vec<i64>,
    pub(crate) heap_lens: Vec<u32>,
    pub(crate) keys: Vec<K>,
    pub(crate) heap_counts: Vec<i64>,
    pub(crate) candidate_complete: Vec<bool>,
    pub(crate) bucket_size: u64,
    pub(crate) update_mode: u8,
}

/// Wire tag of the update mode: `0` unset, `1` standard, `2` terminal-only.
/// The mode is acquired from the stream rather than configured, and it selects
/// the query recurrence, so it is payload state.
pub(crate) fn update_mode_tag(mode: UnivMonUpdateMode) -> u8 {
    match mode {
        UnivMonUpdateMode::Unset => 0,
        UnivMonUpdateMode::Standard => 1,
        UnivMonUpdateMode::Terminal => 2,
    }
}

/// Reads the update-mode tag, rejecting any value outside the three modes.
pub(crate) fn update_mode_of(tag: u8) -> Result<UnivMonUpdateMode, RmpDecodeError> {
    match tag {
        0 => Ok(UnivMonUpdateMode::Unset),
        1 => Ok(UnivMonUpdateMode::Standard),
        2 => Ok(UnivMonUpdateMode::Terminal),
        other => Err(RmpDecodeError::Uncategorized(format!(
            "ASAPv1 UnivMon envelope: update_mode {other} is not a wire mode"
        ))),
    }
}

/// Checks a pyramid's layer count against [`MAX_LAYER_SIZE`], naming `sketch`
/// in the error. Shared by both UnivMon kinds' encoders and decoders.
pub(crate) fn check_layer_size(sketch: &str, layer_size: usize) -> Result<(), String> {
    if layer_size > MAX_LAYER_SIZE {
        return Err(format!(
            "{sketch} layer_size {layer_size} exceeds MAX_LAYER_SIZE {MAX_LAYER_SIZE}"
        ));
    }
    Ok(())
}

/// Rejects a UnivMon shape the algorithm never has: `layer_size`,
/// `sketch_row`, `sketch_col` and `heap_size` are each positive. The encoder
/// and the decoder both call this, so the two doors cannot drift.
pub(crate) fn check_univmon_dimensions(
    layer_size: usize,
    sketch_row: usize,
    sketch_col: usize,
    heap_size: usize,
) -> Result<(), String> {
    if layer_size == 0 || sketch_row == 0 || sketch_col == 0 || heap_size == 0 {
        return Err(format!(
            "UnivMon layer_size, sketch_row, sketch_col and heap_size must be non-zero: layer_size={layer_size}, sketch_row={sketch_row}, sketch_col={sketch_col}, heap_size={heap_size}"
        ));
    }
    Ok(())
}

/// The layers' state in emitted order, ready to be packed.
pub(crate) struct PyramidState<'a> {
    pub(crate) counts: Vec<i64>,
    pub(crate) l2: Vec<i64>,
    pub(crate) heap_lens: Vec<u32>,
    pub(crate) entries: Vec<(&'a HeapItem, i64)>,
    pub(crate) candidate_complete: Vec<bool>,
    pub(crate) bucket_size: u64,
    pub(crate) update_mode: u8,
}

/// Reads every layer's counters, accumulators and heap entries in the emitted
/// order: layers ascending, and within a layer the heap's own emitted order
/// (descending count, ties by the total order over the key).
///
/// Fails when a layer's geometry disagrees with the declared one, when a
/// layer's seed index is not its position, when a layer's heap holds the same
/// wire key twice, or when the heaps hold more entries than `u32` can name.
pub(crate) fn pyramid_state<'a>(
    sketches: &'a Vector1D<L2HH>,
    heaps: &'a Vector1D<HHHeap>,
    geometry: &[(usize, usize)],
    bucket_size: usize,
    update_mode: u8,
    candidate_complete: &[bool],
) -> Result<PyramidState<'a>, RmpEncodeError> {
    let layer_size = geometry.len();
    if sketches.len() != layer_size || heaps.len() != layer_size {
        return Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 UnivMon envelope: {} counters and {} heaps over {layer_size} layers",
            sketches.len(),
            heaps.len()
        )));
    }
    if candidate_complete.len() != layer_size {
        return Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 UnivMon envelope: {} candidate flags over {layer_size} layers",
            candidate_complete.len()
        )));
    }
    let bucket_size = u64::try_from(bucket_size).map_err(|_| {
        RmpEncodeError::Syntax(
            "ASAPv1 UnivMon envelope: bucket_size exceeds the u64 payload field".to_string(),
        )
    })?;
    let mut state = PyramidState {
        counts: Vec::new(),
        l2: Vec::new(),
        heap_lens: Vec::with_capacity(layer_size),
        entries: Vec::new(),
        candidate_complete: candidate_complete.to_vec(),
        bucket_size,
        update_mode,
    };
    for (layer, &(rows, cols)) in geometry.iter().enumerate() {
        let L2HH::COUNT(counter) = &sketches[layer];
        if (counter.rows(), counter.cols()) != (rows, cols) {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 UnivMon envelope: layer {layer} is {}x{} against the declared {rows}x{cols}",
                counter.rows(),
                counter.cols()
            )));
        }
        if counter.seed_idx() != layer {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 UnivMon envelope: layer {layer} hashes at seed index {}",
                counter.seed_idx()
            )));
        }
        let (counts, l2) = l2hh_wire::layer_state(counter)?;
        state.counts.extend_from_slice(counts);
        state.l2.extend_from_slice(l2);
        let entries = heap_entries(&heaps[layer]);
        check_distinct_keys(entries.iter().map(|(key, _)| *key))
            .map_err(|problem| RmpEncodeError::Syntax(format!("{problem} (layer {layer})")))?;
        state
            .heap_lens
            .push(u32::try_from(entries.len()).map_err(|_| {
                RmpEncodeError::Syntax(format!(
                    "ASAPv1 UnivMon envelope: layer {layer} holds more entries than u32 can name"
                ))
            })?);
        state.entries.extend(entries);
    }
    Ok(state)
}

/// The `key_type` the payload will be written in, taken from the first key in
/// emitted order across every layer. Rejects a 128-bit key outright; the
/// homogeneity of the rest is enforced by [`encode_pyramid`].
pub(crate) fn pyramid_key_type(
    entries: &[(&HeapItem, i64)],
) -> Result<&'static str, RmpEncodeError> {
    match entries.first() {
        None => Ok(EMPTY_KEY_TYPE),
        Some((key, _)) => key_type_of(key).ok_or_else(|| {
            RmpEncodeError::Syntax("ASAPv1 UnivMon: 128-bit keys are not wire types".to_string())
        }),
    }
}

fn mixed_variant_error(key_type: &str, key: &HeapItem) -> RmpEncodeError {
    RmpEncodeError::Syntax(format!(
        "ASAPv1 UnivMon: keys mix variants — key_type is {key_type}, but a {} key is held",
        key_type_of(key).unwrap_or("128-bit")
    ))
}

/// Packs the layers into the positional payload, with `keys` typed by
/// `key_type`. Any key that is not of that variant fails the encode.
pub(crate) fn encode_pyramid(
    key_type: &str,
    state: PyramidState<'_>,
) -> Result<Vec<u8>, RmpEncodeError> {
    let heap_counts: Vec<i64> = state.entries.iter().map(|entry| entry.1).collect();

    macro_rules! pack {
        ($variant:ident) => {{
            let mut keys = Vec::with_capacity(state.entries.len());
            for (key, _) in &state.entries {
                match key {
                    HeapItem::$variant(value) => keys.push(*value),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&PyramidPayload {
                counts: state.counts,
                l2: state.l2,
                heap_lens: state.heap_lens,
                keys,
                heap_counts,
                candidate_complete: state.candidate_complete,
                bucket_size: state.bucket_size,
                update_mode: state.update_mode,
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
            let mut keys = Vec::with_capacity(state.entries.len());
            for (key, _) in &state.entries {
                match key {
                    HeapItem::String(value) => keys.push(value.clone()),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&PyramidPayload {
                counts: state.counts,
                l2: state.l2,
                heap_lens: state.heap_lens,
                keys,
                heap_counts,
                candidate_complete: state.candidate_complete,
                bucket_size: state.bucket_size,
                update_mode: state.update_mode,
            })
        }
        "bytes" => {
            let mut keys = Vec::with_capacity(state.entries.len());
            for (key, _) in &state.entries {
                match key {
                    HeapItem::Bytes(value) => keys.push(WireBytes(value.clone())),
                    _ => return Err(mixed_variant_error(key_type, key)),
                }
            }
            rmp_serde::to_vec(&PyramidPayload {
                counts: state.counts,
                l2: state.l2,
                heap_lens: state.heap_lens,
                keys,
                heap_counts,
                candidate_complete: state.candidate_complete,
                bucket_size: state.bucket_size,
                update_mode: state.update_mode,
            })
        }
        other => Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 UnivMon: key_type {other:?} is not a wire key type"
        ))),
    }
}

/// The layer state one pyramid payload decodes into.
pub(crate) struct DecodedPyramid {
    pub(crate) counts: Vec<i64>,
    pub(crate) l2: Vec<i64>,
    pub(crate) heap_lens: Vec<u32>,
    pub(crate) entries: Vec<(HeapItem, i64)>,
    pub(crate) candidate_complete: Vec<bool>,
    pub(crate) bucket_size: u64,
    pub(crate) update_mode: u8,
}

/// Reads the payload with `keys` typed by the metadata `key_type`. Rejects an
/// unknown `key_type` and parallel arrays of unequal length; a `keys` array
/// whose msgpack types do not match `key_type` fails in `from_slice`.
pub(crate) fn decode_pyramid(
    key_type: &str,
    payload: &[u8],
) -> Result<DecodedPyramid, RmpDecodeError> {
    macro_rules! unpack {
        ($variant:ident, $ty:ty) => {{
            let decoded: PyramidPayload<$ty> = from_slice(payload)?;
            (
                decoded
                    .keys
                    .into_iter()
                    .map(HeapItem::$variant)
                    .collect::<Vec<HeapItem>>(),
                DecodedPyramid {
                    counts: decoded.counts,
                    l2: decoded.l2,
                    heap_lens: decoded.heap_lens,
                    entries: Vec::new(),
                    candidate_complete: decoded.candidate_complete,
                    bucket_size: decoded.bucket_size,
                    update_mode: decoded.update_mode,
                },
                decoded.heap_counts,
            )
        }};
    }

    let (keys, mut decoded, heap_counts) = match key_type {
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
            let decoded: PyramidPayload<WireBytes> = from_slice(payload)?;
            (
                decoded
                    .keys
                    .into_iter()
                    .map(|key| HeapItem::Bytes(key.into_vec()))
                    .collect::<Vec<HeapItem>>(),
                DecodedPyramid {
                    counts: decoded.counts,
                    l2: decoded.l2,
                    heap_lens: decoded.heap_lens,
                    entries: Vec::new(),
                    candidate_complete: decoded.candidate_complete,
                    bucket_size: decoded.bucket_size,
                    update_mode: decoded.update_mode,
                },
                decoded.heap_counts,
            )
        }
        other => {
            return Err(RmpDecodeError::Uncategorized(format!(
                "ASAPv1 UnivMon: key_type {other:?} is not a wire key type"
            )));
        }
    };

    if keys.len() != heap_counts.len() {
        return Err(RmpDecodeError::Uncategorized(format!(
            "ASAPv1 UnivMon: {} keys against {} heap counts",
            keys.len(),
            heap_counts.len()
        )));
    }
    decoded.entries = keys.into_iter().zip(heap_counts).collect();
    Ok(decoded)
}

/// The counters and heaps one decoded pyramid rebuilds into.
pub(crate) type DecodedLayers = (Vector1D<L2HH>, Vector1D<HHHeap>, Vec<bool>, usize, u8);

/// Seats the decoded layers, rebuilding every counter matrix and heap index.
///
/// Each declared length is measured against what the payload actually carries
/// before anything is sized from it, so neither a crafted geometry nor a
/// crafted `heap_size` drives an allocation.
pub(crate) fn rebuild_layers(
    geometry: &[(usize, usize)],
    heap_size: usize,
    decoded: DecodedPyramid,
) -> Result<DecodedLayers, RmpDecodeError> {
    let layer_size = geometry.len();
    let complain = |problem: String| RmpDecodeError::Uncategorized(problem);
    if decoded.heap_lens.len() != layer_size {
        return Err(complain(format!(
            "UnivMon heap_lens length {} != layer_size {layer_size}",
            decoded.heap_lens.len()
        )));
    }
    if decoded.candidate_complete.len() != layer_size {
        return Err(complain(format!(
            "UnivMon candidate_complete length {} != layer_size {layer_size}",
            decoded.candidate_complete.len()
        )));
    }
    let mut cells = 0usize;
    let mut accumulators = 0usize;
    for &(rows, cols) in geometry {
        l2hh_wire::check_dimensions(rows, cols).map_err(&complain)?;
        cells = cells
            .checked_add(rows.checked_mul(cols).ok_or_else(|| {
                complain(format!("UnivMon layer geometry {rows}x{cols} overflows"))
            })?)
            .ok_or_else(|| complain("UnivMon total counter count overflows".to_string()))?;
        accumulators = accumulators
            .checked_add(rows)
            .ok_or_else(|| complain("UnivMon total accumulator count overflows".to_string()))?;
    }
    if decoded.counts.len() != cells {
        return Err(complain(format!(
            "UnivMon counts length {} != the declared layers' {cells} cells",
            decoded.counts.len()
        )));
    }
    if decoded.l2.len() != accumulators {
        return Err(complain(format!(
            "UnivMon l2 length {} != the declared layers' {accumulators} rows",
            decoded.l2.len()
        )));
    }
    let mut seated = 0usize;
    for &len in &decoded.heap_lens {
        seated = seated
            .checked_add(len as usize)
            .ok_or_else(|| complain("UnivMon total heap entry count overflows".to_string()))?;
    }
    if decoded.entries.len() != seated {
        return Err(complain(format!(
            "UnivMon carries {} heap entries against the declared {seated}",
            decoded.entries.len()
        )));
    }

    let mut sketches = Vec::with_capacity(layer_size);
    let mut heaps = Vec::with_capacity(layer_size);
    let mut entries = decoded.entries.into_iter();
    let (mut cell, mut accumulator) = (0usize, 0usize);
    for (layer, &(rows, cols)) in geometry.iter().enumerate() {
        let next_cell = cell + rows * cols;
        let next_accumulator = accumulator + rows;
        sketches.push(L2HH::COUNT(l2hh_wire::rebuild_layer::<DefaultXxHasher>(
            rows,
            cols,
            layer,
            &decoded.counts[cell..next_cell],
            &decoded.l2[accumulator..next_accumulator],
        )?));
        cell = next_cell;
        accumulator = next_accumulator;
        let run: Vec<(HeapItem, i64)> = entries
            .by_ref()
            .take(decoded.heap_lens[layer] as usize)
            .collect();
        heaps.push(rebuild_heap(heap_size, run)?);
    }
    let bucket_size = usize::try_from(decoded.bucket_size)
        .map_err(|_| complain("UnivMon bucket_size exceeds this target's usize".to_string()))?;
    Ok((
        Vector1D::from_vec(sketches),
        Vector1D::from_vec(heaps),
        decoded.candidate_complete,
        bucket_size,
        decoded.update_mode,
    ))
}

// Wire serialization for UnivMon. `wire` is a descendant of the sketch module,
// so this impl reads the private fields directly.
impl UnivMon {
    /// Serializes the pyramid into an ASAPv1 MessagePack envelope
    /// (kind_id `0x10 0x00`). The metadata is derived from
    /// [`DefaultXxHasher`]'s [`HashProfile`], the hasher UnivMon is built on.
    ///
    /// Fails when a structural parameter is zero, when a layer's geometry or
    /// seed index disagrees with the declared pyramid, when a layer's heap
    /// holds one wire key twice, when the heaps' keys mix `HeapItem` variants
    /// or hold a 128-bit key, or when a structural parameter overflows its
    /// `u32` metadata field.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        check_univmon_dimensions(
            self.layer_size,
            self.sketch_row,
            self.sketch_col,
            self.heap_size,
        )
        .map_err(RmpEncodeError::Syntax)?;
        check_layer_size("UnivMon", self.layer_size).map_err(RmpEncodeError::Syntax)?;
        let geometry = vec![(self.sketch_row, self.sketch_col); self.layer_size];
        let state = pyramid_state(
            &self.l2_sketch_layers,
            &self.hh_layers,
            &geometry,
            self.bucket_size,
            update_mode_tag(self.update_mode),
            &self.candidate_complete,
        )?;
        let key_type = pyramid_key_type(&state.entries)?;
        let field = |name: &str, value: usize| {
            u32::try_from(value).map_err(|_| {
                RmpEncodeError::Syntax(format!(
                    "ASAPv1 UnivMon envelope: {name} {value} exceeds the u32 metadata field"
                ))
            })
        };
        let metadata = rmp_serde::to_vec_named(&univmon_metadata::<DefaultXxHasher>(
            field("layer_size", self.layer_size)?,
            field("sketch_row", self.sketch_row)?,
            field("sketch_col", self.sketch_col)?,
            field("heap_size", self.heap_size)?,
            key_type,
        ))?;
        let payload = encode_pyramid(key_type, state)?;
        Ok(envelope::encode(UNIVMON_KIND, &metadata, &payload))
    }

    /// Deserializes a pyramid from an ASAPv1 MessagePack envelope. The shape
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
        if kind_id != UNIVMON_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "UnivMon kind_id mismatch: stored {kind_id:?}, expected {UNIVMON_KIND:?}"
            )));
        }
        let meta: UnivMonMetadata = from_slice(metadata)?;
        if meta
            != univmon_metadata::<DefaultXxHasher>(
                meta.layer_size,
                meta.sketch_row,
                meta.sketch_col,
                meta.heap_size,
                &meta.key_type,
            )
        {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 UnivMon envelope: metadata mismatch".to_string(),
            ));
        }
        let (layer_size, sketch_row, sketch_col, heap_size) = (
            meta.layer_size as usize,
            meta.sketch_row as usize,
            meta.sketch_col as usize,
            meta.heap_size as usize,
        );
        check_univmon_dimensions(layer_size, sketch_row, sketch_col, heap_size)
            .map_err(RmpDecodeError::Uncategorized)?;
        check_layer_size("UnivMon", layer_size).map_err(RmpDecodeError::Uncategorized)?;
        let decoded = decode_pyramid(&meta.key_type, payload)?;
        // The declared layer count is measured against the accumulators the
        // payload actually carries before the geometry is built from it.
        if sketch_row.checked_mul(layer_size) != Some(decoded.l2.len()) {
            return Err(RmpDecodeError::Uncategorized(format!(
                "UnivMon declares {layer_size} layers of {sketch_row} rows against a payload carrying {} accumulators",
                decoded.l2.len()
            )));
        }
        let geometry = vec![(sketch_row, sketch_col); layer_size];
        let (l2_sketch_layers, hh_layers, candidate_complete, bucket_size, mode_tag) =
            rebuild_layers(&geometry, heap_size, decoded)?;
        Ok(UnivMon {
            l2_sketch_layers,
            hh_layers,
            layer_size,
            sketch_row,
            sketch_col,
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
    use crate::{
        CANONICAL_HASH_SEED, DataInput, MATRIX_MAX_ROWS, RegularPath, SketchHasher, Vector2D,
    };

    fn populated() -> UnivMon {
        let mut um = UnivMon::init_univmon(4, 2, 16, 3);
        for (key, weight) in [("alpha", 5i64), ("beta", 7), ("gamma", 9), ("delta", 11)] {
            um.insert(&DataInput::Str(key), weight);
        }
        um
    }

    fn metadata_of(bytes: &[u8]) -> UnivMonMetadata {
        let (_, metadata, _) = envelope::split(bytes).expect("split");
        from_slice(metadata).expect("metadata")
    }

    fn crafted<K: Serialize>(meta: &UnivMonMetadata, payload: &PyramidPayload<K>) -> Vec<u8> {
        let metadata = rmp_serde::to_vec_named(meta).expect("metadata");
        let payload = rmp_serde::to_vec(payload).expect("payload");
        envelope::encode(UNIVMON_KIND, &metadata, &payload)
    }

    fn payload_of<K: for<'de> Deserialize<'de>>(bytes: &[u8]) -> PyramidPayload<K> {
        let (_, _, payload) = envelope::split(bytes).expect("split");
        from_slice(payload).expect("payload")
    }

    /// Every layer is a CountL2HH matrix, so a layer past the seed list's row
    /// bound is refused on both sides.
    #[test]
    fn univmon_rejects_too_many_layer_rows() {
        let rows = MATRIX_MAX_ROWS + 1;
        assert!(
            UnivMon::init_univmon(4, rows, 16, 3)
                .serialize_to_bytes()
                .is_err(),
            "a layer past MATRIX_MAX_ROWS must not serialize"
        );

        let um = populated();
        let encoded = um.serialize_to_bytes().expect("serialize");
        let mut meta = metadata_of(&encoded);
        meta.sketch_row = rows as u32;
        let layers = meta.layer_size as usize;
        let cols = meta.sketch_col as usize;
        // Sized to the crafted geometry, so the row bound is what bites.
        let mut payload: PyramidPayload<String> = payload_of(&encoded);
        payload.counts = vec![0; rows * cols * layers];
        payload.l2 = vec![0; rows * layers];
        let problem = UnivMon::deserialize_from_bytes(&crafted(&meta, &payload))
            .expect_err("layer rows past MATRIX_MAX_ROWS must be rejected")
            .to_string();
        assert!(problem.contains("MATRIX_MAX_ROWS"), "got {problem}");

        // The boundary itself is eligible.
        assert!(
            UnivMon::init_univmon(4, MATRIX_MAX_ROWS, 16, 3)
                .serialize_to_bytes()
                .is_ok()
        );
    }

    #[test]
    fn univmon_round_trip_serialization() {
        let um = populated();
        let encoded = um.serialize_to_bytes().expect("serialize UnivMon");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x10, 0x00]); // kind_id_len=2, kind_id=[0x10,0x00]

        let meta = metadata_of(&encoded);
        assert_eq!(meta.metadata_version, 1);
        assert_eq!(
            (
                meta.layer_size,
                meta.sketch_row,
                meta.sketch_col,
                meta.heap_size
            ),
            (3, 2, 16, 4)
        );
        assert_eq!(meta.key_type, "string");

        let decoded = UnivMon::deserialize_from_bytes(&encoded).expect("deserialize UnivMon");
        assert_eq!(decoded.layer_size, um.layer_size);
        assert_eq!(decoded.bucket_size, um.bucket_size);
        assert_eq!(decoded.calc_l1(), um.calc_l1());
        assert_eq!(decoded.calc_l2(), um.calc_l2());
        assert_eq!(decoded.calc_card(), um.calc_card());
        assert_eq!(decoded.calc_entropy(), um.calc_entropy());
        assert_eq!(decoded.candidates_complete(), um.candidates_complete());
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            encoded,
            "a decoded pyramid re-serialized to different bytes"
        );
    }

    /// Layers hold different numbers of entries: each layer's heap contents
    /// and emitted order survive the round trip.
    #[test]
    fn univmon_layers_with_different_heap_loads_round_trip() {
        let mut um = UnivMon::init_univmon(8, 2, 16, 4);
        for key in 0..40u64 {
            um.insert(&DataInput::U64(key), 1 + (key as i64 % 5));
        }
        let loads: Vec<usize> = (0..um.layer_size).map(|i| um.hh_layers[i].len()).collect();
        assert!(
            loads.windows(2).any(|pair| pair[0] != pair[1]),
            "expected layers of different heap loads, got {loads:?}"
        );

        let encoded = um.serialize_to_bytes().expect("serialize");
        let decoded = UnivMon::deserialize_from_bytes(&encoded).expect("decode");
        for layer in 0..um.layer_size {
            let (original, rebuilt) = (&um.hh_layers[layer], &decoded.hh_layers[layer]);
            assert_eq!(original.len(), rebuilt.len(), "layer {layer} lost entries");
            assert_eq!(rebuilt.capacity(), 8);
            for item in original.heap() {
                let found = rebuilt
                    .find_heap_item(&item.key)
                    .unwrap_or_else(|| panic!("layer {layer} lost {:?}", item.key));
                assert_eq!(rebuilt.heap()[found].count, item.count);
            }
        }
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);
    }

    /// `update_mode` and `candidate_complete` are state, not derived: a
    /// terminal-only pyramid and an incomplete candidate set both survive.
    #[test]
    fn univmon_carries_update_mode_and_candidate_flags() {
        let mut terminal = UnivMon::init_univmon(2, 2, 8, 3);
        for key in 0..20u64 {
            terminal.fast_insert(&DataInput::U64(key), 3);
        }
        assert!(
            terminal.candidates_complete().iter().any(|&flag| !flag),
            "expected an evicting layer"
        );
        let encoded = terminal.serialize_to_bytes().expect("serialize");
        assert_eq!(payload_of::<u64>(&encoded).update_mode, 2);

        let decoded = UnivMon::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(
            decoded.candidates_complete(),
            terminal.candidates_complete()
        );
        assert_eq!(decoded.calc_card(), terminal.calc_card());
        assert_eq!(decoded.calc_entropy(), terminal.calc_entropy());
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);

        // A layer marked incomplete keeps its widened threshold: flipping the
        // flag changes what the pyramid reports.
        let mut forged = payload_of::<u64>(&encoded);
        forged.candidate_complete.fill(true);
        let meta = metadata_of(&encoded);
        let relaxed = UnivMon::deserialize_from_bytes(&crafted(&meta, &forged)).expect("decode");
        assert_ne!(
            relaxed.calc_card(),
            terminal.calc_card(),
            "candidate_complete had no effect on a query"
        );
    }

    /// An empty pyramid has exactly one encoding, and its heaps decode empty.
    #[test]
    fn univmon_empty_has_one_encoding() {
        let left = UnivMon::init_univmon(4, 2, 16, 3);
        let mut right = UnivMon::init_univmon(4, 2, 16, 3);
        right.insert(&DataInput::Str("alpha"), 5);
        right.free();

        let encoded = left.serialize_to_bytes().expect("serialize");
        assert_eq!(right.serialize_to_bytes().expect("serialize"), encoded);
        assert_eq!(metadata_of(&encoded).key_type, EMPTY_KEY_TYPE);

        let decoded = UnivMon::deserialize_from_bytes(&encoded).expect("decode");
        assert!((0..3).all(|layer| decoded.hh_layers[layer].is_empty()));
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

    /// UnivMon hashes through the crate default, so it has one truthful
    /// profile: (a) it emits that profile, (b) a custom-profile envelope is
    /// different bytes, and (c) decode fails closed on it.
    #[test]
    fn univmon_pins_its_hash_profile() {
        let um = populated();
        let encoded = um.serialize_to_bytes().expect("serialize");
        let meta = metadata_of(&encoded);
        assert_eq!(meta.hash_profile_id, DefaultXxHasher::PROFILE_ID);
        assert_eq!(meta.seed_list, DefaultXxHasher::seed_list());

        let alt = univmon_metadata::<AltHasher>(
            meta.layer_size,
            meta.sketch_row,
            meta.sketch_col,
            meta.heap_size,
            &meta.key_type,
        );
        let (_, _, payload) = envelope::split(&encoded).expect("split");
        let forged = envelope::encode(
            UNIVMON_KIND,
            &rmp_serde::to_vec_named(&alt).expect("metadata"),
            payload,
        );
        assert_ne!(forged, encoded);
        assert!(
            UnivMon::deserialize_from_bytes(&forged).is_err(),
            "a custom-profile envelope must be rejected"
        );
    }

    /// Each family's envelope is rejected by the other three, and by a plain
    /// Count Sketch envelope.
    #[test]
    fn univmon_rejects_foreign_kind_ids() {
        let count_sketch = crate::Count::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8)
            .serialize_to_bytes()
            .expect("serialize Count Sketch");
        let count_l2hh =
            crate::sketches::countsketch_topk::CountL2HH::<DefaultXxHasher>::with_dimensions(2, 8)
                .serialize_to_bytes()
                .expect("serialize CountL2HH");
        let pyramid = crate::UnivMonPyramid::new(4, 1, 2, 8, 2, 4, 2)
            .serialize_to_bytes()
            .expect("serialize UnivMonPyramid");
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

        for foreign in [count_sketch, count_l2hh, pyramid, univmon_q] {
            assert!(
                UnivMon::deserialize_from_bytes(&foreign).is_err(),
                "a foreign envelope must not decode as a UnivMon"
            );
        }
    }

    /// Fail closed (not panic) on crafted geometry, layer counts and heap
    /// capacities, including a `layer_size` far larger than the payload
    /// carries. Every check precedes an allocation.
    #[test]
    fn univmon_rejects_crafted_shapes() {
        let encoded = populated().serialize_to_bytes().expect("serialize");
        let base = metadata_of(&encoded);
        let payload = payload_of::<String>(&encoded);

        let shaped = |layer_size, sketch_row, sketch_col, heap_size| UnivMonMetadata {
            layer_size,
            sketch_row,
            sketch_col,
            heap_size,
            ..univmon_metadata::<DefaultXxHasher>(0, 0, 0, 0, &base.key_type)
        };
        let cases = [
            shaped(u32::MAX, 2, 16, 4),
            shaped(3, 2, 0, 4),
            shaped(3, 0, 16, 4),
            shaped(0, 2, 16, 4),
            shaped(3, 2, 16, 0),
            shaped(3, MATRIX_MAX_ROWS as u32, 4096, 4),
            shaped(3, 2, 16, 1),
        ];
        for meta in cases {
            assert!(
                UnivMon::deserialize_from_bytes(&crafted(&meta, &payload)).is_err(),
                "a crafted shape must be rejected, not decoded"
            );
        }

        // A zero dimension against an empty payload: the accumulator check
        // alone passes (0 rows over any layer count names 0 accumulators), so
        // the dimensions must be refused before the geometry is sized.
        let mut empty = payload_of::<String>(&encoded);
        empty.counts.clear();
        empty.l2.clear();
        empty.keys.clear();
        empty.heap_counts.clear();
        for meta in [shaped(u32::MAX, 0, 16, 4), shaped(3, 0, 0, 4)] {
            let problem = UnivMon::deserialize_from_bytes(&crafted(&meta, &empty))
                .expect_err("a zero dimension must be rejected, not sized from")
                .to_string();
            assert!(problem.contains("must be non-zero"), "got {problem}");
        }

        // Parallel arrays and per-layer runs must agree with the declared
        // pyramid.
        let mut short = payload_of::<String>(&encoded);
        short.heap_counts.pop();
        assert!(UnivMon::deserialize_from_bytes(&crafted(&base, &short)).is_err());

        let mut flags = payload_of::<String>(&encoded);
        flags.candidate_complete.pop();
        assert!(UnivMon::deserialize_from_bytes(&crafted(&base, &flags)).is_err());

        let mut mode = payload_of::<String>(&encoded);
        mode.update_mode = 7;
        assert!(UnivMon::deserialize_from_bytes(&crafted(&base, &mode)).is_err());
    }

    /// A pyramid whose layers disagree with their declared geometry, seed
    /// index or key variants must not serialize.
    #[test]
    fn univmon_rejects_serializing_an_inconsistent_pyramid() {
        let mut wrong_layer = UnivMon::init_univmon(4, 2, 16, 3);
        wrong_layer.l2_sketch_layers[1] = L2HH::COUNT(
            crate::sketches::countsketch_topk::CountL2HH::with_dimensions_and_seed(2, 16, 1),
        );
        assert!(wrong_layer.serialize_to_bytes().is_ok());

        wrong_layer.l2_sketch_layers[1] = L2HH::COUNT(
            crate::sketches::countsketch_topk::CountL2HH::with_dimensions_and_seed(2, 32, 1),
        );
        assert!(
            wrong_layer.serialize_to_bytes().is_err(),
            "a layer that is not the declared size must not serialize"
        );

        let mut wrong_seed = UnivMon::init_univmon(4, 2, 16, 3);
        wrong_seed.l2_sketch_layers[1] = L2HH::COUNT(
            crate::sketches::countsketch_topk::CountL2HH::with_dimensions_and_seed(2, 16, 5),
        );
        assert!(
            wrong_seed.serialize_to_bytes().is_err(),
            "a layer hashing at another layer's seed index must not serialize"
        );

        let mut mixed = UnivMon::init_univmon(4, 2, 16, 3);
        mixed.hh_layers[0].update(&DataInput::U64(1), 5);
        mixed.hh_layers[0].update(&DataInput::Str("two"), 3);
        let problem = mixed
            .serialize_to_bytes()
            .expect_err("a pyramid mixing key variants must not serialize")
            .to_string();
        assert!(problem.contains("keys mix variants"), "got {problem}");

        let mut wide = UnivMon::init_univmon(4, 2, 16, 3);
        wide.hh_layers[0].update(&DataInput::U128(1), 5);
        assert!(
            wide.serialize_to_bytes().is_err(),
            "a 128-bit key is not a wire type"
        );
    }
    /// A layer's heap can seat one key twice — `update` compares residents with
    /// `HeapItem`'s equality and `NaN != NaN`, so a second `NaN` takes its own
    /// seat — and `rebuild_heap` refuses that payload. The encoder runs the
    /// same `check_distinct_keys` per layer, so the pyramid never emits it.
    #[test]
    fn univmon_refuses_to_serialize_a_layer_holding_a_key_twice() {
        let mut nans = UnivMon::init_univmon(4, 2, 16, 3);
        nans.hh_layers[1].update(&DataInput::F64(f64::NAN), 5);
        nans.hh_layers[1].update(&DataInput::F64(f64::NAN), 4);
        assert_eq!(nans.hh_layers[1].len(), 2, "the two NaNs did not both seat");
        let problem = nans
            .serialize_to_bytes()
            .expect_err("a layer holding one key twice must not serialize")
            .to_string();
        assert!(
            problem.contains("the same key appears twice"),
            "got {problem}"
        );
        assert!(problem.contains("layer 1"), "got {problem}");

        // Two layers each holding the same key is not a duplicate: the check is
        // per layer, as the heaps are.
        let mut shared = UnivMon::init_univmon(4, 2, 16, 3);
        shared.hh_layers[0].update(&DataInput::U64(7), 5);
        shared.hh_layers[1].update(&DataInput::U64(7), 5);
        let encoded = shared.serialize_to_bytes().expect("serialize");
        assert!(UnivMon::deserialize_from_bytes(&encoded).is_ok());
    }

    /// The dimensions are public fields, so a caller can zero one after
    /// construction (the constructor asserts them positive). The decoder
    /// refuses a zero dimension, so the encoder must too. Each case below is
    /// built by writing the public field directly.
    #[test]
    fn univmon_refuses_to_serialize_a_zero_dimension() {
        for zero in ["layer_size", "sketch_row", "sketch_col", "heap_size"] {
            let mut um = UnivMon::init_univmon(4, 2, 16, 3);
            match zero {
                "layer_size" => um.layer_size = 0,
                "sketch_row" => um.sketch_row = 0,
                "sketch_col" => um.sketch_col = 0,
                _ => um.heap_size = 0,
            }
            let problem = match um.serialize_to_bytes() {
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
    fn univmon_rejects_layers_past_the_shift_bound() {
        let deepest = UnivMon::init_univmon(4, 2, 16, MAX_LAYER_SIZE);
        let encoded = deepest.serialize_to_bytes().expect("serialize");
        assert!(UnivMon::deserialize_from_bytes(&encoded).is_ok());

        let layers = MAX_LAYER_SIZE + 1;
        let mut meta = metadata_of(&encoded);
        meta.layer_size = layers as u32;
        let (rows, cols) = (meta.sketch_row as usize, meta.sketch_col as usize);
        let mut payload: PyramidPayload<u64> = payload_of(&encoded);
        payload.counts = vec![0; rows * cols * layers];
        payload.l2 = vec![0; rows * layers];
        payload.heap_lens = vec![0; layers];
        payload.candidate_complete = vec![true; layers];
        let problem = UnivMon::deserialize_from_bytes(&crafted(&meta, &payload))
            .expect_err("a pyramid past the shift bound must be rejected")
            .to_string();
        assert!(problem.contains("MAX_LAYER_SIZE"), "got {problem}");
    }

    #[test]
    #[should_panic(expected = "at most MAX_LAYER_SIZE")]
    fn univmon_refuses_constructing_past_the_shift_bound() {
        UnivMon::init_univmon(4, 2, 16, MAX_LAYER_SIZE + 1);
    }

    /// Fail closed on an unexpected metadata key, and on a missing required
    /// one.
    #[test]
    fn univmon_metadata_rejects_unknown_and_missing_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            layer_size: u32,
            sketch_row: u32,
            sketch_col: u32,
            heap_size: u32,
            key_type: String,
            bogus_field: u8, // key not in UnivMonMetadata
        }
        #[derive(Serialize)]
        struct WithoutKeyType {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            layer_size: u32,
            sketch_row: u32,
            sketch_col: u32,
            heap_size: u32,
        }
        let m = univmon_metadata::<DefaultXxHasher>(3, 2, 16, 4, "u64");
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            layer_size: m.layer_size,
            sketch_row: m.sketch_row,
            sketch_col: m.sketch_col,
            heap_size: m.heap_size,
            key_type: m.key_type.clone(),
            bogus_field: 7,
        };
        let without = WithoutKeyType {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            layer_size: m.layer_size,
            sketch_row: m.sketch_row,
            sketch_col: m.sketch_col,
            heap_size: m.heap_size,
        };
        assert!(
            from_slice::<UnivMonMetadata>(&rmp_serde::to_vec_named(&extra).unwrap()).is_err(),
            "an unknown metadata key must be rejected"
        );
        assert!(
            from_slice::<UnivMonMetadata>(&rmp_serde::to_vec_named(&without).unwrap()).is_err(),
            "a missing required key must be rejected"
        );
    }
}
