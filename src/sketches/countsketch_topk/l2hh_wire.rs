//! ASAPv1 wire serialization for [`CountL2HH`], plus the sub-payload the
//! UnivMon family reuses.
//!
//! Child submodule of [`crate::sketches::countsketch_topk`]: it holds the
//! metadata/payload DTOs, the kind_id constant and the `serialize_to_bytes` /
//! `deserialize_from_bytes` impls, while the algorithm lives in the parent
//! module file. Being a descendant module, it reads the private `counts`,
//! `l2`, `row`, `col` and `seed_idx` fields directly without widening any
//! field visibility. See `docs/asapv1_wire_format.md`.
//!
//! CountL2HH is one algorithm — a single kind_id `0x19 0x00`. The dimensions
//! (`rows` / `cols`) and the seed-list index the sketch hashes with
//! (`seed_index`) are construction config and live in the metadata, so the
//! payload is `[counts, l2]`.
//!
//! ## The shared sub-payload
//!
//! [`CountL2HH`] is also the per-layer counter of `UnivMon`, `UnivMonPyramid`
//! and every other `L2HH::COUNT`. Those sketches inline the same
//! `(counts, l2)` state into their own positional arrays instead of nesting an
//! envelope per layer, so [`layer_state`] and [`rebuild_layer`] are the one
//! place that state is read and rebuilt.
//!
//! ## `l2` is carried, not recomputed
//!
//! A row's `l2` is its running sum of squares, clamped to `[0, i64::MAX]`. It
//! is not `sum(counts[row]^2)`: `fast_insert_with_count_without_l2_and_hash`
//! moves counters without it and saturation is one-way, so it is state.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::common::hash::check_matrix_rows;
use crate::common::structures::matrix_storage::cols_mask_bits;
use crate::message_pack_format::envelope;
use crate::{HashProfile, SketchHasher, Vector1D, Vector2D};

use super::CountL2HH;

/// CountL2HH kind_id: family `0x19`, single algorithm variant `0x00`.
const L2HH_KIND: &[u8] = &[0x19, 0x00];

/// CountL2HH descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it). Hash-spec fields first, then the
/// structural params `seed_index` / `rows` / `cols`.
///
/// There is no `counter_type` and no `mode`: the counters are `i64` and the
/// column derivation is fixed by the algorithm, so `kind_id` already
/// determines both.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct L2hhMetadata {
    pub(crate) metadata_version: u8,
    pub(crate) hash_profile_id: String,
    pub(crate) hash_algorithm: String,
    pub(crate) seed_derivation: String,
    pub(crate) input_encoding: String,
    pub(crate) seed_list: Vec<u64>,
    pub(crate) seed_index: u32,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
}

/// Builds the CountL2HH descriptor metadata from the hasher's [`HashProfile`],
/// so the wire bytes truthfully describe how the sketch was hashed.
/// `seed_index` is the sketch's own seed offset, a construction parameter
/// rather than a profile constant, so it is carried as a structural param.
pub(crate) fn l2hh_metadata<H: HashProfile>(seed_index: u32, rows: u32, cols: u32) -> L2hhMetadata {
    L2hhMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        seed_index,
        rows,
        cols,
    }
}

/// CountL2HH payload (ASAPv1 §3.x), a msgpack **array** (`to_vec`,
/// positional): `[counts, l2]`. `counts` is packed row-major over `rows*cols`
/// signed cells; `l2` is one accumulator per row.
#[derive(Debug, Serialize, Deserialize)]
struct L2hhPayload {
    counts: Vec<i64>,
    l2: Vec<i64>,
}

/// Rejects a zero dimension, a row count past `MATRIX_MAX_ROWS`, and a
/// geometry whose per-row hash fields do not fit in one 128-bit key hash.
///
/// `Vector2D` derives its column mask via `cols.ilog2()`, which panics on
/// `cols == 0`.
///
/// CountL2HH draws one 128-bit hash per key and slices it per row: row `r`
/// reads its column from bits `[r * mask_bits, (r + 1) * mask_bits)` and its
/// sign from bit `127 - r`, where `mask_bits = cols_mask_bits(cols)`. Each row
/// therefore claims `mask_bits + 1` of the 128 bits, so the fields are in
/// range and disjoint exactly while `rows * (mask_bits + 1) <= 128` — the
/// budget `<u128 as MatrixFastHash>::assert_compatible` asserts, restated here
/// as a `Result` so both wire doors fail closed instead of panicking.
pub(crate) fn check_dimensions(rows: usize, cols: usize) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "CountL2HH dimensions must be non-zero: rows={rows}, cols={cols}"
        ));
    }
    check_matrix_rows("CountL2HH", rows)?;
    let mask_bits = cols_mask_bits(cols) as usize;
    let bits_required = mask_bits.saturating_add(1).saturating_mul(rows);
    if bits_required > 128 {
        return Err(format!(
            "CountL2HH rows={rows}, cols={cols} needs {bits_required} hash bits \
             ({mask_bits} column bits + 1 sign bit per row), past the 128 the key hash carries"
        ));
    }
    Ok(())
}

/// Rejects a seed index the hasher's seed list does not hold. Seed selection
/// wraps (`seed_idx % seed_list.len()`), so an out-of-range index would give a
/// second encoding of a state the in-range index already encodes.
pub(crate) fn check_seed_index(seed_idx: usize, seed_list_len: usize) -> Result<(), String> {
    if seed_idx >= seed_list_len {
        return Err(format!(
            "CountL2HH seed_index {seed_idx} is past the seed list's {seed_list_len} entries"
        ));
    }
    Ok(())
}

/// The `(counts, l2)` sub-payload one CountL2HH contributes, in the order the
/// wire emits it. Rejects a matrix whose cell count disagrees with its own
/// dimensions, a mis-sized `l2`, and a negative accumulator — states the
/// decoder refuses.
pub(crate) fn layer_state<H: SketchHasher>(
    sketch: &CountL2HH<H>,
) -> Result<(&[i64], &[i64]), RmpEncodeError> {
    check_dimensions(sketch.row, sketch.col).map_err(RmpEncodeError::Syntax)?;
    let cells = sketch.row.saturating_mul(sketch.col);
    let counts = sketch.counts.as_slice();
    let l2 = sketch.l2.as_slice();
    if counts.len() != cells {
        return Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 CountL2HH state: counts length {} != rows*cols {cells}",
            counts.len()
        )));
    }
    if l2.len() != sketch.row {
        return Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 CountL2HH state: l2 length {} != rows {}",
            l2.len(),
            sketch.row
        )));
    }
    if let Some(negative) = l2.iter().find(|&&value| value < 0) {
        return Err(RmpEncodeError::Syntax(format!(
            "ASAPv1 CountL2HH state: l2 accumulator {negative} is negative"
        )));
    }
    Ok((counts, l2))
}

/// Rebuilds one CountL2HH from a validated sub-payload. The lengths are
/// checked against the declared geometry before the matrix is built, so a
/// crafted `rows`/`cols` never drives an allocation.
pub(crate) fn rebuild_layer<H: SketchHasher>(
    rows: usize,
    cols: usize,
    seed_idx: usize,
    counts: &[i64],
    l2: &[i64],
) -> Result<CountL2HH<H>, RmpDecodeError> {
    check_dimensions(rows, cols).map_err(RmpDecodeError::Uncategorized)?;
    let cells = rows.saturating_mul(cols);
    if counts.len() != cells {
        return Err(RmpDecodeError::Uncategorized(format!(
            "CountL2HH counts length {} != rows*cols {cells}",
            counts.len()
        )));
    }
    if l2.len() != rows {
        return Err(RmpDecodeError::Uncategorized(format!(
            "CountL2HH l2 length {} != rows {rows}",
            l2.len()
        )));
    }
    if let Some(negative) = l2.iter().find(|&&value| value < 0) {
        return Err(RmpDecodeError::Uncategorized(format!(
            "CountL2HH l2 accumulator {negative} is negative"
        )));
    }
    Ok(CountL2HH {
        counts: Vector2D::from_fn(rows, cols, |r, c| counts[r * cols + c]),
        l2: Vector1D::from_vec(l2.to_vec()),
        row: rows,
        col: cols,
        seed_idx,
        _hasher: PhantomData,
    })
}

// Wire serialization for CountL2HH. `l2hh_wire` is a descendant of the sketch
// module, so this impl reads the private fields directly.
impl<H: SketchHasher + HashProfile> CountL2HH<H> {
    /// Serializes the sketch into an ASAPv1 MessagePack envelope
    /// (kind_id `0x19 0x00`). The metadata is derived from the hasher's
    /// [`HashProfile`], so it truthfully describes how the sketch was hashed.
    ///
    /// Fails when the state disagrees with its own dimensions, when an `l2`
    /// accumulator is negative, when the geometry needs more hash bits than
    /// the 128-bit key hash carries, when the seed index is past the hasher's
    /// seed list, or when a dimension or the seed index overflows its `u32`
    /// metadata field.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        check_dimensions(self.row, self.col).map_err(RmpEncodeError::Syntax)?;
        check_seed_index(self.seed_idx, H::seed_list().len()).map_err(RmpEncodeError::Syntax)?;
        let (counts, l2) = layer_state(self)?;
        let field = |name: &str, value: usize| {
            u32::try_from(value).map_err(|_| {
                RmpEncodeError::Syntax(format!(
                    "ASAPv1 CountL2HH envelope: {name} {value} exceeds the u32 metadata field"
                ))
            })
        };
        let metadata = rmp_serde::to_vec_named(&l2hh_metadata::<H>(
            field("seed_index", self.seed_idx)?,
            field("rows", self.row)?,
            field("cols", self.col)?,
        ))?;
        let payload = rmp_serde::to_vec(&L2hhPayload {
            counts: counts.to_vec(),
            l2: l2.to_vec(),
        })?;
        Ok(envelope::encode(L2HH_KIND, &metadata, &payload))
    }

    /// Deserializes a sketch from an ASAPv1 MessagePack envelope. The
    /// dimensions and the seed index are structural (they are properties of
    /// the stored sketch, not of the target), so they are echoed back into the
    /// expected metadata; the hash spec is pinned against this target.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != L2HH_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "CountL2HH kind_id mismatch: stored {kind_id:?}, expected {L2HH_KIND:?}"
            )));
        }
        let meta: L2hhMetadata = from_slice(metadata)?;
        if meta != l2hh_metadata::<H>(meta.seed_index, meta.rows, meta.cols) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 CountL2HH envelope: metadata mismatch".to_string(),
            ));
        }
        check_seed_index(meta.seed_index as usize, meta.seed_list.len())
            .map_err(RmpDecodeError::Uncategorized)?;
        let p: L2hhPayload = from_slice(payload)?;
        rebuild_layer::<H>(
            meta.rows as usize,
            meta.cols as usize,
            meta.seed_index as usize,
            &p.counts,
            &p.l2,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, HeapItem, MATRIX_MAX_ROWS, RegularPath,
    };

    /// The standard-profile sketch, named so decode calls can pick a hasher.
    type Std = CountL2HH<DefaultXxHasher>;

    fn populated() -> CountL2HH {
        let mut sketch = CountL2HH::with_dimensions_and_seed(3, 32, 7);
        for (key, weight) in [("alpha", 9i64), ("beta", 5), ("gamma", -4)] {
            sketch.fast_insert_with_count(&DataInput::Str(key), weight);
        }
        sketch
    }

    fn metadata_of(bytes: &[u8]) -> L2hhMetadata {
        let (_, metadata, _) = envelope::split(bytes).expect("split");
        from_slice(metadata).expect("metadata")
    }

    fn crafted(meta: &L2hhMetadata, counts: Vec<i64>, l2: Vec<i64>) -> Vec<u8> {
        let metadata = rmp_serde::to_vec_named(meta).expect("metadata");
        let payload = rmp_serde::to_vec(&L2hhPayload { counts, l2 }).expect("payload");
        envelope::encode(L2HH_KIND, &metadata, &payload)
    }

    /// More rows than the seed list has seeds is outside the wire-eligible
    /// subset, on both sides.
    #[test]
    fn countl2hh_rejects_too_many_rows() {
        let rows = MATRIX_MAX_ROWS + 1;
        assert!(
            Std::with_dimensions(rows, 32).serialize_to_bytes().is_err(),
            "a matrix past MATRIX_MAX_ROWS must not serialize"
        );

        let bytes = crafted(
            &l2hh_metadata::<DefaultXxHasher>(0, rows as u32, 32),
            vec![0; rows * 32],
            vec![0; rows],
        );
        let problem = Std::deserialize_from_bytes(&bytes)
            .expect_err("rows past MATRIX_MAX_ROWS must be rejected")
            .to_string();
        assert!(problem.contains("MATRIX_MAX_ROWS"), "got {problem}");

        // The boundary itself is eligible.
        assert!(
            Std::with_dimensions(MATRIX_MAX_ROWS, 32)
                .serialize_to_bytes()
                .is_ok()
        );
    }

    #[test]
    fn count_l2hh_round_trip_serialization() {
        let sketch = populated();
        let encoded = sketch.serialize_to_bytes().expect("serialize CountL2HH");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x19, 0x00]); // kind_id_len=2, kind_id=[0x19,0x00]

        let meta = metadata_of(&encoded);
        assert_eq!(meta.metadata_version, 1);
        assert_eq!((meta.seed_index, meta.rows, meta.cols), (7, 3, 32));

        let decoded = Std::deserialize_from_bytes(&encoded).expect("deserialize CountL2HH");
        assert_eq!(decoded.rows(), 3);
        assert_eq!(decoded.cols(), 32);
        assert_eq!(decoded.seed_idx(), 7);
        assert_eq!(
            sketch.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );
        assert_eq!(sketch.l2.as_slice(), decoded.l2.as_slice());
        assert_eq!(sketch.get_l2(), decoded.get_l2());
    }

    /// A decoded sketch re-serializes byte-identically.
    #[test]
    fn count_l2hh_decoded_re_serializes_identically() {
        let sketch = populated();
        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded = Std::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);
    }

    /// Count Sketch cells are signed, so a CountL2HH carries negatives through
    /// unchanged.
    #[test]
    fn count_l2hh_negative_counters_round_trip() {
        let mut sketch: CountL2HH = CountL2HH::with_dimensions(2, 8);
        sketch.fast_insert_with_count(&DataInput::U64(11), -25);
        sketch.fast_insert_with_count(&DataInput::U64(12), -7);
        assert!(sketch.as_storage().as_slice().iter().any(|&v| v < 0));

        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded = Std::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(
            sketch.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );
        assert!(decoded.as_storage().as_slice().iter().any(|&v| v < 0));
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), encoded);
    }

    /// The seed index is state, not a profile constant: two sketches that
    /// differ only in it do not share bytes and do not decode into each other.
    #[test]
    fn count_l2hh_seed_index_travels_with_the_sketch() {
        let zero: CountL2HH = CountL2HH::with_dimensions_and_seed(2, 8, 0);
        let nine: CountL2HH = CountL2HH::with_dimensions_and_seed(2, 8, 9);
        let zero_bytes = zero.serialize_to_bytes().expect("serialize");
        let nine_bytes = nine.serialize_to_bytes().expect("serialize");
        assert_ne!(zero_bytes, nine_bytes);
        assert_eq!(
            Std::deserialize_from_bytes(&nine_bytes)
                .expect("decode")
                .seed_idx(),
            9
        );
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

    #[test]
    fn count_l2hh_custom_hasher_profile_round_trips_and_is_self_describing() {
        // (a) A CountL2HH built with a custom-profile hasher round-trips.
        let mut alt = CountL2HH::<AltHasher>::with_dimensions(3, 32);
        let mut std: CountL2HH = CountL2HH::with_dimensions(3, 32);
        alt.fast_insert_with_count(&DataInput::U64(42), 5);
        std.fast_insert_with_count(&DataInput::U64(42), 5);

        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded =
            CountL2HH::<AltHasher>::deserialize_from_bytes(&alt_bytes).expect("alt decode");
        assert_eq!(alt.as_storage().as_slice(), decoded.as_storage().as_slice());

        // (b) Bytes differ from the standard-profile sketch.
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            Std::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }

    /// Each family's envelope is rejected by the other three, and by a plain
    /// Count Sketch envelope.
    #[test]
    fn count_l2hh_rejects_foreign_kind_ids() {
        let count_sketch = crate::Count::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8)
            .serialize_to_bytes()
            .expect("serialize Count Sketch");
        let univmon = crate::UnivMon::init_univmon(4, 2, 8, 2)
            .serialize_to_bytes()
            .expect("serialize UnivMon");
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

        for foreign in [count_sketch, univmon, pyramid, univmon_q] {
            assert!(
                Std::deserialize_from_bytes(&foreign).is_err(),
                "a foreign envelope must not decode as a CountL2HH"
            );
        }
    }

    /// Fail closed (not panic) on a crafted envelope carrying a zero
    /// dimension, and on one whose declared geometry the payload does not
    /// carry. Both checks precede any allocation.
    #[test]
    fn count_l2hh_rejects_crafted_geometry() {
        let meta = l2hh_metadata::<DefaultXxHasher>(0, 4, 0);
        assert!(Std::deserialize_from_bytes(&crafted(&meta, Vec::new(), vec![0; 4])).is_err());

        let huge = l2hh_metadata::<DefaultXxHasher>(0, 1, 1 << 24);
        assert!(
            Std::deserialize_from_bytes(&crafted(&huge, vec![1, 2, 3], vec![0; 1])).is_err(),
            "counts length must match rows*cols"
        );

        let meta = l2hh_metadata::<DefaultXxHasher>(0, 2, 4);
        assert!(
            Std::deserialize_from_bytes(&crafted(&meta, vec![0; 8], vec![0; 5])).is_err(),
            "l2 length must match rows"
        );
        assert!(
            Std::deserialize_from_bytes(&crafted(&meta, vec![0; 8], vec![-1, 0])).is_err(),
            "a negative l2 accumulator is not a state the algorithm reaches"
        );
    }

    /// An unpopulated matrix carries dimensions its cells do not match.
    /// Serializing it must fail rather than emit bytes the decoder refuses.
    #[test]
    fn count_l2hh_rejects_serializing_an_unfilled_matrix() {
        let mut sketch: CountL2HH = CountL2HH::with_dimensions(2, 4);
        sketch.counts = Vector2D::init(2, 4);
        assert!(
            sketch.serialize_to_bytes().is_err(),
            "a matrix whose cell count disagrees with its dimensions must not serialize"
        );
    }

    /// Fail closed on an unexpected metadata key, and on a missing required
    /// one.
    #[test]
    fn l2hh_metadata_rejects_unknown_and_missing_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            seed_index: u32,
            rows: u32,
            cols: u32,
            bogus_field: u8, // key not in L2hhMetadata
        }
        #[derive(Serialize)]
        struct WithoutCols {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            seed_index: u32,
            rows: u32,
        }
        let m = l2hh_metadata::<DefaultXxHasher>(0, 2, 4);
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            seed_index: m.seed_index,
            rows: m.rows,
            cols: m.cols,
            bogus_field: 7,
        };
        let without = WithoutCols {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            seed_index: m.seed_index,
            rows: m.rows,
        };
        assert!(
            from_slice::<L2hhMetadata>(&rmp_serde::to_vec_named(&extra).unwrap()).is_err(),
            "an unknown metadata key must be rejected"
        );
        assert!(
            from_slice::<L2hhMetadata>(&rmp_serde::to_vec_named(&without).unwrap()).is_err(),
            "a missing required key must be rejected"
        );
    }

    /// One 128-bit key hash gives each row `mask_bits + 1` bits, so a geometry
    /// needing more than 128 of them is not a sketch either door may pass.
    /// `rows = 12, cols = 4096` and `rows = 13, cols = 2048` both need 156:
    /// their last row shifts the hash by 132, past the 127-bit ceiling.
    #[test]
    fn count_l2hh_rejects_a_geometry_past_the_hash_budget() {
        for (rows, cols) in [(12usize, 4096usize), (13, 2048)] {
            let problem = Std::with_dimensions(rows, cols)
                .serialize_to_bytes()
                .expect_err("a geometry past the hash budget must not serialize")
                .to_string();
            assert!(problem.contains("hash bits"), "got {problem}");

            let bytes = crafted(
                &l2hh_metadata::<DefaultXxHasher>(0, rows as u32, cols as u32),
                vec![0; rows * cols],
                vec![0; rows],
            );
            let problem = Std::deserialize_from_bytes(&bytes)
                .expect_err("a geometry past the hash budget must not decode")
                .to_string();
            assert!(problem.contains("hash bits"), "got {problem}");
        }
    }

    /// The same budget covers the case that does not panic: 10 rows of 4096
    /// columns fill bits 0..119 with column fields while the sign bits start at
    /// 118, so two rows would read a column out of bits that are already signs.
    #[test]
    fn count_l2hh_rejects_column_fields_overlapping_the_sign_bits() {
        assert!(
            Std::with_dimensions(10, 4096).serialize_to_bytes().is_err(),
            "column fields must not reach into the sign bits"
        );
        // The widest geometry each column count still admits.
        assert!(Std::with_dimensions(9, 4096).serialize_to_bytes().is_ok());
        assert!(Std::with_dimensions(10, 2048).serialize_to_bytes().is_ok());
        assert!(
            Std::with_dimensions(11, 2048).serialize_to_bytes().is_err(),
            "column fields must not reach into the sign bits"
        );
    }

    /// Seed selection wraps, so a seed index past the seed list is a second
    /// encoding of a state an in-range index already encodes — and two such
    /// sketches would refuse to merge. Neither door accepts it.
    #[test]
    fn count_l2hh_rejects_a_seed_index_past_the_seed_list() {
        let seeds = <DefaultXxHasher as HashProfile>::seed_list().len();
        let wrapped = seeds + 5;

        let problem = CountL2HH::<DefaultXxHasher>::with_dimensions_and_seed(2, 8, wrapped)
            .serialize_to_bytes()
            .expect_err("a wrapped seed index must not serialize")
            .to_string();
        assert!(problem.contains("seed_index"), "got {problem}");

        let bytes = crafted(
            &l2hh_metadata::<DefaultXxHasher>(wrapped as u32, 2, 8),
            vec![0; 16],
            vec![0; 2],
        );
        let problem = Std::deserialize_from_bytes(&bytes)
            .expect_err("a wrapped seed index must not decode")
            .to_string();
        assert!(problem.contains("seed_index"), "got {problem}");

        // The last in-range index is eligible.
        assert!(
            CountL2HH::<DefaultXxHasher>::with_dimensions_and_seed(2, 8, seeds - 1)
                .serialize_to_bytes()
                .is_ok()
        );
    }

    /// An empty sketch has exactly one encoding.
    #[test]
    fn count_l2hh_empty_has_one_encoding() {
        let left: CountL2HH = CountL2HH::with_dimensions(3, 32);
        let mut right: CountL2HH = CountL2HH::with_dimensions(3, 32);
        right.fast_insert_with_count(&DataInput::U64(5), 4);
        right.clear();
        assert_eq!(
            left.serialize_to_bytes().expect("serialize"),
            right.serialize_to_bytes().expect("serialize")
        );
    }
}
