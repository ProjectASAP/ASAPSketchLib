//! ASAPv1 wire serialization for the Count Sketch.
//!
//! Child submodule of [`crate::sketches::countsketch`]: it holds ALL of Count
//! Sketch's serialization (the metadata/payload DTOs, the kind_id constant, the
//! [`CsWireCounter`] / [`CsWireMode`] marker traits, and the
//! `serialize_to_bytes` / `deserialize_from_bytes` impls) while the algorithm
//! lives in the parent module file. Being a descendant module, it reads the
//! sketch's private `counts` field directly without widening any field
//! visibility. See `docs/asapv1_wire_format.md` §3.6.
//!
//! Count Sketch is one algorithm — a single kind_id `0x04 0x00`. The structural
//! parameters — the matrix dimensions (`rows` / `cols`), the **counter type**
//! (i32/i64) and the column-derivation **mode** (fast/regular) — live in the
//! metadata, so the payload itself is just `[counts]` (a 1-element array
//! mirroring Count-Min's).
//!
//! Wire counter types are `i32` and `i64` — the two [`CsWireCounter`] is
//! implemented for. Count-Min's `f64` has no counterpart here because
//! [`CountSketchCounter`] is implemented only for `i32`, `i64` and `i128`, so
//! `f64` is not a Count Sketch counter at all; `i128` is a counter but has no
//! msgpack integer form. `i32` is carried at its own width rather than widened,
//! because a nested Count Sketch — the `Vector2D<i32>` counters `HydraCounter`
//! and `EHSketchList` hold — must decode back into the type it was stored as.
//!
//! [`CountSketchCounter`]: crate::sketches::countsketch::CountSketchCounter

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::common::hash::check_matrix_rows;
use crate::message_pack_format::envelope;
use crate::{FastPath, HashProfile, RegularPath, SketchHasher, Vector2D};

use super::{Count, CountSketchCounter};

/// Count Sketch kind_id: family `0x04`, single algorithm variant `0x00`.
const CS_KIND: &[u8] = &[0x04, 0x00];

/// Names the wire counter type carried in the metadata (`counter_type`).
/// Implemented only for the two wire-eligible counter types.
pub trait CsWireCounter: Copy {
    /// Metadata `counter_type` string — `"i32"` or `"i64"`.
    const COUNTER_TYPE: &'static str;
}
impl CsWireCounter for i32 {
    const COUNTER_TYPE: &'static str = "i32";
}
impl CsWireCounter for i64 {
    const COUNTER_TYPE: &'static str = "i64";
}

/// Names the wire column-derivation mode carried in the metadata (`mode`).
pub trait CsWireMode {
    /// Metadata `mode` string — `"fast"` or `"regular"`.
    const MODE: &'static str;
}
impl CsWireMode for RegularPath {
    const MODE: &'static str = "regular";
}
impl CsWireMode for FastPath {
    const MODE: &'static str = "fast";
}

/// Count Sketch descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it). Hash-spec fields first, then the
/// structural params `rows` / `cols` / `counter_type` / `mode` — the same order
/// Count-Min uses. Per the spec's config→metadata rule, the matrix dimensions
/// are configuration (like HLL's `precision`) and so live here rather than in
/// the payload.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CsMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    matrix_seed_index: u32,
    rows: u32,
    cols: u32,
    counter_type: String,
    mode: String,
}

/// Builds the Count Sketch descriptor metadata from the hasher's
/// [`HashProfile`], so the wire bytes truthfully describe how the sketch was
/// hashed (rather than hardcoding the standard profile). `matrix_seed_index` is
/// the profile's own row seed index; `rows` / `cols` are the sketch's structural
/// dimensions.
fn cs_metadata<H: HashProfile>(rows: u32, cols: u32, counter_type: &str, mode: &str) -> CsMetadata {
    CsMetadata {
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
    }
}

/// Count Sketch payload (ASAPv1 §3.6), a msgpack **array** (`to_vec`,
/// positional): `[counts]` — a 1-element array. The dimensions live in the
/// metadata; `counts` is packed row-major and its element type is fixed by the
/// metadata `counter_type`. Cells are signed: Count Sketch adds `±weight`, so a
/// counter may be negative.
#[derive(Debug, Serialize, Deserialize)]
struct CsPayload<T> {
    counts: Vec<T>,
}

// Wire serialization for the canonical Count Sketch configs only. `wire` is a
// descendant of the sketch module, so this impl reads the private `counts`
// field directly.
impl<T, Mode, H> Count<Vector2D<T>, Mode, H>
where
    // `CountSketchCounter` is required by `Count::from_storage`; `AddAssign` by
    // `Vector2D<T>: MatrixStorage`. Neither is used by the bodies below.
    T: CsWireCounter
        + CountSketchCounter
        + std::ops::AddAssign
        + Serialize
        + for<'de> Deserialize<'de>,
    Mode: CsWireMode,
    H: SketchHasher + HashProfile,
{
    /// Serializes the sketch into an ASAPv1 MessagePack envelope. The metadata is
    /// derived from the hasher's [`HashProfile`], so it truthfully describes how
    /// the sketch was hashed.
    ///
    /// A matrix whose cell count disagrees with its own dimensions is an error
    /// rather than bytes that would be refused on decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let rows = self.counts.rows();
        let cols = self.counts.cols();
        let counts = self.counts.as_slice();
        // The decoder refuses a zero dimension, so the encoder must too: with
        // `rows == 0` the length check below passes trivially (`0 == 0*cols`).
        if rows == 0 || cols == 0 {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 Count Sketch envelope: dimensions must be non-zero: rows={rows}, cols={cols}"
            )));
        }
        check_matrix_rows("Count Sketch", rows)
            .map_err(|e| RmpEncodeError::Syntax(format!("ASAPv1 Count Sketch envelope: {e}")))?;
        if counts.len() != rows.saturating_mul(cols) {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 Count Sketch envelope: counts length {} != rows*cols {}",
                counts.len(),
                rows.saturating_mul(cols)
            )));
        }
        let cols_u32 = u32::try_from(cols).map_err(|_| {
            RmpEncodeError::Syntax(format!(
                "ASAPv1 Count Sketch envelope: cols {cols} exceeds u32"
            ))
        })?;
        let metadata = rmp_serde::to_vec_named(&cs_metadata::<H>(
            rows as u32,
            cols_u32,
            T::COUNTER_TYPE,
            Mode::MODE,
        ))?;
        let payload = rmp_serde::to_vec(&CsPayload::<T> {
            counts: counts.to_vec(),
        })?;
        Ok(envelope::encode(CS_KIND, &metadata, &payload))
    }

    /// Deserializes a sketch from an ASAPv1 MessagePack envelope. The matrix
    /// dimensions are read from the (validated) metadata; the payload carries
    /// only the counts.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != CS_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Count Sketch kind_id mismatch: stored {kind_id:?}, expected {CS_KIND:?}"
            )));
        }
        let meta: CsMetadata = from_slice(metadata)?;
        // Validate the hash spec + counter type + mode against this target;
        // `rows`/`cols` are structural (the sketch is dynamically sized), so they
        // are echoed back into the expected block rather than known a priori.
        if meta != cs_metadata::<H>(meta.rows, meta.cols, T::COUNTER_TYPE, Mode::MODE) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 Count Sketch envelope: metadata mismatch".to_string(),
            ));
        }
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        let p: CsPayload<T> = from_slice(payload)?;
        // Reject zero dimensions before building the matrix: `Vector2D::from_fn`
        // derives its mask via `cols.ilog2()`, which panics on `cols == 0`. Fail
        // closed with an error rather than panicking on crafted bytes.
        if rows == 0 || cols == 0 {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Count Sketch dimensions must be non-zero: rows={rows}, cols={cols}"
            )));
        }
        check_matrix_rows("Count Sketch", rows).map_err(RmpDecodeError::Uncategorized)?;
        // Length check precedes the allocation, so crafted dimensions cannot
        // drive `from_fn` into a huge reserve: the payload must actually carry
        // `rows*cols` decoded counters.
        if p.counts.len() != rows.saturating_mul(cols) {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Count Sketch counts length {} != rows*cols {}",
                p.counts.len(),
                rows.saturating_mul(cols)
            )));
        }
        let storage = Vector2D::from_fn(rows, cols, |r, c| p.counts[r * cols + c]);
        Ok(Count::from_storage(storage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, MATRIX_MAX_ROWS};

    #[test]
    fn count_sketch_round_trip_serialization() {
        let mut sketch = Count::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8);
        sketch.insert(&DataInput::U64(42));
        sketch.insert(&DataInput::U64(7));

        let encoded = sketch.serialize_to_bytes().expect("serialize Count");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x04, 0x00]); // kind_id_len=2, kind_id=[0x04,0x00]

        let decoded = Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&encoded)
            .expect("deserialize Count");

        assert_eq!(sketch.rows(), decoded.rows());
        assert_eq!(sketch.cols(), decoded.cols());
        assert_eq!(
            sketch.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );
    }

    /// Count Sketch cells are signed — the payload must carry negatives through
    /// unchanged (the one shape difference from Count-Min's monotonic counters).
    #[test]
    fn count_sketch_negative_counters_round_trip() {
        let sketch =
            Count::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::from_fn(2, 4, |r, c| {
                let v = (r * 4 + c) as i64;
                if v % 2 == 0 { v } else { -v }
            }));
        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded =
            Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(
            sketch.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );
        assert!(decoded.as_storage().as_slice().iter().any(|&v| v < 0));
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`. Count Sketch metadata is derived from
    // the profile, so an `AltHasher` sketch serializes truthfully. (An
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

    #[test]
    fn count_sketch_custom_hasher_profile_round_trips_and_is_self_describing() {
        // (a) A Count Sketch built with a custom-profile hasher round-trips.
        let mut alt = Count::<Vector2D<i64>, RegularPath, AltHasher>::with_dimensions(3, 8);
        let mut std = Count::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8);
        alt.insert(&DataInput::U64(42));
        alt.insert(&DataInput::U64(7));
        std.insert(&DataInput::U64(42));
        std.insert(&DataInput::U64(7));

        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded =
            Count::<Vector2D<i64>, RegularPath, AltHasher>::deserialize_from_bytes(&alt_bytes)
                .expect("alt decode");
        assert_eq!(alt.as_storage().as_slice(), decoded.as_storage().as_slice());

        // (b) Bytes differ from the standard-profile sketch (metadata derived
        // from the different profile).
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }

    #[test]
    fn count_sketch_mode_in_metadata_round_trips() {
        let mut sketch = Count::<Vector2D<i64>, FastPath>::with_dimensions(4, 16);
        sketch.insert_many(&DataInput::U64(1), 5);
        sketch.insert_many(&DataInput::U64(2), 3);

        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded = Count::<Vector2D<i64>, FastPath>::deserialize_from_bytes(&encoded)
            .expect("deserialize");
        assert_eq!(
            sketch.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );

        // Mode is pinned by the target: a fast payload must not decode into a
        // regular sketch (metadata mismatch).
        assert!(Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&encoded).is_err());
    }

    /// Count Sketch and Count-Min share the payload shape but not the kind_id;
    /// a CMS envelope must not decode as a Count Sketch.
    #[test]
    fn count_sketch_rejects_foreign_kind_id() {
        let cms = crate::CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8);
        let cms_bytes = cms.serialize_to_bytes().expect("serialize CMS");
        assert!(
            Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&cms_bytes).is_err(),
            "CMS bytes must not decode as a Count Sketch"
        );
    }

    /// Fail closed (not panic) on a crafted envelope with a zero dimension:
    /// valid envelope + valid metadata that carries `cols == 0`, with an empty
    /// `[counts]` payload. `Vector2D::from_fn` derives its mask via
    /// `cols.ilog2()`, which panics on `cols == 0`.
    #[test]
    fn count_sketch_rejects_zero_dimension_payload() {
        let metadata =
            rmp_serde::to_vec_named(&cs_metadata::<DefaultXxHasher>(4, 0, "i64", "regular"))
                .unwrap();
        let payload = rmp_serde::to_vec(&CsPayload::<i64> { counts: Vec::new() }).unwrap();
        let bytes = envelope::encode(CS_KIND, &metadata, &payload);
        assert!(
            Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).is_err(),
            "zero-dimension metadata must be rejected, not panic"
        );
    }

    /// Crafted dimensions must not drive a huge allocation: the length check
    /// runs before `Vector2D::from_fn`.
    #[test]
    fn count_sketch_rejects_dimension_length_mismatch() {
        let metadata = rmp_serde::to_vec_named(&cs_metadata::<DefaultXxHasher>(
            MATRIX_MAX_ROWS as u32,
            1 << 24,
            "i64",
            "regular",
        ))
        .unwrap();
        let payload = rmp_serde::to_vec(&CsPayload::<i64> {
            counts: vec![1, 2, 3],
        })
        .unwrap();
        let bytes = envelope::encode(CS_KIND, &metadata, &payload);
        assert!(
            Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).is_err(),
            "counts length must match rows*cols"
        );
    }

    /// More rows than the seed list has seeds is outside the wire-eligible
    /// subset, on both sides.
    #[test]
    fn count_sketch_rejects_too_many_rows() {
        let rows = MATRIX_MAX_ROWS + 1;
        assert!(
            Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, 8)
                .serialize_to_bytes()
                .is_err(),
            "a matrix past MATRIX_MAX_ROWS must not serialize"
        );

        let metadata = rmp_serde::to_vec_named(&cs_metadata::<DefaultXxHasher>(
            rows as u32,
            8,
            "i64",
            "regular",
        ))
        .unwrap();
        let payload = rmp_serde::to_vec(&CsPayload::<i64> {
            counts: vec![0; rows * 8],
        })
        .unwrap();
        let bytes = envelope::encode(CS_KIND, &metadata, &payload);
        let err = Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes)
            .expect_err("rows past MATRIX_MAX_ROWS must be rejected");
        assert!(err.to_string().contains("MATRIX_MAX_ROWS"), "got {err}");

        // The boundary itself is eligible.
        assert!(
            Count::<Vector2D<i64>, RegularPath>::with_dimensions(MATRIX_MAX_ROWS, 8)
                .serialize_to_bytes()
                .is_ok()
        );
    }

    /// An unpopulated matrix carries dimensions its cells do not match.
    /// Serializing it must fail rather than emit bytes the decoder refuses.
    #[test]
    fn count_sketch_rejects_serializing_an_unfilled_matrix() {
        let sketch = Count::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::<i64>::init(2, 4));
        assert!(
            sketch.serialize_to_bytes().is_err(),
            "a matrix whose cell count disagrees with its dimensions must not serialize"
        );
    }

    /// A `rows == 0` matrix satisfies the length check trivially (`0 == 0*cols`)
    /// but is refused on decode, so serializing it must fail rather than emit
    /// bytes the format would not read back.
    #[test]
    fn count_sketch_rejects_serializing_zero_rows() {
        let sketch =
            Count::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::from_fn(0, 4, |_, _| 0i64));
        assert_eq!(sketch.rows(), 0);
        let problem = sketch
            .serialize_to_bytes()
            .expect_err("a zero-row matrix must not serialize")
            .to_string();
        assert!(problem.contains("non-zero"), "got {problem}");
    }

    /// Fail closed on an unexpected metadata key (mirrors the CMS/HLL tests).
    #[test]
    fn cs_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            matrix_seed_index: u32,
            rows: u32,
            cols: u32,
            counter_type: String,
            mode: String,
            bogus_field: u8, // key not in CsMetadata
        }
        let m = cs_metadata::<DefaultXxHasher>(2, 3, "i64", "regular");
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            matrix_seed_index: m.matrix_seed_index,
            rows: m.rows,
            cols: m.cols,
            counter_type: m.counter_type.clone(),
            mode: m.mode.clone(),
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).unwrap();
        assert!(rmp_serde::from_slice::<CsMetadata>(&bytes).is_err());
    }

    /// `counter_type` is required: a Count Sketch metadata map missing it does
    /// not decode, so the key cannot be silently defaulted.
    #[test]
    fn cs_metadata_rejects_a_missing_counter_type_key() {
        #[derive(Serialize)]
        struct WithoutCounterType {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            matrix_seed_index: u32,
            rows: u32,
            cols: u32,
            mode: String,
        }
        let m = cs_metadata::<DefaultXxHasher>(2, 3, "i64", "regular");
        let without = WithoutCounterType {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            matrix_seed_index: m.matrix_seed_index,
            rows: m.rows,
            cols: m.cols,
            mode: m.mode.clone(),
        };
        let bytes = rmp_serde::to_vec_named(&without).unwrap();
        assert!(rmp_serde::from_slice::<CsMetadata>(&bytes).is_err());
    }

    /// `f64` is a Count-Min wire counter but not a Count Sketch one, so its
    /// name must not decode into either wire-eligible type.
    #[test]
    fn cs_metadata_rejects_a_foreign_counter_type_name() {
        let bytes =
            rmp_serde::to_vec_named(&cs_metadata::<DefaultXxHasher>(2, 4, "f64", "regular"))
                .unwrap();
        let payload = rmp_serde::to_vec(&CsPayload::<i64> { counts: vec![0; 8] }).unwrap();
        let envelope_bytes = envelope::encode(CS_KIND, &bytes, &payload);
        assert!(
            Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&envelope_bytes).is_err(),
            "an i64 sketch must reject an f64-labelled envelope"
        );
        assert!(
            Count::<Vector2D<i32>, RegularPath>::deserialize_from_bytes(&envelope_bytes).is_err(),
            "an i32 sketch must reject an f64-labelled envelope"
        );
    }

    /// The `i32` wire config round-trips, and the counter type is pinned by the
    /// target: i32 bytes must not decode into an i64 sketch or the reverse.
    #[test]
    fn count_sketch_i32_round_trips_and_is_pinned_by_counter_type() {
        let cells = |r: usize, c: usize| {
            let v = (r * 4 + c) as i32;
            if v % 2 == 0 { v } else { -v }
        };
        let narrow =
            Count::<Vector2D<i32>, RegularPath>::from_storage(Vector2D::from_fn(2, 4, cells));
        let wide =
            Count::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::from_fn(2, 4, |r, c| {
                cells(r, c) as i64
            }));

        let narrow_bytes = narrow.serialize_to_bytes().expect("serialize i32");
        let decoded = Count::<Vector2D<i32>, RegularPath>::deserialize_from_bytes(&narrow_bytes)
            .expect("decode i32");
        assert_eq!(
            narrow.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );

        // The two sketches hold numerically equal cells, so only the metadata
        // `counter_type` separates their bytes.
        let wide_bytes = wide.serialize_to_bytes().expect("serialize i64");
        assert_ne!(narrow_bytes, wide_bytes);
        assert!(
            Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&narrow_bytes).is_err(),
            "i32 bytes must not decode as an i64 sketch"
        );
        assert!(
            Count::<Vector2D<i32>, RegularPath>::deserialize_from_bytes(&wide_bytes).is_err(),
            "i64 bytes must not decode as an i32 sketch"
        );
    }
}
