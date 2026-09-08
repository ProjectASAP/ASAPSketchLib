//! ASAPv1 wire serialization for the Count-Min sketch.
//!
//! Child submodule of [`crate::sketches::countminsketch`]: it holds ALL of
//! Count-Min's serialization (the metadata/payload DTOs, the kind_id constant,
//! the [`CmsWireCounter`] / [`CmsWireMode`] marker traits, and the
//! `serialize_to_bytes` / `deserialize_from_bytes` impls) while the algorithm
//! lives in the parent module file. Being a descendant module, it reads the
//! sketch's private `counts` field directly without widening any field
//! visibility. See `docs/asapv1_wire_format.md` §3.2.
//!
//! Count-Min is one algorithm — a single kind_id `0x02 0x00`. The structural
//! parameters — the matrix dimensions (`rows` / `cols`), the **counter type**
//! (i32/i64/f64) and the column-derivation **mode** (fast/regular) — all live in
//! the metadata, so the payload itself is just `[counts]` (a 1-element array
//! mirroring HLL Classic's `[registers]`). Only the canonical wire configs
//! (i32/i64/f64 counters × fast/regular) get a serialization; exotic in-memory
//! counters (i128/…) must be converted to a wire type first. `i32` is carried at
//! its own width and is pinned on decode.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use crate::common::hash::check_matrix_rows;
use crate::message_pack_format::envelope;
use crate::{FastPath, HashProfile, RegularPath, SketchHasher, Vector2D};

use super::CountMin;

/// CMS kind_id: family `0x02`, single algorithm variant `0x00`.
const CMS_KIND: &[u8] = &[0x02, 0x00];

/// Names the wire counter type carried in the metadata (`counter_type`).
/// Implemented only for the wire-eligible counter types.
pub trait CmsWireCounter: Copy {
    /// Metadata `counter_type` string — `"i32"`, `"i64"` or `"f64"`.
    const COUNTER_TYPE: &'static str;
}
impl CmsWireCounter for i32 {
    const COUNTER_TYPE: &'static str = "i32";
}
impl CmsWireCounter for i64 {
    const COUNTER_TYPE: &'static str = "i64";
}
impl CmsWireCounter for f64 {
    const COUNTER_TYPE: &'static str = "f64";
}

/// Names the wire column-derivation mode carried in the metadata (`mode`).
pub trait CmsWireMode {
    /// Metadata `mode` string — `"fast"` or `"regular"`.
    const MODE: &'static str;
}
impl CmsWireMode for RegularPath {
    const MODE: &'static str = "regular";
}
impl CmsWireMode for FastPath {
    const MODE: &'static str = "fast";
}

/// CMS descriptor metadata (ASAPv1 §2), a msgpack **map** (`to_vec_named`) with
/// keys in this declaration order — the canonical order the wire spec fixes
/// (Go must mirror it). Hash-spec fields first, then the structural params
/// `rows` / `cols` / `counter_type` / `mode`. Per the spec's config→metadata
/// rule, the matrix dimensions are configuration (like HLL's `precision`) and so
/// live here rather than in the payload.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CmsMetadata {
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

/// Builds the CMS descriptor metadata from the hasher's [`HashProfile`], so the
/// wire bytes truthfully describe how the sketch was hashed (rather than
/// hardcoding the standard profile). `matrix_seed_index` is the profile's own
/// row seed index; `rows` / `cols` are the sketch's structural dimensions.
fn cms_metadata<H: HashProfile>(
    rows: u32,
    cols: u32,
    counter_type: &str,
    mode: &str,
) -> CmsMetadata {
    CmsMetadata {
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

/// CMS payload (ASAPv1 §3.2), a msgpack **array** (`to_vec`, positional):
/// `[counts]` — a 1-element array (mirroring HLL Classic's `[registers]`). The
/// dimensions live in the metadata; `counts` is packed row-major and its element
/// type is fixed by the metadata `counter_type`.
#[derive(Debug, Serialize, Deserialize)]
struct CmsPayload<T> {
    counts: Vec<T>,
}

// Wire serialization for the canonical Count-Min configs only. `wire` is a
// descendant of the sketch module, so this impl reads the private `counts`
// field directly.
impl<T, Mode, H> CountMin<Vector2D<T>, Mode, H>
where
    // `AddAssign` is required for `Vector2D<T>: MatrixStorage` (the struct's
    // bound), not by the bodies below.
    T: CmsWireCounter + std::ops::AddAssign + Serialize + for<'de> Deserialize<'de>,
    Mode: CmsWireMode,
    H: SketchHasher + HashProfile,
{
    /// Serializes the sketch into an ASAPv1 MessagePack envelope. The metadata is
    /// derived from the hasher's [`HashProfile`], so it truthfully describes how
    /// the sketch was hashed.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let rows = self.counts.rows();
        let cols = self.counts.cols();
        let counts = self.counts.as_slice();
        // The decoder refuses a zero dimension, so the encoder must too: with
        // `rows == 0` the length check below passes trivially (`0 == 0*cols`).
        if rows == 0 || cols == 0 {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 CMS envelope: dimensions must be non-zero: rows={rows}, cols={cols}"
            )));
        }
        check_matrix_rows("CMS", rows)
            .map_err(|e| RmpEncodeError::Syntax(format!("ASAPv1 CMS envelope: {e}")))?;
        if counts.len() != rows.saturating_mul(cols) {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 CMS envelope: counts length {} != rows*cols {}",
                counts.len(),
                rows.saturating_mul(cols)
            )));
        }
        let cols_u32 = u32::try_from(cols).map_err(|_| {
            RmpEncodeError::Syntax(format!("ASAPv1 CMS envelope: cols {cols} exceeds u32"))
        })?;
        let metadata = rmp_serde::to_vec_named(&cms_metadata::<H>(
            rows as u32,
            cols_u32,
            T::COUNTER_TYPE,
            Mode::MODE,
        ))?;
        let payload = rmp_serde::to_vec(&CmsPayload::<T> {
            counts: counts.to_vec(),
        })?;
        Ok(envelope::encode(CMS_KIND, &metadata, &payload))
    }

    /// Deserializes a sketch from an ASAPv1 MessagePack envelope. The matrix
    /// dimensions are read from the (validated) metadata; the payload carries
    /// only the counts.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != CMS_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "CMS kind_id mismatch: stored {kind_id:?}, expected {CMS_KIND:?}"
            )));
        }
        let meta: CmsMetadata = from_slice(metadata)?;
        // Validate the hash spec + counter type + mode against this target;
        // `rows`/`cols` are structural (the sketch is dynamically sized), so they
        // are echoed back into the expected block rather than known a priori.
        if meta != cms_metadata::<H>(meta.rows, meta.cols, T::COUNTER_TYPE, Mode::MODE) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 CMS envelope: metadata mismatch".to_string(),
            ));
        }
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        let p: CmsPayload<T> = from_slice(payload)?;
        // Reject zero dimensions before building the matrix: `Vector2D::from_fn`
        // derives its mask via `cols.ilog2()`, which panics on `cols == 0`. Fail
        // closed with an error rather than panicking on crafted bytes.
        if rows == 0 || cols == 0 {
            return Err(RmpDecodeError::Uncategorized(format!(
                "CMS dimensions must be non-zero: rows={rows}, cols={cols}"
            )));
        }
        check_matrix_rows("CMS", rows).map_err(RmpDecodeError::Uncategorized)?;
        if p.counts.len() != rows.saturating_mul(cols) {
            return Err(RmpDecodeError::Uncategorized(format!(
                "CMS counts length {} != rows*cols {}",
                p.counts.len(),
                rows.saturating_mul(cols)
            )));
        }
        let storage = Vector2D::from_fn(rows, cols, |r, c| p.counts[r * cols + c]);
        Ok(CountMin::from_storage(storage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, MATRIX_MAX_ROWS};

    #[test]
    fn count_min_round_trip_serialization() {
        // i64 counters are a wire-eligible config; the ASAPv1 envelope round-trips.
        let mut sketch = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8);
        sketch.insert(&DataInput::U64(42));
        sketch.insert(&DataInput::U64(7));

        let encoded = sketch.serialize_to_bytes().expect("serialize CountMin");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x02, 0x00]); // kind_id_len=2, kind_id=[0x02,0x00]

        let decoded = CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&encoded)
            .expect("deserialize CountMin");

        assert_eq!(sketch.rows(), decoded.rows());
        assert_eq!(sketch.cols(), decoded.cols());
        assert_eq!(
            sketch.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`. CMS metadata is derived from the
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
    fn count_min_custom_hasher_profile_round_trips_and_is_self_describing() {
        // (a) A CMS built with a custom-profile hasher round-trips.
        let mut alt = CountMin::<Vector2D<i64>, RegularPath, AltHasher>::with_dimensions(3, 8);
        let mut std = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(3, 8);
        alt.insert(&DataInput::U64(42));
        alt.insert(&DataInput::U64(7));
        std.insert(&DataInput::U64(42));
        std.insert(&DataInput::U64(7));

        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded =
            CountMin::<Vector2D<i64>, RegularPath, AltHasher>::deserialize_from_bytes(&alt_bytes)
                .expect("alt decode");
        assert_eq!(alt.as_storage().as_slice(), decoded.as_storage().as_slice());

        // (b) Bytes differ from the standard-profile sketch (metadata derived
        // from the different profile).
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }

    #[test]
    fn count_min_f64_and_mode_in_metadata_round_trip() {
        // f64 counters (fractional weights) are the other wire-eligible config.
        let mut sketch = CountMin::<Vector2D<f64>, FastPath>::with_dimensions(4, 16);
        sketch.insert_many(&DataInput::U64(1), 2.5);
        sketch.insert_many(&DataInput::U64(2), 1.25);

        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded = CountMin::<Vector2D<f64>, FastPath>::deserialize_from_bytes(&encoded)
            .expect("deserialize");
        assert_eq!(
            sketch.as_storage().as_slice(),
            decoded.as_storage().as_slice()
        );

        // Counter type + mode are pinned by the target: an f64/fast payload must
        // not decode into an i64/regular sketch (metadata mismatch).
        assert!(CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&encoded).is_err());
    }

    /// Fail closed (not panic) on a crafted envelope with a zero dimension:
    /// valid envelope + valid metadata that carries `cols == 0`, with an empty
    /// `[counts]` payload. Before the guard this panicked in `Vector2D::from_fn`
    /// via `0.ilog2()`.
    #[test]
    fn count_min_rejects_zero_dimension_payload() {
        let metadata =
            rmp_serde::to_vec_named(&cms_metadata::<DefaultXxHasher>(4, 0, "i64", "regular"))
                .unwrap();
        let payload = rmp_serde::to_vec(&CmsPayload::<i64> { counts: Vec::new() }).unwrap();
        let bytes = envelope::encode(CMS_KIND, &metadata, &payload);
        assert!(
            CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes).is_err(),
            "zero-dimension metadata must be rejected, not panic"
        );
    }

    /// More rows than the seed list has seeds is outside the wire-eligible
    /// subset, on both sides.
    #[test]
    fn count_min_rejects_too_many_rows() {
        let rows = MATRIX_MAX_ROWS + 1;
        assert!(
            CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(rows, 8)
                .serialize_to_bytes()
                .is_err(),
            "a matrix past MATRIX_MAX_ROWS must not serialize"
        );

        let metadata = rmp_serde::to_vec_named(&cms_metadata::<DefaultXxHasher>(
            rows as u32,
            8,
            "i64",
            "regular",
        ))
        .unwrap();
        let payload = rmp_serde::to_vec(&CmsPayload::<i64> {
            counts: vec![0; rows * 8],
        })
        .unwrap();
        let bytes = envelope::encode(CMS_KIND, &metadata, &payload);
        let err = CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes)
            .expect_err("rows past MATRIX_MAX_ROWS must be rejected");
        assert!(err.to_string().contains("MATRIX_MAX_ROWS"), "got {err}");

        // The boundary itself is eligible.
        assert!(
            CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(MATRIX_MAX_ROWS, 8)
                .serialize_to_bytes()
                .is_ok()
        );
    }

    /// An unpopulated matrix carries dimensions its cells do not match.
    /// Serializing it must fail rather than emit bytes the decoder refuses.
    #[test]
    fn count_min_rejects_serializing_an_unfilled_matrix() {
        let sketch =
            CountMin::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::<i64>::init(2, 4));
        assert!(
            sketch.serialize_to_bytes().is_err(),
            "a matrix whose cell count disagrees with its dimensions must not serialize"
        );
    }

    /// A `rows == 0` matrix satisfies the length check trivially (`0 == 0*cols`)
    /// but is refused on decode, so serializing it must fail rather than emit
    /// bytes the format would not read back.
    #[test]
    fn count_min_rejects_serializing_zero_rows() {
        let sketch = CountMin::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::from_fn(
            0,
            4,
            |_, _| 0i64,
        ));
        assert_eq!(sketch.rows(), 0);
        let problem = sketch
            .serialize_to_bytes()
            .expect_err("a zero-row matrix must not serialize")
            .to_string();
        assert!(problem.contains("non-zero"), "got {problem}");
    }

    /// Fail closed on an unexpected metadata key (mirrors the HLL test).
    #[test]
    fn cms_metadata_rejects_unknown_keys() {
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
            bogus_field: u8, // key not in CmsMetadata
        }
        let m = cms_metadata::<DefaultXxHasher>(2, 3, "i64", "regular");
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
        assert!(rmp_serde::from_slice::<CmsMetadata>(&bytes).is_err());
    }

    /// The `i32` wire config round-trips, and the counter type is pinned by the
    /// target: i32 bytes must not decode into an i64 sketch or the reverse.
    #[test]
    fn count_min_i32_round_trips_and_is_pinned_by_counter_type() {
        let cells = |r: usize, c: usize| (r * 4 + c) as i32;
        let narrow =
            CountMin::<Vector2D<i32>, RegularPath>::from_storage(Vector2D::from_fn(2, 4, cells));
        let wide = CountMin::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::from_fn(
            2,
            4,
            |r, c| cells(r, c) as i64,
        ));

        let narrow_bytes = narrow.serialize_to_bytes().expect("serialize i32");
        assert_eq!(&narrow_bytes[7..10], &[2u8, 0x02, 0x00]);
        let decoded = CountMin::<Vector2D<i32>, RegularPath>::deserialize_from_bytes(&narrow_bytes)
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
            CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&narrow_bytes).is_err(),
            "i32 bytes must not decode as an i64 sketch"
        );
        assert!(
            CountMin::<Vector2D<i32>, RegularPath>::deserialize_from_bytes(&wide_bytes).is_err(),
            "i64 bytes must not decode as an i32 sketch"
        );
    }

    /// Each of the three wire counter types rejects the others' bytes.
    #[test]
    fn count_min_counter_types_reject_each_other() {
        let float = CountMin::<Vector2D<f64>, RegularPath>::from_storage(Vector2D::from_fn(
            2,
            4,
            |r, c| (r * 4 + c) as f64,
        ));
        let float_bytes = float.serialize_to_bytes().expect("serialize f64");
        assert!(
            CountMin::<Vector2D<i32>, RegularPath>::deserialize_from_bytes(&float_bytes).is_err(),
            "an i32 sketch must reject f64 bytes"
        );
        assert!(
            CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&float_bytes).is_err(),
            "an i64 sketch must reject f64 bytes"
        );

        let narrow = CountMin::<Vector2D<i32>, RegularPath>::from_storage(Vector2D::from_fn(
            2,
            4,
            |r, c| (r * 4 + c) as i32,
        ));
        let narrow_bytes = narrow.serialize_to_bytes().expect("serialize i32");
        assert!(
            CountMin::<Vector2D<f64>, RegularPath>::deserialize_from_bytes(&narrow_bytes).is_err(),
            "an f64 sketch must reject i32 bytes"
        );
    }
}
