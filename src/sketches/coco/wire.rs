//! ASAPv1 wire serialization for CocoSketch.
//!
//! Child submodule of [`crate::sketches::coco`]: it holds ALL of Coco's
//! serialization (the metadata/payload DTOs, the kind_id constant, and the
//! `serialize_to_bytes` / `deserialize_from_bytes` impls) while the algorithm
//! lives in the parent module file. Being a descendant module, it reads the
//! sketch's private `_hasher` marker and rebuilds the struct directly, without
//! widening any field visibility. See the Coco section of
//! `docs/asapv1_wire_format.md`.
//!
//! Coco is one algorithm — a single kind_id `0x0c 0x00`. Its structural
//! parameters are the table geometry: `rows` (the sketch's `d`, the arrays an
//! insert scans) and `cols` (the sketch's `w`, the buckets per array). Both are
//! construction config and live in the metadata, so the payload is the bucket
//! state alone: `[keys, values]`, two parallel row-major arrays of `rows*cols`
//! entries.
//!
//! ## An unoccupied bucket is msgpack `nil`
//!
//! A bucket's key is `Option<String>` and `insert` accepts any `&str`,
//! including `""`. So `Some("")` is an occupied bucket that `estimate_key("")`
//! answers for, and it must not share an encoding with an unoccupied one.
//! `keys[i]` is therefore msgpack `nil` (`0xc0`) when the bucket is free and a
//! msgpack `str` otherwise.
//!
//! ## Emitted order (byte-stable round trips)
//!
//! The arrays are dense and **row-major** over the whole table, index order,
//! the same layout Count-Min packs `counts` in. Every bucket is emitted, so a
//! decoded sketch re-serializes byte-identically.
//!
//! ## Keys are `String`
//!
//! A Coco key is a `String` at every entry point, so there is no `key_type`
//! param: the kind_id fixes the element type of `keys`.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};

use std::collections::HashSet;

use crate::common::hash::check_matrix_rows;
use crate::message_pack_format::envelope;
use crate::{DataInput, HashProfile, SketchHasher, Vector2D};

use super::{Coco, CocoBucket};

/// Coco kind_id: family `0x0c`, single algorithm variant `0x00`.
const COCO_KIND: &[u8] = &[0x0c, 0x00];

/// Coco descriptor metadata (ASAPv1 §2), a msgpack **map** (`to_vec_named`)
/// with keys in this declaration order — the canonical order the wire spec
/// fixes (Go must mirror it). Hash-spec fields first, then the structural
/// params `rows` / `cols`, which are the sketch's `d` and `w`.
///
/// There is **no seed-index key**: Coco hashes array `i` with
/// `hash64_seeded(i, ..)`, a fixed part of the algorithm rather than a profile
/// choice, so no `HashProfile` index field would describe it truthfully.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CocoMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    rows: u32,
    cols: u32,
}

/// Builds the Coco descriptor metadata from the hasher's [`HashProfile`], so
/// the wire bytes truthfully describe how the sketch was hashed (rather than
/// hardcoding the standard profile). `rows` is the sketch's `d`, `cols` its
/// `w`.
fn coco_metadata<H: HashProfile>(rows: u32, cols: u32) -> CocoMetadata {
    CocoMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        rows,
        cols,
    }
}

/// Coco payload (ASAPv1, kind_id `0x0c 0x00`), a msgpack **array** (`to_vec`, positional):
/// `[keys, values]`. Both are dense and row-major over the table, so their
/// length is `rows*cols`; `keys[i]` is `nil` for an unoccupied bucket and a
/// `str` otherwise, and `values[i]` is that bucket's attributed mass.
#[derive(Debug, Serialize, Deserialize)]
struct CocoPayload {
    keys: Vec<Option<String>>,
    values: Vec<u64>,
}

/// Rejects a geometry the format cannot carry. Zero in either dimension is out:
/// `Vector2D` derives its column mask via `cols.ilog2()`, and a table with no
/// buckets records nothing.
fn check_geometry(rows: usize, cols: usize) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "Coco table dimensions must be non-zero: rows={rows}, cols={cols}"
        ));
    }
    check_matrix_rows("Coco table", rows)?;
    Ok(())
}

/// Rejects a bucket layout no insert could produce. `insert` scans all `d`
/// mapped buckets before electing one, so a stored key sits in row `r` only at
/// column `hash64_seeded(r, key) % cols`, and holds at most one bucket in the
/// whole table. `keys` must already be `rows*cols` long and `cols` non-zero.
fn check_placement<H: SketchHasher>(keys: &[Option<String>], cols: usize) -> Result<(), String> {
    let mut seen: HashSet<&str> = HashSet::new();
    for (i, slot) in keys.iter().enumerate() {
        let Some(key) = slot.as_deref() else { continue };
        let (row, col) = (i / cols, i % cols);
        let mapped = H::hash64_seeded(row, &DataInput::Str(key)) as usize % cols;
        if mapped != col {
            return Err(format!(
                "Coco bucket ({row}, {col}) holds a key that maps to column {mapped}"
            ));
        }
        if !seen.insert(key) {
            return Err(format!(
                "Coco bucket ({row}, {col}) repeats a key already stored elsewhere in the table"
            ));
        }
    }
    Ok(())
}

// Wire serialization for Coco. `wire` is a descendant of the sketch module, so
// this impl reads the private hasher marker and rebuilds the struct directly.
impl<H: SketchHasher + HashProfile> Coco<H> {
    /// Serializes the sketch into an ASAPv1 MessagePack envelope. The metadata
    /// is derived from the hasher's [`HashProfile`], so it truthfully describes
    /// how the sketch was hashed.
    ///
    /// A table whose bucket count disagrees with the sketch's own geometry, a
    /// geometry with a zero dimension, and a bucket layout no insert could
    /// produce (a key outside the column its row hashes it to, or a key held by
    /// two buckets) are all errors rather than bytes that would be refused on
    /// decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let (rows, cols) = (self.d, self.w);
        check_geometry(rows, cols).map_err(RmpEncodeError::Syntax)?;
        let buckets = self.table.as_slice();
        if self.table.rows() != rows
            || self.table.cols() != cols
            || buckets.len() != rows.saturating_mul(cols)
        {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 Coco envelope: table {}x{} ({} buckets) != declared {rows}x{cols}",
                self.table.rows(),
                self.table.cols(),
                buckets.len()
            )));
        }
        let keys: Vec<Option<String>> = buckets.iter().map(|b| b.full_key.clone()).collect();
        // The same placement and uniqueness the decoder demands, so a
        // hand-built table fails here rather than on the way back in.
        check_placement::<H>(&keys, cols).map_err(RmpEncodeError::Syntax)?;
        let to_u32 = |v: usize, name: &str| {
            u32::try_from(v).map_err(|_| {
                RmpEncodeError::Syntax(format!("ASAPv1 Coco envelope: {name} {v} exceeds u32"))
            })
        };
        let metadata = rmp_serde::to_vec_named(&coco_metadata::<H>(
            to_u32(rows, "rows")?,
            to_u32(cols, "cols")?,
        ))?;
        let payload = rmp_serde::to_vec(&CocoPayload {
            keys,
            values: buckets.iter().map(|b| b.val).collect(),
        })?;
        Ok(envelope::encode(COCO_KIND, &metadata, &payload))
    }

    /// Deserializes a sketch from an ASAPv1 MessagePack envelope. The table
    /// geometry is read from the (validated) metadata; the payload carries only
    /// the bucket keys and their masses.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != COCO_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Coco kind_id mismatch: stored {kind_id:?}, expected {COCO_KIND:?}"
            )));
        }
        let meta: CocoMetadata = from_slice(metadata)?;
        // Validate the hash spec against this target; `rows`/`cols` are
        // structural (the table is dynamically sized), so they are echoed back
        // into the expected block rather than known a priori.
        if meta != coco_metadata::<H>(meta.rows, meta.cols) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 Coco envelope: metadata mismatch".to_string(),
            ));
        }
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        // The geometry is validated before anything is sized from it, so
        // crafted dimensions never reach an allocation.
        check_geometry(rows, cols).map_err(RmpDecodeError::Uncategorized)?;
        let mut p: CocoPayload = from_slice(payload)?;
        let expected = rows.saturating_mul(cols);
        if p.keys.len() != expected || p.values.len() != expected {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Coco payload lengths (keys {}, values {}) != rows*cols {expected}",
                p.keys.len(),
                p.values.len()
            )));
        }
        // An unoccupied bucket holds no mass: `insert` always elects a key into
        // the bucket it credits. Rejecting mass under a `nil` key keeps the
        // decoded table canonical.
        if let Some(i) = (0..expected).find(|&i| p.keys[i].is_none() && p.values[i] != 0) {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Coco bucket {i} is unoccupied but carries value {}",
                p.values[i]
            )));
        }
        // Placement and uniqueness, checked after the geometry so the scan is
        // bounded by the payload the sender actually shipped.
        check_placement::<H>(&p.keys, cols).map_err(RmpDecodeError::Uncategorized)?;
        let table = Vector2D::from_fn(rows, cols, |r, c| {
            let i = r * cols + c;
            CocoBucket {
                full_key: p.keys[i].take(),
                val: p.values[i],
            }
        });
        Ok(Coco {
            w: cols,
            d: rows,
            table,
            _hasher: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, MATRIX_MAX_ROWS, Vector2D};

    /// More rows than the seed list has seeds is outside the wire-eligible
    /// subset, on both sides: row `i` hashes at seed index `i`.
    #[test]
    fn coco_rejects_too_many_rows() {
        let rows = MATRIX_MAX_ROWS + 1;
        assert!(
            Coco::<DefaultXxHasher>::init_with_size(8, rows)
                .serialize_to_bytes()
                .is_err(),
            "a table past MATRIX_MAX_ROWS must not serialize"
        );

        let metadata =
            rmp_serde::to_vec_named(&coco_metadata::<DefaultXxHasher>(rows as u32, 8)).unwrap();
        let payload = rmp_serde::to_vec(&CocoPayload {
            keys: vec![None; rows * 8],
            values: vec![0; rows * 8],
        })
        .unwrap();
        let bytes = envelope::encode(COCO_KIND, &metadata, &payload);
        let problem = Coco::<DefaultXxHasher>::deserialize_from_bytes(&bytes)
            .expect_err("rows past MATRIX_MAX_ROWS must be rejected")
            .to_string();
        assert!(problem.contains("MATRIX_MAX_ROWS"), "got {problem}");

        // The boundary itself is eligible.
        assert!(
            Coco::<DefaultXxHasher>::init_with_size(8, MATRIX_MAX_ROWS)
                .serialize_to_bytes()
                .is_ok()
        );
    }

    /// The whole table as `(key, value)` pairs in row-major order.
    fn cells<H: SketchHasher>(sketch: &Coco<H>) -> Vec<(Option<String>, u64)> {
        sketch
            .table
            .as_slice()
            .iter()
            .map(|b| (b.full_key.clone(), b.val))
            .collect()
    }

    #[test]
    fn coco_round_trip_serialization() {
        let mut sketch: Coco = Coco::init_with_size(8, 4);
        sketch.insert("19.98.10.26|80", 521);
        sketch.insert("34.52.73.17|118", 856);

        let encoded = sketch.serialize_to_bytes().expect("serialize Coco");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x0c, 0x00]); // kind_id_len=2, kind_id=[0x0c,0x00]

        let decoded: Coco = Coco::deserialize_from_bytes(&encoded).expect("deserialize Coco");
        assert_eq!(decoded.w, 8);
        assert_eq!(decoded.d, 4);
        assert_eq!(cells(&sketch), cells(&decoded));
        assert_eq!(decoded.estimate_key("19.98.10.26|80"), 521);
    }

    /// Coco and Count-Min both carry `rows`/`cols` metadata; a CMS envelope
    /// must not decode as a Coco.
    #[test]
    fn coco_rejects_foreign_kind_id() {
        let cms = crate::CountMin::<Vector2D<i64>, crate::RegularPath>::with_dimensions(4, 8);
        let cms_bytes = cms.serialize_to_bytes().expect("serialize CMS");
        assert!(
            Coco::<DefaultXxHasher>::deserialize_from_bytes(&cms_bytes).is_err(),
            "CMS bytes must not decode as a Coco"
        );
    }

    /// Fail closed on an unexpected metadata key.
    #[test]
    fn coco_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            rows: u32,
            cols: u32,
            bogus_field: u8, // key not in CocoMetadata
        }
        let m = coco_metadata::<DefaultXxHasher>(4, 8);
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            rows: m.rows,
            cols: m.cols,
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).unwrap();
        assert!(rmp_serde::from_slice::<CocoMetadata>(&bytes).is_err());
    }

    /// `cols` is required: a Coco metadata map missing it does not decode, so
    /// the key cannot be silently defaulted.
    #[test]
    fn coco_metadata_rejects_a_missing_cols_key() {
        #[derive(Serialize)]
        struct WithoutCols {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            rows: u32,
        }
        let m = coco_metadata::<DefaultXxHasher>(4, 8);
        let without = WithoutCols {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            rows: m.rows,
        };
        let bytes = rmp_serde::to_vec_named(&without).unwrap();
        assert!(rmp_serde::from_slice::<CocoMetadata>(&bytes).is_err());
    }

    /// Fail closed (not panic) on a crafted envelope with a zero dimension:
    /// `Vector2D` derives its column mask via `cols.ilog2()`, which panics on
    /// `cols == 0`.
    #[test]
    fn coco_rejects_zero_dimension_payload() {
        let metadata = rmp_serde::to_vec_named(&coco_metadata::<DefaultXxHasher>(4, 0)).unwrap();
        let payload = rmp_serde::to_vec(&CocoPayload {
            keys: Vec::new(),
            values: Vec::new(),
        })
        .unwrap();
        let bytes = envelope::encode(COCO_KIND, &metadata, &payload);
        assert!(
            Coco::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
            "zero-dimension metadata must be rejected, not panic"
        );
    }

    /// Crafted dimensions must not drive a huge allocation: the length check
    /// runs before `Vector2D::from_fn`.
    #[test]
    fn coco_rejects_dimension_length_mismatch() {
        let metadata = rmp_serde::to_vec_named(&coco_metadata::<DefaultXxHasher>(
            MATRIX_MAX_ROWS as u32,
            1 << 24,
        ))
        .unwrap();
        let payload = rmp_serde::to_vec(&CocoPayload {
            keys: vec![None, None, None],
            values: vec![0, 0, 0],
        })
        .unwrap();
        let bytes = envelope::encode(COCO_KIND, &metadata, &payload);
        assert!(
            Coco::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
            "payload lengths must match rows*cols"
        );
    }

    /// An unoccupied bucket holds no mass, so a payload crediting one is
    /// rejected rather than decoded into a table no insert could produce.
    #[test]
    fn coco_rejects_mass_under_an_unoccupied_bucket() {
        let metadata = rmp_serde::to_vec_named(&coco_metadata::<DefaultXxHasher>(1, 2)).unwrap();
        let payload = rmp_serde::to_vec(&CocoPayload {
            keys: vec![None, None],
            values: vec![0, 9],
        })
        .unwrap();
        let bytes = envelope::encode(COCO_KIND, &metadata, &payload);
        assert!(
            Coco::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
            "a nil key with a non-zero value must be rejected"
        );
    }

    /// The column a `DefaultXxHasher` Coco maps `key` to in `row`.
    fn mapped_col(row: usize, key: &str, cols: usize) -> usize {
        DefaultXxHasher::hash64_seeded(row, &DataInput::Str(key)) as usize % cols
    }

    /// Encodes a crafted `rows`x`cols` table (row-major `(key, value)` cells).
    fn craft(rows: usize, cols: usize, cells: Vec<(Option<String>, u64)>) -> Vec<u8> {
        let metadata =
            rmp_serde::to_vec_named(&coco_metadata::<DefaultXxHasher>(rows as u32, cols as u32))
                .unwrap();
        let payload = rmp_serde::to_vec(&CocoPayload {
            keys: cells.iter().map(|(k, _)| k.clone()).collect(),
            values: cells.iter().map(|(_, v)| *v).collect(),
        })
        .unwrap();
        envelope::encode(COCO_KIND, &metadata, &payload)
    }

    /// A key only ever lands in the bucket its own row hashes it to, so a
    /// payload parking it anywhere else is rejected: `recorded_flows` would
    /// otherwise report a flow `estimate_key` answers 0 for.
    #[test]
    fn coco_rejects_a_key_in_a_bucket_it_does_not_hash_to() {
        let (rows, cols) = (1, 2);
        let key = "flow::misplaced";
        let wrong = 1 - mapped_col(0, key, cols);
        let mut cells = vec![(None, 0), (None, 0)];
        cells[wrong] = (Some(key.to_string()), 11);

        let problem = Coco::<DefaultXxHasher>::deserialize_from_bytes(&craft(rows, cols, cells))
            .expect_err("a misplaced key must be rejected")
            .to_string();
        assert!(problem.contains("maps to column"), "got {problem}");
    }

    /// An insert scans every mapped bucket before electing one, so a key holds
    /// at most one bucket table-wide. Two correctly placed copies are still
    /// rejected: they would double-count in `estimate_key`.
    #[test]
    fn coco_rejects_a_key_stored_in_two_buckets() {
        let (rows, cols) = (2, 2);
        let key = "flow::twice";
        let mut cells = vec![(None, 0); rows * cols];
        for row in 0..rows {
            cells[row * cols + mapped_col(row, key, cols)] = (Some(key.to_string()), u64::MAX / 2);
        }
        assert_eq!(
            cells.iter().filter(|(k, _)| k.is_some()).count(),
            2,
            "the two rows must place the key in distinct buckets"
        );

        let problem = Coco::<DefaultXxHasher>::deserialize_from_bytes(&craft(rows, cols, cells))
            .expect_err("a duplicated key must be rejected")
            .to_string();
        assert!(problem.contains("repeats a key"), "got {problem}");
    }

    /// A sketch whose declared geometry disagrees with its table must not emit
    /// bytes the decoder would refuse.
    #[test]
    fn coco_rejects_serializing_a_geometry_mismatch() {
        let mut sketch: Coco = Coco::init_with_size(8, 4);
        sketch.w = 16;
        assert!(
            sketch.serialize_to_bytes().is_err(),
            "a table whose bucket count disagrees with its geometry must not serialize"
        );

        let empty: Coco = Coco::init_with_size(8, 0);
        assert!(
            empty.serialize_to_bytes().is_err(),
            "a zero-dimension geometry must not serialize"
        );
    }

    /// `Coco`'s table is public, so a caller can park a key in a bucket its row
    /// does not hash it to, or hold one key in two buckets. Both must fail the
    /// encode rather than emit bytes the decoder refuses.
    #[test]
    fn coco_rejects_serializing_a_hand_built_table() {
        let key = "flow::misplaced";
        let mut misplaced: Coco = Coco::init_with_size(2, 1);
        let wrong = 1 - mapped_col(0, key, 2);
        misplaced.table.as_mut_slice()[wrong] = CocoBucket {
            full_key: Some(key.to_string()),
            val: 11,
        };
        let problem = misplaced
            .serialize_to_bytes()
            .expect_err("a misplaced key must not serialize")
            .to_string();
        assert!(problem.contains("maps to column"), "got {problem}");

        let (rows, cols) = (2, 2);
        let mut twice: Coco = Coco::init_with_size(cols, rows);
        for row in 0..rows {
            twice.table.as_mut_slice()[row * cols + mapped_col(row, key, cols)] = CocoBucket {
                full_key: Some(key.to_string()),
                val: 7,
            };
        }
        let problem = twice
            .serialize_to_bytes()
            .expect_err("a key in two buckets must not serialize")
            .to_string();
        assert!(problem.contains("repeats a key"), "got {problem}");
    }

    /// An unoccupied bucket is `nil`, not an empty string, so an inserted `""`
    /// key stays distinguishable from the buckets that hold nothing.
    #[test]
    fn coco_empty_buckets_are_nil_and_never_collide_with_an_empty_key() {
        let all_empty: Coco = Coco::init_with_size(4, 2);
        let encoded = all_empty.serialize_to_bytes().expect("serialize");
        let (_, _, payload) = envelope::split(&encoded).expect("split");
        assert_eq!(
            payload.iter().filter(|&&b| b == 0xc0).count(),
            8,
            "nil per bucket"
        );
        assert!(!payload.contains(&0xa0), "no empty-string key is emitted");
        let decoded: Coco = Coco::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(cells(&all_empty), cells(&decoded));
        assert!(decoded.recorded_flows().next().is_none());

        let mut with_empty_key: Coco = Coco::init_with_size(4, 2);
        with_empty_key.insert("", 5);
        let encoded = with_empty_key.serialize_to_bytes().expect("serialize");
        let (_, _, payload) = envelope::split(&encoded).expect("split");
        assert!(payload.contains(&0xa0), "the inserted empty key is a str");
        let decoded: Coco = Coco::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(cells(&with_empty_key), cells(&decoded));
        assert_eq!(decoded.recorded_flows().count(), 1);
        assert_eq!(decoded.estimate_key(""), 5);
    }

    /// A table mixing occupied and free buckets round-trips bucket for bucket.
    #[test]
    fn coco_mixed_occupancy_round_trips() {
        let mut sketch: Coco = Coco::init_with_size(16, 3);
        for i in 0..10u64 {
            sketch.insert(&format!("flow::{i}"), i + 1);
        }
        let occupied = sketch.recorded_flows().count();
        assert!(occupied > 0 && occupied < 16 * 3, "mix both bucket states");

        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded: Coco = Coco::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(cells(&sketch), cells(&decoded));
        for i in 0..10u64 {
            let key = format!("flow::{i}");
            assert_eq!(sketch.estimate_key(&key), decoded.estimate_key(&key));
        }
    }

    /// The emitted order is the table's own index order, so a decoded sketch
    /// re-serializes to the bytes it came from.
    #[test]
    fn coco_decoded_sketch_reserializes_byte_identically() {
        let mut sketch: Coco = Coco::init_with_size(16, 3);
        for i in 0..12u64 {
            sketch.insert(&format!("fam{}|item{i}", i % 4), i % 5 + 1);
        }
        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded: Coco = Coco::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(encoded, decoded.serialize_to_bytes().expect("re-serialize"));
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`. Coco metadata is derived from the
    // profile, so an `AltHasher` sketch serializes truthfully.
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
    fn coco_custom_hasher_profile_round_trips_and_is_self_describing() {
        // (a) A Coco built with a custom-profile hasher round-trips.
        let mut alt: Coco<AltHasher> = Coco::init_with_size(8, 4);
        let mut std: Coco = Coco::init_with_size(8, 4);
        alt.insert("flow::42", 7);
        alt.insert("flow::7", 3);
        std.insert("flow::42", 7);
        std.insert("flow::7", 3);

        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded = Coco::<AltHasher>::deserialize_from_bytes(&alt_bytes).expect("alt decode");
        assert_eq!(cells(&alt), cells(&decoded));

        // (b) Bytes differ from the standard-profile sketch (metadata derived
        // from the different profile).
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            Coco::<DefaultXxHasher>::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }
}
