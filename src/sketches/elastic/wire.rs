//! ASAPv1 wire serialization for the Elastic sketch.
//!
//! Child submodule of [`crate::sketches::elastic`]: it holds ALL of Elastic's
//! serialization (the metadata/payload DTOs, the kind_id constant, and the
//! `serialize_to_bytes` / `deserialize_from_bytes` impls) while the algorithm
//! lives in the parent module file. Being a descendant module, it reads the
//! sketch's private `stale_copies` flag and `_hasher` marker and rebuilds the
//! struct directly, without widening any field visibility. See the Elastic
//! section of `docs/asapv1_wire_format.md`.
//!
//! Elastic is one algorithm — a single kind_id `0x0b 0x00`. Its structural
//! parameters are the heavy table's bucket count and the light layer's
//! Count-Min geometry, all construction config, so they live in the metadata
//! and the payload is state alone.
//!
//! ## The light layer is inlined
//!
//! `Elastic::light` is a `CountMin<Vector2D<i32>, RegularPath, H>`. Its
//! counters are inlined into this payload as one row-major array and its
//! structural params (`light_rows`, `light_cols`, `light_counter_type`,
//! `light_mode`) into this metadata, so an Elastic binary is one envelope.
//!
//! ## A free heavy bucket is msgpack `nil`
//!
//! A bucket holds a flow exactly while `vote_pos != 0`, and `insert("")` seats
//! the empty flow id in a bucket that `query("")` answers for. So `flow_ids[i]`
//! is msgpack `nil` (`0xc0`) when the bucket is free and a msgpack `str`
//! otherwise, and the two states never share an encoding.
//!
//! ## Emitted order (byte-stable round trips)
//!
//! The heavy arrays are dense in bucket index order and the light counters are
//! row-major, so a decoded sketch re-serializes byte-identically.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::common::hash::check_matrix_rows;
use crate::message_pack_format::envelope;
use crate::{CountMin, HashProfile, SketchHasher, Vector2D};

use super::{Elastic, HeavyBucket};

/// Elastic kind_id: family `0x0b`, single algorithm variant `0x00`.
const ELASTIC_KIND: &[u8] = &[0x0b, 0x00];

/// Metadata `light_counter_type`: the light layer is `Vector2D<i32>`, carried
/// at its own width.
const LIGHT_COUNTER_TYPE: &str = "i32";

/// Metadata `light_mode`: the light layer is a `RegularPath` Count-Min.
const LIGHT_MODE: &str = "regular";

/// Elastic descriptor metadata (ASAPv1 §2), a msgpack **map**
/// (`to_vec_named`) with keys in this declaration order — the canonical order
/// the wire spec fixes (Go must mirror it). Hash-spec fields first, then the
/// structural params: the heavy table's bucket count, then the inlined light
/// layer's Count-Min params.
///
/// The only seed-index key is `matrix_seed_index`, the light Count-Min's own.
/// The heavy part hashes at the fixed canonical index, a part of the algorithm
/// rather than a profile choice, so no key describes it.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ElasticMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    matrix_seed_index: u32,
    heavy_buckets: u32,
    light_rows: u32,
    light_cols: u32,
    light_counter_type: String,
    light_mode: String,
}

/// Builds the Elastic descriptor metadata from the hasher's [`HashProfile`], so
/// the wire bytes truthfully describe how the sketch was hashed (rather than
/// hardcoding the standard profile).
fn elastic_metadata<H: HashProfile>(
    heavy_buckets: u32,
    light_rows: u32,
    light_cols: u32,
) -> ElasticMetadata {
    ElasticMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        matrix_seed_index: H::MATRIX_SEED_INDEX,
        heavy_buckets,
        light_rows,
        light_cols,
        light_counter_type: LIGHT_COUNTER_TYPE.to_string(),
        light_mode: LIGHT_MODE.to_string(),
    }
}

/// Elastic payload (ASAPv1, kind_id `0x0b 0x00`), a msgpack **array**
/// (`to_vec`, positional): `[flow_ids, vote_pos, vote_neg, evictions,
/// stale_copies, light_counts]`. The first four are parallel and dense in
/// bucket index order, `heavy_buckets` long; `light_counts` is row-major over
/// `light_rows * light_cols`.
#[derive(Debug, Serialize, Deserialize)]
struct ElasticPayload {
    flow_ids: Vec<Option<String>>,
    vote_pos: Vec<i32>,
    vote_neg: Vec<i32>,
    evictions: Vec<bool>,
    stale_copies: bool,
    light_counts: Vec<i32>,
}

/// Rejects a geometry the format cannot carry and returns the light layer's
/// cell count. Zero is out in every dimension, `bktlen` is an `i32`, and the
/// light cell count must not overflow.
fn check_geometry(
    heavy_buckets: usize,
    light_rows: usize,
    light_cols: usize,
) -> Result<usize, String> {
    if heavy_buckets == 0 || light_rows == 0 || light_cols == 0 {
        return Err(format!(
            "Elastic dimensions must be non-zero: heavy_buckets={heavy_buckets}, light={light_rows}x{light_cols}"
        ));
    }
    check_matrix_rows("Elastic light layer", light_rows)?;
    if heavy_buckets > i32::MAX as usize {
        return Err(format!(
            "Elastic heavy bucket count {heavy_buckets} exceeds i32"
        ));
    }
    light_rows.checked_mul(light_cols).ok_or_else(|| {
        format!("Elastic light layer {light_rows}x{light_cols} overflows a cell count")
    })
}

// Wire serialization for Elastic. `wire` is a descendant of the sketch module,
// so this impl reads the private stale-copy flag and rebuilds the struct
// directly.
impl<H: SketchHasher + HashProfile> Elastic<H> {
    /// Serializes the sketch into an ASAPv1 MessagePack envelope. The metadata
    /// is derived from the hasher's [`HashProfile`], so it truthfully describes
    /// how the sketch was hashed.
    ///
    /// A `bktlen` disagreeing with the heavy table, a light layer whose cell
    /// count disagrees with its own dimensions, a zero dimension, and a free
    /// bucket still naming a flow are errors rather than bytes that would be
    /// refused on decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let heavy_buckets = self.heavy.len();
        let (light_rows, light_cols) = (self.light.rows(), self.light.cols());
        let light_cells = check_geometry(heavy_buckets, light_rows, light_cols)
            .map_err(RmpEncodeError::Syntax)?;
        if self.bktlen <= 0 || self.bktlen as usize != heavy_buckets {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 Elastic envelope: bktlen {} != heavy table length {heavy_buckets}",
                self.bktlen
            )));
        }
        let counts = self.light.as_storage().as_slice();
        if counts.len() != light_cells {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 Elastic envelope: light layer holds {} cells != {light_rows}x{light_cols}",
                counts.len()
            )));
        }
        // A free bucket encodes as `nil` and carries no flow id, so one that
        // names a flow has no encoding.
        if let Some(i) = self
            .heavy
            .iter()
            .position(|b| b.is_vacant() && !b.flow_id.is_empty())
        {
            return Err(RmpEncodeError::Syntax(format!(
                "ASAPv1 Elastic envelope: heavy bucket {i} is free but names a flow"
            )));
        }
        let to_u32 = |v: usize, name: &str| {
            u32::try_from(v).map_err(|_| {
                RmpEncodeError::Syntax(format!("ASAPv1 Elastic envelope: {name} {v} exceeds u32"))
            })
        };
        let metadata = rmp_serde::to_vec_named(&elastic_metadata::<H>(
            to_u32(heavy_buckets, "heavy_buckets")?,
            to_u32(light_rows, "light_rows")?,
            to_u32(light_cols, "light_cols")?,
        ))?;
        let payload = rmp_serde::to_vec(&ElasticPayload {
            flow_ids: self
                .heavy
                .iter()
                .map(|b| (!b.is_vacant()).then(|| b.flow_id.clone()))
                .collect(),
            vote_pos: self.heavy.iter().map(|b| b.vote_pos).collect(),
            vote_neg: self.heavy.iter().map(|b| b.vote_neg).collect(),
            evictions: self.heavy.iter().map(|b| b.eviction).collect(),
            stale_copies: self.stale_copies,
            light_counts: counts.to_vec(),
        })?;
        Ok(envelope::encode(ELASTIC_KIND, &metadata, &payload))
    }

    /// Deserializes a sketch from an ASAPv1 MessagePack envelope. Both
    /// geometries are read from the (validated) metadata; the payload carries
    /// the heavy buckets, the stale-copy flag, and the light counters.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != ELASTIC_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Elastic kind_id mismatch: stored {kind_id:?}, expected {ELASTIC_KIND:?}"
            )));
        }
        let meta: ElasticMetadata = from_slice(metadata)?;
        // Validate the hash spec, counter type and mode against this target;
        // the two geometries are structural (both parts are dynamically
        // sized), so they are echoed back into the expected block.
        if meta != elastic_metadata::<H>(meta.heavy_buckets, meta.light_rows, meta.light_cols) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 Elastic envelope: metadata mismatch".to_string(),
            ));
        }
        let heavy_buckets = meta.heavy_buckets as usize;
        let (light_rows, light_cols) = (meta.light_rows as usize, meta.light_cols as usize);
        // The geometry is validated before anything is sized from it, so
        // crafted dimensions never reach an allocation.
        let light_cells = check_geometry(heavy_buckets, light_rows, light_cols)
            .map_err(RmpDecodeError::Uncategorized)?;
        let mut p: ElasticPayload = from_slice(payload)?;
        if p.flow_ids.len() != heavy_buckets
            || p.vote_pos.len() != heavy_buckets
            || p.vote_neg.len() != heavy_buckets
            || p.evictions.len() != heavy_buckets
        {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Elastic heavy payload lengths (flow_ids {}, vote_pos {}, vote_neg {}, evictions {}) != heavy_buckets {heavy_buckets}",
                p.flow_ids.len(),
                p.vote_pos.len(),
                p.vote_neg.len(),
                p.evictions.len()
            )));
        }
        if p.light_counts.len() != light_cells {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Elastic light_counts length {} != light_rows*light_cols {light_cells}",
                p.light_counts.len()
            )));
        }
        // A bucket holds a flow exactly while `vote_pos != 0`, so `nil` and a
        // zero vote are the same state and must agree.
        if let Some(i) =
            (0..heavy_buckets).find(|&i| p.flow_ids[i].is_none() != (p.vote_pos[i] == 0))
        {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Elastic heavy bucket {i} has a flow id and a vote_pos that disagree on occupancy"
            )));
        }
        let heavy: Vec<HeavyBucket> = (0..heavy_buckets)
            .map(|i| HeavyBucket {
                flow_id: p.flow_ids[i].take().unwrap_or_default(),
                vote_pos: p.vote_pos[i],
                vote_neg: p.vote_neg[i],
                eviction: p.evictions[i],
            })
            .collect();
        let light = CountMin::from_storage(Vector2D::from_fn(light_rows, light_cols, |r, c| {
            p.light_counts[r * light_cols + c]
        }));
        Ok(Elastic {
            heavy,
            light,
            bktlen: heavy_buckets as i32,
            stale_copies: p.stale_copies,
            _hasher: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, MATRIX_MAX_ROWS, RegularPath};

    /// The light layer is a Count-Min matrix, so it carries the same row bound:
    /// more rows than the seed list has seeds is refused on both sides.
    #[test]
    fn elastic_rejects_too_many_light_rows() {
        let rows = MATRIX_MAX_ROWS + 1;
        assert!(
            Elastic::<DefaultXxHasher>::init_with_dimensions(4, rows, 256)
                .serialize_to_bytes()
                .is_err(),
            "a light layer past MATRIX_MAX_ROWS must not serialize"
        );

        let metadata =
            rmp_serde::to_vec_named(&elastic_metadata::<DefaultXxHasher>(4, rows as u32, 256))
                .unwrap();
        let payload = rmp_serde::to_vec(&ElasticPayload {
            flow_ids: vec![None; 4],
            vote_pos: vec![0; 4],
            vote_neg: vec![0; 4],
            evictions: vec![false; 4],
            stale_copies: false,
            light_counts: vec![0; rows * 256],
        })
        .unwrap();
        let bytes = envelope::encode(ELASTIC_KIND, &metadata, &payload);
        let problem = Elastic::<DefaultXxHasher>::deserialize_from_bytes(&bytes)
            .expect_err("light rows past MATRIX_MAX_ROWS must be rejected")
            .to_string();
        assert!(problem.contains("MATRIX_MAX_ROWS"), "got {problem}");

        // The boundary itself is eligible.
        assert!(
            Elastic::<DefaultXxHasher>::init_with_dimensions(4, MATRIX_MAX_ROWS, 256)
                .serialize_to_bytes()
                .is_ok()
        );
    }

    /// The whole heavy table as `(flow_id, vote_pos, vote_neg, eviction)` in
    /// bucket index order.
    fn buckets<H: SketchHasher>(sketch: &Elastic<H>) -> Vec<(String, i32, i32, bool)> {
        sketch
            .heavy
            .iter()
            .map(|b| (b.flow_id.clone(), b.vote_pos, b.vote_neg, b.eviction))
            .collect()
    }

    /// The light layer's counters in row-major order.
    fn light_cells<H: SketchHasher>(sketch: &Elastic<H>) -> Vec<i32> {
        sketch.light.as_storage().as_slice().to_vec()
    }

    /// A key that lands in the same heavy bucket as `primary`.
    fn colliding_key(primary: &str, sketch: &Elastic) -> String {
        let target = sketch.bucket_index(primary);
        (0..10_000)
            .map(|i| format!("flow::secondary::{i}"))
            .find(|c| sketch.bucket_index(c) == target && c != primary)
            .expect("unable to find colliding key for test")
    }

    #[test]
    fn elastic_round_trip_serialization() {
        let mut sketch: Elastic = Elastic::init_with_dimensions(8, 2, 256);
        for _ in 0..12 {
            sketch.insert("19.98.10.26|80".to_string());
        }
        sketch.insert("34.52.73.17|118".to_string());

        let encoded = sketch.serialize_to_bytes().expect("serialize Elastic");
        assert!(encoded.starts_with(b"ASAPv1"));
        assert_eq!(&encoded[7..10], &[2u8, 0x0b, 0x00]); // kind_id_len=2, kind_id=[0x0b,0x00]

        let decoded: Elastic =
            Elastic::deserialize_from_bytes(&encoded).expect("deserialize Elastic");
        assert_eq!(decoded.bktlen, 8);
        assert_eq!(decoded.light.rows(), 2);
        assert_eq!(decoded.light.cols(), 256);
        assert_eq!(buckets(&sketch), buckets(&decoded));
        assert_eq!(light_cells(&sketch), light_cells(&decoded));
        assert_eq!(decoded.query("19.98.10.26|80".to_string()), 12);
    }

    /// Elastic inlines a Count-Min; a stand-alone CMS envelope must not decode
    /// as an Elastic.
    #[test]
    fn elastic_rejects_foreign_kind_id() {
        let cms = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(2, 256);
        let cms_bytes = cms.serialize_to_bytes().expect("serialize CMS");
        assert!(
            Elastic::<DefaultXxHasher>::deserialize_from_bytes(&cms_bytes).is_err(),
            "CMS bytes must not decode as an Elastic"
        );
    }

    /// Fail closed on an unexpected metadata key.
    #[test]
    fn elastic_metadata_rejects_unknown_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            matrix_seed_index: u32,
            heavy_buckets: u32,
            light_rows: u32,
            light_cols: u32,
            light_counter_type: String,
            light_mode: String,
            bogus_field: u8, // key not in ElasticMetadata
        }
        let m = elastic_metadata::<DefaultXxHasher>(8, 2, 256);
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            matrix_seed_index: m.matrix_seed_index,
            heavy_buckets: m.heavy_buckets,
            light_rows: m.light_rows,
            light_cols: m.light_cols,
            light_counter_type: m.light_counter_type.clone(),
            light_mode: m.light_mode.clone(),
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).unwrap();
        assert!(rmp_serde::from_slice::<ElasticMetadata>(&bytes).is_err());
    }

    /// `light_cols` is required: an Elastic metadata map missing it does not
    /// decode, so the key cannot be silently defaulted.
    #[test]
    fn elastic_metadata_rejects_a_missing_light_cols_key() {
        #[derive(Serialize)]
        struct WithoutLightCols {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            matrix_seed_index: u32,
            heavy_buckets: u32,
            light_rows: u32,
            light_counter_type: String,
            light_mode: String,
        }
        let m = elastic_metadata::<DefaultXxHasher>(8, 2, 256);
        let without = WithoutLightCols {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            matrix_seed_index: m.matrix_seed_index,
            heavy_buckets: m.heavy_buckets,
            light_rows: m.light_rows,
            light_counter_type: m.light_counter_type.clone(),
            light_mode: m.light_mode.clone(),
        };
        let bytes = rmp_serde::to_vec_named(&without).unwrap();
        assert!(rmp_serde::from_slice::<ElasticMetadata>(&bytes).is_err());
    }

    /// Fail closed (not panic) on a crafted envelope with a zero dimension in
    /// either part: `Vector2D` derives its column mask via `cols.ilog2()`, and
    /// a heavy table of no buckets records nothing.
    #[test]
    fn elastic_rejects_zero_dimension_payload() {
        let empty = ElasticPayload {
            flow_ids: Vec::new(),
            vote_pos: Vec::new(),
            vote_neg: Vec::new(),
            evictions: Vec::new(),
            stale_copies: false,
            light_counts: Vec::new(),
        };
        let payload = rmp_serde::to_vec(&empty).unwrap();
        for (heavy, rows, cols) in [(0u32, 2u32, 256u32), (8, 0, 256), (8, 2, 0)] {
            let metadata =
                rmp_serde::to_vec_named(&elastic_metadata::<DefaultXxHasher>(heavy, rows, cols))
                    .unwrap();
            let bytes = envelope::encode(ELASTIC_KIND, &metadata, &payload);
            assert!(
                Elastic::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
                "zero-dimension metadata must be rejected, not panic"
            );
        }
    }

    /// Crafted dimensions must not drive a huge allocation: both length checks
    /// run before the heavy table and the light matrix are built.
    #[test]
    fn elastic_rejects_dimension_length_mismatch() {
        let metadata = rmp_serde::to_vec_named(&elastic_metadata::<DefaultXxHasher>(
            1024,
            MATRIX_MAX_ROWS as u32,
            1 << 24,
        ))
        .unwrap();
        let payload = rmp_serde::to_vec(&ElasticPayload {
            flow_ids: vec![None, None],
            vote_pos: vec![0, 0],
            vote_neg: vec![0, 0],
            evictions: vec![false, false],
            stale_copies: false,
            light_counts: vec![0, 0, 0],
        })
        .unwrap();
        let bytes = envelope::encode(ELASTIC_KIND, &metadata, &payload);
        assert!(
            Elastic::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
            "payload lengths must match the declared geometry"
        );

        // The heavy arrays agree with each other but not with the light layer.
        let metadata =
            rmp_serde::to_vec_named(&elastic_metadata::<DefaultXxHasher>(2, 2, 256)).unwrap();
        let payload = rmp_serde::to_vec(&ElasticPayload {
            flow_ids: vec![None, None],
            vote_pos: vec![0, 0],
            vote_neg: vec![0, 0],
            evictions: vec![false, false],
            stale_copies: false,
            light_counts: vec![0; 8],
        })
        .unwrap();
        let bytes = envelope::encode(ELASTIC_KIND, &metadata, &payload);
        assert!(
            Elastic::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
            "light_counts must match light_rows*light_cols"
        );
    }

    /// A free bucket is `nil` with no vote; a payload where the two disagree
    /// decodes into a table no insert could produce, so it is rejected.
    #[test]
    fn elastic_rejects_a_flow_and_vote_that_disagree_on_occupancy() {
        let metadata =
            rmp_serde::to_vec_named(&elastic_metadata::<DefaultXxHasher>(2, 1, 2)).unwrap();
        for (flow_ids, vote_pos) in [
            (vec![Some("a".to_string()), None], vec![0, 0]),
            (vec![None, None], vec![0, 5]),
        ] {
            let payload = rmp_serde::to_vec(&ElasticPayload {
                flow_ids,
                vote_pos,
                vote_neg: vec![0, 0],
                evictions: vec![false, false],
                stale_copies: false,
                light_counts: vec![0, 0],
            })
            .unwrap();
            let bytes = envelope::encode(ELASTIC_KIND, &metadata, &payload);
            assert!(
                Elastic::<DefaultXxHasher>::deserialize_from_bytes(&bytes).is_err(),
                "occupancy must agree between flow_ids and vote_pos"
            );
        }
    }

    /// A sketch the decoder would refuse must not serialize.
    #[test]
    fn elastic_rejects_serializing_an_inconsistent_sketch() {
        let mut desynced: Elastic = Elastic::init_with_dimensions(8, 2, 256);
        desynced.bktlen = 16;
        assert!(
            desynced.serialize_to_bytes().is_err(),
            "a bktlen disagreeing with the heavy table must not serialize"
        );

        let mut named_free: Elastic = Elastic::init_with_dimensions(8, 2, 256);
        named_free.heavy[3].flow_id = "ghost".to_string();
        assert!(
            named_free.serialize_to_bytes().is_err(),
            "a free bucket naming a flow must not serialize"
        );
    }

    /// A free bucket is `nil`, not an empty string, so an inserted `""` flow
    /// stays distinguishable from the buckets that hold nothing.
    #[test]
    fn elastic_free_buckets_are_nil_and_never_collide_with_an_empty_flow_id() {
        let all_free: Elastic = Elastic::init_with_dimensions(4, 1, 2);
        let encoded = all_free.serialize_to_bytes().expect("serialize");
        let (_, _, payload) = envelope::split(&encoded).expect("split");
        assert_eq!(
            payload.iter().filter(|&&b| b == 0xc0).count(),
            4,
            "nil per free bucket"
        );
        assert!(
            !payload.contains(&0xa0),
            "no empty-string flow id is emitted"
        );
        let decoded: Elastic = Elastic::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(buckets(&all_free), buckets(&decoded));

        let mut with_empty_flow: Elastic = Elastic::init_with_dimensions(4, 1, 2);
        with_empty_flow.insert(String::new());
        let encoded = with_empty_flow.serialize_to_bytes().expect("serialize");
        let (_, _, payload) = envelope::split(&encoded).expect("split");
        assert!(
            payload.contains(&0xa0),
            "the inserted empty flow id is a str"
        );
        let decoded: Elastic = Elastic::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(buckets(&with_empty_flow), buckets(&decoded));
        assert_eq!(decoded.query(String::new()), 1);
    }

    /// A table mixing occupied and free buckets, with flows that were evicted
    /// into the light layer, round-trips bucket for bucket.
    #[test]
    fn elastic_mixed_occupancy_round_trips() {
        let mut sketch: Elastic = Elastic::init_with_dimensions(16, 3, 512);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &sketch);
        for _ in 0..10 {
            sketch.insert(primary.to_string());
        }
        for _ in 0..(super::super::LAMBDA * 10) {
            sketch.insert(secondary.clone());
        }
        for i in 0..6u32 {
            sketch.insert(format!("flow::{i}"));
        }
        assert!(
            sketch.heavy.iter().any(|b| b.is_vacant()) && sketch.heavy.iter().any(|b| b.eviction),
            "mix free, occupied and flagged buckets"
        );

        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded: Elastic = Elastic::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(buckets(&sketch), buckets(&decoded));
        assert_eq!(light_cells(&sketch), light_cells(&decoded));
        assert_eq!(
            sketch.query(primary.to_string()),
            decoded.query(primary.to_string())
        );
        assert_eq!(sketch.query(secondary.clone()), decoded.query(secondary));
    }

    /// Votes and light counters are signed on the wire and round-trip
    /// unchanged.
    #[test]
    fn elastic_negative_votes_and_light_counters_round_trip() {
        let mut sketch: Elastic = Elastic::init_with_dimensions(4, 2, 8);
        sketch.insert("flow::signed".to_string());
        let idx = sketch.bucket_index("flow::signed");
        sketch.heavy[idx].vote_neg = -300;
        sketch.heavy[idx].vote_pos = -7;
        sketch
            .light
            .insert_many(&DataInput::String("flow::mouse".to_string()), -9);
        assert!(light_cells(&sketch).iter().any(|&c| c < 0));

        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded: Elastic = Elastic::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(buckets(&sketch), buckets(&decoded));
        assert_eq!(light_cells(&sketch), light_cells(&decoded));
    }

    /// `stale_copies` is not derivable from the buckets, so it is carried; a
    /// decoder that dropped it would report every expanded flow twice.
    #[test]
    fn elastic_stale_copies_round_trips_in_both_states() {
        let mut sketch: Elastic = Elastic::init_with_dimensions(8, 2, 256);
        for i in 0..5u32 {
            for _ in 0..(i + 3) {
                sketch.insert(format!("flow::{i}"));
            }
        }
        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded: Elastic = Elastic::deserialize_from_bytes(&encoded).expect("deserialize");
        assert!(!decoded.stale_copies);

        sketch.expand_heavy();
        assert!(sketch.stale_copies);
        let encoded = sketch.serialize_to_bytes().expect("serialize expanded");
        let decoded: Elastic =
            Elastic::deserialize_from_bytes(&encoded).expect("deserialize expanded");
        assert!(decoded.stale_copies);
        assert_eq!(decoded.bktlen, 16);
        assert_eq!(buckets(&sketch), buckets(&decoded));

        // A query path that reads the flag: stale copies are skipped, so no
        // flow is reported twice.
        let hitters = decoded.heavy_hitters(1);
        assert_eq!(hitters, sketch.heavy_hitters(1));
        assert!(!hitters.is_empty());
        let occupied = decoded.heavy.iter().filter(|b| !b.is_vacant()).count();
        assert_eq!(
            occupied,
            2 * hitters.len(),
            "every reported flow still has a stale twin in the table"
        );
    }

    /// The emitted order is bucket index order for the heavy table and
    /// row-major for the light layer, so a decoded sketch re-serializes to the
    /// bytes it came from.
    #[test]
    fn elastic_decoded_sketch_reserializes_byte_identically() {
        let mut sketch: Elastic = Elastic::init_with_dimensions(16, 3, 512);
        for i in 0..40u32 {
            sketch.insert(format!("fam{}|item{i}", i % 5));
        }
        sketch.expand_heavy();
        let encoded = sketch.serialize_to_bytes().expect("serialize");
        let decoded: Elastic = Elastic::deserialize_from_bytes(&encoded).expect("deserialize");
        assert_eq!(encoded, decoded.serialize_to_bytes().expect("re-serialize"));
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`. Elastic metadata is derived from the
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
    fn elastic_custom_hasher_profile_round_trips_and_is_self_describing() {
        // (a) An Elastic built with a custom-profile hasher round-trips.
        let mut alt: Elastic<AltHasher> = Elastic::init_with_dimensions(8, 2, 256);
        let mut std: Elastic = Elastic::init_with_dimensions(8, 2, 256);
        for _ in 0..4 {
            alt.insert("flow::42".to_string());
            std.insert("flow::42".to_string());
        }
        alt.insert("flow::7".to_string());
        std.insert("flow::7".to_string());

        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded = Elastic::<AltHasher>::deserialize_from_bytes(&alt_bytes).expect("alt decode");
        assert_eq!(buckets(&alt), buckets(&decoded));
        assert_eq!(light_cells(&alt), light_cells(&decoded));

        // (b) Bytes differ from the standard-profile sketch (metadata derived
        // from the different profile).
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            Elastic::<DefaultXxHasher>::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }
}
