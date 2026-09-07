//! ASAPv1 wire serialization for the Bloom filter.
//!
//! Child submodule of [`crate::sketches::bloom`]: it holds ALL of Bloom's
//! ASAPv1 serialization (the metadata/payload DTOs, the kind_id constant, and
//! the `serialize_to_bytes` / `deserialize_from_bytes` impls) while the
//! algorithm lives in the parent module file. Being a descendant module, it
//! reads the sketch's private `bits` / `inserted` fields directly without
//! widening any field visibility. See `docs/asapv1_wire_format.md` §3.4.
//!
//! Bloom is one algorithm — a single kind_id `0x17 0x00`. The structural
//! parameters — the grid dimensions (`rows` / `cols`) and the column-derivation
//! **mode** (fast/regular, the [`BloomMode`] tag the filter already carries) —
//! live in the metadata, so the payload is `[words, inserted]`: the packed bit
//! words row-major, plus the insert counter.
//!
//! ## Wire-eligible geometries
//!
//! The wire covers the geometries [`Bloom::with_capacity`] produces: at most
//! [`BLOOM_MAX_SLICES`] slices, a power-of-two `cols`, and at most
//! [`BLOOM_MAX_BITS`] bits. [`Bloom::with_dimensions`] can build a
//! modulo-folded width, which is outside that subset and rejected on **both**
//! sides, so the format never emits bytes it would refuse to read back.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::message_pack_format::envelope;
use crate::{BitMatrix, HashProfile, SketchHasher};

use super::{BLOOM_MAX_BITS, BLOOM_MAX_SLICES, Bloom, BloomMode};

/// Bloom kind_id: family `0x17`, single algorithm variant `0x00`.
const BLOOM_KIND: &[u8] = &[0x17, 0x00];

/// Bits packed into one storage word of the bit grid.
const BITS_PER_WORD: usize = u64::BITS as usize;

/// Bloom descriptor metadata (ASAPv1 §2), a msgpack **map** (`to_vec_named`)
/// with keys in this declaration order — the canonical order the wire spec
/// fixes (Go must mirror it). Hash-spec fields first, then the structural
/// params `rows` / `cols` / `mode`. Per the spec's config→metadata rule, the
/// grid dimensions are configuration (like HLL's `precision`) and so live here
/// rather than in the payload.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BloomMetadata {
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

/// Builds the Bloom descriptor metadata from the hasher's [`HashProfile`], so
/// the wire bytes truthfully describe how the filter was hashed (rather than
/// hardcoding the standard profile). `matrix_seed_index` is the profile's own
/// row seed index; `rows` / `cols` are the filter's structural dimensions and
/// `mode` is the [`BloomMode`] tag.
fn bloom_metadata<H: HashProfile>(rows: u32, cols: u32, mode: &str) -> BloomMetadata {
    BloomMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        matrix_seed_index: H::MATRIX_SEED_INDEX,
        rows,
        cols,
        mode: mode.to_string(),
    }
}

/// Bloom payload (ASAPv1 §3.4), a msgpack **array** (`to_vec`, positional):
/// `[words, inserted]`. `words` is the bit grid packed row-major, one row after
/// another, each row padded out to a whole number of `u64`s; the stride is
/// `ceil(cols / 64)`, derived from the metadata dimensions. `inserted` is the
/// insert counter, which no other field determines.
#[derive(Debug, Serialize, Deserialize)]
struct BloomPayload {
    words: Vec<u64>,
    inserted: u64,
}

/// Checks a `(rows, cols)` pair against the wire-eligible subset: both non-zero,
/// a power-of-two width, at most [`BLOOM_MAX_SLICES`] slices, and a bit capacity
/// that neither overflows nor exceeds [`BLOOM_MAX_BITS`]. Applied before any
/// allocation is sized from the declared dimensions.
fn check_geometry(rows: usize, cols: usize) -> Result<(), String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "Bloom dimensions must be non-zero: rows={rows}, cols={cols}"
        ));
    }
    if !cols.is_power_of_two() {
        return Err(format!("Bloom cols {cols} is not a power of two"));
    }
    if rows > BLOOM_MAX_SLICES {
        return Err(format!(
            "Bloom rows {rows} exceeds BLOOM_MAX_SLICES {BLOOM_MAX_SLICES}"
        ));
    }
    match rows.checked_mul(cols) {
        Some(bits) if bits <= BLOOM_MAX_BITS => Ok(()),
        Some(bits) => Err(format!(
            "Bloom bit capacity {bits} exceeds BLOOM_MAX_BITS {BLOOM_MAX_BITS}"
        )),
        None => Err(format!(
            "Bloom rows*cols overflows: rows={rows}, cols={cols}"
        )),
    }
}

// Wire serialization for the Bloom filter. `wire` is a descendant of the sketch
// module, so these impls read the private `bits` / `inserted` fields and
// construct the struct directly.
impl<Mode, H> Bloom<Mode, H>
where
    Mode: BloomMode,
    H: SketchHasher + HashProfile,
{
    /// Serializes the filter into an ASAPv1 MessagePack envelope
    /// (kind_id `0x17 0x00`). The metadata is derived from the hasher's
    /// [`HashProfile`], so it truthfully describes how the filter was hashed.
    ///
    /// A geometry outside the wire-eligible subset (see the module docs) is an
    /// error rather than bytes that would be refused on decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let (rows, cols) = (self.bits.rows(), self.bits.cols());
        check_geometry(rows, cols)
            .map_err(|e| RmpEncodeError::Syntax(format!("ASAPv1 Bloom envelope: {e}")))?;
        let metadata = rmp_serde::to_vec_named(&bloom_metadata::<H>(
            rows as u32,
            cols as u32,
            Mode::MODE_TAG,
        ))?;
        let payload = rmp_serde::to_vec(&BloomPayload {
            words: self.bits.words().to_vec(),
            inserted: self.inserted,
        })?;
        Ok(envelope::encode(BLOOM_KIND, &metadata, &payload))
    }

    /// Deserializes a filter from an ASAPv1 MessagePack envelope. The grid
    /// dimensions are read from the (validated) metadata and the word stride is
    /// derived from them; the payload carries only the words and the insert
    /// counter. Every inconsistency fails closed.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, metadata, payload) =
            envelope::split(bytes).map_err(RmpDecodeError::Uncategorized)?;
        if kind_id != BLOOM_KIND {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Bloom kind_id mismatch: stored {kind_id:?}, expected {BLOOM_KIND:?}"
            )));
        }
        let meta: BloomMetadata = from_slice(metadata)?;
        // Validate the hash spec + mode against this target; `rows`/`cols` are
        // structural (the filter is dynamically sized), so they are echoed back
        // into the expected block rather than known a priori.
        if meta != bloom_metadata::<H>(meta.rows, meta.cols, Mode::MODE_TAG) {
            return Err(RmpDecodeError::Uncategorized(
                "ASAPv1 Bloom envelope: metadata mismatch".to_string(),
            ));
        }
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        check_geometry(rows, cols).map_err(RmpDecodeError::Uncategorized)?;

        let words_per_row = cols.div_ceil(BITS_PER_WORD);
        let expected_words = rows.checked_mul(words_per_row).ok_or_else(|| {
            RmpDecodeError::Uncategorized(format!(
                "Bloom word count overflows: rows={rows}, words_per_row={words_per_row}"
            ))
        })?;
        let p: BloomPayload = from_slice(payload)?;
        if p.words.len() != expected_words {
            return Err(RmpDecodeError::Uncategorized(format!(
                "Bloom words length {} != rows*words_per_row {expected_words}",
                p.words.len()
            )));
        }
        // Rows are padded out to whole words and no insert can reach past
        // `cols`, but `count_ones` / `fill_ratio` / `estimated_fpp` count every
        // stored bit, so a set padding bit is rejected rather than believed.
        let padding_bits = cols % BITS_PER_WORD;
        if padding_bits != 0 {
            let pad_mask = !((1u64 << padding_bits) - 1);
            if p.words
                .chunks_exact(words_per_row)
                .any(|row| row[words_per_row - 1] & pad_mask != 0)
            {
                return Err(RmpDecodeError::Uncategorized(format!(
                    "Bloom payload: bits set in the row padding past cols {cols}"
                )));
            }
        }

        Ok(Bloom {
            bits: BitMatrix::from_words(rows, cols, p.words)
                .map_err(RmpDecodeError::Uncategorized)?,
            inserted: p.inserted,
            _mode: PhantomData,
            _hasher: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, FastPath, RegularPath};

    const MEMBERS: u64 = 5_000;
    const PROBE_BASE: u64 = 10_000_000;
    const PROBES: u64 = 20_000;

    fn members() -> Vec<DataInput<'static>> {
        (0..MEMBERS).map(DataInput::U64).collect()
    }

    fn probes() -> Vec<DataInput<'static>> {
        (PROBE_BASE..PROBE_BASE + PROBES)
            .map(DataInput::U64)
            .collect()
    }

    /// Every membership answer the source gives, the decoded filter must give:
    /// `true` on all members (no false negative may appear) and the *same*
    /// answer on a large disjoint probe set (a filter decoded onto shifted or
    /// reordered words would agree on the members it saturated but diverge
    /// here).
    fn assert_answers_preserved(
        source: impl Fn(&DataInput) -> bool,
        decoded: impl Fn(&DataInput) -> bool,
    ) {
        for key in members() {
            assert!(decoded(&key), "decoded filter lost a member");
        }
        let differing = probes().iter().filter(|k| source(k) != decoded(k)).count();
        assert_eq!(
            differing, 0,
            "decoded filter answered differently on probes"
        );
    }

    #[test]
    fn bloom_envelope_structure_and_round_trip() {
        let mut filter = Bloom::<RegularPath>::with_dimensions(7, 1 << 14);
        for key in members() {
            filter.insert(&key);
        }
        let bytes = filter.serialize_to_bytes().expect("serialize");

        assert!(bytes.starts_with(envelope::MAGIC));
        assert_eq!(bytes[6], envelope::VERSION);
        assert_eq!(bytes[7], 2, "kind_id_len");
        // Literal, not `BLOOM_KIND`: the constant compared against itself pins
        // nothing, and these bytes are what a Go reader dispatches on.
        assert_eq!(&bytes[8..10], &[0x17, 0x00]);
        let (kind_id, metadata, _) = envelope::split(&bytes).expect("split");
        assert_eq!(kind_id, &[0x17, 0x00]);
        let meta: BloomMetadata = from_slice(metadata).expect("metadata");
        assert_eq!(meta.metadata_version, 1);
        assert_eq!((meta.rows, meta.cols), (7, 1 << 14));
        assert_eq!(meta.mode, "regular");

        let decoded = Bloom::<RegularPath>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            bytes,
            "Bloom serialized bytes differed after round trip"
        );
        assert_eq!(decoded.inserted(), filter.inserted());
        assert_eq!(decoded.fill_ratio(), filter.fill_ratio());
        assert_answers_preserved(|k| filter.contains(k), |k| decoded.contains(k));
    }

    #[test]
    fn bloom_fast_path_round_trip() {
        let mut filter = Bloom::<FastPath>::with_dimensions(7, 1 << 14);
        for key in members() {
            filter.insert(&key);
        }
        let bytes = filter.serialize_to_bytes().expect("serialize");
        let decoded = Bloom::<FastPath>::deserialize_from_bytes(&bytes).expect("decode");

        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
        assert_eq!(decoded.inserted(), filter.inserted());
        assert_answers_preserved(|k| filter.contains(k), |k| decoded.contains(k));
    }

    #[test]
    fn bloom_empty_round_trip() {
        let filter = Bloom::<RegularPath>::default();
        let bytes = filter.serialize_to_bytes().expect("serialize");
        let decoded = Bloom::<RegularPath>::deserialize_from_bytes(&bytes).expect("decode");

        assert!(decoded.is_empty(), "an empty filter decoded with bits set");
        assert_eq!(decoded.inserted(), 0);
        assert_eq!(decoded.rows(), filter.rows());
        assert_eq!(decoded.cols(), filter.cols());
        assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), bytes);
    }

    /// The two paths fold different bits for the same key, so a filter decoded
    /// into the wrong path would answer no about its own members. The metadata
    /// `mode` makes that a decode error instead.
    #[test]
    fn bloom_cross_mode_rejection() {
        let mut regular = Bloom::<RegularPath>::with_dimensions(7, 1 << 12);
        let mut fast = Bloom::<FastPath>::with_dimensions(7, 1 << 12);
        for key in members() {
            regular.insert(&key);
            fast.insert(&key);
        }
        let regular_bytes = regular.serialize_to_bytes().expect("serialize regular");
        let fast_bytes = fast.serialize_to_bytes().expect("serialize fast");
        assert_ne!(regular_bytes, fast_bytes, "the two paths set the same bits");

        assert!(
            Bloom::<FastPath>::deserialize_from_bytes(&regular_bytes).is_err(),
            "regular-path bytes must not decode as fast path"
        );
        assert!(
            Bloom::<RegularPath>::deserialize_from_bytes(&fast_bytes).is_err(),
            "fast-path bytes must not decode as regular path"
        );
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`. Bloom metadata is derived from the
    // profile, so an `AltHasher` filter serializes truthfully. (An *unprofiled*
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
    fn bloom_custom_hasher_profile_round_trips_and_is_self_describing() {
        // (a) A filter built with a custom-profile hasher round-trips.
        let mut alt = Bloom::<RegularPath, AltHasher>::with_dimensions(7, 1 << 12);
        let mut std = Bloom::<RegularPath>::with_dimensions(7, 1 << 12);
        for key in members() {
            alt.insert(&key);
            std.insert(&key);
        }
        let alt_bytes = alt.serialize_to_bytes().expect("alt serialize");
        let decoded = Bloom::<RegularPath, AltHasher>::deserialize_from_bytes(&alt_bytes)
            .expect("alt decode");
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            alt_bytes
        );
        assert_answers_preserved(|k| alt.contains(k), |k| decoded.contains(k));

        // (b) Bytes differ from the standard-profile filter (metadata derived
        // from the different profile).
        let std_bytes = std.serialize_to_bytes().expect("std serialize");
        assert_ne!(alt_bytes, std_bytes);

        // (c) Standard-profile decode fails closed on custom-profile bytes.
        assert!(
            Bloom::<RegularPath>::deserialize_from_bytes(&alt_bytes).is_err(),
            "standard-profile decode must reject custom-profile bytes"
        );
    }

    /// Fail closed on an unexpected metadata key.
    #[test]
    fn bloom_metadata_rejects_unknown_keys() {
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
            mode: String,
            bogus_field: u8, // key not in BloomMetadata
        }
        let m = bloom_metadata::<DefaultXxHasher>(7, 1024, "regular");
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
            mode: m.mode.clone(),
            bogus_field: 7,
        };
        let bytes = rmp_serde::to_vec_named(&extra).expect("encode");
        assert!(
            rmp_serde::from_slice::<BloomMetadata>(&bytes).is_err(),
            "an unexpected metadata key must be rejected"
        );
    }

    /// Builds a real envelope with real metadata around a hand-built payload,
    /// so the crafted-bytes tests exercise the decoder's own rules rather than
    /// the envelope's framing.
    fn crafted(rows: u32, cols: u32, mode: &str, words: Vec<u64>, inserted: u64) -> Vec<u8> {
        let metadata =
            rmp_serde::to_vec_named(&bloom_metadata::<DefaultXxHasher>(rows, cols, mode)).unwrap();
        let payload = rmp_serde::to_vec(&BloomPayload { words, inserted }).unwrap();
        envelope::encode(BLOOM_KIND, &metadata, &payload)
    }

    /// The message a rejected decode fails with, so a test can pin *which*
    /// rule fired rather than settling for any error at all.
    fn decode_error(bytes: &[u8]) -> String {
        Bloom::<RegularPath>::deserialize_from_bytes(bytes)
            .expect_err("decode must fail")
            .to_string()
    }

    /// A crafted envelope carrying another sketch's kind_id must be rejected
    /// even though its metadata and payload parse cleanly.
    #[test]
    fn bloom_rejects_foreign_kind_id() {
        let metadata =
            rmp_serde::to_vec_named(&bloom_metadata::<DefaultXxHasher>(2, 64, "regular")).unwrap();
        let payload = rmp_serde::to_vec(&BloomPayload {
            words: vec![0, 0],
            inserted: 0,
        })
        .unwrap();
        let bytes = envelope::encode(&[0x02, 0x00], &metadata, &payload);
        assert!(
            Bloom::<RegularPath>::deserialize_from_bytes(&bytes).is_err(),
            "a foreign kind_id must be rejected"
        );
    }

    /// `BitMatrix::new` asserts on a zero dimension, so a crafted zero must be
    /// an error rather than a panic.
    #[test]
    fn bloom_rejects_zero_dimension() {
        for (rows, cols) in [(0u32, 64u32), (4, 0)] {
            let err = decode_error(&crafted(rows, cols, "regular", Vec::new(), 0));
            assert!(
                err.contains("dimensions must be non-zero"),
                "rows={rows}, cols={cols} must be rejected by the dimension rule, got {err}"
            );
        }
    }

    /// A modulo-folded width is outside the wire-eligible subset; the same
    /// payload at the neighbouring power of two decodes, so the rule is what
    /// rejects it and not the word count.
    #[test]
    fn bloom_rejects_non_power_of_two_cols() {
        let err = decode_error(&crafted(2, 96, "regular", vec![0; 4], 0));
        assert!(
            err.contains("not a power of two"),
            "a non-power-of-two cols must be rejected as such, got {err}"
        );
        let ok = crafted(2, 128, "regular", vec![0; 4], 0);
        assert!(
            Bloom::<RegularPath>::deserialize_from_bytes(&ok).is_ok(),
            "the power-of-two control case must still decode"
        );
    }

    /// More slices than the seed list has entries is outside the wire-eligible
    /// subset, so crafted bytes declaring them are refused before anything is
    /// sized from the declared row count.
    #[test]
    fn bloom_rejects_too_many_rows() {
        let rows = BLOOM_MAX_SLICES + 1;
        let err = decode_error(&crafted(rows as u32, 64, "regular", vec![0; rows], 0));
        assert!(
            err.contains("BLOOM_MAX_SLICES"),
            "rows past BLOOM_MAX_SLICES must be rejected as such, got {err}"
        );
        // The boundary itself is eligible.
        let ok = crafted(
            BLOOM_MAX_SLICES as u32,
            64,
            "regular",
            vec![0; BLOOM_MAX_SLICES],
            0,
        );
        assert!(Bloom::<RegularPath>::deserialize_from_bytes(&ok).is_ok());
    }

    /// A hostile bit capacity is rejected from the declared dimensions alone,
    /// before any allocation is sized from them (the payload here is two words,
    /// nowhere near the ~2 GiB the metadata claims).
    #[test]
    fn bloom_rejects_oversized_bit_capacity() {
        let cols = (BLOOM_MAX_BITS / 2) as u32;
        let err = decode_error(&crafted(4, cols, "regular", vec![0; 2], 0));
        assert!(
            err.contains("BLOOM_MAX_BITS"),
            "the capacity rule must fire before the word count is even derived, got {err}"
        );
    }

    /// The word stride is derived from `cols`, so a payload whose length
    /// disagrees with it is rejected in either direction.
    #[test]
    fn bloom_rejects_word_count_mismatch() {
        // 3 rows x 128 cols needs 3 * 2 = 6 words.
        for words in [vec![0u64; 5], vec![0u64; 7]] {
            let len = words.len();
            let err = decode_error(&crafted(3, 128, "regular", words, 0));
            assert!(
                err.contains("words length"),
                "{len} words for a 3x128 grid must be rejected by the stride rule, got {err}"
            );
        }
        let ok = crafted(3, 128, "regular", vec![0; 6], 0);
        assert!(Bloom::<RegularPath>::deserialize_from_bytes(&ok).is_ok());
    }

    /// A row's trailing padding is unreachable by `get`, but `count_ones` sums
    /// it — so bits parked there would skew `fill_ratio` and `estimated_fpp`
    /// without changing a single membership answer. Decode rejects them.
    #[test]
    fn bloom_rejects_padding_bits_set() {
        // 2 rows x 8 cols: one word per row, bits 8..64 are padding.
        // Bit 8 is the first padding bit and bit 63 the last; both are rejected,
        // and either row is checked.
        for words in [vec![0xFF, 1 << 8], vec![1 << 63, 0], vec![0, 1 << 40]] {
            let err = decode_error(&crafted(2, 8, "regular", words, 0));
            assert!(
                err.contains("row padding"),
                "bits set in the row padding must be rejected as such, got {err}"
            );
        }
        // Same words with the padding clear decode, and report the fill the
        // eight live bits imply rather than the inflated one.
        let clean = crafted(2, 8, "regular", vec![0xFF, 0], 0);
        let decoded = Bloom::<RegularPath>::deserialize_from_bytes(&clean).expect("clean decode");
        assert_eq!(decoded.fill_ratio(), 0.5);
        // Eight bits set against an insert counter of zero: emptiness is a
        // property of the bits.
        assert_eq!(decoded.inserted(), 0);
        assert!(!decoded.is_empty(), "a filter with bits set reported empty");
    }

    /// A filter sized for a realistic target still delivers it after a round
    /// trip: the decoded bits are the source's, so the measured rate over a
    /// large disjoint probe set stays within the model.
    #[test]
    fn bloom_with_capacity_round_trip_meets_predicted_fpp() {
        const N: usize = 50_000;
        const TARGET: f64 = 0.001;
        let mut filter = Bloom::<RegularPath>::with_capacity(N, TARGET);
        for key in 0..N as u64 {
            filter.insert(&DataInput::U64(key));
        }
        let bytes = filter.serialize_to_bytes().expect("serialize");
        let decoded = Bloom::<RegularPath>::deserialize_from_bytes(&bytes).expect("decode");

        assert_eq!(decoded.rows(), filter.rows());
        assert_eq!(decoded.cols(), filter.cols());
        for key in 0..N as u64 {
            assert!(
                decoded.contains(&DataInput::U64(key)),
                "decoded filter lost member {key}"
            );
        }
        let trials = 200_000u64;
        let false_positives = (PROBE_BASE..PROBE_BASE + trials)
            .filter(|k| decoded.contains(&DataInput::U64(*k)))
            .count();
        let measured = false_positives as f64 / trials as f64;
        let predicted = decoded.predicted_fpp(N);
        // Five binomial standard errors of headroom over `trials` draws.
        let band = 5.0 * (predicted * (1.0 - predicted) / trials as f64).sqrt();
        assert!(
            measured <= predicted + band,
            "decoded filter delivered {measured}, over its predicted {predicted} (+{band})"
        );
    }
}
