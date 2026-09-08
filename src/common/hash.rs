use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use twox_hash::{XxHash3_64, XxHash3_128};

use super::{DataInput, HeapItem, MatrixFastHash, MatrixHashType};
use smallvec::SmallVec;

/// Default seed index used by single-hash sketch operations.
pub const CANONICAL_HASH_SEED: usize = 5; // 18 and 19 will cause hll test to fail...? is 5 faster...?
/// Seed index used for UnivMon bottom-layer selection.
pub const BOTTOM_LAYER_FINDER: usize = 19;
/// Seed index reserved for Hydra hashing.
pub const HYDRA_SEED: usize = 6;

/// Built-in seed values used by the default hasher.
pub const SEEDLIST: [u64; 20] = [
    0xcafe3553,
    0xade3415118,
    0x8cc70208,
    0x2f024b2b,
    0x451a3df5,
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
    0xcbbb9d5d,
    0x629a292a,
    0x9159015a,
    0x152fecd8,
    0x67332667,
    0x8eb44a87,
    0xdb0c2e0d,
];

#[inline(always)]
fn normalized_seed_idx(d: usize) -> usize {
    d % SEEDLIST.len()
}

/// Rows a per-row-seeded matrix sketch hashes independently. Row `r` and row
/// `r + MATRIX_MAX_ROWS` draw the same [`SEEDLIST`] entry.
pub const MATRIX_MAX_ROWS: usize = SEEDLIST.len();

/// Checks a matrix sketch's row count against [`MATRIX_MAX_ROWS`], naming
/// `sketch` in the error. Shared by the ASAPv1 encoders and decoders.
pub(crate) fn check_matrix_rows(sketch: &str, rows: usize) -> Result<(), String> {
    if rows > MATRIX_MAX_ROWS {
        return Err(format!(
            "{sketch} rows {rows} exceeds MATRIX_MAX_ROWS {MATRIX_MAX_ROWS}"
        ));
    }
    Ok(())
}

/// Trait abstracting hash function signatures for probabilistic data structures.
///
/// All methods are static (no `&self`) to enable zero-cost monomorphization.
/// Implement this trait to inject a custom hash algorithm into any sketch struct.
pub trait SketchHasher: Clone + Debug {
    /// Hash representation used by matrix-backed sketches.
    type HashType: MatrixFastHash + Clone + Debug;

    /// Hashes an input into a 64-bit value with the selected seed.
    fn hash64_seeded(d: usize, key: &DataInput) -> u64;
    /// Hashes an input into a 128-bit value with the selected seed.
    fn hash128_seeded(d: usize, key: &DataInput) -> u128;
    /// Hashes a heap-owned key into a 64-bit value with the selected seed.
    fn hash_item64_seeded(d: usize, key: &HeapItem) -> u64;
    /// Hashes a heap-owned key into a 128-bit value with the selected seed.
    fn hash_item128_seeded(d: usize, key: &HeapItem) -> u128;

    /// Produces the matrix hash form used by matrix-backed sketches.
    fn hash_for_matrix_seeded(
        seed_idx: usize,
        rows: usize,
        cols: usize,
        key: &DataInput,
    ) -> Self::HashType;
}

/// Describes, in ASAPv1 wire terms, *how* a [`SketchHasher`] hashes — the hash
/// spec that a serialized sketch carries in its metadata (see
/// `docs/asapv1_wire_format.md` §2). Serialization is bounded on this trait so
/// the metadata is **derived from the hasher** rather than hardcoded: an
/// unprofiled hasher simply cannot be serialized (a compile-time guarantee that
/// fails closed), and a custom hasher that declares a profile serializes
/// truthfully — because `seed_list()` is inlined, its bytes are fully
/// self-describing on the wire.
pub trait HashProfile {
    /// Stable global id, e.g. `"projectasap.xxh3.seedlist.v1"` (authoritative).
    const PROFILE_ID: &'static str;
    /// Hash algorithm identifier, e.g. `"xxh3_64_128"`.
    const ALGORITHM: &'static str;
    /// Seed-derivation scheme, e.g. `"seed_list_index_wrap"`.
    const SEED_DERIVATION: &'static str;
    /// Input-encoding identifier, e.g. `"projectasap.input.v1"`.
    const INPUT_ENCODING: &'static str;
    /// The full seed list, inlined into the metadata so the bytes self-describe.
    fn seed_list() -> Vec<u64>;
    /// Seed-list index a single-hash sketch (HLL) hashes with.
    const CANONICAL_SEED_INDEX: u32;
    /// Seed-list index a matrix-backed sketch (Count-Min) hashes rows with.
    const MATRIX_SEED_INDEX: u32;
}

/// Default hasher using twox_hash (XxHash3). This is the built-in implementation
/// used when no custom hasher is specified.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DefaultXxHasher;

/// The standard ProjectASAP hash profile (`docs/asapv1_wire_format.md` §2).
/// This is the single source of truth for the profile's wire values.
impl HashProfile for DefaultXxHasher {
    const PROFILE_ID: &'static str = "projectasap.xxh3.seedlist.v1";
    const ALGORITHM: &'static str = "xxh3_64_128";
    const SEED_DERIVATION: &'static str = "seed_list_index_wrap";
    const INPUT_ENCODING: &'static str = "projectasap.input.v1";
    fn seed_list() -> Vec<u64> {
        SEEDLIST.to_vec()
    }
    const CANONICAL_SEED_INDEX: u32 = CANONICAL_HASH_SEED as u32;
    const MATRIX_SEED_INDEX: u32 = 0;
}

impl SketchHasher for DefaultXxHasher {
    type HashType = MatrixHashType;

    #[inline(always)]
    fn hash64_seeded(d: usize, key: &DataInput) -> u64 {
        let seed = SEEDLIST[normalized_seed_idx(d)];
        match key {
            DataInput::I32(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::I64(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::U32(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            DataInput::U64(u) => XxHash3_64::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            DataInput::F32(f) => XxHash3_64::oneshot_with_seed(seed, &f.to_ne_bytes()),
            DataInput::F64(f) => XxHash3_64::oneshot_with_seed(seed, &f.to_ne_bytes()),
            DataInput::Str(s) => XxHash3_64::oneshot_with_seed(seed, (*s).as_bytes()),
            DataInput::String(s) => XxHash3_64::oneshot_with_seed(seed, (*s).as_bytes()),
            DataInput::Bytes(items) => XxHash3_64::oneshot_with_seed(seed, items),
            DataInput::I8(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::I16(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::I128(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u128).to_ne_bytes()),
            DataInput::ISIZE(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::U8(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            DataInput::U16(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            DataInput::U128(u) => XxHash3_64::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            DataInput::USIZE(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
        }
    }

    #[inline(always)]
    fn hash128_seeded(d: usize, key: &DataInput) -> u128 {
        let seed = SEEDLIST[normalized_seed_idx(d)];
        match key {
            DataInput::I32(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::I64(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::U32(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            DataInput::U64(u) => XxHash3_128::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            DataInput::F32(f) => XxHash3_128::oneshot_with_seed(seed, &f.to_ne_bytes()),
            DataInput::F64(f) => XxHash3_128::oneshot_with_seed(seed, &f.to_ne_bytes()),
            DataInput::Str(s) => XxHash3_128::oneshot_with_seed(seed, (*s).as_bytes()),
            DataInput::String(s) => XxHash3_128::oneshot_with_seed(seed, (*s).as_bytes()),
            DataInput::Bytes(items) => XxHash3_128::oneshot_with_seed(seed, items),
            DataInput::I8(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::I16(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::I128(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u128).to_ne_bytes()),
            DataInput::ISIZE(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            DataInput::U8(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            DataInput::U16(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            DataInput::U128(u) => XxHash3_128::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            DataInput::USIZE(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
        }
    }

    #[inline(always)]
    fn hash_item128_seeded(d: usize, key: &HeapItem) -> u128 {
        let seed = SEEDLIST[normalized_seed_idx(d)];
        match key {
            HeapItem::I32(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::I64(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::U32(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            HeapItem::U64(u) => XxHash3_128::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            HeapItem::F32(f) => XxHash3_128::oneshot_with_seed(seed, &f.to_ne_bytes()),
            HeapItem::F64(f) => XxHash3_128::oneshot_with_seed(seed, &f.to_ne_bytes()),
            HeapItem::String(s) => XxHash3_128::oneshot_with_seed(seed, (*s).as_bytes()),
            HeapItem::Bytes(b) => XxHash3_128::oneshot_with_seed(seed, b),
            HeapItem::I8(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::I16(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::I128(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u128).to_ne_bytes()),
            HeapItem::ISIZE(i) => XxHash3_128::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::U8(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            HeapItem::U16(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            HeapItem::U128(u) => XxHash3_128::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            HeapItem::USIZE(u) => XxHash3_128::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
        }
    }

    #[inline(always)]
    fn hash_item64_seeded(d: usize, key: &HeapItem) -> u64 {
        let seed = SEEDLIST[normalized_seed_idx(d)];
        match key {
            HeapItem::I32(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::I64(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::U32(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            HeapItem::U64(u) => XxHash3_64::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            HeapItem::F32(f) => XxHash3_64::oneshot_with_seed(seed, &f.to_ne_bytes()),
            HeapItem::F64(f) => XxHash3_64::oneshot_with_seed(seed, &f.to_ne_bytes()),
            HeapItem::String(s) => XxHash3_64::oneshot_with_seed(seed, (*s).as_bytes()),
            HeapItem::Bytes(b) => XxHash3_64::oneshot_with_seed(seed, b),
            HeapItem::I8(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::I16(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::I128(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u128).to_ne_bytes()),
            HeapItem::ISIZE(i) => XxHash3_64::oneshot_with_seed(seed, &(*i as u64).to_ne_bytes()),
            HeapItem::U8(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            HeapItem::U16(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
            HeapItem::U128(u) => XxHash3_64::oneshot_with_seed(seed, &(*u).to_ne_bytes()),
            HeapItem::USIZE(u) => XxHash3_64::oneshot_with_seed(seed, &(*u as u64).to_ne_bytes()),
        }
    }

    #[inline(always)]
    fn hash_for_matrix_seeded(
        seed_idx: usize,
        rows: usize,
        cols: usize,
        key: &DataInput,
    ) -> Self::HashType {
        hash_for_matrix_seeded_generic::<Self>(seed_idx, rows, cols, key)
    }
}

/// Hashes a `u64` that is already a digest, with a finalizing mix instead of
/// a full byte-wise hash.
///
/// The index maps in this crate are keyed by xxh3 digests; running SipHash over
/// them again buys nothing. The finalizer is what a table index still needs: the
/// digests come from a fixed seed list, so passing them through unchanged would
/// let chosen keys share the low bits a `HashMap` buckets on and collapse a
/// bucket into a linear scan.
///
/// Only `write_u64` is on the fast path. The byte fallback keeps the type a
/// valid [`Hasher`](std::hash::Hasher), but it is a weak mixer over more than
/// eight bytes and this type is not meant for byte-slice or `String` keys.
#[derive(Default, Clone, Copy, Debug)]
pub struct DigestHasher(u64);

impl std::hash::Hasher for DigestHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^= h >> 33;
        h
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }

    #[inline(always)]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

/// [`BuildHasher`](std::hash::BuildHasher) for [`DigestHasher`].
pub type DigestBuildHasher = std::hash::BuildHasherDefault<DigestHasher>;

// ---------------------------------------------------------------------------
// Backwards-compatible free functions — delegate to DefaultXxHasher
// ---------------------------------------------------------------------------

/// I32, U32, F32 will all be treated as 64-bit value.
#[inline(always)]
pub fn hash64_seeded(d: usize, key: &DataInput) -> u64 {
    DefaultXxHasher::hash64_seeded(d, key)
}

#[inline(always)]
/// Hashes an input into a 128-bit value with the selected seed.
pub fn hash128_seeded(d: usize, key: &DataInput) -> u128 {
    DefaultXxHasher::hash128_seeded(d, key)
}

// for speed, add separate function
/// Hashes a heap-owned key into a 128-bit value with the selected seed.
#[inline(always)]
pub fn hash_item128_seeded(d: usize, key: &HeapItem) -> u128 {
    DefaultXxHasher::hash_item128_seeded(d, key)
}

// for speed, add separate function
/// Hashes a heap-owned key into a 64-bit value with the selected seed.
#[inline(always)]
pub fn hash_item64_seeded(d: usize, key: &HeapItem) -> u64 {
    DefaultXxHasher::hash_item64_seeded(d, key)
}

// ---------------------------------------------------------------------------
// Matrix hash helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn mask_bits_for_cols(cols: usize) -> u32 {
    if cols.is_power_of_two() {
        cols.ilog2()
    } else {
        cols.ilog2() + 1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Layout used to store per-row fast-path hashes.
pub enum MatrixHashMode {
    /// Packs row hashes into one 64-bit value.
    Packed64,
    /// Packs row hashes into one 128-bit value.
    Packed128,
    /// Stores one row hash per row.
    Rows,
}

/// Chooses a matrix hash layout for the given sketch dimensions.
#[inline(always)]
pub fn hash_mode_for_matrix(rows: usize, cols: usize) -> MatrixHashMode {
    let mask_bits = mask_bits_for_cols(cols) as usize;
    // Reserve one extra bit per row for sketches that use a sign bit (e.g., Count Sketch).
    let bits_per_row = mask_bits + 1;
    let bits_required = bits_per_row.saturating_mul(rows);
    if bits_required <= 64 {
        MatrixHashMode::Packed64
    } else if bits_required <= 128 {
        MatrixHashMode::Packed128
    } else {
        MatrixHashMode::Rows
    }
}

/// Creates a fast-path hash for a matrix-backed sketch using the default seed.
pub fn hash_for_matrix(rows: usize, cols: usize, key: &DataInput) -> MatrixHashType {
    hash_for_matrix_seeded(0, rows, cols, key)
}

/// Creates a fast-path hash for a matrix-backed sketch with a custom seed.
/// Chooses a packed hash when the required bits fit in 128; otherwise uses per-row hashes.
#[inline(always)]
pub fn hash_for_matrix_seeded(
    seed_idx: usize,
    rows: usize,
    cols: usize,
    key: &DataInput,
) -> MatrixHashType {
    let mode = hash_mode_for_matrix(rows, cols);
    hash_for_matrix_seeded_with_mode(seed_idx, mode, rows, key)
}

/// Creates a fast-path hash using a pre-selected hash mode.
#[inline(always)]
pub fn hash_for_matrix_seeded_with_mode(
    seed_idx: usize,
    mode: MatrixHashMode,
    rows: usize,
    key: &DataInput,
) -> MatrixHashType {
    hash_for_matrix_seeded_with_mode_generic::<DefaultXxHasher>(seed_idx, mode, rows, key)
}

/// Generic version of matrix hash that uses a custom hasher.
#[inline(always)]
pub fn hash_for_matrix_seeded_with_mode_generic<H: SketchHasher>(
    seed_idx: usize,
    mode: MatrixHashMode,
    rows: usize,
    key: &DataInput,
) -> MatrixHashType {
    match mode {
        MatrixHashMode::Packed64 => {
            MatrixHashType::Packed64(H::hash64_seeded(seed_idx % SEEDLIST.len(), key))
        }
        MatrixHashMode::Packed128 => {
            MatrixHashType::Packed128(H::hash128_seeded(seed_idx % SEEDLIST.len(), key))
        }
        MatrixHashMode::Rows => {
            let mut hashes = SmallVec::<[u64; 8]>::with_capacity(rows);
            for row in 0..rows {
                let seed = (seed_idx + row) % SEEDLIST.len();
                hashes.push(H::hash64_seeded(seed, key));
            }
            MatrixHashType::Rows(hashes)
        }
    }
}

/// Generic version of hash_for_matrix that uses a custom hasher.
pub fn hash_for_matrix_generic<H: SketchHasher>(
    rows: usize,
    cols: usize,
    key: &DataInput,
) -> MatrixHashType {
    hash_for_matrix_seeded_generic::<H>(0, rows, cols, key)
}

/// Generic version of hash_for_matrix_seeded that uses a custom hasher.
#[inline(always)]
pub fn hash_for_matrix_seeded_generic<H: SketchHasher>(
    seed_idx: usize,
    rows: usize,
    cols: usize,
    key: &DataInput,
) -> MatrixHashType {
    let mode = hash_mode_for_matrix(rows, cols);
    hash_for_matrix_seeded_with_mode_generic::<H>(seed_idx, mode, rows, key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_uniform_f64, sample_zipf_u64};
    use std::collections::HashSet;

    #[derive(Clone, Debug)]
    struct Packed64Hasher;

    impl SketchHasher for Packed64Hasher {
        type HashType = u64;

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
            <u64 as MatrixFastHash>::assert_compatible(rows, cols);
            DefaultXxHasher::hash64_seeded(seed_idx, key)
        }
    }

    #[derive(Clone, Debug)]
    struct Packed128Hasher;

    impl SketchHasher for Packed128Hasher {
        type HashType = u128;

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
            <u128 as MatrixFastHash>::assert_compatible(rows, cols);
            DefaultXxHasher::hash128_seeded(seed_idx, key)
        }
    }

    // Test: ensures the hash collision is not likely to happen
    // the input cardinality should be roughly the same with cardinality of hashed value
    #[test]
    fn hash128_seeded_preserves_cardinality() {
        const SEED_IDX: usize = 0;
        const SAMPLE_SIZE: usize = 5_000;

        let uniform_values = sample_uniform_f64(0.0, 1_000_000.0, SAMPLE_SIZE, 42);
        let uniform_input_cardinality = uniform_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<HashSet<_>>()
            .len();
        let uniform_hash_cardinality = uniform_values
            .iter()
            .map(|value| hash128_seeded(SEED_IDX, &DataInput::F64(*value)))
            .collect::<HashSet<_>>()
            .len();
        assert_eq!(
            uniform_input_cardinality, uniform_hash_cardinality,
            "uniform samples should not collide after hashing"
        );

        let zipf_values = sample_zipf_u64(10_000, 1.1, SAMPLE_SIZE, 7);
        let zipf_input_cardinality = zipf_values.iter().copied().collect::<HashSet<_>>().len();
        let zipf_hash_cardinality = zipf_values
            .iter()
            .map(|value| hash128_seeded(SEED_IDX, &DataInput::U64(*value)))
            .collect::<HashSet<_>>()
            .len();
        assert_eq!(
            zipf_input_cardinality, zipf_hash_cardinality,
            "zipf samples should not collide after hashing"
        );
    }

    #[test]
    fn hash128_seeded_is_deterministic_for_repeated_inputs() {
        const SEED_IDX: usize = 3;
        let key = DataInput::String("deterministic-key".to_string());
        let expected = hash128_seeded(SEED_IDX, &key);
        for _ in 0..100 {
            assert_eq!(expected, hash128_seeded(SEED_IDX, &key));
        }
    }

    #[test]
    fn xxh3_regression_vectors_match_go() {
        let key = DataInput::Bytes(b"projectasap");

        assert_eq!(hash64_seeded(0, &key), 887548862923853302);
        assert_eq!(
            hash64_seeded(CANONICAL_HASH_SEED, &key),
            8535098769003547387
        );
        assert_eq!(
            hash128_seeded(CANONICAL_HASH_SEED, &key),
            199634325175509853918794253804029959851u128
        );
    }

    /// A key stored as `HeapItem::Bytes` must hash exactly like the borrowed
    /// `DataInput::Bytes` a caller queries with, or a sketch that hashes its
    /// stored keys on rebuild would file them under digests no query reaches.
    #[test]
    fn owned_byte_keys_hash_like_the_borrowed_input() {
        for raw in [
            b"".as_slice(),
            b"projectasap",
            &[0xff, 0x00, 0xfe],
            &[0x80; 64],
        ] {
            let borrowed = DataInput::Bytes(raw);
            let owned = HeapItem::Bytes(raw.to_vec());
            for seed in [0usize, 1, CANONICAL_HASH_SEED, SEEDLIST.len() + 3] {
                assert_eq!(
                    hash64_seeded(seed, &borrowed),
                    hash_item64_seeded(seed, &owned),
                    "64-bit digests disagree for {raw:?} at seed {seed}"
                );
                assert_eq!(
                    hash128_seeded(seed, &borrowed),
                    hash_item128_seeded(seed, &owned),
                    "128-bit digests disagree for {raw:?} at seed {seed}"
                );
            }
        }
    }

    #[test]
    fn hash_seed_index_wraps_like_go() {
        let key = DataInput::Bytes(b"projectasap");

        assert_eq!(
            hash64_seeded(SEEDLIST.len() + CANONICAL_HASH_SEED, &key),
            hash64_seeded(CANONICAL_HASH_SEED, &key)
        );
        assert_eq!(
            hash128_seeded(SEEDLIST.len() + CANONICAL_HASH_SEED, &key),
            hash128_seeded(CANONICAL_HASH_SEED, &key)
        );
    }

    #[test]
    fn packed64_hasher_accepts_compatible_dimensions() {
        let key = DataInput::U64(7);
        let hash = Packed64Hasher::hash_for_matrix_seeded(0, 3, 4096, &key);
        assert_eq!(hash, DefaultXxHasher::hash64_seeded(0, &key));
    }

    #[test]
    fn packed128_hasher_accepts_larger_dimensions() {
        let key = DataInput::U64(11);
        let hash = Packed128Hasher::hash_for_matrix_seeded(0, 8, 4096, &key);
        assert_eq!(hash, DefaultXxHasher::hash128_seeded(0, &key));
    }

    #[test]
    #[should_panic(
        expected = "SketchHasher hash type u64 cannot represent fast-path hash for rows=8, cols=4096; use u128 or MatrixHashType"
    )]
    fn packed64_hasher_rejects_oversized_dimensions() {
        let key = DataInput::U64(19);
        let _ = Packed64Hasher::hash_for_matrix_seeded(0, 8, 4096, &key);
    }

    /// A `HashMap` buckets on the low bits of `finish()`. Digests that agree
    /// there must not stay agreed, or one bucket absorbs all of them.
    #[test]
    fn digest_hasher_spreads_digests_that_share_their_low_bits() {
        use std::hash::Hasher;

        let spread = |shift: u32| {
            let mut seen = std::collections::HashSet::new();
            for i in 0..1024u64 {
                let mut h = DigestHasher::default();
                h.write_u64(i << shift);
                seen.insert(h.finish() & 0x3ff);
            }
            seen.len()
        };

        // Every digest is a multiple of 2^16, so all 1024 share ten low bits
        // and a pass-through hash puts every one of them in bucket 0. Well
        // spread, 1024 draws over 1024 buckets fill about 647 of them.
        assert!(
            spread(16) > 550,
            "digests sharing their low bits collapsed into {} buckets",
            spread(16)
        );
        for shift in [10u32, 24, 32, 40, 48] {
            assert!(
                spread(shift) > 550,
                "digests that are multiples of 2^{shift} collapsed into {} buckets",
                spread(shift)
            );
        }
    }

    /// The strict avalanche criterion: flipping one input bit must flip each
    /// output bit about half the time. A mix that keeps some input bits away
    /// from some output bits fails this even though it still spreads buckets.
    #[test]
    fn digest_hasher_avalanche_flips_about_half_the_output_bits() {
        use std::hash::Hasher;

        const SAMPLES: u64 = 4096;
        let once = |v: u64| {
            let mut h = DigestHasher::default();
            h.write_u64(v);
            h.finish()
        };

        for input_bit in 0..64u32 {
            let mut flips_per_output_bit = [0u32; 64];
            let mut total_flips = 0u64;
            for v in 0..SAMPLES {
                let delta = once(v) ^ once(v ^ (1u64 << input_bit));
                total_flips += u64::from(delta.count_ones());
                for (output_bit, flips) in flips_per_output_bit.iter_mut().enumerate() {
                    *flips += ((delta >> output_bit) & 1) as u32;
                }
            }

            let mean = total_flips as f64 / SAMPLES as f64;
            assert!(
                (31.0..=33.0).contains(&mean),
                "flipping input bit {input_bit} flipped {mean} of 64 output bits on average"
            );
            for (output_bit, flips) in flips_per_output_bit.iter().enumerate() {
                let rate = f64::from(*flips) / SAMPLES as f64;
                assert!(
                    (0.42..=0.58).contains(&rate),
                    "input bit {input_bit} flipped output bit {output_bit} at rate {rate}"
                );
            }
        }
    }

    /// The mix loses none of the digest it is handed: distinct digests keep
    /// distinct hashes, so the index maps never collide on the mix itself.
    #[test]
    fn digest_hasher_keeps_distinct_digests_distinct() {
        use std::hash::BuildHasher;

        let build = DigestBuildHasher::default();
        let mut seen = HashSet::with_capacity(4096);
        for v in 0..4096u64 {
            assert!(
                seen.insert(build.hash_one(v)),
                "digest {v} collided with an earlier one"
            );
        }
        assert_ne!(build.hash_one(0u64), build.hash_one(1u64));
    }

    /// The byte path a `String` or byte-slice key reaches. It is the weak half
    /// of this hasher, but it still has to separate keys of eight bytes or
    /// fewer, and it has to depend on the order of the bytes it is handed. It
    /// is also the only path that can observe the starting state, which
    /// `write_u64` overwrites, so the seed-free `Default` is pinned here.
    #[test]
    fn digest_hasher_write_separates_short_byte_slices() {
        use std::hash::{BuildHasher, Hasher};

        let bytes = |slices: &[&[u8]]| {
            let mut h = DigestHasher::default();
            for slice in slices {
                h.write(slice);
            }
            h.finish()
        };

        let mut seen = HashSet::with_capacity(1024);
        for i in 0..1024u64 {
            let key = (i * 0x0001_0001_0001_0001).to_be_bytes();
            assert!(
                seen.insert(bytes(&[&key])),
                "eight-byte key {key:?} collided with an earlier one"
            );
        }

        assert_ne!(bytes(&[b"ab"]), bytes(&[b"ba"]), "byte order ignored");
        assert_ne!(bytes(&[b"ab"]), bytes(&[b"abc"]), "trailing byte ignored");
        assert_eq!(
            bytes(&[b"ab"]),
            bytes(&[b"a", b"b"]),
            "split writes should hash like one write"
        );
        assert_eq!(
            DigestBuildHasher::default().hash_one("seed-free"),
            DigestBuildHasher::default().hash_one("seed-free"),
            "independently built hashers disagreed, so `Default` carries a seed"
        );
    }
}
