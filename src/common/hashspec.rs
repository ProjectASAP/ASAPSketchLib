//! Cross-language hash layer mirroring `sketchlib-go::common`'s
//! `HashSpec` / `DeriveIndex` / `DeriveSign` decomposition for
//! matrix-backed sketches (CountSketch, CountMinSketch).
//!
//! `sketchlib-go` and `asap_sketchlib` share three building blocks for
//! these sketches:
//!
//!   1. A 20-entry seed table (`CANONICAL_HASH_SEED_TABLE`, identical
//!      bit-for-bit to Go's `seedList`).
//!   2. A canonical seed index (`CANONICAL_HASH_SEED` = 5).
//!   3. A single XXH3-64-with-seed call per (key, seed) pair, with the
//!      resulting `u64` then bit-sliced into per-row column indices and
//!      ±1 signs.
//!
//! This module re-exposes that pipeline as a small, byte-key API so
//! sketches that consume keys as raw bytes (e.g. the wire-format
//! `CountSketch::update`) can match Go's emitted matrix
//! cell-for-cell. CountMinSketch's pending byte-parity fix will reuse
//! the same primitives — the seed table, derive_index, and derive_sign
//! are agnostic to whether ±1 signing is applied.
//!
//! # Example
//!
//! ```
//! use asap_sketchlib::common::hashspec::{HashSpec, hash_with_spec, derive_index, derive_sign};
//!
//! let spec = HashSpec::default();             // matches Go's portableHashSpec()
//! let key = b"flow-7";
//! let h = hash_with_spec(&spec, key);         // single XXH3-64 with seed_list[0]
//! let col = derive_index(&spec, 0, h, 512);   // row 0, width 512
//! let sign = derive_sign(&spec, 0, h);        // -1 or +1
//! assert!(col < 512);
//! assert!(sign == -1 || sign == 1);
//! ```

use twox_hash::XxHash3_64;

/// 20-entry seed table shared by all matrix-backed sketches.
///
/// Bit-for-bit identical to `sketchlib-go::common.seedList` — the order
/// MUST match because `derive_index` / `derive_sign` produce byte
/// parity only when both libraries pull the same `u64` from index `i`.
pub const CANONICAL_HASH_SEED_TABLE: [u64; 20] = [
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

/// Default canonical-seed index used by single-hash matrix-sketch
/// operations. Matches `sketchlib-go::common.CanonicalHashSeed` and
/// `asap_sketchlib::common::hash::CANONICAL_HASH_SEED`.
pub const CANONICAL_HASH_SEED: usize = 5;

/// Hash configuration shared by matrix-backed sketches.
///
/// Mirrors `sketchlib-go::common.HashSpec` semantics: a producer
/// records the seed table, the canonical seed index, and the seed
/// derivation strategy so consumers can validate hash compatibility
/// before merging.
///
/// `asap_sketchlib`'s wire-format CountSketch constructs a
/// `HashSpec::default()` and feeds it to [`derive_index`] /
/// [`derive_sign`] on the hot path. The default matches Go's
/// `portableHashSpec()`:
///
/// - `seed_list = CANONICAL_HASH_SEED_TABLE`
/// - `canonical_seed_index = CANONICAL_HASH_SEED` (= 5)
/// - `seed_derivation = SeedDerivation::Packed` — the matrix sketches
///   used today fit in a single packed `u64` (rows × bits_per_row ≤ 64),
///   so a single hash with `seed_list[0]` covers all rows. The
///   `Additive` variant is reserved for the per-row-hash fallback used
///   by larger matrices and is not yet exercised by the wire path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashSpec {
    /// Full seed table, length must be > 0.
    pub seed_list: Vec<u64>,
    /// Index into `seed_list` for the canonical (single-hash) seed.
    pub canonical_seed_index: usize,
    /// Strategy for deriving per-row seeds.
    pub seed_derivation: SeedDerivation,
}

/// Strategy for deriving per-row seeds, mirroring
/// `sketchlib-go::common.SeedDerivation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedDerivation {
    /// All rows share `seed_list[0]`; per-row column index and sign are
    /// extracted via bit slicing on a single `u64` hash. Used when
    /// `rows × bits_per_row ≤ 64`.
    Packed,
    /// Each row r uses `seed_list[(base + r) % len]`; one
    /// XXH3-64-with-seed call per row. Reserved for matrices that
    /// exceed the packed budget.
    Additive,
}

impl Default for HashSpec {
    fn default() -> Self {
        Self {
            seed_list: CANONICAL_HASH_SEED_TABLE.to_vec(),
            canonical_seed_index: CANONICAL_HASH_SEED,
            seed_derivation: SeedDerivation::Packed,
        }
    }
}

impl HashSpec {
    /// Returns the seed used for the single-hash packed path. Matches
    /// Go's `seedList[0]` (because Go's `Hash64` calls `HashIt(0, key)`,
    /// not `HashIt(canonical_seed_index, key)` — the canonical-seed
    /// index governs `CanonicalHash`/`hh_keys` lookups, not the
    /// matrix hot path). The matrix path always uses index 0,
    /// regardless of `canonical_seed_index`.
    #[inline]
    pub fn matrix_seed(&self) -> u64 {
        // Mirror Go's `common.Hash64`: `HashIt(0, key)` reads
        // `seedList[normalizedSeedIdx(0)]` = seedList[0].
        self.seed_list[0]
    }

    /// Returns the seed for row `r` under the additive-offset
    /// derivation (reserved; not used by the packed wire path).
    #[inline]
    pub fn row_seed(&self, row: usize) -> u64 {
        let idx = row % self.seed_list.len();
        self.seed_list[idx]
    }
}

/// Hashes `key` under `spec`'s matrix seed (single XXH3-64-with-seed
/// call, mirroring Go's `common.Hash64`). Producers of byte-parity
/// CountSketch / CMS update paths call this once per key, then invoke
/// [`derive_index`] / [`derive_sign`] per row to populate the matrix.
#[inline]
pub fn hash_with_spec(spec: &HashSpec, key: &[u8]) -> u64 {
    XxHash3_64::oneshot_with_seed(spec.matrix_seed(), key)
}

/// Returns `mask_bits` such that `(1 << mask_bits) - 1` is the column
/// mask for `width`. Mirrors Go's `maskBitsForWidth`.
#[inline]
fn mask_bits_for_width(width: u32) -> u32 {
    if width <= 1 {
        return 1;
    }
    // ceil(log2(width)). Wire-path widths are power-of-two, so this
    // collapses to `width.trailing_zeros()`.
    let mut u = width - 1;
    let mut bits = 0u32;
    while u > 0 {
        bits += 1;
        u >>= 1;
    }
    bits
}

/// Derives the row-local column index from a precomputed `hash`,
/// mirroring `sketchlib-go::common.DeriveIndex`:
///
/// ```text
/// shift = row * mask_bits(width)
/// index = (hash >> shift) & (width - 1)
/// ```
///
/// `width` MUST be power-of-two (the wire-format matrix sketches
/// enforce this on construction); other values produce a mask that
/// does not equal `width - 1` and the caller's matrix-cell layout
/// will diverge from Go's.
#[inline]
pub fn derive_index(_spec: &HashSpec, row: usize, hash: u64, width: u32) -> usize {
    let shift = (row as u32) * mask_bits_for_width(width);
    let mask = (width as u64).wrapping_sub(1);
    ((hash >> shift) & mask) as usize
}

/// Derives the ±1 sign for row `row` from a precomputed `hash`,
/// mirroring `sketchlib-go::common.DeriveSign`:
///
/// ```text
/// bit  = (hash >> (63 - row)) & 1
/// sign = bit == 1 ? +1 : -1
/// ```
///
/// `row` must satisfy `row <= 63`; callers with deeper sketches must
/// switch to the per-row hash fallback (not yet exposed here).
#[inline]
pub fn derive_sign(_spec: &HashSpec, row: usize, hash: u64) -> i64 {
    debug_assert!(
        row <= 63,
        "derive_sign: row {row} out of range for u64 hash"
    );
    let bit = (hash >> (63 - row as u32)) & 1;
    if bit == 0 { -1 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the seed table to Go's `seedList`. Any drift here will
    /// silently desync the matrix-cell layout from `sketchlib-go`'s.
    #[test]
    fn seed_table_matches_sketchlib_go() {
        // Captured from `sketchlib-go::common.seedList` (hash.go:13-34).
        let go_seed_list: [u64; 20] = [
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
        assert_eq!(CANONICAL_HASH_SEED_TABLE, go_seed_list);
        assert_eq!(CANONICAL_HASH_SEED, 5);
    }

    /// Locks `hash_with_spec` to Go's `Hash64` for a representative
    /// key. Captured by running `common.Hash64([]byte("projectasap"))`
    /// in sketchlib-go (see `common/hash_test.go::TestXxh3RegressionVectors`).
    #[test]
    fn hash_with_spec_matches_sketchlib_go() {
        let spec = HashSpec::default();
        let got = hash_with_spec(&spec, b"projectasap");
        // From sketchlib-go::common::hash_test.go: Hash64(key) = 887548862923853302.
        assert_eq!(got, 887548862923853302);
    }

    /// derive_index slices the hash exactly the way Go's DeriveIndex
    /// does. Sanity-check across rows and widths.
    #[test]
    fn derive_index_matches_go_bit_slicing() {
        let spec = HashSpec::default();
        // Same hash Go would compute for "projectasap" with seedList[0].
        let h = hash_with_spec(&spec, b"projectasap");
        // 9-bit width (512 cols) → mask 0x1ff, shift = row*9.
        for row in 0..3 {
            let expected = ((h >> (row as u32 * 9)) & 0x1ff) as usize;
            assert_eq!(derive_index(&spec, row, h, 512), expected);
        }
        // 10-bit width (1024 cols) → mask 0x3ff, shift = row*10.
        for row in 0..3 {
            let expected = ((h >> (row as u32 * 10)) & 0x3ff) as usize;
            assert_eq!(derive_index(&spec, row, h, 1024), expected);
        }
    }

    /// derive_sign extracts the high-bit-minus-row exactly like Go.
    #[test]
    fn derive_sign_matches_go_high_bit() {
        let spec = HashSpec::default();
        let h = hash_with_spec(&spec, b"projectasap");
        for row in 0..5 {
            let bit = (h >> (63 - row as u32)) & 1;
            let expected: i64 = if bit == 0 { -1 } else { 1 };
            assert_eq!(derive_sign(&spec, row, h), expected);
        }
    }

    #[test]
    fn mask_bits_for_width_matches_go() {
        assert_eq!(mask_bits_for_width(1), 1);
        assert_eq!(mask_bits_for_width(2), 1);
        assert_eq!(mask_bits_for_width(4), 2);
        assert_eq!(mask_bits_for_width(512), 9);
        assert_eq!(mask_bits_for_width(1024), 10);
        assert_eq!(mask_bits_for_width(4096), 12);
    }

    #[test]
    fn default_hashspec_has_packed_derivation() {
        let spec = HashSpec::default();
        assert_eq!(spec.seed_derivation, SeedDerivation::Packed);
        assert_eq!(spec.canonical_seed_index, CANONICAL_HASH_SEED);
        assert_eq!(spec.seed_list.len(), CANONICAL_HASH_SEED_TABLE.len());
    }
}
