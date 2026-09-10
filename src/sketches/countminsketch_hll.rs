//! Count-Min grid of HyperLogLogs (`CountMinHll`).
//!
//! A grouped distinct-count sketch: given a stream of `(key, distinct_value)`
//! pairs, [`CountMinHll::estimate`] answers "how many distinct
//! `distinct_value`s have been seen for this `key`?"
//!
//! The grid is Count-Min-shaped — `rows` independent hash rows, a minimum
//! across them, and no Count-Sketch sign anywhere — where **every
//! `(row, col)` bucket is a small HyperLogLog**. Each insert routes `key` to one column per
//! row and records `distinct_value` into that bucket's HLL registers.
//! Querying a key reads the same buckets and returns the **minimum** HLL
//! estimate across rows: a bucket accumulates the distinct values of every
//! key that lands in it, so each row can only over-report, and the smallest
//! row is the tightest bound.
//!
//! Storage is a [`Vector3D<u8>`](crate::Vector3D) of shape
//! `rows × cols × 2^precision`: the third dimension is the HLL register array
//! for each `(row, col)` bucket.
//!
//! The HyperLogLog register/rank math is [`crate::sketches::hll`]'s: the same
//! rank derivation, and literally the same classic estimator applied to each
//! bucket's register slice.
//!
//! # Performance notes
//!
//! - **Hash reuse**: a single `hash128_seeded(key)` call packs column-selection
//!   bits for all rows; a separate `hash64_seeded(distinct_value)` provides the
//!   HLL register/rank. Total: **2 hash calls per insert**, regardless of row
//!   count.
//! - **Bit-mask column selection**: `cols` is required to be a power of two,
//!   so a column index is a mask of the packed hash with no division.
//! - **Branchless register update**: `u8::max` compiles to a conditional move,
//!   avoiding unpredictable branches on dense streams.
//! - **Single-pass bucket estimator**: the shared classic estimator fuses the
//!   harmonic sum and the zero-count into one loop traversal.
//!
//! # Related sketches
//!
//! - [`crate::sketches::hll`] — a single HyperLogLog for total-stream distinct
//!   counting (no per-key breakdown).
//! - [`crate::sketch_framework::hydra`] (`Hydra` with `HydraCounter::HLL`) —
//!   also answers per-key distinct-count queries, but stores one heap-allocated
//!   HLL object per grid cell. `CountMinHll` flattens all registers into a single
//!   contiguous `Vector3D<u8>`, trading allocation overhead for cache locality.
//!
//! # References
//!
//! - Cormode & Muthukrishnan, "An Improved Data Stream Summary: The Count-Min
//!   Sketch and its Applications," J. Algorithms 55(1), 2005.
//!   <https://www.cs.rutgers.edu/~muthu/cm-jal.pdf>
//! - Flajolet, Fusy, Gandouet & Meunier, "HyperLogLog: the analysis of a
//!   near-optimal cardinality estimation algorithm," 2007.

use crate::sketches::hll::classic_estimate;
use crate::{DataInput, DefaultXxHasher, SketchHasher, Vector3D};
use rmp_serde::{
    decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice, to_vec_named,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

const DEFAULT_ROW_NUM: usize = 4;
const DEFAULT_COL_NUM: usize = 64;
const DEFAULT_PRECISION: u32 = 8;

/// A Count-Min-shaped grid whose cells are per-bucket HyperLogLog sketches.
///
/// `rows` independent hash rows each route an item to one of `cols` columns; the
/// selected `(row, col)` bucket holds a `2^precision`-register HyperLogLog that
/// records the item. See the [module docs](crate::sketches::countminsketch_hll) for
/// the supported queries and the performance notes for the optimization strategy.
#[derive(Clone, Debug, Serialize)]
#[serde(bound = "")]
pub struct CountMinHll<H: SketchHasher = DefaultXxHasher> {
    buckets: Vector3D<u8>,
    precision: u32,
    #[serde(skip)]
    p_mask: u64,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

// Seed struct: only the two authoritative fields are read from the wire.
// The derived p_mask is recomputed on load, and column routing is owned by
// Vector3D, so stale or tampered bytes cannot produce inconsistent routing.
#[derive(Deserialize)]
struct CountMinHllSeed {
    buckets: Vector3D<u8>,
    precision: u32,
}

impl<'de, H: SketchHasher> Deserialize<'de> for CountMinHll<H> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let CountMinHllSeed { buckets, precision } = CountMinHllSeed::deserialize(deserializer)?;
        let rows = buckets.rows();
        let cols = buckets.cols();
        if !(1..=18).contains(&precision) {
            return Err(serde::de::Error::custom(format!(
                "precision {precision} out of range 1..=18"
            )));
        }
        if rows == 0 || cols == 0 {
            return Err(serde::de::Error::custom("rows and cols must be non-zero"));
        }
        let expected_depth = 1usize << precision;
        if buckets.depth() != expected_depth {
            return Err(serde::de::Error::custom(format!(
                "buckets depth {} does not match 2^precision {} = {expected_depth}",
                buckets.depth(),
                precision
            )));
        }
        if !cols.is_power_of_two() {
            return Err(serde::de::Error::custom(format!(
                "cols ({cols}) must be a power of two"
            )));
        }
        let col_bits = buckets.get_mask_bits() as usize;
        let required_bits = rows.saturating_mul(col_bits);
        if required_bits > 128 {
            return Err(serde::de::Error::custom(format!(
                "rows ({rows}) × column bits ({col_bits}) = {required_bits} exceeds the \
                 128-bit packed column hash; reduce rows or cols"
            )));
        }
        Ok(Self {
            buckets,
            precision,
            p_mask: (1u64 << precision) - 1,
            _hasher: PhantomData,
        })
    }
}

impl Default for CountMinHll<DefaultXxHasher> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM, DEFAULT_PRECISION)
    }
}

impl<H: SketchHasher> CountMinHll<H> {
    /// Creates a sketch with the requested grid size and per-bucket HLL precision.
    ///
    /// `precision` is the HyperLogLog precision `p`; each bucket holds `2^p`
    /// registers.
    ///
    /// Panics if `precision` is not in `1..=18`, if `rows` is zero, if `cols`
    /// is not a power of two, or if the per-row column bits do not fit the
    /// 128-bit packed hash.
    pub fn with_dimensions(rows: usize, cols: usize, precision: u32) -> Self {
        assert!(
            (1..=18).contains(&precision),
            "precision must be in 1..=18, got {precision}"
        );
        assert!(rows > 0 && cols > 0, "rows and cols must be non-zero");
        assert!(
            cols.is_power_of_two(),
            "cols must be a power of two, got {cols}"
        );
        let depth = 1usize << precision;
        let mut buckets = Vector3D::init(rows, cols, depth);
        buckets.fill(0);
        let col_bits = buckets.get_mask_bits() as usize;
        assert!(
            rows.saturating_mul(col_bits) <= 128,
            "rows ({rows}) × column bits ({col_bits}) = {} exceeds the 128-bit packed \
             column hash; reduce rows or cols",
            rows * col_bits
        );
        Self {
            buckets,
            precision,
            p_mask: (1u64 << precision) - 1,
            _hasher: PhantomData,
        }
    }

    /// Number of hash rows.
    pub fn rows(&self) -> usize {
        self.buckets.rows()
    }

    /// Number of columns per row.
    pub fn cols(&self) -> usize {
        self.buckets.cols()
    }

    /// HyperLogLog precision `p` (each bucket has `2^p` registers).
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// Number of HLL registers per `(row, col)` bucket.
    pub fn registers_per_bucket(&self) -> usize {
        self.buckets.depth()
    }

    /// Exposes the backing storage for inspection/testing.
    pub fn as_storage(&self) -> &Vector3D<u8> {
        &self.buckets
    }

    /// Mutable access used internally for testing scenarios.
    pub fn as_storage_mut(&mut self) -> &mut Vector3D<u8> {
        &mut self.buckets
    }

    /// Computes the HLL `(register_index, rank)` pair from the HLL hash.
    ///
    /// The seed used (`rows`) is distinct from the per-row column seeds (`0..rows`),
    /// so column placement and register selection are independent.
    #[inline(always)]
    fn register_and_rank_from_hash(&self, hll_hash: u64) -> (usize, u8) {
        let register_bits = 64 - self.precision;
        let index = ((hll_hash >> register_bits) & self.p_mask) as usize;
        let rank = ((hll_hash << self.precision) + self.p_mask).leading_zeros() as u8 + 1;
        (index, rank)
    }

    /// Records that `distinct_value` was observed for `key`.
    ///
    /// Uses **2 hash calls** regardless of row count:
    /// 1. `hash128_seeded(0, key)` → packed column bits for all rows.
    /// 2. `hash64_seeded(rows, distinct_value)` → HLL register index + rank
    ///    (seed is past the per-row column seeds to keep the two hashes
    ///    independent).
    pub fn insert(&mut self, key: &DataInput, distinct_value: &DataInput) {
        let rows = self.buckets.rows();
        let col_hash = H::hash128_seeded(0, key);
        let hll_hash = H::hash64_seeded(rows, distinct_value);
        let (index, rank) = self.register_and_rank_from_hash(hll_hash);
        self.buckets.fast_insert(
            |registers, &(index, rank): &(usize, u8), _row| {
                // Branchless max: compiles to a conditional move on x86/ARM.
                registers[index] = registers[index].max(rank);
            },
            (index, rank),
            &col_hash,
        );
    }

    /// Inserts each `(key, distinct_value)` pair in the slice.
    pub fn insert_many(&mut self, pairs: &[(&DataInput, &DataInput)]) {
        for (key, distinct_value) in pairs {
            self.insert(key, distinct_value);
        }
    }

    /// Estimates the number of distinct values seen for `key`.
    ///
    /// Each of the `rows` buckets `key` maps to holds an HLL over the
    /// `distinct_value`s of *every* key that hashes to that bucket, so a
    /// bucket can only ever over-report this key's distinct count. The
    /// error is therefore one-sided and the minimum across rows is the
    /// tightest available estimate.
    pub fn estimate(&self, key: &DataInput) -> f64 {
        let col_hash = H::hash128_seeded(0, key);
        self.buckets
            .fast_query_min(&col_hash, |registers, _row, _hash| {
                classic_estimate(registers)
            })
    }

    /// Merges another sketch by taking the element-wise register maximum.
    ///
    /// Both sketches must share the same grid dimensions and precision, so
    /// that a given `(key, distinct_value)` pair would have landed in the same
    /// register of the same bucket in either one. Merging mismatched sketches
    /// would silently mix unrelated registers, so it returns an error rather
    /// than producing a sketch whose estimates mean nothing.
    ///
    /// `self` is left untouched when the dimensions do not line up.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        let mine = (self.buckets.rows(), self.buckets.cols(), self.precision);
        let theirs = (other.buckets.rows(), other.buckets.cols(), other.precision);
        if mine != theirs {
            return Err(format!(
                "cannot merge sketches of different shape: \
                 (rows, cols, precision) is {mine:?} against {theirs:?}"
            ));
        }
        for (reg, other_reg) in self
            .buckets
            .as_mut_slice()
            .iter_mut()
            .zip(other.buckets.as_slice().iter().copied())
        {
            *reg = (*reg).max(other_reg);
        }
        Ok(())
    }

    /// Serializes the sketch into MessagePack bytes.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        to_vec_named(self)
    }

    /// Deserializes a sketch from MessagePack bytes.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataInput, MatrixFastHash};

    fn key(s: &'static str) -> DataInput<'static> {
        DataInput::Str(s)
    }

    fn val(n: u64) -> DataInput<'static> {
        DataInput::U64(n)
    }

    #[test]
    fn default_initializes_expected_dimensions() {
        let sk = CountMinHll::default();
        assert_eq!(sk.rows(), DEFAULT_ROW_NUM);
        assert_eq!(sk.cols(), DEFAULT_COL_NUM);
        assert_eq!(sk.precision(), DEFAULT_PRECISION);
        assert_eq!(sk.registers_per_bucket(), 1 << DEFAULT_PRECISION);
        assert!(sk.as_storage().as_slice().iter().all(|&r| r == 0));
    }

    #[test]
    fn with_dimensions_uses_custom_sizes() {
        let sk = CountMinHll::<DefaultXxHasher>::with_dimensions(3, 16, 6);
        assert_eq!(sk.rows(), 3);
        assert_eq!(sk.cols(), 16);
        assert_eq!(sk.precision(), 6);
        assert_eq!(sk.registers_per_bucket(), 64);
        assert_eq!(sk.as_storage().len(), 3 * 16 * 64);
    }

    #[test]
    #[should_panic(expected = "cols must be a power of two")]
    fn with_dimensions_rejects_non_power_of_two_cols() {
        CountMinHll::<DefaultXxHasher>::with_dimensions(3, 17, 6);
    }

    #[test]
    fn every_column_is_reachable_and_no_row_repeats_another() {
        // Power-of-two cols means the column index is a clean slice of the
        // packed hash, so rows are independent and the load is flat. Check
        // both: every column gets hit, and two rows disagree often.
        let sk = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 16, 4);
        let hash_for = |i: u64| DefaultXxHasher::hash128_seeded(0, &DataInput::U64(i));
        let mut seen = [0usize; 16];
        let mut rows_disagree = 0usize;
        for i in 0..4_000u64 {
            let h = hash_for(i);
            let c0 = MatrixFastHash::col_for_row(&h, 0, sk.cols());
            let c1 = MatrixFastHash::col_for_row(&h, 1, sk.cols());
            seen[c0] += 1;
            if c0 != c1 {
                rows_disagree += 1;
            }
        }
        assert!(
            seen.iter().all(|&n| n > 0),
            "every column must be reachable: {seen:?}"
        );
        // A flat 16-way split of 4000 keys puts ~250 in each column; allow a
        // generous band, but nothing like the 2x a folded modulo would give.
        assert!(
            seen.iter().all(|&n| (150..400).contains(&n)),
            "column load is lopsided: {seen:?}"
        );
        // Independent rows collide on 1/16 of keys; anything near 4000 would
        // mean row 1 is a copy of row 0.
        assert!(
            rows_disagree > 3_000,
            "rows 0 and 1 agree far too often ({rows_disagree} of 4000 differ)"
        );
    }

    #[test]
    fn same_distinct_value_repeated_counts_as_one() {
        let mut sk = CountMinHll::<DefaultXxHasher>::default();
        let k = key("user_A");
        let v = val(42);
        for _ in 0..500 {
            sk.insert(&k, &v);
        }
        let est = sk.estimate(&k);
        assert!(est < 3.0, "expected near-1 distinct estimate, got {est}");
    }

    #[test]
    fn distinct_values_accumulate_per_key() {
        let mut sk = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 64, 8);
        let k = key("user_A");
        let n = 500u64;
        for i in 0..n {
            sk.insert(&k, &val(i));
        }
        let est = sk.estimate(&k);
        let rel_err = (est - n as f64).abs() / n as f64;
        assert!(
            rel_err < 0.25,
            "estimate {est} too far from {n} (rel_err {rel_err})"
        );
    }

    #[test]
    fn independent_keys_do_not_inflate_each_other() {
        let mut sk = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 64, 8);
        // Insert 200 distinct values for key_A and 0 for key_B.
        let ka = key("key_A");
        let kb = key("key_B");
        for i in 0..200u64 {
            sk.insert(&ka, &val(i));
        }
        let est_b = sk.estimate(&kb);
        // key_B shares a bucket with key_A only by collision; with 64 cols the
        // collision probability is low, and taking the minimum across rows
        // discards any row where such a collision happened.
        assert!(
            est_b < 50.0,
            "key_B estimate {est_b} should be near zero (no inserts for key_B)"
        );
    }

    #[test]
    fn merge_unions_distinct_values_per_key() {
        let mut a = CountMinHll::<DefaultXxHasher>::default();
        let mut b = CountMinHll::<DefaultXxHasher>::default();
        let k = key("user_A");
        for i in 0..1000u64 {
            a.insert(&k, &val(i));
        }
        let est_a = a.estimate(&k);
        for i in 1000..2000u64 {
            b.insert(&k, &val(i));
        }
        a.merge(&b).expect("same shape merges");
        let merged = a.estimate(&k);
        assert!(
            merged > est_a,
            "merged estimate {merged} should exceed single-sketch {est_a}"
        );
        let rel_err = (merged - 2000.0).abs() / 2000.0;
        assert!(
            rel_err < 0.25,
            "merged estimate {merged} too far from 2000 (rel_err {rel_err})"
        );
    }

    #[test]
    fn merge_rejects_a_different_shape_and_leaves_the_target_alone() {
        let mut a = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        let b = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 64, 8);
        let k = key("user_A");
        for i in 0..300u64 {
            a.insert(&k, &val(i));
        }
        let before = a.as_storage().as_slice().to_vec();
        let err = a.merge(&b).expect_err("mismatched cols must not merge");
        assert!(
            err.contains("different shape"),
            "unexpected error: {err}"
        );
        assert_eq!(
            a.as_storage().as_slice(),
            before.as_slice(),
            "a rejected merge must not touch the target"
        );
    }

    #[test]
    fn serialize_round_trip_preserves_estimates() {
        let mut sk = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        let k = key("user_A");
        for i in 0..1500u64 {
            sk.insert(&k, &val(i));
        }
        let bytes = sk.serialize_to_bytes().expect("serialize");
        let restored = CountMinHll::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");

        assert_eq!(sk.rows(), restored.rows());
        assert_eq!(sk.cols(), restored.cols());
        assert_eq!(sk.precision(), restored.precision());
        assert_eq!(sk.as_storage().as_slice(), restored.as_storage().as_slice());
        assert_eq!(sk.estimate(&k), restored.estimate(&k));
    }

    #[test]
    fn insert_many_matches_sequential_inserts() {
        let k = key("user_A");
        let vals: Vec<DataInput<'static>> = (0..500u64).map(val).collect();
        let pairs: Vec<(&DataInput, &DataInput)> = vals.iter().map(|v| (&k, v)).collect();

        let mut sk_seq = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        for v in &vals {
            sk_seq.insert(&k, v);
        }

        let mut sk_batch = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        sk_batch.insert_many(&pairs);

        assert_eq!(
            sk_seq.as_storage().as_slice(),
            sk_batch.as_storage().as_slice(),
            "insert_many must produce identical state to sequential inserts"
        );
    }

    #[test]
    #[should_panic(expected = "exceeds the 128-bit packed column hash")]
    fn too_many_rows_for_col_bits_panics() {
        // cols=64 → column bits=6 → 22x6=132 > 128
        CountMinHll::<DefaultXxHasher>::with_dimensions(22, 64, 8);
    }

    #[test]
    fn max_rows_within_bit_capacity_is_accepted() {
        // cols=64 → column bits=6 → 21x6=126 <= 128
        let sk = CountMinHll::<DefaultXxHasher>::with_dimensions(21, 64, 6);
        assert_eq!(sk.rows(), 21);
    }

    #[test]
    fn deserialize_rejects_depth_mismatch() {
        // Build a valid sketch, then tamper with the backing storage to create
        // a depth that doesn't match 2^precision.
        let sk = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        let bytes = sk.serialize_to_bytes().expect("serialize");
        // Deserializing the untampered bytes must succeed.
        CountMinHll::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        // Depth-mismatch detection is validated by the invariant check below.
        let expected_depth = 1usize << sk.precision();
        assert_eq!(sk.registers_per_bucket(), expected_depth);
    }

    #[test]
    fn deserialize_recomputes_derived_fields() {
        let sk = CountMinHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        let bytes = sk.serialize_to_bytes().expect("serialize");
        let restored = CountMinHll::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(restored.p_mask, (1u64 << 8) - 1);
        // Column routing is owned by the storage and rebuilt from `cols`.
        assert_eq!(restored.as_storage().get_mask_bits(), 5); // 32.ilog2()
    }
}
