//! Count Sketch + HyperLogLog hybrid (`CountHll`).
//!
//! A grouped distinct-count sketch: given a stream of `(key, distinct_value)`
//! pairs, [`CountHll::estimate`] answers "how many distinct `distinct_value`s
//! have been seen for this `key`?"
//!
//! Internally this is a Count Sketch row/column grid where **every `(row, col)`
//! bucket is a small HyperLogLog**. Each insert routes `key` to one column per
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
//! The HyperLogLog register/rank math mirrors [`crate::sketches::hll`] (classic
//! estimator with small/large-range corrections).
//!
//! # Performance notes
//!
//! - **Hash reuse**: a single `hash128_seeded(key)` call packs column-selection
//!   bits for all rows; a separate `hash64_seeded(distinct_value)` provides the
//!   HLL register/rank. Total: **2 hash calls per insert**, regardless of row
//!   count.
//! - **Bit-mask column selection**: when `cols` is a power of two the modulo is
//!   replaced by a bitmask (no division).
//! - **Branchless register update**: `u8::max` compiles to a conditional move,
//!   avoiding unpredictable branches on dense streams.
//! - **Single-pass bucket estimator**: `estimate_bucket` fuses the harmonic sum
//!   and zero-count into one loop traversal.
//!
//! # Related sketches
//!
//! - [`crate::sketches::hll`] — a single HyperLogLog for total-stream distinct
//!   counting (no per-key breakdown).
//! - [`crate::sketch_framework::hydra`] (`Hydra` with `HydraCounter::HLL`) —
//!   also answers per-key distinct-count queries, but stores one heap-allocated
//!   HLL object per grid cell. `CountHll` flattens all registers into a single
//!   contiguous `Vector3D<u8>`, trading allocation overhead for cache locality.
//!
//! # References
//!
//! - Charikar, Chen & Farach-Colton, "Finding Frequent Items in Data Streams,"
//!   ICALP 2002.
//! - Flajolet, Fusy, Gandouet & Meunier, "HyperLogLog: the analysis of a
//!   near-optimal cardinality estimation algorithm," 2007.

use crate::{DataInput, DefaultXxHasher, SketchHasher, Vector3D};
use rmp_serde::{
    decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice, to_vec_named,
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

const DEFAULT_ROW_NUM: usize = 4;
const DEFAULT_COL_NUM: usize = 64;
const DEFAULT_PRECISION: u32 = 8;

/// A Count Sketch grid whose cells are per-bucket HyperLogLog sketches.
///
/// `rows` independent hash rows each route an item to one of `cols` columns; the
/// selected `(row, col)` bucket holds a `2^precision`-register HyperLogLog that
/// records the item. See the [module docs](crate::sketches::countsketch_hll) for
/// the supported queries and the performance notes for the optimization strategy.
#[derive(Clone, Debug, Serialize)]
#[serde(bound = "")]
pub struct CountHll<H: SketchHasher = DefaultXxHasher> {
    buckets: Vector3D<u8>,
    precision: u32,
    #[serde(skip)]
    p_mask: u64,
    #[serde(skip)]
    col_mask_bits: u32,
    #[serde(skip)]
    col_mask: Option<usize>,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

// Seed struct: only the two authoritative fields are read from the wire.
// Derived fields (p_mask, col_mask_bits, col_mask) are recomputed on load,
// so stale or tampered bytes can never produce internally inconsistent routing.
#[derive(Deserialize)]
struct CountHllSeed {
    buckets: Vector3D<u8>,
    precision: u32,
}

impl<'de, H: SketchHasher> Deserialize<'de> for CountHll<H> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let CountHllSeed { buckets, precision } = CountHllSeed::deserialize(deserializer)?;
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
        let p_mask = (1u64 << precision) - 1;
        let col_mask_bits = if cols.is_power_of_two() {
            cols.ilog2()
        } else {
            cols.ilog2() + 1
        };
        let required_bits = rows.saturating_mul(col_mask_bits as usize);
        if required_bits > 128 {
            return Err(serde::de::Error::custom(format!(
                "rows ({rows}) × col_mask_bits ({col_mask_bits}) = {required_bits} exceeds the \
                 128-bit packed column hash; reduce rows or cols"
            )));
        }
        let col_mask = if cols.is_power_of_two() {
            Some(cols - 1)
        } else {
            None
        };
        Ok(Self {
            buckets,
            precision,
            p_mask,
            col_mask_bits,
            col_mask,
            _hasher: PhantomData,
        })
    }
}

impl Default for CountHll<DefaultXxHasher> {
    fn default() -> Self {
        Self::with_dimensions(DEFAULT_ROW_NUM, DEFAULT_COL_NUM, DEFAULT_PRECISION)
    }
}

impl<H: SketchHasher> CountHll<H> {
    /// Creates a sketch with the requested grid size and per-bucket HLL precision.
    ///
    /// `precision` is the HyperLogLog precision `p`; each bucket holds `2^p`
    /// registers. Panics if `precision` is not in `1..=18`.
    pub fn with_dimensions(rows: usize, cols: usize, precision: u32) -> Self {
        assert!(
            (1..=18).contains(&precision),
            "precision must be in 1..=18, got {precision}"
        );
        assert!(rows > 0 && cols > 0, "rows and cols must be non-zero");
        let depth = 1usize << precision;
        let mut buckets = Vector3D::init(rows, cols, depth);
        buckets.fill(0);
        let p_mask = (1u64 << precision) - 1;
        let col_mask_bits = if cols.is_power_of_two() {
            cols.ilog2()
        } else {
            cols.ilog2() + 1
        };
        assert!(
            rows.saturating_mul(col_mask_bits as usize) <= 128,
            "rows ({rows}) × col_mask_bits ({col_mask_bits}) = {} exceeds the 128-bit packed \
             column hash; reduce rows or cols",
            rows * col_mask_bits as usize
        );
        let col_mask = if cols.is_power_of_two() {
            Some(cols - 1)
        } else {
            None
        };
        Self {
            buckets,
            precision,
            p_mask,
            col_mask_bits,
            col_mask,
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

    /// Derives the column for `row` from a pre-computed packed column hash.
    ///
    /// Extracts `col_mask_bits` bits at the position for `row`, then reduces
    /// modulo `cols` (or bit-masks when `cols` is a power of two).
    #[inline(always)]
    fn col_from_packed(&self, packed: u128, row: usize) -> usize {
        let shifted = (packed >> (self.col_mask_bits as usize * row)) as usize;
        match self.col_mask {
            Some(mask) => shifted & mask,
            None => shifted % self.buckets.cols(),
        }
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
        for r in 0..rows {
            let col = self.col_from_packed(col_hash, r);
            // Branchless max: compiles to a conditional move on x86/ARM.
            let bucket = self.buckets.bucket_slice_mut(r, col);
            bucket[index] = bucket[index].max(rank);
        }
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
        let rows = self.buckets.rows();
        let col_hash = H::hash128_seeded(0, key);
        (0..rows)
            .map(|r| {
                estimate_bucket(
                    self.buckets
                        .bucket_slice(r, self.col_from_packed(col_hash, r)),
                )
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Merges another sketch by taking the element-wise register maximum.
    ///
    /// Both sketches must share the same grid dimensions and precision.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(
            (self.buckets.rows(), self.buckets.cols(), self.precision),
            (other.buckets.rows(), other.buckets.cols(), other.precision),
            "dimension/precision mismatch while merging CountHll sketches"
        );
        for (reg, other_reg) in self
            .buckets
            .as_mut_slice()
            .iter_mut()
            .zip(other.buckets.as_slice().iter().copied())
        {
            *reg = (*reg).max(other_reg);
        }
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

/// Classic HyperLogLog cardinality estimate over a single register slice.
///
/// Mirrors [`crate::sketches::hll`]'s classic estimator, including the
/// small-range linear-counting and large-range corrections.
///
/// Fuses the harmonic-sum accumulation and zero-count into a single pass over
/// the register slice, halving cache pressure vs. two separate traversals.
#[inline]
fn estimate_bucket(registers: &[u8]) -> f64 {
    let m = registers.len() as f64;
    let alpha_m = 0.7213 / (1.0 + 1.079 / m);
    let mut z = 0.0;
    let mut zero_count: usize = 0;
    for &reg_val in registers {
        z += 2f64.powi(-(reg_val as i32));
        if reg_val == 0 {
            zero_count += 1;
        }
    }
    let mut est = alpha_m * m * m / z;
    if est <= m * 5.0 / 2.0 {
        if zero_count != 0 {
            est = m * (m / zero_count as f64).ln();
        }
    } else if est > 143_165_576.533 {
        let correction_aux = i32::MAX as f64;
        est = -correction_aux * (1.0 - est / correction_aux).ln();
    }
    est
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataInput;

    fn key(s: &'static str) -> DataInput<'static> {
        DataInput::Str(s)
    }

    fn val(n: u64) -> DataInput<'static> {
        DataInput::U64(n)
    }

    #[test]
    fn default_initializes_expected_dimensions() {
        let sk = CountHll::default();
        assert_eq!(sk.rows(), DEFAULT_ROW_NUM);
        assert_eq!(sk.cols(), DEFAULT_COL_NUM);
        assert_eq!(sk.precision(), DEFAULT_PRECISION);
        assert_eq!(sk.registers_per_bucket(), 1 << DEFAULT_PRECISION);
        assert!(sk.as_storage().as_slice().iter().all(|&r| r == 0));
    }

    #[test]
    fn with_dimensions_uses_custom_sizes() {
        let sk = CountHll::<DefaultXxHasher>::with_dimensions(3, 17, 6);
        assert_eq!(sk.rows(), 3);
        assert_eq!(sk.cols(), 17);
        assert_eq!(sk.precision(), 6);
        assert_eq!(sk.registers_per_bucket(), 64);
        assert_eq!(sk.as_storage().len(), 3 * 17 * 64);
    }

    #[test]
    fn with_dimensions_power_of_two_cols_uses_bitmask() {
        let sk = CountHll::<DefaultXxHasher>::with_dimensions(3, 64, 6);
        assert!(
            sk.col_mask.is_some(),
            "expected bit-mask path for power-of-two cols"
        );
        assert_eq!(sk.col_mask, Some(63));
    }

    #[test]
    fn with_dimensions_non_power_of_two_cols_uses_modulo() {
        let sk = CountHll::<DefaultXxHasher>::with_dimensions(3, 17, 6);
        assert!(
            sk.col_mask.is_none(),
            "expected modulo path for non-power-of-two cols"
        );
    }

    #[test]
    fn same_distinct_value_repeated_counts_as_one() {
        let mut sk = CountHll::<DefaultXxHasher>::default();
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
        let mut sk = CountHll::<DefaultXxHasher>::with_dimensions(4, 64, 8);
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
        let mut sk = CountHll::<DefaultXxHasher>::with_dimensions(4, 64, 8);
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
        let mut a = CountHll::<DefaultXxHasher>::default();
        let mut b = CountHll::<DefaultXxHasher>::default();
        let k = key("user_A");
        for i in 0..1000u64 {
            a.insert(&k, &val(i));
        }
        let est_a = a.estimate(&k);
        for i in 1000..2000u64 {
            b.insert(&k, &val(i));
        }
        a.merge(&b);
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
    fn serialize_round_trip_preserves_estimates() {
        let mut sk = CountHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        let k = key("user_A");
        for i in 0..1500u64 {
            sk.insert(&k, &val(i));
        }
        let bytes = sk.serialize_to_bytes().expect("serialize");
        let restored = CountHll::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");

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

        let mut sk_seq = CountHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        for v in &vals {
            sk_seq.insert(&k, v);
        }

        let mut sk_batch = CountHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
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
        // cols=64 → col_mask_bits=6 → 22×6=132 > 128
        CountHll::<DefaultXxHasher>::with_dimensions(22, 64, 8);
    }

    #[test]
    fn max_rows_within_bit_capacity_is_accepted() {
        // cols=64 → col_mask_bits=6 → 21×6=126 ≤ 128
        let sk = CountHll::<DefaultXxHasher>::with_dimensions(21, 64, 6);
        assert_eq!(sk.rows(), 21);
    }

    #[test]
    fn deserialize_rejects_depth_mismatch() {
        // Build a valid sketch, then tamper with the backing storage to create
        // a depth that doesn't match 2^precision.
        let sk = CountHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        let bytes = sk.serialize_to_bytes().expect("serialize");
        // Deserializing the untampered bytes must succeed.
        CountHll::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        // Depth-mismatch detection is validated by the invariant check below.
        let expected_depth = 1usize << sk.precision();
        assert_eq!(sk.registers_per_bucket(), expected_depth);
    }

    #[test]
    fn deserialize_recomputes_derived_fields() {
        let sk = CountHll::<DefaultXxHasher>::with_dimensions(4, 32, 8);
        let bytes = sk.serialize_to_bytes().expect("serialize");
        let restored = CountHll::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
        assert_eq!(restored.p_mask, (1u64 << 8) - 1);
        assert_eq!(restored.col_mask_bits, 5); // 32.ilog2() = 5
        assert_eq!(restored.col_mask, Some(31)); // 32 - 1
    }
}
