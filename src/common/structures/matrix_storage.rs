//! Trait bound for matrix-backed sketches.

use smallvec::SmallVec;

use crate::DataInput;

/// Fast-path hash container for matrix-backed sketches.
#[derive(Clone, Debug)]
pub enum MatrixHashType {
    /// Packed per-row hashes stored in one `u64`.
    Packed64(u64),
    /// Packed per-row hashes stored in one `u128`.
    Packed128(u128),
    /// One hash value per row.
    Rows(SmallVec<[u64; 8]>),
}

impl MatrixHashType {
    #[inline(always)]
    /// Extracts the row-local hash bits for one row.
    pub fn row_hash(&self, row: usize, mask_bits: u32, mask: u128) -> u128 {
        match self {
            MatrixHashType::Packed64(value) => {
                let shifted = (*value >> (mask_bits as usize * row)) as u128;
                shifted & mask
            }
            MatrixHashType::Packed128(value) => (value >> (mask_bits as usize * row)) & mask,
            MatrixHashType::Rows(values) => {
                debug_assert!(row < values.len(), "row index out of bounds for hash rows");
                (values[row] as u128) & mask
            }
        }
    }

    #[inline(always)]
    /// Returns the Count-Sketch sign for one row.
    pub fn sign_for_row(&self, row: usize) -> i32 {
        let bit = match self {
            MatrixHashType::Packed64(value) => (value >> (63 - row)) & 1,
            MatrixHashType::Packed128(value) => ((value >> (127 - row)) & 1) as u64,
            MatrixHashType::Rows(values) => {
                debug_assert!(row < values.len(), "row index out of bounds for hash rows");
                (values[row] >> 63) & 1
            }
        };
        (bit as i32 * 2) - 1
    }

    #[inline(always)]
    /// Returns the lower 64 bits of the stored hash.
    pub fn lower_64(&self) -> u64 {
        match self {
            MatrixHashType::Packed64(value) => *value,
            MatrixHashType::Packed128(value) => *value as u64,
            MatrixHashType::Rows(values) => values.first().copied().unwrap_or(0),
        }
    }
}

/// Trait for hash values that support fast row/column decoding.
pub trait MatrixFastHash: Clone {
    /// Verifies that the hash type can encode the given dimensions.
    fn assert_compatible(rows: usize, cols: usize);
    /// Extracts the row-local hash bits before column folding.
    ///
    /// Splitting decode from folding lets callers with cached
    /// `(mask_bits, mask)` pairs (e.g. `Vector2D`) skip recomputing them per
    /// row, and lets callers whose `cols` is a compile-time constant fold the
    /// whole decode away.
    fn row_hash(&self, row: usize, mask_bits: u32, mask: u128) -> u128;
    /// Returns the column index for one row.
    fn col_for_row(&self, row: usize, cols: usize) -> usize;
    /// Returns the Count-Sketch sign for one row.
    fn sign_for_row(&self, row: usize) -> i32;
}

#[inline(always)]
pub(crate) fn cols_mask_bits(cols: usize) -> u32 {
    if cols.is_power_of_two() {
        cols.ilog2()
    } else {
        cols.ilog2() + 1
    }
}

#[inline(always)]
pub(crate) fn cols_mask(cols: usize) -> u128 {
    (1u128 << cols_mask_bits(cols)) - 1
}

/// Shared column-folding logic: extract the row bits, then reduce to
/// `[0, cols)` without a division when `cols` is a power of two.
///
/// When `cols` is a power of two, `mask == cols - 1`, so the masked value is
/// already `< cols` and `% cols` would be an identity — but the compiler
/// cannot prove that from a runtime `cols`, and the division (magic multiply)
/// executed on every row of every insert/query. The branch below compiles to
/// a single AND+compare.
#[inline(always)]
pub(crate) fn fold_to_col(raw: u128, cols: usize) -> usize {
    if cols.is_power_of_two() {
        raw as usize
    } else {
        raw as usize % cols
    }
}

impl MatrixFastHash for MatrixHashType {
    #[inline(always)]
    fn assert_compatible(_rows: usize, _cols: usize) {}

    #[inline(always)]
    fn row_hash(&self, row: usize, mask_bits: u32, mask: u128) -> u128 {
        MatrixHashType::row_hash(self, row, mask_bits, mask)
    }

    #[inline(always)]
    fn col_for_row(&self, row: usize, cols: usize) -> usize {
        let mask_bits = cols_mask_bits(cols);
        let mask = cols_mask(cols);
        fold_to_col(self.row_hash(row, mask_bits, mask), cols)
    }

    #[inline(always)]
    fn sign_for_row(&self, row: usize) -> i32 {
        MatrixHashType::sign_for_row(self, row)
    }
}

impl MatrixFastHash for u64 {
    #[inline(always)]
    fn assert_compatible(rows: usize, cols: usize) {
        let mask_bits = cols_mask_bits(cols) as usize;
        let bits_per_row = mask_bits + 1;
        let bits_required = bits_per_row.saturating_mul(rows);
        assert!(
            bits_required <= 64,
            "SketchHasher hash type u64 cannot represent fast-path hash for rows={rows}, cols={cols}; use u128 or MatrixHashType"
        );
    }

    #[inline(always)]
    fn row_hash(&self, row: usize, mask_bits: u32, mask: u128) -> u128 {
        ((*self >> (mask_bits as usize * row)) as u128) & mask
    }

    #[inline(always)]
    fn col_for_row(&self, row: usize, cols: usize) -> usize {
        let mask_bits = cols_mask_bits(cols);
        let mask = cols_mask(cols);
        fold_to_col(((*self >> (mask_bits as usize * row)) as u128) & mask, cols)
    }

    #[inline(always)]
    fn sign_for_row(&self, row: usize) -> i32 {
        let bit = (self >> (63 - row)) & 1;
        (bit as i32 * 2) - 1
    }
}

impl MatrixFastHash for u128 {
    #[inline(always)]
    fn assert_compatible(rows: usize, cols: usize) {
        let mask_bits = cols_mask_bits(cols) as usize;
        let bits_per_row = mask_bits + 1;
        let bits_required = bits_per_row.saturating_mul(rows);
        assert!(
            bits_required <= 128,
            "SketchHasher hash type u128 cannot represent fast-path hash for rows={rows}, cols={cols}; use MatrixHashType"
        );
    }

    #[inline(always)]
    fn row_hash(&self, row: usize, mask_bits: u32, mask: u128) -> u128 {
        (*self >> (mask_bits as usize * row)) & mask
    }

    #[inline(always)]
    fn col_for_row(&self, row: usize, cols: usize) -> usize {
        let mask_bits = cols_mask_bits(cols);
        let mask = cols_mask(cols);
        fold_to_col((*self >> (mask_bits as usize * row)) & mask, cols)
    }

    #[inline(always)]
    fn sign_for_row(&self, row: usize) -> i32 {
        let bit = (self >> (127 - row)) & 1;
        (bit as i32 * 2) - 1
    }
}

/// Storage interface implemented by matrix-backed sketch backends.
pub trait MatrixStorage {
    /// Counter type stored in each cell.
    type Counter: Clone;
    /// Returns the number of rows.
    fn rows(&self) -> usize;
    /// Returns the number of columns.
    fn cols(&self) -> usize;

    /// Updates a single counter at `(row, col)`.
    fn update_one_counter<F, V>(&mut self, row: usize, col: usize, op: F, value: V)
    where
        F: Fn(&mut Self::Counter, V);

    /// Increments one counter by a typed value.
    fn increment_by_row(&mut self, row: usize, col: usize, value: Self::Counter);

    /// Inserts one value into all rows using a precomputed hash.
    ///
    /// NOTE (perf finding, 2026-08): a row-major batch-transposed variant of
    /// this loop was prototyped and measured ~25-40% SLOWER than key-major
    /// across matrix sizes from 96KB to 128MB on M1 Max — the packed hash
    /// staying register-resident across a key's rows beats any multi-stream
    /// store reordering. Do not revisit without new hardware evidence.
    fn fast_insert<Hash, F, V>(&mut self, op: F, value: V, hashed_val: &Hash)
    where
        Hash: MatrixFastHash,
        F: Fn(&mut Self::Counter, &V, usize),
        V: Clone;

    /// Queries the minimum across rows using a precomputed hash.
    fn fast_query_min<Hash, F, R>(&self, hashed_val: &Hash, op: F) -> R
    where
        Hash: MatrixFastHash,
        F: Fn(&Self::Counter, usize, &Hash) -> R,
        R: PartialOrd;

    /// Queries the median across rows using a precomputed hash.
    fn fast_query_median<Hash, F>(&self, hashed_val: &Hash, op: F) -> f64
    where
        Hash: MatrixFastHash,
        F: Fn(&Self::Counter, usize, &Hash) -> f64;

    /// Reads one counter at `(row, col)`.
    fn query_one_counter(&self, row: usize, col: usize) -> Self::Counter;
}

/// Trait for storages that can derive fast-path hashes for their dimensions.
pub trait FastPathHasher<H>: MatrixStorage
where
    H: crate::SketchHasher,
{
    /// Computes a compatible fast-path hash for `value`.
    fn hash_for_matrix(&self, value: &DataInput) -> H::HashType;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The power-of-two fast path (mask-only decode) must agree exactly with
    /// the reference `shift → mask → % cols` computation for every dimension
    /// class: powers of two (modulo is an identity), odd/non-power sizes
    /// (modulo still applies), and the degenerate cols=1.
    #[test]
    fn col_decode_matches_reference_for_all_dim_classes() {
        let hash64 = 0x0123_4567_89AB_CDEF_u64;
        let hash128 = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF_u128;
        let packed = MatrixHashType::Packed64(hash64);
        let packed_wide = MatrixHashType::Packed128(hash128);

        for &cols in &[1usize, 2, 3, 5, 7, 8, 10, 100, 1000, 2048, 4096, 5003] {
            let mask_bits = if cols.is_power_of_two() {
                cols.ilog2()
            } else {
                cols.ilog2() + 1
            } as usize;
            let mask = (1u128 << mask_bits) - 1;

            // Decode shifts must stay within the hash width — mirror the
            // `assert_compatible` contract rather than testing past it.
            let unit = mask_bits.max(1);
            let rows_64 = (64 / unit).clamp(1, 6);
            let rows_128 = (128 / unit).clamp(1, 6);

            for row in 0..rows_128 {
                let expected = |raw: u128| -> usize { (raw as usize) % cols };

                if row < rows_64 {
                    let got_packed = packed.col_for_row(row, cols);
                    let want_packed = expected(packed.row_hash(row, mask_bits as u32, mask));
                    assert_eq!(got_packed, want_packed, "Packed64 cols={cols} row={row}");

                    assert_eq!(
                        hash64.col_for_row(row, cols),
                        expected(((hash64 >> (mask_bits * row)) as u128) & mask),
                        "u64 cols={cols} row={row}"
                    );
                }

                let got_wide = packed_wide.col_for_row(row, cols);
                let want_wide = expected(packed_wide.row_hash(row, mask_bits as u32, mask));
                assert_eq!(got_wide, want_wide, "Packed128 cols={cols} row={row}");

                assert_eq!(
                    hash128.col_for_row(row, cols),
                    expected((hash128 >> (mask_bits * row)) & mask),
                    "u128 cols={cols} row={row}"
                );
            }
        }
    }
}
