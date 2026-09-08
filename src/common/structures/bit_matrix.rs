//! Packed bit storage for membership sketches.
//!
//! [`BitMatrix`] is a `rows x cols` grid of single bits behind the same
//! [`MatrixStorage`] interface the counter matrices use, so a sketch written
//! against that trait runs over bits without a second code path. One bit per
//! cell rather than one byte is the whole point: a Bloom filter sized for a
//! target false-positive rate is a memory structure first.

use super::matrix_storage::{
    MatrixFastHash, MatrixStorage, cols_mask, cols_mask_bits, fold_to_col,
};
use serde::{Deserialize, Serialize};

const BITS_PER_WORD: usize = u64::BITS as usize;

/// A `rows x cols` bit grid packed into `u64` words, one row after another.
#[derive(Clone, Debug)]
pub struct BitMatrix {
    words: Vec<u64>,
    rows: usize,
    cols: usize,
    words_per_row: usize,
    mask_bits: u32,
    mask: u128,
}

// Only the stored fields travel; `words_per_row`, `mask_bits` and `mask` are
// functions of `cols` and are recomputed on decode.
#[derive(Serialize)]
#[serde(rename = "BitMatrix")]
struct BitMatrixSer<'a> {
    words: &'a [u64],
    rows: usize,
    cols: usize,
}

#[derive(Deserialize)]
#[serde(rename = "BitMatrix")]
struct BitMatrixDe {
    words: Vec<u64>,
    rows: usize,
    cols: usize,
}

impl Serialize for BitMatrix {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        BitMatrixSer {
            words: &self.words,
            rows: self.rows,
            cols: self.cols,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BitMatrix {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let input = BitMatrixDe::deserialize(deserializer)?;
        Self::from_words(input.rows, input.cols, input.words).map_err(serde::de::Error::custom)
    }
}

impl BitMatrix {
    /// The packed words, one row after another, each row padded out to a whole
    /// number of words.
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    /// Builds a grid from packed words, recomputing the word stride and the fold
    /// masks from `cols`. `Err` names the mismatch when the word count does not
    /// describe a `rows x cols` grid.
    pub(crate) fn from_words(rows: usize, cols: usize, words: Vec<u64>) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err(format!(
                "a bit matrix needs both dimensions, got {rows}x{cols}"
            ));
        }
        let words_per_row = cols.div_ceil(BITS_PER_WORD);
        let expected = rows
            .checked_mul(words_per_row)
            .ok_or_else(|| "bit matrix dimensions overflow".to_string())?;
        if words.len() != expected {
            return Err(format!(
                "bit matrix of {rows}x{cols} needs {expected} words, got {}",
                words.len()
            ));
        }
        Ok(Self {
            words,
            rows,
            cols,
            words_per_row,
            mask_bits: cols_mask_bits(cols),
            mask: cols_mask(cols),
        })
    }

    /// Creates a `rows x cols` grid with every bit clear.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "a bit matrix needs both dimensions");
        let words_per_row = cols.div_ceil(BITS_PER_WORD);
        Self {
            words: vec![0; rows * words_per_row],
            rows,
            cols,
            words_per_row,
            mask_bits: cols_mask_bits(cols),
            mask: cols_mask(cols),
        }
    }

    /// Number of rows.
    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes of packed storage the grid occupies.
    pub fn size_in_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }

    /// Number of bits currently set.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Fraction of bits currently set.
    pub fn fill_ratio(&self) -> f64 {
        self.count_ones() as f64 / (self.rows * self.cols) as f64
    }

    /// Clears every bit.
    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }

    /// Sets every bit in `self` that is set in `other`.
    ///
    /// Dimensions must match; a union across different geometries is not
    /// defined.
    pub fn union_from(&mut self, other: &Self) {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "bit matrices must have the same dimensions to be unioned"
        );
        for (dst, src) in self.words.iter_mut().zip(other.words.iter()) {
            *dst |= *src;
        }
    }

    /// Word index and bit mask for `(row, col)`.
    ///
    /// Rows are padded to `words_per_row * 64` bits, so an unchecked column
    /// past `cols` reads a padding bit or the next row instead of panicking.
    /// Every caller goes through the bounds check below.
    #[inline(always)]
    fn locate(&self, row: usize, col: usize) -> (usize, u64) {
        debug_assert!(row < self.rows && col < self.cols);
        let idx = row * self.words_per_row + col / BITS_PER_WORD;
        (idx, 1u64 << (col % BITS_PER_WORD))
    }

    #[inline(always)]
    fn check_bounds(&self, row: usize, col: usize) {
        assert!(
            row < self.rows && col < self.cols,
            "({row}, {col}) is outside a {}x{} bit matrix",
            self.rows,
            self.cols
        );
    }

    /// Reads the bit at `(row, col)`.
    ///
    /// # Panics
    /// If `row >= rows` or `col >= cols`.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> bool {
        self.check_bounds(row, col);
        let (idx, bit) = self.locate(row, col);
        self.words[idx] & bit != 0
    }

    /// Sets the bit at `(row, col)`.
    ///
    /// # Panics
    /// If `row >= rows` or `col >= cols`.
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize) {
        self.check_bounds(row, col);
        let (idx, bit) = self.locate(row, col);
        self.words[idx] |= bit;
    }

    /// Writes the bit at `(row, col)`.
    ///
    /// # Panics
    /// If `row >= rows` or `col >= cols`.
    #[inline(always)]
    pub fn put(&mut self, row: usize, col: usize, value: bool) {
        self.check_bounds(row, col);
        let (idx, bit) = self.locate(row, col);
        if value {
            self.words[idx] |= bit;
        } else {
            self.words[idx] &= !bit;
        }
    }

    #[inline(always)]
    fn col_for_row<Hash: MatrixFastHash>(&self, hashed_val: &Hash, row: usize) -> usize {
        let raw = hashed_val.row_hash(row, self.mask_bits, self.mask);
        fold_to_col(raw, self.cols)
    }
}

impl MatrixStorage for BitMatrix {
    type Counter = bool;

    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn update_one_counter<F, V>(&mut self, row: usize, col: usize, op: F, value: V)
    where
        F: Fn(&mut Self::Counter, V),
    {
        let mut cell = self.get(row, col);
        op(&mut cell, value);
        self.put(row, col, cell);
    }

    #[inline(always)]
    fn increment_by_row(&mut self, row: usize, col: usize, value: Self::Counter) {
        if value {
            self.set(row, col);
        }
    }

    #[inline(always)]
    fn fast_insert<Hash, F, V>(&mut self, op: F, value: V, hashed_val: &Hash)
    where
        Hash: MatrixFastHash,
        F: Fn(&mut Self::Counter, &V, usize),
        V: Clone,
    {
        for row in 0..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let mut cell = self.get(row, col);
            op(&mut cell, &value, row);
            self.put(row, col, cell);
        }
    }

    #[inline(always)]
    fn fast_query_min<Hash, F, R>(&self, hashed_val: &Hash, op: F) -> R
    where
        Hash: MatrixFastHash,
        F: Fn(&Self::Counter, usize, &Hash) -> R,
        R: PartialOrd,
    {
        let c0 = self.col_for_row(hashed_val, 0);
        let mut min = op(&self.get(0, c0), 0, hashed_val);
        for row in 1..self.rows {
            let col = self.col_for_row(hashed_val, row);
            let candidate = op(&self.get(row, col), row, hashed_val);
            if candidate < min {
                min = candidate;
            }
        }
        min
    }

    #[inline(always)]
    fn fast_query_median<Hash, F>(&self, hashed_val: &Hash, op: F) -> f64
    where
        Hash: MatrixFastHash,
        F: Fn(&Self::Counter, usize, &Hash) -> f64,
    {
        let mut values: Vec<f64> = (0..self.rows)
            .map(|row| {
                let col = self.col_for_row(hashed_val, row);
                op(&self.get(row, col), row, hashed_val)
            })
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    #[inline(always)]
    fn query_one_counter(&self, row: usize, col: usize) -> Self::Counter {
        self.get(row, col)
    }
}

impl<H> crate::FastPathHasher<H> for BitMatrix
where
    H: crate::SketchHasher,
{
    #[inline(always)]
    fn hash_for_matrix(&self, value: &crate::DataInput) -> H::HashType {
        <H::HashType as MatrixFastHash>::assert_compatible(self.rows, self.cols);
        H::hash_for_matrix_seeded(0, self.rows, self.cols, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataInput, DefaultXxHasher, FastPathHasher};

    const GEOMETRIES: [(usize, usize); 33] = {
        let cols = [1, 2, 7, 63, 64, 65, 100, 127, 128, 129, 255];
        let mut out = [(0, 0); 33];
        let mut i = 0;
        while i < cols.len() {
            out[i * 3] = (1, cols[i]);
            out[i * 3 + 1] = (2, cols[i]);
            out[i * 3 + 2] = (5, cols[i]);
            i += 1;
        }
        out
    };

    /// Every cell owns one bit and no other cell's. Setting the grid one cell
    /// at a time and watching the population rise by exactly one each time
    /// proves both halves at once: a cell that aliased another would not raise
    /// the count, and a cell that spilled into row padding would not read back.
    #[test]
    fn every_cell_owns_exactly_one_bit() {
        for (rows, cols) in GEOMETRIES {
            let mut m = BitMatrix::new(rows, cols);
            let mut set_so_far = 0;
            for row in 0..rows {
                for col in 0..cols {
                    assert!(!m.get(row, col), "{rows}x{cols}: ({row}, {col}) preset");
                    m.set(row, col);
                    set_so_far += 1;
                    assert!(m.get(row, col), "{rows}x{cols}: ({row}, {col}) lost");
                    assert_eq!(
                        m.count_ones(),
                        set_so_far,
                        "{rows}x{cols}: setting ({row}, {col}) moved another bit"
                    );
                }
            }
            assert_eq!(m.count_ones(), rows * cols);
            assert_eq!(m.fill_ratio(), 1.0, "{rows}x{cols} fill after filling");
        }
    }

    /// Rows are padded out to whole words, and the padding never counts.
    #[test]
    fn row_padding_is_never_reachable_or_counted() {
        for (rows, cols) in GEOMETRIES {
            let mut m = BitMatrix::new(rows, cols);
            for row in 0..rows {
                for col in 0..cols {
                    m.set(row, col);
                }
            }
            let words_per_row = cols.div_ceil(BITS_PER_WORD);
            assert_eq!(m.size_in_bytes(), rows * words_per_row * 8);
            // Storage holds `rows * words_per_row * 64` bits; only `rows * cols`
            // of them are addressable, so a full grid leaves the rest clear.
            assert_eq!(m.count_ones(), rows * cols);
        }
    }

    #[test]
    fn put_and_clear_reset_individual_bits() {
        for (rows, cols) in GEOMETRIES {
            let mut m = BitMatrix::new(rows, cols);
            for row in 0..rows {
                for col in 0..cols {
                    m.put(row, col, true);
                }
            }
            for row in 0..rows {
                for col in 0..cols {
                    m.put(row, col, false);
                    assert!(!m.get(row, col));
                }
            }
            assert_eq!(m.count_ones(), 0, "{rows}x{cols} after clearing by put");
            assert_eq!(m.fill_ratio(), 0.0);

            m.set(rows - 1, cols - 1);
            m.clear();
            assert_eq!(m.count_ones(), 0, "{rows}x{cols} after clear");
        }
    }

    /// A column past `cols` lands in row padding or in the next row, so it
    /// cannot be left to the slice index to catch.
    #[test]
    #[should_panic(expected = "(0, 100) is outside a 3x100 bit matrix")]
    fn a_column_past_the_last_one_panics_rather_than_aliasing() {
        let mut m = BitMatrix::new(3, 100);
        m.set(0, 100);
    }

    #[test]
    #[should_panic(expected = "(0, 128) is outside a 3x100 bit matrix")]
    fn a_column_reaching_the_next_row_panics() {
        let mut m = BitMatrix::new(3, 100);
        m.set(0, 128);
    }

    #[test]
    #[should_panic(expected = "(3, 0) is outside a 3x100 bit matrix")]
    fn a_row_past_the_last_one_panics() {
        let m = BitMatrix::new(3, 100);
        let _ = m.get(3, 0);
    }

    #[test]
    #[should_panic(expected = "(0, 120) is outside a 2x100 bit matrix")]
    fn put_checks_bounds_too() {
        let mut m = BitMatrix::new(2, 100);
        m.put(0, 120, true);
    }

    #[test]
    fn union_takes_the_bitwise_or() {
        let mut left = BitMatrix::new(3, 100);
        let mut right = BitMatrix::new(3, 100);
        left.set(0, 5);
        left.set(1, 99);
        right.set(1, 99);
        right.set(2, 0);
        left.union_from(&right);
        assert_eq!(left.count_ones(), 3);
        assert!(left.get(0, 5) && left.get(1, 99) && left.get(2, 0));
    }

    #[test]
    #[should_panic(expected = "bit matrices must have the same dimensions")]
    fn union_across_geometries_panics() {
        let mut left = BitMatrix::new(3, 100);
        let right = BitMatrix::new(3, 101);
        left.union_from(&right);
    }

    #[test]
    #[should_panic(expected = "a bit matrix needs both dimensions")]
    fn a_zero_dimension_is_rejected() {
        let _ = BitMatrix::new(0, 64);
    }

    /// Only `words`, `rows` and `cols` travel; the word stride and the fold
    /// masks are recomputed, so a decoded matrix folds hashes exactly as the
    /// original did.
    #[test]
    fn a_round_trip_recomputes_the_derived_fields() {
        for (rows, cols) in GEOMETRIES {
            let mut m = BitMatrix::new(rows, cols);
            for row in 0..rows {
                m.set(row, (row * 7) % cols);
            }
            let bytes = rmp_serde::to_vec(&m).expect("encode");
            let decoded: BitMatrix = rmp_serde::from_slice(&bytes).expect("decode");

            assert_eq!((decoded.rows(), decoded.cols()), (rows, cols));
            for row in 0..rows {
                for col in 0..cols {
                    assert_eq!(decoded.get(row, col), m.get(row, col), "{rows}x{cols}");
                }
            }

            let key = DataInput::U64(0xfeed);
            let hashed =
                <BitMatrix as FastPathHasher<DefaultXxHasher>>::hash_for_matrix(&decoded, &key);
            let mut original = m.clone();
            let mut copy = decoded;
            original.fast_insert(|c, _, _| *c = true, (), &hashed);
            copy.fast_insert(|c, _, _| *c = true, (), &hashed);
            assert_eq!(
                original.count_ones(),
                copy.count_ones(),
                "{rows}x{cols}: decoded matrix folds a hash differently"
            );
            for row in 0..rows {
                for col in 0..cols {
                    assert_eq!(copy.get(row, col), original.get(row, col));
                }
            }
        }
    }

    /// A grid whose row windows all fit in one 64-bit hash decodes each row
    /// from its own slice of that hash. Rows sharing one window would still set
    /// one bit per row, so only the columns tell them apart: over a stream of
    /// keys, two rows holding identical bits are reading the same window.
    #[test]
    fn a_packed_64_grid_decodes_a_distinct_window_per_row() {
        let (rows, cols) = (5, 1024);
        assert_eq!(
            crate::hash_mode_for_matrix(rows, cols),
            crate::MatrixHashMode::Packed64
        );
        let mut m = BitMatrix::new(rows, cols);
        for key in 0..300u64 {
            let hashed = <BitMatrix as FastPathHasher<DefaultXxHasher>>::hash_for_matrix(
                &m,
                &DataInput::U64(key),
            );
            m.fast_insert(|c, _, _| *c = true, (), &hashed);
        }
        for left in 0..rows {
            for right in left + 1..rows {
                assert!(
                    (0..cols).any(|col| m.get(left, col) != m.get(right, col)),
                    "rows {left} and {right} read the same window"
                );
            }
        }
    }

    #[derive(serde::Serialize)]
    #[serde(rename = "BitMatrix")]
    struct CraftedMatrix {
        words: Vec<u64>,
        rows: usize,
        cols: usize,
    }

    /// A payload whose word count disagrees with its dimensions fails at
    /// decode rather than later, as an index panic or a silent alias.
    #[test]
    fn a_payload_that_does_not_fit_its_dimensions_is_rejected() {
        let short = CraftedMatrix {
            words: vec![0; 2],
            rows: 3,
            cols: 100,
        };
        let bytes = rmp_serde::to_vec(&short).expect("encode");
        let err = rmp_serde::from_slice::<BitMatrix>(&bytes).expect_err("short payload accepted");
        assert!(
            err.to_string().contains("needs 6 words, got 2"),
            "unexpected error: {err}"
        );

        let long = CraftedMatrix {
            words: vec![0; 9],
            rows: 3,
            cols: 100,
        };
        let bytes = rmp_serde::to_vec(&long).expect("encode");
        assert!(rmp_serde::from_slice::<BitMatrix>(&bytes).is_err());

        let empty = CraftedMatrix {
            words: vec![],
            rows: 0,
            cols: 100,
        };
        let bytes = rmp_serde::to_vec(&empty).expect("encode");
        let err = rmp_serde::from_slice::<BitMatrix>(&bytes).expect_err("zero rows accepted");
        assert!(
            err.to_string().contains("needs both dimensions"),
            "unexpected error: {err}"
        );
    }

    /// Dimensions whose word count does not fit a `usize` are named as an
    /// overflow. Wrapping the product to zero would make an empty word list the
    /// right length for a grid of `usize::MAX` rows.
    #[test]
    fn dimensions_whose_word_count_overflows_are_rejected() {
        let overflowing = CraftedMatrix {
            words: vec![],
            rows: usize::MAX,
            cols: 128,
        };
        let bytes = rmp_serde::to_vec(&overflowing).expect("encode");
        let err = rmp_serde::from_slice::<BitMatrix>(&bytes).expect_err("overflow accepted");
        assert!(
            err.to_string().contains("dimensions overflow"),
            "unexpected error: {err}"
        );
    }

    /// The wire carries the three stored fields and nothing derived from them.
    #[test]
    fn the_wire_form_carries_only_the_stored_fields() {
        let m = BitMatrix::new(3, 100);
        let stored = CraftedMatrix {
            words: vec![0; 6],
            rows: 3,
            cols: 100,
        };
        assert_eq!(
            rmp_serde::to_vec(&m).expect("encode"),
            rmp_serde::to_vec(&stored).expect("encode")
        );
    }
}
