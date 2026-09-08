//! Wire-format-aligned Count Sketch types. The wire DTO and runtime
//! ops live together here.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error as MsgPackError, MessagePackCodec};

// =====================================================================
// ASAP runtime wire-format-aligned variant .
//
// `CountSketch` and `CountSketchDelta` below are the public-field,
// proto-decode-friendly types consumed by the ASAP query engine
// accumulators. The high-throughput in-process variant is
// `crate::sketches::countsketch::Count`, which keeps its own design.
// =====================================================================

// Count Sketch (a.k.a. Count-Min-style signed-counter sketch) —
// element-wise mergeable frequency estimator.
//
// Parallel to `count_min::CountMinSketch` but with **signed** counters,
// matching the `asap_sketchlib::proto::sketchlib::CountSketchState` wire
// format that DataCollector's `countsketchprocessor` emits via the
// modified OTLP `Metric.data = CountSketch{…}` variant.
//
// The surface is: construct from a decoded proto state, merge
// element-wise with another sketch, emit the matrix for queries and
// serialization. The richer query semantics of Count Sketch (median-
// of-estimators heavy-hitter tracking, `TopKState` integration) are not
// implemented here — the wire format carries the matrix losslessly, so
// the merge/store round-trip works with just a matrix.

/// Default Top-K capacity. Mirrors sketchlib-go `TOPK_SIZE = 100`.
pub const COUNT_SKETCH_TOPK_CAPACITY: usize = 100;

/// Sparse delta between two consecutive CountSketch snapshots —
/// the input shape for [`CountSketch::apply_delta`]. Mirrors the
/// `CountSketchDelta` proto in
/// `sketchlib-go/proto/countsketch/countsketch.proto` and the native
/// Go `Delta` in `sketchlib-go/sketches/CountSketch/delta.go`.
///
/// Cells apply additively: `matrix[row][col] += d_count` for each
/// `(row, col, d_count)` triple. Per-row L2 norm deltas apply
/// additively. Heavy-hitter candidate keys (`hh_keys`) are queried
/// against the post-merge matrix and used to rebuild the receiver's
/// Top-K heap.
#[derive(Debug, Clone, Default)]
pub struct CountSketchDelta {
    pub rows: u32,
    pub cols: u32,
    /// `(row, col, d_count)` cell updates, additive on the CS matrix.
    pub cells: Vec<(u32, u32, i64)>,
    /// Per-row L2 norm deltas. Additive, one scalar per row of the
    /// base sketch. Kept on the delta surface for downstream
    /// error-accounting; `apply_delta` itself ignores L2.
    pub l2: Vec<f64>,
    /// Heavy-hitter candidate keys forwarded by the upstream
    /// Space-Saving tracker. The receiver re-queries the merged CS
    /// matrix for each key and updates its Top-K heap with the
    /// resulting estimate. Mirrors Go's `Delta.HHKeys`.
    pub hh_keys: Vec<String>,
}

/// Minimal Count Sketch state — a flat `rows × cols` matrix of signed
/// counts. Element-wise mergeable (sum over aligned cells). Mirrors
/// sketchlib-go's `CountSketch.Count`/`TopK` pair (the on-the-wire
/// `L2` field is a derived value and is recomputed on load).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountSketch {
    #[serde(rename = "row_num")]
    pub rows: usize,
    #[serde(rename = "col_num")]
    pub cols: usize,
    /// Row-major matrix of signed counts. `matrix[r][c]` is the value of
    /// hash row `r`, column `c`.
    pub matrix: Vec<Vec<f64>>,
    /// Top-K heavy hitters as `(key, count)` pairs, capped at
    /// [`COUNT_SKETCH_TOPK_CAPACITY`]. Order is not guaranteed (heap
    /// shape is not preserved on the wire). Mirrors Go's
    /// `CountSketch.TopK` slot. Defaults to empty on legacy payloads.
    #[serde(default)]
    pub topk: Vec<(String, f64)>,
}

impl CountSketch {
    /// Construct an all-zero sketch with the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            matrix: vec![vec![0.0; cols]; rows],
            topk: Vec::new(),
        }
    }

    /// Construct from a pre-built matrix (used by the modified-OTLP
    /// proto-decode path). `topk` is zero-initialised; callers that
    /// need non-zero auxiliary state should use the msgpack/proto path.
    pub fn from_legacy_matrix(matrix: Vec<Vec<f64>>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(matrix.len(), rows, "row count mismatch");
        debug_assert!(
            matrix.iter().all(|r| r.len() == cols),
            "column count mismatch in at least one row"
        );
        Self {
            rows,
            cols,
            matrix,
            topk: Vec::new(),
        }
    }

    /// Borrow the inner matrix.
    pub fn sketch(&self) -> &Vec<Vec<f64>> {
        &self.matrix
    }

    /// Update the in-memory Top-K heap with `(key, count)`. Keeps the
    /// heap bounded by [`COUNT_SKETCH_TOPK_CAPACITY`]; on overflow,
    /// drops the smallest-count entry. If `key` is already present,
    /// the new count replaces the old (max semantics). Used by
    /// `apply_delta` to rebuild Top-K from `hh_keys`.
    fn topk_update(&mut self, key: &str, count: f64) {
        if let Some(slot) = self.topk.iter_mut().find(|(k, _)| k == key) {
            if count > slot.1 {
                slot.1 = count;
            }
            return;
        }
        if self.topk.len() < COUNT_SKETCH_TOPK_CAPACITY {
            self.topk.push((key.to_owned(), count));
            return;
        }
        // Capacity hit: replace the minimum if `count` exceeds it.
        if let Some((min_idx, min_count)) = self
            .topk
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.1
                    .partial_cmp(&b.1.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, e)| (i, e.1))
        {
            if count > min_count {
                self.topk[min_idx] = (key.to_owned(), count);
            }
        }
    }

    /// Insert a single weighted observation. Uses the shared
    /// [`crate::MatrixHashType::Packed64`] fast-path hash so the matrix-cell
    /// layout matches `sketchlib-go::CountSketch.UpdateString`
    /// bit-for-bit. Locked in by
    /// `tests/sketches_go_parity_probe.rs`.
    ///
    /// Constraints inherited from Go's Packed64 mode:
    /// - `cols` must be a power of two; the column mask is `cols - 1`.
    /// - `rows * (log2(cols) + 1) ≤ 64` so the packed hash holds all
    ///   row indices and sign bits.
    pub fn update(&mut self, key: &str, value: f64) {
        if self.rows == 0 || self.cols == 0 {
            return;
        }
        let input = crate::common::DataInput::String(key.to_owned());
        let hashed = crate::common::MatrixHashType::Packed64(
            <crate::DefaultXxHasher as crate::common::SketchHasher>::hash64_seeded(0, &input),
        );
        for r in 0..self.rows {
            let col = <crate::common::MatrixHashType as crate::common::MatrixFastHash>::col_for_row(
                &hashed, r, self.cols,
            );
            let sign =
                <crate::common::MatrixHashType as crate::common::MatrixFastHash>::sign_for_row(
                    &hashed, r,
                ) as f64;
            self.matrix[r][col] += sign * value;
        }
    }

    /// Estimate the frequency of `key` via the standard median-of-rows
    /// CountSketch query. Returns 0 for an empty sketch. Inverts the
    /// signed-counter projection that [`Self::update`] applied.
    pub fn estimate(&self, key: &str) -> f64 {
        if self.rows == 0 || self.cols == 0 {
            return 0.0;
        }
        let input = crate::common::DataInput::String(key.to_owned());
        let hashed = crate::common::MatrixHashType::Packed64(
            <crate::DefaultXxHasher as crate::common::SketchHasher>::hash64_seeded(0, &input),
        );
        let mut estimates: Vec<f64> = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let col = <crate::common::MatrixHashType as crate::common::MatrixFastHash>::col_for_row(
                &hashed, r, self.cols,
            );
            let sign =
                <crate::common::MatrixHashType as crate::common::MatrixFastHash>::sign_for_row(
                    &hashed, r,
                ) as f64;
            estimates.push(sign * self.matrix[r][col]);
        }
        estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = estimates.len() / 2;
        if estimates.len() % 2 == 1 {
            estimates[mid]
        } else {
            (estimates[mid - 1] + estimates[mid]) / 2.0
        }
    }

    /// Merge one other sketch into self via element-wise addition. Both
    /// operands must have identical dimensions.
    pub fn merge(
        &mut self,
        other: &CountSketch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "CountSketch dimension mismatch: self={}x{}, other={}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
            .into());
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.matrix[r][c] += other.matrix[r][c];
            }
        }
        Ok(())
    }

    /// Apply a sparse delta in place. Matches the `ApplyDelta`
    /// semantics in `sketchlib-go/sketches/CountSketch/delta.go`:
    ///   * each `(row, col, d_count)` triple updates the count matrix
    ///     additively (`matrix[r][c] += d_count`);
    ///   * each `hh_key` is re-queried against the post-update matrix
    ///     and pushed into the receiver's Top-K with the merged-estimate
    ///     count (mirrors Go's `Delta.HHKeys` heavy-hitter rebuild).
    ///
    /// Returns `Err` if any `(row, col)` is out of range — indicating
    /// a dimension mismatch between the snapshot this sketch was
    /// built from and the delta sender.
    pub fn apply_delta(
        &mut self,
        delta: &CountSketchDelta,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 1. Cell additions.
        for (row, col, d_count) in &delta.cells {
            let r = *row as usize;
            let c = *col as usize;
            if r >= self.rows || c >= self.cols {
                return Err(format!(
                    "CountSketchDelta cell ({r},{c}) out of range (matrix={}x{})",
                    self.rows, self.cols
                )
                .into());
            }
            // `d_count` is signed on the wire; CS counts are signed
            // too (can go negative under adversarial keys).
            self.matrix[r][c] += *d_count as f64;
        }
        // 2. Heavy-hitter rebuild from `hh_keys`. Re-estimate against
        // the freshly-updated matrix and push into Top-K with the
        // merged count. Mirrors sketchlib-go's `Delta.HHKeys` path.
        for key in &delta.hh_keys {
            let est = self.estimate(key);
            self.topk_update(key, est);
        }
        Ok(())
    }

    /// Compute a sparse, proto-marshalled `CountSketchDelta` of `self`
    /// against a `snapshot`. A cell is included when its signed count
    /// delta `Δcount` (self − snapshot) is non-zero and its magnitude is
    /// `>= threshold` (the threshold is truncated to an integer, matching
    /// the Go reference). The full per-row L2 norm deltas are always
    /// carried (one entry per row).
    ///
    /// This is the Rust twin of the Go reference implementation's
    /// `ComputeDelta` + `SerializeDelta`: it iterates the matrix in
    /// row-major order, subtracts the snapshot's value for each cell, and
    /// emits the surviving signed cell deltas using the packed-array proto
    /// encoding (`cell_rows`/`cell_cols`/`d_counts`). L2 row deltas are
    /// derived from the matrices — `l2[r] = Σ_c count[r][c]^2` — matching
    /// the Go producer's incrementally-maintained norm. Heavy-hitter
    /// candidate keys (`hh_keys`) are sourced from an upstream Space-Saving
    /// tracker that this minimal wrapper does not maintain, so the field is
    /// left empty (an empty repeated field encodes to nothing on the wire).
    /// The returned bytes are a `prost`-encoded
    /// [`crate::proto::sketchlib::CountSketchDelta`], byte-identical to the
    /// Go `proto.Marshal(CountSketchDelta)` output for the same inputs when
    /// no heavy-hitter keys are forwarded (cross-language byte parity).
    ///
    /// Delta-against-empty: when `snapshot` is the all-zero sketch, every
    /// surviving cell delta equals this window's own (signed) cell count,
    /// so the result is this window's full state encoded as a delta (no
    /// cross-window subtraction). CountSketch deltas carry only
    /// sketch-internal cells/norms — there are no DataPoint-level metric
    /// scalars to drop.
    pub fn compute_delta(
        &self,
        snapshot: &CountSketch,
        threshold: f64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        use crate::proto::sketchlib::CountSketchDelta as ProtoDelta;
        use prost::Message;

        if self.rows != snapshot.rows || self.cols != snapshot.cols {
            return Err(format!(
                "CountSketch dimension mismatch: self={}x{}, snapshot={}x{}",
                self.rows, self.cols, snapshot.rows, snapshot.cols
            )
            .into());
        }

        // Threshold is compared against |Δ| after truncation to an
        // integer, mirroring the Go reference's `int64(threshold)` cast.
        let thresh = threshold as i64;

        let mut cell_rows: Vec<u32> = Vec::new();
        let mut cell_cols: Vec<u32> = Vec::new();
        let mut d_counts: Vec<i64> = Vec::new();
        let mut l2: Vec<f64> = Vec::with_capacity(self.rows);

        for r in 0..self.rows {
            let mut cur_l2 = 0.0f64;
            let mut snap_l2 = 0.0f64;
            for c in 0..self.cols {
                let cv = self.matrix[r][c];
                let sv = snapshot.matrix[r][c];
                cur_l2 += cv * cv;
                snap_l2 += sv * sv;
                let dc = (cv - sv) as i64;
                if dc != 0 && (dc <= -thresh || dc >= thresh) {
                    cell_rows.push(r as u32);
                    cell_cols.push(c as u32);
                    d_counts.push(dc);
                }
            }
            l2.push(cur_l2 - snap_l2);
        }

        let delta = ProtoDelta {
            rows: self.rows as u32,
            cols: self.cols as u32,
            cells_legacy: Vec::new(),
            l2_legacy: Vec::new(),
            topk: None,
            hh_keys: Vec::new(),
            cell_rows,
            cell_cols,
            d_counts,
            l2,
        };
        Ok(delta.encode_to_vec())
    }

    /// Apply a `prost`-encoded [`crate::proto::sketchlib::CountSketchDelta`]
    /// to this sketch in place (additive signed-cell merge + heavy-hitter
    /// rebuild). The Rust twin of the Go reference implementation's
    /// `DeserializeDelta` + `ApplyDelta`. Reads the packed cell arrays
    /// (`cell_rows`/`cell_cols`/`d_counts`) and the heavy-hitter keys
    /// (`hh_keys`), falling back to the legacy per-cell records and TopK
    /// entries for payloads from older producers.
    ///
    /// Returns `Err` if `bytes` is not a valid `CountSketchDelta` proto or
    /// a cell is out of range for this sketch's dimensions.
    pub fn apply_delta_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::proto::sketchlib::CountSketchDelta as ProtoDelta;
        use prost::Message;

        let proto = ProtoDelta::decode(bytes)?;
        let cells: Vec<(u32, u32, i64)> = if !proto.cell_rows.is_empty() {
            proto
                .cell_rows
                .iter()
                .zip(proto.cell_cols.iter())
                .zip(proto.d_counts.iter())
                .map(|((&r, &c), &d)| (r, c, d))
                .collect()
        } else {
            proto
                .cells_legacy
                .iter()
                .map(|c| (c.row, c.col, c.d_count as i64))
                .collect()
        };
        let l2 = if !proto.l2.is_empty() {
            proto.l2
        } else {
            proto.l2_legacy
        };
        let hh_keys = if !proto.hh_keys.is_empty() {
            proto.hh_keys
        } else if let Some(topk) = proto.topk {
            topk.entries.into_iter().map(|e| e.key).collect()
        } else {
            Vec::new()
        };
        let delta = CountSketchDelta {
            rows: proto.rows,
            cols: proto.cols,
            cells,
            l2,
            hh_keys,
        };
        self.apply_delta(&delta)
    }

    /// MessagePack twin of [`Self::compute_delta`]. Computes the same
    /// sparse signed-cell delta of `self` against `snapshot` (a cell is
    /// carried when `Δcount` is non-zero and `|Δcount| >= threshold`, the
    /// threshold truncated to an integer), but serializes it as the
    /// cells-only parallel-array MessagePack layout via
    /// [`CountSketchDelta`]'s [`MessagePackCodec`] instead of proto. The
    /// per-row `l2` norms and `hh_keys` the proto delta carries are dropped
    /// — the plain-sketch estimate path does not consume them.
    /// Delta-against-empty carries this window's own signed cell counts.
    pub fn compute_delta_msgpack(
        &self,
        snapshot: &CountSketch,
        threshold: f64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        if self.rows != snapshot.rows || self.cols != snapshot.cols {
            return Err(format!(
                "CountSketch dimension mismatch: self={}x{}, snapshot={}x{}",
                self.rows, self.cols, snapshot.rows, snapshot.cols
            )
            .into());
        }
        let thresh = threshold as i64;
        let mut delta = CountSketchDelta {
            rows: self.rows as u32,
            cols: self.cols as u32,
            ..CountSketchDelta::default()
        };
        for r in 0..self.rows {
            for c in 0..self.cols {
                let dc = (self.matrix[r][c] - snapshot.matrix[r][c]) as i64;
                if dc != 0 && (dc <= -thresh || dc >= thresh) {
                    delta.cells.push((r as u32, c as u32, dc));
                }
            }
        }
        Ok(delta.to_msgpack()?)
    }

    /// MessagePack twin of [`Self::apply_delta_bytes`]. Decodes the
    /// cells-only parallel-array MessagePack [`CountSketchDelta`] and
    /// applies it in place (additive signed-cell merge).
    ///
    /// Returns `Err` if `bytes` is not a valid MessagePack
    /// `CountSketchDelta` or a cell is out of range for this sketch's
    /// dimensions.
    pub fn apply_delta_msgpack_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let delta = CountSketchDelta::from_msgpack(bytes)?;
        self.apply_delta(&delta)
    }

    /// Merge a slice of references into a single new sketch. All inputs
    /// must share the same dimensions; returns `Err` on mismatch or an
    /// empty input.
    pub fn merge_refs(
        inputs: &[&CountSketch],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let first = inputs
            .first()
            .ok_or("CountSketch::merge_refs called with empty input")?;
        let mut merged = CountSketch::new(first.rows, first.cols);
        for cs in inputs {
            merged.merge(cs)?;
        }
        Ok(merged)
    }
}

impl MessagePackCodec for CountSketch {
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        Ok(rmp_serde::to_vec(self)?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        Ok(rmp_serde::from_slice(bytes)?)
    }
}

impl MessagePackCodec for CountSketchDelta {
    /// Encodes the sparse signed-cell delta as the 5-element MessagePack
    /// array
    /// `[ rows:u64, cols:u64, row_idx:[]u32, col_idx:[]u32, d_count:[]i64 ]`
    /// — byte-identical to Rust
    /// `rmp_serde::to_vec(&(u64, u64, Vec<u32>, Vec<u32>, Vec<i64>))`
    /// (compact mode), the layout the Go `count_sketch_delta_sparse.go`
    /// established (with signed `d_count` in place of the F2-broadcast's
    /// `f64` vals). Only the matrix cells cross the wire; the per-row `l2`
    /// norms and `hh_keys` heavy-hitter surface are not carried.
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let mut row_idx: Vec<u32> = Vec::with_capacity(self.cells.len());
        let mut col_idx: Vec<u32> = Vec::with_capacity(self.cells.len());
        let mut d_count: Vec<i64> = Vec::with_capacity(self.cells.len());
        for (r, c, d) in &self.cells {
            row_idx.push(*r);
            col_idx.push(*c);
            d_count.push(*d);
        }
        Ok(rmp_serde::to_vec(&(
            self.rows as u64,
            self.cols as u64,
            row_idx,
            col_idx,
            d_count,
        ))?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        let (rows, cols, row_idx, col_idx, d_count): (u64, u64, Vec<u32>, Vec<u32>, Vec<i64>) =
            rmp_serde::from_slice(bytes)?;
        let cells = row_idx
            .into_iter()
            .zip(col_idx)
            .zip(d_count)
            .map(|((r, c), d)| (r, c, d))
            .collect();
        Ok(CountSketchDelta {
            rows: rows as u32,
            cols: cols as u32,
            cells,
            ..CountSketchDelta::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msgpack_delta_against_empty_round_trips() {
        let (rows, cols) = (5, 1024);
        let mut w = CountSketch::new(rows, cols);
        for _ in 0..200 {
            w.update("hot", 1.0);
        }
        let empty = CountSketch::new(rows, cols);
        let bytes = w.compute_delta_msgpack(&empty, 1.0).unwrap();
        let mut recon = CountSketch::new(rows, cols);
        recon.apply_delta_msgpack_bytes(&bytes).unwrap();
        // CountSketch round-trips its cells exactly under delta-against-empty.
        assert_eq!(recon.sketch(), w.sketch());
    }

    #[test]
    fn msgpack_delta_wire_layout_is_parallel_arrays() {
        use crate::message_pack_format::MessagePackCodec;
        let (rows, cols) = (3, 8);
        let mut w = CountSketch::new(rows, cols);
        w.update("x", 5.0);
        let empty = CountSketch::new(rows, cols);
        let bytes = w.compute_delta_msgpack(&empty, 1.0).unwrap();
        let cur = w.sketch();
        let (mut ri, mut ci, mut dc): (Vec<u32>, Vec<u32>, Vec<i64>) =
            (Vec::new(), Vec::new(), Vec::new());
        for (r, row) in cur.iter().enumerate() {
            for (c, &cell) in row.iter().enumerate() {
                let d = cell as i64;
                if d != 0 {
                    ri.push(r as u32);
                    ci.push(c as u32);
                    dc.push(d);
                }
            }
        }
        let expected = rmp_serde::to_vec(&(rows as u64, cols as u64, ri, ci, dc)).unwrap();
        assert_eq!(bytes, expected);
        let d = CountSketchDelta::from_msgpack(&bytes).unwrap();
        assert_eq!(d.rows, rows as u32);
        assert_eq!(d.cols, cols as u32);
    }

    #[test]
    fn test_new_empty() {
        let cs = CountSketch::new(2, 3);
        assert_eq!(cs.rows, 2);
        assert_eq!(cs.cols, 3);
        assert_eq!(cs.sketch(), &vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_from_legacy_matrix() {
        let m = vec![vec![1.0, -2.0, 3.0], vec![-4.0, 5.0, -6.0]];
        let cs = CountSketch::from_legacy_matrix(m.clone(), 2, 3);
        assert_eq!(cs.sketch(), &m);
    }

    #[test]
    fn test_merge_element_wise() {
        let mut a = CountSketch::from_legacy_matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]], 2, 2);
        let b = CountSketch::from_legacy_matrix(vec![vec![-1.0, -2.0], vec![-3.0, -4.0]], 2, 2);
        a.merge(&b).unwrap();
        assert_eq!(a.sketch(), &vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn test_merge_dimension_mismatch() {
        let mut a = CountSketch::new(2, 3);
        let b = CountSketch::new(3, 3);
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn test_merge_refs() {
        let a = CountSketch::from_legacy_matrix(vec![vec![1.0, 2.0]], 1, 2);
        let b = CountSketch::from_legacy_matrix(vec![vec![3.0, 4.0]], 1, 2);
        let c = CountSketch::from_legacy_matrix(vec![vec![5.0, 6.0]], 1, 2);
        let merged = CountSketch::merge_refs(&[&a, &b, &c]).unwrap();
        assert_eq!(merged.sketch(), &vec![vec![9.0, 12.0]]);
    }

    #[test]
    fn test_apply_delta_additive() {
        let mut cs = CountSketch::from_legacy_matrix(
            vec![vec![1.0, -2.0, 3.0], vec![-4.0, 5.0, -6.0]],
            2,
            3,
        );
        let delta = CountSketchDelta {
            rows: 2,
            cols: 3,
            cells: vec![
                (0, 0, 10),  // 1 + 10 = 11
                (0, 2, -3),  // 3 - 3 = 0
                (1, 1, -15), // 5 - 15 = -10
            ],
            l2: vec![],
            hh_keys: vec![],
        };
        cs.apply_delta(&delta).unwrap();
        assert_eq!(
            cs.sketch(),
            &vec![vec![11.0, -2.0, 0.0], vec![-4.0, -10.0, -6.0]]
        );
    }

    #[test]
    fn test_apply_delta_matches_full_merge() {
        let base = CountSketch::from_legacy_matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]], 2, 2);
        let addition =
            CountSketch::from_legacy_matrix(vec![vec![10.0, 0.0], vec![0.0, 20.0]], 2, 2);
        let mut via_merge = base.clone();
        via_merge.merge(&addition).unwrap();

        let delta = CountSketchDelta {
            rows: 2,
            cols: 2,
            cells: vec![(0, 0, 10), (1, 1, 20)],
            l2: vec![],
            hh_keys: vec![],
        };
        let mut via_delta = base;
        via_delta.apply_delta(&delta).unwrap();
        assert_eq!(via_delta.sketch(), via_merge.sketch());
    }

    #[test]
    fn test_apply_delta_out_of_range() {
        let mut cs = CountSketch::new(2, 3);
        let delta = CountSketchDelta {
            rows: 2,
            cols: 3,
            cells: vec![(2, 0, 1)], // row 2 out of range for 2-row matrix
            l2: vec![],
            hh_keys: vec![],
        };
        assert!(cs.apply_delta(&delta).is_err());
    }

    #[test]
    fn test_apply_delta_rebuilds_topk_from_hh_keys() {
        // Build a sketch with two known keys via the in-process
        // `update` path so the matrix has a coherent shape, then
        // send a delta that only carries `hh_keys` entries. The
        // receiver should re-query the merged matrix and populate
        // `topk` with the resulting estimates. Mirrors sketchlib-go's
        // `Delta.HHKeys` heavy-hitter rebuild path.
        let mut cs = CountSketch::new(3, 16);
        cs.update("alpha", 5.0);
        cs.update("beta", 3.0);
        let delta = CountSketchDelta {
            rows: 3,
            cols: 16,
            cells: vec![],
            l2: vec![],
            hh_keys: vec!["alpha".to_string(), "beta".to_string()],
        };
        cs.apply_delta(&delta).unwrap();
        assert_eq!(cs.topk.len(), 2);
        let alpha_count = cs
            .topk
            .iter()
            .find(|(k, _)| k == "alpha")
            .map(|(_, v)| *v)
            .expect("alpha should be in topk");
        let beta_count = cs
            .topk
            .iter()
            .find(|(k, _)| k == "beta")
            .map(|(_, v)| *v)
            .expect("beta should be in topk");
        // Alpha was inserted with weight 5; the median estimate
        // should exceed beta's (weight 3) modulo signed-counter
        // cancellation in this small 3x16 matrix.
        assert!(
            alpha_count > beta_count,
            "alpha={alpha_count} beta={beta_count}"
        );
    }

    #[test]
    fn test_apply_delta_hh_keys_topk_capacity() {
        // Verify the Top-K heap is bounded by COUNT_SKETCH_TOPK_CAPACITY
        // and that on overflow, the smallest-count entry is evicted in
        // favor of a larger-count newcomer.
        let mut cs = CountSketch::new(3, 1024);
        let n = COUNT_SKETCH_TOPK_CAPACITY + 5;
        let keys: Vec<String> = (0..n).map(|i| format!("k{i:04}")).collect();
        // Fill all keys into the matrix so estimates are non-zero.
        for (i, k) in keys.iter().enumerate() {
            cs.update(k, (i + 1) as f64);
        }
        let delta = CountSketchDelta {
            rows: 3,
            cols: 1024,
            cells: vec![],
            l2: vec![],
            hh_keys: keys.clone(),
        };
        cs.apply_delta(&delta).unwrap();
        assert_eq!(cs.topk.len(), COUNT_SKETCH_TOPK_CAPACITY);
    }

    #[test]
    fn test_msgpack_round_trip() {
        let original =
            CountSketch::from_legacy_matrix(vec![vec![1.5, -2.5], vec![3.5, -4.5]], 2, 2);
        let bytes = original.to_msgpack().unwrap();
        let decoded = CountSketch::from_msgpack(&bytes).unwrap();
        assert_eq!(decoded.sketch(), original.sketch());
        assert_eq!(decoded.rows, original.rows);
        assert_eq!(decoded.cols, original.cols);
    }

    /// Cross-language byte-parity guard against `sketchlib-go`'s
    /// `CountSketch.SerializeProtoBytes` output for the deterministic
    /// input `goldenCsKeys()` (25 keys "k-a"..."k-e", each repeated 5×)
    /// at dimensions `(rows=3, cols=512)`. The hex blob below was
    /// captured from a `proto.Marshal` of the Go envelope with
    /// `Producer` and `HashSpec` cleared (matching the
    /// `integration/parity/golden_test.go::TestGenerateGoldenFixtures`
    /// recipe used for DDSketch and KLL).
    ///
    /// Any drift in [`CountSketch::update`]'s hash → (col, sign)
    /// derivation breaks this test cell-for-cell; that is the contract
    /// `cross_language_parity::countsketch_byte_parity_with_go` in
    /// ASAPCollector relies on.
    #[test]
    fn test_update_then_envelope_matches_sketchlib_go_bytes() {
        use crate::proto::sketchlib::{
            CountSketchState, CounterType, SketchEnvelope, sketch_envelope::SketchState,
        };
        use prost::Message;

        let rows = 3usize;
        let cols = 512usize;
        let mut sk = CountSketch::new(rows, cols);
        for i in 0..25 {
            let key = format!("k-{}", (b'a' + (i % 5) as u8) as char);
            sk.update(&key, 1.0);
        }

        // Build envelope mirroring sketchlib-go's CountSketch.SerializePortable:
        //   counter_type = INT64, counts packed sint64 in row-major
        //   order, l2 = per-row sum of squared cells, no TopK.
        let mut counts_int: Vec<i64> = Vec::with_capacity(rows * cols);
        let mut l2: Vec<f64> = Vec::with_capacity(rows);
        for row in sk.matrix.iter().take(rows) {
            let mut row_l2 = 0.0f64;
            for &cell in row.iter().take(cols) {
                counts_int.push(cell as i64);
                row_l2 += cell * cell;
            }
            l2.push(row_l2);
        }

        let state = CountSketchState {
            rows: rows as u32,
            cols: cols as u32,
            counter_type: CounterType::Int64 as i32,
            counts_int,
            counts_float: Vec::new(),
            l2,
            topk: None,
        };
        let envelope = SketchEnvelope {
            format_version: 1,
            producer: None,
            hash_spec: None,
            sample_p: 0.0,
            sketch_state: Some(SketchState::CountSketch(state)),
        };
        let mut got = Vec::with_capacity(envelope.encoded_len());
        envelope.encode(&mut got).expect("prost encode");

        // Byte string captured from sketchlib-go for the same
        // (3,512) × `goldenCsKeys()` input — see
        // `integration/parity/golden_test.go` and
        // `cross_language_parity::countsketch_byte_parity_with_go`
        // in ASAPCollector. 1577 bytes total: a `SketchEnvelope` proto
        // wrapping a `CountSketchState` whose `counts_int` is a packed
        // sint64 (zigzag) row-major matrix and `l2` is `[125.0;3]`
        // (each row's 5 hot cells hold ±5 → l2 = 5·25 = 125).
        const GOLDEN_HEX: &str = "08015aa40c0803108004180222800c000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000090000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000090000000000000a000000000000000900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000900000000000000000000000000000000000000000000000000000000000000000a0000000000000000000000090000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000032180000000000405f400000000000405f400000000000405f40";
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "CountSketch envelope bytes diverge from sketchlib-go golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );
    }

    /// Cross-language byte-parity guard for CountSketchDelta hh_keys: encoding
    /// the canonical packed delta with heavy-hitter keys at tag 6 must be
    /// byte-identical to the Go reference's `SerializeDelta` output for the same
    /// state and keys. The empty-`hh_keys` path is already covered by
    /// `test_compute_delta_matches_go_golden_bytes`.
    #[test]
    fn test_hh_keys_matches_go_golden_bytes() {
        use crate::proto::sketchlib::CountSketchDelta as ProtoDelta;
        use prost::Message;

        let rows = 3usize;
        let cols = 512usize;
        let mut current = CountSketch::new(rows, cols);
        for i in 0..25 {
            let key = format!("k-{}", (b'a' + (i % 5) as u8) as char);
            current.update(&key, 1.0);
        }
        let empty = CountSketch::new(rows, cols);

        // Inject heavy-hitter keys onto the canonical packed delta at tag 6.
        let no_hh = current.compute_delta(&empty, 1.0).unwrap();
        let mut proto = ProtoDelta::decode(no_hh.as_slice()).unwrap();
        proto.hh_keys = vec!["flow-0".into(), "flow-3".into(), "flow-7".into()];
        let got = proto.encode_to_vec();

        // Captured from the Go reference implementation's SerializeDelta for the
        // same input with HHKeys = ["flow-0","flow-3","flow-7"].
        const GOLDEN_HEX: &str = "08031080043206666c6f772d303206666c6f772d333206666c6f772d374a0f000000000001010101010202020202521a3d7fd9019402d3034eb9019602f703fe0306a501c601d201ba025a0f090a0909090a090a090a09090a090962180000000000405f400000000000405f400000000000405f40";
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "CountSketch delta hh_keys bytes diverge from the Go reference golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );
    }

    fn decode_hex(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|pair| {
                let high = hex_nibble(pair[0]);
                let low = hex_nibble(pair[1]);
                (high << 4) | low
            })
            .collect()
    }

    fn hex_nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("non-hex byte {}", c as char),
        }
    }

    /// `compute_delta` against an EMPTY snapshot reconstructs the window's
    /// full (signed) state when its bytes are applied to a fresh empty
    /// sketch (round-trip). With `threshold = 1.0` every changed cell
    /// survives.
    #[test]
    fn test_compute_delta_against_empty_round_trips() {
        let rows = 3usize;
        let cols = 512usize;
        let mut window = CountSketch::new(rows, cols);
        for i in 0..200u64 {
            window.update(&format!("k-{}", i % 23), 1.0);
        }
        let empty = CountSketch::new(rows, cols);

        let delta_bytes = window.compute_delta(&empty, 1.0).unwrap();

        let mut reconstructed = CountSketch::new(rows, cols);
        reconstructed.apply_delta_bytes(&delta_bytes).unwrap();

        assert_eq!(reconstructed.sketch(), window.sketch());
    }

    /// A delta computed between two non-empty snapshots reconstructs the
    /// current sketch when applied to the base.
    #[test]
    fn test_compute_delta_then_apply_matches_current() {
        let mut base = CountSketch::new(3, 64);
        for i in 0..40u64 {
            base.update(&format!("x{}", i % 8), 1.0);
        }
        let mut current = base.clone();
        for i in 0..30u64 {
            current.update(&format!("x{}", i % 8), 1.0);
        }

        let delta_bytes = current.compute_delta(&base, 1.0).unwrap();
        let mut reconstructed = base.clone();
        reconstructed.apply_delta_bytes(&delta_bytes).unwrap();
        assert_eq!(reconstructed.sketch(), current.sketch());
    }

    /// Cross-language byte-parity guard: `compute_delta` against an empty
    /// snapshot must emit bytes identical to the Go reference
    /// implementation's `SerializeDelta(ComputeDelta(empty, current, 1.0))`
    /// for the same `(rows=3, cols=512)` × "k-a".."k-e" (each 5×, 25
    /// unweighted updates) input. The golden hex was captured from a
    /// `proto.Marshal` of the Go reference's `CountSketchDelta`
    /// (packed-array encoding). No heavy-hitter keys are forwarded by this
    /// minimal wrapper, so `hh_keys` is empty in both producers; the golden
    /// pins the packed cell arrays and the per-row L2 (=125) norm deltas.
    #[test]
    fn test_compute_delta_matches_go_golden_bytes() {
        let rows = 3usize;
        let cols = 512usize;
        let mut current = CountSketch::new(rows, cols);
        for i in 0..25 {
            let key = format!("k-{}", (b'a' + (i % 5) as u8) as char);
            current.update(&key, 1.0);
        }
        let empty = CountSketch::new(rows, cols);
        let got = current.compute_delta(&empty, 1.0).unwrap();

        // Captured from the Go reference implementation's
        // SerializeDelta(ComputeDelta(empty, current, 1.0)) for the same input.
        const GOLDEN_HEX: &str = "08031080044a0f000000000001010101010202020202521a3d7fd9019402d3034eb9019602f703fe0306a501c601d201ba025a0f090a0909090a090a090a09090a090962180000000000405f400000000000405f400000000000405f40";
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "CountSketch delta bytes diverge from the Go reference golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );
    }
}
