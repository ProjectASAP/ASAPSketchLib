//! Wire-format-aligned Count-Min sketch types. The wire DTO and
//! runtime ops live together here.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error as MsgPackError, MessagePackCodec};
use crate::sketches::countminsketch::CountMin;
use crate::{DataInput, FastPath, Vector2D};

// =====================================================================
// asap_sketchlib wire-format-aligned variant.
//
// `CountMinSketch` and `CountMinDelta` below are the public-field,
// proto-decode-friendly types consumed by the ASAP query engine
// accumulators, backed by `asap_sketchlib`'s in-tree CountMin. The
// high-throughput in-process variant is
// `crate::sketches::countminsketch::CountMin`, which keeps its own design.
// =====================================================================

// ----- asap_sketchlib-backed Count-Min helpers -----
// Used below by `CountMinSketch`. Lives in this file so the wire-format
// type and its backend share a single home.

/// Concrete Count-Min type backing the wire-format `CountMinSketch`.
/// Uses f64 counters (`Vector2D<f64>`) for weighted updates without
/// integer rounding, and the `FastPath` packed-hash strategy so the
/// matrix-cell layout is byte-parity with `sketchlib-go`. Locked in by
/// `tests/sketches_go_parity_probe.rs`.
pub type SketchlibCms = CountMin<Vector2D<f64>, FastPath>;

/// Creates a fresh sketchlib Count-Min sketch with the given dimensions.
pub fn new_sketchlib_cms(row_num: usize, col_num: usize) -> SketchlibCms {
    SketchlibCms::with_dimensions(row_num, col_num)
}

/// Builds a sketchlib Count-Min sketch from an existing `sketch` matrix.
pub fn sketchlib_cms_from_matrix(
    row_num: usize,
    col_num: usize,
    sketch: &[Vec<f64>],
) -> SketchlibCms {
    let matrix = Vector2D::from_fn(row_num, col_num, |r, c| {
        sketch
            .get(r)
            .and_then(|row| row.get(c))
            .copied()
            .unwrap_or(0.0)
    });
    SketchlibCms::from_storage(matrix)
}

/// Converts a sketchlib Count-Min sketch into a `Vec<Vec<f64>>` matrix.
pub fn matrix_from_sketchlib_cms(inner: &SketchlibCms) -> Vec<Vec<f64>> {
    let storage: &Vector2D<f64> = inner.as_storage();
    let rows = storage.rows();
    let cols = storage.cols();
    let mut sketch = vec![vec![0.0; cols]; rows];

    for (r, row) in sketch.iter_mut().enumerate().take(rows) {
        for (c, cell) in row.iter_mut().enumerate().take(cols) {
            if let Some(v) = storage.get(r, c) {
                *cell = *v;
            }
        }
    }

    sketch
}

/// Helper to update a sketchlib Count-Min with a weighted key.
pub fn sketchlib_cms_update(inner: &mut SketchlibCms, key: &str, value: f64) {
    if value <= 0.0 {
        return;
    }
    inner.insert_many(&DataInput::String(key.to_owned()), value);
}

/// Helper to query a sketchlib Count-Min for a key, returning f64.
pub fn sketchlib_cms_query(inner: &SketchlibCms, key: &str) -> f64 {
    inner.estimate(&DataInput::String(key.to_owned()))
}

/// Sparse delta between two consecutive CountMinSketch snapshots —
/// the input shape for [`CountMinSketch::apply_delta`]. Mirrors the
/// `CountMinSketchDelta` proto in
/// `sketchlib-go/proto/countminsketch/countminsketch.proto` (packed
/// encoding only).
///
/// Cells apply additively: `matrix[row][col] += d_count`. Per-row
/// L1 and L2 norm deltas are carried for downstream error-accounting
/// but are not consumed by `apply_delta` itself.
///
/// Structurally identical to [`super::countsketch::CountSketchDelta`]: the
/// optional `hh_keys` channel carries heavy-hitter candidate keys forwarded by
/// an upstream tracker. Count-Min can track heavy hitters, but whether
/// `hh_keys` is populated is a control-plane decision — it is empty when
/// heavy-hitter tracking is not enabled. The receiver re-queries the merged
/// matrix for each key to (re)build its Top-K. Mirrors the Go reference
/// implementation's `Delta.HHKeys`.
#[derive(Debug, Clone, Default)]
pub struct CountMinSketchDelta {
    pub rows: u32,
    pub cols: u32,
    pub cells: Vec<(u32, u32, i64)>,
    pub l1: Vec<f64>,
    pub l2: Vec<f64>,
    pub hh_keys: Vec<String>,
}

/// Provides approximate frequency counts with error bounds.
/// The msgpack wire format is the contract between sketch producers and
/// the query engine consumer.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    pub rows: usize,
    pub cols: usize,
    pub(crate) backend: SketchlibCms,
}

impl CountMinSketch {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            backend: new_sketchlib_cms(rows, cols),
        }
    }

    /// Number of hash rows in the sketch matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns (width) in the sketch matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the sketch matrix (for wire format, serialization, tests).
    pub fn sketch(&self) -> Vec<Vec<f64>> {
        matrix_from_sketchlib_cms(&self.backend)
    }

    /// Construct from a `Vec<Vec<f64>>` matrix (used by deserialization and query engine).
    pub fn from_legacy_matrix(sketch: Vec<Vec<f64>>, rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            backend: sketchlib_cms_from_matrix(rows, cols, &sketch),
        }
    }

    /// Insert a single weighted observation. Delegates to the
    /// `FastPath` backend so the matrix-cell layout matches
    /// `sketchlib-go::CountMinSketch.InsertWithHash` bit-for-bit; the
    /// parity is locked in by `tests/sketches_go_parity_probe.rs`.
    ///
    /// Negative or zero values are skipped, mirroring Go's
    /// `UpdateWeight` behavior on `many == 0`.
    pub fn update(&mut self, key: &str, value: f64) {
        if value <= 0.0 || self.rows == 0 || self.cols == 0 {
            return;
        }
        self.backend
            .insert_many(&DataInput::String(key.to_owned()), value);
    }

    /// Estimate the frequency of `key` (CountMin point query).
    /// Delegates to the `FastPath` backend, which derives the same
    /// per-row column index that [`Self::update`] used.
    pub fn estimate(&self, key: &str) -> f64 {
        if self.rows == 0 || self.cols == 0 {
            return 0.0;
        }
        self.backend.estimate(&DataInput::String(key.to_owned()))
    }

    /// Merge another CountMinSketch into self in place. Both operands
    /// must have identical dimensions.
    pub fn merge(
        &mut self,
        other: &CountMinSketch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "CountMinSketch dimension mismatch: self={}x{}, other={}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
            .into());
        }
        self.backend.merge(&other.backend);
        Ok(())
    }

    /// Merge from references, allocating only the output — no input clones.
    pub fn merge_refs(
        accumulators: &[&Self],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if accumulators.is_empty() {
            return Err("No accumulators to merge".into());
        }

        let rows = accumulators[0].rows;
        let cols = accumulators[0].cols;

        for acc in accumulators {
            if acc.rows != rows || acc.cols != cols {
                return Err(
                    "Cannot merge CountMinSketch accumulators with different dimensions".into(),
                );
            }
        }

        let mut merged = CountMinSketch::new(rows, cols);
        for acc in accumulators {
            merged.backend.merge(&acc.backend);
        }
        Ok(merged)
    }

    /// Apply a sparse delta in place. Matches the `ApplyDelta`
    /// semantics in `sketchlib-go/sketches/CountMinSketch/delta.go`:
    /// `matrix[row][col] += d_count` for each cell in the delta.
    ///
    /// The FFI handle is opaque, so we snapshot the matrix, apply
    /// cell updates, and rebuild the backend. The rebuild is
    /// O(rows × cols) per delta and is acceptable for ingest-side
    /// reconstitution — no delta should fire more than once per
    /// window (10s–300s in the paper's B3 / B4 configs).
    pub fn apply_delta(
        &mut self,
        delta: &CountMinSketchDelta,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for (row, col, _) in &delta.cells {
            let r = *row as usize;
            let c = *col as usize;
            if r >= self.rows || c >= self.cols {
                return Err(format!(
                    "CountMinSketchDelta cell ({r},{c}) out of range (matrix={}x{})",
                    self.rows, self.cols
                )
                .into());
            }
        }
        let mut matrix = self.sketch();
        for (row, col, d_count) in &delta.cells {
            matrix[*row as usize][*col as usize] += *d_count as f64;
        }
        self.backend = sketchlib_cms_from_matrix(self.rows, self.cols, &matrix);
        Ok(())
    }

    /// Compute a sparse, proto-marshalled `CountMinDelta` of `self`
    /// against a `snapshot`. A cell is included when its absolute count
    /// delta `|Δcount|` (self − snapshot) is `>= threshold` and non-zero.
    /// The full per-row L1/L2 norm deltas are always carried (one entry
    /// per row, negligible size).
    ///
    /// This is the Rust twin of the Go reference implementation's
    /// `ComputeDelta` + `SerializeDelta`: it iterates the matrix in
    /// row-major order, subtracts the snapshot's value for each cell, and
    /// emits the surviving cell deltas using the packed-array proto
    /// encoding (`cell_rows`/`cell_cols`/`d_counts`). L1/L2 row deltas are
    /// derived from the matrices — `l1[r] = Σ_c count[r][c]` and
    /// `l2[r] = Σ_c count[r][c]^2` — which telescopes to the same value the
    /// Go producer maintains incrementally. Heavy-hitter candidate keys
    /// (`hh_keys`) are sourced from an upstream tracker that this minimal
    /// wrapper does not maintain, so the field is left empty (an empty repeated
    /// field encodes to nothing on the wire), exactly as CountSketch's
    /// `compute_delta` does. The returned bytes are a `prost`-encoded
    /// [`crate::proto::sketchlib::CountMinDelta`], byte-identical to the Go
    /// `proto.Marshal(CountMinDelta)` output for the same inputs when no
    /// heavy-hitter keys are forwarded (cross-language byte parity).
    ///
    /// Delta-against-empty: when `snapshot` is the all-zero sketch, every
    /// surviving cell delta equals this window's own cell count, so the
    /// result is this window's full state encoded as a delta (no
    /// cross-window subtraction). CMS deltas carry only sketch-internal
    /// cells/norms — there are no DataPoint-level metric scalars to drop.
    pub fn compute_delta(
        &self,
        snapshot: &CountMinSketch,
        threshold: f64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        use crate::proto::sketchlib::CountMinDelta as ProtoDelta;
        use prost::Message;

        if self.rows != snapshot.rows || self.cols != snapshot.cols {
            return Err(format!(
                "CountMinSketch dimension mismatch: self={}x{}, snapshot={}x{}",
                self.rows, self.cols, snapshot.rows, snapshot.cols
            )
            .into());
        }

        let cur = self.sketch();
        let snap = snapshot.sketch();

        let mut cell_rows: Vec<u32> = Vec::new();
        let mut cell_cols: Vec<u32> = Vec::new();
        let mut d_counts: Vec<i64> = Vec::new();
        let mut l1: Vec<f64> = Vec::with_capacity(self.rows);
        let mut l2: Vec<f64> = Vec::with_capacity(self.rows);

        for r in 0..self.rows {
            let mut cur_l1 = 0.0f64;
            let mut cur_l2 = 0.0f64;
            let mut snap_l1 = 0.0f64;
            let mut snap_l2 = 0.0f64;
            for c in 0..self.cols {
                let cv = cur[r][c];
                let sv = snap[r][c];
                cur_l1 += cv;
                cur_l2 += cv * cv;
                snap_l1 += sv;
                snap_l2 += sv * sv;
                // CMS counts are non-negative integers in the wire form;
                // mirror the Go reference's signed Δ + |Δ| threshold test.
                let dc = (cv - sv) as i64;
                if dc != 0 && (dc.unsigned_abs() as f64) >= threshold {
                    cell_rows.push(r as u32);
                    cell_cols.push(c as u32);
                    d_counts.push(dc);
                }
            }
            l1.push(cur_l1 - snap_l1);
            l2.push(cur_l2 - snap_l2);
        }

        let delta = ProtoDelta {
            rows: self.rows as u32,
            cols: self.cols as u32,
            cells_legacy: Vec::new(),
            l1,
            l2,
            // Heavy-hitter keys are control-plane-gated; this wrapper has no
            // tracker, so the field is empty (mirrors CountSketch::compute_delta).
            hh_keys: Vec::new(),
            cell_rows,
            cell_cols,
            d_counts,
        };
        Ok(delta.encode_to_vec())
    }

    /// Apply a `prost`-encoded [`crate::proto::sketchlib::CountMinDelta`]
    /// to this sketch in place (additive cell merge). The Rust twin of the
    /// Go reference implementation's `DeserializeDelta` + `ApplyDelta`.
    /// Reads the packed cell arrays (`cell_rows`/`cell_cols`/`d_counts`)
    /// and falls back to the legacy per-cell records for payloads from
    /// older producers. Heavy-hitter keys (`hh_keys`) are read symmetrically
    /// (mirroring CountSketch) and carried onto the decoded delta; a plain CMS
    /// keeps no Top-K, so they are not otherwise consumed here.
    ///
    /// Returns `Err` if `bytes` is not a valid `CountMinDelta` proto or a
    /// cell is out of range for this sketch's dimensions.
    pub fn apply_delta_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::proto::sketchlib::CountMinDelta as ProtoDelta;
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
        let delta = CountMinSketchDelta {
            rows: proto.rows,
            cols: proto.cols,
            cells,
            l1: proto.l1,
            l2: proto.l2,
            // Heavy-hitter keys are carried through symmetrically (mirrors
            // CountSketch). This wrapper keeps no Top-K, so they are not
            // otherwise consumed; a control-plane sink would re-query the
            // merged matrix for each key. Empty for a plain CMS.
            hh_keys: proto.hh_keys,
        };
        self.apply_delta(&delta)
    }

    /// MessagePack twin of [`Self::compute_delta`]. Computes the same
    /// sparse cell delta of `self` against `snapshot` (a cell is carried
    /// when `|Δcount|` is non-zero and `>= threshold`), but serializes it
    /// as the cells-only parallel-array MessagePack layout via
    /// [`CountMinSketchDelta`]'s [`MessagePackCodec`] instead of proto. The
    /// per-row `l1`/`l2` norms and `hh_keys` the proto delta carries are
    /// dropped — the plain-sketch estimate path does not consume them.
    /// Delta-against-empty carries this window's own cell counts.
    pub fn compute_delta_msgpack(
        &self,
        snapshot: &CountMinSketch,
        threshold: f64,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        if self.rows != snapshot.rows || self.cols != snapshot.cols {
            return Err(format!(
                "CountMinSketch dimension mismatch: self={}x{}, snapshot={}x{}",
                self.rows, self.cols, snapshot.rows, snapshot.cols
            )
            .into());
        }
        let cur = self.sketch();
        let snap = snapshot.sketch();
        let mut delta = CountMinSketchDelta {
            rows: self.rows as u32,
            cols: self.cols as u32,
            ..CountMinSketchDelta::default()
        };
        for r in 0..self.rows {
            for c in 0..self.cols {
                let dc = (cur[r][c] - snap[r][c]) as i64;
                if dc != 0 && (dc.unsigned_abs() as f64) >= threshold {
                    delta.cells.push((r as u32, c as u32, dc));
                }
            }
        }
        Ok(delta.to_msgpack()?)
    }

    /// MessagePack twin of [`Self::apply_delta_bytes`]. Decodes the
    /// cells-only parallel-array MessagePack [`CountMinSketchDelta`] and
    /// applies it in place (additive cell merge).
    ///
    /// Returns `Err` if `bytes` is not a valid MessagePack
    /// `CountMinSketchDelta` or a cell is out of range for this sketch's
    /// dimensions.
    pub fn apply_delta_msgpack_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let delta = CountMinSketchDelta::from_msgpack(bytes)?;
        self.apply_delta(&delta)
    }

    /// One-shot aggregation: build a sketch from parallel key/value slices
    /// and return the msgpack bytes.
    pub fn aggregate_count(
        depth: usize,
        width: usize,
        keys: &[&str],
        values: &[f64],
    ) -> Option<Vec<u8>> {
        if keys.is_empty() {
            return None;
        }
        let mut sketch = Self::new(depth, width);
        for (key, &value) in keys.iter().zip(values.iter()) {
            sketch.update(key, value);
        }
        sketch.to_msgpack().ok()
    }

    /// Same as aggregate_count — CMS accumulates sums by construction.
    pub fn aggregate_sum(
        depth: usize,
        width: usize,
        keys: &[&str],
        values: &[f64],
    ) -> Option<Vec<u8>> {
        Self::aggregate_count(depth, width, keys, values)
    }
}

// ----- Wire format -----

/// Wire DTO for [`CountMinSketch`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountMinSketchWire {
    pub sketch: Vec<Vec<f64>>,
    #[serde(rename = "row_num")]
    pub rows: usize,
    #[serde(rename = "col_num")]
    pub cols: usize,
}

impl MessagePackCodec for CountMinSketch {
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let wire = CountMinSketchWire {
            sketch: self.sketch(),
            rows: self.rows,
            cols: self.cols,
        };
        Ok(rmp_serde::to_vec(&wire)?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        let wire: CountMinSketchWire = rmp_serde::from_slice(bytes)?;
        let backend = sketchlib_cms_from_matrix(wire.rows, wire.cols, &wire.sketch);
        Ok(Self {
            rows: wire.rows,
            cols: wire.cols,
            backend,
        })
    }
}

impl MessagePackCodec for CountMinSketchDelta {
    /// Encodes the sparse cell delta as the 5-element MessagePack array
    /// `[ rows:u64, cols:u64, row_idx:[]u32, col_idx:[]u32, d_count:[]i64 ]`
    /// — byte-identical to Rust
    /// `rmp_serde::to_vec(&(u64, u64, Vec<u32>, Vec<u32>, Vec<i64>))`
    /// (compact mode), the same struct-of-parallel-arrays shape the Go
    /// `count_sketch_delta_sparse.go` established. Only the matrix cells
    /// cross the wire; the per-row `l1`/`l2` norms and the `hh_keys`
    /// heavy-hitter surface are not carried.
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
        Ok(CountMinSketchDelta {
            rows: rows as u32,
            cols: cols as u32,
            cells,
            ..CountMinSketchDelta::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msgpack_delta_against_empty_round_trips() {
        let (rows, cols) = (5, 1024);
        let mut w = CountMinSketch::new(rows, cols);
        for _ in 0..200 {
            w.update("hot", 1.0);
        }
        let empty = CountMinSketch::new(rows, cols);
        let bytes = w.compute_delta_msgpack(&empty, 1.0).unwrap();
        let mut recon = CountMinSketch::new(rows, cols);
        recon.apply_delta_msgpack_bytes(&bytes).unwrap();
        let (got, want) = (recon.estimate("hot"), w.estimate("hot"));
        assert!((got - want).abs() <= want * 0.1, "got={got} want={want}");
    }

    #[test]
    fn msgpack_delta_wire_layout_is_parallel_arrays() {
        use crate::message_pack_format::MessagePackCodec;
        let (rows, cols) = (3, 8);
        let mut w = CountMinSketch::new(rows, cols);
        w.update("x", 5.0);
        let empty = CountMinSketch::new(rows, cols);
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
        let d = CountMinSketchDelta::from_msgpack(&bytes).unwrap();
        assert_eq!(d.rows, rows as u32);
        assert_eq!(d.cols, cols as u32);
    }

    #[test]
    fn test_count_min_sketch_creation() {
        let cms = CountMinSketch::new(4, 1000);
        assert_eq!(cms.rows, 4);
        assert_eq!(cms.cols, 1000);
        let sketch = cms.sketch();
        assert_eq!(sketch.len(), 4);
        assert_eq!(sketch[0].len(), 1000);

        // Check all values are initialized to 0
        for row in &sketch {
            for &value in row {
                assert_eq!(value, 0.0);
            }
        }
    }

    #[test]
    fn test_count_min_sketch_update() {
        let mut cms = CountMinSketch::new(2, 10);
        cms.update("key1", 1.0);
        // Query should return at least the updated value
        let result = cms.estimate("key1");
        assert!(result >= 1.0);
    }

    #[test]
    fn test_count_min_sketch_query_empty() {
        let cms = CountMinSketch::new(2, 10);
        assert_eq!(cms.estimate("anything"), 0.0);
    }

    #[test]
    fn test_count_min_sketch_merge() {
        // Use from_legacy_matrix so the test works regardless of sketchlib/legacy config
        let mut sketch1 = vec![vec![0.0; 3]; 2];
        sketch1[0][0] = 5.0;
        sketch1[1][2] = 10.0;
        let mut cms1 = CountMinSketch::from_legacy_matrix(sketch1, 2, 3);

        let mut sketch2 = vec![vec![0.0; 3]; 2];
        sketch2[0][0] = 3.0;
        sketch2[0][1] = 7.0;
        let cms2 = CountMinSketch::from_legacy_matrix(sketch2, 2, 3);

        cms1.merge(&cms2).unwrap();
        let merged_sketch = cms1.sketch();

        assert_eq!(merged_sketch[0][0], 8.0); // 5 + 3
        assert_eq!(merged_sketch[0][1], 7.0); // 0 + 7
        assert_eq!(merged_sketch[1][2], 10.0); // 10 + 0
    }

    #[test]
    fn test_count_min_sketch_merge_dimension_mismatch() {
        let mut cms1 = CountMinSketch::new(2, 3);
        let cms2 = CountMinSketch::new(3, 3);
        assert!(cms1.merge(&cms2).is_err());
    }

    #[test]
    fn test_count_min_sketch_msgpack_round_trip() {
        let mut cms = CountMinSketch::new(4, 256);
        cms.update("apple", 5.0);
        cms.update("banana", 3.0);
        cms.update("apple", 2.0); // total "apple" = 7

        let bytes = cms.to_msgpack().unwrap();
        let deserialized = CountMinSketch::from_msgpack(&bytes).unwrap();

        assert_eq!(deserialized.rows, 4);
        assert_eq!(deserialized.cols, 256);
        assert!(deserialized.estimate("apple") >= 7.0);
        assert!(deserialized.estimate("banana") >= 3.0);
    }

    #[test]
    fn test_aggregate_count() {
        let keys = ["a", "b", "a"];
        let values = [1.0, 2.0, 3.0];
        let bytes = CountMinSketch::aggregate_count(4, 100, &keys, &values).unwrap();
        let cms = CountMinSketch::from_msgpack(&bytes).unwrap();
        // "a" was updated twice (1.0 + 3.0 = 4.0), "b" once (2.0)
        assert!(cms.estimate("a") >= 4.0);
        assert!(cms.estimate("b") >= 2.0);
    }

    #[test]
    fn test_aggregate_count_empty() {
        assert!(CountMinSketch::aggregate_count(4, 100, &[], &[]).is_none());
    }

    #[test]
    fn test_apply_delta_additive() {
        let mut cms = CountMinSketch::from_legacy_matrix(
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            2,
            3,
        );
        let delta = CountMinSketchDelta {
            rows: 2,
            cols: 3,
            cells: vec![(0, 0, 10), (1, 2, 100)],
            l1: vec![],
            l2: vec![],
            hh_keys: vec![],
        };
        cms.apply_delta(&delta).unwrap();
        assert_eq!(
            cms.sketch(),
            vec![vec![11.0, 2.0, 3.0], vec![4.0, 5.0, 106.0]]
        );
    }

    #[test]
    fn test_apply_delta_matches_full_merge() {
        let base = CountMinSketch::from_legacy_matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]], 2, 2);
        let addition =
            CountMinSketch::from_legacy_matrix(vec![vec![10.0, 0.0], vec![0.0, 20.0]], 2, 2);
        let mut via_merge = base.clone();
        via_merge.merge(&addition).unwrap();

        let delta = CountMinSketchDelta {
            rows: 2,
            cols: 2,
            cells: vec![(0, 0, 10), (1, 1, 20)],
            l1: vec![],
            l2: vec![],
            hh_keys: vec![],
        };
        let mut via_delta = base;
        via_delta.apply_delta(&delta).unwrap();
        assert_eq!(via_delta.sketch(), via_merge.sketch());
    }

    #[test]
    fn test_apply_delta_out_of_range() {
        let mut cms = CountMinSketch::new(2, 3);
        let delta = CountMinSketchDelta {
            rows: 2,
            cols: 3,
            cells: vec![(5, 0, 1)],
            l1: vec![],
            l2: vec![],
            hh_keys: vec![],
        };
        assert!(cms.apply_delta(&delta).is_err());
    }

    /// Cross-language byte-parity guard against `sketchlib-go`'s
    /// `CountMinSketch.SerializeProtoBytesFO` output for the
    /// deterministic input `goldenCmsKeys()` (10 keys "flow-0"..."flow-9",
    /// each repeated 5×, 50 unweighted updates total) at dimensions
    /// `(rows=4, cols=2048)` — the
    /// `asap-precompute-rs::CMSWrapper::new(4, 2048)` default. The hex
    /// blob below was captured from `proto.Marshal` of the Go envelope
    /// with `Producer` and `HashSpec` cleared, matching the
    /// `integration/parity/golden_test.go::TestGenerateGoldenFixtures`
    /// recipe (and the byte-payload that the Rust wrapper's
    /// `CMSWrapper::encode_envelope` emits to satisfy
    /// `cross_language_parity::cms_byte_parity_with_go`).
    ///
    /// Frequency-Only payload structure (matches Go's
    /// `SerializePortableFO`):
    /// - `rows = 4`, `cols = 2048`, `counter_type = INT64`
    /// - `counts_int` = packed sint64 row-major, `4*2048 = 8192` cells
    /// - `sum_counts` and `sum2_counts` deliberately omitted
    /// - `l1[r] = 50` (each row sees 50 unweighted inserts)
    /// - `l2[r] = 250` (10 unique cells per row each holding count 5,
    ///   so `Σ count^2 = 10 * 25 = 250`)
    ///
    /// Any drift in [`CountMinSketch::update`]'s hash → column-index
    /// derivation breaks this test cell-for-cell; that is the contract
    /// `cross_language_parity::cms_byte_parity_with_go` in ASAPCollector
    /// relies on.
    #[test]
    fn test_update_then_envelope_matches_sketchlib_go_bytes() {
        use crate::proto::sketchlib::{
            CountMinState, CounterType, SketchEnvelope, sketch_envelope::SketchState,
        };
        use prost::Message;

        let rows = 4usize;
        let cols = 2048usize;
        let mut sk = CountMinSketch::new(rows, cols);
        for i in 0..50u64 {
            let key = format!("flow-{}", i % 10);
            sk.update(&key, 1.0);
        }

        // Build the Frequency-Only envelope mirroring sketchlib-go's
        // `CountMinSketch.SerializePortableFO`:
        //   - counter_type = INT64
        //   - counts_int = packed sint64 row-major, len = rows * cols
        //   - sum_counts / sum2_counts deliberately omitted
        //   - l1[r] = Σ_c count[r][c]   (Go maintains this incrementally
        //     in InsertWithHash; equals `weight * Σ inserts in row r`,
        //     which collapses to 50 for an unweighted 50-insert stream)
        //   - l2[r] = Σ_c count[r][c]^2 (Go maintains this as
        //     `L2[r] += curr*curr - prev*prev`, telescoping to the same
        //     sum-of-squares for unweighted streams)
        let matrix = sk.sketch();
        let mut counts_int: Vec<i64> = Vec::with_capacity(rows * cols);
        let mut l1: Vec<f64> = Vec::with_capacity(rows);
        let mut l2: Vec<f64> = Vec::with_capacity(rows);
        for row in matrix.iter().take(rows) {
            let mut row_l1 = 0.0f64;
            let mut row_l2 = 0.0f64;
            for &cell in row.iter().take(cols) {
                counts_int.push(cell as i64);
                row_l1 += cell;
                row_l2 += cell * cell;
            }
            l1.push(row_l1);
            l2.push(row_l2);
        }

        let state = CountMinState {
            rows: rows as u32,
            cols: cols as u32,
            counter_type: CounterType::Int64 as i32,
            counts_int,
            counts_float: Vec::new(),
            sum_counts: Vec::new(),
            sum2_counts: Vec::new(),
            l1,
            l2,
        };
        let envelope = SketchEnvelope {
            format_version: 1,
            producer: None,
            hash_spec: None,
            sample_p: 0.0,
            sketch_state: Some(SketchState::CountMin(state)),
        };
        let mut got = Vec::with_capacity(envelope.encoded_len());
        envelope.encode(&mut got).expect("prost encode");

        // 8275-byte hex blob captured from
        // `sketchlib-go::CountMinSketch.SerializeProtoBytesFO` for the
        // same `(4, 2048) × goldenCmsKeys()` input — see
        // `integration/parity/golden_test.go` and
        // `cross_language_parity::cms_byte_parity_with_go` in
        // ASAPCollector.
        const GOLDEN_HEX: &str = include_str!("../../sketches/testdata/cms_envelope_golden.hex");
        let want = decode_hex_cms(GOLDEN_HEX);
        assert_eq!(
            got.len(),
            want.len(),
            "CMS envelope length differs: got {} bytes, want {} bytes",
            got.len(),
            want.len(),
        );
        assert_eq!(
            got, want,
            "CMS envelope bytes diverge from sketchlib-go golden"
        );
    }

    /// Cross-language byte-parity guard for CountMinDelta hh_keys: encoding the
    /// canonical packed delta with heavy-hitter keys at tag 6 must be
    /// byte-identical to the Go reference's `SerializeDelta` output for the same
    /// state and keys. The empty-`hh_keys` path is already covered by
    /// `test_compute_delta_matches_go_golden_bytes`.
    #[test]
    fn test_hh_keys_matches_go_golden_bytes() {
        use crate::proto::sketchlib::CountMinDelta as ProtoDelta;
        use prost::Message;

        let rows = 4usize;
        let cols = 2048usize;
        let mut current = CountMinSketch::new(rows, cols);
        for i in 0..50u64 {
            current.update(&format!("flow-{}", i % 10), 1.0);
        }
        let empty = CountMinSketch::new(rows, cols);

        // Inject heavy-hitter keys onto the canonical packed delta at tag 6.
        let no_hh = current.compute_delta(&empty, 1.0).unwrap();
        let mut proto = ProtoDelta::decode(no_hh.as_slice()).unwrap();
        proto.hh_keys = vec!["flow-0".into(), "flow-3".into(), "flow-7".into()];
        let got = proto.encode_to_vec();

        // Captured from the Go reference implementation's SerializeDelta for the
        // same input with HHKeys = ["flow-0","flow-3","flow-7"].
        const GOLDEN_HEX: &str = "0804108010222000000000000049400000000000004940000000000000494000000000000049402a200000000000406f400000000000406f400000000000406f400000000000406f403206666c6f772d303206666c6f772d333206666c6f772d374a28000000000000000000000101010101010101010102020202020202020202030303030303030303035250c101b202b608b7088809bc09c20dc60d910fbb0f9f018d04a604c004d6058507a909920b900ec70ead038304bc058007c207cc0a820bd50cbe0fcd0fe2019e02cd02e702a707b809a00ac20cee0dfe0f5a280a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
        let want = decode_hex_cms(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "CountMin delta hh_keys bytes diverge from the Go reference golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );

        // Symmetric decode: apply_delta_bytes round-trips hh_keys without error.
        let mut sink = CountMinSketch::new(rows, cols);
        sink.apply_delta_bytes(&got).unwrap();
    }

    fn decode_hex_cms(s: &str) -> Vec<u8> {
        let s = s.trim();
        s.as_bytes()
            .chunks(2)
            .map(|pair| {
                let high = hex_nibble_cms(pair[0]);
                let low = hex_nibble_cms(pair[1]);
                (high << 4) | low
            })
            .collect()
    }

    fn hex_nibble_cms(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("non-hex byte {}", c as char),
        }
    }

    /// `compute_delta` against an EMPTY snapshot reconstructs the window's
    /// full state when its bytes are applied to a fresh empty sketch
    /// (round-trip). With `threshold = 1.0` every changed cell survives, so
    /// applying the delta to a zero base yields the same matrix as the
    /// original window.
    #[test]
    fn test_compute_delta_against_empty_round_trips() {
        let rows = 4usize;
        let cols = 2048usize;
        let mut window = CountMinSketch::new(rows, cols);
        for i in 0..200u64 {
            window.update(&format!("flow-{}", i % 37), 1.0);
        }
        let empty = CountMinSketch::new(rows, cols);

        let delta_bytes = window.compute_delta(&empty, 1.0).unwrap();

        let mut reconstructed = CountMinSketch::new(rows, cols);
        reconstructed.apply_delta_bytes(&delta_bytes).unwrap();

        assert_eq!(reconstructed.sketch(), window.sketch());
    }

    /// A delta computed between two non-empty snapshots reconstructs the
    /// current sketch when applied to the base — matching a direct merge of
    /// the cell-wise difference.
    #[test]
    fn test_compute_delta_then_apply_matches_current() {
        let mut base = CountMinSketch::new(2, 64);
        for i in 0..40u64 {
            base.update(&format!("k{}", i % 8), 1.0);
        }
        let mut current = base.clone();
        for i in 0..30u64 {
            current.update(&format!("k{}", i % 8), 1.0);
        }

        let delta_bytes = current.compute_delta(&base, 1.0).unwrap();
        let mut reconstructed = base.clone();
        reconstructed.apply_delta_bytes(&delta_bytes).unwrap();
        assert_eq!(reconstructed.sketch(), current.sketch());
    }

    /// Cross-language byte-parity guard: `compute_delta` against an empty
    /// snapshot must emit bytes identical to the Go reference
    /// implementation's `SerializeDelta(ComputeDelta(empty, current, 1.0))`
    /// for the same `(rows=4, cols=2048)` × "flow-0".."flow-9" (each 5×,
    /// 50 unweighted updates) input. The golden hex was captured from a
    /// `proto.Marshal` of the Go reference's `CountMinDelta` (packed-array
    /// encoding) for that input. A delta-against-empty carries the window's
    /// full state, so this also pins the per-row L1 (=50) / L2 (=250) norm
    /// deltas and the packed cell arrays.
    #[test]
    fn test_compute_delta_matches_go_golden_bytes() {
        let rows = 4usize;
        let cols = 2048usize;
        let mut current = CountMinSketch::new(rows, cols);
        for i in 0..50u64 {
            current.update(&format!("flow-{}", i % 10), 1.0);
        }
        let empty = CountMinSketch::new(rows, cols);
        let got = current.compute_delta(&empty, 1.0).unwrap();

        // Captured from the Go reference implementation's
        // SerializeDelta(ComputeDelta(empty, current, 1.0)) for the same input.
        const GOLDEN_HEX: &str = "0804108010222000000000000049400000000000004940000000000000494000000000000049402a200000000000406f400000000000406f400000000000406f400000000000406f404a28000000000000000000000101010101010101010102020202020202020202030303030303030303035250c101b202b608b7088809bc09c20dc60d910fbb0f9f018d04a604c004d6058507a909920b900ec70ead038304bc058007c207cc0a820bd50cbe0fcd0fe2019e02cd02e702a707b809a00ac20cee0dfe0f5a280a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
        let want = decode_hex_cms(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "CountMin delta bytes diverge from the Go reference golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );
    }
}
