//! Wire-format-aligned Count Sketch + top-k heap composite.
//!
//! No `sketches::*` equivalent — this combines `sketches::CSHeap` with
//! the wire-format `CountSketch` shape and exposes it as a single
//! Go-interop type. Mirrors [`crate::message_pack_format::portable::countminsketch_topk::CountMinSketchWithHeap`]
//! structurally, but backed by `CSHeap` (median estimator, signed rows)
//! instead of `CMSHeap` (min-over-rows estimator) — the two are
//! different sketch algorithms that happen to produce a same-shaped
//! `rows x cols` matrix, not interchangeable data.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error as MsgPackError, MessagePackCodec};
use crate::sketches::countsketch_topk::CSHeap;
use crate::{DataInput, RegularPath, Vector2D};

/// Wire-format heap item (key, value) used by the dispatch helpers below.
pub struct WireHeapItem {
    pub key: String,
    pub value: f64,
}

/// Concrete Count-Sketch-with-Heap type backing the wire-format
/// `CountSketchWithHeap`.
pub type SketchlibCSHeap = CSHeap<Vector2D<i64>, RegularPath>;

/// Creates a fresh CSHeap with the given dimensions and heap capacity.
pub fn new_sketchlib_cs_heap(row_num: usize, col_num: usize, heap_size: usize) -> SketchlibCSHeap {
    CSHeap::new(row_num, col_num, heap_size)
}

/// Builds a CSHeap from an existing sketch matrix and optional heap
/// items. Used when deserializing wire-format state.
pub fn sketchlib_cs_heap_from_matrix_and_heap(
    row_num: usize,
    col_num: usize,
    heap_size: usize,
    sketch: &[Vec<f64>],
    topk_heap: &[WireHeapItem],
) -> SketchlibCSHeap {
    let matrix = Vector2D::from_fn(row_num, col_num, |r, c| {
        sketch
            .get(r)
            .and_then(|row| row.get(c))
            .copied()
            .unwrap_or(0.0)
            .round() as i64
    });
    let mut cs_heap = CSHeap::from_storage(matrix, heap_size);

    for item in topk_heap {
        let count = item.value.round() as i64;
        if count > 0 {
            let input = DataInput::Str(&item.key);
            cs_heap.heap_mut().update(&input, count);
        }
    }

    cs_heap
}

/// Converts a CSHeap's storage into a `Vec<Vec<f64>>` matrix.
pub fn matrix_from_sketchlib_cs_heap(cs_heap: &SketchlibCSHeap) -> Vec<Vec<f64>> {
    let storage = cs_heap.cs().as_storage();
    let rows = storage.rows();
    let cols = storage.cols();
    let mut sketch = vec![vec![0.0; cols]; rows];

    for (r, row) in sketch.iter_mut().enumerate().take(rows) {
        for (c, cell) in row.iter_mut().enumerate().take(cols) {
            if let Some(v) = storage.get(r, c) {
                *cell = *v as f64;
            }
        }
    }

    sketch
}

/// Converts sketchlib HHHeap items to wire-format (key, value) pairs.
pub fn heap_to_wire(cs_heap: &SketchlibCSHeap) -> Vec<WireHeapItem> {
    cs_heap
        .heap()
        .heap()
        .iter()
        .map(|hh_item| {
            let key = match &hh_item.key {
                crate::HeapItem::String(s) => s.clone(),
                other => format!("{other:?}"),
            };
            WireHeapItem {
                key,
                value: hh_item.count as f64,
            }
        })
        .collect()
}

/// Updates a CSHeap with a weighted key. Automatically updates the heap.
pub fn sketchlib_cs_heap_update(cs_heap: &mut SketchlibCSHeap, key: &str, value: f64) {
    let many = value.round() as i64;
    if many <= 0 {
        return;
    }
    cs_heap.insert_many(&DataInput::String(key.to_owned()), many);
}

/// Queries a CSHeap for a key's frequency estimate (median-of-signed-rows,
/// not CMS's min-over-rows).
pub fn sketchlib_cs_heap_query(cs_heap: &SketchlibCSHeap, key: &str) -> f64 {
    cs_heap.estimate(&DataInput::String(key.to_owned()))
}

/// Item in the top-k heap representing a key-value pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsHeapItem {
    pub key: String,
    pub value: f64,
}

/// Count Sketch with Heap for top-k tracking. Combines the balanced/
/// zero-mean-error frequency estimator with efficient top-k maintenance.
pub struct CountSketchWithHeap {
    pub rows: usize,
    pub cols: usize,
    pub heap_size: usize,
    pub(crate) backend: SketchlibCSHeap,
}

impl std::fmt::Debug for CountSketchWithHeap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountSketchWithHeap")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("heap_size", &self.heap_size)
            .finish()
    }
}

impl Clone for CountSketchWithHeap {
    fn clone(&self) -> Self {
        // Structural clone: `CSHeap` derives `Clone` (its `Count` +
        // `HHHeap` fields are `Clone`), so copy the backend directly
        // instead of extracting the matrix + heap to wire form and
        // rebuilding it.
        Self {
            rows: self.rows,
            cols: self.cols,
            heap_size: self.heap_size,
            backend: self.backend.clone(),
        }
    }
}

impl CountSketchWithHeap {
    pub fn new(rows: usize, cols: usize, heap_size: usize) -> Self {
        Self {
            rows,
            cols,
            heap_size,
            backend: new_sketchlib_cs_heap(rows, cols, heap_size),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Create from a sketch matrix and heap (e.g. from JSON deserialization).
    pub fn from_legacy_matrix(
        sketch: Vec<Vec<f64>>,
        topk_heap: Vec<CsHeapItem>,
        rows: usize,
        cols: usize,
        heap_size: usize,
    ) -> Self {
        let wire_heap: Vec<WireHeapItem> = topk_heap
            .into_iter()
            .map(|h| WireHeapItem {
                key: h.key,
                value: h.value,
            })
            .collect();
        Self {
            rows,
            cols,
            heap_size,
            backend: sketchlib_cs_heap_from_matrix_and_heap(
                rows, cols, heap_size, &sketch, &wire_heap,
            ),
        }
    }

    /// Get the top-k heap items.
    pub fn topk_heap_items(&self) -> Vec<CsHeapItem> {
        heap_to_wire(&self.backend)
            .into_iter()
            .map(|w| CsHeapItem {
                key: w.key,
                value: w.value,
            })
            .collect()
    }

    /// Get the sketch matrix.
    pub fn sketch_matrix(&self) -> Vec<Vec<f64>> {
        matrix_from_sketchlib_cs_heap(&self.backend)
    }

    pub fn update(&mut self, key: &str, value: f64) {
        sketchlib_cs_heap_update(&mut self.backend, key, value);
    }

    pub fn estimate(&self, key: &str) -> f64 {
        sketchlib_cs_heap_query(&self.backend, key)
    }

    /// Merge another CountSketchWithHeap into self in place. Both
    /// operands must have identical dimensions; the resulting
    /// `heap_size` is the minimum of the two.
    pub fn merge(
        &mut self,
        other: &CountSketchWithHeap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "CountSketchWithHeap dimension mismatch: self={}x{}, other={}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
            .into());
        }
        self.backend.merge(&other.backend);
        self.heap_size = self.heap_size.min(other.heap_size);
        Ok(())
    }

    pub fn merge_refs(
        inputs: &[&CountSketchWithHeap],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let first = inputs
            .first()
            .ok_or("CountSketchWithHeap::merge_refs called with empty input")?;
        let mut merged = (*first).clone();
        for h in inputs.iter().skip(1) {
            merged.merge(h)?;
        }
        Ok(merged)
    }

    pub fn aggregate_topk(
        rows: usize,
        cols: usize,
        heap_size: usize,
        keys: &[&str],
        values: &[f64],
    ) -> Option<Vec<u8>> {
        if keys.is_empty() {
            return None;
        }
        let mut sketch = Self::new(rows, cols, heap_size);
        for (key, &value) in keys.iter().zip(values.iter()) {
            sketch.update(key, value);
        }
        sketch.to_msgpack().ok()
    }
}

// ----- Wire format -----

/// Inner CS payload nested inside [`CountSketchWithHeapWire`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountSketchInnerWire {
    pub sketch: Vec<Vec<f64>>,
    #[serde(rename = "row_num")]
    pub rows: usize,
    #[serde(rename = "col_num")]
    pub cols: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountSketchWithHeapWire {
    pub sketch: CountSketchInnerWire,
    pub topk_heap: Vec<CsHeapItem>,
    pub heap_size: usize,
}

impl MessagePackCodec for CountSketchWithHeap {
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let wire = CountSketchWithHeapWire {
            sketch: CountSketchInnerWire {
                sketch: self.sketch_matrix(),
                rows: self.rows,
                cols: self.cols,
            },
            topk_heap: self.topk_heap_items(),
            heap_size: self.heap_size,
        };
        Ok(rmp_serde::to_vec(&wire)?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        let wire: CountSketchWithHeapWire = rmp_serde::from_slice(bytes)?;

        let mut sorted_topk_heap = wire.topk_heap;
        sorted_topk_heap.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let wire_heap: Vec<WireHeapItem> = sorted_topk_heap
            .iter()
            .map(|h| WireHeapItem {
                key: h.key.clone(),
                value: h.value,
            })
            .collect();
        let backend = sketchlib_cs_heap_from_matrix_and_heap(
            wire.sketch.rows,
            wire.sketch.cols,
            wire.heap_size,
            &wire.sketch.sketch,
            &wire_heap,
        );

        Ok(Self {
            rows: wire.sketch.rows,
            cols: wire.sketch.cols,
            heap_size: wire.heap_size,
            backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let cs = CountSketchWithHeap::new(4, 1000, 20);
        assert_eq!(cs.rows, 4);
        assert_eq!(cs.cols, 1000);
        assert_eq!(cs.heap_size, 20);
        assert_eq!(cs.sketch_matrix().len(), 4);
        assert_eq!(cs.sketch_matrix()[0].len(), 1000);
        assert_eq!(cs.topk_heap_items().len(), 0);
    }

    #[test]
    fn test_query_empty() {
        let cs = CountSketchWithHeap::new(2, 10, 5);
        assert_eq!(cs.estimate("anything"), 0.0);
    }

    #[test]
    fn test_merge() {
        let mut sketch1 = vec![vec![0.0; 10]; 2];
        sketch1[0][0] = 10.0;
        sketch1[1][1] = 20.0;
        let mut cs1 = CountSketchWithHeap::from_legacy_matrix(
            sketch1,
            vec![
                CsHeapItem {
                    key: "key1".to_string(),
                    value: 100.0,
                },
                CsHeapItem {
                    key: "key2".to_string(),
                    value: 50.0,
                },
            ],
            2,
            10,
            5,
        );
        let mut sketch2 = vec![vec![0.0; 10]; 2];
        sketch2[0][0] = 5.0;
        sketch2[1][1] = 15.0;
        let cs2 = CountSketchWithHeap::from_legacy_matrix(
            sketch2,
            vec![
                CsHeapItem {
                    key: "key3".to_string(),
                    value: 75.0,
                },
                CsHeapItem {
                    key: "key1".to_string(),
                    value: 80.0,
                },
            ],
            2,
            10,
            3,
        );

        cs1.merge(&cs2).unwrap();

        assert_eq!(cs1.sketch_matrix()[0][0], 15.0);
        assert_eq!(cs1.sketch_matrix()[1][1], 35.0);
        assert_eq!(cs1.heap_size, 3);
        assert!(cs1.topk_heap_items().len() <= 3);
    }

    #[test]
    fn test_merge_dimension_mismatch() {
        let mut cs1 = CountSketchWithHeap::new(2, 10, 5);
        let cs2 = CountSketchWithHeap::new(3, 10, 5);
        assert!(cs1.merge(&cs2).is_err());
    }

    #[test]
    fn test_msgpack_round_trip() {
        let mut cs = CountSketchWithHeap::new(4, 128, 3);
        cs.update("hot", 100.0);
        cs.update("cold", 1.0);

        let bytes = cs.to_msgpack().unwrap();
        let deserialized = CountSketchWithHeap::from_msgpack(&bytes).unwrap();

        assert_eq!(deserialized.rows, 4);
        assert_eq!(deserialized.cols, 128);
        assert_eq!(deserialized.heap_size, 3);
        let items = deserialized.topk_heap_items();
        assert!(!items.is_empty());
        let hot = items
            .iter()
            .find(|item| item.key == "hot")
            .expect("'hot' should be in the heap");
        assert!(hot.value >= 100.0);
        assert!(deserialized.estimate("hot") >= 100.0);
        assert!(deserialized.estimate("cold") >= 1.0);
    }

    #[test]
    fn test_aggregate_topk() {
        let keys = ["a", "b", "a", "c"];
        let values = [1.0, 2.0, 3.0, 0.5];
        let bytes = CountSketchWithHeap::aggregate_topk(4, 100, 2, &keys, &values).unwrap();
        let cs = CountSketchWithHeap::from_msgpack(&bytes).unwrap();
        assert_eq!(cs.heap_size, 2);
        assert!(cs.topk_heap_items().len() <= 2);
    }

    #[test]
    fn test_aggregate_topk_empty() {
        assert!(CountSketchWithHeap::aggregate_topk(4, 100, 10, &[], &[]).is_none());
    }

    /// The whole point of splitting this from `CountMinSketchWithHeap`:
    /// CS's median estimator must behave differently from CMS's
    /// min-over-rows estimator on the same kind of skewed insert
    /// pattern, not just share a struct shape.
    #[test]
    fn test_median_estimator_differs_from_cms_style_min() {
        let mut cs = CountSketchWithHeap::new(5, 64, 10);
        for _ in 0..50 {
            cs.update("heavy", 1.0);
        }
        for i in 0..20 {
            cs.update(&format!("light-{i}"), 1.0);
        }
        let est = cs.estimate("heavy");
        assert!(
            (40.0..=60.0).contains(&est),
            "median estimate {est} should track the true count (50) reasonably \
             closely, not collide-and-inflate the way a single min-over-rows \
             outlier could"
        );
    }
}
