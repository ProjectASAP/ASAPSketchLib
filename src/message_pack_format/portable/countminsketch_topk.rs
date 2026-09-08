//! Wire-format-aligned Count-Min sketch + top-k heap composite.
//!
//! No `sketches::*` equivalent — this combines `sketches::CMSHeap`
//! with the wire-format `CountMinSketch` shape and exposes it as a
//! single Go-interop type.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error as MsgPackError, MessagePackCodec};
use crate::sketches::countminsketch_topk::CMSHeap;
use crate::{DataInput, RegularPath, Vector2D};

/// Wire-format heap item (key, value) used by the dispatch helpers below.
pub struct WireHeapItem {
    pub key: String,
    pub value: f64,
}

/// Concrete Count-Min-with-Heap type backing the wire-format
/// `CountMinSketchWithHeap`.
pub type SketchlibCMSHeap = CMSHeap<Vector2D<i64>, RegularPath>;

/// Creates a fresh CMSHeap with the given dimensions and heap capacity.
pub fn new_sketchlib_cms_heap(
    row_num: usize,
    col_num: usize,
    heap_size: usize,
) -> SketchlibCMSHeap {
    CMSHeap::new(row_num, col_num, heap_size)
}

/// Builds a CMSHeap from an existing sketch matrix and optional heap
/// items. Used when deserializing wire-format state.
pub fn sketchlib_cms_heap_from_matrix_and_heap(
    row_num: usize,
    col_num: usize,
    heap_size: usize,
    sketch: &[Vec<f64>],
    topk_heap: &[WireHeapItem],
) -> SketchlibCMSHeap {
    let matrix = Vector2D::from_fn(row_num, col_num, |r, c| {
        sketch
            .get(r)
            .and_then(|row| row.get(c))
            .copied()
            .unwrap_or(0.0)
            .round() as i64
    });
    let mut cms_heap = CMSHeap::from_storage(matrix, heap_size);

    for item in topk_heap {
        let count = item.value.round() as i64;
        if count > 0 {
            let input = DataInput::Str(&item.key);
            cms_heap.heap_mut().update(&input, count);
        }
    }

    cms_heap
}

/// Converts a CMSHeap's storage into a `Vec<Vec<f64>>` matrix.
pub fn matrix_from_sketchlib_cms_heap(cms_heap: &SketchlibCMSHeap) -> Vec<Vec<f64>> {
    let storage = cms_heap.cms().as_storage();
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
pub fn heap_to_wire(cms_heap: &SketchlibCMSHeap) -> Vec<WireHeapItem> {
    cms_heap
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

/// Updates a CMSHeap with a weighted key. Automatically updates the heap.
pub fn sketchlib_cms_heap_update(cms_heap: &mut SketchlibCMSHeap, key: &str, value: f64) {
    let many = value.round() as i64;
    if many <= 0 {
        return;
    }
    cms_heap.insert_many(&DataInput::String(key.to_owned()), many);
}

/// Queries a CMSHeap for a key's frequency estimate.
pub fn sketchlib_cms_heap_query(cms_heap: &SketchlibCMSHeap, key: &str) -> f64 {
    cms_heap.estimate(&DataInput::String(key.to_owned())) as f64
}

/// Item in the top-k heap representing a key-value pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmsHeapItem {
    pub key: String,
    pub value: f64,
}

/// Count-Min Sketch with Heap for top-k tracking. Combines probabilistic
/// frequency counting with efficient top-k maintenance.
pub struct CountMinSketchWithHeap {
    pub rows: usize,
    pub cols: usize,
    pub heap_size: usize,
    pub(crate) backend: SketchlibCMSHeap,
}

impl std::fmt::Debug for CountMinSketchWithHeap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountMinSketchWithHeap")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("heap_size", &self.heap_size)
            .finish()
    }
}

impl Clone for CountMinSketchWithHeap {
    fn clone(&self) -> Self {
        // Structural clone: `CMSHeap` derives `Clone` (its `CountMin` + `HHHeap`
        // fields are `Clone`), so copy the backend directly instead of
        // extracting the matrix + heap to wire form and rebuilding it.
        Self {
            rows: self.rows,
            cols: self.cols,
            heap_size: self.heap_size,
            backend: self.backend.clone(),
        }
    }
}

impl CountMinSketchWithHeap {
    pub fn new(rows: usize, cols: usize, heap_size: usize) -> Self {
        Self {
            rows,
            cols,
            heap_size,
            backend: new_sketchlib_cms_heap(rows, cols, heap_size),
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
        topk_heap: Vec<CmsHeapItem>,
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
            backend: sketchlib_cms_heap_from_matrix_and_heap(
                rows, cols, heap_size, &sketch, &wire_heap,
            ),
        }
    }

    /// Get the top-k heap items.
    pub fn topk_heap_items(&self) -> Vec<CmsHeapItem> {
        heap_to_wire(&self.backend)
            .into_iter()
            .map(|w| CmsHeapItem {
                key: w.key,
                value: w.value,
            })
            .collect()
    }

    /// Get the sketch matrix.
    pub fn sketch_matrix(&self) -> Vec<Vec<f64>> {
        matrix_from_sketchlib_cms_heap(&self.backend)
    }

    pub fn update(&mut self, key: &str, value: f64) {
        sketchlib_cms_heap_update(&mut self.backend, key, value);
    }

    pub fn estimate(&self, key: &str) -> f64 {
        sketchlib_cms_heap_query(&self.backend, key)
    }

    /// Merge another CountMinSketchWithHeap into self in place. Both
    /// operands must have identical dimensions; the resulting `heap_size`
    /// is the minimum of the two.
    pub fn merge(
        &mut self,
        other: &CountMinSketchWithHeap,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(format!(
                "CountMinSketchWithHeap dimension mismatch: self={}x{}, other={}x{}",
                self.rows, self.cols, other.rows, other.cols
            )
            .into());
        }
        self.backend.merge(&other.backend);
        self.heap_size = self.heap_size.min(other.heap_size);
        Ok(())
    }

    pub fn merge_refs(
        inputs: &[&CountMinSketchWithHeap],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let first = inputs
            .first()
            .ok_or("CountMinSketchWithHeap::merge_refs called with empty input")?;
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

/// Inner CMS payload nested inside [`CountMinSketchWithHeapWire`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountMinSketchInnerWire {
    pub sketch: Vec<Vec<f64>>,
    #[serde(rename = "row_num")]
    pub rows: usize,
    #[serde(rename = "col_num")]
    pub cols: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountMinSketchWithHeapWire {
    pub sketch: CountMinSketchInnerWire,
    pub topk_heap: Vec<CmsHeapItem>,
    pub heap_size: usize,
}

impl MessagePackCodec for CountMinSketchWithHeap {
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let wire = CountMinSketchWithHeapWire {
            sketch: CountMinSketchInnerWire {
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
        let wire: CountMinSketchWithHeapWire = rmp_serde::from_slice(bytes)?;

        let mut sorted_topk_heap = wire.topk_heap;
        sorted_topk_heap.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let wire_heap: Vec<WireHeapItem> = sorted_topk_heap
            .iter()
            .map(|h| WireHeapItem {
                key: h.key.clone(),
                value: h.value,
            })
            .collect();
        let backend = sketchlib_cms_heap_from_matrix_and_heap(
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
        let cms = CountMinSketchWithHeap::new(4, 1000, 20);
        assert_eq!(cms.rows, 4);
        assert_eq!(cms.cols, 1000);
        assert_eq!(cms.heap_size, 20);
        assert_eq!(cms.sketch_matrix().len(), 4);
        assert_eq!(cms.sketch_matrix()[0].len(), 1000);
        assert_eq!(cms.topk_heap_items().len(), 0);
    }

    #[test]
    fn test_query_empty() {
        let cms = CountMinSketchWithHeap::new(2, 10, 5);
        assert_eq!(cms.estimate("anything"), 0.0);
    }

    #[test]
    fn test_merge() {
        let mut sketch1 = vec![vec![0.0; 10]; 2];
        sketch1[0][0] = 10.0;
        sketch1[1][1] = 20.0;
        let mut cms1 = CountMinSketchWithHeap::from_legacy_matrix(
            sketch1,
            vec![
                CmsHeapItem {
                    key: "key1".to_string(),
                    value: 100.0,
                },
                CmsHeapItem {
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
        let cms2 = CountMinSketchWithHeap::from_legacy_matrix(
            sketch2,
            vec![
                CmsHeapItem {
                    key: "key3".to_string(),
                    value: 75.0,
                },
                CmsHeapItem {
                    key: "key1".to_string(),
                    value: 80.0,
                },
            ],
            2,
            10,
            3,
        );

        cms1.merge(&cms2).unwrap();

        assert_eq!(cms1.sketch_matrix()[0][0], 15.0);
        assert_eq!(cms1.sketch_matrix()[1][1], 35.0);
        assert_eq!(cms1.heap_size, 3);
        assert!(cms1.topk_heap_items().len() <= 3);
    }

    #[test]
    fn test_merge_dimension_mismatch() {
        let mut cms1 = CountMinSketchWithHeap::new(2, 10, 5);
        let cms2 = CountMinSketchWithHeap::new(3, 10, 5);
        assert!(cms1.merge(&cms2).is_err());
    }

    #[test]
    fn test_msgpack_round_trip() {
        let mut cms = CountMinSketchWithHeap::new(4, 128, 3);
        cms.update("hot", 100.0);
        cms.update("cold", 1.0);

        let bytes = cms.to_msgpack().unwrap();
        let deserialized = CountMinSketchWithHeap::from_msgpack(&bytes).unwrap();

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
        let bytes = CountMinSketchWithHeap::aggregate_topk(4, 100, 2, &keys, &values).unwrap();
        let cms = CountMinSketchWithHeap::from_msgpack(&bytes).unwrap();
        assert_eq!(cms.heap_size, 2);
        assert!(cms.topk_heap_items().len() <= 2);
    }

    #[test]
    fn test_aggregate_topk_empty() {
        assert!(CountMinSketchWithHeap::aggregate_topk(4, 100, 10, &[], &[]).is_none());
    }
}
