//! Core sketch implementations.
//!
//! This module is the main entry point for the library's sketch algorithms.
//! Broadly, the sketches fall into a few common use cases:
//!
//! - frequency estimation: [`CountMin`], [`Count`]
//! - cardinality estimation: [`HyperLogLog`]
//! - quantiles and distributions: [`KLL`], [`DDSketch`]
//! - specialized or composite structures: heap-backed, folded, and
//!   runtime-oriented variants such as [`CMSHeap`], [`CSHeap`], [`FoldCMS`], and
//!   [`FoldCS`]
//!
//! For most users:
//!
//! - choose [`CountMin`] for fast approximate non-negative frequency queries
//! - choose [`Count`] for Count Sketch style frequency estimation
//! - choose [`HyperLogLog`] for distinct counts
//! - choose [`KLL`] for general quantile estimation
//! - choose [`DDSketch`] when relative-error quantiles are important
//!
//! [`FastPath`] and [`RegularPath`] control how some matrix-backed frequency
//! sketches map values to rows and columns. Several specialized sketches in
//! this module are feature-gated behind `experimental`.
//!
//! ## `*sketch` vs `*sketch_topk` files
//!
//! For each Count-Min / Count Sketch style algorithm we ship two files:
//!
//! - [`countminsketch`] / [`countsketch`] — the core sketch (matrix of
//!   counters only). Answers point-frequency queries: "how often did key `k`
//!   appear?" These do not track which keys are heavy hitters.
//! - [`countminsketch_topk`] / [`countsketch_topk`] — the same sketch paired
//!   with a min-heap that tracks the top-`k` heavy hitters as items stream in
//!   ([`CMSHeap`], [`CSHeap`] / [`CountL2HH`]). Use these when the question is
//!   "which keys are the most frequent?" rather than "how often did this
//!   specific key appear?"
//!
//! The split keeps the core sketch lean for callers that only need point
//! queries, while the `_topk` variants compose the sketch with heap
//! bookkeeping for heavy-hitter workloads.

pub mod coco;
pub use coco::Coco;
pub use coco::CocoBucket;

/// Fixed-counter heavy-hitter tracking over a Stream-Summary.
pub mod space_saving;
pub use space_saving::{SPACE_SAVING_DEFAULT_CAPACITY, SpaceSaving};

/// Approximate set membership over a packed bit grid.
pub mod bloom;
pub use bloom::{
    BLOOM_DEFAULT_COLS, BLOOM_DEFAULT_ROWS, BLOOM_MAX_BITS, BLOOM_MAX_SLICES, Bloom, BloomMode,
};

pub mod countsketch;
pub use countsketch::Count;

/// Count Sketch grid with a per-bucket HyperLogLog (distinct-count) sketch.
pub mod countsketch_hll;
pub use countsketch_hll::CountHll;

/// Hashing path markers for matrix-backed sketches.
pub mod mode;
pub use mode::{FastPath, RegularPath};

pub mod countminsketch;
pub use crate::MatrixStorage;
pub use countminsketch::{CountMin, QUICKSTART_COL_NUM, QUICKSTART_ROW_NUM};

pub mod elastic;
pub use elastic::Elastic;
pub use elastic::HeavyBucket;

/// HyperLogLog implementations and aliases.
pub mod hll;
pub use hll::{
    Classic, ErtlMLE, HyperLogLog, HyperLogLogHIP, HyperLogLogHIPP12, HyperLogLogHIPP14,
    HyperLogLogHIPP16, HyperLogLogP12, HyperLogLogP14, HyperLogLogP16,
};

pub mod kll;
pub use kll::KLL;

pub mod kll_dynamic;
pub use kll_dynamic::KLLDynamic;

#[cfg(feature = "experimental")]
pub mod kmv;
#[cfg(feature = "experimental")]
pub use kmv::KMV;

#[cfg(feature = "experimental")]
pub mod uniform;
#[cfg(feature = "experimental")]
pub use uniform::UniformSampling;

pub mod ddsketch;
pub use ddsketch::DDSketch;

pub mod countminsketch_topk;
pub use countminsketch_topk::CMSHeap;

pub mod countsketch_topk;
pub use countsketch_topk::{CSHeap, cs_heap_count};
pub use countsketch_topk::{CountL2HH, l2hh_cell_for_row};

pub mod octo_delta;
pub use octo_delta::{
    CM_PROMASK, COCO_PROMASK, COUNT_PROMASK, CmDelta, CocoDelta, CountDelta, DD_PROMASK, DdDelta,
    ELASTIC_PROMASK, ElasticDelta, HLL_PROMASK, HllDelta, KeyedCmDelta, KeyedCountDelta,
    LayeredCountDelta, MAX_PROMASK, OctoThreshold, UNIVMON_PROMASK,
};

pub mod fold_cms;
pub use fold_cms::{FoldCMS, FoldCell, FoldEntry};

pub mod fold_cs;
pub use fold_cs::FoldCS;
