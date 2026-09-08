//! Higher-level frameworks built on top of the core sketches.
//!
//! These APIs compose, extend, or orchestrate sketches for more advanced
//! workflows:
//!
//! - windowed analytics: [`ExponentialHistogram`], [`TumblingWindow`]
//! - subpopulation queries: [`Hydra`]
//! - universal multi-metric monitoring: [`UnivMon`] and experimental [`UnivMonQ`]
//! - batch update acceleration: [`NitroBatch`]
//! - shared-hash or multi-sketch coordination: [`HashSketchEnsemble`]
//! - runtime and parallel execution helpers: the `octo` family
//!
//! Most users can begin with the core algorithms in [`crate::sketches`]. Reach
//! for this module when you need to add windows, grouped queries, coordinated
//! execution, or higher-level sketch composition.
//!
//! Some parts of this module are feature-gated:
//!
//! - `eh_univ_optimized` requires `experimental`
//! - the `OctoRuntime` execution runtime requires `octo-runtime`

pub mod eh;
pub use eh::EHBucket;
pub use eh::ExponentialHistogram;

/// Sketch-type adapters used by exponential histograms.
pub mod eh_sketch_list;
pub use eh_sketch_list::EHSketchList;
pub use eh_sketch_list::SketchNorm;

pub mod hashlayer;
pub use hashlayer::{EnsembleSketch, HashSketchEnsemble};

pub mod sketch_catalog;

pub mod hydra;
pub use hydra::Hydra;

pub mod univmon;
pub use univmon::{UnivMon, UnivMonDeltaFidelity, bottom_layer_for_hash};

pub mod univmon_q;
pub use univmon_q::{
    OrderedQueryDiagnostics, UnivMonQ, UnivMonQConfig, UnivMonQError, UnivMonQPoint, UnivMonQQuery,
};

pub mod univmon_optimized;
pub use univmon_optimized::{UnivMonPyramid, UnivSketchPool};

pub mod nitro;
pub use nitro::{
    NitroBatch, NitroEstimate, NitroTarget, nitro_delta_saturated_i32, nitro_delta_saturated_u32,
};

#[cfg(feature = "experimental")]
pub mod eh_univ_optimized;
#[cfg(feature = "experimental")]
pub use eh_univ_optimized::{EHMapBucket, EHUnivMonBucket, EHUnivOptimized, EHUnivQueryResult};

pub mod octo;
pub use octo::{
    CmOctoAggregator, CmOctoPlan, CmOctoWorker, CmTopKOctoAggregator, CmTopKOctoPlan,
    CmTopKOctoWorker, CmWorkerSketch, CocoOctoAggregator, CocoOctoPlan, CocoOctoWorker,
    CocoWorkerSketch, CountOctoAggregator, CountOctoPlan, CountOctoWorker, CountTopKOctoAggregator,
    CountTopKOctoPlan, CountTopKOctoWorker, CountWorkerSketch, DEFAULT_OCTO_TOP_K,
    DdOctoAggregator, DdOctoPlan, DdOctoWorker, DdWorkerSketch, ElasticOctoAggregator,
    ElasticOctoPlan, ElasticOctoWorker, ElasticWorkerSketch, HllOctoAggregator, HllOctoPlan,
    HllOctoWorker, KeyedHashes, L2hhWorkerSketch, OctoAggregator, OctoPlan, OctoWorker, RowHashes,
    UnivMonInput, UnivMonOctoAggregator, UnivMonOctoPlan, UnivMonOctoWorker, flow_key_string,
    max_hll_threshold, threshold_for_error, univmon_layer_threshold,
};
#[cfg(feature = "octo-runtime")]
pub use octo::{
    OctoAdaptiveThreshold, OctoConfig, OctoPartition, OctoReadHandle, OctoResult, OctoRuntime,
    run_octo,
};

pub mod tumbling;
pub use tumbling::{
    FoldCMSConfig, FoldCSConfig, KLLConfig, SketchPool, TumblingWindow, TumblingWindowSketch,
};
