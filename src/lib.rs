#![doc = include_str!("../README.md")]
//! `asap_sketchlib` is organized into the following layers:
//!
//! - [`sketches`]: core sketch data structures such as Count-Min, HyperLogLog,
//!   KLL, and DDSketch.
//! - [`common`]: shared input types, hashing abstractions, storage backends, and
//!   reusable utilities used across sketches.
//! - [`proto`]: portable protobuf message types for sketch interchange.
//! - [`sketch_framework`]: higher-level composition layers such as Hydra,
//!   UnivMon, tumbling windows, and batch/parallel execution helpers.
//! - [`message_pack_format`]: MessagePack/proto wire format shared with
//!   `sketchlib-go`. Owns the wire-DTO sketch types consumed by the ASAP
//!   query engine (`CountMinSketch`, `CountSketch`, `KllSketch`, `HllSketch`,
//!   `DdSketch`, `HydraKllSketch`, `CountMinSketchWithHeap`,
//!   `SetAggregator`, `DeltaResult`).
//!
//! Most users can start with the crate-root re-exports such as [`DataInput`],
//! [`CountMin`], [`HyperLogLog`], [`KLL`], and [`DDSketch`]. Reach for the
//! submodules directly when you need lower-level storage, hashing, or
//! framework-specific APIs.

/// Shared primitives used across sketches, including input wrappers, hashers,
/// storage backends, and reusable utilities.
pub mod common;
pub mod message_pack_format;
/// Portable protobuf message types for sketch interchange.
pub mod proto;
/// Higher-level composition layers such as Hydra, UnivMon, tumbling windows,
/// and batch/parallel execution helpers.
pub mod sketch_framework;
/// Core sketch implementations such as Count-Min, HyperLogLog, KLL, and DDSketch.
pub mod sketches;
#[cfg(test)]
pub mod test_utils;

#[doc(hidden)]
pub mod __private {
    pub use serde;
    pub use serde_big_array;
}

pub use common::*;
pub use message_pack_format::MessagePackCodec;
pub use message_pack_format::portable::countminsketch::{CountMinSketch, CountMinSketchDelta};
pub use message_pack_format::portable::countminsketch_topk::{CmsHeapItem, CountMinSketchWithHeap};
pub use message_pack_format::portable::countsketch::{
    COUNT_SKETCH_TOPK_CAPACITY, CountSketch, CountSketchDelta,
};
pub use message_pack_format::portable::countsketch_topk::{CountSketchWithHeap, CsHeapItem};
pub use message_pack_format::portable::ddsketch::{DDSKETCH_GROW_CHUNK, DdSketch, DdSketchDelta};
pub use message_pack_format::portable::delta_set_aggregator::DeltaResult;
pub use message_pack_format::portable::hll::{HllSketch, HllSketchDelta, HllVariant};
pub use message_pack_format::portable::hydra_kll::HydraKllSketch;
pub use message_pack_format::portable::kll::{KllSketch, KllSketchData};
pub use message_pack_format::portable::sampling::{
    effective_sample_p, is_quantile_scale_invariant, rescale_count, rescale_count_with_env,
    sample_p_or_default,
};
pub use message_pack_format::portable::set_aggregator::SetAggregator;
pub use sketch_framework::*;
pub use sketches::*;
