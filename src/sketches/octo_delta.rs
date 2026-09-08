//! Delta entry types for OctoSketch-style multi-threaded sketch updates.
//!
//! Each delta represents an accumulated counter change emitted by a child
//! worker sketch when a local counter crosses the promotion threshold τ
//! (Algorithm 1 of the OctoSketch paper, NSDI '24).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::HeapItem;

/// Default promotion threshold τ for Count-Min workers.
pub const CM_PROMASK: u32 = 0x1f;

/// Default promotion threshold τ for Count sketch workers, applied to `|counter|`.
pub const COUNT_PROMASK: u32 = 0x1f;

/// Default promotion threshold τ for DDSketch workers.
///
/// A bucket histogram spreads a stream over many buckets, and a bucket that
/// never reaches τ never reaches the aggregator at all - so the sparse tail
/// disappears rather than merely lagging. DDSketch therefore wants a much
/// lower threshold than a Count-Min row, whose counters are dense and whose
/// query is a minimum. Size it against the samples-per-bucket you expect.
pub const DD_PROMASK: u32 = 4;

/// Default base promotion threshold τ for UnivMon workers.
///
/// Higher than a plain Count sketch's because a UnivMon insert touches several
/// layers and the aggregator does the heavy-hitter work the workers dropped,
/// which makes it the pipeline's ceiling. Measured with
/// `cargo run --release --example octo_throughput_probe`, the sustainable rate
/// on a 2M-insert Zipf stream runs 0.41 M/s at τ=31 and 1.00 M/s at τ=64 for
/// the same 0.02% gap to a single-threaded sketch, so 31 is simply dominated.
pub const UNIVMON_PROMASK: u32 = 64;

/// Default HLL promotion threshold τ: 0 promotes every register improvement.
pub const HLL_PROMASK: u8 = 0;

/// Default promotion threshold τ for CocoSketch workers.
///
/// A Coco bucket counter is exactly as dense as a Count-Min cell - every
/// insert lands on one of them and increments it by one - so it takes the same
/// τ.
pub const COCO_PROMASK: u32 = 0x1f;

/// Default promotion threshold τ for Elastic sketch workers.
///
/// One τ covers both halves: a heavy-part vote counter and a light-part
/// Count-Min counter each advance by one per insert.
pub const ELASTIC_PROMASK: u32 = 0x1f;

/// Largest threshold a worker may be configured with.
///
/// Counter storage in the compact worker sketches is one byte wide, and the
/// Count-sketch and UnivMon workers hold *signed* counters, so a counter must
/// stay representable in `i8` right up to the promotion that clears it. The
/// ceiling is the same for every sketch on purpose: one shared `OctoThreshold`
/// serves workers of different kinds, and a τ that silently meant 200 in one
/// and 127 in another would put a dead band in the aggregator's controller.
pub const MAX_PROMASK: u32 = i8::MAX as u32;

/// Delta emitted by a CountMin child worker.
/// Represents an accumulated unsigned count for a single cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CmDelta {
    /// Row index of the updated cell.
    pub row: u32,
    /// Column index of the updated cell.
    pub col: u32,
    /// Accumulated delta for the cell.
    pub value: u32,
}

/// Delta emitted by a Count sketch child worker.
/// Represents a signed accumulated count for a single cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountDelta {
    /// Row index of the updated cell.
    pub row: u32,
    /// Column index of the updated cell.
    pub col: u32,
    /// Signed accumulated delta for the cell.
    pub value: i32,
}

/// Delta emitted by an HLL child worker.
/// Represents a register improvement (max-register semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HllDelta {
    /// Register position to update.
    pub pos: u32,
    /// New register value.
    pub value: u8,
}

/// Delta emitted by a DDSketch child worker for one bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdDelta {
    /// Absolute bucket index, as used by `DDSketch::store_offset`.
    pub index: i32,
    /// Accumulated count for the bucket.
    pub value: u64,
}

/// A `CmDelta` carrying the flow key that produced it.
///
/// The paper's worker-to-aggregator message is a 4-tuple `<flow key, row,
/// column, value>`; the key lets the aggregator maintain the heavy-hitter heap
/// that workers no longer keep (§3.2, Idea 3).
#[derive(Clone, Debug, PartialEq)]
pub struct KeyedCmDelta {
    /// Flow key whose insertion triggered the promotion.
    pub key: HeapItem,
    /// The promoted cell update.
    pub delta: CmDelta,
}

/// A `CountDelta` carrying the flow key that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyedCountDelta {
    /// Flow key whose insertion triggered the promotion.
    pub key: HeapItem,
    /// The promoted cell update.
    pub delta: CountDelta,
}

/// Delta emitted by a CocoSketch child worker: a whole bucket.
///
/// §4.4, "Handling counters with flow keys": a sketch that keeps a flow key
/// beside every counter ships the key together with the counter, and the
/// aggregator replays the pair through the parent's own insertion logic. There
/// is no cell index because the parent re-derives one - its victim choice
/// depends on what its own buckets hold, not on the worker's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CocoDelta {
    /// Key the promoted bucket holds.
    pub key: String,
    /// Mass attributed to that key since the bucket last promoted.
    pub value: u64,
}

/// Delta emitted by an Elastic sketch child worker.
///
/// Appendix C keeps both halves in the worker and promotes them differently:
/// the heavy part ships `<key, votes>` the way CocoSketch does, while the light
/// part is a Count-Min and ships the ordinary cell delta.
///
/// Two of the three variants go beyond §4.4's `<key, counter>`, because this
/// crate implements the Elastic Sketch of the original paper, whose heavy
/// bucket carries an eviction flag that §4.4's rule has no way to move. The
/// flag rides with the counter word it qualifies, and the evicted resident
/// travels under its own key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElasticDelta {
    /// Heavy-part bucket: the resident flow, the votes it accumulated, and the
    /// bucket's eviction flag.
    Heavy {
        /// Flow the promoted bucket holds.
        key: String,
        /// Votes recorded for it since the bucket last promoted.
        value: u32,
        /// Whether the worker's bucket holds this flow by eviction, so part of
        /// the flow's mass may sit in a light layer.
        eviction: bool,
    },
    /// A resident evicted from a worker's heavy bucket, handed to the light
    /// part under its own key rather than folded into an unkeyed cell delta.
    ///
    /// Emitted on every worker eviction, `votes` of zero included: a promotion
    /// zeroes the counter while the flow stays resident, so a later eviction
    /// carries nothing but still tells the aggregator that the flow's remaining
    /// mass will arrive through the light layer.
    Evicted {
        /// The evicted flow.
        key: String,
        /// Votes it held when it was evicted.
        votes: u32,
    },
    /// Light-part cell, promoted exactly as a Count-Min worker's would be.
    /// Carries the arrivals that lose a bucket contest, one at a time.
    Light(CmDelta),
}

/// A `CountDelta` tagged with the UnivMon pyramid layer it belongs to.
#[derive(Clone, Debug, PartialEq)]
pub struct LayeredCountDelta {
    /// Pyramid layer index.
    pub layer: u32,
    /// Flow key whose insertion triggered the promotion.
    pub key: HeapItem,
    /// The promoted cell update.
    pub delta: CountDelta,
    /// Which worker emitted this, so an aggregator can keep per-worker totals.
    pub worker_id: u32,
    /// Total weight the emitting worker has seen so far.
    ///
    /// UnivMon divides by this in its g-sum queries, and it cannot be recovered
    /// from thresholded counter deltas. Riding along on a message the worker is
    /// already sending keeps the aggregator's copy current without adding one.
    pub weight_total: u64,
}

/// The promotion threshold τ, shared by every worker and adjustable at runtime.
///
/// The paper keeps τ in one atomic that the aggregator raises or lowers to
/// match its receive rate against the workers' send rate (§4.3).
#[derive(Clone, Debug)]
pub struct OctoThreshold(Arc<AtomicU32>);

impl OctoThreshold {
    /// Creates a shared threshold, clamped to `1..=MAX_PROMASK`.
    pub fn new(tau: u32) -> Self {
        Self(Arc::new(AtomicU32::new(tau.clamp(1, MAX_PROMASK))))
    }

    /// Reads the current threshold.
    #[inline(always)]
    pub fn get(&self) -> u32 {
        self.0.load(Ordering::Relaxed)
    }

    /// Replaces the threshold, clamped to `1..=MAX_PROMASK`.
    pub fn set(&self, tau: u32) {
        self.0.store(tau.clamp(1, MAX_PROMASK), Ordering::Relaxed);
    }

    /// Adds `step` to the threshold, saturating at `MAX_PROMASK`.
    pub fn increase(&self, step: u32) {
        self.set(self.get().saturating_add(step));
    }

    /// Subtracts `step` from the threshold, saturating at 1.
    pub fn decrease(&self, step: u32) {
        self.set(self.get().saturating_sub(step).max(1));
    }
}

impl Default for OctoThreshold {
    fn default() -> Self {
        Self::new(CM_PROMASK)
    }
}
