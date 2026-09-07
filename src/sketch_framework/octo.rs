//! OctoSketch multi-threaded sketch framework.
//!
//! Implements the parent-child delta-promotion architecture from OctoSketch (NSDI 2024).
//! Worker threads maintain lightweight child sketches with one-byte counters and emit
//! compact delta entries via per-worker channels when a counter reaches the promotion
//! threshold τ. An aggregator thread applies deltas to a full-precision parent sketch
//! and adjusts τ to keep its receive rate matched to the workers' send rate.
//!
//! The three ideas of §3.2 map onto this module as follows:
//!
//! - Idea 1, change-based updates: [`OctoWorker`] emits one counter at a time
//!   instead of shipping whole sketches.
//! - Idea 2, adaptive resource allocation: [`OctoAdaptiveThreshold`] drives the
//!   shared [`OctoThreshold`] from the aggregator's queue occupancy.
//! - Idea 3, reconstructed data structures: [`CmWorkerSketch`] and
//!   [`CountWorkerSketch`] drop flow-key storage and use one-byte counters,
//!   while the `*TopK*` aggregators hold the only heavy-hitter heap.
//!
//! Reference:
//! - <https://www.usenix.org/conference/nsdi24/presentation/zhang-yinda>

#[cfg(feature = "octo-runtime")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "octo-runtime")]
use std::sync::{Arc, RwLock, Weak};
#[cfg(feature = "octo-runtime")]
use std::thread;
#[cfg(feature = "octo-runtime")]
use std::time::{Duration, Instant};

#[cfg(feature = "octo-runtime")]
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};

use std::collections::HashMap;

use crate::common::input::data_input_to_f64;
use crate::common::structures::matrix_storage::cols_mask_bits;
use crate::octo_delta::{
    DD_PROMASK, DdDelta, KeyedCmDelta, KeyedCountDelta, MAX_PROMASK, OctoThreshold, UNIVMON_PROMASK,
};
use crate::sketch_framework::univmon::{UnivMonDeltaFidelity, bottom_layer_for_hash};
use crate::sketches::coco::Coco;
use crate::sketches::countminsketch_topk::CMSHeap;
use crate::sketches::countsketch_topk::{CSHeap, l2hh_cell_for_row};
use crate::sketches::elastic::{Elastic, LAMBDA};
use crate::{
    BOTTOM_LAYER_FINDER, CM_PROMASK, COUNT_PROMASK, Classic, CmDelta, Count, CountDelta, CountMin,
    DDSketch, DataInput, HLL_PROMASK, HeapItem, HllDelta, HyperLogLog, LayeredCountDelta,
    RegularPath, UnivMon, Vector2D, hash64_seeded, hash128_seeded, heap_item_to_sketch_input,
    input_to_owned,
};
use crate::{CANONICAL_HASH_SEED, COCO_PROMASK, CocoDelta, ELASTIC_PROMASK, ElasticDelta};
use rand::Rng;
use rand::rngs::ThreadRng;
use smallvec::SmallVec;

#[cfg(feature = "octo-runtime")]
/// Legacy queue capacity default retained for config compatibility.
const DEFAULT_QUEUE_CAPACITY: usize = 65536;

/// Default heavy-hitter heap capacity for the `*TopK*` aggregators.
pub const DEFAULT_OCTO_TOP_K: usize = 128;

const LOWER_32_MASK: u64 = (1u64 << 32) - 1;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Per-row hashes for a worker sketch, inline for any realistic row count.
pub type RowHashes = SmallVec<[u64; 8]>;

/// Worker-side trait: processes prepared inputs and emits deltas.
pub trait OctoWorker: Send {
    /// Delta type emitted by the worker.
    ///
    /// Deltas that carry a flow key own it, so this is `Send + 'static` rather
    /// than `Copy`.
    type Delta: Send + 'static;

    /// What actually crosses to the worker thread.
    ///
    /// A `DataInput` can borrow, and a worker runs on another thread, so the
    /// borrow has to end before the hand-off. [`OctoPlan::prepare`] turns the
    /// input into this on the calling thread: hashes for a worker that only
    /// hashes, an owned key for one that must store it. The caller is then free
    /// to drop the key immediately - it never has to outlive the runtime.
    type Payload: Send + 'static;

    /// Process one prepared input and emit zero or more deltas.
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta);

    /// Promote everything still held back, so the aggregator can be queried
    /// against the whole stream so far.
    ///
    /// A worker holds any counter that has not reached τ, and every worker
    /// holds its own, so a shared cell trails by up to `workers · (τ - 1)`.
    /// For Count-Min and Count that is only a low counter, but for a bucket
    /// histogram or a max-register sketch an un-promoted cell is *absent* from
    /// the parent rather than lagging, which is a different kind of wrong.
    /// Flushing costs one message per non-empty cell - as much as shipping the
    /// sketch - so it belongs at the point a stream is handed over for
    /// querying, not on a timer.
    ///
    /// The default does nothing, for workers that hold nothing back.
    fn flush<F>(&mut self, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        let _ = emit;
    }
}

/// Builds a fleet of workers and prepares inputs for them.
///
/// One object holds the geometry, because both jobs need it: `worker` needs the
/// dimensions to allocate counters, and `prepare` needs them to know how many
/// hashes to compute. `prepare` runs on the calling thread, which is what lets
/// the borrow in a `DataInput` end at the call rather than having to outlive
/// the runtime.
pub trait OctoPlan: Send + 'static {
    /// Worker type this plan builds.
    type Worker: OctoWorker;

    /// Builds worker `worker_id`. Called once per worker at startup.
    fn worker(&self, worker_id: usize) -> Self::Worker;

    /// Converts one input into the worker's transport form.
    fn prepare(&self, input: &DataInput<'_>) -> <Self::Worker as OctoWorker>::Payload;
}

/// Aggregator-side trait: absorbs deltas into a full-precision sketch.
pub trait OctoAggregator: Send {
    /// Delta type consumed by the aggregator.
    type Delta: Send + 'static;

    /// Apply a single delta to the parent sketch.
    fn apply(&mut self, delta: Self::Delta);
}

// ---------------------------------------------------------------------------
// Compact worker sketches (§4.1, §3.2 Idea 3)
// ---------------------------------------------------------------------------

/// Count-Min worker sketch: one-byte counters, no flow-key storage.
///
/// A worker counter is cleared the moment it reaches τ, so it never exceeds
/// `MAX_PROMASK` and `⌈log τ⌉` bits suffice. Against the 32-bit counters a
/// full `CountMin` uses, this is the 4x memory saving of §4.1.
#[derive(Clone, Debug)]
pub struct CmWorkerSketch {
    counters: Vec<u8>,
    rows: usize,
    cols: usize,
}

impl CmWorkerSketch {
    /// Creates a `rows` x `cols` worker sketch with all counters cleared.
    pub fn new(rows: usize, cols: usize) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        Self {
            counters: vec![0u8; rows * cols],
            rows,
            cols,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns per row.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes held by the counter array.
    pub fn counter_bytes(&self) -> usize {
        self.counters.len()
    }

    /// Counters still held back from the aggregator, one per cell.
    pub fn residual(&self) -> &[u8] {
        &self.counters
    }

    /// Per-row hashes this geometry needs for `value`.
    pub fn hashes(rows: usize, value: &DataInput) -> RowHashes {
        (0..rows.max(1)).map(|r| hash64_seeded(r, value)).collect()
    }

    /// Inserts a key, emitting and clearing every counter that reaches
    /// `threshold` (Algorithm 1).
    #[inline(always)]
    pub fn insert_emit_delta(
        &mut self,
        value: &DataInput,
        threshold: u32,
        emit: &mut impl FnMut(CmDelta),
    ) {
        self.insert_hashes_emit_delta(&Self::hashes(self.rows, value), threshold, emit);
    }

    /// As `insert_emit_delta`, from hashes already computed by `hashes`.
    #[inline(always)]
    pub fn insert_hashes_emit_delta(
        &mut self,
        hashes: &[u64],
        threshold: u32,
        emit: &mut impl FnMut(CmDelta),
    ) {
        assert_eq!(
            hashes.len(),
            self.rows,
            "one hash per row; a plan built for a different geometry would \
             leave the untouched rows at zero and every estimate at zero"
        );
        let threshold = threshold.clamp(1, MAX_PROMASK) as u8;
        for (r, hashed) in hashes.iter().copied().enumerate() {
            let col = ((hashed & LOWER_32_MASK) as usize) % self.cols;
            let cell = &mut self.counters[r * self.cols + col];
            *cell += 1;
            if *cell >= threshold {
                emit(CmDelta {
                    row: r as u32,
                    col: col as u32,
                    value: *cell as u32,
                });
                *cell = 0;
            }
        }
    }

    /// Promotes and clears every counter still holding a partial count.
    pub fn flush(&mut self, emit: &mut impl FnMut(CmDelta)) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = &mut self.counters[row * self.cols + col];
                if *cell != 0 {
                    emit(CmDelta {
                        row: row as u32,
                        col: col as u32,
                        value: *cell as u32,
                    });
                    *cell = 0;
                }
            }
        }
    }

    /// As `insert_hashes_emit_delta`, but each delta carries the flow key so
    /// the aggregator can maintain the heavy-hitter heap.
    #[inline(always)]
    pub fn insert_hashes_emit_keyed_delta(
        &mut self,
        hashes: &[u64],
        key: &HeapItem,
        threshold: u32,
        emit: &mut impl FnMut(KeyedCmDelta),
    ) {
        self.insert_hashes_emit_delta(hashes, threshold, &mut |delta| {
            emit(KeyedCmDelta {
                key: key.clone(),
                delta,
            })
        });
    }
}

/// Count-sketch worker sketch: one-byte signed counters, no flow-key storage.
#[derive(Clone, Debug)]
pub struct CountWorkerSketch {
    counters: Vec<i8>,
    rows: usize,
    cols: usize,
}

impl CountWorkerSketch {
    /// Creates a `rows` x `cols` worker sketch with all counters cleared.
    pub fn new(rows: usize, cols: usize) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        Self {
            counters: vec![0i8; rows * cols],
            rows,
            cols,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns per row.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes held by the counter array.
    pub fn counter_bytes(&self) -> usize {
        self.counters.len()
    }

    /// Counters still held back from the aggregator, one per cell.
    pub fn residual(&self) -> &[i8] {
        &self.counters
    }

    /// Inserts a key, emitting and clearing every counter whose magnitude
    /// reaches `threshold` (§4.4: signed counters compare on `|counter|`).
    #[inline(always)]
    pub fn insert_emit_delta(
        &mut self,
        value: &DataInput,
        threshold: u32,
        emit: &mut impl FnMut(CountDelta),
    ) {
        self.insert_hashes_emit_delta(&CmWorkerSketch::hashes(self.rows, value), threshold, emit);
    }

    /// As `insert_emit_delta`, from hashes already computed by
    /// `CmWorkerSketch::hashes` - the two share a per-row hashing scheme.
    #[inline(always)]
    pub fn insert_hashes_emit_delta(
        &mut self,
        hashes: &[u64],
        threshold: u32,
        emit: &mut impl FnMut(CountDelta),
    ) {
        assert_eq!(
            hashes.len(),
            self.rows,
            "one hash per row; a plan built for a different geometry would \
             leave the untouched rows at zero and every estimate at zero"
        );
        let threshold = threshold.clamp(1, MAX_PROMASK) as i8;
        for (r, hashed) in hashes.iter().copied().enumerate() {
            let col = ((hashed & LOWER_32_MASK) as usize) % self.cols;
            let sign: i8 = if ((hashed >> 63) & 1) == 1 { 1 } else { -1 };
            let cell = &mut self.counters[r * self.cols + col];
            *cell += sign;
            if cell.unsigned_abs() >= threshold as u8 {
                emit(CountDelta {
                    row: r as u32,
                    col: col as u32,
                    value: *cell as i32,
                });
                *cell = 0;
            }
        }
    }

    /// Promotes and clears every counter still holding a partial count.
    pub fn flush(&mut self, emit: &mut impl FnMut(CountDelta)) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = &mut self.counters[row * self.cols + col];
                if *cell != 0 {
                    emit(CountDelta {
                        row: row as u32,
                        col: col as u32,
                        value: *cell as i32,
                    });
                    *cell = 0;
                }
            }
        }
    }

    /// As `insert_hashes_emit_delta`, but each delta carries the flow key.
    #[inline(always)]
    pub fn insert_hashes_emit_keyed_delta(
        &mut self,
        hashes: &[u64],
        key: &HeapItem,
        threshold: u32,
        emit: &mut impl FnMut(KeyedCountDelta),
    ) {
        self.insert_hashes_emit_delta(hashes, threshold, &mut |delta| {
            emit(KeyedCountDelta {
                key: key.clone(),
                delta,
            })
        });
    }
}

/// UnivMon-layer worker sketch: one-byte signed counters over the cell mapping
/// `CountL2HH` uses.
///
/// A UnivMon layer is not a plain `Count` sketch: it slices one 128-bit hash
/// into per-row columns and takes each row's sign from a different high bit,
/// and every layer hashes under its own seed. This worker calls the same
/// `l2hh_cell_for_row` the parent does, so the two cannot drift apart.
#[derive(Clone, Debug)]
pub struct L2hhWorkerSketch {
    counters: Vec<i8>,
    rows: usize,
    cols: usize,
    seed_idx: usize,
    mask_bits: u32,
}

impl L2hhWorkerSketch {
    /// Creates a worker over the cell space of a `CountL2HH` with these
    /// dimensions and seed.
    pub fn new(rows: usize, cols: usize, seed_idx: usize) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        let mask_bits = cols_mask_bits(cols);
        // Every row slices its column out of the same 128-bit hash, so the
        // rows collectively have 128 bits to spend. Past that the shift is
        // undefined: debug builds panic and release builds wrap, aliasing a
        // deep row onto a shallow one's column bits.
        assert!(
            rows * mask_bits as usize <= 128,
            "{rows} rows x {mask_bits} column bits exceeds the 128-bit hash budget; \
             reduce rows or columns"
        );
        Self {
            counters: vec![0i8; rows * cols],
            rows,
            cols,
            seed_idx,
            mask_bits,
        }
    }

    /// Bytes held by the counter array.
    pub fn counter_bytes(&self) -> usize {
        self.counters.len()
    }

    /// Counters still held back from the aggregator, one per cell.
    pub fn residual(&self) -> &[i8] {
        &self.counters
    }

    /// Promotes and clears every counter still holding a partial count.
    pub fn flush(&mut self, emit: &mut impl FnMut(CountDelta)) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let cell = &mut self.counters[row * self.cols + col];
                if *cell != 0 {
                    emit(CountDelta {
                        row: row as u32,
                        col: col as u32,
                        value: *cell as i32,
                    });
                    *cell = 0;
                }
            }
        }
    }

    /// Inserts a key, emitting and clearing every counter whose magnitude
    /// reaches `threshold`.
    #[inline(always)]
    pub fn insert_emit_delta(
        &mut self,
        value: &DataInput,
        threshold: u32,
        emit: &mut impl FnMut(CountDelta),
    ) {
        self.insert_hash_emit_delta(hash128_seeded(self.seed_idx, value), threshold, emit);
    }

    /// Seed this layer hashes under.
    pub fn seed_idx(&self) -> usize {
        self.seed_idx
    }

    /// As `insert_emit_delta`, from a hash already computed under `seed_idx`.
    #[inline(always)]
    pub fn insert_hash_emit_delta(
        &mut self,
        hashed: u128,
        threshold: u32,
        emit: &mut impl FnMut(CountDelta),
    ) {
        let threshold = threshold.clamp(1, MAX_PROMASK) as u8;
        for row in 0..self.rows {
            let (col, sign) = l2hh_cell_for_row(hashed, row, self.cols, self.mask_bits);
            let cell = &mut self.counters[row * self.cols + col];
            *cell += sign as i8;
            if cell.unsigned_abs() >= threshold {
                emit(CountDelta {
                    row: row as u32,
                    col: col as u32,
                    value: *cell as i32,
                });
                *cell = 0;
            }
        }
    }
}

/// CocoSketch worker sketch: the parent's table with one-byte counters.
///
/// §3.2 Idea 3 strips flow-key storage out of a worker, but CocoSketch *is*
/// key storage - a bucket's key is the only record of what its counter counts -
/// so this worker keeps the keys and shrinks only the counters. Every step of
/// `insert_emit_delta` mirrors `Coco::insert`, down to the uniform draw among
/// tied minima and the `v / val` election, so the two cannot drift apart.
#[derive(Clone, Debug)]
pub struct CocoWorkerSketch {
    keys: Vec<Option<String>>,
    counters: Vec<u8>,
    w: usize,
    d: usize,
}

impl CocoWorkerSketch {
    /// Creates a `d` x `w` worker table, empty and cleared. The argument order
    /// is `Coco::init_with_size`'s.
    pub fn new(w: usize, d: usize) -> Self {
        Self {
            keys: vec![None; w * d],
            counters: vec![0u8; w * d],
            w,
            d,
        }
    }

    /// Buckets per array.
    pub fn width(&self) -> usize {
        self.w
    }

    /// Number of arrays.
    pub fn depth(&self) -> usize {
        self.d
    }

    /// Bytes held by the counter array, excluding the keys.
    pub fn counter_bytes(&self) -> usize {
        self.counters.len()
    }

    /// Counters still held back from the aggregator, one per bucket.
    pub fn residual(&self) -> &[u8] {
        &self.counters
    }

    /// Inserts `key`, emitting and clearing the bucket that reaches `threshold`.
    ///
    /// The promoted message carries whichever key the bucket holds *after* the
    /// election, which is the arriving key when it won the bucket and the
    /// incumbent when it did not.
    pub fn insert_emit_delta(
        &mut self,
        key: &str,
        threshold: u32,
        emit: &mut impl FnMut(CocoDelta),
    ) {
        if self.d == 0 || self.w == 0 {
            return;
        }
        let threshold = threshold.clamp(1, MAX_PROMASK) as u8;
        let key_input = DataInput::Str(key);
        let mut rng: Option<ThreadRng> = None;
        let mut victim = 0usize;
        let mut victim_val = u8::MAX;
        let mut tied = 0u32;

        for i in 0..self.d {
            let cell = i * self.w + hash64_seeded(i, &key_input) as usize % self.w;
            if self.keys[cell].as_deref() == Some(key) {
                self.counters[cell] += 1;
                if self.counters[cell] >= threshold {
                    emit(CocoDelta {
                        key: key.to_string(),
                        value: self.counters[cell] as u64,
                    });
                    self.counters[cell] = 0;
                }
                return;
            }
            if self.counters[cell] < victim_val {
                victim_val = self.counters[cell];
                victim = cell;
                tied = 1;
            } else if self.counters[cell] == victim_val {
                tied += 1;
                if rng.get_or_insert_with(rand::rng).random_range(0..tied) == 0 {
                    victim = cell;
                }
            }
        }

        self.counters[victim] += 1;
        let elected = match self.keys[victim] {
            None => true,
            Some(_) => {
                let draw = rng
                    .get_or_insert_with(rand::rng)
                    .random_range(0.0..=1.0_f64);
                1.0 > draw * self.counters[victim] as f64
            }
        };
        if elected {
            self.keys[victim] = Some(key.to_string());
        }
        if self.counters[victim] >= threshold {
            if let Some(resident) = self.keys[victim].clone() {
                emit(CocoDelta {
                    key: resident,
                    value: self.counters[victim] as u64,
                });
            }
            self.counters[victim] = 0;
        }
    }

    /// Promotes and clears every bucket still holding a partial count, leaving
    /// the keys resident.
    pub fn flush(&mut self, emit: &mut impl FnMut(CocoDelta)) {
        for cell in 0..self.counters.len() {
            if self.counters[cell] == 0 {
                continue;
            }
            if let Some(resident) = self.keys[cell].clone() {
                emit(CocoDelta {
                    key: resident,
                    value: self.counters[cell] as u64,
                });
            }
            self.counters[cell] = 0;
        }
    }
}

/// One heavy-part slot of an [`ElasticWorkerSketch`].
///
/// Occupancy is `flow_id`, not `vote_pos` as in the parent's
/// `HeavyBucket::is_vacant`: a promotion clears the votes and the flow stays
/// resident.
///
/// `eviction` carries the parent's own semantics -- set on takeover by
/// eviction, clear on seating a previously unoccupied slot.
#[derive(Clone, Debug, Default)]
struct ElasticWorkerBucket {
    flow_id: Option<String>,
    vote_pos: u8,
    vote_neg: i32,
    eviction: bool,
}

/// Elastic sketch worker: a heavy part with one-byte vote counters over a
/// compact Count-Min light layer.
///
/// Appendix C keeps both halves in the worker. The heavy part promotes the
/// resident flow, its votes and its eviction flag; the light part is an
/// ordinary [`CmWorkerSketch`], promoting cell deltas with no key attached, and
/// takes the arrivals that lose a bucket contest. An evicted resident does not
/// go there: it is handed over keyed, as [`ElasticDelta::Evicted`].
#[derive(Clone, Debug)]
pub struct ElasticWorkerSketch {
    heavy: Vec<ElasticWorkerBucket>,
    light: CmWorkerSketch,
}

impl ElasticWorkerSketch {
    /// Creates a worker mirroring an `Elastic` of these dimensions.
    pub fn new(bucket_count: usize, light_rows: usize, light_cols: usize) -> Self {
        Self {
            heavy: vec![ElasticWorkerBucket::default(); bucket_count.max(1)],
            light: CmWorkerSketch::new(light_rows, light_cols),
        }
    }

    /// Number of heavy buckets.
    pub fn bucket_count(&self) -> usize {
        self.heavy.len()
    }

    /// Borrows the light layer's counter array.
    pub fn light(&self) -> &CmWorkerSketch {
        &self.light
    }

    /// Records one occurrence of `id`, promoting whichever half crosses
    /// `threshold`.
    ///
    /// Every branch is `Elastic::insert`'s: a vacant bucket seats the flow and
    /// clears the slot's eviction flag, a matching one takes a positive vote,
    /// and otherwise the bucket takes a negative vote and either the arrival
    /// goes to the light layer or, once `vote_neg >= LAMBDA * vote_pos`, the
    /// slot is flagged and the resident is handed over as
    /// [`ElasticDelta::Evicted`] with its whole positive vote.
    pub fn insert_emit_delta(
        &mut self,
        id: &str,
        threshold: u32,
        emit: &mut impl FnMut(ElasticDelta),
    ) {
        let threshold = threshold.clamp(1, MAX_PROMASK);
        let idx =
            hash64_seeded(CANONICAL_HASH_SEED, &DataInput::Str(id)) as usize % self.heavy.len();

        if self.heavy[idx].flow_id.is_none() {
            self.heavy[idx].flow_id = Some(id.to_string());
            self.heavy[idx].vote_pos = 1;
            self.heavy[idx].vote_neg = 0;
            self.heavy[idx].eviction = false;
            self.promote_heavy(idx, threshold, emit);
            return;
        }
        if self.heavy[idx].flow_id.as_deref() == Some(id) {
            self.heavy[idx].vote_pos += 1;
            self.promote_heavy(idx, threshold, emit);
            return;
        }

        self.heavy[idx].vote_neg += 1;
        if self.heavy[idx].vote_neg < LAMBDA * self.heavy[idx].vote_pos as i32 {
            self.spill(id, threshold, emit);
            return;
        }

        let evicted_votes = self.heavy[idx].vote_pos;
        let evicted_id = self.heavy[idx].flow_id.replace(id.to_string());
        self.heavy[idx].vote_pos = 1;
        self.heavy[idx].vote_neg = 1;
        self.heavy[idx].eviction = true;
        if let Some(evicted_id) = evicted_id {
            emit(ElasticDelta::Evicted {
                key: evicted_id,
                votes: evicted_votes as u32,
            });
        }
        self.promote_heavy(idx, threshold, emit);
    }

    /// Promotes and clears every counter still holding a partial count, in both
    /// halves. Heavy flows stay resident, having only their votes taken.
    pub fn flush(&mut self, emit: &mut impl FnMut(ElasticDelta)) {
        for bucket in &mut self.heavy {
            if bucket.vote_pos == 0 {
                continue;
            }
            if let Some(resident) = bucket.flow_id.clone() {
                emit(ElasticDelta::Heavy {
                    key: resident,
                    value: bucket.vote_pos as u32,
                    eviction: bucket.eviction,
                });
            }
            bucket.vote_pos = 0;
        }
        self.light
            .flush(&mut |delta| emit(ElasticDelta::Light(delta)));
    }

    /// Adds one to the light layer's cells for `id`, the only unkeyed write
    /// the worker makes.
    fn spill(&mut self, id: &str, threshold: u32, emit: &mut impl FnMut(ElasticDelta)) {
        let hashes = CmWorkerSketch::hashes(self.light.rows(), &DataInput::Str(id));
        self.light
            .insert_hashes_emit_delta(&hashes, threshold, &mut |delta| {
                emit(ElasticDelta::Light(delta))
            });
    }

    /// Ships the bucket's votes and its flag, and clears the votes, if they
    /// reached `threshold`.
    fn promote_heavy(&mut self, idx: usize, threshold: u32, emit: &mut impl FnMut(ElasticDelta)) {
        let bucket = &mut self.heavy[idx];
        if (bucket.vote_pos as u32) < threshold {
            return;
        }
        if let Some(resident) = bucket.flow_id.clone() {
            emit(ElasticDelta::Heavy {
                key: resident,
                value: bucket.vote_pos as u32,
                eviction: bucket.eviction,
            });
        }
        bucket.vote_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Configuration & Runtime (requires "octo-runtime" feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "octo-runtime")]
/// How inputs are handed to workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OctoPartition {
    /// Hash the key so one flow always lands on one worker.
    ///
    /// This is the paper's setting - each worker owns a slice of the traffic -
    /// and makes k', the number of workers a flow may pass by, equal to 1. The
    /// additive term k'τ in the error bounds is then as small as it can be.
    #[default]
    HashByKey,
    /// Hand inputs to workers in turn, regardless of key.
    ///
    /// Spreads load perfectly even under a skewed key distribution, at the cost
    /// of k' = k: every flow reaches every worker, so each one may hold back up
    /// to τ of that flow's count.
    RoundRobin,
}

#[cfg(feature = "octo-runtime")]
/// Aggregator-driven threshold control (§4.3).
///
/// The aggregator watches how much work is queued and nudges the shared τ so
/// its receive rate tracks the workers' send rate: a short queue means it can
/// afford more updates, a long one means it is falling behind.
#[derive(Clone, Debug)]
pub struct OctoAdaptiveThreshold {
    /// Target total queue length Q across all worker channels (paper: 10).
    pub target_queue_len: usize,
    /// Dead band α around the target (paper: 0.25).
    pub alpha: f64,
    /// How often the threshold is re-evaluated (paper: 100µs).
    pub interval: Duration,
    /// Floor for τ. Derive it from an accuracy target with `threshold_for_error`.
    pub min_threshold: u32,
    /// Ceiling for τ.
    pub max_threshold: u32,
}

#[cfg(feature = "octo-runtime")]
impl Default for OctoAdaptiveThreshold {
    fn default() -> Self {
        Self {
            target_queue_len: 10,
            alpha: 0.25,
            interval: Duration::from_micros(100),
            min_threshold: 1,
            max_threshold: MAX_PROMASK,
        }
    }
}

/// The threshold that meets an additive error target, from Equation 4: `τ = εL1/k'`.
///
/// `k_prime` is the number of workers a single flow may reach: 1 under
/// [`OctoPartition::HashByKey`], and the worker count under
/// [`OctoPartition::RoundRobin`].
pub fn threshold_for_error(epsilon: f64, l1: f64, k_prime: usize) -> u32 {
    if !(epsilon.is_finite() && l1.is_finite()) || k_prime == 0 {
        return 1;
    }
    let tau = epsilon * l1 / k_prime as f64;
    if !tau.is_finite() || tau < 1.0 {
        return 1;
    }
    (tau as u32).min(MAX_PROMASK)
}

#[cfg(feature = "octo-runtime")]
/// Configuration for `run_octo` and [`OctoRuntime`].
#[derive(Clone, Debug)]
pub struct OctoConfig {
    /// Number of worker threads (default: 4).
    pub num_workers: usize,
    /// Pin worker threads to cores (default: true).
    /// Worker i is pinned to core i, aggregator to core num_workers.
    /// Silently skipped if pinning fails.
    pub pin_cores: bool,
    /// Queue capacity for bounded worker-input and worker-delta channels (default: 65536).
    pub queue_capacity: usize,
    /// Promotion threshold τ shared by every worker.
    ///
    /// Clone it into the worker factory so the workers and the aggregator's
    /// controller refer to the same value.
    pub threshold: OctoThreshold,
    /// How inputs are routed to workers (default: [`OctoPartition::HashByKey`]).
    pub partition: OctoPartition,
    /// Threshold control, or `None` to hold τ fixed (default: `None`).
    pub adaptive: Option<OctoAdaptiveThreshold>,
}

#[cfg(feature = "octo-runtime")]
impl Default for OctoConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            pin_cores: true,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            threshold: OctoThreshold::new(CM_PROMASK),
            partition: OctoPartition::default(),
            adaptive: None,
        }
    }
}

#[cfg(feature = "octo-runtime")]
/// Result of an `run_octo` execution.
pub struct OctoResult<P> {
    /// Final parent sketch after all deltas are applied.
    pub parent: P,
}

#[cfg(feature = "octo-runtime")]
enum WorkerMsg<T> {
    Data(T),
    /// Promote everything held back, then acknowledge.
    Flush(Sender<()>),
    End,
}

#[cfg(feature = "octo-runtime")]
enum AggregatorMsg {
    /// Apply everything already queued, then acknowledge.
    Drain(Sender<()>),
}

#[cfg(feature = "octo-runtime")]
/// Streaming Octo runtime that accepts incremental inserts and finalizes into a parent sketch.
///
/// Nothing borrowed crosses a thread: [`OctoPlan::prepare`] turns each input
/// into the worker's payload on the calling thread, so a borrowed key is
/// finished with by the time `insert` returns.
pub struct OctoRuntime<L, P>
where
    L: OctoPlan,
    P: OctoAggregator<Delta = <L::Worker as OctoWorker>::Delta> + Send + Sync + 'static,
{
    core: Option<OctoCore<<L::Worker as OctoWorker>::Payload, P>>,
    plan: L,
}

#[cfg(feature = "octo-runtime")]
/// Read-only handle for querying the live aggregator state while runtime is active.
pub struct OctoReadHandle<P> {
    parent: Weak<RwLock<P>>,
}

#[cfg(feature = "octo-runtime")]
impl<P> Clone for OctoReadHandle<P> {
    fn clone(&self) -> Self {
        Self {
            parent: Weak::clone(&self.parent),
        }
    }
}

#[cfg(feature = "octo-runtime")]
impl<P> OctoReadHandle<P> {
    /// Executes a read-only closure over the live parent state.
    pub fn with_parent<R>(&self, f: impl FnOnce(&P) -> R) -> R {
        let parent = self
            .parent
            .upgrade()
            .expect("Octo runtime has been finished and parent state was dropped");
        let guard = parent.read().expect("parent lock poisoned");
        f(&guard)
    }
}

#[cfg(feature = "octo-runtime")]
struct OctoCore<T, P> {
    worker_input_txs: Vec<Sender<WorkerMsg<T>>>,
    control_tx: Sender<AggregatorMsg>,
    next_worker: AtomicUsize,
    partition: OctoPartition,
    worker_handles: Vec<thread::JoinHandle<()>>,
    aggregator_handle: Option<thread::JoinHandle<()>>,
    parent: Arc<RwLock<P>>,
    closed: AtomicBool,
}

#[cfg(feature = "octo-runtime")]
impl<T, P> OctoCore<T, P> {
    fn read_handle(&self) -> OctoReadHandle<P> {
        OctoReadHandle {
            parent: Arc::downgrade(&self.parent),
        }
    }

    fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        for tx in &self.worker_input_txs {
            let _ = tx.send(WorkerMsg::End);
        }
    }

    fn worker_for(&self, input: &DataInput<'_>) -> usize {
        let workers = self.worker_input_txs.len();
        match self.partition {
            OctoPartition::RoundRobin => self.next_worker.fetch_add(1, Ordering::AcqRel) % workers,
            OctoPartition::HashByKey => (hash64_seeded(0, input) % workers as u64) as usize,
        }
    }
}

#[cfg(feature = "octo-runtime")]
impl<T, P> OctoCore<T, P> {
    fn into_parent(mut self) -> P {
        self.close();

        for handle in self.worker_handles.drain(..) {
            handle.join().expect("worker thread panicked during finish");
        }

        if let Some(aggregator) = self.aggregator_handle.take() {
            aggregator
                .join()
                .expect("aggregator thread panicked during finish");
        }

        let parent_lock = match Arc::try_unwrap(self.parent) {
            Ok(lock) => lock,
            Err(_) => panic!("Octo parent still has external strong references at finish"),
        };
        parent_lock.into_inner().expect("parent lock poisoned")
    }
}

#[cfg(feature = "octo-runtime")]
impl<T, P> OctoCore<T, P>
where
    T: Send + 'static,
    P: Send + Sync + 'static,
{
    fn start<W>(workers: Vec<W>, parent: P, config: &OctoConfig) -> Self
    where
        W: OctoWorker<Payload = T> + 'static,
        P: OctoAggregator<Delta = W::Delta>,
    {
        let num_workers = config.num_workers.max(1);
        assert_eq!(workers.len(), num_workers);
        let queue_capacity = config.queue_capacity.max(1);
        let pin_cores = config.pin_cores;

        let parent = Arc::new(RwLock::new(parent));
        let parent_for_aggregator = Arc::clone(&parent);
        let mut delta_txs: Vec<Sender<W::Delta>> = Vec::with_capacity(num_workers);
        let mut delta_rxs: Vec<Option<Receiver<W::Delta>>> = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (tx, rx) = bounded::<W::Delta>(queue_capacity);
            delta_txs.push(tx);
            delta_rxs.push(Some(rx));
        }

        let adaptive = config.adaptive.clone();
        let threshold = config.threshold.clone();
        let (control_tx, control_rx) = bounded::<AggregatorMsg>(1);
        let aggregator_handle = thread::spawn(move || {
            if pin_cores {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: num_workers });
            }

            let mut controller = adaptive.as_ref().map(ThresholdController::new);
            let mut disconnected_workers = 0usize;
            while disconnected_workers < num_workers {
                let mut made_progress = false;
                for rx_slot in &mut delta_rxs {
                    let Some(rx) = rx_slot else {
                        continue;
                    };
                    match rx.try_recv() {
                        Ok(delta) => {
                            let mut guard = parent_for_aggregator
                                .write()
                                .expect("parent lock poisoned in aggregator");
                            guard.apply(delta);
                            made_progress = true;
                        }
                        Err(TryRecvError::Empty) => {}
                        Err(TryRecvError::Disconnected) => {
                            *rx_slot = None;
                            disconnected_workers += 1;
                        }
                    }
                }
                if let Some(controller) = controller.as_mut() {
                    controller.maybe_adjust(&delta_rxs, &threshold);
                }
                if !made_progress {
                    // Only answer a drain request once every queue is empty.
                    // The caller has already collected each worker's flush
                    // acknowledgement, so no worker is still producing and
                    // "empty" means "applied".
                    if let Ok(AggregatorMsg::Drain(ack)) = control_rx.try_recv() {
                        let _ = ack.send(());
                    }
                    std::hint::spin_loop();
                }
            }
        });

        let mut worker_input_txs = Vec::with_capacity(num_workers);
        let mut worker_handles = Vec::with_capacity(num_workers);
        for (worker_id, (mut worker, delta_tx_worker)) in
            workers.into_iter().zip(delta_txs).enumerate()
        {
            let (worker_tx, worker_rx) = bounded::<WorkerMsg<T>>(queue_capacity);
            worker_input_txs.push(worker_tx);
            worker_handles.push(thread::spawn(move || {
                if pin_cores {
                    let _ = core_affinity::set_for_current(core_affinity::CoreId { id: worker_id });
                }
                while let Ok(msg) = worker_rx.recv() {
                    match msg {
                        WorkerMsg::Data(payload) => worker.process(&payload, &mut |delta| {
                            delta_tx_worker
                                .send(delta)
                                .expect("aggregator receiver dropped while workers still running");
                        }),
                        WorkerMsg::Flush(ack) => {
                            worker.flush(&mut |delta| {
                                delta_tx_worker.send(delta).expect(
                                    "aggregator receiver dropped while workers still running",
                                );
                            });
                            let _ = ack.send(());
                        }
                        WorkerMsg::End => {
                            // Hand over whatever is still held back, so a
                            // finished run answers against the whole stream.
                            worker.flush(&mut |delta| {
                                delta_tx_worker.send(delta).expect(
                                    "aggregator receiver dropped while workers still running",
                                );
                            });
                            break;
                        }
                    }
                }
            }));
        }

        Self {
            worker_input_txs,
            control_tx,
            next_worker: AtomicUsize::new(0),
            partition: config.partition,
            worker_handles,
            aggregator_handle: Some(aggregator_handle),
            parent,
            closed: AtomicBool::new(false),
        }
    }
}

#[cfg(feature = "octo-runtime")]
/// Applies Equations 1 and 2 of §4.3 on the aggregator thread.
struct ThresholdController {
    settings: OctoAdaptiveThreshold,
    previous_queue_len: f64,
    next_check: Instant,
    polls_since_check: u32,
}

#[cfg(feature = "octo-runtime")]
impl ThresholdController {
    /// Reading the clock on every poll would cost more than the control loop
    /// saves, so the elapsed-time check is itself sampled.
    const POLLS_BETWEEN_CLOCK_READS: u32 = 256;

    fn new(settings: &OctoAdaptiveThreshold) -> Self {
        let mut settings = settings.clone();
        // `OctoAdaptiveThreshold` has public fields, and `clamp` panics on an
        // inverted range. Normalising here keeps a misconfigured band from
        // killing the aggregator thread and surfacing as an unrelated
        // "worker receiver dropped" panic on the caller's thread.
        settings.min_threshold = settings.min_threshold.clamp(1, MAX_PROMASK);
        settings.max_threshold = settings
            .max_threshold
            .clamp(settings.min_threshold, MAX_PROMASK);
        let next_check = Instant::now() + settings.interval;
        Self {
            settings,
            previous_queue_len: 0.0,
            next_check,
            polls_since_check: 0,
        }
    }

    fn maybe_adjust<D>(&mut self, receivers: &[Option<Receiver<D>>], threshold: &OctoThreshold) {
        self.polls_since_check += 1;
        if self.polls_since_check < Self::POLLS_BETWEEN_CLOCK_READS {
            return;
        }
        self.polls_since_check = 0;

        let now = Instant::now();
        if now < self.next_check {
            return;
        }
        self.next_check = now + self.settings.interval;

        let queue_len: usize = receivers.iter().flatten().map(|rx| rx.len()).sum();
        let queue_len = queue_len as f64;
        // Equation 1: predict the next window from the last two observations.
        let predicted = queue_len + (queue_len - self.previous_queue_len);
        self.previous_queue_len = queue_len;

        // Equation 2: nudge τ by one in whichever direction the prediction misses.
        let target = self.settings.target_queue_len.max(1) as f64;
        let tau = threshold.get();
        let adjusted = if predicted < (1.0 - self.settings.alpha) * target {
            tau.saturating_sub(1)
        } else if predicted > (1.0 + self.settings.alpha) * target {
            tau.saturating_add(1)
        } else {
            tau
        };
        threshold.set(adjusted.clamp(self.settings.min_threshold, self.settings.max_threshold));
    }
}

#[cfg(feature = "octo-runtime")]
impl<L, P> OctoRuntime<L, P>
where
    L: OctoPlan,
    P: OctoAggregator<Delta = <L::Worker as OctoWorker>::Delta> + Send + Sync + 'static,
{
    /// Starts the worker and aggregator threads described by `config`.
    pub fn new<PF>(config: &OctoConfig, plan: L, parent_factory: PF) -> Self
    where
        PF: FnOnce() -> P,
    {
        let num_workers = config.num_workers.max(1);
        let workers: Vec<L::Worker> = (0..num_workers).map(|id| plan.worker(id)).collect();
        let parent = parent_factory();
        let core = OctoCore::start(workers, parent, config);

        Self {
            core: Some(core),
            plan,
        }
    }

    /// Borrows the plan the runtime was built with.
    pub fn plan(&self) -> &L {
        &self.plan
    }

    /// Returns a handle for reading the parent while the runtime is live.
    pub fn read_handle(&self) -> OctoReadHandle<P> {
        self.core
            .as_ref()
            .expect("runtime core missing")
            .read_handle()
    }

    /// Signals every worker to stop after draining what is already queued.
    pub fn close(&self) {
        self.core.as_ref().expect("runtime core missing").close();
    }

    /// Routes one input to a worker according to the configured partition.
    ///
    /// The input is hashed for partitioning and converted to the worker's
    /// payload here, on the calling thread, so `input` may borrow from storage
    /// that is dropped as soon as this returns.
    pub fn insert(&mut self, input: DataInput<'_>) {
        let core = self.core.as_ref().expect("runtime core missing");
        if core.closed.load(Ordering::Acquire) {
            panic!("cannot insert after runtime has been closed");
        }

        let worker_id = core.worker_for(&input);
        let payload = self.plan.prepare(&input);
        core.worker_input_txs[worker_id]
            .send(WorkerMsg::Data(payload))
            .expect("worker receiver dropped while runtime is active");
    }

    /// Promotes everything the workers still hold and waits for the aggregator
    /// to apply it, so the parent answers against every input accepted so far.
    ///
    /// This is the point at which a stream is handed over for querying. Between
    /// flushes a worker holds any counter below τ, which for Count-Min and
    /// Count only makes the parent low by under τ per cell, but for DDSketch or
    /// a thresholded HyperLogLog means whole cells are missing - and a quantile
    /// or a cardinality is exactly a statement about which cells exist. It
    /// costs one message per non-empty cell, the same as shipping the sketch,
    /// so call it when a query needs to be right rather than on a timer.
    ///
    /// Inserting afterwards is fine; the runtime is not sealed.
    ///
    /// After `close` this returns immediately without draining anything.
    /// `close` has already asked every worker to flush, but nothing here waits
    /// for that to land, so read the result through `finish` rather than
    /// through a live handle.
    pub fn flush(&mut self) {
        let core = self.core.as_ref().expect("runtime core missing");
        if core.closed.load(Ordering::Acquire) {
            return;
        }

        // Phase one: every worker hands over its residue. Taking &mut self
        // means no insert can race this, so once each has acknowledged, the
        // delta queues hold everything and nothing more is coming.
        let (ack_tx, ack_rx) = bounded::<()>(core.worker_input_txs.len());
        for tx in &core.worker_input_txs {
            tx.send(WorkerMsg::Flush(ack_tx.clone()))
                .expect("worker receiver dropped while runtime is active");
        }
        // Drop the local sender, or `recv` can never report `Disconnected` and a
        // worker that dies mid-flush - the aggregator panicking behind it, say -
        // leaves this waiting forever instead of failing.
        drop(ack_tx);
        for _ in 0..core.worker_input_txs.len() {
            ack_rx.recv().expect("worker dropped during flush");
        }

        // Phase two: the aggregator applies what is queued.
        let (drain_tx, drain_rx) = bounded::<()>(1);
        core.control_tx
            .send(AggregatorMsg::Drain(drain_tx))
            .expect("aggregator dropped during flush");
        drain_rx.recv().expect("aggregator dropped during flush");
    }

    /// Inserts a batch, one element at a time.
    pub fn insert_batch(&mut self, inputs: &[DataInput<'_>]) {
        for input in inputs {
            self.insert(input.clone());
        }
    }

    /// Drains the pipeline and returns the finished parent sketch.
    pub fn finish(mut self) -> OctoResult<P> {
        let parent = self
            .core
            .take()
            .expect("runtime core missing")
            .into_parent();

        OctoResult { parent }
    }
}

// ---------------------------------------------------------------------------
// Prepared inputs
// ---------------------------------------------------------------------------

/// Per-row hashes plus the key, for a worker whose deltas carry the key.
///
/// The key is owned: a worker that must store it has to copy it, and copying
/// once at the hand-off is what lets the caller drop the original immediately.
#[derive(Clone, Debug)]
pub struct KeyedHashes {
    /// Per-row hashes, as `CmWorkerSketch::hashes` computes them.
    pub hashes: RowHashes,
    /// The flow key.
    pub key: HeapItem,
}

/// A prepared UnivMon insert.
#[derive(Clone, Debug)]
pub struct UnivMonInput {
    /// Deepest layer this key reaches; it touches `0..=bottom`.
    pub bottom: usize,
    /// One 128-bit hash per touched layer, under that layer's seed. Layer depth
    /// is geometric, so this is nearly always one or two entries.
    pub layer_hashes: SmallVec<[u128; 4]>,
    /// The flow key.
    pub key: HeapItem,
}

// ---------------------------------------------------------------------------
// Concrete worker/parent implementations
// ---------------------------------------------------------------------------

// -- CountMin --

/// OctoSketch worker backed by a compact one-byte Count-Min sketch.
pub struct CmOctoWorker {
    sketch: CmWorkerSketch,
    threshold: OctoThreshold,
}

impl CmOctoWorker {
    /// Creates a worker with a private threshold fixed at `CM_PROMASK`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_threshold(rows, cols, OctoThreshold::new(CM_PROMASK))
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(rows: usize, cols: usize, threshold: OctoThreshold) -> Self {
        Self {
            sketch: CmWorkerSketch::new(rows, cols),
            threshold,
        }
    }

    /// Borrows the worker's counter array.
    pub fn sketch(&self) -> &CmWorkerSketch {
        &self.sketch
    }
}

impl OctoWorker for CmOctoWorker {
    type Delta = CmDelta;
    type Payload = RowHashes;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch
            .insert_hashes_emit_delta(payload, self.threshold.get(), emit);
    }

    fn flush<F>(&mut self, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch.flush(emit);
    }
}

/// OctoSketch parent wrapping a full-precision `CountMin`.
pub struct CmOctoAggregator {
    /// Parent Count-Min sketch updated by worker deltas.
    pub sketch: CountMin<Vector2D<i32>, RegularPath>,
}

impl CmOctoAggregator {
    /// Creates an aggregator with a `rows` x `cols` parent sketch.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            sketch: CountMin::with_dimensions(rows, cols),
        }
    }
}

impl OctoAggregator for CmOctoAggregator {
    type Delta = CmDelta;

    #[inline(always)]
    fn apply(&mut self, delta: CmDelta) {
        self.sketch.apply_delta(delta);
    }
}

// -- CountMin with heavy-hitter tracking (§3.2, Idea 3) --

/// Count-Min worker that ships the flow key alongside each promoted counter.
pub struct CmTopKOctoWorker {
    sketch: CmWorkerSketch,
    threshold: OctoThreshold,
}

impl CmTopKOctoWorker {
    /// Creates a worker with a private threshold fixed at `CM_PROMASK`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_threshold(rows, cols, OctoThreshold::new(CM_PROMASK))
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(rows: usize, cols: usize, threshold: OctoThreshold) -> Self {
        Self {
            sketch: CmWorkerSketch::new(rows, cols),
            threshold,
        }
    }

    /// Borrows the worker's counter array.
    pub fn sketch(&self) -> &CmWorkerSketch {
        &self.sketch
    }
}

impl OctoWorker for CmTopKOctoWorker {
    type Delta = KeyedCmDelta;
    type Payload = KeyedHashes;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch.insert_hashes_emit_keyed_delta(
            &payload.hashes,
            &payload.key,
            self.threshold.get(),
            emit,
        );
    }

    // No flush: every delta carries the key that produced it, and a worker
    // keeps no key storage, so a residual cell cannot be attributed back to a
    // key. A key that has never promoted is absent from the aggregator's heap
    // entirely and estimates zero. See
    // `a_key_below_the_threshold_never_reaches_the_topk_aggregator`.
}

/// Count-Min parent that also maintains the only heavy-hitter heap in the
/// pipeline, rebuilt from the keys the workers ship (Algorithm 2).
pub struct CmTopKOctoAggregator {
    /// Parent sketch plus heavy-hitter heap.
    pub sketch: CMSHeap<Vector2D<i32>, RegularPath>,
}

impl CmTopKOctoAggregator {
    /// Creates an aggregator tracking the `top_k` heaviest keys.
    pub fn new(rows: usize, cols: usize, top_k: usize) -> Self {
        Self {
            sketch: CMSHeap::new(rows, cols, top_k),
        }
    }
}

impl OctoAggregator for CmTopKOctoAggregator {
    type Delta = KeyedCmDelta;

    #[inline(always)]
    fn apply(&mut self, delta: KeyedCmDelta) {
        self.sketch.cms_mut().apply_delta(delta.delta);
        let key = heap_item_to_sketch_input(&delta.key);
        let estimate = self.sketch.cms().estimate(&key);
        self.sketch.heap_mut().update(&key, estimate as i64);
    }
}

// -- Count Sketch --

/// OctoSketch worker backed by a compact one-byte Count sketch.
pub struct CountOctoWorker {
    sketch: CountWorkerSketch,
    threshold: OctoThreshold,
}

impl CountOctoWorker {
    /// Creates a worker with a private threshold fixed at `COUNT_PROMASK`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_threshold(rows, cols, OctoThreshold::new(COUNT_PROMASK))
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(rows: usize, cols: usize, threshold: OctoThreshold) -> Self {
        Self {
            sketch: CountWorkerSketch::new(rows, cols),
            threshold,
        }
    }

    /// Borrows the worker's counter array.
    pub fn sketch(&self) -> &CountWorkerSketch {
        &self.sketch
    }
}

impl OctoWorker for CountOctoWorker {
    type Delta = CountDelta;
    type Payload = RowHashes;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch
            .insert_hashes_emit_delta(payload, self.threshold.get(), emit);
    }

    fn flush<F>(&mut self, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch.flush(emit);
    }
}

/// OctoSketch parent wrapping a full-precision `Count`.
pub struct CountOctoAggregator {
    /// Parent Count Sketch updated by worker deltas.
    pub sketch: Count<Vector2D<i32>, RegularPath>,
}

impl CountOctoAggregator {
    /// Creates an aggregator with a `rows` x `cols` parent sketch.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            sketch: Count::with_dimensions(rows, cols),
        }
    }
}

impl OctoAggregator for CountOctoAggregator {
    type Delta = CountDelta;

    #[inline(always)]
    fn apply(&mut self, delta: CountDelta) {
        self.sketch.apply_delta(delta);
    }
}

// -- Count Sketch with heavy-hitter tracking --

/// Count-sketch worker that ships the flow key alongside each promoted counter.
pub struct CountTopKOctoWorker {
    sketch: CountWorkerSketch,
    threshold: OctoThreshold,
}

impl CountTopKOctoWorker {
    /// Creates a worker with a private threshold fixed at `COUNT_PROMASK`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_threshold(rows, cols, OctoThreshold::new(COUNT_PROMASK))
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(rows: usize, cols: usize, threshold: OctoThreshold) -> Self {
        Self {
            sketch: CountWorkerSketch::new(rows, cols),
            threshold,
        }
    }
}

impl OctoWorker for CountTopKOctoWorker {
    type Delta = KeyedCountDelta;
    type Payload = KeyedHashes;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch.insert_hashes_emit_keyed_delta(
            &payload.hashes,
            &payload.key,
            self.threshold.get(),
            emit,
        );
    }

    // No flush, for the same reason as CmTopKOctoWorker: a key the aggregator
    // has never been sent is absent from the heap, not merely undercounted.
}

/// Count-sketch parent holding the pipeline's only heavy-hitter heap.
pub struct CountTopKOctoAggregator {
    /// Parent sketch plus heavy-hitter heap.
    pub sketch: CSHeap<Vector2D<i32>, RegularPath>,
}

impl CountTopKOctoAggregator {
    /// Creates an aggregator tracking the `top_k` heaviest keys.
    pub fn new(rows: usize, cols: usize, top_k: usize) -> Self {
        Self {
            sketch: CSHeap::new(rows, cols, top_k),
        }
    }
}

impl OctoAggregator for CountTopKOctoAggregator {
    type Delta = KeyedCountDelta;

    #[inline(always)]
    fn apply(&mut self, delta: KeyedCountDelta) {
        self.sketch.cs_mut().apply_delta(delta.delta);
        let key = heap_item_to_sketch_input(&delta.key);
        let estimate = self.sketch.cs().estimate(&key);
        self.sketch.heap_mut().update(&key, estimate as i64);
    }
}

// -- DDSketch --

/// DDSketch worker: one-byte counters over the parent's bucket space.
///
/// Appendix C applies OctoSketch to DDSketch the same way as to Count-Min - one
/// array of counters, promote and clear at τ. Bucket indices are sparse and
/// signed, so the counters are held in a map rather than a dense row.
#[derive(Clone, Debug)]
pub struct DdWorkerSketch {
    /// An empty sketch kept only for its logarithmic bucket mapping.
    mapping: DDSketch,
    counters: HashMap<i32, u8>,
}

impl DdWorkerSketch {
    /// Creates a worker over the bucket space of a DDSketch with this `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            mapping: DDSketch::new(alpha),
            counters: HashMap::new(),
        }
    }

    /// Relative accuracy of the bucket space this worker maps into.
    pub fn alpha(&self) -> f64 {
        self.mapping.alpha()
    }

    /// Counts still held back from the aggregator, by bucket index.
    pub fn residual(&self) -> &HashMap<i32, u8> {
        &self.counters
    }

    /// Total sample count still held back from the aggregator.
    ///
    /// A quantile read from the parent is off by at most this many ranks, so
    /// dividing it by the stream length bounds the rank error that promotion
    /// contributes on top of the sketch's own α.
    pub fn held_back(&self) -> u64 {
        self.counters.values().map(|&c| c as u64).sum()
    }

    /// Promotes and clears every bucket still holding a partial count.
    ///
    /// Until this runs, a bucket under the threshold is missing from the parent
    /// entirely - so `count`, `min`, `max`, `sum` and the extreme quantiles are
    /// wrong without bound, not within alpha. Flush before reading those.
    pub fn flush(&mut self, emit: &mut impl FnMut(DdDelta)) {
        for (index, count) in self.counters.drain() {
            if count != 0 {
                emit(DdDelta {
                    index,
                    value: count as u64,
                });
            }
        }
    }

    /// Adds a sample, promoting and clearing the bucket that reaches `threshold`.
    ///
    /// Values the parent sketch would itself drop - non-positive, non-finite or
    /// outside the indexable range - are dropped here too.
    pub fn add_emit_delta(&mut self, value: f64, threshold: u32, emit: &mut impl FnMut(DdDelta)) {
        let Some(index) = self.mapping.bucket_index_for(value) else {
            return;
        };
        let threshold = threshold.clamp(1, MAX_PROMASK) as u8;
        let counter = self.counters.entry(index).or_insert(0);
        *counter += 1;
        if *counter >= threshold {
            emit(DdDelta {
                index,
                value: *counter as u64,
            });
            self.counters.remove(&index);
        }
    }
}

/// OctoSketch worker backed by a compact DDSketch bucket store.
pub struct DdOctoWorker {
    sketch: DdWorkerSketch,
    threshold: OctoThreshold,
}

impl DdOctoWorker {
    /// Creates a worker with a private threshold fixed at `DD_PROMASK`.
    pub fn new(alpha: f64) -> Self {
        Self::with_threshold(alpha, OctoThreshold::new(DD_PROMASK))
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(alpha: f64, threshold: OctoThreshold) -> Self {
        Self {
            sketch: DdWorkerSketch::new(alpha),
            threshold,
        }
    }
}

impl OctoWorker for DdOctoWorker {
    type Delta = DdDelta;
    /// `None` for an input DDSketch cannot index; the conversion happens at
    /// preparation so a non-numeric key never crosses a thread.
    type Payload = Option<f64>;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        let Some(value) = payload else {
            return;
        };
        self.sketch
            .add_emit_delta(*value, self.threshold.get(), emit);
    }

    fn flush<F>(&mut self, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch.flush(emit);
    }
}

/// OctoSketch parent wrapping a full-precision `DDSketch`.
pub struct DdOctoAggregator {
    /// Parent quantile sketch updated by worker deltas.
    pub sketch: DDSketch,
}

impl DdOctoAggregator {
    /// Creates an aggregator with the given relative accuracy.
    pub fn new(alpha: f64) -> Self {
        Self {
            sketch: DDSketch::new(alpha),
        }
    }
}

impl OctoAggregator for DdOctoAggregator {
    type Delta = DdDelta;

    #[inline(always)]
    fn apply(&mut self, delta: DdDelta) {
        self.sketch.apply_delta(delta);
    }
}

// -- HyperLogLog --

/// OctoSketch worker backed by `HyperLogLog`.
///
/// Cardinality sketches merge by `max`, so this worker never clears a register;
/// its threshold is the exponent τ in the paper's `|2^C' - 2^C| >= 2^τ` rule and
/// is held fixed rather than driven by the aggregator's queue controller.
pub struct HllOctoWorker {
    child: HyperLogLog<Classic>,
    threshold: u8,
}

/// Largest HLL threshold exponent that can still fire at precision `p`.
///
/// A register holds a leading-zero count of at most `64 - p + 1`, so the gain
/// `2^C' - 2^C` never reaches `2^(64 - p)`. A threshold at or above that
/// silently promotes nothing at all, leaving the parent empty.
#[inline(always)]
pub const fn max_hll_threshold(precision: u8) -> u8 {
    64 - precision
}

impl HllOctoWorker {
    /// Creates a HyperLogLog-backed Octo worker at the default threshold,
    /// which promotes every register improvement.
    pub fn new() -> Self {
        Self::with_threshold(HLL_PROMASK)
    }

    /// Creates a worker that promotes a register only once the improvement in
    /// `2^register` reaches `2^threshold`.
    ///
    /// # Panics
    ///
    /// If `threshold` is large enough that no register improvement can ever
    /// reach it. Such a worker promotes nothing at all and leaves the parent
    /// empty, which is worse than any accuracy trade the caller was after, so
    /// it is refused rather than silently accepted. Useful values are small -
    /// the paper finds 2 sufficient, and 0 makes the parent exact.
    pub fn with_threshold(threshold: u8) -> Self {
        let child = HyperLogLog::<Classic>::default();
        let precision = child.registers_as_slice().len().ilog2() as u8;
        let ceiling = max_hll_threshold(precision);
        assert!(
            threshold < ceiling,
            "HLL threshold {threshold} can never fire at precision {precision}; \
             the largest register gain is below 2^{ceiling}"
        );
        Self { child, threshold }
    }
}

impl Default for HllOctoWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl OctoWorker for HllOctoWorker {
    type Delta = HllDelta;
    /// The single canonical-seed hash; HLL never needs the key itself.
    type Payload = u64;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.child
            .insert_emit_delta_with_hash_and_threshold(*payload, self.threshold, emit);
    }

    /// Ships every non-empty register. At the default threshold of 0 nothing is
    /// ever held back and this is redundant, but a worker running a positive
    /// threshold has registers the parent has not seen, and a missing register
    /// reads there as an empty bucket rather than a low one.
    fn flush<F>(&mut self, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        for (pos, value) in self.child.registers_as_slice().iter().enumerate() {
            if *value != 0 {
                emit(HllDelta {
                    pos: pos as u32,
                    value: *value,
                });
            }
        }
    }
}

/// OctoSketch parent wrapping a full-precision `HyperLogLog<Classic>`.
pub struct HllOctoAggregator {
    /// Parent HyperLogLog sketch updated by worker deltas.
    pub sketch: HyperLogLog<Classic>,
}

impl HllOctoAggregator {
    /// Creates an aggregator over a default-precision HyperLogLog.
    pub fn new() -> Self {
        Self {
            sketch: HyperLogLog::default(),
        }
    }
}

impl Default for HllOctoAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl OctoAggregator for HllOctoAggregator {
    type Delta = HllDelta;

    #[inline(always)]
    fn apply(&mut self, delta: HllDelta) {
        self.sketch.apply_delta(delta);
    }
}

// -- UnivMon --

/// Promotion threshold for UnivMon layer `layer`, given the pyramid's base τ.
///
/// A UnivMon layer only receives the keys that survive `layer` coin flips, so
/// layer L carries roughly `n / 2^L` of the stream. One threshold across the
/// whole pyramid would be sized for layer 0 and would starve the deep layers
/// outright - and those are exactly the layers the recursive estimator leans on
/// for cardinality. Halving τ per layer keeps each layer's promotion rate in
/// proportion to the traffic it actually sees, and the floor of 1 makes the
/// deepest layers exact, where they have too little traffic to threshold at all.
#[inline(always)]
pub fn univmon_layer_threshold(base: u32, layer: usize) -> u32 {
    if layer >= u32::BITS as usize {
        return 1;
    }
    (base >> layer).max(1)
}

/// OctoSketch worker backed by one compact sketch per UnivMon layer.
///
/// An insert reaches layers `0..=bottom_layer_for(key)`, exactly the set a
/// single-threaded `UnivMon::insert` touches, because the layer selector is a
/// pure function of the key's hash. The worker keeps no heavy-hitter heap;
/// that lives only in the aggregator.
pub struct UnivMonOctoWorker {
    layers: Vec<L2hhWorkerSketch>,
    threshold: OctoThreshold,
    worker_id: u32,
    weight_total: u64,
}

impl UnivMonOctoWorker {
    /// Creates a worker mirroring a `UnivMon` of these dimensions.
    pub fn new(worker_id: usize, sketch_row: usize, sketch_col: usize, layer_size: usize) -> Self {
        Self::with_threshold(
            worker_id,
            sketch_row,
            sketch_col,
            layer_size,
            OctoThreshold::new(UNIVMON_PROMASK),
        )
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(
        worker_id: usize,
        sketch_row: usize,
        sketch_col: usize,
        layer_size: usize,
        threshold: OctoThreshold,
    ) -> Self {
        let layer_size = layer_size.max(1);
        Self {
            layers: (0..layer_size)
                .map(|layer| L2hhWorkerSketch::new(sketch_row, sketch_col, layer))
                .collect(),
            threshold,
            worker_id: worker_id as u32,
            weight_total: 0,
        }
    }

    /// Bytes held across every layer's counter array.
    pub fn counter_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.counter_bytes()).sum()
    }
}

impl OctoWorker for UnivMonOctoWorker {
    type Delta = LayeredCountDelta;
    type Payload = UnivMonInput;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.weight_total += 1;
        let threshold = self.threshold.get();
        let (worker_id, weight_total) = (self.worker_id, self.weight_total);
        let bottom = payload.bottom.min(self.layers.len() - 1);
        for (layer, hashed) in payload
            .layer_hashes
            .iter()
            .copied()
            .enumerate()
            .take(bottom + 1)
        {
            let layer_threshold = univmon_layer_threshold(threshold, layer);
            let key = &payload.key;
            self.layers[layer].insert_hash_emit_delta(hashed, layer_threshold, &mut |delta| {
                emit(LayeredCountDelta {
                    layer: layer as u32,
                    key: key.clone(),
                    delta,
                    worker_id,
                    weight_total,
                })
            });
        }
    }
}

/// OctoSketch parent wrapping a full-precision `UnivMon`.
///
/// It holds the pipeline's only heavy-hitter heaps and restores the total
/// weight UnivMon's g-sum queries divide by from the running totals its
/// workers report.
pub struct UnivMonOctoAggregator {
    /// Parent UnivMon updated by worker deltas.
    pub sketch: UnivMon,
    threshold: OctoThreshold,
    /// τ the per-layer fidelity was last computed from.
    fidelity_threshold: u32,
    fidelity: Vec<UnivMonDeltaFidelity>,
    worker_weights: Vec<u64>,
    total_weight: u64,
}

impl UnivMonOctoAggregator {
    /// Creates an aggregator whose workers promote at `threshold`.
    ///
    /// The threshold decides whether candidate sets can ever be called
    /// complete: only a threshold of 1 delivers every insert to the heaps.
    pub fn new(
        heap_size: usize,
        sketch_row: usize,
        sketch_col: usize,
        layer_size: usize,
        threshold: u32,
    ) -> Self {
        Self::with_threshold(
            heap_size,
            sketch_row,
            sketch_col,
            layer_size,
            OctoThreshold::new(threshold),
        )
    }

    /// Creates an aggregator that reads the same live threshold its workers do.
    ///
    /// Completeness has to track τ rather than a snapshot of it: with the
    /// adaptive controller running, a layer that started at τ = 1 can be
    /// raised above it mid-stream, and a stale `EveryInsert` verdict would let
    /// `heavy_threshold` take its permissive branch and overcount.
    pub fn with_threshold(
        heap_size: usize,
        sketch_row: usize,
        sketch_col: usize,
        layer_size: usize,
        threshold: OctoThreshold,
    ) -> Self {
        let mut this = Self {
            sketch: UnivMon::init_univmon(heap_size, sketch_row, sketch_col, layer_size),
            fidelity_threshold: u32::MAX,
            fidelity: vec![UnivMonDeltaFidelity::EveryInsert; layer_size],
            threshold,
            worker_weights: Vec::new(),
            total_weight: 0,
        };
        this.refresh_fidelity();
        this
    }

    /// Recomputes per-layer fidelity from the current τ, withdrawing
    /// completeness wherever a layer now thresholds. Withdrawal is one-way: a
    /// layer that has ever thresholded stays partial for the rest of the run.
    fn refresh_fidelity(&mut self) {
        let threshold = self.threshold.get();
        if threshold == self.fidelity_threshold {
            return;
        }
        self.fidelity_threshold = threshold;
        for layer in 0..self.fidelity.len() {
            if univmon_layer_threshold(threshold, layer) > 1 {
                self.fidelity[layer] = UnivMonDeltaFidelity::PromotedOnly;
                // A layer that receives traffic but never promotes any of it
                // sends nothing, so the flag has to come down before a delta
                // arrives rather than when one does.
                self.sketch.mark_layer_candidates_incomplete(layer);
            }
        }
    }
}

impl OctoAggregator for UnivMonOctoAggregator {
    type Delta = LayeredCountDelta;

    #[inline(always)]
    fn apply(&mut self, delta: LayeredCountDelta) {
        self.refresh_fidelity();

        let worker = delta.worker_id as usize;
        if worker >= self.worker_weights.len() {
            self.worker_weights.resize(worker + 1, 0);
        }
        // A single worker's running total only grows, so a report that moves
        // backwards means two workers were built with the same id and their
        // totals are being folded into one slot - which would silently turn
        // the fleet total into a maximum instead of a sum.
        assert!(
            delta.weight_total >= self.worker_weights[worker],
            "worker id {worker} reported a total of {} after {}; worker ids must be distinct",
            delta.weight_total,
            self.worker_weights[worker]
        );
        self.total_weight += delta.weight_total - self.worker_weights[worker];
        self.worker_weights[worker] = delta.weight_total;

        let fidelity = self.fidelity[delta.layer as usize];
        self.sketch.apply_layered_delta(&delta, fidelity);
        self.sketch.set_total_weight(self.total_weight as usize);
    }
}

// -- CocoSketch --

/// Renders an input as the `String` flow key `Coco` and `Elastic` are defined
/// over. Bytes render as lowercase hex, so distinct byte strings stay distinct.
///
/// Both sketches key on a `String`, so a plan has to settle on one rendering:
/// this is what the aggregator will hold, and what `Coco::estimate_key` or
/// `Elastic::query` must be asked for.
pub fn flow_key_string(input: &DataInput<'_>) -> String {
    use std::fmt::Write as _;

    match input {
        DataInput::I8(v) => v.to_string(),
        DataInput::I16(v) => v.to_string(),
        DataInput::I32(v) => v.to_string(),
        DataInput::I64(v) => v.to_string(),
        DataInput::I128(v) => v.to_string(),
        DataInput::ISIZE(v) => v.to_string(),
        DataInput::U8(v) => v.to_string(),
        DataInput::U16(v) => v.to_string(),
        DataInput::U32(v) => v.to_string(),
        DataInput::U64(v) => v.to_string(),
        DataInput::U128(v) => v.to_string(),
        DataInput::USIZE(v) => v.to_string(),
        DataInput::F32(v) => v.to_string(),
        DataInput::F64(v) => v.to_string(),
        DataInput::Str(s) => (*s).to_string(),
        DataInput::String(s) => s.clone(),
        DataInput::Bytes(b) => {
            let mut hex = String::with_capacity(b.len() * 2);
            for byte in *b {
                let _ = write!(hex, "{byte:02x}");
            }
            hex
        }
    }
}

/// OctoSketch worker backed by a compact CocoSketch table.
pub struct CocoOctoWorker {
    sketch: CocoWorkerSketch,
    threshold: OctoThreshold,
}

impl CocoOctoWorker {
    /// Creates a worker with a private threshold fixed at `COCO_PROMASK`.
    pub fn new(w: usize, d: usize) -> Self {
        Self::with_threshold(w, d, OctoThreshold::new(COCO_PROMASK))
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(w: usize, d: usize, threshold: OctoThreshold) -> Self {
        Self {
            sketch: CocoWorkerSketch::new(w, d),
            threshold,
        }
    }

    /// Borrows the worker's table.
    pub fn sketch(&self) -> &CocoWorkerSketch {
        &self.sketch
    }
}

impl OctoWorker for CocoOctoWorker {
    type Delta = CocoDelta;
    /// The rendered flow key; a Coco bucket is nothing without one.
    type Payload = String;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch
            .insert_emit_delta(payload, self.threshold.get(), emit);
    }

    /// Ships every bucket still holding a partial count. Unlike the `*TopK*`
    /// workers, a Coco worker keeps the key beside every counter, so a residual
    /// bucket can be attributed and flushed.
    fn flush<F>(&mut self, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch.flush(emit);
    }
}

/// OctoSketch parent wrapping a full-precision `Coco`.
///
/// §4.4: each `<key, counter>` message is replayed through the parent's own
/// insertion logic, which for CocoSketch is the weighted `Coco::insert` - the
/// aggregator picks its own victim bucket and runs its own election, so a key
/// the worker held lands wherever the *parent's* table would have put it.
pub struct CocoOctoAggregator {
    /// Parent CocoSketch updated by worker deltas.
    pub sketch: Coco,
}

impl CocoOctoAggregator {
    /// Creates an aggregator with a `d` x `w` parent table.
    pub fn new(w: usize, d: usize) -> Self {
        Self {
            sketch: Coco::init_with_size(w, d),
        }
    }
}

impl OctoAggregator for CocoOctoAggregator {
    type Delta = CocoDelta;

    #[inline(always)]
    fn apply(&mut self, delta: CocoDelta) {
        self.sketch.insert(&delta.key, delta.value);
    }
}

// -- Elastic sketch --

/// OctoSketch worker backed by a compact Elastic sketch, both halves.
pub struct ElasticOctoWorker {
    sketch: ElasticWorkerSketch,
    threshold: OctoThreshold,
}

impl ElasticOctoWorker {
    /// Creates a worker with a private threshold fixed at `ELASTIC_PROMASK`.
    pub fn new(bucket_count: usize, light_rows: usize, light_cols: usize) -> Self {
        Self::with_threshold(
            bucket_count,
            light_rows,
            light_cols,
            OctoThreshold::new(ELASTIC_PROMASK),
        )
    }

    /// Creates a worker sharing `threshold` with its peers and the aggregator.
    pub fn with_threshold(
        bucket_count: usize,
        light_rows: usize,
        light_cols: usize,
        threshold: OctoThreshold,
    ) -> Self {
        Self {
            sketch: ElasticWorkerSketch::new(bucket_count, light_rows, light_cols),
            threshold,
        }
    }

    /// Borrows the worker's heavy and light parts.
    pub fn sketch(&self) -> &ElasticWorkerSketch {
        &self.sketch
    }
}

impl OctoWorker for ElasticOctoWorker {
    type Delta = ElasticDelta;
    /// The rendered flow key. The light half could travel as hashes alone, but
    /// the heavy half stores the key and an eviction spills a key the caller
    /// never sent, so the worker hashes for both.
    type Payload = String;

    #[inline(always)]
    fn process<F>(&mut self, payload: &Self::Payload, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch
            .insert_emit_delta(payload, self.threshold.get(), emit);
    }

    /// Ships both halves: residual votes with their resident flow, and every
    /// light-layer counter still under τ.
    fn flush<F>(&mut self, emit: &mut F)
    where
        F: FnMut(Self::Delta),
    {
        self.sketch.flush(emit);
    }
}

/// OctoSketch parent wrapping a full-precision `Elastic`.
///
/// Appendix C splits the two halves: a heavy message is replayed through the
/// parent's own insertion logic as `Elastic::merge_heavy`, so the arrival
/// contests the *parent's* bucket and may evict a different flow than it did on
/// the worker; a light message is an ordinary Count-Min cell delta and is added
/// to the light layer as-is. A worker's evicted resident arrives keyed and goes
/// through `Elastic::absorb_evicted`, which is where the parent learns that a
/// flow it still holds has mass coming through the light layer.
pub struct ElasticOctoAggregator {
    /// Parent Elastic sketch updated by worker deltas.
    pub sketch: Elastic,
}

impl ElasticOctoAggregator {
    /// Creates an aggregator with a `bucket_count` heavy table over a
    /// `light_rows` by `light_cols` light layer.
    pub fn new(bucket_count: i32, light_rows: usize, light_cols: usize) -> Self {
        Self {
            sketch: Elastic::init_with_dimensions(bucket_count, light_rows, light_cols),
        }
    }
}

impl OctoAggregator for ElasticOctoAggregator {
    type Delta = ElasticDelta;

    #[inline(always)]
    fn apply(&mut self, delta: ElasticDelta) {
        match delta {
            ElasticDelta::Heavy {
                key,
                value,
                eviction,
            } => self.sketch.merge_heavy(key, value as i32, eviction),
            ElasticDelta::Evicted { key, votes } => self.sketch.absorb_evicted(key, votes as i32),
            ElasticDelta::Light(delta) => self.sketch.light.apply_delta(delta),
        }
    }
}

// ---------------------------------------------------------------------------
// Plans: geometry plus the preparation that strips the borrow
// ---------------------------------------------------------------------------

/// Builds `CmOctoWorker`s and prepares their per-row hashes.
#[derive(Clone, Debug)]
pub struct CmOctoPlan {
    rows: usize,
    cols: usize,
    threshold: OctoThreshold,
}

/// Builds `CountOctoWorker`s and prepares their per-row hashes.
#[derive(Clone, Debug)]
pub struct CountOctoPlan {
    rows: usize,
    cols: usize,
    threshold: OctoThreshold,
}

/// Builds `CmTopKOctoWorker`s; payloads carry the key as well as the hashes.
#[derive(Clone, Debug)]
pub struct CmTopKOctoPlan {
    rows: usize,
    cols: usize,
    threshold: OctoThreshold,
}

/// Builds `CountTopKOctoWorker`s; payloads carry the key as well as the hashes.
#[derive(Clone, Debug)]
pub struct CountTopKOctoPlan {
    rows: usize,
    cols: usize,
    threshold: OctoThreshold,
}

/// Builds `HllOctoWorker`s; payloads are the single canonical-seed hash.
#[derive(Clone, Debug)]
pub struct HllOctoPlan {
    threshold: u8,
}

/// Builds `DdOctoWorker`s; payloads are the numeric value.
#[derive(Clone, Debug)]
pub struct DdOctoPlan {
    alpha: f64,
    threshold: OctoThreshold,
}

/// Builds `UnivMonOctoWorker`s; payloads carry one hash per touched layer plus
/// the key, since the aggregator's heaps need it.
#[derive(Clone, Debug)]
pub struct UnivMonOctoPlan {
    rows: usize,
    cols: usize,
    layers: usize,
    threshold: OctoThreshold,
}

macro_rules! counter_plan {
    ($plan:ident, $worker:ident, $default:expr) => {
        impl $plan {
            /// Creates a plan at the sketch's default threshold.
            pub fn new(rows: usize, cols: usize) -> Self {
                Self::with_threshold(rows, cols, OctoThreshold::new($default))
            }

            /// Creates a plan whose workers share `threshold`.
            pub fn with_threshold(rows: usize, cols: usize, threshold: OctoThreshold) -> Self {
                Self {
                    rows,
                    cols,
                    threshold,
                }
            }

            /// The threshold this plan's workers read. Clone it into
            /// `OctoConfig::threshold` so the controller drives the same value.
            pub fn threshold(&self) -> &OctoThreshold {
                &self.threshold
            }
        }
    };
}

impl CmOctoPlan {
    /// Builds the parent this plan's workers feed.
    ///
    /// Worker and parent geometry are two independent arguments to `run_octo`,
    /// and a mismatch is silent: deltas name rows the parent has, the rows
    /// beyond them stay zero, and Count-Min's min-over-rows estimate is zero
    /// for every key. Building the parent from the plan removes the chance.
    pub fn aggregator(&self) -> CmOctoAggregator {
        CmOctoAggregator::new(self.rows, self.cols)
    }
}

impl CountOctoPlan {
    /// Builds the parent this plan's workers feed. See `CmOctoPlan::aggregator`.
    pub fn aggregator(&self) -> CountOctoAggregator {
        CountOctoAggregator::new(self.rows, self.cols)
    }
}

impl CmTopKOctoPlan {
    /// Builds the parent this plan's workers feed, tracking `top_k` keys.
    pub fn aggregator(&self, top_k: usize) -> CmTopKOctoAggregator {
        CmTopKOctoAggregator::new(self.rows, self.cols, top_k)
    }
}

impl CountTopKOctoPlan {
    /// Builds the parent this plan's workers feed, tracking `top_k` keys.
    pub fn aggregator(&self, top_k: usize) -> CountTopKOctoAggregator {
        CountTopKOctoAggregator::new(self.rows, self.cols, top_k)
    }
}

impl DdOctoPlan {
    /// Builds the parent this plan's workers feed, over the same bucket space.
    pub fn aggregator(&self) -> DdOctoAggregator {
        DdOctoAggregator::new(self.alpha)
    }
}

impl HllOctoPlan {
    /// Builds the parent this plan's workers feed.
    pub fn aggregator(&self) -> HllOctoAggregator {
        HllOctoAggregator::new()
    }
}

impl UnivMonOctoPlan {
    /// Builds the parent this plan's workers feed, sharing their threshold.
    pub fn aggregator(&self, heap_size: usize) -> UnivMonOctoAggregator {
        UnivMonOctoAggregator::with_threshold(
            heap_size,
            self.rows,
            self.cols,
            self.layers,
            self.threshold.clone(),
        )
    }
}

counter_plan!(CmOctoPlan, CmOctoWorker, CM_PROMASK);
counter_plan!(CountOctoPlan, CountOctoWorker, COUNT_PROMASK);
counter_plan!(CmTopKOctoPlan, CmTopKOctoWorker, CM_PROMASK);
counter_plan!(CountTopKOctoPlan, CountTopKOctoWorker, COUNT_PROMASK);

impl OctoPlan for CmOctoPlan {
    type Worker = CmOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        CmOctoWorker::with_threshold(self.rows, self.cols, self.threshold.clone())
    }

    fn prepare(&self, input: &DataInput<'_>) -> RowHashes {
        CmWorkerSketch::hashes(self.rows, input)
    }
}

impl OctoPlan for CountOctoPlan {
    type Worker = CountOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        CountOctoWorker::with_threshold(self.rows, self.cols, self.threshold.clone())
    }

    fn prepare(&self, input: &DataInput<'_>) -> RowHashes {
        CmWorkerSketch::hashes(self.rows, input)
    }
}

impl OctoPlan for CmTopKOctoPlan {
    type Worker = CmTopKOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        CmTopKOctoWorker::with_threshold(self.rows, self.cols, self.threshold.clone())
    }

    fn prepare(&self, input: &DataInput<'_>) -> KeyedHashes {
        KeyedHashes {
            hashes: CmWorkerSketch::hashes(self.rows, input),
            key: input_to_owned(input),
        }
    }
}

impl OctoPlan for CountTopKOctoPlan {
    type Worker = CountTopKOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        CountTopKOctoWorker::with_threshold(self.rows, self.cols, self.threshold.clone())
    }

    fn prepare(&self, input: &DataInput<'_>) -> KeyedHashes {
        KeyedHashes {
            hashes: CmWorkerSketch::hashes(self.rows, input),
            key: input_to_owned(input),
        }
    }
}

impl HllOctoPlan {
    /// Creates a plan at the default threshold, which promotes every register
    /// improvement and leaves the parent exact.
    pub fn new() -> Self {
        Self::with_threshold(HLL_PROMASK)
    }

    /// Creates a plan at a custom threshold exponent.
    pub fn with_threshold(threshold: u8) -> Self {
        // Fail here rather than at the first worker, where it would surface on
        // a background thread.
        let _ = HllOctoWorker::with_threshold(threshold);
        Self { threshold }
    }
}

impl Default for HllOctoPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl OctoPlan for HllOctoPlan {
    type Worker = HllOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        HllOctoWorker::with_threshold(self.threshold)
    }

    fn prepare(&self, input: &DataInput<'_>) -> u64 {
        HyperLogLog::<Classic>::canonical_hash(input)
    }
}

impl DdOctoPlan {
    /// Creates a plan at the DDSketch default threshold.
    pub fn new(alpha: f64) -> Self {
        Self::with_threshold(alpha, OctoThreshold::new(DD_PROMASK))
    }

    /// Creates a plan whose workers share `threshold`.
    pub fn with_threshold(alpha: f64, threshold: OctoThreshold) -> Self {
        Self { alpha, threshold }
    }

    /// Relative accuracy of the bucket space.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The threshold this plan's workers read.
    pub fn threshold(&self) -> &OctoThreshold {
        &self.threshold
    }
}

impl OctoPlan for DdOctoPlan {
    type Worker = DdOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        DdOctoWorker::with_threshold(self.alpha, self.threshold.clone())
    }

    fn prepare(&self, input: &DataInput<'_>) -> Option<f64> {
        data_input_to_f64(input).ok()
    }
}

/// Builds `CocoOctoWorker`s; payloads are the rendered flow key.
#[derive(Clone, Debug)]
pub struct CocoOctoPlan {
    w: usize,
    d: usize,
    threshold: OctoThreshold,
}

impl CocoOctoPlan {
    /// Creates a plan at the CocoSketch default threshold.
    pub fn new(w: usize, d: usize) -> Self {
        Self::with_threshold(w, d, OctoThreshold::new(COCO_PROMASK))
    }

    /// Creates a plan whose workers share `threshold`.
    pub fn with_threshold(w: usize, d: usize, threshold: OctoThreshold) -> Self {
        Self { w, d, threshold }
    }

    /// The threshold this plan's workers read.
    pub fn threshold(&self) -> &OctoThreshold {
        &self.threshold
    }

    /// Builds the parent this plan's workers feed. See `CmOctoPlan::aggregator`.
    pub fn aggregator(&self) -> CocoOctoAggregator {
        CocoOctoAggregator::new(self.w, self.d)
    }
}

impl OctoPlan for CocoOctoPlan {
    type Worker = CocoOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        CocoOctoWorker::with_threshold(self.w, self.d, self.threshold.clone())
    }

    fn prepare(&self, input: &DataInput<'_>) -> String {
        flow_key_string(input)
    }
}

/// Builds `ElasticOctoWorker`s; payloads are the rendered flow key.
#[derive(Clone, Debug)]
pub struct ElasticOctoPlan {
    bucket_count: i32,
    light_rows: usize,
    light_cols: usize,
    threshold: OctoThreshold,
}

impl ElasticOctoPlan {
    /// Creates a plan at the Elastic sketch default threshold.
    pub fn new(bucket_count: i32, light_rows: usize, light_cols: usize) -> Self {
        Self::with_threshold(
            bucket_count,
            light_rows,
            light_cols,
            OctoThreshold::new(ELASTIC_PROMASK),
        )
    }

    /// Creates a plan whose workers share `threshold`.
    pub fn with_threshold(
        bucket_count: i32,
        light_rows: usize,
        light_cols: usize,
        threshold: OctoThreshold,
    ) -> Self {
        Self {
            bucket_count,
            light_rows,
            light_cols,
            threshold,
        }
    }

    /// The threshold this plan's workers read.
    pub fn threshold(&self) -> &OctoThreshold {
        &self.threshold
    }

    /// Builds the parent this plan's workers feed.
    ///
    /// Both halves have to match: a light delta names a cell by index, and a
    /// heavy delta lands in the bucket the parent hashes the key to, so a
    /// parent of different dimensions silently misplaces every message.
    pub fn aggregator(&self) -> ElasticOctoAggregator {
        ElasticOctoAggregator::new(self.bucket_count, self.light_rows, self.light_cols)
    }
}

impl OctoPlan for ElasticOctoPlan {
    type Worker = ElasticOctoWorker;

    fn worker(&self, _worker_id: usize) -> Self::Worker {
        ElasticOctoWorker::with_threshold(
            self.bucket_count.max(1) as usize,
            self.light_rows,
            self.light_cols,
            self.threshold.clone(),
        )
    }

    fn prepare(&self, input: &DataInput<'_>) -> String {
        flow_key_string(input)
    }
}

impl UnivMonOctoPlan {
    /// Creates a plan at the UnivMon default threshold.
    pub fn new(rows: usize, cols: usize, layers: usize) -> Self {
        Self::with_threshold(rows, cols, layers, OctoThreshold::new(UNIVMON_PROMASK))
    }

    /// Creates a plan whose workers share `threshold`.
    pub fn with_threshold(
        rows: usize,
        cols: usize,
        layers: usize,
        threshold: OctoThreshold,
    ) -> Self {
        Self {
            rows,
            cols,
            layers: layers.max(1),
            threshold,
        }
    }

    /// The threshold this plan's workers read.
    pub fn threshold(&self) -> &OctoThreshold {
        &self.threshold
    }
}

impl OctoPlan for UnivMonOctoPlan {
    type Worker = UnivMonOctoWorker;

    fn worker(&self, worker_id: usize) -> Self::Worker {
        UnivMonOctoWorker::with_threshold(
            worker_id,
            self.rows,
            self.cols,
            self.layers,
            self.threshold.clone(),
        )
    }

    fn prepare(&self, input: &DataInput<'_>) -> UnivMonInput {
        // Only the layers this key actually reaches get hashed. Layer depth is
        // geometric, so that is nearly always one or two.
        let bottom = bottom_layer_for_hash(hash64_seeded(BOTTOM_LAYER_FINDER, input), self.layers);
        UnivMonInput {
            bottom,
            layer_hashes: (0..=bottom).map(|l| hash128_seeded(l, input)).collect(),
            key: input_to_owned(input),
        }
    }
}

// ---------------------------------------------------------------------------
// Core execution engine
// ---------------------------------------------------------------------------

#[cfg(feature = "octo-runtime")]
/// Runs the OctoSketch multi-threaded insert protocol.
///
/// 1. Routes `inputs` to workers per `config.partition`.
/// 2. Each worker maintains a compact child sketch, emitting deltas on its own channel.
/// 3. The aggregator applies deltas to the parent and, if configured, adjusts τ.
/// 4. Returns the fully-merged parent sketch.
pub fn run_octo<L, P>(
    inputs: &[DataInput<'_>],
    config: &OctoConfig,
    plan: L,
    parent_factory: impl FnOnce() -> P,
) -> OctoResult<P>
where
    L: OctoPlan,
    P: OctoAggregator<Delta = <L::Worker as OctoWorker>::Delta> + Send + Sync + 'static,
{
    let mut runtime = OctoRuntime::new(config, plan, parent_factory);
    for input in inputs {
        runtime.insert(input.clone());
    }
    runtime.finish()
}

#[cfg(test)]
mod worker_tests {
    use super::*;

    #[test]
    fn compact_worker_clears_the_counter_it_promotes() {
        let mut worker = CmWorkerSketch::new(3, 64);
        let key = DataInput::U64(42);
        let mut deltas: Vec<CmDelta> = Vec::new();

        for _ in 0..(CM_PROMASK - 1) {
            worker.insert_emit_delta(&key, CM_PROMASK, &mut |d| deltas.push(d));
        }
        assert!(deltas.is_empty(), "should not promote before the threshold");

        worker.insert_emit_delta(&key, CM_PROMASK, &mut |d| deltas.push(d));
        assert_eq!(deltas.len(), 3, "one promotion per row");
        for d in &deltas {
            assert_eq!(d.value, CM_PROMASK);
        }
        assert!(
            worker.residual().iter().all(|&c| c == 0),
            "Algorithm 1 clears the counter it promotes"
        );
    }

    #[test]
    fn compact_worker_uses_a_quarter_of_the_counter_memory() {
        let (rows, cols) = (3usize, 4096usize);
        let compact = CmWorkerSketch::new(rows, cols).counter_bytes();
        let full = rows * cols * std::mem::size_of::<i32>();
        assert_eq!(compact * 4, full, "one byte per counter instead of four");
    }

    #[test]
    fn compact_worker_honours_a_custom_threshold() {
        let mut worker = CmWorkerSketch::new(1, 16);
        let key = DataInput::U64(7);
        let mut promotions = 0usize;
        for _ in 0..100 {
            worker.insert_emit_delta(&key, 10, &mut |_| promotions += 1);
        }
        assert_eq!(
            promotions, 10,
            "a threshold of 10 promotes every 10 inserts"
        );
    }

    #[test]
    fn compact_count_worker_promotes_on_magnitude_and_clears() {
        let mut worker = CountWorkerSketch::new(3, 64);
        let key = DataInput::U64(99);
        let mut deltas: Vec<CountDelta> = Vec::new();
        for _ in 0..200 {
            worker.insert_emit_delta(&key, COUNT_PROMASK, &mut |d| deltas.push(d));
        }
        assert!(!deltas.is_empty());
        for d in &deltas {
            assert_eq!(d.value.unsigned_abs(), COUNT_PROMASK);
        }
        assert!(worker.residual().iter().all(|&c| c.unsigned_abs() < 31));
    }

    #[test]
    fn keyed_deltas_carry_the_key_that_triggered_them() {
        let mut worker = CmWorkerSketch::new(2, 8);
        let key = DataInput::Str("flow-a");
        let mut keyed: Vec<KeyedCmDelta> = Vec::new();
        for _ in 0..CM_PROMASK {
            worker.insert_hashes_emit_keyed_delta(
                &CmWorkerSketch::hashes(2, &key),
                &input_to_owned(&key),
                CM_PROMASK,
                &mut |d| keyed.push(d),
            );
        }
        assert!(!keyed.is_empty());
        for d in &keyed {
            assert_eq!(d.key, input_to_owned(&key));
        }
    }

    #[test]
    fn dd_worker_promotes_and_clears_a_bucket() {
        let mut worker = DdWorkerSketch::new(0.01);
        let mut deltas: Vec<DdDelta> = Vec::new();
        for _ in 0..CM_PROMASK {
            worker.add_emit_delta(42.0, CM_PROMASK, &mut |d| deltas.push(d));
        }
        assert_eq!(deltas.len(), 1, "one promotion per tau samples");
        assert_eq!(deltas[0].value as u32, CM_PROMASK);
        assert!(
            worker.residual().is_empty(),
            "the promoted bucket must be cleared"
        );
    }

    #[test]
    fn dd_worker_drops_what_the_parent_would_drop() {
        let mut worker = DdWorkerSketch::new(0.01);
        let mut promoted = 0usize;
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            for _ in 0..1_000 {
                worker.add_emit_delta(value, 1, &mut |_| promoted += 1);
            }
        }
        assert_eq!(
            promoted, 0,
            "non-positive and non-finite values are dropped"
        );
    }

    #[test]
    fn dd_promotion_bounds_the_quantile_rank_error_by_what_it_holds_back() {
        let alpha = 0.01;
        let mut worker = DdWorkerSketch::new(alpha);
        let mut parent = DDSketch::new(alpha);
        let mut exact: Vec<f64> = Vec::new();

        let n = 200_000u64;
        for i in 1..=n {
            let value = 1.0 + (i as f64 * 7.0) % 999.0;
            exact.push(value);
            worker.add_emit_delta(value, DD_PROMASK, &mut |d| parent.apply_delta(d));
        }
        exact.sort_by(f64::total_cmp);

        // Unlike a Count-Min row, an un-promoted bucket is missing from the
        // parent outright, so the promotion error shows up as a rank shift
        // bounded by the mass the worker still holds.
        assert_eq!(
            worker.held_back() + parent.get_count(),
            n,
            "every sample is either promoted or still held"
        );
        let rank_slack = worker.held_back() as f64 / n as f64;
        assert!(
            rank_slack < 0.05,
            "a threshold of {DD_PROMASK} held back {rank_slack:.3} of the stream"
        );

        for q in [0.1, 0.5, 0.9, 0.99] {
            let got = parent
                .get_value_at_quantile(q)
                .expect("parent received samples");
            let lo = exact[(((q - rank_slack).max(0.0)) * (exact.len() - 1) as f64) as usize]
                * (1.0 - alpha);
            let hi = exact[(((q + rank_slack).min(1.0)) * (exact.len() - 1) as f64) as usize]
                * (1.0 + alpha);
            assert!(
                (lo..=hi).contains(&got),
                "q={q}: {got} outside [{lo}, {hi}] at rank slack {rank_slack:.3}"
            );
        }
    }

    #[test]
    fn dd_a_higher_threshold_holds_back_more_of_a_sparse_stream() {
        let alpha = 0.01;
        let n = 20_000u64;
        let mut held = Vec::new();
        for threshold in [2u32, 8, 31] {
            let mut worker = DdWorkerSketch::new(alpha);
            for i in 1..=n {
                worker.add_emit_delta(1.0 + (i as f64 * 7.0) % 999.0, threshold, &mut |_| {});
            }
            held.push((threshold, worker.held_back()));
        }
        assert!(
            held[0].1 < held[1].1 && held[1].1 < held[2].1,
            "held-back mass must grow with the threshold: {held:?}"
        );
    }

    #[test]
    #[should_panic(expected = "exceeds the 128-bit hash budget")]
    fn a_univmon_worker_refuses_a_geometry_that_outruns_the_hash() {
        L2hhWorkerSketch::new(13, 2048, 0);
    }

    #[test]
    fn the_widest_accepted_univmon_geometry_still_addresses_distinct_rows() {
        // 11 rows x 11 bits = 121, just inside the budget.
        let mut worker = L2hhWorkerSketch::new(11, 2048, 0);
        let mut cells: Vec<(u32, u32)> = Vec::new();
        worker.insert_emit_delta(&DataInput::U64(7), 1, &mut |d| cells.push((d.row, d.col)));
        assert_eq!(cells.len(), 11, "every row promotes at threshold 1");
        let rows: std::collections::HashSet<u32> = cells.iter().map(|(r, _)| *r).collect();
        assert_eq!(rows.len(), 11, "each row must land on its own index");
    }

    #[test]
    fn every_worker_reads_one_threshold_the_same_way() {
        let tau = 200u32;
        let key = DataInput::U64(9);

        let mut compact = CountWorkerSketch::new(1, 64);
        let mut compact_values: Vec<i32> = Vec::new();
        let mut full = Count::<Vector2D<i32>, RegularPath>::with_dimensions(1, 64);
        let mut full_values: Vec<i32> = Vec::new();
        for _ in 0..400 {
            compact.insert_emit_delta(&key, tau, &mut |d| compact_values.push(d.value));
            full.insert_emit_delta_with_threshold(&key, tau, &mut |d| full_values.push(d.value));
        }

        assert!(!compact_values.is_empty(), "tau must be reachable at all");
        assert_eq!(
            compact_values, full_values,
            "the compact worker and the full sketch must promote identically"
        );
        for value in &compact_values {
            assert_eq!(value.unsigned_abs(), MAX_PROMASK);
        }
    }

    #[test]
    #[should_panic(expected = "can never fire at precision")]
    fn an_hll_worker_refuses_a_threshold_no_register_can_reach() {
        // A register tops out at 64 - PRECISION + 1, so the gain 2^C' - 2^C
        // never reaches 2^(64 - PRECISION). Such a worker promoted nothing at
        // all and left the parent estimating zero.
        HllOctoWorker::with_threshold(max_hll_threshold(14));
    }

    #[test]
    fn any_hll_threshold_above_zero_costs_cardinality_accuracy() {
        // HLL merges by max, so a register the worker has not promoted reads at
        // the parent as an *empty* bucket rather than a low one, and the
        // harmonic mean the estimator is built on collapses. Theorem 4 says the
        // same thing from the other side: it only promises equality once
        // Z > 2*alpha_m*m^2*2^(tau-2), which at m = 16384 is around 3.9e8 for
        // tau = 2 - far above the 50k here. Zero is the only threshold that
        // keeps the estimate.
        let truth = 50_000u64;
        let mut sweep: Vec<(u8, usize, usize)> = Vec::new();
        for threshold in [0u8, 1, 2, 4] {
            let mut child = HyperLogLog::<Classic>::default();
            let mut parent = HyperLogLog::<Classic>::default();
            let mut promoted = 0usize;
            for i in 0..truth {
                child.insert_emit_delta_with_threshold(&DataInput::U64(i), threshold, &mut |d| {
                    promoted += 1;
                    parent.apply_delta(d);
                });
            }
            sweep.push((threshold, promoted, parent.estimate()));
        }

        let mut reference = HyperLogLog::<Classic>::default();
        for i in 0..truth {
            reference.insert(&DataInput::U64(i));
        }
        assert_eq!(
            sweep[0].2,
            reference.estimate(),
            "threshold 0 must leave the parent exactly ideal"
        );

        for pair in sweep.windows(2) {
            let (lo_tau, lo_msgs, lo_est) = pair[0];
            let (hi_tau, hi_msgs, hi_est) = pair[1];
            assert!(
                hi_msgs < lo_msgs,
                "tau {hi_tau} should send fewer messages than {lo_tau}: {hi_msgs} vs {lo_msgs}"
            );
            assert!(
                hi_est <= lo_est,
                "tau {hi_tau} should not estimate higher than {lo_tau}: {hi_est} vs {lo_est}"
            );
        }
        assert!(
            (sweep[3].2 as f64) < truth as f64 * 0.5,
            "a threshold of 4 loses most of the cardinality: {:?}",
            sweep[3]
        );
        assert_eq!(max_hll_threshold(14), 50);
    }

    #[test]
    fn a_ddsketch_delta_from_a_finer_mapping_is_dropped_not_allocated() {
        // The worker's index space is bounded by its own alpha; a delta from a
        // much finer sketch names an index this store cannot hold, and growing
        // the dense array across that gap would allocate gigabytes.
        let mut parent = DDSketch::new(0.01);
        parent.add(&50.0);
        let before = parent.store_counts().len();
        parent.apply_delta(DdDelta {
            index: i32::MAX / 2,
            value: 4,
        });
        parent.apply_delta(DdDelta {
            index: i32::MIN / 2,
            value: 4,
        });
        assert_eq!(parent.store_counts().len(), before, "store must not grow");
        assert_eq!(parent.get_count(), 1, "out-of-range deltas are dropped");
    }

    #[test]
    #[should_panic(expected = "one hash per row")]
    fn a_count_worker_refuses_hashes_built_for_a_different_geometry() {
        let mut worker = CountWorkerSketch::new(4, 16);
        let hashes = CmWorkerSketch::hashes(2, &DataInput::U64(1));
        worker.insert_hashes_emit_delta(&hashes, COUNT_PROMASK, &mut |_| {});
    }

    #[test]
    #[should_panic(expected = "one hash per row")]
    fn a_worker_refuses_hashes_built_for_a_different_geometry() {
        // The hashes are a public entry point, so a plan/worker mismatch has to
        // fail loudly. Silently using the rows it was given would leave the rest
        // at zero, and Count-Min's min-over-rows estimate is then zero for
        // every key.
        let mut worker = CmWorkerSketch::new(4, 16);
        let hashes = CmWorkerSketch::hashes(2, &DataInput::U64(1));
        worker.insert_hashes_emit_delta(&hashes, CM_PROMASK, &mut |_| {});
    }

    #[test]
    fn coco_worker_promotes_the_key_its_bucket_holds_and_clears_it() {
        let mut worker = CocoWorkerSketch::new(64, 2);
        let key = "flow::coco";
        let mut deltas: Vec<CocoDelta> = Vec::new();

        for _ in 0..(COCO_PROMASK - 1) {
            worker.insert_emit_delta(key, COCO_PROMASK, &mut |d| deltas.push(d));
        }
        assert!(deltas.is_empty(), "should not promote before the threshold");

        worker.insert_emit_delta(key, COCO_PROMASK, &mut |d| deltas.push(d));
        assert_eq!(deltas.len(), 1, "one bucket, one promotion");
        assert_eq!(deltas[0].key, key, "the message carries the bucket's key");
        assert_eq!(deltas[0].value as u32, COCO_PROMASK);
        assert!(
            worker.residual().iter().all(|&c| c == 0),
            "§4.4 clears the bucket it promotes"
        );
    }

    #[test]
    fn coco_promotion_conserves_the_inserted_mass() {
        // Every insert lands in exactly one bucket, so what the parent received
        // plus what the worker still holds is the whole stream.
        let mut worker = CocoWorkerSketch::new(16, 2);
        let mut parent = CocoOctoAggregator::new(16, 2);
        let inserted = 5_000u64;
        for i in 0..inserted {
            worker.insert_emit_delta(&format!("flow::{}", i % 200), COCO_PROMASK, &mut |d| {
                parent.apply(d)
            });
        }
        let held: u64 = worker.residual().iter().map(|&c| c as u64).sum();
        let promoted: u64 = parent.sketch.recorded_flows().map(|(_, v)| v).sum();
        assert_eq!(promoted + held, inserted);

        worker.flush(&mut |d| parent.apply(d));
        assert_eq!(
            parent.sketch.recorded_flows().map(|(_, v)| v).sum::<u64>(),
            inserted,
            "a flush hands over the rest"
        );
    }

    #[test]
    fn elastic_worker_promotes_each_half_on_its_own_terms() {
        let mut worker = ElasticWorkerSketch::new(1, 2, 64);
        let mut heavy = 0usize;
        let mut light = 0usize;
        let mut evicted = 0usize;
        // One heavy bucket and two flows: the resident promotes its votes, the
        // loser goes to the light layer and promotes cell deltas, and a
        // takeover hands the loser over keyed.
        for i in 0..1_000u32 {
            let key = if i % 2 == 0 { "flow::a" } else { "flow::b" };
            worker.insert_emit_delta(key, ELASTIC_PROMASK, &mut |d| match d {
                ElasticDelta::Heavy { .. } => heavy += 1,
                ElasticDelta::Evicted { .. } => evicted += 1,
                ElasticDelta::Light(_) => light += 1,
            });
        }
        assert!(heavy > 0, "the heavy part must promote");
        assert!(light > 0, "the light part must promote");
        assert!(evicted > 0, "a takeover must hand the resident over keyed");
    }

    #[test]
    fn threshold_for_error_follows_equation_four() {
        // tau = eps * L1 / k'
        assert_eq!(threshold_for_error(0.0004, 1_000_000.0, 4), 100);
        assert_eq!(threshold_for_error(0.0001, 1_000_000.0, 1), 100);
        assert_eq!(threshold_for_error(1e-9, 1_000.0, 1), 1, "never below one");
        // An accuracy target loose enough to want a threshold wider than a
        // one-byte worker counter gets the widest one that fits.
        assert_eq!(threshold_for_error(0.001, 1_000_000.0, 1), MAX_PROMASK);
    }

    #[test]
    fn shared_threshold_is_visible_to_every_holder() {
        let shared = OctoThreshold::new(31);
        let clone = shared.clone();
        shared.increase(4);
        assert_eq!(clone.get(), 35);
        clone.decrease(34);
        assert_eq!(shared.get(), 1, "the floor is one");
        clone.set(u32::MAX);
        assert_eq!(
            shared.get(),
            MAX_PROMASK,
            "the ceiling is a signed one-byte worker counter"
        );
    }

    #[test]
    fn hll_threshold_zero_promotes_every_improvement() {
        let mut child = HyperLogLog::<Classic>::default();
        let mut deltas = 0usize;
        child.insert_emit_delta_with_threshold(&DataInput::U64(1), 0, &mut |_| deltas += 1);
        assert_eq!(deltas, 1);
        child.insert_emit_delta_with_threshold(&DataInput::U64(1), 0, &mut |_| deltas += 1);
        assert_eq!(deltas, 1, "a duplicate cannot improve a register");
    }

    #[test]
    fn hll_threshold_holds_back_small_register_gains() {
        let n = 20_000u64;
        let mut promoted_at_zero = 0usize;
        let mut promoted_at_four = 0usize;
        let mut a = HyperLogLog::<Classic>::default();
        let mut b = HyperLogLog::<Classic>::default();
        for i in 0..n {
            a.insert_emit_delta_with_threshold(&DataInput::U64(i), 0, &mut |_| {
                promoted_at_zero += 1
            });
            b.insert_emit_delta_with_threshold(&DataInput::U64(i), 4, &mut |_| {
                promoted_at_four += 1
            });
        }
        assert!(
            promoted_at_four < promoted_at_zero,
            "a larger threshold must send fewer messages: {promoted_at_four} vs {promoted_at_zero}"
        );
    }
}

#[cfg(all(test, feature = "octo-runtime"))]
mod runtime_tests {
    use super::*;

    fn config(num_workers: usize) -> OctoConfig {
        OctoConfig {
            num_workers,
            // CI runners may have fewer cores than the widest configuration here.
            pin_cores: false,
            queue_capacity: 8192,
            ..OctoConfig::default()
        }
    }

    #[test]
    fn run_octo_cm_tracks_a_single_threaded_sketch() {
        let (rows, cols) = (3usize, 4096usize);
        let n = 100_000u64;
        let inputs: Vec<DataInput<'_>> = (0..n).map(|i| DataInput::U64(i % 1024)).collect();

        let mut reference = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(rows, cols);
        for input in &inputs {
            reference.insert(input);
        }

        let result = run_octo(&inputs, &config(4), CmOctoPlan::new(rows, cols), || {
            CmOctoAggregator::new(rows, cols)
        });

        // A counter is shared by whatever flows hash into it, and each worker
        // may hold back up to tau of its own share, so the provable ceiling is
        // k*tau whatever the partition. Hash partitioning bounds the *queried
        // flow's own* held-back count at tau, not the counter's.
        let ceiling = 4 * CM_PROMASK as i32;
        for key_val in 0u64..1024 {
            let key = DataInput::U64(key_val);
            let deficit = reference.estimate(&key) - result.parent.sketch.estimate(&key);
            assert!(
                (0..ceiling).contains(&deficit),
                "key {key_val}: deficit {deficit} outside [0, k*tau)"
            );
        }
    }

    #[test]
    fn run_octo_hll_matches_a_single_threaded_sketch_exactly() {
        let n = 50_000u64;
        let inputs: Vec<DataInput<'_>> = (0..n).map(DataInput::U64).collect();
        let mut reference = HyperLogLog::<Classic>::default();
        for input in &inputs {
            reference.insert(input);
        }

        let result = run_octo(
            &inputs,
            &config(4),
            HllOctoPlan::new(),
            HllOctoAggregator::new,
        );
        assert_eq!(
            result.parent.sketch.registers_as_slice(),
            reference.registers_as_slice()
        );
    }

    #[test]
    fn hash_partitioning_sends_a_key_to_exactly_one_worker() {
        let workers = 4usize;
        let cfg = OctoConfig {
            partition: OctoPartition::HashByKey,
            ..config(workers)
        };
        let mut runtime = OctoRuntime::new(&cfg, WorkerIdPlan, || WorkerLoadAggregator {
            loads: vec![0; workers],
        });
        // One key, many inserts: a hash partition must pile them all on one worker.
        for _ in 0..1_000 {
            runtime.insert(DataInput::U64(4_242));
        }
        let loads = runtime.finish().parent.loads;
        assert_eq!(loads.iter().filter(|&&l| l > 0).count(), 1, "{loads:?}");
        assert_eq!(loads.iter().sum::<u64>(), 1_000);
    }

    #[test]
    fn round_robin_partitioning_still_spreads_inputs_evenly() {
        let workers = 3usize;
        let cfg = OctoConfig {
            partition: OctoPartition::RoundRobin,
            ..config(workers)
        };
        let mut runtime = OctoRuntime::new(&cfg, WorkerIdPlan, || WorkerLoadAggregator {
            loads: vec![0; workers],
        });
        for i in 0..10u64 {
            runtime.insert(DataInput::U64(i));
        }
        assert_eq!(runtime.finish().parent.loads, vec![4, 3, 3]);
    }

    /// Reports the threshold each worker is actually reading.
    struct ThresholdReporter {
        threshold: OctoThreshold,
    }

    impl OctoWorker for ThresholdReporter {
        type Delta = u32;
        type Payload = ();

        fn process<F>(&mut self, _payload: &(), emit: &mut F)
        where
            F: FnMut(Self::Delta),
        {
            emit(self.threshold.get());
        }
    }

    #[derive(Clone)]
    struct ThresholdReporterPlan {
        threshold: OctoThreshold,
    }

    impl OctoPlan for ThresholdReporterPlan {
        type Worker = ThresholdReporter;

        fn worker(&self, _worker_id: usize) -> Self::Worker {
            ThresholdReporter {
                threshold: self.threshold.clone(),
            }
        }

        fn prepare(&self, _input: &DataInput<'_>) {}
    }

    struct ThresholdCollector {
        seen: Vec<u32>,
    }

    impl OctoAggregator for ThresholdCollector {
        type Delta = u32;

        fn apply(&mut self, delta: Self::Delta) {
            self.seen.push(delta);
        }
    }

    #[test]
    fn a_shared_threshold_reaches_every_worker() {
        let workers = 4usize;
        let threshold = OctoThreshold::new(37);
        let cfg = OctoConfig {
            threshold: threshold.clone(),
            ..config(workers)
        };
        let inputs: Vec<DataInput<'_>> = (0..2_000u64).map(DataInput::U64).collect();

        let result = run_octo(
            &inputs,
            &cfg,
            ThresholdReporterPlan {
                threshold: threshold.clone(),
            },
            || ThresholdCollector { seen: Vec::new() },
        );

        assert_eq!(result.parent.seen.len(), inputs.len());
        assert!(
            result.parent.seen.iter().all(|t| *t == 37),
            "every worker must read the threshold the config was built with"
        );
    }

    #[test]
    fn the_threshold_still_decides_how_much_is_promoted() {
        // Post-flush the parent is exact whatever tau was, so the threshold's
        // effect shows up in traffic rather than in the final answer.
        let (rows, cols) = (2usize, 256usize);
        let stream: Vec<u64> = (0..4_000u64).map(|i| i % 8).collect();
        let mut volumes = Vec::new();
        for tau in [4u32, 64] {
            let mut worker = CmWorkerSketch::new(rows, cols);
            let mut promoted = 0usize;
            for key in &stream {
                worker.insert_emit_delta(&DataInput::U64(*key), tau, &mut |_| promoted += 1);
            }
            volumes.push((tau, promoted));
        }
        assert!(
            volumes[0].1 > volumes[1].1,
            "a lower threshold must send more: {volumes:?}"
        );
    }

    /// Drives `ThresholdController` over channels held at a fixed occupancy.
    fn adjust_against_queue_len(
        settings: &OctoAdaptiveThreshold,
        threshold: &OctoThreshold,
        queued: usize,
        rounds: usize,
    ) {
        let (tx, rx) = bounded::<u8>(queued.max(1));
        for _ in 0..queued {
            tx.send(0).expect("queue has room");
        }
        let receivers = vec![Some(rx)];
        let mut controller = ThresholdController::new(settings);
        for _ in 0..rounds {
            for _ in 0..ThresholdController::POLLS_BETWEEN_CLOCK_READS {
                controller.maybe_adjust(&receivers, threshold);
            }
        }
    }

    #[test]
    fn the_controller_raises_tau_when_the_queue_runs_long() {
        let settings = OctoAdaptiveThreshold {
            target_queue_len: 10,
            alpha: 0.25,
            interval: Duration::ZERO,
            min_threshold: 1,
            max_threshold: MAX_PROMASK,
        };
        let threshold = OctoThreshold::new(31);
        adjust_against_queue_len(&settings, &threshold, 500, 8);
        assert!(
            threshold.get() > 31,
            "a queue far above target must push tau up, got {}",
            threshold.get()
        );
    }

    #[test]
    fn the_controller_lowers_tau_when_the_queue_runs_short() {
        let settings = OctoAdaptiveThreshold {
            target_queue_len: 10,
            alpha: 0.25,
            interval: Duration::ZERO,
            min_threshold: 4,
            max_threshold: MAX_PROMASK,
        };
        let threshold = OctoThreshold::new(31);
        adjust_against_queue_len(&settings, &threshold, 0, 64);
        assert_eq!(
            threshold.get(),
            4,
            "an idle queue must walk tau down to its floor"
        );
    }

    #[test]
    fn an_inverted_control_band_is_normalised_rather_than_fatal() {
        let settings = OctoAdaptiveThreshold {
            target_queue_len: 10,
            alpha: 0.25,
            interval: Duration::ZERO,
            min_threshold: 32,
            max_threshold: 8,
        };
        let threshold = OctoThreshold::new(31);
        adjust_against_queue_len(&settings, &threshold, 500, 8);
        assert_eq!(
            threshold.get(),
            32,
            "an inverted band collapses onto its floor instead of panicking"
        );
    }

    #[test]
    fn the_controller_holds_tau_inside_the_dead_band() {
        let settings = OctoAdaptiveThreshold {
            target_queue_len: 10,
            alpha: 0.25,
            interval: Duration::ZERO,
            min_threshold: 1,
            max_threshold: 200,
        };
        let threshold = OctoThreshold::new(31);
        adjust_against_queue_len(&settings, &threshold, 10, 8);
        // Equation 1 has no previous sample on the first evaluation, so it
        // predicts 10 + (10 - 0) = 20 and nudges tau once. Every later
        // evaluation predicts 10, inside the [7.5, 12.5] dead band.
        assert_eq!(
            threshold.get(),
            32,
            "a queue held at target must settle after the initial transient"
        );
    }

    #[test]
    fn the_adaptive_controller_moves_tau_during_a_real_run() {
        let (rows, cols) = (3usize, 1024usize);
        let threshold = OctoThreshold::new(60);
        let settings = OctoAdaptiveThreshold {
            target_queue_len: 8,
            alpha: 0.25,
            interval: Duration::from_micros(1),
            min_threshold: 4,
            max_threshold: 120,
        };
        let cfg = OctoConfig {
            num_workers: 4,
            pin_cores: false,
            queue_capacity: 4096,
            threshold: threshold.clone(),
            partition: OctoPartition::HashByKey,
            adaptive: Some(settings.clone()),
        };

        let inputs: Vec<DataInput<'_>> =
            (0..200_000u64).map(|i| DataInput::U64(i % 4096)).collect();
        let result = run_octo(
            &inputs,
            &cfg,
            CmOctoPlan::with_threshold(rows, cols, threshold.clone()),
            || CmOctoAggregator::new(rows, cols),
        );

        // Whether tau moves, and how far, depends on live queue occupancy, so
        // this asserts only what holds regardless of scheduling. That the rule
        // itself raises, lowers and holds tau is pinned deterministically by
        // the_controller_* tests above.
        let settled = threshold.get();
        assert!(
            (settings.min_threshold..=settings.max_threshold).contains(&settled),
            "tau {settled} left the configured band"
        );
        let mut reference = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(rows, cols);
        for input in &inputs {
            reference.insert(input);
        }
        let key = DataInput::U64(0);
        let deficit = reference.estimate(&key) - result.parent.sketch.estimate(&key);
        assert!(
            deficit >= 0 && deficit <= (cfg.num_workers as i32) * MAX_PROMASK as i32,
            "a controller run must still leave the parent within k*tau: deficit {deficit}"
        );
    }

    #[test]
    fn the_topk_aggregator_rebuilds_the_heavy_hitter_heap_from_worker_keys() {
        let (rows, cols, top_k) = (4usize, 2048usize, 8usize);
        let mut inputs: Vec<DataInput<'_>> = Vec::new();
        // Four unmistakable heavy hitters over a wide light tail.
        for i in 0..4u64 {
            for _ in 0..4_000 {
                inputs.push(DataInput::U64(i));
            }
        }
        for i in 1_000..6_000u64 {
            inputs.push(DataInput::U64(i));
        }

        let result = run_octo(&inputs, &config(4), CmTopKOctoPlan::new(rows, cols), || {
            CmTopKOctoAggregator::new(rows, cols, top_k)
        });

        let heap = result.parent.sketch.heap();
        assert!(!heap.is_empty(), "the aggregator must have built a heap");
        for hot in 0..4u64 {
            assert!(
                result
                    .parent
                    .sketch
                    .heap()
                    .find(&DataInput::U64(hot))
                    .is_some(),
                "heavy hitter {hot} missing from the aggregator heap"
            );
        }
    }

    #[test]
    fn a_parent_wider_than_its_workers_estimates_zero() {
        // Worker and parent geometry are separate arguments and a mismatch is
        // silent: every delta names a row the parent has, the rows past the
        // worker's never receive one, and min-over-rows is zero for every key.
        // `CmOctoPlan::aggregator` is the way not to hit this; this pins the
        // hazard so a future check has to update the test rather than pass
        // quietly.
        let inputs: Vec<DataInput<'_>> = (0..40_000u64).map(|i| DataInput::U64(i % 64)).collect();
        let plan = CmOctoPlan::new(3, 1024);

        let mismatched = run_octo(&inputs, &config(4), plan.clone(), || {
            CmOctoAggregator::new(5, 1024)
        });
        assert_eq!(
            mismatched.parent.sketch.estimate(&DataInput::U64(0)),
            0,
            "a wider parent has rows no worker ever writes"
        );

        let matched = run_octo(&inputs, &config(4), plan.clone(), || plan.aggregator());
        assert!(
            matched.parent.sketch.estimate(&DataInput::U64(0)) >= 600,
            "building the parent from the plan keeps the geometries together"
        );
    }

    #[test]
    fn a_key_below_the_threshold_never_reaches_the_topk_aggregator() {
        // The keyed workers keep no key storage and so cannot flush, and the
        // aggregator only learns a key exists from a delta. A key promotes the
        // first time an increment it caused takes some row to tau on its
        // worker - which is exactly its tau-th occurrence only while its cells
        // are collision-free, so the geometry is checked here rather than
        // assumed.
        let (rows, cols, top_k) = (3usize, 1024usize, 32usize);
        let tau = CM_PROMASK;
        let keys: Vec<u64> = (0..20).collect();

        let mut occupied: std::collections::HashSet<(usize, usize, usize)> =
            std::collections::HashSet::new();
        for key in &keys {
            let input = DataInput::U64(*key);
            let worker = (hash64_seeded(0, &input) % 4) as usize;
            for (row, hashed) in CmWorkerSketch::hashes(rows, &input).iter().enumerate() {
                let col = ((hashed & LOWER_32_MASK) as usize) % cols;
                assert!(
                    occupied.insert((worker, row, col)),
                    "key {key} shares a cell with another; the tau-th-occurrence \
                     boundary only holds for collision-free cells"
                );
            }
        }

        let run = |occurrences: u32| {
            let mut inputs: Vec<DataInput<'_>> = Vec::new();
            for key in &keys {
                for _ in 0..occurrences {
                    inputs.push(DataInput::U64(*key));
                }
            }
            let plan = CmTopKOctoPlan::new(rows, cols);
            run_octo(&inputs, &config(4), plan.clone(), || plan.aggregator(top_k))
                .parent
                .sketch
        };

        let below = run(tau - 1);
        assert_eq!(
            below.heap().len(),
            0,
            "a key that never reaches tau is absent from the heap entirely"
        );
        assert_eq!(
            below.estimate(&DataInput::U64(0)),
            0,
            "and its counter never left the worker"
        );

        let at = run(tau);
        assert_eq!(
            at.heap().len(),
            keys.len(),
            "one more occurrence per key makes every one of them a candidate"
        );
        assert!(at.estimate(&DataInput::U64(0)) >= tau as i32);
    }

    #[test]
    fn the_count_topk_aggregator_also_tracks_heavy_hitters() {
        let (rows, cols, top_k) = (5usize, 2048usize, 8usize);
        let mut inputs: Vec<DataInput<'_>> = Vec::new();
        for i in 0..3u64 {
            for _ in 0..5_000 {
                inputs.push(DataInput::U64(i));
            }
        }
        for i in 500..3_000u64 {
            inputs.push(DataInput::U64(i));
        }

        let result = run_octo(
            &inputs,
            &config(4),
            CountTopKOctoPlan::new(rows, cols),
            || CountTopKOctoAggregator::new(rows, cols, top_k),
        );
        for hot in 0..3u64 {
            assert!(
                result
                    .parent
                    .sketch
                    .heap()
                    .find(&DataInput::U64(hot))
                    .is_some(),
                "heavy hitter {hot} missing from the aggregator heap"
            );
        }
    }

    #[test]
    fn run_octo_ddsketch_tracks_quantiles_of_the_stream() {
        let alpha = 0.01;
        let inputs: Vec<DataInput<'_>> = (1..=200_000u64)
            .map(|i| DataInput::F64(1.0 + (i as f64 * 7.0) % 999.0))
            .collect();
        let mut exact: Vec<f64> = inputs
            .iter()
            .map(|i| match i {
                DataInput::F64(v) => *v,
                _ => unreachable!(),
            })
            .collect();
        exact.sort_by(f64::total_cmp);

        let result = run_octo(&inputs, &config(4), DdOctoPlan::new(alpha), || {
            DdOctoAggregator::new(alpha)
        });

        // Four workers each hold back their own partial buckets, so the rank
        // slack is what the whole fleet has yet to promote.
        let held_back = inputs.len() as u64 - result.parent.sketch.get_count();
        let rank_slack = held_back as f64 / inputs.len() as f64;
        assert!(rank_slack < 0.05, "held back {rank_slack:.3} of the stream");

        for q in [0.1, 0.5, 0.9] {
            let got = result
                .parent
                .sketch
                .get_value_at_quantile(q)
                .expect("parent received samples");
            let lo = exact[(((q - rank_slack).max(0.0)) * (exact.len() - 1) as f64) as usize]
                * (1.0 - alpha);
            let hi = exact[(((q + rank_slack).min(1.0)) * (exact.len() - 1) as f64) as usize]
                * (1.0 + alpha);
            assert!(
                (lo..=hi).contains(&got),
                "q={q}: {got} outside [{lo}, {hi}] at rank slack {rank_slack:.3}"
            );
        }
    }

    #[test]
    fn run_octo_univmon_reaches_the_deepest_layer() {
        use crate::L2HH;
        // The structural half of `a_flat_threshold_starves_the_deep_univmon_layers`,
        // checked against the shipped pipeline rather than the replica that
        // test needs in order to run the flat counterfactual at all.
        let (heap, rows, cols, layers) = (64usize, 5usize, 1_024usize, 12usize);
        let inputs: Vec<DataInput<'_>> =
            (0..60_000u64).map(|i| DataInput::U64(i % 4_096)).collect();
        let plan = UnivMonOctoPlan::new(rows, cols, layers);
        let result = run_octo(&inputs, &config(1), plan.clone(), || plan.aggregator(heap));

        let deepest = layers - 1;
        let L2HH::COUNT(inner) = &result.parent.sketch.l2_sketch_layers[deepest];
        let nonzero = (0..inner.rows())
            .flat_map(|r| (0..inner.cols()).map(move |c| (r, c)))
            .filter(|(r, c)| inner.as_storage().query_one_counter(*r, *c) != 0)
            .count();
        assert!(
            nonzero > 0,
            "the scaled per-layer threshold must let the deepest layer through"
        );
    }

    #[test]
    fn run_octo_univmon_tracks_heavy_hitters_and_total_weight() {
        let (heap, rows, cols, layers) = (32usize, 3usize, 256usize, 6usize);
        let tau = 8u32;
        let threshold = OctoThreshold::new(tau);

        let mut inputs: Vec<DataInput<'_>> = Vec::new();
        for hot in 0..5u64 {
            for _ in 0..4_000 {
                inputs.push(DataInput::U64(hot));
            }
        }
        for cold in 10_000..15_000u64 {
            inputs.push(DataInput::U64(cold));
        }

        let cfg = OctoConfig {
            threshold: threshold.clone(),
            ..config(4)
        };
        let result = run_octo(
            &inputs,
            &cfg,
            UnivMonOctoPlan::with_threshold(rows, cols, layers, threshold.clone()),
            || UnivMonOctoAggregator::new(heap, rows, cols, layers, tau),
        );
        let univmon = &result.parent.sketch;

        // The fleet reports its running weight on messages it was already
        // sending, so the aggregator's total trails by whatever arrived after
        // each worker's last promotion - never overshoots.
        assert!(
            univmon.bucket_size <= inputs.len(),
            "reported weight {} exceeds the stream",
            univmon.bucket_size
        );
        assert!(
            univmon.bucket_size as f64 > inputs.len() as f64 * 0.9,
            "reported weight {} trails the stream of {} by too much",
            univmon.bucket_size,
            inputs.len()
        );

        for hot in 0..5u64 {
            assert!(
                univmon.hh_layers[0].find(&DataInput::U64(hot)).is_some(),
                "heavy hitter {hot} missing from the aggregator's layer-0 heap"
            );
        }
        assert!(
            univmon.candidates_complete().iter().all(|c| !c),
            "a promoted-only aggregator must not claim complete candidate sets"
        );
    }

    #[test]
    fn octo_runtime_streaming_matches_the_batch_helper() {
        let (rows, cols) = (3usize, 4096usize);
        let inputs: Vec<DataInput<'_>> = (0..30_000u64).map(|i| DataInput::U64(i % 1024)).collect();
        let cfg = config(4);

        let batch = run_octo(&inputs, &cfg, CmOctoPlan::new(rows, cols), || {
            CmOctoAggregator::new(rows, cols)
        });

        let mut runtime = OctoRuntime::new(&cfg, CmOctoPlan::new(rows, cols), move || {
            CmOctoAggregator::new(rows, cols)
        });
        for input in &inputs {
            runtime.insert(input.clone());
        }
        let streamed = runtime.finish();

        for key_val in 0u64..128 {
            let key = DataInput::U64(key_val);
            assert_eq!(
                batch.parent.sketch.estimate(&key),
                streamed.parent.sketch.estimate(&key),
                "key {key_val}"
            );
        }
    }

    #[test]
    fn finishing_a_ddsketch_run_answers_against_the_whole_stream() {
        // Without the flush on End, a stream of near-distinct values leaves no
        // bucket at the threshold, the parent holds nothing, and every quantile
        // query returns None.
        let alpha = 0.01;
        let inputs: Vec<DataInput<'_>> = (1..=20_000u64)
            .map(|i| DataInput::F64(1.0 + i as f64 * 0.37))
            .collect();
        let mut exact: Vec<f64> = inputs
            .iter()
            .map(|i| match i {
                DataInput::F64(v) => *v,
                _ => unreachable!(),
            })
            .collect();
        exact.sort_by(f64::total_cmp);

        let result = run_octo(&inputs, &config(4), DdOctoPlan::new(alpha), || {
            DdOctoAggregator::new(alpha)
        });
        let parent = &result.parent.sketch;

        assert_eq!(
            parent.get_count(),
            inputs.len() as u64,
            "every sample must have reached the parent"
        );
        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let truth = exact[((exact.len() - 1) as f64 * q) as usize];
            let got = parent
                .get_value_at_quantile(q)
                .expect("a flushed parent answers every quantile");
            let relative = (got - truth).abs() / truth;
            assert!(
                relative <= 2.0 * alpha,
                "q={q}: got {got}, truth {truth}, relative {relative:.4}"
            );
        }
        let (min, max) = (parent.min().unwrap(), parent.max().unwrap());
        assert!(min <= exact[0] * (1.0 + alpha) && max >= exact[exact.len() - 1] * (1.0 - alpha));
    }

    #[test]
    fn a_mid_stream_flush_makes_the_live_parent_answerable() {
        let alpha = 0.01;
        let values: Vec<f64> = (1..=20_000u64).map(|i| 1.0 + i as f64 * 0.37).collect();
        let mut runtime = OctoRuntime::new(&config(4), DdOctoPlan::new(alpha), || {
            DdOctoAggregator::new(alpha)
        });
        let reader = runtime.read_handle();

        let half = values.len() / 2;
        for value in &values[..half] {
            runtime.insert(DataInput::F64(*value));
        }

        runtime.flush();
        let counted = reader.with_parent(|p| p.sketch.get_count());
        assert_eq!(
            counted, half as u64,
            "a flush must land every input accepted so far"
        );
        let median = reader.with_parent(|p| p.sketch.get_value_at_quantile(0.5));
        assert!(median.is_some(), "a flushed parent answers mid-stream");

        // The runtime is not sealed by a flush.
        for value in &values[half..] {
            runtime.insert(DataInput::F64(*value));
        }
        assert_eq!(
            runtime.finish().parent.sketch.get_count(),
            values.len() as u64
        );
    }

    struct PanicOnFlushWorker;

    impl OctoWorker for PanicOnFlushWorker {
        type Delta = u64;
        type Payload = ();

        fn process<F>(&mut self, _payload: &(), emit: &mut F)
        where
            F: FnMut(Self::Delta),
        {
            emit(1);
        }

        fn flush<F>(&mut self, _emit: &mut F)
        where
            F: FnMut(Self::Delta),
        {
            panic!("worker died during flush");
        }
    }

    struct PanicOnFlushPlan;

    impl OctoPlan for PanicOnFlushPlan {
        type Worker = PanicOnFlushWorker;

        fn worker(&self, _worker_id: usize) -> Self::Worker {
            PanicOnFlushWorker
        }

        fn prepare(&self, _input: &DataInput<'_>) {}
    }

    #[test]
    #[should_panic(expected = "worker dropped during flush")]
    fn a_worker_that_dies_mid_flush_fails_the_flush_rather_than_hanging_it() {
        // The acknowledgement channel has to observe every sender going away,
        // which means the caller's own copy must be dropped first. Otherwise
        // this waits forever instead of reporting.
        let mut runtime = OctoRuntime::new(&config(2), PanicOnFlushPlan, || CountingAggregator {
            total: 0,
        });
        runtime.insert(DataInput::U64(1));
        runtime.flush();
    }

    #[test]
    fn flushing_twice_is_harmless() {
        let alpha = 0.01;
        let mut runtime = OctoRuntime::new(&config(2), DdOctoPlan::new(alpha), || {
            DdOctoAggregator::new(alpha)
        });
        for i in 1..=5_000u64 {
            runtime.insert(DataInput::F64(i as f64));
        }
        runtime.flush();
        runtime.flush();
        assert_eq!(runtime.finish().parent.sketch.get_count(), 5_000);
    }

    #[test]
    fn a_flushed_count_min_parent_matches_a_single_pass_exactly() {
        // Flush is not DDSketch-only: it makes every unkeyed sketch exact at
        // the point of query, where before the parent was low by under tau.
        let (rows, cols) = (3usize, 1024usize);
        let inputs: Vec<DataInput<'_>> = (0..40_000u64).map(|i| DataInput::U64(i % 512)).collect();
        let got = run_octo(&inputs, &config(4), CmOctoPlan::new(rows, cols), || {
            CmOctoAggregator::new(rows, cols)
        });
        let mut reference = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(rows, cols);
        for input in &inputs {
            reference.insert(input);
        }
        for row in 0..rows {
            for col in 0..cols {
                assert_eq!(
                    got.parent.sketch.as_storage().query_one_counter(row, col),
                    reference.as_storage().query_one_counter(row, col),
                    "cell ({row},{col})"
                );
            }
        }
    }

    #[test]
    fn octo_runtime_close_is_idempotent() {
        let runtime = OctoRuntime::new(&config(2), HllOctoPlan::new(), HllOctoAggregator::new);
        runtime.close();
        runtime.close();
        assert_eq!(runtime.finish().parent.sketch.estimate(), 0);
    }

    #[test]
    #[should_panic(expected = "cannot insert after runtime has been closed")]
    fn octo_runtime_insert_after_close_panics() {
        let mut runtime = OctoRuntime::new(&config(2), HllOctoPlan::new(), HllOctoAggregator::new);
        runtime.close();
        runtime.insert(DataInput::U64(1));
    }

    #[test]
    fn octo_runtime_empty_stream_finishes() {
        let runtime = OctoRuntime::new(&config(4), HllOctoPlan::new(), HllOctoAggregator::new);
        assert_eq!(runtime.finish().parent.sketch.estimate(), 0);
    }

    #[test]
    fn octo_runtime_live_read_handle_tracks_the_aggregator() {
        let n = 64u64;
        let mut runtime =
            OctoRuntime::new(&config(2), CountingPlan, || CountingAggregator { total: 0 });
        let reader = runtime.read_handle();
        assert_eq!(reader.with_parent(|p| p.total), 0, "nothing inserted yet");

        for i in 0..n {
            runtime.insert(DataInput::U64(i));
        }

        // Poll rather than sleep-and-hope: the point is that the handle reaches
        // the live total without the runtime being finished. A handle wired to
        // a constant would never get here.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut last = 0u64;
        loop {
            let observed = reader.with_parent(|p| p.total);
            assert!(
                observed >= last,
                "live total went backwards: {last} -> {observed}"
            );
            assert!(observed <= n, "live total {observed} exceeded the stream");
            last = observed;
            if observed == n {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "read handle stalled at {observed}/{n}"
            );
            std::hint::spin_loop();
        }

        assert_eq!(runtime.finish().parent.total, n);
    }

    #[test]
    fn octo_runtime_close_preserves_queued_items() {
        let n = 257u64;
        let mut runtime =
            OctoRuntime::new(&config(4), CountingPlan, || CountingAggregator { total: 0 });
        for i in 0..n {
            runtime.insert(DataInput::U64(i + 42));
        }
        runtime.close();
        assert_eq!(runtime.finish().parent.total, n);
    }

    struct CountingWorker;

    impl OctoWorker for CountingWorker {
        type Delta = u64;
        type Payload = ();

        fn process<F>(&mut self, _payload: &(), emit: &mut F)
        where
            F: FnMut(Self::Delta),
        {
            emit(1);
        }
    }

    struct CountingPlan;

    impl OctoPlan for CountingPlan {
        type Worker = CountingWorker;

        fn worker(&self, _worker_id: usize) -> Self::Worker {
            CountingWorker
        }

        fn prepare(&self, _input: &DataInput<'_>) {}
    }

    struct CountingAggregator {
        total: u64,
    }

    impl OctoAggregator for CountingAggregator {
        type Delta = u64;

        fn apply(&mut self, delta: Self::Delta) {
            self.total += delta;
        }
    }

    struct WorkerIdEmitter {
        worker_id: usize,
    }

    impl OctoWorker for WorkerIdEmitter {
        type Delta = usize;
        type Payload = ();

        fn process<F>(&mut self, _payload: &(), emit: &mut F)
        where
            F: FnMut(Self::Delta),
        {
            emit(self.worker_id);
        }
    }

    struct WorkerIdPlan;

    impl OctoPlan for WorkerIdPlan {
        type Worker = WorkerIdEmitter;

        fn worker(&self, worker_id: usize) -> Self::Worker {
            WorkerIdEmitter { worker_id }
        }

        fn prepare(&self, _input: &DataInput<'_>) {}
    }

    struct WorkerLoadAggregator {
        loads: Vec<u64>,
    }

    impl OctoAggregator for WorkerLoadAggregator {
        type Delta = usize;

        fn apply(&mut self, delta: Self::Delta) {
            self.loads[delta] += 1;
        }
    }
}

#[cfg(test)]
mod univmon_tests {
    use super::*;
    use crate::L2HH;

    const HEAP: usize = 32;
    const ROWS: usize = 3;
    const COLS: usize = 256;
    const LAYERS: usize = 6;

    fn layer_counters(sketch: &UnivMon, layer: usize) -> Vec<i64> {
        let L2HH::COUNT(inner) = &sketch.l2_sketch_layers[layer];
        (0..inner.rows())
            .flat_map(|r| (0..inner.cols()).map(move |c| (r, c)))
            .map(|(r, c)| inner.as_storage().query_one_counter(r, c))
            .collect()
    }

    fn heap_contents(sketch: &UnivMon, layer: usize) -> Vec<(HeapItem, i64)> {
        let mut items: Vec<(HeapItem, i64)> = sketch.hh_layers[layer]
            .heap()
            .iter()
            .map(|i| (i.key.clone(), i.count))
            .collect();
        items.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)))
        });
        items
    }

    fn stream(n: u64) -> Vec<DataInput<'static>> {
        // A skewed stream: a few keys carry most of the weight.
        (0..n)
            .map(|i| {
                let key = match i % 10 {
                    0..=4 => i % 5,
                    5..=7 => 100 + (i % 40),
                    _ => 1_000 + (i % 900),
                };
                DataInput::U64(key)
            })
            .collect()
    }

    #[test]
    fn a_worker_picks_the_same_layers_as_a_single_threaded_insert() {
        let reference = UnivMon::init_univmon(HEAP, ROWS, COLS, LAYERS);
        for key in 0..5_000u64 {
            let input = DataInput::U64(key);
            let from_hash =
                bottom_layer_for_hash(hash64_seeded(BOTTOM_LAYER_FINDER, &input), LAYERS);
            assert_eq!(
                from_hash,
                reference.bottom_layer_for(&input),
                "key {key} routed to a different layer set"
            );
            assert!(from_hash < LAYERS);
        }
    }

    #[test]
    fn one_worker_at_threshold_one_reproduces_a_single_threaded_univmon() {
        // A threshold of 1 promotes every counter touch, and a single worker
        // applies them in stream order, so the aggregator must land in exactly
        // the state a single-threaded insert loop would.
        let inputs = stream(20_000);

        let mut reference = UnivMon::init_univmon(HEAP, ROWS, COLS, LAYERS);
        for input in &inputs {
            reference.insert(input, 1);
        }

        let preparer = UnivMonOctoPlan::with_threshold(ROWS, COLS, LAYERS, OctoThreshold::new(1));
        let mut worker =
            UnivMonOctoWorker::with_threshold(0, ROWS, COLS, LAYERS, OctoThreshold::new(1));
        let mut aggregator = UnivMonOctoAggregator::new(HEAP, ROWS, COLS, LAYERS, 1);
        for input in &inputs {
            let mut promoted = Vec::new();
            worker.process(&preparer.prepare(input), &mut |d| promoted.push(d));
            for delta in promoted {
                aggregator.apply(delta);
            }
        }
        let octo = &aggregator.sketch;

        assert_eq!(octo.bucket_size, reference.bucket_size, "total weight");
        assert_eq!(
            octo.candidates_complete(),
            reference.candidates_complete(),
            "candidate completeness"
        );
        for layer in 0..LAYERS {
            assert_eq!(
                layer_counters(octo, layer),
                layer_counters(&reference, layer),
                "layer {layer} counters"
            );
            assert_eq!(
                octo.l2_sketch_layers[layer].get_l2(),
                reference.l2_sketch_layers[layer].get_l2(),
                "layer {layer} L2"
            );
            assert_eq!(
                heap_contents(octo, layer),
                heap_contents(&reference, layer),
                "layer {layer} heavy-hitter heap"
            );
        }
        assert_eq!(
            octo.calc_g_sum(|x| x, false),
            reference.calc_g_sum(|x| x, false),
            "g-sum query"
        );
    }

    #[test]
    fn promotion_conserves_every_layer_counter() {
        let inputs = stream(30_000);
        let threshold = 31u32;

        let mut reference = UnivMon::init_univmon(HEAP, ROWS, COLS, LAYERS);
        for input in &inputs {
            reference.insert(input, 1);
        }

        let preparer =
            UnivMonOctoPlan::with_threshold(ROWS, COLS, LAYERS, OctoThreshold::new(threshold));
        let mut worker =
            UnivMonOctoWorker::with_threshold(0, ROWS, COLS, LAYERS, OctoThreshold::new(threshold));
        let mut aggregator = UnivMonOctoAggregator::new(HEAP, ROWS, COLS, LAYERS, threshold);
        for input in &inputs {
            let mut promoted = Vec::new();
            worker.process(&preparer.prepare(input), &mut |d| promoted.push(d));
            for delta in promoted {
                aggregator.apply(delta);
            }
        }

        for layer in 0..LAYERS {
            let promoted = layer_counters(&aggregator.sketch, layer);
            let exact = layer_counters(&reference, layer);
            let residual = worker.layers[layer].residual();
            for (cell, ((&p, &e), &r)) in
                promoted.iter().zip(exact.iter()).zip(residual).enumerate()
            {
                assert_eq!(
                    p + r as i64,
                    e,
                    "layer {layer} cell {cell}: promoted {p} plus residual {r} != {e}"
                );
                assert!(
                    r.unsigned_abs() < threshold as u8,
                    "layer {layer} cell {cell}: residual {r} should have been promoted"
                );
            }
        }
    }

    #[test]
    fn completeness_is_withdrawn_exactly_where_the_layer_thresholds() {
        let base = 8u32;
        let inputs = stream(5_000);
        let preparer =
            UnivMonOctoPlan::with_threshold(ROWS, COLS, LAYERS, OctoThreshold::new(base));
        let mut worker =
            UnivMonOctoWorker::with_threshold(0, ROWS, COLS, LAYERS, OctoThreshold::new(base));
        let mut aggregator = UnivMonOctoAggregator::new(HEAP, ROWS, COLS, LAYERS, base);
        for input in &inputs {
            let mut promoted = Vec::new();
            worker.process(&preparer.prepare(input), &mut |d| promoted.push(d));
            for delta in promoted {
                aggregator.apply(delta);
            }
        }

        let (mut thresholding, mut exact) = (0usize, 0usize);
        for (layer, complete) in aggregator.sketch.candidates_complete().iter().enumerate() {
            if univmon_layer_threshold(base, layer) > 1 {
                // The aggregator provably never saw the keys that stayed under
                // this layer's threshold.
                thresholding += 1;
                assert!(
                    !complete,
                    "layer {layer} thresholds at {} yet claims completeness",
                    univmon_layer_threshold(base, layer)
                );
            } else {
                // A layer at threshold 1 promotes every insert, so its verdict
                // must come from the heap rather than from the withdrawal - and
                // this stream is small enough that the heap holds everything.
                exact += 1;
                assert!(
                    complete,
                    "layer {layer} promotes every insert yet lost completeness"
                );
            }
        }
        // Both halves of "exactly where" have to be exercised, or the test
        // passes on an aggregator that simply withdraws everything.
        assert!(thresholding > 0 && exact > 0, "{thresholding} / {exact}");
    }

    fn plan(threshold: &OctoThreshold) -> UnivMonOctoPlan {
        UnivMonOctoPlan::with_threshold(ROWS, COLS, LAYERS, threshold.clone())
    }

    /// Drives one worker/aggregator pair over `inputs`.
    fn drive(
        plan: &UnivMonOctoPlan,
        worker: &mut UnivMonOctoWorker,
        aggregator: &mut UnivMonOctoAggregator,
        inputs: &[DataInput<'_>],
    ) {
        for input in inputs {
            let mut promoted = Vec::new();
            worker.process(&plan.prepare(input), &mut |d| promoted.push(d));
            for delta in promoted {
                aggregator.apply(delta);
            }
        }
    }

    #[test]
    fn raising_tau_mid_run_withdraws_completeness_that_was_already_granted() {
        // With the adaptive controller live, a layer that started at tau = 1
        // can be pushed above it. A fidelity verdict frozen at construction
        // would leave that layer claiming a complete candidate set, sending
        // heavy_threshold down its permissive branch.
        let threshold = OctoThreshold::new(1);
        let mut worker =
            UnivMonOctoWorker::with_threshold(0, ROWS, COLS, LAYERS, threshold.clone());
        let mut aggregator =
            UnivMonOctoAggregator::with_threshold(HEAP, ROWS, COLS, LAYERS, threshold.clone());
        let pl = plan(&threshold);
        let inputs = stream(4_000);
        drive(&pl, &mut worker, &mut aggregator, &inputs);
        assert!(
            aggregator.sketch.candidates_complete().iter().any(|c| *c),
            "at tau = 1 the aggregator sees every insert, so some layer should still be complete"
        );

        threshold.set(64);
        drive(&pl, &mut worker, &mut aggregator, &inputs);
        for (layer, complete) in aggregator.sketch.candidates_complete().iter().enumerate() {
            if univmon_layer_threshold(64, layer) > 1 {
                assert!(
                    !complete,
                    "layer {layer} kept a completeness verdict from before tau rose"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "worker ids must be distinct")]
    fn workers_sharing_an_id_are_refused_rather_than_summed_wrongly() {
        // Two workers in one slot would make the fleet total a maximum instead
        // of a sum, and calc_l1 would report about N/k.
        let threshold = OctoThreshold::new(1);
        let mut first = UnivMonOctoWorker::with_threshold(0, ROWS, COLS, LAYERS, threshold.clone());
        let mut second =
            UnivMonOctoWorker::with_threshold(0, ROWS, COLS, LAYERS, threshold.clone());
        let pl = plan(&threshold);
        let mut aggregator =
            UnivMonOctoAggregator::with_threshold(HEAP, ROWS, COLS, LAYERS, threshold);
        let inputs = stream(200);
        drive(&pl, &mut first, &mut aggregator, &inputs);
        drive(&pl, &mut second, &mut aggregator, &inputs);
    }

    #[test]
    fn the_layer_threshold_tracks_how_little_traffic_a_deep_layer_sees() {
        // Layer L receives roughly n / 2^L of the stream, so its threshold
        // halves with it and bottoms out at 1.
        assert_eq!(univmon_layer_threshold(32, 0), 32);
        assert_eq!(univmon_layer_threshold(32, 1), 16);
        assert_eq!(univmon_layer_threshold(32, 5), 1);
        assert_eq!(univmon_layer_threshold(32, 40), 1, "no shift overflow");
        assert_eq!(univmon_layer_threshold(1, 0), 1);
    }

    #[test]
    fn the_worker_holds_one_byte_per_counter_and_no_keys() {
        let mut worker = UnivMonOctoWorker::new(0, ROWS, COLS, LAYERS);
        // One byte per cell against the i64 counters a UnivMon layer holds.
        assert_eq!(worker.counter_bytes(), ROWS * COLS * LAYERS);
        assert_eq!(
            worker.counter_bytes(),
            ROWS * COLS * LAYERS * std::mem::size_of::<i64>() / 8
        );

        // And no key storage: a worker's footprint must not grow with the
        // number of distinct keys it has seen, however long the strings are.
        let before = worker.counter_bytes();
        let preparer = UnivMonOctoPlan::new(ROWS, COLS, LAYERS);
        let keys: Vec<String> = (0..5_000).map(|i| format!("flow-{i:040}")).collect();
        for key in &keys {
            worker.process(&preparer.prepare(&DataInput::Str(key)), &mut |_| {});
        }
        assert_eq!(
            worker.counter_bytes(),
            before,
            "a worker that kept keys would grow here"
        );
    }
}
