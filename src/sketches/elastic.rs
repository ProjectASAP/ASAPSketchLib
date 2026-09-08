//! Elastic Sketch.
//!
//! Reference:
//! - Yang et al., "Elastic Sketch: Adaptive and Fast Network-wide Measurements,"
//!   SIGCOMM 2018.
//!   <https://dl.acm.org/doi/10.1145/3230543.3230544>

use crate::{CANONICAL_HASH_SEED, DataInput, DefaultXxHasher, SketchHasher};

use super::{CountMin, RegularPath};
use crate::Vector2D;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

mod wire;

/// Eviction threshold `lambda` from the paper. A resident flow is replaced once
/// its negative votes reach `LAMBDA` times its positive votes, which is the
/// paper's `vote-/vote+ >= lambda`; BlockLiu/ElasticSketchCode swaps one packet
/// later, on `>`.
pub const LAMBDA: i32 = 8;

/// Rows in the light layer built by [`Elastic::new`] and
/// [`Elastic::init_with_length`].
pub const DEFAULT_LIGHT_ROWS: usize = 3;

/// Columns per light-layer row built by [`Elastic::new`] and
/// [`Elastic::init_with_length`].
pub const DEFAULT_LIGHT_COLS: usize = 4096;

/// One slot of the heavy part: the resident flow, its vote pair, and the flag
/// marking that part of its size lives in the light layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HeavyBucket {
    pub flow_id: String,
    pub vote_pos: i32,
    pub vote_neg: i32,
    pub eviction: bool,
}

/// Heavy/light frequency estimator: a heavy hash table over flow ids backed by
/// a Count-Min light layer that absorbs evicted and unelected flows.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "")]
pub struct Elastic<H: SketchHasher = DefaultXxHasher> {
    pub heavy: Vec<HeavyBucket>,
    pub light: CountMin<Vector2D<i32>, RegularPath, H>,
    pub bktlen: i32,
    /// Set by [`Elastic::expand_heavy`] while copies of pre-expansion
    /// residents may still sit in the half they no longer hash to.
    #[serde(default)]
    stale_copies: bool,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

impl Default for HeavyBucket {
    fn default() -> Self {
        Self::new()
    }
}

impl HeavyBucket {
    pub fn new() -> Self {
        HeavyBucket {
            flow_id: String::new(),
            vote_pos: 0,
            vote_neg: 0,
            eviction: false,
        }
    }

    /// A bucket holds no flow exactly while it has no positive vote.
    #[inline]
    pub fn is_vacant(&self) -> bool {
        self.vote_pos == 0
    }

    /// Seats `id` in a vacant bucket. The `eviction` flag is carried over so a
    /// bucket vacated by [`Elastic::merge`] still reports its light-layer mass.
    pub fn occupy(&mut self, id: String) {
        self.occupy_many(id, 1);
    }

    /// Seats `id` in a vacant bucket with `count` positive votes already to its
    /// name, for [`Elastic::insert_many`].
    pub fn occupy_many(&mut self, id: String, count: i32) {
        self.flow_id = id;
        self.vote_pos = count;
        self.vote_neg = 0;
    }

    /// Replaces the resident flow with `id`, per the paper's takeover rule.
    pub fn evict(&mut self, id: String) -> String {
        self.evict_many(id, 1)
    }

    /// Replaces the resident flow with `id`, which arrives carrying `count`
    /// votes rather than one.
    pub fn evict_many(&mut self, id: String, count: i32) -> String {
        let evicted = std::mem::replace(&mut self.flow_id, id);
        self.vote_pos = count;
        self.vote_neg = count;
        self.eviction = true;
        evicted
    }
}

impl Default for Elastic {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: SketchHasher> Elastic<H> {
    pub fn new() -> Self {
        Elastic::init_with_length(8)
    }

    /// Heavy table of `l` buckets over the default light layer.
    pub fn init_with_length(l: i32) -> Self {
        Elastic::init_with_dimensions(l, DEFAULT_LIGHT_ROWS, DEFAULT_LIGHT_COLS)
    }

    /// Heavy table of `bucket_count` buckets over a `light_rows` by
    /// `light_cols` Count-Min. Bucket count sets the elephant collision rate;
    /// the light dimensions set the error carried by every non-resident flow.
    pub fn init_with_dimensions(bucket_count: i32, light_rows: usize, light_cols: usize) -> Self {
        assert!(
            bucket_count > 0,
            "Elastic needs at least one heavy bucket, got {bucket_count}"
        );
        assert!(
            light_rows > 0 && light_cols > 0,
            "Elastic needs a non-empty light layer, got {light_rows}x{light_cols}"
        );

        let heavy = (0..bucket_count).map(|_| HeavyBucket::new()).collect();
        let light =
            CountMin::<Vector2D<i32>, RegularPath, H>::with_dimensions(light_rows, light_cols);
        Elastic {
            heavy,
            light,
            bktlen: bucket_count,
            stale_copies: false,
            _hasher: PhantomData,
        }
    }

    /// Records one occurrence of `id`.
    ///
    /// A vacant bucket seats the flow; a matching bucket takes a positive vote.
    /// Otherwise the bucket takes a negative vote and either the arriving flow
    /// goes to the light layer, or, once `vote_neg >= LAMBDA * vote_pos`, the
    /// resident flow is evicted into the light layer with its full positive
    /// vote and `id` takes the bucket.
    pub fn insert(&mut self, id: String) {
        self.insert_many(id, 1);
    }

    /// Records `count` occurrences of `id` in one step.
    ///
    /// The weighted form of [`Self::insert`]: a matching bucket takes `count`
    /// positive votes, a non-matching one takes `count` negative votes, and a
    /// takeover seats the arrival with `count` of each. `count` of 1 is
    /// [`Self::insert`] exactly.
    ///
    /// This is the insertion an OctoSketch aggregator applies to a heavy-part
    /// `<key, votes>` message; it absorbs the whole promoted counter in one
    /// pass rather than replaying it `count` times.
    pub fn insert_many(&mut self, id: String, count: i32) {
        if count <= 0 {
            return;
        }
        let idx = self.bucket_index(&id);
        if self.stale_at(idx) {
            self.seat_over_stale_copy(idx, id, count);
            return;
        }
        let bucket = &mut self.heavy[idx];

        if bucket.is_vacant() {
            bucket.occupy_many(id, count);
            return;
        }
        if bucket.flow_id == id {
            bucket.vote_pos += count;
            return;
        }

        bucket.vote_neg += count;
        if bucket.vote_neg < LAMBDA * bucket.vote_pos {
            self.light.insert_many(&DataInput::String(id), count);
            return;
        }

        let evicted_votes = bucket.vote_pos;
        let evicted_id = bucket.evict_many(id, count);
        self.light
            .insert_many(&DataInput::String(evicted_id), evicted_votes);
    }

    /// Absorbs a `<flow, votes, eviction>` message from another sketch's heavy
    /// part, the insertion an OctoSketch aggregator applies to a heavy-part
    /// delta (§4.4) once the flag is carried with the counter.
    ///
    /// [`Self::insert_many`], plus the sender's flag OR-ed in when the arrival
    /// ends up resident here. The counter word handed between the two parts
    /// *is* the flag. A sender whose bucket is unflagged holds the flow's whole
    /// mass in its heavy part, so flagging it here would make the estimate read
    /// Count-Min noise it does not own.
    pub fn merge_heavy(&mut self, id: String, votes: i32, eviction: bool) {
        if votes <= 0 {
            return;
        }
        let idx = self.bucket_index(&id);
        let arrival = id.clone();
        self.insert_many(id, votes);
        if eviction && !self.heavy[idx].is_vacant() && self.heavy[idx].flow_id == arrival {
            self.heavy[idx].eviction = true;
        }
    }

    /// Absorbs a resident another sketch's heavy part evicted, under its own
    /// key.
    ///
    /// The votes go to the light layer under `id`, and `id`'s bucket here is
    /// flagged if it still holds it. `votes` of zero still flags: the sender
    /// has stopped being able to speak for this flow's heavy part, so whatever
    /// it sees of it next arrives through the light layer.
    pub fn absorb_evicted(&mut self, id: String, votes: i32) {
        let idx = self.bucket_index(&id);
        if votes > 0 {
            self.light
                .insert_many(&DataInput::String(id.clone()), votes);
        }
        if !self.heavy[idx].is_vacant() && self.heavy[idx].flow_id == id {
            self.heavy[idx].eviction = true;
        }
    }

    /// Records one occurrence of `id` against the heavy part alone, the
    /// overload path from the paper's section 3.3. The light layer is read by
    /// queries but never written, so an unelected flow is dropped outright.
    ///
    /// A vacant bucket seats the flow and a matching bucket takes a positive
    /// vote, both as in [`Self::insert`]. A non-matching arrival takes a
    /// negative vote and is discarded. On takeover the arrival keeps the
    /// evicted flow's positive vote instead of starting at 1, and the evicted
    /// flow's size is lost rather than spilled.
    ///
    /// The arrival also inherits the bucket's eviction flag, and the negative
    /// vote resets to 0: this path leaves the counter and its flag bit
    /// untouched.
    pub fn insert_heavy_only(&mut self, id: String) {
        let idx = self.bucket_index(&id);
        if self.stale_at(idx) {
            self.seat_over_stale_copy(idx, id, 1);
            return;
        }
        let bucket = &mut self.heavy[idx];

        if bucket.is_vacant() {
            bucket.occupy(id);
            return;
        }
        if bucket.flow_id == id {
            bucket.vote_pos += 1;
            return;
        }

        bucket.vote_neg += 1;
        if bucket.vote_neg < LAMBDA * bucket.vote_pos {
            return;
        }

        // vote_pos and eviction are left alone: the arrival inherits the
        // evicted flow's size and its flag
        bucket.flow_id = id;
        bucket.vote_neg = 0;
    }

    /// Frequency estimate for `id`: the resident vote count, plus the light
    /// layer whenever the bucket carries the eviction flag.
    pub fn query(&self, id: String) -> i32 {
        let idx = self.bucket_index(&id);
        let bucket = &self.heavy[idx];
        if !bucket.is_vacant() && bucket.flow_id == id {
            if bucket.eviction {
                bucket.vote_pos + self.light.estimate(&DataInput::String(id))
            } else {
                bucket.vote_pos
            }
        } else {
            self.light.estimate(&DataInput::String(id))
        }
    }

    /// Merges `other` in, keeping elephants in the heavy part and adding the
    /// light layers counter by counter -- the paper's Sum merging for the
    /// light half. Correct whatever the two sketches saw, including flows that
    /// appear on both sides.
    pub fn merge(&mut self, other: &Elastic<H>) {
        assert_eq!(
            self.bktlen, other.bktlen,
            "bucket length mismatch while merging Elastic sketches"
        );

        let losers = self.contest_heavy_against(other);
        self.light.merge(&other.light);
        self.spill(losers);
    }

    /// Merges `other` in, keeping elephants in the heavy part and keeping the
    /// larger of each light counter pair -- the paper's Maximum merging, and
    /// what the technical report's B.5 prescribes for combining light layers.
    ///
    /// The light half requires the two sketches to have observed **disjoint
    /// flow sets**: a mouse flow both sides saw reads back as the larger side
    /// rather than the sum. Flows held by both heavy parts are summed either
    /// way. Use [`Self::merge`] whenever mouse flows can repeat across
    /// sketches.
    pub fn merge_max(&mut self, other: &Elastic<H>) {
        assert_eq!(
            self.bktlen, other.bktlen,
            "bucket length mismatch while merging Elastic sketches"
        );

        let losers = self.contest_heavy_against(other);
        self.light.merge_max(&other.light);
        self.spill(losers);
    }

    /// Combines the two heavy parts bucket by bucket, returning the flows that
    /// lost their bucket and owe their votes to the light layer.
    ///
    /// Section 3.4 merges two buckets by querying both keys and keeping the
    /// larger, and that is the rule here, applied across sketches rather than
    /// within one. Each side is queried against its own sketch, before
    /// anything is written, so no decision reads a light layer this merge has
    /// already changed. A flow both sides held keeps one bucket with the votes
    /// summed.
    ///
    /// Every surviving bucket ends up flagged. The flow that keeps a bucket
    /// may have mass in the peer's light layer -- it could have been a mouse
    /// there -- and nothing short of trusting the peer's Count-Min can rule
    /// that out, so the flag is set and the estimate reads through. That
    /// overestimates rather than underestimates, which is the direction
    /// Elastic's guarantee allows.
    ///
    /// `vote_neg` takes the larger of the pair. The paper does not say what
    /// becomes of the votes, and the larger one evicts sooner.
    fn contest_heavy_against(&mut self, other: &Elastic<H>) -> Vec<(String, i32)> {
        let mut losers: Vec<(String, i32)> = Vec::new();
        let mut merged: Vec<Option<HeavyBucket>> = Vec::with_capacity(self.heavy.len());

        for idx in 0..self.heavy.len() {
            let mine = (!self.heavy[idx].is_vacant() && !self.stale_at(idx))
                .then(|| self.heavy[idx].clone());
            let theirs = (!other.heavy[idx].is_vacant() && !other.stale_at(idx))
                .then(|| other.heavy[idx].clone());

            merged.push(match (mine, theirs) {
                (None, None) => None,
                (Some(kept), None) | (None, Some(kept)) => Some(kept),
                (Some(mut kept), Some(peer)) if kept.flow_id == peer.flow_id => {
                    kept.vote_pos += peer.vote_pos;
                    kept.vote_neg = kept.vote_neg.max(peer.vote_neg);
                    Some(kept)
                }
                (Some(mine), Some(theirs)) => {
                    let my_size = self.query(mine.flow_id.clone());
                    let their_size = other.query(theirs.flow_id.clone());
                    let (kept, lost) = if my_size >= their_size {
                        (mine, theirs)
                    } else {
                        (theirs, mine)
                    };
                    losers.push((lost.flow_id, lost.vote_pos));
                    Some(kept)
                }
            });
        }

        for (idx, slot) in merged.into_iter().enumerate() {
            let bucket = &mut self.heavy[idx];
            match slot {
                Some(mut kept) => {
                    kept.eviction = true;
                    *bucket = kept;
                }
                None => {
                    bucket.flow_id = String::new();
                    bucket.vote_pos = 0;
                    bucket.vote_neg = 0;
                    bucket.eviction = true;
                }
            }
        }
        self.stale_copies = false;
        losers
    }

    /// Adds each flow's votes to the light layer.
    fn spill(&mut self, flows: Vec<(String, i32)>) {
        for (flow_id, votes) in flows {
            self.light.insert_many(&DataInput::String(flow_id), votes);
        }
    }

    /// Doubles the heavy table by appending a copy of itself, the paper's
    /// copy operation. Bucket count goes from `w` to `2w`, and by lemma 3.2
    /// every resident still hashes to a bucket holding it.
    ///
    /// Each flow now sits in both halves; the half it no longer hashes to is a
    /// stale copy, dropped lazily as inserts land on it. Two sketches expanded
    /// a different number of times can no longer merge -- [`Self::merge`]
    /// asserts on the bucket count.
    pub fn expand_heavy(&mut self) {
        let doubled = self
            .bktlen
            .checked_mul(2)
            .expect("heavy table size overflowed i32 while expanding");
        let copy = self.heavy.clone();
        self.heavy.extend(copy);
        self.bktlen = doubled;
        self.stale_copies = true;
    }

    /// Flows the heavy part holds, each with its whole estimate. Stale copies
    /// left by an expansion are skipped, so no flow is reported twice.
    fn resident_flows(&self) -> Vec<(String, i32)> {
        let mut flows: Vec<(String, i32)> = (0..self.heavy.len())
            .filter(|idx| !self.heavy[*idx].is_vacant() && !self.stale_at(*idx))
            .map(|idx| {
                let id = self.heavy[idx].flow_id.clone();
                let size = self.query(id.clone());
                (id, size)
            })
            .collect();
        flows.sort_unstable();
        flows
    }

    /// Section 5's heavy hitter detection: every flow in the heavy part whose
    /// estimate reaches `threshold`, sorted by flow id.
    ///
    /// The paper queries the size of each flow in the heavy part rather than
    /// reading `vote_pos`, so a flow that was evicted and came back reads
    /// through the light layer too.
    pub fn heavy_hitters(&self, threshold: i32) -> Vec<(String, i32)> {
        self.resident_flows()
            .into_iter()
            .filter(|(_, size)| *size >= threshold)
            .collect()
    }

    /// Section 5's heavy change detection over two adjacent windows: every flow
    /// held by either heavy part whose size moved by more than `threshold`.
    ///
    /// Each entry is `(flow, size in self, size in other)`; a flow absent from
    /// one window reads whatever that window's light layer holds for it.
    pub fn heavy_changes(&self, other: &Elastic<H>, threshold: i32) -> Vec<(String, i32, i32)> {
        let mut ids: Vec<String> = self
            .resident_flows()
            .into_iter()
            .map(|(id, _)| id)
            .chain(other.resident_flows().into_iter().map(|(id, _)| id))
            .collect();
        ids.sort_unstable();
        ids.dedup();

        ids.into_iter()
            .map(|id| {
                let before = self.query(id.clone());
                let after = other.query(id.clone());
                (id, before, after)
            })
            .filter(|(_, before, after)| (after - before).abs() > threshold)
            .collect()
    }

    /// Buckets whose resident flow is larger than `t2`, the paper's count of
    /// full buckets. Compare against a `T1` of your own to decide when to call
    /// [`Self::expand_heavy`].
    pub fn full_bucket_count(&self, t2: i32) -> usize {
        self.heavy
            .iter()
            .filter(|bucket| !bucket.is_vacant() && bucket.vote_pos > t2)
            .count()
    }

    /// Shrinks the heavy table by `ratio`, the paper's active compression.
    /// New bucket `j` absorbs old buckets `j`, `j + w'`, `j + 2w'`, ...; the
    /// largest resident of each group keeps its bucket, the rest spill.
    ///
    /// `ratio` must divide the bucket count. That is what lemma 3.2 needs for
    /// `(i % w) % w' == i % w'`, so every resident still hashes to the group
    /// that holds it.
    ///
    /// A loser is queried for its whole size but spills only its `vote_pos`;
    /// whatever the light layer already held for it is still there. The
    /// winner's bucket carries over untouched -- the paper does not say what
    /// becomes of the votes, and keeping its own pair leaves
    /// `vote_neg / vote_pos` a record of the contests it actually fought.
    pub fn compress_heavy(&mut self, ratio: i32) {
        assert!(
            ratio >= 1,
            "Elastic compression ratio must be at least 1, got {ratio}"
        );
        assert!(
            self.bktlen % ratio == 0,
            "Elastic compression ratio {ratio} must divide the bucket count {}",
            self.bktlen
        );
        if ratio == 1 {
            return;
        }

        self.drop_stale_copies();

        let width = (self.bktlen / ratio) as usize;
        let mut winner: Vec<Option<usize>> = vec![None; width];
        let mut best: Vec<i32> = vec![0; width];
        let mut flagged: Vec<bool> = vec![false; width];

        for idx in 0..self.heavy.len() {
            let group = idx % width;
            if self.heavy[idx].is_vacant() {
                flagged[group] |= self.heavy[idx].eviction;
                continue;
            }
            let size = self.query(self.heavy[idx].flow_id.clone());
            if winner[group].is_none() || size > best[group] {
                best[group] = size;
                winner[group] = Some(idx);
            }
        }

        let old = std::mem::take(&mut self.heavy);
        let mut compressed: Vec<HeavyBucket> = (0..width).map(|_| HeavyBucket::new()).collect();
        let mut losers: Vec<(String, i32)> = Vec::new();

        for (idx, bucket) in old.into_iter().enumerate() {
            let group = idx % width;
            if bucket.is_vacant() {
                continue;
            }
            if winner[group] == Some(idx) {
                compressed[group] = bucket;
            } else {
                losers.push((bucket.flow_id, bucket.vote_pos));
            }
        }

        for (group, bucket) in compressed.iter_mut().enumerate() {
            if bucket.is_vacant() && flagged[group] {
                bucket.eviction = true;
            }
        }

        self.heavy = compressed;
        self.bktlen = width as i32;

        for (flow_id, votes) in losers {
            self.light.insert_many(&DataInput::String(flow_id), votes);
        }
    }

    /// Empties every bucket holding a copy left behind by an expansion. The
    /// copy is not spilled: its live twin still carries the flow. The flag is
    /// kept so a later resident of the slot still reads the light layer.
    fn drop_stale_copies(&mut self) {
        if !self.stale_copies {
            return;
        }
        for idx in 0..self.heavy.len() {
            if self.stale_at(idx) {
                let bucket = &mut self.heavy[idx];
                bucket.flow_id = String::new();
                bucket.vote_pos = 0;
                bucket.vote_neg = 0;
                bucket.eviction = true;
            }
        }
        self.stale_copies = false;
    }

    /// Whether the bucket at `idx` holds a copy left behind by an expansion,
    /// meaning its resident hashes somewhere else now.
    #[inline]
    fn stale_at(&self, idx: usize) -> bool {
        if !self.stale_copies {
            return false;
        }
        let bucket = &self.heavy[idx];
        !bucket.is_vacant() && self.bucket_index(&bucket.flow_id) != idx
    }

    /// Drops a stale copy and seats `id` in its place. The copy is not spilled:
    /// the flow's live entry is in the twin bucket. The flag is set because the
    /// arrival shared this bucket before the expansion and lost to its
    /// resident, so the light layer already holds part of it.
    fn seat_over_stale_copy(&mut self, idx: usize, id: String, count: i32) {
        let bucket = &mut self.heavy[idx];
        bucket.flow_id = id;
        bucket.vote_pos = count;
        bucket.vote_neg = 0;
        bucket.eviction = true;
    }

    #[inline]
    fn bucket_index(&self, id: &str) -> usize {
        let hash = H::hash64_seeded(CANONICAL_HASH_SEED, &DataInput::Str(id));
        hash as usize % self.bktlen as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CANONICAL_HASH_SEED, DataInput, hash64_seeded};

    fn bucket_for(id: &str, sketch: &Elastic) -> usize {
        let hash = hash64_seeded(CANONICAL_HASH_SEED, &DataInput::Str(id));
        hash as usize % sketch.bktlen as usize
    }

    fn colliding_key(primary: &str, sketch: &Elastic) -> String {
        let target = bucket_for(primary, sketch);
        (0..10_000)
            .map(|idx| format!("flow::secondary::{idx}"))
            .find(|candidate| bucket_for(candidate, sketch) == target && candidate != primary)
            .expect("unable to find colliding key for test")
    }

    #[test]
    fn init_with_dimensions_sizes_both_parts() {
        let sketch: Elastic = Elastic::init_with_dimensions(12, 2, 256);

        assert_eq!(sketch.heavy.len(), 12);
        assert_eq!(sketch.bktlen, 12);
        assert_eq!(sketch.light.rows(), 2);
        assert_eq!(sketch.light.cols(), 256);
    }

    #[test]
    fn init_with_length_keeps_the_default_light_layer() {
        let sketch: Elastic = Elastic::init_with_length(8);

        assert_eq!(sketch.heavy.len(), 8);
        assert_eq!(sketch.light.rows(), DEFAULT_LIGHT_ROWS);
        assert_eq!(sketch.light.cols(), DEFAULT_LIGHT_COLS);
    }

    #[test]
    #[should_panic(expected = "at least one heavy bucket")]
    fn an_empty_heavy_table_is_rejected() {
        let _: Elastic = Elastic::init_with_length(0);
    }

    #[test]
    #[should_panic(expected = "at least one heavy bucket")]
    fn a_negative_heavy_table_is_rejected() {
        let _: Elastic = Elastic::init_with_length(-1);
    }

    #[test]
    #[should_panic(expected = "non-empty light layer")]
    fn an_empty_light_layer_is_rejected() {
        let _: Elastic = Elastic::init_with_dimensions(8, 0, 4096);
    }

    #[test]
    #[should_panic(expected = "non-empty light layer")]
    fn a_zero_width_light_layer_is_rejected() {
        let _: Elastic = Elastic::init_with_dimensions(8, 3, 0);
    }

    #[test]
    fn heavy_bucket_tracks_repeated_flow_exactly() {
        // repeated inserts of the same flow should accumulate in the heavy bucket
        let mut sketch: Elastic = Elastic::init_with_length(8);
        let flow = "flow::primary".to_string();

        for _ in 0..12 {
            sketch.insert(flow.clone());
        }

        assert_eq!(sketch.query(flow.clone()), 12);
        assert_eq!(sketch.query("other".to_string()), 0);
    }

    #[test]
    fn light_sketch_counts_colliding_flows() {
        // simulate two flows mapped to the same bucket so the light CountMin tracks the second one
        let mut sketch: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &sketch);

        for _ in 0..10 {
            sketch.insert(primary.to_string());
        }
        for _ in 0..6 {
            sketch.insert(secondary.clone());
        }

        let heavy_est = sketch.query(primary.to_string());
        let light_est = sketch.query(secondary.clone());

        assert!(
            heavy_est >= 10,
            "expected heavy bucket >= 10 after repeated inserts, got {heavy_est}"
        );
        assert!(
            light_est >= 6,
            "colliding flow should accumulate in CountMin, expected >= 6, got {light_est}"
        );
    }

    #[test]
    fn eviction_moves_the_resident_flow_into_the_light_layer() {
        // the paper evicts the flow sitting in the bucket, not the arriving one
        let mut sketch: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &sketch);

        for _ in 0..10 {
            sketch.insert(primary.to_string());
        }
        // vote_neg reaches LAMBDA * 10 on the 80th arrival, which triggers takeover
        for _ in 0..(LAMBDA * 10) {
            sketch.insert(secondary.clone());
        }

        let idx = bucket_for(primary, &sketch);
        assert_eq!(sketch.heavy[idx].flow_id, secondary, "takeover must happen");
        assert!(sketch.heavy[idx].eviction);
        assert_eq!(sketch.heavy[idx].vote_pos, 1);
        assert_eq!(sketch.heavy[idx].vote_neg, 1);

        assert_eq!(
            sketch.query(primary.to_string()),
            10,
            "evicted flow keeps its full size in the light layer"
        );
        assert_eq!(
            sketch.query(secondary.clone()),
            LAMBDA * 10,
            "arriving flow must not absorb the evicted flow's votes"
        );
    }

    #[test]
    fn merge_keeps_uncontested_flows_in_the_heavy_part() {
        let mut left: Elastic = Elastic::init_with_length(16);
        let mut right: Elastic = Elastic::init_with_length(16);

        for _ in 0..30 {
            left.insert("flow::left".to_string());
        }
        for _ in 0..18 {
            right.insert("flow::right".to_string());
        }

        left.merge(&right);

        assert_eq!(left.query("flow::left".to_string()), 30);
        assert_eq!(left.query("flow::right".to_string()), 18);

        // both keep a bucket: neither was flushed into the light layer
        let residents: Vec<(&str, i32)> = left
            .heavy
            .iter()
            .filter(|bucket| !bucket.is_vacant())
            .map(|bucket| (bucket.flow_id.as_str(), bucket.vote_pos))
            .collect();
        assert_eq!(residents.len(), 2, "both flows must stay resident");
        assert!(residents.contains(&("flow::left", 30)));
        assert!(residents.contains(&("flow::right", 18)));
        assert!(
            left.heavy.iter().all(|bucket| bucket.eviction),
            "every merged bucket reads through the light layer"
        );
    }

    #[test]
    fn merge_preserves_colliding_flow_mass() {
        let mut left: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &left);

        for _ in 0..20 {
            left.insert(primary.to_string());
        }

        let mut right: Elastic = Elastic::init_with_length(8);
        for _ in 0..9 {
            right.insert(secondary.clone());
        }

        left.merge(&right);

        assert!(left.query(primary.to_string()) >= 20);
        assert!(left.query(secondary.clone()) >= 9);
    }

    #[test]
    fn a_bucket_reoccupied_after_merge_still_reads_the_light_layer() {
        let mut left: Elastic = Elastic::init_with_length(8);
        for _ in 0..30 {
            left.insert("flow::left".to_string());
        }
        let right: Elastic = Elastic::init_with_length(8);

        left.merge(&right);
        left.insert("flow::left".to_string());

        assert_eq!(
            left.query("flow::left".to_string()),
            31,
            "a flow that kept its bucket through a merge goes on accumulating"
        );
    }

    #[test]
    fn merge_keeps_the_larger_flow_on_a_contested_bucket() {
        let mut left: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &left);

        for _ in 0..20 {
            left.insert(primary.to_string());
        }
        let mut right: Elastic = Elastic::init_with_length(8);
        for _ in 0..9 {
            right.insert(secondary.clone());
        }

        left.merge(&right);

        let idx = bucket_for(primary, &left);
        assert_eq!(left.heavy[idx].flow_id, primary, "the larger flow keeps it");
        assert_eq!(left.heavy[idx].vote_pos, 20);
        assert!(
            left.query(secondary.clone()) >= 9,
            "the loser's votes must reach the light layer"
        );
    }

    #[test]
    fn merge_sums_the_votes_of_a_flow_both_sides_held() {
        let mut left: Elastic = Elastic::init_with_length(8);
        let mut right: Elastic = Elastic::init_with_length(8);
        for _ in 0..30 {
            left.insert("flow::shared".to_string());
        }
        for _ in 0..20 {
            right.insert("flow::shared".to_string());
        }

        left.merge(&right);

        let idx = bucket_for("flow::shared", &left);
        assert_eq!(left.heavy[idx].flow_id, "flow::shared");
        assert_eq!(left.heavy[idx].vote_pos, 50);
        assert_eq!(left.query("flow::shared".to_string()), 50);
    }

    #[test]
    fn merge_keeps_the_peers_flow_when_it_is_the_larger() {
        // each side must be sized against its own sketch: the peer's flow is
        // absent from ours, so asking ours would read a light estimate near 0
        let mut left: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &left);

        for _ in 0..3 {
            left.insert(primary.to_string());
        }
        let mut right: Elastic = Elastic::init_with_length(8);
        for _ in 0..50 {
            right.insert(secondary.clone());
        }

        left.merge(&right);

        let idx = bucket_for(primary, &left);
        assert_eq!(
            left.heavy[idx].flow_id, secondary,
            "the peer's larger flow takes the bucket"
        );
        assert_eq!(left.heavy[idx].vote_pos, 50);
        assert!(left.query(primary.to_string()) >= 3);
    }

    #[test]
    fn merge_does_not_leave_a_stale_copy_as_a_resident() {
        // the merge clears the stale flag, so a copy it kept would look live
        let (mut sk, _) = seeded_sketch(8, 24);
        sk.expand_heavy();

        let empty: Elastic = Elastic::init_with_length(16);
        sk.merge(&empty);

        let mut ids: Vec<String> = sk.heavy_hitters(1).into_iter().map(|(id, _)| id).collect();
        let reported = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            reported,
            "a stale copy survived the merge as a resident"
        );
    }

    /// Two sketches sharing half their flows, over a light layer tight enough
    /// that contested buckets and spills both happen.
    fn overlapping_pair() -> (Elastic, Elastic, Vec<(String, i32)>) {
        let mut left: Elastic = Elastic::init_with_dimensions(8, 2, 64);
        let mut right: Elastic = Elastic::init_with_dimensions(8, 2, 64);
        let mut truth: Vec<(String, i32)> = Vec::new();

        for i in 0..60i32 {
            let key = format!("flow::{i}");
            let in_left = (i % 3) + 1;
            let in_right = if i % 2 == 0 { (i % 4) + 1 } else { 0 };
            for _ in 0..in_left {
                left.insert(key.clone());
            }
            for _ in 0..in_right {
                right.insert(key.clone());
            }
            truth.push((key, in_left + in_right));
        }
        (left, right, truth)
    }

    #[test]
    fn merge_never_underestimates_across_a_large_flow_set() {
        let (mut left, right, truth) = overlapping_pair();
        left.merge(&right);

        for (key, count) in &truth {
            let est = left.query(key.clone());
            assert!(est >= *count, "merge underestimated {key}: {est} < {count}");
        }
    }

    #[test]
    fn maximum_merging_keeps_the_larger_flow_on_a_contested_bucket() {
        let mut left: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &left);

        for _ in 0..20 {
            left.insert(primary.to_string());
        }
        let mut right: Elastic = Elastic::init_with_length(8);
        for _ in 0..9 {
            right.insert(secondary.clone());
        }

        left.merge_max(&right);

        let idx = bucket_for(primary, &left);
        assert_eq!(left.heavy[idx].flow_id, primary);
        assert_eq!(left.heavy[idx].vote_pos, 20);
        assert!(left.query(secondary.clone()) >= 9);
    }

    /// Builds two sketches over disjoint flow sets, plus the truth table.
    fn disjoint_pair() -> (Elastic, Elastic, Vec<(String, i32)>) {
        let mut left: Elastic = Elastic::init_with_dimensions(8, 2, 64);
        let mut right: Elastic = Elastic::init_with_dimensions(8, 2, 64);
        let mut truth = Vec::new();

        for i in 0..40i32 {
            let key = format!("left::{i}");
            let count = (i % 7) + 1;
            for _ in 0..count {
                left.insert(key.clone());
            }
            truth.push((key, count));
        }
        for i in 0..40i32 {
            let key = format!("right::{i}");
            let count = (i % 5) + 1;
            for _ in 0..count {
                right.insert(key.clone());
            }
            truth.push((key, count));
        }
        (left, right, truth)
    }

    #[test]
    fn maximum_merging_never_underestimates_disjoint_flows() {
        let (mut left, right, truth) = disjoint_pair();
        left.merge_max(&right);

        for (key, count) in &truth {
            let est = left.query(key.clone());
            assert!(
                est >= *count,
                "maximum merging underestimated {key}: {est} < {count}"
            );
        }
    }

    #[test]
    fn maximum_merging_is_tighter_than_sum_merging() {
        let (mut summed, right, truth) = disjoint_pair();
        summed.merge(&right);
        let (mut maxed, right, _) = disjoint_pair();
        maxed.merge_max(&right);

        let mut strictly_tighter = 0;
        for (key, _) in &truth {
            let sum_est = summed.query(key.clone());
            let max_est = maxed.query(key.clone());
            assert!(
                max_est <= sum_est,
                "maximum merging must not be looser on {key}: {max_est} > {sum_est}"
            );
            if max_est < sum_est {
                strictly_tighter += 1;
            }
        }
        assert!(
            strictly_tighter > 0,
            "no flow got a tighter estimate, so this input cannot tell the merges apart"
        );
    }

    /// One bucket held by a hot flow, so `flow::mouse` never gets a bucket and
    /// lives entirely in the light layer on both sides.
    fn shared_mouse_pair(left_count: i32, right_count: i32) -> (Elastic, Elastic) {
        let mut left: Elastic = Elastic::init_with_dimensions(1, 2, 64);
        let mut right: Elastic = Elastic::init_with_dimensions(1, 2, 64);
        for sketch in [&mut left, &mut right] {
            for _ in 0..500 {
                sketch.insert("flow::hot".to_string());
            }
        }
        for _ in 0..left_count {
            left.insert("flow::mouse".to_string());
        }
        for _ in 0..right_count {
            right.insert("flow::mouse".to_string());
        }
        assert_eq!(
            left.heavy[0].flow_id, "flow::hot",
            "the mouse must not seat"
        );
        (left, right)
    }

    #[test]
    fn maximum_merging_underestimates_a_mouse_flow_both_sides_saw() {
        // the paper's precondition, pinned: MM is for disjoint flow sets, and a
        // shared mouse reads back as the larger side instead of the sum
        let (mut maxed, right) = shared_mouse_pair(30, 20);
        maxed.merge_max(&right);
        assert_eq!(maxed.query("flow::mouse".to_string()), 30);

        let (mut summed, right) = shared_mouse_pair(30, 20);
        summed.merge(&right);
        assert_eq!(summed.query("flow::mouse".to_string()), 50);
    }

    #[test]
    fn maximum_merging_sums_a_flow_both_heavy_parts_held() {
        // the heavy halves are combined bucket by bucket either way, so a
        // shared elephant is summed even under maximum merging
        let mut left: Elastic = Elastic::init_with_length(8);
        let mut right: Elastic = Elastic::init_with_length(8);
        for _ in 0..30 {
            left.insert("flow::shared".to_string());
        }
        for _ in 0..20 {
            right.insert("flow::shared".to_string());
        }

        left.merge_max(&right);

        let idx = bucket_for("flow::shared", &left);
        assert_eq!(left.heavy[idx].vote_pos, 50);
        assert_eq!(left.query("flow::shared".to_string()), 50);
    }

    /// Every counter of the light layer, so a test can prove it went untouched.
    fn light_snapshot(sketch: &Elastic) -> Vec<i32> {
        let storage = sketch.light.as_storage();
        (0..storage.rows())
            .flat_map(|i| (0..storage.cols()).map(move |j| (i, j)))
            .map(|(i, j)| storage.query_one_counter(i, j))
            .collect()
    }

    #[test]
    fn heavy_only_insert_never_touches_the_light_layer() {
        // seed the light layer through the normal path so the comparison is
        // against real counters rather than an all-zero table
        let mut sketch: Elastic = Elastic::init_with_dimensions(8, 2, 64);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &sketch);
        for _ in 0..4 {
            sketch.insert(primary.to_string());
        }
        for _ in 0..3 {
            sketch.insert(secondary.clone());
        }

        let before = light_snapshot(&sketch);
        assert!(
            before.iter().any(|count| *count > 0),
            "the light layer must hold something for this test to mean anything"
        );

        // vacant bucket, then a match
        let vacant = (0..10_000)
            .map(|idx| format!("flow::fresh::{idx}"))
            .find(|candidate| sketch.heavy[bucket_for(candidate, &sketch)].is_vacant())
            .expect("a vacant bucket must exist in an 8-bucket table");
        sketch.insert_heavy_only(vacant.clone());
        sketch.insert_heavy_only(vacant.clone());

        // non-matching arrivals: discarded first, then a takeover
        for _ in 0..(LAMBDA * 4 + 4) {
            sketch.insert_heavy_only(secondary.clone());
        }
        assert_eq!(
            sketch.heavy[bucket_for(primary, &sketch)].flow_id,
            secondary,
            "the run must reach a takeover to cover that case"
        );

        assert_eq!(
            light_snapshot(&sketch),
            before,
            "the heavy-only path must leave every light counter alone"
        );
    }

    #[test]
    fn heavy_only_takeover_inherits_the_evicted_flow_size() {
        let mut sketch: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &sketch);

        for _ in 0..10 {
            sketch.insert_heavy_only(primary.to_string());
        }
        // vote_neg reaches LAMBDA * 10 on the 80th arrival, which takes over
        for _ in 0..(LAMBDA * 10) {
            sketch.insert_heavy_only(secondary.clone());
        }

        let bucket = &sketch.heavy[bucket_for(primary, &sketch)];
        assert_eq!(bucket.flow_id, secondary);
        assert_eq!(
            bucket.vote_pos, 10,
            "the arrival inherits the evicted flow's size, not a fresh 1"
        );
        assert_eq!(bucket.vote_neg, 0);
    }

    #[test]
    fn heavy_only_takeover_inherits_the_eviction_flag() {
        // the counter and its flag bit stay in place, so the arrival takes
        // over both
        for seeded_flag in [false, true] {
            let mut sketch: Elastic = Elastic::init_with_length(8);
            let primary = "flow::primary";
            let secondary = colliding_key(primary, &sketch);
            let idx = bucket_for(primary, &sketch);

            for _ in 0..10 {
                sketch.insert_heavy_only(primary.to_string());
            }
            sketch.heavy[idx].eviction = seeded_flag;

            for _ in 0..(LAMBDA * 10) {
                sketch.insert_heavy_only(secondary.clone());
            }

            assert_eq!(sketch.heavy[idx].flow_id, secondary);
            assert_eq!(
                sketch.heavy[idx].eviction, seeded_flag,
                "the arrival must inherit the bucket's flag, not overwrite it"
            );
        }
    }

    #[test]
    fn heavy_only_takeover_discards_the_evicted_flow_as_designed() {
        // the paper trades the evicted flow's size for one probe per packet
        let mut sketch: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &sketch);

        for _ in 0..10 {
            sketch.insert_heavy_only(primary.to_string());
        }
        for _ in 0..(LAMBDA * 10) {
            sketch.insert_heavy_only(secondary.clone());
        }

        assert_eq!(
            sketch.query(primary.to_string()),
            0,
            "the evicted flow's 10 packets are gone, not spilled to the light layer"
        );
    }

    #[test]
    fn heavy_only_matches_insert_while_buckets_seat_and_match() {
        let mut normal: Elastic = Elastic::init_with_length(16);
        let mut overload: Elastic = Elastic::init_with_length(16);

        for i in 0..6 {
            let flow = format!("flow::{i}");
            for _ in 0..(i + 3) {
                normal.insert(flow.clone());
                overload.insert_heavy_only(flow.clone());
            }
        }

        for (lhs, rhs) in normal.heavy.iter().zip(overload.heavy.iter()) {
            assert_eq!(lhs.flow_id, rhs.flow_id);
            assert_eq!(lhs.vote_pos, rhs.vote_pos);
            assert_eq!(lhs.vote_neg, rhs.vote_neg);
            assert_eq!(lhs.eviction, rhs.eviction);
        }
        for i in 0..6 {
            let flow = format!("flow::{i}");
            assert_eq!(
                normal.query(flow.clone()),
                overload.query(flow.clone()),
                "seating and matching must agree between the two paths"
            );
        }
    }

    /// Flows whose estimate should survive a doubling, with the truth table.
    fn seeded_sketch(buckets: i32, flows: usize) -> (Elastic, Vec<(String, i32)>) {
        let mut sk: Elastic = Elastic::init_with_length(buckets);
        let mut truth = Vec::new();
        for i in 0..flows {
            let key = format!("flow::{i}");
            let count = (i as i32 % 5) + 1;
            for _ in 0..count {
                sk.insert(key.clone());
            }
            truth.push((key, count));
        }
        (sk, truth)
    }

    #[test]
    fn expansion_doubles_the_heavy_table() {
        let mut sk: Elastic = Elastic::init_with_length(8);
        sk.insert("flow::a".to_string());

        sk.expand_heavy();

        assert_eq!(sk.bktlen, 16);
        assert_eq!(sk.heavy.len(), 16);
        assert!(sk.stale_copies);
    }

    #[test]
    fn expansion_preserves_every_existing_estimate() {
        // lemma 3.2: h(f) % 2w lands on a half that already holds f
        let (mut sk, truth) = seeded_sketch(8, 24);
        let before: Vec<i32> = truth.iter().map(|(k, _)| sk.query(k.clone())).collect();

        sk.expand_heavy();

        for ((key, _), was) in truth.iter().zip(before) {
            assert_eq!(sk.query(key.clone()), was, "estimate for {key} moved");
        }
    }

    #[test]
    fn repeated_expansion_keeps_estimates_intact() {
        let (mut sk, truth) = seeded_sketch(8, 24);
        let before: Vec<i32> = truth.iter().map(|(k, _)| sk.query(k.clone())).collect();

        sk.expand_heavy();
        sk.expand_heavy();

        assert_eq!(sk.bktlen, 32);
        for ((key, _), was) in truth.iter().zip(before) {
            assert_eq!(sk.query(key.clone()), was, "estimate for {key} moved");
        }
    }

    #[test]
    fn an_insert_onto_a_stale_copy_replaces_it() {
        let (mut sk, _) = seeded_sketch(8, 24);
        sk.expand_heavy();

        let stale_idx = (0..sk.heavy.len())
            .find(|idx| sk.stale_at(*idx))
            .expect("a doubling must leave at least one stale copy");
        let displaced = sk.heavy[stale_idx].flow_id.clone();

        let arrival = (0..10_000)
            .map(|i| format!("late::{i}"))
            .find(|key| sk.bucket_index(key) == stale_idx)
            .expect("unable to find a key for the stale bucket");
        sk.insert(arrival.clone());

        assert_eq!(sk.heavy[stale_idx].flow_id, arrival);
        assert_eq!(sk.heavy[stale_idx].vote_pos, 1);
        assert!(!sk.stale_at(stale_idx));
        // the displaced copy was not the flow's live entry, which still answers
        assert!(sk.query(displaced.clone()) > 0, "{displaced} lost its mass");
    }

    #[test]
    fn merge_after_expansion_does_not_double_count() {
        // every flow sits in both halves after a doubling; flushing both copies
        // would spill it twice
        let (mut sk, truth) = seeded_sketch(8, 24);
        sk.expand_heavy();

        let empty: Elastic = Elastic::init_with_length(16);
        sk.merge(&empty);

        for (key, count) in &truth {
            assert_eq!(
                sk.query(key.clone()),
                *count,
                "{key} came back doubled or short after merging an expanded table"
            );
        }
    }

    #[test]
    fn merge_does_not_double_count_an_expanded_peer() {
        // the peer's stale copies must be skipped too, not just our own
        let (mut right, truth) = seeded_sketch(8, 24);
        right.expand_heavy();

        let mut left: Elastic = Elastic::init_with_length(16);
        left.merge(&right);

        for (key, count) in &truth {
            assert_eq!(
                left.query(key.clone()),
                *count,
                "{key} came back doubled or short after merging an expanded peer"
            );
        }
    }

    #[test]
    fn maximum_merging_does_not_double_count_an_expanded_peer() {
        let (mut right, truth) = seeded_sketch(8, 24);
        right.expand_heavy();

        let mut left: Elastic = Elastic::init_with_length(16);
        left.merge_max(&right);

        for (key, count) in &truth {
            assert_eq!(
                left.query(key.clone()),
                *count,
                "{key} came back doubled or short after max merging an expanded peer"
            );
        }
    }

    #[test]
    fn full_bucket_count_counts_residents_above_the_threshold() {
        let mut sk: Elastic = Elastic::init_with_length(64);
        for i in 0..6 {
            let key = format!("hot::{i}");
            for _ in 0..10 {
                sk.insert(key.clone());
            }
        }

        assert_eq!(sk.full_bucket_count(9), 6);
        assert_eq!(sk.full_bucket_count(10), 0);
    }

    /// Sums every flow's estimate, for checking mass is neither lost nor doubled.
    fn total_estimate(sk: &Elastic, truth: &[(String, i32)]) -> i32 {
        truth.iter().map(|(k, _)| sk.query(k.clone())).sum()
    }

    #[test]
    fn compression_shrinks_the_heavy_table() {
        let (mut sk, _) = seeded_sketch(16, 24);

        sk.compress_heavy(4);

        assert_eq!(sk.bktlen, 4);
        assert_eq!(sk.heavy.len(), 4);
    }

    #[test]
    fn compression_keeps_the_larger_flow_and_spills_the_smaller() {
        let mut sk: Elastic = Elastic::init_with_length(8);
        // after halving, buckets j and j+4 merge, so pick a pair four apart
        let big = (0..10_000)
            .map(|i| format!("big::{i}"))
            .find(|key| sk.bucket_index(key) == 0)
            .expect("no key for bucket 0");
        let small = (0..10_000)
            .map(|i| format!("small::{i}"))
            .find(|key| sk.bucket_index(key) == 4)
            .expect("no key for bucket 4");

        for _ in 0..30 {
            sk.insert(big.clone());
        }
        for _ in 0..3 {
            sk.insert(small.clone());
        }

        sk.compress_heavy(2);

        assert_eq!(sk.heavy[0].flow_id, big, "the larger flow keeps the bucket");
        assert_eq!(sk.query(big.clone()), 30);
        assert!(
            sk.heavy.iter().all(|b| b.flow_id != small),
            "the smaller flow must leave the heavy part"
        );
        assert!(
            sk.query(small.clone()) >= 3,
            "the spilled flow underestimated: {}",
            sk.query(small.clone())
        );
    }

    #[test]
    fn compression_neither_loses_nor_doubles_mass() {
        let (mut sk, truth) = seeded_sketch(16, 40);
        let before = total_estimate(&sk, &truth);

        sk.compress_heavy(4);

        for (key, count) in &truth {
            assert!(
                sk.query(key.clone()) >= *count,
                "{key} underestimated after compression"
            );
        }
        let after = total_estimate(&sk, &truth);
        assert!(
            after >= before,
            "compression may only add error, went {before} -> {after}"
        );
        assert!(
            after < before * 2,
            "compression doubled the mass, went {before} -> {after}"
        );
    }

    #[test]
    #[should_panic(expected = "must divide the bucket count")]
    fn a_ratio_that_does_not_divide_the_table_is_rejected() {
        let mut sk: Elastic = Elastic::init_with_length(8);
        sk.compress_heavy(3);
    }

    #[test]
    fn compression_after_expansion_does_not_double_count() {
        // 12 -> 24 puts a flow's twin 12 buckets away; compressing to 8 lands
        // the two in different groups, where the copy would win its own group
        let (mut sk, truth) = seeded_sketch(12, 40);
        sk.expand_heavy();
        assert!(sk.stale_copies);

        sk.compress_heavy(3);

        assert!(!sk.stale_copies);
        assert_eq!(sk.bktlen, 8);
        let twice: Vec<&String> = truth
            .iter()
            .map(|(k, _)| k)
            .filter(|k| sk.heavy.iter().filter(|b| &&b.flow_id == k).count() > 1)
            .collect();
        assert!(twice.is_empty(), "flows resident twice: {twice:?}");

        // merging reads every flow back through the light layer, where a
        // spurious spill of the copy's votes would show up as doubled mass
        let empty: Elastic = Elastic::init_with_length(8);
        sk.merge(&empty);
        for (key, count) in &truth {
            assert_eq!(
                sk.query(key.clone()),
                *count,
                "{key} came back doubled or short after expand then compress"
            );
        }
    }

    #[test]
    fn expand_then_compress_returns_to_the_original_size() {
        let (mut sk, truth) = seeded_sketch(8, 24);

        sk.expand_heavy();
        sk.compress_heavy(2);

        assert_eq!(sk.bktlen, 8);
        assert_eq!(sk.heavy.len(), 8);
        for (key, count) in &truth {
            assert!(
                sk.query(key.clone()) >= *count,
                "{key} underestimated after a round trip"
            );
        }
    }
    /// Seats each `(flow, count)` in its own sketch and asserts every one of
    /// them really is resident, so a hash collision fails the fixture loudly
    /// rather than quietly changing what the test covers.
    fn sketch_with_resident_flows(buckets: i32, flows: &[(&str, i32)]) -> Elastic {
        let mut sk: Elastic = Elastic::init_with_length(buckets);
        for (id, count) in flows {
            for _ in 0..*count {
                sk.insert((*id).to_string());
            }
        }
        for (id, count) in flows {
            let idx = bucket_for(id, &sk);
            assert_eq!(
                sk.heavy[idx].flow_id, *id,
                "fixture flow {id} is not resident; pick different keys"
            );
            assert_eq!(sk.query((*id).to_string()), *count, "fixture flow {id}");
        }
        sk
    }

    #[test]
    fn heavy_hitters_reports_every_resident_above_the_threshold() {
        let sk = sketch_with_resident_flows(
            256,
            &[
                ("flow::alpha", 50),
                ("flow::beta", 30),
                ("flow::gamma", 12),
                ("flow::delta", 3),
            ],
        );

        assert_eq!(
            sk.heavy_hitters(20),
            vec![
                ("flow::alpha".to_string(), 50),
                ("flow::beta".to_string(), 30),
            ]
        );
        assert_eq!(sk.heavy_hitters(100), vec![]);
        assert_eq!(sk.heavy_hitters(1).len(), 4);
    }

    #[test]
    fn heavy_hitters_size_a_flagged_resident_through_the_light_layer() {
        // a resident that took its bucket over carries most of its size in the
        // light layer, so vote_pos alone would not clear the threshold
        let mut sketch: Elastic = Elastic::init_with_length(8);
        let primary = "flow::primary";
        let secondary = colliding_key(primary, &sketch);

        for _ in 0..10 {
            sketch.insert(primary.to_string());
        }
        for _ in 0..(LAMBDA * 10) {
            sketch.insert(secondary.clone());
        }

        let idx = bucket_for(primary, &sketch);
        assert_eq!(
            sketch.heavy[idx].vote_pos, 1,
            "the takeover leaves one vote"
        );
        assert!(sketch.heavy[idx].eviction);

        assert_eq!(
            sketch.heavy_hitters(50),
            vec![(secondary, LAMBDA * 10)],
            "a flagged resident is sized by query, not by vote_pos"
        );
    }

    #[test]
    fn heavy_hitters_includes_a_flow_sitting_exactly_on_the_threshold() {
        // the reference reports on `val >= threshold`
        let sk = sketch_with_resident_flows(256, &[("flow::on", 20), ("flow::under", 19)]);

        assert_eq!(
            sk.heavy_hitters(20),
            vec![("flow::on".to_string(), 20)],
            "a flow equal to the threshold is a heavy hitter"
        );
    }

    #[test]
    fn heavy_hitters_does_not_report_a_flow_twice_after_expansion() {
        let mut sk = sketch_with_resident_flows(
            8,
            &[("flow::alpha", 50), ("flow::beta", 30), ("flow::gamma", 25)],
        );
        sk.expand_heavy();

        // every resident now has a copy in the half it no longer hashes to
        assert!(sk.stale_copies);
        assert_eq!(
            sk.heavy_hitters(20),
            vec![
                ("flow::alpha".to_string(), 50),
                ("flow::beta".to_string(), 30),
                ("flow::gamma".to_string(), 25),
            ]
        );
    }

    #[test]
    fn heavy_changes_reports_only_moves_past_the_threshold() {
        let before = sketch_with_resident_flows(
            256,
            &[
                ("flow::rising", 10),
                ("flow::falling", 60),
                ("flow::steady", 40),
            ],
        );
        let after = sketch_with_resident_flows(
            256,
            &[
                ("flow::rising", 55),
                ("flow::falling", 8),
                ("flow::steady", 42),
            ],
        );

        assert_eq!(
            before.heavy_changes(&after, 20),
            vec![
                ("flow::falling".to_string(), 60, 8),
                ("flow::rising".to_string(), 10, 55),
            ],
            "steady moved by 2 and must not be reported"
        );
    }

    #[test]
    fn heavy_changes_covers_a_flow_present_in_only_one_window() {
        let before = sketch_with_resident_flows(256, &[("flow::gone", 40), ("flow::kept", 30)]);
        let after = sketch_with_resident_flows(256, &[("flow::kept", 31), ("flow::new", 45)]);

        assert_eq!(
            before.heavy_changes(&after, 20),
            vec![
                ("flow::gone".to_string(), 40, 0),
                ("flow::new".to_string(), 0, 45),
            ],
            "a flow in one window only is a change against zero"
        );
    }

    #[test]
    fn heavy_changes_reports_each_flow_once() {
        // the flow is resident in both windows and each expansion leaves it a
        // stale copy, so it reaches the id list four times
        let mut before = sketch_with_resident_flows(8, &[("flow::rising", 10)]);
        let mut after = sketch_with_resident_flows(8, &[("flow::rising", 55)]);
        before.expand_heavy();
        after.expand_heavy();

        assert_eq!(
            before.heavy_changes(&after, 20),
            vec![("flow::rising".to_string(), 10, 55)]
        );
    }
}
