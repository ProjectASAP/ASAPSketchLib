//! # Space-Saving (Metwally, Agrawal, El Abbadi — ICDT 2005)
//!
//! Tracks the most frequent keys of a stream in a fixed number of counters.
//! Every arrival either lands on a monitored key or displaces the smallest one,
//! taking over its count as an error allowance, so a monitored key's recorded
//! count never underestimates its true count and overstates it by at most
//! [`SpaceSaving::error`].
//!
//! ## Structure
//!
//! The Stream-Summary of section 3.1: buckets carrying a count, ordered by
//! count in a doubly linked list, each owning a doubly linked list of the
//! counters at that count, plus a key index into those counters. A unit arrival
//! moves one counter to the neighbouring bucket and an eviction takes the head
//! of the lowest bucket, so [`SpaceSaving::insert`] touches a constant number of
//! links whatever the capacity. A weighted arrival lands further along the
//! bucket list and walks it to reach its destination, one step per bucket it
//! passes.
//!
//! Both lists are arenas of indices rather than pointers: `monitored` is
//! allocated once up to `capacity` and reused in place, and `buckets` recycles
//! through a free list. Only the `(key, count, error)` triples reach the wire;
//! the arenas and the key index are rebuilt from them on load.
//!
//! ## Reference
//! * "Efficient Computation of Frequent and Top-k Elements in Data Streams",
//!   ICDT 2005. <https://doi.org/10.1007/978-3-540-30570-5_27>

use crate::common::DigestBuildHasher;
use crate::{DataInput, DefaultXxHasher, HeapItem, SketchHasher, input_to_owned};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::marker::PhantomData;

pub mod wire;

const NIL: usize = usize::MAX;

/// Counters in a default sketch.
pub const SPACE_SAVING_DEFAULT_CAPACITY: usize = 1024;

/// Positions of the monitored keys sharing one digest. Two monitored keys colliding on a
/// 64-bit digest is rare enough that the inline pair is effectively never
/// spilled.
type Slot = SmallVec<[usize; 2]>;

type Index = HashMap<u64, Slot, DigestBuildHasher>;

/// One monitored key.
#[derive(Clone, Debug)]
struct MonitoredKey {
    key: HeapItem,
    digest: u64,
    error: u64,
    bucket: usize,
    prev: usize,
    next: usize,
}

/// One count value, owning every counter currently at that count.
#[derive(Clone, Debug)]
struct Bucket {
    count: u64,
    head: usize,
    prev: usize,
    next: usize,
}

/// A Space-Saving summary over a fixed number of counters.
#[derive(Clone, Debug)]
pub struct SpaceSaving<H: SketchHasher = DefaultXxHasher> {
    capacity: usize,
    monitored: Vec<MonitoredKey>,
    buckets: Vec<Bucket>,
    bucket_free: Vec<usize>,
    bucket_head: usize,
    bucket_tail: usize,
    index: Index,
    total: u64,
    /// Largest count known to have left the summary, so the ceiling on any key
    /// that is no longer monitored.
    discarded_max: u64,
    _hasher: PhantomData<H>,
}

impl<H: SketchHasher> Default for SpaceSaving<H> {
    fn default() -> Self {
        Self::with_capacity(SPACE_SAVING_DEFAULT_CAPACITY)
    }
}

impl<H: SketchHasher> SpaceSaving<H> {
    /// Creates a summary that monitors at most `capacity` keys at a time.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            monitored: Vec::with_capacity(capacity),
            buckets: Vec::new(),
            bucket_free: Vec::new(),
            bucket_head: NIL,
            bucket_tail: NIL,
            index: Index::with_capacity_and_hasher(capacity, DigestBuildHasher::default()),
            total: 0,
            discarded_max: 0,
            _hasher: PhantomData,
        }
    }

    /// Counters this summary can hold.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Keys currently monitored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.monitored.len()
    }

    /// True while nothing has been recorded.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.monitored.is_empty()
    }

    /// Total weight recorded, monitored or displaced.
    #[inline(always)]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// The ceiling on the true count of any key the summary does not monitor.
    ///
    /// The larger of the smallest count still held, once every counter is in
    /// use, and the largest count that has left the summary through an eviction
    /// or a merge. Zero while the summary has spare counters and has dropped
    /// nothing.
    #[inline(always)]
    pub fn min_count(&self) -> u64 {
        let lowest = if self.monitored.len() == self.capacity && self.bucket_head != NIL {
            self.buckets[self.bucket_head].count
        } else {
            0
        };
        self.discarded_max.max(lowest)
    }

    /// Drops every counter.
    pub fn clear(&mut self) {
        self.monitored.clear();
        self.buckets.clear();
        self.bucket_free.clear();
        self.bucket_head = NIL;
        self.bucket_tail = NIL;
        self.index.clear();
        self.total = 0;
        self.discarded_max = 0;
    }

    /// Records one occurrence of `value`.
    #[inline]
    pub fn insert(&mut self, value: &DataInput) {
        self.insert_many(value, 1);
    }

    /// Records `count` occurrences of `value` in one step.
    ///
    /// A weighted arrival on a monitored key raises it by `count`; one that
    /// takes a free or displaced counter seats itself at [`Self::min_count`]
    /// plus `count` and carries that minimum as its error. `count` of zero is a
    /// no-op, and counts saturate at [`u64::MAX`].
    pub fn insert_many(&mut self, value: &DataInput, count: u64) {
        if count == 0 {
            return;
        }
        self.total = self.total.saturating_add(count);
        let digest = H::hash64_seeded(0, value);

        if let Some(cid) = self.find(digest, value) {
            self.raise(cid, count);
            return;
        }

        if self.monitored.len() < self.capacity {
            let seated = self.discarded_max.saturating_add(count);
            let discarded_max = self.discarded_max;
            self.seat(digest, input_to_owned(value), seated, discarded_max);
            return;
        }

        let victim = self.buckets[self.bucket_head].head;
        let lowest = self.buckets[self.bucket_head].count;
        debug_assert!(
            self.discarded_max <= lowest,
            "the ceiling {} sits above the lowest live count {lowest}",
            self.discarded_max
        );
        self.discarded_max = lowest;
        self.unindex(self.monitored[victim].digest, victim);
        self.monitored[victim].key = input_to_owned(value);
        self.monitored[victim].digest = digest;
        self.monitored[victim].error = lowest;
        self.index.entry(digest).or_default().push(victim);
        self.raise(victim, count);
    }

    /// Records every value in `values`.
    pub fn bulk_insert(&mut self, values: &[DataInput]) {
        for value in values {
            self.insert(value);
        }
    }

    /// The recorded count for `value`, or zero when it is not monitored.
    ///
    /// A monitored key's count never falls below its true count and exceeds it
    /// by at most [`Self::error`]. An unmonitored key reports zero; its true
    /// count cannot exceed [`Self::min_count`], which [`Self::upper_bound`]
    /// reports instead.
    pub fn estimate(&self, value: &DataInput) -> u64 {
        let digest = H::hash64_seeded(0, value);
        match self.find(digest, value) {
            Some(cid) => self.buckets[self.monitored[cid].bucket].count,
            None => 0,
        }
    }

    /// The largest count `value` can have, monitored or not.
    ///
    /// Equal to [`Self::estimate`] for a monitored key and to
    /// [`Self::min_count`] otherwise, so it never falls below the truth for any
    /// key in the stream.
    pub fn upper_bound(&self, value: &DataInput) -> u64 {
        let digest = H::hash64_seeded(0, value);
        match self.find(digest, value) {
            Some(cid) => self.buckets[self.monitored[cid].bucket].count,
            None => self.min_count(),
        }
    }

    /// How far [`Self::estimate`] may sit from the truth for `value`: above it
    /// by this much for a monitored key, and below it by this much for one the
    /// summary does not hold, where the estimate reads zero.
    pub fn error(&self, value: &DataInput) -> u64 {
        let digest = H::hash64_seeded(0, value);
        match self.find(digest, value) {
            Some(cid) => self.monitored[cid].error,
            None => self.min_count(),
        }
    }

    /// True when `value`'s count is above every unmonitored key's ceiling, so
    /// its place among the heavy hitters is certain rather than probable.
    pub fn is_guaranteed(&self, value: &DataInput) -> bool {
        let digest = H::hash64_seeded(0, value);
        match self.find(digest, value) {
            Some(cid) => {
                let count = self.buckets[self.monitored[cid].bucket].count;
                count.saturating_sub(self.monitored[cid].error) > self.min_count()
            }
            None => false,
        }
    }

    /// The `k` monitored keys with the largest counts, highest first, as
    /// `(key, count, error)`.
    pub fn top_k(&self, k: usize) -> Vec<(HeapItem, u64, u64)> {
        let mut out = Vec::with_capacity(k.min(self.monitored.len()));
        let mut bid = self.bucket_tail;
        while bid != NIL && out.len() < k {
            let count = self.buckets[bid].count;
            let mut cid = self.buckets[bid].head;
            while cid != NIL && out.len() < k {
                out.push((
                    self.monitored[cid].key.clone(),
                    count,
                    self.monitored[cid].error,
                ));
                cid = self.monitored[cid].next;
            }
            bid = self.buckets[bid].prev;
        }
        out
    }

    /// Every monitored key as `(key, count, error)`, in no particular order.
    pub fn entries(&self) -> Vec<(HeapItem, u64, u64)> {
        self.monitored
            .iter()
            .map(|c| (c.key.clone(), self.buckets[c.bucket].count, c.error))
            .collect()
    }

    /// Absorbs `other`.
    ///
    /// Counts for a shared key add, and a key held by only one side takes the
    /// other's `min_count` as both extra count and extra error, since that is
    /// all the other side can say about it. The union is then trimmed back to
    /// `capacity`. This is not equivalent to running one summary over the
    /// concatenated streams — a key evicted on both sides cannot be recovered.
    ///
    /// The `min_count` a one-sided key picks up is added weight that the stream
    /// never carried, so the merged counts sum to more than [`Self::total`] and
    /// an estimate divided by the total is no longer a frequency.
    pub fn merge(&mut self, other: &Self) {
        let mine_min = self.min_count();
        let theirs_min = other.min_count();

        let mut merged: HashMap<u64, SmallVec<[MergeEntry; 2]>, DigestBuildHasher> =
            HashMap::default();
        for c in &self.monitored {
            merged.entry(c.digest).or_default().push(MergeEntry {
                key: c.key.clone(),
                digest: c.digest,
                count: self.buckets[c.bucket].count,
                error: c.error,
                paired: false,
            });
        }
        for c in &other.monitored {
            let count = other.buckets[c.bucket].count;
            let slot = merged.entry(c.digest).or_default();
            match slot.iter_mut().find(|entry| entry.key == c.key) {
                Some(entry) => {
                    entry.count = entry.count.saturating_add(count);
                    entry.error = entry.error.saturating_add(c.error);
                    entry.paired = true;
                }
                None => slot.push(MergeEntry {
                    key: c.key.clone(),
                    digest: c.digest,
                    count: count.saturating_add(mine_min),
                    error: c.error.saturating_add(mine_min),
                    paired: true,
                }),
            }
        }

        let mut flat: Vec<MergeEntry> = merged.into_values().flatten().collect();
        for entry in &mut flat {
            if !entry.paired {
                entry.count = entry.count.saturating_add(theirs_min);
                entry.error = entry.error.saturating_add(theirs_min);
            }
        }
        flat.sort_unstable_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| key_order(&a.key).cmp(&key_order(&b.key)))
        });

        let mut discarded_max = mine_min.saturating_add(theirs_min);
        if flat.len() > self.capacity {
            discarded_max = discarded_max.max(flat[self.capacity].count);
            flat.truncate(self.capacity);
        }

        let total = self.total.saturating_add(other.total);
        self.clear();
        self.total = total;
        self.discarded_max = discarded_max;
        for entry in flat {
            self.seat(entry.digest, entry.key, entry.count, entry.error);
        }
    }

    // -- Stream-Summary internals -------------------------------------------

    fn find(&self, digest: u64, value: &DataInput) -> Option<usize> {
        self.index.get(&digest).and_then(|ids| {
            ids.iter()
                .copied()
                .find(|cid| self.monitored[*cid].key == *value)
        })
    }

    fn find_key(&self, digest: u64, key: &HeapItem) -> Option<usize> {
        self.index.get(&digest).and_then(|ids| {
            ids.iter()
                .copied()
                .find(|cid| self.monitored[*cid].key == *key)
        })
    }

    fn unindex(&mut self, digest: u64, cid: usize) {
        if let Some(ids) = self.index.get_mut(&digest) {
            ids.retain(|id| *id != cid);
            if ids.is_empty() {
                self.index.remove(&digest);
            }
        }
    }

    /// Takes a free counter for `key` at `count` with `error`.
    fn seat(&mut self, digest: u64, key: HeapItem, count: u64, error: u64) {
        let cid = self.monitored.len();
        self.monitored.push(MonitoredKey {
            key,
            digest,
            error,
            bucket: NIL,
            prev: NIL,
            next: NIL,
        });
        let bucket = self.bucket_for(NIL, count);
        self.attach(cid, bucket);
        self.index.entry(digest).or_default().push(cid);
    }

    /// Moves `cid` up by `count`, creating the destination bucket if needed.
    fn raise(&mut self, cid: usize, count: u64) {
        let from = self.monitored[cid].bucket;
        let target_count = self.buckets[from].count.saturating_add(count);
        if target_count == self.buckets[from].count {
            return;
        }
        let target = self.bucket_for(from, target_count);
        self.detach(cid);
        self.attach(cid, target);
    }

    /// Returns the bucket holding `count`, inserting one after `after` if no
    /// such bucket exists. `after` of `NIL` searches from the low end, and any
    /// other `after` holds a count below `count`.
    fn bucket_for(&mut self, after: usize, count: u64) -> usize {
        let mut prev = after;
        let mut next = if after == NIL {
            self.bucket_head
        } else {
            self.buckets[after].next
        };
        while next != NIL && self.buckets[next].count < count {
            prev = next;
            next = self.buckets[next].next;
        }
        if next != NIL && self.buckets[next].count == count {
            return next;
        }
        self.insert_bucket(prev, next, count)
    }

    fn insert_bucket(&mut self, prev: usize, next: usize, count: u64) -> usize {
        let bucket = Bucket {
            count,
            head: NIL,
            prev,
            next,
        };
        let bid = match self.bucket_free.pop() {
            Some(id) => {
                self.buckets[id] = bucket;
                id
            }
            None => {
                self.buckets.push(bucket);
                self.buckets.len() - 1
            }
        };
        if prev != NIL {
            self.buckets[prev].next = bid;
        } else {
            self.bucket_head = bid;
        }
        if next != NIL {
            self.buckets[next].prev = bid;
        } else {
            self.bucket_tail = bid;
        }
        bid
    }

    fn attach(&mut self, cid: usize, bid: usize) {
        let head = self.buckets[bid].head;
        self.monitored[cid].prev = NIL;
        self.monitored[cid].next = head;
        self.monitored[cid].bucket = bid;
        if head != NIL {
            self.monitored[head].prev = cid;
        }
        self.buckets[bid].head = cid;
    }

    fn detach(&mut self, cid: usize) {
        let bid = self.monitored[cid].bucket;
        let prev = self.monitored[cid].prev;
        let next = self.monitored[cid].next;
        if prev != NIL {
            self.monitored[prev].next = next;
        } else {
            self.buckets[bid].head = next;
        }
        if next != NIL {
            self.monitored[next].prev = prev;
        }
        self.monitored[cid].prev = NIL;
        self.monitored[cid].next = NIL;
        self.monitored[cid].bucket = NIL;
        if self.buckets[bid].head == NIL {
            self.drop_bucket(bid);
        }
    }

    /// Unlinks an emptied bucket into the free list. `raise` inserts the
    /// destination bucket before detaching, so `bid` is never the tail.
    fn drop_bucket(&mut self, bid: usize) {
        let prev = self.buckets[bid].prev;
        let next = self.buckets[bid].next;
        if prev != NIL {
            self.buckets[prev].next = next;
        } else {
            self.bucket_head = next;
        }
        debug_assert_ne!(next, NIL, "the tail bucket is never emptied");
        self.buckets[next].prev = prev;
        self.buckets[bid].head = NIL;
        self.buckets[bid].prev = NIL;
        self.buckets[bid].next = NIL;
        self.bucket_free.push(bid);
    }
}

/// One key of the union while a merge is in flight. `paired` marks an entry
/// that has already taken the other side's contribution.
struct MergeEntry {
    key: HeapItem,
    digest: u64,
    count: u64,
    error: u64,
    paired: bool,
}

/// A total order over keys, breaking the ties that counts leave open.
fn key_order(key: &HeapItem) -> (u8, u128, &[u8]) {
    match key {
        HeapItem::I8(v) => (0, *v as i128 as u128, b""),
        HeapItem::I16(v) => (1, *v as i128 as u128, b""),
        HeapItem::I32(v) => (2, *v as i128 as u128, b""),
        HeapItem::I64(v) => (3, *v as i128 as u128, b""),
        HeapItem::I128(v) => (4, *v as u128, b""),
        HeapItem::ISIZE(v) => (5, *v as i128 as u128, b""),
        HeapItem::U8(v) => (6, u128::from(*v), b""),
        HeapItem::U16(v) => (7, u128::from(*v), b""),
        HeapItem::U32(v) => (8, u128::from(*v), b""),
        HeapItem::U64(v) => (9, u128::from(*v), b""),
        HeapItem::U128(v) => (10, *v, b""),
        HeapItem::USIZE(v) => (11, *v as u128, b""),
        HeapItem::F32(v) => (12, u128::from(v.to_bits()), b""),
        HeapItem::F64(v) => (13, u128::from(v.to_bits()), b""),
        HeapItem::String(v) => (14, 0, v.as_bytes()),
        HeapItem::Bytes(v) => (15, 0, v.as_slice()),
    }
}

/// Serialized form: the summary as the triples it answers with, since the
/// bucket and counter lists and the key index all follow from them.
#[derive(Serialize)]
struct SpaceSavingRef<'a> {
    capacity: usize,
    total: u64,
    discarded_max: u64,
    entries: Vec<(&'a HeapItem, u64, u64)>,
}

#[derive(Deserialize)]
struct SpaceSavingState {
    capacity: usize,
    total: u64,
    discarded_max: u64,
    entries: Vec<(HeapItem, u64, u64)>,
}

impl<H: SketchHasher> Serialize for SpaceSaving<H> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SpaceSavingRef {
            capacity: self.capacity,
            total: self.total,
            discarded_max: self.discarded_max,
            entries: self
                .monitored
                .iter()
                .map(|c| (&c.key, self.buckets[c.bucket].count, c.error))
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de, H: SketchHasher> Deserialize<'de> for SpaceSaving<H> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let state = SpaceSavingState::deserialize(deserializer)?;
        Self::rebuild(state).map_err(serde::de::Error::custom)
    }
}

impl<H: SketchHasher> SpaceSaving<H> {
    /// Rebuilds a summary from decoded triples, rejecting any that no run of
    /// the algorithm could have produced: a duplicate or zero-count key, an
    /// error above its count, a ceiling above the lowest count, or a total
    /// under the weight the counters account for.
    fn rebuild(state: SpaceSavingState) -> Result<Self, String> {
        if state.capacity == 0 {
            return Err("space-saving capacity is zero".to_string());
        }
        if state.entries.len() > state.capacity {
            return Err(format!(
                "space-saving carries {} counters over a capacity of {}",
                state.entries.len(),
                state.capacity
            ));
        }
        for (_, count, error) in &state.entries {
            if *count == 0 {
                return Err("space-saving carries a counter at zero".to_string());
            }
            if *error > *count {
                return Err(format!(
                    "space-saving carries an error of {error} against a count of {count}"
                ));
            }
        }

        let mut entries = state.entries;
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.1));

        let mut summary = Self {
            capacity: state.capacity,
            monitored: Vec::with_capacity(entries.len()),
            buckets: Vec::new(),
            bucket_free: Vec::new(),
            bucket_head: NIL,
            bucket_tail: NIL,
            index: Index::with_capacity_and_hasher(entries.len(), DigestBuildHasher::default()),
            total: state.total,
            discarded_max: state.discarded_max,
            _hasher: PhantomData,
        };
        for (key, count, error) in entries {
            let digest = H::hash_item64_seeded(0, &key);
            if summary.find_key(digest, &key).is_some() {
                return Err("space-saving carries the same key twice".to_string());
            }
            summary.seat(digest, key, count, error);
        }

        let smallest = summary
            .monitored
            .iter()
            .map(|c| summary.buckets[c.bucket].count)
            .min()
            .unwrap_or(0);
        if summary.discarded_max > smallest {
            return Err(format!(
                "space-saving carries a ceiling of {} above its lowest count of {smallest}",
                summary.discarded_max
            ));
        }
        let recorded = summary
            .monitored
            .iter()
            .map(|c| summary.buckets[c.bucket].count.saturating_sub(c.error))
            .fold(0u64, u64::saturating_add);
        if recorded > summary.total {
            return Err(format!(
                "space-saving carries a total of {} under the {recorded} its counters account for",
                summary.total
            ));
        }
        Ok(summary)
    }
}

#[cfg(test)]
impl<H: SketchHasher> SpaceSaving<H> {
    /// Checks every Stream-Summary invariant: both directions of both linked
    /// lists, strict count ordering, arena bookkeeping and index agreement.
    fn validate(&self) -> Result<(), String> {
        if self.monitored.len() > self.capacity {
            return Err(format!(
                "{} counters over a capacity of {}",
                self.monitored.len(),
                self.capacity
            ));
        }
        if self.monitored.is_empty() != (self.bucket_head == NIL) {
            return Err("the bucket list disagrees with counter residency".to_string());
        }

        let mut live: Vec<usize> = Vec::new();
        let mut previous = NIL;
        let mut bid = self.bucket_head;
        while bid != NIL {
            if bid >= self.buckets.len() {
                return Err(format!("bucket {bid} is outside the arena"));
            }
            if live.len() > self.buckets.len() {
                return Err("the bucket list cycles".to_string());
            }
            let bucket = &self.buckets[bid];
            if bucket.prev != previous {
                return Err(format!("bucket {bid} does not point back at {previous}"));
            }
            if bucket.head == NIL {
                return Err(format!("live bucket {bid} holds no counter"));
            }
            if bucket.count == 0 {
                return Err(format!("live bucket {bid} sits at zero"));
            }
            if let Some(lower) = live.last()
                && self.buckets[*lower].count >= bucket.count
            {
                return Err("the bucket counts are not strictly increasing".to_string());
            }
            live.push(bid);
            previous = bid;
            bid = bucket.next;
        }
        if previous != self.bucket_tail {
            return Err("the bucket list does not end at the tail".to_string());
        }

        let mut backwards: Vec<usize> = Vec::new();
        let mut bid = self.bucket_tail;
        while bid != NIL {
            if backwards.len() > self.buckets.len() {
                return Err("the bucket list cycles backwards".to_string());
            }
            backwards.push(bid);
            bid = self.buckets[bid].prev;
        }
        backwards.reverse();
        if backwards != live {
            return Err("the bucket list reads differently in each direction".to_string());
        }

        let mut seen = vec![false; self.monitored.len()];
        for bid in &live {
            let count = self.buckets[*bid].count;
            let mut chain: Vec<usize> = Vec::new();
            let mut previous = NIL;
            let mut cid = self.buckets[*bid].head;
            while cid != NIL {
                if cid >= self.monitored.len() {
                    return Err(format!("counter {cid} is outside the arena"));
                }
                if seen[cid] {
                    return Err(format!("counter {cid} is reached twice"));
                }
                let counter = &self.monitored[cid];
                if counter.prev != previous {
                    return Err(format!("counter {cid} does not point back at {previous}"));
                }
                if counter.bucket != *bid {
                    return Err(format!("counter {cid} points at bucket {}", counter.bucket));
                }
                if counter.error > count {
                    return Err(format!(
                        "counter {cid} carries an error of {} against a count of {count}",
                        counter.error
                    ));
                }
                seen[cid] = true;
                chain.push(cid);
                previous = cid;
                cid = counter.next;
            }

            let mut backwards: Vec<usize> = Vec::new();
            let mut cid = previous;
            while cid != NIL {
                if backwards.len() > chain.len() {
                    return Err(format!("bucket {bid}'s counter list cycles backwards"));
                }
                backwards.push(cid);
                cid = self.monitored[cid].prev;
            }
            backwards.reverse();
            if backwards != chain {
                return Err(format!(
                    "bucket {bid}'s counter list reads differently in each direction"
                ));
            }
        }
        if let Some(cid) = seen.iter().position(|reached| !reached) {
            return Err(format!("counter {cid} hangs off no bucket"));
        }

        let mut free = vec![false; self.buckets.len()];
        for bid in &self.bucket_free {
            if *bid >= self.buckets.len() {
                return Err(format!("free bucket {bid} is outside the arena"));
            }
            if free[*bid] {
                return Err(format!("bucket {bid} is freed twice"));
            }
            free[*bid] = true;
        }
        for bid in &live {
            if free[*bid] {
                return Err(format!("bucket {bid} is both live and free"));
            }
        }
        if live.len() + self.bucket_free.len() != self.buckets.len() {
            return Err(format!(
                "{} live and {} free buckets in an arena of {}",
                live.len(),
                self.bucket_free.len(),
                self.buckets.len()
            ));
        }

        let mut indexed = vec![false; self.monitored.len()];
        for (digest, slot) in &self.index {
            if slot.is_empty() {
                return Err(format!("digest {digest} indexes nothing"));
            }
            for cid in slot {
                if *cid >= self.monitored.len() {
                    return Err(format!("digest {digest} indexes counter {cid}"));
                }
                if indexed[*cid] {
                    return Err(format!("counter {cid} is indexed twice"));
                }
                if self.monitored[*cid].digest != *digest {
                    return Err(format!("counter {cid} is filed under the wrong digest"));
                }
                indexed[*cid] = true;
            }
        }
        if let Some(cid) = indexed.iter().position(|filed| !filed) {
            return Err(format!("counter {cid} is not indexed"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_of(item: &HeapItem) -> i64 {
        match item {
            HeapItem::I64(v) => *v,
            other => panic!("unexpected key form {other:?}"),
        }
    }

    fn walk(summary: &SpaceSaving) -> Vec<(i64, u64)> {
        summary
            .top_k(usize::MAX)
            .iter()
            .map(|(key, count, _)| (key_of(key), *count))
            .collect()
    }

    fn next_random(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Every key the summary reports must bracket its truth, and every key it
    /// does not must sit under the ceiling.
    fn assert_sound_against(summary: &SpaceSaving, truth: &HashMap<i64, u64>) {
        for (key, count) in truth {
            let probe = DataInput::I64(*key);
            assert!(
                summary.upper_bound(&probe) >= *count,
                "key {key} has true count {count} above the {} ceiling",
                summary.upper_bound(&probe)
            );
            let estimate = summary.estimate(&probe);
            if estimate == 0 {
                continue;
            }
            assert!(
                estimate >= *count,
                "monitored key {key} reads {estimate} against a truth of {count}"
            );
            assert!(
                estimate - summary.error(&probe) <= *count,
                "monitored key {key} reads {estimate} with too small an error for {count}"
            );
        }
    }

    fn fuzzed(
        capacity: usize,
        domain: i64,
        steps: usize,
        seed: u64,
    ) -> (SpaceSaving, HashMap<i64, u64>) {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(capacity);
        let mut truth: HashMap<i64, u64> = HashMap::new();
        let mut state = seed;
        for step in 0..steps {
            let draw = next_random(&mut state);
            let key = (draw % domain as u64) as i64;
            let weight = match (draw >> 40) % 8 {
                0..=4 => 1,
                5 => 3,
                6 => 11,
                _ => 97,
            };
            summary.insert_many(&DataInput::I64(key), weight);
            *truth.entry(key).or_default() += weight;
            if let Err(problem) = summary.validate() {
                panic!("capacity {capacity} step {step}: {problem}");
            }
        }
        (summary, truth)
    }

    #[test]
    fn a_fresh_summary_is_well_formed() {
        let summary: SpaceSaving = SpaceSaving::with_capacity(4);
        summary.validate().expect("empty summary");
        assert_eq!(summary.min_count(), 0);
        assert_eq!(summary.capacity(), 4);
    }

    #[test]
    fn a_capacity_of_zero_floors_at_one() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(0);
        assert_eq!(summary.capacity(), 1);
        summary.insert(&DataInput::I64(1));
        summary.insert(&DataInput::I64(2));
        summary.validate().expect("single counter");
        assert_eq!(summary.len(), 1);
        assert_eq!(summary.estimate(&DataInput::I64(2)), 2);
    }

    #[test]
    fn a_weighted_arrival_displaces_the_minimum_and_starts_above_it() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(2);
        for _ in 0..5 {
            summary.insert(&DataInput::I64(1));
        }
        for _ in 0..2 {
            summary.insert(&DataInput::I64(2));
        }

        summary.insert_many(&DataInput::I64(3), 4);

        summary.validate().expect("after a weighted eviction");
        assert_eq!(summary.len(), 2);
        assert_eq!(summary.estimate(&DataInput::I64(2)), 0);
        assert_eq!(summary.estimate(&DataInput::I64(3)), 6);
        assert_eq!(summary.error(&DataInput::I64(3)), 2);
        assert_eq!(summary.estimate(&DataInput::I64(1)), 5);
        assert_eq!(summary.min_count(), 5);
        assert_eq!(summary.total(), 11);
    }

    #[test]
    fn a_weighted_raise_passes_every_bucket_below_its_destination() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(4);
        for (key, count) in [(1i64, 1u64), (2, 2), (3, 3), (4, 4)] {
            summary.insert_many(&DataInput::I64(key), count);
        }
        assert_eq!(walk(&summary), vec![(4, 4), (3, 3), (2, 2), (1, 1)]);

        summary.insert_many(&DataInput::I64(1), 9);

        summary.validate().expect("after a multi-hop raise");
        assert_eq!(walk(&summary), vec![(1, 10), (4, 4), (3, 3), (2, 2)]);
    }

    #[test]
    fn counts_saturate_and_keep_the_bucket_order() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(3);
        summary.insert_many(&DataInput::I64(3), 7);
        summary.insert_many(&DataInput::I64(1), u64::MAX - 2);
        summary.insert_many(&DataInput::I64(2), u64::MAX);
        summary.insert_many(&DataInput::I64(1), 10);
        summary.validate().expect("after a saturating raise");

        summary.insert_many(&DataInput::I64(1), 5);
        summary
            .validate()
            .expect("after raising a saturated counter");

        assert_eq!(summary.estimate(&DataInput::I64(1)), u64::MAX);
        assert_eq!(summary.estimate(&DataInput::I64(2)), u64::MAX);
        assert_eq!(summary.estimate(&DataInput::I64(3)), 7);
        assert_eq!(summary.total(), u64::MAX);
        let walked = walk(&summary);
        assert_eq!(walked.len(), 3);
        for pair in walked.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "the walk is out of order");
        }
        assert_eq!(
            walked[2],
            (3, 7),
            "the small counter must be walked last, not first"
        );
    }

    #[test]
    fn an_eviction_from_a_saturated_counter_stays_sound() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(1);
        summary.insert_many(&DataInput::I64(1), u64::MAX);
        summary.insert(&DataInput::I64(2));
        summary
            .validate()
            .expect("after evicting a saturated counter");

        assert_eq!(summary.len(), 1);
        assert_eq!(summary.estimate(&DataInput::I64(2)), u64::MAX);
        assert_eq!(summary.error(&DataInput::I64(2)), u64::MAX);
        assert_eq!(summary.upper_bound(&DataInput::I64(1)), u64::MAX);
    }

    #[test]
    fn a_merge_saturates_instead_of_wrapping() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(2);
        left.insert_many(&DataInput::I64(1), u64::MAX);
        left.insert_many(&DataInput::I64(2), u64::MAX - 1);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(3);
        right.insert_many(&DataInput::I64(1), u64::MAX / 2);
        right.insert_many(&DataInput::I64(3), 100);
        right.insert_many(&DataInput::I64(4), 50);

        left.merge(&right);

        left.validate().expect("after a saturating merge");
        assert_eq!(left.len(), 2);
        assert_eq!(left.total(), u64::MAX);
        for (key, count) in walk(&left) {
            assert_eq!(count, u64::MAX, "key {key} wrapped past the ceiling");
        }
        for key in 1..=4i64 {
            assert_eq!(left.upper_bound(&DataInput::I64(key)), u64::MAX);
        }
    }

    /// A summary with counters to spare whose ceiling is `ceiling`, built by
    /// merging in a single counter that displaced a count that large. `dropped`
    /// truly reached `ceiling - 1`; `held` arrived once.
    fn ceilinged(capacity: usize, ceiling: u64, held: i64, dropped: i64) -> SpaceSaving {
        let mut source: SpaceSaving = SpaceSaving::with_capacity(1);
        source.insert_many(&DataInput::I64(dropped), ceiling - 1);
        source.insert(&DataInput::I64(held));
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(capacity);
        summary.merge(&source);
        assert_eq!(summary.min_count(), ceiling, "the fixture ceiling");
        summary
    }

    /// Left holds key 2 at `u64::MAX` with an error of `u64::MAX`; right holds
    /// the same key at 6 with an error of 5. Key 2 truly arrived twice.
    fn a_saturated_overlap() -> (SpaceSaving, SpaceSaving) {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(1);
        left.insert_many(&DataInput::I64(1), u64::MAX);
        left.insert(&DataInput::I64(2));
        let mut right: SpaceSaving = SpaceSaving::with_capacity(1);
        right.insert_many(&DataInput::I64(3), 5);
        right.insert(&DataInput::I64(2));
        assert_eq!(left.estimate(&DataInput::I64(2)), u64::MAX);
        assert_eq!(right.estimate(&DataInput::I64(2)), 6);
        (left, right)
    }

    /// A key seated above an already enormous ceiling stops at `u64::MAX`
    /// rather than wrapping under its own error.
    #[test]
    fn a_seat_above_the_ceiling_saturates() {
        let mut summary = ceilinged(4, u64::MAX - 3, 8, 9);

        summary.insert_many(&DataInput::I64(5), 10);

        summary.validate().expect("after seating above the ceiling");
        assert_eq!(summary.estimate(&DataInput::I64(5)), u64::MAX);
        assert_eq!(summary.error(&DataInput::I64(5)), u64::MAX - 3);
        assert_eq!(summary.estimate(&DataInput::I64(8)), u64::MAX - 3);
    }

    /// A shared key's counts add to `u64::MAX` rather than wrapping to a count
    /// below the error it carries.
    #[test]
    fn a_merge_saturates_a_shared_keys_count() {
        let (mut left, right) = a_saturated_overlap();

        left.merge(&right);

        left.validate().expect("after a saturating merge");
        assert_eq!(left.len(), 1);
        assert_eq!(
            left.estimate(&DataInput::I64(2)),
            u64::MAX,
            "key 2's paired count wrapped"
        );
    }

    /// A shared key's errors add to `u64::MAX` rather than wrapping to a small
    /// allowance, which would claim the key is nearly as large as its count.
    #[test]
    fn a_merge_saturates_a_shared_keys_error() {
        let (mut left, right) = a_saturated_overlap();

        left.merge(&right);

        left.validate().expect("after a saturating merge");
        assert_eq!(
            left.error(&DataInput::I64(2)),
            u64::MAX,
            "key 2's paired error wrapped"
        );
        let probe = DataInput::I64(2);
        assert!(
            left.estimate(&probe) - left.error(&probe) <= 2,
            "key 2 arrived twice, but the merge claims at least {}",
            left.estimate(&probe) - left.error(&probe)
        );
    }

    /// A key only the other side holds picks up this side's ceiling and stops
    /// at `u64::MAX`.
    #[test]
    fn a_merge_saturates_a_key_only_the_other_side_holds() {
        let mut left = ceilinged(4, u64::MAX - 3, 8, 9);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(4);
        right.insert_many(&DataInput::I64(3), 7);

        left.merge(&right);

        left.validate().expect("after a saturating merge");
        assert_eq!(
            left.estimate(&DataInput::I64(3)),
            u64::MAX,
            "key 3's count wrapped past the ceiling it inherited"
        );
        assert_eq!(left.error(&DataInput::I64(3)), u64::MAX - 3);
        assert_eq!(left.estimate(&DataInput::I64(8)), u64::MAX - 3);
    }

    /// A key only this side holds picks up the other side's ceiling and stops
    /// at `u64::MAX`.
    #[test]
    fn a_merge_saturates_a_key_only_this_side_holds() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(4);
        left.insert_many(&DataInput::I64(3), 7);
        let right = ceilinged(4, u64::MAX - 3, 8, 9);

        left.merge(&right);

        left.validate().expect("after a saturating merge");
        assert_eq!(
            left.estimate(&DataInput::I64(3)),
            u64::MAX,
            "key 3's count wrapped past the ceiling it inherited"
        );
        assert_eq!(left.error(&DataInput::I64(3)), u64::MAX - 3);
        assert_eq!(left.estimate(&DataInput::I64(8)), u64::MAX - 3);
    }

    /// The ceiling a merge leaves behind saturates: wrapping it would report a
    /// tiny bound for a key that truly reached nearly `u64::MAX`.
    #[test]
    fn a_merge_saturates_the_ceiling() {
        let mut left = ceilinged(4, u64::MAX - 3, 8, 9);
        let right = ceilinged(4, 10, 5, 6);

        left.merge(&right);

        left.validate().expect("after a saturating merge");
        assert_eq!(left.min_count(), u64::MAX, "the merged ceiling wrapped");
        assert!(
            left.upper_bound(&DataInput::I64(9)) >= u64::MAX - 4,
            "key 9 truly reached {} but is capped at {}",
            u64::MAX - 4,
            left.upper_bound(&DataInput::I64(9))
        );
    }

    /// The ceiling a merge leaves behind lives only in `discarded_max`; an under-full
    /// summary reads it from no counter, so a serde round trip that dropped it
    /// would answer below the truth.
    #[test]
    fn a_serde_round_trip_carries_a_ceiling_no_counter_holds() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(33);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(1);
        for _ in 0..10 {
            right.insert(&DataInput::I64(7));
        }
        for _ in 0..20 {
            right.insert(&DataInput::I64(8));
        }
        left.merge(&right);
        assert!(left.len() < left.capacity(), "the merge left room to spare");
        assert_eq!(left.min_count(), 30);

        let bytes = rmp_serde::to_vec(&left).expect("serialize");
        let decoded: SpaceSaving = rmp_serde::from_slice(&bytes).expect("deserialize");

        decoded.validate().expect("decoded summary");
        assert_eq!(
            decoded.min_count(),
            30,
            "the merged ceiling was not written"
        );
        assert!(
            decoded.upper_bound(&DataInput::I64(7)) >= 10,
            "key 7 truly reached 10 but decodes capped at {}",
            decoded.upper_bound(&DataInput::I64(7))
        );
    }

    /// `is_guaranteed` is strict: a key whose count less its error only reaches
    /// the ceiling ties whatever the summary may have dropped.
    #[test]
    fn a_key_that_only_ties_the_ceiling_is_not_guaranteed() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(2);
        for _ in 0..3 {
            summary.insert(&DataInput::I64(1));
        }
        for _ in 0..3 {
            summary.insert(&DataInput::I64(2));
        }
        summary.insert(&DataInput::I64(3));

        summary.validate().expect("after the eviction");
        assert_eq!(summary.estimate(&DataInput::I64(1)), 3);
        assert_eq!(summary.error(&DataInput::I64(1)), 0);
        assert_eq!(summary.min_count(), 3);
        assert!(
            !summary.is_guaranteed(&DataInput::I64(1)),
            "key 1 only ties the 3 key 2 reached before it was dropped"
        );
        assert!(!summary.is_guaranteed(&DataInput::I64(3)));

        for _ in 0..2 {
            summary.insert(&DataInput::I64(1));
        }
        assert_eq!(summary.estimate(&DataInput::I64(1)), 5);
        assert_eq!(summary.min_count(), 4);
        assert!(summary.is_guaranteed(&DataInput::I64(1)));
    }

    /// `clear` leaves a summary that answers like a fresh one, ceiling and
    /// total included.
    #[test]
    fn clear_resets_every_answer() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(2);
        for _ in 0..5 {
            summary.insert(&DataInput::I64(1));
        }
        for _ in 0..3 {
            summary.insert(&DataInput::I64(2));
        }
        summary.insert(&DataInput::I64(3));
        assert!(summary.min_count() > 0);
        assert!(summary.total() > 0);

        summary.clear();

        summary.validate().expect("cleared summary");
        assert!(summary.is_empty());
        assert_eq!(summary.len(), 0);
        assert_eq!(summary.capacity(), 2);
        assert_eq!(summary.total(), 0, "clear left the recorded weight behind");
        assert_eq!(summary.min_count(), 0, "clear left the ceiling behind");
        assert_eq!(summary.upper_bound(&DataInput::I64(1)), 0);
        assert_eq!(summary.error(&DataInput::I64(1)), 0);
        assert!(summary.top_k(4).is_empty());
        assert!(summary.entries().is_empty());

        summary.insert(&DataInput::I64(4));
        summary.validate().expect("after refilling");
        assert_eq!(summary.estimate(&DataInput::I64(4)), 1);
        assert_eq!(summary.total(), 1);
    }

    #[test]
    fn a_merge_carries_the_ceiling_into_an_under_full_summary() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(33);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(1);
        for _ in 0..10 {
            right.insert(&DataInput::I64(7));
        }
        for _ in 0..20 {
            right.insert(&DataInput::I64(8));
        }

        left.merge(&right);

        left.validate()
            .expect("after merging into an empty summary");
        assert_eq!(left.len(), 1);
        assert!(left.len() < left.capacity(), "the merge left room to spare");
        assert!(
            left.min_count() >= 10,
            "key 7 truly reached 10 but the ceiling is {}",
            left.min_count()
        );
        assert!(left.upper_bound(&DataInput::I64(7)) >= 10);
        assert!(
            !left.is_guaranteed(&DataInput::I64(8)),
            "nothing outranks a ceiling it does not clear"
        );
    }

    /// A count tie straddling the capacity boundary is cut by `key_order`
    /// alone, so a truncating merge keeps the keys the encoder emits first and
    /// the digests do not choose the survivors.
    #[test]
    fn a_truncating_merge_keeps_the_keys_the_encoder_emits_first() {
        fn left_side() -> SpaceSaving {
            let mut left: SpaceSaving = SpaceSaving::with_capacity(4);
            left.insert_many(&DataInput::I64(1), 9);
            for key in [10i64, 20, 30] {
                left.insert_many(&DataInput::I64(key), 3);
            }
            left
        }
        fn right_side() -> SpaceSaving {
            let mut right: SpaceSaving = SpaceSaving::with_capacity(3);
            for key in [40i64, 50, 60] {
                right.insert_many(&DataInput::I64(key), 2);
            }
            right
        }
        fn keys_of(summary: &SpaceSaving) -> Vec<i64> {
            summary
                .entries()
                .iter()
                .map(|(key, _, _)| match key {
                    HeapItem::I64(v) => *v,
                    other => panic!("unexpected key form {other:?}"),
                })
                .collect()
        }
        fn sorted_keys(summary: &SpaceSaving) -> Vec<i64> {
            let mut keys = keys_of(summary);
            keys.sort_unstable();
            keys
        }

        let mut merged = left_side();
        merged.merge(&right_side());
        merged.validate().expect("after a truncating merge");
        let mut swapped = right_side();
        swapped.merge(&left_side());
        swapped.validate().expect("after the swapped merge");

        for key in [10i64, 20, 30, 40, 50, 60] {
            assert_eq!(
                merged.upper_bound(&DataInput::I64(key)),
                5,
                "key {key} did not join the tie at the capacity boundary"
            );
        }
        assert_eq!(
            sorted_keys(&merged),
            vec![1, 10, 20, 30],
            "the tie was cut somewhere other than the key order"
        );
        assert_eq!(
            sorted_keys(&swapped),
            vec![1, 10, 20],
            "a narrower cut of the same tie did not follow the key order"
        );
        for dropped in [40i64, 50, 60] {
            assert!(
                key_order(&HeapItem::I64(30)) < key_order(&HeapItem::I64(dropped)),
                "key {dropped} was dropped although it sorts before the last survivor"
            );
        }

        let bytes = merged
            .serialize_to_bytes()
            .expect("serialize the survivors");
        let emitted: SpaceSaving =
            SpaceSaving::deserialize_from_bytes(&bytes).expect("deserialize the survivors");
        assert_eq!(
            keys_of(&emitted),
            vec![1, 10, 20, 30],
            "the encoder emits an order the merge did not keep"
        );
    }

    #[test]
    fn a_chain_of_merges_keeps_the_ceiling_above_everything_dropped() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(5);
        let mut middle: SpaceSaving = SpaceSaving::with_capacity(1);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(1);
        for _ in 0..10 {
            middle.insert(&DataInput::I64(7));
        }
        for _ in 0..20 {
            middle.insert(&DataInput::I64(8));
        }
        for _ in 0..5 {
            right.insert(&DataInput::I64(9));
        }
        for _ in 0..7 {
            right.insert(&DataInput::I64(10));
        }

        left.merge(&middle);
        left.validate().expect("after the first merge");
        left.merge(&right);
        left.validate().expect("after the second merge");

        assert!(left.len() < left.capacity(), "the chain left room to spare");
        for (key, truth) in [(7i64, 10u64), (9, 5)] {
            assert!(
                left.upper_bound(&DataInput::I64(key)) >= truth,
                "key {key} truly reached {truth} but is capped at {}",
                left.upper_bound(&DataInput::I64(key))
            );
        }
        assert!(left.min_count() >= 15, "the two ceilings did not compound");
    }

    #[test]
    fn a_key_that_re_enters_after_a_merge_never_reads_low() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(8);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(1);
        for _ in 0..12 {
            right.insert(&DataInput::I64(7));
        }
        for _ in 0..30 {
            right.insert(&DataInput::I64(8));
        }
        left.merge(&right);

        let ceiling = left.min_count();
        left.insert(&DataInput::I64(7));

        left.validate().expect("after re-entry");
        let estimate = left.estimate(&DataInput::I64(7));
        assert!(
            estimate >= 13,
            "key 7 truly reached 13 but reads {estimate} on re-entry"
        );
        assert_eq!(estimate, ceiling + 1);
        assert_eq!(left.error(&DataInput::I64(7)), ceiling);
    }

    #[test]
    fn randomized_operations_keep_the_structure_sound() {
        for (capacity, seed) in [(1usize, 11u64), (2, 22), (7, 33), (64, 44), (257, 55)] {
            let (summary, truth) = fuzzed(capacity, 96, 4_000, seed);
            assert_sound_against(&summary, &truth);
            assert_eq!(summary.len(), capacity.min(truth.len()));
            assert_eq!(summary.total(), truth.values().sum::<u64>());
            let walked = walk(&summary);
            assert_eq!(walked.len(), summary.len());
            for pair in walked.windows(2) {
                assert!(pair[0].1 >= pair[1].1, "capacity {capacity}: walk order");
            }
        }
    }

    #[test]
    fn randomized_merges_keep_the_structure_sound() {
        for (capacity, domain, seed) in [
            (1usize, 64i64, 101u64),
            (3, 64, 202),
            (32, 64, 303),
            (128, 64, 404),
            (256, 40, 505),
        ] {
            let (mut left, mut truth) = fuzzed(capacity, domain, 900, seed);
            let (right, right_truth) = fuzzed(2, 80, 700, seed ^ 0xabcd);
            let (third, third_truth) = fuzzed(capacity + 5, 70, 500, seed ^ 0x1234);

            left.merge(&right);
            left.validate().expect("after the first merge");
            left.merge(&third);
            left.validate().expect("after the second merge");

            for (key, count) in right_truth.iter().chain(third_truth.iter()) {
                *truth.entry(*key).or_default() += *count;
            }
            assert_sound_against(&left, &truth);
            assert_eq!(left.total(), truth.values().sum::<u64>());
            assert!(left.len() <= left.capacity());

            let mut state = seed;
            for _ in 0..200 {
                let key = (next_random(&mut state) % 90) as i64;
                left.insert(&DataInput::I64(key));
                *truth.entry(key).or_default() += 1;
            }
            left.validate()
                .expect("after inserting into a merged summary");
            assert_sound_against(&left, &truth);
        }
    }

    #[test]
    fn a_decoded_summary_rebuilds_both_link_directions() {
        let (summary, truth) = fuzzed(48, 200, 3_000, 77);
        let bytes = rmp_serde::to_vec(&summary).expect("serialize");
        let decoded: SpaceSaving = rmp_serde::from_slice(&bytes).expect("deserialize");

        decoded.validate().expect("decoded summary");
        assert_eq!(decoded.len(), summary.len());
        assert_eq!(decoded.min_count(), summary.min_count());
        assert_eq!(decoded.total(), summary.total());
        assert_sound_against(&decoded, &truth);

        let mut walked = walk(&decoded);
        let mut expected = walk(&summary);
        walked.sort_unstable();
        expected.sort_unstable();
        assert_eq!(walked, expected);
    }

    #[test]
    fn a_crafted_state_fails_closed() {
        let over_capacity = SpaceSavingState {
            capacity: 1,
            total: 4,
            discarded_max: 0,
            entries: vec![(HeapItem::I64(1), 2, 0), (HeapItem::I64(2), 2, 0)],
        };
        let cases = [
            (
                SpaceSavingState {
                    capacity: 0,
                    total: 0,
                    discarded_max: 0,
                    entries: Vec::new(),
                },
                "capacity is zero",
            ),
            (over_capacity, "over a capacity"),
            (
                SpaceSavingState {
                    capacity: 4,
                    total: 1,
                    discarded_max: 0,
                    entries: vec![(HeapItem::I64(1), 0, 0)],
                },
                "at zero",
            ),
            (
                SpaceSavingState {
                    capacity: 4,
                    total: 1,
                    discarded_max: 0,
                    entries: vec![(HeapItem::I64(1), 3, 4)],
                },
                "error of 4",
            ),
            (
                SpaceSavingState {
                    capacity: 4,
                    total: 2,
                    discarded_max: 0,
                    entries: vec![(HeapItem::I64(1), 2, 0), (HeapItem::I64(1), 1, 0)],
                },
                "same key twice",
            ),
            (
                SpaceSavingState {
                    capacity: 4,
                    total: 3,
                    discarded_max: u64::MAX,
                    entries: vec![(HeapItem::I64(1), 3, 0)],
                },
                "ceiling of 18446744073709551615 above its lowest count of 3",
            ),
            (
                SpaceSavingState {
                    capacity: 4,
                    total: 0,
                    discarded_max: 0,
                    entries: vec![(HeapItem::I64(1), 9, 0)],
                },
                "total of 0 under the 9",
            ),
        ];
        for (state, expected) in cases {
            let problem = SpaceSaving::<DefaultXxHasher>::rebuild(state)
                .expect_err("a crafted state must be rejected");
            assert!(
                problem.contains(expected),
                "expected a complaint about {expected}, got {problem}"
            );
        }
    }

    #[test]
    fn a_declared_capacity_is_not_allocated_on_decode() {
        let state = SpaceSavingState {
            capacity: 1 << 40,
            total: 3,
            discarded_max: 0,
            entries: vec![(HeapItem::I64(1), 3, 0)],
        };
        let summary = SpaceSaving::<DefaultXxHasher>::rebuild(state).expect("a sparse state");
        summary.validate().expect("decoded summary");
        assert_eq!(summary.capacity(), 1 << 40);
        assert_eq!(summary.len(), 1);
    }

    // -- Byte-array keys ----------------------------------------------------

    /// Bytes that are not UTF-8 at all, so a summary that stored them as a
    /// `String` could not have held them.
    const RAW: &[u8] = &[0xff, 0x00, 0xfe];

    fn bytes_of(summary: &SpaceSaving) -> Vec<Vec<u8>> {
        let mut keys: Vec<Vec<u8>> = summary
            .entries()
            .iter()
            .map(|(key, _, _)| match key {
                HeapItem::Bytes(v) => v.clone(),
                other => panic!("unexpected key form {other:?}"),
            })
            .collect();
        keys.sort_unstable();
        keys
    }

    /// A byte array that is not UTF-8 is a key like any other: it seats, it
    /// answers, and repeats of it land on the counter already holding it.
    #[test]
    fn a_non_utf8_byte_key_is_monitored_and_queried() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(4);
        for _ in 0..3 {
            summary.insert(&DataInput::Bytes(RAW));
        }
        summary.insert_many(&DataInput::Bytes(&[0x00, 0x80]), 2);

        summary.validate().expect("two byte keys");
        assert_eq!(summary.len(), 2, "a repeat took a second counter");
        assert_eq!(bytes_of(&summary), vec![vec![0x00, 0x80], RAW.to_vec()]);
        assert_eq!(summary.estimate(&DataInput::Bytes(RAW)), 3);
        assert_eq!(summary.estimate(&DataInput::Bytes(&[0x00, 0x80])), 2);
        assert_eq!(summary.estimate(&DataInput::Bytes(&[0x01])), 0);
    }

    /// `Bytes(b"abc")` and `Str("abc")` are two keys. They may share a digest,
    /// so the counter that answers is settled by the full key.
    #[test]
    fn a_byte_key_and_a_string_key_are_separate_counters() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(4);
        summary.insert_many(&DataInput::Bytes(b"abc"), 5);
        summary.insert_many(&DataInput::Str("abc"), 2);

        summary.validate().expect("a byte key beside a string key");
        assert_eq!(summary.len(), 2, "the two keys shared a counter");
        assert_eq!(summary.estimate(&DataInput::Bytes(b"abc")), 5);
        assert_eq!(summary.estimate(&DataInput::Str("abc")), 2);
        assert_eq!(summary.estimate(&DataInput::String("abc".to_string())), 2);
    }

    /// An evicted byte key still reads under the ceiling, and the key that took
    /// its counter carries the displaced count as its error.
    #[test]
    fn an_evicted_byte_key_keeps_its_bound() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(2);
        for _ in 0..3 {
            summary.insert(&DataInput::Bytes(RAW));
        }
        for _ in 0..2 {
            summary.insert(&DataInput::Bytes(b"\x00mid"));
        }
        summary.insert(&DataInput::Bytes(&[0xfd]));

        summary.validate().expect("after a byte-key eviction");
        assert_eq!(summary.len(), 2);
        assert_eq!(summary.estimate(&DataInput::Bytes(b"\x00mid")), 0);
        assert_eq!(summary.min_count(), 3);
        assert!(
            summary.upper_bound(&DataInput::Bytes(b"\x00mid")) >= 2,
            "the evicted byte key truly reached 2"
        );
        assert_eq!(summary.estimate(&DataInput::Bytes(&[0xfd])), 3);
        assert_eq!(summary.error(&DataInput::Bytes(&[0xfd])), 2);
    }

    /// A merge pairs byte keys by their bytes: the shared one adds, and the one
    /// only the other side holds arrives with that side's ceiling.
    #[test]
    fn a_merge_pairs_byte_keys_by_their_bytes() {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(4);
        left.insert_many(&DataInput::Bytes(RAW), 5);
        left.insert_many(&DataInput::Bytes(&[0x01]), 3);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(4);
        right.insert_many(&DataInput::Bytes(RAW), 2);
        right.insert_many(&DataInput::Bytes(&[0x02]), 7);

        left.merge(&right);

        left.validate().expect("after a byte-key merge");
        assert_eq!(left.len(), 3);
        assert_eq!(
            bytes_of(&left),
            vec![vec![0x01], vec![0x02], RAW.to_vec()],
            "the byte keys did not survive the merge"
        );
        assert_eq!(left.estimate(&DataInput::Bytes(RAW)), 7, "the shared key");
        assert_eq!(left.estimate(&DataInput::Bytes(&[0x01])), 3);
        assert_eq!(left.estimate(&DataInput::Bytes(&[0x02])), 7);
    }

    /// A serde round trip carries the raw bytes, and the decoded summary
    /// answers the same `DataInput::Bytes` the original did — which it can only
    /// do if the rebuild hashed the stored key to the digest a query reaches.
    #[test]
    fn a_serde_round_trip_keeps_a_byte_key() {
        let mut summary: SpaceSaving = SpaceSaving::with_capacity(4);
        summary.insert_many(&DataInput::Bytes(RAW), 9);
        summary.insert_many(&DataInput::Bytes(&[0x00; 5]), 4);

        let bytes = rmp_serde::to_vec(&summary).expect("serialize");
        let decoded: SpaceSaving = rmp_serde::from_slice(&bytes).expect("deserialize");

        decoded.validate().expect("decoded summary");
        assert_eq!(bytes_of(&decoded), bytes_of(&summary));
        assert_eq!(decoded.estimate(&DataInput::Bytes(RAW)), 9);
        assert_eq!(decoded.estimate(&DataInput::Bytes(&[0x00; 5])), 4);
        assert_eq!(decoded.total(), summary.total());
    }
}

/// The collision guards, which a 64-bit digest makes unreachable in practice:
/// every key here hashes to one digest, so the key index holds a single slot
/// and identity is decided by the key alone.
#[cfg(test)]
mod collisions {
    use super::*;

    /// Files every key under one digest.
    #[derive(Clone, Debug)]
    struct OneDigest;

    const ONE: u64 = 0x5151_5151_5151_5151;

    impl SketchHasher for OneDigest {
        type HashType = <DefaultXxHasher as SketchHasher>::HashType;

        fn hash64_seeded(_: usize, _: &DataInput) -> u64 {
            ONE
        }
        fn hash128_seeded(_: usize, _: &DataInput) -> u128 {
            u128::from(ONE)
        }
        fn hash_item64_seeded(_: usize, _: &HeapItem) -> u64 {
            ONE
        }
        fn hash_item128_seeded(_: usize, _: &HeapItem) -> u128 {
            u128::from(ONE)
        }
        fn hash_for_matrix_seeded(
            seed_idx: usize,
            rows: usize,
            cols: usize,
            key: &DataInput,
        ) -> Self::HashType {
            DefaultXxHasher::hash_for_matrix_seeded(seed_idx, rows, cols, key)
        }
    }

    type Colliding = SpaceSaving<OneDigest>;

    fn keys_of(summary: &Colliding) -> Vec<i64> {
        let mut keys: Vec<i64> = summary
            .entries()
            .iter()
            .map(|(key, _, _)| match key {
                HeapItem::I64(v) => *v,
                other => panic!("unexpected key form {other:?}"),
            })
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Keys sharing a digest are separate counters, each answering for itself.
    #[test]
    fn colliding_keys_stay_distinct() {
        let mut summary: Colliding = SpaceSaving::with_capacity(4);
        for (key, weight) in [(10i64, 3u64), (20, 5), (30, 1)] {
            summary.insert_many(&DataInput::I64(key), weight);
        }

        summary.validate().expect("three keys under one digest");
        assert_eq!(summary.len(), 3);
        assert_eq!(keys_of(&summary), vec![10, 20, 30]);
        for (key, weight) in [(10i64, 3u64), (20, 5), (30, 1)] {
            assert_eq!(summary.estimate(&DataInput::I64(key)), weight, "key {key}");
        }
        assert_eq!(summary.estimate(&DataInput::I64(40)), 0);
    }

    /// An eviction reuses the victim's counter under the victim's own digest,
    /// so the index must lose the old key before it gains the new one.
    #[test]
    fn an_eviction_under_collision_keeps_the_index_straight() {
        let mut summary: Colliding = SpaceSaving::with_capacity(2);
        for _ in 0..3 {
            summary.insert(&DataInput::I64(10));
        }
        for _ in 0..2 {
            summary.insert(&DataInput::I64(20));
        }

        summary.insert(&DataInput::I64(30));

        summary.validate().expect("after a colliding eviction");
        assert_eq!(summary.len(), 2);
        assert_eq!(keys_of(&summary), vec![10, 30]);
        assert_eq!(summary.estimate(&DataInput::I64(30)), 3);
        assert_eq!(summary.error(&DataInput::I64(30)), 2);
        assert_eq!(summary.estimate(&DataInput::I64(10)), 3);
        assert_eq!(summary.estimate(&DataInput::I64(20)), 0);
    }

    /// A merge pairs the two sides by key, not by the digest slot they share.
    #[test]
    fn a_merge_under_collision_pairs_by_key() {
        let mut left: Colliding = SpaceSaving::with_capacity(4);
        left.insert_many(&DataInput::I64(20), 3);
        left.insert_many(&DataInput::I64(10), 5);
        let mut right: Colliding = SpaceSaving::with_capacity(4);
        right.insert_many(&DataInput::I64(10), 2);
        right.insert_many(&DataInput::I64(30), 7);

        left.merge(&right);

        left.validate().expect("after a colliding merge");
        assert_eq!(left.len(), 3);
        assert_eq!(keys_of(&left), vec![10, 20, 30]);
        assert_eq!(left.estimate(&DataInput::I64(10)), 7, "the shared key");
        assert_eq!(left.estimate(&DataInput::I64(20)), 3);
        assert_eq!(left.estimate(&DataInput::I64(30)), 7);
    }

    /// With one digest for everything, the key order is the only tie-break the
    /// merge has left, so which of four equal counts survive is still fixed.
    #[test]
    fn a_merge_under_collision_breaks_ties_by_key() {
        fn merged(swap: bool) -> Vec<i64> {
            let mut left: Colliding = SpaceSaving::with_capacity(2);
            let mut right: Colliding = SpaceSaving::with_capacity(2);
            for key in [30i64, 40] {
                left.insert(&DataInput::I64(key));
            }
            for key in [10i64, 20] {
                right.insert(&DataInput::I64(key));
            }
            if swap {
                right.merge(&left);
                right.validate().expect("after a colliding merge");
                keys_of(&right)
            } else {
                left.merge(&right);
                left.validate().expect("after a colliding merge");
                keys_of(&left)
            }
        }

        assert_eq!(merged(false), vec![10, 20], "the tie was not broken by key");
        assert_eq!(merged(true), merged(false), "the merge is order dependent");
    }

    /// A decode seats the keys one at a time and rejects a repeat, so the
    /// duplicate check must compare keys rather than digests.
    #[test]
    fn a_decoded_summary_under_collision_keeps_its_keys() {
        let mut summary: Colliding = SpaceSaving::with_capacity(4);
        for (key, weight) in [(10i64, 5u64), (20, 3), (30, 9)] {
            summary.insert_many(&DataInput::I64(key), weight);
        }

        let bytes = rmp_serde::to_vec(&summary).expect("serialize");
        let decoded: Colliding = rmp_serde::from_slice(&bytes).expect("deserialize");

        decoded.validate().expect("decoded summary");
        assert_eq!(decoded.len(), 3);
        assert_eq!(keys_of(&decoded), vec![10, 20, 30]);
        for (key, weight) in [(10i64, 5u64), (20, 3), (30, 9)] {
            assert_eq!(decoded.estimate(&DataInput::I64(key)), weight, "key {key}");
        }

        let repeated = SpaceSavingState {
            capacity: 4,
            total: 9,
            discarded_max: 0,
            entries: vec![(HeapItem::I64(10), 5, 0), (HeapItem::I64(10), 4, 0)],
        };
        let problem = Colliding::rebuild(repeated).expect_err("a repeated key");
        assert!(problem.contains("same key twice"), "got {problem}");
    }
}
