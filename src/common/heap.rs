//! Heavy-hitter heap shared by the top-k sketches and the frameworks.
//!
//! A capacity-bounded min-heap of [`HHItem`] beside an index from key digest to
//! heap position. The index is patched through each sift rather than rebuilt,
//! so an update costs one hash, one map probe and `O(log k)` index writes
//! whatever the capacity.

use crate::common::input::HHItem;
use crate::common::structures::heap::PREALLOCATED_SLOTS;
use crate::common::{CommonHeap, DigestBuildHasher, KeepSmallest};
use crate::{DataInput, HeapItem, hash_item64_seeded, hash64_seeded, input_to_owned};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Heap positions sharing one digest. Two residents colliding on a 64-bit
/// digest is rare enough that the inline pair is effectively never spilled.
type Slot = SmallVec<[usize; 2]>;

type Index = HashMap<u64, Slot, DigestBuildHasher>;

/// Serialized form. The index is derived from the heap, so it is rebuilt on
/// load rather than carried.
#[derive(Deserialize)]
struct HHHeapState {
    heap: CommonHeap<HHItem, KeepSmallest>,
    k: usize,
}

/// Bounded min-heap over [`HHItem`] with key lookup, retaining the `k` largest
/// counts.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(from = "HHHeapState")]
pub struct HHHeap {
    heap: CommonHeap<HHItem, KeepSmallest>,
    /// Digest of each resident, parallel to the heap array, so a sift patches
    /// the index without re-hashing the keys it moves.
    #[serde(skip)]
    slots: Vec<u64>,
    #[serde(skip)]
    positions: Index,
    k: usize,
}

impl From<HHHeapState> for HHHeap {
    fn from(state: HHHeapState) -> Self {
        let mut heap = HHHeap {
            heap: state.heap,
            slots: Vec::new(),
            positions: Index::default(),
            k: state.k,
        };
        heap.rebuild_index();
        heap
    }
}

impl HHHeap {
    /// Creates a new HHHeap with capacity k.
    pub fn new(k: usize) -> Self {
        Self::with_expected_len(k, k)
    }

    /// Creates a heap of capacity `k` whose heap array and index reserve for
    /// `expected_len` residents rather than for `k`. Capacity semantics are
    /// unchanged — [`Self::capacity`] is still `k`, and the heap still turns a
    /// key away once it holds `k` — so this only moves the reservation off a
    /// declared capacity the caller does not yet have the entries to fill.
    pub fn with_expected_len(k: usize, expected_len: usize) -> Self {
        let reserved = expected_len.min(k).min(PREALLOCATED_SLOTS);
        HHHeap {
            heap: CommonHeap::with_expected_len(k, expected_len, KeepSmallest),
            slots: Vec::with_capacity(reserved),
            positions: Index::with_capacity_and_hasher(reserved, DigestBuildHasher::default()),
            k,
        }
    }

    /// Finds an item by key, returns the index if found.
    pub fn find(&self, key: &DataInput) -> Option<usize> {
        let slot = self.slot_for_input(key);
        self.lookup(slot, |item| item.key == *key)
    }

    /// Finds an owned heap item by key, returning its index if present.
    pub fn find_heap_item(&self, key: &HeapItem) -> Option<usize> {
        let slot = self.slot_for_item(key);
        self.lookup(slot, |item| &item.key == key)
    }

    /// Updates an existing key's count or inserts it if it earns a place.
    ///
    /// Returns whether every key offered so far is still retained, which stops
    /// being true once the heap has turned one away.
    pub fn update(&mut self, key: &DataInput, count: i64) -> bool {
        let slot = self.slot_for_input(key);
        if let Some(idx) = self.lookup(slot, |item| item.key == *key) {
            self.rescore(idx, count);
            return true;
        }

        let retained_every_key = self.heap.len() < self.k;
        if !self.should_accept_new(count) {
            return retained_every_key;
        }
        self.seat(input_to_owned(key), slot, count);
        retained_every_key
    }

    /// Updates an existing owned item or inserts it if needed.
    ///
    /// The return value has the same completeness meaning as [`Self::update`].
    pub fn update_heap_item(&mut self, key: &HeapItem, count: i64) -> bool {
        let slot = self.slot_for_item(key);
        if let Some(idx) = self.lookup(slot, |item| &item.key == key) {
            self.rescore(idx, count);
            return true;
        }

        let retained_every_key = self.heap.len() < self.k;
        if !self.should_accept_new(count) {
            return retained_every_key;
        }
        self.seat(key.to_owned(), slot, count);
        retained_every_key
    }

    /// Provides access to the underlying data as a slice.
    /// Named `heap` for API compatibility with TopKHeap.
    pub fn heap(&self) -> &[HHItem] {
        self.heap.as_slice()
    }

    /// Prints all items in the heap.
    pub fn print_heap(&self) {
        println!("======== Beginning of Heap ========");
        for item in self.heap.iter() {
            item.print_item();
        }
        println!("============ Heap Ends ============");
    }

    /// Clears the heap.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.slots.clear();
        self.positions.clear();
    }

    /// Returns the number of items in the heap.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns true if the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Creates a copy of another HHHeap.
    pub fn from_heap(other: &HHHeap) -> Self {
        other.clone()
    }

    /// Returns the capacity of the heap.
    pub fn capacity(&self) -> usize {
        self.k
    }

    // -- index maintenance ---------------------------------------------------

    fn lookup(&self, slot: u64, matches: impl Fn(&HHItem) -> bool) -> Option<usize> {
        self.positions
            .get(&slot)?
            .iter()
            .copied()
            .find(|idx| matches(&self.heap[*idx]))
    }

    /// Writes a new count onto a resident and re-sifts it.
    fn rescore(&mut self, idx: usize, count: i64) {
        self.heap[idx].count = count;
        let Self {
            heap,
            slots,
            positions,
            ..
        } = self;
        heap.update_at_with(idx, &mut |i, j| swap_entry(slots, positions, i, j));
    }

    /// Places a key the heap has decided to accept, evicting the root when the
    /// heap is already at capacity.
    fn seat(&mut self, key: HeapItem, slot: u64, count: i64) {
        let item = HHItem::create_item(key, count);
        if self.heap.len() < self.k {
            let idx = self.heap.len();
            self.slots.push(slot);
            self.positions.entry(slot).or_default().push(idx);
            let Self {
                heap,
                slots,
                positions,
                ..
            } = self;
            heap.push_back_with(item, &mut |i, j| swap_entry(slots, positions, i, j));
        } else {
            let displaced = self.slots[0];
            drop_entry(&mut self.positions, displaced, 0);
            self.slots[0] = slot;
            self.positions.entry(slot).or_default().push(0);
            let Self {
                heap,
                slots,
                positions,
                ..
            } = self;
            heap.replace_root_with(item, &mut |i, j| swap_entry(slots, positions, i, j));
        }
    }

    /// Rebuilds `slots` and `positions` from the heap array.
    fn rebuild_index(&mut self) {
        let Self {
            heap,
            slots,
            positions,
            ..
        } = self;
        slots.clear();
        positions.clear();
        slots.reserve(heap.len());
        for (idx, item) in heap.iter().enumerate() {
            let slot = hash_item64_seeded(0, &item.key);
            slots.push(slot);
            positions.entry(slot).or_default().push(idx);
        }
    }

    #[inline]
    fn should_accept_new(&self, count: i64) -> bool {
        if self.k == 0 {
            return false;
        }
        if self.heap.len() < self.k {
            return true;
        }
        debug_assert!(
            !self.heap.is_empty(),
            "a full heap of capacity k > 0 has a root"
        );
        self.heap
            .peek()
            .is_some_and(|min_item| count > min_item.count)
    }

    #[inline]
    fn slot_for_input(&self, key: &DataInput) -> u64 {
        hash64_seeded(0, key)
    }

    #[inline]
    fn slot_for_item(&self, key: &HeapItem) -> u64 {
        hash_item64_seeded(0, key)
    }
}

/// Follows one heap swap into the index.
///
/// Residents sharing a digest sit in the same bucket, so exchanging their two
/// positions leaves that bucket unchanged and there is nothing to write.
#[inline]
fn swap_entry(slots: &mut [u64], positions: &mut Index, i: usize, j: usize) {
    let (left, right) = (slots[i], slots[j]);
    if left == right {
        return;
    }
    retarget(positions, left, i, j);
    retarget(positions, right, j, i);
    slots.swap(i, j);
}

#[inline]
fn retarget(positions: &mut Index, digest: u64, from: usize, to: usize) {
    if let Some(bucket) = positions.get_mut(&digest)
        && let Some(entry) = bucket.iter_mut().find(|entry| **entry == from)
    {
        *entry = to;
    }
}

#[inline]
fn drop_entry(positions: &mut Index, digest: u64, idx: usize) {
    if let Some(bucket) = positions.get_mut(&digest) {
        if let Some(at) = bucket.iter().position(|entry| *entry == idx) {
            bucket.swap_remove(at);
        }
        if bucket.is_empty() {
            positions.remove(&digest);
        }
    }
}

/// Every element sits in heap order beneath the parent the array layout gives
/// it, with the parent index computed here rather than taken from the heap.
#[cfg(test)]
fn assert_ordered<T, O: crate::CommonHeapOrder<T>>(items: &[T], order: &O, context: &str) {
    for i in 1..items.len() {
        assert!(
            !order.should_swap(&items[(i - 1) / 2], &items[i]),
            "{context}: position {i} breaks heap order against position {}",
            (i - 1) / 2
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CommonHeap, CommonHeapOrder, DataInput, HeapItem, KeepLargest, KeepSmallest,
        common::input::HHItem,
    };

    fn heap_item_from_str(value: &str) -> HeapItem {
        HeapItem::String(value.to_string())
    }

    #[test]
    fn heap_retains_top_k_items_by_count() {
        // confirm inserting beyond capacity keeps only the k largest counts
        let mut heap = HHHeap::new(3);
        for i in 1..=5 {
            let key = format!("key-{i}");
            let key_item = heap_item_from_str(&key);
            heap.update_heap_item(&key_item, i as i64);
        }

        assert_eq!(heap.heap.len(), 3);
        let mut counts: Vec<i64> = heap.heap.iter().map(|item| item.count).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![3, 4, 5]);
    }

    #[test]
    fn update_count_increments_existing_entry() {
        // ensure update_count bumps stored counter instead of replacing the entry
        let mut heap = HHHeap::new(4);
        let key_item = heap_item_from_str("alpha");
        let mut count = 0;
        for _ in 0..3 {
            count += 1;
            heap.update_heap_item(&key_item, count);
        }

        let idx = heap.find_heap_item(&key_item).expect("alpha present");
        assert_eq!(heap.heap[idx].count, 3);
    }

    #[test]
    fn clean_resets_heap_state() {
        // cleaning should drop all entries and reclaim capacity
        let mut heap = HHHeap::new(2);
        let key_a = heap_item_from_str("a");
        let key_b = heap_item_from_str("b");
        heap.update_heap_item(&key_a, 5);
        heap.update_heap_item(&key_b, 6);
        assert_eq!(heap.heap.len(), 2);

        heap.clear();
        assert!(heap.heap.is_empty());
    }

    #[test]
    fn test_min_heap_basic() {
        let mut heap = CommonHeap::<i32, KeepSmallest>::new_min(5);
        heap.push(5);
        heap.push(3);
        heap.push(7);
        heap.push(1);

        assert_eq!(heap.peek(), Some(&1));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(7));
        assert_eq!(heap.pop(), None);

        // Past index 4, where a child's parent is no longer its own half.
        let mut deep = CommonHeap::<i32, KeepSmallest>::new_min(16);
        for (step, value) in [95, 55, 37, 58, 81, 67, 42, 71, 27, 97, 13, 21]
            .into_iter()
            .enumerate()
        {
            deep.push(value);
            assert_ordered(deep.as_slice(), &KeepSmallest, &format!("min push {step}"));
        }
        assert_eq!(deep.len(), 12);
        let mut drained = Vec::new();
        while let Some(value) = deep.pop() {
            drained.push(value);
        }
        assert_eq!(
            drained,
            vec![13, 21, 27, 37, 42, 55, 58, 67, 71, 81, 95, 97]
        );
    }

    #[test]
    fn test_max_heap_basic() {
        let mut heap = CommonHeap::<i32, KeepLargest>::new_max(5);
        heap.push(5);
        heap.push(3);
        heap.push(7);
        heap.push(1);

        assert_eq!(heap.peek(), Some(&7));
        assert_eq!(heap.pop(), Some(7));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);

        // Past index 4, where a child's parent is no longer its own half.
        let mut deep = CommonHeap::<i32, KeepLargest>::new_max(16);
        for (step, value) in [62, 12, 38, 71, 33, 31, 43, 94, 14, 76, 80, 34]
            .into_iter()
            .enumerate()
        {
            deep.push(value);
            assert_ordered(deep.as_slice(), &KeepLargest, &format!("max push {step}"));
        }
        assert_eq!(deep.len(), 12);
        let mut drained = Vec::new();
        while let Some(value) = deep.pop() {
            drained.push(value);
        }
        assert_eq!(
            drained,
            vec![94, 80, 76, 71, 62, 43, 38, 34, 33, 31, 14, 12]
        );
    }

    #[test]
    fn test_bounded_heap_capacity() {
        let mut heap = CommonHeap::<i32, KeepSmallest>::new_min(3);

        heap.push(5);
        heap.push(3);
        heap.push(7);
        assert_eq!(heap.len(), 3);

        // Should not grow beyond capacity
        heap.push(1);
        assert_eq!(heap.len(), 3);

        // Smallest should be replaced by larger value since it's a min heap
        heap.push(10);
        assert_eq!(heap.len(), 3);

        // Should contain 5, 7, 10 (1 and 3 were kicked out)
        let mut vals: Vec<i32> = vec![];
        while let Some(v) = heap.pop() {
            vals.push(v);
        }
        vals.sort();
        assert_eq!(vals, vec![5, 7, 10]);
    }

    #[test]
    fn test_update_at() {
        let mut heap = CommonHeap::<i32, KeepSmallest>::new_min(5);
        heap.push(10);
        heap.push(20);
        heap.push(5);

        // Modify element and update heap
        heap[1] = 3;
        assert!(heap.update_at(1));

        assert_eq!(heap.peek(), Some(&3));

        // In range and already in place.
        assert!(heap.update_at(0));

        // One past the last element, and far past it.
        assert!(!heap.update_at(3));
        assert!(!heap.update_at(99));
        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some(&3));
    }

    #[test]
    fn test_custom_struct_with_ord() {
        let mut heap = CommonHeap::<HHItem, KeepSmallest>::new_min(3);
        heap.push(HHItem::new(DataInput::String("five".to_owned()), 5));
        heap.push(HHItem::new(DataInput::String("three".to_owned()), 3));
        heap.push(HHItem::new(DataInput::String("seven".to_owned()), 7));

        assert_eq!(heap.peek().map(|item| item.count), Some(3));
    }

    #[test]
    fn test_topk_use_case() {
        // Simulates TopKHeap use case: maintain top-K items by count
        // Use min-heap so smallest is at root and can be evicted

        // Create a min-heap with capacity 3 to keep top-3 items
        let mut heap = CommonHeap::<HHItem, KeepSmallest>::new_min(3);

        // Insert items (simulating TopKHeap behavior)
        for i in 1..=5 {
            heap.push(HHItem::new(
                DataInput::String(format!("key-{i}").to_owned()),
                i,
            ));
        }

        // Should keep top 3: counts 3, 4, 5
        assert_eq!(heap.len(), 3);
        let mut counts: Vec<i64> = heap.iter().map(|item| item.count).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![3, 4, 5]);

        // Test finding an item (linear search like TopKHeap::find)
        let found = heap
            .iter()
            .find(|item| item.key == HeapItem::String("key-4".to_owned()));
        assert!(found.is_some());
        assert_eq!(found.unwrap().count, 4);
    }

    /// The shipped orderings carry no state, so a heap costs its `Vec` plus the
    /// capacity field and nothing more.
    #[test]
    fn test_heap_size() {
        use std::mem::size_of;

        assert_eq!(size_of::<KeepSmallest>(), 0);
        assert_eq!(size_of::<KeepLargest>(), 0);

        let vec_size = size_of::<Vec<u64>>();
        assert_eq!(
            size_of::<CommonHeap<u64, KeepSmallest>>(),
            vec_size + size_of::<usize>()
        );
        assert_eq!(
            size_of::<CommonHeap<u64, KeepLargest>>(),
            vec_size + size_of::<usize>()
        );
    }

    #[test]
    fn test_topk_with_custom_comparator() {
        // Example of custom heap ordering (though Item already has Ord by count)
        // This demonstrates how to create custom orderings
        #[derive(Clone)]
        struct CompareByCount;

        impl CommonHeapOrder<HHItem> for CompareByCount {
            fn should_swap(&self, parent: &HHItem, child: &HHItem) -> bool {
                child.count < parent.count
            }

            fn should_replace_root(&self, root: &HHItem, new_value: &HHItem) -> bool {
                new_value.count > root.count
            }
        }

        let mut heap = CommonHeap::<HHItem, CompareByCount>::with_capacity(3, CompareByCount);

        heap.push(HHItem::new(DataInput::String("a".to_owned()), 5));
        heap.push(HHItem::new(DataInput::String("b".to_owned()), 3));
        heap.push(HHItem::new(DataInput::String("c".to_owned()), 7));
        heap.push(HHItem::new(DataInput::String("d".to_owned()), 1)); // Won't be added
        heap.push(HHItem::new(DataInput::String("e".to_owned()), 10)); // Will replace min

        assert_eq!(heap.len(), 3);
        let min_count = heap.peek().map(|item| item.count);
        assert_eq!(min_count, Some(5)); // 5 is now the minimum in the heap
    }

    #[test]
    fn test_exact_topk_heap_replacement() {
        // This test demonstrates EXACT TopKHeap behavior using generic Heap

        // TopKHeap::init_heap(3) equivalent:
        let mut heap = CommonHeap::<HHItem, KeepSmallest>::new_min(3);

        // TopKHeap::update("key-1", 1) equivalent:
        let find_and_update =
            |heap: &mut CommonHeap<HHItem, KeepSmallest>, key: &str, count: i64| {
                // TopKHeap::find() equivalent:
                let idx_opt = heap
                    .iter()
                    .position(|item| item.key == HeapItem::String(key.to_owned()));

                if let Some(idx) = idx_opt {
                    // Found: update count
                    heap[idx].count = count;
                    heap.update_at(idx);
                } else {
                    // Not found: insert (TopKHeap::insert equivalent)
                    heap.push(HHItem::new(DataInput::Str(key), count));
                }
            };

        // Replicate the exact test from TopKHeap
        for i in 1..=5 {
            let key = format!("key-{i}");
            find_and_update(&mut heap, &key, i);
        }

        // Should match TopKHeap behavior exactly
        assert_eq!(heap.len(), 3);
        let mut counts: Vec<i64> = heap.iter().map(|item| item.count).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![3, 4, 5]); // Same as TopKHeap test!

        // TopKHeap::find() equivalent:
        let found = heap
            .iter()
            .find(|item| item.key == HeapItem::String("key-4".to_owned()));
        assert!(found.is_some());
        assert_eq!(found.unwrap().count, 4);

        // TopKHeap::clean() equivalent:
        heap.clear();
        assert!(heap.is_empty());
    }

    /// A value that only ties the root leaves it in place; the bounded push
    /// replaces the root on a strict improvement.
    #[test]
    fn a_value_tying_the_root_does_not_displace_it() {
        fn resident<O: CommonHeapOrder<HHItem>>(heap: &CommonHeap<HHItem, O>, name: &str) -> bool {
            heap.iter()
                .any(|item| item.key == HeapItem::String(name.to_owned()))
        }

        let mut smallest = CommonHeap::<HHItem, KeepSmallest>::new_min(3);
        for (name, count) in [("a", 5i64), ("b", 7), ("c", 9)] {
            smallest.push(HHItem::new(DataInput::Str(name), count));
        }
        smallest.push(HHItem::new(DataInput::Str("tie"), 5));
        assert!(resident(&smallest, "a"), "the tied root was displaced");
        assert!(!resident(&smallest, "tie"));
        smallest.push(HHItem::new(DataInput::Str("larger"), 6));
        assert!(!resident(&smallest, "a"));
        assert!(resident(&smallest, "larger"));

        let mut largest = CommonHeap::<HHItem, KeepLargest>::new_max(3);
        for (name, count) in [("a", 9i64), ("b", 7), ("c", 5)] {
            largest.push(HHItem::new(DataInput::Str(name), count));
        }
        largest.push(HHItem::new(DataInput::Str("tie"), 9));
        assert!(resident(&largest, "a"), "the tied root was displaced");
        assert!(!resident(&largest, "tie"));
        largest.push(HHItem::new(DataInput::Str("smaller"), 8));
        assert!(!resident(&largest, "a"));
        assert!(resident(&largest, "smaller"));
    }

    /// The `(destination, source)` pairs one sift reports.
    type Sift = Vec<(usize, usize)>;

    /// A min-heap of five values whose sifts are all forced, used by the
    /// `_with` tests below. Returns the heap holding `[10, 20, 40, 50, 30]`.
    fn five_forced_sifts() -> (CommonHeap<i32, KeepSmallest>, Vec<Sift>) {
        let mut heap = CommonHeap::<i32, KeepSmallest>::new_min(8);
        let mut reported = Vec::new();
        for value in [50, 40, 30, 20, 10] {
            let mut swaps = Vec::new();
            heap.push_back_with(value, &mut |i, j| swaps.push((i, j)));
            reported.push(swaps);
        }
        (heap, reported)
    }

    /// `push_back_with` seats the value at `len()` and reports each swap as
    /// `(destination, source)`, in the order the sift performs them.
    #[test]
    fn push_back_with_reports_every_swap_of_its_sift() {
        let (heap, reported) = five_forced_sifts();

        assert_eq!(
            reported,
            vec![
                vec![],
                vec![(0, 1)],
                vec![(0, 2)],
                vec![(1, 3), (0, 1)],
                vec![(1, 4), (0, 1)],
            ]
        );
        assert_eq!(heap.as_slice(), &[10, 20, 40, 50, 30]);

        // Each sift starts from the index the value was appended at.
        for (step, swaps) in reported.iter().enumerate() {
            if let Some((_, source)) = swaps.first() {
                assert_eq!(
                    *source, step,
                    "sift {step} did not start at the appended slot"
                );
            }
        }
    }

    /// `replace_root_with` hands back the value it displaced and reports the
    /// sift of the value it wrote to index 0.
    #[test]
    fn replace_root_with_returns_the_displaced_root() {
        let (mut heap, _) = five_forced_sifts();

        let mut swaps = Vec::new();
        let displaced = heap.replace_root_with(60, &mut |i, j| swaps.push((i, j)));

        assert_eq!(displaced, 10);
        assert_eq!(swaps, vec![(0, 1), (1, 4)]);
        assert_eq!(swaps[0].0, 0, "the first swap does not leave index 0");
        assert_eq!(heap.as_slice(), &[20, 30, 40, 50, 60]);
    }

    /// `update_at_with` reports the sift in whichever direction it runs, and
    /// reports nothing for an index it refuses.
    #[test]
    fn update_at_with_reports_the_sift_in_both_directions() {
        let (mut heap, _) = five_forced_sifts();
        heap.replace_root_with(60, &mut |_, _| {});
        assert_eq!(heap.as_slice(), &[20, 30, 40, 50, 60]);

        heap[4] = 5;
        let mut up = Vec::new();
        assert!(heap.update_at_with(4, &mut |i, j| up.push((i, j))));
        assert_eq!(up, vec![(1, 4), (0, 1)]);
        assert_eq!(heap.as_slice(), &[5, 20, 40, 50, 30]);

        heap[0] = 100;
        let mut down = Vec::new();
        assert!(heap.update_at_with(0, &mut |i, j| down.push((i, j))));
        assert_eq!(down, vec![(0, 1), (1, 4)]);
        assert_eq!(heap.as_slice(), &[20, 30, 40, 50, 100]);

        let mut refused = Vec::new();
        assert!(!heap.update_at_with(5, &mut |i, j| refused.push((i, j))));
        assert!(refused.is_empty());
        assert_eq!(heap.as_slice(), &[20, 30, 40, 50, 100]);
    }

    /// A caller that appends to its own array and applies each reported swap
    /// ends with that array aligned to the heap.
    #[test]
    fn a_parallel_array_follows_the_reported_swaps() {
        let mut heap = CommonHeap::<i32, KeepSmallest>::new_min(8);
        let mut tags: Vec<char> = Vec::new();

        for (value, tag) in [(50, 'a'), (40, 'b'), (30, 'c'), (20, 'd'), (10, 'e')] {
            tags.push(tag);
            heap.push_back_with(value, &mut |i, j| tags.swap(i, j));
        }

        assert_eq!(heap.as_slice(), &[10, 20, 40, 50, 30]);
        assert_eq!(tags, vec!['e', 'd', 'b', 'a', 'c']);

        heap[0] = 100;
        heap.update_at_with(0, &mut |i, j| tags.swap(i, j));
        assert_eq!(heap.as_slice(), &[20, 30, 40, 50, 100]);
        assert_eq!(tags, vec!['d', 'c', 'b', 'a', 'e']);
    }
}

#[cfg(test)]
mod index_invariants {
    use super::*;
    use crate::DataInput;
    use std::collections::HashMap;

    impl HHHeap {
        /// Every resident is reachable through the index at its own position,
        /// `slots` agrees with the keys, and the index holds nothing else.
        fn assert_index_consistent(&self, context: &str) {
            assert_ordered(self.heap(), &KeepSmallest, context);
            assert_eq!(
                self.slots.len(),
                self.heap.len(),
                "{context}: slots and heap disagree on length"
            );

            let mut counted = 0usize;
            for (idx, item) in self.heap.iter().enumerate() {
                let digest = hash_item64_seeded(0, &item.key);
                assert_eq!(
                    self.slots[idx], digest,
                    "{context}: slot {idx} holds a stale digest"
                );
                let bucket = self
                    .positions
                    .get(&digest)
                    .unwrap_or_else(|| panic!("{context}: no bucket for position {idx}"));
                assert!(
                    bucket.contains(&idx),
                    "{context}: bucket for position {idx} does not list it"
                );
                assert_eq!(
                    self.find_heap_item(&item.key),
                    Some(idx),
                    "{context}: lookup for position {idx} disagrees"
                );
                counted += 1;
            }

            let indexed: usize = self.positions.values().map(|b| b.len()).sum();
            assert_eq!(
                indexed, counted,
                "{context}: the index carries {indexed} entries for {counted} residents"
            );
        }
    }

    fn zipfish(n: usize, domain: usize, seed: u64) -> Vec<u64> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..n)
            .map(|_| {
                let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
                let skewed = u.powf(3.0);
                (skewed * domain as f64) as u64 % domain as u64
            })
            .collect()
    }

    /// The index stays exact through eviction, promotion and re-entry.
    #[test]
    fn the_index_survives_a_long_churning_stream() {
        for capacity in [1usize, 2, 7, 64, 512] {
            let mut heap = HHHeap::new(capacity);
            let mut counts: HashMap<u64, i64> = HashMap::new();

            for (step, key) in zipfish(20_000, 2_048, 0x5eed).into_iter().enumerate() {
                let entry = counts.entry(key).or_default();
                *entry += 1;
                heap.update(&DataInput::U64(key), *entry);
                assert_ordered(
                    heap.heap(),
                    &KeepSmallest,
                    &format!("cap {capacity} step {step}"),
                );
                if step % 97 == 0 {
                    heap.assert_index_consistent(&format!("cap {capacity} step {step}"));
                }
            }
            heap.assert_index_consistent(&format!("cap {capacity} final"));
            assert_eq!(heap.len(), capacity.min(counts.len()));
        }
    }

    /// The residents are exactly the `k` largest counts, which is what the
    /// index is there to serve. Checked against a brute-force ranking.
    #[test]
    fn the_residents_are_the_k_largest_counts() {
        let capacity = 32;
        let mut heap = HHHeap::new(capacity);
        let mut counts: HashMap<u64, i64> = HashMap::new();

        for key in zipfish(20_000, 512, 0xfeed) {
            let entry = counts.entry(key).or_default();
            *entry += 1;
            heap.update(&DataInput::U64(key), *entry);
            assert_ordered(heap.heap(), &KeepSmallest, "ranking stream");
        }

        let mut ranked: Vec<(u64, i64)> = counts.iter().map(|(k, c)| (*k, *c)).collect();
        ranked.sort_by_key(|(key, count)| (-*count, *key));
        let cutoff = ranked[capacity - 1].1;

        for item in heap.heap() {
            let key = match item.key {
                HeapItem::U64(v) => v,
                ref other => panic!("unexpected key {other:?}"),
            };
            assert_eq!(item.count, counts[&key], "resident count is stale");
            assert!(
                item.count >= cutoff,
                "resident {key} at {} is below the {cutoff} cutoff",
                item.count
            );
        }
        assert_eq!(heap.len(), capacity);
    }

    /// A key already resident is re-scored in place rather than seated twice.
    #[test]
    fn re_scoring_a_resident_never_duplicates_it() {
        let mut heap = HHHeap::new(4);
        for round in 1..=50i64 {
            heap.update(&DataInput::Str("hot"), round);
            heap.update(&DataInput::Str("warm"), round / 2);
        }
        heap.assert_index_consistent("re-score");
        assert_eq!(heap.len(), 2);
        assert!(heap.find(&DataInput::Str("hot")).is_some());

        let hot = heap
            .heap()
            .iter()
            .find(|item| item.key == DataInput::Str("hot"))
            .expect("hot is resident");
        assert_eq!(hot.count, 50);
    }

    /// A zero-capacity heap accepts nothing and stays consistent.
    #[test]
    fn a_zero_capacity_heap_turns_everything_away() {
        let mut heap = HHHeap::new(0);
        for key in 0..32u64 {
            assert!(!heap.update(&DataInput::U64(key), key as i64));
        }
        assert!(heap.is_empty());
        assert_eq!(heap.find(&DataInput::U64(0)), None);
        heap.assert_index_consistent("zero capacity");
    }

    /// `clear` drops the index with the heap, and the heap refills correctly.
    #[test]
    fn clearing_drops_the_index_with_the_heap() {
        let mut heap = HHHeap::new(8);
        for key in 0..32u64 {
            heap.update(&DataInput::U64(key), key as i64);
        }
        heap.clear();
        assert!(heap.is_empty());
        assert_eq!(heap.find(&DataInput::U64(31)), None);
        heap.assert_index_consistent("cleared");

        for key in 100..140u64 {
            heap.update(&DataInput::U64(key), key as i64);
        }
        heap.assert_index_consistent("refilled");
        assert_eq!(heap.len(), 8);
    }

    /// The index is derived, so it is rebuilt on load rather than carried. A
    /// decoded heap answers lookups and takes further updates.
    #[test]
    fn a_decoded_heap_rebuilds_its_index() {
        let mut heap = HHHeap::new(16);
        let mut counts: HashMap<u64, i64> = HashMap::new();
        for key in zipfish(5_000, 256, 0xd00d) {
            let entry = counts.entry(key).or_default();
            *entry += 1;
            heap.update(&DataInput::U64(key), *entry);
        }

        let bytes = rmp_serde::to_vec(&heap).expect("serialize");
        let mut decoded: HHHeap = rmp_serde::from_slice(&bytes).expect("deserialize");

        decoded.assert_index_consistent("decoded");
        assert_eq!(decoded.len(), heap.len());
        assert_eq!(decoded.capacity(), heap.capacity());
        for item in heap.heap() {
            assert!(
                decoded.find_heap_item(&item.key).is_some(),
                "decoded heap lost {:?}",
                item.key
            );
        }

        let resident = heap.heap()[0].key.clone();
        decoded.update_heap_item(&resident, 1_000_000);
        decoded.assert_index_consistent("decoded then updated");
        let idx = decoded.find_heap_item(&resident).expect("still resident");
        assert_eq!(decoded.heap()[idx].count, 1_000_000);
    }

    /// The shape every sketch uses after a decode: string keys reaching the
    /// heap as [`DataInput`] through [`HHHeap::update`], which hashes by a
    /// different entry point from the one `rebuild_index` uses. A rebuilt index
    /// the update path cannot read seats a resident key a second time.
    #[test]
    fn a_decoded_heap_takes_string_keyed_updates() {
        let mut heap = HHHeap::new(8);
        for i in 0..8u64 {
            heap.update(&DataInput::String(format!("flow-{i}")), i as i64 + 1);
        }
        assert_eq!(heap.len(), 8);

        let bytes = rmp_serde::to_vec(&heap).expect("serialize");
        let mut decoded: HHHeap = rmp_serde::from_slice(&bytes).expect("deserialize");

        for i in 0..8u64 {
            let probe = DataInput::String(format!("flow-{i}"));
            let idx = decoded
                .find(&probe)
                .unwrap_or_else(|| panic!("decoded heap cannot find flow-{i}"));
            assert_eq!(decoded.heap()[idx].count, i as i64 + 1);
        }

        decoded.update(&DataInput::String("flow-7".to_owned()), 999);
        assert_eq!(decoded.len(), 8, "an update seated a resident key again");
        let seated = decoded
            .heap()
            .iter()
            .filter(|item| item.key == HeapItem::String("flow-7".to_owned()))
            .count();
        assert_eq!(seated, 1, "flow-7 is seated {seated} times");
        let idx = decoded
            .find(&DataInput::String("flow-7".to_owned()))
            .expect("flow-7 is resident");
        assert_eq!(decoded.heap()[idx].count, 999);
        decoded.assert_index_consistent("decoded then updated by data input");
    }
}

#[cfg(test)]
mod differential {
    //! Retention does not depend on how the index is maintained.
    //!
    //! [`Reference`] holds the same `CommonHeap` with the key index rebuilt in
    //! full after every accepted update. It is compared against the shipped
    //! heap element by element, so any divergence in which key is seated,
    //! which is evicted, or where either lands, fails.
    //!
    //! [`RetentionOracle`] shares nothing with either: it decides which keys
    //! survive from a plain vector scanned for its smallest count, so it also
    //! catches a heap whose root is not its minimum.
    //!
    //! `CommonHeap`'s own sift is covered by the tests above it in this file;
    //! what is under test here is the incremental index maintenance.

    use super::*;
    use crate::DataInput;
    use std::collections::HashMap;

    struct Reference {
        heap: CommonHeap<HHItem, KeepSmallest>,
        positions: HashMap<u64, Vec<(HeapItem, usize)>>,
        k: usize,
    }

    impl Reference {
        fn new(k: usize) -> Self {
            Self {
                heap: CommonHeap::new_min(k),
                positions: HashMap::with_capacity(k),
                k,
            }
        }

        fn find(&self, key: &DataInput) -> Option<usize> {
            let slot = hash64_seeded(0, key);
            self.positions.get(&slot).and_then(|bucket| {
                bucket
                    .iter()
                    .find_map(|(value, idx)| if value == key { Some(*idx) } else { None })
            })
        }

        fn update(&mut self, key: &DataInput, count: i64) -> bool {
            if let Some(idx) = self.find(key) {
                self.heap[idx].count = count;
                self.heap.update_at(idx);
                self.refresh_positions();
                return true;
            }

            let retained_every_key = self.heap.len() < self.k;
            if !self.should_accept_new(count) {
                return retained_every_key;
            }

            let owned = input_to_owned(key);
            self.heap.push(HHItem::create_item(owned, count));
            self.refresh_positions();
            retained_every_key
        }

        fn should_accept_new(&self, count: i64) -> bool {
            if self.heap.len() < self.k {
                return true;
            }
            self.heap
                .peek()
                .map(|min_item| count > min_item.count)
                .unwrap_or(true)
        }

        fn refresh_positions(&mut self) {
            self.positions.clear();
            for (idx, item) in self.heap.iter().enumerate() {
                let slot = hash_item64_seeded(0, &item.key);
                self.positions
                    .entry(slot)
                    .or_default()
                    .push((item.key.clone(), idx));
            }
        }
    }

    /// Retention decided without a heap: the residents live in a plain vector
    /// and the smallest count is found by scanning it.
    struct RetentionOracle {
        residents: Vec<HHItem>,
        k: usize,
    }

    impl RetentionOracle {
        fn new(k: usize) -> Self {
            Self {
                residents: Vec::new(),
                k,
            }
        }

        fn update(&mut self, key: &DataInput, count: i64) -> bool {
            if let Some(item) = self.residents.iter_mut().find(|item| item.key == *key) {
                item.count = count;
                return true;
            }

            let retained_every_key = self.residents.len() < self.k;
            if self.k == 0 {
                return retained_every_key;
            }
            if self.residents.len() < self.k {
                self.residents
                    .push(HHItem::create_item(input_to_owned(key), count));
                return retained_every_key;
            }

            let smallest = self
                .residents
                .iter()
                .enumerate()
                .min_by_key(|(_, item)| item.count)
                .map(|(idx, _)| idx)
                .expect("a full heap of capacity k > 0 has residents");
            if count > self.residents[smallest].count {
                self.residents[smallest] = HHItem::create_item(input_to_owned(key), count);
            }
            retained_every_key
        }
    }

    fn sorted_u64_residents(items: &[HHItem]) -> Vec<(u64, i64)> {
        let mut listed: Vec<(u64, i64)> = items
            .iter()
            .map(|item| match item.key {
                HeapItem::U64(value) => (value, item.count),
                ref other => panic!("unexpected key {other:?}"),
            })
            .collect();
        listed.sort_unstable();
        listed
    }

    fn skewed(n: usize, domain: u64, seed: u64) -> Vec<u64> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..n)
            .map(|_| {
                let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
                (u.powf(3.0) * domain as f64) as u64 % domain
            })
            .collect()
    }

    fn assert_same(shipped: &HHHeap, reference: &Reference, context: &str) {
        assert_ordered(shipped.heap(), &KeepSmallest, context);
        assert_eq!(
            shipped.len(),
            reference.heap.len(),
            "{context}: residency differs"
        );
        for (idx, (a, b)) in shipped
            .heap()
            .iter()
            .zip(reference.heap.as_slice().iter())
            .enumerate()
        {
            assert_eq!(
                a.key, b.key,
                "{context}: position {idx} holds a different key"
            );
            assert_eq!(a.count, b.count, "{context}: position {idx} count differs");
        }
    }

    /// Element-for-element agreement with the rebuild implementation, over a
    /// stream long enough to force sustained eviction at every capacity, and
    /// on both key forms.
    #[test]
    fn the_indexed_heap_matches_the_rebuild_implementation() {
        for capacity in [0usize, 1, 2, 3, 7, 8, 64, 257] {
            let mut shipped = HHHeap::new(capacity);
            let mut reference = Reference::new(capacity);
            let mut counts: HashMap<u64, i64> = HashMap::new();

            for (step, key) in skewed(30_000, 4_096, 0xabcd).into_iter().enumerate() {
                let entry = counts.entry(key).or_default();
                *entry += 1;

                let probe = DataInput::U64(key);
                let a = shipped.update(&probe, *entry);
                let b = reference.update(&probe, *entry);
                assert_eq!(
                    a, b,
                    "cap {capacity} step {step}: completeness flag differs for key {key}"
                );
                assert_same(&shipped, &reference, &format!("cap {capacity} step {step}"));
            }
        }
    }

    /// The same agreement on string keys, which take the owned-key path and a
    /// different hash.
    #[test]
    fn the_indexed_heap_matches_the_rebuild_implementation_on_string_keys() {
        let capacity = 32;
        let mut shipped = HHHeap::new(capacity);
        let mut reference = Reference::new(capacity);
        let mut counts: HashMap<String, i64> = HashMap::new();

        for (step, raw) in skewed(20_000, 2_048, 0x1234).into_iter().enumerate() {
            let key = format!("flow::{raw}");
            let entry = counts.entry(key.clone()).or_default();
            *entry += 1;

            let probe = DataInput::Str(&key);
            let a = shipped.update(&probe, *entry);
            let b = reference.update(&probe, *entry);
            assert_eq!(a, b, "step {step}: completeness flag differs");
            assert_same(&shipped, &reference, &format!("step {step}"));
        }
    }

    /// Counts that fall as well as rise, so the resident sinks back down and
    /// the sift runs in both directions.
    #[test]
    fn the_two_agree_when_counts_move_in_both_directions() {
        let capacity = 16;
        let mut shipped = HHHeap::new(capacity);
        let mut reference = Reference::new(capacity);

        for (step, raw) in skewed(10_000, 128, 0x99).into_iter().enumerate() {
            // A count that oscillates rather than only climbing.
            let count = ((step as i64 * 7) % 61) - 30;
            let probe = DataInput::U64(raw);
            assert_eq!(
                shipped.update(&probe, count),
                reference.update(&probe, count),
                "step {step}: completeness flag differs"
            );
            assert_same(&shipped, &reference, &format!("step {step}"));
        }
    }

    /// Which keys survive, checked against a scan of a plain vector. Every
    /// count in the stream is distinct, so the smallest resident is
    /// unambiguous and the oracle needs no knowledge of the array layout.
    #[test]
    fn retention_matches_a_heapless_oracle() {
        for capacity in [0usize, 1, 2, 3, 5, 8, 11, 16] {
            let mut shipped = HHHeap::new(capacity);
            let mut oracle = RetentionOracle::new(capacity);

            for (step, raw) in skewed(4_000, 64, 0x2f2f).into_iter().enumerate() {
                let count = ((step as i64 * 7_919) % 1_000_003) + 1;
                let probe = DataInput::U64(raw);
                let context = format!("cap {capacity} step {step}");
                assert_eq!(
                    shipped.update(&probe, count),
                    oracle.update(&probe, count),
                    "{context}: completeness flag differs"
                );
                assert_ordered(shipped.heap(), &KeepSmallest, &context);
                assert_eq!(
                    sorted_u64_residents(shipped.heap()),
                    sorted_u64_residents(&oracle.residents),
                    "{context}: a different set of keys is retained"
                );
            }
        }
    }

    /// The same stream twice gives the same heap, array position included, and
    /// the same index down to the order its buckets are laid out in - the
    /// digest hasher is seeded by nothing the run supplies.
    #[test]
    fn the_heap_is_reproducible_across_runs() {
        let stream = skewed(20_000, 1_024, 0x5150);
        let mut first: Option<Vec<(HeapItem, i64)>> = None;
        let mut first_index: Option<Vec<(u64, Vec<usize>)>> = None;

        for _ in 0..5 {
            let mut heap = HHHeap::new(64);
            let mut counts: HashMap<u64, i64> = HashMap::new();
            for key in &stream {
                let entry = counts.entry(*key).or_default();
                *entry += 1;
                heap.update(&DataInput::U64(*key), *entry);
            }
            let snapshot: Vec<(HeapItem, i64)> = heap
                .heap()
                .iter()
                .map(|item| (item.key.clone(), item.count))
                .collect();
            match &first {
                None => first = Some(snapshot),
                Some(expected) => assert_eq!(&snapshot, expected, "run-to-run divergence"),
            }

            let index_layout: Vec<(u64, Vec<usize>)> = heap
                .positions
                .iter()
                .map(|(digest, bucket)| (*digest, bucket.to_vec()))
                .collect();
            assert_eq!(index_layout.len(), heap.len());
            match &first_index {
                None => first_index = Some(index_layout),
                Some(expected) => {
                    let diverged = index_layout
                        .iter()
                        .zip(expected.iter())
                        .position(|(seen, want)| seen != want);
                    assert_eq!(
                        diverged, None,
                        "the index is laid out differently from one run to the next"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod colliding_digests {
    //! Index maintenance when residents share a digest.
    //!
    //! Real xxh3 digests do not collide across a test-sized key set, so the
    //! multi-entry bucket path is driven here against the shipped helpers.

    use super::*;

    fn listed(positions: &Index, digest: u64) -> Vec<usize> {
        let mut entries: Vec<usize> = positions
            .get(&digest)
            .map(|bucket| bucket.to_vec())
            .unwrap_or_default();
        entries.sort_unstable();
        entries
    }

    /// The bucket as it is stored, order included.
    fn stored(positions: &Index, digest: u64) -> Vec<usize> {
        positions
            .get(&digest)
            .map(|bucket| bucket.to_vec())
            .unwrap_or_default()
    }

    #[test]
    fn a_bucket_holding_both_sides_of_a_swap_is_unchanged() {
        let mut slots = vec![7u64, 7];
        let mut positions = Index::default();
        positions.entry(7).or_default().extend([0, 1]);

        swap_entry(&mut slots, &mut positions, 0, 1);

        assert_eq!(slots, vec![7, 7]);
        assert_eq!(listed(&positions, 7), vec![0, 1]);
        assert_eq!(stored(&positions, 7), vec![0, 1]);

        // Stored order included: retargeting each end of the swap in turn
        // would leave the same two positions listed the other way round.
        let mut positions = Index::default();
        positions.entry(7).or_default().extend([1, 0]);

        swap_entry(&mut slots, &mut positions, 0, 1);

        assert_eq!(slots, vec![7, 7]);
        assert_eq!(stored(&positions, 7), vec![1, 0]);
    }

    /// A bucket naming two positions is answered by the key at each, not by
    /// the first position listed. Real digests do not collide across a
    /// test-sized key set, so the second resident is placed into the first
    /// one's bucket by hand.
    #[test]
    fn a_lookup_reads_the_key_at_every_position_in_a_bucket() {
        let mut heap = HHHeap::new(4);
        heap.update(&DataInput::U64(11), 10);
        heap.update(&DataInput::U64(22), 20);
        let wanted = heap.find(&DataInput::U64(11)).expect("11 is resident");
        let other = heap.find(&DataInput::U64(22)).expect("22 is resident");
        assert_ne!(wanted, other);

        let digest = heap.slots[wanted];
        heap.slots[other] = digest;
        heap.positions = Index::default();
        heap.positions
            .entry(digest)
            .or_default()
            .extend([other, wanted]);

        assert_eq!(heap.find(&DataInput::U64(11)), Some(wanted));
        assert_eq!(heap.find_heap_item(&HeapItem::U64(11)), Some(wanted));
        assert_eq!(heap.heap()[wanted].count, 10);
    }

    #[test]
    fn a_swap_moves_only_its_own_entry_out_of_a_shared_bucket() {
        // Position 0 is second in its bucket, so patching the bucket head
        // instead of the matching entry corrupts position 2.
        let mut slots = vec![7u64, 9, 7];
        let mut positions = Index::default();
        positions.entry(7).or_default().extend([2, 0]);
        positions.entry(9).or_default().push(1);

        swap_entry(&mut slots, &mut positions, 0, 1);

        assert_eq!(slots, vec![9, 7, 7]);
        assert_eq!(listed(&positions, 7), vec![1, 2]);
        assert_eq!(listed(&positions, 9), vec![0]);
    }

    #[test]
    fn dropping_one_entry_keeps_the_rest_of_its_bucket() {
        let mut positions = Index::default();
        positions.entry(7).or_default().extend([0, 3, 5]);

        drop_entry(&mut positions, 7, 3);

        assert_eq!(listed(&positions, 7), vec![0, 5]);
    }

    #[test]
    fn dropping_the_last_entry_removes_the_bucket() {
        let mut positions = Index::default();
        positions.entry(7).or_default().push(4);

        drop_entry(&mut positions, 7, 4);

        assert!(!positions.contains_key(&7));
    }

    /// Every position is listed in the bucket its slot names, and no bucket
    /// lists a position twice or one the heap does not hold.
    fn assert_buckets_track_slots(heap: &HHHeap, context: &str) {
        for (idx, digest) in heap.slots.iter().enumerate() {
            assert!(
                listed(&heap.positions, *digest).contains(&idx),
                "{context}: bucket {digest:#x} does not list position {idx}"
            );
        }
        let mut all: Vec<usize> = heap
            .positions
            .values()
            .flat_map(|b| b.iter().copied())
            .collect();
        all.sort_unstable();
        assert_eq!(
            all,
            (0..heap.len()).collect::<Vec<_>>(),
            "{context}: buckets do not name every heap position exactly once"
        );
    }

    /// `swap_entry`, `retarget` and `drop_entry` read only `slots` and
    /// `positions`, so they are driven here against an index seeded with two
    /// synthetic digests. That index does not agree with the keys the heap
    /// holds, so this establishes nothing about `find` or about any state the
    /// live update path can reach.
    #[test]
    fn two_shared_buckets_follow_the_sift() {
        let mut heap = HHHeap::new(16);
        for i in 0..16u64 {
            heap.update_heap_item(&HeapItem::U64(i), i as i64 + 1);
        }

        // Two digests across sixteen residents, so a sift both exchanges two
        // members of one bucket and moves a member between buckets.
        let resident = heap.len();
        heap.positions = Index::default();
        for idx in 0..resident {
            let digest = if idx % 2 == 0 { 0xabcu64 } else { 0xdefu64 };
            heap.slots[idx] = digest;
            heap.positions.entry(digest).or_default().push(idx);
        }

        for step in 0..200usize {
            heap.rescore((step * 7) % resident, (step as i64 * 13) % 97);
            assert_buckets_track_slots(&heap, &format!("step {step}"));
        }
    }
}
