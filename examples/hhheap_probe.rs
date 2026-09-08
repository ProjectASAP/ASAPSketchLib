//! Measures `HHHeap::update` across capacities.
//!
//! `HHHeap` maintains its key -> position index through each sift, so its rate
//! should not fall as capacity grows. This probe runs it beside a standalone
//! indexed heap and a linear scan over the same Zipf stream, and prints updates
//! per second at several capacities. The scan column is what an unindexed heap
//! costs; the indexed column is the floor `HHHeap` should stay near.
//!
//! Run with `cargo run --release --example hhheap_probe`.

use asap_sketchlib::{
    CMSHeap, CountMin, DataInput, FastPath, HHHeap, HHItem, HeapItem, Vector2D, hash64_seeded,
    input_to_owned,
};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

const STREAM: usize = 200_000;
const DOMAIN: usize = 20_000;
const EXPONENT: f64 = 1.1;
const CAPACITIES: [usize; 5] = [8, 32, 128, 512, 2048];

// ---------------------------------------------------------------------------
// Prototype: the same bounded min-heap with an incrementally maintained index.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*b);
        }
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type SlotMap = HashMap<u64, Vec<usize>, BuildHasherDefault<IdentityHasher>>;

struct IndexedHeap {
    data: Vec<HHItem>,
    slots: Vec<u64>,
    index: SlotMap,
    k: usize,
}

impl IndexedHeap {
    fn new(k: usize) -> Self {
        IndexedHeap {
            data: Vec::with_capacity(k),
            slots: Vec::with_capacity(k),
            index: SlotMap::default(),
            k,
        }
    }

    fn find(&self, slot: u64, key: &DataInput) -> Option<usize> {
        self.index
            .get(&slot)?
            .iter()
            .copied()
            .find(|idx| self.data[*idx].key == *key)
    }

    fn update(&mut self, key: &DataInput, count: i64) -> bool {
        let slot = hash64_seeded(0, key);

        if let Some(idx) = self.find(slot, key) {
            self.data[idx].count = count;
            self.sift(idx);
            return true;
        }

        let retained_every_key = self.data.len() < self.k;
        if self.data.len() < self.k {
            let idx = self.data.len();
            self.data
                .push(HHItem::create_item(input_to_owned(key), count));
            self.slots.push(slot);
            self.register(slot, idx);
            self.bubble_up(idx);
            return retained_every_key;
        }

        if count <= self.data[0].count {
            return retained_every_key;
        }

        let old_slot = self.slots[0];
        self.unregister(old_slot, 0);
        self.data[0] = HHItem::create_item(input_to_owned(key), count);
        self.slots[0] = slot;
        self.register(slot, 0);
        self.bubble_down(0);
        retained_every_key
    }

    fn register(&mut self, slot: u64, idx: usize) {
        self.index.entry(slot).or_default().push(idx);
    }

    fn unregister(&mut self, slot: u64, idx: usize) {
        if let Some(bucket) = self.index.get_mut(&slot) {
            if let Some(at) = bucket.iter().position(|p| *p == idx) {
                bucket.swap_remove(at);
            }
            if bucket.is_empty() {
                self.index.remove(&slot);
            }
        }
    }

    fn move_pos(&mut self, slot: u64, from: usize, to: usize) {
        if let Some(bucket) = self.index.get_mut(&slot) {
            for p in bucket.iter_mut() {
                if *p == from {
                    *p = to;
                    break;
                }
            }
        }
    }

    fn swap_nodes(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
        self.slots.swap(i, j);
        if self.slots[i] != self.slots[j] {
            self.move_pos(self.slots[i], j, i);
            self.move_pos(self.slots[j], i, j);
        }
    }

    fn sift(&mut self, idx: usize) {
        if !self.bubble_down(idx) {
            self.bubble_up(idx);
        }
    }

    fn bubble_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx].count < self.data[parent].count {
                self.swap_nodes(parent, idx);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn bubble_down(&mut self, mut idx: usize) -> bool {
        let start = idx;
        let len = self.data.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut target = idx;
            if left < len && self.data[left].count < self.data[target].count {
                target = left;
            }
            if right < len && self.data[right].count < self.data[target].count {
                target = right;
            }
            if target == idx {
                break;
            }
            self.swap_nodes(idx, target);
            idx = target;
        }
        idx != start
    }

    fn counts(&self) -> Vec<i64> {
        let mut out: Vec<i64> = self.data.iter().map(|item| item.count).collect();
        out.sort_unstable();
        out
    }

    fn keys(&self) -> HashSet<String> {
        self.data
            .iter()
            .map(|item| format!("{:?}", item.key))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Prototype: the same bounded min-heap with no index, scanned linearly.
// ---------------------------------------------------------------------------

struct ScanHeap {
    data: Vec<HHItem>,
    k: usize,
}

impl ScanHeap {
    fn new(k: usize) -> Self {
        ScanHeap {
            data: Vec::with_capacity(k),
            k,
        }
    }

    fn update(&mut self, key: &DataInput, count: i64) -> bool {
        if let Some(idx) = self.data.iter().position(|item| item.key == *key) {
            self.data[idx].count = count;
            if !self.bubble_down(idx) {
                self.bubble_up(idx);
            }
            return true;
        }

        let retained_every_key = self.data.len() < self.k;
        if self.data.len() < self.k {
            self.data
                .push(HHItem::create_item(input_to_owned(key), count));
            self.bubble_up(self.data.len() - 1);
            return retained_every_key;
        }
        if count <= self.data[0].count {
            return retained_every_key;
        }
        self.data[0] = HHItem::create_item(input_to_owned(key), count);
        self.bubble_down(0);
        retained_every_key
    }

    fn bubble_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx].count < self.data[parent].count {
                self.data.swap(parent, idx);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn bubble_down(&mut self, mut idx: usize) -> bool {
        let start = idx;
        let len = self.data.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut target = idx;
            if left < len && self.data[left].count < self.data[target].count {
                target = left;
            }
            if right < len && self.data[right].count < self.data[target].count {
                target = right;
            }
            if target == idx {
                break;
            }
            self.data.swap(idx, target);
            idx = target;
        }
        idx != start
    }

    fn counts(&self) -> Vec<i64> {
        let mut out: Vec<i64> = self.data.iter().map(|item| item.count).collect();
        out.sort_unstable();
        out
    }
}

// ---------------------------------------------------------------------------
// Workload
// ---------------------------------------------------------------------------

fn zipf_keys(n: usize, domain: usize, exponent: f64, seed: u64) -> Vec<u64> {
    let weights: Vec<f64> = (1..=domain)
        .map(|r| 1.0 / (r as f64).powf(exponent))
        .collect();
    let total: f64 = weights.iter().sum();
    let mut cdf = Vec::with_capacity(domain);
    let mut acc = 0.0;
    for w in &weights {
        acc += w / total;
        cdf.push(acc);
    }

    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        let rank = cdf.partition_point(|c| *c < u).min(domain - 1);
        out.push(rank as u64);
    }
    out
}

fn build_stream(string_keys: bool) -> Vec<(HeapItem, i64)> {
    let mut truth: HashMap<u64, i64> = HashMap::new();
    zipf_keys(STREAM, DOMAIN, EXPONENT, 0x5eed_1234)
        .into_iter()
        .map(|id| {
            let count = truth.entry(id).or_insert(0);
            *count += 1;
            let key = if string_keys {
                HeapItem::String(format!("flow::{id:08}"))
            } else {
                HeapItem::U64(id)
            };
            (key, *count)
        })
        .collect()
}

fn as_input(item: &HeapItem) -> DataInput<'_> {
    match item {
        HeapItem::String(s) => DataInput::Str(s),
        HeapItem::U64(v) => DataInput::U64(*v),
        _ => unreachable!(),
    }
}

fn sketch_ceiling() {
    let stream = build_stream(true);

    let mut plain = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(4, 4096);
    let started = Instant::now();
    for (key, _) in &stream {
        plain.insert(&as_input(key));
    }
    let plain_rate = stream.len() as f64 / started.elapsed().as_secs_f64() / 1e6;

    println!("\nCMSHeap::insert with the current HHHeap (rows=4, cols=4096)");
    println!("  bare CountMin::insert: {plain_rate:.3} Minsert/s");
    println!("{:>6}  {:>16}  {:>12}", "top_k", "Minsert/s", "heap share");
    for cap in CAPACITIES {
        let mut sketch = CMSHeap::<Vector2D<i64>, FastPath>::new(4, 4096, cap);
        let started = Instant::now();
        for (key, _) in &stream {
            sketch.insert(&as_input(key));
        }
        let rate = stream.len() as f64 / started.elapsed().as_secs_f64() / 1e6;
        let share = 1.0 - rate / plain_rate;
        println!("{cap:>6}  {rate:>16.3}  {:>11.1}%", share * 100.0);
    }
}

fn main() {
    for string_keys in [true, false] {
        let label = if string_keys {
            "string keys"
        } else {
            "u64 keys"
        };
        let stream = build_stream(string_keys);
        println!(
            "\n{label}: {} updates over {DOMAIN} distinct keys (zipf {EXPONENT})",
            stream.len()
        );
        println!(
            "{:>6}  {:>14}  {:>14}  {:>14}  {:>8}",
            "cap", "HHHeap Mups/s", "indexed Mups/s", "scan Mups/s", "speedup"
        );

        for cap in CAPACITIES {
            let mut baseline = HHHeap::new(cap);
            let started = Instant::now();
            for (key, count) in &stream {
                baseline.update(&as_input(key), *count);
            }
            let base_rate = stream.len() as f64 / started.elapsed().as_secs_f64() / 1e6;

            let mut fast = IndexedHeap::new(cap);
            let started = Instant::now();
            for (key, count) in &stream {
                fast.update(&as_input(key), *count);
            }
            let fast_rate = stream.len() as f64 / started.elapsed().as_secs_f64() / 1e6;

            let mut scan = ScanHeap::new(cap);
            let started = Instant::now();
            for (key, count) in &stream {
                scan.update(&as_input(key), *count);
            }
            let scan_rate = stream.len() as f64 / started.elapsed().as_secs_f64() / 1e6;

            let mut base_counts: Vec<i64> = baseline.heap().iter().map(|i| i.count).collect();
            base_counts.sort_unstable();
            assert_eq!(
                base_counts,
                fast.counts(),
                "the indexed heap must retain the same top-{cap} counts"
            );
            let base_keys: HashSet<String> = baseline
                .heap()
                .iter()
                .map(|item| format!("{:?}", item.key))
                .collect();
            assert_eq!(
                base_keys.len(),
                baseline.len(),
                "no duplicate baseline keys"
            );
            assert_eq!(
                base_keys,
                fast.keys(),
                "the indexed heap must retain the same top-{cap} keys"
            );

            assert_eq!(
                scan.counts(),
                fast.counts(),
                "the scanned heap must retain the same top-{cap} counts"
            );

            println!(
                "{cap:>6}  {base_rate:>14.2}  {fast_rate:>14.2}  {scan_rate:>14.2}  {:>7.1}x",
                fast_rate / base_rate
            );
        }
    }

    sketch_ceiling();
}
