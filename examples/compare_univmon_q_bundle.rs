//! Task-complete UnivMon-Q versus specialized-sketch bundle comparison.
//!
//! Run with:
//!
//!   cargo run --release --example compare_univmon_q_bundle -- 200000

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use asap_sketchlib::{KLL, UnivMonQ, UnivMonQConfig};

const LEVELS: usize = 12;
const DEPTH: usize = 5;
const ORDERED_SAMPLES: usize = 1_024;
const BUNDLE_WIDTH: usize = 4_096;
const BUNDLE_CANDIDATES: usize = 256;
const BUNDLE_RESERVOIR: usize = 1_024;
const BUNDLE_KLL_K: usize = 200;
const TOP_K: usize = 20;
const UPDATE_REPEATS: usize = 5;
const MERGE_REPEATS: usize = 7;
const QUERY_REPEATS: u32 = 100;
const QUANTILES: [f64; 7] = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
type Generator = fn(usize, u64) -> Vec<i64>;

fn main() {
    let n = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("n must be a positive integer"))
        .unwrap_or(200_000);
    assert!(n > 0, "n must be a positive integer");
    let seed = 0x2026_0804_u64;
    let q_matched = UnivMonQConfig {
        levels: LEVELS,
        width: 256,
        width_halving_period: 3,
        depth: DEPTH,
        counter_bits: 32,
        candidates: 128,
        ordered_samples: ORDERED_SAMPLES,
        hash_seed: 5,
    };
    let q_quality = UnivMonQConfig {
        width: 1_024,
        candidates: 256,
        ..q_matched
    };

    println!("UnivMon-Q versus task-complete specialized bundle");
    println!("n={n} per workload; accuracy is scored after a 50/50 merge");
    println!(
        "bundle: HLL(p=14) + {DEPTH}x{BUNDLE_WIDTH} CountSketch(i32) + SpaceSaving({BUNDLE_CANDIDATES}) + occurrence bottom-k({BUNDLE_RESERVOIR}) + KLL(k={BUNDLE_KLL_K}) + exact count/extrema"
    );
    println!(
        "Q-matched: width={}, candidates={}; Q-quality: width={}, candidates={}; both use {} ordered samples",
        q_matched.width,
        q_matched.candidates,
        q_quality.width,
        q_quality.candidates,
        q_matched.ordered_samples
    );

    let generators: [(&str, Generator); 5] = [
        ("uniform", uniform),
        ("normal", normal),
        ("exponential", exponential),
        ("zipf", zipf),
        ("elephants", elephants),
    ];

    let mut saved = None;
    for (offset, (workload, generator)) in generators.into_iter().enumerate() {
        let workload_seed = seed.wrapping_add(offset as u64);
        let data = generator(n, workload_seed);
        let truth = Truth::new(&data);

        println!();
        println!(
            "[{workload}] distinct={}, range=[{}, {}], entropy={:.3}",
            truth.frequencies.len(),
            truth.min,
            truth.max,
            truth.entropy
        );
        println!(
            "  {:<13} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "construction",
            "ns/up",
            "KiB",
            "merge-us",
            "point",
            "F0-rel",
            "F2-rel",
            "H-rel",
            "HH-mass"
        );

        let (q_matched_one, q_matched_update) = timed_q(&data, q_matched);
        let (q_matched_merged, q_matched_merge) = merged_q(&data, q_matched);
        let q_matched_score = score_q(&q_matched_merged, &truth);
        print_score(
            "Q-matched",
            q_matched_update,
            q_matched_one.estimated_memory_bytes(),
            q_matched_merge,
            &q_matched_score,
        );

        let (q_quality_one, q_quality_update) = timed_q(&data, q_quality);
        let (q_quality_merged, q_quality_merge) = merged_q(&data, q_quality);
        let q_quality_score = score_q(&q_quality_merged, &truth);
        print_score(
            "Q-quality",
            q_quality_update,
            q_quality_one.estimated_memory_bytes(),
            q_quality_merge,
            &q_quality_score,
        );

        let (bundle_one, bundle_update) = timed_bundle(&data, workload_seed);
        let (bundle_merged, bundle_merge) = merged_bundle(&data, workload_seed);
        let bundle_score = score_bundle(&bundle_merged, &truth);
        print_score(
            "bundle",
            bundle_update,
            bundle_one.estimated_memory_bytes(),
            bundle_merge,
            &bundle_score,
        );

        println!(
            "  ordered error: {:<13} rank-mean={:.5}, CDF-max={:.5}, value/range={:.5}",
            "Q-matched",
            q_matched_score.rank_mean,
            q_matched_score.cdf_max,
            q_matched_score.value_range
        );
        println!(
            "                 {:<13} rank-mean={:.5}, CDF-max={:.5}, value/range={:.5}",
            "Q-quality",
            q_quality_score.rank_mean,
            q_quality_score.cdf_max,
            q_quality_score.value_range
        );
        println!(
            "                 {:<13} rank-mean={:.5}, CDF-max={:.5}, value/range={:.5}",
            "bundle", bundle_score.rank_mean, bundle_score.cdf_max, bundle_score.value_range
        );

        if workload == "elephants" {
            saved = Some((truth, q_matched_one, q_quality_one, bundle_one));
        }
    }

    let (truth, q_matched, q_quality, bundle) = saved.expect("elephants workload is present");
    println!();
    println!(
        "query latency on elephants (mean of {QUERY_REPEATS} release queries; per-metric rows use prepared views):"
    );
    print_query_latency(&truth, &q_matched, &q_quality, &bundle);
    println!();
    println!("Notes:");
    println!("  Count and min/max are exact for all three constructions.");
    println!("  point = RMSE over present and absent keys, normalized by exact sqrt(F2).");
    println!("  HH-mass = true mass captured by returned keys / exact top-{TOP_K} mass.");
    println!(
        "  value/range normalizes quantile value error by max-min, avoiding zero-value artifacts."
    );
    println!(
        "  The bundle is a straightforward reference, not a fused/hash-sharing implementation."
    );
    println!("  prepare and all-task batch include materializing the query view.");
}

fn print_score(name: &str, update: Duration, memory: usize, merge: Duration, score: &Score) {
    println!(
        "  {name:<13} {:>9.1} {:>9.1} {:>9.1} {:>9.5} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
        update.as_nanos() as f64 / score.n as f64,
        memory as f64 / 1024.0,
        merge.as_secs_f64() * 1e6,
        score.point,
        score.f0,
        score.f2,
        score.entropy,
        score.hh_mass
    );
}

fn timed_q(data: &[i64], config: UnivMonQConfig) -> (UnivMonQ, Duration) {
    let mut times = Vec::with_capacity(UPDATE_REPEATS);
    let mut retained = None;
    for _ in 0..UPDATE_REPEATS {
        let mut sketch = UnivMonQ::new(config).unwrap();
        let start = Instant::now();
        for value in data {
            sketch.add(value);
        }
        times.push(start.elapsed());
        retained = Some(sketch);
    }
    (retained.unwrap(), median(times))
}

fn timed_bundle(data: &[i64], seed: u64) -> (SpecializedBundle, Duration) {
    let mut times = Vec::with_capacity(UPDATE_REPEATS);
    let mut retained = None;
    for repeat in 0..UPDATE_REPEATS {
        let mut sketch = SpecializedBundle::new(seed, repeat as u64);
        let start = Instant::now();
        for value in data {
            sketch.update(*value);
        }
        times.push(start.elapsed());
        retained = Some(sketch);
    }
    (retained.unwrap(), median(times))
}

fn merged_q(data: &[i64], config: UnivMonQConfig) -> (UnivMonQ, Duration) {
    let midpoint = data.len() / 2;
    let mut left = UnivMonQ::new(config).unwrap();
    let mut right = UnivMonQ::new(config).unwrap();
    for value in &data[..midpoint] {
        left.add(value);
    }
    for value in &data[midpoint..] {
        right.add(value);
    }
    let mut times = Vec::with_capacity(MERGE_REPEATS);
    for _ in 0..MERGE_REPEATS {
        let mut merged = left.clone();
        let start = Instant::now();
        merged.merge(&right).unwrap();
        times.push(start.elapsed());
        black_box(merged.count());
    }
    let mut merged = left;
    merged.merge(&right).unwrap();
    (merged, median(times))
}

fn merged_bundle(data: &[i64], seed: u64) -> (SpecializedBundle, Duration) {
    let midpoint = data.len() / 2;
    let mut left = SpecializedBundle::new(seed, 1);
    let mut right = SpecializedBundle::new(seed, 2);
    for value in &data[..midpoint] {
        left.update(*value);
    }
    for value in &data[midpoint..] {
        right.update(*value);
    }
    let mut times = Vec::with_capacity(MERGE_REPEATS);
    for _ in 0..MERGE_REPEATS {
        let mut merged = left.clone();
        let start = Instant::now();
        merged.merge(&right);
        times.push(start.elapsed());
        black_box(merged.count);
    }
    let mut merged = left;
    merged.merge(&right);
    (merged, median(times))
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

#[derive(Debug)]
struct Score {
    n: usize,
    point: f64,
    f0: f64,
    f2: f64,
    entropy: f64,
    hh_mass: f64,
    rank_mean: f64,
    cdf_max: f64,
    value_range: f64,
}

fn score_q(sketch: &UnivMonQ, truth: &Truth) -> Score {
    let query = sketch.prepare_queries();
    assert_eq!(query.count(), truth.n as u64);
    assert_eq!(query.min(), Some(truth.min as f64));
    assert_eq!(query.max(), Some(truth.max as f64));
    let points: Vec<u64> = truth
        .point_queries
        .iter()
        .map(|(key, _)| query.estimate_frequency(*key as f64))
        .collect();
    let heavy: Vec<(i64, u64)> = query
        .heavy_hitters(TOP_K)
        .into_iter()
        .map(|(key, frequency)| (key as i64, frequency))
        .collect();
    let quantiles: Vec<i64> = query
        .quantiles(&QUANTILES)
        .into_iter()
        .map(|value| value.unwrap() as i64)
        .collect();
    let ranks: Vec<(i64, f64)> = truth
        .cdf_probes
        .iter()
        .map(|value| {
            (
                *value,
                query.rank(*value as f64).unwrap() as f64 / truth.n as f64,
            )
        })
        .collect();
    truth.score(
        &points,
        query.estimate_distinct(),
        query.estimate_f2(),
        query.estimate_entropy(),
        &heavy,
        &quantiles,
        &ranks,
    )
}

fn score_bundle(sketch: &SpecializedBundle, truth: &Truth) -> Score {
    assert_eq!(sketch.count, truth.n as u64);
    assert_eq!(sketch.min, Some(truth.min));
    assert_eq!(sketch.max, Some(truth.max));
    let points: Vec<u64> = truth
        .point_queries
        .iter()
        .map(|(key, _)| sketch.frequency(*key))
        .collect();
    let cdf = sketch.kll.cdf();
    let quantiles: Vec<i64> = QUANTILES
        .iter()
        .map(|q| cdf.query(*q).round() as i64)
        .collect();
    let ranks: Vec<(i64, f64)> = truth
        .cdf_probes
        .iter()
        .map(|value| {
            (
                *value,
                sketch.kll.rank(*value as f64) as f64 / truth.n as f64,
            )
        })
        .collect();
    truth.score(
        &points,
        sketch.hll.estimate(),
        sketch.frequencies.estimate_f2(),
        sketch.estimate_entropy(),
        &sketch.heavy_hitters.top(TOP_K),
        &quantiles,
        &ranks,
    )
}

struct Truth {
    n: usize,
    frequencies: HashMap<i64, u64>,
    sorted: Vec<i64>,
    point_queries: Vec<(i64, u64)>,
    cdf_probes: Vec<i64>,
    top_mass: f64,
    min: i64,
    max: i64,
    f0: f64,
    f2: f64,
    entropy: f64,
}

impl Truth {
    fn new(data: &[i64]) -> Self {
        let mut frequencies = HashMap::<i64, u64>::new();
        for value in data {
            *frequencies.entry(*value).or_default() += 1;
        }
        let mut sorted = data.to_vec();
        sorted.sort_unstable();
        let n = data.len() as f64;
        let f2 = frequencies.values().map(|f| (*f as f64).powi(2)).sum();
        let entropy = frequencies
            .values()
            .map(|f| {
                let p = *f as f64 / n;
                -p * p.ln()
            })
            .sum();
        let mut counts: Vec<u64> = frequencies.values().copied().collect();
        counts.sort_unstable_by(|left, right| right.cmp(left));
        let top_mass = counts.iter().take(TOP_K).sum::<u64>() as f64;
        let mut point_queries: Vec<_> = frequencies
            .iter()
            .map(|(key, frequency)| (*key, *frequency))
            .collect();
        point_queries.sort_unstable_by_key(|(key, _)| mix64(*key as u64 ^ 0x0050_4f49_4e54));
        point_queries.truncate(256);
        let mut absent = 0_u64;
        while point_queries.len() < 512 {
            absent = mix64(absent ^ 0x4142_5345_4e54);
            let key = absent as i64;
            if !frequencies.contains_key(&key) {
                point_queries.push((key, 0));
            }
        }
        let cdf_probes = (0..=100)
            .map(|index| sorted[(index * (data.len() - 1)) / 100])
            .collect();
        Self {
            n: data.len(),
            f0: frequencies.len() as f64,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            frequencies,
            sorted,
            point_queries,
            cdf_probes,
            top_mass,
            f2,
            entropy,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn score(
        &self,
        points: &[u64],
        f0: f64,
        f2: f64,
        entropy: f64,
        heavy: &[(i64, u64)],
        quantiles: &[i64],
        ranks: &[(i64, f64)],
    ) -> Score {
        let point_mse = self
            .point_queries
            .iter()
            .zip(points)
            .map(|((_, exact), estimate)| (*estimate as f64 - *exact as f64).powi(2))
            .sum::<f64>()
            / points.len() as f64;
        let keys: HashSet<i64> = heavy.iter().map(|entry| entry.0).collect();
        let captured: u64 = keys
            .iter()
            .map(|key| self.frequencies.get(key).copied().unwrap_or(0))
            .sum();
        let rank_mean = QUANTILES
            .iter()
            .zip(quantiles)
            .map(|(q, estimate)| self.quantile_rank_error(*q, *estimate))
            .sum::<f64>()
            / QUANTILES.len() as f64;
        let range = (i128::from(self.max) - i128::from(self.min))
            .unsigned_abs()
            .max(1) as f64;
        let value_range = QUANTILES
            .iter()
            .zip(quantiles)
            .map(|(q, estimate)| {
                let exact =
                    self.sorted[((q * (self.n - 1) as f64).round() as usize).min(self.n - 1)];
                (i128::from(*estimate) - i128::from(exact)).unsigned_abs() as f64 / range
            })
            .sum::<f64>()
            / QUANTILES.len() as f64;
        let cdf_max = ranks
            .iter()
            .map(|(value, estimate)| {
                let exact =
                    self.sorted.partition_point(|item| *item <= *value) as f64 / self.n as f64;
                (estimate - exact).abs()
            })
            .fold(0.0, f64::max);
        Score {
            n: self.n,
            point: point_mse.sqrt() / self.f2.sqrt().max(1.0),
            f0: relative_error(f0, self.f0),
            f2: relative_error(f2, self.f2),
            entropy: relative_error(entropy, self.entropy),
            hh_mass: (captured as f64 / self.top_mass.max(1.0)).min(1.0),
            rank_mean,
            cdf_max,
            value_range,
        }
    }

    fn quantile_rank_error(&self, q: f64, estimate: i64) -> f64 {
        let target = (q * self.n as f64).ceil().max(1.0) as usize;
        let lower = self.sorted.partition_point(|item| *item < estimate);
        let upper = self.sorted.partition_point(|item| *item <= estimate);
        if target < lower {
            (lower - target) as f64 / self.n as f64
        } else {
            target.saturating_sub(upper) as f64 / self.n as f64
        }
    }
}

fn relative_error(estimate: f64, exact: f64) -> f64 {
    (estimate - exact).abs() / exact.abs().max(1.0)
}

#[derive(Clone)]
struct SpecializedBundle {
    hll: HyperLogLog,
    frequencies: FrequencyCountSketch,
    heavy_hitters: SpaceSaving,
    reservoir: OccurrenceBottomK,
    kll: KLL<i64>,
    count: u64,
    min: Option<i64>,
    max: Option<i64>,
}

impl SpecializedBundle {
    fn new(seed: u64, stream_id: u64) -> Self {
        Self {
            hll: HyperLogLog::new(14, seed ^ 0x0048_4c4c),
            frequencies: FrequencyCountSketch::new(BUNDLE_WIDTH, DEPTH, seed ^ 0x4353),
            heavy_hitters: SpaceSaving::new(BUNDLE_CANDIDATES),
            reservoir: OccurrenceBottomK::new(BUNDLE_RESERVOIR, seed ^ 0x0052_4553, stream_id),
            kll: KLL::init_kll_with_seed(BUNDLE_KLL_K as i32, seed ^ stream_id ^ 0x004b_4c4c),
            count: 0,
            min: None,
            max: None,
        }
    }

    fn update(&mut self, value: i64) {
        self.count += 1;
        self.min = Some(self.min.map_or(value, |old| old.min(value)));
        self.max = Some(self.max.map_or(value, |old| old.max(value)));
        self.hll.update(value);
        self.frequencies.update(value);
        self.heavy_hitters.update(value);
        self.reservoir.update(value);
        self.kll.update(&value);
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.min = match (self.min, other.min) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        self.max = match (self.max, other.max) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
        self.hll.merge(&other.hll);
        self.frequencies.merge(&other.frequencies);
        self.heavy_hitters.merge(&other.heavy_hitters);
        self.reservoir.merge(&other.reservoir);
        self.kll.merge(&other.kll);
    }

    fn frequency(&self, value: i64) -> u64 {
        self.frequencies.estimate(value).max(0) as u64
    }

    fn estimate_entropy(&self) -> f64 {
        if self.reservoir.samples.is_empty() || self.count == 0 {
            return 0.0;
        }
        let n = self.count as f64;
        self.reservoir
            .samples
            .iter()
            .map(|(_, value)| (n / self.frequency(*value).max(1) as f64).ln())
            .sum::<f64>()
            / self.reservoir.samples.len() as f64
    }

    fn estimated_memory_bytes(&self) -> usize {
        self.hll.bytes()
            + self.frequencies.bytes()
            + self.heavy_hitters.bytes()
            + self.reservoir.bytes()
            + kll_reserved_bytes(BUNDLE_KLL_K, 8)
            + 3 * size_of::<u64>()
    }
}

#[derive(Clone)]
struct HyperLogLog {
    precision: u32,
    registers: Vec<u8>,
    seed: u64,
}

impl HyperLogLog {
    fn new(precision: u32, seed: u64) -> Self {
        Self {
            precision,
            registers: vec![0; 1 << precision],
            seed,
        }
    }

    fn update(&mut self, value: i64) {
        let hash = mix64(value as u64 ^ self.seed);
        let index = (hash >> (64 - self.precision)) as usize;
        let rank = (hash << self.precision).leading_zeros() + 1;
        self.registers[index] = self.registers[index].max(rank as u8);
    }

    fn merge(&mut self, other: &Self) {
        assert_eq!(self.precision, other.precision);
        assert_eq!(self.seed, other.seed);
        for (left, right) in self.registers.iter_mut().zip(&other.registers) {
            *left = (*left).max(*right);
        }
    }

    fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let harmonic: f64 = self
            .registers
            .iter()
            .map(|register| 2.0_f64.powi(-i32::from(*register)))
            .sum();
        let raw = alpha * m * m / harmonic;
        let zeroes = self.registers.iter().filter(|value| **value == 0).count();
        if raw <= 2.5 * m && zeroes > 0 {
            m * (m / zeroes as f64).ln()
        } else {
            raw
        }
    }

    fn bytes(&self) -> usize {
        self.registers.capacity()
    }
}

#[derive(Clone)]
struct FrequencyCountSketch {
    width: usize,
    depth: usize,
    seed: u64,
    counters: Vec<i32>,
}

impl FrequencyCountSketch {
    fn new(width: usize, depth: usize, seed: u64) -> Self {
        Self {
            width,
            depth,
            seed,
            counters: vec![0; width * depth],
        }
    }

    fn update(&mut self, value: i64) {
        for row in 0..self.depth {
            let hash = self.hash(value, row);
            let bucket = hash as usize % self.width;
            let sign = if hash >> 63 == 0 { 1 } else { -1 };
            self.counters[row * self.width + bucket] += sign;
        }
    }

    fn merge(&mut self, other: &Self) {
        assert_eq!(
            (self.width, self.depth, self.seed),
            (other.width, other.depth, other.seed)
        );
        for (left, right) in self.counters.iter_mut().zip(&other.counters) {
            *left = left.saturating_add(*right);
        }
    }

    fn estimate(&self, value: i64) -> i64 {
        let mut estimates = [0_i64; 64];
        for (row, estimate) in estimates.iter_mut().enumerate().take(self.depth) {
            let hash = self.hash(value, row);
            let bucket = hash as usize % self.width;
            let sign = if hash >> 63 == 0 { 1 } else { -1 };
            *estimate = i64::from(sign * self.counters[row * self.width + bucket]);
        }
        estimates[..self.depth].sort_unstable();
        estimates[self.depth / 2]
    }

    fn estimate_f2(&self) -> f64 {
        let mut rows = Vec::with_capacity(self.depth);
        for row in 0..self.depth {
            let start = row * self.width;
            rows.push(
                self.counters[start..start + self.width]
                    .iter()
                    .map(|counter| f64::from(*counter).powi(2))
                    .sum(),
            );
        }
        rows.sort_unstable_by(f64::total_cmp);
        rows[self.depth / 2]
    }

    fn hash(&self, value: i64, row: usize) -> u64 {
        mix64(value as u64 ^ mix64(self.seed ^ row as u64))
    }

    fn bytes(&self) -> usize {
        self.counters.capacity() * size_of::<i32>()
    }
}

#[derive(Clone)]
struct SpaceSaving {
    capacity: usize,
    counts: HashMap<i64, u64>,
    heap: BinaryHeap<Reverse<(u64, i64)>>,
}

impl SpaceSaving {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            counts: HashMap::with_capacity(capacity),
            heap: BinaryHeap::with_capacity(capacity * 4),
        }
    }

    fn update(&mut self, value: i64) {
        if let Some(count) = self.counts.get_mut(&value) {
            *count += 1;
            self.heap.push(Reverse((*count, value)));
            self.compact();
            return;
        }
        if self.counts.len() < self.capacity {
            self.counts.insert(value, 1);
            self.heap.push(Reverse((1, value)));
            return;
        }
        let (minimum, evicted) = loop {
            let Reverse((stored, candidate)) = self.heap.pop().unwrap();
            if self.counts.get(&candidate) == Some(&stored) {
                break (stored, candidate);
            }
        };
        self.counts.remove(&evicted);
        self.counts.insert(value, minimum + 1);
        self.heap.push(Reverse((minimum + 1, value)));
        self.compact();
    }

    fn merge(&mut self, other: &Self) {
        let left_min = if self.counts.len() == self.capacity {
            self.counts.values().copied().min().unwrap_or(0)
        } else {
            0
        };
        let right_min = if other.counts.len() == other.capacity {
            other.counts.values().copied().min().unwrap_or(0)
        } else {
            0
        };
        let mut combined = HashMap::with_capacity(self.capacity * 2);
        for (key, count) in &self.counts {
            combined.insert(
                *key,
                count.saturating_add(other.counts.get(key).copied().unwrap_or(right_min)),
            );
        }
        for (key, count) in &other.counts {
            combined
                .entry(*key)
                .or_insert(count.saturating_add(left_min));
        }
        let mut retained: Vec<_> = combined.into_iter().collect();
        retained.sort_unstable_by_key(|entry| Reverse(entry.1));
        retained.truncate(self.capacity);
        self.counts.clear();
        self.heap.clear();
        for (key, count) in retained {
            self.counts.insert(key, count);
            self.heap.push(Reverse((count, key)));
        }
    }

    fn top(&self, k: usize) -> Vec<(i64, u64)> {
        let mut values: Vec<_> = self
            .counts
            .iter()
            .map(|(key, count)| (*key, *count))
            .collect();
        values.sort_unstable_by_key(|entry| Reverse(entry.1));
        values.truncate(k);
        values
    }

    fn compact(&mut self) {
        if self.heap.len() < self.capacity * 4 {
            return;
        }
        self.heap.clear();
        self.heap.extend(
            self.counts
                .iter()
                .map(|(key, count)| Reverse((*count, *key))),
        );
    }

    fn bytes(&self) -> usize {
        self.capacity * (24 + 4 * size_of::<Reverse<(u64, i64)>>())
    }
}

#[derive(Clone)]
struct OccurrenceBottomK {
    capacity: usize,
    samples: BinaryHeap<(u64, i64)>,
    seed: u64,
    stream_id: u64,
    sequence: u64,
}

impl OccurrenceBottomK {
    fn new(capacity: usize, seed: u64, stream_id: u64) -> Self {
        Self {
            capacity,
            samples: BinaryHeap::with_capacity(capacity),
            seed,
            stream_id,
            sequence: 0,
        }
    }

    fn update(&mut self, value: i64) {
        let priority = mix64(self.seed ^ mix64(self.stream_id) ^ mix64(self.sequence));
        self.sequence += 1;
        let item = (priority, value);
        if self.samples.len() < self.capacity {
            self.samples.push(item);
        } else if self.samples.peek().is_some_and(|largest| item < *largest) {
            self.samples.pop();
            self.samples.push(item);
        }
    }

    fn merge(&mut self, other: &Self) {
        let mut combined: Vec<_> = self.samples.iter().chain(&other.samples).copied().collect();
        combined.sort_unstable();
        combined.truncate(self.capacity);
        self.samples = BinaryHeap::from(combined);
        self.sequence += other.sequence;
    }

    fn bytes(&self) -> usize {
        self.samples.capacity() * size_of::<(u64, i64)>()
    }
}

fn kll_reserved_bytes(k: usize, m: usize) -> usize {
    let mut scale = 1.0;
    let mut items = 0_usize;
    for _ in 0..61 {
        items += ((k as f64 * scale).ceil() as usize).max(m);
        scale *= 2.0 / 3.0;
    }
    items * size_of::<i64>() + 62 * size_of::<usize>() + k * size_of::<i64>()
}

fn average_query<T>(mut query: impl FnMut() -> T, repeats: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..repeats {
        black_box(query());
    }
    start.elapsed() / repeats
}

fn latency(value: Duration) -> String {
    if value.is_zero() {
        "<1ns".to_owned()
    } else {
        format!("{value:?}")
    }
}

fn print_query_latency(
    truth: &Truth,
    matched: &UnivMonQ,
    quality: &UnivMonQ,
    bundle: &SpecializedBundle,
) {
    let matched_view = matched.prepare_queries();
    let quality_view = quality.prepare_queries();
    let bundle_cdf = bundle.kll.cdf();
    let probe = truth.cdf_probes[truth.cdf_probes.len() / 2];
    let columns = |metric: &str, a: Duration, b: Duration, c: Duration| {
        println!(
            "  {metric:<16} {:>13} {:>13} {:>13}",
            latency(a),
            latency(b),
            latency(c)
        );
    };
    println!(
        "  {:<16} {:>13} {:>13} {:>13}",
        "metric", "Q-matched", "Q-quality", "bundle"
    );
    columns(
        "prepare",
        average_query(|| black_box(matched.prepare_queries()), QUERY_REPEATS),
        average_query(|| black_box(quality.prepare_queries()), QUERY_REPEATS),
        average_query(|| black_box(bundle.kll.cdf()), QUERY_REPEATS),
    );
    columns(
        "count",
        average_query(|| black_box(matched_view.count()), QUERY_REPEATS),
        average_query(|| black_box(quality_view.count()), QUERY_REPEATS),
        average_query(|| black_box(bundle.count), QUERY_REPEATS),
    );
    columns(
        "min/max",
        average_query(
            || black_box((matched_view.min(), matched_view.max())),
            QUERY_REPEATS,
        ),
        average_query(
            || black_box((quality_view.min(), quality_view.max())),
            QUERY_REPEATS,
        ),
        average_query(|| black_box((bundle.min, bundle.max)), QUERY_REPEATS),
    );
    columns(
        "point frequency",
        average_query(
            || black_box(matched_view.estimate_frequency(probe as f64)),
            QUERY_REPEATS,
        ),
        average_query(
            || black_box(quality_view.estimate_frequency(probe as f64)),
            QUERY_REPEATS,
        ),
        average_query(|| black_box(bundle.frequency(probe)), QUERY_REPEATS),
    );
    columns(
        "F0",
        average_query(
            || black_box(matched_view.estimate_distinct()),
            QUERY_REPEATS,
        ),
        average_query(
            || black_box(quality_view.estimate_distinct()),
            QUERY_REPEATS,
        ),
        average_query(|| black_box(bundle.hll.estimate()), QUERY_REPEATS),
    );
    columns(
        "F2",
        average_query(|| black_box(matched_view.estimate_f2()), QUERY_REPEATS),
        average_query(|| black_box(quality_view.estimate_f2()), QUERY_REPEATS),
        average_query(
            || black_box(bundle.frequencies.estimate_f2()),
            QUERY_REPEATS,
        ),
    );
    columns(
        "entropy",
        average_query(|| black_box(matched_view.estimate_entropy()), QUERY_REPEATS),
        average_query(|| black_box(quality_view.estimate_entropy()), QUERY_REPEATS),
        average_query(|| black_box(bundle.estimate_entropy()), QUERY_REPEATS),
    );
    columns(
        "top-20",
        average_query(
            || black_box(matched_view.heavy_hitters(TOP_K)),
            QUERY_REPEATS,
        ),
        average_query(
            || black_box(quality_view.heavy_hitters(TOP_K)),
            QUERY_REPEATS,
        ),
        average_query(|| black_box(bundle.heavy_hitters.top(TOP_K)), QUERY_REPEATS),
    );
    columns(
        "rank",
        average_query(|| black_box(matched_view.rank(probe as f64)), QUERY_REPEATS),
        average_query(|| black_box(quality_view.rank(probe as f64)), QUERY_REPEATS),
        average_query(
            || black_box(bundle_cdf.quantile(probe as f64)),
            QUERY_REPEATS,
        ),
    );
    columns(
        "7 quantiles",
        average_query(
            || black_box(matched_view.quantiles(&QUANTILES)),
            QUERY_REPEATS,
        ),
        average_query(
            || black_box(quality_view.quantiles(&QUANTILES)),
            QUERY_REPEATS,
        ),
        average_query(
            || black_box(QUANTILES.map(|q| bundle_cdf.query(q))),
            QUERY_REPEATS,
        ),
    );
    columns(
        "CDF",
        average_query(|| black_box(matched_view.cdf()), QUERY_REPEATS),
        average_query(|| black_box(quality_view.cdf()), QUERY_REPEATS),
        average_query(|| black_box(&bundle_cdf), QUERY_REPEATS),
    );
    columns(
        "all-task batch",
        average_query(|| query_all_q(matched, truth), QUERY_REPEATS),
        average_query(|| query_all_q(quality, truth), QUERY_REPEATS),
        average_query(|| query_all_bundle(bundle, truth), QUERY_REPEATS),
    );
}

fn query_all_q(sketch: &UnivMonQ, truth: &Truth) {
    let query = sketch.prepare_queries();
    black_box(query.count());
    black_box((query.min(), query.max()));
    black_box(query.estimate_frequency(truth.cdf_probes[50] as f64));
    black_box(query.estimate_distinct());
    black_box(query.estimate_f2());
    black_box(query.estimate_entropy());
    black_box(query.heavy_hitters(TOP_K));
    black_box(query.rank(truth.cdf_probes[50] as f64));
    black_box(query.quantiles(&QUANTILES));
    black_box(query.cdf());
}

fn query_all_bundle(bundle: &SpecializedBundle, truth: &Truth) {
    black_box(bundle.count);
    black_box((bundle.min, bundle.max));
    black_box(bundle.frequency(truth.cdf_probes[50]));
    black_box(bundle.hll.estimate());
    black_box(bundle.frequencies.estimate_f2());
    black_box(bundle.estimate_entropy());
    black_box(bundle.heavy_hitters.top(TOP_K));
    black_box(bundle.kll.rank(truth.cdf_probes[50] as f64));
    let cdf = bundle.kll.cdf();
    black_box(QUANTILES.map(|q| cdf.query(q)));
    black_box(cdf);
}

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }

    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn uniform(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| (rng.next() % 1_000_001) as i64 - 500_000)
        .collect()
}

fn normal(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng(seed);
    let mut values = Vec::with_capacity(n);
    while values.len() < n {
        let radius = (-2.0 * rng.unit().ln()).sqrt();
        let angle = std::f64::consts::TAU * rng.unit();
        values.push((100_000.0 * radius * angle.cos()).round() as i64);
        if values.len() < n {
            values.push((100_000.0 * radius * angle.sin()).round() as i64);
        }
    }
    values
}

fn exponential(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| (-rng.unit().ln() * 100_000.0).round() as i64)
        .collect()
}

fn zipf(n: usize, seed: u64) -> Vec<i64> {
    let domain = 20_000_usize;
    let mut cdf = Vec::with_capacity(domain);
    let mut total = 0.0;
    for rank in 1..=domain {
        total += 1.0 / (rank as f64).powf(1.1);
        cdf.push(total);
    }
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let target = rng.unit() * total;
            (cdf.partition_point(|value| *value < target) + 1) as i64
        })
        .collect()
}

fn elephants(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|index| match index % 25 {
            0..=3 => 0,
            4..=6 => 1_000,
            7..=8 => 100_000,
            9 => 1_000_000,
            _ => 100 + (rng.next() % 999_901) as i64,
        })
        .collect()
}
