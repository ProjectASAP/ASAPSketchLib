//! Compare two theory-oriented ordered-query extensions of UnivMon-Q.
//!
//! Mode 1 evaluates fixed ranks directly through the UnivMon recurrence.
//! Mode 2 keeps the same UnivMon core and adds a mergeable bottom-k sample of
//! stream occurrences. Every occurrence must have a globally unique ID.
//!
//! Run with:
//!
//!   cargo run --release --example compare_univmon_rank_modes -- 50000 5

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use asap_sketchlib::{UnivMonQ, UnivMonQConfig};

const OCCURRENCE_SIZES: [usize; 2] = [1_024, 4_096];
const HEAVY_SIGMAS: [f64; 4] = [1.0, 2.0, 4.0, 8.0];
const QUANTILES: [f64; 13] = [
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999,
];
const MERGE_FANOUT: usize = 8;
type Generator = fn(usize, u64) -> Vec<i64>;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("n must be positive"))
        .unwrap_or(50_000);
    let trials: usize = std::env::args()
        .nth(2)
        .map(|value| value.parse().expect("trials must be positive"))
        .unwrap_or(5);
    assert!(n > 0 && trials > 0);

    let workloads: [(&str, Generator); 5] = [
        ("uniform", uniform),
        ("exponential", exponential),
        ("bimodal", bimodal),
        ("zipf", zipf),
        ("elephants", elephants),
    ];

    println!("UnivMon ordered-query mode comparison: n={n}, trials={trials}");
    println!("Errors are normalized rank errors; p95 is across trials.");
    println!(
        "UM quantiles invert 2049 fixed value thresholds (not an arbitrary-domain guarantee)."
    );
    println!();

    for capacity in OCCURRENCE_SIZES {
        let epsilon = (f64::ln(2.0 / 0.01) / (2.0 * capacity as f64)).sqrt();
        println!(
            "Occurrence sample {capacity}: finite-sample Hoeffding/DKW target at delta=0.01 is {:.3}%",
            epsilon * 100.0
        );
    }

    println!();
    println!("accuracy and resource comparison");
    println!(
        "  {:<12} {:<10} {:>8} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "workload",
        "mode",
        "KiB",
        "update-ns",
        "fixed-p95",
        "CDF-p95",
        "Qmean-p95",
        "Qtail-p95",
        "query-us"
    );

    for (workload_index, (name, generator)) in workloads.iter().enumerate() {
        let mut pure_scores = Vec::with_capacity(trials);
        let mut pure_update = Vec::with_capacity(trials);
        let mut pure_query = Vec::with_capacity(trials);
        let mut pure_memory = 0;
        let mut occurrence_scores = vec![Vec::with_capacity(trials); OCCURRENCE_SIZES.len()];
        let mut occurrence_update = vec![Vec::with_capacity(trials); OCCURRENCE_SIZES.len()];
        let mut occurrence_query = vec![Vec::with_capacity(trials); OCCURRENCE_SIZES.len()];
        let mut occurrence_memory = vec![0; OCCURRENCE_SIZES.len()];

        for trial in 0..trials {
            let seed = trial_seed(workload_index, trial);
            let data = generator(n, seed);
            let truth = Truth::new(&data);
            let config = config(trial);

            let pure = build_pure(&data, config, 1);
            pure_memory = pure.estimated_memory_bytes();
            pure_scores.push(score_pure(&pure, &truth));
            pure_query.push(benchmark_pure_queries(&pure, &truth));

            for (index, &capacity) in OCCURRENCE_SIZES.iter().enumerate() {
                let (core, sample) = build_occurrence(&data, config, capacity, seed, 1);
                occurrence_memory[index] =
                    core.estimated_memory_bytes() + sample.estimated_memory_bytes();
                occurrence_scores[index].push(score_occurrence(&sample, &truth));
                occurrence_query[index].push(benchmark_occurrence_queries(&sample, &truth));
            }

            // Alternate measurement order across trials to reduce warm-cache
            // bias between the core-only and core-plus-sampler modes.
            if trial % 2 == 0 {
                pure_update.push(benchmark_pure_updates(&data, config));
                for (index, &capacity) in OCCURRENCE_SIZES.iter().enumerate() {
                    occurrence_update[index]
                        .push(benchmark_occurrence_updates(&data, config, capacity, seed));
                }
            } else {
                for (index, &capacity) in OCCURRENCE_SIZES.iter().enumerate().rev() {
                    occurrence_update[index]
                        .push(benchmark_occurrence_updates(&data, config, capacity, seed));
                }
                pure_update.push(benchmark_pure_updates(&data, config));
            }
        }

        print_row(
            name,
            "UM-fixed",
            pure_memory,
            &pure_update,
            &pure_scores,
            &pure_query,
        );
        for (index, &capacity) in OCCURRENCE_SIZES.iter().enumerate() {
            print_row(
                name,
                &format!("occ-{capacity}"),
                occurrence_memory[index],
                &occurrence_update[index],
                &occurrence_scores[index],
                &occurrence_query[index],
            );
        }
    }

    println!();
    println!("core-assisted occurrence-1024 sweep");
    println!("Heavy threshold = sigma * sqrt(estimated F2 / CountSketch width).");
    println!(
        "  {:<12} {:<10} {:>9} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "workload", "mode", "heavies", "fixed-p95", "CDF-p95", "Qmean-p95", "cold-us", "warm-us"
    );
    for (workload_index, (name, generator)) in workloads.iter().enumerate() {
        let mut raw_scores = Vec::with_capacity(trials);
        let mut raw_queries = Vec::with_capacity(trials);
        let mut assisted_scores = vec![Vec::with_capacity(trials); HEAVY_SIGMAS.len()];
        let mut assisted_queries = vec![Vec::with_capacity(trials); HEAVY_SIGMAS.len()];
        let mut heavy_counts = vec![Vec::with_capacity(trials); HEAVY_SIGMAS.len()];
        let mut adaptive_scores = Vec::with_capacity(trials);
        let mut adaptive_queries = Vec::with_capacity(trials);
        let mut adaptive_warm_queries = Vec::with_capacity(trials);
        let mut adaptive_heavies = Vec::with_capacity(trials);
        for trial in 0..trials {
            let seed = trial_seed(workload_index, trial);
            let data = generator(n, seed);
            let truth = Truth::new(&data);
            let (core, sample) = build_occurrence(&data, config(trial), 1_024, seed, 1);
            raw_scores.push(score_occurrence(&sample, &truth));
            raw_queries.push(benchmark_occurrence_queries(&sample, &truth));
            for (index, &sigma) in HEAVY_SIGMAS.iter().enumerate() {
                let query = AssistedOccurrenceQuery::new(&core, &sample, sigma);
                heavy_counts[index].push(query.heavy_count as f64);
                assisted_scores[index].push(score_assisted(&query, &truth));
                assisted_queries[index]
                    .push(benchmark_assisted_queries(&core, &sample, &truth, sigma));
            }
            let adaptive = AssistedOccurrenceQuery::new_adaptive(&core, &sample);
            adaptive_heavies.push(adaptive.heavy_count as f64);
            adaptive_scores.push(score_assisted(&adaptive, &truth));
            adaptive_queries.push(benchmark_adaptive_queries(&core, &sample, &truth));
            adaptive_warm_queries.push(benchmark_adaptive_warm_queries(&core, &sample, &truth));
        }
        print_assisted_row(name, "raw", &raw_scores, &raw_queries, None, None);
        for (index, &sigma) in HEAVY_SIGMAS.iter().enumerate() {
            print_assisted_row(
                name,
                &format!("sigma-{sigma:.0}"),
                &assisted_scores[index],
                &assisted_queries[index],
                Some(percentile(&heavy_counts[index], 0.5)),
                None,
            );
        }
        print_assisted_row(
            name,
            "adaptive",
            &adaptive_scores,
            &adaptive_queries,
            Some(percentile(&adaptive_heavies, 0.5)),
            Some(&adaptive_warm_queries),
        );
    }

    println!();
    println!("merge comparison: one pass versus {MERGE_FANOUT} shards");
    println!(
        "  {:<12} {:<10} {:>13} {:>14} {:>15}",
        "workload", "mode", "CDF-drift-p95", "Q-drift-p95", "sample-identical"
    );
    for (workload_index, (name, generator)) in workloads.iter().enumerate() {
        let mut pure_cdf = Vec::with_capacity(trials);
        let mut pure_quantile = Vec::with_capacity(trials);
        let mut occurrence_cdf = vec![Vec::with_capacity(trials); OCCURRENCE_SIZES.len()];
        let mut occurrence_quantile = vec![Vec::with_capacity(trials); OCCURRENCE_SIZES.len()];
        let mut adaptive_cdf = Vec::with_capacity(trials);
        let mut adaptive_quantile = Vec::with_capacity(trials);
        let mut identical = vec![true; OCCURRENCE_SIZES.len()];
        for trial in 0..trials {
            let seed = trial_seed(workload_index, trial);
            let data = generator(n, seed);
            let truth = Truth::new(&data);
            let config = config(trial);
            let pure_one = build_pure(&data, config, 1);
            let pure_merged = build_pure(&data, config, MERGE_FANOUT);
            let (cdf_drift, quantile_drift) = pure_drift(&pure_one, &pure_merged, &truth);
            pure_cdf.push(cdf_drift);
            pure_quantile.push(quantile_drift);

            for (index, &capacity) in OCCURRENCE_SIZES.iter().enumerate() {
                let (one_core, one) = build_occurrence(&data, config, capacity, seed, 1);
                let (merged_core, merged) =
                    build_occurrence(&data, config, capacity, seed, MERGE_FANOUT);
                identical[index] &= one.sorted_records() == merged.sorted_records();
                let (cdf_drift, quantile_drift) = occurrence_drift(&one, &merged, &truth);
                occurrence_cdf[index].push(cdf_drift);
                occurrence_quantile[index].push(quantile_drift);
                if capacity == 1_024 {
                    let one_query = AssistedOccurrenceQuery::new_adaptive(&one_core, &one);
                    let merged_query = AssistedOccurrenceQuery::new_adaptive(&merged_core, &merged);
                    let (cdf_drift, quantile_drift) =
                        assisted_drift(&one_query, &merged_query, &truth);
                    adaptive_cdf.push(cdf_drift);
                    adaptive_quantile.push(quantile_drift);
                }
            }
        }
        println!(
            "  {name:<12} {:<10} {:>13.5} {:>14.5} {:>15}",
            "UM-fixed",
            percentile(&pure_cdf, 0.95),
            percentile(&pure_quantile, 0.95),
            "n/a"
        );
        for (index, &capacity) in OCCURRENCE_SIZES.iter().enumerate() {
            println!(
                "  {name:<12} {:<10} {:>13.5} {:>14.5} {:>15}",
                format!("occ-{capacity}"),
                percentile(&occurrence_cdf[index], 0.95),
                percentile(&occurrence_quantile[index], 0.95),
                identical[index]
            );
        }
        println!(
            "  {name:<12} {:<10} {:>13.5} {:>14.5} {:>15}",
            "adaptive",
            percentile(&adaptive_cdf, 0.95),
            percentile(&adaptive_quantile, 0.95),
            identical[0]
        );
    }
}

fn print_assisted_row(
    workload: &str,
    mode: &str,
    scores: &[Score],
    queries: &[Duration],
    heavies: Option<f64>,
    warm_queries: Option<&[Duration]>,
) {
    let fixed: Vec<_> = scores.iter().map(|score| score.fixed_mean).collect();
    let cdf: Vec<_> = scores.iter().map(|score| score.cdf_max).collect();
    let quantile: Vec<_> = scores.iter().map(|score| score.quantile_mean).collect();
    let query_us: Vec<_> = queries
        .iter()
        .map(|duration| duration.as_secs_f64() * 1e6)
        .collect();
    let heavy_label = heavies.map_or_else(|| "-".to_owned(), |value| format!("{value:.0}"));
    let warm_label = warm_queries.map_or_else(
        || "-".to_owned(),
        |durations| {
            let values: Vec<_> = durations
                .iter()
                .map(|duration| duration.as_secs_f64() * 1e6)
                .collect();
            format!("{:.1}", percentile(&values, 0.5))
        },
    );
    println!(
        "  {workload:<12} {mode:<10} {heavy_label:>9} {:>11.5} {:>11.5} {:>11.5} {:>11.1} {warm_label:>11}",
        percentile(&fixed, 0.95),
        percentile(&cdf, 0.95),
        percentile(&quantile, 0.95),
        percentile(&query_us, 0.5),
    );
}

fn print_row(
    workload: &str,
    mode: &str,
    memory: usize,
    updates: &[f64],
    scores: &[Score],
    queries: &[Duration],
) {
    let fixed: Vec<_> = scores.iter().map(|score| score.fixed_mean).collect();
    let cdf: Vec<_> = scores.iter().map(|score| score.cdf_max).collect();
    let quantile: Vec<_> = scores.iter().map(|score| score.quantile_mean).collect();
    let tail: Vec<_> = scores.iter().map(|score| score.tail_max).collect();
    let query_us: Vec<_> = queries
        .iter()
        .map(|duration| duration.as_secs_f64() * 1e6)
        .collect();
    println!(
        "  {workload:<12} {mode:<10} {:>8.1} {:>10.1} {:>11.5} {:>11.5} {:>11.5} {:>11.5} {:>11.1}",
        memory as f64 / 1024.0,
        percentile(updates, 0.5),
        percentile(&fixed, 0.95),
        percentile(&cdf, 0.95),
        percentile(&quantile, 0.95),
        percentile(&tail, 0.95),
        percentile(&query_us, 0.5),
    );
}

fn config(trial: usize) -> UnivMonQConfig {
    UnivMonQConfig {
        levels: 12,
        width: 1_024,
        width_halving_period: 3,
        depth: 5,
        counter_bits: 32,
        candidates: 256,
        ordered_samples: 0,
        hash_seed: 5 + trial,
    }
}

fn build_pure(data: &[i64], config: UnivMonQConfig, fanout: usize) -> UnivMonQ {
    let fanout = fanout.min(data.len()).max(1);
    let mut shards = Vec::with_capacity(fanout);
    for shard in 0..fanout {
        let start = shard * data.len() / fanout;
        let end = (shard + 1) * data.len() / fanout;
        let mut sketch = UnivMonQ::new(config).unwrap();
        for value in &data[start..end] {
            sketch.add(value);
        }
        shards.push(sketch);
    }
    merge_cores(shards)
}

fn build_occurrence(
    data: &[i64],
    config: UnivMonQConfig,
    capacity: usize,
    seed: u64,
    fanout: usize,
) -> (UnivMonQ, OccurrenceBottomK) {
    let fanout = fanout.min(data.len()).max(1);
    let mut shards = Vec::with_capacity(fanout);
    for shard in 0..fanout {
        let start = shard * data.len() / fanout;
        let end = (shard + 1) * data.len() / fanout;
        let mut core = UnivMonQ::new(config).unwrap();
        let mut sample = OccurrenceBottomK::new(capacity, seed);
        for (offset, &value) in data[start..end].iter().enumerate() {
            core.add(&value);
            sample.update(value, (start + offset) as u64);
        }
        shards.push((core, sample));
    }
    while shards.len() > 1 {
        let mut merged = Vec::with_capacity(shards.len().div_ceil(2));
        let mut iter = shards.into_iter();
        while let Some((mut left_core, mut left_sample)) = iter.next() {
            if let Some((right_core, right_sample)) = iter.next() {
                left_core.merge(&right_core).unwrap();
                left_sample.merge(&right_sample).unwrap();
            }
            merged.push((left_core, left_sample));
        }
        shards = merged;
    }
    shards.pop().unwrap()
}

fn merge_cores(mut sketches: Vec<UnivMonQ>) -> UnivMonQ {
    while sketches.len() > 1 {
        let mut merged = Vec::with_capacity(sketches.len().div_ceil(2));
        let mut iter = sketches.into_iter();
        while let Some(mut left) = iter.next() {
            if let Some(right) = iter.next() {
                left.merge(&right).unwrap();
            }
            merged.push(left);
        }
        sketches = merged;
    }
    sketches.pop().unwrap()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Occurrence {
    priority: u64,
    id: u64,
    value: i64,
}

impl Ord for Occurrence {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.priority, self.id).cmp(&(other.priority, other.id))
    }
}

impl PartialOrd for Occurrence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
struct OccurrenceBottomK {
    capacity: usize,
    seed: u64,
    count: u64,
    heap: BinaryHeap<Occurrence>,
}

impl OccurrenceBottomK {
    fn new(capacity: usize, seed: u64) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            seed,
            count: 0,
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    fn update(&mut self, value: i64, id: u64) {
        self.count = self.count.saturating_add(1);
        self.retain(Occurrence {
            priority: mix64(id ^ self.seed),
            id,
            value,
        });
    }

    fn retain(&mut self, occurrence: Occurrence) {
        if self.heap.len() < self.capacity {
            self.heap.push(occurrence);
        } else if self
            .heap
            .peek()
            .is_some_and(|largest| occurrence < *largest)
        {
            self.heap.pop();
            self.heap.push(occurrence);
        }
    }

    fn merge(&mut self, other: &Self) -> Result<(), &'static str> {
        if self.capacity != other.capacity || self.seed != other.seed {
            return Err("occurrence samples have different configurations");
        }
        self.count = self.count.saturating_add(other.count);
        for &occurrence in &other.heap {
            self.retain(occurrence);
        }
        Ok(())
    }

    fn prepare(&self) -> OccurrenceQuery {
        let mut values: Vec<_> = self.heap.iter().map(|item| item.value).collect();
        values.sort_unstable();
        OccurrenceQuery { values }
    }

    fn sorted_records(&self) -> Vec<Occurrence> {
        let mut records = self.heap.clone().into_vec();
        records.sort_unstable();
        records
    }

    fn estimated_memory_bytes(&self) -> usize {
        self.capacity * size_of::<Occurrence>()
    }
}

struct OccurrenceQuery {
    values: Vec<i64>,
}

impl OccurrenceQuery {
    fn rank_fraction(&self, value: i64) -> f64 {
        self.values.partition_point(|item| *item <= value) as f64 / self.values.len() as f64
    }

    fn quantile(&self, q: f64) -> i64 {
        let index = ((q * self.values.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(self.values.len() - 1);
        self.values[index]
    }
}

/// Query-time post-stratification using high-confidence heavy values recovered
/// by the existing UnivMon core. No additional update-time state is required.
struct AssistedOccurrenceQuery {
    points: Vec<(i64, f64)>,
    min: i64,
    max: i64,
    heavy_count: usize,
}

impl AssistedOccurrenceQuery {
    fn new(core: &UnivMonQ, sample: &OccurrenceBottomK, sigma: f64) -> Self {
        let core_query = core.prepare_queries();
        let count = core.count() as f64;
        let f2 = core_query.estimate_f2();
        Self::from_prepared(core, sample, &core_query, sigma, f2, count)
    }

    fn new_adaptive(core: &UnivMonQ, sample: &OccurrenceBottomK) -> Self {
        let core_query = core.prepare_queries();
        Self::new_adaptive_from_prepared(core, sample, &core_query)
    }

    fn new_adaptive_from_prepared(
        core: &UnivMonQ,
        sample: &OccurrenceBottomK,
        core_query: &asap_sketchlib::UnivMonQQuery<'_>,
    ) -> Self {
        let count = core.count() as f64;
        let f2 = core_query.estimate_f2();
        let concentration = f2 / count.powi(2);
        let sigma = if concentration >= 1.0 / sample.capacity as f64 {
            1.0
        } else {
            f64::INFINITY
        };
        Self::from_prepared(core, sample, core_query, sigma, f2, count)
    }

    fn from_prepared(
        core: &UnivMonQ,
        sample: &OccurrenceBottomK,
        core_query: &asap_sketchlib::UnivMonQQuery<'_>,
        sigma: f64,
        f2: f64,
        count: f64,
    ) -> Self {
        let frequency_error_scale = (f2 / core.config().width as f64).sqrt();
        let threshold = sigma * frequency_error_scale;
        let mut heavy: BTreeMap<i64, f64> = core_query
            .heavy_hitters(64)
            .into_iter()
            .filter(|(_, frequency)| *frequency as f64 >= threshold)
            .map(|(value, frequency)| (value.round() as i64, frequency as f64))
            .collect();

        // Estimated heavy frequencies share the exact total-mass constraint.
        // Scaling is needed only if CountSketch overestimates collectively.
        let estimated_heavy_total: f64 = heavy.values().sum();
        if estimated_heavy_total > count {
            let scale = count / estimated_heavy_total;
            for frequency in heavy.values_mut() {
                *frequency *= scale;
            }
        }
        let heavy_total: f64 = heavy.values().sum();
        let residual_total = (count - heavy_total).max(0.0);
        let heavy_values: HashSet<_> = heavy.keys().copied().collect();
        let mut residual_values: Vec<_> = sample
            .heap
            .iter()
            .filter(|occurrence| !heavy_values.contains(&occurrence.value))
            .map(|occurrence| occurrence.value)
            .collect();
        residual_values.sort_unstable();

        // If sampling happens to retain no residual occurrence, retaining the
        // raw empirical distribution is safer than inventing its placement.
        if residual_values.is_empty() && residual_total > 0.0 {
            heavy.clear();
            residual_values = sample.heap.iter().map(|item| item.value).collect();
            residual_values.sort_unstable();
        }

        let residual_total = if heavy.is_empty() {
            count
        } else {
            residual_total
        };
        let residual_weight = if residual_values.is_empty() {
            0.0
        } else {
            residual_total / residual_values.len() as f64
        };
        let heavy_count = heavy.len();
        let mut weighted_values: Vec<(i64, f64)> = heavy.into_iter().collect();
        let mut residual = residual_values.into_iter().peekable();
        while let Some(value) = residual.next() {
            let mut copies = 1_usize;
            while residual.peek() == Some(&value) {
                residual.next();
                copies += 1;
            }
            weighted_values.push((value, residual_weight * copies as f64));
        }
        let min = core.min().unwrap().round() as i64;
        let max = core.max().unwrap().round() as i64;
        weighted_values.push((min, 0.0));
        weighted_values.push((max, 0.0));
        weighted_values.sort_unstable_by_key(|point| point.0);
        let mut combined: Vec<(i64, f64)> = Vec::with_capacity(weighted_values.len());
        for (value, weight) in weighted_values {
            if let Some(last) = combined.last_mut().filter(|last| last.0 == value) {
                last.1 += weight;
            } else {
                combined.push((value, weight));
            }
        }
        let mut running = 0.0;
        let mut points: Vec<_> = combined
            .into_iter()
            .map(|(value, weight)| {
                running += weight;
                (value, (running / count).clamp(0.0, 1.0))
            })
            .collect();
        if let Some(last) = points.last_mut() {
            last.1 = 1.0;
        }
        Self {
            points,
            min,
            max,
            heavy_count,
        }
    }

    fn rank_fraction(&self, value: i64) -> f64 {
        let index = self.points.partition_point(|point| point.0 <= value);
        if index == 0 {
            0.0
        } else {
            self.points[index - 1].1
        }
    }

    fn quantile(&self, q: f64) -> i64 {
        if q == 0.0 {
            return self.min;
        }
        if q == 1.0 {
            return self.max;
        }
        let index = self.points.partition_point(|point| point.1 < q);
        self.points[index.min(self.points.len() - 1)].0
    }
}

#[derive(Clone, Copy, Debug)]
struct Score {
    fixed_mean: f64,
    cdf_max: f64,
    quantile_mean: f64,
    tail_max: f64,
}

fn score_pure(sketch: &UnivMonQ, truth: &Truth) -> Score {
    let query = sketch.prepare_queries();
    let fixed_errors: Vec<_> = truth
        .probes
        .iter()
        .map(|&(value, exact)| {
            let estimate = query.estimate_rank_universal(value as f64).unwrap() as f64
                / truth.sorted.len() as f64;
            (estimate - exact).abs()
        })
        .collect();
    let grid = fixed_value_grid(truth, 2_049);
    let mut estimated_cdf: Vec<_> = grid
        .into_iter()
        .map(|value| {
            let rank = query.estimate_rank_universal(value as f64).unwrap() as f64
                / truth.sorted.len() as f64;
            (value, rank)
        })
        .collect();
    isotonicize(&mut estimated_cdf);
    let quantile_errors = quantile_errors_from_cdf(&estimated_cdf, truth);
    Score {
        fixed_mean: fixed_errors.iter().sum::<f64>() / fixed_errors.len() as f64,
        cdf_max: fixed_errors.into_iter().fold(0.0, f64::max),
        quantile_mean: quantile_errors.iter().sum::<f64>() / quantile_errors.len() as f64,
        tail_max: tail_max(&quantile_errors),
    }
}

fn score_occurrence(sample: &OccurrenceBottomK, truth: &Truth) -> Score {
    let query = sample.prepare();
    let fixed_errors: Vec<_> = truth
        .probes
        .iter()
        .map(|&(value, exact)| (query.rank_fraction(value) - exact).abs())
        .collect();
    let quantile_errors: Vec<_> = QUANTILES
        .iter()
        .map(|&q| truth.quantile_rank_error(q, query.quantile(q)))
        .collect();
    Score {
        fixed_mean: fixed_errors.iter().sum::<f64>() / fixed_errors.len() as f64,
        cdf_max: fixed_errors.into_iter().fold(0.0, f64::max),
        quantile_mean: quantile_errors.iter().sum::<f64>() / quantile_errors.len() as f64,
        tail_max: tail_max(&quantile_errors),
    }
}

fn score_assisted(query: &AssistedOccurrenceQuery, truth: &Truth) -> Score {
    let fixed_errors: Vec<_> = truth
        .probes
        .iter()
        .map(|&(value, exact)| (query.rank_fraction(value) - exact).abs())
        .collect();
    let quantile_errors: Vec<_> = QUANTILES
        .iter()
        .map(|&q| truth.quantile_rank_error(q, query.quantile(q)))
        .collect();
    Score {
        fixed_mean: fixed_errors.iter().sum::<f64>() / fixed_errors.len() as f64,
        cdf_max: fixed_errors.into_iter().fold(0.0, f64::max),
        quantile_mean: quantile_errors.iter().sum::<f64>() / quantile_errors.len() as f64,
        tail_max: tail_max(&quantile_errors),
    }
}

fn benchmark_pure_queries(sketch: &UnivMonQ, truth: &Truth) -> Duration {
    benchmark(|| {
        let query = black_box(sketch).prepare_queries();
        for &(value, _) in &truth.probes {
            black_box(query.estimate_rank_universal(black_box(value as f64)));
        }
        for value in fixed_value_grid(truth, 2_049) {
            black_box(query.estimate_rank_universal(black_box(value as f64)));
        }
    })
}

fn benchmark_pure_updates(data: &[i64], config: UnivMonQConfig) -> f64 {
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        let started = Instant::now();
        let sketch = build_pure(black_box(data), config, 1);
        let elapsed = started.elapsed();
        black_box(sketch.count());
        samples.push(elapsed.as_nanos() as f64 / data.len() as f64);
    }
    percentile(&samples, 0.5)
}

fn benchmark_occurrence_updates(
    data: &[i64],
    config: UnivMonQConfig,
    capacity: usize,
    seed: u64,
) -> f64 {
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        let started = Instant::now();
        let (core, sample) = build_occurrence(black_box(data), config, capacity, seed, 1);
        let elapsed = started.elapsed();
        black_box((core.count(), sample.heap.len()));
        samples.push(elapsed.as_nanos() as f64 / data.len() as f64);
    }
    percentile(&samples, 0.5)
}

fn benchmark_occurrence_queries(sample: &OccurrenceBottomK, truth: &Truth) -> Duration {
    benchmark(|| {
        let query = black_box(sample).prepare();
        for &(value, _) in &truth.probes {
            black_box(query.rank_fraction(black_box(value)));
        }
        for q in QUANTILES {
            black_box(query.quantile(black_box(q)));
        }
    })
}

fn benchmark_assisted_queries(
    core: &UnivMonQ,
    sample: &OccurrenceBottomK,
    truth: &Truth,
    sigma: f64,
) -> Duration {
    benchmark(|| {
        let query = AssistedOccurrenceQuery::new(black_box(core), black_box(sample), sigma);
        for &(value, _) in &truth.probes {
            black_box(query.rank_fraction(black_box(value)));
        }
        for q in QUANTILES {
            black_box(query.quantile(black_box(q)));
        }
    })
}

fn benchmark_adaptive_queries(
    core: &UnivMonQ,
    sample: &OccurrenceBottomK,
    truth: &Truth,
) -> Duration {
    benchmark(|| {
        let query = AssistedOccurrenceQuery::new_adaptive(black_box(core), black_box(sample));
        for &(value, _) in &truth.probes {
            black_box(query.rank_fraction(black_box(value)));
        }
        for q in QUANTILES {
            black_box(query.quantile(black_box(q)));
        }
    })
}

fn benchmark_adaptive_warm_queries(
    core: &UnivMonQ,
    sample: &OccurrenceBottomK,
    truth: &Truth,
) -> Duration {
    let core_query = core.prepare_queries();
    benchmark(|| {
        let query = AssistedOccurrenceQuery::new_adaptive_from_prepared(
            black_box(core),
            black_box(sample),
            black_box(&core_query),
        );
        for &(value, _) in &truth.probes {
            black_box(query.rank_fraction(black_box(value)));
        }
        for q in QUANTILES {
            black_box(query.quantile(black_box(q)));
        }
    })
}

fn benchmark(mut operation: impl FnMut()) -> Duration {
    let mut elapsed = Vec::with_capacity(5);
    for _ in 0..5 {
        let started = Instant::now();
        operation();
        elapsed.push(started.elapsed());
    }
    elapsed.sort_unstable();
    elapsed[elapsed.len() / 2]
}

fn pure_drift(left: &UnivMonQ, right: &UnivMonQ, truth: &Truth) -> (f64, f64) {
    let left_query = left.prepare_queries();
    let right_query = right.prepare_queries();
    let cdf = truth
        .probes
        .iter()
        .map(|&(value, _)| {
            let left = left_query.estimate_rank_universal(value as f64).unwrap() as f64;
            let right = right_query.estimate_rank_universal(value as f64).unwrap() as f64;
            (left - right).abs() / truth.sorted.len() as f64
        })
        .fold(0.0, f64::max);
    let left_cdf = universal_grid(&left_query, truth);
    let right_cdf = universal_grid(&right_query, truth);
    let quantile = QUANTILES
        .iter()
        .map(|&q| {
            let left = invert_cdf(&left_cdf, q);
            let right = invert_cdf(&right_cdf, q);
            truth.rank_interval_distance(left, right)
        })
        .fold(0.0, f64::max);
    (cdf, quantile)
}

fn occurrence_drift(
    left: &OccurrenceBottomK,
    right: &OccurrenceBottomK,
    truth: &Truth,
) -> (f64, f64) {
    let left = left.prepare();
    let right = right.prepare();
    let cdf = truth
        .probes
        .iter()
        .map(|&(value, _)| (left.rank_fraction(value) - right.rank_fraction(value)).abs())
        .fold(0.0, f64::max);
    let quantile = QUANTILES
        .iter()
        .map(|&q| truth.rank_interval_distance(left.quantile(q), right.quantile(q)))
        .fold(0.0, f64::max);
    (cdf, quantile)
}

fn assisted_drift(
    left: &AssistedOccurrenceQuery,
    right: &AssistedOccurrenceQuery,
    truth: &Truth,
) -> (f64, f64) {
    let cdf = truth
        .probes
        .iter()
        .map(|&(value, _)| (left.rank_fraction(value) - right.rank_fraction(value)).abs())
        .fold(0.0, f64::max);
    let quantile = QUANTILES
        .iter()
        .map(|&q| truth.rank_interval_distance(left.quantile(q), right.quantile(q)))
        .fold(0.0, f64::max);
    (cdf, quantile)
}

fn universal_grid(query: &asap_sketchlib::UnivMonQQuery<'_>, truth: &Truth) -> Vec<(i64, f64)> {
    let mut points: Vec<_> = fixed_value_grid(truth, 2_049)
        .into_iter()
        .map(|value| {
            (
                value,
                query.estimate_rank_universal(value as f64).unwrap() as f64
                    / truth.sorted.len() as f64,
            )
        })
        .collect();
    isotonicize(&mut points);
    points
}

fn quantile_errors_from_cdf(cdf: &[(i64, f64)], truth: &Truth) -> Vec<f64> {
    QUANTILES
        .iter()
        .map(|&q| truth.quantile_rank_error(q, invert_cdf(cdf, q)))
        .collect()
}

fn invert_cdf(cdf: &[(i64, f64)], q: f64) -> i64 {
    let index = cdf.partition_point(|point| point.1 < q);
    cdf[index.min(cdf.len() - 1)].0
}

fn fixed_value_grid(truth: &Truth, points: usize) -> Vec<i64> {
    let min = truth.sorted[0] as i128;
    let span = truth.sorted[truth.sorted.len() - 1] as i128 - min;
    (0..points)
        .map(|index| (min + span * index as i128 / (points - 1) as i128) as i64)
        .collect()
}

fn isotonicize(points: &mut [(i64, f64)]) {
    #[derive(Clone, Copy)]
    struct Block {
        start: usize,
        end: usize,
        sum: f64,
        count: usize,
    }
    let mut blocks: Vec<Block> = Vec::new();
    for (index, &(_, rank)) in points.iter().enumerate() {
        blocks.push(Block {
            start: index,
            end: index + 1,
            sum: rank,
            count: 1,
        });
        while blocks.len() >= 2 {
            let right = blocks[blocks.len() - 1];
            let left = blocks[blocks.len() - 2];
            if left.sum / left.count as f64 <= right.sum / right.count as f64 {
                break;
            }
            blocks.pop();
            blocks.pop();
            blocks.push(Block {
                start: left.start,
                end: right.end,
                sum: left.sum + right.sum,
                count: left.count + right.count,
            });
        }
    }
    for block in blocks {
        let rank = (block.sum / block.count as f64).clamp(0.0, 1.0);
        for point in &mut points[block.start..block.end] {
            point.1 = rank;
        }
    }
}

fn tail_max(errors: &[f64]) -> f64 {
    errors
        .iter()
        .enumerate()
        .filter(|(index, _)| *index <= 2 || *index >= errors.len() - 3)
        .map(|(_, &error)| error)
        .fold(0.0, f64::max)
}

struct Truth {
    sorted: Vec<i64>,
    probes: Vec<(i64, f64)>,
}

impl Truth {
    fn new(data: &[i64]) -> Self {
        let mut sorted = data.to_vec();
        sorted.sort_unstable();
        let mut probes: Vec<_> = (0..=1_000)
            .map(|index| sorted[index * (sorted.len() - 1) / 1_000])
            .map(|value| {
                let rank =
                    sorted.partition_point(|item| *item <= value) as f64 / sorted.len() as f64;
                (value, rank)
            })
            .collect();
        probes.dedup_by_key(|point| point.0);
        Self { sorted, probes }
    }

    fn quantile_rank_error(&self, q: f64, estimate: i64) -> f64 {
        let target = (q * self.sorted.len() as f64).ceil().max(1.0) as usize;
        let lower = self.sorted.partition_point(|item| *item < estimate);
        let upper = self.sorted.partition_point(|item| *item <= estimate);
        if target < lower {
            (lower - target) as f64 / self.sorted.len() as f64
        } else {
            target.saturating_sub(upper) as f64 / self.sorted.len() as f64
        }
    }

    fn rank_interval_distance(&self, left: i64, right: i64) -> f64 {
        let left_rank = self.sorted.partition_point(|item| *item <= left);
        let right_rank = self.sorted.partition_point(|item| *item <= right);
        left_rank.abs_diff(right_rank) as f64 / self.sorted.len() as f64
    }
}

fn percentile(values: &[f64], q: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_unstable_by(f64::total_cmp);
    values[((q * values.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1)]
}

fn trial_seed(workload: usize, trial: usize) -> u64 {
    0x2026_0804_u64 ^ ((workload as u64) << 32) ^ trial as u64
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

fn exponential(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| (-rng.unit().ln() * 100_000.0).round() as i64)
        .collect()
}

fn bimodal(n: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|index| {
            let center = if index % 2 == 0 {
                -1_000_000.0
            } else {
                1_000_000.0
            };
            let radius = (-2.0 * rng.unit().ln()).sqrt();
            let angle = std::f64::consts::TAU * rng.unit();
            (center + 20_000.0 * radius * angle.cos()).round() as i64
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occurrence_sample_is_exact_below_capacity() {
        let mut sample = OccurrenceBottomK::new(8, 7);
        for (id, value) in [9, 1, 5, 5].into_iter().enumerate() {
            sample.update(value, id as u64);
        }
        let query = sample.prepare();
        assert_eq!(query.values, vec![1, 5, 5, 9]);
        assert_eq!(query.rank_fraction(5), 0.75);
        assert_eq!(query.quantile(0.5), 5);
    }

    #[test]
    fn occurrence_merge_matches_one_pass_exactly() {
        let data: Vec<_> = (0..10_000).map(|index| (index * 17 % 991) as i64).collect();
        let config = config(0);
        let (_, one) = build_occurrence(&data, config, 256, 99, 1);
        let (_, merged) = build_occurrence(&data, config, 256, 99, 16);
        assert_eq!(one.count, merged.count);
        assert_eq!(one.sorted_records(), merged.sorted_records());
    }

    #[test]
    fn adaptive_assistance_is_raw_on_a_diffuse_stream() {
        let data: Vec<_> = (0..10_000).map(i64::from).collect();
        let (core, sample) = build_occurrence(&data, config(0), 256, 99, 1);
        let raw = sample.prepare();
        let assisted = AssistedOccurrenceQuery::new_adaptive(&core, &sample);
        assert_eq!(assisted.heavy_count, 0);
        for value in [0, 100, 1_000, 5_000, 9_999] {
            assert_eq!(assisted.rank_fraction(value), raw.rank_fraction(value));
        }
        for q in [0.01, 0.5, 0.99] {
            assert_eq!(assisted.quantile(q), raw.quantile(q));
        }
    }

    #[test]
    fn adaptive_assistance_uses_core_on_a_concentrated_stream() {
        let data: Vec<_> = (0..10_000)
            .map(|index| if index < 5_000 { 0 } else { index as i64 })
            .collect();
        let (core, sample) = build_occurrence(&data, config(0), 256, 99, 1);
        let assisted = AssistedOccurrenceQuery::new_adaptive(&core, &sample);
        assert!(assisted.heavy_count >= 1);
        assert_eq!(assisted.quantile(0.25), 0);
    }
}
