//! Large synthetic skew sweep for the experimental UnivMon-Q sketch.
//!
//! The evaluator checks all advertised query families against exact answers,
//! verifies merge-order and native-wire behavior, and measures update, merge,
//! prepared-query construction, and warm batch-query latency.
//!
//! Run with:
//!
//!   cargo run --release --example evaluate_univmon_q_skew -- 1000000 5 100000 256
//!   cargo run --release --example evaluate_univmon_q_skew -- 1000000 5 100000 256 all equal-memory

use std::collections::HashSet;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use asap_sketchlib::{
    BOTTOM_LAYER_FINDER, DataInput, DefaultXxHasher, HeapItem, UnivMon, UnivMonQ, UnivMonQConfig,
    hash64_seeded,
};

const SKEWS: [f64; 7] = [0.0, 0.5, 0.9, 1.1, 1.3, 1.6, 2.0];
const QUANTILES: [f64; 13] = [
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999,
];
const SHARDS: usize = 8;
const TOP_K: usize = 20;
const PREPARE_REPEATS: usize = 9;
const WARM_QUERY_REPEATS: usize = 200;
const UM_QUERY_REPEATS: usize = 9;

fn main() {
    let n: usize = argument(1, 1_000_000, "n");
    let trials: usize = argument(2, 5, "trials");
    let domain: usize = argument(3, 100_000, "domain");
    let candidates: usize = argument(4, 256, "candidates");
    let skew_filter = std::env::args().nth(5).and_then(|value| {
        (value != "all").then(|| {
            value
                .parse::<usize>()
                .expect("skew index must be 0..6 or `all`")
        })
    });
    let profile = std::env::args()
        .nth(6)
        .unwrap_or_else(|| "compact".to_string());
    assert!(n > 0 && trials > 0 && domain > 1 && candidates > 0);
    assert!(skew_filter.is_none_or(|index| index < SKEWS.len()));
    assert!(
        matches!(profile.as_str(), "compact" | "equal-memory"),
        "profile must be `compact` or `equal-memory`"
    );

    let mut base_config = UnivMonQConfig {
        levels: 12,
        width: 4_096,
        width_halving_period: 3,
        depth: 5,
        counter_bits: 32,
        candidates,
        ordered_samples: 4_096,
        hash_seed: 5,
    }
    .with_window_bound(n as u64, 1e-9)
    .expect("the evaluation window must fit within 63 levels");
    let um_candidates = candidates;
    let um_counter_bytes =
        base_config.levels * base_config.depth * base_config.width * size_of::<i64>();
    if profile == "equal-memory" {
        base_config.width_halving_period = 0;
        base_config.candidates = largest_candidate_budget(base_config, um_counter_bytes);
    }
    let q_reserved_bytes = UnivMonQ::new(base_config)
        .expect("evaluation config must be valid")
        .estimated_memory_bytes();

    println!("UnivMon versus UnivMon-Q large synthetic skew evaluation");
    println!("n={n}, trials={trials}, domain={domain}, shards={SHARDS}");
    println!("profile={profile}");
    println!(
        "config: levels={}, width={}, halve/{} levels, depth={}, candidates={}, ordered_samples={}",
        base_config.levels,
        base_config.width,
        base_config.width_halving_period,
        base_config.depth,
        base_config.candidates,
        base_config.ordered_samples,
    );
    println!(
        "memory budget: Q reserved={:.3} MiB; UM counters={:.3} MiB before heap metadata (UM candidates={um_candidates})",
        q_reserved_bytes as f64 / (1024.0 * 1024.0),
        um_counter_bytes as f64 / (1024.0 * 1024.0),
    );
    println!("errors are relative except point nRMSE and normalized rank errors");

    let mut summaries = Vec::with_capacity(SKEWS.len());
    for (skew_index, skew) in SKEWS
        .into_iter()
        .enumerate()
        .filter(|(index, _)| skew_filter.is_none_or(|filter| *index == filter))
    {
        let sampler = ZipfSampler::new(domain, skew);
        let mut results = Vec::with_capacity(trials);
        for trial in 0..trials {
            let seed = trial_seed(skew_index, trial);
            let data = sampler.generate(n, seed);
            let truth = Truth::new(&data, domain);
            let result = evaluate_trial(&data, &truth, base_config, um_candidates, seed);
            eprintln!(
                "completed alpha={skew:.1} trial={}/{}: UM/Q {:.1}/{:.1} ns/update, Q CDF max={:.5}",
                trial + 1,
                trials,
                result.um_update_ns,
                result.q_update_ns,
                result.q_accuracy.cdf_max,
            );
            results.push(result);
        }
        summaries.push(Summary::new(skew, &results));
    }

    println!();
    println!("shared-metric correctness after an 8-way merge (p95 across trials; UM/Q)");
    println!(
        "  {:>5} {:>9} {:>17} {:>17} {:>17} {:>17} {:>17}",
        "alpha", "distinct", "point nRMSE", "F0 rel", "F2 rel", "H rel", "L1 rel"
    );
    for summary in &summaries {
        println!(
            "  {:>5.1} {:>9.0} {:>8.5}/{:<8.5} {:>8.5}/{:<8.5} {:>8.5}/{:<8.5} {:>8.5}/{:<8.5} {:>8.5}/{:<8.5}",
            summary.skew,
            summary.distinct_p50,
            summary.um_point_p95,
            summary.point_p95,
            summary.um_f0_p95,
            summary.f0_p95,
            summary.um_f2_p95,
            summary.f2_p95,
            summary.um_entropy_p95,
            summary.entropy_p95,
            summary.um_l1_p95,
            summary.l1_p95,
        );
    }

    println!();
    println!("ordered and heavy-hitter correctness after an 8-way merge");
    println!(
        "  {:>5} {:>17} {:>15} {:>15} {:>13} {:>15}",
        "alpha", "HH mass UM/Q", "Q rank mean", "Q tail max", "CDF max", "violations UM/Q"
    );
    for summary in &summaries {
        println!(
            "  {:>5.1} {:>8.5}/{:<8.5} {:>7.5}/{:<7.5} {:>7.5}/{:<7.5} {:>6.5}/{:<6.5} {:>7}/{:<7}",
            summary.skew,
            summary.um_hh_mass_p05,
            summary.hh_mass_p05,
            summary.quantile_p50,
            summary.quantile_p95,
            summary.tail_p50,
            summary.tail_p95,
            summary.cdf_p50,
            summary.cdf_p95,
            summary.um_violations,
            summary.q_violations,
        );
    }

    println!();
    println!("entropy-error context (entropy is measured in nats)");
    println!(
        "  {:>5} {:>17} {:>19} {:>19}",
        "alpha", "exact H p50", "absolute UM/Q", "relative UM/Q"
    );
    for summary in &summaries {
        println!(
            "  {:>5.1} {:>17.5} {:>9.5}/{:<9.5} {:>9.5}/{:<9.5}",
            summary.skew,
            summary.entropy_exact_p50,
            summary.um_entropy_abs_p95,
            summary.entropy_abs_p95,
            summary.um_entropy_p95,
            summary.entropy_p95,
        );
    }

    println!();
    println!("performance (median / p95 across trials; UM-terminal/Q)");
    println!(
        "  {:>5} {:>19} {:>19} {:>19} {:>19} {:>19}",
        "alpha", "UM update m/p95", "Q update m/p95", "merge UM/Q", "query UM/Q", "wire KiB UM/Q"
    );
    for summary in &summaries {
        println!(
            "  {:>5.1} {:>8.1}/{:<8.1} {:>8.1}/{:<8.1} {:>8.1}/{:<8.1} {:>8.1}/{:<8.1} {:>8.1}/{:<8.1}",
            summary.skew,
            summary.um_update_p50,
            summary.um_update_p95,
            summary.update_p50,
            summary.update_p95,
            summary.um_merge_p50_us,
            summary.merge_p50_us,
            summary.um_query_p50_us,
            summary.prepare_p50_us + summary.warm_p50_ns / 1_000.0,
            summary.um_wire_kib,
            summary.q_wire_kib,
        );
    }

    println!();
    println!("additional performance context");
    println!(
        "  Q reserved memory={:.3} MiB; UnivMon counters alone={:.3} MiB before heap metadata.",
        summaries[0].memory_mib,
        (base_config.levels * base_config.depth * base_config.width * size_of::<i64>()) as f64
            / (1024.0 * 1024.0),
    );
    println!(
        "  Q prepared warm all-task batch={:.2}–{:.2} us median.",
        summaries
            .iter()
            .map(|summary| summary.warm_p50_ns / 1_000.0)
            .fold(f64::INFINITY, f64::min),
        summaries
            .iter()
            .map(|summary| summary.warm_p50_ns / 1_000.0)
            .fold(0.0, f64::max),
    );
    println!("  UnivMon uses terminal-only `fast_insert`; cumulative-layer `insert` is excluded.");

    let q_violations: usize = summaries.iter().map(|summary| summary.q_violations).sum();
    let um_violations: usize = summaries.iter().map(|summary| summary.um_violations).sum();
    let um_warnings: usize = summaries.iter().map(|summary| summary.um_warnings).sum();
    println!();
    println!("verification checks: UnivMon={um_violations}, UnivMon-Q={q_violations} violation(s)");
    println!("  UnivMon merge-order heavy-hitter tie warnings: {um_warnings}.");
    println!(
        "  Checks cover exact counts, finite estimates, Q extrema and monotone ordered queries,"
    );
    println!("  merge-order query equivalence, and native MessagePack round-trip equivalence.");
    println!("  HH mass is true mass covered by returned keys / exact top-{TOP_K} mass.");
    println!("  Q tail max covers p0.1/p0.5/p1/p99/p99.5/p99.9 rank error.");
}

fn argument(index: usize, default: usize, name: &str) -> usize {
    std::env::args()
        .nth(index)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be positive"))
        })
        .unwrap_or(default)
}

fn largest_candidate_budget(config: UnivMonQConfig, target_bytes: usize) -> usize {
    let mut low = config.candidates;
    let mut high = config.candidates;
    assert!(
        estimated_q_memory(config, low) <= target_bytes,
        "initial UnivMon-Q configuration exceeds the equal-memory target"
    );
    while estimated_q_memory(config, high) <= target_bytes {
        low = high;
        high = high.checked_mul(2).expect("candidate search overflowed");
    }
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if estimated_q_memory(config, middle) <= target_bytes {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn estimated_q_memory(mut config: UnivMonQConfig, candidates: usize) -> usize {
    config.candidates = candidates;
    UnivMonQ::new(config)
        .expect("candidate search must preserve a valid config")
        .estimated_memory_bytes()
}

fn evaluate_trial(
    data: &[i64],
    truth: &Truth,
    config: UnivMonQConfig,
    um_candidates: usize,
    seed: u64,
) -> TrialResult {
    let update_start = Instant::now();
    let q_shards = build_shards(data, config, seed);
    let q_update_time = update_start.elapsed();

    let update_start = Instant::now();
    let um_shards = build_um_shards(data, config, um_candidates);
    let um_update_time = update_start.elapsed();

    let mut q_tree_input = q_shards.clone();
    let merge_start = Instant::now();
    let q_merged_tree = merge_tree(&mut q_tree_input);
    let q_merge_time = merge_start.elapsed();
    let q_merged_left = merge_left(q_shards);

    let mut um_tree_input = um_shards.clone();
    let merge_start = Instant::now();
    let um_merged_tree = merge_um_tree(&mut um_tree_input);
    let um_merge_time = merge_start.elapsed();
    let um_merged_left = merge_um_left(um_shards);

    let q_accuracy = score(&q_merged_tree, truth);
    let um_accuracy = score_um(&um_merged_tree, truth);
    let mut q_violations = validate(&q_merged_tree, truth);
    let mut um_violations = validate_um(&um_merged_tree, truth);
    let mut um_warnings = 0;
    if !equivalent_queries(&q_merged_tree, &q_merged_left, truth) {
        q_violations += 1;
    }
    let um_merge_mismatches = um_query_mismatches(&um_merged_tree, &um_merged_left, truth);
    if !um_merge_mismatches.is_empty() {
        if um_merge_mismatches == ["heavy hitters"] {
            eprintln!(
                "UM merge-order heavy-hitter tie warning seed={seed:#x}: identity set differs"
            );
            um_warnings += 1;
        } else {
            eprintln!("UM merge-order mismatch seed={seed:#x}: {um_merge_mismatches:?}");
            um_violations += 1;
        }
    }

    let q_wire = q_merged_tree.serialize_to_bytes().unwrap();
    let q_decoded = UnivMonQ::<DefaultXxHasher>::deserialize_from_bytes(&q_wire).unwrap();
    if !equivalent_queries(&q_merged_tree, &q_decoded, truth) {
        q_violations += 1;
    }
    let um_wire = um_merged_tree.serialize_to_bytes().unwrap();
    let um_decoded = UnivMon::deserialize_from_bytes(&um_wire).unwrap();
    let um_wire_mismatches = um_query_mismatches(&um_merged_tree, &um_decoded, truth);
    if !um_wire_mismatches.is_empty() {
        eprintln!("UM wire mismatch seed={seed:#x}: {um_wire_mismatches:?}");
        um_violations += 1;
    }

    let prepare_ns = benchmark_prepare(&q_merged_tree);
    let warm_ns = benchmark_warm_queries(&q_merged_tree, truth);
    let um_query_us = benchmark_um_queries(&um_merged_tree, truth) / 1_000.0;
    TrialResult {
        q_accuracy,
        um_accuracy,
        distinct: truth.distinct as f64,
        entropy_exact: truth.entropy,
        q_update_ns: q_update_time.as_secs_f64() * 1e9 / data.len() as f64,
        um_update_ns: um_update_time.as_secs_f64() * 1e9 / data.len() as f64,
        q_merge_us: q_merge_time.as_secs_f64() * 1e6,
        um_merge_us: um_merge_time.as_secs_f64() * 1e6,
        q_prepare_us: prepare_ns / 1_000.0,
        q_warm_ns: warm_ns,
        um_query_us,
        q_memory_mib: q_merged_tree.estimated_memory_bytes() as f64 / (1024.0 * 1024.0),
        q_wire_kib: q_wire.len() as f64 / 1024.0,
        um_wire_kib: um_wire.len() as f64 / 1024.0,
        q_violations,
        um_violations,
        um_warnings,
    }
}

fn new_univmon(config: UnivMonQConfig, candidates: usize) -> UnivMon {
    UnivMon::init_univmon(candidates, config.depth, config.width, config.levels)
}

fn build_shards(data: &[i64], config: UnivMonQConfig, seed: u64) -> Vec<UnivMonQ> {
    (0..SHARDS)
        .map(|shard| {
            let start = shard * data.len() / SHARDS;
            let end = (shard + 1) * data.len() / SHARDS;
            let mut sketch =
                UnivMonQ::new_with_source_id(config, source_id(seed, shard + 1)).unwrap();
            for value in &data[start..end] {
                sketch.add(value);
            }
            sketch
        })
        .collect()
}

fn merge_tree(sketches: &mut Vec<UnivMonQ>) -> UnivMonQ {
    while sketches.len() > 1 {
        let mut merged = Vec::with_capacity(sketches.len().div_ceil(2));
        let mut iter = std::mem::take(sketches).into_iter();
        while let Some(mut left) = iter.next() {
            if let Some(right) = iter.next() {
                left.merge(&right).unwrap();
            }
            merged.push(left);
        }
        *sketches = merged;
    }
    sketches.pop().unwrap()
}

fn merge_left(mut sketches: Vec<UnivMonQ>) -> UnivMonQ {
    let mut result = sketches.remove(0);
    for sketch in sketches {
        result.merge(&sketch).unwrap();
    }
    result
}

fn build_um_shards(data: &[i64], config: UnivMonQConfig, candidates: usize) -> Vec<UnivMon> {
    (0..SHARDS)
        .map(|shard| {
            let start = shard * data.len() / SHARDS;
            let end = (shard + 1) * data.len() / SHARDS;
            let mut sketch = new_univmon(config, candidates);
            for &value in &data[start..end] {
                sketch.fast_insert(&DataInput::U64(value as u64), 1);
            }
            sketch
        })
        .collect()
}

fn merge_um_tree(sketches: &mut Vec<UnivMon>) -> UnivMon {
    while sketches.len() > 1 {
        let mut merged = Vec::with_capacity(sketches.len().div_ceil(2));
        let mut iter = std::mem::take(sketches).into_iter();
        while let Some(mut left) = iter.next() {
            if let Some(right) = iter.next() {
                left.merge(&right);
            }
            merged.push(left);
        }
        *sketches = merged;
    }
    sketches.pop().unwrap()
}

fn merge_um_left(mut sketches: Vec<UnivMon>) -> UnivMon {
    let mut result = sketches.remove(0);
    for sketch in sketches {
        result.merge(&sketch);
    }
    result
}

fn benchmark_prepare(sketch: &UnivMonQ) -> f64 {
    let mut durations = Vec::with_capacity(PREPARE_REPEATS);
    for _ in 0..PREPARE_REPEATS {
        let start = Instant::now();
        black_box(sketch.prepare_queries());
        durations.push(start.elapsed());
    }
    duration_percentile(&durations, 0.5).as_secs_f64() * 1e9
}

fn benchmark_warm_queries(sketch: &UnivMonQ, truth: &Truth) -> f64 {
    let query = sketch.prepare_queries();
    let probe = truth.domain / 2;
    let start = Instant::now();
    for _ in 0..WARM_QUERY_REPEATS {
        black_box(query.count());
        black_box((query.min(), query.max()));
        black_box(query.estimate_frequency(black_box(probe as f64)));
        black_box(query.estimate_distinct());
        black_box(query.estimate_f2());
        black_box(query.estimate_entropy());
        black_box(query.estimate_g_sum(|frequency| frequency));
        black_box(query.heavy_hitters(TOP_K));
        black_box(query.rank(black_box(probe as f64)));
        black_box(query.quantiles(&QUANTILES));
        black_box(query.cdf());
    }
    start.elapsed().as_secs_f64() * 1e9 / WARM_QUERY_REPEATS as f64
}

fn benchmark_um_queries(sketch: &UnivMon, truth: &Truth) -> f64 {
    let mut durations = Vec::with_capacity(UM_QUERY_REPEATS);
    for _ in 0..UM_QUERY_REPEATS {
        let start = Instant::now();
        black_box(sketch.bucket_size);
        black_box(univmon_frequency(sketch, truth.domain / 2));
        black_box(sketch.calc_card());
        black_box(sketch.calc_l2());
        black_box(sketch.calc_entropy());
        black_box(sketch.calc_l1());
        black_box(univmon_heavy_hitters(sketch, TOP_K));
        durations.push(start.elapsed());
    }
    duration_percentile(&durations, 0.5).as_secs_f64() * 1e9
}

#[derive(Clone, Copy)]
struct Accuracy {
    point_nrmse: f64,
    f0_rel: f64,
    f2_rel: f64,
    entropy_rel: f64,
    entropy_abs: f64,
    l1_rel: f64,
    hh_mass: f64,
    quantile_mean: f64,
    tail_max: f64,
    cdf_max: f64,
}

fn score(sketch: &UnivMonQ, truth: &Truth) -> Accuracy {
    let query = sketch.prepare_queries();
    let mut squared_error = 0.0;
    let mut point_count = 0;
    for key in 1..=truth.domain {
        let estimate = query.estimate_frequency(key as f64) as f64;
        squared_error += (estimate - truth.counts[key] as f64).powi(2);
        point_count += 1;
    }
    for key in truth.domain + 1..=truth.domain + 256 {
        squared_error += (query.estimate_frequency(key as f64) as f64).powi(2);
        point_count += 1;
    }

    let quantile_errors: Vec<f64> = QUANTILES
        .iter()
        .zip(query.quantiles(&QUANTILES))
        .map(|(&q, estimate)| truth.quantile_rank_error(q, estimate.unwrap().round() as usize))
        .collect();
    let cdf_max = (1..=truth.domain)
        .map(|value| {
            let estimate = query.rank(value as f64).unwrap() as f64 / truth.n as f64;
            (estimate - truth.cdf(value)).abs()
        })
        .fold(0.0, f64::max);
    let heavy_keys: HashSet<usize> = query
        .heavy_hitters(TOP_K)
        .into_iter()
        .map(|(value, _)| value.round() as usize)
        .collect();
    let captured_mass: u64 = heavy_keys
        .iter()
        .filter_map(|key| truth.counts.get(*key))
        .sum();
    let l1 = query.estimate_l1();

    let estimated_entropy = query.estimate_entropy();
    Accuracy {
        point_nrmse: (squared_error / point_count as f64).sqrt() / truth.f2.sqrt().max(1.0),
        f0_rel: relative_error(query.estimate_distinct(), truth.distinct as f64),
        f2_rel: relative_error(query.estimate_f2(), truth.f2),
        entropy_rel: relative_error(estimated_entropy, truth.entropy),
        entropy_abs: (estimated_entropy - truth.entropy).abs(),
        l1_rel: relative_error(l1, truth.n as f64),
        hh_mass: (captured_mass as f64 / truth.top_mass.max(1) as f64).min(1.0),
        quantile_mean: quantile_errors.iter().sum::<f64>() / quantile_errors.len() as f64,
        tail_max: quantile_errors
            .iter()
            .enumerate()
            .filter(|(index, _)| matches!(index, 0 | 1 | 2 | 10 | 11 | 12))
            .map(|(_, error)| *error)
            .fold(0.0, f64::max),
        cdf_max,
    }
}

fn score_um(sketch: &UnivMon, truth: &Truth) -> Accuracy {
    let mut squared_error = 0.0;
    let mut point_count = 0;
    for key in 1..=truth.domain {
        let estimate = univmon_frequency(sketch, key);
        squared_error += (estimate - truth.counts[key] as f64).powi(2);
        point_count += 1;
    }
    for key in truth.domain + 1..=truth.domain + 256 {
        squared_error += univmon_frequency(sketch, key).powi(2);
        point_count += 1;
    }
    let heavy_keys: HashSet<usize> = univmon_heavy_hitters(sketch, TOP_K)
        .into_iter()
        .map(|(value, _)| value)
        .collect();
    let captured_mass: u64 = heavy_keys
        .iter()
        .filter_map(|key| truth.counts.get(*key))
        .sum();
    let entropy = sketch.calc_entropy() * std::f64::consts::LN_2;
    Accuracy {
        point_nrmse: (squared_error / point_count as f64).sqrt() / truth.f2.sqrt().max(1.0),
        f0_rel: relative_error(sketch.calc_card(), truth.distinct as f64),
        f2_rel: relative_error(sketch.calc_l2().powi(2), truth.f2),
        entropy_rel: relative_error(entropy, truth.entropy),
        entropy_abs: (entropy - truth.entropy).abs(),
        l1_rel: relative_error(sketch.calc_l1(), truth.n as f64),
        hh_mass: (captured_mass as f64 / truth.top_mass.max(1) as f64).min(1.0),
        quantile_mean: 0.0,
        tail_max: 0.0,
        cdf_max: 0.0,
    }
}

fn univmon_frequency(sketch: &UnivMon, value: usize) -> f64 {
    let input = DataInput::U64(value as u64);
    let hash = hash64_seeded(BOTTOM_LAYER_FINDER, &input);
    let mut terminal = sketch.layer_size - 1;
    for level in 1..sketch.layer_size {
        if ((hash >> level) & 1) == 0 {
            terminal = level - 1;
            break;
        }
    }
    sketch.l2_sketch_layers[terminal].estimate(&input).max(0.0)
}

fn univmon_heavy_hitters(sketch: &UnivMon, k: usize) -> Vec<(usize, u64)> {
    let mut keys = HashSet::new();
    for heap in sketch.hh_layers.iter() {
        for item in heap.heap() {
            if let HeapItem::U64(value) = item.key {
                keys.insert(value as usize);
            }
        }
    }
    let mut recovered: Vec<_> = keys
        .into_iter()
        .map(|key| (key, univmon_frequency(sketch, key).round() as u64))
        .collect();
    recovered.sort_unstable_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    recovered.truncate(k);
    recovered
}

fn validate(sketch: &UnivMonQ, truth: &Truth) -> usize {
    let query = sketch.prepare_queries();
    let mut violations = 0;
    violations += usize::from(query.count() != truth.n as u64);
    violations += usize::from(query.min() != Some(truth.min as f64));
    violations += usize::from(query.max() != Some(truth.max as f64));
    violations += usize::from(!query.estimate_distinct().is_finite());
    violations += usize::from(!query.estimate_f2().is_finite());
    violations += usize::from(!query.estimate_entropy().is_finite());
    violations += usize::from(query.estimate_distinct() < 0.0);
    violations += usize::from(query.estimate_distinct() > truth.n as f64);
    violations += usize::from(query.estimate_f2() < 0.0);

    let cdf = query.cdf();
    violations += usize::from(cdf.is_empty());
    violations += cdf
        .windows(2)
        .filter(|pair| {
            !pair[0].value.total_cmp(&pair[1].value).is_lt() || pair[0].rank > pair[1].rank
        })
        .count();
    violations += cdf
        .iter()
        .filter(|point| !point.rank.is_finite() || !(0.0..=1.0).contains(&point.rank))
        .count();
    let quantiles = query.quantiles(&QUANTILES);
    violations += quantiles
        .windows(2)
        .filter(|pair| {
            pair[0]
                .zip(pair[1])
                .is_some_and(|(left, right)| left > right)
        })
        .count();
    violations
}

fn validate_um(sketch: &UnivMon, truth: &Truth) -> usize {
    let estimates = [
        sketch.calc_card(),
        sketch.calc_l2(),
        sketch.calc_entropy(),
        sketch.calc_l1(),
    ];
    usize::from(sketch.bucket_size != truth.n)
        + estimates.iter().filter(|value| !value.is_finite()).count()
        + estimates.iter().filter(|&&value| value < 0.0).count()
}

fn equivalent_queries(left: &UnivMonQ, right: &UnivMonQ, truth: &Truth) -> bool {
    let left = left.prepare_queries();
    let right = right.prepare_queries();
    left.count() == right.count()
        && left.min().map(f64::to_bits) == right.min().map(f64::to_bits)
        && left.max().map(f64::to_bits) == right.max().map(f64::to_bits)
        && left.estimate_distinct().to_bits() == right.estimate_distinct().to_bits()
        && left.estimate_f2().to_bits() == right.estimate_f2().to_bits()
        && left.estimate_entropy().to_bits() == right.estimate_entropy().to_bits()
        && left.heavy_hitters(TOP_K) == right.heavy_hitters(TOP_K)
        && left.quantiles(&QUANTILES) == right.quantiles(&QUANTILES)
        && (0..=100).all(|index| {
            let value = 1 + index * (truth.domain - 1) / 100;
            left.rank(value as f64) == right.rank(value as f64)
        })
        && left.cdf().len() == right.cdf().len()
        && left.cdf().iter().zip(right.cdf()).all(|(left, right)| {
            left.value.to_bits() == right.value.to_bits()
                && left.rank.to_bits() == right.rank.to_bits()
        })
}

fn um_query_mismatches(left: &UnivMon, right: &UnivMon, truth: &Truth) -> Vec<&'static str> {
    let mut mismatches = Vec::new();
    if left.bucket_size != right.bucket_size {
        mismatches.push("count");
    }
    if left.calc_card().to_bits() != right.calc_card().to_bits() {
        mismatches.push("F0");
    }
    if left.calc_l2().to_bits() != right.calc_l2().to_bits() {
        mismatches.push("F2");
    }
    if left.calc_entropy().to_bits() != right.calc_entropy().to_bits() {
        mismatches.push("entropy");
    }
    if left.calc_l1().to_bits() != right.calc_l1().to_bits() {
        mismatches.push("L1");
    }
    if univmon_heavy_hitters(left, TOP_K) != univmon_heavy_hitters(right, TOP_K) {
        mismatches.push("heavy hitters");
    }
    if !(0..=100).all(|index| {
        let value = 1 + index * (truth.domain - 1) / 100;
        univmon_frequency(left, value).to_bits() == univmon_frequency(right, value).to_bits()
    }) {
        mismatches.push("point frequency");
    }
    mismatches
}

struct Truth {
    n: usize,
    domain: usize,
    counts: Vec<u64>,
    cumulative: Vec<u64>,
    distinct: usize,
    min: usize,
    max: usize,
    f2: f64,
    entropy: f64,
    top_mass: u64,
}

impl Truth {
    fn new(data: &[i64], domain: usize) -> Self {
        let mut counts = vec![0_u64; domain + 1];
        for &value in data {
            counts[value as usize] += 1;
        }
        let mut cumulative = vec![0_u64; domain + 1];
        for key in 1..=domain {
            cumulative[key] = cumulative[key - 1] + counts[key];
        }
        let distinct = counts.iter().filter(|&&count| count > 0).count();
        let min = counts.iter().position(|&count| count > 0).unwrap();
        let max = counts.iter().rposition(|&count| count > 0).unwrap();
        let f2 = counts.iter().map(|&count| (count as f64).powi(2)).sum();
        let entropy = counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let probability = count as f64 / data.len() as f64;
                -probability * probability.ln()
            })
            .sum();
        let mut frequencies = counts.clone();
        frequencies.sort_unstable_by(|left, right| right.cmp(left));
        let top_mass = frequencies.into_iter().take(TOP_K).sum();
        Self {
            n: data.len(),
            domain,
            counts,
            cumulative,
            distinct,
            min,
            max,
            f2,
            entropy,
            top_mass,
        }
    }

    fn cdf(&self, value: usize) -> f64 {
        self.cumulative[value.min(self.domain)] as f64 / self.n as f64
    }

    fn quantile_rank_error(&self, q: f64, estimate: usize) -> f64 {
        let target = (q * self.n as f64).ceil().max(1.0) as u64;
        let below = self.cumulative[estimate.saturating_sub(1).min(self.domain)];
        let at_or_below = self.cumulative[estimate.min(self.domain)];
        if target < below {
            (below - target) as f64 / self.n as f64
        } else if target > at_or_below {
            (target - at_or_below) as f64 / self.n as f64
        } else {
            0.0
        }
    }
}

struct TrialResult {
    q_accuracy: Accuracy,
    um_accuracy: Accuracy,
    distinct: f64,
    entropy_exact: f64,
    q_update_ns: f64,
    um_update_ns: f64,
    q_merge_us: f64,
    um_merge_us: f64,
    q_prepare_us: f64,
    q_warm_ns: f64,
    um_query_us: f64,
    q_memory_mib: f64,
    q_wire_kib: f64,
    um_wire_kib: f64,
    q_violations: usize,
    um_violations: usize,
    um_warnings: usize,
}

struct Summary {
    skew: f64,
    distinct_p50: f64,
    point_p95: f64,
    f0_p95: f64,
    f2_p95: f64,
    entropy_p95: f64,
    entropy_exact_p50: f64,
    entropy_abs_p95: f64,
    l1_p95: f64,
    hh_mass_p05: f64,
    um_point_p95: f64,
    um_f0_p95: f64,
    um_f2_p95: f64,
    um_entropy_p95: f64,
    um_entropy_abs_p95: f64,
    um_l1_p95: f64,
    um_hh_mass_p05: f64,
    quantile_p50: f64,
    quantile_p95: f64,
    tail_p50: f64,
    tail_p95: f64,
    cdf_p50: f64,
    cdf_p95: f64,
    update_p50: f64,
    update_p95: f64,
    merge_p50_us: f64,
    prepare_p50_us: f64,
    warm_p50_ns: f64,
    memory_mib: f64,
    q_wire_kib: f64,
    um_update_p50: f64,
    um_update_p95: f64,
    um_merge_p50_us: f64,
    um_query_p50_us: f64,
    um_wire_kib: f64,
    q_violations: usize,
    um_violations: usize,
    um_warnings: usize,
}

impl Summary {
    fn new(skew: f64, results: &[TrialResult]) -> Self {
        let values = |field: fn(&TrialResult) -> f64| results.iter().map(field).collect::<Vec<_>>();
        let distinct = values(|result| result.distinct);
        let point = values(|result| result.q_accuracy.point_nrmse);
        let f0 = values(|result| result.q_accuracy.f0_rel);
        let f2 = values(|result| result.q_accuracy.f2_rel);
        let entropy = values(|result| result.q_accuracy.entropy_rel);
        let entropy_exact = values(|result| result.entropy_exact);
        let entropy_abs = values(|result| result.q_accuracy.entropy_abs);
        let l1 = values(|result| result.q_accuracy.l1_rel);
        let hh_mass = values(|result| result.q_accuracy.hh_mass);
        let quantile = values(|result| result.q_accuracy.quantile_mean);
        let tail = values(|result| result.q_accuracy.tail_max);
        let cdf = values(|result| result.q_accuracy.cdf_max);
        let update = values(|result| result.q_update_ns);
        let merge = values(|result| result.q_merge_us);
        let prepare = values(|result| result.q_prepare_us);
        let warm = values(|result| result.q_warm_ns);
        let um_point = values(|result| result.um_accuracy.point_nrmse);
        let um_f0 = values(|result| result.um_accuracy.f0_rel);
        let um_f2 = values(|result| result.um_accuracy.f2_rel);
        let um_entropy = values(|result| result.um_accuracy.entropy_rel);
        let um_entropy_abs = values(|result| result.um_accuracy.entropy_abs);
        let um_l1 = values(|result| result.um_accuracy.l1_rel);
        let um_hh_mass = values(|result| result.um_accuracy.hh_mass);
        let um_update = values(|result| result.um_update_ns);
        let um_merge = values(|result| result.um_merge_us);
        let um_query = values(|result| result.um_query_us);
        Self {
            skew,
            distinct_p50: percentile(&distinct, 0.5),
            point_p95: percentile(&point, 0.95),
            f0_p95: percentile(&f0, 0.95),
            f2_p95: percentile(&f2, 0.95),
            entropy_p95: percentile(&entropy, 0.95),
            entropy_exact_p50: percentile(&entropy_exact, 0.5),
            entropy_abs_p95: percentile(&entropy_abs, 0.95),
            l1_p95: percentile(&l1, 0.95),
            hh_mass_p05: percentile(&hh_mass, 0.05),
            um_point_p95: percentile(&um_point, 0.95),
            um_f0_p95: percentile(&um_f0, 0.95),
            um_f2_p95: percentile(&um_f2, 0.95),
            um_entropy_p95: percentile(&um_entropy, 0.95),
            um_entropy_abs_p95: percentile(&um_entropy_abs, 0.95),
            um_l1_p95: percentile(&um_l1, 0.95),
            um_hh_mass_p05: percentile(&um_hh_mass, 0.05),
            quantile_p50: percentile(&quantile, 0.5),
            quantile_p95: percentile(&quantile, 0.95),
            tail_p50: percentile(&tail, 0.5),
            tail_p95: percentile(&tail, 0.95),
            cdf_p50: percentile(&cdf, 0.5),
            cdf_p95: percentile(&cdf, 0.95),
            update_p50: percentile(&update, 0.5),
            update_p95: percentile(&update, 0.95),
            merge_p50_us: percentile(&merge, 0.5),
            prepare_p50_us: percentile(&prepare, 0.5),
            warm_p50_ns: percentile(&warm, 0.5),
            memory_mib: percentile(&values(|result| result.q_memory_mib), 0.5),
            q_wire_kib: percentile(&values(|result| result.q_wire_kib), 0.5),
            um_update_p50: percentile(&um_update, 0.5),
            um_update_p95: percentile(&um_update, 0.95),
            um_merge_p50_us: percentile(&um_merge, 0.5),
            um_query_p50_us: percentile(&um_query, 0.5),
            um_wire_kib: percentile(&values(|result| result.um_wire_kib), 0.5),
            q_violations: results.iter().map(|result| result.q_violations).sum(),
            um_violations: results.iter().map(|result| result.um_violations).sum(),
            um_warnings: results.iter().map(|result| result.um_warnings).sum(),
        }
    }
}

struct ZipfSampler {
    cdf: Vec<f64>,
    total: f64,
}

impl ZipfSampler {
    fn new(domain: usize, skew: f64) -> Self {
        let mut total = 0.0;
        let mut cdf = Vec::with_capacity(domain);
        for rank in 1..=domain {
            total += 1.0 / (rank as f64).powf(skew);
            cdf.push(total);
        }
        Self { cdf, total }
    }

    fn generate(&self, n: usize, seed: u64) -> Vec<i64> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|_| {
                let target = rng.unit() * self.total;
                (self.cdf.partition_point(|value| *value < target) + 1) as i64
            })
            .collect()
    }
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

fn source_id(seed: u64, shard: usize) -> u64 {
    mix64(seed ^ (shard as u64).wrapping_mul(0xd6e8_feb8_6659_fd93))
}

fn trial_seed(skew: usize, trial: usize) -> u64 {
    0x2026_0807_554d_5100_u64 ^ ((skew as u64) << 32) ^ trial as u64
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn relative_error(estimate: f64, exact: f64) -> f64 {
    (estimate - exact).abs() / exact.abs().max(1.0)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let index = ((quantile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn duration_percentile(values: &[Duration], quantile: f64) -> Duration {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((quantile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}
