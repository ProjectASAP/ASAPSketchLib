use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

use asap_sketchlib::{
    BOTTOM_LAYER_FINDER, DataInput, HeapItem, UnivMon, UnivMonQ, UnivMonQConfig, hash64_seeded,
};

const N: usize = 250_000;
const DOMAIN: u64 = 20_000;
const LEVELS: usize = 12;
const WIDTH: usize = 2_048;
const DEPTH: usize = 5;
const CANDIDATES: usize = 64;
const MEMORY_MATCHED_CANDIDATES: usize = 768;
const ORDERED_SAMPLES: usize = 1_024;
const REPEATS: usize = 5;
const MERGE_REPEATS: usize = 11;
const QUERY_REPEATS: u32 = 100;
const TOP_K: usize = 4;

fn stream() -> Vec<u64> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..N)
        .map(|i| {
            // Four separated heavy hitters consume 40% of the stream; the
            // remaining 60% is a long numeric tail for F0 and quantiles.
            match i % 25 {
                0..=3 => 0,
                4..=6 => 1,
                7..=8 => 2,
                9 => 3,
                _ => {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    100 + state % (DOMAIN - 100)
                }
            }
        })
        .collect()
}

fn median_duration(mut runs: Vec<Duration>) -> Duration {
    runs.sort_unstable();
    runs[runs.len() / 2]
}

fn rate(duration: Duration) -> f64 {
    N as f64 / duration.as_secs_f64() / 1_000_000.0
}

fn relative_error(estimate: f64, exact: f64) -> f64 {
    (estimate - exact).abs() / exact.abs().max(f64::EPSILON)
}

fn average_query<T>(mut query: impl FnMut() -> T, repeats: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..repeats {
        black_box(query());
    }
    start.elapsed() / repeats
}

fn latency(duration: Duration) -> String {
    if duration.is_zero() {
        "<1ns".to_owned()
    } else {
        format!("{duration:?}")
    }
}

fn bottom_layer(value: u64) -> usize {
    let hash = hash64_seeded(BOTTOM_LAYER_FINDER, &DataInput::U64(value));
    for level in 1..LEVELS {
        if ((hash >> level) & 1) == 0 {
            return level - 1;
        }
    }
    LEVELS - 1
}

fn univmon_frequency(sketch: &UnivMon, value: u64, terminal_only: bool) -> f64 {
    let level = if terminal_only {
        bottom_layer(value)
    } else {
        0
    };
    sketch.l2_sketch_layers[level].estimate(&DataInput::U64(value))
}

fn heap_key_u64(key: &HeapItem) -> Option<u64> {
    match key {
        HeapItem::U64(value) => Some(*value),
        _ => None,
    }
}

fn univmon_heavy_hitters(sketch: &UnivMon, terminal_only: bool, k: usize) -> Vec<(u64, u64)> {
    let mut keys = HashSet::new();
    let levels = if terminal_only { 0..LEVELS } else { 0..1 };
    for level in levels {
        for item in sketch.hh_layers[level].heap() {
            if let Some(key) = heap_key_u64(&item.key) {
                keys.insert(key);
            }
        }
    }
    let mut recovered: Vec<_> = keys
        .into_iter()
        .map(|key| {
            (
                key,
                univmon_frequency(sketch, key, terminal_only).max(0.0) as u64,
            )
        })
        .collect();
    recovered
        .sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    recovered.truncate(k);
    recovered
}

fn heavy_hitter_recall(recovered: &[(u64, u64)], exact: &[(u64, u64)]) -> f64 {
    let expected: HashSet<_> = exact.iter().map(|(key, _)| *key).collect();
    recovered
        .iter()
        .filter(|(key, _)| expected.contains(key))
        .count() as f64
        / exact.len() as f64
}

fn main() {
    let values = stream();
    let mut exact = HashMap::<u64, u64>::new();
    for &value in &values {
        *exact.entry(value).or_default() += 1;
    }
    let exact_l1 = N as f64;
    let exact_f0 = exact.len() as f64;
    let exact_f2 = exact.values().map(|&f| (f as f64).powi(2)).sum::<f64>();
    let exact_entropy_bits = exact
        .values()
        .map(|&f| {
            let p = f as f64 / N as f64;
            -p * p.log2()
        })
        .sum::<f64>();
    let mut sorted = values.clone();
    sorted.sort_unstable();
    let exact_min = sorted[0] as f64;
    let exact_max = sorted[N - 1] as f64;
    let exact_p50 = sorted[((N - 1) as f64 * 0.50).round() as usize] as f64;
    let exact_p90 = sorted[((N - 1) as f64 * 0.90).round() as usize] as f64;
    let exact_p99 = sorted[((N - 1) as f64 * 0.99).round() as usize] as f64;
    let rank_probe = DOMAIN / 2;
    let exact_rank = sorted.partition_point(|value| *value <= rank_probe) as u64;
    let mut exact_top: Vec<_> = exact.iter().map(|(&key, &count)| (key, count)).collect();
    exact_top
        .sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    exact_top.truncate(TOP_K);

    let q_base = UnivMonQConfig {
        levels: LEVELS,
        width: WIDTH,
        width_halving_period: 0,
        depth: DEPTH,
        counter_bits: 32,
        candidates: CANDIDATES,
        ordered_samples: 0,
        hash_seed: 5,
    };
    let q_ordered = UnivMonQConfig {
        ordered_samples: ORDERED_SAMPLES,
        ..q_base
    };
    let q_memory_matched = UnivMonQConfig {
        candidates: MEMORY_MATCHED_CANDIDATES,
        ordered_samples: ORDERED_SAMPLES,
        ..q_base
    };

    let mut regular_times = Vec::new();
    let mut fast_times = Vec::new();
    let mut q_base_times = Vec::new();
    let mut q_ordered_times = Vec::new();
    let mut q_memory_matched_times = Vec::new();

    for _ in 0..REPEATS {
        let mut sketch = UnivMon::init_univmon(CANDIDATES, DEPTH, WIDTH, LEVELS);
        let start = Instant::now();
        for &value in &values {
            sketch.insert(&DataInput::U64(black_box(value)), 1);
        }
        regular_times.push(start.elapsed());
        black_box(sketch.bucket_size);

        let mut sketch = UnivMon::init_univmon(CANDIDATES, DEPTH, WIDTH, LEVELS);
        let start = Instant::now();
        for &value in &values {
            sketch.fast_insert(&DataInput::U64(black_box(value)), 1);
        }
        fast_times.push(start.elapsed());
        black_box(sketch.bucket_size);

        let mut sketch = UnivMonQ::new(q_base).unwrap();
        let start = Instant::now();
        for &value in &values {
            sketch.add(&black_box(value));
        }
        q_base_times.push(start.elapsed());
        black_box(sketch.count());

        let mut sketch = UnivMonQ::new(q_ordered).unwrap();
        let start = Instant::now();
        for &value in &values {
            sketch.add(&black_box(value));
        }
        q_ordered_times.push(start.elapsed());
        black_box(sketch.count());

        let mut sketch = UnivMonQ::new(q_memory_matched).unwrap();
        let start = Instant::now();
        for &value in &values {
            sketch.add(&black_box(value));
        }
        q_memory_matched_times.push(start.elapsed());
        black_box(sketch.count());
    }

    let regular_time = median_duration(regular_times);
    let fast_time = median_duration(fast_times);
    let q_base_time = median_duration(q_base_times);
    let q_ordered_time = median_duration(q_ordered_times);
    let q_memory_matched_time = median_duration(q_memory_matched_times);

    // Build once more for accuracy, query latency, serialized size, and merge.
    let mut regular = UnivMon::init_univmon(CANDIDATES, DEPTH, WIDTH, LEVELS);
    let mut fast = UnivMon::init_univmon(CANDIDATES, DEPTH, WIDTH, LEVELS);
    let mut q_base_sketch = UnivMonQ::new(q_base).unwrap();
    let mut q_ordered_sketch = UnivMonQ::new(q_ordered).unwrap();
    let mut q_memory_matched_sketch = UnivMonQ::new(q_memory_matched).unwrap();
    for &value in &values {
        let input = DataInput::U64(value);
        regular.insert(&input, 1);
        fast.fast_insert(&input, 1);
        q_base_sketch.add(&value);
        q_ordered_sketch.add(&value);
        q_memory_matched_sketch.add(&value);
    }

    let regular_wire = regular.serialize_to_bytes().unwrap().len();
    let fast_wire = fast.serialize_to_bytes().unwrap().len();
    let q_base_wire = q_base_sketch.serialize_to_bytes().unwrap().len();
    let q_ordered_wire = q_ordered_sketch.serialize_to_bytes().unwrap().len();
    let q_memory_matched_wire = q_memory_matched_sketch.serialize_to_bytes().unwrap().len();

    let regular_l1 = regular.calc_l1();
    let regular_card = regular.calc_card();
    let regular_f2 = regular.calc_l2().powi(2);
    let regular_entropy = regular.calc_entropy();
    let regular_frequency = univmon_frequency(&regular, 0, false);
    let regular_hh = univmon_heavy_hitters(&regular, false, TOP_K);
    let fast_l1 = fast.calc_l1();
    let fast_card = fast.calc_card();
    let fast_f2 = fast.calc_l2().powi(2);
    let fast_entropy = fast.calc_entropy();
    let fast_frequency = univmon_frequency(&fast, 0, true);
    let fast_hh = univmon_heavy_hitters(&fast, true, TOP_K);
    let q_l1 = q_ordered_sketch.count() as f64;
    let q_card = q_base_sketch.estimate_distinct();
    let q_f2 = q_base_sketch.estimate_f2();
    let q_entropy_bits = q_base_sketch.estimate_entropy() / std::f64::consts::LN_2;
    let q_frequency = q_ordered_sketch.estimate_frequency(0.0) as f64;
    let q_hh: Vec<_> = q_ordered_sketch
        .heavy_hitters(TOP_K)
        .into_iter()
        .map(|(key, count)| (key as u64, count))
        .collect();
    let q_memory_l1 = q_memory_matched_sketch.count() as f64;
    let q_memory_frequency = q_memory_matched_sketch.estimate_frequency(0.0) as f64;
    let q_memory_hh: Vec<_> = q_memory_matched_sketch
        .heavy_hitters(TOP_K)
        .into_iter()
        .map(|(key, count)| (key as u64, count))
        .collect();

    let q_prepare_query = average_query(|| q_ordered_sketch.prepare_queries(), QUERY_REPEATS);
    let q_memory_prepare_query =
        average_query(|| q_memory_matched_sketch.prepare_queries(), QUERY_REPEATS);
    let q_prepared = q_ordered_sketch.prepare_queries();
    let q_memory_prepared = q_memory_matched_sketch.prepare_queries();

    let regular_l1_query = average_query(|| regular.calc_l1(), QUERY_REPEATS);
    let fast_l1_query = average_query(|| fast.calc_l1(), QUERY_REPEATS);
    let q_l1_query = average_query(|| q_ordered_sketch.count(), QUERY_REPEATS);
    let q_memory_l1_query = average_query(|| q_memory_matched_sketch.count(), QUERY_REPEATS);
    let regular_frequency_query = average_query(
        || univmon_frequency(&regular, black_box(0), false),
        QUERY_REPEATS,
    );
    let fast_frequency_query = average_query(
        || univmon_frequency(&fast, black_box(0), true),
        QUERY_REPEATS,
    );
    let q_frequency_query = average_query(
        || q_ordered_sketch.estimate_frequency(black_box(0.0)),
        QUERY_REPEATS,
    );
    let q_memory_frequency_query = average_query(
        || q_memory_matched_sketch.estimate_frequency(black_box(0.0)),
        QUERY_REPEATS,
    );
    let regular_f0_query = average_query(|| regular.calc_card(), QUERY_REPEATS);
    let fast_f0_query = average_query(|| fast.calc_card(), QUERY_REPEATS);
    let q_f0_query = average_query(|| q_ordered_sketch.estimate_distinct(), QUERY_REPEATS);
    let q_memory_f0_query = average_query(
        || q_memory_matched_sketch.estimate_distinct(),
        QUERY_REPEATS,
    );
    let regular_f2_query = average_query(|| regular.calc_l2(), QUERY_REPEATS);
    let fast_f2_query = average_query(|| fast.calc_l2(), QUERY_REPEATS);
    let q_f2_query = average_query(|| q_ordered_sketch.estimate_f2(), QUERY_REPEATS);
    let q_memory_f2_query = average_query(|| q_memory_matched_sketch.estimate_f2(), QUERY_REPEATS);
    let regular_entropy_query = average_query(|| regular.calc_entropy(), QUERY_REPEATS);
    let fast_entropy_query = average_query(|| fast.calc_entropy(), QUERY_REPEATS);
    let q_entropy_query = average_query(|| q_ordered_sketch.estimate_entropy(), QUERY_REPEATS);
    let q_memory_entropy_query =
        average_query(|| q_memory_matched_sketch.estimate_entropy(), QUERY_REPEATS);
    let regular_hh_query = average_query(
        || univmon_heavy_hitters(&regular, false, TOP_K),
        QUERY_REPEATS,
    );
    let fast_hh_query = average_query(|| univmon_heavy_hitters(&fast, true, TOP_K), QUERY_REPEATS);
    let q_hh_query = average_query(|| q_ordered_sketch.heavy_hitters(TOP_K), QUERY_REPEATS);
    let q_memory_hh_query = average_query(
        || q_memory_matched_sketch.heavy_hitters(TOP_K),
        QUERY_REPEATS,
    );
    let q_min_max_query = average_query(
        || (q_ordered_sketch.min(), q_ordered_sketch.max()),
        QUERY_REPEATS,
    );
    let q_memory_min_max_query = average_query(
        || (q_memory_matched_sketch.min(), q_memory_matched_sketch.max()),
        QUERY_REPEATS,
    );
    let q_rank_query = average_query(
        || q_ordered_sketch.rank(black_box(rank_probe as f64)),
        QUERY_REPEATS,
    );
    let q_memory_rank_query = average_query(
        || q_memory_matched_sketch.rank(black_box(rank_probe as f64)),
        QUERY_REPEATS,
    );
    let q_quantile_query = average_query(
        || q_ordered_sketch.quantiles(&[0.5, 0.9, 0.99]),
        QUERY_REPEATS,
    );
    let q_memory_quantile_query = average_query(
        || q_memory_matched_sketch.quantiles(&[0.5, 0.9, 0.99]),
        QUERY_REPEATS,
    );
    let q_cdf_query = average_query(|| q_ordered_sketch.cdf(), QUERY_REPEATS);
    let q_memory_cdf_query = average_query(|| q_memory_matched_sketch.cdf(), QUERY_REPEATS);

    let q_prepared_f0_query = average_query(|| q_prepared.estimate_distinct(), QUERY_REPEATS);
    let q_memory_prepared_f0_query =
        average_query(|| q_memory_prepared.estimate_distinct(), QUERY_REPEATS);
    let q_prepared_f2_query = average_query(|| q_prepared.estimate_f2(), QUERY_REPEATS);
    let q_memory_prepared_f2_query =
        average_query(|| q_memory_prepared.estimate_f2(), QUERY_REPEATS);
    let q_prepared_entropy_query = average_query(|| q_prepared.estimate_entropy(), QUERY_REPEATS);
    let q_memory_prepared_entropy_query =
        average_query(|| q_memory_prepared.estimate_entropy(), QUERY_REPEATS);
    let q_prepared_hh_query = average_query(|| q_prepared.heavy_hitters(TOP_K), QUERY_REPEATS);
    let q_memory_prepared_hh_query =
        average_query(|| q_memory_prepared.heavy_hitters(TOP_K), QUERY_REPEATS);
    let q_prepared_rank_query = average_query(
        || q_prepared.rank(black_box(rank_probe as f64)),
        QUERY_REPEATS,
    );
    let q_memory_prepared_rank_query = average_query(
        || q_memory_prepared.rank(black_box(rank_probe as f64)),
        QUERY_REPEATS,
    );
    let q_prepared_quantile_query =
        average_query(|| q_prepared.quantiles(&[0.5, 0.9, 0.99]), QUERY_REPEATS);
    let q_memory_prepared_quantile_query = average_query(
        || q_memory_prepared.quantiles(&[0.5, 0.9, 0.99]),
        QUERY_REPEATS,
    );
    let q_prepared_cdf_query = average_query(|| q_prepared.cdf(), QUERY_REPEATS);
    let q_memory_prepared_cdf_query = average_query(|| q_memory_prepared.cdf(), QUERY_REPEATS);

    println!(
        "workload: n={N}, domain={DOMAIN}, levels={LEVELS}, width={WIDTH}, depth={DEPTH}, candidates={CANDIDATES}"
    );
    println!(
        "exact: L1={exact_l1:.0}, freq(0)={}, F0={exact_f0:.0}, F2={exact_f2:.0}, entropy={exact_entropy_bits:.6} bits",
        exact[&0]
    );
    println!(
        "ordered truth: min={exact_min:.0}, max={exact_max:.0}, rank({rank_probe})={exact_rank}, p50={exact_p50:.0}, p90={exact_p90:.0}, p99={exact_p99:.0}, top-{TOP_K}={exact_top:?}"
    );
    println!();
    println!("update (median of {REPEATS} release runs):");
    println!(
        "  UnivMon insert       {:>9.3} M/s  {:?}",
        rate(regular_time),
        regular_time
    );
    println!(
        "  UnivMon fast_insert  {:>9.3} M/s  {:?}",
        rate(fast_time),
        fast_time
    );
    println!(
        "  UnivMon-Q universal  {:>9.3} M/s  {:?}",
        rate(q_base_time),
        q_base_time
    );
    println!(
        "  UnivMon-Q +quantile  {:>9.3} M/s  {:?}",
        rate(q_ordered_time),
        q_ordered_time
    );
    println!(
        "  UnivMon-Q mem-match  {:>9.3} M/s  {:?}",
        rate(q_memory_matched_time),
        q_memory_matched_time
    );
    println!();
    println!("state size:");
    println!(
        "  UnivMon insert       wire={regular_wire} bytes, counters>={} bytes",
        LEVELS * DEPTH * WIDTH * 8
    );
    println!(
        "  UnivMon fast_insert  wire={fast_wire} bytes, counters>={} bytes",
        LEVELS * DEPTH * WIDTH * 8
    );
    println!(
        "  UnivMon-Q universal  wire={q_base_wire} bytes, reserved~{} bytes",
        q_base_sketch.estimated_memory_bytes()
    );
    println!(
        "  UnivMon-Q +quantile  wire={q_ordered_wire} bytes, reserved~{} bytes",
        q_ordered_sketch.estimated_memory_bytes()
    );
    println!(
        "  UnivMon-Q mem-match  wire={q_memory_matched_wire} bytes, reserved~{} bytes",
        q_memory_matched_sketch.estimated_memory_bytes()
    );
    println!();
    println!("query latency (mean of {QUERY_REPEATS} release queries):");
    println!(
        "  metric          UnivMon std  UnivMon terminal  Q equal-candidates  Q memory-matched"
    );
    println!(
        "  L1/count        {:>11}  {:>16}  {:>18}  {:>16}",
        latency(regular_l1_query),
        latency(fast_l1_query),
        latency(q_l1_query),
        latency(q_memory_l1_query)
    );
    println!(
        "  frequency       {:>11}  {:>16}  {:>18}  {:>16}",
        latency(regular_frequency_query),
        latency(fast_frequency_query),
        latency(q_frequency_query),
        latency(q_memory_frequency_query)
    );
    println!(
        "  F0              {:>11}  {:>16}  {:>18}  {:>16}",
        latency(regular_f0_query),
        latency(fast_f0_query),
        latency(q_f0_query),
        latency(q_memory_f0_query)
    );
    println!(
        "  F2              {:>11}  {:>16}  {:>18}  {:>16}",
        latency(regular_f2_query),
        latency(fast_f2_query),
        latency(q_f2_query),
        latency(q_memory_f2_query)
    );
    println!(
        "  entropy         {:>11}  {:>16}  {:>18}  {:>16}",
        latency(regular_entropy_query),
        latency(fast_entropy_query),
        latency(q_entropy_query),
        latency(q_memory_entropy_query)
    );
    println!(
        "  top-{TOP_K}           {:>11}  {:>16}  {:>18}  {:>16}",
        latency(regular_hh_query),
        latency(fast_hh_query),
        latency(q_hh_query),
        latency(q_memory_hh_query)
    );
    println!(
        "  min+max        {:>11}  {:>16}  {:>18}  {:>16}",
        "n/a",
        "n/a",
        latency(q_min_max_query),
        latency(q_memory_min_max_query)
    );
    println!(
        "  rank           {:>11}  {:>16}  {:>18}  {:>16}",
        "n/a",
        "n/a",
        latency(q_rank_query),
        latency(q_memory_rank_query)
    );
    println!(
        "  p50+p90+p99    {:>11}  {:>16}  {:>18}  {:>16}",
        "n/a",
        "n/a",
        latency(q_quantile_query),
        latency(q_memory_quantile_query)
    );
    println!(
        "  full CDF       {:>11}  {:>16}  {:>18}  {:>16}",
        "n/a",
        "n/a",
        latency(q_cdf_query),
        latency(q_memory_cdf_query)
    );
    println!();
    println!("prepared UnivMon-Q view (one reconstruction, then reused):");
    println!("  metric                 Q equal-candidates  Q memory-matched");
    println!(
        "  prepare                        {:>11}        {:>11}",
        latency(q_prepare_query),
        latency(q_memory_prepare_query)
    );
    println!(
        "  F0                             {:>11}        {:>11}",
        latency(q_prepared_f0_query),
        latency(q_memory_prepared_f0_query)
    );
    println!(
        "  F2                             {:>11}        {:>11}",
        latency(q_prepared_f2_query),
        latency(q_memory_prepared_f2_query)
    );
    println!(
        "  entropy                        {:>11}        {:>11}",
        latency(q_prepared_entropy_query),
        latency(q_memory_prepared_entropy_query)
    );
    println!(
        "  top-{TOP_K}                          {:>11}        {:>11}",
        latency(q_prepared_hh_query),
        latency(q_memory_prepared_hh_query)
    );
    println!(
        "  rank                           {:>11}        {:>11}",
        latency(q_prepared_rank_query),
        latency(q_memory_prepared_rank_query)
    );
    println!(
        "  p50+p90+p99                   {:>11}        {:>11}",
        latency(q_prepared_quantile_query),
        latency(q_memory_prepared_quantile_query)
    );
    println!(
        "  full CDF                       {:>11}        {:>11}",
        latency(q_prepared_cdf_query),
        latency(q_memory_prepared_cdf_query)
    );
    println!();
    println!("accuracy (relative error):");
    println!(
        "  UnivMon insert       L1={:.4}, freq={:.4}, F0={:.4}, F2={:.4}, H={:.4}, HH-recall={:.2}",
        relative_error(regular_l1, exact_l1),
        relative_error(regular_frequency, exact[&0] as f64),
        relative_error(regular_card, exact_f0),
        relative_error(regular_f2, exact_f2),
        relative_error(regular_entropy, exact_entropy_bits),
        heavy_hitter_recall(&regular_hh, &exact_top),
    );
    println!(
        "  UnivMon fast_insert  L1={:.4}, freq={:.4}, F0={:.4}, F2={:.4}, H={:.4}, HH-recall={:.2}",
        relative_error(fast_l1, exact_l1),
        relative_error(fast_frequency, exact[&0] as f64),
        relative_error(fast_card, exact_f0),
        relative_error(fast_f2, exact_f2),
        relative_error(fast_entropy, exact_entropy_bits),
        heavy_hitter_recall(&fast_hh, &exact_top),
    );
    println!(
        "  UnivMon-Q universal  L1={:.4}, freq={:.4}, F0={:.4}, F2={:.4}, H={:.4}, HH-recall={:.2}",
        relative_error(q_l1, exact_l1),
        relative_error(q_frequency, exact[&0] as f64),
        relative_error(q_card, exact_f0),
        relative_error(q_f2, exact_f2),
        relative_error(q_entropy_bits, exact_entropy_bits),
        heavy_hitter_recall(&q_hh, &exact_top),
    );
    println!(
        "  UnivMon-Q mem-match L1={:.4}, freq={:.4}, F0={:.4}, F2={:.4}, H={:.4}, HH-recall={:.2}",
        relative_error(q_memory_l1, exact_l1),
        relative_error(q_memory_frequency, exact[&0] as f64),
        relative_error(q_memory_matched_sketch.estimate_distinct(), exact_f0),
        relative_error(q_memory_matched_sketch.estimate_f2(), exact_f2),
        relative_error(
            q_memory_matched_sketch.estimate_entropy() / std::f64::consts::LN_2,
            exact_entropy_bits
        ),
        heavy_hitter_recall(&q_memory_hh, &exact_top),
    );
    println!(
        "  recovered HH: std={regular_hh:?}, terminal={fast_hh:?}, Q={q_hh:?}, Q-memory={q_memory_hh:?}"
    );
    println!(
        "  UnivMon-Q ordered    min={:?}, max={:?}, rank-error={:.4}, p50={:?} ({:.4}), p90={:?} ({:.4}), p99={:?} ({:.4})",
        q_ordered_sketch.min(),
        q_ordered_sketch.max(),
        relative_error(
            q_ordered_sketch.rank(rank_probe as f64).unwrap() as f64,
            exact_rank as f64
        ),
        q_ordered_sketch.quantile(0.5),
        relative_error(q_ordered_sketch.quantile(0.5).unwrap(), exact_p50),
        q_ordered_sketch.quantile(0.9),
        relative_error(q_ordered_sketch.quantile(0.9).unwrap(), exact_p90),
        q_ordered_sketch.quantile(0.99),
        relative_error(q_ordered_sketch.quantile(0.99).unwrap(), exact_p99)
    );
    println!(
        "  UnivMon-Q mem-order  min={:?}, max={:?}, rank-error={:.4}, p50={:?} ({:.4}), p90={:?} ({:.4}), p99={:?} ({:.4})",
        q_memory_matched_sketch.min(),
        q_memory_matched_sketch.max(),
        relative_error(
            q_memory_matched_sketch.rank(rank_probe as f64).unwrap() as f64,
            exact_rank as f64
        ),
        q_memory_matched_sketch.quantile(0.5),
        relative_error(q_memory_matched_sketch.quantile(0.5).unwrap(), exact_p50),
        q_memory_matched_sketch.quantile(0.9),
        relative_error(q_memory_matched_sketch.quantile(0.9).unwrap(), exact_p90),
        q_memory_matched_sketch.quantile(0.99),
        relative_error(q_memory_matched_sketch.quantile(0.99).unwrap(), exact_p99)
    );

    let midpoint = N / 2;
    let mut left = UnivMon::init_univmon(CANDIDATES, DEPTH, WIDTH, LEVELS);
    let mut right = UnivMon::init_univmon(CANDIDATES, DEPTH, WIDTH, LEVELS);
    for &value in &values[..midpoint] {
        left.fast_insert(&DataInput::U64(value), 1);
    }
    for &value in &values[midpoint..] {
        right.fast_insert(&DataInput::U64(value), 1);
    }
    let mut univmon_merge_times = Vec::with_capacity(MERGE_REPEATS);
    for _ in 0..MERGE_REPEATS {
        let mut merged = left.clone();
        let start = Instant::now();
        merged.merge(&right);
        univmon_merge_times.push(start.elapsed());
        black_box(merged.bucket_size);
    }
    let univmon_merge = median_duration(univmon_merge_times);

    let mut left = UnivMonQ::new(q_ordered).unwrap();
    let mut right = UnivMonQ::new(q_ordered).unwrap();
    for &value in &values[..midpoint] {
        left.add(&value);
    }
    for &value in &values[midpoint..] {
        right.add(&value);
    }
    let mut q_merge_times = Vec::with_capacity(MERGE_REPEATS);
    for _ in 0..MERGE_REPEATS {
        let mut merged = left.clone();
        let start = Instant::now();
        merged.merge(&right).unwrap();
        q_merge_times.push(start.elapsed());
        black_box(merged.count());
    }
    let q_merge = median_duration(q_merge_times);
    println!();
    println!("50/50 merge (median of {MERGE_REPEATS} release runs):");
    println!("  UnivMon fast state   {:?}", univmon_merge);
    println!("  UnivMon-Q +quantile  {:?}", q_merge);
}
