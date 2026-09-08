//! Rank/CDF/quantile audit for UnivMon-Q.
//!
//! Run with:
//!
//!   cargo run --release --example audit_univmon_q_order -- 50000 5

use asap_sketchlib::{KLL, UnivMonQ, UnivMonQConfig};

const SAMPLE_SIZES: [usize; 7] = [128, 256, 512, 1_024, 2_048, 4_096, 8_192];
const CANDIDATE_SIZES: [usize; 5] = [64, 128, 256, 512, 1_024];
const FANOUTS: [usize; 4] = [1, 2, 8, 32];
const KLL_K: i32 = 200;
const QUANTILES: [f64; 13] = [
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999,
];
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

    println!("UnivMon-Q ordered-query audit: n={n}, trials={trials}");
    println!("Errors are normalized rank errors; lower is better.");
    println!();
    println!("sample-size sweep (one-pass sketches)");
    println!(
        "  {:<12} {:>7} {:>8} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "workload",
        "sample",
        "KiB",
        "Q mean-p50",
        "res CDF-p95",
        "UM fixed95",
        "UM CDF-p95",
        "Q tail-p95",
        "KLL CDF-p95"
    );
    for (workload_index, (name, generator)) in workloads.iter().enumerate() {
        let mut kll_cdf_errors = Vec::with_capacity(trials);
        let mut datasets = Vec::with_capacity(trials);
        for trial in 0..trials {
            let seed = trial_seed(workload_index, trial);
            let data = generator(n, seed);
            let truth = Truth::new(&data);
            let kll = build_kll(&data, seed, 1);
            kll_cdf_errors.push(score_kll(&kll, &truth).cdf_max);
            datasets.push((seed, data, truth));
        }
        let kll_p95 = percentile(&kll_cdf_errors, 0.95);
        for samples in SAMPLE_SIZES {
            let mut mean_errors = Vec::with_capacity(trials);
            let mut cdf_errors = Vec::with_capacity(trials);
            let mut universal_errors = Vec::with_capacity(trials);
            let mut universal_fixed_errors = Vec::with_capacity(trials);
            let mut tail_errors = Vec::with_capacity(trials);
            let mut memory = 0;
            for (trial, (_, data, truth)) in datasets.iter().enumerate() {
                let config = config(samples, 256, trial);
                let sketch = build_q(data, config, 1);
                memory = sketch.estimated_memory_bytes();
                let score = score_q(&sketch, truth);
                mean_errors.push(score.quantile_mean);
                cdf_errors.push(score.cdf_max);
                let universal = score_universal_rank(&sketch, truth);
                universal_fixed_errors.push(universal.fixed);
                universal_errors.push(universal.maximum);
                tail_errors.push(score.tail_max);
            }
            println!(
                "  {name:<12} {samples:>7} {:>8.1} {:>11.5} {:>11.5} {:>11.5} {:>11.5} {:>11.5} {:>11.5}",
                memory as f64 / 1024.0,
                percentile(&mean_errors, 0.5),
                percentile(&cdf_errors, 0.95),
                percentile(&universal_fixed_errors, 0.95),
                percentile(&universal_errors, 0.95),
                percentile(&tail_errors, 0.95),
                kll_p95,
            );
        }
    }

    println!();
    println!("worst CDF breakpoint (trial 0, 1024 ordered samples)");
    println!(
        "  {:<12} {:>14} {:>11} {:>11} {:>11}",
        "workload", "value", "exact", "estimate", "error"
    );
    for (workload_index, (name, generator)) in workloads.iter().enumerate() {
        let data = generator(n, trial_seed(workload_index, 0));
        let truth = Truth::new(&data);
        let sketch = build_q(&data, config(1_024, 256, 0), 1);
        let (value, exact, estimate, error) = worst_cdf_q(&sketch, &truth);
        println!("  {name:<12} {value:>14} {exact:>11.5} {estimate:>11.5} {error:>11.5}");
    }

    println!();
    println!("candidate-size sweep on Zipf (1024 ordered samples)");
    println!(
        "  {:>10} {:>9} {:>12} {:>12} {:>12}",
        "candidates", "KiB", "mean-rank-p50", "CDF-p95", "tail-p95"
    );
    for candidates in CANDIDATE_SIZES {
        let mut mean_errors = Vec::with_capacity(trials);
        let mut cdf_errors = Vec::with_capacity(trials);
        let mut tail_errors = Vec::with_capacity(trials);
        let mut memory = 0;
        for trial in 0..trials {
            let data = zipf(n, trial_seed(3, trial));
            let truth = Truth::new(&data);
            let sketch = build_q(&data, config(1_024, candidates, trial), 1);
            memory = sketch.estimated_memory_bytes();
            let score = score_q(&sketch, &truth);
            mean_errors.push(score.quantile_mean);
            cdf_errors.push(score.cdf_max);
            tail_errors.push(score.tail_max);
        }
        println!(
            "  {candidates:>10} {:>9.1} {:>12.5} {:>12.5} {:>12.5}",
            memory as f64 / 1024.0,
            percentile(&mean_errors, 0.5),
            percentile(&cdf_errors, 0.95),
            percentile(&tail_errors, 0.95),
        );
    }

    println!();
    println!("merge-tree sweep over identical source sketches (1024 ordered samples)");
    println!(
        "  {:<12} {:>6} {:>11} {:>12} {:>12} {:>13}",
        "workload", "shards", "Q CDF-p95", "Q tree-p95", "KLL CDF-p95", "KLL tree-p95"
    );
    for (workload_index, (name, generator)) in workloads.iter().enumerate() {
        for fanout in FANOUTS {
            let mut q_errors = Vec::with_capacity(trials);
            let mut q_drifts = Vec::with_capacity(trials);
            let mut kll_errors = Vec::with_capacity(trials);
            let mut kll_drifts = Vec::with_capacity(trials);
            for trial in 0..trials {
                let seed = trial_seed(workload_index, trial);
                let data = generator(n, seed);
                let truth = Truth::new(&data);
                let config = config(1_024, 256, trial);
                let q_shards = build_q_shards(&data, config, fanout, seed);
                let q_merged = merge_q_tree(q_shards.clone());
                let q_left = merge_q_left(q_shards);
                q_errors.push(score_q(&q_merged, &truth).cdf_max);
                q_drifts.push(cdf_drift_q(&q_left, &q_merged, &truth));

                let kll_shards = build_kll_shards(&data, seed, fanout);
                let kll_merged = merge_kll_tree(kll_shards.clone());
                let kll_left = merge_kll_left(kll_shards);
                kll_errors.push(score_kll(&kll_merged, &truth).cdf_max);
                kll_drifts.push(cdf_drift_kll(&kll_left, &kll_merged, &truth));
            }
            println!(
                "  {name:<12} {fanout:>6} {:>11.5} {:>12.5} {:>12.5} {:>13.5}",
                percentile(&q_errors, 0.95),
                percentile(&q_drifts, 0.95),
                percentile(&kll_errors, 0.95),
                percentile(&kll_drifts, 0.95),
            );
        }
    }
}

fn config(ordered_samples: usize, candidates: usize, trial: usize) -> UnivMonQConfig {
    UnivMonQConfig {
        levels: 12,
        width: 1_024,
        width_halving_period: 3,
        depth: 5,
        counter_bits: 32,
        candidates,
        ordered_samples,
        hash_seed: 5 + trial,
    }
}

fn build_q(data: &[i64], config: UnivMonQConfig, fanout: usize) -> UnivMonQ {
    merge_q_tree(build_q_shards(data, config, fanout, 0))
}

fn build_q_shards(
    data: &[i64],
    config: UnivMonQConfig,
    fanout: usize,
    source_base: u64,
) -> Vec<UnivMonQ> {
    let fanout = fanout.min(data.len()).max(1);
    let mut shards = Vec::with_capacity(fanout);
    for shard in 0..fanout {
        let start = shard * data.len() / fanout;
        let end = (shard + 1) * data.len() / fanout;
        let source_id = source_base
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(shard as u64 + 1);
        let mut sketch = UnivMonQ::new_with_source_id(config, source_id).unwrap();
        for value in &data[start..end] {
            sketch.add(value);
        }
        shards.push(sketch);
    }
    shards
}

fn merge_q_tree(mut sketches: Vec<UnivMonQ>) -> UnivMonQ {
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

fn merge_q_left(mut sketches: Vec<UnivMonQ>) -> UnivMonQ {
    let mut merged = sketches.remove(0);
    for sketch in sketches {
        merged.merge(&sketch).unwrap();
    }
    merged
}

fn build_kll(data: &[i64], seed: u64, fanout: usize) -> KLL<i64> {
    merge_kll_tree(build_kll_shards(data, seed, fanout))
}

fn build_kll_shards(data: &[i64], seed: u64, fanout: usize) -> Vec<KLL<i64>> {
    let fanout = fanout.min(data.len()).max(1);
    let mut shards = Vec::with_capacity(fanout);
    for shard in 0..fanout {
        let start = shard * data.len() / fanout;
        let end = (shard + 1) * data.len() / fanout;
        let mut sketch = KLL::init_kll_with_seed(KLL_K, seed ^ shard as u64);
        for value in &data[start..end] {
            sketch.update(value);
        }
        shards.push(sketch);
    }
    shards
}

fn merge_kll_tree(mut shards: Vec<KLL<i64>>) -> KLL<i64> {
    while shards.len() > 1 {
        let mut merged = Vec::with_capacity(shards.len().div_ceil(2));
        let mut iter = shards.into_iter();
        while let Some(mut left) = iter.next() {
            if let Some(right) = iter.next() {
                left.merge(&right);
            }
            merged.push(left);
        }
        shards = merged;
    }
    shards.pop().unwrap()
}

fn merge_kll_left(mut sketches: Vec<KLL<i64>>) -> KLL<i64> {
    let mut merged = sketches.remove(0);
    for sketch in sketches {
        merged.merge(&sketch);
    }
    merged
}

#[derive(Clone, Copy)]
struct OrderedScore {
    quantile_mean: f64,
    tail_max: f64,
    cdf_max: f64,
}

fn score_q(sketch: &UnivMonQ, truth: &Truth) -> OrderedScore {
    let query = sketch.prepare_queries();
    let errors: Vec<_> = QUANTILES
        .iter()
        .zip(query.quantiles(&QUANTILES))
        .map(|(&q, estimate)| truth.quantile_rank_error(q, estimate.unwrap().round() as i64))
        .collect();
    OrderedScore {
        quantile_mean: errors.iter().sum::<f64>() / errors.len() as f64,
        tail_max: errors
            .iter()
            .enumerate()
            .filter(|(index, _)| *index <= 2 || *index >= QUANTILES.len() - 3)
            .map(|(_, error)| *error)
            .fold(0.0, f64::max),
        cdf_max: truth
            .probes
            .iter()
            .map(|&(value, exact)| {
                let estimate = query.rank(value as f64).unwrap() as f64 / truth.n as f64;
                (estimate - exact).abs()
            })
            .fold(0.0, f64::max),
    }
}

fn score_kll(sketch: &KLL<i64>, truth: &Truth) -> OrderedScore {
    let cdf = sketch.cdf();
    let errors: Vec<_> = QUANTILES
        .iter()
        .map(|&q| truth.quantile_rank_error(q, cdf.query(q).round() as i64))
        .collect();
    OrderedScore {
        quantile_mean: errors.iter().sum::<f64>() / errors.len() as f64,
        tail_max: errors
            .iter()
            .enumerate()
            .filter(|(index, _)| *index <= 2 || *index >= QUANTILES.len() - 3)
            .map(|(_, error)| *error)
            .fold(0.0, f64::max),
        cdf_max: truth
            .probes
            .iter()
            .map(|&(value, exact)| (cdf.quantile(value as f64) - exact).abs())
            .fold(0.0, f64::max),
    }
}

struct UniversalRankScore {
    fixed: f64,
    maximum: f64,
}

fn score_universal_rank(sketch: &UnivMonQ, truth: &Truth) -> UniversalRankScore {
    let query = sketch.prepare_queries();
    let errors: Vec<_> = truth
        .probes
        .iter()
        .map(|&(value, exact)| {
            let estimate =
                query.estimate_rank_universal(value as f64).unwrap() as f64 / truth.n as f64;
            (estimate - exact).abs()
        })
        .collect();
    UniversalRankScore {
        fixed: errors[errors.len() / 2],
        maximum: errors.into_iter().fold(0.0, f64::max),
    }
}

fn cdf_drift_q(left: &UnivMonQ, right: &UnivMonQ, truth: &Truth) -> f64 {
    let left = left.prepare_queries();
    let right = right.prepare_queries();
    truth
        .probes
        .iter()
        .map(|&(value, _)| {
            let a = left.rank(value as f64).unwrap() as f64 / truth.n as f64;
            let b = right.rank(value as f64).unwrap() as f64 / truth.n as f64;
            (a - b).abs()
        })
        .fold(0.0, f64::max)
}

fn worst_cdf_q(sketch: &UnivMonQ, truth: &Truth) -> (i64, f64, f64, f64) {
    let query = sketch.prepare_queries();
    truth
        .probes
        .iter()
        .map(|&(value, exact)| {
            let estimate = query.rank(value as f64).unwrap() as f64 / truth.n as f64;
            (value, exact, estimate, (estimate - exact).abs())
        })
        .max_by(|left, right| left.3.total_cmp(&right.3))
        .unwrap()
}

fn cdf_drift_kll(left: &KLL<i64>, right: &KLL<i64>, truth: &Truth) -> f64 {
    let left = left.cdf();
    let right = right.cdf();
    truth
        .probes
        .iter()
        .map(|&(value, _)| (left.quantile(value as f64) - right.quantile(value as f64)).abs())
        .fold(0.0, f64::max)
}

struct Truth {
    n: usize,
    sorted: Vec<i64>,
    probes: Vec<(i64, f64)>,
}

impl Truth {
    fn new(data: &[i64]) -> Self {
        let mut sorted = data.to_vec();
        sorted.sort_unstable();
        let probes = (0..=1_000)
            .map(|index| sorted[index * (sorted.len() - 1) / 1_000])
            .map(|value| {
                let rank =
                    sorted.partition_point(|item| *item <= value) as f64 / sorted.len() as f64;
                (value, rank)
            })
            .collect();
        Self {
            n: sorted.len(),
            sorted,
            probes,
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
