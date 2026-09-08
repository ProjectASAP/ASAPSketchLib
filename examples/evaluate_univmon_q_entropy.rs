//! Stress evaluation for experimental UnivMon-Q entropy estimators.
//!
//! Run with:
//!
//!   cargo run --release --example evaluate_univmon_q_entropy -- 200000 8 50000

use std::fmt;
use std::mem::size_of;

use asap_sketchlib::{UnivMonQ, UnivMonQConfig};

const SHARDS: usize = 8;

fn main() {
    let n = argument(1, 200_000);
    let trials = argument(2, 8);
    let domain = argument(3, 50_000);
    assert!(n > 0 && trials > 0 && domain > 1);

    let workloads = workloads(domain);
    let profiles = profiles(n);
    let mut results = vec![vec![Vec::<ResultRow>::new(); workloads.len()]; profiles.len()];

    println!("UnivMon-Q entropy stress evaluation");
    println!(
        "n={n}, trials={trials}, max_domain={domain}, workloads={}, shards={SHARDS}",
        workloads.len()
    );
    for profile in &profiles {
        let empty = UnivMonQ::new(profile.config).unwrap();
        println!(
            "profile={}: {:.3} MiB, width={}, halve/{}, candidates={}, samples={}",
            profile.name,
            empty.estimated_memory_bytes() as f64 / (1024.0 * 1024.0),
            profile.config.width,
            profile.config.width_halving_period,
            profile.config.candidates,
            profile.config.ordered_samples,
        );
    }

    for (workload_index, workload) in workloads.iter().enumerate() {
        for trial in 0..trials {
            let data = workload.generate(n, seed(workload_index, trial));
            let truth = exact_entropy(&data, workload.domain());
            for (profile_index, profile) in profiles.iter().enumerate() {
                let mut shards = build_shards(&data, profile.config, seed(profile_index, trial));
                let left = merge_left(shards.clone());
                let merged = merge_tree(&mut shards);
                let query = merged.prepare_queries();
                assert_eq!(query.estimate_l1(), n as f64);
                assert!(query.estimate_entropy().is_finite());
                assert_eq!(
                    query.estimate_entropy().to_bits(),
                    left.prepare_queries().estimate_entropy().to_bits(),
                    "merge-order entropy mismatch for {} / {}",
                    profile.name,
                    workload.name
                );
                results[profile_index][workload_index].push(ResultRow::new(
                    truth,
                    query.estimate_entropy_universal(),
                    query.estimate_entropy_occurrence(),
                    query.estimate_entropy(),
                ));
            }
        }
        eprintln!("completed {}", workload.name);
    }

    println!();
    println!(
        "{:>14} {:>20} {:>8} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "profile", "workload", "H", "U p95", "O p95", "A p95", "A abs", "O uses"
    );
    for (profile_index, profile) in profiles.iter().enumerate() {
        for (workload_index, workload) in workloads.iter().enumerate() {
            let summary = Summary::new(&results[profile_index][workload_index]);
            println!(
                "{:>14} {:>20} {:>8.4} {:>8.2}% {:>8.2}% {:>8.2}% {:>9.5} {:>3}/{:<3}",
                profile.name,
                workload.name,
                summary.truth,
                100.0 * summary.universal_p95,
                100.0 * summary.occurrence_p95,
                100.0 * summary.adaptive_p95,
                summary.adaptive_abs_p95,
                summary.occurrence_uses,
                trials,
            );
        }
    }

    println!();
    println!("worst adaptive cases");
    for (profile_index, profile) in profiles.iter().enumerate() {
        let mut worst: Vec<_> = workloads
            .iter()
            .enumerate()
            .map(|(index, workload)| {
                (
                    Summary::new(&results[profile_index][index]).adaptive_p95,
                    &workload.name,
                )
            })
            .collect();
        worst.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
        println!(
            "  {:>14}: {}",
            profile.name,
            worst
                .iter()
                .take(3)
                .map(|(error, name)| format!("{name}={:.2}%", 100.0 * error))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

fn argument(index: usize, default: usize) -> usize {
    std::env::args()
        .nth(index)
        .map(|value| value.parse().expect("arguments must be positive integers"))
        .unwrap_or(default)
}

#[derive(Clone)]
struct Profile {
    name: &'static str,
    config: UnivMonQConfig,
}

fn profiles(n: usize) -> Vec<Profile> {
    let compact = UnivMonQConfig {
        levels: 12,
        width: 4_096,
        width_halving_period: 3,
        depth: 5,
        counter_bits: 32,
        candidates: 256,
        ordered_samples: 4_096,
        hash_seed: 5,
    }
    .with_window_bound(n as u64, 1e-9)
    .unwrap();
    let mut small_sample = compact;
    small_sample.ordered_samples = 512;
    let mut large_sample = compact;
    large_sample.ordered_samples = 16_384;
    let mut equal_memory = compact;
    equal_memory.width_halving_period = 0;
    let target = compact.levels * compact.depth * compact.width * size_of::<i64>();
    equal_memory.candidates = largest_candidate_budget(equal_memory, target);
    vec![
        Profile {
            name: "compact-k512",
            config: small_sample,
        },
        Profile {
            name: "compact-k4096",
            config: compact,
        },
        Profile {
            name: "compact-k16k",
            config: large_sample,
        },
        Profile {
            name: "equal-memory",
            config: equal_memory,
        },
    ]
}

fn largest_candidate_budget(config: UnivMonQConfig, target: usize) -> usize {
    let memory = |candidates| {
        let mut candidate = config;
        candidate.candidates = candidates;
        UnivMonQ::new(candidate).unwrap().estimated_memory_bytes()
    };
    let mut low = config.candidates;
    let mut high = low * 2;
    assert!(memory(low) <= target);
    while memory(high) <= target {
        low = high;
        high *= 2;
    }
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if memory(middle) <= target {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

#[derive(Clone)]
struct Workload {
    name: String,
    kind: WorkloadKind,
}

#[derive(Clone)]
enum WorkloadKind {
    Uniform(usize),
    Zipf(Vec<f64>),
    HeadTail { head_mass: f64, domain: usize },
}

impl Workload {
    fn domain(&self) -> usize {
        match &self.kind {
            WorkloadKind::Uniform(domain) | WorkloadKind::HeadTail { domain, .. } => *domain,
            WorkloadKind::Zipf(cdf) => cdf.len(),
        }
    }

    fn generate(&self, n: usize, seed: u64) -> Vec<usize> {
        let mut rng = SplitMix64(seed);
        match &self.kind {
            WorkloadKind::Uniform(domain) => (0..n).map(|_| rng.next() as usize % domain).collect(),
            WorkloadKind::Zipf(cdf) => (0..n)
                .map(|_| {
                    let draw = rng.unit();
                    cdf.partition_point(|probability| *probability < draw)
                        .min(cdf.len() - 1)
                })
                .collect(),
            WorkloadKind::HeadTail { head_mass, domain } => (0..n)
                .map(|_| {
                    if rng.unit() < *head_mass {
                        0
                    } else {
                        1 + rng.next() as usize % (domain - 1)
                    }
                })
                .collect(),
        }
    }
}

impl fmt::Display for Workload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.name)
    }
}

fn workloads(domain: usize) -> Vec<Workload> {
    let mut result = Vec::new();
    for support in [100, 1_000, 10_000, domain] {
        let support = support.min(domain);
        result.push(Workload {
            name: format!("uniform-{support}"),
            kind: WorkloadKind::Uniform(support),
        });
    }
    for alpha in [0.5, 0.75, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 2.5] {
        result.push(Workload {
            name: format!("zipf-{alpha:.2}"),
            kind: WorkloadKind::Zipf(zipf_cdf(domain, alpha)),
        });
    }
    for head_mass in [0.5, 0.9, 0.99, 0.999, 0.9999] {
        result.push(Workload {
            name: format!("head-{head_mass:.4}"),
            kind: WorkloadKind::HeadTail { head_mass, domain },
        });
    }
    result
}

fn zipf_cdf(domain: usize, alpha: f64) -> Vec<f64> {
    let normalization: f64 = (1..=domain).map(|rank| (rank as f64).powf(-alpha)).sum();
    let mut running = 0.0;
    let mut cdf: Vec<f64> = (1..=domain)
        .map(|rank| {
            running += (rank as f64).powf(-alpha) / normalization;
            running
        })
        .collect();
    *cdf.last_mut().unwrap() = 1.0;
    cdf
}

fn exact_entropy(data: &[usize], domain: usize) -> f64 {
    let mut counts = vec![0_u64; domain];
    for &value in data {
        counts[value] += 1;
    }
    let total = data.len() as f64;
    counts
        .into_iter()
        .filter(|count| *count > 0)
        .map(|count| {
            let probability = count as f64 / total;
            -probability * probability.ln()
        })
        .sum()
}

fn build_shards(data: &[usize], config: UnivMonQConfig, seed: u64) -> Vec<UnivMonQ> {
    (0..SHARDS)
        .map(|shard| {
            let start = shard * data.len() / SHARDS;
            let end = (shard + 1) * data.len() / SHARDS;
            let mut sketch =
                UnivMonQ::new_with_source_id(config, seed ^ (shard as u64 + 1)).unwrap();
            for &value in &data[start..end] {
                sketch.add(&value);
            }
            sketch
        })
        .collect()
}

fn merge_left(mut sketches: Vec<UnivMonQ>) -> UnivMonQ {
    let mut result = sketches.remove(0);
    for sketch in sketches {
        result.merge(&sketch).unwrap();
    }
    result
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

#[derive(Clone, Copy)]
struct ResultRow {
    truth: f64,
    universal_rel: f64,
    occurrence_rel: f64,
    adaptive_rel: f64,
    adaptive_abs: f64,
    used_occurrence: bool,
}

impl ResultRow {
    fn new(truth: f64, universal: f64, occurrence: Option<f64>, adaptive: f64) -> Self {
        let occurrence = occurrence.unwrap_or(universal);
        Self {
            truth,
            universal_rel: relative_error(universal, truth),
            occurrence_rel: relative_error(occurrence, truth),
            adaptive_rel: relative_error(adaptive, truth),
            adaptive_abs: (adaptive - truth).abs(),
            used_occurrence: adaptive.to_bits() == occurrence.to_bits()
                && adaptive.to_bits() != universal.to_bits(),
        }
    }
}

struct Summary {
    truth: f64,
    universal_p95: f64,
    occurrence_p95: f64,
    adaptive_p95: f64,
    adaptive_abs_p95: f64,
    occurrence_uses: usize,
}

impl Summary {
    fn new(rows: &[ResultRow]) -> Self {
        Self {
            truth: percentile(&rows.iter().map(|row| row.truth).collect::<Vec<_>>(), 0.5),
            universal_p95: percentile(
                &rows.iter().map(|row| row.universal_rel).collect::<Vec<_>>(),
                0.95,
            ),
            occurrence_p95: percentile(
                &rows
                    .iter()
                    .map(|row| row.occurrence_rel)
                    .collect::<Vec<_>>(),
                0.95,
            ),
            adaptive_p95: percentile(
                &rows.iter().map(|row| row.adaptive_rel).collect::<Vec<_>>(),
                0.95,
            ),
            adaptive_abs_p95: percentile(
                &rows.iter().map(|row| row.adaptive_abs).collect::<Vec<_>>(),
                0.95,
            ),
            occurrence_uses: rows.iter().filter(|row| row.used_occurrence).count(),
        }
    }
}

fn relative_error(estimate: f64, truth: f64) -> f64 {
    (estimate - truth).abs() / truth.max(1e-12)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * quantile).ceil() as usize]
}

fn seed(workload: usize, trial: usize) -> u64 {
    0x2026_0809_5eed_0000 ^ (workload as u64) << 16 ^ trial as u64
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}
