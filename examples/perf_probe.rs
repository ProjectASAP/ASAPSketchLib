//! Throughput micro-benchmark probe for core sketches (CountMin, CountSketch, KLL).
//!
//! Prints machine-readable `BENCH` lines (median ns/item and Mitems/s) suitable
//! for parsing by scripts. Deterministic output across runs: inline xorshift64*
//! RNG with fixed seeds, no external dependencies.
//!
//! Run with:
//!
//!   cargo run --release --example perf_probe

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use asap_sketchlib::{Count, CountMin, DataInput, FastPath, KLL, RegularPath, Vector2D};

const N_ITEMS: usize = 200_000;
const ESTIMATE_N: usize = 100_000;
const QUANTILE_N: usize = 1_000;
const WARMUP_RUNS: usize = 5;
const MEASURED_RUNS: usize = 7;
const ROWS: usize = 3;
const COLS: usize = 4096;
const ZIPF_DOMAIN: usize = 8192;
const ZIPF_EXPONENT: f64 = 1.1;

struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Xorshift(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

fn sequential_keys(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn zipf_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut cum = Vec::with_capacity(ZIPF_DOMAIN);
    let mut acc = 0.0f64;
    for i in 0..ZIPF_DOMAIN {
        acc += 1.0 / (i as f64 + 1.0).powf(ZIPF_EXPONENT);
        cum.push(acc);
    }
    let total = acc;
    let mut rng = Xorshift::new(seed);
    (0..n)
        .map(|_| {
            let target = rng.next_f64_unit() * total;
            let idx = cum.partition_point(|&c| c < target).min(ZIPF_DOMAIN - 1);
            idx as u64
        })
        .collect()
}

fn bench(name: &str, n_items: usize, mut pass: impl FnMut() -> Duration) {
    for _ in 0..WARMUP_RUNS {
        pass();
    }
    let mut runs = Vec::with_capacity(MEASURED_RUNS);
    for _ in 0..MEASURED_RUNS {
        runs.push(pass());
    }
    runs.sort();
    let med = runs[runs.len() / 2];
    let ns_per_item = med.as_nanos() as f64 / n_items as f64;
    let mitems = n_items as f64 / med.as_secs_f64() / 1e6;
    println!("BENCH name={name} ns_per_item={ns_per_item:.2} mitems={mitems:.1}");
}

fn main() {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs();
    println!(
        "# perf_probe MUST be run in release mode (cargo run --release --example perf_probe); debug-build timings are meaningless"
    );
    println!(
        "HEADER timestamp_unix_secs={ts} warmup_runs={WARMUP_RUNS} measured_runs={MEASURED_RUNS}"
    );

    let seq_keys = sequential_keys(N_ITEMS);
    let zipf = zipf_keys(N_ITEMS, 0xDEAD_BEEF);

    bench("cms_fast_insert", N_ITEMS, || {
        let mut sk = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
        let start = Instant::now();
        for &k in &seq_keys {
            sk.insert(&DataInput::U64(k));
        }
        let elapsed = start.elapsed();
        std::hint::black_box(sk.estimate(&DataInput::U64(seq_keys[0])));
        elapsed
    });

    bench("cms_fast_insert_zipf", N_ITEMS, || {
        let mut sk = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
        let start = Instant::now();
        for &k in &zipf {
            sk.insert(&DataInput::U64(k));
        }
        let elapsed = start.elapsed();
        std::hint::black_box(sk.estimate(&DataInput::U64(zipf[0])));
        elapsed
    });

    let mut filled = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    for &k in &zipf {
        filled.insert(&DataInput::U64(k));
    }
    bench("cms_fast_estimate", ESTIMATE_N, || {
        let start = Instant::now();
        let mut acc = 0i64;
        for &k in &zipf[..ESTIMATE_N] {
            acc += filled.estimate(&DataInput::U64(k));
        }
        let elapsed = start.elapsed();
        std::hint::black_box(acc);
        elapsed
    });

    bench("cms_regular_insert", N_ITEMS, || {
        let mut sk = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(ROWS, COLS);
        let start = Instant::now();
        for &k in &seq_keys {
            sk.insert(&DataInput::U64(k));
        }
        let elapsed = start.elapsed();
        std::hint::black_box(sk.estimate(&DataInput::U64(seq_keys[0])));
        elapsed
    });

    bench("cs_fast_insert", N_ITEMS, || {
        let mut sk = Count::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
        let start = Instant::now();
        for &k in &seq_keys {
            sk.insert(&DataInput::U64(k));
        }
        let elapsed = start.elapsed();
        std::hint::black_box(sk.estimate(&DataInput::U64(seq_keys[0])));
        elapsed
    });

    let kll_values: Vec<f64> = {
        let mut rng = Xorshift::new(0x5EED_C0DE);
        (0..N_ITEMS).map(|_| rng.next_f64_unit() * 1000.0).collect()
    };
    bench("kll_update", N_ITEMS, || {
        let mut sk = KLL::<f64>::init_with_seed(200, 8, 42);
        let start = Instant::now();
        for &v in &kll_values {
            sk.update_data_input(&DataInput::F64(v)).unwrap();
        }
        let elapsed = start.elapsed();
        std::hint::black_box(sk.quantile(0.5));
        elapsed
    });

    let mut kll_filled = KLL::<f64>::init_with_seed(200, 8, 42);
    for &v in &kll_values {
        kll_filled.update_data_input(&DataInput::F64(v)).unwrap();
    }
    bench("kll_quantile_after", QUANTILE_N, || {
        let start = Instant::now();
        let mut acc = 0.0f64;
        for i in 0..QUANTILE_N {
            acc += kll_filled.quantile((i + 1) as f64 / (QUANTILE_N + 1) as f64);
        }
        let elapsed = start.elapsed();
        std::hint::black_box(acc);
        elapsed
    });

    // Same workload through the memoized CDF: rebuild cost paid once, then
    // binary search per query.
    let mut kll_cached = KLL::<f64>::init_with_seed(200, 8, 42);
    for &v in &kll_values {
        kll_cached.update_data_input(&DataInput::F64(v)).unwrap();
    }
    bench("kll_quantile_cached", QUANTILE_N, || {
        let start = Instant::now();
        let mut acc = 0.0f64;
        for i in 0..QUANTILE_N {
            acc += kll_cached.quantile_cached((i + 1) as f64 / (QUANTILE_N + 1) as f64);
        }
        let elapsed = start.elapsed();
        std::hint::black_box(acc);
        elapsed
    });
}
