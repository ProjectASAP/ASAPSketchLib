use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Stream generators
// ---------------------------------------------------------------------------

/// `n` draws from Zipf(s) over `[0, domain)`.
pub fn zipf_u64(n: usize, domain: usize, exponent: f64, seed: u64) -> Vec<u64> {
    let mut cdf: Vec<f64> = (0..domain)
        .map(|i| 1.0 / (i as f64 + 1.0).powf(exponent))
        .collect();
    for i in 1..cdf.len() {
        cdf[i] += cdf[i - 1];
    }
    let total = cdf[domain - 1];
    for x in cdf.iter_mut() {
        *x /= total;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            // Draw the variate ONCE; comparing inside the binary-search
            // closure would use a fresh random per probe and corrupt results.
            let u: f64 = rng.random();
            match cdf.binary_search_by(|p| p.partial_cmp(&u).unwrap()) {
                Ok(i) | Err(i) => (i as u64).min(domain as u64 - 1),
            }
        })
        .collect()
}

/// `n` draws from Uniform{[0, domain)}.
pub fn uniform_u64(n: usize, domain: u64, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random::<u64>() % domain).collect()
}

/// `n` iid Normal(mean, std) samples (Box-Muller).
pub fn normal_f64(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            mean + std * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect()
}

/// `n` iid Exponential(lambda) samples.
pub fn exponential_f64(n: usize, lambda: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| -rng.random::<f64>().max(1e-12).ln() / lambda)
        .collect()
}

/// Log-uniform adversarial values: `gamma^k * (1 + frac*(gamma-1))`, mixing
/// bucket-edge hits (frac ≈ 0) with interior values. Targets DDSketch mappings.
pub fn log_uniform_f64(n: usize, gamma: f64, k_range: std::ops::Range<i32>, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let k = rng.random_range(k_range.start..k_range.end);
            let frac: f64 = if rng.random::<f64>() < 0.2 {
                1e-9
            } else {
                rng.random()
            };
            gamma.powi(k) * (1.0 + frac * (gamma - 1.0))
        })
        .collect()
}

/// `n` values drawn from a small domain so most values repeat many times.
/// Rank-error checks must survive heavy ties, where one value legitimately
/// spans a wide rank interval.
pub fn duplicate_heavy_f64(n: usize, distinct: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (rng.random::<u64>() % distinct as u64) as f64)
        .collect()
}

/// Strictly increasing values: the worst case for a compactor that assumes
/// arrivals are unordered.
pub fn monotonic_f64(n: usize, start: f64, step: f64) -> Vec<f64> {
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Adversarial ordering of a fixed multiset: the sorted values are emitted
/// alternately from the two ends inward, so every compaction sees a stream
/// whose local order is maximally unlike its global order.
pub fn outside_in_ordering(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (mut lo, mut hi) = (0usize, values.len());
    let mut out = Vec::with_capacity(values.len());
    while lo < hi {
        out.push(values[lo]);
        lo += 1;
        if lo < hi {
            hi -= 1;
            out.push(values[hi]);
        }
    }
    out
}

/// `n` draws from Zipf(s) mapped onto a fixed value grid spanning
/// `[lo, hi]`, so a heavy-tailed key distribution becomes a heavy-tailed
/// *value* distribution usable by quantile sketches.
pub fn zipf_f64(n: usize, domain: usize, exponent: f64, lo: f64, hi: f64, seed: u64) -> Vec<f64> {
    let step = (hi - lo) / (domain - 1) as f64;
    zipf_u64(n, domain, exponent, seed)
        .into_iter()
        .map(|i| lo + step * i as f64)
        .collect()
}
