//! E2E suites for feature-gated (`experimental`) sketches: KMV cardinality,
//! UniformSampling's retention rate, and both tiers of EHUnivOptimized.
//!
//! CocoSketch and the Elastic sketch are a family of their own;
//! `tests/e2e_heavy_hitters.rs` covers them.
//!
//! Compiled only under `--features experimental`.

#![cfg(feature = "experimental")]

mod common;

use common::specs::{CardinalityConfidenceSpec, PrioritySampleSpec, Tally};
use common::{FreqTruth, assert_between, uniform_u64, zipf_u64};

use asap_sketchlib::{
    DataInput, EHUnivOptimized, EHUnivQueryResult, HeapItem, KMV, UniformSampling,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------- KMV

/// Gaussian quantile for KMV's bands.
///
/// `z = 4` is a two-sided tail of 6.3e-5 **under the normal approximation**.
/// KMV's estimator is a reciprocal Beta variate and is only asymptotically
/// normal, so this is an asymptotic, model-based band, not an exact tail —
/// which is why the coverage matrix files it as `asymptotic model` rather than
/// `theorem`. See `CardinalityConfidenceSpec` for the derivation.
const KMV_Z: f64 = 4.0;

/// Seed-list indices used as independent hash functions.
///
/// `KMV::insert_by_hash` is public and `DefaultXxHasher::hash64_seeded(d, ..)`
/// selects seed `d` from the library's 20-entry table, so a trial can be run
/// under a genuinely different hash rather than under a relabelled key set. The
/// randomness KMV's error model quantifies over *is* the hash, so this is the
/// only construction that makes a binomial over trials legitimate.
const KMV_HASH_SEEDS: [usize; 8] = [0, 1, 2, 3, 7, 11, 13, 17];

/// Distinct-count regimes, expressed relative to `k`, so every `k` is probed
/// below, exactly at, and above the point where the estimator switches on.
#[derive(Clone, Copy, Debug)]
enum KmvRegime {
    /// `n < k`: every hash is retained and `estimate()` returns the count.
    BelowK(usize),
    /// `n == k`: the buffer is full, so `estimate()` uses `(k-1)/U_(k)`.
    AtK,
    /// `n = multiple * k`.
    AboveK(usize),
}

impl KmvRegime {
    fn n(self, k: usize) -> usize {
        match self {
            KmvRegime::BelowK(sub) => k.saturating_sub(sub),
            KmvRegime::AtK => k,
            KmvRegime::AboveK(mult) => k * mult,
        }
    }
}

const KMV_REGIMES: [KmvRegime; 6] = [
    KmvRegime::BelowK(1),
    KmvRegime::BelowK(2),
    KmvRegime::AtK,
    KmvRegime::AboveK(2),
    KmvRegime::AboveK(8),
    KmvRegime::AboveK(32),
];

/// Feeds `n` distinct identities from a private namespace under hash seed `d`.
///
/// The namespace makes two trials that share a hash seed still see disjoint
/// identities, so no two trials in the battery share any randomness at all.
fn kmv_trial(k: usize, n: usize, seed_idx: usize, namespace: u64) -> KMV {
    let mut sketch: KMV = KMV::new(k);
    for i in 0..n as u64 {
        let hashed =
            <asap_sketchlib::DefaultXxHasher as asap_sketchlib::SketchHasher>::hash64_seeded(
                seed_idx,
                &DataInput::U64(namespace.wrapping_add(i)),
            );
        sketch.insert_by_hash(hashed);
    }
    sketch
}

/// KMV's relative standard error across every regime, over independent hash
/// seeds.
///
/// # The estimator, and the two numbers that follow from it
///
/// `KMV::estimate` returns `(k - 1) / U_(k)`, where `U_(k)` is the largest of
/// the `k` smallest normalized hashes. With `n` distinct uniform hashes
/// `U_(k) ~ Beta(k, n-k+1)`, so
///
/// ```text
///   E[(k-1)/U_(k)] = n                                  (unbiased for k > 1)
///   Var           = n (n - k + 1) / (k - 2)
///   RSE(n, k)     = sqrt( (n - k + 1) / (n (k - 2)) )  ->  1/sqrt(k - 2)
/// ```
///
/// The bands use the exact finite-`n` form, not the `1/sqrt(k - 2)` limit: it
/// is tighter at small `n` and correct at every `n`.
///
/// # Trial unit
///
/// One `(k, hash seed, regime)` triple is one sketch, one estimate, one trial —
/// and no two trials share a hash function or an identity. Rising checkpoints
/// read off one accumulating sketch would instead be *nested* estimates
/// sharing every retained hash.
#[test]
fn kmv_estimates_stay_inside_their_relative_standard_error_band_over_independent_hash_seeds() {
    const KS: [usize; 3] = [64, 1_024, 4_096];

    let mut tally = Tally::default();
    let mut namespace = 0u64;
    for &k in &KS {
        let spec = CardinalityConfidenceSpec::kmv(k, KMV_Z);
        for (s, &seed_idx) in KMV_HASH_SEEDS.iter().enumerate() {
            for regime in KMV_REGIMES {
                let n = regime.n(k);
                namespace = namespace.wrapping_add(NAMESPACE_STRIDE);
                let mut sketch = kmv_trial(k, n, seed_idx, namespace);
                let estimate = sketch.estimate();
                let outcome = spec.check(estimate, n);
                tally.record(outcome.is_ok(), || {
                    format!(
                        "k={k} seed_idx={seed_idx} (trial {s}) {regime:?} n={n} \
                         namespace={namespace:#x}: {}",
                        outcome.unwrap_err()
                    )
                });
            }
        }
    }
    tally.assert_independent_binomial(
        "KMV / relative standard error band",
        CardinalityConfidenceSpec::kmv(4_096, KMV_Z).per_check_failure(),
        &format!(
            "one trial = one sketch under one of {} independent hash seeds over its own \
             identity namespace; k in {KS:?}, regimes {KMV_REGIMES:?}, z={KMV_Z}, \
             sigma_rel = sqrt((n-k+1)/(n(k-2)))",
            KMV_HASH_SEEDS.len()
        ),
    );
}

/// Identity namespaces are spaced far enough apart that no two trials collide.
const NAMESPACE_STRIDE: u64 = 1 << 40;

/// The exact/estimated boundary is at `n < k`, **not** `n <= k`.
///
/// `KMV::estimate` returns the retained count verbatim only while
/// `k_vals.len() < k`. At `n == k` the buffer is full and the estimator runs,
/// so the answer is `(k-1)/U_(k)` over the maximum of `k` uniforms — unbiased,
/// but with a standard deviation of `sqrt(k / (k-2))`, about one element. A
/// spec that treated `n == k` as exact would be demanding exactness of a
/// genuinely random number; the earlier `n <= k` form did exactly that and only
/// passed because its checkpoint grid never landed on `k`.
#[test]
fn kmv_is_exact_below_k_and_estimates_at_k() {
    const K: usize = 512;
    let spec = CardinalityConfidenceSpec::kmv(K, KMV_Z);

    assert!(
        spec.is_exact_regime(K - 1),
        "n = k-1 must be the exact regime"
    );
    assert!(
        !spec.is_exact_regime(K),
        "n = k must be the estimated regime: the buffer is full, so estimate() \
         switches to (k-1)/U_(k)"
    );

    for n in [1usize, 2, K / 2, K - 2, K - 1] {
        let mut sketch: KMV = kmv_trial(K, n, 0, NAMESPACE_STRIDE * 900 + n as u64 * 4096);
        assert_eq!(
            sketch.estimate(),
            n as f64,
            "n={n} < k={K}: KMV retains every hash it has seen, so the count is exact"
        );
    }

    // At n == k the estimator runs. Averaged over independent hash seeds it is
    // unbiased; the point of this assertion is that the answer is *not* pinned
    // to k, so the exact-regime branch must not claim it.
    let mut estimates = Vec::new();
    for (i, &seed_idx) in KMV_HASH_SEEDS.iter().enumerate() {
        let mut sketch = kmv_trial(K, K, seed_idx, NAMESPACE_STRIDE * (1_000 + i as u64));
        let est = sketch.estimate();
        assert!(
            spec.check(est, K).is_ok(),
            "n = k = {K} under seed {seed_idx}: {}",
            spec.check(est, K).unwrap_err()
        );
        estimates.push(est);
    }
    assert!(
        estimates.iter().any(|e| *e != K as f64),
        "n = k = {K}: the estimator must be running here, but every seed returned \
         exactly {K}, which would mean the exact-regime branch is still active. \
         estimates: {estimates:?}"
    );
}

/// Duplicates are inert and a shard merge is **exact**, not merely in-band.
///
/// KMV retains the `k` smallest hashes of everything it has seen. Any hash
/// among the global `k` smallest is also among the `k` smallest of whichever
/// shard produced it, so merging the shards recovers exactly the single pass's
/// retained set — the estimate is the *same number*, not a second draw, so it
/// is asserted as an equality rather than scored against a confidence band.
#[test]
fn kmv_duplicates_are_inert_and_a_shard_merge_reproduces_the_single_pass_exactly() {
    const K: usize = 4_096;
    const STREAM_SEED: u64 = 5001;

    let spec = CardinalityConfidenceSpec::kmv(K, KMV_Z);
    let stream = uniform_u64(60_000, 500_000, STREAM_SEED);
    let truth: std::collections::HashSet<u64> = stream.iter().copied().collect();

    let mut full: KMV = KMV::new(K);
    let mut a: KMV = KMV::new(K);
    let mut b: KMV = KMV::new(K);
    for (i, k) in stream.iter().enumerate() {
        full.insert(&DataInput::U64(*k));
        if i % 2 == 0 {
            a.insert(&DataInput::U64(*k));
        } else {
            b.insert(&DataInput::U64(*k));
        }
    }

    let single = full.estimate();
    assert!(
        spec.check(single, truth.len()).is_ok(),
        "single pass over a duplicate-bearing stream: {}",
        spec.check(single, truth.len()).unwrap_err()
    );

    // Replaying the whole stream must not move the estimate at all: a hash
    // already retained is skipped, and one that was not retained cannot
    // displace anything smaller.
    for k in &stream {
        full.insert(&DataInput::U64(*k));
    }
    assert_eq!(
        full.estimate(),
        single,
        "replaying a stream KMV has already seen must leave the estimate untouched \
         (k={K}, stream_seed={STREAM_SEED}, distinct={})",
        truth.len()
    );

    let mut merged = a.clone();
    let mut rhs = b.clone();
    merged.merge(&mut rhs);
    assert_eq!(
        merged.estimate(),
        single,
        "an even/odd shard merge must reproduce the single pass's retained set exactly \
         (k={K}, stream_seed={STREAM_SEED}, distinct={})",
        truth.len()
    );
}

// ------------------------------------------------------------ UniformSampling

/// `UniformSampling` is **priority (bottom-k) sampling**, and that determines
/// which of its properties are exact and which are statistical.
///
/// Each update draws an independent uniform 64-bit priority, the entry is
/// inserted into a priority-ordered list, and the list is truncated to
/// `ceil(total_seen * rate)`. So:
///
/// - the retained size is `ceil(n * rate)` **exactly** — it is computed, never
///   sampled, so there is no slack for a band to assert around;
/// - because the priorities are i.i.d. and independent of the values, the
///   retained set is a uniform sample **without replacement** of that size from
///   the `n` values seen. That is where the statistics live.
///
/// This covers the exact half. The distributional half is the next test.
#[test]
fn uniform_sampling_retention_is_exact_at_every_rate_and_stream_size() {
    const RATES: [f64; 5] = [1.0, 0.5, 0.25, 0.1, 0.01];
    const SIZES: [usize; 6] = [0, 1, 7, 1_000, 10_000, 50_000];

    for &rate in &RATES {
        let spec = PrioritySampleSpec::new(rate, 4.0);
        for (i, &n) in SIZES.iter().enumerate() {
            let seed = 0x5A91_0100 + i as u64;
            let mut us = UniformSampling::with_seed(rate, seed);
            let stream: Vec<f64> = uniform_u64(n, u32::MAX as u64, 5002 + i as u64)
                .into_iter()
                .map(|v| v as f64)
                .collect();
            for v in &stream {
                us.update(*v);
            }

            assert_eq!(
                us.total_seen(),
                n as u64,
                "rate={rate} n={n}: total_seen counts every input"
            );
            assert_eq!(
                us.len(),
                spec.retained(n as u64),
                "rate={rate} n={n}: the retained size is ceil(n*rate) exactly, \
                 not a band (seed={seed:#x})"
            );
            assert!(
                us.len() <= n,
                "rate={rate} n={n}: cannot retain more than was seen"
            );

            // Every retained value came from the stream, and no value is
            // retained more often than it was fed: the sampler stores entries,
            // it does not synthesise or duplicate them.
            let mut budget: HashMap<u64, usize> = HashMap::new();
            for v in &stream {
                *budget.entry(v.to_bits()).or_default() += 1;
            }
            for sampled in us.samples() {
                let slot = budget.get_mut(&sampled.to_bits()).unwrap_or_else(|| {
                    panic!("rate={rate} n={n}: sample {sampled} was never fed in")
                });
                assert!(
                    *slot > 0,
                    "rate={rate} n={n}: sample {sampled} retained more times than it \
                     was fed"
                );
                *slot -= 1;
            }

            // Full sampling keeps everything, in the sense of a multiset.
            if rate >= 1.0 {
                assert_eq!(
                    us.len(),
                    n,
                    "rate=1.0 n={n}: full sampling must retain the whole stream"
                );
            }
        }
    }
}

/// The distributional half: the retained set really is a uniform sample without
/// replacement, so the sample mean tracks the population mean at the variance
/// that fact predicts.
///
/// ```text
///   Var[mean_hat] = (sigma_N^2 / m) * (N - m) / (N - 1)
/// ```
///
/// with `sigma_N^2` the population variance and the trailing factor the
/// finite-population correction. The band is `z` standard deviations of that,
/// so it is derived from the sampler's own algorithm and the stream's own
/// spread — not a percentage.
///
/// # Trial unit
///
/// One seed is one draw of the entire priority sequence, so **one sampler is
/// one trial**. Several statistics off one sampler are not independent, and
/// neither are two rates run from the same seed. The binomial therefore runs
/// over seeds, one outcome each.
#[test]
fn uniform_sampling_is_a_uniform_sample_without_replacement() {
    const RATES: [f64; 3] = [0.5, 0.1, 0.01];
    const N: usize = 20_000;
    const Z: f64 = 4.0;
    const TRIALS: usize = 24;

    // A deliberately skewed population: a uniform sample of it still has the
    // population mean, but a sampler biased toward early or late arrivals would
    // not, because the value is correlated with arrival order here.
    let stream: Vec<f64> = (0..N).map(|i| (i as f64).powf(1.7)).collect();
    let population = stream.len();
    let mean: f64 = stream.iter().sum::<f64>() / population as f64;
    let variance: f64 =
        stream.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / population as f64;

    for &rate in &RATES {
        let spec = PrioritySampleSpec::new(rate, Z);
        let m = spec.retained(population as u64);
        let sigma = spec.mean_sigma(population, m, variance);
        let mut tally = Tally::default();
        for t in 0..TRIALS {
            let seed = 0x5A91_0200 + t as u64;
            let mut us = UniformSampling::with_seed(rate, seed);
            for v in &stream {
                us.update(*v);
            }
            let samples = us.samples();
            assert_eq!(
                samples.len(),
                m,
                "rate={rate} seed={seed:#x}: retained size"
            );
            let sample_mean: f64 = samples.iter().sum::<f64>() / m as f64;
            let deviation = (sample_mean - mean).abs();
            tally.record(deviation <= Z * sigma, || {
                format!(
                    "seed={seed:#x}: sample mean {sample_mean:.3} vs population mean \
                     {mean:.3}, |deviation| {deviation:.3} > z*sigma = {Z}*{sigma:.3} = \
                     {:.3}",
                    Z * sigma
                )
            });
        }
        tally.assert_independent_binomial(
            &format!("UniformSampling rate={rate} / sample mean under SRSWOR"),
            spec.per_check_failure(),
            &format!(
                "one trial = one seed; population N={population} (values i^1.7, so value \
                 and arrival order are correlated), sample m={m}, sigma_N^2={variance:.4e}, \
                 sigma(mean)={sigma:.4}, seeds 0x5A910200..",
            ),
        );
    }
}

/// Merging two samplers keeps the combined budget and the union of the two
/// sample pools, truncated by priority.
///
/// Every part of this is exact: the merged size is `ceil((n1 + n2) * rate)`
/// capped by how many entries the two pools actually hold, the totals add, and
/// every survivor came from one of the two inputs. Rate mismatch is rejected.
#[test]
fn uniform_sampling_merge_keeps_the_combined_budget_exactly() {
    const RATES: [f64; 4] = [1.0, 0.5, 0.1, 0.01];

    for &rate in &RATES {
        let spec = PrioritySampleSpec::new(rate, 4.0);
        let mut left = UniformSampling::with_seed(rate, 42);
        let mut right = UniformSampling::with_seed(rate, 43);

        let left_stream: Vec<f64> = uniform_u64(10_000, u32::MAX as u64, 5002)
            .into_iter()
            .map(|v| v as f64)
            .collect();
        let right_stream: Vec<f64> = uniform_u64(5_000, u32::MAX as u64, 5003)
            .into_iter()
            .map(|v| v as f64 + 0.5)
            .collect();
        for v in &left_stream {
            left.update(*v);
        }
        for v in &right_stream {
            right.update(*v);
        }

        let pooled = left.len() + right.len();
        left.merge(&right).expect("same-rate merge");

        assert_eq!(
            left.total_seen(),
            15_000,
            "rate={rate}: merge must sum the totals"
        );
        assert_eq!(
            left.len(),
            spec.retained(15_000).min(pooled),
            "rate={rate}: the merged sample is the combined budget ceil(n*rate), capped \
             by the {pooled} entries the two pools actually held"
        );

        let allowed: std::collections::HashSet<u64> = left_stream
            .iter()
            .chain(right_stream.iter())
            .map(|v| v.to_bits())
            .collect();
        for sampled in left.samples() {
            assert!(
                allowed.contains(&sampled.to_bits()),
                "rate={rate}: merged sample {sampled} came from neither input"
            );
        }

        let other_rate = if rate == 1.0 { 0.5 } else { 1.0 };
        let mismatched = UniformSampling::with_seed(other_rate, 44);
        assert!(
            left.merge(&mismatched).is_err(),
            "rate={rate}: merging a rate-{other_rate} sampler must be rejected"
        );
    }
}

/// Replaying the same stream from the same seed must reproduce the sample
/// exactly, and a different seed must not: the priorities are the sampler's
/// whole randomness, and they are seeded.
#[test]
fn uniform_sampling_is_reproducible_from_its_seed() {
    const RATE: f64 = 0.1;
    let stream: Vec<f64> = uniform_u64(5_000, u32::MAX as u64, 5004)
        .into_iter()
        .map(|v| v as f64)
        .collect();

    let run = |seed: u64| {
        let mut us = UniformSampling::with_seed(RATE, seed);
        for v in &stream {
            us.update(*v);
        }
        us.samples()
    };

    assert_eq!(run(7), run(7), "the same seed must give the same sample");
    assert_ne!(
        run(7),
        run(8),
        "different seeds must draw different priorities, or the sample is not random \
         at all"
    );
}

// --------------------------------------------------------- EHUnivOptimized

/// A configuration whose promotion threshold (`layer_size * rows * cols`) is
/// small enough that the sketch tier is actually populated by a test-sized
/// stream, while `cols` stays wide enough for the UnivMon layers to be
/// meaningful.
fn promoting_eh(k: usize, window: u64) -> EHUnivOptimized {
    EHUnivOptimized::new(k, window, 32, EH_ROWS, EH_COLS, EH_LAYERS)
}

const EH_ROWS: usize = 3;
const EH_COLS: usize = 512;
const EH_LAYERS: usize = 2;
/// Smaller `k` merges buckets more aggressively, so the oldest map bucket
/// reaches the promotion threshold (`layer_size * rows * cols / 2` distinct
/// keys) within a test-sized stream.
const EH_K: usize = 2;

#[test]
fn eh_univ_optimized_map_tier_exact_windows() {
    let window = 100u64;
    let mut eh = EHUnivOptimized::with_defaults(2, window);

    for t in 0..150u64 {
        eh.update(t, &DataInput::U32((t % 10) as u32), (t as i64 % 3) + 1);
    }

    // Interval fully inside the retained range: map tier answers EXACTLY.
    match eh.query_interval(120, 149) {
        Some(EHUnivQueryResult::Map {
            freq_map,
            total_count,
        }) => {
            let expect_total: usize = (120..=149u64).map(|t| (t as i64 % 3 + 1) as usize).sum();
            assert_eq!(total_count, expect_total, "interval total");
            let mut expect_freq: HashMap<u32, i64> = HashMap::new();
            for t in 120..=149 {
                *expect_freq.entry((t % 10) as u32).or_insert(0) += (t as i64 % 3) + 1;
            }
            assert_eq!(
                expect_freq.len(),
                freq_map.len(),
                "distinct keys in interval"
            );
            for (k, v) in expect_freq {
                assert_eq!(
                    freq_map.get(&HeapItem::U32(k)),
                    Some(&v),
                    "interval count for key {k}"
                );
            }
        }
        _ => panic!("expected exact Map-tier result"),
    }
}

/// The map tier fills, promotes into the sketch tier, and the sketch tier then
/// answers interval queries — with the error model of a UnivMon sketch, not
/// the map tier's exactness.
///
/// `UnivMon::calc_l1` returns the maintained `bucket_size`, an exactly
/// accumulated weight, so L1 is asserted by equality on both tiers. L2 comes
/// out of UnivMon's recursive g-sum recurrence across `layer_size` sampled
/// substreams, for which the crate publishes no closed-form constant, so it is
/// a documented empirical band and is named as such.
///
/// The stream is deliberately **skewed**. UnivMon recovers a g-sum from the
/// heavy hitters it can isolate at each layer, and promotion guarantees the
/// promoted sketch holds at least `layer_size * rows * cols / 2` distinct keys
/// — so a flat stream leaves the counters uniformly loaded with nothing
/// recoverable. Measured on this configuration, L2 lands within 3% of exact on
/// a Zipf(1.1) stream and 57% *below* exact on a uniform one. That is a
/// property of the estimator, not of this test, and the skew is stated here
/// rather than hidden inside a wide tolerance.
///
/// Band source: measured on this exact configuration and stream (Zipf(1.1)
/// over 30k keys, n=200k, k=2, rows=3, cols=512, layers=2, stream seed 4242),
/// where L2 lands 3.0% below exact and entropy 20.4% above it.
#[test]
fn eh_univ_optimized_promotes_into_the_sketch_tier_and_answers_from_it() {
    const N: usize = 80_000;
    const DOMAIN: usize = 30_000;
    const STREAM_SEED: u64 = 4242;

    let keys = zipf_u64(N, DOMAIN, 1.1, STREAM_SEED);
    let mut eh = promoting_eh(EH_K, 100_000_000);
    for (t, k) in keys.iter().enumerate() {
        eh.update(t as u64, &DataInput::U64(*k), 1);
    }

    assert!(
        eh.um_buckets.len() >= 2,
        "test premise: the map tier must have promoted at least two sketch \
         buckets (max_map_size={}, got {} sketch and {} map buckets)",
        eh.max_map_size,
        eh.um_buckets.len(),
        eh.map_buckets.len()
    );

    // Query exactly the span the sketch tier covers. Both endpoints are public
    // bucket boundaries, so the reference window is known exactly.
    let lo = eh.um_buckets[0].min_time;
    let hi = eh.um_buckets[eh.um_buckets.len() - 1].max_time;
    let result = eh.query_interval(lo, hi).expect("sketch-tier interval");
    match &result {
        EHUnivQueryResult::Sketch(_) => {}
        EHUnivQueryResult::Map { .. } => panic!(
            "expected a Sketch-tier result for the promoted span [{lo}, {hi}]; \
             got the exact map tier, so promotion did not take effect"
        ),
    }

    // Exact truth over the same window.
    let mut truth = FreqTruth::default();
    for t in lo..=hi {
        truth.observe(keys[t as usize] as i64);
    }
    let context = format!(
        "k={EH_K} rows={EH_ROWS} cols={EH_COLS} layers={EH_LAYERS} n={N} domain={DOMAIN} \
         zipf(1.1) stream_seed={STREAM_SEED}, promoted span [{lo}, {hi}] with {} distinct \
         keys and L1={}",
        truth.distinct(),
        truth.total()
    );

    // L1 is maintained, not estimated.
    assert_eq!(
        result.calc_l1(),
        truth.total() as f64,
        "sketch-tier L1 must be exact. {context}"
    );

    let l2_truth = truth.l2_norm();
    assert_between(
        result.calc_l2(),
        l2_truth * 0.90,
        l2_truth * 1.10,
        &format!("EHUnivOptimized sketch-tier L2 (empirical band). {context}"),
    );

    let entropy_bits = truth.entropy(true);
    assert_between(
        result.calc_entropy(),
        entropy_bits * 0.90,
        entropy_bits * 1.40,
        &format!("EHUnivOptimized sketch-tier entropy in bits (empirical band). {context}"),
    );
}

/// The sketch tier's cardinality estimate is structurally unrecoverable, and
/// this test pins that rather than letting a wide tolerance hide it.
///
/// UnivMon recovers `F0` from the deepest sampled layer, which needs roughly
/// `log2(distinct)` layers before the surviving substream is small enough for
/// the per-layer heap to hold it. But promotion only fires once the oldest map
/// bucket holds `layer_size * rows * cols / 2` distinct keys, so the promoted
/// sketch always carries at least `192 * layer_size` distinct keys (at the
/// smallest sensible `rows = 3`, `cols = 128`), needing about
/// `7.6 + log2(layer_size)` layers — more than `layer_size` for every
/// configuration below about 13 layers, and raising `layer_size` raises the
/// promotion threshold in lockstep. The two requirements cannot both be met by
/// tuning.
///
/// Measured consequence: on the configurations probed here the sketch tier
/// reports 0 to 11 distinct values for windows holding 16k to 40k of them —
/// a ~100% underestimate, not a noisy one. Callers must take cardinality from
/// the map tier or from a dedicated distinct-count sketch.
///
/// The lower guard makes a future fix fail this test loudly instead of leaving
/// a stale claim in the suite.
#[test]
fn eh_univ_optimized_sketch_tier_cardinality_is_documented_as_unrecoverable() {
    const N: usize = 80_000;
    const DOMAIN: usize = 30_000;

    let keys = zipf_u64(N, DOMAIN, 1.1, 4242);
    let mut eh = promoting_eh(EH_K, 100_000_000);
    for (t, k) in keys.iter().enumerate() {
        eh.update(t as u64, &DataInput::U64(*k), 1);
    }
    let lo = eh.um_buckets[0].min_time;
    let hi = eh.um_buckets[eh.um_buckets.len() - 1].max_time;
    let result = eh.query_interval(lo, hi).expect("sketch-tier interval");

    let mut truth = FreqTruth::default();
    for t in lo..=hi {
        truth.observe(keys[t as usize] as i64);
    }
    let distinct = truth.distinct() as f64;
    let reported = result.calc_card();

    // The promotion threshold really does outrun the layer count.
    let needed_layers = distinct.log2().ceil();
    assert!(
        needed_layers > EH_LAYERS as f64,
        "premise: recovering F0 for {distinct} distinct keys needs about \
         {needed_layers} sampling layers, but this configuration has {EH_LAYERS}"
    );
    assert!(
        reported < distinct * 0.10,
        "sketch-tier cardinality reported {reported} for {distinct} distinct keys. \
         If UnivMon's F0 recovery was fixed or EHUnivOptimized stopped coupling the \
         promotion threshold to layer_size, delete this test and assert a real band \
         instead of leaving this stale documentation in place"
    );
}

/// A span crossing the tier boundary is answered as a Sketch: the map buckets
/// inside the interval are replayed into a merged UnivMon. The result must
/// still carry the whole window's L1 exactly.
#[test]
fn eh_univ_optimized_answers_a_mixed_map_and_sketch_interval() {
    const N: u64 = 60_000;
    const DOMAIN: u64 = 20_000;

    let mut eh = promoting_eh(EH_K, 1_000_000);
    for t in 0..N {
        eh.update(t, &DataInput::U64(t % DOMAIN), 1);
    }
    assert!(!eh.um_buckets.is_empty() && !eh.map_buckets.is_empty());

    // From inside the sketch tier to the newest map bucket.
    let lo = eh.um_buckets[0].min_time;
    let hi = eh.map_buckets[eh.map_buckets.len() - 1].max_time;
    let result = eh.query_interval(lo, hi).expect("mixed interval");
    match &result {
        EHUnivQueryResult::Sketch(_) => {}
        EHUnivQueryResult::Map { .. } => {
            panic!("a span starting in the sketch tier must be answered as a Sketch")
        }
    }

    let mut truth = FreqTruth::default();
    for t in lo..=hi {
        truth.observe((t % DOMAIN) as i64);
    }
    assert_eq!(
        result.calc_l1(),
        truth.total() as f64,
        "mixed-tier L1 must still be the exact window weight over [{lo}, {hi}]"
    );
}

/// Expiry drops buckets older than the window on both tiers, and the retained
/// span shrinks accordingly. Bucket bookkeeping is arithmetic, so these are
/// equalities rather than bands.
#[test]
fn eh_univ_optimized_expires_buckets_past_the_window() {
    const WINDOW: u64 = 5_000;
    let mut eh = promoting_eh(8, WINDOW);
    for t in 0..40_000u64 {
        eh.update(t, &DataInput::U64(t % 8_000), 1);
    }

    let min_time = eh.get_min_time().expect("buckets present");
    let max_time = eh.get_max_time().expect("buckets present");
    assert_eq!(max_time, 39_999, "newest retained time");

    // The expiry rule drops a bucket once its *max_time* falls below the
    // cutoff, so a retained bucket may still reach back before the cutoff with
    // its min_time — that is bucket granularity, not a leak. What must hold is
    // that no retained bucket lies entirely in the past.
    let cutoff = max_time - WINDOW;
    assert!(
        min_time <= max_time,
        "retained span [{min_time}, {max_time}] must be ordered"
    );
    for b in &eh.um_buckets {
        assert!(
            b.max_time >= cutoff,
            "sketch bucket [{}, {}] is entirely older than the cutoff {cutoff}",
            b.min_time,
            b.max_time
        );
    }
    for b in &eh.map_buckets {
        assert!(
            b.max_time >= cutoff,
            "map bucket [{}, {}] is entirely older than the cutoff {cutoff}",
            b.min_time,
            b.max_time
        );
    }
    assert!(
        eh.cover(min_time, max_time),
        "the structure must cover its own retained span"
    );
    assert!(
        !eh.cover(0, max_time),
        "expired times must no longer be covered"
    );

    // Continuing past the window must not grow the structure without bound.
    let buckets_before = eh.bucket_count();
    for t in 40_000..70_000u64 {
        eh.update(t, &DataInput::U64(t % 8_000), 1);
    }
    assert!(
        eh.bucket_count() <= buckets_before * 2,
        "bucket count grew from {buckets_before} to {} over a fixed window",
        eh.bucket_count()
    );
}

/// Recycling through the sketch pool must not leak state between windows: a
/// sketch taken from the pool has to behave like a fresh one.
#[test]
fn eh_univ_optimized_reuses_pooled_sketches_without_leaking_state() {
    const DOMAIN: u64 = 20_000;
    // Long run with a short window, so buckets are created, promoted, merged
    // and expired repeatedly and the pool is exercised.
    let mut eh = promoting_eh(8, 20_000);
    for t in 0..80_000u64 {
        eh.update(t, &DataInput::U64(t % DOMAIN), 1);
    }

    let lo = eh.get_min_time().expect("buckets");
    let hi = eh.get_max_time().expect("buckets");
    let result = eh.query_interval(lo, hi).expect("interval");
    let mut truth = FreqTruth::default();
    for t in lo..=hi {
        truth.observe((t % DOMAIN) as i64);
    }
    // A pooled sketch that kept counters from a previous window would inflate
    // L1 above the retained window's exact weight.
    assert_eq!(
        result.calc_l1(),
        truth.total() as f64,
        "L1 over the retained span [{lo}, {hi}] must equal the exact window weight; \
         a larger value means a recycled sketch carried state across windows"
    );
}

/// The two tiers must agree on a span they can both answer: a weighted,
/// skewed stream small enough to stay in the map tier gives exact per-key
/// counts, which is the reference the sketch tier's estimates are compared to
/// elsewhere in this file.
#[test]
fn eh_univ_optimized_map_tier_matches_exact_per_key_counts_on_a_skewed_stream() {
    const N: u64 = 4_000;
    let mut eh = EHUnivOptimized::with_defaults(4, 1_000_000);
    let keys = zipf_u64(N as usize, 64, 1.2, 9_001);
    let mut truth = FreqTruth::default();
    for (t, k) in keys.iter().enumerate() {
        let w = 1 + (t % 3) as i64;
        eh.update(t as u64, &DataInput::U64(*k), w);
        truth.observe_weighted(*k as i64, w);
    }

    match eh.query_interval(0, N - 1) {
        Some(EHUnivQueryResult::Map {
            freq_map,
            total_count,
        }) => {
            assert_eq!(total_count as i64, truth.total(), "map-tier total weight");
            for (k, c) in truth.pairs() {
                assert_eq!(
                    freq_map.get(&HeapItem::U64(k as u64)),
                    Some(&c),
                    "map-tier count for key {k}"
                );
            }
        }
        other => panic!(
            "expected the map tier to answer a {N}-update stream exactly; got {}",
            match other {
                Some(EHUnivQueryResult::Sketch(_)) => "a Sketch result",
                _ => "nothing",
            }
        ),
    }
}

// ---------------------------------------------------------------------------
// The documented input matrix
// ---------------------------------------------------------------------------

/// The document's KMV row is the twelve numbered inputs at `k = 32` and
/// `k = 128`, and its UniformSampling row is `(1)`, `(2)`, `(7)` and `(8)` at
/// rate 0.1.
///
/// # The two KMV bands
///
/// KMV's estimator is `(k - 1) / U_(k)`, whose relative standard error is
/// `1 / sqrt(k - 2)`: 18.3% at `k = 32` and 8.9% at `k = 128`. The document's
/// per-`k` bands are `z = 4` of that — 73% and 36% — and are asserted with no
/// tolerance. Each input is additionally checked against the spec's own band at
/// its cardinality, which uses the exact finite-`n` standard error and is
/// therefore slightly tighter than the asymptotic number the document quotes.
mod documented_matrix {
    use super::common::inputs::{KEY_INPUT_IDS, key_input};
    use super::common::specs::{CardinalityConfidenceSpec, PrioritySampleSpec};
    use super::*;

    use std::collections::HashSet;

    /// The document's relative-error band per `k`.
    fn documented_band(k: usize) -> f64 {
        match k {
            32 => 0.73,
            128 => 0.36,
            other => panic!("({other}) is not one of the documented k values"),
        }
    }

    fn kmv_documented_input(k: usize) {
        let spec = CardinalityConfidenceSpec::kmv(k, KMV_Z);
        let band = documented_band(k);

        for id in KEY_INPUT_IDS {
            let input = key_input(id);
            let distinct = input.keys.iter().copied().collect::<HashSet<i64>>().len();
            let mut sketch: KMV = KMV::new(k);
            for key in &input.keys {
                sketch.insert(&input.data(*key));
            }
            let estimate = sketch.estimate();
            let context = input.context();

            if let Err(detail) = spec.check(estimate, distinct) {
                panic!("KMV k={k}: {detail}. distinct={distinct} {context}");
            }

            let rel = ((estimate - distinct as f64) / distinct as f64).abs();
            assert!(
                rel <= band,
                "KMV k={k} on ({id}): estimate {estimate:.0} against {distinct} distinct is \
                 {:.2}% off, past the documented {:.0}%. {context}",
                rel * 100.0,
                band * 100.0
            );

            // Duplicate replay is inert: KMV retains the k smallest hashes of
            // what it has seen, and it has seen all of these already.
            let before = sketch.estimate();
            for key in &input.keys {
                sketch.insert(&input.data(*key));
            }
            assert_eq!(
                sketch.estimate(),
                before,
                "replaying ({id}) moved the k={k} KMV estimate"
            );
        }
    }

    #[test]
    fn kmv_k32_over_the_documented_inputs_holds_its_error_model() {
        kmv_documented_input(32);
    }

    #[test]
    fn kmv_k128_over_the_documented_inputs_holds_its_error_model() {
        kmv_documented_input(128);
    }

    /// UniformSampling at the documented rate over one documented input.
    ///
    /// The retained size is asserted twice: against the document's 15% band
    /// around `n * rate`, and against `ceil(n * rate)` exactly, because this
    /// implementation is a priority sample with a deterministic budget rather
    /// than an independent Bernoulli filter. Only the second would notice a
    /// sampler that started retaining one entry too few.
    fn uniform_sampling_documented_input(id: u8) {
        const RATE: f64 = 0.1;

        let input = key_input(id);
        let values = input.values();
        let seed = 0x5A91_0300 + id as u64;
        let mut us = UniformSampling::with_seed(RATE, seed);
        for v in &values {
            us.update(*v);
        }
        let context = format!("{} rate={RATE} seed={seed:#x}", input.context());

        let n = values.len();
        let expected = n as f64 * RATE;
        let retained = us.len() as f64;
        assert!(
            (retained - expected).abs() <= expected * 0.15,
            "retained {retained} samples against an expected {expected}, past the documented \
             15%. {context}"
        );
        let spec = PrioritySampleSpec::new(RATE, 4.0);
        assert_eq!(
            us.len(),
            spec.retained(n as u64),
            "the retained size is the deterministic budget ceil(n*rate). {context}"
        );
        assert_eq!(
            us.total_seen(),
            n as u64,
            "total_seen must be exact. {context}"
        );

        let seen: HashSet<u64> = values.iter().map(|v| v.to_bits()).collect();
        for sample in us.samples() {
            assert!(
                seen.contains(&sample.to_bits()),
                "sample {sample} was never in the stream. {context}"
            );
        }

        // A same-rate merge unions the pools and sums the totals. The second
        // half of the stream is fed to a second sketch so the merge has
        // something to union that the first has not seen.
        let mut left = UniformSampling::with_seed(RATE, seed);
        let mut right = UniformSampling::with_seed(RATE, seed + 1);
        let half = n / 2;
        for v in &values[..half] {
            left.update(*v);
        }
        for v in &values[half..] {
            right.update(*v);
        }
        let pooled = left.len() + right.len();
        left.merge(&right).expect("same-rate merge");
        assert_eq!(
            left.total_seen(),
            n as u64,
            "a merge must sum total_seen. {context}"
        );
        assert_eq!(
            left.len(),
            spec.retained(n as u64).min(pooled),
            "a merge keeps the combined budget, capped by the {pooled} entries the two \
             pools held. {context}"
        );
        for sample in left.samples() {
            assert!(
                seen.contains(&sample.to_bits()),
                "merged sample {sample} was never in the stream. {context}"
            );
        }
    }

    macro_rules! documented_uniform_sampling_inputs {
        ($($name:ident => $id:literal;)*) => {
            $(
                #[test]
                fn $name() {
                    uniform_sampling_documented_input($id);
                }
            )*
        };
    }

    documented_uniform_sampling_inputs! {
        uniform_sampling_input_1_retains_its_documented_budget => 1;
        uniform_sampling_input_2_retains_its_documented_budget => 2;
        uniform_sampling_input_7_retains_its_documented_budget => 7;
        uniform_sampling_input_8_retains_its_documented_budget => 8;
    }
}
