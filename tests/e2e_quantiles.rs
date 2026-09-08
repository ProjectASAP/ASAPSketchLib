//! E2E quantile pipelines on synthetic numeric streams.
//!
//! The two quantile families here answer *different questions* and are held to
//! *different* error metrics:
//!
//! - **KLL** promises **rank** error: the returned value's rank must be within
//!   `eps(k)` of the requested `q`. Its value error is unbounded — on a
//!   heavy-tailed stream a correct KLL can return a value 100x off and still
//!   be within its guarantee — so checking `|est - true| / true` against a
//!   percentage tests nothing KLL claims.
//! - **DDSketch** promises **relative value** error: the returned value must
//!   be within `alpha` of the exact order statistic. Its rank error is
//!   unbounded, so a rank band tests nothing DDSketch claims.
//!
//! Neither may be run through the other's battery. `KllRankSpec` and
//! `RelativeQuantileSpec` in `common::specs` keep them apart.
//!
//! Every sketch here is built with an explicit compaction seed. KLL's coin is
//! the sketch's own randomness and is entirely separate from the stream seed;
//! the wall-clock-seeded `KLL::init_kll` / `KLLDynamic::init_kll` constructors
//! cannot be used in an accuracy test because a failure would not reproduce.
//!
//! # Trial units
//!
//! The KLL number `eps(k) = 2.446 / k^0.9433` is Apache DataSketches'
//! characterization fit to the 99th percentile of the **maximum** rank error
//! over a whole quantile grid — not a theorem about this implementation, and
//! not a per-`q` failure probability. So a KLL battery reduces each sketch to
//! one number (its worst rank error over the grid), each sketch gets its **own
//! compaction seed**, and the binomial is taken over those independent seeds.
//!
//! DDSketch's guarantee is deterministic — bucket width alone, no hash and no
//! sampling — so its batteries tolerate zero violations and no statistical
//! model applies. The two shipped implementations answer a quantile query with
//! **different order statistics**, so each is compared against the truth for
//! its own convention; see `DdRankConvention`.

mod common;

use common::specs::{DdRankConvention, KllRankSpec, RelativeQuantileSpec, Tally, rank_error};
use common::{
    NumericTruth, assert_between, duplicate_heavy_f64, exponential_f64, log_uniform_f64,
    monotonic_f64, normal_f64, outside_in_ordering, uniform_u64, zipf_f64,
};

use asap_sketchlib::message_pack_format::portable::ddsketch::DdSketch as PortableDds;
use asap_sketchlib::message_pack_format::portable::hydra_kll::HydraKllSketch;
use asap_sketchlib::{
    DDSketch, DataInput, KLL, KLLConfig, KLLDynamic, TumblingWindow, UnivMonQ, UnivMonQConfig,
};
use std::collections::HashMap;

// --------------------------------------------------------------------- KLL

/// The quantile grid every rank-error battery runs, including both endpoints
/// and the far tails the interior grid would otherwise never reach.
const RANK_QS: [f64; 7] = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0];

/// Fixed sketch (compaction-coin) seeds. Independent of the stream seeds
/// below: one controls which values arrive, the other controls which of them
/// survive compaction, and conflating the two makes a failure impossible to
/// localise.
const KLL_SKETCH_SEEDS: [u64; 4] = [0x5EED_0001, 0x5EED_0002, 0x5EED_0003, 0x5EED_0004];

/// A distinct compaction seed per trial index, so that no two sketches in a
/// battery share a coin sequence.
///
/// The rank-error batteries are binomials over these trials, which is only
/// valid if each trial is a fresh draw of KLL's randomness. Reusing one seed
/// across shapes or feed modes would silently correlate the outcomes; the
/// multiplier is a large odd constant so consecutive indices land far apart in
/// the seed space.
fn kll_trial_seed(trial: u64) -> u64 {
    0x5EED_0000_0000_0001u64.wrapping_add(trial.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Named stream shapes with a fixed seed each. Covers the light-tailed,
/// heavy-tailed, tie-dense, sorted and adversarially-ordered cases a
/// compaction scheme can behave differently on.
fn rank_streams(trial: usize, n: usize) -> Vec<(&'static str, Vec<f64>)> {
    let s = 0xA5A5_0000u64 + trial as u64 * 7919;
    vec![
        (
            "uniform",
            uniform_u64(n, 100_000_000, s)
                .into_iter()
                .map(|v| v as f64)
                .collect(),
        ),
        ("normal", normal_f64(n, 1_000.0, 250.0, s + 1)),
        ("zipf", zipf_f64(n, 8_192, 1.1, 1e6, 1e7, s + 2)),
        // Fifty distinct values over tens of thousands of observations: a
        // single value legitimately spans several percent of the rank space,
        // which is exactly where a value-error check would misfire and a
        // rank-interval check must not.
        ("duplicate-heavy", duplicate_heavy_f64(n, 50, s + 3)),
        // Sorted input: every compaction sees a run that is already in global
        // order.
        ("monotonic", monotonic_f64(n, 0.0, 1.0)),
        // Adversarial ordering: the same multiset emitted from both ends
        // inward, so no prefix resembles the whole.
        (
            "outside-in",
            outside_in_ordering(normal_f64(n, 5_000.0, 900.0, s + 4)),
        ),
    ]
}

/// How a sketch was fed. Every mode must satisfy the same rank contract —
/// bulk ingestion and merging are supposed to be equivalent to the loop, and
/// a mode that quietly loses weight shows up here as a rank-error failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feed {
    SinglePass,
    BulkUpdate,
    ShardMerge,
    TumblingMerge,
}

const FEEDS: [Feed; 4] = [
    Feed::SinglePass,
    Feed::BulkUpdate,
    Feed::ShardMerge,
    Feed::TumblingMerge,
];

fn feed_kll(feed: Feed, k: i32, seed: u64, values: &[f64]) -> KLL<f64> {
    match feed {
        Feed::SinglePass => {
            let mut s = KLL::init_kll_with_seed(k, seed);
            for v in values {
                s.update(v);
            }
            s
        }
        Feed::BulkUpdate => {
            let mut s = KLL::init_kll_with_seed(k, seed);
            s.bulk_update(values);
            s
        }
        Feed::ShardMerge => {
            // Four shards, each with its own coin seed, merged pairwise into a
            // tree rather than a chain so merge order is exercised too.
            let mut shards: Vec<KLL<f64>> = (0..4)
                .map(|i| KLL::init_kll_with_seed(k, seed.wrapping_add(0x1000 * (i + 1))))
                .collect();
            for (i, v) in values.iter().enumerate() {
                shards[i % 4].update(v);
            }
            let (mut a, b) = (shards.remove(0), shards.remove(0));
            let (mut c, d) = (shards.remove(0), shards.remove(0));
            a.merge(&b);
            c.merge(&d);
            a.merge(&c);
            a
        }
        Feed::TumblingMerge => {
            // Windows are closed and recycled through the pool; `query_all`
            // merges every retained window plus the active one. Pool reuse
            // runs `clear()`, which re-seeds from the stored seed, so the
            // result is reproducible across rotations.
            let cfg = KLLConfig {
                k: k as usize,
                m: 8,
                seed: Some(seed),
            };
            let window = (values.len() / 8).max(1) as u64;
            let mut tw: TumblingWindow<KLL> = TumblingWindow::new(window, 32, cfg, 4);
            for (t, v) in values.iter().enumerate() {
                tw.insert(t as u64, &DataInput::F64(*v), 0);
            }
            tw.query_all()
        }
    }
}

fn feed_kll_dynamic(feed: Feed, k: i32, seed: u64, values: &[f64]) -> Option<KLLDynamic<f64>> {
    match feed {
        Feed::SinglePass => {
            let mut s = KLLDynamic::init_kll_with_seed(k, seed);
            for v in values {
                s.update(v);
            }
            Some(s)
        }
        Feed::BulkUpdate => {
            let mut s = KLLDynamic::init_kll_with_seed(k, seed);
            s.bulk_update(values);
            Some(s)
        }
        Feed::ShardMerge => {
            let mut shards: Vec<KLLDynamic<f64>> = (0..4)
                .map(|i| KLLDynamic::init_kll_with_seed(k, seed.wrapping_add(0x1000 * (i + 1))))
                .collect();
            for (i, v) in values.iter().enumerate() {
                shards[i % 4].update(v);
            }
            let (mut a, b) = (shards.remove(0), shards.remove(0));
            let (mut c, d) = (shards.remove(0), shards.remove(0));
            a.merge(&b);
            c.merge(&d);
            a.merge(&c);
            Some(a)
        }
        // KLLDynamic is not a `TumblingWindowSketch`; the framework hosts the
        // fixed-layout `KLL` only. Nothing to cover.
        Feed::TumblingMerge => None,
    }
}

/// Both KLL implementations against the Apache DataSketches maximum-rank-error
/// characterization `eps(k) = 2.446 / k^0.9433`, over a grid of
/// `(k, distribution, feed mode)` with an independent compaction seed each.
///
/// # Why the acceptance rule looks like this
///
/// The constant is a least-squares fit to the 99th percentile of the
/// **maximum** rank error DataSketches' characterization runs observed across a
/// whole quantile grid. Two things follow.
///
/// First, the quantity it bounds is a per-sketch maximum, so the seven `q`
/// values of one sketch are *one* outcome, not seven Bernoulli draws: they
/// share a single compaction history and are strongly dependent (a compaction
/// that displaces the median displaces its neighbours too).
///
/// Second, the 1% is that single outcome's failure probability, so the only
/// legitimate battery is over **independent compaction seeds**. Every trial
/// below therefore gets its own seed, derived from its index, and the binomial
/// acceptance rule runs over trials.
///
/// This is a characterization target imported from another implementation of
/// the same compact KLL layout — not a theorem proved about this code — and the
/// test name says `characterization`, not `theorem`.
#[test]
fn kll_family_stays_within_the_datasketches_maximum_rank_error_characterization() {
    const N: usize = 30_000;
    const KS: [i32; 3] = [64, 200, 800];
    const REPEATS: usize = 4;

    let mut tallies: HashMap<String, Tally> = HashMap::new();
    let mut trial = 0u64;
    for repeat in 0..REPEATS {
        for (shape, values) in rank_streams(repeat, N) {
            let truth = NumericTruth::new(values.clone());
            for &k in &KS {
                let spec = KllRankSpec::datasketches(k as usize);
                for &feed in &FEEDS {
                    // One fresh compaction seed per trial: that is what makes
                    // the trials independent draws of KLL's own randomness.
                    let seed = kll_trial_seed(trial);
                    trial += 1;
                    let fixed = feed_kll(feed, k, seed, &values);
                    spec.record_trial(
                        tallies.entry(format!("KLL/{feed:?}")).or_default(),
                        &format!("k={k} shape={shape} seed={seed:#x} n={N}"),
                        truth.sorted(),
                        &RANK_QS,
                        |q| fixed.quantile(q),
                    );

                    let seed = kll_trial_seed(trial);
                    trial += 1;
                    if let Some(dynamic) = feed_kll_dynamic(feed, k, seed, &values) {
                        spec.record_trial(
                            tallies.entry(format!("KLLDynamic/{feed:?}")).or_default(),
                            &format!("k={k} shape={shape} seed={seed:#x} n={N}"),
                            truth.sorted(),
                            &RANK_QS,
                            |q| dynamic.quantile(q),
                        );
                    }
                }
            }
        }
    }

    let mut labels: Vec<String> = tallies.keys().cloned().collect();
    labels.sort();
    for label in labels {
        let tally = tallies.remove(&label).expect("label just enumerated");
        tally.assert_independent_binomial(
            &format!("{label} / maximum normalized rank error per sketch"),
            KllRankSpec::datasketches(200).trial_failure_probability,
            &format!(
                "one trial = one sketch with its own compaction seed, scored on its \
                 worst rank error over the whole q grid. n={N}, k in {KS:?}, \
                 seeds kll_trial_seed(0..), stream shapes {:?}, q grid {RANK_QS:?}",
                rank_streams(0, 1)
                    .iter()
                    .map(|(s, _)| *s)
                    .collect::<Vec<_>>()
            ),
        );
    }
}

/// Rank error must shrink as `k` grows, at the rate the characterization
/// states. A hard-coded tolerance cannot see this: it passes identically at
/// `k = 64` and `k = 800`, so it would not notice a `k` that stopped being
/// wired through to the compactor capacities at all.
///
/// This is a **structural** claim about the `k` -> accuracy wiring, checked on
/// four fixed compaction seeds per `k`. The absolute check that each `k`'s
/// worst-of-four stays inside `eps(k)` is a deterministic regression pin, not a
/// tail test: with only four trials it has no power to measure a 1% per-trial
/// failure rate, and the batteries above are where that is judged.
#[test]
fn kll_rank_error_shrinks_with_k_as_the_characterization_predicts() {
    const N: usize = 100_000;
    let values: Vec<f64> = uniform_u64(N, 100_000_000, 0xC0FF_EE01)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let truth = NumericTruth::new(values.clone());
    let qs: Vec<f64> = (1..20).map(|i| i as f64 / 20.0).collect();

    let mut worst: Vec<(i32, f64)> = Vec::new();
    for k in [64i32, 256, 1024] {
        let mut w: f64 = 0.0;
        for &seed in &KLL_SKETCH_SEEDS {
            let mut s = KLL::init_kll_with_seed(k, seed);
            for v in &values {
                s.update(v);
            }
            for &q in &qs {
                let v = s.quantile(q);
                let (excl, incl) = truth.rank_interval(v);
                let err = if q < excl {
                    excl - q
                } else if q > incl {
                    q - incl
                } else {
                    0.0
                };
                w = w.max(err);
            }
        }
        worst.push((k, w));
    }

    for (k, w) in &worst {
        let eps = KllRankSpec::datasketches(*k as usize).epsilon();
        assert!(
            *w <= eps,
            "KLL k={k}: worst rank error {w:.5} exceeds eps(k)={eps:.5} \
             (n={N}, sketch seeds {KLL_SKETCH_SEEDS:02x?}, stream seed 0xC0FFEE01)"
        );
    }
    // 16x more capacity must buy at least a 4x tighter rank error; the
    // characterization predicts 16^0.9433 = 13.4x.
    let (k_lo, w_lo) = worst[0];
    let (k_hi, w_hi) = worst[2];
    assert!(
        w_lo >= w_hi * 4.0,
        "raising k from {k_lo} to {k_hi} moved the worst rank error only from \
         {w_lo:.5} to {w_hi:.5}; the characterization predicts a {:.1}x improvement, so \
         k is not reaching the compactor capacities",
        (k_hi as f64 / k_lo as f64).powf(0.9433)
    );
}

// ------------------------------------------------------- KLL bulk ingestion

/// A distinct compaction seed per bulk-ingestion case, kept away from the
/// rank batteries' trial space so the two never share a coin sequence.
fn kll_bulk_seed(case: u64) -> u64 {
    kll_trial_seed(0x8B14_0000u64.wrapping_add(case))
}

/// The rank batteries' shapes plus three the compaction path can behave
/// differently on: two heavy-tailed spreads, and a run short enough that no
/// compactor ever fills.
fn bulk_cases(trial: usize, n: usize) -> Vec<(&'static str, Vec<f64>)> {
    let alpha = 0.01;
    let gamma = (1.0 + alpha) / (1.0 - alpha);
    let mut cases = rank_streams(trial, n);
    cases.push(("exponential", exponential_f64(n, 1e-3, 3007)));
    cases.push(("log-uniform", log_uniform_f64(n, gamma, 5..40, 3005)));
    cases.push((
        "sequential-10",
        (0..10).map(|i| i as f64 * 1.7 + 11.0).collect(),
    ));
    cases
}

/// `bulk_update` and a loop of `update` must produce the *same sketch*, not
/// two sketches that happen to land in the same rank band.
///
/// `kll_family_stays_within_the_datasketches_maximum_rank_error_characterization`
/// already runs `Feed::BulkUpdate` against `eps(k)`. What a band cannot see is
/// a bulk path that compacts in a different order and lands on a different —
/// still legal — sketch, so what is asserted here is equality: serialized
/// bytes, retained count, and quantile bits over `RANK_QS`.
#[test]
fn kll_bulk_update_is_byte_identical_to_the_update_loop() {
    const N: usize = 20_000;
    for (case, (shape, values)) in bulk_cases(0, N).iter().enumerate() {
        let seed = kll_bulk_seed(case as u64);
        let mut via_loop: KLL<f64> = KLL::init_kll_with_seed(200, seed);
        for v in values {
            via_loop.update(v);
        }
        let mut via_bulk: KLL<f64> = KLL::init_kll_with_seed(200, seed);
        via_bulk.bulk_update(values);

        let ctx = format!("{shape} n={} sketch_seed={seed:#x}", values.len());
        assert_eq!(
            via_loop.count(),
            via_bulk.count(),
            "{ctx}: retained count diverged"
        );
        assert_eq!(
            via_loop.serialize_to_bytes().unwrap(),
            via_bulk.serialize_to_bytes().unwrap(),
            "{ctx}: serialized bytes diverged"
        );
        for &q in &RANK_QS {
            assert_eq!(
                via_loop.quantile(q).to_bits(),
                via_bulk.quantile(q).to_bits(),
                "{ctx} q={q}: quantile bits differ"
            );
        }
    }
}

/// `KLLDynamic` under the same equality, on its own wire form.
#[test]
fn kll_dynamic_bulk_update_is_byte_identical_to_the_update_loop() {
    const N: usize = 10_000;
    for (case, (shape, values)) in bulk_cases(1, N).iter().enumerate() {
        let seed = kll_bulk_seed(0x0100 + case as u64);
        let mut via_loop = KLLDynamic::<f64>::init_kll_with_seed(200, seed);
        for v in values {
            via_loop.update(v);
        }
        let mut via_bulk = KLLDynamic::<f64>::init_kll_with_seed(200, seed);
        via_bulk.bulk_update(values);

        let ctx = format!("{shape} n={} sketch_seed={seed:#x}", values.len());
        assert_eq!(
            via_loop.count(),
            via_bulk.count(),
            "{ctx}: retained mass diverged"
        );
        assert_eq!(
            via_loop.serialize_to_bytes().unwrap(),
            via_bulk.serialize_to_bytes().unwrap(),
            "{ctx}: serialized bytes diverged"
        );
        for &q in &RANK_QS {
            assert_eq!(
                via_loop.quantile(q).to_bits(),
                via_bulk.quantile(q).to_bits(),
                "{ctx} q={q}: quantile bits differ"
            );
        }
    }
}

/// The two degenerate slice lengths: an empty `bulk_update` leaves the sketch
/// byte-for-byte untouched, and a one-element `bulk_update` equals a single
/// `update`.
#[test]
fn kll_bulk_update_on_a_degenerate_slice_matches_the_loop() {
    let seed = kll_bulk_seed(0x0200);
    let mut sk: KLL<f64> = KLL::init_kll_with_seed(200, seed);
    for v in &[1.0f64, 2.0, 3.0] {
        sk.update(v);
    }
    let count = sk.count();
    let median = sk.quantile(0.5);
    let bytes = sk.serialize_to_bytes().unwrap();

    sk.bulk_update(&[]);
    assert_eq!(sk.count(), count, "an empty bulk_update changed the count");
    assert_eq!(
        sk.quantile(0.5).to_bits(),
        median.to_bits(),
        "an empty bulk_update moved the median"
    );
    assert_eq!(
        sk.serialize_to_bytes().unwrap(),
        bytes,
        "an empty bulk_update changed the serialized state"
    );

    let mut one: KLL<f64> = KLL::init_kll_with_seed(200, seed);
    let mut once: KLL<f64> = KLL::init_kll_with_seed(200, seed);
    one.bulk_update(&[42.0]);
    once.update(&42.0);
    assert_eq!(
        one.serialize_to_bytes().unwrap(),
        once.serialize_to_bytes().unwrap(),
        "a one-element bulk_update diverged from one update"
    );
}

/// The `DataInput` batch path matches the loop, and a non-numeric element
/// stops it at that element with the preceding prefix already applied.
#[test]
fn kll_bulk_update_data_input_matches_the_loop_and_stops_at_the_first_non_numeric() {
    let seed = kll_bulk_seed(0x0300);
    let values = normal_f64(5_000, 100.0, 20.0, 8001);
    let inputs: Vec<DataInput> = values.iter().map(|v| DataInput::F64(*v)).collect();

    let mut via_loop: KLL<f64> = KLL::init_kll_with_seed(200, seed);
    for v in &inputs {
        via_loop.update_data_input(v).unwrap();
    }
    let mut via_bulk: KLL<f64> = KLL::init_kll_with_seed(200, seed);
    via_bulk.bulk_update_data_input(&inputs).unwrap();
    assert_eq!(
        via_loop.serialize_to_bytes().unwrap(),
        via_bulk.serialize_to_bytes().unwrap(),
        "DataInput bulk vs loop bytes diverged (sketch_seed={seed:#x}, stream seed 8001)"
    );

    let mixed = vec![
        DataInput::F64(1.0),
        DataInput::String("x".into()),
        DataInput::F64(2.0),
    ];
    let mut stopped: KLL<f64> = KLL::init_kll_with_seed(200, seed);
    assert!(
        stopped.bulk_update_data_input(&mixed).is_err(),
        "a non-numeric element must return an error"
    );
    assert_eq!(
        stopped.count(),
        1,
        "the prefix before the non-numeric element must stay applied"
    );

    let mut empty: KLL<f64> = KLL::init_kll_with_seed(200, seed);
    empty.bulk_update_data_input(&[]).unwrap();
    assert_eq!(empty.count(), 0, "an empty DataInput batch must be a no-op");
}

// ---------------------------------------------------------------- DDSketch

/// Accuracy parameters spanning three orders of magnitude. The tightest is
/// where floating-point round-off in the logarithmic mapping is most likely
/// to matter, the loosest is where the bucket span per key is widest.
const DDS_ALPHAS: [f64; 4] = [0.001, 0.01, 0.05, 0.1];

/// Same q grid as the rank battery, so the two contracts are compared on the
/// same questions rather than on grids chosen to flatter each sketch.
const DDS_QS: [f64; 7] = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0];

/// Streams for the relative-error battery. `adversarial` places a fifth of its
/// mass exactly on bucket lower edges, where the mapping's error is at its
/// maximum of exactly `alpha` and one ULP of drift decides the bucket.
fn dds_streams(alpha: f64, n: usize, seed: u64) -> Vec<(&'static str, Vec<f64>)> {
    let gamma = (1.0 + alpha) / (1.0 - alpha);
    vec![
        (
            "adversarial-bucket-edges",
            log_uniform_f64(n, gamma, 5..40, seed),
        ),
        (
            "normal",
            normal_f64(n, 1_000.0, 250.0, seed + 1)
                .into_iter()
                .filter(|v| *v > 0.0)
                .collect(),
        ),
        ("exponential", exponential_f64(n, 1e-3, seed + 2)),
        (
            "uniform",
            uniform_u64(n, 9_000_000, seed + 3)
                .into_iter()
                .map(|v| 1_000_000.0 + v as f64)
                .collect(),
        ),
        ("zipf", zipf_f64(n, 8_192, 1.1, 1e6, 1e7, seed + 4)),
        // Nine decades in one stream: the bucket store spans a large index
        // range and the mapping is exercised far from 1.0 in both directions.
        (
            "wide-dynamic-range",
            uniform_u64(n, 1_000_000, seed + 5)
                .into_iter()
                .enumerate()
                .map(|(i, v)| 10f64.powi((i % 10) as i32 - 4) * (1.0 + v as f64 / 1_000_000.0))
                .collect(),
        ),
    ]
}

/// DDSketch's relative-value-error guarantee, for the core sketch and the
/// portable wire twin, at every supported alpha.
///
/// Each implementation is compared against the exact order statistic **of its
/// own rank convention**: `DDSketch::get_value_at_quantile` answers
/// `sorted[ceil(q*n) - 1]`, while the portable `DdSketch::quantile` answers
/// `sorted[floor(q*(n-1))]`. They are different questions (see
/// `ddsketch_core_and_portable_answer_different_order_statistics`), so a single
/// truth helper would score one of them against the other one's question and
/// quietly absorb the difference into `alpha`.
///
/// The tolerance is `alpha + numerical_slack`, where the slack is a few ULP of
/// the logarithmic mapping — never a percentage of alpha, which would license
/// breaking the advertised guarantee by that percentage.
#[test]
fn ddsketch_core_and_portable_satisfy_the_relative_value_error_contract() {
    const SAMPLE_SIZES: [usize; 3] = [1_000, 20_000, 100_000];

    for &alpha in &DDS_ALPHAS {
        let core_spec = RelativeQuantileSpec::core(alpha);
        let port_spec = RelativeQuantileSpec::portable(alpha);
        let mut core_tally = Tally::default();
        let mut port_tally = Tally::default();
        for (i, &n) in SAMPLE_SIZES.iter().enumerate() {
            let seed = 3_005_000u64 + i as u64 * 101 + (alpha * 1e6) as u64;
            for (label, values) in dds_streams(alpha, n, seed) {
                let mut core = DDSketch::new(alpha);
                let mut port = PortableDds::new(alpha);
                for v in &values {
                    core.add(v);
                    port.update(*v);
                }
                let truth = NumericTruth::new(values.clone());
                assert_eq!(
                    core.get_count() as usize,
                    truth.len(),
                    "{label} alpha={alpha} n={n} seed={seed}: core dropped samples"
                );
                assert_eq!(
                    port.total_count() as usize,
                    truth.len(),
                    "{label} alpha={alpha} n={n} seed={seed}: portable dropped samples"
                );
                core_spec.tally_into(&mut core_tally, truth.sorted(), &DDS_QS, |q| {
                    core.get_value_at_quantile(q)
                });
                port_spec.tally_into(&mut port_tally, truth.sorted(), &DDS_QS, |q| {
                    port.quantile(q)
                });
            }
        }
        // The guarantee is deterministic, not probabilistic: DDSketch's error
        // comes from bucket width alone, with no hash and no sampling, so a
        // single violation is a defect and none are tolerated.
        let context = format!(
            "alpha={alpha} sizes={SAMPLE_SIZES:?} shapes={:?} q grid {DDS_QS:?}",
            dds_streams(alpha, 1, 0)
                .iter()
                .map(|(s, _)| *s)
                .collect::<Vec<_>>()
        );
        core_tally.assert_none(&format!("core DDSketch alpha={alpha}"), &context);
        port_tally.assert_none(&format!("portable DdSketch alpha={alpha}"), &context);
    }
}

/// The two implementations answer a quantile query with **different order
/// statistics**, and this pins the divergence rather than letting `alpha`
/// absorb it.
///
/// - `DDSketch::get_value_at_quantile` uses `rank = ceil(q * n)`, 1-based.
/// - Portable `DdSketch::quantile` uses `target = floor(q * (n - 1))`, 0-based
///   — the lower-quantile convention of the DDSketch paper and of DataDog's
///   reference implementation, which is also what the wire format's Go twin
///   answers.
///
/// **The decision taken here is to keep both.** The portable type exists to be
/// byte- and answer-compatible with `sketchlib-go`, so its convention is fixed
/// by an external contract; the core type's `ceil` convention is what its own
/// callers have been reading for the life of the API, and it is what lets `q=0`
/// and `q=1` return the exactly retained minimum and maximum. Changing either
/// silently moves numbers under existing callers, and the divergence is only
/// observable at small `n` or ragged `q` — precisely the cases pinned below.
/// What is *not* acceptable is leaving it undocumented, or scoring both against
/// one truth helper, which is what the previous revision did.
///
/// The probes are chosen so the two formulas disagree: at `n = 3, q = 0.4` the
/// core answers `sorted[1]` and the portable `sorted[0]`.
#[test]
fn ddsketch_core_and_portable_answer_different_order_statistics() {
    // (n, q, expected core 0-based index, expected portable 0-based index)
    const PROBES: [(usize, f64, usize, usize); 8] = [
        (3, 0.4, 1, 0),
        (4, 0.34, 1, 1),
        (4, 0.3, 1, 0),
        (7, 0.2, 1, 1),
        (7, 0.6, 4, 3),
        (5, 0.5, 2, 2),
        (10, 0.25, 2, 2),
        (10, 0.15, 1, 1),
    ];

    // Values one bucket apart at the coarsest alpha, so a one-rank difference
    // is a different bucket and therefore a different answer — not two ranks
    // that happen to share a representative.
    const ALPHA: f64 = 0.01;
    let gamma = (1.0 + ALPHA) / (1.0 - ALPHA);

    let mut disagreements = 0usize;
    for &(n, q, core_idx, port_idx) in &PROBES {
        let values: Vec<f64> = (0..n).map(|i| 100.0 * gamma.powi(3 * i as i32)).collect();
        let sorted = values.clone();

        assert_eq!(
            DdRankConvention::CeilNearestRank.index(n, q),
            core_idx,
            "core convention ceil(q*n)-1 at n={n} q={q}"
        );
        assert_eq!(
            DdRankConvention::LowerFloor.index(n, q),
            port_idx,
            "portable convention floor(q*(n-1)) at n={n} q={q}"
        );

        let mut core = DDSketch::new(ALPHA);
        let mut port = PortableDds::new(ALPHA);
        for v in &values {
            core.add(v);
            port.update(*v);
        }

        let core_spec = RelativeQuantileSpec::core(ALPHA);
        let port_spec = RelativeQuantileSpec::portable(ALPHA);
        let core_est = core.get_value_at_quantile(q).expect("non-empty");
        let port_est = port.quantile(q).expect("non-empty");

        if let Err(detail) = core_spec.check(q, core_est, sorted[core_idx]) {
            panic!("core DDSketch n={n} q={q}: {detail}");
        }
        if let Err(detail) = port_spec.check(q, port_est, sorted[port_idx]) {
            panic!("portable DdSketch n={n} q={q}: {detail}");
        }

        if core_idx != port_idx {
            disagreements += 1;
            // Each implementation must be answering *its own* order statistic,
            // so the two answers must differ here: if they agreed, one of them
            // would have silently changed convention.
            assert!(
                core_est != port_est,
                "n={n} q={q}: the conventions pick different order statistics \
                 ({core_idx} vs {port_idx}) but both returned {core_est}"
            );
        }
    }
    assert!(
        disagreements >= 3,
        "the probe set must contain cases where the two conventions genuinely \
         disagree; only {disagreements} did"
    );
}

/// Endpoint behaviour, which is **not** the same on the two implementations.
///
/// - The core sketch tracks the exact minimum and maximum beside the bucket
///   store, and `get_value_at_quantile` short-circuits `q <= 0` and `q >= 1` to
///   them. Its endpoints are therefore *exact*, with zero error.
/// - The portable sketch carries no min/max scalars at all — they were removed
///   from the wire — so its endpoints are ordinary bucket representatives and
///   are only guaranteed within `alpha`. The previous revision of this test
///   claimed it "clamps its bucket representative into [min, max], so its
///   endpoints are exact too", which is not what the code does.
#[test]
fn ddsketch_core_endpoints_are_exact_and_portable_endpoints_are_alpha_relative() {
    for &alpha in &DDS_ALPHAS {
        let values = log_uniform_f64(5_000, (1.0 + alpha) / (1.0 - alpha), 3..30, 4_242);
        let truth = NumericTruth::new(values.clone());
        let mut core = DDSketch::new(alpha);
        let mut port = PortableDds::new(alpha);
        for v in &values {
            core.add(v);
            port.update(*v);
        }
        assert_eq!(
            core.get_value_at_quantile(0.0),
            Some(truth.min()),
            "core alpha={alpha}: q=0 must be the exact minimum"
        );
        assert_eq!(
            core.get_value_at_quantile(1.0),
            Some(truth.max()),
            "core alpha={alpha}: q=1 must be the exact maximum"
        );
        assert_eq!(
            core.min(),
            Some(truth.min()),
            "core alpha={alpha}: min() must be exact"
        );
        assert_eq!(
            core.max(),
            Some(truth.max()),
            "core alpha={alpha}: max() must be exact"
        );

        // The portable twin holds only buckets, so its endpoints get the same
        // relative-value guarantee as any other quantile and nothing more.
        let spec = RelativeQuantileSpec::portable(alpha);
        let (p0, p1) = (port.quantile(0.0).unwrap(), port.quantile(1.0).unwrap());
        if let Err(detail) = spec.check(0.0, p0, truth.min()) {
            panic!("portable alpha={alpha} at q=0: {detail}");
        }
        if let Err(detail) = spec.check(1.0, p1, truth.max()) {
            panic!("portable alpha={alpha} at q=1: {detail}");
        }
    }
}

/// Values placed deliberately on and around a bucket boundary. Whichever side
/// of the edge the mapping lands on, the returned representative must stay
/// within `alpha` — a value at the lower edge is over-estimated by exactly
/// `alpha`, and one that slips into the bucket below is under-estimated by
/// exactly `alpha`, so both sides are tight and neither may exceed it.
#[test]
fn ddsketch_satisfies_the_relative_error_contract_at_bucket_boundaries() {
    for &alpha in &DDS_ALPHAS {
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        // A one-sample sketch answers every q with that one sample, so the two
        // rank conventions coincide here by construction (both index 0) and the
        // probe isolates the mapping from any rank effect.
        let core_spec = RelativeQuantileSpec::core(alpha);
        let port_spec = RelativeQuantileSpec::portable(alpha);
        for k in [-40i32, -7, 0, 1, 13, 60, 200] {
            let edge = gamma.powi(k);
            if edge <= 0.0 || !edge.is_finite() {
                continue;
            }
            let probes = [
                edge,                         // exactly the lower edge
                edge * (1.0 - f64::EPSILON),  // one ULP below it
                edge * (1.0 + f64::EPSILON),  // one ULP above it
                edge * gamma.sqrt(),          // bucket interior
                edge * gamma * (1.0 - 1e-12), // just under the upper edge
            ];
            for probe in probes {
                let mut core = DDSketch::new(alpha);
                let mut port = PortableDds::new(alpha);
                core.add(&probe);
                port.update(probe);
                if core.get_count() == 0 {
                    continue; // outside the indexable range at this alpha
                }
                // A single-sample sketch answers every q with that sample's
                // bucket, so this isolates the mapping from any rank effect.
                let est = core.get_value_at_quantile(0.5).unwrap();
                if let Err(detail) = core_spec.check(0.5, est, probe) {
                    panic!("core DDSketch alpha={alpha} gamma^{k} boundary probe: {detail}");
                }
                let pest = port.quantile(0.5).unwrap();
                if let Err(detail) = port_spec.check(0.5, pest, probe) {
                    panic!("portable DdSketch alpha={alpha} gamma^{k} boundary probe: {detail}");
                }
            }
        }
    }
}

/// Merging shards and replaying promotion deltas must both leave the relative
/// error contract intact, because both write into the same bucket store the
/// guarantee is stated over.
#[test]
fn ddsketch_merge_and_delta_replay_preserve_the_relative_error_contract() {
    use asap_sketchlib::message_pack_format::portable::ddsketch::DdSketchDelta;
    use asap_sketchlib::octo_delta::DdDelta;

    const N: usize = 40_000;
    for &alpha in &DDS_ALPHAS {
        let spec = RelativeQuantileSpec::core(alpha);
        let port_spec = RelativeQuantileSpec::portable(alpha);
        let values = zipf_f64(N, 8_192, 1.1, 1e3, 1e7, 7_654_321 + (alpha * 1e5) as u64);
        let truth = NumericTruth::new(values.clone());

        // Four-way shard merge, core side.
        let mut shards: Vec<DDSketch> = (0..4).map(|_| DDSketch::new(alpha)).collect();
        for (i, v) in values.iter().enumerate() {
            shards[i % 4].add(v);
        }
        let mut merged = shards.remove(0);
        for s in &shards {
            merged.merge(s).expect("same-alpha merge");
        }
        assert_eq!(
            merged.get_count() as usize,
            N,
            "alpha={alpha}: merge must preserve total count"
        );
        let mut merge_tally = Tally::default();
        spec.tally_into(&mut merge_tally, truth.sorted(), &DDS_QS, |q| {
            merged.get_value_at_quantile(q)
        });
        merge_tally.assert_none(
            &format!("core DDSketch alpha={alpha} after a 4-way shard merge"),
            &format!("zipf n={N} over [1e3, 1e7]"),
        );

        // Delta replay, core side: rebuild a sketch from bucket deltas only
        // and require the same contract of it. The replayed sketch's own
        // min/max come from bucket representatives, so its endpoints are
        // within alpha rather than exact — the interior q grid is what the
        // contract covers here.
        let single = {
            let mut s = DDSketch::new(alpha);
            for v in &values {
                s.add(v);
            }
            s
        };
        let mut replayed = DDSketch::new(alpha);
        for (i, &count) in single.store_counts().iter().enumerate() {
            if count > 0 {
                replayed.apply_delta(DdDelta {
                    index: single.store_offset() + i as i32,
                    value: count,
                });
            }
        }
        assert_eq!(
            replayed.get_count(),
            single.get_count(),
            "alpha={alpha}: delta replay must reproduce the total count"
        );
        let mut replay_tally = Tally::default();
        spec.tally_into(&mut replay_tally, truth.sorted(), &DDS_QS[1..6], |q| {
            replayed.get_value_at_quantile(q)
        });
        replay_tally.assert_none(
            &format!("core DDSketch alpha={alpha} rebuilt from bucket deltas"),
            &format!(
                "zipf n={N} over [1e3, 1e7], interior q grid {:?}",
                &DDS_QS[1..6]
            ),
        );

        // Portable side: merge, then a delta carrying the same buckets.
        let mut pa = PortableDds::new(alpha);
        let mut pb = PortableDds::new(alpha);
        for (i, v) in values.iter().enumerate() {
            if i % 2 == 0 {
                pa.update(*v);
            } else {
                pb.update(*v);
            }
        }
        pa.merge(&pb).expect("same-alpha portable merge");
        let mut port_tally = Tally::default();
        port_spec.tally_into(&mut port_tally, truth.sorted(), &DDS_QS, |q| pa.quantile(q));
        port_tally.assert_none(
            &format!("portable DdSketch alpha={alpha} after merge"),
            &format!("zipf n={N} over [1e3, 1e7]"),
        );

        let mut pr = PortableDds::new(alpha);
        let delta = DdSketchDelta {
            buckets: pa
                .store_counts
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(i, c)| (pa.store_offset + i as i32, *c))
                .collect(),
            d_count: pa.total_count() as i64,
            ..DdSketchDelta::default()
        };
        pr.apply_delta(&delta).expect("benign delta");
        let mut pr_tally = Tally::default();
        port_spec.tally_into(&mut pr_tally, truth.sorted(), &DDS_QS[1..6], |q| {
            pr.quantile(q)
        });
        pr_tally.assert_none(
            &format!("portable DdSketch alpha={alpha} rebuilt from a delta"),
            &format!(
                "zipf n={N} over [1e3, 1e7], interior q grid {:?}",
                &DDS_QS[1..6]
            ),
        );
    }
}

// ---------------------------------------------------------------- UnivMonQ

/// Exact aggregates are exact: `count`, `min` and `max` are maintained
/// directly, not estimated, so no band applies to them at all.
#[test]
fn univmonq_count_min_and_max_are_exact() {
    let mut q = UnivMonQ::new(Default::default()).expect("default config valid");
    let values: Vec<f64> = uniform_u64(20_000, 900, 3008)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    for v in &values {
        q.update(v);
    }
    let truth = NumericTruth::new(values.clone());
    assert_eq!(q.count() as usize, values.len(), "count must be exact");
    assert_eq!(q.min(), Some(truth.min()), "min must be exact");
    assert_eq!(q.max(), Some(truth.max()), "max must be exact");
}

/// UnivMon-Q's bottom layer sees every update, so a point frequency query is
/// a Count Sketch query at `(depth, width)` and carries Count Sketch's L2
/// bound — not a percentage. Its `F2` estimate is the AMS row-sum estimator
/// over the same matrix and carries `sqrt(2*kappa/w)`.
#[test]
fn univmonq_frequency_and_f2_satisfy_the_count_sketch_bounds() {
    use common::FreqTruth;
    use common::specs::{CountSketchSpec, SecondMomentSpec};

    let config = UnivMonQConfig::default();
    let mut q = UnivMonQ::new(config).expect("default config valid");
    let mut truth = FreqTruth::default();

    // Well-separated heavy hitters plus a uniform background.
    for i in 0..40u64 {
        for _ in 0..(i + 1) * 60 {
            q.update(&(i as f64));
            truth.observe(i as i64);
        }
    }
    for k in uniform_u64(20_000, 900, 3008) {
        let key = 100u64 + k % 800;
        q.update(&(key as f64));
        truth.observe(key as i64);
    }

    let context = format!(
        "UnivMonQConfig {{ levels: {}, width: {}, depth: {}, hash_seed: {} }}, \
         40 separated heavy keys + uniform background (stream seed 3008)",
        config.levels, config.width, config.depth, config.hash_seed
    );
    CountSketchSpec::new(config.depth, config.width).assert_contract(
        "UnivMonQ::estimate_frequency (bottom layer Count Sketch)",
        &truth,
        |k| q.estimate_frequency(k as f64) as f64,
        &context,
    );

    let f2_spec = SecondMomentSpec::new(config.depth, config.width);
    if let Err(detail) = f2_spec.check(q.estimate_f2(), truth.f2()) {
        panic!("UnivMonQ::estimate_f2: {detail}\n  context: {context}");
    }
}

/// Ordered queries against the full documented bound: the Kolmogorov distance
/// of the estimated CDF, and the rank acceptance of `quantile` answers.
///
/// The contract in `docs/api/api_univmon_q.md` is
///
/// ```text
///   sup_x |F_hat(x) - F(x)| <= eta = 2 E_H + P_hat_R * epsilon_R
///   E_H       = sum_h |f_hat_h - f_h| / N        (over the recovered heavy set)
///   epsilon_R = sqrt(log(2 / delta) / (2 m_R))   (0 when m_R = 0 and P_hat_R = 0)
/// ```
///
/// `m_R` is the number of retained occurrence samples backing the residual —
/// `OrderedQueryDiagnostics::residual_samples` — and nothing else. The CDF's
/// breakpoint count is not a sample count: it also carries the heavy values and
/// the retained minimum and maximum, so substituting it inflates `m_R` and
/// quietly shrinks `epsilon_R`. `m_R = 0` with `P_hat_R > 0` is an invalid
/// estimator state — mass attributed to a residual with no sample behind it —
/// and fails outright rather than being given a fabricated `epsilon_R`.
///
/// # Two assertions, one sketch
///
/// The supremum and the rank acceptance are different statements, and each
/// keeps its own tally and its own failure message. They are scored on the
/// *same* seeded sketch: rebuilding an identical sketch to ask it a second
/// question doubles the work and adds no independence.
///
/// - **CDF supremum.** The left-hand side is a supremum over all `x`, so one
///   sketch answers it with one pass/fail. `common::specs::cdf_sup_distance`
///   sweeps the ordered union of the exact truth support and the estimated
///   breakpoints, which is exact for step functions because both are constant
///   between consecutive jump points.
/// - **Quantile rank acceptance.** `quantile(q)` must return a value whose
///   true rank interval meets `[q - eta, q + eta]`. This follows from the same
///   `eta`: the answer `v` satisfies `F_hat(v) >= q >= F_hat(v-)`, and
///   `|F_hat - F| <= eta` everywhere, so `F(v) >= q - eta` and
///   `F(v-) <= q + eta`.
///
/// # Trial unit
///
/// One trial is one sketch. Independent trials come from the two places
/// UnivMon-Q's randomness lives — the CountSketch hash (`config.hash_seed`)
/// and the occurrence-priority stream (`source_id`) — and both move per trial.
#[test]
fn univmonq_ordered_queries_satisfy_the_documented_cdf_and_rank_bounds() {
    use common::specs::{cdf_sup_distance, occurrence_sample_epsilon, rank_violation};

    const DELTA: f64 = 0.01;
    const TRIALS: usize = 12;
    const QS: [f64; 7] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

    let mut gate_open_seen = 0usize;
    let mut gate_closed_seen = 0usize;

    for (regime, values) in univmonq_ordered_regimes() {
        let truth = NumericTruth::new(values.clone());
        let mut exact_counts: HashMap<u64, u64> = HashMap::new();
        for v in &values {
            *exact_counts.entry(v.to_bits()).or_default() += 1;
        }

        let mut cdf_tally = Tally::default();
        let mut rank_tally = Tally::default();
        let mut gate_fired = 0usize;
        let mut heavy_seen = 0usize;
        let mut last_context = String::new();

        for t in 0..TRIALS {
            let config = UnivMonQConfig {
                hash_seed: 3 + t,
                ..UnivMonQConfig::default()
            };
            let mut q: UnivMonQ =
                UnivMonQ::with_hasher_and_source_id(config, 0x0DDE_1000 + t as u64)
                    .expect("config valid");
            for v in &values {
                q.update(v);
            }

            let n = q.count() as f64;
            let view = q.prepare_queries();
            let diag = view.ordered_query_diagnostics();
            let estimated: Vec<(f64, f64)> = view.cdf().iter().map(|p| (p.value, p.rank)).collect();

            let gate = view.estimate_f2() / (n * n);
            let threshold = 1.0 / config.ordered_samples as f64;
            if gate >= threshold {
                gate_fired += 1;
            } else {
                assert!(
                    diag.heavy.is_empty(),
                    "{regime} trial {t}: the adaptive gate is closed \
                     (F2_hat/N^2 = {gate:.3e} < 1/ordered_samples = {threshold:.3e}) but \
                     the CDF still used {} heavy values",
                    diag.heavy.len()
                );
            }
            if !diag.heavy.is_empty() {
                heavy_seen += 1;
            }

            // E_H, as the document defines it: the **sum** of absolute
            // frequency errors over the recovered heavy set, normalized by N.
            let e_h: f64 = diag
                .heavy
                .iter()
                .map(|(value, frequency)| {
                    let exact = *exact_counts.get(&value.to_bits()).unwrap_or(&0) as f64;
                    (frequency - exact).abs()
                })
                .sum::<f64>()
                / n;
            let p_hat_r = diag.residual_mass_fraction(q.count());
            let m_r = diag.residual_samples;
            assert!(
                m_r > 0 || p_hat_r == 0.0,
                "{regime} trial {t}: P_hat_R = {p_hat_r:.6} of the mass is attributed to \
                 a residual backed by m_R = 0 occurrence samples. There is no valid \
                 epsilon_R for that state and inventing one would make the bound \
                 unfalsifiable"
            );
            let eps_r = if m_r == 0 {
                0.0
            } else {
                occurrence_sample_epsilon(m_r, DELTA)
            };
            let eta = 2.0 * e_h + p_hat_r * eps_r;

            let context = format!(
                "{regime} trial {t}: hash_seed={} source_id={:#x}, n={n}, gate \
                 F2_hat/N^2={gate:.3e} vs 1/ordered_samples={threshold:.3e}, heavy set \
                 {} values, E_H={e_h:.6} (sum form), P_hat_R={p_hat_r:.6}, m_R={m_r}, \
                 epsilon_R={eps_r:.6} at delta={DELTA} -> eta {eta:.6}; \
                 {} CDF breakpoints, {} distinct truth values",
                config.hash_seed,
                0x0DDE_1000 + t as u64,
                diag.heavy.len(),
                estimated.len(),
                exact_counts.len(),
            );
            last_context = context.clone();

            assert!(
                m_r > 0 || !diag.heavy.is_empty(),
                "{context}: the CDF was built from neither heavy values nor occurrence \
                 samples, so there is nothing for the bound to cover"
            );

            // --- Assertion 1: the Kolmogorov distance.
            let (sup, at) = cdf_sup_distance(&estimated, truth.sorted());
            cdf_tally.record(sup <= eta, || {
                format!(
                    "{context}\n      sup_x |F_hat - F| = {sup:.6} > eta, attained at \
                     x = {at}"
                )
            });

            // --- Assertion 2: quantile answers under the same eta.
            let mut worst: Option<String> = None;
            for &qq in &QS {
                let est = view.quantile(qq).expect("ordered_samples enabled");
                if let Some(detail) = rank_violation(truth.sorted(), qq, est, eta) {
                    worst.get_or_insert(detail);
                }
            }
            rank_tally.record(worst.is_none(), || {
                format!("{context}\n      {}", worst.clone().unwrap())
            });

            // `rank` and `quantile` must read the same CDF. The comparison is
            // against `rank_incl(v)` — the exact share of observations at or
            // below `v`, which is what `rank` estimates — and not against `q`:
            // on a tie-dense stream one value legitimately spans a wide band.
            //
            // |rank(v)/n - F(v)| <= |rank(v)/n - F_hat(v)| + |F_hat(v) - F(v)|.
            // The second term is eta. The first is zero when `rank` and
            // `quantile` resolve to the same breakpoint and is at most one
            // further CDF-against-CDF discrepancy when they do not, so the sum
            // is bounded by 2*eta.
            for &qq in &[0.1f64, 0.5, 0.9] {
                let v = view.quantile(qq).expect("quantile");
                let r = view.rank(v).expect("rank") as f64 / n;
                let (_, incl) = truth.rank_interval(v);
                assert!(
                    (r - incl).abs() <= 2.0 * eta + 1e-9,
                    "rank(quantile({qq})) = {r:.6} but {v} truly occupies ranks up to \
                     {incl:.6}, off by {:.6} > 2*eta = {:.6}. {context}",
                    (r - incl).abs(),
                    2.0 * eta
                );
            }

            // Monotonicity is structural: a CDF that decreases is malformed
            // whatever the sampling error.
            assert!(!estimated.is_empty(), "{context}: cdf must not be empty");
            for w in estimated.windows(2) {
                assert!(
                    w[0].1 <= w[1].1 + 1e-9,
                    "cdf ranks not monotone at {:?} -> {:?}. {context}",
                    w[0],
                    w[1]
                );
            }
        }

        // Each regime must actually be the regime it claims to be.
        match regime {
            "diffuse" => assert_eq!(
                gate_fired, 0,
                "the diffuse stream fired the adaptive gate on {gate_fired} of {TRIALS} \
                 trials; the regime premise is broken. {last_context}"
            ),
            "heavy" => {
                assert_eq!(
                    gate_fired, TRIALS,
                    "the heavy stream must fire the gate on every trial, or the open-gate \
                     path is not being exercised. {last_context}"
                );
                assert!(
                    heavy_seen > 0,
                    "the heavy stream never produced a non-empty heavy set, so E_H was \
                     zero throughout and the full bound was never exercised. \
                     {last_context}"
                );
            }
            _ => assert!(
                gate_fired > 0,
                "the mixed stream never fired the gate. {last_context}"
            ),
        }
        gate_open_seen += gate_fired;
        gate_closed_seen += TRIALS - gate_fired;

        let trial_unit = format!(
            "one trial = one sketch with its own hash seed and source id; {TRIALS} \
             trials. Last: {last_context}"
        );
        cdf_tally.assert_independent_binomial(
            &format!(
                "UnivMonQ ordered queries ({regime}) / sup_x |F_hat - F| <= 2 E_H + P_hat_R eps_R"
            ),
            DELTA,
            &format!(
                "scored on the exact Kolmogorov distance over the union of the truth \
                      support and the estimated breakpoints. {trial_unit}"
            ),
        );
        rank_tally.assert_independent_binomial(
            &format!("UnivMonQ::quantile ({regime}) / rank acceptance within eta"),
            DELTA,
            &format!(
                "q grid {QS:?}, each answer's true rank interval must meet \
                      [q - eta, q + eta]. {trial_unit}"
            ),
        );
    }

    // Both branches of the adaptive gate must have run somewhere in this test,
    // or half the code path is untested whatever the individual regimes did.
    assert!(
        gate_open_seen > 0 && gate_closed_seen > 0,
        "the adaptive gate must be exercised both open ({gate_open_seen} trials) and \
         closed ({gate_closed_seen} trials)"
    );
}

/// The three streams the ordered-query bound is checked on.
///
/// - **diffuse**: 200k observations over 200k distinct values. `F2/N^2` is tiny,
///   the gate cannot fire, the heavy set is empty and the bound collapses to
///   `epsilon_R`.
/// - **heavy**: a sharp Zipf head. The gate fires on every trial, so `E_H > 0`
///   and `P_hat_R < 1` and all three terms are live.
/// - **mixed**: a heavy head over a broad diffuse tail, so the residual carries
///   most of the mass while the heavy set is still non-empty.
fn univmonq_ordered_regimes() -> Vec<(&'static str, Vec<f64>)> {
    let diffuse: Vec<f64> = uniform_u64(200_000, 10_000_000, 0x0DDE_0001)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let heavy = zipf_f64(200_000, 4_096, 1.4, 1.0, 1e6, 0x0DDE_0002);
    let mut mixed = zipf_f64(60_000, 64, 1.6, 1.0, 1e3, 0x0DDE_0003);
    mixed.extend(
        uniform_u64(140_000, 5_000_000, 0x0DDE_0004)
            .into_iter()
            .map(|v| 1e4 + v as f64),
    );
    vec![("diffuse", diffuse), ("heavy", heavy), ("mixed", mixed)]
}

/// The exact CDF sweep must catch an error a breakpoint scan cannot see.
///
/// This is a hand-built fixture, not a sketch: it isolates the *measurement*,
/// which is what the previous revision got wrong.
///
/// The estimate reports two breakpoints and both are exactly right at their own
/// values, so scoring the estimate one breakpoint at a time gives **zero**
/// error. But it has no breakpoint anywhere across `[2, 8]`, where the truth
/// puts 60% of its mass — so its CDF stays flat at 0.2 while the true one
/// climbs to 0.8, and the real Kolmogorov distance is 0.6.
#[test]
fn cdf_sup_distance_detects_a_gap_a_breakpoint_scan_misses() {
    use common::specs::{breakpoint_rank_interval_distance, cdf_sup_distance};

    // Truth: 10 observations. Two at value 1, six at value 5, two at value 9.
    let sorted_truth: Vec<f64> = vec![1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0, 9.0];

    // The estimate keeps only the extremes: F_hat(1) = 0.2, F_hat(9) = 1.0.
    // Both are exactly the true CDF at those two values.
    let estimated: Vec<(f64, f64)> = vec![(1.0, 0.2), (9.0, 1.0)];

    // `breakpoint_rank_interval_distance` scores each breakpoint's reported
    // rank against its own value's true rank interval. Value 1 occupies ranks
    // [0.0, 0.2] and is reported at 0.2; value 9 occupies [0.8, 1.0] and is
    // reported at 1.0. Both are inside, so it sees a perfect estimate.
    let breakpoint_only = breakpoint_rank_interval_distance(&estimated, &sorted_truth);
    assert_eq!(
        breakpoint_only, 0.0,
        "the fixture must be one the breakpoint scan reports as perfect, or it does \
         not demonstrate anything"
    );

    // The real distance is attained at x = 5, where F = 0.8 and F_hat = 0.2.
    let (sup, at) = cdf_sup_distance(&estimated, &sorted_truth);
    assert_eq!(at, 5.0, "the supremum must be attained at the missing atom");
    assert!(
        (sup - 0.6).abs() < 1e-12,
        "sup_x |F_hat - F| must be 0.6 here, got {sup}"
    );

    // Sanity in the other direction: an estimate that reproduces the truth
    // exactly has zero distance under the sweep too.
    let exact: Vec<(f64, f64)> = vec![(1.0, 0.2), (5.0, 0.8), (9.0, 1.0)];
    let (zero, _) = cdf_sup_distance(&exact, &sorted_truth);
    assert_eq!(zero, 0.0, "an exact step CDF must have zero distance");

    // And the sweep must look strictly below the first breakpoint as well as
    // between them: an estimate that starts too high is caught at the truth's
    // own smallest value.
    let too_high: Vec<(f64, f64)> = vec![(1.0, 0.9), (5.0, 0.9), (9.0, 1.0)];
    let (high, at_high) = cdf_sup_distance(&too_high, &sorted_truth);
    assert_eq!(at_high, 1.0);
    assert!((high - 0.7).abs() < 1e-12, "expected 0.7, got {high}");
}

/// Distinct count, entropy and heavy-hitter recall for UnivMon-Q's recursive
/// sketch hierarchy.
///
/// Documented empirical regression, not a theorem. UnivMon's `g`-sum
/// recurrence composes per-layer Count Sketch errors across `levels`
/// geometrically sampled substreams; the resulting constant depends on the
/// stream's own layer occupancy and the crate publishes no closed form for it,
/// so there is no theoretical band to assert.
///
/// Band source: measured on this exact stream (40 separated heavy keys with
/// weights 60..2400, plus 20k uniform background over 800 keys, stream seed
/// 3008, default `UnivMonQConfig`). Observed relative errors are distinct
/// 1.6%, entropy 0.3%; the bands below are 10% for both, roughly six times the
/// observed movement, and heavy-hitter recall is 10/10 against a target of 8.
#[test]
fn univmonq_distinct_entropy_and_recall_stay_within_the_documented_empirical_band() {
    use common::FreqTruth;

    let mut q = UnivMonQ::new(Default::default()).expect("default config valid");
    let mut truth = FreqTruth::default();
    for i in 0..40u64 {
        for _ in 0..(i + 1) * 60 {
            q.update(&(i as f64));
            truth.observe(i as i64);
        }
    }
    for k in uniform_u64(20_000, 900, 3008) {
        let key = 100u64 + k % 800;
        q.update(&(key as f64));
        truth.observe(key as i64);
    }

    let distinct = truth.distinct() as f64;
    assert_between(
        q.estimate_distinct(),
        distinct * 0.90,
        distinct * 1.10,
        "UnivMonQ distinct (empirical band, stream seed 3008)",
    );
    let entropy = truth.entropy(false);
    assert_between(
        q.estimate_entropy(),
        entropy * 0.90,
        entropy * 1.10,
        "UnivMonQ entropy in nats (empirical band, stream seed 3008)",
    );

    // The ten heaviest separated keys carry weights 1860..2400, each at least
    // 3x the heaviest background key, so a working heavy-hitter path finds
    // essentially all of them.
    let top_true: std::collections::HashSet<u64> = (30..40u64).collect();
    let hits = q
        .heavy_hitters(10)
        .iter()
        .filter(|(v, _)| top_true.contains(&(*v as u64)))
        .count();
    assert!(
        hits >= 8,
        "UnivMonQ recovered only {hits}/10 known heavy keys (empirical band, stream seed 3008)"
    );
}

// -------------------------------------------------------- TumblingWindow<KLL>

/// Window bookkeeping is exact (which observations land in which window) while
/// the answers inside a window carry KLL's rank error. Both halves are checked
/// against their own standard: window membership by equality, quantiles by
/// `eps(k)`.
#[test]
fn tumbling_kll_windows_are_exact_and_answers_satisfy_the_rank_contract() {
    const K: i32 = 200;
    const SKETCH_SEED: u64 = 0x7717_0001;
    const N: usize = 4_000;
    const WINDOW: u64 = 400;

    let cfg = KLLConfig {
        k: K as usize,
        m: 8,
        seed: Some(SKETCH_SEED),
    };
    let mut tw: TumblingWindow<KLL> = TumblingWindow::new(WINDOW, 16, cfg, 4);
    let all: Vec<f64> = uniform_u64(N, 1_000_000, 3009)
        .iter()
        .map(|v| *v as f64)
        .collect();
    for (t, v) in all.iter().enumerate() {
        tw.insert(t as u64, &DataInput::F64(*v), 0);
    }

    // Window boundaries are arithmetic, so this is an equality, not a band.
    assert_eq!(
        tw.closed_count(),
        (N as u64 / WINDOW - 1) as usize,
        "windows [0, {}) should be closed at t={}",
        N as u64 - WINDOW,
        N - 1
    );

    // Sixteen independent compaction seeds, one trial each. Within a seed the
    // three window views (`query_all`, `query_recent`, the active sketch) are
    // built from the same compactors and are strongly dependent, so they are
    // reduced to a single worst-case rank error before the trial is scored.
    const TRIAL_SEEDS: usize = 16;
    let spec = KllRankSpec::datasketches(K as usize);
    let context = format!(
        "k={K} stream_seed=3009 n={N} window={WINDOW}, {TRIAL_SEEDS} independent \
         compaction seeds from kll_trial_seed(0x7717_0000..)"
    );
    let mut tally = Tally::default();
    for t in 0..TRIAL_SEEDS {
        let seed = kll_trial_seed(0x7717_0000 + t as u64);
        let cfg = KLLConfig {
            k: K as usize,
            m: 8,
            seed: Some(seed),
        };
        let mut w: TumblingWindow<KLL> = TumblingWindow::new(WINDOW, 16, cfg, 4);
        for (t, v) in all.iter().enumerate() {
            w.insert(t as u64, &DataInput::F64(*v), 0);
        }
        let views: [(&str, KLL<f64>, &[f64]); 3] = [
            ("query_all", w.query_all(), &all[..]),
            // query_recent(1) = the active window plus the last closed one.
            (
                "query_recent(1)",
                w.query_recent(1),
                &all[N - 2 * WINDOW as usize..],
            ),
            (
                "active_sketch",
                w.active_sketch().clone(),
                &all[N - WINDOW as usize..],
            ),
        ];
        let mut worst = 0.0f64;
        let mut detail = String::new();
        for (label, sketch, slice) in views {
            let truth = NumericTruth::new(slice.to_vec());
            let (e, d) = spec.max_rank_error(truth.sorted(), &[0.1, 0.25, 0.5, 0.75, 0.9], |q| {
                sketch.quantile(q)
            });
            if e >= worst {
                worst = e;
                detail = format!("{label}: {d}");
            }
        }
        let eps = spec.epsilon();
        tally.record(worst <= eps, || {
            format!("seed={seed:#x}: max rank error {worst:.6} > eps(k={K}) = {eps:.6}; {detail}")
        });
    }
    tally.assert_independent_binomial(
        "TumblingWindow<KLL> window queries / maximum rank error per compaction seed",
        spec.trial_failure_probability,
        &context,
    );

    // Rotation must not drop windows. `KLL::count()` is the *retained weighted
    // mass*, not an exact counter — compaction promotes half a buffer at
    // double weight, so the total is an unbiased randomized estimate of `n`
    // and single-pass sketches already report `n +- a few` before any merge.
    // The crate documents no exactness guarantee for it, so this is a
    // consistency check rather than a theorem: the estimate must stay inside
    // the same `eps(k)` band the rank contract works in, which a pipeline that
    // silently discarded a whole window (12.5% of the stream here) could not.
    let merged_count = tw.query_all().count() as f64;
    let drift = (merged_count - N as f64).abs() / N as f64;
    assert!(
        drift <= spec.epsilon(),
        "TumblingWindow<KLL>::query_all reported a weighted mass of {merged_count} for {N} \
         inserts (drift {drift:.5} > eps(k) {:.5}); a dropped or duplicated window would \
         show up here. {context}",
        spec.epsilon()
    );
}

// --------------------------------------------- Portable HydraKll per-key

/// Hydra routes each key to its own KLL cell, so each cell answers under the
/// same rank characterization as a standalone KLL over that key's observations.
///
/// Every cell of a `HydraKllSketch` is cloned from one seeded prototype, so the
/// two keys of a single grid share a compaction seed and are **not** independent
/// trials. Each grid is therefore reduced to its worst rank error across both
/// keys and the whole q grid, and the binomial runs over twelve independent
/// prototype seeds.
#[test]
fn portable_hydra_kll_per_key_medians_satisfy_the_rank_characterization() {
    const K: usize = 200;
    const TRIALS: usize = 12;
    const QS: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];

    let spec = KllRankSpec::datasketches(K);
    let mut truths: Vec<(&str, NumericTruth)> = Vec::new();
    for (name, base, seed) in [("svc-a", 100.0f64, 3010u64), ("svc-b", 900.0, 3011)] {
        let vals: Vec<f64> = normal_f64(4000, base, base * 0.05, seed)
            .into_iter()
            .map(f64::abs) // HydraKll cells are KLL: positive domain
            .collect();
        truths.push((name, NumericTruth::new(vals)));
    }

    let mut tally = Tally::default();
    for t in 0..TRIALS {
        let seed = kll_trial_seed(0x5EED_0500 + t as u64);
        let mut hk = HydraKllSketch::with_seed(3, 256, K as u16, seed);
        for (name, truth) in &truths {
            for v in truth.sorted() {
                hk.update(name, *v);
            }
        }
        let mut worst = 0.0f64;
        let mut detail = String::new();
        for (name, truth) in &truths {
            let (e, d) = spec.max_rank_error(truth.sorted(), &QS, |q| hk.quantile(name, q));
            if e >= worst {
                worst = e;
                detail = format!("key {name}: {d}");
            }
        }
        let eps = spec.epsilon();
        tally.record(worst <= eps, || {
            format!("seed={seed:#x}: max rank error {worst:.6} > eps(k={K}) = {eps:.6}; {detail}")
        });
    }
    tally.assert_independent_binomial(
        "portable HydraKll per-key quantiles / maximum rank error per prototype seed",
        spec.trial_failure_probability,
        &format!(
            "k={K}, 3x256 grid, normal(100, 5) and normal(900, 45), stream seeds 3010/3011, \
             {TRIALS} independent prototype seeds from kll_trial_seed(0x5EED_0500..), \
             q grid {QS:?}"
        ),
    );
}

// ------------------------------------------- DDSketch input rejection

/// Structural guards, not accuracy: values the mapping cannot index and
/// deltas spanning an implausible bucket range must be rejected without
/// corrupting state. No error bound applies to any of it.
#[test]
fn ddsketch_rejects_untrackable_values_and_mapping_mismatches() {
    let alpha = 0.01;
    let mut core = DDSketch::new(alpha);
    let mut port = PortableDds::new(alpha);

    // Non-finite / non-positive / beyond-indexable-range values must be
    // dropped by BOTH implementations: silently, without corrupting bucket 0
    // (NaN floor-casts to 0) and without letting one sample force a distant-
    // bucket allocation (unguarded, f64::MAX maps ~35k buckets away at
    // alpha=0.01 — ~277 KiB of amplification per sample, scaling with 1/lnγ).
    for v in [
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        -5.0,
        0.0,
        f64::MIN_POSITIVE, // below min-indexable
        5e-324,            // smallest subnormal
        f64::MAX,          // above max-indexable
        1e308,
    ] {
        core.add(&v);
        port.update(v);
    }
    assert_eq!(core.get_count(), 0, "core must drop untrackable extremes");
    assert_eq!(
        port.total_count(),
        0,
        "portable must drop untrackable extremes"
    );
    assert!(
        port.store_counts.len() < 10_000,
        "portable store grew to {} buckets from untrackable input",
        port.store_counts.len()
    );
    assert_eq!(port.quantile(0.5), None, "nothing trackable was added");

    // Normal samples still work after the rejected inputs.
    let good = [1.0f64, 2.0, 4.0, 8.0, 16.0];
    for v in good {
        core.add(&v);
        port.update(v);
    }
    assert_eq!(core.get_count(), 5);
    assert_eq!(port.total_count(), 5);
    assert_between(
        port.store_counts.len() as f64,
        1.0,
        2.0 * 256.0, // initial GROW_CHUNK seed + at most one more
        "store stays compact",
    );

    // Boundary acceptance: values just INSIDE the indexable range must still
    // be tracked, so the guards reject only genuinely unmappable extremes.
    // Bounds come from the shared production helper — core and portable MUST
    // agree because they compute from the same function.
    let (min_idx, max_idx) = asap_sketchlib::sketches::ddsketch::ddsketch_indexable_bounds(alpha);

    let mut port_boundary = PortableDds::new(alpha);
    let mut core_boundary = DDSketch::new(alpha);
    let just_inside_min = min_idx * (1.0 + 1e-9); // a hair above the floor
    let just_inside_max = max_idx * (1.0 - 1e-9); // a hair below the ceiling
    port_boundary.update(just_inside_min);
    port_boundary.update(just_inside_max);
    core_boundary.add(&just_inside_min);
    core_boundary.add(&just_inside_max);
    assert_eq!(
        port_boundary.total_count(),
        2,
        "in-range extremes near boundaries must be kept"
    );
    assert_eq!(core_boundary.get_count(), 2, "core boundary agreement");
    assert_eq!(port_boundary.total_count(), core_boundary.get_count());

    // And one step past each boundary must be rejected by both.
    port_boundary.update(min_idx * 0.5);
    port_boundary.update(max_idx * (1.0 + 1e-6));
    core_boundary.add(&(min_idx * 0.5));
    assert_eq!(
        port_boundary.total_count(),
        2,
        "out-of-range neighbors rejected"
    );
    assert_eq!(
        core_boundary.get_count(),
        2,
        "core rejects out-of-range neighbor"
    );

    // Item 2 contract: mismatched mappings are a runtime error in BOTH types,
    // not a debug-only assertion.
    let other_alpha = 0.05;
    let mut core_other = DDSketch::new(other_alpha);
    core_other.add(&1.0);
    assert!(
        core.merge(&core_other).is_err(),
        "core merge must reject alpha mismatch"
    );
    let mut port_other = PortableDds::new(other_alpha);
    port_other.update(1.0);
    assert!(
        port.merge(&port_other).is_err(),
        "portable merge must reject alpha mismatch"
    );

    // Tiny-alpha regression: at alpha=1e-9 ln(gamma) ~ 2e-9, and naive
    // reciprocal-multiplied guard formulas diverge from core's — admitting
    // v=1e-300 whose bucket index saturates i32 and overflows ensure_bucket.
    // Both implementations must drop it identically via the shared helper.
    let mut tiny = PortableDds::new(1e-9);
    let mut tiny_core = DDSketch::new(1e-9);
    tiny.update(1e-300);
    tiny_core.add(&1e-300);
    assert_eq!(
        tiny.total_count(),
        0,
        "portable drops sub-indexable value at tiny alpha"
    );
    assert_eq!(
        tiny_core.get_count(),
        0,
        "core drops sub-indexable value at tiny alpha"
    );
    assert_eq!(
        tiny.store_counts.len(),
        0,
        "no allocation may occur for rejected values"
    );
}

#[test]
fn portable_ddsketch_rejects_hostile_delta_spans() {
    use asap_sketchlib::message_pack_format::portable::ddsketch::DdSketchDelta;

    let alpha = 0.01;
    let mut base = PortableDds::new(alpha);
    base.update(1.0);
    let (len_before, offset_before) = (base.store_counts.len(), base.store_offset);

    // A corrupt/hostile delta pointing near i32::MAX must be rejected with an
    // error BEFORE any allocation: the naive pad would be ~2e9 buckets.
    let hostile = DdSketchDelta {
        buckets: vec![(i32::MAX - 1, 1), (7, 3)],
        ..DdSketchDelta::default()
    };
    assert!(
        base.apply_delta(&hostile).is_err(),
        "hostile far-span delta must be rejected"
    );
    assert_eq!(
        base.store_counts.len(),
        len_before,
        "state untouched on rejection"
    );
    assert_eq!(
        base.store_offset, offset_before,
        "offset untouched on rejection"
    );

    // Benign deltas still apply: bucket index 6 carries count 5 afterward,
    // and its representative gamma^6*(1+alpha) becomes visible in queries.
    let benign = DdSketchDelta {
        buckets: vec![(6, 5)],
        ..DdSketchDelta::default()
    };
    base.apply_delta(&benign).expect("benign delta");
    let gamma = (1.0 + alpha) / (1.0 - alpha);
    assert_eq!(
        base.quantile(0.9),
        Some(gamma.powf(6.0) * (1.0 + alpha)),
        "applied delta count visible through queries"
    );
}

const KLL_M_AXIS: [usize; 5] = [2, 4, 8, 16, 64];
const KLL_M_AT_OR_ABOVE_DEFAULT: [usize; 3] = [8, 16, 64];
const KLL_M_BELOW_DEFAULT: [usize; 2] = [2, 4];
const KLL_SMALL_K: [i32; 3] = [8, 16, 32];
const KLL_EXACT_REGIME_N: [usize; 5] = [1, 2, 7, 50, 150];
const KLL_WIDENED_EPSILON_FACTOR: f64 = 3.0;

fn kll_axis_seed(case: u64) -> u64 {
    kll_trial_seed(0x9C2A_0000u64.wrapping_add(case))
}

fn kll_axis_values(n: usize, seed: u64) -> Vec<f64> {
    uniform_u64(n, 1_000_000, seed)
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

#[test]
fn kll_minimum_compactor_capacity_axis_keeps_the_count_exact_and_reports_its_own_geometry() {
    let values = kll_axis_values(20_000, 0x4B11_0001);
    let truth = NumericTruth::new(values.clone());
    for &m in &KLL_M_AXIS {
        let seed = kll_axis_seed(m as u64);
        let mut sketch: KLL<f64> = KLL::init_with_seed(200, m, seed);
        for v in &values {
            sketch.update(v);
        }
        let ctx = format!("m={m} k=200 n={} sketch_seed={seed:#x}", values.len());
        assert_eq!(sketch.k(), 200, "{ctx}: k moved");
        assert_eq!(sketch.wire_m() as usize, m, "{ctx}: reported m");
        assert_eq!(sketch.wire_k() as usize, 200, "{ctx}: reported k");
        assert_between(
            sketch.count() as f64,
            values.len() as f64 * 0.95,
            values.len() as f64 * 1.05,
            &format!("{ctx}: weighted retained count"),
        );
        for &q in &RANK_QS {
            assert_between(
                sketch.quantile(q),
                truth.min(),
                truth.max(),
                &format!("{ctx} q={q}"),
            );
        }
    }
}

#[test]
fn kll_normalizes_a_minimum_compactor_capacity_outside_its_supported_range() {
    let clamped: KLL<f64> = KLL::init_with_seed(200, 0, kll_axis_seed(0x0100));
    assert_eq!(clamped.wire_m(), 2, "m=0 must normalize to 2");
    assert_eq!(clamped.wire_k(), 200, "m=0 must leave k alone");

    let raised: KLL<f64> = KLL::init_with_seed(4, 64, kll_axis_seed(0x0101));
    assert_eq!(raised.wire_m(), 64, "m=64 must survive");
    assert_eq!(raised.wire_k(), 64, "k below m must be raised to m");

    let mut usable: KLL<f64> = KLL::init_with_seed(4, 64, kll_axis_seed(0x0102));
    let values = kll_axis_values(5_000, 0x4B11_0102);
    for v in &values {
        usable.update(v);
    }
    assert_between(
        usable.count() as f64,
        values.len() as f64 * 0.85,
        values.len() as f64 * 1.15,
        "a normalized geometry must still account for every value it ingested",
    );
}

#[test]
fn kll_minimum_compactor_capacity_at_or_above_the_default_satisfies_the_rank_characterization() {
    const N: usize = 30_000;
    const K: usize = 200;
    let spec = KllRankSpec::datasketches(K);
    let mut tally = Tally::default();
    for (case, &m) in KLL_M_AT_OR_ABOVE_DEFAULT.iter().enumerate() {
        for (i, (shape, values)) in rank_streams(case, N).into_iter().enumerate() {
            let seed = kll_axis_seed(0x0200 + (case * 16 + i) as u64);
            let mut sketch: KLL<f64> = KLL::init_with_seed(K, m, seed);
            for v in &values {
                sketch.update(v);
            }
            let truth = NumericTruth::new(values);
            spec.record_trial(
                &mut tally,
                &format!("m={m} {shape} sketch_seed={seed:#x}"),
                truth.sorted(),
                &RANK_QS,
                |q| sketch.quantile(q),
            );
        }
    }
    let context = format!(
        "KLL k={K}, m in {KLL_M_AT_OR_ABOVE_DEFAULT:?}, eps={:.6}, n={N}, \
         one trial = one (m, stream) sketch with its own compaction seed",
        spec.epsilon()
    );
    tally.assert_independent_binomial(
        "KLL minimum-compactor-capacity axis / max rank error",
        spec.trial_failure_probability,
        &context,
    );
}

#[test]
fn kll_minimum_compactor_capacity_below_the_default_stays_within_a_widened_rank_band() {
    const N: usize = 30_000;
    const K: usize = 200;
    let spec = KllRankSpec::datasketches(K);
    let widened = KLL_WIDENED_EPSILON_FACTOR * spec.epsilon();
    let mut tally = Tally::default();
    for (case, &m) in KLL_M_BELOW_DEFAULT.iter().enumerate() {
        for (i, (shape, values)) in rank_streams(case + 8, N).into_iter().enumerate() {
            let seed = kll_axis_seed(0x0300 + (case * 16 + i) as u64);
            let mut sketch: KLL<f64> = KLL::init_with_seed(K, m, seed);
            for v in &values {
                sketch.update(v);
            }
            let truth = NumericTruth::new(values);
            let (worst, detail) =
                spec.max_rank_error(truth.sorted(), &RANK_QS, |q| sketch.quantile(q));
            tally.record(worst <= widened, || {
                format!(
                    "m={m} {shape} sketch_seed={seed:#x}: max rank error {worst:.6} > \
                     {KLL_WIDENED_EPSILON_FACTOR} * eps(k={K}) = {widened:.6}; worst {detail}"
                )
            });
        }
    }
    let context =
        format!("KLL k={K}, m in {KLL_M_BELOW_DEFAULT:?}, widened band {widened:.6}, n={N}");
    tally.assert_none(
        "KLL sub-default minimum-compactor-capacity / widened rank band",
        &context,
    );
}

#[test]
fn kll_small_k_satisfies_the_rank_characterization_at_its_own_epsilon() {
    const N: usize = 20_000;
    let mut tally = Tally::default();
    let mut described = Vec::new();
    for (case, &k) in KLL_SMALL_K.iter().enumerate() {
        let spec = KllRankSpec::datasketches(k as usize);
        described.push(format!("k={k} eps={:.6}", spec.epsilon()));
        for (i, (shape, values)) in rank_streams(case + 16, N).into_iter().enumerate() {
            let seed = kll_axis_seed(0x0400 + (case * 16 + i) as u64);
            let mut sketch: KLL<f64> = KLL::init_kll_with_seed(k, seed);
            for v in &values {
                sketch.update(v);
            }
            assert_between(
                sketch.count() as f64,
                values.len() as f64 * 0.85,
                values.len() as f64 * 1.15,
                &format!("k={k} {shape}: weighted retained count"),
            );
            let truth = NumericTruth::new(values);
            spec.record_trial(
                &mut tally,
                &format!("k={k} {shape} sketch_seed={seed:#x}"),
                truth.sorted(),
                &RANK_QS,
                |q| sketch.quantile(q),
            );
        }
    }
    let context = format!(
        "small-k KLL, n={N}, one trial = one (k, stream) sketch; {}",
        described.join("; ")
    );
    tally.assert_independent_binomial("small-k KLL / max rank error", 0.01, &context);
}

#[test]
fn a_kll_stream_shorter_than_k_is_answered_exactly() {
    for (case, &n) in KLL_EXACT_REGIME_N.iter().enumerate() {
        let seed = kll_axis_seed(0x0500 + case as u64);
        let values = kll_axis_values(n, 0x4B11_0500 + case as u64);
        let mut sketch: KLL<f64> = KLL::init_kll_with_seed(200, seed);
        for v in &values {
            sketch.update(v);
        }
        let truth = NumericTruth::new(values.clone());
        let ctx = format!("n={n} k=200 sketch_seed={seed:#x}");
        assert_eq!(sketch.count(), n, "{ctx}: retained count");
        for &q in &RANK_QS {
            let est = sketch.quantile(q);
            assert_eq!(
                rank_error(truth.sorted(), q, est),
                0.0,
                "{ctx} q={q}: answer {est} is not an exact order statistic"
            );
            assert!(
                values.iter().any(|v| v.to_bits() == est.to_bits()),
                "{ctx} q={q}: answer {est} was never in the stream"
            );
        }
        assert_eq!(
            sketch.quantile(0.0).to_bits(),
            truth.min().to_bits(),
            "{ctx}: q=0 must be the exact minimum"
        );
        assert_eq!(
            sketch.quantile(1.0).to_bits(),
            truth.max().to_bits(),
            "{ctx}: q=1 must be the exact maximum"
        );
    }
}

const DDS_EXTREME_ALPHAS: [f64; 5] = [1e-5, 1e-4, 0.3, 0.5, 0.9];

#[test]
fn ddsketch_satisfies_the_relative_value_error_contract_at_extreme_accuracy_parameters() {
    const N: usize = 20_000;

    for (i, &alpha) in DDS_EXTREME_ALPHAS.iter().enumerate() {
        let core_spec = RelativeQuantileSpec::core(alpha);
        let port_spec = RelativeQuantileSpec::portable(alpha);
        let mut core_tally = Tally::default();
        let mut port_tally = Tally::default();
        let seed = 0x0DDA_0000u64 + i as u64 * 7919;
        let (min_indexable, max_indexable) =
            asap_sketchlib::sketches::ddsketch::ddsketch_indexable_bounds(alpha);
        for (label, raw) in dds_streams(alpha, N, seed) {
            let values: Vec<f64> = raw
                .into_iter()
                .filter(|v| v.is_finite() && *v >= min_indexable && *v <= max_indexable)
                .collect();
            if values.len() < 100 {
                continue;
            }
            let mut core = DDSketch::new(alpha);
            let mut port = PortableDds::new(alpha);
            for v in &values {
                core.add(v);
                port.update(*v);
            }
            let truth = NumericTruth::new(values.clone());
            assert_eq!(
                core.get_count() as usize,
                truth.len(),
                "{label} alpha={alpha}: core dropped an indexable sample"
            );
            assert_eq!(
                port.total_count() as usize,
                truth.len(),
                "{label} alpha={alpha}: portable dropped an indexable sample"
            );
            core_spec.tally_into(&mut core_tally, truth.sorted(), &DDS_QS, |q| {
                core.get_value_at_quantile(q)
            });
            port_spec.tally_into(&mut port_tally, truth.sorted(), &DDS_QS, |q| {
                port.quantile(q)
            });
        }
        let context = format!(
            "alpha={alpha} n={N} indexable=[{min_indexable:.3e}, {max_indexable:.3e}] \
             q grid {DDS_QS:?}"
        );
        core_tally.assert_none(&format!("core DDSketch alpha={alpha}"), &context);
        port_tally.assert_none(&format!("portable DdSketch alpha={alpha}"), &context);
    }
}

#[test]
fn ddsketch_bucket_width_widens_monotonically_with_the_accuracy_parameter() {
    let values = kll_axis_values(20_000, 0x0DDA_9001)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<f64>>();
    let mut previous: Option<(f64, usize)> = None;
    for &alpha in &DDS_EXTREME_ALPHAS {
        let mut sketch = DDSketch::new(alpha);
        for v in &values {
            sketch.add(v);
        }
        assert_eq!(
            sketch.get_count() as usize,
            values.len(),
            "alpha={alpha}: dropped a trackable sample"
        );
        let buckets = sketch.store_counts().iter().filter(|c| **c > 0).count();
        if let Some((prev_alpha, prev_buckets)) = previous {
            assert!(
                buckets <= prev_buckets,
                "alpha={alpha} used {buckets} buckets, more than alpha={prev_alpha} at \
                 {prev_buckets}, but a looser accuracy parameter cannot need finer buckets"
            );
        }
        previous = Some((alpha, buckets));
    }
}

#[test]
fn univmonq_l1_is_exact_and_the_generic_g_sum_reproduces_the_named_estimators() {
    let mut q = UnivMonQ::new(Default::default()).expect("default config valid");
    let values = kll_axis_values(50_000, 0x0DDE_9001);
    for v in &values {
        q.update(v);
    }
    assert_eq!(
        q.estimate_l1(),
        values.len() as f64,
        "L1 of an insertion-only stream is the observation count"
    );
    assert_eq!(
        q.estimate_l1(),
        q.count() as f64,
        "L1 must agree with the sketch's own count"
    );
    assert_eq!(
        q.estimate_g_sum(|_| 1.0).clamp(0.0, q.count() as f64),
        q.estimate_distinct(),
        "g(f)=1 is the distinct-count estimator"
    );
    assert_eq!(
        q.estimate_g_sum(|f| f * f).max(0.0),
        q.estimate_f2(),
        "g(f)=f^2 is the second-moment estimator"
    );
    assert_eq!(
        q.estimate_g_sum(|_| 0.0),
        0.0,
        "the zero function must sum to zero"
    );
}

#[test]
fn univmonq_universal_entropy_tracks_the_exact_entropy_on_a_diffuse_stream() {
    use common::FreqTruth;

    let mut q = UnivMonQ::new(Default::default()).expect("default config valid");
    let mut truth = FreqTruth::default();
    for v in uniform_u64(200_000, 50_000, 0x0DDE_9101) {
        q.update(&(v as f64));
        truth.observe(v as i64);
    }
    let exact = truth.entropy(false);
    assert_between(
        q.estimate_entropy_universal(),
        exact * 0.80,
        exact * 1.20,
        "UnivMonQ universal entropy in nats (diffuse stream, seed 0x0DDE9101)",
    );
}

#[test]
fn univmonq_occurrence_entropy_is_present_only_when_ordered_samples_are_configured() {
    let mut sampled = UnivMonQ::new(UnivMonQConfig {
        ordered_samples: 1_024,
        ..Default::default()
    })
    .expect("sampled config valid");
    let mut unsampled = UnivMonQ::new(UnivMonQConfig {
        ordered_samples: 0,
        ..Default::default()
    })
    .expect("unsampled config valid");
    for v in zipf_f64(120_000, 4_096, 1.2, 1.0, 1e6, 0x0DDE_9201) {
        sampled.update(&v);
        unsampled.update(&v);
    }
    assert!(
        unsampled.estimate_entropy_occurrence().is_none(),
        "no ordered sample means no occurrence entropy"
    );
    let occurrence = sampled
        .estimate_entropy_occurrence()
        .expect("a configured ordered sample must produce an occurrence entropy");
    assert!(
        occurrence.is_finite() && occurrence >= 0.0,
        "occurrence entropy {occurrence} must be a finite non-negative number of nats"
    );
    assert!(
        sampled.estimate_entropy().is_finite(),
        "the assisted entropy estimate must stay finite"
    );
}

#[test]
fn univmonq_universal_rank_is_exact_outside_the_observed_range_and_banded_inside_it() {
    let values = zipf_f64(120_000, 2_048, 1.2, 1.0, 1e6, 0x0DDE_9301);
    let mut q = UnivMonQ::new(Default::default()).expect("default config valid");
    for v in &values {
        q.update(v);
    }
    let truth = NumericTruth::new(values.clone());
    let n = values.len() as f64;

    assert_eq!(
        q.estimate_rank_universal(truth.min() - 1.0),
        Some(0),
        "a value below the observed minimum has rank 0"
    );
    assert_eq!(
        q.estimate_rank_universal(truth.max()),
        Some(values.len() as u64),
        "the observed maximum carries the whole stream"
    );
    assert_eq!(
        q.estimate_rank_universal(truth.max() + 1.0),
        Some(values.len() as u64),
        "a value above the observed maximum carries the whole stream"
    );

    let empty = UnivMonQ::new(Default::default()).expect("default config valid");
    assert_eq!(
        empty.estimate_rank_universal(1.0),
        None,
        "an empty sketch has no rank to report"
    );

    let mut tally = Tally::default();
    for &probe_q in &[0.25f64, 0.5, 0.75] {
        let probe = truth.quantile(probe_q);
        let exact = truth.sorted().partition_point(|v| *v <= probe) as f64;
        let estimated =
            q.estimate_rank_universal(probe)
                .expect("a populated sketch answers every interior rank") as f64;
        let error = (estimated - exact).abs() / n;
        tally.record(error <= 0.25, || {
            format!(
                "probe q={probe_q} value {probe}: estimated rank {estimated} vs exact {exact}, \
                 normalized error {error:.5} > 0.25"
            )
        });
    }
    tally.assert_none(
        "UnivMonQ universal rank / interior band",
        "zipf(1.2) 120k observations over 2048 values, seed 0x0DDE9301, default config",
    );
}

// ---------------------------------------------------------------------------
// The documented input matrix
// ---------------------------------------------------------------------------

/// `tests/TEST_COVERAGE.md` writes this suite as the twelve numbered inputs
/// crossed with `k = 50, 200, 800` for both KLL implementations and
/// `alpha = 0.1, 0.01, 0.001` for DDSketch, queried on the `P0 .. P100` decile
/// grid, single-pass and through a shard merge.
///
/// The batteries above run on the stream *shapes* a compaction scheme can
/// behave differently on (sorted, tie-dense, adversarially ordered) at
/// `k = 64, 200, 800`; this is the document's grid, at the document's `k = 50`,
/// on the document's streams.
///
/// # Where the document's percentages come from
///
/// The rank errors it quotes — 6.11%, 1.65%, 0.447% — are
/// `eps(k) = 2.446 / k^0.9433` evaluated at the three `k` values, which is
/// exactly what `KllRankSpec::datasketches` computes; the first assertion in
/// the KLL half checks that identity, so the tests and the table cannot drift
/// apart silently. DDSketch's quoted relative errors are `alpha` itself.
///
/// # Acceptance rules
///
/// A KLL trial is one sketch scored on its worst rank error over the whole
/// grid, with a 1% failure probability, so a handful of trials per input has no
/// power as a tail test: these are deterministic regression pins on fixed
/// compaction seeds, and the tail test over independent seeds is
/// `kll_family_stays_within_the_datasketches_maximum_rank_error_characterization`
/// above. DDSketch's guarantee is deterministic — bucket width alone, no hash
/// and no sampling — so its checks tolerate no violations at all.
mod documented_matrix {
    use super::common::inputs::key_input;
    use super::*;

    /// The document's query grid, `P0` through `P100` in steps of ten.
    const DOC_QS: [f64; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    /// The document's `k` values and the rank error it quotes for each.
    const DOC_KS: [(i32, f64); 3] = [(50, 0.0611), (200, 0.0165), (800, 0.00447)];

    /// The document's accuracy parameters and the relative error each promises.
    const DOC_ALPHAS: [f64; 3] = [0.1, 0.01, 0.001];

    fn kll_documented_matrix(id: u8) {
        let input = key_input(id);
        let values = input.values();
        let truth = NumericTruth::new(values.clone());

        for (k, documented_eps) in DOC_KS {
            let spec = KllRankSpec::datasketches(k as usize);
            assert!(
                (spec.epsilon() - documented_eps).abs() <= 5e-5,
                "eps(k={k}) = {:.6} is no longer the {documented_eps} the coverage document \
                 quotes",
                spec.epsilon()
            );

            let mut tally = Tally::default();
            for (i, feed) in [Feed::SinglePass, Feed::ShardMerge].into_iter().enumerate() {
                let seed = kll_trial_seed(0x00D0_0000 + id as u64 * 16 + i as u64);
                let label = format!("k={k} feed={feed:?} seed={seed:#x} {}", input.context());

                let fixed = feed_kll(feed, k, seed, &values);
                spec.record_trial(
                    &mut tally,
                    &format!("KLL {label}"),
                    truth.sorted(),
                    &DOC_QS,
                    |q| fixed.quantile(q),
                );

                let dynamic = feed_kll_dynamic(feed, k, seed, &values)
                    .expect("both feeds are supported by KLLDynamic");
                spec.record_trial(
                    &mut tally,
                    &format!("KLLDynamic {label}"),
                    truth.sorted(),
                    &DOC_QS,
                    |q| dynamic.quantile(q),
                );
            }
            tally.assert_none(
                &format!("KLL family k={k} / maximum rank error over the documented grid"),
                &format!(
                    "{}, q grid {DOC_QS:?}, eps(k)={:.6}",
                    input.context(),
                    spec.epsilon()
                ),
            );
        }
    }

    fn ddsketch_documented_matrix(id: u8) {
        let input = key_input(id);
        // DDSketch is defined on positive values and documents that it drops
        // everything else, so the reference is the positive subsequence — the
        // sketch is scored on the question it actually answers.
        let values: Vec<f64> = input
            .values()
            .into_iter()
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();
        let truth = NumericTruth::new(values.clone());
        let n = values.len();

        for alpha in DOC_ALPHAS {
            let spec = RelativeQuantileSpec::core(alpha);
            let context = format!(
                "{} alpha={alpha} over the {n} positive values, q grid {DOC_QS:?}",
                input.context()
            );

            let mut single = DDSketch::new(alpha);
            for v in &values {
                single.add(v);
            }
            assert_eq!(
                single.get_count() as usize,
                n,
                "DDSketch dropped a positive value. {context}"
            );
            let mut tally = Tally::default();
            spec.tally_into(&mut tally, truth.sorted(), &DOC_QS, |q| {
                single.get_value_at_quantile(q)
            });
            tally.assert_none(&format!("core DDSketch alpha={alpha}"), &context);

            // Four-way shard merge: the same contract.
            let mut shards: Vec<DDSketch> = (0..4).map(|_| DDSketch::new(alpha)).collect();
            for (i, v) in values.iter().enumerate() {
                shards[i % 4].add(v);
            }
            let mut merged = shards.remove(0);
            for s in &shards {
                merged.merge(s).expect("same-alpha merge");
            }
            let mut merge_tally = Tally::default();
            spec.tally_into(&mut merge_tally, truth.sorted(), &DOC_QS, |q| {
                merged.get_value_at_quantile(q)
            });
            merge_tally.assert_none(
                &format!("core DDSketch alpha={alpha} after a 4-way shard merge"),
                &context,
            );
        }
    }

    /// The document's value streams, `(15)` to `(18)`, which carry no key
    /// identity and exist for the quantile sketches alone: two Normal streams,
    /// an Exponential one, and the log-uniform stream that lands on a
    /// DDSketch's own bucket edges for a given alpha.
    ///
    /// `(18)` is defined relative to the sketch under test — its `gamma` comes
    /// from the alpha — so it is rebuilt per alpha rather than being one fixed
    /// stream.
    #[test]
    fn the_documented_value_inputs_satisfy_both_quantile_contracts() {
        for id in [15u8, 16, 17] {
            let (label, values) = common::inputs::value_input(id);
            let positive: Vec<f64> = values
                .iter()
                .copied()
                .filter(|v| v.is_finite() && *v > 0.0)
                .collect();
            let truth = NumericTruth::new(values.clone());
            let positive_truth = NumericTruth::new(positive.clone());

            for (k, _) in DOC_KS {
                let spec = KllRankSpec::datasketches(k as usize);
                let seed = kll_trial_seed(0x00E0_0000 + id as u64 * 16 + k as u64);
                let sketch = feed_kll(Feed::SinglePass, k, seed, &values);
                let mut tally = Tally::default();
                spec.record_trial(
                    &mut tally,
                    &format!("KLL k={k} on ({id}) {label} seed={seed:#x}"),
                    truth.sorted(),
                    &DOC_QS,
                    |q| sketch.quantile(q),
                );
                tally.assert_none(
                    &format!("KLL k={k} / ({id}) maximum rank error"),
                    &format!("({id}) {label} n={}, q grid {DOC_QS:?}", values.len()),
                );
            }

            for alpha in DOC_ALPHAS {
                let spec = RelativeQuantileSpec::core(alpha);
                let mut dd = DDSketch::new(alpha);
                for v in &positive {
                    dd.add(v);
                }
                let mut tally = Tally::default();
                spec.tally_into(&mut tally, positive_truth.sorted(), &DOC_QS, |q| {
                    dd.get_value_at_quantile(q)
                });
                tally.assert_none(
                    &format!("core DDSketch alpha={alpha} / ({id})"),
                    &format!(
                        "({id}) {label}, {} positive values, q grid {DOC_QS:?}",
                        positive.len()
                    ),
                );
            }
        }

        // (18): one stream per alpha, built on that alpha's bucket edges.
        for alpha in DOC_ALPHAS {
            let (label, values) = common::inputs::ddsketch_edge_input(alpha);
            let truth = NumericTruth::new(values.clone());
            let spec = RelativeQuantileSpec::core(alpha);
            let mut dd = DDSketch::new(alpha);
            for v in &values {
                dd.add(v);
            }
            assert_eq!(
                dd.get_count() as usize,
                values.len(),
                "({}) {label}: DDSketch dropped a value it can index",
                18
            );
            let mut tally = Tally::default();
            spec.tally_into(&mut tally, truth.sorted(), &DOC_QS, |q| {
                dd.get_value_at_quantile(q)
            });
            tally.assert_none(
                &format!("core DDSketch alpha={alpha} / (18)"),
                &format!("(18) {label}, n={}, q grid {DOC_QS:?}", values.len()),
            );
        }
    }

    macro_rules! documented_quantile_matrix {
        ($($kll:ident, $dd:ident => $id:literal;)*) => {
            $(
                #[test]
                fn $kll() {
                    kll_documented_matrix($id);
                }

                #[test]
                fn $dd() {
                    ddsketch_documented_matrix($id);
                }
            )*
        };
    }

    documented_quantile_matrix! {
        kll_family_on_input_1_holds_its_rank_error,
        ddsketch_on_input_1_holds_its_relative_error => 1;
        kll_family_on_input_2_holds_its_rank_error,
        ddsketch_on_input_2_holds_its_relative_error => 2;
        kll_family_on_input_3_holds_its_rank_error,
        ddsketch_on_input_3_holds_its_relative_error => 3;
        kll_family_on_input_4_holds_its_rank_error,
        ddsketch_on_input_4_holds_its_relative_error => 4;
        kll_family_on_input_5_holds_its_rank_error,
        ddsketch_on_input_5_holds_its_relative_error => 5;
        kll_family_on_input_6_holds_its_rank_error,
        ddsketch_on_input_6_holds_its_relative_error => 6;
        kll_family_on_input_7_holds_its_rank_error,
        ddsketch_on_input_7_holds_its_relative_error => 7;
        kll_family_on_input_8_holds_its_rank_error,
        ddsketch_on_input_8_holds_its_relative_error => 8;
        kll_family_on_input_9_holds_its_rank_error,
        ddsketch_on_input_9_holds_its_relative_error => 9;
        kll_family_on_input_10_holds_its_rank_error,
        ddsketch_on_input_10_holds_its_relative_error => 10;
        kll_family_on_input_11_holds_its_rank_error,
        ddsketch_on_input_11_holds_its_relative_error => 11;
        kll_family_on_input_12_holds_its_rank_error,
        ddsketch_on_input_12_holds_its_relative_error => 12;
    }
}
