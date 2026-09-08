//! Nitro sampling, end to end: the row-level accelerator behind
//! `CountMin::fast_insert_nitro` / `Count::fast_insert_nitro`, and the
//! `NitroBatch` wrapper's two ingestion paths.
//!
//! # The model every band here is computed from
//!
//! Nitro admits each unit of work with probability `p` and compensates by
//! writing a weight of `1/p`. Counters are integers, so the weight is rounded
//! **stochastically** — `floor(1/p) + Bernoulli(frac(1/p))` — which makes the
//! estimator unbiased at every rate rather than only at rates whose reciprocal
//! is an integer. With `r = frac(1/p)` and `f` units of true work:
//!
//! ```text
//!   E[estimate]   = f
//!   Var[estimate] = f * ( (1-p)/p  +  p * r * (1-r) )
//!                       \_________/    \___________/
//!                        admission       rounding
//! ```
//!
//! Every band below is `z` standard deviations of that, computed per rate from
//! the sketch's own configuration. Nothing here is a percentage.
//!
//! Unbiasedness is exact given uniform draws; that the shared skip table and
//! the rounding stream supply them is a modelling assumption, and the normal
//! approximation behind the `z`-sigma band is another. The coverage matrix
//! files Nitro as `asymptotic model` for both reasons.
//!
//! # The unit of work differs between the two APIs
//!
//! - **Row-level** (`fast_insert_nitro`): NitroSketch's per-row sampling. A
//!   `d`-row sketch turns `f` updates of a key into `f * d` *row slots*, each
//!   admitted independently. So row `r`'s counter is an estimate of `f` from
//!   `f` slots.
//! - **`NitroBatch`**: samples per *update*, and an admitted update writes its
//!   weight into every row. So the unit is `f`, and every row carries the same
//!   total.
//!
//! Either way the per-row statistic estimates `f` with the variance above,
//! which is what lets [`PATHS`] score all five ingestion paths against one
//! spec. Getting the unit wrong by a factor of `d` is exactly the kind of
//! error these tests exist to catch, so `nitro_estimate` / `estimate_median`
//! are pinned exactly at `p = 1` rather than only inside a band.
//!
//! # Trial units
//!
//! The randomness is the sampling schedule. One `(path, rate, seed)` triple is
//! one draw; the rows *inside* one sketch share an interleaved skip stream and
//! are not independent, so a trial's rows are reduced to a single pass/fail
//! before the binomial runs over seeds.
//!
//! Seeds are spaced so that no two trials read overlapping stretches of the
//! shared skip table — see `TRIAL_STRIDE`.

mod common;

use common::specs::{CountMinSpec, SIMULTANEOUS_LEVEL, SamplingConfidenceSpec, Tally};
use common::{FreqTruth, zipf_u64};

use asap_sketchlib::{Count, CountMin, DataInput, FastPath, NitroBatch, Vector2D};

/// Every rate the public constructors accept that the suite covers.
///
/// `0.3` and `0.07` are the load-bearing ones: their reciprocals are not
/// integers, so they are the only rates where the rounding correction is
/// observable at all. `1.0` is the degenerate full-sampling case.
const RATES: [f64; 6] = [1.0, 0.5, 0.3, 0.1, 0.07, 0.01];

/// Gaussian quantile for the sampling bands. Two-sided 6.3e-5 per trial under
/// the normal approximation to a sum of `f` i.i.d. weighted admissions.
const Z: f64 = 4.0;

const TRIALS: usize = 12;

/// Distance between two trials' starting offsets in the shared skip table.
///
/// A trial consumes one table entry per admission. `WORK_PER_TRIAL` fixes the
/// admission count at roughly 1000 per trial regardless of rate, so a
/// 5000-entry stride leaves every trial reading a disjoint stretch and the
/// twelve of them stay inside the 65536-entry table without wrapping into each
/// other.
const TRIAL_STRIDE: u64 = 5_000;

fn trial_seed(trial: usize) -> u64 {
    0x0117_0000 + trial as u64 * TRIAL_STRIDE
}

/// Admissions each trial should make, held constant across rates.
///
/// The true count is `WORK_PER_TRIAL / p`, so `f * p` — the admission count,
/// and therefore the table consumption — does not move with the rate, while
/// the relative width of the band stays usable at every rate (about 9% at
/// `p = 0.5` and 13% at `p = 0.01`). Holding `f` fixed instead would make the
/// band at `p = 0.01` wider than 100%, which would test nothing.
const WORK_PER_TRIAL: f64 = 1_000.0;

fn true_count_for(rate: f64) -> usize {
    (WORK_PER_TRIAL / rate).round() as usize
}

const ROWS: usize = 3;
const COLS: usize = 4_096;

/// A single key, so every counter it touches holds its mass and nothing else.
const SOLO_KEY: u64 = 0xC0FF_EE01;

// ---------------------------------------------------------------------------
// The ingestion paths, parameterised
// ---------------------------------------------------------------------------

/// One Nitro ingestion path, reduced to `(rate, seed, work)` in and one
/// unbiased estimate of `f` per row out.
///
/// Every path's per-row statistic has mean `f` and variance
/// `f((1-p)/p + p r(1-r))`, so all five answer to a single
/// [`SamplingConfidenceSpec`] and none of the trial/tally machinery is written
/// twice.
struct Path {
    name: &'static str,
    /// Per-row statistics after `f` updates of [`SOLO_KEY`], each an unbiased
    /// estimate of `f`.
    run: fn(f64, u64, usize) -> Vec<f64>,
    /// Rows the statistic is reported over.
    rows: usize,
}

const PATHS: [Path; 5] = [
    Path {
        name: "CountMin::fast_insert_nitro",
        run: row_countmin,
        rows: ROWS,
    },
    Path {
        name: "Count::fast_insert_nitro",
        run: row_countsketch,
        rows: ROWS,
    },
    Path {
        name: "NitroBatch::insert",
        run: batch_live,
        rows: 1,
    },
    Path {
        name: "NitroBatch::insert_cached_step",
        run: batch_cached,
        rows: 1,
    },
    Path {
        name: "NitroBatch<Vector2D<u32>>::insert",
        run: batch_bare,
        rows: 5,
    },
];

/// Total mass each row of a signed `i32` matrix holds.
fn signed_row_masses(storage: &Vector2D<i32>) -> Vec<f64> {
    (0..storage.rows())
        .map(|r| {
            (0..storage.cols())
                .map(|c| storage.query_one_counter(r, c) as f64)
                .sum()
        })
        .collect()
}

fn row_countmin(rate: f64, seed: u64, f: usize) -> Vec<f64> {
    let mut cm = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
    cm.enable_nitro_with_seed(rate, seed);
    let key = DataInput::U64(SOLO_KEY);
    for _ in 0..f {
        cm.fast_insert_nitro(&key);
    }
    signed_row_masses(cm.as_storage())
}

/// Count Sketch applies a per-row sign taken from the row's own hash bit, so a
/// single key's row mass is `±` its admitted mass; the magnitude carries the
/// same distribution as the Count-Min path.
fn row_countsketch(rate: f64, seed: u64, f: usize) -> Vec<f64> {
    let mut cs = Count::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
    cs.enable_nitro_with_seed(rate, seed);
    let key = DataInput::U64(SOLO_KEY);
    for _ in 0..f {
        cs.fast_insert_nitro(&key);
    }
    signed_row_masses(cs.as_storage())
        .into_iter()
        .map(f64::abs)
        .collect()
}

fn batch_target(rate: f64, seed: u64) -> NitroBatch<CountMin<Vector2D<i32>, FastPath>> {
    NitroBatch::with_target_and_seed(
        rate,
        CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS),
        seed,
    )
}

fn batch_live(rate: f64, seed: u64, f: usize) -> Vec<f64> {
    let mut nb = batch_target(rate, seed);
    nb.insert(&vec![SOLO_KEY as i64; f]);
    vec![nb.estimate_median(&DataInput::I64(SOLO_KEY as i64))]
}

fn batch_cached(rate: f64, seed: u64, f: usize) -> Vec<f64> {
    let mut nb = batch_target(rate, seed);
    nb.insert_cached_step(&vec![SOLO_KEY as i64; f]);
    vec![nb.estimate_median(&DataInput::I64(SOLO_KEY as i64))]
}

/// `NitroBatch<Vector2D<u32>>` — the bare-storage target reached by
/// `init_nitro` — has **no public per-key query path**: `NitroEstimate` is
/// implemented only for the `Vector2D<i32>`-backed Count-Min and Count Sketch,
/// and `CountMin::estimate` cannot be instantiated over a `u32` counter
/// because it needs `Counter: From<i32>`. This does not invent one by
/// re-deriving the fast-path hash — a test that re-derives it asserts its own
/// arithmetic, not the estimator's.
///
/// What *is* publicly observable, and is what Nitro controls, is the admitted
/// mass: every admitted update writes its weight into one cell of every row,
/// so a row's total is the admitted mass, computed without touching a hash.
/// That every row carries the identical total is a structural invariant of the
/// batch path and is asserted here rather than left to the band.
fn batch_bare(rate: f64, seed: u64, f: usize) -> Vec<f64> {
    let mut nb = NitroBatch::init_nitro_with_seed(rate, seed);
    nb.insert(&vec![SOLO_KEY as i64; f]);
    let target = nb.target();
    let masses: Vec<f64> = (0..target.rows())
        .map(|r| {
            (0..target.cols())
                .map(|c| target.query_one_counter(r, c) as f64)
                .sum()
        })
        .collect();
    for (r, mass) in masses.iter().enumerate() {
        assert_eq!(
            *mass, masses[0],
            "row {r} carries {mass} but row 0 carries {}; every row must receive each \
             admitted update (rate={rate} seed={seed:#x})",
            masses[0]
        );
    }
    masses
}

// ---------------------------------------------------------------------------
// The sampling band, over every path and every rate
// ---------------------------------------------------------------------------

/// Averaging independent trials drives the sampling noise down by
/// `sqrt(trials)` and leaves any systematic rounding bias standing.
fn assert_unbiased_across_trials(
    label: &str,
    observations: &[f64],
    truth: f64,
    spec: &SamplingConfidenceSpec,
    context: &str,
) {
    let n = observations.len() as f64;
    let mean = observations.iter().sum::<f64>() / n;
    let allowed = Z * spec.sigma(truth) / n.sqrt();
    assert!(
        (mean - truth).abs() <= allowed,
        "{label}: the mean over {n} independent seeds is {mean:.1} against a true \
         {truth:.0} — relative bias {:.4}, outside the mean's band \
         z*sigma/sqrt(trials) = {allowed:.1}. A weight whose expectation is not 1/p \
         shows up here even when every single trial sits inside its own band. \
         {context}",
        (mean - truth) / truth,
    );
}

/// Every ingestion path, at every rate the public API accepts, inside the band
/// its own configuration earns — and unbiased across seeds.
///
/// Two assertions per `(path, rate)`, and the second is the one the rounding
/// correction exists for:
///
/// 1. every row of every trial lands in the `z`-sigma band (rows within a
///    trial share an interleaved skip stream, so a trial passes only if *all*
///    its rows do, and the binomial runs over seeds);
/// 2. the **mean across independent seeds** is unbiased. A single-trial band
///    cannot see a systematic error: a `ceil(1/p)` weight puts every estimate
///    at `p = 0.3` 20% high, which averaging exposes and a band the width of
///    one trial's noise does not.
#[test]
fn every_nitro_path_is_unbiased_inside_its_sampling_band_at_every_rate() {
    for path in &PATHS {
        for &rate in &RATES {
            let spec = SamplingConfidenceSpec::new(rate, Z);
            let f = true_count_for(rate);
            let mut tally = Tally::default();
            let mut first_row = Vec::with_capacity(TRIALS);

            for trial in 0..TRIALS {
                let seed = trial_seed(trial);
                let stats = (path.run)(rate, seed, f);
                assert_eq!(
                    stats.len(),
                    path.rows,
                    "{}: expected {} per-row statistics",
                    path.name,
                    path.rows
                );
                first_row.push(stats[0]);

                let failures: Vec<String> = stats
                    .iter()
                    .enumerate()
                    .filter_map(|(r, m)| {
                        spec.check(*m, f as f64, 0.0)
                            .err()
                            .map(|d| format!("row {r}: {d}"))
                    })
                    .collect();
                tally.record(failures.is_empty(), || {
                    format!("seed={seed:#x}: {}", failures.join("; "))
                });
            }

            let (q, r) = spec.weight_parts();
            let context = format!(
                "{} rate={rate} rows={} cols={COLS} f={f} (f*p={:.0} admissions), \
                 weight {q}+Bernoulli({r:.4}) so E[W]={:.4}, sigma={:.2}, z={Z}; one \
                 trial per seed, all {} rows must pass",
                path.name,
                path.rows,
                f as f64 * rate,
                spec.expected_weight(),
                spec.sigma(f as f64),
                path.rows,
            );
            tally.assert_independent_binomial(
                &format!("{} rate={rate} / sampling band", path.name),
                spec.per_check_failure(),
                &context,
            );
            assert_unbiased_across_trials(
                &format!("{} rate={rate}", path.name),
                &first_row,
                f as f64,
                &spec,
                &context,
            );
        }
    }
}

/// Full sampling is not approximate on any path, and the query reads back
/// exactly what the insert wrote.
///
/// At `p = 1` every unit of work is admitted at weight 1, so each of the three
/// numbers below is pinned exactly. Two specific failure modes are named
/// because both have shipped:
///
/// - **`0`** — the insert derives its cells from a raw `hash128_seeded` while
///   the estimator queries through `hash_for_matrix` / `col_for_row`. Those
///   disagree whenever the matrix hash is not the identity on the raw hash, so
///   every estimate reads a cell the insert never touched.
/// - **`f / rows`** — an ingestion path that assigns each sampled record to
///   one row instead of all of them, which divides every per-row counter by
///   the depth.
#[test]
fn full_sampling_is_exact_and_the_query_reads_the_cells_the_insert_wrote() {
    const F: usize = 5_000;
    let key = DataInput::U64(SOLO_KEY);

    for path in &PATHS {
        let stats = (path.run)(1.0, trial_seed(0), F);
        for (r, mass) in stats.iter().enumerate() {
            assert_eq!(
                *mass,
                F as f64,
                "{} rate=1.0: row {r} holds {mass} for {F} updates. 0 means insert and \
                 query disagree on the hash domain; {:.1} means the path divided the \
                 work across its {} rows",
                path.name,
                F as f64 / path.rows as f64,
                path.rows,
            );
        }
    }

    // The public per-key estimators, pinned on the same workload.
    let mut cm = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
    cm.enable_nitro_with_seed(1.0, trial_seed(0));
    for _ in 0..F {
        cm.fast_insert_nitro(&key);
    }
    assert_eq!(
        cm.nitro_estimate(&key),
        F as f64,
        "CountMin::nitro_estimate must return the exact count at rate 1.0"
    );

    for (name, run) in [
        ("NitroBatch::insert", false),
        ("NitroBatch::insert_cached_step", true),
    ]
    .map(|(n, cached)| (n, cached))
    {
        let mut nb = batch_target(1.0, trial_seed(0));
        let data = vec![SOLO_KEY as i64; F];
        if run {
            nb.insert_cached_step(&data);
        } else {
            nb.insert(&data);
        }
        assert_eq!(
            nb.estimate_median(&DataInput::I64(SOLO_KEY as i64)),
            F as f64,
            "{name}::estimate_median must return the exact count at rate 1.0"
        );
    }
}

/// Every path reproduces from its seed, and different seeds genuinely differ.
#[test]
fn every_nitro_path_is_reproducible_from_its_seed() {
    const RATE: f64 = 0.3;
    let f = true_count_for(RATE);

    for path in &PATHS {
        let first = (path.run)(RATE, trial_seed(0), f);
        assert_eq!(
            (path.run)(RATE, trial_seed(0), f),
            first,
            "{}: the same seed must produce the same admitted subset",
            path.name
        );
        let distinct: std::collections::HashSet<u64> = (1..TRIALS)
            .map(|t| (path.run)(RATE, trial_seed(t), f)[0].to_bits())
            .collect();
        assert!(
            distinct.len() > 1,
            "{}: {} distinct seeds produced {} distinct results; the seed is not \
             reaching the sampler",
            path.name,
            TRIALS - 1,
            distinct.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/// Merging two same-rate Nitro sketches sums their admitted mass, so a key
/// split across both must land in the band for the combined work — on the
/// row-level path and on `NitroBatch` alike.
#[test]
fn nitro_merge_lands_in_the_combined_band() {
    type Merged = fn(f64, u64, u64, usize) -> f64;

    let row_level: Merged = |rate, sa, sb, half| {
        let build = |seed: u64| {
            let mut cm = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
            cm.enable_nitro_with_seed(rate, seed);
            let key = DataInput::U64(SOLO_KEY);
            for _ in 0..half {
                cm.fast_insert_nitro(&key);
            }
            cm
        };
        let mut a = build(sa);
        a.merge(&build(sb));
        signed_row_masses(a.as_storage())[0]
    };
    let batch: Merged = |rate, sa, sb, half| {
        let build = |seed: u64| {
            let mut nb = batch_target(rate, seed);
            nb.insert(&vec![SOLO_KEY as i64; half]);
            nb
        };
        let mut a = build(sa);
        a.merge(&build(sb));
        a.estimate_median(&DataInput::I64(SOLO_KEY as i64))
    };

    for (name, merged) in [
        ("CountMin::fast_insert_nitro + CountMin::merge", row_level),
        ("NitroBatch::insert + NitroBatch::merge", batch),
    ] {
        for &rate in &RATES {
            let spec = SamplingConfidenceSpec::new(rate, Z);
            let half = true_count_for(rate);
            let combined = (2 * half) as f64;
            let mut tally = Tally::default();

            for pair in 0..TRIALS / 2 {
                let (sa, sb) = (trial_seed(2 * pair), trial_seed(2 * pair + 1));
                let outcome = spec.check(merged(rate, sa, sb, half), combined, 0.0);
                tally.record(outcome.is_ok(), || {
                    format!("seeds {sa:#x}/{sb:#x}: {}", outcome.unwrap_err())
                });
            }
            tally.assert_independent_binomial(
                &format!("{name} rate={rate} / combined band"),
                spec.per_check_failure(),
                &format!(
                    "two shards of {half} updates each, merged; one trial per disjoint \
                     seed pair; sigma over the combined work = {:.2}, z={Z}",
                    spec.sigma(combined)
                ),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Row-level Nitro under collisions
// ---------------------------------------------------------------------------

/// A Zipf stream through the row-level Count-Min path, per key.
///
/// With collisions the two error sources compose, and the band is the sum of
/// the two the sketch actually earns — no third constant:
///
/// - **below:** a cell only ever *gains* mass from collisions, so the minimum
///   over rows is at least the key's own admitted mass in some row, and that
///   is within `z*sigma(f)` of `f`.
/// - **above:** the minimum is at most row 0, which holds the key's own
///   admitted mass plus the colliding keys' — bounded by Count-Min's own
///   simultaneous budget, with `z*sigma(N)` for the noise, `sigma` being
///   increasing in the mass and a cell holding at most the whole stream.
#[test]
fn row_level_countmin_nitro_tracks_a_zipf_stream_within_the_combined_band() {
    const N: usize = 200_000;
    const DOMAIN: usize = 512;
    const STREAM_SEED: u64 = 0x0117_2199;

    let stream = zipf_u64(N, DOMAIN, 1.1, STREAM_SEED);
    let mut truth = FreqTruth::default();
    for k in &stream {
        truth.observe(*k as i64);
    }
    let total = truth.total() as f64;
    let distinct = truth.distinct();
    let cm_spec = CountMinSpec::new(ROWS, COLS);

    // Rates whose admission count keeps the whole stream inside one disjoint
    // stretch of the skip table.
    for &rate in &[1.0f64, 0.3, 0.1, 0.07] {
        let spec = SamplingConfidenceSpec::new(rate, Z);
        let mut cm = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
        cm.enable_nitro_with_seed(rate, trial_seed(0));
        for k in &stream {
            cm.fast_insert_nitro(&DataInput::U64(*k));
        }

        let collision_budget =
            |f: f64| cm_spec.simultaneous_bound(total, f, distinct, SIMULTANEOUS_LEVEL);
        let context = format!(
            "row-level CountMin Nitro over zipf(1.1) domain={DOMAIN} n={N} \
             seed={STREAM_SEED:#x}, rate={rate}, rows={ROWS} cols={COLS}"
        );

        let mut low = Tally::default();
        let mut high = Tally::default();
        for (k, c) in truth.pairs() {
            let f = c as f64;
            let est = cm.nitro_estimate(&DataInput::U64(k as u64));
            let floor = f - Z * spec.sigma(f);
            let ceiling = f + collision_budget(f) + Z * spec.sigma(total);
            low.record(est >= floor, || {
                format!("key {k}: true {f}, est {est:.1} below f - z*sigma(f) = {floor:.1}")
            });
            high.record(est <= ceiling, || {
                format!(
                    "key {k}: true {f}, est {est:.1} above f + collisions + z*sigma(N) = \
                     {ceiling:.1}"
                )
            });
        }
        // Both halves are union-bounded over the probed keys, so neither
        // tolerates a violation.
        low.assert_none(
            &format!("row-level CountMin Nitro rate={rate} / lower band"),
            &context,
        );
        high.assert_none(
            &format!("row-level CountMin Nitro rate={rate} / upper band"),
            &context,
        );
    }
}

/// Several keys of comparable frequency, so no single key dominates the cells
/// it shares and the per-key estimates cannot be carried by one hot flow.
#[test]
fn row_level_countmin_nitro_separates_keys_of_similar_frequency() {
    const KEYS: usize = 16;
    const PER_KEY: usize = 20_000;

    for &rate in &[1.0f64, 0.5, 0.3, 0.1] {
        let spec = SamplingConfidenceSpec::new(rate, Z);
        let mut cm = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
        cm.enable_nitro_with_seed(rate, trial_seed(1));
        let keys: Vec<u64> = (0..KEYS as u64).map(|i| 0x5EED_0000 + i * 7919).collect();
        for _ in 0..PER_KEY {
            for k in &keys {
                cm.fast_insert_nitro(&DataInput::U64(*k));
            }
        }

        let total = (KEYS * PER_KEY) as f64;
        let f = PER_KEY as f64;
        let cm_spec = CountMinSpec::new(ROWS, COLS);
        let ceiling = f
            + cm_spec.simultaneous_bound(total, f, KEYS, SIMULTANEOUS_LEVEL)
            + Z * spec.sigma(total);
        let floor = f - Z * spec.sigma(f);
        let mut tally = Tally::default();
        for k in &keys {
            let est = cm.nitro_estimate(&DataInput::U64(*k));
            tally.record(est >= floor && est <= ceiling, || {
                format!("key {k:#x}: est {est:.1} outside [{floor:.1}, {ceiling:.1}]")
            });
        }
        tally.assert_none(
            &format!("row-level CountMin Nitro rate={rate} / {KEYS} equal-frequency keys"),
            &format!("each key inserted {PER_KEY} times, rows={ROWS} cols={COLS}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Weight saturation
// ---------------------------------------------------------------------------

/// The scaled increment is clamped into the counter's domain rather than
/// wrapping into a decrement.
#[test]
fn nitro_saturates_oversized_weights_instead_of_wrapping() {
    assert_eq!(
        asap_sketchlib::nitro_delta_saturated_i32(u64::MAX),
        i32::MAX,
        "an oversized weight must clamp to i32::MAX, not wrap negative"
    );
    assert_eq!(
        asap_sketchlib::nitro_delta_saturated_u32(u64::MAX),
        u32::MAX,
        "an oversized weight must clamp to u32::MAX"
    );
    assert_eq!(asap_sketchlib::nitro_delta_saturated_i32(7), 7);
    assert_eq!(asap_sketchlib::nitro_delta_saturated_u32(7), 7);

    // A rate low enough to push `1/rate` past `i32::MAX` must still leave
    // counters non-negative, on the batch path and the row-level path alike.
    let mut tiny = batch_target(1e-9, trial_seed(0));
    tiny.insert(&vec![1i64; 4_000]);
    assert!(
        tiny.estimate_median(&DataInput::I64(1)) >= 0.0,
        "a saturating weight must not turn Count-Min counters negative"
    );

    let mut row_level = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
    row_level.enable_nitro_with_seed(1e-9, trial_seed(0));
    for _ in 0..4_000 {
        row_level.fast_insert_nitro(&DataInput::U64(SOLO_KEY));
    }
    assert!(
        row_level.nitro_estimate(&DataInput::U64(SOLO_KEY)) >= 0.0,
        "a saturating weight must not turn row-level counters negative"
    );
}

// ---------------------------------------------------------------------------
// Continuation across a serde round trip and across a context snapshot
// ---------------------------------------------------------------------------

/// Rates whose reciprocal is not an integer, so the stochastic-rounding stream
/// is live and its state actually has to survive.
const CONTINUATION_RATES: [f64; 2] = [0.3, 0.07];

fn round_trip(cm: &CountMin<Vector2D<i32>, FastPath>) -> CountMin<Vector2D<i32>, FastPath> {
    let bytes = rmp_serde::to_vec_named(cm).expect("serialize a Nitro-enabled CountMin");
    rmp_serde::from_slice(&bytes).expect("deserialize a Nitro-enabled CountMin")
}

fn nitro_enabled(rate: f64, seed: u64) -> CountMin<Vector2D<i32>, FastPath> {
    let mut cm = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(ROWS, COLS);
    cm.enable_nitro_with_seed(rate, seed);
    cm
}

fn cells(cm: &CountMin<Vector2D<i32>, FastPath>) -> Vec<i32> {
    let s = cm.as_storage();
    (0..s.rows())
        .flat_map(|r| (0..s.cols()).map(move |c| (r, c)))
        .map(|(r, c)| s.query_one_counter(r, c))
        .collect()
}

/// A sketch that is encoded and decoded after every update must end up
/// identical to one that was never interrupted.
///
/// This is the property the sampling state has to be *complete* for. The
/// rounding stream is part of it: it is serialized, so a decoded sketch
/// continues the Bernoulli sequence rather than restarting it from a constant,
/// which matters at any rate whose reciprocal is not an integer.
#[test]
fn a_serde_round_trip_after_every_update_reproduces_the_uninterrupted_run() {
    const UPDATES: usize = 600;
    const TAIL: usize = 200;
    let key = DataInput::U64(SOLO_KEY);

    for &rate in &CONTINUATION_RATES {
        let seed = trial_seed(0);
        let mut continuous = nitro_enabled(rate, seed);
        let mut interrupted = nitro_enabled(rate, seed);

        for _ in 0..UPDATES {
            continuous.fast_insert_nitro(&key);
            interrupted.fast_insert_nitro(&key);
            interrupted = round_trip(&interrupted);
        }

        assert_eq!(
            cells(&interrupted),
            cells(&continuous),
            "rate={rate}: a sketch round-tripped after every one of {UPDATES} updates \
             must hold the same counters, cell for cell, as one that was never \
             serialized"
        );
        assert_eq!(
            interrupted.nitro_estimate(&key),
            continuous.nitro_estimate(&key),
            "rate={rate}: the estimates must agree exactly"
        );

        // And the *future* has to agree too: a decoded sketch that resumed
        // from a reset rounding stream can hold the right counters and still
        // diverge on the next update.
        for _ in 0..TAIL {
            continuous.fast_insert_nitro(&key);
            interrupted.fast_insert_nitro(&key);
        }
        assert_eq!(
            cells(&interrupted),
            cells(&continuous),
            "rate={rate}: {TAIL} further updates after the last round trip must follow \
             the same admission and weight sequence"
        );
    }
}

/// A payload written before `Nitro::rounding_state` existed must still decode.
///
/// `rounding_state` is a trailing serialized field carrying
/// `#[serde(default)]`, so a map-encoded payload that simply lacks the key
/// decodes with the default stream. This builds exactly that payload from a
/// mirror of the field layout without it — there is no other way to produce
/// one once the field exists — and checks that the counters survive and the
/// decoded sketch is usable.
///
/// `mask_bits` and `mask` are present because such payloads carry them;
/// `Vector2D`'s decoder recomputes both from `cols` and ignores what the
/// payload says, so the values below only have to be well-formed.
#[test]
fn a_payload_written_before_the_rounding_field_existed_still_decodes() {
    #[derive(serde::Serialize)]
    struct LegacyNitro {
        is_nitro_mode: bool,
        sampling_rate: f64,
        to_skip: usize,
        inv_ln_one_minus_p: f64,
        delta: u64,
        idx: usize,
        mask: usize,
    }
    #[derive(serde::Serialize)]
    struct LegacyVector2D {
        data: Vec<i32>,
        rows: usize,
        cols: usize,
        mask_bits: u32,
        mask: u128,
        nitro: LegacyNitro,
    }
    #[derive(serde::Serialize)]
    struct LegacyCountMin {
        counts: LegacyVector2D,
        row: usize,
        col: usize,
    }

    const RATE: f64 = 0.3;
    // Row 0 column 17 holds 4; nothing else is set. Any recognisable pattern
    // does, since the point is that the counters survive the decode.
    let mut data = vec![0i32; ROWS * COLS];
    data[17] = 4;

    let legacy = LegacyCountMin {
        counts: LegacyVector2D {
            data: data.clone(),
            rows: ROWS,
            cols: COLS,
            mask_bits: COLS.trailing_zeros(),
            mask: (COLS - 1) as u128,
            nitro: LegacyNitro {
                is_nitro_mode: true,
                sampling_rate: RATE,
                to_skip: 2,
                inv_ln_one_minus_p: 1.0 / (1.0 - RATE).ln(),
                delta: (1.0 / RATE).floor() as u64,
                idx: 9,
                mask: 0x1_0000,
            },
        },
        row: ROWS,
        col: COLS,
    };

    let bytes = rmp_serde::to_vec_named(&legacy).expect("encode the legacy shape");
    let mut decoded: CountMin<Vector2D<i32>, FastPath> =
        rmp_serde::from_slice(&bytes).expect("a payload without `rounding_state` must decode");

    assert_eq!(
        cells(&decoded),
        data,
        "the counters must survive a decode of the pre-`rounding_state` shape"
    );

    // And the decoded sketch keeps sampling: the rounding stream starts from
    // its default rather than failing to decode.
    let key = DataInput::U64(SOLO_KEY);
    for _ in 0..2_000 {
        decoded.fast_insert_nitro(&key);
    }
    let spec = SamplingConfidenceSpec::new(RATE, Z);
    let mass = signed_row_masses(decoded.as_storage())[0] - 4.0;
    assert!(
        spec.check(mass, 2_000.0, 0.0).is_ok(),
        "a decoded legacy sketch must keep sampling at its stored rate: row-0 mass \
         {mass} for 2000 updates at p={RATE}, band +-{:.1}",
        spec.half_width(2_000.0)
    );
}

/// A context snapshot taken mid-stream and restored onto a fresh sketch must
/// replay the rest of the stream exactly.
///
/// [`asap_sketchlib::NitroContext`] carries the skip cursor, the outstanding
/// skip, and the rounding stream. The legacy `get_ctx` / `commit_ctx` pair
/// carries only the first two, which is why it is documented as a partial
/// restore and is not used here.
#[test]
fn a_context_snapshot_restores_the_rest_of_the_stream_exactly() {
    const PREFIX: usize = 500;
    const SUFFIX: usize = 500;
    let key = DataInput::U64(SOLO_KEY);

    for &rate in &CONTINUATION_RATES {
        let seed = trial_seed(3);
        let mut continuous = nitro_enabled(rate, seed);
        for _ in 0..PREFIX {
            continuous.fast_insert_nitro(&key);
        }
        let snapshot = continuous.as_storage().nitro().context();
        let prefix_cells = cells(&continuous);

        for _ in 0..SUFFIX {
            continuous.fast_insert_nitro(&key);
        }

        // A fresh sketch at the same rate, wound forward to the snapshot.
        let mut restored = nitro_enabled(rate, seed);
        restored
            .as_storage_mut()
            .nitro_mut()
            .restore_context(snapshot);
        for _ in 0..SUFFIX {
            restored.fast_insert_nitro(&key);
        }

        let suffix_of_continuous: Vec<i32> = cells(&continuous)
            .into_iter()
            .zip(prefix_cells)
            .map(|(after, before)| after - before)
            .collect();
        assert_eq!(
            cells(&restored),
            suffix_of_continuous,
            "rate={rate}: the {SUFFIX} updates after a restored snapshot must write \
             exactly what the uninterrupted run wrote over the same stretch"
        );
    }
}

const INTEGRATION_ROWS: usize = 4;
const INTEGRATION_COLS: usize = 2_048;
const INTEGRATION_N: usize = 40_000;
const INTEGRATION_DOMAIN: usize = 2_048;
const INTEGRATION_SEED: u64 = 0x4E17_0001;

fn integration_stream() -> (Vec<u64>, FreqTruth) {
    let stream = zipf_u64(INTEGRATION_N, INTEGRATION_DOMAIN, 1.1, INTEGRATION_SEED);
    let mut truth = FreqTruth::default();
    for k in &stream {
        truth.observe(*k as i64);
    }
    (stream, truth)
}

fn counters(storage: &Vector2D<i32>) -> Vec<i32> {
    let mut out = Vec::with_capacity(storage.rows() * storage.cols());
    for row in 0..storage.rows() {
        for col in 0..storage.cols() {
            out.push(storage.query_one_counter(row, col));
        }
    }
    out
}

#[test]
fn count_min_at_full_nitro_sampling_writes_exactly_what_a_plain_insert_writes() {
    let (stream, truth) = integration_stream();
    let mut plain =
        CountMin::<Vector2D<i32>, FastPath>::with_dimensions(INTEGRATION_ROWS, INTEGRATION_COLS);
    let mut sampled =
        CountMin::<Vector2D<i32>, FastPath>::with_dimensions(INTEGRATION_ROWS, INTEGRATION_COLS);
    sampled.enable_nitro(1.0);
    for k in &stream {
        plain.insert(&DataInput::U64(*k));
        sampled.fast_insert_nitro(&DataInput::U64(*k));
    }
    assert_eq!(
        counters(sampled.as_storage()),
        counters(plain.as_storage()),
        "rate=1 Nitro insertion must write the same counters as insert"
    );
    for (key, count) in truth.pairs() {
        let probe = DataInput::U64(key as u64);
        assert_eq!(
            sampled.estimate(&probe),
            plain.estimate(&probe),
            "key {key}: rate=1 Nitro estimate diverged"
        );
        assert!(
            sampled.nitro_estimate(&probe) >= count as f64,
            "key {key}: the row median must not fall below the true count {count}"
        );
        assert!(
            sampled.nitro_estimate(&probe) >= plain.estimate(&probe) as f64,
            "key {key}: the row median must not fall below the row minimum"
        );
    }
}

#[test]
fn count_min_after_disable_nitro_leaves_the_sampled_insert_path_inert() {
    let (stream, _) = integration_stream();
    let mut sketch =
        CountMin::<Vector2D<i32>, FastPath>::with_dimensions(INTEGRATION_ROWS, INTEGRATION_COLS);
    sketch.enable_nitro(1.0);
    for k in &stream {
        sketch.fast_insert_nitro(&DataInput::U64(*k));
    }
    let before = counters(sketch.as_storage());

    sketch.disable_nitro();
    for k in &stream {
        sketch.fast_insert_nitro(&DataInput::U64(*k));
    }
    assert_eq!(
        counters(sketch.as_storage()),
        before,
        "with sampling disabled the Nitro insert path must not move a counter"
    );

    for k in &stream {
        sketch.insert(&DataInput::U64(*k));
    }
    assert_ne!(
        counters(sketch.as_storage()),
        before,
        "disabling sampling must not disable the ordinary insert path"
    );
}

#[test]
fn count_min_seeded_nitro_sampling_is_reproducible_and_admits_a_strict_subset() {
    const RATE: f64 = 0.1;
    let (stream, _) = integration_stream();

    let build = |seed: u64| {
        let mut sketch = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
            INTEGRATION_ROWS,
            INTEGRATION_COLS,
        );
        sketch.enable_nitro_with_seed(RATE, seed);
        for k in &stream {
            sketch.fast_insert_nitro(&DataInput::U64(*k));
        }
        sketch
    };

    let first = build(0x5A11_0001);
    let second = build(0x5A11_0001);
    assert_eq!(
        counters(first.as_storage()),
        counters(second.as_storage()),
        "two sketches at the same rate and seed must admit the same subset"
    );

    let mut full =
        CountMin::<Vector2D<i32>, FastPath>::with_dimensions(INTEGRATION_ROWS, INTEGRATION_COLS);
    for k in &stream {
        full.insert(&DataInput::U64(*k));
    }
    let sampled_slots = counters(first.as_storage())
        .iter()
        .filter(|c| **c > 0)
        .count();
    let full_slots = counters(full.as_storage())
        .iter()
        .filter(|c| **c > 0)
        .count();
    assert!(
        sampled_slots < full_slots,
        "sampling at rate {RATE} touched {sampled_slots} counters, not fewer than the \
         {full_slots} an unsampled pass touches"
    );
}

#[test]
fn count_sketch_at_full_nitro_sampling_writes_exactly_what_a_plain_insert_writes() {
    let (stream, truth) = integration_stream();
    let mut plain =
        Count::<Vector2D<i32>, FastPath>::with_dimensions(INTEGRATION_ROWS, INTEGRATION_COLS);
    let mut sampled =
        Count::<Vector2D<i32>, FastPath>::with_dimensions(INTEGRATION_ROWS, INTEGRATION_COLS);
    sampled.enable_nitro(1.0);
    for k in &stream {
        plain.insert(&DataInput::U64(*k));
        sampled.fast_insert_nitro(&DataInput::U64(*k));
    }
    assert_eq!(
        counters(sampled.as_storage()),
        counters(plain.as_storage()),
        "rate=1 Nitro insertion must write the same signed counters as insert"
    );
    for (key, _) in truth.pairs() {
        let probe = DataInput::U64(key as u64);
        assert_eq!(
            sampled.estimate(&probe),
            plain.estimate(&probe),
            "key {key}: rate=1 Nitro estimate diverged"
        );
    }
}

#[test]
fn count_sketch_seeded_nitro_sampling_is_reproducible() {
    const RATE: f64 = 0.25;
    let (stream, _) = integration_stream();
    let build = |seed: u64| {
        let mut sketch =
            Count::<Vector2D<i32>, FastPath>::with_dimensions(INTEGRATION_ROWS, INTEGRATION_COLS);
        sketch.enable_nitro_with_seed(RATE, seed);
        for k in &stream {
            sketch.fast_insert_nitro(&DataInput::U64(*k));
        }
        sketch
    };
    assert_eq!(
        counters(build(0x5A11_0002).as_storage()),
        counters(build(0x5A11_0002).as_storage()),
        "two Count sketches at the same rate and seed must admit the same subset"
    );
}
