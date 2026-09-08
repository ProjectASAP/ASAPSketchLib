//! Windowed frameworks: every `EHSketchList` variant inside an
//! `ExponentialHistogram`, and every `TumblingWindowSketch` implementation
//! inside a `TumblingWindow`.
//!
//! The window machinery is shared, but the payloads are not, so a single
//! tolerance across all of them would be meaningless. Each variant is checked
//! against **its own** error metric — Count-Min's additive bound, Count
//! Sketch's L2 bound, HLL's register model, KLL's rank error, DDSketch's
//! relative value error — over the exact contents of the window the query
//! actually covered.
//!
//! ## Reading the ground truth off a bucket span
//!
//! `query_interval_merge` snaps a requested interval to bucket boundaries, so
//! an arbitrary `[t1, t2]` has no exact reference. Every query below therefore
//! asks for the histogram's **full retained span**, `[payload[0].min_time,
//! payload.last().max_time]`, which merges every retained bucket and nothing
//! else. Both endpoints are public, so the reference window is known exactly
//! and no tolerance is spent on bucket granularity.

mod common;

use common::specs::{
    CardinalityConfidenceSpec, CountMinSpec, CountSketchSpec, KllRankSpec, RelativeQuantileSpec,
    Tally,
};
use common::{FreqTruth, NumericTruth, uniform_u64, zipf_u64};

use asap_sketchlib::{
    Coco, Count, CountL2HH, CountMin, DDSketch, DataInput, EHSketchList, Elastic, ErtlMLE,
    ExponentialHistogram, FastPath, FoldCMS, FoldCMSConfig, FoldCS, FoldCSConfig, HyperLogLog, KLL,
    KLLConfig, SketchNorm, SketchPool, TumblingWindow, UnivMon, UnivMonQ, UnivMonQConfig, Vector2D,
};

const EH_K: usize = 8;
const EH_WINDOW: u64 = 1_000_000; // no expiry inside the accuracy runs
// Sized so the histogram's per-update prototype clone (an `ExponentialHistogram`
// copies its prototype sketch on every insert) stays affordable in an
// unoptimised `cargo test` run. The bounds are computed from each instance's
// own dimensions, so a smaller grid narrows nothing about what is asserted.
const N: usize = 10_000;
const DOMAIN: usize = 2_048;
const STREAM_SEED: u64 = 0x0E11_0001;

/// Matrix dimensions for the counter-backed variants, chosen so their bounds
/// are meaningful at `N` updates over `DOMAIN` keys.
/// UnivMon layer geometry used by the `UNIVMON` payload below. Its L2 band is
/// derived from these, so a change here moves the band with it.
const UNIVMON_ROWS: usize = 5;
const UNIVMON_COLS: usize = 2_048;

const ROWS: usize = 3;
const COLS: usize = 512;

/// The full retained span of a histogram: querying it merges every bucket, so
/// the reference window is the whole stream that has not expired.
fn full_span(eh: &ExponentialHistogram) -> (u64, u64) {
    let first = eh.payload.first().expect("histogram has buckets");
    let last = eh.payload.last().expect("histogram has buckets");
    (first.min_time, last.max_time)
}

// ------------------------------------------------- Norm policy per variant

/// Which merge rule each payload selects. `COUNTL2HH` and `UNIVMON` carry an
/// L2 mass and are merged by it; every other variant has no L2 mass to read
/// and falls back to the L1 bucket-size rule. Getting this wrong would silently
/// change how buckets consolidate, and therefore how much of the window a query
/// actually covers, so it is asserted per variant rather than assumed.
#[test]
fn every_eh_variant_selects_the_documented_merge_norm() {
    let l2_variants = [
        (
            "COUNTL2HH",
            asap_sketchlib::EHSketchList::COUNTL2HH(CountL2HH::with_dimensions(ROWS, COLS)),
        ),
        (
            "UNIVMON",
            asap_sketchlib::EHSketchList::UNIVMON(UnivMon::init_univmon(32, ROWS, COLS, 4)),
        ),
    ];
    for (name, proto) in l2_variants {
        assert!(
            proto.supports_norm(SketchNorm::L2) && !proto.supports_norm(SketchNorm::L1),
            "{name} must be an L2-merged payload"
        );
        let eh = ExponentialHistogram::new(EH_K, EH_WINDOW, proto);
        assert_eq!(
            eh.merge_norm,
            SketchNorm::L2,
            "{name} histogram must merge by L2 mass"
        );
    }

    let l1_variants: Vec<(&str, asap_sketchlib::EHSketchList)> = vec![
        (
            "CM",
            asap_sketchlib::EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
                ROWS, COLS,
            )),
        ),
        (
            "CS",
            asap_sketchlib::EHSketchList::CS(Count::<Vector2D<i32>, FastPath>::with_dimensions(
                ROWS, COLS,
            )),
        ),
        (
            "COCO",
            asap_sketchlib::EHSketchList::COCO(Coco::init_with_size(512, 4)),
        ),
        (
            "ELASTIC",
            asap_sketchlib::EHSketchList::ELASTIC(Elastic::init_with_length(512)),
        ),
        (
            "HLL",
            asap_sketchlib::EHSketchList::HLL(HyperLogLog::<ErtlMLE>::new()),
        ),
        (
            "KLL",
            asap_sketchlib::EHSketchList::KLL(KLL::init_kll_with_seed(200, 0x5EED_0100)),
        ),
        (
            "DDS",
            asap_sketchlib::EHSketchList::DDS(DDSketch::new(0.01)),
        ),
    ];
    for (name, proto) in l1_variants {
        assert!(
            proto.supports_norm(SketchNorm::L1),
            "{name} must be an L1-merged payload"
        );
        let eh = ExponentialHistogram::new(EH_K, EH_WINDOW, proto);
        assert_eq!(
            eh.merge_norm,
            SketchNorm::L1,
            "{name} histogram must merge by bucket size"
        );
    }
}

// ------------------------------------------------ Counter-backed variants

/// Feeds a Zipf key stream through a histogram and returns the merged payload
/// over the full retained span, together with the exact truth for that span.
fn run_keyed_variant(
    proto: asap_sketchlib::EHSketchList,
) -> (asap_sketchlib::EHSketchList, FreqTruth, String) {
    let keys = zipf_u64(N, DOMAIN, 1.1, STREAM_SEED);
    let mut eh = ExponentialHistogram::new(EH_K, EH_WINDOW, proto);
    for (t, k) in keys.iter().enumerate() {
        eh.update(t as u64, &DataInput::U64(*k));
    }
    let (lo, hi) = full_span(&eh);
    let merged = eh
        .query_interval_merge(lo, hi)
        .expect("full-span interval must be covered");
    let mut truth = FreqTruth::default();
    for t in lo..=hi {
        truth.observe(keys[t as usize] as i64);
    }
    let ctx = format!(
        "k={EH_K} window={EH_WINDOW} zipf(1.1) domain={DOMAIN} n={N} seed={STREAM_SEED:#x}, \
         retained span [{lo}, {hi}] over {} buckets",
        eh.payload.len()
    );
    (merged, truth, ctx)
}

#[test]
fn eh_count_min_variant_conforms_to_the_count_min_model_over_the_retained_window() {
    let (merged, truth, ctx) = run_keyed_variant(asap_sketchlib::EHSketchList::CM(CountMin::<
        Vector2D<i32>,
        FastPath,
    >::with_dimensions(
        ROWS, COLS
    )));
    CountMinSpec::new(ROWS, COLS).assert_contract(
        "EHSketchList::CM",
        &truth,
        |k| merged.query(&DataInput::U64(k as u64)).expect("CM query"),
        &ctx,
    );
}

#[test]
fn eh_count_sketch_variant_conforms_to_the_l2_model_over_the_retained_window() {
    let (merged, truth, ctx) = run_keyed_variant(asap_sketchlib::EHSketchList::CS(Count::<
        Vector2D<i32>,
        FastPath,
    >::with_dimensions(
        ROWS, COLS
    )));
    CountSketchSpec::new(ROWS, COLS).assert_contract(
        "EHSketchList::CS",
        &truth,
        |k| merged.query(&DataInput::U64(k as u64)).expect("CS query"),
        &ctx,
    );
}

#[test]
fn eh_countl2hh_variant_satisfies_the_l2_bound_over_the_retained_window() {
    let (merged, truth, ctx) = run_keyed_variant(asap_sketchlib::EHSketchList::COUNTL2HH(
        CountL2HH::with_dimensions(ROWS, COLS),
    ));
    CountSketchSpec::new(ROWS, COLS).assert_contract(
        "EHSketchList::COUNTL2HH",
        &truth,
        |k| merged.query(&DataInput::U64(k as u64)).expect("L2HH query"),
        &ctx,
    );
}

/// The heavy-hitter payloads keep a flow key beside each counter and evict on
/// pressure, so their guarantee is one-sided on the keys they retain: a
/// reported count never reads below the truth. Their full error sandwiches are
/// covered in `e2e_heavy_hitters.rs`; what is new here is that the guarantee
/// survives EH bucket merging.
#[test]
fn eh_heavy_hitter_variants_stay_one_sided_over_the_retained_window() {
    for (name, proto) in [
        (
            "COCO",
            asap_sketchlib::EHSketchList::COCO(Coco::init_with_size(1024, 4)),
        ),
        (
            "ELASTIC",
            asap_sketchlib::EHSketchList::ELASTIC(Elastic::init_with_length(1024)),
        ),
    ] {
        // Both payloads key on strings, so the stream is fed as strings and
        // the truth is keyed by the same identity.
        let keys = zipf_u64(N, DOMAIN, 1.1, STREAM_SEED);
        let mut eh = ExponentialHistogram::new(EH_K, EH_WINDOW, proto);
        for (t, k) in keys.iter().enumerate() {
            eh.update(t as u64, &DataInput::String(format!("f{k}")));
        }
        let (lo, hi) = full_span(&eh);
        let merged = eh.query_interval_merge(lo, hi).expect("full span");
        let mut truth = FreqTruth::default();
        for t in lo..=hi {
            truth.observe(keys[t as usize] as i64);
        }
        let ctx = format!(
            "{name} k={EH_K} zipf(1.1) domain={DOMAIN} n={N} seed={STREAM_SEED:#x}, \
             retained span [{lo}, {hi}] over {} buckets",
            eh.payload.len()
        );

        // Only the truly heavy keys are guaranteed to be retained; the
        // one-sided property is asserted over those.
        let mut tally = Tally::default();
        for (k, c) in truth.top_k(32) {
            let est = merged
                .query(&DataInput::String(format!("f{k}")))
                .expect("heavy-hitter query");
            tally.record(est >= c as f64, || {
                format!("key f{k}: true {c}, reported {est} (must never read low)")
            });
        }
        tally.assert_none(
            &format!("EHSketchList::{name} one-sided on heavy keys"),
            &ctx,
        );
    }
}

#[test]
fn eh_hll_variant_satisfies_the_register_error_model_over_the_retained_window() {
    let keys = uniform_u64(N, 200_000, STREAM_SEED);
    let mut eh = ExponentialHistogram::new(
        EH_K,
        EH_WINDOW,
        asap_sketchlib::EHSketchList::HLL(HyperLogLog::<ErtlMLE>::new()),
    );
    for (t, k) in keys.iter().enumerate() {
        eh.update(t as u64, &DataInput::U64(*k));
    }
    let (lo, hi) = full_span(&eh);
    let merged = eh.query_interval_merge(lo, hi).expect("full span");
    let distinct: std::collections::HashSet<u64> = (lo..=hi).map(|t| keys[t as usize]).collect();

    // `HyperLogLog<ErtlMLE>` here is the p14 default: m = 2^14 registers.
    let spec = CardinalityConfidenceSpec::hll(14, 4.0);
    let mut tally = Tally::default();
    spec.tally_into(
        &mut tally,
        merged.query(&DataInput::Str("card")).expect("HLL query"),
        distinct.len(),
    );
    // One merged HLL, one estimate: a single independent trial. The binomial
    // acceptance rule at n = 1 is exactly "this must pass", which is the honest
    // reading of a one-draw experiment.
    tally.assert_independent_binomial(
        "EHSketchList::HLL / register error model",
        spec.per_check_failure(),
        &format!(
            "k={EH_K} uniform n={N} domain=200000 seed={STREAM_SEED:#x}, retained span \
             [{lo}, {hi}] with {} distinct, tolerance={:.5}",
            distinct.len(),
            spec.tolerance()
        ),
    );
}

#[test]
fn eh_kll_variant_satisfies_the_rank_error_characterization_over_the_retained_window() {
    const KLL_K: i32 = 200;
    let values: Vec<f64> = uniform_u64(N, 1_000_000, STREAM_SEED)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let mut eh = ExponentialHistogram::new(
        EH_K,
        EH_WINDOW,
        asap_sketchlib::EHSketchList::KLL(KLL::init_kll_with_seed(KLL_K, 0x5EED_0200)),
    );
    for (t, v) in values.iter().enumerate() {
        eh.update(t as u64, &DataInput::F64(*v));
    }
    let (lo, hi) = full_span(&eh);
    let truth = NumericTruth::new((lo..=hi).map(|t| values[t as usize]).collect());

    // Twelve independent compaction seeds, one trial each: the five quantiles
    // of one merged KLL share a compaction history and are not five Bernoulli
    // draws, so each seed is reduced to its worst rank error first.
    const QS: [f64; 5] = [0.1, 0.25, 0.5, 0.75, 0.9];
    const TRIALS: u64 = 12;
    let spec = KllRankSpec::datasketches(KLL_K as usize);
    let mut tally = Tally::default();
    for t in 0..TRIALS {
        let seed = 0x5EED_0200u64.wrapping_add(t.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut trial_eh = ExponentialHistogram::new(
            EH_K,
            EH_WINDOW,
            asap_sketchlib::EHSketchList::KLL(KLL::init_kll_with_seed(KLL_K, seed)),
        );
        for (t, v) in values.iter().enumerate() {
            trial_eh.update(t as u64, &DataInput::F64(*v));
        }
        let (tlo, thi) = full_span(&trial_eh);
        let trial_merged = trial_eh.query_interval_merge(tlo, thi).expect("full span");
        let trial_truth = NumericTruth::new((tlo..=thi).map(|t| values[t as usize]).collect());
        // The KLL payload answers `query(q)` as `quantile(q)`.
        spec.record_trial(
            &mut tally,
            &format!("EHSketchList::KLL seed={seed:#x} span=[{tlo}, {thi}]"),
            trial_truth.sorted(),
            &QS,
            |q| trial_merged.query(&DataInput::F64(q)).expect("KLL query"),
        );
    }
    tally.assert_independent_binomial(
        "EHSketchList::KLL / maximum normalized rank error per compaction seed",
        spec.trial_failure_probability,
        &format!(
            "k={EH_K} kll_k={KLL_K} uniform n={N} seed={STREAM_SEED:#x}, retained span \
             [{lo}, {hi}] with {} observations, {TRIALS} independent compaction seeds, \
             q grid {QS:?}",
            truth.len()
        ),
    );
}

#[test]
fn eh_ddsketch_variant_satisfies_the_relative_value_error_contract_over_the_window() {
    const ALPHA: f64 = 0.01;
    let values: Vec<f64> = uniform_u64(N, 9_000_000, STREAM_SEED)
        .into_iter()
        .map(|v| 1_000_000.0 + v as f64)
        .collect();
    let mut eh = ExponentialHistogram::new(
        EH_K,
        EH_WINDOW,
        asap_sketchlib::EHSketchList::DDS(DDSketch::new(ALPHA)),
    );
    for (t, v) in values.iter().enumerate() {
        eh.update(t as u64, &DataInput::F64(*v));
    }
    let (lo, hi) = full_span(&eh);
    let merged = eh.query_interval_merge(lo, hi).expect("full span");
    let truth = NumericTruth::new((lo..=hi).map(|t| values[t as usize]).collect());

    // The DDS payload answers `query(q)` as `get_value_at_quantile(q)`.
    let spec = RelativeQuantileSpec::core(ALPHA);
    let mut tally = Tally::default();
    spec.tally_into(
        &mut tally,
        truth.sorted(),
        &[0.1, 0.25, 0.5, 0.75, 0.9],
        |q| merged.query(&DataInput::F64(q)).ok(),
    );
    tally.assert_none(
        "EHSketchList::DDS / relative value error",
        &format!(
            "alpha={ALPHA} k={EH_K} uniform n={N} seed={STREAM_SEED:#x}, retained span \
             [{lo}, {hi}] with {} observations",
            truth.len()
        ),
    );
    // Count is maintained, so the merged payload holds the whole window.
    assert_eq!(
        merged.query(&DataInput::Str("count")).expect("count"),
        truth.len() as f64,
        "DDS payload must retain every observation in the merged span"
    );
}

#[test]
fn eh_univmon_variant_reports_the_exact_l1_over_the_retained_window() {
    let keys = zipf_u64(N, DOMAIN, 1.1, STREAM_SEED);
    let mut eh = ExponentialHistogram::new(
        EH_K,
        EH_WINDOW,
        asap_sketchlib::EHSketchList::UNIVMON(UnivMon::init_univmon(
            32,
            UNIVMON_ROWS,
            UNIVMON_COLS,
            8,
        )),
    );
    for (t, k) in keys.iter().enumerate() {
        eh.update(t as u64, &DataInput::U64(*k));
    }
    let (lo, hi) = full_span(&eh);
    let merged = eh.query_interval_merge(lo, hi).expect("full span");
    let mut truth = FreqTruth::default();
    for t in lo..=hi {
        truth.observe(keys[t as usize] as i64);
    }
    let ctx = format!(
        "k={EH_K} zipf(1.1) domain={DOMAIN} n={N} seed={STREAM_SEED:#x}, retained span [{lo}, {hi}]"
    );

    // L1 is the maintained bucket size: exact, not estimated.
    assert_eq!(
        merged.query(&DataInput::Str("l1")).expect("l1"),
        truth.total() as f64,
        "UnivMon L1 over the merged span must be exact. {ctx}"
    );

    // L2 through UnivMon's recursive g-sum, held to the AMS second-moment
    // bound of the layer the answer comes from rather than to a written
    // percentage.
    //
    // `l2` is `sqrt(F2_hat)` where `F2_hat` is the row-median AMS estimate over
    // the sketch's own 5x2048 counters, so `SecondMomentSpec`'s relative bound
    // `b = sqrt(2*kappa/w)` on F2 becomes `[sqrt(1-b), sqrt(1+b)]` on the norm.
    // At 5x2048 that is about -4.2%/+4.1%, tied to the configuration: halving
    // `cols` widens the band automatically.
    //
    // This is the *terminal* layer's bound. UnivMon's recurrence composes
    // per-layer estimates, and the crate publishes no closed form for the
    // composed constant, so what is asserted here is the bound of the estimator
    // the answer is actually read from — with the recurrence's own contribution
    // covered by the exact L1 equality above.
    let l2 = merged.query(&DataInput::Str("l2")).expect("l2");
    let l2_truth = truth.l2_norm();
    let b = common::specs::SecondMomentSpec::new(UNIVMON_ROWS, UNIVMON_COLS).relative_bound();
    let (band_lo, band_hi) = (
        l2_truth * (1.0 - b).max(0.0).sqrt(),
        l2_truth * (1.0 + b).sqrt(),
    );
    assert!(
        l2 >= band_lo && l2 <= band_hi,
        "UnivMon L2 {l2:.1} vs exact {l2_truth:.1} outside [{band_lo:.1}, {band_hi:.1}], the \
         AMS second-moment bound sqrt(2*kappa/w)={b:.5} at {UNIVMON_ROWS}x{UNIVMON_COLS} \
         carried through the square root. {ctx}"
    );
}

// ---------------------------------------------------------- Window semantics

/// Expiry and interval queries, checked on the payload every variant shares.
/// Bucket bookkeeping is arithmetic, so these are equalities.
#[test]
fn eh_expires_buckets_past_the_window_and_reports_its_retained_span() {
    const WINDOW: u64 = 100;
    let mut eh = ExponentialHistogram::new(
        EH_K,
        WINDOW,
        asap_sketchlib::EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
            ROWS, COLS,
        )),
    );
    for t in 0..500u64 {
        if t % 3 == 0 {
            eh.update(t, &DataInput::Str("req"));
        }
    }

    let retained_min = eh.get_min_time().expect("buckets present");
    let retained_max = eh.get_max_time().expect("buckets present");
    assert_eq!(retained_max, 498, "newest retained timestamp");
    assert!(eh.cover(retained_min, 498), "must cover its retained span");
    assert!(!eh.cover(500, 600), "cannot cover beyond the observed span");

    // Every retained bucket must reach at least to the cutoff.
    let cutoff = retained_max.saturating_sub(WINDOW);
    for b in &eh.payload {
        assert!(
            b.max_time >= cutoff,
            "bucket [{}, {}] lies entirely before the cutoff {cutoff}",
            b.min_time,
            b.max_time
        );
    }

    // The full-span query merges every retained bucket, so its count is the
    // exact number of retained events — no bucket-granularity slack needed.
    let (lo, hi) = full_span(&eh);
    let merged = eh.query_interval_merge(lo, hi).expect("full span");
    let retained_events = (lo..=hi).filter(|t| t % 3 == 0).count();
    let est = merged.query(&DataInput::Str("req")).expect("CM query");
    assert!(
        est >= retained_events as f64,
        "Count-Min must not underestimate the retained event count: {est} < {retained_events}"
    );
}

// ------------------------------------------------------------ TumblingWindow

/// `TumblingWindow<FoldCS>`: window bookkeeping is exact, and the folded Count
/// Sketch inside each window carries the L2 bound at its *folded* width.
#[test]
fn tumbling_fold_cs_windows_are_exact_and_answers_satisfy_the_l2_bound() {
    const FULL_COLS: usize = 4_096;
    const FOLD_LEVEL: u32 = 2;
    const WINDOW: u64 = 500;
    const TOTAL: u64 = 3_000;

    let cfg = FoldCSConfig {
        rows: ROWS,
        full_cols: FULL_COLS,
        fold_level: FOLD_LEVEL,
        top_k: 32,
    };
    let mut tw: TumblingWindow<FoldCS> = TumblingWindow::new(WINDOW, 16, cfg, 4);
    let keys = zipf_u64(TOTAL as usize, 512, 1.1, STREAM_SEED);
    for (t, k) in keys.iter().enumerate() {
        tw.insert(t as u64, &DataInput::U64(*k), 1);
    }

    assert_eq!(
        tw.closed_count(),
        (TOTAL / WINDOW - 1) as usize,
        "windows [0, {}) must be closed at t={}",
        TOTAL - WINDOW,
        TOTAL - 1
    );

    let folded_cols = FULL_COLS >> FOLD_LEVEL;
    let spec = CountSketchSpec::new(ROWS, folded_cols);

    // The active window is the exact last-`WINDOW` slice.
    let mut active_truth = FreqTruth::default();
    for k in &keys[(TOTAL - WINDOW) as usize..] {
        active_truth.observe(*k as i64);
    }
    let active = tw.active_sketch();
    spec.assert_contract(
        "TumblingWindow<FoldCS> active window",
        &active_truth,
        |k| active.query(&DataInput::U64(k as u64)) as f64,
        &format!(
            "rows={ROWS} full_cols={FULL_COLS} fold_level={FOLD_LEVEL} -> {folded_cols} cols, \
             window={WINDOW} zipf(1.1) domain=512 seed={STREAM_SEED:#x}"
        ),
    );

    // `query_all` covers every observation.
    let mut all_truth = FreqTruth::default();
    for k in &keys {
        all_truth.observe(*k as i64);
    }
    let all = tw.query_all();
    spec.assert_contract(
        "TumblingWindow<FoldCS> query_all",
        &all_truth,
        |k| all.query(&DataInput::U64(k as u64)) as f64,
        &format!("all {TOTAL} observations across {} windows", TOTAL / WINDOW),
    );

    // `query_recent(2)` is the active window plus the two most recent closed
    // ones — an exact time slice.
    let mut recent_truth = FreqTruth::default();
    for k in &keys[(TOTAL - 3 * WINDOW) as usize..] {
        recent_truth.observe(*k as i64);
    }
    let recent = tw.query_recent(2);
    spec.assert_contract(
        "TumblingWindow<FoldCS> query_recent(2)",
        &recent_truth,
        |k| recent.query(&DataInput::U64(k as u64)) as f64,
        "last three windows",
    );

    // Rotation and pool reuse: flushing then inserting again must produce a
    // clean active window, not one carrying the previous window's counters.
    tw.flush(TOTAL);
    let closed_after_flush = tw.closed_count();
    tw.insert(TOTAL, &DataInput::U64(7), 1);
    assert_eq!(
        tw.active_sketch().query(&DataInput::U64(7)),
        1,
        "a recycled window sketch must start empty; got a carried-over count"
    );
    assert!(
        closed_after_flush >= (TOTAL / WINDOW) as usize,
        "flush must close the active window"
    );
}

/// `TumblingWindow<UnivMonQ>`: the same window bookkeeping, with UnivMon-Q's
/// exact aggregates checked exactly and its estimates left to their own
/// suite. What this test adds is that windowing, pooling and `clear()` do not
/// corrupt the sketch.
#[test]
fn tumbling_univmon_q_windows_carry_exact_aggregates_through_rotation() {
    const WINDOW: u64 = 500;
    const TOTAL: u64 = 3_000;

    let cfg = UnivMonQConfig {
        levels: 6,
        width: 1_024,
        depth: 5,
        candidates: 256,
        ordered_samples: 512,
        ..UnivMonQConfig::default()
    };
    let mut tw: TumblingWindow<UnivMonQ> = TumblingWindow::new(WINDOW, 16, cfg, 4);
    let values: Vec<f64> = uniform_u64(TOTAL as usize, 100_000, STREAM_SEED)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    for (t, v) in values.iter().enumerate() {
        tw.insert(t as u64, &DataInput::F64(*v), 0);
    }

    assert_eq!(
        tw.closed_count(),
        (TOTAL / WINDOW - 1) as usize,
        "closed window count"
    );

    // Count, min and max are maintained exactly on every window slice.
    for (label, sketch, slice) in [
        ("query_all", tw.query_all(), &values[..]),
        (
            "query_recent(1)",
            tw.query_recent(1),
            &values[(TOTAL - 2 * WINDOW) as usize..],
        ),
        (
            "active_sketch",
            tw.active_sketch().clone(),
            &values[(TOTAL - WINDOW) as usize..],
        ),
    ] {
        let truth = NumericTruth::new(slice.to_vec());
        assert_eq!(
            sketch.count() as usize,
            slice.len(),
            "{label}: UnivMon-Q count is maintained exactly"
        );
        assert_eq!(sketch.min(), Some(truth.min()), "{label}: exact minimum");
        assert_eq!(sketch.max(), Some(truth.max()), "{label}: exact maximum");
    }

    // Rotation through the pool must hand back a cleared sketch.
    tw.flush(TOTAL);
    tw.insert(TOTAL, &DataInput::F64(42.0), 0);
    let active = tw.active_sketch();
    assert_eq!(
        active.count(),
        1,
        "a recycled UnivMon-Q must start empty after clear()"
    );
    assert_eq!(active.min(), Some(42.0), "recycled sketch min");
    assert_eq!(active.max(), Some(42.0), "recycled sketch max");
}

/// Every payload variant that owns a mergeable sketch must have a merge arm in
/// `EHSketchList::merge`. A missing arm is invisible at runtime — `EHBucket::to_merge`
/// discards the `Result` — so the histogram keeps counting bucket sizes while
/// silently dropping the merged sketch's contents, and every query afterwards
/// reads a single bucket. `ELASTIC` shipped without its arm and lost the whole
/// window; this pins every variant so the next one cannot.
#[test]
fn every_eh_variant_can_merge_into_its_own_kind() {
    use asap_sketchlib::EHSketchList;

    let variants: Vec<(&str, EHSketchList)> = vec![
        (
            "CM",
            EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 256)),
        ),
        (
            "CS",
            EHSketchList::CS(Count::<Vector2D<i32>, FastPath>::with_dimensions(3, 256)),
        ),
        ("COCO", EHSketchList::COCO(Coco::init_with_size(256, 4))),
        (
            "COUNTL2HH",
            EHSketchList::COUNTL2HH(CountL2HH::with_dimensions(3, 256)),
        ),
        ("DDS", EHSketchList::DDS(DDSketch::new(0.01))),
        (
            "ELASTIC",
            EHSketchList::ELASTIC(Elastic::init_with_length(256)),
        ),
        ("HLL", EHSketchList::HLL(HyperLogLog::<ErtlMLE>::new())),
        (
            "KLL",
            EHSketchList::KLL(KLL::init_kll_with_seed(200, 0x5EED_0300)),
        ),
        (
            "UNIVMON",
            EHSketchList::UNIVMON(UnivMon::init_univmon(32, 3, 256, 4)),
        ),
        #[cfg(feature = "experimental")]
        (
            "UNIFORM",
            EHSketchList::UNIFORM(asap_sketchlib::UniformSampling::with_seed(0.5, 7)),
        ),
    ];

    for (name, proto) in variants {
        let mut left = proto.clone();
        let mut right = proto.clone();
        // Feed each side something it can actually ingest, then merge.
        for i in 0..64u64 {
            left.insert(&DataInput::String(format!("k{i}")));
            left.insert(&DataInput::F64(1.0 + i as f64));
            right.insert(&DataInput::String(format!("k{}", i + 64)));
            right.insert(&DataInput::F64(100.0 + i as f64));
        }
        assert!(
            left.merge(&right).is_ok(),
            "EHSketchList::{name} has no merge arm, so an ExponentialHistogram over it \
             silently discards data on every bucket merge"
        );
    }
}

/// `UNIFORM` is the experimental reservoir payload: its window answers are
/// retained-sample bookkeeping, not an estimate, so they are checked exactly.
#[cfg(feature = "experimental")]
#[test]
fn eh_uniform_sampling_variant_reports_exact_retention_bookkeeping() {
    use asap_sketchlib::UniformSampling;

    const RATE: f64 = 0.1;
    const SAMPLER_SEED: u64 = 0x5A_9101;
    let values: Vec<f64> = uniform_u64(N, 1_000_000, STREAM_SEED)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let mut eh = ExponentialHistogram::new(
        EH_K,
        EH_WINDOW,
        asap_sketchlib::EHSketchList::UNIFORM(UniformSampling::with_seed(RATE, SAMPLER_SEED)),
    );
    for (t, v) in values.iter().enumerate() {
        eh.update(t as u64, &DataInput::F64(*v));
    }
    let (lo, hi) = full_span(&eh);
    let merged = eh.query_interval_merge(lo, hi).expect("full span");

    // `total_seen` is a maintained counter: exact over the merged span.
    let seen = merged
        .query(&DataInput::Str("total_seen"))
        .expect("total_seen");
    assert_eq!(
        seen,
        (hi - lo + 1) as f64,
        "UniformSampling total_seen must equal the number of merged observations \
         over [{lo}, {hi}]"
    );

    // Retained samples must be a subset of the window's observations and
    // bounded by the rate's budget.
    let retained = merged.query(&DataInput::Str("len")).expect("len") as usize;
    assert!(
        retained > 0 && retained as f64 <= seen,
        "retained {retained} samples out of {seen} observed"
    );
    let observed: std::collections::HashSet<u64> =
        (lo..=hi).map(|t| values[t as usize].to_bits()).collect();
    for i in 0..retained {
        let s = merged
            .query(&DataInput::U64(i as u64))
            .expect("sample index in range");
        assert!(
            observed.contains(&s.to_bits()),
            "retained sample {s} was never observed in the merged window [{lo}, {hi}]"
        );
    }
}

#[test]
fn tumbling_fold_cms_hierarchical_merge_covers_every_observation() {
    const FULL_COLS: usize = 4_096;
    const FOLD_LEVEL: u32 = 2;
    const WINDOW: u64 = 500;
    const TOTAL: u64 = 3_000;

    let cfg = FoldCMSConfig {
        rows: ROWS,
        full_cols: FULL_COLS,
        fold_level: FOLD_LEVEL,
        top_k: 32,
    };
    let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(WINDOW, 16, cfg, 4);
    let keys = zipf_u64(TOTAL as usize, 512, 1.1, STREAM_SEED);
    let mut truth = FreqTruth::default();
    for (t, k) in keys.iter().enumerate() {
        tw.insert(t as u64, &DataInput::U64(*k), 1);
        truth.observe(*k as i64);
    }

    let hierarchical = tw.query_all_hierarchical();
    let flat = tw.query_all();
    assert!(
        hierarchical.fold_cols() >= flat.fold_cols(),
        "a hierarchical merge must not narrow the sketch: {} < {}",
        hierarchical.fold_cols(),
        flat.fold_cols()
    );

    let spec = CountMinSpec::new(ROWS, hierarchical.fold_cols());
    spec.assert_contract(
        "TumblingWindow<FoldCMS> query_all_hierarchical",
        &truth,
        |k| hierarchical.query(&DataInput::U64(k as u64)) as f64,
        &format!(
            "rows={ROWS} full_cols={FULL_COLS} fold_level={FOLD_LEVEL} -> {} cols after \
             hierarchical merge, window={WINDOW}, {TOTAL} observations",
            hierarchical.fold_cols()
        ),
    );

    for (key, count) in truth.pairs() {
        let probe = DataInput::U64(key as u64);
        let h = hierarchical.query(&probe);
        let f = flat.query(&probe);
        assert!(
            h >= count,
            "key {key}: hierarchical merge underestimated {count} as {h}"
        );
        assert!(
            h <= f,
            "key {key}: unfolding raised the estimate from {f} to {h}"
        );
    }
    assert!(
        hierarchical.total_entries() > 0,
        "a hierarchical merge over a populated stream must retain entries"
    );
}

#[test]
fn tumbling_window_pool_accounting_tracks_every_recycled_sketch() {
    const WINDOW: u64 = 100;
    const MAX_WINDOWS: usize = 2;
    const POOL_CAP: usize = 4;

    let cfg = KLLConfig {
        k: 200,
        m: 8,
        seed: Some(0x7001_0001),
    };
    let mut tw: TumblingWindow<KLL> = TumblingWindow::new(WINDOW, MAX_WINDOWS, cfg, POOL_CAP);
    assert_eq!(
        tw.pool_total_allocated(),
        POOL_CAP,
        "the pool must pre-allocate exactly its capacity"
    );
    assert_eq!(
        tw.pool_available(),
        POOL_CAP - 1,
        "the initial active window must come out of the pool"
    );

    for t in 0..(WINDOW * 8) {
        tw.insert(t, &DataInput::U64(t % 97), 1);
    }
    assert_eq!(
        tw.closed_count(),
        MAX_WINDOWS,
        "at most {MAX_WINDOWS} closed windows are retained"
    );
    assert!(
        tw.pool_available() > 0,
        "evicted windows must return their sketches to the pool"
    );
    assert!(
        tw.pool_total_allocated() >= POOL_CAP,
        "the pool must never report fewer allocations than it started with"
    );
    assert_eq!(
        tw.pool_available() + tw.closed_count() + 1,
        tw.pool_total_allocated(),
        "every allocated sketch is either pooled, in a retained window, or active"
    );
}

#[test]
fn a_sketch_pool_reuses_a_returned_sketch_before_allocating_a_new_one() {
    const CAP: usize = 2;
    let cfg = KLLConfig {
        k: 200,
        m: 8,
        seed: Some(0x7001_0002),
    };
    let mut pool: SketchPool<KLL> = SketchPool::new(CAP, cfg);
    assert_eq!(pool.available(), CAP);
    assert_eq!(pool.total_allocated(), CAP);

    let first = pool.take();
    let second = pool.take();
    assert_eq!(pool.available(), 0);
    assert_eq!(pool.total_allocated(), CAP);

    let third = pool.take();
    assert_eq!(
        pool.total_allocated(),
        CAP + 1,
        "an empty pool must allocate rather than block"
    );

    let mut dirty = first;
    for v in 0..1_000u64 {
        dirty.update(&(v as f64));
    }
    pool.put(dirty);
    assert_eq!(pool.available(), 1, "a returned sketch is available again");

    let recycled = pool.take();
    assert_eq!(
        recycled.count(),
        0,
        "a recycled sketch must come back cleared"
    );
    assert_eq!(
        pool.total_allocated(),
        CAP + 1,
        "taking a recycled sketch must not allocate"
    );
    drop((second, third, recycled));
}

#[test]
fn an_exponential_histogram_expires_against_the_window_length_it_was_last_given() {
    let proto = EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
        ROWS, COLS,
    ));
    let mut eh = ExponentialHistogram::new(EH_K, 10_000, proto);
    for t in 0..200u64 {
        eh.update(t * 10, &DataInput::U64(t % 32));
    }
    let wide_span = full_span(&eh);
    assert_eq!(
        wide_span.0, 0,
        "nothing may expire while the window covers the whole stream"
    );

    eh.update_window(100);
    eh.update(2_000, &DataInput::U64(0));
    let narrow_span = full_span(&eh);
    assert!(
        narrow_span.0 > wide_span.0,
        "shortening the window must drop the oldest buckets, span still starts at {}",
        narrow_span.0
    );
    assert!(
        eh.get_max_time() == Some(2_000),
        "the newest bucket must carry the timestamp just inserted"
    );

    eh.update_window(10_000);
    eh.update(2_010, &DataInput::U64(0));
    assert!(
        full_span(&eh).0 >= narrow_span.0,
        "widening the window cannot resurrect an expired bucket"
    );
}

#[test]
fn an_exponential_histogram_custom_bucket_update_matches_repeated_inserts() {
    let proto = EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
        ROWS, COLS,
    ));
    let mut by_update = ExponentialHistogram::new(EH_K, EH_WINDOW, proto.clone());
    let mut by_custom = ExponentialHistogram::new(EH_K, EH_WINDOW, proto);

    for t in 0..500u64 {
        let key = DataInput::U64(t % 64);
        by_update.update(t, &key);
        by_custom.update_with(t, |sketch| {
            sketch.insert(&key);
        });
    }

    assert_eq!(
        by_update.bucket_count(),
        by_custom.bucket_count(),
        "the custom updater must produce the same bucket structure"
    );
    let (lo, hi) = full_span(&by_update);
    let merged_update = by_update
        .query_interval_merge(lo, hi)
        .expect("update path merges its own span");
    let merged_custom = by_custom
        .query_interval_merge(lo, hi)
        .expect("custom path merges its own span");
    for k in 0..64u64 {
        let probe = DataInput::U64(k);
        assert_eq!(
            merged_update.query(&probe).expect("count-min answers"),
            merged_custom.query(&probe).expect("count-min answers"),
            "key {k}: the custom updater diverged from update"
        );
    }

    let mut doubled = ExponentialHistogram::new(
        EH_K,
        EH_WINDOW,
        EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
            ROWS, COLS,
        )),
    );
    for t in 0..500u64 {
        let key = DataInput::U64(t % 64);
        doubled.update_with(t, |sketch| {
            sketch.insert(&key);
            sketch.insert(&key);
        });
    }
    let (dlo, dhi) = full_span(&doubled);
    let merged_doubled = doubled
        .query_interval_merge(dlo, dhi)
        .expect("doubled path merges its own span");
    for k in 0..64u64 {
        let probe = DataInput::U64(k);
        assert!(
            merged_doubled.query(&probe).expect("count-min answers")
                >= merged_update.query(&probe).expect("count-min answers"),
            "key {k}: a bucket updated twice must not read below one updated once"
        );
    }
}

// ---------------------------------------------------------------------------
// The documented input matrix
// ---------------------------------------------------------------------------

/// `tests/TEST_COVERAGE.md` gives the sliding-window histogram its own row:
/// `k = 8`, `window 100`, a Count-Min payload at `row 3, col 2048` on the fast
/// path, over inputs `(3) ~ (6)` and `(14)`, with an interval count held to 21%
/// relative error and the retained span reported honestly.
///
/// The variant matrix above runs at `window 1,000,000` so that nothing expires
/// inside the accuracy runs; this row is the opposite case, where expiry is the
/// point. Each stream is laid over 1000 time units, so a window of 100 retains
/// the last tenth of it and the rest has to be gone.
///
/// # Why the interval count is two-sided
///
/// `query_interval_merge` snaps the requested interval to bucket boundaries in
/// both directions, so the merged payload can cover slightly more or slightly
/// less than what was asked for. The Count-Min payload can only over-count
/// within whatever it does cover, but the snapping can drop events the
/// requested interval contained, so the answer is not one-sided and the 21% is
/// a two-sided granularity band rather than a sketch bound.
mod documented_matrix {
    use super::common::inputs::{key_input, string_input};
    use super::common::specs::SIMULTANEOUS_LEVEL;
    use super::*;

    use std::collections::HashMap;
    use std::hash::Hash;

    /// The document's configuration for this row.
    const DOC_K: usize = 8;
    const DOC_WINDOW: u64 = 100;
    const DOC_ROWS: usize = 3;
    const DOC_COLS: usize = 2_048;
    /// Time units the stream is spread over, so that `DOC_WINDOW` retains a
    /// tenth of it.
    const SPAN: u64 = 1_000;
    /// The document's interval-count band.
    const INTERVAL_RELATIVE_ERROR: f64 = 0.21;

    fn documented_histogram() -> ExponentialHistogram {
        ExponentialHistogram::new(
            DOC_K,
            DOC_WINDOW,
            EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
                DOC_ROWS, DOC_COLS,
            )),
        )
    }

    /// Feeds a keyed stream through the documented histogram and checks the
    /// interval count, the retained span and expiry past it.
    fn eh_documented_stream<K, F>(keys: &[K], to_input: F, context: &str)
    where
        K: Eq + Hash + Clone + std::fmt::Debug,
        F: Fn(&K) -> DataInput<'static>,
    {
        let n = keys.len() as u64;
        let mut eh = documented_histogram();
        let mut at: Vec<(u64, K)> = Vec::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            let t = i as u64 * SPAN / n;
            eh.update(t, &to_input(key));
            at.push((t, key.clone()));
        }

        let min_time = eh.get_min_time().expect("buckets present");
        let max_time = eh.get_max_time().expect("buckets present");
        assert_eq!(
            max_time,
            at.last().expect("non-empty stream").0,
            "the histogram must report the last timestamp it saw. {context}"
        );
        assert!(
            eh.cover(min_time, max_time),
            "the retained span must be covered. {context}"
        );
        assert!(
            !eh.cover(max_time + 1, max_time + SPAN),
            "nothing past the observed maximum can be covered. {context}"
        );
        // Expiry is by bucket, not by timestamp: the oldest retained bucket may
        // have started before the window boundary and is kept whole, so the
        // retained span overhangs the window by that bucket's own span
        // (measured: 120 units for a window of 100). What must not happen is
        // the span growing to a multiple of the window, which is what an
        // expiry that never fired would look like.
        assert!(
            min_time + 2 * DOC_WINDOW >= max_time,
            "the retained span [{min_time}, {max_time}] is more than twice the \
             {DOC_WINDOW} window. {context}"
        );

        // Everything the retained buckets can hold is a subset of the events at
        // or after `min_time` — a timestamp on the expiry boundary can have part
        // of its events in an expired bucket and part in a retained one, so this
        // is a superset rather than the exact window contents, and it is used
        // only where a superset is the right side of the inequality.
        let mut retained: HashMap<K, i64> = HashMap::new();
        for (t, key) in &at {
            if *t >= min_time {
                *retained.entry(key.clone()).or_insert(0) += 1;
            }
        }
        let retained_mass: i64 = retained.values().sum();

        // The payload's own bound over the full retained span: whatever subset
        // of these events the buckets hold, a Count-Min over them cannot exceed
        // this key's count in the superset by more than the additive budget.
        let merged_span = eh
            .query_interval_merge(min_time, max_time)
            .expect("the retained span is covered");
        let spec = CountMinSpec::new(DOC_ROWS, DOC_COLS);
        let probed = retained.len();
        let mut over = Tally::default();
        for (key, count) in &retained {
            let est = merged_span
                .query(&to_input(key))
                .expect("the Count-Min payload answers a frequency query");
            let f = *count as f64;
            let budget =
                spec.simultaneous_bound(retained_mass as f64, f, probed, SIMULTANEOUS_LEVEL);
            over.record(est - f <= budget, || {
                format!(
                    "key {key:?}: the merged span reads {est} against at most {f} retained \
                     events, an excess of {:.1} past the additive budget {budget:.1}",
                    est - f
                )
            });
        }
        over.assert_none(
            "EH Count-Min payload / additive budget over the retained span",
            &format!("{context}; retained {retained_mass} events over {probed} keys"),
        );

        // An interval strictly inside the retained span, so the query is about
        // bucket granularity rather than about expiry.
        let lo = min_time + (max_time - min_time) / 4;
        let hi = max_time;
        let mut truth: HashMap<K, i64> = HashMap::new();
        for (t, key) in &at {
            if *t >= lo && *t <= hi {
                *truth.entry(key.clone()).or_insert(0) += 1;
            }
        }
        let (heaviest, count) = truth
            .iter()
            .max_by_key(|(_, c)| **c)
            .expect("the interval carries events");

        let merged = eh
            .query_interval_merge(lo, hi)
            .expect("an interval inside the retained span is covered");
        let est = merged
            .query(&to_input(heaviest))
            .expect("the Count-Min payload answers a frequency query");
        let truth_count = *count as f64;
        let rel = ((est - truth_count) / truth_count).abs();
        assert!(
            rel <= INTERVAL_RELATIVE_ERROR,
            "interval [{lo}, {hi}] count for {heaviest:?}: {est} against the exact \
             {truth_count} is {:.2}% off, past the documented {:.0}%. {context}",
            rel * 100.0,
            INTERVAL_RELATIVE_ERROR * 100.0
        );
    }

    fn eh_documented_key_input(id: u8) {
        let input = key_input(id);
        let context = format!(
            "{} k={DOC_K} window={DOC_WINDOW} payload CountMin {DOC_ROWS}x{DOC_COLS} FastPath, \
             stream spread over {SPAN} time units",
            input.context()
        );
        eh_documented_stream(&input.keys, |k| input.data(*k), &context);
    }

    fn eh_documented_string_input(id: u8) {
        let input = string_input(id);
        let context = format!(
            "{} k={DOC_K} window={DOC_WINDOW} payload CountMin {DOC_ROWS}x{DOC_COLS} FastPath, \
             stream spread over {SPAN} time units",
            input.context()
        );
        eh_documented_stream(
            &input.keys,
            |k: &String| DataInput::String(k.clone()),
            &context,
        );
    }

    macro_rules! documented_eh_inputs {
        ($($name:ident => $id:literal, $kind:ident;)*) => {
            $(
                #[test]
                fn $name() {
                    $kind($id);
                }
            )*
        };
    }

    documented_eh_inputs! {
        eh_input_3_interval_counts_and_expiry => 3, eh_documented_key_input;
        eh_input_4_interval_counts_and_expiry => 4, eh_documented_key_input;
        eh_input_5_interval_counts_and_expiry => 5, eh_documented_key_input;
        eh_input_6_interval_counts_and_expiry => 6, eh_documented_key_input;
        eh_input_14_interval_counts_and_expiry => 14, eh_documented_string_input;
    }
}
