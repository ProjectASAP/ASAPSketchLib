//! Every built-in `NumericalValue` type carried through the generic sketches:
//! `KLL<T>`, `KLLDynamic<T>` and `DDSketch::add<T>`.
//!
//! The library implements `NumericalValue` for fourteen concrete types —
//! `i8 i16 i32 i64 i128 isize`, `u8 u16 u32 u64 u128 usize`, `f32 f64` — and
//! each is a public instance a caller can reach. A suite that only ever
//! instantiates `f64` would not notice a type whose ordering or `to_f64`
//! projection was wrong for its own domain: `u128` overflowing an intermediate,
//! `i8` wrapping in a comparison, `f32` losing a tie.
//!
//! Both sketches keep their own error metric here — rank error for the KLL
//! family, relative value error for DDSketch. The type is what varies, not the
//! contract.
//!
//! ## The `to_f64` projection
//!
//! `NumericalValue::to_f64` is how a value reaches the sketch's arithmetic, and
//! for `i128`/`u128` (and `i64`/`u64` past `2^53`) that projection is itself
//! lossy — 64 and 128-bit integers have more distinct values than `f64` has
//! mantissa bits. Ground truth here is therefore taken over the **projected**
//! values, which is precisely what the sketch was handed. The precision limit
//! that belongs to the caller's type choice is covered separately and
//! explicitly by `the_f64_projection_is_exact_below_two_to_the_53`.

mod common;

use common::specs::{KllRankSpec, RelativeQuantileSpec, Tally};
use common::{NumericTruth, uniform_u64};

use asap_sketchlib::{DDSketch, KLL, KLLDynamic, NumericalValue};

/// KLL parameter used throughout; `eps(200) = 1.65%` at 99% confidence.
const K: i32 = 200;
/// Compaction-coin seeds. Fixed, and separate from the stream seed.
const SKETCH_SEEDS: [u64; 3] = [0x4E17_0001, 0x4E17_0002, 0x4E17_0003];

/// A fresh compaction seed per KLL trial, so no two sketches in a battery share
/// a coin sequence. See `tests/e2e_quantiles.rs` for why the rank batteries are
/// binomials over seeds rather than over quantiles.
fn kll_trial_seed(trial: u64) -> u64 {
    0x4E17_0000_0000_0001u64.wrapping_add(trial.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}
const STREAM_SEED: u64 = 0x57EA_0001;
const N: usize = 20_000;
const QS: [f64; 7] = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0];

/// A positive-valued stream of `$ty`, drawn from one seeded `u64` source and
/// folded into the type's own range so every type sees the same shape at its
/// own scale. Values stay strictly positive because DDSketch only tracks
/// positive reals, and the same stream feeds all three sketches.
macro_rules! typed_stream {
    ($ty:ty, $span:expr) => {{
        let span: u64 = $span;
        uniform_u64(N, span, STREAM_SEED)
            .into_iter()
            .map(|v| (v + 1) as $ty)
            .collect::<Vec<$ty>>()
    }};
}

/// Ground truth over the projected values — what the sketch actually saw.
fn projected_truth<T: NumericalValue>(values: &[T]) -> NumericTruth {
    NumericTruth::new(values.iter().map(|v| v.to_f64()).collect())
}

// --------------------------------------------------------------- KLL family

/// One numeric type through both KLL implementations, in three feed modes,
/// against the DataSketches maximum-rank-error characterization.
///
/// Each sketch is a **trial**: it gets its own compaction seed and is scored on
/// its worst rank error over the whole `q` grid, because that maximum is the
/// quantity the characterization's 1% is quoted for. Six trials per repeat per
/// type — `KLL` and `KLLDynamic` x {single pass, two-shard merge, bulk}.
macro_rules! kll_type_case {
    ($tally:ident, $trial:ident, $ty:ty, $span:expr) => {{
        let values = typed_stream!($ty, $span);
        let truth = projected_truth(&values);
        let spec = KllRankSpec::datasketches(K as usize);
        let type_name = stringify!($ty);

        macro_rules! trial {
            ($label:expr, $build:expr) => {{
                let seed = kll_trial_seed($trial);
                $trial += 1;
                let sketch = $build(seed);
                spec.record_trial(
                    &mut $tally,
                    &format!("{}<{type_name}> {} seed={seed:#x}", $label.0, $label.1),
                    truth.sorted(),
                    &QS,
                    |q| sketch.quantile(q),
                );
            }};
        }

        for _ in 0..SKETCH_SEEDS.len() {
            trial!(("KLL", "single pass"), |seed| {
                let mut s = KLL::<$ty>::init_kll_with_seed(K, seed);
                for v in &values {
                    s.update(v);
                }
                s
            });
            trial!(("KLLDynamic", "single pass"), |seed| {
                let mut s = KLLDynamic::<$ty>::init_kll_with_seed(K, seed);
                for v in &values {
                    s.update(v);
                }
                s
            });

            // Two shards merged must answer under the same characterization.
            trial!(("KLL", "two-shard merge"), |seed: u64| {
                let mut left = KLL::<$ty>::init_kll_with_seed(K, seed);
                let mut right = KLL::<$ty>::init_kll_with_seed(K, seed ^ 0x5555);
                for (i, v) in values.iter().enumerate() {
                    if i % 2 == 0 {
                        left.update(v);
                    } else {
                        right.update(v);
                    }
                }
                left.merge(&right);
                left
            });
            trial!(("KLLDynamic", "two-shard merge"), |seed: u64| {
                let mut left = KLLDynamic::<$ty>::init_kll_with_seed(K, seed);
                let mut right = KLLDynamic::<$ty>::init_kll_with_seed(K, seed ^ 0x5555);
                for (i, v) in values.iter().enumerate() {
                    if i % 2 == 0 {
                        left.update(v);
                    } else {
                        right.update(v);
                    }
                }
                left.merge(&right);
                left
            });

            // Bulk ingestion must be equivalent to the loop.
            trial!(("KLL", "bulk update"), |seed| {
                let mut s = KLL::<$ty>::init_kll_with_seed(K, seed);
                s.bulk_update(&values);
                s
            });
            trial!(("KLLDynamic", "bulk update"), |seed| {
                let mut s = KLLDynamic::<$ty>::init_kll_with_seed(K, seed);
                s.bulk_update(&values);
                s
            });
        }
    }};
}

/// Every built-in `NumericalValue` type through `KLL<T>` and `KLLDynamic<T>`.
///
/// The spans are the type's own usable range, so the narrow types genuinely
/// exercise the tie-dense case: an `i8` stream of 20,000 observations over 127
/// distinct values gives every value a rank interval nearly 1% wide, which the
/// rank-interval predicate handles and a value-error check could not.
#[test]
fn every_numeric_type_satisfies_the_kll_rank_error_characterization() {
    let mut tally = Tally::default();
    let mut trial = 0u64;
    kll_type_case!(tally, trial, i8, 127);
    kll_type_case!(tally, trial, i16, 32_767);
    kll_type_case!(tally, trial, i32, 1_000_000_000);
    kll_type_case!(tally, trial, i64, 1_000_000_000_000_000);
    kll_type_case!(tally, trial, i128, 1_000_000_000_000_000);
    kll_type_case!(tally, trial, isize, 1_000_000_000_000_000);
    kll_type_case!(tally, trial, u8, 255);
    kll_type_case!(tally, trial, u16, 65_535);
    kll_type_case!(tally, trial, u32, 4_000_000_000);
    kll_type_case!(tally, trial, u64, 1_000_000_000_000_000);
    kll_type_case!(tally, trial, u128, 1_000_000_000_000_000);
    kll_type_case!(tally, trial, usize, 1_000_000_000_000_000);
    kll_type_case!(tally, trial, f32, 1_000_000);
    kll_type_case!(tally, trial, f64, 1_000_000_000_000_000);

    tally.assert_independent_binomial(
        "KLL<T> / KLLDynamic<T> across every built-in NumericalValue type",
        KllRankSpec::datasketches(K as usize).trial_failure_probability,
        &format!(
            "one trial = one sketch with its own compaction seed, scored on its worst \
             rank error over the q grid. k={K}, seeds kll_trial_seed(0..), \
             stream_seed={STREAM_SEED:#x} n={N}, modes: single pass, bulk update, \
             two-shard merge; q grid {QS:?}"
        ),
    );
}

/// The KLL family must also order negative and mixed-sign values correctly.
/// The unsigned types cannot represent them, so this covers the signed and
/// floating types only — and it is where a `total_cmp` that fell back to a
/// bitwise comparison would show up.
#[test]
fn signed_numeric_types_order_negative_values_correctly_in_kll() {
    macro_rules! signed_case {
        ($tally:ident, $trial:ident, $ty:ty, $span:expr) => {{
            let span: i64 = $span;
            let values: Vec<$ty> = uniform_u64(N, (span * 2) as u64, STREAM_SEED)
                .into_iter()
                .map(|v| (v as i64 - span) as $ty)
                .collect();
            let truth = projected_truth(&values);
            let spec = KllRankSpec::datasketches(K as usize);
            let type_name = stringify!($ty);
            for _ in 0..SKETCH_SEEDS.len() {
                let seed = kll_trial_seed($trial);
                $trial += 1;
                let mut fixed = KLL::<$ty>::init_kll_with_seed(K, seed);
                for v in &values {
                    fixed.update(v);
                }
                spec.record_trial(
                    &mut $tally,
                    &format!("KLL<{type_name}> mixed-sign seed={seed:#x}"),
                    truth.sorted(),
                    &QS,
                    |q| fixed.quantile(q),
                );

                let seed = kll_trial_seed($trial);
                $trial += 1;
                let mut dynamic = KLLDynamic::<$ty>::init_kll_with_seed(K, seed);
                for v in &values {
                    dynamic.update(v);
                }
                spec.record_trial(
                    &mut $tally,
                    &format!("KLLDynamic<{type_name}> mixed-sign seed={seed:#x}"),
                    truth.sorted(),
                    &QS,
                    |q| dynamic.quantile(q),
                );
            }
            // KLL answers with *retained* items, and compaction may discard
            // the actual extremes — so q=0 is not required to be the exact
            // minimum (its rank-error obligation at q=0 is already tallied
            // above). What is structural is that the sketch never invents a
            // value: every answer must lie inside the observed range, on the
            // correct side of zero. A `total_cmp` that ordered negatives by
            // their bit pattern would break this immediately.
            let mut probe = KLL::<$ty>::init_kll_with_seed(K, kll_trial_seed($trial));
            for v in &values {
                probe.update(v);
            }
            for q in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
                let got = probe.quantile(q);
                assert!(
                    got >= truth.min() && got <= truth.max(),
                    concat!(
                        stringify!($ty),
                        ": quantile({}) returned {}, outside the observed range [{}, {}]"
                    ),
                    q,
                    got,
                    truth.min(),
                    truth.max()
                );
            }
            // The median of a symmetric mixed-sign stream must land near zero,
            // which a broken sign ordering could not produce.
            let median = probe.quantile(0.5);
            assert!(
                median.abs() <= truth.max() * 0.10,
                concat!(
                    stringify!($ty),
                    ": median {} is far from zero on a stream symmetric about it \
                     (range [{}, {}]); check the type's ordering"
                ),
                median,
                truth.min(),
                truth.max()
            );
        }};
    }

    let mut tally = Tally::default();
    let mut trial = 0x5164_0000u64;
    signed_case!(tally, trial, i8, 100);
    signed_case!(tally, trial, i16, 30_000);
    signed_case!(tally, trial, i32, 1_000_000_000);
    signed_case!(tally, trial, i64, 1_000_000_000_000_000);
    signed_case!(tally, trial, i128, 1_000_000_000_000_000);
    signed_case!(tally, trial, isize, 1_000_000_000_000_000);
    signed_case!(tally, trial, f32, 1_000_000);
    signed_case!(tally, trial, f64, 1_000_000_000_000_000);

    tally.assert_independent_binomial(
        "KLL family over signed and mixed-sign values",
        KllRankSpec::datasketches(K as usize).trial_failure_probability,
        &format!(
            "one trial = one sketch with its own compaction seed, scored on its worst \
             rank error over the q grid. k={K}, seeds kll_trial_seed(0x5164_0000..), \
             stream_seed={STREAM_SEED:#x} n={N}"
        ),
    );
}

// ------------------------------------------------------------------ DDSketch

/// `DDSketch::add<T>` accepts any `NumericalValue`. The relative-error
/// guarantee is on the projected value, so converting through a wider integer
/// type must not change it: the same numbers must come back within `alpha`
/// whether they arrived as `u8` or `u128`.
macro_rules! dds_type_case {
    ($alpha:expr, $tally:ident, $ty:ty, $span:expr) => {{
        let values = typed_stream!($ty, $span);
        let truth = projected_truth(&values);
        let spec = RelativeQuantileSpec::core($alpha);
        let mut sketch = DDSketch::new($alpha);
        for v in &values {
            sketch.add(v);
        }
        assert_eq!(
            sketch.get_count() as usize,
            values.len(),
            concat!(
                stringify!($ty),
                ": every positive value must be trackable at this alpha"
            )
        );
        spec.tally_into(&mut $tally, truth.sorted(), &QS, |q| {
            sketch.get_value_at_quantile(q)
        });
    }};
}

#[test]
fn every_numeric_type_satisfies_the_ddsketch_relative_value_error_contract() {
    for alpha in [0.001f64, 0.01, 0.05] {
        let mut tally = Tally::default();
        dds_type_case!(alpha, tally, i8, 127);
        dds_type_case!(alpha, tally, i16, 32_767);
        dds_type_case!(alpha, tally, i32, 1_000_000_000);
        dds_type_case!(alpha, tally, i64, 1_000_000_000_000_000);
        dds_type_case!(alpha, tally, i128, 1_000_000_000_000_000);
        dds_type_case!(alpha, tally, isize, 1_000_000_000_000_000);
        dds_type_case!(alpha, tally, u8, 255);
        dds_type_case!(alpha, tally, u16, 65_535);
        dds_type_case!(alpha, tally, u32, 4_000_000_000);
        dds_type_case!(alpha, tally, u64, 1_000_000_000_000_000);
        dds_type_case!(alpha, tally, u128, 1_000_000_000_000_000);
        dds_type_case!(alpha, tally, usize, 1_000_000_000_000_000);
        dds_type_case!(alpha, tally, f32, 1_000_000);
        dds_type_case!(alpha, tally, f64, 1_000_000_000_000_000);

        // Deterministic guarantee: no hashing, no sampling, so no violations
        // are tolerated at any alpha.
        tally.assert_none(
            "DDSketch::add<T> across every built-in NumericalValue type",
            &format!("alpha={alpha} stream_seed={STREAM_SEED:#x} n={N} q grid {QS:?}"),
        );
    }
}

/// The 128-bit types past `f64`'s exact-integer range.
///
/// `2^53` is where `f64` stops representing consecutive integers, and
/// `NumericalValue::to_f64` is a plain `as` cast, so an `i128`/`u128` above it
/// reaches the sketch already rounded. The relative-error contract still holds
/// on the projected value — the rounding is at most one part in `2^53`, twelve
/// orders of magnitude below the smallest alpha the library supports — but the
/// *identity* of the value is not preserved, and that is a documented limit of
/// the caller's type choice rather than a sketch defect.
#[test]
fn the_f64_projection_is_exact_below_two_to_the_53() {
    const TWO_53: u128 = 1u128 << 53;

    // Exact below the boundary, in both 128-bit types.
    for v in [1u128, 1_000, TWO_53 - 1, TWO_53] {
        assert_eq!(
            (v as f64) as u128,
            v,
            "u128 {v} must project into f64 exactly (at or below 2^53)"
        );
        assert_eq!(
            ((v as i128).to_f64()) as i128,
            v as i128,
            "i128 {v} must project into f64 exactly (at or below 2^53)"
        );
    }
    // Just past it, consecutive integers collapse — this is the documented
    // limit, asserted so the boundary cannot move unnoticed.
    assert_eq!(
        (TWO_53 + 1) as f64,
        TWO_53 as f64,
        "2^53 + 1 is expected to collapse onto 2^53 in f64"
    );

    // The relative-error contract survives regardless: a stream of very large
    // 128-bit values still answers within alpha of the projected truth.
    const ALPHA: f64 = 0.01;
    let values: Vec<u128> = uniform_u64(5_000, 1_000_000, STREAM_SEED)
        .into_iter()
        .map(|v| (v as u128 + 1) * (1u128 << 70))
        .collect();
    let truth = projected_truth(&values);
    let mut sketch = DDSketch::new(ALPHA);
    for v in &values {
        sketch.add(v);
    }
    assert_eq!(sketch.get_count() as usize, values.len());
    let mut tally = Tally::default();
    RelativeQuantileSpec::core(ALPHA).tally_into(&mut tally, truth.sorted(), &QS, |q| {
        sketch.get_value_at_quantile(q)
    });
    tally.assert_none(
        "DDSketch over u128 values above 2^70",
        &format!("alpha={ALPHA} values in [2^70, 10^6 * 2^70], stream_seed={STREAM_SEED:#x}"),
    );

    // And the KLL family orders them correctly at that magnitude. Sixteen
    // independent compaction seeds, one trial each, scored on the worst rank
    // error over the q grid.
    let spec = KllRankSpec::datasketches(K as usize);
    let mut rank_tally = Tally::default();
    for t in 0..16u64 {
        let seed = kll_trial_seed(0x2E70_0000 + t);
        let mut kll = KLL::<u128>::init_kll_with_seed(K, seed);
        for v in &values {
            kll.update(v);
        }
        spec.record_trial(
            &mut rank_tally,
            &format!("KLL<u128> above 2^70 seed={seed:#x}"),
            truth.sorted(),
            &QS,
            |q| kll.quantile(q),
        );
    }
    rank_tally.assert_independent_binomial(
        "KLL<u128> over values above 2^70",
        spec.trial_failure_probability,
        &format!("k={K}, 16 seeds from kll_trial_seed(0x2E70_0000..)"),
    );
}
