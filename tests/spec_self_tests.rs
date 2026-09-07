//! Self-tests for the statistical specs in `tests/common/specs.rs`.
//!
//! These verify the *bounds*, not any sketch: the depth threshold the median
//! estimator earns, the binomial tails that threshold selects, and the
//! simultaneous `kappa` search built on them. They live in their own
//! integration binary because `tests/common/specs.rs` is compiled into every
//! suite that says `mod common;` — a `#[test]` there runs once per suite, so
//! these four ran fourteen times over for one bound each.

mod common;

use common::specs::{CountSketchSpec, SIMULTANEOUS_LEVEL, SecondMomentSpec, binomial_tail_ge};

/// The bad-row threshold is `ceil(d/2)`, not `d/2 + 1`.
///
/// They agree for odd `d` and differ for even `d`, which is exactly where
/// the estimator changes shape: an even-depth sketch reports the average of
/// the two middle order statistics rather than a single one.
#[test]
fn bad_row_threshold_is_ceil_half_the_depth() {
    for (rows, expected) in [
        (1usize, 1usize),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 4),
    ] {
        let spec = CountSketchSpec::new(rows, 1024);
        assert_eq!(
            spec.bad_row_threshold(),
            expected,
            "d={rows}: threshold must be ceil(d/2)"
        );
        if rows % 2 == 1 {
            assert_eq!(
                spec.bad_row_threshold(),
                rows / 2 + 1,
                "d={rows}: for odd depth ceil(d/2) and d/2+1 coincide"
            );
        } else {
            assert!(
                spec.bad_row_threshold() < rows / 2 + 1,
                "d={rows}: for even depth ceil(d/2) must be strictly smaller than \
                 d/2+1, the naive majority threshold"
            );
        }
    }
    // `SecondMomentSpec` reports the same statistic and must agree.
    for rows in 1usize..=8 {
        assert_eq!(
            SecondMomentSpec::new(rows, 1024).bad_row_threshold(),
            CountSketchSpec::new(rows, 1024).bad_row_threshold(),
            "d={rows}: the two specs must use the same threshold"
        );
    }
}

/// The binomial tails the threshold selects, at the marginal `kappa = 3`.
///
/// Pinned numerically so a change to either the threshold or the tail
/// routine is visible. `d = 4` is where `d/2 + 1` diverges: it gives 0.111
/// where the estimator earns 0.407, a bound almost four times stronger than
/// the median supports.
#[test]
fn marginal_failure_probabilities_match_the_hand_computed_tails() {
    let cases = [
        (3usize, 2usize, 0.259_259_259_259),
        (4, 2, 0.407_407_407_407),
        (5, 3, 0.209_876_543_209),
        (6, 3, 0.319_615_912_208),
    ];
    for (rows, threshold, tail) in cases {
        let spec = CountSketchSpec::new(rows, 1024);
        assert_eq!(spec.bad_row_threshold(), threshold, "d={rows}");
        assert!(
            (spec.marginal_failure() - tail).abs() < 1e-9,
            "d={rows}: P[Bin({rows}, 1/3) >= {threshold}] = {} but hand computation \
             gives {tail}",
            spec.marginal_failure()
        );
    }
    // `d/2 + 1` at d=4, kept as an explicit contrast.
    assert!(
        (binomial_tail_ge(4, 3, 1.0 / 3.0) - 0.111_111_111_111).abs() < 1e-9,
        "P[Bin(4, 1/3) >= 3] is the number a `d/2 + 1` threshold reports"
    );
}

/// Two same-direction bad rows really do break a four-row median.
///
/// This is the concrete counter-example to a `d/2 + 1` threshold: with
/// `d = 4` the reported value is `(X_(2) + X_(3)) / 2`, so two bad rows on
/// the same side put a bad value at position 3 and drag the average out of
/// the band — while three good rows and one bad row cannot, because both
/// middle order statistics are then good and the band is convex.
#[test]
fn two_same_direction_bad_rows_move_an_averaged_four_row_median_out_of_band() {
    // Median of four, exactly as `compute_median_inline_f64` computes it.
    fn median4(mut v: [f64; 4]) -> f64 {
        v.sort_by(f64::total_cmp);
        (v[1] + v[2]) / 2.0
    }

    let f = 100.0f64;
    let t = 10.0f64; // the error scale; "bad" means outside [90, 110]
    let good = |x: f64| (x - f).abs() <= t;

    // One bad row: the outlier sorts to an end, both middle values are
    // good, the average stays in band.
    let one_bad = [95.0, 105.0, 100.0, 10_000.0];
    assert_eq!(one_bad.iter().filter(|x| !good(**x)).count(), 1);
    assert!(
        good(median4(one_bad)),
        "one bad row must not be able to break the band, or the threshold \
         would have to be 1"
    );

    // Two bad rows on the same side: one of them occupies position 3, and
    // the average leaves the band by an unbounded amount.
    let two_bad_same_side = [95.0, 105.0, 10_000.0, 10_000.0];
    assert_eq!(
        two_bad_same_side.iter().filter(|x| !good(**x)).count(),
        2,
        "the fixture must have exactly two bad rows"
    );
    let broken = median4(two_bad_same_side);
    assert!(
        !good(broken),
        "two same-direction bad rows must break a four-row averaged median; \
         got {broken}, which is inside [{}, {}]",
        f - t,
        f + t
    );

    // Two bad rows on *opposite* sides cancel, which is why the bound is
    // stated on the count of bad rows rather than on their arrangement:
    // `ceil(d/2)` is the smallest count for which *some* arrangement fails,
    // which is what a failure-probability upper bound needs.
    let two_bad_opposite = [95.0, 105.0, -10_000.0, 10_000.0];
    assert!(
        good(median4(two_bad_opposite)),
        "opposite-side outliers cancel here; the threshold is a worst-case \
         count, not a claim that every arrangement of that many fails"
    );
}

/// Raising `kappa` until the union bound over `D` keys closes must respect
/// `ceil(d/2)`, which gives an even-depth sketch a wider simultaneous band
/// than `d/2 + 1` would.
#[test]
fn simultaneous_kappa_reflects_the_corrected_threshold() {
    let spec = CountSketchSpec::new(4, 2048);
    let kappa = spec.simultaneous_kappa(512, SIMULTANEOUS_LEVEL);
    assert!(
        spec.key_failure_at(kappa) <= SIMULTANEOUS_LEVEL / 512.0,
        "the search must actually reach the target: got {} for a target of {}",
        spec.key_failure_at(kappa),
        SIMULTANEOUS_LEVEL / 512.0
    );
    // Two bad rows out of four is a `kappa^-2` tail, so the required kappa
    // is far larger than the `kappa^-3` a `d/2 + 1` threshold implies.
    let old_style = {
        let mut lo = 3.0f64;
        let mut hi = lo;
        for _ in 0..80 {
            hi *= 2.0;
            if binomial_tail_ge(4, 3, 1.0 / hi) <= SIMULTANEOUS_LEVEL / 512.0 {
                break;
            }
            lo = hi;
        }
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if binomial_tail_ge(4, 3, 1.0 / mid) <= SIMULTANEOUS_LEVEL / 512.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    };
    assert!(
        kappa > old_style,
        "the corrected threshold must demand a larger simultaneous kappa at d=4: \
         got {kappa}, old formula gave {old_style}"
    );
}
