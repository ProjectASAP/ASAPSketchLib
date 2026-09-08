//! E2E suite for the Bloom filter: the properties a membership sketch is
//! bought for, which the shared batteries do not model.
//!
//! `tests/conformance_kit.rs` runs the standard `membership_battery`. Here the
//! filter is pushed on the three things it promises exactly - no false
//! negative, a union that is the filter of the concatenated stream, and a
//! delivered false-positive rate its own sizing predicts - plus the two hash
//! paths, serialization, and the degenerate geometries.

use asap_sketchlib::bloom::{BLOOM_MAX_BITS, BLOOM_MAX_SLICES};
use asap_sketchlib::{
    BLOOM_DEFAULT_COLS, BLOOM_DEFAULT_ROWS, BitMatrix, Bloom, DataInput, FastPath, MatrixHashMode,
    RegularPath, hash_mode_for_matrix,
};

const MEMBERS: i64 = 20_000;
const PROBES: i64 = 200_000;

fn members() -> Vec<i64> {
    (0..MEMBERS).collect()
}

fn probes() -> Vec<i64> {
    (10_000_000..10_000_000 + PROBES).collect()
}

fn all_bits(bits: &BitMatrix) -> Vec<bool> {
    (0..bits.rows())
        .flat_map(|row| (0..bits.cols()).map(move |col| (row, col)))
        .map(|(row, col)| bits.get(row, col))
        .collect()
}

/// Row pairs holding exactly the same bits. Two slices that agree on every one
/// of thousands of columns are the same hash function, not a coincidence.
fn duplicate_slice_pairs(bits: &BitMatrix) -> Vec<(usize, usize)> {
    let mut pairs = vec![];
    for left in 0..bits.rows() {
        for right in left + 1..bits.rows() {
            if (0..bits.cols()).all(|col| bits.get(left, col) == bits.get(right, col)) {
                pairs.push((left, right));
            }
        }
    }
    pairs
}

/// Five standard errors of a binomial rate over `PROBES` draws. Wide enough
/// that a correct filter never trips it, narrow enough that it stays under the
/// sizing target at every rate the suite exercises.
fn sampling_band(rate: f64) -> f64 {
    5.0 * (rate * (1.0 - rate) / PROBES as f64).sqrt()
}

fn filled_regular(target: f64) -> Bloom<RegularPath> {
    let mut filter = Bloom::<RegularPath>::with_capacity(MEMBERS as usize, target);
    for key in members() {
        filter.insert(&DataInput::I64(key));
    }
    filter
}

fn false_positive_rate<F: Fn(i64) -> bool>(contains: F) -> f64 {
    let probes = probes();
    let hits = probes.iter().filter(|k| contains(**k)).count();
    hits as f64 / probes.len() as f64
}

/// The one exact guarantee: a key that was inserted is never reported absent.
/// Asserted on both hash paths, since they decode columns differently.
#[test]
fn an_inserted_key_is_never_reported_absent() {
    let regular = filled_regular(0.01);
    let absent: Vec<i64> = members()
        .into_iter()
        .filter(|k| !regular.contains(&DataInput::I64(*k)))
        .collect();
    assert!(
        absent.is_empty(),
        "regular path lost {} of {MEMBERS} members, first {:?}",
        absent.len(),
        absent.first()
    );

    let mut fast = Bloom::<FastPath>::with_capacity(MEMBERS as usize, 0.01);
    for key in members() {
        fast.insert(&DataInput::I64(key));
    }
    let absent_fast = members()
        .into_iter()
        .filter(|k| !fast.contains(&DataInput::I64(*k)))
        .count();
    assert_eq!(absent_fast, 0, "fast path lost {absent_fast} of {MEMBERS}");
}

/// A filter with nothing in it rejects everything, on both paths.
#[test]
fn an_empty_filter_rejects_every_probe() {
    let regular = Bloom::<RegularPath>::with_capacity(MEMBERS as usize, 0.01);
    let fast = Bloom::<FastPath>::with_capacity(MEMBERS as usize, 0.01);
    assert!(regular.is_empty());
    assert_eq!(regular.inserted(), 0);
    assert!(fast.is_empty());
    assert_eq!(fast.inserted(), 0);
    for key in probes().into_iter().take(10_000) {
        assert!(!regular.contains(&DataInput::I64(key)));
        assert!(!fast.contains(&DataInput::I64(key)));
    }
}

/// The delivered rate lands at or under the target, and `predicted_fpp`
/// describes it. Sizing rounds each slice up to a power of two, so the filter
/// is larger than the formula's minimum and the rate correspondingly lower.
#[test]
fn the_measured_false_positive_rate_matches_what_the_sizing_predicts() {
    for target in [0.1, 0.01, 0.001] {
        let filter = filled_regular(target);
        let measured = false_positive_rate(|k| filter.contains(&DataInput::I64(k)));
        let predicted = filter.predicted_fpp(MEMBERS as usize);

        assert!(
            predicted <= target,
            "target {target}: predicted {predicted:.5} above the target"
        );
        // The prediction is the model. A five-sigma binomial band around it is
        // the tightest window the probe set supports, and it stays below the
        // target at every rate here, so the two assertions cannot disagree.
        let band = sampling_band(predicted);
        assert!(
            predicted + band <= target,
            "target {target}: the {band:.6} band around {predicted:.5} reaches past the target"
        );
        assert!(
            (measured - predicted).abs() <= band,
            "target {target}: measured {measured:.6} is more than {band:.6} from predicted {predicted:.6}"
        );
        assert!(
            measured <= target,
            "target {target}: measured {measured:.5} above the target"
        );
    }
}

/// `estimated_fpp` reads the bits actually set rather than the insert count,
/// so it tracks the measured rate on a filter that saw duplicates.
#[test]
fn the_fill_based_estimate_tracks_the_measured_rate() {
    let mut filter = Bloom::<RegularPath>::with_capacity(MEMBERS as usize, 0.01);
    for _ in 0..3 {
        for key in members() {
            filter.insert(&DataInput::I64(key));
        }
    }
    assert_eq!(filter.inserted(), 3 * MEMBERS as u64);

    let measured = false_positive_rate(|k| filter.contains(&DataInput::I64(k)));
    let estimated = filter.estimated_fpp();
    assert!(
        measured >= estimated * 0.75 && measured <= estimated * 1.25,
        "measured {measured:.5} outside 0.75-1.25x of estimated {estimated:.5}"
    );

    // The same keys once set the same bits and a third of the inserts, so an
    // estimate that read the counter would move and this one does not.
    let mut once = Bloom::<RegularPath>::with_capacity(MEMBERS as usize, 0.01);
    for key in members() {
        once.insert(&DataInput::I64(key));
    }
    assert_eq!(all_bits(once.as_bits()), all_bits(filter.as_bits()));
    assert_ne!(once.inserted(), filter.inserted());
    assert_eq!(
        once.estimated_fpp(),
        estimated,
        "estimated_fpp moved with the insert count"
    );
}

/// Inserting a key twice sets no bit the first insert did not, so the filter's
/// fill is a function of the distinct keys alone.
#[test]
fn repeated_inserts_leave_the_bits_unchanged() {
    let mut once = Bloom::<RegularPath>::with_capacity(1_000, 0.01);
    let mut twice = Bloom::<RegularPath>::with_capacity(1_000, 0.01);
    for key in 0..1_000i64 {
        once.insert(&DataInput::I64(key));
        twice.insert(&DataInput::I64(key));
        twice.insert(&DataInput::I64(key));
    }
    assert_eq!(once.as_bits().count_ones(), twice.as_bits().count_ones());
    // The same population is not the same bits; compare the grids cell by cell.
    assert_eq!(all_bits(once.as_bits()), all_bits(twice.as_bits()));
    assert_eq!(once.inserted(), 1_000);
    assert_eq!(twice.inserted(), 2_000);
}

/// A union is exact: merging two filters gives the same bits as one filter
/// over both streams. This is the property that makes the filter shardable.
#[test]
fn a_union_equals_the_filter_of_the_concatenated_stream() {
    let mut left = Bloom::<RegularPath>::with_dimensions(7, 1 << 14);
    let mut right = Bloom::<RegularPath>::with_dimensions(7, 1 << 14);
    let mut whole = Bloom::<RegularPath>::with_dimensions(7, 1 << 14);

    for key in 0..10_000i64 {
        if key % 2 == 0 {
            left.insert(&DataInput::I64(key));
        } else {
            right.insert(&DataInput::I64(key));
        }
        whole.insert(&DataInput::I64(key));
    }

    left.merge(&right);
    assert_eq!(left.as_bits().count_ones(), whole.as_bits().count_ones());
    assert_eq!(left.inserted(), whole.inserted());

    for key in 0..10_000i64 {
        assert!(left.contains(&DataInput::I64(key)), "union lost key {key}");
    }
    // Same bits means the same answers on non-members too, not just members.
    for key in probes().into_iter().take(20_000) {
        assert_eq!(
            left.contains(&DataInput::I64(key)),
            whole.contains(&DataInput::I64(key)),
            "union and single-pass disagree on non-member {key}"
        );
    }
}

/// Sizing is checked against its contract rather than against its own
/// expression: the chosen geometry meets the target, and one power of two
/// narrower does not. That pins tightness without restating the code.
#[test]
fn sizing_meets_the_target_and_is_one_power_of_two_from_missing_it() {
    for (n, p) in [
        (1_000usize, 0.01),
        (20_000, 0.1),
        (20_000, 0.001),
        (10_000, 1e-7),
        (10_000, 1e-12),
        (1, 0.5),
    ] {
        let (rows, cols) = Bloom::<RegularPath>::dimensions_for(n, p);
        assert!(
            cols.is_power_of_two(),
            "n={n} p={p} gave a non-power-of-two slice of {cols}"
        );
        assert!(
            rows <= BLOOM_MAX_SLICES,
            "n={n} p={p} asked for {rows} slices, past the {BLOOM_MAX_SLICES} the seed list has"
        );

        let chosen = Bloom::<RegularPath>::with_dimensions(rows, cols);
        assert!(
            chosen.predicted_fpp(n) <= p,
            "n={n} p={p}: {rows}x{cols} predicts {:e}, above the target",
            chosen.predicted_fpp(n)
        );
        if cols > 1 {
            let narrower = Bloom::<RegularPath>::with_dimensions(rows, cols / 2);
            assert!(
                narrower.predicted_fpp(n) > p,
                "n={n} p={p}: {rows}x{} already meets the target, so {cols} is oversized",
                cols / 2
            );
        }
    }
}

/// The slice count is capped at the number of seeds the hasher actually has,
/// and every slice a sized filter builds is a distinct hash function.
#[test]
fn sizing_never_asks_for_more_slices_than_the_seed_list_has() {
    for (n, p) in [(10_000usize, 1e-7), (10_000, 1e-9), (10_000, 1e-12)] {
        let (rows, cols) = Bloom::<RegularPath>::dimensions_for(n, p);
        assert!(
            rows <= BLOOM_MAX_SLICES,
            "n={n} p={p:e} asked for {rows} slices"
        );

        let mut filter = Bloom::<RegularPath>::with_dimensions(rows, cols);
        for key in 0..n as i64 {
            filter.insert(&DataInput::I64(key));
        }
        assert_eq!(
            duplicate_slice_pairs(filter.as_bits()),
            vec![],
            "n={n} p={p:e}: {rows}x{cols} has slices that are the same hash function"
        );
    }
}

/// A target the seed list cannot reach within [`BLOOM_MAX_BITS`] yields the
/// widest slices that fit rather than an unbounded allocation, and a huge
/// `expected_items` does not wrap the power-of-two rounding to zero.
#[test]
fn sizing_stays_inside_the_allocation_ceiling() {
    for (n, p) in [
        (usize::MAX, 0.01),
        (1usize << 50, 0.01),
        (1usize << 40, 1e-9),
        (1_000, f64::MIN_POSITIVE),
        (1_000, 0.0),
        (1_000, 2.0),
    ] {
        let (rows, cols) = Bloom::<RegularPath>::dimensions_for(n, p);
        assert!(cols >= 1, "n={n} p={p:e} gave {cols} columns");
        assert!(cols.is_power_of_two(), "n={n} p={p:e} gave {cols} columns");
        assert!(
            (1..=BLOOM_MAX_SLICES).contains(&rows),
            "n={n} p={p:e} gave {rows} slices"
        );
        assert!(
            rows * cols <= BLOOM_MAX_BITS,
            "n={n} p={p:e}: {rows}x{cols} is {} bits, past the ceiling",
            rows * cols
        );
    }
}

/// The allocation ceiling is a documented number, and the geometry a target
/// past it settles on is that number divided by the slice count and rounded
/// down to a power of two.
#[test]
fn the_allocation_ceiling_is_the_documented_number() {
    assert_eq!(BLOOM_MAX_BITS, 1 << 31);
    assert_eq!(
        Bloom::<RegularPath>::dimensions_for(1 << 40, 1e-9),
        (20, 1 << 26)
    );
}

/// A degenerate sizing input yields a real geometry rather than the widest
/// slice count over one bit each, which would answer yes to everything.
#[test]
fn degenerate_sizing_inputs_give_a_usable_geometry() {
    assert_eq!(Bloom::<RegularPath>::dimensions_for(0, 0.01), (7, 2));
    assert_eq!(
        Bloom::<RegularPath>::dimensions_for(1_000, 0.0),
        (20, 1 << 26)
    );
    assert_eq!(Bloom::<RegularPath>::dimensions_for(1_000, 2.0), (1, 32));
}

/// The default filter is the documented geometry.
#[test]
fn the_default_filter_has_the_documented_dimensions() {
    let filter = Bloom::<RegularPath>::default();
    assert_eq!(
        (filter.rows(), filter.cols()),
        (BLOOM_DEFAULT_ROWS, BLOOM_DEFAULT_COLS)
    );
    assert!(filter.is_empty());
}

/// A width that is not a power of two folds through a modulo, and the two ends
/// of a query must fold the same bits: a key inserted under one reduction and
/// looked up under another goes missing. Every geometry here is checked on both
/// paths, with few enough members that the slices are not saturated.
#[test]
fn non_power_of_two_widths_answer_membership_on_both_paths() {
    const KEYS: i64 = 200;
    for (rows, cols) in [
        (1usize, 1usize),
        (3, 65),
        (7, 100),
        (5, 127),
        (9, 129),
        (5, 1_000),
    ] {
        let mut regular = Bloom::<RegularPath>::with_dimensions(rows, cols);
        let mut fast = Bloom::<FastPath>::with_dimensions(rows, cols);
        for key in 0..KEYS {
            regular.insert(&DataInput::I64(key));
            fast.insert(&DataInput::I64(key));
        }
        let lost_regular = (0..KEYS)
            .filter(|k| !regular.contains(&DataInput::I64(*k)))
            .count();
        let lost_fast = (0..KEYS)
            .filter(|k| !fast.contains(&DataInput::I64(*k)))
            .count();
        assert_eq!(
            lost_regular, 0,
            "{rows}x{cols} regular path lost {lost_regular} of {KEYS} members"
        );
        assert_eq!(
            lost_fast, 0,
            "{rows}x{cols} fast path lost {lost_fast} of {KEYS} members"
        );
    }
}

/// A geometry whose row windows all fit in one 64-bit hash is the fast path's
/// third layout, and each row must read its own window. Collapsing the rows
/// onto one window keeps every member present and costs the whole rate, so
/// only a measured rate catches it.
#[test]
fn a_packed_64_geometry_gives_each_row_its_own_window() {
    const ROWS: usize = 5;
    const COLS: usize = 1024;
    const KEYS: i64 = 300;
    assert_eq!(
        hash_mode_for_matrix(ROWS, COLS),
        MatrixHashMode::Packed64,
        "{ROWS}x{COLS} no longer selects the packed 64-bit layout"
    );

    let mut filter = Bloom::<FastPath>::with_dimensions(ROWS, COLS);
    for key in 0..KEYS {
        filter.insert(&DataInput::I64(key));
    }
    for key in 0..KEYS {
        assert!(filter.contains(&DataInput::I64(key)), "lost member {key}");
    }

    let measured = false_positive_rate(|k| filter.contains(&DataInput::I64(k)));
    let predicted = filter.predicted_fpp(KEYS as usize);
    let band = sampling_band(predicted);
    assert!(
        (measured - predicted).abs() <= band,
        "measured {measured:e} is more than {band:e} from predicted {predicted:e}"
    );
    // One shared window would leave a single slice's rate, which this bound is
    // two orders of magnitude under.
    assert!(
        measured <= 0.01,
        "measured {measured:e}: the five slices are not independent"
    );
}

#[test]
#[should_panic(expected = "target false-positive rate must be finite")]
fn a_nan_target_rate_is_rejected() {
    let _ = Bloom::<RegularPath>::with_capacity(1_000, f64::NAN);
}

#[test]
#[should_panic(expected = "target false-positive rate must be finite")]
fn an_infinite_target_rate_is_rejected() {
    let _ = Bloom::<RegularPath>::with_capacity(1_000, f64::INFINITY);
}

#[test]
#[should_panic(expected = "target false-positive rate must be finite")]
fn a_negative_infinite_target_rate_is_rejected() {
    let _ = Bloom::<RegularPath>::with_capacity(1_000, f64::NEG_INFINITY);
}

/// A single-bit filter is degenerate but must still answer, and a single row
/// must still hold its members.
#[test]
fn degenerate_geometries_still_answer() {
    let mut one = Bloom::<RegularPath>::with_dimensions(1, 1);
    assert!(!one.contains(&DataInput::I64(7)));
    one.insert(&DataInput::I64(7));
    assert!(one.contains(&DataInput::I64(7)));
    // One bit, so everything now reads present.
    assert!(one.contains(&DataInput::I64(8)));
    assert_eq!(one.estimated_fpp(), 1.0);

    let mut row = Bloom::<RegularPath>::with_dimensions(1, 4096);
    for key in 0..500i64 {
        row.insert(&DataInput::I64(key));
    }
    for key in 0..500i64 {
        assert!(row.contains(&DataInput::I64(key)));
    }
}

/// `clear` returns the filter to its constructed state, dimensions kept.
#[test]
fn clearing_restores_an_empty_filter() {
    let mut filter = filled_regular(0.01);
    let (rows, cols) = (filter.rows(), filter.cols());
    filter.clear();
    assert!(filter.is_empty());
    assert_eq!(filter.inserted(), 0);
    assert_eq!((filter.rows(), filter.cols()), (rows, cols));
    for key in members().into_iter().take(1_000) {
        assert!(!filter.contains(&DataInput::I64(key)));
    }
}

/// A round-trip through serde preserves every answer, members and probes
/// alike.
#[test]
fn a_serde_round_trip_preserves_every_answer() {
    let filter = filled_regular(0.01);
    let bytes = rmp_serde::to_vec(&filter).expect("serialize");
    let decoded: Bloom<RegularPath> = rmp_serde::from_slice(&bytes).expect("deserialize");

    assert_eq!(decoded.rows(), filter.rows());
    assert_eq!(decoded.cols(), filter.cols());
    assert_eq!(decoded.inserted(), filter.inserted());
    assert_eq!(
        decoded.as_bits().count_ones(),
        filter.as_bits().count_ones()
    );
    for key in members() {
        assert!(decoded.contains(&DataInput::I64(key)));
    }
    for key in probes().into_iter().take(20_000) {
        assert_eq!(
            decoded.contains(&DataInput::I64(key)),
            filter.contains(&DataInput::I64(key))
        );
    }
}

/// Packed storage is one bit per cell, not one byte. A slice that is not a
/// whole number of words is padded out to one, and the padding is the only
/// slack there is.
#[test]
fn storage_is_one_bit_per_cell() {
    let filter = Bloom::<RegularPath>::with_dimensions(8, 4096);
    assert_eq!(filter.bit_capacity(), 8 * 4096);
    assert_eq!(filter.size_in_bytes(), 8 * 4096 / 8);

    for (rows, cols) in [(1usize, 1usize), (3, 65), (7, 100), (5, 127), (9, 129)] {
        let mut filter = Bloom::<RegularPath>::with_dimensions(rows, cols);
        assert_eq!(filter.bit_capacity(), rows * cols);
        assert_eq!(filter.size_in_bytes(), rows * cols.div_ceil(64) * 8);

        // Padding bits sit outside `cols`, so filling every slice cannot reach
        // them and `fill_ratio` still tops out at 1.
        for key in 0..20_000i64 {
            filter.insert(&DataInput::I64(key));
        }
        assert!(
            filter.fill_ratio() <= 1.0,
            "{rows}x{cols} fill {} counts padding",
            filter.fill_ratio()
        );
        assert!(filter.as_bits().count_ones() <= rows * cols);
    }
}

/// The two paths decode a key's columns differently, and whether they land on
/// the same bits is a property of the geometry, not of the filter. A packed
/// layout splits one hash into per-row windows; once the rows no longer fit in
/// 128 bits the layout falls back to one seeded hash per row, which is exactly
/// what the regular path computes.
#[test]
fn the_two_hash_paths_agree_only_where_the_geometry_gives_each_row_its_own_hash() {
    // 8 x 65536 needs 8 * 17 = 136 bits, past a packed hash, so both paths run
    // one seeded hash per row and set the same bits.
    let (mut regular, mut fast) = (
        Bloom::<RegularPath>::with_dimensions(8, 1 << 16),
        Bloom::<FastPath>::with_dimensions(8, 1 << 16),
    );
    for key in members() {
        regular.insert(&DataInput::I64(key));
        fast.insert(&DataInput::I64(key));
    }
    assert_eq!(
        all_bits(regular.as_bits()),
        all_bits(fast.as_bits()),
        "8 x 65536 falls back to per-row hashes, so the paths must agree"
    );

    // 7 x 65536 fits in 128 bits, so the fast path slices one hash into seven
    // windows and lands somewhere else entirely.
    let (mut regular, mut fast) = (
        Bloom::<RegularPath>::with_dimensions(7, 1 << 16),
        Bloom::<FastPath>::with_dimensions(7, 1 << 16),
    );
    for key in members() {
        regular.insert(&DataInput::I64(key));
        fast.insert(&DataInput::I64(key));
    }
    assert_ne!(
        all_bits(regular.as_bits()),
        all_bits(fast.as_bits()),
        "7 x 65536 packs the row hashes, so the paths must differ"
    );
    // The default geometry is this one, so interchangeability validated on the
    // 8-row shape does not carry over.
    assert_eq!(
        (regular.rows(), regular.cols()),
        (
            asap_sketchlib::BLOOM_DEFAULT_ROWS,
            asap_sketchlib::BLOOM_DEFAULT_COLS
        )
    );
}

/// The serialized form carries which path built the filter, so bytes cannot be
/// decoded into the other one. Without the tag this succeeds and the filter
/// then reports almost every one of its own members absent.
#[test]
fn a_filter_cannot_be_decoded_into_the_other_hash_path() {
    let regular = filled_regular(0.01);
    let mut fast = Bloom::<FastPath>::with_capacity(MEMBERS as usize, 0.01);
    for key in members() {
        fast.insert(&DataInput::I64(key));
    }

    let regular_bytes = rmp_serde::to_vec(&regular).expect("serialize");
    let fast_bytes = rmp_serde::to_vec(&fast).expect("serialize");

    let err = rmp_serde::from_slice::<Bloom<FastPath>>(&regular_bytes)
        .expect_err("regular bytes decoded as fast path");
    assert!(
        err.to_string().contains("regular") && err.to_string().contains("fast"),
        "unexpected error: {err}"
    );
    assert!(rmp_serde::from_slice::<Bloom<RegularPath>>(&fast_bytes).is_err());
    // `RegularPath` is the default parameter, so the unannotated form is the
    // one a caller reaches for and it must fail too.
    assert!(rmp_serde::from_slice::<Bloom>(&fast_bytes).is_err());

    let decoded: Bloom<FastPath> = rmp_serde::from_slice(&fast_bytes).expect("same path decodes");
    for key in members() {
        assert!(decoded.contains(&DataInput::I64(key)));
    }
}

/// The fast path round-trips like the regular one, tag and all.
#[test]
fn a_fast_path_serde_round_trip_preserves_every_answer() {
    let mut filter = Bloom::<FastPath>::with_capacity(MEMBERS as usize, 0.01);
    for key in members() {
        filter.insert(&DataInput::I64(key));
    }
    let bytes = rmp_serde::to_vec(&filter).expect("serialize");
    let decoded: Bloom<FastPath> = rmp_serde::from_slice(&bytes).expect("deserialize");

    assert_eq!(
        (decoded.rows(), decoded.cols()),
        (filter.rows(), filter.cols())
    );
    assert_eq!(decoded.inserted(), filter.inserted());
    assert_eq!(all_bits(decoded.as_bits()), all_bits(filter.as_bits()));
    for key in probes().into_iter().take(20_000) {
        assert_eq!(
            decoded.contains(&DataInput::I64(key)),
            filter.contains(&DataInput::I64(key))
        );
    }
}

/// The union is exact on the fast path too.
#[test]
fn a_fast_path_union_equals_the_filter_of_the_concatenated_stream() {
    let mut left = Bloom::<FastPath>::with_dimensions(7, 1 << 14);
    let mut right = Bloom::<FastPath>::with_dimensions(7, 1 << 14);
    let mut whole = Bloom::<FastPath>::with_dimensions(7, 1 << 14);

    for key in 0..10_000i64 {
        if key % 2 == 0 {
            left.insert(&DataInput::I64(key));
        } else {
            right.insert(&DataInput::I64(key));
        }
        whole.insert(&DataInput::I64(key));
    }
    left.merge(&right);
    assert_eq!(all_bits(left.as_bits()), all_bits(whole.as_bits()));
    assert_eq!(left.inserted(), whole.inserted());
    for key in 0..10_000i64 {
        assert!(left.contains(&DataInput::I64(key)), "union lost key {key}");
    }
}

#[test]
#[should_panic(expected = "bit matrices must have the same dimensions")]
fn merging_filters_of_different_widths_panics() {
    let mut left = Bloom::<RegularPath>::with_dimensions(7, 1 << 14);
    let right = Bloom::<RegularPath>::with_dimensions(7, 1 << 13);
    left.merge(&right);
}

#[test]
#[should_panic(expected = "bit matrices must have the same dimensions")]
fn merging_filters_of_different_slice_counts_panics() {
    let mut left = Bloom::<RegularPath>::with_dimensions(7, 1 << 14);
    let right = Bloom::<RegularPath>::with_dimensions(8, 1 << 14);
    left.merge(&right);
}

/// `bulk_insert` is the loop, not a different filter.
#[test]
fn bulk_insert_matches_inserting_one_at_a_time() {
    let batch: Vec<DataInput> = members().into_iter().map(DataInput::I64).collect();

    let mut one_by_one = Bloom::<RegularPath>::with_capacity(MEMBERS as usize, 0.01);
    for key in members() {
        one_by_one.insert(&DataInput::I64(key));
    }
    let mut bulk = Bloom::<RegularPath>::with_capacity(MEMBERS as usize, 0.01);
    bulk.bulk_insert(&batch);
    assert_eq!(all_bits(bulk.as_bits()), all_bits(one_by_one.as_bits()));
    assert_eq!(bulk.inserted(), one_by_one.inserted());

    let mut fast_one_by_one = Bloom::<FastPath>::with_capacity(MEMBERS as usize, 0.01);
    for key in members() {
        fast_one_by_one.insert(&DataInput::I64(key));
    }
    let mut fast_bulk = Bloom::<FastPath>::with_capacity(MEMBERS as usize, 0.01);
    fast_bulk.bulk_insert(&batch);
    assert_eq!(
        all_bits(fast_bulk.as_bits()),
        all_bits(fast_one_by_one.as_bits())
    );
    assert_eq!(fast_bulk.inserted(), fast_one_by_one.inserted());
}
