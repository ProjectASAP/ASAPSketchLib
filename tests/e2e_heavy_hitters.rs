//! E2E suite for the heavy-hitter sketches: Space-Saving, CocoSketch (SIGCOMM
//! '21) and the Elastic sketch (SIGCOMM '18). All three answer "which flows are
//! big" by keeping a flow key beside every counter, so a query is served from
//! the keys the structure still holds rather than from an unkeyed counter array
//! -- Space-Saving through its Stream-Summary of monitored counters, Coco
//! through partial-key aggregation over its recorded flows, Elastic through its
//! heavy/light split.
//!
//! `tests/conformance_kit.rs` runs the standard `frequency_battery` over 1024
//! Space-Saving counters. Here the summary is pushed on what that battery does
//! not model - the error sandwich around every monitored key, the ceiling an
//! unmonitored key cannot exceed, `is_guaranteed` as a claim about the real
//! top-k, and the bucket and counter lists staying well formed under sustained
//! eviction.
//!
//! Coco and Elastic run the standard conformance batteries their documented
//! contracts justify, then the depth no battery models: Coco's
//! over-attribution under substring matching, its point-query mass partition,
//! its unbiasedness under eviction and its recall at the paper's worked
//! operating point; Elastic's hot-flow tracking, its one-sided estimator under
//! eviction pressure, and the reach of the light layer's dimensions.
//!
//! `tests/e2e_octo.rs` covers the multi-threaded OctoSketch variants of Coco
//! and Elastic in its `heavy_hitters` module; everything here is the
//! single-threaded sketch. The top-k heap sketches (`CMSHeap`, `CSHeap`) answer
//! the same question from an unkeyed sketch plus a heap, and stay with their
//! own family in `tests/e2e_frequency.rs`.

mod common;

use common::{FreqTruth, zipf_u64};

use asap_sketchlib::{DataInput, HeapItem, SPACE_SAVING_DEFAULT_CAPACITY, SpaceSaving};
use std::collections::{HashMap, HashSet};

const STREAM: usize = 60_000;
const DOMAIN: usize = 2_048;
const SEED: u64 = 9_001;

fn stream() -> Vec<i64> {
    zipf_u64(STREAM, DOMAIN, 1.1, SEED)
        .iter()
        .map(|v| *v as i64)
        .collect()
}

fn truth_of(stream: &[i64]) -> FreqTruth {
    let mut truth = FreqTruth::default();
    for key in stream {
        truth.observe(*key);
    }
    truth
}

fn filled(capacity: usize, stream: &[i64]) -> SpaceSaving {
    let mut summary = SpaceSaving::with_capacity(capacity);
    for key in stream {
        summary.insert(&DataInput::I64(*key));
    }
    summary
}

fn key_of(item: &HeapItem) -> i64 {
    match item {
        HeapItem::I64(v) => *v,
        other => panic!("unexpected key form {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Space-Saving
// ---------------------------------------------------------------------------

/// The counter guarantee, both halves at once: a monitored key's count is at
/// least its true count and at most its true count plus its own error.
#[test]
fn a_monitored_key_is_sandwiched_by_its_error() {
    let stream = stream();
    let truth = truth_of(&stream);
    let summary = filled(256, &stream);

    let mut monitored = 0;
    for (key, count) in truth.pairs() {
        let probe = DataInput::I64(key);
        let estimate = summary.estimate(&probe);
        if estimate == 0 {
            continue;
        }
        monitored += 1;
        let error = summary.error(&probe);
        assert!(
            estimate >= count as u64,
            "key {key} true {count} estimated {estimate} - a monitored key must never read low"
        );
        assert!(
            estimate - error <= count as u64,
            "key {key} true {count} estimated {estimate} error {error} - the sandwich is open at the bottom"
        );
    }
    assert_eq!(monitored, summary.len());
    assert_eq!(summary.len(), 256, "the summary should be saturated");
}

/// No key the summary dropped can be larger than the smallest it kept, which
/// is what makes `min_count` a usable ceiling and `upper_bound` one-sided for
/// every key in the stream.
#[test]
fn an_unmonitored_key_never_exceeds_the_minimum_count() {
    let stream = stream();
    let truth = truth_of(&stream);
    let summary = filled(256, &stream);
    let floor = summary.min_count();
    assert!(floor > 0, "a saturated summary has a positive minimum");

    for (key, count) in truth.pairs() {
        let probe = DataInput::I64(key);
        if summary.estimate(&probe) != 0 {
            continue;
        }
        assert!(
            count as u64 <= floor,
            "dropped key {key} has true count {count}, above the {floor} ceiling"
        );
        assert_eq!(summary.upper_bound(&probe), floor);
        assert_eq!(summary.error(&probe), floor);
    }
}

/// The heavy keys the summary is bought for come back exactly, in order.
#[test]
fn the_true_heavy_hitters_are_reported_exactly_and_in_order() {
    let stream = stream();
    let truth = truth_of(&stream);
    let summary = filled(256, &stream);

    let mut expected: Vec<(i64, i64)> = truth.pairs();
    expected.sort_by_key(|(key, count)| (-*count, *key));
    expected.truncate(20);

    let reported = summary.top_k(20);
    assert_eq!(reported.len(), 20);

    let mut previous = u64::MAX;
    for (slot, (item, count, error)) in reported.iter().enumerate() {
        assert!(*count <= previous, "top_k is not descending at slot {slot}");
        previous = *count;

        let (want_key, want_count) = expected[slot];
        assert_eq!(key_of(item), want_key, "slot {slot} holds the wrong key");
        assert_eq!(*count as i64, want_count, "slot {slot} count");
        assert_eq!(
            *error, 0,
            "a true heavy hitter is never displaced, so it carries no error"
        );
    }
}

/// `is_guaranteed` is a claim about the real stream: every key it accepts is
/// genuinely above every key the summary dropped.
#[test]
fn a_guaranteed_key_really_outranks_everything_dropped() {
    let stream = stream();
    let truth = truth_of(&stream);
    let summary = filled(256, &stream);

    let dropped_max = truth
        .pairs()
        .into_iter()
        .filter(|(key, _)| summary.estimate(&DataInput::I64(*key)) == 0)
        .map(|(_, count)| count)
        .max()
        .unwrap_or(0);

    let mut guaranteed = 0;
    for (key, count) in truth.pairs() {
        if !summary.is_guaranteed(&DataInput::I64(key)) {
            continue;
        }
        guaranteed += 1;
        assert!(
            count > dropped_max,
            "key {key} is claimed guaranteed at true count {count}, but a dropped key reached {dropped_max}"
        );
    }
    assert!(
        guaranteed > 0,
        "a saturated summary should guarantee someone"
    );
}

/// Eviction takes the minimum and hands over its count as the arrival's error.
/// Hand-built so the whole sequence is forced rather than sampled.
#[test]
fn an_arrival_displaces_the_minimum_and_inherits_its_count() {
    let mut summary: SpaceSaving = SpaceSaving::with_capacity(2);
    for _ in 0..3 {
        summary.insert(&DataInput::I64(1));
    }
    for _ in 0..2 {
        summary.insert(&DataInput::I64(2));
    }
    assert_eq!(summary.min_count(), 2);

    summary.insert(&DataInput::I64(3));

    assert_eq!(summary.len(), 2);
    assert_eq!(summary.estimate(&DataInput::I64(1)), 3);
    assert_eq!(summary.error(&DataInput::I64(1)), 0);
    // Key 2 held the minimum, so key 3 took its slot at min + 1 with min as
    // the allowance for what key 3 may not have contributed.
    assert_eq!(summary.estimate(&DataInput::I64(2)), 0);
    assert_eq!(summary.estimate(&DataInput::I64(3)), 3);
    assert_eq!(summary.error(&DataInput::I64(3)), 2);
    assert_eq!(summary.min_count(), 3);
    assert_eq!(summary.total(), 6);
}

/// The bucket and counter lists stay well formed under sustained eviction:
/// walking them reaches every counter exactly once, in descending count order.
#[test]
fn the_stream_summary_lists_stay_well_formed_under_eviction() {
    let stream = stream();
    for capacity in [1usize, 2, 17, 256, 1_024] {
        let summary = filled(capacity, &stream);
        let expected = capacity.min(truth_of(&stream).distinct());
        assert_eq!(summary.len(), expected, "capacity {capacity} residency");

        let walked = summary.top_k(usize::MAX);
        assert_eq!(
            walked.len(),
            summary.len(),
            "capacity {capacity}: the bucket walk reached {} of {} counters",
            walked.len(),
            summary.len()
        );

        let mut previous = u64::MAX;
        let mut seen: HashSet<i64> = HashSet::new();
        for (item, count, _) in &walked {
            assert!(count <= &previous, "capacity {capacity}: walk out of order");
            previous = *count;
            assert!(
                seen.insert(key_of(item)),
                "capacity {capacity}: key {} reached twice",
                key_of(item)
            );
        }

        // `entries` reaches the counters directly; the two views must agree on
        // every triple, not just on which keys are held.
        let triples = |view: &[(HeapItem, u64, u64)]| {
            let mut out: Vec<(i64, u64, u64)> = view
                .iter()
                .map(|(key, count, error)| (key_of(key), *count, *error))
                .collect();
            out.sort_unstable();
            out
        };
        let direct = triples(&summary.entries());
        assert_eq!(direct.len(), seen.len());
        assert_eq!(
            direct,
            triples(&walked),
            "capacity {capacity}: the two views disagree"
        );
        assert!(
            direct.iter().any(|(_, _, error)| *error > 0),
            "capacity {capacity}: eviction left no error behind"
        );
    }
}

/// A weighted arrival is the same as that many single arrivals.
#[test]
fn a_weighted_arrival_matches_repeating_it() {
    let stream = stream();
    let mut singles: SpaceSaving = SpaceSaving::with_capacity(2_048);
    let mut weighted: SpaceSaving = SpaceSaving::with_capacity(2_048);

    let mut counts: HashMap<i64, u64> = HashMap::new();
    for key in &stream {
        singles.insert(&DataInput::I64(*key));
        *counts.entry(*key).or_default() += 1;
    }
    let mut keys: Vec<i64> = counts.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        weighted.insert_many(&DataInput::I64(key), counts[&key]);
    }

    assert_eq!(weighted.total(), singles.total());
    assert_eq!(weighted.len(), singles.len());
    for (key, count) in &counts {
        assert_eq!(
            weighted.estimate(&DataInput::I64(*key)),
            *count,
            "weighted estimate for key {key}"
        );
        assert_eq!(singles.estimate(&DataInput::I64(*key)), *count);
    }
}

/// With room for every distinct key the summary is exact and carries no error.
#[test]
fn a_summary_larger_than_the_domain_is_exact() {
    let stream = stream();
    let truth = truth_of(&stream);
    let summary = filled(DOMAIN * 2, &stream);

    assert_eq!(summary.len(), truth.distinct());
    assert_eq!(summary.min_count(), 0);
    for (key, count) in truth.pairs() {
        let probe = DataInput::I64(key);
        assert_eq!(summary.estimate(&probe), count as u64);
        assert_eq!(summary.error(&probe), 0);
    }
}

/// Merging keeps the total, respects capacity, and never reports a shared key
/// below what either side saw.
#[test]
fn a_merge_keeps_the_total_and_never_reads_low() {
    let stream = stream();
    let (left_stream, right_stream): (Vec<i64>, Vec<i64>) =
        stream.iter().partition(|key| *key % 2 == 0);

    let mut left = filled(256, &left_stream);
    let right = filled(256, &right_stream);
    let left_truth = truth_of(&left_stream);
    let right_truth = truth_of(&right_stream);

    left.merge(&right);

    assert_eq!(left.total(), stream.len() as u64);
    assert!(left.len() <= left.capacity());

    let mut checked: HashSet<i64> = HashSet::new();
    for (key, count) in left_truth.pairs().into_iter().chain(right_truth.pairs()) {
        let estimate = left.estimate(&DataInput::I64(key));
        if estimate == 0 {
            continue;
        }
        checked.insert(key);
        assert!(
            estimate >= count as u64,
            "merged key {key} reads {estimate} against a per-side truth of {count}"
        );
    }
    assert_eq!(
        checked.len(),
        left.len(),
        "the loop reached {} of the {} merged counters",
        checked.len(),
        left.len()
    );
}

/// A round-trip through serde preserves every answer, including the key index
/// the summary looks up on.
#[test]
fn a_serde_round_trip_preserves_every_answer() {
    let stream = stream();
    let truth = truth_of(&stream);
    let summary = filled(256, &stream);

    let bytes = rmp_serde::to_vec(&summary).expect("serialize");
    let decoded: SpaceSaving = rmp_serde::from_slice(&bytes).expect("deserialize");

    assert_eq!(decoded.len(), summary.len());
    assert_eq!(decoded.capacity(), summary.capacity());
    assert_eq!(decoded.total(), summary.total());
    assert_eq!(decoded.min_count(), summary.min_count());
    for (key, _) in truth.pairs() {
        let probe = DataInput::I64(key);
        assert_eq!(decoded.estimate(&probe), summary.estimate(&probe));
        assert_eq!(decoded.error(&probe), summary.error(&probe));
    }

    // The decoded summary still takes inserts against its rebuilt index.
    let mut decoded = decoded;
    let hottest = key_of(&summary.top_k(1)[0].0);
    let before = decoded.estimate(&DataInput::I64(hottest));
    decoded.insert(&DataInput::I64(hottest));
    assert_eq!(decoded.estimate(&DataInput::I64(hottest)), before + 1);
}

/// A summary that saw nothing answers without panicking.
#[test]
fn an_empty_summary_answers() {
    let summary: SpaceSaving = SpaceSaving::with_capacity(8);
    assert!(summary.is_empty());
    assert_eq!(summary.len(), 0);
    assert_eq!(summary.min_count(), 0);
    assert_eq!(summary.total(), 0);
    assert_eq!(summary.estimate(&DataInput::I64(1)), 0);
    assert_eq!(summary.upper_bound(&DataInput::I64(1)), 0);
    assert!(!summary.is_guaranteed(&DataInput::I64(1)));
    assert!(summary.top_k(5).is_empty());
    assert!(summary.entries().is_empty());
}

/// A weighted arrival that has to displace the minimum takes the same slot,
/// count and error as the arrivals it stands in for.
#[test]
fn a_weighted_arrival_matches_repeating_it_under_eviction() {
    let stream = stream();
    let mut counts: HashMap<i64, u64> = HashMap::new();
    for key in &stream {
        *counts.entry(*key).or_default() += 1;
    }
    let mut keys: Vec<i64> = counts.keys().copied().collect();
    keys.sort_unstable();

    let mut singles: SpaceSaving = SpaceSaving::with_capacity(64);
    let mut weighted: SpaceSaving = SpaceSaving::with_capacity(64);
    for key in &keys {
        for _ in 0..counts[key] {
            singles.insert(&DataInput::I64(*key));
        }
        weighted.insert_many(&DataInput::I64(*key), counts[key]);
    }

    assert_eq!(weighted.len(), 64, "the run should have evicted heavily");
    assert_eq!(weighted.total(), singles.total());
    assert_eq!(weighted.min_count(), singles.min_count());

    let mut left: Vec<(i64, u64, u64)> = weighted
        .entries()
        .iter()
        .map(|(key, count, error)| (key_of(key), *count, *error))
        .collect();
    let mut right: Vec<(i64, u64, u64)> = singles
        .entries()
        .iter()
        .map(|(key, count, error)| (key_of(key), *count, *error))
        .collect();
    left.sort_unstable();
    right.sort_unstable();
    assert_eq!(left, right, "a weighted arrival diverged from repeating it");
}

/// `bulk_insert` is the loop it replaces, and a zero weight records nothing.
#[test]
fn bulk_insert_matches_repeated_inserts_and_a_zero_weight_is_inert() {
    let stream = stream();
    let values: Vec<DataInput> = stream.iter().map(|key| DataInput::I64(*key)).collect();

    let mut bulk: SpaceSaving = SpaceSaving::with_capacity(128);
    bulk.bulk_insert(&values);
    let one_at_a_time = filled(128, &stream);

    assert_eq!(bulk.total(), one_at_a_time.total());
    assert_eq!(bulk.top_k(usize::MAX), one_at_a_time.top_k(usize::MAX));

    let before = bulk.top_k(usize::MAX);
    bulk.insert_many(&DataInput::I64(stream[0]), 0);
    bulk.insert_many(&DataInput::I64(i64::MIN), 0);
    assert_eq!(bulk.total(), one_at_a_time.total());
    assert_eq!(bulk.len(), one_at_a_time.len());
    assert_eq!(bulk.top_k(usize::MAX), before);
}

/// A summary asked for no counters still holds one.
#[test]
fn a_zero_capacity_summary_still_answers() {
    let mut summary: SpaceSaving = SpaceSaving::with_capacity(0);
    assert_eq!(summary.capacity(), 1);
    for key in 0..10i64 {
        summary.insert(&DataInput::I64(key));
    }
    assert_eq!(summary.len(), 1);
    assert_eq!(summary.total(), 10);
    let held = summary.top_k(4);
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].1, 10);
    assert_eq!(held[0].2, 9);
}

/// A merge into a summary with counters to spare still reports a ceiling that
/// covers everything the other side had already dropped.
#[test]
fn a_merge_into_an_under_full_summary_keeps_the_ceiling_honest() {
    let mut left: SpaceSaving = SpaceSaving::with_capacity(33);
    let mut right: SpaceSaving = SpaceSaving::with_capacity(1);
    for _ in 0..10 {
        right.insert(&DataInput::I64(7));
    }
    for _ in 0..20 {
        right.insert(&DataInput::I64(8));
    }

    left.merge(&right);

    assert_eq!(left.len(), 1);
    assert!(left.len() < left.capacity());
    assert!(
        left.upper_bound(&DataInput::I64(7)) >= 10,
        "key 7 truly reached 10 but is capped at {}",
        left.upper_bound(&DataInput::I64(7))
    );
    assert!(
        !left.is_guaranteed(&DataInput::I64(8)),
        "a merged summary that dropped a key cannot guarantee anything for free"
    );
}

/// A chain of merges over asymmetric capacities: every key of the combined
/// stream stays under its ceiling, and every guarantee holds against the truth.
#[test]
fn a_merge_chain_stays_one_sided_against_the_truth() {
    let stream = stream();
    let mut shards: Vec<Vec<i64>> = vec![Vec::new(); 3];
    for (position, key) in stream.iter().enumerate() {
        shards[position % 3].push(*key);
    }
    let truth = truth_of(&stream);

    let mut merged = filled(64, &shards[0]);
    merged.merge(&filled(512, &shards[1]));
    merged.merge(&filled(7, &shards[2]));

    assert_eq!(merged.total(), stream.len() as u64);
    assert!(merged.len() <= merged.capacity());

    let mut dropped_max = 0;
    for (key, count) in truth.pairs() {
        let probe = DataInput::I64(key);
        assert!(
            merged.upper_bound(&probe) >= count as u64,
            "key {key} truly reached {count} but is capped at {}",
            merged.upper_bound(&probe)
        );
        if merged.estimate(&probe) == 0 {
            dropped_max = dropped_max.max(count);
        } else {
            assert!(
                merged.estimate(&probe) >= count as u64,
                "monitored key {key} reads below its true {count}"
            );
        }
    }
    let mut guaranteed = 0;
    for (key, count) in truth.pairs() {
        if merged.is_guaranteed(&DataInput::I64(key)) {
            guaranteed += 1;
            assert!(
                count > dropped_max,
                "key {key} is claimed guaranteed at {count}, under a dropped key at {dropped_max}"
            );
        }
    }
    assert!(guaranteed > 0, "the merge chain guarantees nobody");
}

/// The same two summaries merge to the same answer every time, ties included,
/// and that answer depends on what they hold rather than on the order the keys
/// happened to arrive in.
#[test]
fn a_merge_picks_the_same_survivors_every_time() {
    fn shape_of(ascending: bool) -> Vec<(i64, u64, u64)> {
        let mut left: SpaceSaving = SpaceSaving::with_capacity(200);
        let mut right: SpaceSaving = SpaceSaving::with_capacity(200);
        let mut left_keys: Vec<i64> = (0..150).collect();
        let mut right_keys: Vec<i64> = (100..250).collect();
        if !ascending {
            left_keys.reverse();
            right_keys.reverse();
        }
        for key in left_keys {
            left.insert(&DataInput::I64(key));
        }
        for key in right_keys {
            right.insert(&DataInput::I64(key));
        }
        left.merge(&right);
        left.top_k(usize::MAX)
            .iter()
            .map(|(key, count, error)| (key_of(key), *count, *error))
            .collect()
    }

    let reference = shape_of(true);
    assert_eq!(reference.len(), 200, "the union should fill the capacity");
    assert_eq!(
        reference.iter().filter(|(_, count, _)| *count == 2).count(),
        50,
        "the fifty shared keys should be the heavy ones"
    );
    for round in 0..5 {
        assert_eq!(
            shape_of(true),
            reference,
            "round {round} merged differently"
        );
    }
    assert_eq!(
        shape_of(false),
        reference,
        "the merge depends on the order the keys arrived in"
    );
}

/// The ceiling a merge established survives the wire.
#[test]
fn a_round_trip_carries_the_merged_ceiling() {
    let stream = stream();
    let mut merged = filled(32, &stream[..stream.len() / 2]);
    merged.merge(&filled(4, &stream[stream.len() / 2..]));

    let bytes = rmp_serde::to_vec(&merged).expect("serialize");
    let decoded: SpaceSaving = rmp_serde::from_slice(&bytes).expect("deserialize");

    assert_eq!(decoded.min_count(), merged.min_count());
    assert_eq!(decoded.total(), merged.total());
    assert_eq!(decoded.len(), merged.len());
    for (key, count) in truth_of(&stream).pairs() {
        let probe = DataInput::I64(key);
        assert_eq!(decoded.estimate(&probe), merged.estimate(&probe));
        assert!(
            decoded.upper_bound(&probe) >= count as u64,
            "key {key} truly reached {count} but decodes capped at {}",
            decoded.upper_bound(&probe)
        );
    }
}

/// Serialized state that no run of the algorithm could have produced is
/// refused, rather than decoded into a summary that panics or loops.
#[test]
fn crafted_state_fails_closed() {
    #[derive(serde::Serialize)]
    struct CraftedState {
        capacity: usize,
        total: u64,
        discarded_max: u64,
        entries: Vec<(HeapItem, u64, u64)>,
    }

    let crafted =
        |capacity: usize, total: u64, discarded_max: u64, entries: Vec<(HeapItem, u64, u64)>| {
            rmp_serde::to_vec(&CraftedState {
                capacity,
                total,
                discarded_max,
                entries,
            })
            .expect("serialize")
        };

    let refused = [
        (crafted(0, 100, 0, Vec::new()), "capacity is zero"),
        (
            crafted(
                1,
                100,
                0,
                vec![(HeapItem::I64(1), 2, 0), (HeapItem::I64(2), 2, 0)],
            ),
            "over a capacity of 1",
        ),
        (
            crafted(4, 100, 0, vec![(HeapItem::I64(1), 0, 0)]),
            "counter at zero",
        ),
        (
            crafted(4, 100, 0, vec![(HeapItem::I64(1), 3, 9)]),
            "error of 9 against a count of 3",
        ),
        (
            crafted(
                4,
                100,
                0,
                vec![(HeapItem::I64(1), 3, 0), (HeapItem::I64(1), 2, 0)],
            ),
            "same key twice",
        ),
        (
            crafted(4, 100, 200, vec![(HeapItem::I64(1), 3, 0)]),
            "ceiling of 200 above its lowest count of 3",
        ),
        (
            crafted(4, 2, 0, vec![(HeapItem::I64(1), 9, 0)]),
            "total of 2 under the 9",
        ),
    ];
    for (bytes, expected) in &refused {
        let problem = rmp_serde::from_slice::<SpaceSaving>(bytes)
            .expect_err("a crafted state must be refused, not decoded")
            .to_string();
        assert!(
            problem.contains(expected),
            "expected a complaint about {expected}, got {problem}"
        );
    }

    let bytes = crafted(
        4,
        100,
        0,
        vec![(HeapItem::I64(1), 3, 1), (HeapItem::I64(2), 9, 2)],
    );
    let decoded: SpaceSaving = rmp_serde::from_slice(&bytes).expect("a well formed state");
    assert_eq!(decoded.top_k(8).len(), 2);
    assert_eq!(decoded.estimate(&DataInput::I64(2)), 9);
}

/// `SpaceSaving::default()` is the documented counter budget, and it fills.
#[test]
fn the_default_summary_holds_the_default_capacity() {
    assert_eq!(SPACE_SAVING_DEFAULT_CAPACITY, 1_024);

    let mut summary: SpaceSaving = SpaceSaving::default();
    assert_eq!(summary.capacity(), SPACE_SAVING_DEFAULT_CAPACITY);
    assert!(summary.is_empty());

    let stream = stream();
    for key in &stream {
        summary.insert(&DataInput::I64(*key));
    }

    assert_eq!(summary.len(), SPACE_SAVING_DEFAULT_CAPACITY);
    assert_eq!(summary.total(), stream.len() as u64);
    assert_eq!(
        summary.top_k(usize::MAX).len(),
        SPACE_SAVING_DEFAULT_CAPACITY
    );
}

// ---------------------------------------------------------------------------
// CocoSketch and Elastic: the keyed-bucket tables
// ---------------------------------------------------------------------------

mod keyed_bucket {
    use super::common::assert_between;
    use super::common::conformance::{self, FrequencyOps, FrequencySpec, MergeOps};
    use super::*;

    use asap_sketchlib::{Coco, DefaultXxHasher, Elastic};

    // -----------------------------------------------------------------------
    // Conformance adapters: both sketches are string-keyed, so the kit's integer
    // keys are rendered through one shared flow-id format.
    // -----------------------------------------------------------------------

    fn flow_key(key: i64) -> String {
        format!("flow::{key}")
    }

    /// Coco at its documented default, `1024 x 4`: 4096 buckets against the ~2000
    /// distinct flows the battery stream carries, which is the regime the sizing
    /// note asks for -- the table attributes mass to at most `w * d` keys at once.
    struct CocoAdapter(Coco<DefaultXxHasher>);

    impl CocoAdapter {
        fn new() -> Self {
            Self(Coco::new())
        }
    }

    impl FrequencyOps<i64> for CocoAdapter {
        fn ingest(&mut self, key: &i64) {
            self.0.insert(&flow_key(*key), 1);
        }
        fn estimate(&self, key: &i64) -> f64 {
            self.0.estimate_key(&flow_key(*key)) as f64
        }
    }

    impl MergeOps for CocoAdapter {
        fn merge_from(&mut self, other: &Self) {
            self.0.merge(&other.0);
        }
    }

    /// Elastic at 256 heavy buckets over the default 3 x 4096 light layer.
    /// Section 3.1.2 puts the elephant collision rate at `1 - (H/w + 1) e^(-H/w)`;
    /// the battery stream carries 243 dense flows, so `H/w ~ 0.95` and a quarter of
    /// the buckets hold more than one elephant. Contested buckets are the point:
    /// the losers read through the light layer, which is where the one-sided claim
    /// is worth checking.
    struct ElasticAdapter(Elastic<DefaultXxHasher>);

    impl ElasticAdapter {
        fn new() -> Self {
            Self(Elastic::init_with_length(256))
        }
    }

    impl FrequencyOps<i64> for ElasticAdapter {
        fn ingest(&mut self, key: &i64) {
            self.0.insert(flow_key(*key));
        }
        fn estimate(&self, key: &i64) -> f64 {
            self.0.query(flow_key(*key)) as f64
        }
    }

    impl MergeOps for ElasticAdapter {
        fn merge_from(&mut self, other: &Self) {
            self.0.merge(&other.0);
        }
    }

    // -----------------------------------------------------------------------
    // Battery runs
    // -----------------------------------------------------------------------

    /// Coco is *unbiased*, not one-sided: an estimate comes back either side of the
    /// truth, so `one_sided` stays false and the spec is Count Sketch's two-sided
    /// reference spec from `conformance_kit.rs`, unchanged.
    ///
    /// `turnstile_battery` does not fit: `insert` takes an unsigned weight and the
    /// sketch has no decrement path.
    #[test]
    fn coco_passes_frequency_and_merge_conformance() {
        let stream = stream();
        let truth = truth_of(&stream);
        let spec = FrequencySpec {
            one_sided: false,
            rel_tol: 0.06,
            abs_tol: 25.0,
        };

        conformance::frequency_battery("Coco", CocoAdapter::new, &stream, &truth, spec).assert_ok();
        conformance::merge_equivalence_battery("Coco", CocoAdapter::new, &stream, spec).assert_ok();
    }

    /// `docs/api/api_elastic.md`: "The estimator is one-sided: it never returns
    /// less than the true count." A resident, unflagged flow reports its own votes
    /// exactly; every other flow reads the light layer, so the excess above the
    /// truth is that layer's Count-Min error and the whole tolerance is absolute.
    ///
    /// `turnstile_battery` does not fit: `insert_many` is documented as repeated
    /// positive votes, and there is no decrement path. The battery ingests through
    /// `insert` alone -- overload mode is documented as breaking the one-sided
    /// guarantee, so a battery holding `one_sided: true` must not reach it.
    #[test]
    fn elastic_passes_frequency_and_merge_conformance() {
        let stream = stream();
        let truth = truth_of(&stream);
        // Count-Min's additive bound over the light layer: eps * N with
        // eps = e / cols. Measured worst dense-key excess is 8.
        let spec = FrequencySpec {
            one_sided: true,
            rel_tol: 0.0,
            abs_tol: std::f64::consts::E / 4096.0 * stream.len() as f64,
        };

        conformance::frequency_battery("Elastic", ElasticAdapter::new, &stream, &truth, spec)
            .assert_ok();
        conformance::merge_equivalence_battery("Elastic", ElasticAdapter::new, &stream, spec)
            .assert_ok();
    }

    // -----------------------------------------------------------------------
    // CocoSketch
    // -----------------------------------------------------------------------

    #[test]
    fn coco_over_attribution_bounds_with_disjoint_prefixes() {
        // Table sized well above distinct-key count keeps eviction loss bounded;
        // Coco remains an approximate estimator either way.
        let mut coco = Coco::<asap_sketchlib::DefaultXxHasher>::init_with_size(256, 2);
        let mut truth: HashMap<String, u64> = HashMap::new();
        let mut total = 0u64;
        for i in 0..3000u64 {
            let key = format!("aaa{}", i % 50);
            coco.insert(&key, 7);
            *truth.entry("aaa".to_string()).or_insert(0) += 7;
            total += 7;
            let _ = i;
        }
        for i in 0..2000u64 {
            let key = format!("zzz{}", i % 30);
            coco.insert(&key, 3);
            *truth.entry("zzz".to_string()).or_insert(0) += 3;
            total += 3;
        }

        // Substring matching: "aaa" only matches aaa* buckets, never zzz*.
        let got_aaa = coco.estimate_substring("aaa");
        let true_aaa = truth["aaa"];
        assert!(
            got_aaa >= true_aaa * 3 / 4 && got_aaa <= total,
            "coco 'aaa' estimate {got_aaa} outside [{}, {total}]",
            true_aaa * 3 / 4
        );

        // Exact-match UDF pins down the precise family sum.
        let exact =
            coco.estimate_with_udf("zzz", |full: &str, partial: &str| full.starts_with(partial));
        let true_zzz = truth["zzz"];
        assert!(
            exact >= true_zzz * 3 / 4 && exact <= total,
            "coco 'zzz' estimate {exact} outside [{}, {total}]",
            true_zzz * 3 / 4
        );
    }

    /// CocoSketch attributes every increment to exactly one bucket, so the paper's
    /// point query partitions the stream: summing it over the observed keys returns
    /// the total inserted mass, never more.
    #[test]
    fn coco_point_queries_partition_the_inserted_mass() {
        const COCO_BUCKETS: usize = 128;
        let mut coco = Coco::<asap_sketchlib::DefaultXxHasher>::init_with_size(COCO_BUCKETS, 3);
        let mut truth: HashMap<String, u64> = HashMap::new();
        let mut total = 0u64;

        for key in zipf_u64(40_000, 2_000, 1.2, 4242) {
            let id = format!("key::{key}");
            coco.insert(&id, 2);
            *truth.entry(id).or_insert(0) += 2;
            total += 2;
        }

        let attributed: u64 = truth.keys().map(|k| coco.estimate_key(k)).sum();
        assert_eq!(
            attributed, total,
            "point queries must partition the inserted mass"
        );

        // Heavy keys hold their own bucket, so their estimates track the truth.
        //
        // The **ceiling is derived**: CocoSketch's stochastic-variance-
        // minimizing eviction attributes each increment to exactly one bucket,
        // so a key's estimate can only exceed its count by the mass of the other
        // keys that landed in the same bucket. With `w` buckets per array that
        // expectation is `(total - count) / w`, and Markov at `e` — the same
        // step Count-Min's Theorem 1 takes — gives a per-array ceiling of
        // `count + e (total - count) / w` that holds with probability
        // `1 - 1/e`. The union bound over the ten probed keys is what the
        // failure message quotes.
        //
        // The **floor is empirical and is named as such below**: eviction can
        // take a key's whole bucket at any moment, so nothing forbids a low
        // read, and 0.5x is a measured regression pin (the worst of the ten on
        // this stream and seed is 1.00x, i.e. exact) rather than a bound.
        let mut ranked: Vec<(&String, &u64)> = truth.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (key, count) in ranked.iter().take(10) {
            let est = coco.estimate_key(key);
            let ceiling = **count as f64
                + std::f64::consts::E * (total - **count) as f64 / COCO_BUCKETS as f64;
            assert!(
                est as f64 <= ceiling,
                "coco heavy key {key}: estimate {est} above count + e*(total-count)/w = \
                 {ceiling:.1} (true {count}, total {total}, w={COCO_BUCKETS})"
            );
            assert!(
                est as f64 >= **count as f64 * 0.5,
                "coco heavy key {key}: estimate {est} below the documented empirical floor \
                 of 0.5x its true count {count}; measured worst on this stream is 1.00x"
            );
        }
    }

    // Theorems 3 and 4 are stated for the section 4.2 hardware variant, which this
    // crate does not implement. The test below asserts the paper's own worked 99%
    // figure at the paper's own configuration.

    /// Section 3.2 claims stochastic variance minimization "yields unbiased size
    /// estimation", and Theorem 1 gives the per-bucket update distribution that
    /// makes it so. Unbiasedness is a statement about the mean, not any one run.
    #[test]
    fn coco_point_estimates_are_unbiased_under_heavy_eviction() {
        const TRIALS: usize = 800;
        const BG_KEYS: usize = 200;
        const BG_WEIGHT: u64 = 10;
        const TARGET: u64 = 20;

        // 201 flows over 64 buckets: the target is evicted outright in roughly a
        // third of the runs, and the surviving runs must overshoot to compensate.
        let keys: Vec<String> = (0..BG_KEYS).map(|i| format!("bg::{i}")).collect();
        let mut estimates: Vec<u64> = Vec::with_capacity(TRIALS);

        for _ in 0..TRIALS {
            let mut coco = Coco::<asap_sketchlib::DefaultXxHasher>::init_with_size(32, 2);
            let mut sent = 0u64;
            for (i, key) in keys.iter().enumerate() {
                for _ in 0..BG_WEIGHT {
                    coco.insert(key, 1);
                }
                // spread the target's packets evenly through the background stream
                while sent * (BG_KEYS as u64) < TARGET * (i as u64 + 1) {
                    coco.insert("flow::target", 1);
                    sent += 1;
                }
            }
            while sent < TARGET {
                coco.insert("flow::target", 1);
                sent += 1;
            }
            estimates.push(coco.estimate_key("flow::target"));
        }

        let dropped = estimates.iter().filter(|est| **est == 0).count();
        assert!(
            dropped > TRIALS / 10,
            "the table must actually evict the target sometimes, dropped {dropped}/{TRIALS}"
        );

        let mean = estimates.iter().map(|est| *est as f64).sum::<f64>() / TRIALS as f64;
        // the mean's standard error here is ~0.55, so this band is over 5 sigma
        assert_between(
            mean,
            TARGET as f64 - 3.0,
            TARGET as f64 + 3.0,
            "coco mean point estimate over independent runs",
        );
    }

    /// Theorem 4 bounds how often a flow is recorded at all. Section 5.3 works the
    /// bound at d=2, l=900 for a heavy hitter holding 1% of the traffic and reads
    /// off a 99% recall target; this reproduces that operating point.
    #[test]
    fn coco_recall_meets_the_papers_heavy_hitter_target() {
        const TRIALS: usize = 200;
        const BG_KEYS: usize = 5_000;
        const HEAVY: u64 = 51;
        const WIDTH: usize = 900;
        const DEPTH: usize = 2;

        // Theorem 4: P[Z(e) = 1] >= 1 - (1 + l * f(e) / f_bar(e))^-d. The bound is
        // read off the configuration, so it cannot be the assertion target -- it
        // would sink with the table. It only confirms the setup reaches section
        // 5.3's worked case, and TARGET_RECALL is what the run must clear.
        const TARGET_RECALL: f64 = 0.99;
        let ratio = HEAVY as f64 / BG_KEYS as f64;
        let bound = 1.0 - (1.0 + WIDTH as f64 * ratio).powi(-(DEPTH as i32));
        assert!(
            bound >= TARGET_RECALL,
            "this configuration does not reach the paper's 99% case: bound {bound:.4}"
        );

        let keys: Vec<String> = (0..BG_KEYS).map(|i| format!("bg::{i}")).collect();
        let mut recorded = 0usize;

        for _ in 0..TRIALS {
            let mut coco = Coco::<asap_sketchlib::DefaultXxHasher>::init_with_size(WIDTH, DEPTH);
            let mut sent = 0u64;
            for (i, key) in keys.iter().enumerate() {
                coco.insert(key, 1);
                while sent * (BG_KEYS as u64) < HEAVY * (i as u64 + 1) {
                    coco.insert("flow::heavy", 1);
                    sent += 1;
                }
            }
            while sent < HEAVY {
                coco.insert("flow::heavy", 1);
                sent += 1;
            }
            if coco.recorded_flows().any(|(key, _)| key == "flow::heavy") {
                recorded += 1;
            }
        }

        let recall = recorded as f64 / TRIALS as f64;
        assert!(
            recall >= TARGET_RECALL,
            "coco recall {recall:.4} below the paper's {TARGET_RECALL} target ({recorded}/{TRIALS})"
        );
    }

    // -----------------------------------------------------------------------
    // Elastic
    // -----------------------------------------------------------------------

    #[test]
    fn elastic_tracks_hot_flows() {
        let mut sk = Elastic::<asap_sketchlib::DefaultXxHasher>::init_with_length(64);

        // Three hot flows amid background chatter.
        let mut expected: HashMap<String, i64> = HashMap::new();
        for i in 0..12_000u64 {
            let id = match i % 10 {
                0 => "hot-alpha".to_string(),
                1 => "hot-beta".to_string(),
                2 => "hot-gamma".to_string(),
                _ => format!("bg{}", i % 977),
            };
            sk.insert(id.clone());
            *expected.entry(id).or_insert(0) += 1;
        }

        for hot in ["hot-alpha", "hot-beta", "hot-gamma"] {
            let c = sk.query(hot.to_string()) as i64;
            let t = expected[hot];
            assert_between(
                c as f64,
                t as f64 * 0.80,
                t as f64 * 1.20,
                &format!("elastic flow {hot}"),
            );
        }
    }

    /// The light layer carries every flow the heavy part evicted, so its
    /// dimensions set the error on those flows. Section 4.1 discusses picking the
    /// depth; this checks the choice actually reaches the estimate.
    #[test]
    fn elastic_light_dimensions_set_the_error_on_evicted_flows() {
        const TARGET: i32 = 50;
        const BG_KEYS: usize = 20_000;

        // 8 heavy buckets against 20k distinct flows evicts the target every time.
        let evicted_estimate = |rows: usize, cols: usize| {
            let mut sk =
                Elastic::<asap_sketchlib::DefaultXxHasher>::init_with_dimensions(8, rows, cols);
            for _ in 0..TARGET {
                sk.insert("flow::target".to_string());
            }
            for i in 0..BG_KEYS {
                sk.insert(format!("bg::{i}"));
            }
            assert!(
                !sk.heavy.iter().any(|b| b.flow_id == "flow::target"),
                "the target must be evicted for this to measure the light layer"
            );
            sk.query("flow::target".to_string())
        };

        let narrow = evicted_estimate(1, 64);
        let wide = evicted_estimate(3, 4096);

        // Elastic never underestimates, whatever the light layer costs in error.
        assert!(narrow >= TARGET, "narrow light underestimated: {narrow}");
        assert!(wide >= TARGET, "wide light underestimated: {wide}");

        // Measured 379 against 50: a 1x64 light collides ~7.6x worse than 3x4096.
        assert!(
            wide <= TARGET * 2,
            "a 3x4096 light should stay near the truth, got {wide} for {TARGET}"
        );
        assert!(
            narrow >= wide * 4,
            "light dimensions must reach the estimate: 1x64 gave {narrow}, 3x4096 gave {wide}"
        );
    }

    /// Elastic's estimator is one-sided: a resident flow reports its own votes, a
    /// displaced flow keeps its full size in the light layer, and every other flow
    /// reads a Count-Min over-estimate. No flow may come back short.
    #[test]
    fn elastic_never_underestimates_under_eviction_pressure() {
        // A small heavy table against a wide key space forces repeated takeovers.
        let mut sk = Elastic::<asap_sketchlib::DefaultXxHasher>::init_with_length(16);
        let mut truth: HashMap<String, i64> = HashMap::new();

        for (i, key) in zipf_u64(60_000, 4_000, 1.1, 909).into_iter().enumerate() {
            let id = if i % 500 == 0 {
                "flow::elephant".to_string()
            } else {
                format!("flow::{key}")
            };
            sk.insert(id.clone());
            *truth.entry(id).or_insert(0) += 1;
        }

        let mut evicted_seen = 0usize;
        for (id, count) in &truth {
            let est = sk.query(id.clone()) as i64;
            assert!(
                est >= *count,
                "elastic underestimated {id}: got {est}, true {count}"
            );
            if sk.heavy.iter().any(|b| b.flow_id == *id) {
                continue;
            }
            evicted_seen += 1;
        }
        assert!(
            evicted_seen > 0,
            "the workload must actually push flows out of the heavy part"
        );

        // The elephant is resident in the heavy table, so its estimate is its
        // own vote count plus whatever the light layer adds if it was ever
        // displaced. The ceiling is Count-Min's on the light layer at its own
        // dimensions — `e * (N - f) / cols` — not a written 5%: the light layer
        // *is* a Count-Min, and this is the bound it promises.
        let elephant = sk.query("flow::elephant".to_string()) as i64;
        let true_elephant = truth["flow::elephant"];
        let total: i64 = truth.values().sum();
        let light_ceiling = true_elephant as f64
            + std::f64::consts::E * (total - true_elephant) as f64 / sk.light.cols() as f64;
        assert!(
            elephant >= true_elephant,
            "elastic must never read the elephant low: got {elephant}, true {true_elephant}"
        );
        assert!(
            elephant as f64 <= light_ceiling,
            "elastic elephant {elephant} above true {true_elephant} + e*(N-f)/cols = \
             {light_ceiling:.1} (N={total}, light cols={})",
            sk.light.cols()
        );
    }
}

mod partial_key_and_heavy_maintenance {
    use asap_sketchlib::{Coco, CocoBucket, DataInput, DefaultXxHasher, Elastic, HeavyBucket};
    use std::collections::HashMap;

    const TENANTS: usize = 8;
    const FLOWS_PER_TENANT: usize = 64;
    const REPEATS: u64 = 5;

    fn tenant_of(full: &str) -> &str {
        full.split_once('/').map(|(t, _)| t).unwrap_or(full)
    }

    fn coco_keys() -> Vec<String> {
        let mut keys = Vec::new();
        for t in 0..TENANTS {
            for f in 0..FLOWS_PER_TENANT {
                keys.push(format!("tenant-{t:02}/flow-{f:04}"));
            }
        }
        keys
    }

    fn filled_coco() -> (Coco<DefaultXxHasher>, u64) {
        let mut coco = Coco::<DefaultXxHasher>::init_with_size(2_048, 4);
        let mut inserted = 0u64;
        for (i, key) in coco_keys().iter().enumerate() {
            let weight = REPEATS + (i % 3) as u64;
            coco.insert(key, weight);
            inserted += weight;
        }
        (coco, inserted)
    }

    #[test]
    fn coco_group_by_partitions_exactly_the_mass_the_table_records() {
        let (coco, inserted) = filled_coco();
        let recorded: u64 = coco.recorded_flows().map(|(_, v)| v).sum();
        let groups: HashMap<String, u64> = coco.group_by(tenant_of);
        let grouped: u64 = groups.values().sum();

        assert_eq!(
            grouped, recorded,
            "group_by must fold every recorded flow exactly once"
        );
        assert!(
            recorded <= inserted,
            "the table records {recorded} of {inserted} inserted units, which is more than arrived"
        );
        assert!(
            !groups.is_empty(),
            "a populated table must project onto at least one group"
        );
        for tenant in groups.keys() {
            assert!(
                tenant.starts_with("tenant-"),
                "group key {tenant} is not a tenant projection"
            );
        }
    }

    #[test]
    fn coco_projected_and_udf_partial_key_queries_agree_with_group_by() {
        let (coco, _) = filled_coco();
        let groups: HashMap<String, u64> = coco.group_by(tenant_of);

        for t in 0..TENANTS {
            let tenant = format!("tenant-{t:02}");
            let expected = groups.get(&tenant).copied().unwrap_or(0);
            assert_eq!(
                coco.estimate_projected(&tenant, tenant_of),
                expected,
                "{tenant}: estimate_projected disagreed with group_by"
            );
            assert_eq!(
                coco.estimate_with_udf(&tenant, |full, partial| tenant_of(full) == partial),
                expected,
                "{tenant}: estimate_with_udf disagreed with group_by"
            );
            assert!(
                coco.estimate_substring(&tenant) >= expected,
                "{tenant}: substring containment must collect at least the projected group"
            );
        }

        assert_eq!(
            coco.estimate_projected("tenant-99", tenant_of),
            0,
            "an absent projection carries no mass"
        );
    }

    #[test]
    fn coco_per_key_estimates_sum_to_their_own_group() {
        let (coco, _) = filled_coco();
        let mut per_group: HashMap<String, u64> = HashMap::new();
        for (full, val) in coco.recorded_flows() {
            *per_group.entry(tenant_of(full).to_string()).or_insert(0) += val;
        }
        for (tenant, total) in per_group {
            assert_eq!(
                coco.estimate_projected(&tenant, tenant_of),
                total,
                "{tenant}: projected query disagreed with the recorded flows it covers"
            );
        }
    }

    #[test]
    fn a_coco_bucket_reports_its_own_partial_key_membership() {
        let mut bucket = CocoBucket::new();
        assert!(
            !bucket.is_partial_key("tenant-00"),
            "an empty bucket matches nothing"
        );
        assert!(
            !bucket.is_partial_key_with_udf("tenant-00", |full, partial| full == partial),
            "an empty bucket matches no user-defined predicate either"
        );

        bucket.update_key("tenant-00/flow-0001");
        bucket.add_v(7);
        bucket.add_v(5);

        assert!(bucket.is_partial_key("tenant-00"), "prefix containment");
        assert!(bucket.is_partial_key("flow-0001"), "suffix containment");
        assert!(
            !bucket.is_partial_key("tenant-01"),
            "a different tenant must not match"
        );
        assert!(
            bucket.is_partial_key_with_udf("tenant-00", |full, partial| full
                .split_once('/')
                .map(|(t, _)| t)
                .unwrap_or(full)
                == partial),
            "a projection predicate must match its own tenant"
        );
        assert!(
            !bucket.is_partial_key_with_udf("flow-0001", |full, partial| full
                .split_once('/')
                .map(|(t, _)| t)
                .unwrap_or(full)
                == partial),
            "a projection predicate must reject a non-projection"
        );
        assert_eq!(bucket.val, 12, "add_v must accumulate");
    }

    fn elephant_stream(prefix: &str, elephants: usize, weight: usize) -> Vec<String> {
        let mut out = Vec::new();
        for e in 0..elephants {
            for _ in 0..weight {
                out.push(format!("{prefix}-elephant-{e:03}"));
            }
        }
        for m in 0..2_000 {
            out.push(format!("{prefix}-mouse-{m:04}"));
        }
        out
    }

    fn filled_elastic(stream: &[String]) -> Elastic<DefaultXxHasher> {
        let mut sk = Elastic::<DefaultXxHasher>::init_with_dimensions(64, 3, 1_024);
        for id in stream {
            sk.insert(id.clone());
        }
        sk
    }

    #[test]
    fn elastic_merge_max_never_reads_below_either_side_on_disjoint_flow_sets() {
        let left_stream = elephant_stream("l", 16, 300);
        let right_stream = elephant_stream("r", 16, 300);
        let left = filled_elastic(&left_stream);
        let right = filled_elastic(&right_stream);

        let mut merged = left.clone();
        merged.merge_max(&right);

        for id in left_stream.iter().chain(right_stream.iter()) {
            let own = if id.starts_with("l-") {
                left.query(id.clone())
            } else {
                right.query(id.clone())
            };
            assert!(
                merged.query(id.clone()) >= own,
                "flow {id}: maximum merging read {} below its own side's {own}",
                merged.query(id.clone())
            );
        }
    }

    #[test]
    fn elastic_merge_max_and_sum_merging_agree_on_every_heavy_flow() {
        let left_stream = elephant_stream("l", 16, 300);
        let right_stream = elephant_stream("r", 16, 300);
        let left = filled_elastic(&left_stream);
        let right = filled_elastic(&right_stream);

        let mut by_max = left.clone();
        by_max.merge_max(&right);
        let mut by_sum = left.clone();
        by_sum.merge(&right);

        for (id, _) in by_sum.heavy_hitters(200) {
            let max_side = by_max.query(id.clone());
            let sum_side = by_sum.query(id.clone());
            assert!(
                max_side <= sum_side,
                "flow {id}: maximum merging read {max_side}, above summing at {sum_side}"
            );
            assert!(
                max_side > 0,
                "flow {id}: a heavy flow must survive maximum merging"
            );
        }
    }

    #[test]
    fn elastic_merge_heavy_and_absorb_evicted_carry_a_transferred_resident() {
        let mut target = Elastic::<DefaultXxHasher>::init_with_dimensions(64, 3, 1_024);
        target.merge_heavy("transferred".to_string(), 500, false);
        assert_eq!(
            target.query("transferred".to_string()),
            500,
            "an unflagged transfer must land its whole vote count in the heavy part"
        );

        target.merge_heavy("transferred".to_string(), 0, true);
        assert_eq!(
            target.query("transferred".to_string()),
            500,
            "a zero-vote transfer must not move the estimate"
        );

        let mut flagged = Elastic::<DefaultXxHasher>::init_with_dimensions(64, 3, 1_024);
        flagged.merge_heavy("spilled".to_string(), 400, true);
        flagged.absorb_evicted("spilled".to_string(), 90);
        assert!(
            flagged.query("spilled".to_string()) >= 400,
            "absorbing an eviction must not lose the heavy votes already held"
        );

        let mut light_only = Elastic::<DefaultXxHasher>::init_with_dimensions(64, 3, 1_024);
        light_only.absorb_evicted("mouse".to_string(), 75);
        assert!(
            light_only.query("mouse".to_string()) >= 75,
            "an absorbed flow with no heavy bucket must be readable through the light layer"
        );
    }

    #[test]
    fn elastic_insert_heavy_only_never_writes_the_light_layer() {
        let mut sk = Elastic::<DefaultXxHasher>::init_with_dimensions(8, 3, 256);
        for e in 0..64 {
            for _ in 0..20 {
                sk.insert_heavy_only(format!("flow-{e:03}"));
            }
        }
        for e in 0..64 {
            let id = format!("flow-{e:03}");
            let light = sk.light.estimate(&DataInput::String(id.clone()));
            assert_eq!(
                light, 0,
                "flow {id}: the heavy-only path wrote {light} into the light layer"
            );
        }
    }

    #[test]
    fn elastic_expansion_keeps_every_resident_readable_and_compression_restores_the_width() {
        let stream = elephant_stream("x", 24, 250);
        let mut sk = filled_elastic(&stream);

        let before: Vec<(String, i32)> = sk.heavy_hitters(200);
        assert!(
            !before.is_empty(),
            "the fixture must seat at least one elephant before expanding"
        );
        let full_before = sk.full_bucket_count(100);

        sk.expand_heavy();
        for (id, size) in &before {
            assert!(
                sk.query(id.clone()) >= *size,
                "flow {id}: expansion lost mass, {} < {size}",
                sk.query(id.clone())
            );
        }
        assert!(
            sk.full_bucket_count(100) >= full_before,
            "expansion copies the table, so it cannot reduce the full-bucket count"
        );

        sk.compress_heavy(2);
        for (id, size) in &before {
            assert!(
                sk.query(id.clone()) > 0,
                "flow {id}: compression dropped a flow that held {size}"
            );
        }
    }

    #[test]
    fn elastic_heavy_changes_reports_only_flows_that_moved_past_the_threshold() {
        let first = filled_elastic(&elephant_stream("w", 16, 200));
        let mut second_stream = elephant_stream("w", 16, 200);
        second_stream.extend(std::iter::repeat_n("w-elephant-000".to_string(), 600));
        let second = filled_elastic(&second_stream);

        let changes = second.heavy_changes(&first, 100);
        assert!(
            changes.iter().any(|(id, _, _)| id == "w-elephant-000"),
            "the flow that gained 600 observations must be reported, got {changes:?}"
        );
        assert!(
            changes
                .iter()
                .all(|(_, before, after)| (after - before).abs() > 100),
            "every reported change must clear the threshold it was asked for"
        );
        assert!(
            second.heavy_changes(&first, 10_000).is_empty(),
            "no flow moved by ten thousand observations"
        );
        let unchanged = second.heavy_changes(&second, 0);
        assert!(
            unchanged.is_empty(),
            "a window compared against itself has no heavy change, got {unchanged:?}"
        );
    }

    #[test]
    fn a_heavy_bucket_seats_evicts_and_reports_vacancy() {
        let mut bucket = HeavyBucket::new();
        assert!(bucket.is_vacant(), "a fresh bucket holds no flow");

        bucket.occupy("first".to_string());
        assert!(!bucket.is_vacant(), "an occupied bucket is not vacant");
        assert_eq!(bucket.flow_id, "first");
        assert_eq!(bucket.vote_pos, 1);
        assert_eq!(bucket.vote_neg, 0);
        assert!(!bucket.eviction, "seating a flow raises no eviction flag");

        let mut weighted = HeavyBucket::new();
        weighted.occupy_many("bulk".to_string(), 40);
        assert_eq!(weighted.vote_pos, 40, "occupy_many seats the whole count");

        let evicted = weighted.evict_many("takeover".to_string(), 12);
        assert_eq!(evicted, "bulk", "eviction returns the displaced flow");
        assert_eq!(weighted.flow_id, "takeover");
        assert_eq!(weighted.vote_pos, 12);
        assert_eq!(weighted.vote_neg, 12);
        assert!(weighted.eviction, "a takeover raises the eviction flag");

        let displaced = weighted.evict("single".to_string());
        assert_eq!(displaced, "takeover");
        assert_eq!(weighted.vote_pos, 1);
    }
}

// ---------------------------------------------------------------------------
// The documented input matrix
// ---------------------------------------------------------------------------

/// `tests/TEST_COVERAGE.md` files two more heavy-hitter families under this
/// suite: the counter matrices queried at their heaviest keys, and Space-Saving
/// swept across the capacities its `N / m` ceiling is tabulated at. Both are
/// written against the twelve numbered inputs, so both are swept here one test
/// per input.
///
/// # What "heavy hitter" means here
///
/// The document's definition is positional, not thresholded: the 1% of keys
/// with the highest counts. That set is taken from the exact frequency vector,
/// so the sketches are asked about the keys the workload actually made heavy
/// rather than the ones they happen to think are heavy — a sketch cannot pass
/// by nominating a convenient set.
///
/// # Why the bound is not the table's percentage
///
/// The table's `per-key relative error` column is `e*N/w` divided by `f_HH`,
/// the count of the key at the 1% rank. That is one number for the whole
/// column because it is evaluated at one key. Every heavy hitter above the
/// rank boundary has a larger `f`, so reading the column back as a per-key
/// relative tolerance would hand those keys a budget several times larger than
/// the theorem grants. The additive form `e*(N-f)/w` is asserted instead, at
/// each key's own mass, which is the same theorem evaluated correctly.
mod documented_matrix {
    use super::common::inputs::{KeyInput, key_input, top_percent_keys};
    use super::common::specs::{CountMinSpec, CountSketchSpec, SIMULTANEOUS_LEVEL, Tally};
    use super::common::{
        FixMat1, FixMat2, FixMat3, FixMat4, FixMat5, FixMat6, FixMat7, FixMat8, FixMat9, FixMat10,
        FixMat11, FixMat12, FixMat13, FixMat14, FixMat15, FreqTruth,
    };
    use super::*;

    use asap_sketchlib::{Count, CountMin, FastPath, RegularPath, Vector2D};

    /// The document's `(row, column)` grid.
    const DOC_ROWS: [usize; 3] = [3, 5, 7];
    const DOC_COLS: [usize; 5] = [2_048, 4_096, 8_192, 16_384, 32_768];

    /// The document's heavy-hitter definition.
    const HEAVY_HITTER_SHARE: f64 = 0.01;

    /// Count-Min's contract restricted to the heavy hitters: one-sidedness with
    /// no tolerance, the union-bounded additive excess over the probed set with
    /// no tolerance, and the marginal `e*(N-f)/w` as a rate pin at the
    /// theorem's own `e^-d`.
    fn assert_countmin_heavy_hitters<F>(
        spec: &CountMinSpec,
        truth: &FreqTruth,
        heavy: &[(i64, i64)],
        label: &str,
        estimate: F,
        context: &str,
    ) where
        F: Fn(i64) -> f64,
    {
        let total = truth.total() as f64;
        let probed = heavy.len();
        let factor = spec.simultaneous_factor(probed, SIMULTANEOUS_LEVEL);
        let mut one_sided = Tally::default();
        let mut simultaneous = Tally::default();
        let mut marginal = Tally::default();
        for (key, count) in heavy {
            let f = *count as f64;
            let est = estimate(*key);
            one_sided.record(est >= f, || {
                format!("heavy hitter {key}: est {est} < true {f} (Count-Min must never read low)")
            });
            let simul = spec.simultaneous_bound(total, f, probed, SIMULTANEOUS_LEVEL);
            simultaneous.record(est - f <= simul, || {
                format!(
                    "heavy hitter {key}: excess {:.1} > b*(N-f)/w = {simul:.1} with \
                     b={factor:.2} (true {f}, est {est})",
                    est - f
                )
            });
            let marg = spec.marginal_bound(total, f);
            marginal.record(est - f <= marg, || {
                format!(
                    "heavy hitter {key}: excess {:.1} > e*(N-f)/w = {marg:.1} (true {f}, \
                     est {est})",
                    est - f
                )
            });
        }
        let ctx = format!(
            "{context}; rows={} cols={} N={total:.0} over the top {probed} keys \
             (1% of {} distinct), simultaneous b={factor:.3}",
            spec.rows,
            spec.cols,
            truth.distinct(),
        );
        one_sided.assert_none(&format!("{label} / heavy hitters one-sided"), &ctx);
        simultaneous.assert_none(&format!("{label} / heavy hitters simultaneous"), &ctx);
        marginal.assert_rate_at_most(
            &format!("{label} / heavy hitters marginal e*(N-f)/w"),
            spec.marginal_failure(),
            &ctx,
        );
    }

    /// Count Sketch's L2 contract restricted to the heavy hitters. The residual
    /// norm is recomputed per key, which matters far more here than on a
    /// uniform key set: a heavy hitter carries a large share of `F2`, so
    /// `||f_-i||_2` is much smaller than `||f||_2` and the budget is
    /// correspondingly tighter.
    fn assert_countsketch_heavy_hitters<F>(
        spec: &CountSketchSpec,
        truth: &FreqTruth,
        heavy: &[(i64, i64)],
        label: &str,
        estimate: F,
        context: &str,
    ) where
        F: Fn(i64) -> f64,
    {
        let f2 = truth.f2();
        let probed = heavy.len();
        let kappa = spec.simultaneous_kappa(probed, SIMULTANEOUS_LEVEL);
        let mut simultaneous = Tally::default();
        let mut marginal = Tally::default();
        for (key, count) in heavy {
            let f = *count as f64;
            let residual_l2 = (f2 - f * f).max(0.0).sqrt();
            let est = estimate(*key);
            let err = (est - f).abs();

            let simul = spec.scale_at(kappa, residual_l2);
            simultaneous.record(err <= simul, || {
                format!(
                    "heavy hitter {key}: |{est:.1} - {f}| = {err:.1} > \
                     sqrt(kappa/w)*||f_-i||_2 = {simul:.1} at kappa={kappa:.1}"
                )
            });
            let marg = spec.marginal_scale(residual_l2);
            marginal.record(err <= marg, || {
                format!(
                    "heavy hitter {key}: |{est:.1} - {f}| = {err:.1} > \
                     sqrt(3/w)*||f_-i||_2 = {marg:.1}"
                )
            });
        }
        let ctx = format!(
            "{context}; rows={} cols={} over the top {probed} keys (1% of {} distinct), \
             F2={f2:.3e}, simultaneous kappa={kappa:.1}",
            spec.rows,
            spec.cols,
            truth.distinct(),
        );
        simultaneous.assert_none(&format!("{label} / heavy hitters simultaneous L2"), &ctx);
        marginal.assert_rate_at_most(
            &format!("{label} / heavy hitters marginal sqrt(3/w)*||f_-i||_2"),
            spec.marginal_failure(),
            &ctx,
        );
    }

    fn countmin_heavy_hitter_matrix(id: u8) {
        let input = key_input(id);
        let truth = input.truth();
        let heavy = top_percent_keys(&truth, HEAVY_HITTER_SHARE);
        let context = input.context();

        for rows in DOC_ROWS {
            for cols in DOC_COLS {
                let mut regular =
                    CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);
                let mut fast = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(rows, cols);
                for key in &input.keys {
                    let d = input.data(*key);
                    regular.insert(&d);
                    fast.insert(&d);
                }
                let spec = CountMinSpec::new(rows, cols);
                assert_countmin_heavy_hitters(
                    &spec,
                    &truth,
                    &heavy,
                    &format!("CountMin<Vector2D<i64>, RegularPath> {rows}x{cols}"),
                    |k| regular.estimate(&input.data(k)) as f64,
                    &context,
                );
                assert_countmin_heavy_hitters(
                    &spec,
                    &truth,
                    &heavy,
                    &format!("CountMin<Vector2D<i64>, FastPath> {rows}x{cols}"),
                    |k| fast.estimate(&input.data(k)) as f64,
                    &context,
                );
            }
        }

        macro_rules! assert_fixed_matrix {
            ($($mat:ident => ($rows:expr, $cols:expr)),* $(,)?) => {
                $({
                    let mut regular = CountMin::<$mat, RegularPath>::from_storage($mat::default());
                    let mut fast = CountMin::<$mat, FastPath>::from_storage($mat::default());
                    for key in &input.keys {
                        let d = input.data(*key);
                        regular.insert(&d);
                        fast.insert(&d);
                    }
                    let spec = CountMinSpec::new($rows, $cols);
                    assert_countmin_heavy_hitters(
                        &spec,
                        &truth,
                        &heavy,
                        concat!("CountMin<", stringify!($mat), ", RegularPath>"),
                        |k| regular.estimate(&input.data(k)) as f64,
                        &context,
                    );
                    assert_countmin_heavy_hitters(
                        &spec,
                        &truth,
                        &heavy,
                        concat!("CountMin<", stringify!($mat), ", FastPath>"),
                        |k| fast.estimate(&input.data(k)) as f64,
                        &context,
                    );
                })*
            };
        }

        assert_fixed_matrix! {
            FixMat1 => (3, 2048),
            FixMat2 => (5, 2048),
            FixMat3 => (7, 2048),
            FixMat4 => (3, 4096),
            FixMat5 => (5, 4096),
            FixMat6 => (7, 4096),
            FixMat7 => (3, 8192),
            FixMat8 => (5, 8192),
            FixMat9 => (7, 8192),
            FixMat10 => (3, 16384),
            FixMat11 => (5, 16384),
            FixMat12 => (7, 16384),
            FixMat13 => (3, 32768),
            FixMat14 => (5, 32768),
            FixMat15 => (7, 32768),
        }
    }

    fn countsketch_heavy_hitter_matrix(id: u8) {
        let input = key_input(id);
        let truth = input.truth();
        let heavy = top_percent_keys(&truth, HEAVY_HITTER_SHARE);
        let context = input.context();

        for rows in DOC_ROWS {
            for cols in DOC_COLS {
                let mut regular = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);
                let mut fast = Count::<Vector2D<i64>, FastPath>::with_dimensions(rows, cols);
                for key in &input.keys {
                    let d = input.data(*key);
                    regular.insert(&d);
                    fast.insert(&d);
                }
                let spec = CountSketchSpec::new(rows, cols);
                assert_countsketch_heavy_hitters(
                    &spec,
                    &truth,
                    &heavy,
                    &format!("Count<Vector2D<i64>, RegularPath> {rows}x{cols}"),
                    |k| regular.estimate(&input.data(k)),
                    &context,
                );
                assert_countsketch_heavy_hitters(
                    &spec,
                    &truth,
                    &heavy,
                    &format!("Count<Vector2D<i64>, FastPath> {rows}x{cols}"),
                    |k| fast.estimate(&input.data(k)),
                    &context,
                );
            }
        }

        macro_rules! assert_fixed_matrix {
            ($($mat:ident => ($rows:expr, $cols:expr)),* $(,)?) => {
                $({
                    let mut regular = Count::<$mat, RegularPath>::from_storage($mat::default());
                    let mut fast = Count::<$mat, FastPath>::from_storage($mat::default());
                    for key in &input.keys {
                        let d = input.data(*key);
                        regular.insert(&d);
                        fast.insert(&d);
                    }
                    let spec = CountSketchSpec::new($rows, $cols);
                    assert_countsketch_heavy_hitters(
                        &spec,
                        &truth,
                        &heavy,
                        concat!("Count<", stringify!($mat), ", RegularPath>"),
                        |k| regular.estimate(&input.data(k)),
                        &context,
                    );
                    assert_countsketch_heavy_hitters(
                        &spec,
                        &truth,
                        &heavy,
                        concat!("Count<", stringify!($mat), ", FastPath>"),
                        |k| fast.estimate(&input.data(k)),
                        &context,
                    );
                })*
            };
        }

        assert_fixed_matrix! {
            FixMat1 => (3, 2048),
            FixMat2 => (5, 2048),
            FixMat3 => (7, 2048),
            FixMat4 => (3, 4096),
            FixMat5 => (5, 4096),
            FixMat6 => (7, 4096),
            FixMat7 => (3, 8192),
            FixMat8 => (5, 8192),
            FixMat9 => (7, 8192),
            FixMat10 => (3, 16384),
            FixMat11 => (5, 16384),
            FixMat12 => (7, 16384),
            FixMat13 => (3, 32768),
            FixMat14 => (5, 32768),
            FixMat15 => (7, 32768),
        }
    }

    /// Space-Saving's counter guarantee at every capacity the document
    /// tabulates, over one documented input.
    ///
    /// The table's cells are `N / m`: the largest over-estimate the summary can
    /// carry at that capacity, which is `min_count`'s own ceiling. Both halves
    /// of the sandwich are asserted for every monitored key, and the ceiling
    /// itself is asserted against `min_count`, so a summary that stayed inside
    /// the band by dropping the keys that would have violated it still fails.
    fn space_saving_documented_capacities(id: u8) {
        const CAPACITIES: [usize; 5] = [2, 8, 64, SPACE_SAVING_DEFAULT_CAPACITY, 2_048];

        let input = key_input(id);
        let truth = input.truth();
        let n = truth.total() as f64;

        for capacity in CAPACITIES {
            let mut summary: SpaceSaving = SpaceSaving::with_capacity(capacity);
            for key in &input.keys {
                summary.insert(&input.data(*key));
            }
            let ceiling = n / capacity as f64;
            let context = format!(
                "{} capacity={capacity} N={n:.0} documented ceiling N/m={ceiling:.1}",
                input.context()
            );

            assert!(
                summary.min_count() as f64 <= ceiling,
                "min_count {} exceeds the documented N/m ceiling. {context}",
                summary.min_count()
            );

            let mut monitored = 0usize;
            let mut sandwich = Tally::default();
            let mut ceiling_tally = Tally::default();
            for (key, count) in truth.pairs() {
                let est = summary.estimate(&input.data(key));
                if est == 0 {
                    continue;
                }
                monitored += 1;
                let f = count as u64;
                sandwich.record(est >= f, || {
                    format!("key {key}: est {est} < true {f}, a monitored key must never read low")
                });
                ceiling_tally.record((est.saturating_sub(f)) as f64 <= ceiling, || {
                    format!(
                        "key {key}: over-estimate {} exceeds the documented N/m = {ceiling:.1} \
                         (true {f}, est {est})",
                        est.saturating_sub(f)
                    )
                });
            }
            assert_eq!(
                monitored,
                summary.len(),
                "every monitored counter must belong to a key the stream carried. {context}"
            );
            sandwich.assert_none("Space-Saving / monitored key floor", &context);
            ceiling_tally.assert_none("Space-Saving / N/m over-estimate ceiling", &context);
        }
    }

    /// Elastic at the documented `length 64`, with the document's hot-flow
    /// workload laid over one of the skewed inputs.
    ///
    /// The document specifies both the background stream (the numbered input)
    /// and the heavy part (three flows at roughly a tenth of the stream each),
    /// so the two are combined: the input supplies the background flows, and
    /// three synthetic flows are injected at 10% of the final total each. The
    /// bound is the document's own 20% per hot flow, which is empirical
    /// because the split between the heavy and light parts is data dependent.
    fn elastic_documented_hot_flows(input: &KeyInput) {
        use asap_sketchlib::{DefaultXxHasher, Elastic};

        const HOT: [&str; 3] = ["hot-alpha", "hot-beta", "hot-gamma"];
        const HOT_SHARE: f64 = 0.10;

        // Each hot flow gets a tenth of the *final* stream, so the background
        // supplies the remaining 70%.
        let background = input.keys.len();
        let per_hot =
            (background as f64 * HOT_SHARE / (1.0 - HOT.len() as f64 * HOT_SHARE)).round() as usize;

        let mut sk = Elastic::<DefaultXxHasher>::init_with_length(64);
        let mut expected: HashMap<String, i64> = HashMap::new();
        let mut hot_cursor = 0usize;
        let stride = background / (per_hot * HOT.len()).max(1);
        for (i, key) in input.keys.iter().enumerate() {
            let id = format!("bg::{key}");
            sk.insert(id.clone());
            *expected.entry(id).or_insert(0) += 1;
            // Interleave the hot flows through the stream rather than
            // appending them, so they compete for buckets the whole way.
            if stride > 0 && i % stride == 0 {
                let hot = HOT[hot_cursor % HOT.len()].to_string();
                hot_cursor += 1;
                sk.insert(hot.clone());
                *expected.entry(hot).or_insert(0) += 1;
            }
        }

        let context = format!(
            "{} with {} hot flows at ~{}% each, elastic length=64",
            input.context(),
            HOT.len(),
            (HOT_SHARE * 100.0) as u32
        );
        for hot in HOT {
            let truth = expected[hot] as f64;
            let est = sk.query(hot.to_string()) as f64;
            let rel = ((est - truth) / truth).abs();
            assert!(
                rel <= 0.20,
                "elastic hot flow {hot}: est {est} vs true {truth} is {:.2}% off, past the \
                 documented 20%. {context}",
                rel * 100.0
            );
        }
    }

    macro_rules! documented_heavy_hitter_matrix {
        ($($cm:ident, $cs:ident, $ss:ident => $id:literal;)*) => {
            $(
                #[test]
                fn $cm() {
                    countmin_heavy_hitter_matrix($id);
                }

                #[test]
                fn $cs() {
                    countsketch_heavy_hitter_matrix($id);
                }

                #[test]
                fn $ss() {
                    space_saving_documented_capacities($id);
                }
            )*
        };
    }

    documented_heavy_hitter_matrix! {
        countmin_input_1_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_1_heavy_hitters_hold_the_l2_bound,
        space_saving_input_1_holds_its_capacity_ceiling => 1;
        countmin_input_2_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_2_heavy_hitters_hold_the_l2_bound,
        space_saving_input_2_holds_its_capacity_ceiling => 2;
        countmin_input_3_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_3_heavy_hitters_hold_the_l2_bound,
        space_saving_input_3_holds_its_capacity_ceiling => 3;
        countmin_input_4_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_4_heavy_hitters_hold_the_l2_bound,
        space_saving_input_4_holds_its_capacity_ceiling => 4;
        countmin_input_5_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_5_heavy_hitters_hold_the_l2_bound,
        space_saving_input_5_holds_its_capacity_ceiling => 5;
        countmin_input_6_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_6_heavy_hitters_hold_the_l2_bound,
        space_saving_input_6_holds_its_capacity_ceiling => 6;
        countmin_input_7_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_7_heavy_hitters_hold_the_l2_bound,
        space_saving_input_7_holds_its_capacity_ceiling => 7;
        countmin_input_8_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_8_heavy_hitters_hold_the_l2_bound,
        space_saving_input_8_holds_its_capacity_ceiling => 8;
        countmin_input_9_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_9_heavy_hitters_hold_the_l2_bound,
        space_saving_input_9_holds_its_capacity_ceiling => 9;
        countmin_input_10_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_10_heavy_hitters_hold_the_l2_bound,
        space_saving_input_10_holds_its_capacity_ceiling => 10;
        countmin_input_11_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_11_heavy_hitters_hold_the_l2_bound,
        space_saving_input_11_holds_its_capacity_ceiling => 11;
        countmin_input_12_heavy_hitters_hold_the_count_min_bound,
        countsketch_input_12_heavy_hitters_hold_the_l2_bound,
        space_saving_input_12_holds_its_capacity_ceiling => 12;
    }

    macro_rules! documented_elastic_inputs {
        ($($name:ident => $id:literal;)*) => {
            $(
                #[test]
                fn $name() {
                    elastic_documented_hot_flows(&key_input($id));
                }
            )*
        };
    }

    documented_elastic_inputs! {
        elastic_input_3_hot_flows_stay_within_twenty_percent => 3;
        elastic_input_4_hot_flows_stay_within_twenty_percent => 4;
        elastic_input_5_hot_flows_stay_within_twenty_percent => 5;
        elastic_input_6_hot_flows_stay_within_twenty_percent => 6;
        elastic_input_9_hot_flows_stay_within_twenty_percent => 9;
        elastic_input_10_hot_flows_stay_within_twenty_percent => 10;
        elastic_input_11_hot_flows_stay_within_twenty_percent => 11;
        elastic_input_12_hot_flows_stay_within_twenty_percent => 12;
    }
}
