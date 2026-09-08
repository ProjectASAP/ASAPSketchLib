//! E2E frequency-sketch pipelines on synthetic streams with exact ground
//! truth.
//!
//! Two different error metrics live in this file and must not be confused:
//!
//! - **Count-Min** is one-sided and *additive*: `est >= f` always, and the
//!   excess is bounded by `e * (N - f) / w` with probability `1 - e^-d`.
//! - **Count Sketch** is two-sided and *L2*: the error is
//!   `sqrt(kappa / w) * ||f_-i||_2`, rank-independent, and has nothing to do
//!   with `eps * N`. A Count Sketch checked against `e/w * N` is not being
//!   checked against its own theorem — that bound is enormously looser on a
//!   skewed stream, and passing it says almost nothing.
//!
//! Both bounds live in `common::specs` alongside the binomial acceptance rules
//! that turn a per-key failure probability into a pass/fail decision.

mod common;

use common::specs::{CountMinSpec, CountSketchSpec, SecondMomentSpec, Tally};
use common::{
    FixMat1, FixMat2, FixMat3, FixMat4, FixMat5, FixMat6, FixMat7, FixMat8, FixMat9, FixMat10,
    FixMat11, FixMat12, FixMat13, FixMat14, FixMat15,
};
use common::{FreqTruth, uniform_u64, zipf_u64};
use std::collections::HashMap;

use asap_sketchlib::message_pack_format::portable::countminsketch::CountMinSketch;
use asap_sketchlib::message_pack_format::portable::countsketch::CountSketch;
use asap_sketchlib::{
    Count, CountL2HH, CountMin, DataInput, DefaultXxHasher, FastPath, FoldCMS, FoldCS, RegularPath,
    Vector2D,
};

// ----------------------------------------------------------------- CountMin

/// Count-Min's own theorem end to end: the one-sided guarantee on every key,
/// the additive `e*(N-f)/w` excess under the binomial acceptance rule, and
/// exact equality between a shard-merged sketch and a single-pass one.
#[test]
fn countmin_fast_path_zipf_conforms_to_the_count_min_model_and_merges_shards_exactly() {
    const ROWS: usize = 4;
    const COLS: usize = 4096;
    const STREAM_SEED: u64 = 1001;

    let stream = zipf_u64(100_000, 8192, 1.1, STREAM_SEED);
    let mut single = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let (mut shard_a, mut shard_b, mut shard_c) = (
        CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS),
        CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS),
        CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS),
    );
    let mut truth = FreqTruth::default();
    for (i, k) in stream.iter().enumerate() {
        let d = DataInput::I64(*k as i64);
        truth.observe(*k as i64);
        single.insert(&d);
        match i % 3 {
            0 => shard_a.insert(&d),
            1 => shard_b.insert(&d),
            _ => shard_c.insert(&d),
        }
    }
    shard_b.merge(&shard_c);
    shard_a.merge(&shard_b);

    let spec = CountMinSpec::new(ROWS, COLS);
    let context = format!("zipf(1.1) domain=8192 n=100000 stream_seed={STREAM_SEED}");
    spec.assert_contract(
        "CountMin<Vector2D<i64>, FastPath> single-pass",
        &truth,
        |k| single.estimate(&DataInput::I64(k)) as f64,
        &context,
    );
    spec.assert_contract(
        "CountMin<Vector2D<i64>, FastPath> 3-way shard merge",
        &truth,
        |k| shard_a.estimate(&DataInput::I64(k)) as f64,
        &context,
    );

    // Merging counter matrices is exact addition, so a merged sketch is not
    // merely close to the single-pass one — it is identical on every key.
    let mut merge_equality = Tally::default();
    for (k, _) in truth.pairs() {
        let a = single.estimate(&DataInput::I64(k));
        let b = shard_a.estimate(&DataInput::I64(k));
        merge_equality.record(a == b, || {
            format!("key {k}: single-pass {a} != shard-merged {b}")
        });
    }
    merge_equality.assert_none("CountMin merge equality", &context);
}

#[test]
fn countmin_vecbased_error_bound() {
    const ROWS: [usize; 3] = [3, 5, 7];
    const COLS: [usize; 5] = [2048, 4096, 8192, 16384, 32768];
    const STREAM_SEED: u64 = 1002;

    for r in ROWS {
        for c in COLS {
            let mut reg = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(r, c);
            let mut fast = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(r, c);
            let mut truth = FreqTruth::default();
            let stream = zipf_u64(20_000, 512, 1.2, STREAM_SEED);
            for k in &stream {
                truth.observe(*k as i64);
                reg.insert(&DataInput::I64(*k as i64));
                fast.insert(&DataInput::I64(*k as i64));
            }
            let spec = CountMinSpec::new(r, c);
            let context = format!("zipf(1.2) domain=512 n=20000 stream_seed={STREAM_SEED}");
            spec.assert_contract(
                "CountMin<Vector2D<i64>, RegularPath>",
                &truth,
                |k| reg.estimate(&DataInput::I64(k)) as f64,
                &context,
            );
            spec.assert_contract(
                "CountMin<Vector2D<i32>, FastPath>",
                &truth,
                |k| fast.estimate(&DataInput::I64(k)) as f64,
                &context,
            );
        }
    }
}

#[test]
fn count_vecbased_error_bound() {
    const ROWS: [usize; 3] = [3, 5, 7];
    const COLS: [usize; 5] = [2048, 4096, 8192, 16384, 32768];
    const STREAM_SEED: u64 = 1002;

    for r in ROWS {
        for c in COLS {
            let mut reg = Count::<Vector2D<i64>, RegularPath>::with_dimensions(r, c);
            let mut fast = Count::<Vector2D<i64>, FastPath>::with_dimensions(r, c);
            let mut truth = FreqTruth::default();
            let stream = zipf_u64(20_000, 512, 1.2, STREAM_SEED);
            for k in &stream {
                truth.observe(*k as i64);
                reg.insert(&DataInput::I64(*k as i64));
                fast.insert(&DataInput::I64(*k as i64));
            }
            let spec = CountSketchSpec::new(r, c);
            let context = format!("zipf(1.2) domain=512 n=20000 stream_seed={STREAM_SEED}");
            spec.assert_contract(
                "Count<Vector2D<i64>, RegularPath>",
                &truth,
                |k| reg.estimate(&DataInput::I64(k)),
                &context,
            );
            spec.assert_contract(
                "Count<Vector2D<i32>, FastPath>",
                &truth,
                |k| fast.estimate(&DataInput::I64(k)),
                &context,
            );
        }
    }
}

#[test]
fn countmin_matbased_error_bound() {
    const STREAM_SEED: u64 = 1002;

    let stream = zipf_u64(20_000, 512, 1.2, STREAM_SEED);
    let mut truth = FreqTruth::default();
    for k in &stream {
        truth.observe(*k as i64);
    }
    let context = format!("zipf(1.2) domain=512 n=20000 stream_seed={STREAM_SEED}");

    // Each `FixMatN` is a distinct type carrying its own compile-time
    // dimensions, so the sweep has to be unrolled rather than looped. The
    // spec's `(rows, cols)` must match the matrix's, or the assertions check
    // the wrong theorem.
    macro_rules! assert_fixed_matrix {
        ($($mat:ident => ($rows:expr, $cols:expr)),* $(,)?) => {
            $({
                let mut reg = CountMin::<$mat, RegularPath>::from_storage($mat::default());
                let mut fast = CountMin::<$mat, FastPath>::from_storage($mat::default());
                for k in &stream {
                    reg.insert(&DataInput::I64(*k as i64));
                    fast.insert(&DataInput::I64(*k as i64));
                }
                let spec = CountMinSpec::new($rows, $cols);
                spec.assert_contract(
                    concat!("CountMin<", stringify!($mat), ", RegularPath>"),
                    &truth,
                    |k| reg.estimate(&DataInput::I64(k)) as f64,
                    &context,
                );
                spec.assert_contract(
                    concat!("CountMin<", stringify!($mat), ", FastPath>"),
                    &truth,
                    |k| fast.estimate(&DataInput::I64(k)) as f64,
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

#[test]
fn count_matbased_error_bound() {
    const STREAM_SEED: u64 = 1002;

    let stream = zipf_u64(20_000, 512, 1.2, STREAM_SEED);
    let mut truth = FreqTruth::default();
    for k in &stream {
        truth.observe(*k as i64);
    }
    let context = format!("zipf(1.2) domain=512 n=20000 stream_seed={STREAM_SEED}");

    // Each `FixMatN` is a distinct type carrying its own compile-time
    // dimensions, so the sweep has to be unrolled rather than looped. The
    // spec's `(rows, cols)` must match the matrix's, or the assertions check
    // the wrong theorem.
    macro_rules! assert_fixed_matrix {
        ($($mat:ident => ($rows:expr, $cols:expr)),* $(,)?) => {
            $({
                let mut reg = Count::<$mat, RegularPath>::from_storage($mat::default());
                let mut fast = Count::<$mat, FastPath>::from_storage($mat::default());
                for k in &stream {
                    reg.insert(&DataInput::I64(*k as i64));
                    fast.insert(&DataInput::I64(*k as i64));
                }
                let spec = CountSketchSpec::new($rows, $cols);
                spec.assert_contract(
                    concat!("CountMin<", stringify!($mat), ", RegularPath>"),
                    &truth,
                    |k| reg.estimate(&DataInput::I64(k)) as f64,
                    &context,
                );
                spec.assert_contract(
                    concat!("CountMin<", stringify!($mat), ", FastPath>"),
                    &truth,
                    |k| fast.estimate(&DataInput::I64(k)) as f64,
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

/// Bounded integer draws mapped onto distinct f64 values in `[100, 1000)`,
/// identified by bit pattern so exact counts stay comparable.
fn uniform_f64_key(v: u64) -> i64 {
    (100.0 + v as f64 * (900.0 / 4096.0)).to_bits() as i64
}

fn f64_input(key: i64) -> DataInput<'static> {
    DataInput::F64(f64::from_bits(key as u64))
}

fn u64_input(key: i64) -> DataInput<'static> {
    DataInput::U64(key as u64)
}

/// A named key stream paired with the `DataInput` constructor for its keys.
type BoundStream = (&'static str, Vec<i64>, fn(i64) -> DataInput<'static>);

const BOUND_ROWS: usize = 3;
const BOUND_COLS: usize = 4096;
const BOUND_N: usize = 120_000;
/// Independent key populations per shape. The library fixes its hash seed
/// table, so a test cannot resample the hash function; what it *can* resample
/// is the key population, which under the random-oracle model the sketches
/// assume yields an independent collision configuration per trial. This is
/// stated plainly rather than dressed up as seed independence.
const BOUND_TRIALS: u64 = 3;

/// Zipf over `u64` keys and uniform over `f64` keys for one trial, exercising
/// both `DataInput` hashing paths. Each trial uses a disjoint key domain so
/// its collisions are unrelated to the previous trial's.
fn bound_streams(trial: u64) -> [BoundStream; 2] {
    [
        (
            "zipf/u64",
            zipf_u64(BOUND_N, 8192, 1.1, 1005 + trial * 977)
                .into_iter()
                .map(|v| v as i64 + (trial as i64) * 100_000)
                .collect(),
            u64_input as fn(i64) -> DataInput<'static>,
        ),
        (
            "uniform/f64",
            uniform_u64(BOUND_N, 4096, 1006 + trial * 977)
                .into_iter()
                .map(|v| uniform_f64_key(v + trial * 8192))
                .collect(),
            f64_input as fn(i64) -> DataInput<'static>,
        ),
    ]
}

/// Count-Min Theorem 1 on the **RegularPath**, over independent key
/// populations.
///
/// # Why the two hash paths are separate tests
///
/// Theorem 1's `b^-d` comes from the `d` rows failing independently, which
/// needs the `d` row hashes to be independent draws from a suitable family.
/// The two paths get there differently:
///
/// - `RegularPath` makes one hash call per row with a distinct seed from the
///   table, so the rows are separate hash functions. Under the standard
///   assumption that XXH3-with-distinct-seeds behaves as an independent family
///   — stated here rather than left implicit — the theorem applies as written.
/// - `FastPath` makes a **single** call and slices each row's column index out
///   of bit fields of that one 128-bit output. The rows are then deterministic
///   functions of one hash value, not independent hash functions. Treating
///   disjoint bit fields of a good hash as independent is a modelling
///   assumption about XXH3's avalanche, not something the theorem grants.
///
/// So this test carries the theorem and
/// `countmin_fast_path_conforms_to_the_count_min_model` carries the same
/// arithmetic under an honest label. The coverage matrix classifies them
/// `theorem` and `asymptotic model` respectively.
#[test]
fn countmin_regular_path_satisfies_the_count_min_theorem() {
    let spec = CountMinSpec::new(BOUND_ROWS, BOUND_COLS);
    for trial in 0..BOUND_TRIALS {
        for (name, keys, to_input) in bound_streams(trial) {
            let mut truth = FreqTruth::default();
            let mut regular =
                CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(BOUND_ROWS, BOUND_COLS);
            for k in &keys {
                truth.observe(*k);
                regular.insert(&to_input(*k));
            }
            let context = format!(
                "{name} trial={trial} n={BOUND_N} (see bound_streams for stream seeds); \
                 RegularPath hashes each row with its own seed from the table"
            );
            spec.assert_contract(
                &format!("CountMin RegularPath {name}"),
                &truth,
                |k| regular.estimate(&to_input(k)) as f64,
                &context,
            );
        }
    }
}

/// The same arithmetic on the **FastPath**, which is a model check rather than
/// a theorem.
///
/// `FastPath` derives all `d` column indices from bit fields of one 128-bit
/// hash. Every bound evaluated here is Count-Min's, at the sketch's own
/// dimensions — nothing is widened — but the `b^-d` it is quoted at assumes
/// row independence that a single hash does not literally provide. What the
/// test establishes is that the fast path *conforms to the same model* as the
/// independently-hashed path, which is the useful statement and is how the
/// coverage matrix labels it.
#[test]
fn countmin_fast_path_conforms_to_the_count_min_model() {
    let spec = CountMinSpec::new(BOUND_ROWS, BOUND_COLS);
    for trial in 0..BOUND_TRIALS {
        for (name, keys, to_input) in bound_streams(trial) {
            let mut truth = FreqTruth::default();
            let mut fast =
                CountMin::<Vector2D<i32>, FastPath>::with_dimensions(BOUND_ROWS, BOUND_COLS);
            for k in &keys {
                truth.observe(*k);
                fast.insert(&to_input(*k));
            }
            let context = format!(
                "{name} trial={trial} n={BOUND_N} (see bound_streams for stream seeds); \
                 FastPath slices all {BOUND_ROWS} row indices out of one 128-bit hash, so \
                 row independence is a model, not a hypothesis the theorem supplies"
            );
            spec.assert_contract(
                &format!("CountMin FastPath {name}"),
                &truth,
                |k| fast.estimate(&to_input(k)) as f64,
                &context,
            );
        }
    }
}

/// Count Sketch's L2 bound on both insert paths and both `DataInput` hash
/// domains, over several independent key populations.
///
/// The two paths are tallied separately and the coverage matrix classifies them
/// differently: `RegularPath` hashes each row with its own seed, so the median
/// amplification's independence hypothesis is met (under the usual assumption
/// about the hash family); `FastPath` slices every row index out of one
/// 128-bit hash, so row independence there is a **model**. The arithmetic is
/// identical — nothing is widened for the fast path — but the label is not.
/// See `countmin_fast_path_conforms_to_the_count_min_model`.
///
/// # Two assertions, two acceptance rules
///
/// The library's hash is fixed by the sketch's type parameter, so every sketch
/// here draws from the same hash function and the keys inside one sketch share
/// it. They are therefore *not* independent Bernoulli trials, and a binomial
/// over keys assumes exactly the independence that does not hold.
///
/// What is asserted instead:
///
/// 1. the **simultaneous** bound, whose `kappa` is raised until a union bound
///    over the whole probed key set leaves `SIMULTANEOUS_LEVEL` overall. That
///    needs no independence at all and tolerates zero violations;
/// 2. the **marginal** bound at `kappa = 3` as a violation-*rate* pin against
///    the theorem's own per-key failure probability — a regression pin on a
///    fixed realisation, not a tail test.
#[test]
fn countsketch_both_paths_meet_the_l2_median_bound() {
    let spec = CountSketchSpec::new(BOUND_ROWS, BOUND_COLS);
    // One pair of tallies per (shape, path): all trials pool into a single
    // acceptance rule, so the decision is made on the whole population rather
    // than on whichever trial happened to look worst.
    let mut tallies: HashMap<String, (Tally, Tally)> = HashMap::new();
    for trial in 0..BOUND_TRIALS {
        for (name, keys, to_input) in bound_streams(trial) {
            let mut truth = FreqTruth::default();
            let mut regular = asap_sketchlib::Count::<Vector2D<i32>, RegularPath>::with_dimensions(
                BOUND_ROWS, BOUND_COLS,
            );
            let mut fast = asap_sketchlib::Count::<Vector2D<i32>, FastPath>::with_dimensions(
                BOUND_ROWS, BOUND_COLS,
            );
            for k in &keys {
                let d = to_input(*k);
                truth.observe(*k);
                regular.insert(&d);
                fast.insert(&d);
            }
            {
                let (simul, marg) = tallies.entry(format!("{name}/RegularPath")).or_default();
                spec.tally_into(simul, marg, &truth, |k| regular.estimate(&to_input(k)));
            }
            {
                let (simul, marg) = tallies.entry(format!("{name}/FastPath")).or_default();
                spec.tally_into(simul, marg, &truth, |k| fast.estimate(&to_input(k)));
            }
        }
    }
    let mut labels: Vec<String> = tallies.keys().cloned().collect();
    labels.sort();
    for label in labels {
        let (simul, marg) = tallies.remove(&label).expect("label just enumerated");
        let context = format!(
            "rows={BOUND_ROWS} cols={BOUND_COLS} marginal kappa={} n={BOUND_N} \
             trials={BOUND_TRIALS} (independent key populations, fixed library hash seed \
             — the keys of one sketch share it and are not independent trials)",
            spec.kappa
        );
        simul.assert_none(&format!("Count {label} / simultaneous L2 bound"), &context);
        marg.assert_rate_at_most(
            &format!("Count {label} / marginal L2 median bound"),
            spec.marginal_failure(),
            &context,
        );
    }
}

/// Count Sketch's error is *rank-independent*: a cold key carries the same
/// absolute error band as the hottest one, which is exactly what separates it
/// from Count-Min. Documented empirical regression, not a theorem: the
/// theorem bounds each key individually and says nothing about how the
/// measured mean error compares across frequency strata.
///
/// Band source: measured on this exact stream (zipf(1.1), domain 8192,
/// n=200_000, seed 1007, rows=5, cols=4096) where the mean |error| per decile
/// spans 20.6 to 26.5 — a spread of 1.29x. The 3x ceiling below leaves ample
/// room for run-to-run movement while still failing loudly if the error ever
/// starts tracking frequency the way Count-Min's does (on this stream
/// Count-Min's per-decile mean error spans more than 40x).
#[test]
fn countsketch_error_stays_rank_independent_within_the_documented_empirical_band() {
    const ROWS: usize = 5;
    const COLS: usize = 4096;
    const STREAM_SEED: u64 = 1007;
    const DECILES: usize = 10;

    let stream = zipf_u64(200_000, 8192, 1.1, STREAM_SEED);
    let mut truth = FreqTruth::default();
    let mut cs = asap_sketchlib::Count::<Vector2D<i64>, RegularPath>::with_dimensions(ROWS, COLS);
    for k in &stream {
        truth.observe(*k as i64);
        cs.insert(&DataInput::I64(*k as i64));
    }

    let mut pairs = truth.pairs();
    pairs.sort_by_key(|(_, c)| *c);
    let per = pairs.len() / DECILES;
    let means: Vec<f64> = (0..DECILES)
        .map(|d| {
            let slice = &pairs[d * per..(d + 1) * per];
            slice
                .iter()
                .map(|(k, c)| (cs.estimate(&DataInput::I64(*k)) - *c as f64).abs())
                .sum::<f64>()
                / slice.len() as f64
        })
        .collect();
    let lo = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = means.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        hi <= lo * 3.0,
        "Count Sketch mean |error| per frequency decile spans {lo:.1}..{hi:.1} ({:.2}x), \
         beyond the documented 3x empirical band — the error has started tracking key \
         frequency, which Count Sketch's L2 guarantee says it must not. \
         stream_seed={STREAM_SEED} rows={ROWS} cols={COLS} deciles={means:?}",
        hi / lo
    );
}

// -------------------------------------------------------------- CountL2HH

/// CountL2HH is a Count Sketch carrying a running F2, so its point estimates
/// obey the Count Sketch L2 bound and its `get_l2_sqr` tracks the exact F2.
#[test]
fn countl2hh_weighted_turnstile_satisfies_the_l2_median_bound() {
    const ROWS: usize = 4;
    const COLS: usize = 2048;
    const HASH_SEED_IDX: usize = 11;
    const STREAM_SEED: u64 = 1005;

    let mut sk = CountL2HH::<DefaultXxHasher>::with_dimensions_and_seed(ROWS, COLS, HASH_SEED_IDX);
    let mut truth = FreqTruth::default();
    let stream = zipf_u64(30_000, 512, 1.3, STREAM_SEED);
    for (i, k) in stream.iter().enumerate() {
        let w = 1 + (i % 5) as i64;
        sk.fast_insert_with_count(&DataInput::I64(*k as i64), w);
        truth.observe_weighted(*k as i64, w);
    }
    // Turnstile decrement on the hottest key must track through the median path.
    let hot = truth.top_k(1)[0].0;
    sk.fast_insert_with_count(&DataInput::I64(hot), -10);
    truth.observe_weighted(hot, -10);

    let context = format!(
        "zipf(1.3) domain=512 n=30000 weighted 1..5 with one -10 decrement, \
         stream_seed={STREAM_SEED} hash_seed_idx={HASH_SEED_IDX}"
    );
    CountSketchSpec::new(ROWS, COLS).assert_contract(
        "CountL2HH point estimate",
        &truth,
        |k| sk.fast_get_est(&DataInput::I64(k)),
        &context,
    );

    // F2 comes from the median of the per-row counter sums, i.e. the AMS
    // tug-of-war estimator over this sketch's own matrix — an estimate with a
    // real bound, not an exactly maintained total.
    let f2_spec = SecondMomentSpec::new(ROWS, COLS);
    if let Err(detail) = f2_spec.check(sk.get_l2_sqr(), truth.f2()) {
        panic!("CountL2HH F2: {detail}\n  context: {context}");
    }
}

// ------------------------------------------- FoldCMS / FoldCS counts + merge

#[test]
fn fold_cms_foldcs_counts_and_hierarchical_merge() {
    // Weighted per-key counts stay exact on sparse dims.
    let mut a = FoldCMS::<DefaultXxHasher>::new(3, 2048, 0, 32);
    let mut b = FoldCMS::<DefaultXxHasher>::new(3, 2048, 0, 32);
    for _ in 0..100 {
        a.insert(&DataInput::Str("alpha"), 2);
        b.insert(&DataInput::Str("alpha"), 2);
        b.insert(&DataInput::Str("beta"), 1);
    }
    assert_eq!(a.query(&DataInput::Str("alpha")), 200);
    assert_eq!(b.query(&DataInput::Str("alpha")), 200);
    assert_eq!(b.query(&DataInput::Str("beta")), 100);

    // Same-level merge sums disjoint contributions.
    a.merge_same_level(&b);
    assert_eq!(a.query(&DataInput::Str("alpha")), 400);
    assert_eq!(a.query(&DataInput::Str("beta")), 100);

    // Signed weighted updates through FoldCS.
    let mut fs = FoldCS::<DefaultXxHasher>::new(3, 2048, 0, 32);
    fs.insert(&DataInput::Str("gamma"), 60);
    fs.insert(&DataInput::Str("gamma"), -20);
    assert_eq!(fs.query(&DataInput::Str("gamma")), 40);

    // Hierarchical merge of level-matched sketches preserves totals.
    let s1 = build_fold_cms("s1", 30);
    let s2 = build_fold_cms("s2", 45);
    let merged = FoldCMS::hierarchical_merge(&[s1, s2]);
    assert_eq!(merged.query(&DataInput::Str("s1")), 90);
    assert_eq!(merged.query(&DataInput::Str("s2")), 135);
}

fn build_fold_cms(key: &'static str, n: usize) -> FoldCMS<DefaultXxHasher> {
    let mut s = FoldCMS::<DefaultXxHasher>::new(3, 2048, 0, 32);
    for _ in 0..n {
        s.insert(&DataInput::Str(key), 3);
    }
    s
}

/// Sixteen sub-window sketches folded to a quarter width, then hierarchically
/// merged. Folding halves the width `fold_level` times, so the merged sketches
/// answer at `FULL_COLS >> FOLD_LEVEL` columns — and that, not the unfolded
/// width, is the `w` each theorem is evaluated at. FoldCMS keeps Count-Min's
/// additive bound; FoldCS keeps Count Sketch's L2 bound.
#[test]
fn folded_sketches_keep_their_own_bounds_through_a_sixteen_way_merge() {
    const ROWS: usize = 3;
    const FULL_COLS: usize = 4096;
    const FOLD_LEVEL: u32 = 4;
    const TOP_K: usize = 20;
    const WINDOWS: usize = 16;
    // Enough per window that folding to 256 columns is a real test, while an
    // unoptimised `cargo test` run stays affordable. The bounds are computed
    // from the exact truth, so a shorter stream narrows nothing.
    const N: usize = 240_000;
    const STREAM_SEED: u64 = 1009;

    let stream = zipf_u64(N, 10_000, 1.1, STREAM_SEED);
    let per_window = N / WINDOWS;
    let mut truth = FreqTruth::default();
    let mut cms_windows = Vec::with_capacity(WINDOWS);
    let mut cs_windows = Vec::with_capacity(WINDOWS);

    for w in 0..WINDOWS {
        let mut cms = FoldCMS::<DefaultXxHasher>::new(ROWS, FULL_COLS, FOLD_LEVEL, TOP_K);
        let mut cs = FoldCS::<DefaultXxHasher>::new(ROWS, FULL_COLS, FOLD_LEVEL, TOP_K);
        for &v in &stream[w * per_window..(w + 1) * per_window] {
            let d = DataInput::U64(v);
            cms.insert(&d, 1);
            cs.insert(&d, 1);
            truth.observe(v as i64);
        }
        cms_windows.push(cms);
        cs_windows.push(cs);
    }

    let cms_merged = FoldCMS::hierarchical_merge(&cms_windows);
    let cs_merged = FoldCS::hierarchical_merge(&cs_windows);
    let folded_cols = FULL_COLS >> FOLD_LEVEL;
    let context = format!(
        "zipf(1.1) domain=10000 n={N} stream_seed={STREAM_SEED}, {WINDOWS} windows folded \
         {FOLD_LEVEL} levels: {FULL_COLS} -> {folded_cols} columns"
    );

    CountMinSpec::new(ROWS, folded_cols).assert_contract(
        "FoldCMS after a 16-way hierarchical merge",
        &truth,
        |k| cms_merged.query(&DataInput::U64(k as u64)) as f64,
        &context,
    );
    CountSketchSpec::new(ROWS, folded_cols).assert_contract(
        "FoldCS after a 16-way hierarchical merge",
        &truth,
        |k| cs_merged.query(&DataInput::U64(k as u64)) as f64,
        &context,
    );
}

// ------------------------------------------------------ Portable wire twins

/// The portable wire twins answer over string keys and must satisfy the same
/// two theorems as their core counterparts — Count-Min additive, Count Sketch
/// L2 — with no extra slack for being on the wire side.
#[test]
fn portable_cms_and_cs_string_keys_satisfy_their_own_bounds() {
    const CM_ROWS: usize = 3;
    const CS_ROWS: usize = 5;
    const COLS: usize = 4096;
    const STREAM_SEED: u64 = 1006;

    let stream = zipf_u64(50_000, 2048, 1.1, STREAM_SEED);
    // The portable types key on strings; `FreqTruth` keys on i64. The stream's
    // u64 values are the identity behind both, so `k -> "k{k}"` is injective
    // and the two truths agree key for key.
    let mut truth = FreqTruth::default();
    let mut pcs = CountMinSketch::new(CM_ROWS, COLS);
    let mut pcss = CountSketch::new(CS_ROWS, COLS);
    for k in &stream {
        truth.observe(*k as i64);
        let key = format!("k{k}");
        pcs.update(&key, 1.0);
        pcss.update(&key, 1.0);
    }
    let context = format!("zipf(1.1) domain=2048 n=50000 string keys, stream_seed={STREAM_SEED}");

    CountMinSpec::new(CM_ROWS, COLS).assert_contract(
        "portable CountMinSketch",
        &truth,
        |k| pcs.estimate(&format!("k{k}")),
        &context,
    );
    CountSketchSpec::new(CS_ROWS, COLS).assert_contract(
        "portable CountSketch",
        &truth,
        |k| pcss.estimate(&format!("k{k}")),
        &context,
    );
}

// ---------------------------------------------------------------------------
// The documented input matrix
// ---------------------------------------------------------------------------
//
// `tests/TEST_COVERAGE.md` writes this suite's Count-Min and Count Sketch
// tables as twelve numbered inputs crossed with a `(row, column)` grid, on both
// storage backends and both hash paths. The tests above sweep the grid on one
// hand-rolled stream; these sweep the grid on the streams the table is actually
// written against, one test per input so a failure names the row it came from.
//
// The tolerance is never the table's percentage column read back as a literal:
// that column *is* `e*(N-f)/w` (Count-Min) and `sqrt(e/w)*||f||_2` (Count
// Sketch) evaluated at `f(k) = 1`, and `CountMinSpec` / `CountSketchSpec`
// evaluate the same expressions at every key's own mass. Asserting the derived
// percentage instead would hand every key above the first the wrong budget.

/// The `(row, column)` grid both tables sweep.
const DOC_ROWS: [usize; 3] = [3, 5, 7];
const DOC_COLS: [usize; 5] = [2_048, 4_096, 8_192, 16_384, 32_768];

/// Every documented `(row, column)` cell on the growable backend, both paths.
fn countmin_documented_vector_grid(
    input: &common::inputs::KeyInput,
    truth: &FreqTruth,
    context: &str,
) {
    for rows in DOC_ROWS {
        for cols in DOC_COLS {
            let mut regular = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);
            let mut fast = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(rows, cols);
            for key in &input.keys {
                let d = input.data(*key);
                regular.insert(&d);
                fast.insert(&d);
            }
            let spec = CountMinSpec::new(rows, cols);
            spec.assert_contract(
                &format!("CountMin<Vector2D<i64>, RegularPath> {rows}x{cols}"),
                truth,
                |k| regular.estimate(&input.data(k)) as f64,
                context,
            );
            spec.assert_contract(
                &format!("CountMin<Vector2D<i64>, FastPath> {rows}x{cols}"),
                truth,
                |k| fast.estimate(&input.data(k)) as f64,
                context,
            );
        }
    }
}

/// The same grid on the compile-time matrices. Each `FixMatN` is its own type,
/// so the sweep is unrolled and the spec's dimensions must match the matrix's.
fn countmin_documented_fixed_grid(
    input: &common::inputs::KeyInput,
    truth: &FreqTruth,
    context: &str,
) {
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
                spec.assert_contract(
                    concat!("CountMin<", stringify!($mat), ", RegularPath>"),
                    truth,
                    |k| regular.estimate(&input.data(k)) as f64,
                    context,
                );
                spec.assert_contract(
                    concat!("CountMin<", stringify!($mat), ", FastPath>"),
                    truth,
                    |k| fast.estimate(&input.data(k)) as f64,
                    context,
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

fn countmin_documented_matrix(id: u8) {
    let input = common::inputs::key_input(id);
    let truth = input.truth();
    let context = input.context();
    countmin_documented_vector_grid(&input, &truth, &context);
    countmin_documented_fixed_grid(&input, &truth, &context);
}

fn countsketch_documented_vector_grid(
    input: &common::inputs::KeyInput,
    truth: &FreqTruth,
    context: &str,
) {
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
            spec.assert_contract(
                &format!("Count<Vector2D<i64>, RegularPath> {rows}x{cols}"),
                truth,
                |k| regular.estimate(&input.data(k)),
                context,
            );
            spec.assert_contract(
                &format!("Count<Vector2D<i64>, FastPath> {rows}x{cols}"),
                truth,
                |k| fast.estimate(&input.data(k)),
                context,
            );
        }
    }
}

fn countsketch_documented_fixed_grid(
    input: &common::inputs::KeyInput,
    truth: &FreqTruth,
    context: &str,
) {
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
                spec.assert_contract(
                    concat!("Count<", stringify!($mat), ", RegularPath>"),
                    truth,
                    |k| regular.estimate(&input.data(k)),
                    context,
                );
                spec.assert_contract(
                    concat!("Count<", stringify!($mat), ", FastPath>"),
                    truth,
                    |k| fast.estimate(&input.data(k)),
                    context,
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

fn countsketch_documented_matrix(id: u8) {
    let input = common::inputs::key_input(id);
    let truth = input.truth();
    let context = input.context();
    countsketch_documented_vector_grid(&input, &truth, &context);
    countsketch_documented_fixed_grid(&input, &truth, &context);
}

macro_rules! documented_frequency_matrix {
    ($($cm:ident, $cs:ident => $id:literal;)*) => {
        $(
            #[test]
            fn $cm() {
                countmin_documented_matrix($id);
            }

            #[test]
            fn $cs() {
                countsketch_documented_matrix($id);
            }
        )*
    };
}

documented_frequency_matrix! {
    countmin_input_1_grid_satisfies_the_count_min_model,
    countsketch_input_1_grid_satisfies_the_l2_bound => 1;
    countmin_input_2_grid_satisfies_the_count_min_model,
    countsketch_input_2_grid_satisfies_the_l2_bound => 2;
    countmin_input_3_grid_satisfies_the_count_min_model,
    countsketch_input_3_grid_satisfies_the_l2_bound => 3;
    countmin_input_4_grid_satisfies_the_count_min_model,
    countsketch_input_4_grid_satisfies_the_l2_bound => 4;
    countmin_input_5_grid_satisfies_the_count_min_model,
    countsketch_input_5_grid_satisfies_the_l2_bound => 5;
    countmin_input_6_grid_satisfies_the_count_min_model,
    countsketch_input_6_grid_satisfies_the_l2_bound => 6;
    countmin_input_7_grid_satisfies_the_count_min_model,
    countsketch_input_7_grid_satisfies_the_l2_bound => 7;
    countmin_input_8_grid_satisfies_the_count_min_model,
    countsketch_input_8_grid_satisfies_the_l2_bound => 8;
    countmin_input_9_grid_satisfies_the_count_min_model,
    countsketch_input_9_grid_satisfies_the_l2_bound => 9;
    countmin_input_10_grid_satisfies_the_count_min_model,
    countsketch_input_10_grid_satisfies_the_l2_bound => 10;
    countmin_input_11_grid_satisfies_the_count_min_model,
    countsketch_input_11_grid_satisfies_the_l2_bound => 11;
    countmin_input_12_grid_satisfies_the_count_min_model,
    countsketch_input_12_grid_satisfies_the_l2_bound => 12;
}

/// CountL2HH over the eight skewed documented inputs, at the documented
/// `row 4, col 2048`, with weighted arrivals and one negative update.
///
/// Two assertions come straight from the coverage document — the hottest key
/// within 2% after its decrement, and `F2` within 10% — and both are checked
/// alongside the sketch's own theorems (`CountSketchSpec` per key,
/// `SecondMomentSpec` for `F2`), so a regression that stays inside the
/// document's round numbers but leaves the estimator's model still fails.
fn countl2hh_documented_input(id: u8) {
    const ROWS: usize = 4;
    const COLS: usize = 2_048;
    const HASH_SEED_IDX: usize = 11;

    let input = common::inputs::key_input(id);
    let mut sk = CountL2HH::<DefaultXxHasher>::with_dimensions_and_seed(ROWS, COLS, HASH_SEED_IDX);
    let mut truth = FreqTruth::default();
    for (i, key) in input.keys.iter().enumerate() {
        let w = 1 + (i % 5) as i64;
        sk.fast_insert_with_count(&input.data(*key), w);
        truth.observe_weighted(*key, w);
    }

    // The decrement lands on the hottest key, so it has to survive the median
    // path rather than being absorbed by a counter nobody reads.
    let hot = truth.top_k(1)[0].0;
    let decrement = -(truth.get(hot) / 10);
    sk.fast_insert_with_count(&input.data(hot), decrement);
    truth.observe_weighted(hot, decrement);

    let context = format!(
        "{} weighted 1..5 with a {decrement} decrement on the hottest key, \
         rows={ROWS} cols={COLS} hash_seed_idx={HASH_SEED_IDX}",
        input.context()
    );

    CountSketchSpec::new(ROWS, COLS).assert_contract(
        "CountL2HH point estimate",
        &truth,
        |k| sk.fast_get_est(&input.data(k)),
        &context,
    );

    let hot_truth = truth.get(hot) as f64;
    let hot_est = sk.fast_get_est(&input.data(hot));
    let hot_rel = ((hot_est - hot_truth) / hot_truth).abs();
    assert!(
        hot_rel <= 0.02,
        "CountL2HH hottest key after the decrement: est {hot_est:.1} vs true {hot_truth} \
         is {:.3}% off, past the documented 2%. {context}",
        hot_rel * 100.0
    );

    let f2_truth = truth.f2();
    let f2_est = sk.get_l2_sqr();
    let f2_rel = ((f2_est - f2_truth) / f2_truth).abs();
    assert!(
        f2_rel <= 0.10,
        "CountL2HH F2: est {f2_est:.3e} vs true {f2_truth:.3e} is {:.2}% off, past the \
         documented 10%. {context}",
        f2_rel * 100.0
    );

    let f2_spec = SecondMomentSpec::new(ROWS, COLS);
    if let Err(detail) = f2_spec.check(f2_est, f2_truth) {
        panic!("CountL2HH F2 model: {detail}\n  context: {context}");
    }
}

macro_rules! documented_countl2hh_inputs {
    ($($name:ident => $id:literal;)*) => {
        $(
            #[test]
            fn $name() {
                countl2hh_documented_input($id);
            }
        )*
    };
}

documented_countl2hh_inputs! {
    countl2hh_input_3_weighted_turnstile_holds_its_bounds => 3;
    countl2hh_input_4_weighted_turnstile_holds_its_bounds => 4;
    countl2hh_input_5_weighted_turnstile_holds_its_bounds => 5;
    countl2hh_input_6_weighted_turnstile_holds_its_bounds => 6;
    countl2hh_input_9_weighted_turnstile_holds_its_bounds => 9;
    countl2hh_input_10_weighted_turnstile_holds_its_bounds => 10;
    countl2hh_input_11_weighted_turnstile_holds_its_bounds => 11;
    countl2hh_input_12_weighted_turnstile_holds_its_bounds => 12;
}
