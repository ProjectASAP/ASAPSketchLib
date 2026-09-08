//! Every built-in `(storage, hashing path)` instance of the four matrix-backed
//! frequency families, held to its own family's bound.
//!
//! The bound is the same on both hashing paths and is evaluated at each
//! instance's own dimensions, but the two paths do not earn the same label:
//! `RegularPath` hashes each row with its own seed from the table, so the row
//! independence the `b^-d` / median amplification rests on is supplied by the
//! hash family; `FastPath` slices every row index out of one 128-bit hash, so
//! that independence is a modelling assumption about the hash's avalanche. The
//! coverage matrix classifies the fast-path rows `asymptotic model` for this
//! reason. Nothing is widened for them — see
//! `countmin_fast_path_conforms_to_the_count_min_model` in `e2e_frequency.rs`
//! for the split written out in full.
//!
//! The deep, production-sized accuracy runs live in `e2e_frequency.rs` on each
//! family's headline instance. This file is about *coverage of the instance
//! matrix*: a storage backend that silently mis-indexes, a counter width that
//! wraps, or a fast path that reads cells the regular path never wrote, would
//! all pass a suite that only ever exercised `Vector2D<i64>`.
//!
//! Each instance therefore gets a medium stream and three checks:
//!
//! 1. its family's error bound (`CountMinSpec` or `CountSketchSpec`) evaluated
//!    at the dimensions the instance itself reports, so the fixed 5x2048 and
//!    3x4096 layouts are judged at their own `w` rather than a borrowed one;
//! 2. agreement between the regular and fast paths on the same storage, which
//!    is exact — they are two ways of deriving the same cells;
//! 3. shard merge, which for counter matrices is exact addition.
//!
//! Counter-width overflow is covered separately, once per width, because that
//! is where `i32`, `i64` and `i128` actually differ.

mod common;

use common::specs::{CountMinSpec, CountSketchSpec, SIMULTANEOUS_LEVEL, Tally};
use common::{FreqTruth, zipf_u64};

use asap_sketchlib::{
    CMSHeap, CSHeap, Count, CountMin, DataInput, DefaultMatrixI32, DefaultMatrixI64,
    DefaultMatrixI128, FastPath, FixedMatrix, HeapItem, QuickMatrixI64, QuickMatrixI128,
    RegularPath, Vector2D, cs_heap_count, hash_for_matrix,
};

/// Medium stream: large enough that collisions are statistically meaningful at
/// 2048 and 4096 columns, small enough that seventy instances stay quick.
const N: usize = 40_000;
const DOMAIN: usize = 4_096;
const STREAM_SEED: u64 = 0x10BE_C700;

fn stream_and_truth() -> (Vec<u64>, FreqTruth) {
    let stream = zipf_u64(N, DOMAIN, 1.1, STREAM_SEED);
    let mut truth = FreqTruth::default();
    for k in &stream {
        truth.observe(*k as i64);
    }
    (stream, truth)
}

fn context(label: &str, rows: usize, cols: usize) -> String {
    format!("{label} rows={rows} cols={cols} zipf(1.1) domain={DOMAIN} n={N} seed={STREAM_SEED:#x}")
}

/// Counter values reach the assertions as `f64`; the concrete counter type is
/// known at each macro expansion, so this stays a plain conversion rather than
/// a lossy generic cast scattered through the tests.
trait CounterValue: Copy {
    fn as_f64(self) -> f64;
}
impl CounterValue for i32 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}
impl CounterValue for i64 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}
impl CounterValue for i128 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}
impl CounterValue for f64 {
    fn as_f64(self) -> f64 {
        self
    }
}

// ----------------------------------------------------------------- CountMin

/// One `CountMin<$storage, $path>` instance: build it from `Default` (which is
/// what fixes each fixed-layout storage's dimensions), run the stream, and
/// assert Count-Min's own contract at the dimensions it reports.
macro_rules! countmin_instance {
    ($storage:ty, $path:ty, $stream:expr, $truth:expr) => {{
        let mut single = CountMin::<$storage, $path>::default();
        let mut left = CountMin::<$storage, $path>::default();
        let mut right = CountMin::<$storage, $path>::default();
        for (i, k) in $stream.iter().enumerate() {
            let d = DataInput::U64(*k);
            single.insert(&d);
            if i % 2 == 0 {
                left.insert(&d);
            } else {
                right.insert(&d);
            }
        }
        left.merge(&right);

        let (rows, cols) = (single.rows(), single.cols());
        let label = concat!(
            "CountMin<",
            stringify!($storage),
            ", ",
            stringify!($path),
            ">"
        );
        let ctx = context(label, rows, cols);
        let spec = CountMinSpec::new(rows, cols);
        spec.assert_contract(
            label,
            $truth,
            |k| single.estimate(&DataInput::U64(k as u64)).as_f64(),
            &ctx,
        );
        spec.assert_contract(
            &format!("{label} shard merge"),
            $truth,
            |k| left.estimate(&DataInput::U64(k as u64)).as_f64(),
            &ctx,
        );

        // Merging counter matrices is exact addition, so the merged sketch is
        // identical to the single-pass one, not merely close.
        let mut equality = Tally::default();
        for (k, _) in $truth.pairs() {
            let a = single.estimate(&DataInput::U64(k as u64)).as_f64();
            let b = left.estimate(&DataInput::U64(k as u64)).as_f64();
            equality.record(a == b, || format!("key {k}: single {a} merged {b}"));
        }
        equality.assert_none(&format!("{label} merge equality"), &ctx);
        single
    }};
}

/// On a workload small enough to be collision-free, *both* paths must return
/// exact counts — every probed cell belongs to the queried key alone.
///
/// This, and not estimate-for-estimate equality, is the cross-path contract.
/// `RegularPath` makes one hash call per row with `seed_list[r]`; `FastPath`
/// makes a single call with `seed_list[0]` and slices row bits out of it.
/// They are different hash functions, so they place keys in different columns
/// and legitimately disagree wherever either one collides. Asserting equality
/// on a colliding stream would be asserting a coincidence: on a 40k-update
/// Zipf stream at 3x4096 the two paths differ on roughly a third of keys, all
/// of them inside Count-Min's bound.
macro_rules! assert_both_paths_exact_without_collisions {
    ($sketch:ident, $storage:ty, $keys:expr) => {{
        let mut regular = $sketch::<$storage, RegularPath>::default();
        let mut fast = $sketch::<$storage, FastPath>::default();
        let mut truth = FreqTruth::default();
        for (i, k) in $keys.iter().enumerate() {
            for _ in 0..(i + 1) * 10 {
                let d = DataInput::U64(*k);
                regular.insert(&d);
                fast.insert(&d);
                truth.observe(*k as i64);
            }
        }
        let label = concat!(
            stringify!($sketch),
            "<",
            stringify!($storage),
            "> collision-free exactness on both paths"
        );
        let ctx = context(label, regular.rows(), regular.cols());
        let mut tally = Tally::default();
        for (k, c) in truth.pairs() {
            let r = regular.estimate(&DataInput::U64(k as u64)).as_f64();
            let f = fast.estimate(&DataInput::U64(k as u64)).as_f64();
            tally.record(r == c as f64 && f == c as f64, || {
                format!("key {k}: true {c}, regular {r}, fast {f}")
            });
        }
        tally.assert_none(label, &ctx);
    }};
}

/// Eight well-separated keys in a 2048- or 4096-column grid: the chance that
/// any pair collides in every row is negligible, and with fixed seeds the
/// outcome is deterministic, so a failure here means a storage backend is
/// mis-indexing rather than that the stream was unlucky.
const COLLISION_FREE_KEYS: [u64; 8] = [
    1,
    7,
    4_242,
    90_210,
    1_000_003,
    2_147_483_647,
    4_294_967_311,
    9_007_199_254_740_993,
];

macro_rules! countmin_matrix_test {
    ($name:ident, $path:ty, $( $storage:ty ),+ $(,)?) => {
        #[test]
        fn $name() {
            let (stream, truth) = stream_and_truth();
            $( { let _ = countmin_instance!($storage, $path, stream, &truth); } )+
        }
    };
}

countmin_matrix_test!(
    countmin_regular_path_instances_satisfy_the_count_min_bound,
    RegularPath,
    Vector2D<i32>,
    Vector2D<i64>,
    Vector2D<i128>,
    Vector2D<f64>,
    FixedMatrix,
    DefaultMatrixI32,
    QuickMatrixI64,
    QuickMatrixI128,
    DefaultMatrixI64,
    DefaultMatrixI128,
);

countmin_matrix_test!(
    countmin_fast_path_instances_conform_to_the_count_min_model,
    FastPath,
    Vector2D<i32>,
    Vector2D<i64>,
    Vector2D<i128>,
    Vector2D<f64>,
    FixedMatrix,
    DefaultMatrixI32,
    QuickMatrixI64,
    QuickMatrixI128,
    DefaultMatrixI64,
    DefaultMatrixI128,
);

#[test]
fn countmin_both_paths_are_exact_on_a_collision_free_workload() {
    assert_both_paths_exact_without_collisions!(CountMin, Vector2D<i32>, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, Vector2D<i64>, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, Vector2D<i128>, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, Vector2D<f64>, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, FixedMatrix, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, DefaultMatrixI32, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, QuickMatrixI64, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, QuickMatrixI128, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, DefaultMatrixI64, COLLISION_FREE_KEYS);
    assert_both_paths_exact_without_collisions!(CountMin, DefaultMatrixI128, COLLISION_FREE_KEYS);
}

// -------------------------------------------------------------- CountSketch

macro_rules! count_instance {
    ($storage:ty, $path:ty, $stream:expr, $truth:expr) => {{
        let mut single = Count::<$storage, $path>::default();
        let mut left = Count::<$storage, $path>::default();
        let mut right = Count::<$storage, $path>::default();
        for (i, k) in $stream.iter().enumerate() {
            let d = DataInput::U64(*k);
            single.insert(&d);
            if i % 2 == 0 {
                left.insert(&d);
            } else {
                right.insert(&d);
            }
        }
        left.merge(&right);

        let (rows, cols) = (single.rows(), single.cols());
        let label = concat!("Count<", stringify!($storage), ", ", stringify!($path), ">");
        let ctx = context(label, rows, cols);
        let spec = CountSketchSpec::new(rows, cols);
        spec.assert_contract(
            label,
            $truth,
            |k| single.estimate(&DataInput::U64(k as u64)),
            &ctx,
        );
        spec.assert_contract(
            &format!("{label} shard merge"),
            $truth,
            |k| left.estimate(&DataInput::U64(k as u64)),
            &ctx,
        );

        // Signed counters still add exactly on merge.
        let mut equality = Tally::default();
        for (k, _) in $truth.pairs() {
            let a = single.estimate(&DataInput::U64(k as u64));
            let b = left.estimate(&DataInput::U64(k as u64));
            equality.record(a == b, || format!("key {k}: single {a} merged {b}"));
        }
        equality.assert_none(&format!("{label} merge equality"), &ctx);
    }};
}

macro_rules! count_matrix_test {
    ($name:ident, $path:ty, $( $storage:ty ),+ $(,)?) => {
        #[test]
        fn $name() {
            let (stream, truth) = stream_and_truth();
            $( count_instance!($storage, $path, stream, &truth); )+
        }
    };
}

count_matrix_test!(
    countsketch_regular_path_instances_satisfy_the_l2_bound,
    RegularPath,
    Vector2D<i32>,
    Vector2D<i64>,
    Vector2D<i128>,
    FixedMatrix,
    DefaultMatrixI32,
    QuickMatrixI64,
    QuickMatrixI128,
    DefaultMatrixI64,
    DefaultMatrixI128,
);

count_matrix_test!(
    countsketch_fast_path_instances_conform_to_the_l2_model,
    FastPath,
    Vector2D<i32>,
    Vector2D<i64>,
    Vector2D<i128>,
    FixedMatrix,
    DefaultMatrixI32,
    QuickMatrixI64,
    QuickMatrixI128,
    DefaultMatrixI64,
    DefaultMatrixI128,
);

/// The same collision-free exactness contract for the signed family: with no
/// collisions every row reports `sign * sign * f = f`, so the median is exact
/// on both paths.
#[test]
fn countsketch_both_paths_are_exact_on_a_collision_free_workload() {
    macro_rules! exact {
        ($storage:ty) => {{
            let mut regular = Count::<$storage, RegularPath>::default();
            let mut fast = Count::<$storage, FastPath>::default();
            let mut truth = FreqTruth::default();
            for (i, k) in COLLISION_FREE_KEYS.iter().enumerate() {
                for _ in 0..(i + 1) * 10 {
                    let d = DataInput::U64(*k);
                    regular.insert(&d);
                    fast.insert(&d);
                    truth.observe(*k as i64);
                }
            }
            let label = concat!(
                "Count<",
                stringify!($storage),
                "> collision-free exactness on both paths"
            );
            let ctx = context(label, regular.rows(), regular.cols());
            let mut tally = Tally::default();
            for (k, c) in truth.pairs() {
                let r = regular.estimate(&DataInput::U64(k as u64));
                let f = fast.estimate(&DataInput::U64(k as u64));
                tally.record(r == c as f64 && f == c as f64, || {
                    format!("key {k}: true {c}, regular {r}, fast {f}")
                });
            }
            tally.assert_none(label, &ctx);
        }};
    }
    exact!(Vector2D<i32>);
    exact!(Vector2D<i64>);
    exact!(Vector2D<i128>);
    exact!(FixedMatrix);
    exact!(DefaultMatrixI32);
    exact!(QuickMatrixI64);
    exact!(QuickMatrixI128);
    exact!(DefaultMatrixI64);
    exact!(DefaultMatrixI128);
}

// ------------------------------------------------------------ Heap variants

/// `CMSHeap` is a Count-Min plus a top-k heap. The point estimate carries
/// Count-Min's bound; the heap must stay consistent with it and must recover
/// the keys the top-k contract actually guarantees.
macro_rules! cmsheap_instance {
    ($storage:ty, $path:ty, $stream:expr, $truth:expr) => {{
        let mut single = CMSHeap::<$storage, $path>::default();
        let mut left = CMSHeap::<$storage, $path>::default();
        let mut right = CMSHeap::<$storage, $path>::default();
        for (i, k) in $stream.iter().enumerate() {
            let d = DataInput::U64(*k);
            single.insert(&d);
            if i % 2 == 0 {
                left.insert(&d);
            } else {
                right.insert(&d);
            }
        }
        left.merge(&right);

        let (rows, cols) = (single.rows(), single.cols());
        let label = concat!(
            "CMSHeap<",
            stringify!($storage),
            ", ",
            stringify!($path),
            ">"
        );
        let ctx = context(label, rows, cols);
        let spec = CountMinSpec::new(rows, cols);
        spec.assert_contract(
            label,
            $truth,
            |k| single.estimate(&DataInput::U64(k as u64)).as_f64(),
            &ctx,
        );
        spec.assert_contract(
            &format!("{label} shard merge"),
            $truth,
            |k| left.estimate(&DataInput::U64(k as u64)).as_f64(),
            &ctx,
        );
        assert_heap_matches_sketch(
            label,
            &ctx,
            single.heap().heap(),
            $truth,
            |k| single.estimate(&DataInput::U64(k as u64)).as_f64(),
            |est| est as i64,
        );
    }};
}

/// `CSHeap` is a Count Sketch plus the same heap, so the point estimate
/// carries the L2 bound instead.
macro_rules! csheap_instance {
    ($storage:ty, $path:ty, $stream:expr, $truth:expr) => {
        csheap_instance!(
            @build $storage, $path, $stream, $truth,
            CSHeap::<$storage, $path>::default()
        )
    };
    // `Vector2D<i128>` has no `Default` impl for either heap family, but it is
    // a perfectly ordinary public instance through `CSHeap::new`; it is built
    // at the same 3x4096 / top-32 geometry the `Default` impls use so it is
    // judged at comparable dimensions.
    (sized $storage:ty, $path:ty, $stream:expr, $truth:expr) => {
        csheap_instance!(
            @build $storage, $path, $stream, $truth,
            CSHeap::<$storage, $path>::new(3, 4096, 32)
        )
    };
    (@build $storage:ty, $path:ty, $stream:expr, $truth:expr, $ctor:expr) => {{
        let mut single = $ctor;
        let mut left = $ctor;
        let mut right = $ctor;
        for (i, k) in $stream.iter().enumerate() {
            let d = DataInput::U64(*k);
            single.insert(&d);
            if i % 2 == 0 {
                left.insert(&d);
            } else {
                right.insert(&d);
            }
        }
        left.merge(&right);

        let (rows, cols) = (single.rows(), single.cols());
        let label = concat!(
            "CSHeap<",
            stringify!($storage),
            ", ",
            stringify!($path),
            ">"
        );
        let ctx = context(label, rows, cols);
        let spec = CountSketchSpec::new(rows, cols);
        spec.assert_contract(
            label,
            $truth,
            |k| single.estimate(&DataInput::U64(k as u64)),
            &ctx,
        );
        spec.assert_contract(
            &format!("{label} shard merge"),
            $truth,
            |k| left.estimate(&DataInput::U64(k as u64)),
            &ctx,
        );
        assert_heap_matches_sketch(
            label,
            &ctx,
            single.heap().heap(),
            $truth,
            |k| single.estimate(&DataInput::U64(k as u64)),
            cs_heap_count,
        );
    }};
}

/// Shared heap checks: every entry agrees with the sketch's own estimate, and
/// the heap holds keys the top-k contract guarantees.
///
/// `count_of` maps the sketch's estimate to the integer a heap entry holds. It
/// is the identity for `CMSHeap`, whose estimate is already the counter type,
/// and `cs_heap_count` for `CSHeap`, whose estimate is an `f64` median and is
/// documented to reach the heap by a saturating, truncating conversion.
fn assert_heap_matches_sketch<F, C>(
    label: &str,
    ctx: &str,
    items: &[asap_sketchlib::HHItem],
    truth: &FreqTruth,
    estimate: F,
    count_of: C,
) where
    F: Fn(i64) -> f64,
    C: Fn(f64) -> i64,
{
    let mut consistency = Tally::default();
    let capacity = items.len().max(1);
    let kth = truth.top_k(capacity)[capacity.min(truth.distinct()) - 1].1;
    let mut recall = 0usize;
    for it in items {
        let key = match it.key {
            HeapItem::U64(v) => v as i64,
            ref other => panic!("{label}: unexpected heap key {other:?}"),
        };
        let est = estimate(key);
        consistency.record(it.count == count_of(est), || {
            format!(
                "key {key}: heap holds {} but the sketch estimates {est} (-> {})",
                it.count,
                count_of(est)
            )
        });
        if truth.get(key) >= kth {
            recall += 1;
        }
    }
    consistency.assert_none(&format!("{label} heap/sketch consistency"), ctx);
    assert!(
        recall + 1 >= items.len(),
        "{label}: only {recall} of {} heap entries are at or above the true k-th count \
         ({kth}). {ctx}",
        items.len()
    );
}

macro_rules! heap_matrix_test {
    ($name:ident, $mac:ident, $path:ty, $( $kind:ident $storage:ty ),+ $(,)?) => {
        #[test]
        fn $name() {
            let (stream, truth) = stream_and_truth();
            $( heap_matrix_test!(@one $mac, $kind, $storage, $path, stream, &truth); )+
        }
    };
    (@one $mac:ident, default, $storage:ty, $path:ty, $stream:expr, $truth:expr) => {
        $mac!($storage, $path, $stream, $truth);
    };
    (@one $mac:ident, sized, $storage:ty, $path:ty, $stream:expr, $truth:expr) => {
        $mac!(sized $storage, $path, $stream, $truth);
    };
}

// The two heap families do **not** have the same instance coverage, and the
// reason is a difference in their insert bounds rather than an oversight:
//
// - `CMSHeap`'s insert path is bounded on `S::Counter: Copy + Ord + From<i32> +
//   Into<i64> + AddAssign`. `i128` has no `Into<i64>` and `f64` has neither
//   `Ord` nor `Into<i64>`, so those instances construct but have no `insert`,
//   `estimate` or `merge` at all. They are enumerated and pinned as inert by
//   `cmsheap_instances_without_an_insert_impl_are_inert_by_construction`.
// - `CSHeap`'s insert path is bounded on `S::Counter: CountSketchCounter`,
//   which `i128` *does* satisfy. All six `i128` instances are fully
//   operational and are covered below; the previous revision of this file and
//   of the coverage matrix wrongly claimed they were not insertable.
heap_matrix_test!(
    cmsheap_regular_path_instances_satisfy_the_count_min_bound,
    cmsheap_instance,
    RegularPath,
    default Vector2D<i32>,
    default Vector2D<i64>,
    default FixedMatrix,
    default DefaultMatrixI32,
    default QuickMatrixI64,
    default DefaultMatrixI64,
);

heap_matrix_test!(
    cmsheap_fast_path_instances_conform_to_the_count_min_model,
    cmsheap_instance,
    FastPath,
    default Vector2D<i32>,
    default Vector2D<i64>,
    default FixedMatrix,
    default DefaultMatrixI32,
    default QuickMatrixI64,
    default DefaultMatrixI64,
);

heap_matrix_test!(
    csheap_regular_path_instances_satisfy_the_l2_bound,
    csheap_instance,
    RegularPath,
    default Vector2D<i32>,
    default Vector2D<i64>,
    sized Vector2D<i128>,
    default FixedMatrix,
    default DefaultMatrixI32,
    default QuickMatrixI64,
    default QuickMatrixI128,
    default DefaultMatrixI64,
    default DefaultMatrixI128,
);

heap_matrix_test!(
    csheap_fast_path_instances_conform_to_the_l2_model,
    csheap_instance,
    FastPath,
    default Vector2D<i32>,
    default Vector2D<i64>,
    sized Vector2D<i128>,
    default FixedMatrix,
    default DefaultMatrixI32,
    default QuickMatrixI64,
    default QuickMatrixI128,
    default DefaultMatrixI64,
    default DefaultMatrixI128,
);

// ------------------------------------------------- 128-bit heap semantics

/// `CSHeap` over an `i128` counter really does accept counts past `i64::MAX`,
/// and this pins what happens to the heap entry when it does.
///
/// The insert path is `heap.update(key, cs_heap_count(estimate))`, where the
/// estimate is the row **median** as `f64`. Two lossy steps are unavoidable and
/// both are documented API rather than accidents:
///
/// 1. `i128 -> f64` inside `Count::estimate`, exact only below `2^53`;
/// 2. `f64 -> i64` in `cs_heap_count`, which **saturates** at `i64::MAX`.
///
/// Saturating is the chosen semantics. Wrapping would turn a huge positive
/// count into a negative one and corrupt the heap's ordering, which is the one
/// failure mode a top-k structure cannot survive; returning an error would put
/// a fallible result on an infallible `insert`; and widening `HHItem::count` to
/// `i128` would change the heap wire payload for every family that shares it.
/// This test fails if any of that silently changes.
#[test]
fn csheap_i128_counters_saturate_into_the_heap_instead_of_wrapping() {
    macro_rules! probe {
        ($storage:ty, $path:ty, $ctor:expr) => {{
            let label = concat!(
                "CSHeap<",
                stringify!($storage),
                ", ",
                stringify!($path),
                ">"
            );
            let key = DataInput::U64(0xC0FF_EE01);

            // Below 2^53 the whole pipeline is exact.
            let mut small = $ctor;
            let exact: i128 = 1_i128 << 40;
            small.insert_many(&key, exact);
            assert_eq!(
                small.estimate(&key),
                exact as f64,
                "{label}: an i128 count below 2^53 must round-trip exactly"
            );
            assert_eq!(
                heap_count(small.heap().heap(), &key),
                Some(exact as i64),
                "{label}: the heap entry must carry the same exact count"
            );

            // Past i64::MAX the heap entry saturates. It must not wrap
            // negative, and it must not silently become a different positive
            // number: `i64::MAX` is the documented ceiling.
            let mut huge = $ctor;
            let over: i128 = (i64::MAX as i128) * 4;
            huge.insert_many(&key, over);
            let est = huge.estimate(&key);
            assert!(
                est >= i64::MAX as f64,
                "{label}: the sketch itself must still hold the i128 mass, got {est}"
            );
            assert_eq!(
                cs_heap_count(est),
                i64::MAX,
                "{label}: cs_heap_count must saturate at i64::MAX"
            );
            assert_eq!(
                heap_count(huge.heap().heap(), &key),
                Some(i64::MAX),
                "{label}: the heap entry must saturate, never wrap"
            );

            // Merging two saturated sketches must stay saturated and positive.
            let mut other = $ctor;
            other.insert_many(&key, over);
            huge.merge(&other);
            assert_eq!(
                heap_count(huge.heap().heap(), &key),
                Some(i64::MAX),
                "{label}: a merge of two saturated sketches must stay at the ceiling"
            );
        }};
    }

    probe!(
        Vector2D<i128>,
        RegularPath,
        CSHeap::<Vector2D<i128>, RegularPath>::new(3, 4096, 32)
    );
    probe!(
        Vector2D<i128>,
        FastPath,
        CSHeap::<Vector2D<i128>, FastPath>::new(3, 4096, 32)
    );
    probe!(
        QuickMatrixI128,
        RegularPath,
        CSHeap::<QuickMatrixI128, RegularPath>::default()
    );
    probe!(
        QuickMatrixI128,
        FastPath,
        CSHeap::<QuickMatrixI128, FastPath>::default()
    );
    probe!(
        DefaultMatrixI128,
        RegularPath,
        CSHeap::<DefaultMatrixI128, RegularPath>::default()
    );
    probe!(
        DefaultMatrixI128,
        FastPath,
        CSHeap::<DefaultMatrixI128, FastPath>::default()
    );
}

/// The count a heap holds for `key`, or `None` if it is not in the heap.
fn heap_count(items: &[asap_sketchlib::HHItem], key: &DataInput) -> Option<i64> {
    items
        .iter()
        .find(|it| asap_sketchlib::heap_item_to_sketch_input(&it.key) == *key)
        .map(|it| it.count)
}

/// The `CMSHeap` instances that construct but have no operations at all.
///
/// `CMSHeap::insert` / `insert_many` / `estimate` / `merge` all live in an impl
/// bounded on `S::Counter: Copy + Ord + From<i32> + Into<i64> + AddAssign`.
/// Four counter types reach `CMSHeap` through a public constructor and two of
/// them fail that bound:
///
/// | counter | `Ord` | `Into<i64>` | operations |
/// | --- | --- | --- | --- |
/// | `i32`, `i64` | yes | yes | full |
/// | `i128` | yes | **no** | none |
/// | `f64` | **no** | **no** | none |
///
/// So `CMSHeap<Vector2D<i128>>`, `CMSHeap<Vector2D<f64>>`,
/// `CMSHeap<QuickMatrixI128>` and `CMSHeap<DefaultMatrixI128>` are
/// *constructible but inert* on both hashing paths — eight instances that
/// allocate a sketch and a heap and can then do nothing but report their
/// dimensions.
///
/// # What this test is, and what it is not
///
/// It is **constructibility coverage only**. It shows that the constructors
/// exist, that the geometry accessors answer, and that the heap starts empty.
/// It does **not** verify that `insert` is uncallable — a Rust test cannot
/// assert the absence of a method without a compile-fail harness such as
/// `trybuild`, and an earlier revision of this comment wrongly claimed the test
/// "fails to compile if an insert impl appears". It would not: adding an impl
/// leaves every assertion below compiling and passing.
///
/// A `trybuild` fixture would close that hole and is the right thing if these
/// instances ever matter; it was not added here because it means a new
/// dev-dependency for a set of instances that, by construction, have no
/// behaviour to protect. The coverage matrix lists all eight under **gap**, not
/// under covered instances, and this test is filed as `structural
/// (constructibility only)`.
///
/// # Why the API is not being changed
///
/// Every alternative costs more than it buys: `TryInto<i64>` would make
/// `insert` fallible or silently lossy for exactly the counters that motivated
/// `i128`; widening `HHItem::count` to `i128` changes the heap wire payload
/// shared with `CSHeap`, Space-Saving and the Octo top-k plans; and removing
/// the constructors is a breaking change to a type that composes generically.
/// `CSHeap` already covers `i128` end to end for callers who need it.
#[test]
fn cmsheap_inert_instances_construct_and_report_their_geometry() {
    macro_rules! inert {
        ($storage:ty, $path:ty, $ctor:expr, $rows:expr, $cols:expr) => {{
            let label = concat!(
                "CMSHeap<",
                stringify!($storage),
                ", ",
                stringify!($path),
                ">"
            );
            let sketch = $ctor;
            // Everything a caller can actually do with one of these.
            assert_eq!(sketch.rows(), $rows, "{label}: rows()");
            assert_eq!(sketch.cols(), $cols, "{label}: cols()");
            assert_eq!(sketch.heap().len(), 0, "{label}: the heap starts empty");
            assert_eq!(
                sketch.cms().rows(),
                $rows,
                "{label}: the wrapped CountMin reports the same geometry"
            );
        }};
    }

    inert!(
        Vector2D<i128>,
        RegularPath,
        CMSHeap::<Vector2D<i128>, RegularPath>::new(3, 4096, 32),
        3,
        4096
    );
    inert!(
        Vector2D<i128>,
        FastPath,
        CMSHeap::<Vector2D<i128>, FastPath>::new(3, 4096, 32),
        3,
        4096
    );
    inert!(
        Vector2D<f64>,
        RegularPath,
        CMSHeap::<Vector2D<f64>, RegularPath>::new(3, 4096, 32),
        3,
        4096
    );
    inert!(
        Vector2D<f64>,
        FastPath,
        CMSHeap::<Vector2D<f64>, FastPath>::new(3, 4096, 32),
        3,
        4096
    );
    inert!(
        QuickMatrixI128,
        RegularPath,
        CMSHeap::<QuickMatrixI128, RegularPath>::default(),
        5,
        2048
    );
    inert!(
        QuickMatrixI128,
        FastPath,
        CMSHeap::<QuickMatrixI128, FastPath>::default(),
        5,
        2048
    );
    inert!(
        DefaultMatrixI128,
        RegularPath,
        CMSHeap::<DefaultMatrixI128, RegularPath>::default(),
        3,
        4096
    );
    inert!(
        DefaultMatrixI128,
        FastPath,
        CMSHeap::<DefaultMatrixI128, FastPath>::default(),
        3,
        4096
    );
}

// ------------------------------------------------------- Counter-width edges

/// The only thing the three counter widths actually promise differently is how
/// much mass a single cell can hold. Each width must carry a count the next
/// smaller one cannot, and the estimate must come back intact rather than
/// wrapped.
#[test]
fn countmin_counter_widths_carry_the_mass_their_type_allows() {
    let key = DataInput::U64(0xFEED_FACE);

    // i32: just under the signed 32-bit ceiling.
    let mut cm32 = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(3, 64);
    let near_i32 = i32::MAX - 1;
    cm32.insert_many(&key, near_i32);
    assert_eq!(
        cm32.estimate(&key),
        near_i32,
        "i32 counters must hold {near_i32} without wrapping"
    );

    // i64: a count an i32 cell could not represent at all.
    let mut cm64 = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(3, 64);
    let beyond_i32 = i32::MAX as i64 * 4;
    cm64.insert_many(&key, beyond_i32);
    assert_eq!(
        cm64.estimate(&key),
        beyond_i32,
        "i64 counters must hold {beyond_i32}, which overflows i32"
    );

    // i128: a count an i64 cell could not represent.
    let mut cm128 = CountMin::<Vector2D<i128>, RegularPath>::with_dimensions(3, 64);
    let beyond_i64 = i64::MAX as i128 * 4;
    cm128.insert_many(&key, beyond_i64);
    assert_eq!(
        cm128.estimate(&key),
        beyond_i64,
        "i128 counters must hold {beyond_i64}, which overflows i64"
    );

    // Merging two sketches each holding half the i64 range must not wrap the
    // i128 target either.
    let mut a = CountMin::<Vector2D<i128>, RegularPath>::with_dimensions(3, 64);
    let mut b = CountMin::<Vector2D<i128>, RegularPath>::with_dimensions(3, 64);
    a.insert_many(&key, i64::MAX as i128);
    b.insert_many(&key, i64::MAX as i128);
    a.merge(&b);
    assert_eq!(
        a.estimate(&key),
        i64::MAX as i128 * 2,
        "merging two i64::MAX counts into i128 storage must not wrap"
    );
}

/// The signed families have a symmetric requirement: a decrement must reach
/// the negative end of the counter's range without wrapping.
#[test]
fn countsketch_counter_widths_carry_signed_mass_in_both_directions() {
    let key = DataInput::U64(0xDEAD_BEEF);

    let mut cs32 = Count::<Vector2D<i32>, RegularPath>::with_dimensions(3, 64);
    cs32.insert_many(&key, i32::MAX / 2);
    cs32.insert_many(&key, -(i32::MAX / 2));
    assert_eq!(
        cs32.estimate(&key),
        0.0,
        "i32 Count Sketch must cancel a half-range increment exactly"
    );

    let mut cs64 = Count::<Vector2D<i64>, RegularPath>::with_dimensions(3, 64);
    let big = i32::MAX as i64 * 4;
    cs64.insert_many(&key, big);
    assert_eq!(cs64.estimate(&key), big as f64);
    cs64.insert_many(&key, -big * 2);
    assert_eq!(
        cs64.estimate(&key),
        -(big as f64),
        "i64 Count Sketch must reach the negative side of the i32 range"
    );

    let mut cs128 = Count::<Vector2D<i128>, RegularPath>::with_dimensions(3, 64);
    let huge = i64::MAX as i128 * 4;
    cs128.insert_many(&key, huge);
    assert_eq!(cs128.estimate(&key), huge as f64);
    cs128.insert_many(&key, -huge * 2);
    assert_eq!(
        cs128.estimate(&key),
        -(huge as f64),
        "i128 Count Sketch must reach the negative side of the i64 range"
    );
}

const DEPTH_AXIS: [usize; 4] = [1, 2, 3, 9];
const WIDTH_AXIS: [usize; 4] = [64, 512, 4_096, 8_192];
const COUNTSKETCH_DEPTH_AXIS: [usize; 3] = [3, 5, 9];
const NON_POWER_OF_TWO_WIDTHS: [usize; 4] = [3, 100, 1_000, 4_095];
const WIDTH_EXCESS_DECAY: f64 = 2.0;

fn countmin_axis_contract(
    label: &str,
    rows: usize,
    cols: usize,
    truth: &FreqTruth,
    estimate: impl Fn(i64) -> f64,
) -> f64 {
    let spec = CountMinSpec::new(rows, cols);
    let total = truth.total() as f64;
    let distinct = truth.distinct();
    let mut one_sided = Tally::default();
    let mut simultaneous = Tally::default();
    let mut excess_sum = 0.0;
    let mut keys = 0usize;
    for (key, count) in truth.pairs() {
        let est = estimate(key);
        let f = count as f64;
        one_sided.record(est >= f, || {
            format!("key {key}: est {est} < true {f} (Count-Min must never underestimate)")
        });
        let bound = spec.simultaneous_bound(total, f, distinct, SIMULTANEOUS_LEVEL);
        simultaneous.record(est - f <= bound, || {
            format!("key {key}: excess {:.1} > b*(N-f)/w = {bound:.1}", est - f)
        });
        excess_sum += est - f;
        keys += 1;
    }
    let ctx = context(label, rows, cols);
    one_sided.assert_none(&format!("{label} / one-sided"), &ctx);
    simultaneous.assert_none(&format!("{label} / simultaneous"), &ctx);
    excess_sum / keys.max(1) as f64
}

#[test]
fn countmin_holds_its_contract_across_the_depth_and_width_axis_on_both_paths() {
    let (stream, truth) = stream_and_truth();
    for &rows in &DEPTH_AXIS {
        for &cols in &WIDTH_AXIS {
            let mut regular = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);
            let mut fast = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(rows, cols);
            for k in &stream {
                regular.insert(&DataInput::U64(*k));
                fast.insert(&DataInput::U64(*k));
            }
            countmin_axis_contract(
                "CountMin<Vector2D<i64>, RegularPath> axis",
                rows,
                cols,
                &truth,
                |key| regular.estimate(&DataInput::U64(key as u64)) as f64,
            );
            countmin_axis_contract(
                "CountMin<Vector2D<i64>, FastPath> axis",
                rows,
                cols,
                &truth,
                |key| fast.estimate(&DataInput::U64(key as u64)) as f64,
            );
        }
    }
}

#[test]
fn countmin_mean_excess_falls_as_the_width_axis_grows() {
    let (stream, truth) = stream_and_truth();
    for &rows in &DEPTH_AXIS {
        let mut previous: Option<(usize, f64)> = None;
        for &cols in &[64usize, 512, 4_096] {
            let mut sketch = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(rows, cols);
            for k in &stream {
                sketch.insert(&DataInput::U64(*k));
            }
            let mut excess_sum = 0.0;
            for (key, count) in truth.pairs() {
                excess_sum += sketch.estimate(&DataInput::U64(key as u64)) as f64 - count as f64;
            }
            let mean = excess_sum / truth.distinct() as f64;
            if let Some((prev_cols, prev_mean)) = previous {
                assert!(
                    mean * WIDTH_EXCESS_DECAY <= prev_mean,
                    "d={rows}: mean excess {mean:.3} at w={cols} is not at most 1/{WIDTH_EXCESS_DECAY} \
                     of {prev_mean:.3} at w={prev_cols}"
                );
            }
            previous = Some((cols, mean));
        }
    }
}

#[test]
fn countsketch_holds_its_l2_contract_across_the_depth_and_width_axis() {
    let (stream, truth) = stream_and_truth();
    for &rows in &COUNTSKETCH_DEPTH_AXIS {
        for &cols in &WIDTH_AXIS {
            let spec = CountSketchSpec::new(rows, cols);
            let mut regular = Count::<Vector2D<i64>, RegularPath>::with_dimensions(rows, cols);
            let mut fast = Count::<Vector2D<i64>, FastPath>::with_dimensions(rows, cols);
            for k in &stream {
                regular.insert(&DataInput::U64(*k));
                fast.insert(&DataInput::U64(*k));
            }
            let mut regular_simultaneous = Tally::default();
            let mut regular_marginal = Tally::default();
            spec.tally_into(
                &mut regular_simultaneous,
                &mut regular_marginal,
                &truth,
                |key| regular.estimate(&DataInput::U64(key as u64)),
            );
            regular_simultaneous.assert_none(
                "Count<Vector2D<i64>, RegularPath> axis / simultaneous L2",
                &context("Count<Vector2D<i64>, RegularPath> axis", rows, cols),
            );
            let mut fast_simultaneous = Tally::default();
            let mut fast_marginal = Tally::default();
            spec.tally_into(&mut fast_simultaneous, &mut fast_marginal, &truth, |key| {
                fast.estimate(&DataInput::U64(key as u64))
            });
            fast_simultaneous.assert_none(
                "Count<Vector2D<i64>, FastPath> axis / simultaneous L2",
                &context("Count<Vector2D<i64>, FastPath> axis", rows, cols),
            );
        }
    }
}

#[test]
fn countmin_answers_a_non_power_of_two_width_on_both_paths() {
    let (stream, truth) = stream_and_truth();
    for &cols in &NON_POWER_OF_TWO_WIDTHS {
        assert!(
            !cols.is_power_of_two(),
            "the width axis must stay off the power-of-two grid, got {cols}"
        );
        let mut regular = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(3, cols);
        let mut fast = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(3, cols);
        for k in &stream {
            regular.insert(&DataInput::U64(*k));
            fast.insert(&DataInput::U64(*k));
        }
        assert_eq!(regular.cols(), cols, "regular path reported width");
        assert_eq!(fast.cols(), cols, "fast path reported width");
        countmin_axis_contract(
            "CountMin<Vector2D<i64>, RegularPath> non-power-of-two",
            3,
            cols,
            &truth,
            |key| regular.estimate(&DataInput::U64(key as u64)) as f64,
        );
        countmin_axis_contract(
            "CountMin<Vector2D<i64>, FastPath> non-power-of-two",
            3,
            cols,
            &truth,
            |key| fast.estimate(&DataInput::U64(key as u64)) as f64,
        );
    }
}

#[test]
fn countsketch_answers_a_non_power_of_two_width_on_both_paths() {
    let (stream, truth) = stream_and_truth();
    for &cols in &NON_POWER_OF_TWO_WIDTHS {
        let spec = CountSketchSpec::new(5, cols);
        let mut regular = Count::<Vector2D<i64>, RegularPath>::with_dimensions(5, cols);
        let mut fast = Count::<Vector2D<i64>, FastPath>::with_dimensions(5, cols);
        for k in &stream {
            regular.insert(&DataInput::U64(*k));
            fast.insert(&DataInput::U64(*k));
        }
        assert_eq!(regular.cols(), cols, "regular path reported width");
        assert_eq!(fast.cols(), cols, "fast path reported width");
        let mut regular_simultaneous = Tally::default();
        let mut regular_marginal = Tally::default();
        spec.tally_into(
            &mut regular_simultaneous,
            &mut regular_marginal,
            &truth,
            |key| regular.estimate(&DataInput::U64(key as u64)),
        );
        regular_simultaneous.assert_none(
            "Count<Vector2D<i64>, RegularPath> non-power-of-two / simultaneous L2",
            &context("Count non-power-of-two regular", 5, cols),
        );
        let mut fast_simultaneous = Tally::default();
        let mut fast_marginal = Tally::default();
        spec.tally_into(&mut fast_simultaneous, &mut fast_marginal, &truth, |key| {
            fast.estimate(&DataInput::U64(key as u64))
        });
        fast_simultaneous.assert_none(
            "Count<Vector2D<i64>, FastPath> non-power-of-two / simultaneous L2",
            &context("Count non-power-of-two fast", 5, cols),
        );
    }
}

#[test]
fn a_non_power_of_two_width_keeps_every_column_index_inside_the_matrix() {
    for &cols in &NON_POWER_OF_TWO_WIDTHS {
        let mut fast = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(3, cols);
        for k in 0..20_000u64 {
            fast.insert(&DataInput::U64(k.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        }
        let occupied = (0..fast.rows())
            .map(|row| {
                (0..cols)
                    .filter(|col| fast.as_storage().query_one_counter(row, *col) > 0)
                    .count()
            })
            .sum::<usize>();
        assert!(
            occupied > 0,
            "w={cols}: no counter was written, so the fold never reached the matrix"
        );
        let total: i64 = (0..fast.rows())
            .flat_map(|row| (0..cols).map(move |col| (row, col)))
            .map(|(row, col)| fast.as_storage().query_one_counter(row, col))
            .sum();
        assert_eq!(
            total,
            20_000 * fast.rows() as i64,
            "w={cols}: every insert must land in exactly one counter per row"
        );
    }
}

#[test]
fn countmin_merge_max_dominates_both_sides_on_disjoint_key_sets() {
    let (stream, _) = stream_and_truth();
    let mut left = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(4, 2_048);
    let mut right = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(4, 2_048);
    let mut left_truth = FreqTruth::default();
    let mut right_truth = FreqTruth::default();
    for k in &stream {
        if k % 2 == 0 {
            left.insert(&DataInput::U64(*k));
            left_truth.observe(*k as i64);
        } else {
            right.insert(&DataInput::U64(*k));
            right_truth.observe(*k as i64);
        }
    }

    let left_before: Vec<(i64, i64)> = left_truth
        .pairs()
        .into_iter()
        .map(|(key, _)| (key, left.estimate(&DataInput::U64(key as u64))))
        .collect();
    let right_before: Vec<(i64, i64)> = right_truth
        .pairs()
        .into_iter()
        .map(|(key, _)| (key, right.estimate(&DataInput::U64(key as u64))))
        .collect();

    let mut merged = left.clone();
    merged.merge_max(&right);

    for (key, before) in &left_before {
        let after = merged.estimate(&DataInput::U64(*key as u64));
        assert!(
            after >= *before,
            "key {key}: merge_max lowered the left estimate from {before} to {after}"
        );
        assert!(
            after >= left_truth.get(*key),
            "key {key}: merge_max underestimated the left true count {}",
            left_truth.get(*key)
        );
    }
    for (key, before) in &right_before {
        let after = merged.estimate(&DataInput::U64(*key as u64));
        assert!(
            after >= *before,
            "key {key}: merge_max lowered the right estimate from {before} to {after}"
        );
        assert!(
            after >= right_truth.get(*key),
            "key {key}: merge_max underestimated the right true count {}",
            right_truth.get(*key)
        );
    }

    let summed = {
        let mut s = left.clone();
        s.merge(&right);
        s
    };
    for (key, _) in left_truth.pairs() {
        let by_max = merged.estimate(&DataInput::U64(key as u64));
        let by_sum = summed.estimate(&DataInput::U64(key as u64));
        assert!(
            by_max <= by_sum,
            "key {key}: elementwise max {by_max} exceeded elementwise sum {by_sum}"
        );
    }
}

#[test]
fn countmin_merge_max_is_idempotent_and_absorbs_an_empty_sketch() {
    let (stream, truth) = stream_and_truth();
    let mut sketch = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(4, 2_048);
    for k in &stream {
        sketch.insert(&DataInput::U64(*k));
    }
    let baseline: Vec<(i64, i64)> = truth
        .pairs()
        .into_iter()
        .map(|(key, _)| (key, sketch.estimate(&DataInput::U64(key as u64))))
        .collect();

    let empty = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(4, 2_048);
    let mut with_empty = sketch.clone();
    with_empty.merge_max(&empty);
    let mut with_self = sketch.clone();
    let twin = sketch.clone();
    with_self.merge_max(&twin);

    for (key, before) in baseline {
        assert_eq!(
            with_empty.estimate(&DataInput::U64(key as u64)),
            before,
            "key {key}: merging an empty sketch by max moved the estimate"
        );
        assert_eq!(
            with_self.estimate(&DataInput::U64(key as u64)),
            before,
            "key {key}: merge_max with an identical sketch is not idempotent"
        );
    }
}

#[test]
fn countmin_precomputed_hash_entry_points_match_the_value_entry_points() {
    const ROWS: usize = 4;
    const COLS: usize = 2_048;
    let (stream, truth) = stream_and_truth();

    let mut by_value = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let mut by_hash = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let mut by_bulk_hash = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let hashes: Vec<_> = stream
        .iter()
        .map(|k| hash_for_matrix(ROWS, COLS, &DataInput::U64(*k)))
        .collect();
    for k in &stream {
        by_value.insert(&DataInput::U64(*k));
    }
    for h in &hashes {
        by_hash.fast_insert_with_hash_value(h);
    }
    by_bulk_hash.bulk_insert_with_hashes(&hashes);

    for (key, _) in truth.pairs() {
        let probe = DataInput::U64(key as u64);
        let hashed = hash_for_matrix(ROWS, COLS, &probe);
        let expected = by_value.estimate(&probe);
        assert_eq!(
            by_hash.estimate(&probe),
            expected,
            "key {key}: fast_insert_with_hash_value diverged from insert"
        );
        assert_eq!(
            by_bulk_hash.estimate(&probe),
            expected,
            "key {key}: bulk_insert_with_hashes diverged from insert"
        );
        assert_eq!(
            by_value.fast_estimate_with_hash(&hashed),
            expected,
            "key {key}: fast_estimate_with_hash diverged from estimate"
        );
    }
}

#[test]
fn countmin_weighted_batch_entry_points_match_a_loop_of_single_inserts() {
    const ROWS: usize = 4;
    const COLS: usize = 2_048;
    let (stream, truth) = stream_and_truth();

    let mut by_loop = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let mut by_bulk_many = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let mut by_hashed_many = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);

    let weighted: Vec<(DataInput, i64)> = stream
        .iter()
        .enumerate()
        .map(|(i, k)| (DataInput::U64(*k), (i % 5) as i64 + 1))
        .collect();
    let hashed_weighted: Vec<_> = stream
        .iter()
        .enumerate()
        .map(|(i, k)| {
            (
                hash_for_matrix(ROWS, COLS, &DataInput::U64(*k)),
                (i % 5) as i64 + 1,
            )
        })
        .collect();

    for (value, many) in &weighted {
        by_loop.insert_many(value, *many);
    }
    by_bulk_many.bulk_insert_many(&weighted);
    by_hashed_many.bulk_insert_many_with_hashes(&hashed_weighted);

    for (key, _) in truth.pairs() {
        let probe = DataInput::U64(key as u64);
        let expected = by_loop.estimate(&probe);
        assert_eq!(
            by_bulk_many.estimate(&probe),
            expected,
            "key {key}: bulk_insert_many diverged from a loop of insert_many"
        );
        assert_eq!(
            by_hashed_many.estimate(&probe),
            expected,
            "key {key}: bulk_insert_many_with_hashes diverged from a loop of insert_many"
        );
    }

    let mut single = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let hashed = hash_for_matrix(ROWS, COLS, &DataInput::U64(7));
    single.fast_insert_many_with_hash_value(&hashed, 9);
    assert_eq!(
        single.estimate(&DataInput::U64(7)),
        9,
        "fast_insert_many_with_hash_value must apply the whole weight"
    );
}

#[test]
fn countsketch_precomputed_hash_entry_points_match_the_value_entry_points() {
    const ROWS: usize = 5;
    const COLS: usize = 2_048;
    let (stream, truth) = stream_and_truth();

    let mut by_value = Count::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let mut by_hash = Count::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    for k in &stream {
        by_value.insert(&DataInput::U64(*k));
        by_hash.fast_insert_with_hash_value(&hash_for_matrix(ROWS, COLS, &DataInput::U64(*k)));
    }

    for (key, _) in truth.pairs() {
        let probe = DataInput::U64(key as u64);
        let hashed = hash_for_matrix(ROWS, COLS, &probe);
        let expected = by_value.estimate(&probe);
        assert_eq!(
            by_hash.estimate(&probe),
            expected,
            "key {key}: fast_insert_with_hash_value diverged from insert"
        );
        assert_eq!(
            by_value.fast_estimate_with_hash(&hashed),
            expected,
            "key {key}: fast_estimate_with_hash diverged from estimate"
        );
    }

    let mut weighted = Count::<Vector2D<i64>, FastPath>::with_dimensions(ROWS, COLS);
    let hashed = hash_for_matrix(ROWS, COLS, &DataInput::U64(11));
    weighted.fast_insert_many_with_hash_value(&hashed, 6);
    assert_eq!(
        weighted.estimate(&DataInput::U64(11)),
        6.0,
        "fast_insert_many_with_hash_value must apply the whole weight"
    );
}
