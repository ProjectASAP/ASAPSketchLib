//! E2E OctoSketch suite: delta-promotion conservation invariants for the
//! CountMin, Count and HyperLogLog child/parent protocol, and the
//! `octo-runtime` multi-threaded pipeline checked against single-threaded
//! replays of the same round-robin partition.
//!
//! The protocol is fully deterministic: `run_octo` dispatches round-robin from
//! the calling thread, and the aggregator only ever applies commutative
//! updates (`+=` for the counter sketches, `max` for HLL registers). Thread
//! interleaving therefore cannot change the parent, so every runtime test here
//! asserts *exact* equality against a replay instead of a statistical band.
//!
//! The second half checks the implementation against the OctoSketch paper
//! itself (NSDI '24, Zhang/Chen/Liu): whether estimates land inside Theorems
//! 1, 3 and 4, and whether delta promotion actually beats the periodic
//! sketch-merge baseline the paper measures against (Theorem 2, Figure 6).
//! The runtime defaults to `OctoPartition::HashByKey`, so a flow lands on one
//! worker and the paper's k' - the number of workers a flow may pass by - is 1.
//! The bound tests sweep `RoundRobin` as well, where k' = k. Note that k' is
//! about a *flow*: a counter is shared by whatever flows hash into it, and each
//! worker may hold back up to tau of its own share, so the provable per-counter
//! gap is k*tau under either partition.
//!
//! The last section covers the two families that keep a flow key beside every
//! counter - CocoSketch and the Elastic sketch, section 4.4 and appendix C.
//! Only the Octo variants are here; the single-threaded sketches are
//! `tests/e2e_heavy_hitters.rs`. Neither family is covered by a theorem, and
//! Coco's aggregator elects from an unseeded RNG, so those tests split: exact
//! mass identities that hold under any interleaving, and the paper's own
//! measured comparison against sketch-merge for everything else.

mod common;

#[cfg(feature = "octo-runtime")]
use asap_sketchlib::DD_PROMASK;
use asap_sketchlib::common::BOTTOM_LAYER_FINDER;
use asap_sketchlib::{
    CM_PROMASK, COUNT_PROMASK, Classic, CmDelta, Count, CountDelta, CountMin, DDSketch, DataInput,
    DdWorkerSketch, ErtlMLE, FastPath, HLL_PROMASK, HllDelta, HyperLogLog, HyperLogLogP12,
    HyperLogLogP16, L2HH, L2hhWorkerSketch, LayeredCountDelta, OctoAggregator, OctoPlan,
    OctoThreshold, OctoWorker, RegularPath, UnivMon, UnivMonDeltaFidelity, UnivMonOctoAggregator,
    UnivMonOctoPlan, UnivMonOctoWorker, Vector2D, bottom_layer_for_hash, hash64_seeded,
    hash128_seeded, input_to_owned, univmon_layer_threshold,
};
use common::{FreqTruth, zipf_u64};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

type Cm = CountMin<Vector2D<i32>, RegularPath>;
type CmFast = CountMin<Vector2D<i32>, FastPath>;
type Cs = Count<Vector2D<i32>, RegularPath>;
type CsFast = Count<Vector2D<i32>, FastPath>;

const ROWS: usize = 5;
const COLS: usize = 2048;

/// τ in the paper's notation (NSDI '24, §4.1): the worker promotion threshold.
const TAU: i32 = CM_PROMASK as i32;

/// Largest amount a single cell can lag its exact count: promotion fires at a
/// multiple of τ, so at most `τ - 1` un-promoted increments remain.
const MAX_CELL_RESIDUAL: i32 = TAU - 1;

fn keys(n: usize, domain: usize, seed: u64) -> Vec<u64> {
    zipf_u64(n, domain, 1.1, seed)
}

#[cfg(feature = "octo-runtime")]
fn inputs_from(keys: &[u64]) -> Vec<DataInput<'static>> {
    keys.iter().copied().map(DataInput::U64).collect()
}

// ---------------------------------------------------------------------------
// Child-run helpers: drive one child sketch over a stream, capture its deltas
// ---------------------------------------------------------------------------

fn cm_child_run(stream: &[u64], rows: usize, cols: usize) -> (Cm, Vec<CmDelta>) {
    let mut child = Cm::with_dimensions(rows, cols);
    let mut deltas = Vec::new();
    for k in stream {
        child.insert_emit_delta(&DataInput::U64(*k), &mut |d| deltas.push(d));
    }
    (child, deltas)
}

fn cs_child_run(stream: &[u64], rows: usize, cols: usize) -> (Cs, Vec<CountDelta>) {
    let mut child = Cs::with_dimensions(rows, cols);
    let mut deltas = Vec::new();
    for k in stream {
        child.insert_emit_delta(&DataInput::U64(*k), &mut |d| deltas.push(d));
    }
    (child, deltas)
}

fn hll_child_run(stream: &[u64]) -> (HyperLogLog<Classic>, Vec<HllDelta>) {
    let mut child = HyperLogLog::<Classic>::default();
    let mut deltas = Vec::new();
    for k in stream {
        child.insert_emit_delta(&DataInput::U64(*k), &mut |d| deltas.push(d));
    }
    (child, deltas)
}

fn cm_cells(sketch: &Cm) -> Vec<i32> {
    let (rows, cols) = (sketch.rows(), sketch.cols());
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| sketch.as_storage().query_one_counter(r, c))
        .collect()
}

fn cs_cells(sketch: &Cs) -> Vec<i32> {
    let (rows, cols) = (sketch.rows(), sketch.cols());
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| sketch.as_storage().query_one_counter(r, c))
        .collect()
}

// ---------------------------------------------------------------------------
// CountMin delta protocol
// ---------------------------------------------------------------------------

#[test]
fn cm_promotion_is_lossless_modulo_the_child_residual() {
    let stream = keys(20_000, 512, 9_101);
    let (child, deltas) = cm_child_run(&stream, ROWS, COLS);

    let mut parent = Cm::with_dimensions(ROWS, COLS);
    for d in &deltas {
        parent.apply_delta(*d);
    }
    let mut reference = Cm::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    // Algorithm 1 clears the counter it promotes, so what the child still
    // holds plus what the parent received must reconstruct a single pass.
    let (child_cells, parent_cells, ref_cells) =
        (cm_cells(&child), cm_cells(&parent), cm_cells(&reference));
    for (i, ((&c, &p), &r)) in child_cells
        .iter()
        .zip(parent_cells.iter())
        .zip(ref_cells.iter())
        .enumerate()
    {
        assert_eq!(p + c, r, "cell {i}: promoted {p} plus residual {c} != {r}");
        assert!(c <= MAX_CELL_RESIDUAL, "cell {i}: residual {c} un-promoted");
    }
}

#[test]
fn cm_deltas_are_well_formed_and_carry_exactly_one_promotion() {
    let stream = keys(20_000, 512, 9_102);
    let (_, deltas) = cm_child_run(&stream, ROWS, COLS);

    assert!(!deltas.is_empty(), "a 20k stream must promote something");
    for d in &deltas {
        assert_eq!(d.value, CM_PROMASK, "CM promotes exactly the mask value");
        assert!((d.row as usize) < ROWS, "row {} out of range", d.row);
        assert!((d.col as usize) < COLS, "col {} out of range", d.col);
    }
}

#[test]
fn cm_parent_holds_every_completed_promotion() {
    let stream = keys(30_000, 512, 9_103);
    let (_, deltas) = cm_child_run(&stream, ROWS, COLS);

    let mut parent = Cm::with_dimensions(ROWS, COLS);
    for d in &deltas {
        parent.apply_delta(*d);
    }

    let mut reference = Cm::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    let mask = CM_PROMASK as i32;
    let (parent_cells, ref_cells) = (cm_cells(&parent), cm_cells(&reference));
    let mut worst_residual = 0;
    for (i, (&p, &r)) in parent_cells.iter().zip(ref_cells.iter()).enumerate() {
        assert_eq!(
            p,
            mask * (r / mask),
            "cell {i}: parent must hold every completed promotion of {r}"
        );
        worst_residual = worst_residual.max(r - p);
    }
    assert!(worst_residual <= MAX_CELL_RESIDUAL);
    assert!(
        worst_residual > 0,
        "a skewed stream should leave some un-promoted remainder"
    );
}

#[test]
fn cm_octo_estimate_trails_the_single_thread_estimate_by_under_one_promotion() {
    let stream = keys(40_000, 256, 9_104);
    let (_, deltas) = cm_child_run(&stream, ROWS, COLS);

    let mut parent = Cm::with_dimensions(ROWS, COLS);
    for d in &deltas {
        parent.apply_delta(*d);
    }
    let mut reference = Cm::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    // Per row the parent lags by `count mod tau`, so the row-wise minimum
    // lags by strictly less than one promotion window and never overshoots.
    for k in 0u64..256 {
        let key = DataInput::U64(k);
        let single = reference.estimate(&key);
        let octo = parent.estimate(&key);
        assert!(
            octo <= single,
            "key {k}: octo {octo} must not exceed single-thread {single}"
        );
        assert!(
            single - octo <= MAX_CELL_RESIDUAL,
            "key {k}: deficit {} exceeds one promotion window",
            single - octo
        );
    }
}

#[test]
fn cm_fast_path_promotes_on_the_same_schedule_as_the_regular_path() {
    let stream = keys(20_000, 512, 9_105);

    let mut fast_child = CmFast::with_dimensions(ROWS, COLS);
    let mut fast_parent = CmFast::with_dimensions(ROWS, COLS);
    let mut fast_deltas = 0usize;
    for k in &stream {
        fast_child.insert_emit_delta(&DataInput::U64(*k), &mut |d| {
            assert_eq!(d.value, CM_PROMASK);
            fast_parent.apply_delta(d);
            fast_deltas += 1;
        });
    }

    assert!(fast_deltas > 0, "a 20k stream must promote something");

    // The fast path derives all columns from one hash, so its collision
    // pattern - and therefore its promotion count - differs from the regular
    // path. Only the per-cell promotion rule is shared.
    let mut reference = CmFast::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }
    for r in 0..ROWS {
        for c in 0..COLS {
            assert_eq!(
                fast_parent.as_storage().query_one_counter(r, c)
                    + fast_child.as_storage().query_one_counter(r, c),
                reference.as_storage().query_one_counter(r, c),
                "cell ({r},{c})"
            );
        }
    }
}

#[test]
fn cm_delta_addressing_survives_columns_past_the_u16_ceiling() {
    // `CmDelta`'s row and col fields are u32, so a geometry wider than
    // 65_536 columns addresses every cell exactly.
    let rows = 3;
    let cols = 100_000;
    let stream = keys(60_000, 4_096, 9_108);

    let mut child = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(rows, cols);
    let mut parent = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(rows, cols);
    for k in &stream {
        child.insert_emit_delta(&DataInput::U64(*k), &mut |d| parent.apply_delta(d));
    }

    let mut reference = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(rows, cols);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(
                parent.as_storage().query_one_counter(r, c)
                    + child.as_storage().query_one_counter(r, c),
                reference.as_storage().query_one_counter(r, c),
                "cell ({r},{c}) mis-addressed past the u16 ceiling"
            );
        }
    }
}

#[test]
fn cm_delta_application_is_order_independent() {
    let stream = keys(15_000, 512, 9_106);
    let (_, deltas) = cm_child_run(&stream, ROWS, COLS);

    let mut in_order = Cm::with_dimensions(ROWS, COLS);
    for d in &deltas {
        in_order.apply_delta(*d);
    }

    let mut shuffled = deltas.clone();
    shuffled.shuffle(&mut StdRng::seed_from_u64(9_106));
    let mut out_of_order = Cm::with_dimensions(ROWS, COLS);
    for d in &shuffled {
        out_of_order.apply_delta(*d);
    }

    assert_eq!(
        cm_cells(&in_order),
        cm_cells(&out_of_order),
        "the aggregator applies commutative updates, so delivery order is free"
    );
}

#[test]
fn cm_sharded_children_conserve_counts_against_a_single_pass() {
    let stream = keys(40_000, 512, 9_107);
    let shards = 4;

    let mut children: Vec<Cm> = (0..shards)
        .map(|_| Cm::with_dimensions(ROWS, COLS))
        .collect();
    let mut parent = Cm::with_dimensions(ROWS, COLS);
    for (i, k) in stream.iter().enumerate() {
        let mut out = Vec::new();
        children[i % shards].insert_emit_delta(&DataInput::U64(*k), &mut |d| out.push(d));
        for d in out {
            parent.apply_delta(d);
        }
    }

    let mut reference = Cm::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    // parent + Σ per-shard residuals == the single-pass sketch, cell for cell.
    for r in 0..ROWS {
        for c in 0..COLS {
            let promoted = parent.as_storage().query_one_counter(r, c);
            let residual: i32 = children
                .iter()
                .map(|ch| ch.as_storage().query_one_counter(r, c))
                .sum();
            assert_eq!(
                promoted + residual,
                reference.as_storage().query_one_counter(r, c),
                "cell ({r},{c}) lost or duplicated counts across {shards} shards"
            );
            assert!(
                residual <= shards as i32 * MAX_CELL_RESIDUAL,
                "cell ({r},{c}) residual {residual} exceeds one window per shard"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Count sketch delta protocol
// ---------------------------------------------------------------------------

#[test]
fn count_deltas_carry_exactly_the_signed_threshold() {
    let stream = keys(30_000, 256, 9_201);
    let (_, deltas) = cs_child_run(&stream, ROWS, COLS);

    assert!(!deltas.is_empty(), "a 30k stream must promote something");
    for d in &deltas {
        assert_eq!(
            d.value.unsigned_abs(),
            COUNT_PROMASK,
            "Count resets on promotion, so a delta is always ±the mask"
        );
        assert!((d.row as usize) < ROWS, "row {} out of range", d.row);
        assert!((d.col as usize) < COLS, "col {} out of range", d.col);
    }
}

#[test]
fn count_promotion_is_lossless_modulo_the_child_residual() {
    let stream = keys(30_000, 256, 9_202);
    let (child, deltas) = cs_child_run(&stream, ROWS, COLS);

    let mut parent = Cs::with_dimensions(ROWS, COLS);
    for d in &deltas {
        parent.apply_delta(*d);
    }

    let mut reference = Cs::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    let (child_cells, parent_cells, ref_cells) =
        (cs_cells(&child), cs_cells(&parent), cs_cells(&reference));
    for (i, ((&c, &p), &r)) in child_cells
        .iter()
        .zip(parent_cells.iter())
        .zip(ref_cells.iter())
        .enumerate()
    {
        assert_eq!(
            p + c,
            r,
            "cell {i}: promoted {p} plus residual {c} must reconstruct {r}"
        );
        assert!(
            c.abs() <= MAX_CELL_RESIDUAL,
            "cell {i}: residual {c} should have been promoted"
        );
    }
}

#[test]
fn count_octo_estimate_stays_within_one_residual_of_the_single_thread_estimate() {
    let stream = keys(40_000, 256, 9_203);
    let (_, deltas) = cs_child_run(&stream, ROWS, COLS);

    let mut parent = Cs::with_dimensions(ROWS, COLS);
    for d in &deltas {
        parent.apply_delta(*d);
    }
    let mut reference = Cs::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    // Every row cell is within `MAX_CELL_RESIDUAL` of the reference, and the
    // estimate is an order statistic over those rows, so the same bound holds.
    for k in 0u64..256 {
        let key = DataInput::U64(k);
        let gap = (parent.estimate(&key) - reference.estimate(&key)).abs();
        assert!(
            gap <= MAX_CELL_RESIDUAL as f64,
            "key {k}: estimate gap {gap} exceeds one promotion window"
        );
    }
}

#[test]
fn count_fast_path_conserves_counts_like_the_regular_path() {
    let stream = keys(20_000, 256, 9_204);

    let mut child = CsFast::with_dimensions(ROWS, COLS);
    let mut parent = CsFast::with_dimensions(ROWS, COLS);
    for k in &stream {
        child.insert_emit_delta(&DataInput::U64(*k), &mut |d| {
            assert_eq!(d.value.unsigned_abs(), COUNT_PROMASK);
            parent.apply_delta(d);
        });
    }

    let mut reference = CsFast::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            assert_eq!(
                parent.as_storage().query_one_counter(r, c)
                    + child.as_storage().query_one_counter(r, c),
                reference.as_storage().query_one_counter(r, c),
                "cell ({r},{c})"
            );
        }
    }
}

#[test]
fn count_delta_application_is_order_independent() {
    let stream = keys(15_000, 256, 9_205);
    let (_, deltas) = cs_child_run(&stream, ROWS, COLS);

    let mut in_order = Cs::with_dimensions(ROWS, COLS);
    for d in &deltas {
        in_order.apply_delta(*d);
    }
    let mut shuffled = deltas.clone();
    shuffled.shuffle(&mut StdRng::seed_from_u64(9_205));
    let mut out_of_order = Cs::with_dimensions(ROWS, COLS);
    for d in &shuffled {
        out_of_order.apply_delta(*d);
    }

    assert_eq!(cs_cells(&in_order), cs_cells(&out_of_order));
}

#[test]
fn count_sharded_children_conserve_signed_counts() {
    let stream = keys(40_000, 256, 9_206);
    let shards = 3;

    let mut children: Vec<Cs> = (0..shards)
        .map(|_| Cs::with_dimensions(ROWS, COLS))
        .collect();
    let mut parent = Cs::with_dimensions(ROWS, COLS);
    for (i, k) in stream.iter().enumerate() {
        let mut out = Vec::new();
        children[i % shards].insert_emit_delta(&DataInput::U64(*k), &mut |d| out.push(d));
        for d in out {
            parent.apply_delta(d);
        }
    }

    let mut reference = Cs::with_dimensions(ROWS, COLS);
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            let residual: i32 = children
                .iter()
                .map(|ch| ch.as_storage().query_one_counter(r, c))
                .sum();
            assert_eq!(
                parent.as_storage().query_one_counter(r, c) + residual,
                reference.as_storage().query_one_counter(r, c),
                "cell ({r},{c})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// HyperLogLog delta protocol
// ---------------------------------------------------------------------------

#[test]
fn hll_promotion_reproduces_the_single_thread_registers_exactly() {
    let stream: Vec<u64> = (0..60_000u64).collect();
    let (child, deltas) = hll_child_run(&stream);

    let mut parent = HyperLogLog::<Classic>::default();
    for d in &deltas {
        parent.apply_delta(*d);
    }

    let mut reference = HyperLogLog::<Classic>::default();
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    assert_eq!(
        HLL_PROMASK, 0,
        "the exactness argument relies on every improvement being promoted"
    );
    assert_eq!(
        parent.registers_as_slice(),
        reference.registers_as_slice(),
        "max-register promotion is lossless"
    );
    assert_eq!(
        child.registers_as_slice(),
        reference.registers_as_slice(),
        "the emit path must leave the child identical to a plain insert"
    );
    assert_eq!(parent.estimate(), reference.estimate());
}

#[test]
fn hll_promotion_is_exact_for_every_variant_and_precision() {
    let stream: Vec<u64> = (0..40_000u64)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();

    macro_rules! check {
        ($ty:ty, $label:literal) => {{
            let mut child = <$ty>::default();
            let mut parent = <$ty>::default();
            for k in &stream {
                child.insert_emit_delta(&DataInput::U64(*k), &mut |d| parent.apply_delta(d));
            }
            let mut reference = <$ty>::default();
            for k in &stream {
                reference.insert(&DataInput::U64(*k));
            }
            assert_eq!(
                parent.registers_as_slice(),
                reference.registers_as_slice(),
                "{} registers diverged",
                $label
            );
            assert_eq!(
                child.registers_as_slice(),
                reference.registers_as_slice(),
                "{} child perturbed by the emit path",
                $label
            );
        }};
    }

    check!(HyperLogLogP12<Classic>, "P12/Classic");
    check!(HyperLogLog<Classic>, "P14/Classic");
    check!(HyperLogLogP16<Classic>, "P16/Classic");
    check!(HyperLogLogP12<ErtlMLE>, "P12/ErtlMLE");
    check!(HyperLogLog<ErtlMLE>, "P14/ErtlMLE");
    check!(HyperLogLogP16<ErtlMLE>, "P16/ErtlMLE");
}

#[test]
fn hll_deltas_are_strictly_increasing_per_register_and_never_repeat() {
    let stream: Vec<u64> = (0..30_000u64).collect();
    let (child, deltas) = hll_child_run(&stream);

    let register_count = child.registers_as_slice().len();
    let mut best = vec![0u8; register_count];
    for d in &deltas {
        let pos = d.pos as usize;
        assert!(pos < register_count, "register {pos} out of range");
        assert!(
            d.value > best[pos],
            "register {pos} re-promoted a non-improving value {}",
            d.value
        );
        best[pos] = d.value;
    }
    assert_eq!(
        best,
        child.registers_as_slice(),
        "the delta stream must replay the child's final register state"
    );
}

#[test]
fn hll_duplicate_keys_promote_nothing_after_the_first_improvement() {
    let mut child = HyperLogLog::<Classic>::default();
    let mut deltas = 0usize;
    for _ in 0..1_000 {
        child.insert_emit_delta(&DataInput::U64(7), &mut |_| deltas += 1);
    }
    assert_eq!(deltas, 1, "only the first insert can improve a register");
}

#[test]
fn hll_delta_application_is_order_independent() {
    let stream: Vec<u64> = (0..30_000u64).collect();
    let (_, deltas) = hll_child_run(&stream);

    let mut in_order = HyperLogLog::<Classic>::default();
    for d in &deltas {
        in_order.apply_delta(*d);
    }
    let mut shuffled = deltas.clone();
    shuffled.shuffle(&mut StdRng::seed_from_u64(9_301));
    let mut out_of_order = HyperLogLog::<Classic>::default();
    for d in &shuffled {
        out_of_order.apply_delta(*d);
    }

    assert_eq!(
        in_order.registers_as_slice(),
        out_of_order.registers_as_slice(),
        "max is commutative, so delivery order is free"
    );
}

#[test]
fn hll_sharded_children_match_a_single_pass_exactly() {
    let stream: Vec<u64> = (0..50_000u64).collect();
    let shards = 4;

    let mut children: Vec<HyperLogLog<Classic>> =
        (0..shards).map(|_| HyperLogLog::default()).collect();
    let mut parent = HyperLogLog::<Classic>::default();
    for (i, k) in stream.iter().enumerate() {
        let mut out = Vec::new();
        children[i % shards].insert_emit_delta(&DataInput::U64(*k), &mut |d| out.push(d));
        for d in out {
            parent.apply_delta(d);
        }
    }

    let mut reference = HyperLogLog::<Classic>::default();
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    assert_eq!(
        parent.registers_as_slice(),
        reference.registers_as_slice(),
        "sharded HLL promotion carries no partition penalty"
    );
}

#[test]
fn hll_cardinality_survives_the_delta_round_trip_exactly() {
    let truth = 100_000u64;
    let stream: Vec<u64> = (0..truth).collect();
    let (_, deltas) = hll_child_run(&stream);
    let mut parent = HyperLogLog::<Classic>::default();
    for d in &deltas {
        parent.apply_delta(*d);
    }
    let mut reference = HyperLogLog::<Classic>::default();
    for k in &stream {
        reference.insert(&DataInput::U64(*k));
    }

    // At HLL_PROMASK = 0 every improvement is promoted, so this is an equality,
    // not a tolerance: a band would pass on an implementation that dropped a
    // slice of the register improvements.
    assert_eq!(
        parent.registers_as_slice(),
        reference.registers_as_slice(),
        "delta-fed registers must match a single pass byte for byte"
    );
    assert_eq!(parent.estimate(), reference.estimate());
    // The sketch's own accuracy, for context: P14 standard error is
    // 1.04/sqrt(2^14) ~ 0.81%.
    let error = (parent.estimate() as f64 - truth as f64).abs() / truth as f64;
    assert!(error < 0.025, "P14 error {error:.4} exceeds 3 sigma");
}

// ---------------------------------------------------------------------------
// Runtime: `run_octo` / `OctoRuntime`
// ---------------------------------------------------------------------------

#[cfg(feature = "octo-runtime")]
mod runtime {
    use super::*;
    use asap_sketchlib::{
        CmOctoAggregator, CmOctoPlan, CmTopKOctoAggregator, CmTopKOctoPlan, CountOctoAggregator,
        CountOctoPlan, DdOctoAggregator, DdOctoPlan, HllOctoAggregator, HllOctoPlan,
        OctoAggregator, OctoConfig, OctoPartition, OctoPlan, OctoRuntime, OctoWorker, run_octo,
    };

    fn config(num_workers: usize) -> OctoConfig {
        OctoConfig {
            num_workers,
            // CI runners have fewer cores than the widest configuration here.
            pin_cores: false,
            queue_capacity: 4096,
            ..OctoConfig::default()
        }
    }

    fn cm_runtime(inputs: &[DataInput<'_>], workers: usize, rows: usize, cols: usize) -> Cm {
        run_octo(
            inputs,
            &config(workers),
            CmOctoPlan::new(rows, cols),
            || CmOctoAggregator {
                sketch: Cm::with_dimensions(rows, cols),
            },
        )
        .parent
        .sketch
    }

    fn cm_replay(inputs: &[DataInput<'_>], workers: usize, rows: usize, cols: usize) -> Cm {
        let mut replay = OctoCm::new(workers, rows, cols, Route::HashByKey);
        for (i, input) in inputs.iter().enumerate() {
            replay.insert(i, input);
        }
        // `run_octo` flushes when the stream ends, so the replay must too.
        replay.flush();
        replay.parent
    }

    fn cs_replay(inputs: &[DataInput<'_>], workers: usize, rows: usize, cols: usize) -> Cs {
        let mut replay = OctoCs::new(workers, rows, cols, Route::HashByKey);
        for (i, input) in inputs.iter().enumerate() {
            replay.insert(i, input);
        }
        replay.flush();
        replay.parent
    }

    #[test]
    fn run_octo_cm_matches_a_single_threaded_replay_of_the_same_partition() {
        let inputs = inputs_from(&keys(40_000, 512, 9_401));
        for workers in [1usize, 2, 3, 4, 7] {
            assert_eq!(
                cm_cells(&cm_runtime(&inputs, workers, ROWS, COLS)),
                cm_cells(&cm_replay(&inputs, workers, ROWS, COLS)),
                "{workers} workers: runtime diverged from the deterministic replay"
            );
        }
    }

    #[test]
    fn run_octo_count_matches_a_single_threaded_replay_of_the_same_partition() {
        let inputs = inputs_from(&keys(40_000, 256, 9_402));
        for workers in [1usize, 2, 4] {
            let got = run_octo(
                &inputs,
                &config(workers),
                CountOctoPlan::new(ROWS, COLS),
                || CountOctoAggregator {
                    sketch: Cs::with_dimensions(ROWS, COLS),
                },
            )
            .parent
            .sketch;
            assert_eq!(
                cs_cells(&got),
                cs_cells(&cs_replay(&inputs, workers, ROWS, COLS)),
                "{workers} workers: runtime diverged from the deterministic replay"
            );
        }
    }

    #[test]
    fn run_octo_hll_is_bit_exact_and_worker_count_invariant() {
        let inputs = inputs_from(&(0..80_000u64).collect::<Vec<_>>());
        let mut reference = HyperLogLog::<Classic>::default();
        for input in &inputs {
            reference.insert(input);
        }

        for workers in [1usize, 2, 3, 4, 8] {
            let got = run_octo(&inputs, &config(workers), HllOctoPlan::new(), || {
                HllOctoAggregator {
                    sketch: HyperLogLog::<Classic>::default(),
                }
            })
            .parent
            .sketch;
            assert_eq!(
                got.registers_as_slice(),
                reference.registers_as_slice(),
                "{workers} workers: HLL promotion must carry no partition penalty"
            );
        }
    }

    #[test]
    fn run_octo_ddsketch_matches_a_single_threaded_replay() {
        // Bucket deltas are additive, so the parent is a pure function of the
        // partition and thread interleaving cannot change it.
        let alpha = 0.01;
        let values: Vec<f64> = (1..=40_000u64)
            .map(|i| 1.0 + (i as f64 * 7.0) % 999.0)
            .collect();
        let inputs: Vec<DataInput<'_>> = values.iter().map(|v| DataInput::F64(*v)).collect();
        let workers = 4usize;

        let got = run_octo(&inputs, &config(workers), DdOctoPlan::new(alpha), || {
            DdOctoAggregator::new(alpha)
        })
        .parent
        .sketch;

        let mut replay = OctoDd::new(workers, alpha);
        for (i, value) in values.iter().enumerate() {
            replay.insert(i, *value, DD_PROMASK);
        }
        // `run_octo` flushes when the stream ends, so the replay must too.
        let mut children = std::mem::take(&mut replay.children);
        for child in children.iter_mut() {
            child.flush(&mut |d| replay.parent.apply_delta(d));
        }

        // The dense store grows in chunks anchored on whichever bucket is
        // touched first, so its offset and array length depend on delta arrival
        // order. What must match is the logical bucket-to-count mapping.
        let buckets = |sketch: &DDSketch| -> Vec<(i32, u64)> {
            sketch
                .store_counts()
                .iter()
                .enumerate()
                .filter(|(_, count)| **count != 0)
                .map(|(i, count)| (sketch.store_offset() + i as i32, *count))
                .collect()
        };
        assert_eq!(got.get_count(), replay.parent.get_count());
        assert_eq!(buckets(&got), buckets(&replay.parent));
    }

    #[test]
    fn run_octo_is_deterministic_across_repeated_runs() {
        let inputs = inputs_from(&keys(30_000, 512, 9_403));
        let first = cm_runtime(&inputs, 4, ROWS, COLS);
        let second = cm_runtime(&inputs, 4, ROWS, COLS);
        assert_eq!(
            cm_cells(&first),
            cm_cells(&second),
            "commutative aggregation makes the parent independent of interleaving"
        );
    }

    #[test]
    fn streaming_runtime_matches_the_batch_helper() {
        let inputs = inputs_from(&keys(30_000, 512, 9_404));
        let batch = cm_runtime(&inputs, 4, ROWS, COLS);

        let mut runtime = OctoRuntime::new(&config(4), CmOctoPlan::new(ROWS, COLS), || {
            CmOctoAggregator {
                sketch: Cm::with_dimensions(ROWS, COLS),
            }
        });
        for input in &inputs {
            runtime.insert(input.clone());
        }
        let streamed = runtime.finish().parent.sketch;

        assert_eq!(cm_cells(&batch), cm_cells(&streamed));
    }

    #[test]
    fn insert_batch_matches_element_wise_inserts() {
        let inputs = inputs_from(&keys(20_000, 512, 9_405));

        let mut one_by_one = OctoRuntime::new(&config(3), CmOctoPlan::new(ROWS, COLS), || {
            CmOctoAggregator {
                sketch: Cm::with_dimensions(ROWS, COLS),
            }
        });
        for input in &inputs {
            one_by_one.insert(input.clone());
        }

        let mut batched = OctoRuntime::new(&config(3), CmOctoPlan::new(ROWS, COLS), || {
            CmOctoAggregator {
                sketch: Cm::with_dimensions(ROWS, COLS),
            }
        });
        batched.insert_batch(&inputs);

        assert_eq!(
            cm_cells(&one_by_one.finish().parent.sketch),
            cm_cells(&batched.finish().parent.sketch)
        );
    }

    #[test]
    fn degenerate_config_is_clamped_rather_than_rejected() {
        let inputs = inputs_from(&(0..5_000u64).collect::<Vec<_>>());
        let cfg = OctoConfig {
            num_workers: 0,
            pin_cores: false,
            queue_capacity: 0,
            ..OctoConfig::default()
        };
        let got = run_octo(&inputs, &cfg, HllOctoPlan::new(), || HllOctoAggregator {
            sketch: HyperLogLog::<Classic>::default(),
        })
        .parent
        .sketch;

        let mut reference = HyperLogLog::<Classic>::default();
        for input in &inputs {
            reference.insert(input);
        }
        assert_eq!(
            got.registers_as_slice(),
            reference.registers_as_slice(),
            "zero workers/capacity must clamp to one, not drop data"
        );
    }

    #[test]
    fn a_one_slot_queue_applies_backpressure_without_deadlocking() {
        let inputs = inputs_from(&(0..20_000u64).collect::<Vec<_>>());
        let cfg = OctoConfig {
            num_workers: 4,
            pin_cores: false,
            queue_capacity: 1,
            ..OctoConfig::default()
        };
        let got = run_octo(&inputs, &cfg, HllOctoPlan::new(), || HllOctoAggregator {
            sketch: HyperLogLog::<Classic>::default(),
        })
        .parent
        .sketch;

        let mut reference = HyperLogLog::<Classic>::default();
        for input in &inputs {
            reference.insert(input);
        }
        assert_eq!(got.registers_as_slice(), reference.registers_as_slice());
    }

    #[test]
    fn a_borrowed_key_may_be_dropped_the_moment_insert_returns() {
        // The point of a borrowed key is that its owner does not have to
        // outlive the sketch. `insert` hashes for partitioning and prepares the
        // payload on this thread, so nothing borrowed crosses to a worker and
        // the string below is free the instant the call returns.
        let n = 20_000u64;
        let mut runtime = OctoRuntime::new(&config(4), HllOctoPlan::new(), HllOctoAggregator::new);
        for i in 0..n {
            let owned = format!("session-{i:020}");
            runtime.insert(DataInput::Str(&owned));
            drop(owned);
        }
        let got = runtime.finish().parent.sketch;

        let mut reference = HyperLogLog::<Classic>::default();
        for i in 0..n {
            reference.insert(&DataInput::Str(&format!("session-{i:020}")));
        }
        assert_eq!(
            got.registers_as_slice(),
            reference.registers_as_slice(),
            "a key freed right after insert must still have been hashed correctly"
        );
    }

    #[test]
    fn a_borrowed_key_reaches_a_keyed_aggregator_by_value() {
        // A worker that stores keys copies at preparation, which is the one
        // place a copy is unavoidable - and it happens on this thread, so the
        // borrow still ends at the call.
        let (rows, cols, top_k) = (4usize, 2048usize, 8usize);
        let mut runtime = OctoRuntime::new(&config(4), CmTopKOctoPlan::new(rows, cols), || {
            CmTopKOctoAggregator::new(rows, cols, top_k)
        });
        for _ in 0..4_000 {
            for hot in 0..3usize {
                let owned = format!("hot-{hot}");
                runtime.insert(DataInput::Str(&owned));
                drop(owned);
            }
        }
        let result = runtime.finish();
        for hot in 0..3usize {
            let key = format!("hot-{hot}");
            assert!(
                result
                    .parent
                    .sketch
                    .heap()
                    .find(&DataInput::Str(&key))
                    .is_some(),
                "heavy hitter {key} missing from the aggregator heap"
            );
        }
    }

    #[test]
    fn an_empty_stream_finishes_with_a_pristine_parent() {
        let got = run_octo(&[], &config(4), CmOctoPlan::new(ROWS, COLS), || {
            CmOctoAggregator {
                sketch: Cm::with_dimensions(ROWS, COLS),
            }
        })
        .parent
        .sketch;
        assert!(cm_cells(&got).iter().all(|&c| c == 0));
    }

    // -- custom worker/aggregator: the public trait surface must be usable
    //    from outside the crate, not just by the bundled adapters.

    struct SumWorker {
        worker_id: usize,
        seen: u64,
    }

    #[derive(Clone, Copy)]
    struct SumDelta {
        worker_id: usize,
        keys_seen: u64,
    }

    impl OctoWorker for SumWorker {
        type Delta = SumDelta;
        type Payload = ();

        fn process<F>(&mut self, _payload: &(), emit: &mut F)
        where
            F: FnMut(Self::Delta),
        {
            self.seen += 1;
            emit(SumDelta {
                worker_id: self.worker_id,
                keys_seen: 1,
            });
        }
    }

    struct SumPlan;

    impl OctoPlan for SumPlan {
        type Worker = SumWorker;

        fn worker(&self, worker_id: usize) -> Self::Worker {
            SumWorker { worker_id, seen: 0 }
        }

        fn prepare(&self, _input: &DataInput<'_>) {}
    }

    struct SumAggregator {
        per_worker: Vec<u64>,
    }

    impl OctoAggregator for SumAggregator {
        type Delta = SumDelta;

        fn apply(&mut self, delta: SumDelta) {
            self.per_worker[delta.worker_id] += delta.keys_seen;
        }
    }

    #[test]
    fn a_user_defined_worker_and_aggregator_round_trip_every_input() {
        let workers = 3;
        let n = 10_001u64;
        let inputs = inputs_from(&(0..n).collect::<Vec<_>>());

        let cfg = OctoConfig {
            partition: OctoPartition::RoundRobin,
            ..config(workers)
        };
        let result = run_octo(&inputs, &cfg, SumPlan, || SumAggregator {
            per_worker: vec![0; workers],
        });

        assert_eq!(
            result.parent.per_worker.iter().sum::<u64>(),
            n,
            "no input may be dropped"
        );
        // Round-robin from a single dispatching thread: worker w takes every
        // index ≡ w (mod workers).
        for (w, &load) in result.parent.per_worker.iter().enumerate() {
            let expected = (0..n as usize).filter(|i| i % workers == w).count() as u64;
            assert_eq!(load, expected, "worker {w} received the wrong share");
        }
    }

    #[test]
    fn the_read_handle_observes_a_monotone_prefix_of_the_final_state() {
        let workers = 2;
        let n = 4_000u64;
        let mut runtime = OctoRuntime::new(&config(workers), SumPlan, || SumAggregator {
            per_worker: vec![0; workers],
        });
        let reader = runtime.read_handle();

        assert_eq!(
            reader.with_parent(|p| p.per_worker.iter().sum::<u64>()),
            0,
            "nothing inserted yet"
        );

        let mut last = 0u64;
        for i in 0..n {
            runtime.insert(DataInput::U64(i));
            if i % 500 == 0 {
                let observed = reader.with_parent(|p| p.per_worker.iter().sum::<u64>());
                assert!(
                    observed >= last,
                    "live total went backwards: {last} -> {observed}"
                );
                assert!(observed <= n, "live total {observed} exceeded the stream");
                last = observed;
            }
        }

        // Monotonicity alone is satisfied by a handle stuck at zero, so wait
        // for it to actually reach the live total while the runtime is still
        // open. That is the property the handle exists for.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let observed = reader.with_parent(|p| p.per_worker.iter().sum::<u64>());
            assert!(observed >= last, "live total went backwards");
            last = observed;
            if observed == n {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "read handle stalled at {observed}/{n}"
            );
            std::hint::spin_loop();
        }

        assert_eq!(runtime.finish().parent.per_worker.iter().sum::<u64>(), n);
    }

    #[test]
    #[should_panic(expected = "Octo runtime has been finished")]
    fn reading_through_a_stale_handle_panics() {
        let workers = 2;
        let runtime = OctoRuntime::new(&config(workers), SumPlan, || SumAggregator {
            per_worker: vec![0; workers],
        });
        let reader = runtime.read_handle();
        let _ = runtime.finish();
        reader.with_parent(|p| p.per_worker.len());
    }

    #[test]
    fn close_is_idempotent_and_preserves_already_queued_work() {
        let workers = 4;
        let n = 3_001u64;
        let mut runtime = OctoRuntime::new(&config(workers), SumPlan, || SumAggregator {
            per_worker: vec![0; workers],
        });
        for i in 0..n {
            runtime.insert(DataInput::U64(i));
        }
        runtime.close();
        runtime.close();

        let result = runtime.finish();
        assert_eq!(
            result.parent.per_worker.iter().sum::<u64>(),
            n,
            "close must drain what was already queued"
        );
    }

    #[test]
    #[should_panic(expected = "cannot insert after runtime has been closed")]
    fn inserting_after_close_panics() {
        let workers = 2;
        let mut runtime = OctoRuntime::new(&config(workers), SumPlan, || SumAggregator {
            per_worker: vec![0; workers],
        });
        runtime.close();
        runtime.insert(DataInput::U64(1));
    }

    // -- accuracy end to end -------------------------------------------------

    #[test]
    fn a_finished_run_carries_only_count_mins_own_one_sided_error() {
        let rows = 5;
        let cols = 4096;
        let stream = keys(200_000, 4_096, 9_501);
        let inputs = inputs_from(&stream);

        let mut truth = FreqTruth::default();
        for k in &stream {
            truth.observe(*k as i64);
        }

        let parent = cm_runtime(&inputs, 4, rows, cols);

        // `run_octo` flushes when the stream ends, so no promotion residue
        // survives and the protocol contributes nothing: what is left is
        // Count-Min's own one-sided error, never below truth and at most eps*N
        // above it with eps = e/cols. Between flushes the parent would instead
        // sit up to k*tau low - that band is what the theorem tests measure.
        let epsilon_n = (std::f64::consts::E / cols as f64) * truth.total() as f64;
        for k in 0u64..4_096 {
            let exact = truth.get(k as i64) as f64;
            let est = parent.estimate(&DataInput::U64(k)) as f64;
            assert!(
                est >= exact,
                "key {k}: a flushed Count-Min must never underestimate: {est} < {exact}"
            );
            assert!(
                est <= exact + epsilon_n,
                "key {k}: octo estimate {est} exceeded truth {exact} + {epsilon_n:.1}"
            );
        }
    }

    #[test]
    fn run_octo_hll_cardinality_error_stays_within_three_sigma() {
        let truth = 200_000u64;
        let inputs = inputs_from(&(0..truth).collect::<Vec<_>>());
        let got = run_octo(&inputs, &config(4), HllOctoPlan::new(), || {
            HllOctoAggregator {
                sketch: HyperLogLog::<Classic>::default(),
            }
        })
        .parent
        .sketch;

        // Max-register promotion is lossless, so the runtime's parent must be
        // the single-threaded sketch exactly, not merely within 3 sigma of it.
        let mut reference = HyperLogLog::<Classic>::default();
        for input in &inputs {
            reference.insert(input);
        }
        assert_eq!(got.registers_as_slice(), reference.registers_as_slice());
        let estimate = got.estimate() as f64;
        let error = (estimate - truth as f64).abs() / truth as f64;
        assert!(
            error < 0.025,
            "octo HLL error {error:.4} exceeds 3σ (estimate {estimate}, truth {truth})"
        );
    }

    #[test]
    fn run_octo_count_frequencies_track_a_zipf_stream() {
        let rows = 5;
        let cols = 4096;
        let stream = keys(200_000, 2_048, 9_502);
        let inputs = inputs_from(&stream);

        let mut truth = FreqTruth::default();
        for k in &stream {
            truth.observe(*k as i64);
        }

        let parent = run_octo(&inputs, &config(4), CountOctoPlan::new(rows, cols), || {
            CountOctoAggregator {
                sketch: Cs::with_dimensions(rows, cols),
            }
        })
        .parent
        .sketch;

        // A finished run is flushed, so the promotion protocol adds nothing
        // and what is left is the Count-Sketch bound itself: +/- ||f||_2/sqrt(cols) w.h.p.
        let tolerance = 3.0 * truth.l2_norm() / (cols as f64).sqrt();
        let mut violations = Vec::new();
        for (key, exact) in truth.top_k(64) {
            let est = parent.estimate(&DataInput::U64(key as u64));
            if (est - exact as f64).abs() > tolerance {
                violations.push(format!("key {key}: est {est} vs exact {exact}"));
            }
        }
        assert!(
            violations.is_empty(),
            "count-sketch estimates outside ±{tolerance:.1}: {violations:?}"
        );
    }
}

// ===========================================================================
// Paper conformance: OctoSketch (NSDI '24) Theorems 1-4 and its baseline
// ===========================================================================

// ---------------------------------------------------------------------------
// Replays: OctoSketch delta promotion vs. the periodic sketch-merge baseline
// ---------------------------------------------------------------------------

/// How a replay routes an input to a worker, mirroring `OctoPartition`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// One flow, one worker: the paper's setting, k' = 1.
    HashByKey,
    /// Every flow reaches every worker: k' = k, the worst case of each bound.
    RoundRobin,
}

impl Route {
    fn worker(self, index: usize, key: &DataInput, workers: usize) -> usize {
        match self {
            Route::RoundRobin => index % workers,
            Route::HashByKey => (hash64_seeded(0, key) % workers as u64) as usize,
        }
    }

    /// k' in the paper's bounds: the number of workers one flow may reach.
    fn k_prime(self, workers: usize) -> usize {
        match self {
            Route::RoundRobin => workers,
            Route::HashByKey => 1,
        }
    }
}

/// OctoSketch: workers promote single counters once they cross τ.
struct OctoCm {
    children: Vec<Cm>,
    parent: Cm,
    route: Route,
    /// Counters shipped to the aggregator, the unit Theorem 2 counts in.
    sent_counters: usize,
}

impl OctoCm {
    fn new(workers: usize, rows: usize, cols: usize, route: Route) -> Self {
        Self {
            children: (0..workers)
                .map(|_| Cm::with_dimensions(rows, cols))
                .collect(),
            parent: Cm::with_dimensions(rows, cols),
            route,
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = self.route.worker(index, key, self.children.len());
        let mut promoted = Vec::new();
        self.children[worker].insert_emit_delta(key, &mut |d| promoted.push(d));
        self.sent_counters += promoted.len();
        for d in promoted {
            self.parent.apply_delta(d);
        }
    }

    /// Hands over every residual counter, mirroring what the runtime does when
    /// a stream is finished or handed over for querying. The bound and
    /// conservation tests deliberately do not call this: the residue is what
    /// they are measuring.
    #[cfg(feature = "octo-runtime")]
    fn flush(&mut self) {
        let (rows, cols) = (self.parent.rows(), self.parent.cols());
        for child in self.children.iter_mut() {
            for row in 0..rows {
                for col in 0..cols {
                    let held = child.as_storage().query_one_counter(row, col);
                    if held != 0 {
                        self.parent.apply_delta(CmDelta {
                            row: row as u32,
                            col: col as u32,
                            value: held as u32,
                        });
                        child
                            .as_storage_mut()
                            .update_one_counter(row, col, |c, _| *c = 0, ());
                        self.sent_counters += 1;
                    }
                }
            }
        }
    }
}

/// Sketch-merge: each worker keeps a full sketch and ships all `rows*cols`
/// counters every `period` items it sees, then restarts from zero.
struct MergeCm {
    children: Vec<Cm>,
    seen: Vec<usize>,
    parent: Cm,
    period: usize,
    rows: usize,
    cols: usize,
    route: Route,
    sent_counters: usize,
}

impl MergeCm {
    fn new(workers: usize, rows: usize, cols: usize, period: usize, route: Route) -> Self {
        Self {
            children: (0..workers)
                .map(|_| Cm::with_dimensions(rows, cols))
                .collect(),
            seen: vec![0; workers],
            parent: Cm::with_dimensions(rows, cols),
            period,
            rows,
            cols,
            route,
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = self.route.worker(index, key, self.children.len());
        self.children[worker].insert(key);
        self.seen[worker] += 1;
        if self.seen[worker] % self.period == 0 {
            self.parent.merge(&self.children[worker]);
            self.children[worker] = Cm::with_dimensions(self.rows, self.cols);
            self.sent_counters += self.rows * self.cols;
        }
    }
}

/// Count-sketch flavour of `OctoCm`.
struct OctoCs {
    children: Vec<Cs>,
    parent: Cs,
    route: Route,
    sent_counters: usize,
}

impl OctoCs {
    fn new(workers: usize, rows: usize, cols: usize, route: Route) -> Self {
        Self {
            children: (0..workers)
                .map(|_| Cs::with_dimensions(rows, cols))
                .collect(),
            parent: Cs::with_dimensions(rows, cols),
            route,
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = self.route.worker(index, key, self.children.len());
        let mut promoted = Vec::new();
        self.children[worker].insert_emit_delta(key, &mut |d| promoted.push(d));
        self.sent_counters += promoted.len();
        for d in promoted {
            self.parent.apply_delta(d);
        }
    }

    /// See `OctoCm::flush`.
    #[cfg(feature = "octo-runtime")]
    fn flush(&mut self) {
        let (rows, cols) = (self.parent.rows(), self.parent.cols());
        for child in self.children.iter_mut() {
            for row in 0..rows {
                for col in 0..cols {
                    let held = child.as_storage().query_one_counter(row, col);
                    if held != 0 {
                        self.parent.apply_delta(CountDelta {
                            row: row as u32,
                            col: col as u32,
                            value: held,
                        });
                        child
                            .as_storage_mut()
                            .update_one_counter(row, col, |c, _| *c = 0, ());
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Theorem 1: Count-Min error bound
// ---------------------------------------------------------------------------

#[test]
fn theorem_1_bounds_the_count_min_error() {
    // "let d = log2 δ⁻¹ and l = 2ε⁻¹. For any flow e and any traffic whose
    //  L1 > ε⁻¹k'τ, Pr[|f̂(e) − f(e)| > εL1] < δ"
    let d = 5usize;
    let l = 4096usize;
    let workers = 4usize;
    let epsilon = 2.0 / l as f64;
    let delta = 2f64.powi(-(d as i32));

    let stream = zipf_u64(400_000, 8_192, 1.1, 11_001);
    let mut truth = FreqTruth::default();
    for key in &stream {
        truth.observe(*key as i64);
    }
    let mut measured: Vec<(Route, f64)> = Vec::new();
    for route in [Route::HashByKey, Route::RoundRobin] {
        let k_prime_tau = (route.k_prime(workers) as i32 * TAU) as f64;
        // A counter is shared by every flow that hashes into it, and each worker
        // may hold back up to tau of its own share, so the provable per-counter
        // gap is k*tau whatever the partition. The paper's k'*tau bounds only the
        // queried flow's own held-back count.
        let counter_ceiling = (workers as i32 * TAU) as f64;
        let mut octo = OctoCm::new(workers, d, l, route);
        for (i, key) in stream.iter().enumerate() {
            octo.insert(i, &DataInput::U64(*key));
        }

        let l1 = truth.total() as f64;
        assert!(
            l1 > k_prime_tau / epsilon,
            "theorem precondition L1 > ε⁻¹k'τ unmet: L1 = {l1}, ε⁻¹k'τ = {}",
            k_prime_tau / epsilon
        );

        let band = epsilon * l1;
        let mut violations = 0usize;
        let mut worst_deficit = 0.0f64;
        let mut worst_excess = 0.0f64;
        let pairs = truth.pairs();
        for (key, exact) in &pairs {
            let est = octo.parent.estimate(&DataInput::U64(*key as u64)) as f64;
            let exact = *exact as f64;
            if (est - exact).abs() > band {
                violations += 1;
            }
            worst_deficit = worst_deficit.max(exact - est);
            worst_excess = worst_excess.max(est - exact);
        }

        // The proof's deterministic half: f̂ ≤ f̂' and |f̂' − f̂| < k'τ, and Count-Min
        // never underestimates, so f(e) − k'τ < f̂(e) for *every* key, not merely
        // with probability 1 − δ.
        assert!(
            worst_deficit < counter_ceiling,
            "worst underestimate {worst_deficit} reached the k*tau = {counter_ceiling} ceiling"
        );
        measured.push((route, worst_deficit));

        let rate = violations as f64 / pairs.len() as f64;
        println!(
            "Theorem 1 [{route:?}]: eps={epsilon:.2e} delta={delta:.4} eps*L1={band:.1} \
         k'tau={k_prime_tau} violations={rate:.5} worst_deficit={worst_deficit} \
         worst_excess={worst_excess}"
        );
        assert!(
            rate < delta,
            "{route:?}: violation rate {rate:.5} exceeded δ = {delta:.5} over {} keys",
            pairs.len()
        );
    }

    // Hash partitioning keeps a flow's whole count on one worker, so it should
    // never trail further behind than round-robin does.
    let hashed = measured[0].1;
    let round_robin = measured[1].1;
    assert!(
        hashed <= round_robin,
        "hash partitioning lagged more than round-robin: {hashed} vs {round_robin}"
    );
}

// ---------------------------------------------------------------------------
// Theorem 3: Count sketch error bound
// ---------------------------------------------------------------------------

#[test]
fn theorem_3_bounds_the_count_sketch_error() {
    // "let d = O(log2 δ⁻¹) and l = 8ε⁻². For any flow e and any traffic whose
    //  L2 > 2ε⁻¹k'τ, Pr[|f̂(e) − f(e)| > εL2] < δ"
    assert_eq!(
        COUNT_PROMASK, CM_PROMASK,
        "both sketches share one τ in this implementation"
    );
    let d = 5usize;
    let l = 4096usize;
    let workers = 4usize;
    let epsilon = (8.0 / l as f64).sqrt();
    let delta = 2f64.powi(-(d as i32));

    let stream = zipf_u64(400_000, 8_192, 1.1, 11_003);
    let mut truth = FreqTruth::default();
    let mut single_core = Cs::with_dimensions(d, l);
    for key in &stream {
        truth.observe(*key as i64);
        single_core.insert(&DataInput::U64(*key));
    }
    let mut measured: Vec<(Route, f64)> = Vec::new();
    for route in [Route::HashByKey, Route::RoundRobin] {
        let k_prime_tau = (route.k_prime(workers) as i32 * TAU) as f64;
        // See Theorem 1: the per-counter gap ceiling is k*tau, not k'*tau.
        let counter_ceiling = (workers as i32 * TAU) as f64;
        let mut octo = OctoCs::new(workers, d, l, route);
        for (i, key) in stream.iter().enumerate() {
            octo.insert(i, &DataInput::U64(*key));
        }

        let l2 = truth.l2_norm();
        assert!(
            l2 > 2.0 * k_prime_tau / epsilon,
            "theorem precondition L2 > 2ε⁻¹k'τ unmet: L2 = {l2}, 2ε⁻¹k'τ = {}",
            2.0 * k_prime_tau / epsilon
        );

        let band = epsilon * l2;
        let mut violations = 0usize;
        let mut worst_octo_gap = 0.0f64;
        let pairs = truth.pairs();
        for (key, exact) in &pairs {
            let input = DataInput::U64(*key as u64);
            let est = octo.parent.estimate(&input);
            if (est - *exact as f64).abs() > band {
                violations += 1;
            }
            // The proof's deterministic half: |f̂'(e) − f̂(e)| < k'τ.
            worst_octo_gap = worst_octo_gap.max((single_core.estimate(&input) - est).abs());
        }

        assert!(
            worst_octo_gap < counter_ceiling,
            "gap to the single-core sketch {worst_octo_gap} reached the k*tau = {counter_ceiling} ceiling"
        );
        measured.push((route, worst_octo_gap));

        let rate = violations as f64 / pairs.len() as f64;
        println!(
            "Theorem 3 [{route:?}]: eps={epsilon:.4} delta={delta:.4} eps*L2={band:.1} \
         k'tau={k_prime_tau} violations={rate:.5} worst_gap_to_single_core={worst_octo_gap}"
        );
        assert!(
            rate < delta,
            "{route:?}: violation rate {rate:.5} exceeded δ = {delta:.5} over {} keys",
            pairs.len()
        );
    }

    let hashed = measured[0].1;
    let round_robin = measured[1].1;
    assert!(
        hashed <= round_robin,
        "hash partitioning lagged more than round-robin: {hashed} vs {round_robin}"
    );
}

// ---------------------------------------------------------------------------
// Theorem 4: HyperLogLog exactness
// ---------------------------------------------------------------------------

#[test]
fn theorem_4_holds_unconditionally_because_this_implementation_pins_tau_to_zero() {
    // Theorem 4 only promises Ẑ = Ẑ' once Ẑ > 2·α_m·m²·2^(τ−2), because a
    // worker holds back register improvements below τ−1. This implementation
    // sets HLL_PROMASK = 0, so C'[i] = C[i] for every i and the equality holds
    // at any cardinality - including ones orders of magnitude below the
    // precondition the paper would otherwise need.
    assert_eq!(HLL_PROMASK, 0);

    let m = HyperLogLog::<Classic>::default().registers_as_slice().len() as f64;
    let alpha_m = 0.7213 / (1.0 + 1.079 / m);
    // The precondition evaluated at the smallest threshold the paper suggests
    // for HLL (Appendix B: "the threshold τ = 2 is often enough").
    let precondition = 2.0 * alpha_m * m * m * 2f64.powi(2 - 2);

    for n in [10u64, 100, 1_000, 10_000] {
        let mut child = HyperLogLog::<Classic>::default();
        let mut parent = HyperLogLog::<Classic>::default();
        for i in 0..n {
            child.insert_emit_delta(&DataInput::U64(i), &mut |d| parent.apply_delta(d));
        }
        let mut ideal = HyperLogLog::<Classic>::default();
        for i in 0..n {
            ideal.insert(&DataInput::U64(i));
        }

        assert!(
            (n as f64) < precondition,
            "n = {n} is above the paper's precondition {precondition:.3e}, \
             so this case would not demonstrate the unconditional guarantee"
        );
        assert_eq!(
            parent.registers_as_slice(),
            ideal.registers_as_slice(),
            "n = {n}: Ẑ' must equal Ẑ even far below the τ = 2 precondition"
        );
        assert_eq!(parent.estimate(), ideal.estimate(), "n = {n}");
    }
    println!(
        "Theorem 4: tau=0 gives Z' = Z unconditionally; the tau=2 precondition \
         would only bind above Z > {precondition:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Theorem 2: communication cost against sketch-merge
// ---------------------------------------------------------------------------

#[test]
fn theorem_2_delta_promotion_sends_l_times_fewer_counters_for_the_same_goal() {
    // "To achieve the accuracy goal, sketch-merge needs to send O(Δ⁻¹·k·N·d·l)
    //  counters, while OctoSketch needs to send O(Δ⁻¹·k·N·d) counters."
    //
    // The proof holds both sides to the same accuracy goal Δ: sketch-merge
    // ships the whole sketch every Δ/k items per worker, and OctoSketch sets
    // its threshold to τ = Δ/k. This implementation pins τ = 31, so the goal
    // it is solving for is Δ = k·τ.
    let d = 3usize;
    let l = 1_024usize;
    let workers = 4usize;
    // A counter is shared across workers whatever the partition, so the goal a
    // shipped tau solves for is k*tau, and sketch-merge must match it by
    // shipping every tau items per worker.
    let accuracy_goal = (workers as i32 * TAU) as f64;
    let merge_period = TAU as usize;

    let stream = zipf_u64(40_000, 2_048, 1.1, 11_005);
    let mut octo = OctoCm::new(workers, d, l, Route::HashByKey);
    let mut merge = MergeCm::new(workers, d, l, merge_period, Route::HashByKey);
    let mut single_core = Cm::with_dimensions(d, l);
    for (i, key) in stream.iter().enumerate() {
        let input = DataInput::U64(*key);
        octo.insert(i, &input);
        merge.insert(i, &input);
        single_core.insert(&input);
    }

    // Both configurations must actually meet the goal before their costs are
    // comparable - that is the premise of the theorem, not a side check.
    let mut worst_octo = 0.0f64;
    let mut worst_merge = 0.0f64;
    for key in 0u64..2_048 {
        let input = DataInput::U64(key);
        let ideal = single_core.estimate(&input) as f64;
        worst_octo = worst_octo.max(ideal - octo.parent.estimate(&input) as f64);
        worst_merge = worst_merge.max(ideal - merge.parent.estimate(&input) as f64);
    }
    assert!(
        worst_octo <= accuracy_goal,
        "octo missed the accuracy goal: {worst_octo} > {accuracy_goal}"
    );
    assert!(
        worst_merge <= accuracy_goal,
        "sketch-merge missed the accuracy goal: {worst_merge} > {accuracy_goal}"
    );

    let ratio = merge.sent_counters as f64 / octo.sent_counters as f64;
    println!(
        "Theorem 2: goal={accuracy_goal} octo_counters={} merge_counters={} ratio={ratio:.0}x (l={l})",
        octo.sent_counters, merge.sent_counters
    );
    assert!(
        ratio > l as f64 / 4.0,
        "expected roughly an l = {l} fold saving, measured {ratio:.0}x"
    );

    // Counting bytes rather than counters keeps the claim honest: an Octo
    // message carries indices as well as a value, a merged counter does not.
    let octo_bytes = octo.sent_counters * std::mem::size_of::<CmDelta>();
    let merge_bytes = merge.sent_counters * std::mem::size_of::<i32>();
    let byte_ratio = merge_bytes as f64 / octo_bytes as f64;
    println!("Theorem 2 (bytes): octo={octo_bytes} merge={merge_bytes} ratio={byte_ratio:.0}x");
    assert!(
        byte_ratio > l as f64 / 16.0,
        "byte-level saving {byte_ratio:.0}x is smaller than expected"
    );
}

// ---------------------------------------------------------------------------
// Figure 6 / Table 1: online accuracy against sketch-merge
// ---------------------------------------------------------------------------

#[test]
fn delta_promotion_beats_sketch_merge_on_online_accuracy() {
    // The paper's headline claim is *online* accuracy: the error a query sees
    // while the stream is still running. Sketch-merge is stale by up to one
    // merge period; delta promotion is stale by at most k'τ whatever the
    // arrival rate. Sweeping the baseline's communication budget from parity
    // up to 10x Octo's shows the gap is a property of the mechanism, not of a
    // budget chosen to flatter it.
    let d = 3usize;
    let l = 1_024usize;
    let workers = 4usize;
    let n = 60_000usize;
    let checkpoints = 8usize;
    let budget_multiples = [1usize, 10];

    let stream = zipf_u64(n, 4_096, 1.1, 11_007);
    let mut final_truth = FreqTruth::default();
    for key in &stream {
        final_truth.observe(*key as i64);
    }
    let watched: Vec<i64> = final_truth.top_k(32).into_iter().map(|(k, _)| k).collect();

    // Sizing pass: what does delta promotion spend over this stream?
    let mut probe = OctoCm::new(workers, d, l, Route::HashByKey);
    for (i, key) in stream.iter().enumerate() {
        probe.insert(i, &DataInput::U64(*key));
    }
    let mut merges: Vec<MergeCm> = budget_multiples
        .iter()
        .map(|mult| {
            let merges_per_worker = ((mult * probe.sent_counters) as f64 / (d * l * workers) as f64)
                .ceil()
                .max(1.0) as usize;
            // Divide by one more than the merge count so boundaries land
            // mid-stream: a period equal to a worker's whole share would make
            // it a coin flip whether that worker ever merges at all.
            let period = ((n / workers) / (merges_per_worker + 1)).max(1);
            MergeCm::new(workers, d, l, period, Route::HashByKey)
        })
        .collect();

    let mut truth = FreqTruth::default();
    let mut octo = OctoCm::new(workers, d, l, Route::HashByKey);
    let mut single_core = Cm::with_dimensions(d, l);
    let mut octo_error = 0.0f64;
    let mut ideal_error = 0.0f64;
    let mut merge_error = vec![0.0f64; merges.len()];
    let mut samples = 0usize;
    let stride = n / checkpoints;

    for (i, key) in stream.iter().enumerate() {
        let input = DataInput::U64(*key);
        truth.observe(*key as i64);
        octo.insert(i, &input);
        single_core.insert(&input);
        for merge in merges.iter_mut() {
            merge.insert(i, &input);
        }

        // Query at points that deliberately do not line up with a merge
        // boundary: an online query arrives whenever it arrives.
        if i > 0 && i % stride == stride / 3 {
            for watched_key in &watched {
                let probe_input = DataInput::U64(*watched_key as u64);
                let exact = truth.get(*watched_key) as f64;
                octo_error += (octo.parent.estimate(&probe_input) as f64 - exact).abs();
                ideal_error += (single_core.estimate(&probe_input) as f64 - exact).abs();
                for (slot, merge) in merge_error.iter_mut().zip(merges.iter()) {
                    *slot += (merge.parent.estimate(&probe_input) as f64 - exact).abs();
                }
                samples += 1;
            }
        }
    }

    let octo_mae = octo_error / samples as f64;
    let ideal_mae = ideal_error / samples as f64;
    println!("Online accuracy over {samples} queries: ideal={ideal_mae:.1} octo={octo_mae:.1}");
    for ((mult, merge), error) in budget_multiples
        .iter()
        .zip(merges.iter())
        .zip(merge_error.iter())
    {
        let merge_mae = error / samples as f64;
        let spend = merge.sent_counters as f64 / octo.sent_counters as f64;
        println!(
            "  merge at {mult}x budget (period={}, actual spend {spend:.1}x): \
             mae={merge_mae:.1} = {:.1}x octo",
            merge.period,
            merge_mae / octo_mae
        );
        assert!(
            merge.sent_counters > octo.sent_counters,
            "the baseline must not be starved of budget for this comparison to mean anything"
        );
        // Empirical and deliberately loose: measured 14.0x at parity and 5.4x
        // when the baseline is handed ten times the budget.
        assert!(
            merge_mae > octo_mae * 2.0,
            "delta promotion should stay well ahead at {mult}x budget: \
             octo={octo_mae:.1} vs merge={merge_mae:.1}"
        );
    }

    let parity_mae = merge_error[0] / samples as f64;
    assert!(
        parity_mae > octo_mae * 8.0,
        "at equal communication budget the gap should be large: \
         octo={octo_mae:.1} vs merge={parity_mae:.1}"
    );
    assert!(
        octo_mae <= ideal_mae + TAU as f64,
        "octo online error {octo_mae:.1} drifted more than tau from ideal {ideal_mae:.1}"
    );
}

#[test]
fn sketch_merge_staleness_grows_with_the_merge_period_while_promotion_does_not() {
    // The mechanism behind the previous test, isolated: sketch-merge error is
    // proportional to its period, delta promotion error is capped at k'τ no
    // matter how much traffic arrives between queries. The deficit is averaged
    // over many query points - sampling only at the end of the stream would
    // land on a merge boundary whenever the period divides the per-worker
    // item count, and report a staleness of zero.
    let d = 3usize;
    let l = 1_024usize;
    let workers = 4usize;
    let n = 40_000usize;
    let query_points = 40usize;
    let stream = zipf_u64(n, 2_048, 1.1, 11_009);

    let mut whole_stream_truth = FreqTruth::default();
    for key in &stream {
        whole_stream_truth.observe(*key as i64);
    }
    let hottest = whole_stream_truth.top_k(1)[0].0;
    let probe = DataInput::U64(hottest as u64);

    let periods = [500usize, 1_000, 2_000, 4_000];
    let mut truth = FreqTruth::default();
    let mut octo = OctoCm::new(workers, d, l, Route::HashByKey);
    let mut merges: Vec<MergeCm> = periods
        .iter()
        .map(|p| MergeCm::new(workers, d, l, *p, Route::HashByKey))
        .collect();
    let mut octo_deficit = 0.0f64;
    let mut merge_deficit = vec![0.0f64; periods.len()];
    let mut samples = 0usize;
    let stride = n / query_points;

    for (i, key) in stream.iter().enumerate() {
        let input = DataInput::U64(*key);
        truth.observe(*key as i64);
        octo.insert(i, &input);
        for merge in merges.iter_mut() {
            merge.insert(i, &input);
        }
        if i > 0 && i % stride == stride / 3 {
            let exact = truth.get(hottest) as f64;
            octo_deficit += exact - octo.parent.estimate(&probe) as f64;
            for (slot, merge) in merge_deficit.iter_mut().zip(merges.iter()) {
                *slot += exact - merge.parent.estimate(&probe) as f64;
            }
            samples += 1;
        }
    }

    let octo_mean = octo_deficit / samples as f64;
    let merge_means: Vec<f64> = merge_deficit.iter().map(|d| d / samples as f64).collect();
    println!(
        "hottest key: octo mean deficit={octo_mean:.1}, merge by period {periods:?} = {:?}",
        merge_means
            .iter()
            .map(|m| format!("{m:.1}"))
            .collect::<Vec<_>>()
    );

    assert!(
        octo_mean <= TAU as f64,
        "octo mean deficit {octo_mean:.1} exceeded tau"
    );
    for (window, pair) in merge_means.windows(2).zip(periods.windows(2)) {
        assert!(
            window[1] > window[0],
            "doubling the merge period from {} to {} should increase staleness: \
             {:.1} -> {:.1}",
            pair[0],
            pair[1],
            window[0],
            window[1]
        );
    }
    assert!(
        merge_means[periods.len() - 1] > octo_mean * 3.0,
        "expected the widest merge period to be far staler than delta promotion"
    );
}

// ===========================================================================
// Online accuracy per sketch: delta promotion vs periodic sketch-merge
// ===========================================================================
//
// Correctness is settled by the conservation identities above. What these
// measure is the paper's actual claim: the error a query sees *while the
// stream is still running*. Every comparison hands sketch-merge at least as
// much communication as delta promotion spent, then queries both at points
// that do not line up with a merge boundary.

/// One row of the comparison: mean error at query time under each scheme.
struct OnlineComparison {
    ideal: f64,
    octo: f64,
    merge_at_parity: f64,
    merge_at_ten_x: f64,
    octo_counters: usize,
    parity_counters: usize,
    ten_x_counters: usize,
}

impl OnlineComparison {
    fn report(&self, label: &str) {
        println!(
            "{label:<10} ideal={:.4} octo={:.4} merge@{:.1}x={:.4} merge@{:.1}x={:.4} \
             (octo sent {} counters)",
            self.ideal,
            self.octo,
            self.parity_counters as f64 / self.octo_counters.max(1) as f64,
            self.merge_at_parity,
            self.ten_x_counters as f64 / self.octo_counters.max(1) as f64,
            self.merge_at_ten_x,
            self.octo_counters,
        );
    }

    /// Harness self-check, not evidence: `merge_periods` already rounds the
    /// baseline's spend up past its budget. This catches a sizing bug that
    /// would starve the baseline and make the comparison meaningless; the
    /// printed spend multiples are what say how well it was funded.
    fn assert_baselines_were_funded(&self) {
        assert!(
            self.parity_counters >= self.octo_counters,
            "parity baseline was starved: {} < {}",
            self.parity_counters,
            self.octo_counters
        );
        assert!(self.ten_x_counters > self.parity_counters);
    }
}

/// Merge periods that spend roughly `1x` and `10x` the given counter budget.
fn merge_periods(
    octo_counters: usize,
    sketch_counters: usize,
    per_worker: usize,
) -> (usize, usize) {
    let period_for = |budget: usize| {
        let merges = (budget as f64 / sketch_counters as f64).ceil().max(1.0) as usize;
        (per_worker / (merges + 1)).max(1)
    };
    (period_for(octo_counters), period_for(10 * octo_counters))
}

// --------------------------------------------------------------- Count sketch

struct MergeCs {
    children: Vec<Cs>,
    seen: Vec<usize>,
    parent: Cs,
    period: usize,
    rows: usize,
    cols: usize,
    sent_counters: usize,
}

impl MergeCs {
    fn new(workers: usize, rows: usize, cols: usize, period: usize) -> Self {
        Self {
            children: (0..workers)
                .map(|_| Cs::with_dimensions(rows, cols))
                .collect(),
            seen: vec![0; workers],
            parent: Cs::with_dimensions(rows, cols),
            period,
            rows,
            cols,
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = Route::HashByKey.worker(index, key, self.children.len());
        self.children[worker].insert(key);
        self.seen[worker] += 1;
        if self.seen[worker] % self.period == 0 {
            self.parent.merge(&self.children[worker]);
            self.children[worker] = Cs::with_dimensions(self.rows, self.cols);
            self.sent_counters += self.rows * self.cols;
        }
    }
}

#[test]
fn count_sketch_online_accuracy_beats_sketch_merge() {
    let (d, l, workers, n) = (5usize, 1_024usize, 4usize, 60_000usize);
    let stream = zipf_u64(n, 4_096, 1.1, 12_001);
    let mut truth = FreqTruth::default();
    for key in &stream {
        truth.observe(*key as i64);
    }
    let watched: Vec<i64> = truth.top_k(32).into_iter().map(|(k, _)| k).collect();

    // Sizing pass: what does delta promotion spend over this stream?
    let mut probe = OctoCs::new(workers, d, l, Route::HashByKey);
    for (i, key) in stream.iter().enumerate() {
        probe.insert(i, &DataInput::U64(*key));
    }
    let octo_counters = probe.sent_counters;
    let (parity_period, ten_x_period) = merge_periods(octo_counters, d * l, n / workers);

    let mut running = FreqTruth::default();
    let mut octo = OctoCs::new(workers, d, l, Route::HashByKey);
    let mut ideal = Cs::with_dimensions(d, l);
    let mut parity = MergeCs::new(workers, d, l, parity_period);
    let mut ten_x = MergeCs::new(workers, d, l, ten_x_period);
    let (mut e_ideal, mut e_octo, mut e_parity, mut e_ten) = (0.0, 0.0, 0.0, 0.0);
    let mut samples = 0usize;
    let stride = n / 8;

    for (i, key) in stream.iter().enumerate() {
        let input = DataInput::U64(*key);
        running.observe(*key as i64);
        octo.insert(i, &input);
        ideal.insert(&input);
        parity.insert(i, &input);
        ten_x.insert(i, &input);
        if i > 0 && i % stride == stride / 3 {
            for probe_key in &watched {
                let probe_input = DataInput::U64(*probe_key as u64);
                let exact = running.get(*probe_key) as f64;
                e_ideal += (ideal.estimate(&probe_input) - exact).abs();
                e_octo += (octo.parent.estimate(&probe_input) - exact).abs();
                e_parity += (parity.parent.estimate(&probe_input) - exact).abs();
                e_ten += (ten_x.parent.estimate(&probe_input) - exact).abs();
                samples += 1;
            }
        }
    }

    let scale = samples as f64;
    let comparison = OnlineComparison {
        ideal: e_ideal / scale,
        octo: e_octo / scale,
        merge_at_parity: e_parity / scale,
        merge_at_ten_x: e_ten / scale,
        octo_counters,
        parity_counters: parity.sent_counters,
        ten_x_counters: ten_x.sent_counters,
    };
    comparison.report("Count");
    comparison.assert_baselines_were_funded();
    assert!(comparison.octo < comparison.merge_at_parity / 2.0);
    assert!(comparison.octo < comparison.merge_at_ten_x);
}

// ---------------------------------------------------------------- HyperLogLog

struct OctoHll {
    children: Vec<HyperLogLog<Classic>>,
    parent: HyperLogLog<Classic>,
    sent_counters: usize,
}

impl OctoHll {
    fn new(workers: usize) -> Self {
        Self {
            children: (0..workers).map(|_| HyperLogLog::default()).collect(),
            parent: HyperLogLog::default(),
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = Route::HashByKey.worker(index, key, self.children.len());
        let mut promoted = Vec::new();
        self.children[worker].insert_emit_delta(key, &mut |d| promoted.push(d));
        self.sent_counters += promoted.len();
        for d in promoted {
            self.parent.apply_delta(d);
        }
    }
}

struct MergeHll {
    children: Vec<HyperLogLog<Classic>>,
    seen: Vec<usize>,
    parent: HyperLogLog<Classic>,
    period: usize,
    sent_counters: usize,
}

impl MergeHll {
    fn new(workers: usize, period: usize) -> Self {
        Self {
            children: (0..workers).map(|_| HyperLogLog::default()).collect(),
            seen: vec![0; workers],
            parent: HyperLogLog::default(),
            period,
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = Route::HashByKey.worker(index, key, self.children.len());
        self.children[worker].insert(key);
        self.seen[worker] += 1;
        if self.seen[worker] % self.period == 0 {
            // A max-merge does not clear the worker: its registers stay valid.
            self.parent.merge(&self.children[worker]);
            self.sent_counters += self.children[worker].registers_as_slice().len();
        }
    }
}

#[test]
fn hyperloglog_online_accuracy_beats_sketch_merge() {
    let (workers, n) = (4usize, 200_000usize);
    let stream: Vec<u64> = (0..n as u64).collect();
    let registers = HyperLogLog::<Classic>::default().registers_as_slice().len();

    let mut probe = OctoHll::new(workers);
    for (i, key) in stream.iter().enumerate() {
        probe.insert(i, &DataInput::U64(*key));
    }
    let octo_counters = probe.sent_counters;
    let (parity_period, ten_x_period) = merge_periods(octo_counters, registers, n / workers);

    let mut octo = OctoHll::new(workers);
    let mut ideal = HyperLogLog::<Classic>::default();
    let mut parity = MergeHll::new(workers, parity_period);
    let mut ten_x = MergeHll::new(workers, ten_x_period);
    let (mut e_ideal, mut e_octo, mut e_parity, mut e_ten) = (0.0, 0.0, 0.0, 0.0);
    let mut samples = 0usize;
    let stride = n / 8;

    for (i, key) in stream.iter().enumerate() {
        let input = DataInput::U64(*key);
        octo.insert(i, &input);
        ideal.insert(&input);
        parity.insert(i, &input);
        ten_x.insert(i, &input);
        if i > 0 && i % stride == stride / 3 {
            // Every key so far is distinct, so the exact cardinality is i + 1.
            let exact = (i + 1) as f64;
            let rel = |estimate: usize| (estimate as f64 - exact).abs() / exact;
            e_ideal += rel(ideal.estimate());
            e_octo += rel(octo.parent.estimate());
            e_parity += rel(parity.parent.estimate());
            e_ten += rel(ten_x.parent.estimate());
            samples += 1;
        }
    }

    let scale = samples as f64;
    let comparison = OnlineComparison {
        ideal: e_ideal / scale,
        octo: e_octo / scale,
        merge_at_parity: e_parity / scale,
        merge_at_ten_x: e_ten / scale,
        octo_counters,
        parity_counters: parity.sent_counters,
        ten_x_counters: ten_x.sent_counters,
    };
    comparison.report("HLL");
    comparison.assert_baselines_were_funded();
    // Max-register promotion loses nothing, so the parent is the ideal sketch
    // at every query point - not merely close to it.
    assert_eq!(
        comparison.octo, comparison.ideal,
        "delta promotion must leave HLL exactly ideal online"
    );
    assert!(comparison.octo < comparison.merge_at_parity);
    assert!(comparison.octo < comparison.merge_at_ten_x);
}

// -------------------------------------------------------------------- DDSketch

struct OctoDd {
    children: Vec<DdWorkerSketch>,
    parent: DDSketch,
    sent_counters: usize,
}

impl OctoDd {
    fn new(workers: usize, alpha: f64) -> Self {
        Self {
            children: (0..workers).map(|_| DdWorkerSketch::new(alpha)).collect(),
            parent: DDSketch::new(alpha),
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, value: f64, threshold: u32) {
        let key = DataInput::F64(value);
        let worker = Route::HashByKey.worker(index, &key, self.children.len());
        let mut promoted = Vec::new();
        self.children[worker].add_emit_delta(value, threshold, &mut |d| promoted.push(d));
        self.sent_counters += promoted.len();
        for d in promoted {
            self.parent.apply_delta(d);
        }
    }
}

struct MergeDd {
    children: Vec<DDSketch>,
    seen: Vec<usize>,
    parent: DDSketch,
    period: usize,
    alpha: f64,
    sent_counters: usize,
}

impl MergeDd {
    fn new(workers: usize, alpha: f64, period: usize) -> Self {
        Self {
            children: (0..workers).map(|_| DDSketch::new(alpha)).collect(),
            seen: vec![0; workers],
            parent: DDSketch::new(alpha),
            period,
            alpha,
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, value: f64) {
        let worker = Route::HashByKey.worker(index, &DataInput::F64(value), self.children.len());
        self.children[worker].add(&value);
        self.seen[worker] += 1;
        if self.seen[worker] % self.period == 0 {
            self.sent_counters += self.children[worker].store_counts().len();
            self.parent
                .merge(&self.children[worker])
                .expect("same alpha");
            self.children[worker] = DDSketch::new(self.alpha);
        }
    }
}

/// Runs the DDSketch comparison over one value stream at one threshold.
fn ddsketch_online_comparison(
    label: &str,
    values: &[f64],
    alpha: f64,
    threshold: u32,
) -> OnlineComparison {
    let workers = 4usize;
    let n = values.len();

    let mut sizing = DDSketch::new(alpha);
    for v in values {
        sizing.add(v);
    }
    let sketch_counters = sizing.store_counts().len();

    let mut probe = OctoDd::new(workers, alpha);
    for (i, v) in values.iter().enumerate() {
        probe.insert(i, *v, threshold);
    }
    let octo_counters = probe.sent_counters;
    let (parity_period, ten_x_period) = merge_periods(octo_counters, sketch_counters, n / workers);

    let mut octo = OctoDd::new(workers, alpha);
    let mut ideal = DDSketch::new(alpha);
    let mut parity = MergeDd::new(workers, alpha, parity_period);
    let mut ten_x = MergeDd::new(workers, alpha, ten_x_period);
    let mut seen: Vec<f64> = Vec::with_capacity(n);
    let (mut e_ideal, mut e_octo, mut e_parity, mut e_ten) = (0.0, 0.0, 0.0, 0.0);
    let mut samples = 0usize;
    let stride = n / 8;

    for (i, v) in values.iter().enumerate() {
        octo.insert(i, *v, threshold);
        ideal.add(v);
        parity.insert(i, *v);
        ten_x.insert(i, *v);
        seen.push(*v);
        if i > 0 && i % stride == stride / 3 {
            let mut sorted = seen.clone();
            sorted.sort_by(f64::total_cmp);
            for q in [0.1, 0.5, 0.9] {
                let exact = sorted[((sorted.len() - 1) as f64 * q) as usize];
                let rel = |sketch: &DDSketch| {
                    sketch
                        .get_value_at_quantile(q)
                        .map(|got| (got - exact).abs() / exact)
                        .unwrap_or(1.0)
                };
                e_ideal += rel(&ideal);
                e_octo += rel(&octo.parent);
                e_parity += rel(&parity.parent);
                e_ten += rel(&ten_x.parent);
                samples += 1;
            }
        }
    }

    let scale = samples as f64;
    let comparison = OnlineComparison {
        ideal: e_ideal / scale,
        octo: e_octo / scale,
        merge_at_parity: e_parity / scale,
        merge_at_ten_x: e_ten / scale,
        octo_counters,
        parity_counters: parity.sent_counters,
        ten_x_counters: ten_x.sent_counters,
    };
    comparison.report(label);
    comparison
}

#[test]
fn ddsketch_delta_promotion_trades_quantile_accuracy_for_messages() {
    // DDSketch is the one integrated sketch where delta promotion does not pay
    // off, and this pins the reason. A Count-Min counter that lags is still
    // counted, just low; a DDSketch bucket that never reaches the threshold is
    // absent from the parent entirely, and a quantile is exactly a statement
    // about where the mass sits. Sparse buckets are also the common case: a
    // logarithmic histogram over a skewed stream has many of them.
    let (n, alpha) = (200_000usize, 0.01f64);
    let values = common::exponential_f64(n, 0.05, 12_101)
        .into_iter()
        .map(|v| v.max(1e-3))
        .collect::<Vec<f64>>();

    let mut sweep: Vec<(u32, f64, usize)> = Vec::new();
    for threshold in [1u32, 2, 4, 8] {
        let result =
            ddsketch_online_comparison(&format!("DDS/tau={threshold}"), &values, alpha, threshold);
        sweep.push((threshold, result.octo, result.octo_counters));
    }

    // A threshold of 1 promotes every sample, so the parent is the ideal
    // sketch - at the cost of one message per sample.
    let (_, exact_error, exact_counters) = sweep[0];
    assert_eq!(exact_counters, n, "tau=1 must promote every sample");
    let ideal_error = ddsketch_online_comparison("DDS/ideal-ref", &values, alpha, 1).ideal;
    assert!(
        (exact_error - ideal_error).abs() < 1e-12,
        "tau=1 must leave the parent exactly ideal: {exact_error} vs {ideal_error}"
    );

    // Above that, every step up buys fewer messages and costs accuracy.
    for pair in sweep.windows(2) {
        let (lo_tau, lo_error, lo_counters) = pair[0];
        let (hi_tau, hi_error, hi_counters) = pair[1];
        assert!(
            hi_counters < lo_counters,
            "tau {hi_tau} should send fewer counters than {lo_tau}: {hi_counters} vs {lo_counters}"
        );
        assert!(
            hi_error >= lo_error,
            "tau {hi_tau} should not be more accurate than {lo_tau}: {hi_error} vs {lo_error}"
        );
    }
}

// --------------------------------------------------------------------- UnivMon

struct OctoUniv {
    plan: UnivMonOctoPlan,
    workers: Vec<UnivMonOctoWorker>,
    aggregator: UnivMonOctoAggregator,
    sent_counters: usize,
}

impl OctoUniv {
    fn new(workers: usize, heap: usize, rows: usize, cols: usize, layers: usize, tau: u32) -> Self {
        let threshold = OctoThreshold::new(tau);
        let plan = UnivMonOctoPlan::with_threshold(rows, cols, layers, threshold.clone());
        Self {
            workers: (0..workers).map(|id| plan.worker(id)).collect(),
            plan,
            aggregator: UnivMonOctoAggregator::with_threshold(heap, rows, cols, layers, threshold),
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = Route::HashByKey.worker(index, key, self.workers.len());
        let payload = self.plan.prepare(key);
        let mut promoted = Vec::new();
        self.workers[worker].process(&payload, &mut |d| promoted.push(d));
        self.sent_counters += promoted.len();
        for delta in promoted {
            self.aggregator.apply(delta);
        }
    }
}

struct MergeUniv {
    children: Vec<UnivMon>,
    seen: Vec<usize>,
    parent: UnivMon,
    period: usize,
    dims: (usize, usize, usize, usize),
    sent_counters: usize,
}

impl MergeUniv {
    fn new(
        workers: usize,
        heap: usize,
        rows: usize,
        cols: usize,
        layers: usize,
        period: usize,
    ) -> Self {
        Self {
            children: (0..workers)
                .map(|_| UnivMon::init_univmon(heap, rows, cols, layers))
                .collect(),
            seen: vec![0; workers],
            parent: UnivMon::init_univmon(heap, rows, cols, layers),
            period,
            dims: (heap, rows, cols, layers),
            sent_counters: 0,
        }
    }

    fn insert(&mut self, index: usize, key: &DataInput) {
        let worker = Route::HashByKey.worker(index, key, self.children.len());
        self.children[worker].insert(key, 1);
        self.seen[worker] += 1;
        if self.seen[worker] % self.period == 0 {
            let (heap, rows, cols, layers) = self.dims;
            self.parent.merge(&self.children[worker]);
            self.children[worker] = UnivMon::init_univmon(heap, rows, cols, layers);
            self.sent_counters += rows * cols * layers;
        }
    }
}

#[test]
fn univmon_online_accuracy_beats_sketch_merge() {
    // UnivMon's recursive estimator needs about log2(distinct) sampling layers
    // before the deepest sampled substream fits the heap. This stream carries
    // ~3.5k distinct keys, so 12 layers; too few and the sketch's own error
    // swamps anything a distribution scheme does to it.
    let (workers, heap, rows, cols, layers) = (4usize, 64usize, 5usize, 1_024usize, 12usize);
    let n = 60_000usize;
    let tau = COUNT_PROMASK;
    let stream = zipf_u64(n, 4_096, 1.1, 12_201);

    let mut probe = OctoUniv::new(workers, heap, rows, cols, layers, tau);
    for (i, key) in stream.iter().enumerate() {
        probe.insert(i, &DataInput::U64(*key));
    }
    let octo_counters = probe.sent_counters;
    let (parity_period, ten_x_period) =
        merge_periods(octo_counters, rows * cols * layers, n / workers);

    let mut truth = FreqTruth::default();
    let mut octo = OctoUniv::new(workers, heap, rows, cols, layers, tau);
    let mut ideal = UnivMon::init_univmon(heap, rows, cols, layers);
    let mut parity = MergeUniv::new(workers, heap, rows, cols, layers, parity_period);
    let mut ten_x = MergeUniv::new(workers, heap, rows, cols, layers, ten_x_period);

    let (mut e_ideal, mut e_octo, mut e_parity, mut e_ten) = (0.0, 0.0, 0.0, 0.0);
    let mut samples = 0usize;
    let stride = n / 8;

    for (i, key) in stream.iter().enumerate() {
        let input = DataInput::U64(*key);
        truth.observe(*key as i64);
        octo.insert(i, &input);
        ideal.insert(&input, 1);
        parity.insert(i, &input);
        ten_x.insert(i, &input);
        if i > 0 && i % stride == stride / 3 {
            // UnivMon's headline queries are g-sums over the whole stream:
            // distinct-count and Shannon entropy. Both are compared against the
            // single-core sketch rather than against exact truth, which is the
            // paper's Definition 1: what is being measured here is the gap a
            // distributed scheme opens up, not UnivMon's own approximation
            // error, which at these dimensions dwarfs it.
            let (ideal_card, ideal_entropy) = (ideal.calc_card(), ideal.calc_entropy());
            let gap_to_ideal = |sketch: &UnivMon| {
                let card = (sketch.calc_card() - ideal_card).abs() / ideal_card.abs().max(1.0);
                let entropy =
                    (sketch.calc_entropy() - ideal_entropy).abs() / ideal_entropy.abs().max(1e-9);
                (card + entropy) / 2.0
            };
            let exact_card = truth.distinct() as f64;
            let exact_entropy = truth.entropy(true);
            e_ideal += ((ideal_card - exact_card).abs() / exact_card
                + (ideal_entropy - exact_entropy).abs() / exact_entropy.abs())
                / 2.0;
            e_octo += gap_to_ideal(&octo.aggregator.sketch);
            e_parity += gap_to_ideal(&parity.parent);
            e_ten += gap_to_ideal(&ten_x.parent);
            samples += 1;
        }
    }

    let scale = samples as f64;
    let comparison = OnlineComparison {
        ideal: e_ideal / scale,
        octo: e_octo / scale,
        merge_at_parity: e_parity / scale,
        merge_at_ten_x: e_ten / scale,
        octo_counters,
        parity_counters: parity.sent_counters,
        ten_x_counters: ten_x.sent_counters,
    };
    // `ideal` here is UnivMon's own error against exact truth, for context;
    // the other three are gaps to that ideal, per Definition 1.
    comparison.report("UnivMon");
    comparison.assert_baselines_were_funded();
    assert!(
        comparison.octo < comparison.merge_at_parity / 2.0,
        "delta promotion should track the single-core answer at least twice as closely \
         at comparable budget: octo={:.4} merge={:.4}",
        comparison.octo,
        comparison.merge_at_parity
    );
    // Sketch-merge does close the gap - it just has to buy its way there. On
    // this workload it needs an order of magnitude more traffic before it
    // matches delta promotion, which is Theorem 2 showing up as a measurement.
    assert!(
        comparison.merge_at_ten_x < comparison.merge_at_parity,
        "more budget must help the baseline"
    );
    assert!(
        comparison.ten_x_counters as f64 / comparison.octo_counters as f64 > 10.0,
        "the catch-up budget should be an order of magnitude larger"
    );
}

#[test]
fn univmon_heavy_hitter_recall_beats_sketch_merge() {
    let (workers, heap, rows, cols, layers) = (4usize, 64usize, 5usize, 1_024usize, 12usize);
    let n = 60_000usize;
    let tau = COUNT_PROMASK;
    let stream = zipf_u64(n, 4_096, 1.1, 12_203);

    let mut truth = FreqTruth::default();
    for key in &stream {
        truth.observe(*key as i64);
    }
    let hottest: Vec<i64> = truth.top_k(16).into_iter().map(|(k, _)| k).collect();

    let mut probe = OctoUniv::new(workers, heap, rows, cols, layers, tau);
    for (i, key) in stream.iter().enumerate() {
        probe.insert(i, &DataInput::U64(*key));
    }
    let (parity_period, _) = merge_periods(probe.sent_counters, rows * cols * layers, n / workers);

    let mut octo = OctoUniv::new(workers, heap, rows, cols, layers, tau);
    let mut parity = MergeUniv::new(workers, heap, rows, cols, layers, parity_period);
    let (mut octo_recall, mut merge_recall) = (0.0, 0.0);
    let mut samples = 0usize;
    let stride = n / 8;

    for (i, key) in stream.iter().enumerate() {
        let input = DataInput::U64(*key);
        octo.insert(i, &input);
        parity.insert(i, &input);
        if i > 0 && i % stride == stride / 3 {
            let found = |sketch: &UnivMon| {
                hottest
                    .iter()
                    .filter(|k| {
                        sketch.hh_layers[0]
                            .find(&DataInput::U64(**k as u64))
                            .is_some()
                    })
                    .count() as f64
                    / hottest.len() as f64
            };
            octo_recall += found(&octo.aggregator.sketch);
            merge_recall += found(&parity.parent);
            samples += 1;
        }
    }

    let (octo_recall, merge_recall) = (octo_recall / samples as f64, merge_recall / samples as f64);
    println!(
        "UnivMon top-16 recall at layer 0: octo={octo_recall:.3} merge={merge_recall:.3} \
         (merge spent {}x octo's counters)",
        parity.sent_counters / probe.sent_counters.max(1)
    );
    assert!(
        octo_recall >= merge_recall,
        "the aggregator's heap should not trail sketch-merge: {octo_recall:.3} vs {merge_recall:.3}"
    );
    assert!(
        octo_recall > 0.9,
        "heavy hitters should be found nearly always, got {octo_recall:.3}"
    );
}

// ---------------------------------------------------------------------------
// Why UnivMon scales its threshold per layer
// ---------------------------------------------------------------------------

/// Drives a UnivMon pyramid through the delta path with a caller-chosen
/// per-layer threshold, so a flat threshold can be compared against the scaled
/// one the shipped worker uses.
///
/// Deliberately a replica rather than `UnivMonOctoWorker`: the shipped worker
/// applies the scaled rule unconditionally, so the flat counterfactual cannot
/// be run through it. Layer selection, the per-layer seeds and the fidelity
/// choice mirror the shipped code; the weight bookkeeping does not - this sets
/// the exact total on every insert where the aggregator sums lagging per-worker
/// reports - so the scaled arm here is a slightly better case than the pipeline
/// delivers. `run_octo_univmon_reaches_the_deepest_layer` checks the structural
/// half against the real pipeline.
fn univmon_through_deltas(
    stream: &[u64],
    heap: usize,
    rows: usize,
    cols: usize,
    layers: usize,
    layer_threshold: impl Fn(usize) -> u32,
) -> UnivMon {
    let mut workers: Vec<L2hhWorkerSketch> = (0..layers)
        .map(|layer| L2hhWorkerSketch::new(rows, cols, layer))
        .collect();
    let mut parent = UnivMon::init_univmon(heap, rows, cols, layers);
    let mut weight = 0u64;

    for key in stream {
        let input = DataInput::U64(*key);
        weight += 1;
        let bottom = bottom_layer_for_hash(hash64_seeded(BOTTOM_LAYER_FINDER, &input), layers);
        let owned = input_to_owned(&input);
        for (layer, worker) in workers.iter_mut().enumerate().take(bottom + 1) {
            let hashed = hash128_seeded(layer, &input);
            let mut promoted = Vec::new();
            worker
                .insert_hash_emit_delta(hashed, layer_threshold(layer), &mut |d| promoted.push(d));
            for delta in promoted {
                parent.apply_layered_delta(
                    &LayeredCountDelta {
                        layer: layer as u32,
                        key: owned.clone(),
                        delta,
                        worker_id: 0,
                        weight_total: weight,
                    },
                    if layer_threshold(layer) <= 1 {
                        UnivMonDeltaFidelity::EveryInsert
                    } else {
                        UnivMonDeltaFidelity::PromotedOnly
                    },
                );
            }
        }
        parent.set_total_weight(weight as usize);
    }
    parent
}

#[test]
fn a_flat_threshold_starves_the_deep_univmon_layers() {
    // A UnivMon layer only receives the keys that survive L coin flips, so
    // layer L carries about n / 2^L of the stream. One threshold across the
    // whole pyramid is sized for layer 0 and leaves the deep layers - the ones
    // the recursive estimator leans on for cardinality - with nothing. This is
    // the measurement `univmon_layer_threshold` exists for.
    let (heap, rows, cols, layers) = (64usize, 5usize, 1_024usize, 12usize);
    let base = 31u32;
    let stream = zipf_u64(60_000, 4_096, 1.1, 12_201);

    let mut ideal = UnivMon::init_univmon(heap, rows, cols, layers);
    for key in &stream {
        ideal.insert(&DataInput::U64(*key), 1);
    }

    let flat = univmon_through_deltas(&stream, heap, rows, cols, layers, |_| base);
    let scaled = univmon_through_deltas(&stream, heap, rows, cols, layers, |layer| {
        univmon_layer_threshold(base, layer)
    });

    let nonzero = |sketch: &UnivMon, layer: usize| -> usize {
        let L2HH::COUNT(inner) = &sketch.l2_sketch_layers[layer];
        (0..inner.rows())
            .flat_map(|r| (0..inner.cols()).map(move |c| (r, c)))
            .filter(|(r, c)| inner.as_storage().query_one_counter(*r, *c) != 0)
            .count()
    };
    let deepest = layers - 1;
    println!(
        "UnivMon card: ideal={:.0} flat-tau={:.0} scaled-tau={:.0} | deepest layer non-zero cells: \
         ideal={} flat={} scaled={}",
        ideal.calc_card(),
        flat.calc_card(),
        scaled.calc_card(),
        nonzero(&ideal, deepest),
        nonzero(&flat, deepest),
        nonzero(&scaled, deepest),
    );

    // The deep layers are what a flat threshold empties.
    assert!(
        nonzero(&ideal, deepest) > 0,
        "the ideal sketch reaches this layer"
    );
    assert_eq!(
        nonzero(&flat, deepest),
        0,
        "a flat threshold leaves the deepest layer empty"
    );
    assert_eq!(
        nonzero(&scaled, deepest),
        nonzero(&ideal, deepest),
        "scaling the threshold per layer keeps the deep layers exact"
    );

    // And that is what wrecks the estimate the deep layers feed.
    let relative = |sketch: &UnivMon| {
        (sketch.calc_card() - ideal.calc_card()).abs() / ideal.calc_card().abs().max(1.0)
    };
    assert!(
        relative(&flat) > 0.5,
        "a flat threshold should lose most of the cardinality, got {:.3}",
        relative(&flat)
    );
    assert!(
        relative(&scaled) < 0.1,
        "the scaled threshold should track it, got {:.3}",
        relative(&scaled)
    );
}

// ===========================================================================
// Keyed-bucket families: CocoSketch and Elastic sketch (§4.4, appendix C)
// ===========================================================================
//
// Neither family is covered by a theorem, so the assertions here are of two
// kinds: exact conservation identities that hold under every interleaving, and
// the paper's measured comparison against the sketch-merge baseline - relative
// error (figures 21a and 21c) and F1 at the heavy-hitter threshold (figures
// 20a and 20c).
//
// Coco's aggregator elects from an unseeded RNG, so a promoted batch of τ takes
// a bucket with probability τ/val rather than 1/val: mass is conserved exactly,
// which key holds it is not. Its tests assert the mass identities exactly and
// everything else over enough runs to state a band in standard errors. Elastic
// has no randomness and is asserted exactly throughout.

mod heavy_hitters {
    use super::*;
    use asap_sketchlib::CmWorkerSketch;
    use asap_sketchlib::sketches::elastic::LAMBDA;
    use asap_sketchlib::{
        CANONICAL_HASH_SEED, COCO_PROMASK, Coco, CocoDelta, CocoOctoAggregator, CocoOctoPlan,
        CocoOctoWorker, ELASTIC_PROMASK, Elastic, ElasticDelta, ElasticOctoAggregator,
        ElasticOctoPlan, ElasticOctoWorker, flow_key_string,
    };
    use std::collections::{HashMap, HashSet};

    /// τ for both families.
    const COCO_TAU: u64 = COCO_PROMASK as u64;
    const ELASTIC_TAU: u32 = ELASTIC_PROMASK;

    /// The paper's heavy-hitter threshold: a flow is heavy when its count is
    /// *strictly* above `ALPHA * N`.
    const HH_ALPHA: f64 = 0.0002;

    fn flow_stream(n: usize, domain: usize, seed: u64) -> Vec<u64> {
        zipf_u64(n, domain, 1.1, seed)
    }

    /// The rendering `CocoOctoPlan` and `ElasticOctoPlan` apply to every input,
    /// so a test queries the parent with exactly the key the plan shipped.
    fn key_of(raw: u64) -> String {
        flow_key_string(&DataInput::U64(raw))
    }

    fn truth_of(stream: &[u64]) -> HashMap<String, u64> {
        let mut truth: HashMap<String, u64> = HashMap::new();
        for raw in stream {
            *truth.entry(key_of(*raw)).or_insert(0) += 1;
        }
        truth
    }

    // -- mass accessors ------------------------------------------------------

    /// Every `Coco::insert` adds its whole weight to exactly one bucket, so the
    /// table's total is the mass it has absorbed.
    fn coco_mass(sketch: &Coco) -> u64 {
        (0..sketch.d)
            .flat_map(|i| (0..sketch.w).map(move |j| (i, j)))
            .map(|(i, j)| sketch.table[i][j].val)
            .sum()
    }

    /// Mass resident in the heavy part: one positive vote per absorbed packet.
    fn elastic_heavy_mass(sketch: &Elastic) -> i64 {
        sketch.heavy.iter().map(|b| b.vote_pos as i64).sum()
    }

    /// Mass in one light-layer row. An Elastic spill adds its weight to every
    /// row, so each row independently accounts for the light half.
    fn elastic_light_row(sketch: &Elastic, row: usize) -> i64 {
        (0..sketch.light.cols())
            .map(|col| sketch.light.as_storage().query_one_counter(row, col) as i64)
            .sum()
    }

    fn worker_light_row(worker: &ElasticOctoWorker, row: usize) -> i64 {
        let light = worker.sketch().light();
        (0..light.cols())
            .map(|col| light.residual()[row * light.cols() + col] as i64)
            .sum()
    }

    // -- replay harnesses ----------------------------------------------------

    /// OctoSketch over CocoSketch, driven through the shipped plan, worker and
    /// aggregator rather than a re-implementation of them.
    struct OctoCoco {
        children: Vec<CocoOctoWorker>,
        parent: CocoOctoAggregator,
        route: Route,
        sent_counters: usize,
    }

    impl OctoCoco {
        fn new(workers: usize, w: usize, d: usize, route: Route) -> Self {
            let plan = CocoOctoPlan::new(w, d);
            Self {
                children: (0..workers).map(|id| plan.worker(id)).collect(),
                parent: plan.aggregator(),
                route,
                sent_counters: 0,
            }
        }

        fn insert(&mut self, index: usize, raw: u64) {
            let input = DataInput::U64(raw);
            let worker = self.route.worker(index, &input, self.children.len());
            let payload = key_of(raw);
            let mut promoted = Vec::new();
            self.children[worker].process(&payload, &mut |d: CocoDelta| promoted.push(d));
            self.sent_counters += promoted.len();
            for delta in promoted {
                self.parent.apply(delta);
            }
        }

        /// Counter mass the workers still hold back, one byte per bucket.
        fn residual(&self) -> u64 {
            self.children
                .iter()
                .map(|c| c.sketch().residual().iter().map(|v| *v as u64).sum::<u64>())
                .sum()
        }

        fn flush(&mut self) -> Vec<CocoDelta> {
            let mut shipped = Vec::new();
            for child in self.children.iter_mut() {
                child.flush(&mut |d: CocoDelta| shipped.push(d));
            }
            for delta in shipped.iter().cloned() {
                self.sent_counters += 1;
                self.parent.apply(delta);
            }
            shipped
        }

        fn sketch(&self) -> &Coco {
            &self.parent.sketch
        }
    }

    /// Sketch-merge over CocoSketch: each worker keeps a full table and replays
    /// every occupied bucket into the parent every `period` items, then starts
    /// over. `Coco::merge` is the paper's baseline "merge the whole sketch".
    struct MergeCoco {
        children: Vec<Coco>,
        seen: Vec<usize>,
        parent: Coco,
        period: usize,
        w: usize,
        d: usize,
        route: Route,
        sent_counters: usize,
    }

    impl MergeCoco {
        fn new(workers: usize, w: usize, d: usize, period: usize, route: Route) -> Self {
            Self {
                children: (0..workers).map(|_| Coco::init_with_size(w, d)).collect(),
                seen: vec![0; workers],
                parent: Coco::init_with_size(w, d),
                period,
                w,
                d,
                route,
                sent_counters: 0,
            }
        }

        fn insert(&mut self, index: usize, raw: u64) {
            let input = DataInput::U64(raw);
            let worker = self.route.worker(index, &input, self.children.len());
            self.children[worker].insert(&key_of(raw), 1);
            self.seen[worker] += 1;
            if self.seen[worker] % self.period == 0 {
                let full = std::mem::replace(
                    &mut self.children[worker],
                    Coco::init_with_size(self.w, self.d),
                );
                self.parent.merge(&full);
                self.sent_counters += self.w * self.d;
            }
        }
    }

    /// OctoSketch over the Elastic sketch, through the shipped plan/worker/
    /// aggregator. Fully deterministic: neither half draws a random number.
    struct OctoElastic {
        children: Vec<ElasticOctoWorker>,
        parent: ElasticOctoAggregator,
        route: Route,
        sent_counters: usize,
    }

    impl OctoElastic {
        fn new(workers: usize, buckets: i32, rows: usize, cols: usize, route: Route) -> Self {
            let plan = ElasticOctoPlan::new(buckets, rows, cols);
            Self {
                children: (0..workers).map(|id| plan.worker(id)).collect(),
                parent: plan.aggregator(),
                route,
                sent_counters: 0,
            }
        }

        fn insert(&mut self, index: usize, raw: u64) {
            let input = DataInput::U64(raw);
            let worker = self.route.worker(index, &input, self.children.len());
            let payload = key_of(raw);
            let mut promoted = Vec::new();
            self.children[worker].process(&payload, &mut |d: ElasticDelta| promoted.push(d));
            self.sent_counters += promoted.len();
            for delta in promoted {
                self.parent.apply(delta);
            }
        }

        fn flush(&mut self) -> Vec<ElasticDelta> {
            let mut shipped = Vec::new();
            for child in self.children.iter_mut() {
                child.flush(&mut |d: ElasticDelta| shipped.push(d));
            }
            for delta in shipped.iter().cloned() {
                self.sent_counters += 1;
                self.parent.apply(delta);
            }
            shipped
        }

        fn sketch(&self) -> &Elastic {
            &self.parent.sketch
        }
    }

    /// Sketch-merge over the Elastic sketch, using the paper's Sum merging.
    struct MergeElastic {
        children: Vec<Elastic>,
        seen: Vec<usize>,
        parent: Elastic,
        period: usize,
        buckets: i32,
        rows: usize,
        cols: usize,
        route: Route,
        sent_counters: usize,
    }

    impl MergeElastic {
        fn new(
            workers: usize,
            buckets: i32,
            rows: usize,
            cols: usize,
            period: usize,
            route: Route,
        ) -> Self {
            Self {
                children: (0..workers)
                    .map(|_| Elastic::init_with_dimensions(buckets, rows, cols))
                    .collect(),
                seen: vec![0; workers],
                parent: Elastic::init_with_dimensions(buckets, rows, cols),
                period,
                buckets,
                rows,
                cols,
                route,
                sent_counters: 0,
            }
        }

        fn insert(&mut self, index: usize, raw: u64) {
            let input = DataInput::U64(raw);
            let worker = self.route.worker(index, &input, self.children.len());
            self.children[worker].insert(key_of(raw));
            self.seen[worker] += 1;
            if self.seen[worker] % self.period == 0 {
                let full = std::mem::replace(
                    &mut self.children[worker],
                    Elastic::init_with_dimensions(self.buckets, self.rows, self.cols),
                );
                self.parent.merge(&full);
                self.sent_counters += self.buckets as usize + self.rows * self.cols;
            }
        }
    }

    // -- heavy-hitter scoring -------------------------------------------------

    /// `<flow, estimate>` for every flow the sketch currently records above
    /// `threshold`.
    fn coco_reported(sketch: &Coco, threshold: u64) -> HashMap<String, u64> {
        sketch
            .recorded_flows()
            .filter(|(_, size)| *size > threshold)
            .map(|(key, size)| (key.to_string(), size))
            .collect()
    }

    /// The Elastic counterpart: `query_all` reports only heavy-part residents,
    /// each estimated as its bucket plus the light layer.
    fn elastic_reported(sketch: &Elastic, threshold: u64) -> HashMap<String, u64> {
        sketch
            .heavy
            .iter()
            .filter(|b| b.vote_pos > 0)
            .map(|b| {
                (
                    b.flow_id.clone(),
                    sketch.query(b.flow_id.clone()).max(0) as u64,
                )
            })
            .filter(|(_, size)| *size > threshold)
            .collect()
    }

    /// Recall and precision over the heavy-hitter sets, and relative error
    /// averaged over the true positives alone rather than the whole query set.
    struct HhScore {
        recall: f64,
        precision: f64,
        are: f64,
    }

    impl HhScore {
        fn f1(&self) -> f64 {
            if self.recall + self.precision == 0.0 {
                0.0
            } else {
                2.0 * self.recall * self.precision / (self.recall + self.precision)
            }
        }
    }

    fn score(
        reported: &HashMap<String, u64>,
        truth: &HashMap<String, u64>,
        threshold: u64,
    ) -> HhScore {
        let real: HashSet<&String> = truth
            .iter()
            .filter(|(_, size)| **size > threshold)
            .map(|(key, _)| key)
            .collect();
        let hits: Vec<&String> = reported.keys().filter(|k| real.contains(*k)).collect();
        let both = hits.len() as f64;
        let are = hits
            .iter()
            .map(|k| {
                let exact = truth[*k] as f64;
                (reported[*k] as f64 - exact).abs() / exact
            })
            .sum::<f64>()
            / both.max(1.0);
        HhScore {
            recall: both / real.len().max(1) as f64,
            precision: both / reported.len().max(1) as f64,
            are,
        }
    }

    // -----------------------------------------------------------------------
    // CocoSketch delta protocol
    // -----------------------------------------------------------------------

    /// A promotion fires the moment a bucket counter *reaches* τ and the
    /// counter is zeroed, so every insert-driven message carries exactly τ.
    /// And the key it names is always a key the stream actually contained: the
    /// worker ships either the arrival or the bucket's incumbent, never a
    /// synthesised one.
    #[test]
    fn coco_deltas_carry_exactly_one_promotion_window_and_a_key_the_stream_used() {
        let stream = flow_stream(40_000, 2_048, 21_001);
        let seen: HashSet<String> = stream.iter().map(|raw| key_of(*raw)).collect();

        let mut worker = CocoOctoPlan::new(512, 2).worker(0);
        let mut promoted = Vec::new();
        for raw in &stream {
            worker.process(&key_of(*raw), &mut |d: CocoDelta| promoted.push(d));
        }

        assert!(
            !promoted.is_empty(),
            "a 40k stream must cross τ = {COCO_TAU} somewhere"
        );
        for delta in &promoted {
            assert_eq!(
                delta.value, COCO_TAU,
                "CocoSketch promotes exactly one window of τ, got {}",
                delta.value
            );
            assert!(
                seen.contains(&delta.key),
                "delta named {} which never appeared in the stream",
                delta.key
            );
        }
    }

    /// The counter a worker promotes is cleared, and every increment - promoted
    /// or held back - lands in exactly one bucket. So the parent's table plus
    /// the workers' one-byte residuals reconstruct the stream exactly, and no
    /// residual can have reached τ without being shipped.
    #[test]
    fn coco_promotion_conserves_the_stream_mass_exactly() {
        let stream = flow_stream(60_000, 2_048, 21_002);
        let mut octo = OctoCoco::new(4, 512, 2, Route::HashByKey);
        for (i, raw) in stream.iter().enumerate() {
            octo.insert(i, *raw);
        }

        let promoted = coco_mass(octo.sketch());
        let residual = octo.residual();
        assert_eq!(
            promoted + residual,
            stream.len() as u64,
            "promoted {promoted} plus residual {residual} lost mass against {}",
            stream.len()
        );
        assert!(
            residual > 0,
            "a skewed 60k stream must leave partial counters behind"
        );
        for child in &octo.children {
            for (bucket, held) in child.sketch().residual().iter().enumerate() {
                assert!(
                    (*held as u64) < COCO_TAU,
                    "bucket {bucket} holds {held}, at or past the promotion window"
                );
            }
        }
        // §7.2: "OctoSketch tends to underestimate compared to the ideal
        // accuracy since there is some information left in each worker."
        assert!(
            promoted < stream.len() as u64,
            "an unflushed parent must trail the stream, not match it"
        );
    }

    /// `flush` ships every bucket that still holds a partial count, and only
    /// those: each flush message carries between 1 and τ - 1, and afterwards
    /// the parent holds the whole stream and the workers hold nothing.
    #[test]
    fn coco_flush_hands_over_every_residual_bucket() {
        let stream = flow_stream(60_000, 2_048, 21_003);
        let mut octo = OctoCoco::new(4, 512, 2, Route::HashByKey);
        for (i, raw) in stream.iter().enumerate() {
            octo.insert(i, *raw);
        }
        let expected_residual = octo.residual();
        let shipped = octo.flush();

        assert_eq!(
            shipped.iter().map(|d| d.value).sum::<u64>(),
            expected_residual,
            "flush must ship exactly what the workers were holding"
        );
        for delta in &shipped {
            assert!(
                delta.value > 0 && delta.value < COCO_TAU,
                "a flushed bucket carries a partial count, got {}",
                delta.value
            );
        }
        assert_eq!(octo.residual(), 0, "flush must leave the workers empty");
        assert_eq!(
            coco_mass(octo.sketch()),
            stream.len() as u64,
            "a flushed parent holds the whole stream"
        );
    }

    /// The branch that makes CocoSketch's payload a key rather than a cell
    /// index: an arrival that loses the `v/val` election still pushed the
    /// bucket over τ, and what ships is the *incumbent*. Shipping the arrival
    /// instead would attribute the window to the wrong flow.
    #[test]
    fn coco_promotes_the_bucket_incumbent_when_the_arrival_loses_the_election() {
        // A 32-bucket table under 2k distinct flows keeps almost every arrival
        // in the losing branch, so incumbent promotions are the common case.
        let mut promotions = 0usize;
        let mut incumbent = 0usize;
        for seed in 0..4u64 {
            let stream = flow_stream(20_000, 2_048, 21_100 + seed);
            let mut worker = CocoOctoPlan::new(16, 2).worker(0);
            for raw in &stream {
                let arrival = key_of(*raw);
                worker.process(&arrival, &mut |d: CocoDelta| {
                    promotions += 1;
                    if d.key != arrival {
                        incumbent += 1;
                    }
                });
            }
        }

        assert!(promotions > 1_000, "expected a busy promotion schedule");
        // Measured 21-39% across these seeds; the assertion only has to rule
        // out a worker that always names the arriving key.
        let share = incumbent as f64 / promotions as f64;
        assert!(
            share > 0.05,
            "only {incumbent}/{promotions} promotions named the incumbent ({share:.3})"
        );
    }

    /// Sharding changes which worker sees a flow but not the accounting: the
    /// parent still ends up holding the whole stream, and CocoSketch's point
    /// query still partitions it, because an insert leaves a key in at most one
    /// bucket and every occupied bucket holds a key the stream contained.
    #[test]
    fn coco_sharded_workers_conserve_mass_and_partition_the_point_queries() {
        let stream = flow_stream(60_000, 2_048, 21_004);
        let distinct: HashSet<String> = stream.iter().map(|raw| key_of(*raw)).collect();

        for workers in [1usize, 2, 4, 8] {
            let mut octo = OctoCoco::new(workers, 512, 2, Route::HashByKey);
            for (i, raw) in stream.iter().enumerate() {
                octo.insert(i, *raw);
            }
            octo.flush();

            assert_eq!(
                coco_mass(octo.sketch()),
                stream.len() as u64,
                "{workers} workers lost mass"
            );
            let attributed: u64 = distinct
                .iter()
                .map(|key| octo.sketch().estimate_key(key))
                .sum();
            assert_eq!(
                attributed,
                stream.len() as u64,
                "{workers} workers: point queries must partition the stream"
            );
        }
    }

    /// CocoSketch §3.2 claims stochastic variance minimization "yields unbiased
    /// size estimation", and the OctoSketch aggregator only preserves that if
    /// it replays a promoted window as a *weighted* insert, electing at
    /// `rng() % C < value` rather than `rng() % C == 0`. An aggregator that
    /// dropped the weight, or elected at the single-packet rate, would show up
    /// as bias here.
    ///
    /// Unbiasedness is a statement about the mean, so the band is stated in
    /// standard errors of the sample mean rather than as a fixed tolerance.
    #[test]
    fn coco_octo_point_estimates_stay_unbiased_across_independent_runs() {
        const TRIALS: usize = 120;
        let stream = flow_stream(20_000, 1_000, 21_005);
        let truth = truth_of(&stream);
        let mut ranked: Vec<(&String, &u64)> = truth.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let watched: Vec<String> = ranked.iter().take(32).map(|(k, _)| (*k).clone()).collect();
        let exact: u64 = ranked.iter().take(32).map(|(_, v)| **v).sum();

        // 128 buckets under 1000 flows: the watched flows are evicted and
        // re-elected constantly, which is where a biased replay would show.
        let mut totals = Vec::with_capacity(TRIALS);
        for _ in 0..TRIALS {
            let mut octo = OctoCoco::new(1, 64, 2, Route::HashByKey);
            for (i, raw) in stream.iter().enumerate() {
                octo.insert(i, *raw);
            }
            octo.flush();
            totals.push(
                watched
                    .iter()
                    .map(|key| octo.sketch().estimate_key(key))
                    .sum::<u64>() as f64,
            );
        }

        let mean = totals.iter().sum::<f64>() / TRIALS as f64;
        let variance =
            totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (TRIALS as f64 - 1.0);
        let standard_error = (variance / TRIALS as f64).sqrt();
        println!(
            "Coco octo top-32 mass: truth={exact} mean={mean:.1} se={standard_error:.1} \
             ({:.2} sigma)",
            (mean - exact as f64) / standard_error
        );

        // The band has to be narrow to mean anything: measured at 0.19% of the
        // watched mass per standard error, so 5 sigma is under 1%.
        assert!(
            standard_error * 5.0 < 0.02 * exact as f64,
            "the estimator is too noisy for this to assert anything: 5 se = {:.1} on {exact}",
            standard_error * 5.0
        );
        assert!(
            (mean - exact as f64).abs() < 5.0 * standard_error,
            "mean {mean:.1} is {:.2} standard errors from the exact {exact}",
            (mean - exact as f64) / standard_error
        );
    }

    // -----------------------------------------------------------------------
    // Elastic sketch delta protocol
    // -----------------------------------------------------------------------

    /// Appendix C splits the Elastic promotion in two: the heavy part ships
    /// `<key, votes>` the way CocoSketch does, and "the light part's insertion
    /// logic is the same as that of the OctoSketch-optimized Count-Min sketch",
    /// so it ships an unkeyed cell delta. This crate adds the eviction flag to
    /// the heavy message and a third, keyed message for an evicted resident,
    /// which is the only counter that ever moved by more than one at a time.
    /// With it gone from the light half, a light cell promotes at exactly τ,
    /// like any other Count-Min worker's.
    #[test]
    fn elastic_deltas_split_into_keyed_heavy_votes_keyed_evictions_and_unkeyed_light_cells() {
        let stream = flow_stream(40_000, 2_048, 22_001);
        let seen: HashSet<String> = stream.iter().map(|raw| key_of(*raw)).collect();
        let (rows, cols) = (3usize, 1_024usize);

        let mut worker = ElasticOctoPlan::new(64, rows, cols).worker(0);
        let mut promoted = Vec::new();
        for raw in &stream {
            worker.process(&key_of(*raw), &mut |d: ElasticDelta| promoted.push(d));
        }

        let mut heavy = 0usize;
        let mut light = 0usize;
        let mut evicted = 0usize;
        let mut evicted_empty = 0usize;
        let mut flagged = 0usize;
        for delta in &promoted {
            match delta {
                ElasticDelta::Heavy {
                    key,
                    value,
                    eviction,
                } => {
                    heavy += 1;
                    flagged += usize::from(*eviction);
                    assert_eq!(*value, ELASTIC_TAU, "a heavy bucket promotes one window");
                    assert!(seen.contains(key), "heavy delta named an unseen flow {key}");
                }
                ElasticDelta::Evicted { key, votes } => {
                    evicted += 1;
                    evicted_empty += usize::from(*votes == 0);
                    assert!(
                        *votes < ELASTIC_TAU,
                        "an evicted resident carries less than one window, got {votes}"
                    );
                    assert!(
                        seen.contains(key),
                        "eviction delta named an unseen flow {key}"
                    );
                }
                ElasticDelta::Light(cell) => {
                    light += 1;
                    assert_eq!(
                        cell.value, ELASTIC_TAU,
                        "the light half moves one at a time, so it promotes exactly τ"
                    );
                    assert!((cell.row as usize) < rows, "row {} out of range", cell.row);
                    assert!((cell.col as usize) < cols, "col {} out of range", cell.col);
                }
            }
        }
        assert!(
            heavy > 0 && light > 0 && evicted > 0,
            "all three message kinds must appear"
        );
        assert!(
            flagged > 0,
            "a bucket taken over by eviction must ship its flag"
        );
        assert!(
            evicted_empty > 0,
            "a resident evicted right after a promotion carries zero votes and must still ship"
        );
    }

    /// Elastic conserves mass per key, and per light-layer row: a packet
    /// becomes one positive vote or one light increment on every row, an
    /// eviction moves the resident's whole vote from the heavy part into every
    /// row, and a promotion moves it to the parent unchanged. So for each row
    /// independently, parent heavy + parent light + worker heavy + worker light
    /// is the stream, exactly, at every worker count.
    #[test]
    fn elastic_promotion_conserves_the_stream_mass_row_by_row() {
        let stream = flow_stream(60_000, 2_048, 22_002);
        let (rows, cols) = (3usize, 2_048usize);

        for workers in [1usize, 2, 4, 8] {
            let mut octo = OctoElastic::new(workers, 256, rows, cols, Route::HashByKey);
            for (i, raw) in stream.iter().enumerate() {
                octo.insert(i, *raw);
            }

            let parent_heavy = elastic_heavy_mass(octo.sketch());
            let worker_light: Vec<i64> = (0..rows)
                .map(|r| octo.children.iter().map(|c| worker_light_row(c, r)).sum())
                .collect();
            let parent_light: Vec<i64> = (0..rows)
                .map(|r| elastic_light_row(octo.sketch(), r))
                .collect();

            // The votes the workers still hold are what `flush` is about to
            // ship, which is the only part not visible through an accessor.
            let held: i64 = octo
                .flush()
                .iter()
                .filter_map(|d| match d {
                    ElasticDelta::Heavy { value, .. } => Some(*value as i64),
                    ElasticDelta::Evicted { .. } | ElasticDelta::Light(_) => None,
                })
                .sum();

            for row in 0..rows {
                assert_eq!(
                    parent_heavy + parent_light[row] + worker_light[row] + held,
                    stream.len() as i64,
                    "{workers} workers, row {row}: mass leaked out of the pipeline"
                );
            }
            assert!(held > 0, "{workers} workers: some votes must still be held");
        }
    }

    /// `flush` empties both halves: every light counter goes out as a partial
    /// cell delta, every resident's remaining votes go out as a heavy message,
    /// and afterwards the parent alone accounts for the stream on every row.
    #[test]
    fn elastic_flush_empties_both_halves_of_every_worker() {
        let stream = flow_stream(60_000, 2_048, 22_003);
        let (rows, cols) = (3usize, 2_048usize);
        let mut octo = OctoElastic::new(4, 256, rows, cols, Route::HashByKey);
        for (i, raw) in stream.iter().enumerate() {
            octo.insert(i, *raw);
        }

        let shipped = octo.flush();
        for delta in &shipped {
            let value = match delta {
                ElasticDelta::Heavy { value, .. } => *value,
                ElasticDelta::Light(cell) => cell.value,
                ElasticDelta::Evicted { key, .. } => {
                    panic!("a flush evicts nobody, but handed over {key}")
                }
            };
            assert!(
                value > 0 && value < ELASTIC_TAU,
                "a flushed counter carries a partial count, got {value}"
            );
        }
        for child in &octo.children {
            assert!(
                child.sketch().light().residual().iter().all(|c| *c == 0),
                "the light layer must be empty after a flush"
            );
        }

        let heavy = elastic_heavy_mass(octo.sketch());
        for row in 0..rows {
            assert_eq!(
                heavy + elastic_light_row(octo.sketch(), row),
                stream.len() as i64,
                "row {row}: a flushed parent holds the whole stream"
            );
        }
    }

    /// An eviction hands the evicted resident to the light part under *its
    /// own* key, not under the arriving packet's. A worker cannot do that
    /// through an unkeyed cell
    /// delta, so the eviction travels as `ElasticDelta::Evicted` and the
    /// aggregator addresses the parent's light layer by the victim's key.
    /// The losing arrivals stay unkeyed and batched.
    #[test]
    fn elastic_eviction_hands_the_evicted_resident_over_under_its_own_key() {
        const RESIDENT_VOTES: usize = 3;
        let (rows, cols) = (1usize, 64usize);
        // τ above anything this workload reaches, so nothing is promoted away
        // and the losing arrivals stay visible in the worker's residual.
        let quiet = OctoThreshold::new(127);
        let plan = ElasticOctoPlan::with_threshold(1, rows, cols, quiet);
        let mut worker = plan.worker(0);
        let mut parent = plan.aggregator();

        let resident = "flow::resident".to_string();
        let challenger = "flow::challenger".to_string();
        let cell_of = |key: &str| {
            (CmWorkerSketch::hashes(rows, &DataInput::Str(key))[0] & 0xffff_ffff) as usize % cols
        };
        let (resident_cell, challenger_cell) = (cell_of(&resident), cell_of(&challenger));
        assert_ne!(
            resident_cell, challenger_cell,
            "the two keys must land on different light cells for this to measure anything"
        );

        let mut shipped = Vec::new();
        for _ in 0..RESIDENT_VOTES {
            worker.process(&resident, &mut |d| shipped.push(d));
        }
        // vote_neg reaches LAMBDA * vote_pos on this arrival, which evicts.
        for _ in 0..(LAMBDA as usize * RESIDENT_VOTES) {
            worker.process(&challenger, &mut |d| shipped.push(d));
        }
        assert_eq!(
            shipped,
            vec![ElasticDelta::Evicted {
                key: resident.clone(),
                votes: RESIDENT_VOTES as u32,
            }],
            "τ = 127 leaves the eviction as the only message the worker must send"
        );

        let residual = worker.sketch().light().residual();
        assert_eq!(
            residual[resident_cell], 0,
            "the evicted resident never touches the worker's light layer"
        );
        assert_eq!(
            residual[challenger_cell] as usize,
            LAMBDA as usize * RESIDENT_VOTES - 1,
            "the challenger spills every losing arrival but the one that evicted"
        );

        for delta in shipped {
            parent.apply(delta);
        }
        let light = &parent.sketch.light;
        assert_eq!(
            light.as_storage().query_one_counter(0, resident_cell),
            RESIDENT_VOTES as i32,
            "the parent's light layer takes the vote on the victim's own cell"
        );
        assert_eq!(
            light.as_storage().query_one_counter(0, challenger_cell),
            0,
            "and puts nothing on the arrival that displaced it"
        );
    }

    /// Elastic's light part is additive and never decremented, so it never
    /// underestimates the mass hashed into a cell. Under hash-by-key routing a
    /// flow meets exactly one worker, so every path by which its mass can reach
    /// the parent's light layer also flags its parent bucket: a worker bucket
    /// held by eviction ships `eviction: true` with its votes, and a worker
    /// eviction ships `ElasticDelta::Evicted` under the victim's key. Together
    /// those make a flushed parent one-sided: no flow, resident or evicted, may
    /// read back below its true size.
    ///
    /// Drop either mechanism and a heavy flow whose parent bucket never evicted
    /// anyone loses whatever a *worker* eviction had already put in the light
    /// layer.
    #[test]
    fn elastic_octo_never_underestimates_any_flow_after_a_flush() {
        let stream = flow_stream(60_000, 4_096, 22_004);
        let truth = truth_of(&stream);

        for workers in [1usize, 2, 4, 8] {
            let mut octo = OctoElastic::new(workers, 128, 3, 2_048, Route::HashByKey);
            for (i, raw) in stream.iter().enumerate() {
                octo.insert(i, *raw);
            }
            octo.flush();

            let mut evicted = 0usize;
            for (key, exact) in &truth {
                let estimate = octo.sketch().query(key.clone());
                assert!(
                    estimate >= *exact as i32,
                    "{workers} workers: {key} read back {estimate} against a true {exact}"
                );
                if !octo.sketch().heavy.iter().any(|b| b.flow_id == *key) {
                    evicted += 1;
                }
            }
            assert!(
                evicted > 0,
                "{workers} workers: a 128-bucket table must push flows into the light layer"
            );
        }
    }

    /// The property the flag exists to protect. A heavy flow that never lost a
    /// bucket contest and was never evicted - not at its worker, not at the
    /// parent - holds all of its mass in the parent's heavy part, so its
    /// estimate must be *exact*. Reading through to the light layer would
    /// charge it for Count-Min mass that belongs to other flows, which is what
    /// flagging every heavy message unconditionally does.
    ///
    /// A single light column puts every other flow's spill on this flow's own
    /// cell, so "exact" is a claim the arrangement can actually falsify.
    #[test]
    fn a_heavy_flow_that_never_spilled_is_estimated_exactly() {
        const BUCKETS: i32 = 4;
        const LONE_ARRIVALS: usize = 200;
        const CONTESTED_ARRIVALS: usize = 400;
        let (rows, cols) = (1usize, 1usize);

        let bucket_of = |key: &str| {
            hash64_seeded(CANONICAL_HASH_SEED, &DataInput::Str(key)) as usize % BUCKETS as usize
        };
        let in_bucket = |bucket: usize, want: usize| -> Vec<String> {
            (0..10_000usize)
                .map(|i| format!("flow::{i}"))
                .filter(|key| bucket_of(key) == bucket)
                .take(want)
                .collect()
        };
        let lone = in_bucket(0, 1).pop().expect("a key hashing to bucket 0");
        let contenders = in_bucket(1, 2);
        assert_eq!(contenders.len(), 2, "two keys hashing to bucket 1");

        let plan = ElasticOctoPlan::new(BUCKETS, rows, cols);
        let mut worker = plan.worker(0);
        let mut parent = plan.aggregator();
        let mut stream: Vec<String> = (0..LONE_ARRIVALS).map(|_| lone.clone()).collect();
        for i in 0..CONTESTED_ARRIVALS {
            stream.push(contenders[i % 2].clone());
        }
        for key in &stream {
            let mut promoted = Vec::new();
            worker.process(key, &mut |d: ElasticDelta| promoted.push(d));
            for delta in promoted {
                parent.apply(delta);
            }
        }
        let mut shipped = Vec::new();
        worker.flush(&mut |d: ElasticDelta| shipped.push(d));
        for delta in shipped {
            parent.apply(delta);
        }

        let noise = parent
            .sketch
            .light
            .estimate(&DataInput::String(lone.clone()));
        assert!(
            noise > 0,
            "the contested pair must have left mass on the lone flow's light cell"
        );
        let bucket = parent
            .sketch
            .heavy
            .iter()
            .find(|b| b.flow_id == lone)
            .expect("the lone flow must be resident at the parent");
        assert!(
            !bucket.eviction,
            "a flow no worker ever evicted must not be flagged"
        );
        assert_eq!(
            parent.sketch.query(lone.clone()),
            LONE_ARRIVALS as i32,
            "an unspilled heavy flow must read back exactly, not {noise} above"
        );
    }

    /// The other half of the same protocol. A flow that *did* acquire
    /// light-layer mass at a worker must read through to it at the parent, even
    /// though the parent's own bucket never evicted anyone. Here the message
    /// that carries the news is an eviction of *zero* votes - the worker had
    /// just promoted the counter, so the takeover found nothing left to hand
    /// over - which is why the worker ships it anyway.
    #[test]
    fn a_resident_that_spilled_at_a_worker_reads_through_to_the_light_layer() {
        // One more arrival than this and the challenger is evicted in turn.
        const SPILLS: usize = LAMBDA as usize - 2;
        let (rows, cols) = (1usize, 64usize);

        let cell_of = |key: &str| {
            (CmWorkerSketch::hashes(rows, &DataInput::Str(key))[0] & 0xffff_ffff) as usize % cols
        };
        let resident = "flow::resident".to_string();
        let challenger = (0..10_000usize)
            .map(|i| format!("flow::challenger::{i}"))
            .find(|key| cell_of(key) != cell_of(&resident))
            .expect("a challenger on a different light cell");

        let plan = ElasticOctoPlan::new(1, rows, cols);
        let mut worker = plan.worker(0);
        let mut parent = plan.aggregator();
        let mut shipped = Vec::new();
        let mut stream: Vec<String> = (0..ELASTIC_TAU).map(|_| resident.clone()).collect();
        stream.push(challenger.clone());
        stream.extend((0..SPILLS).map(|_| resident.clone()));
        for key in &stream {
            worker.process(key, &mut |d: ElasticDelta| shipped.push(d));
        }

        assert_eq!(
            shipped,
            vec![
                ElasticDelta::Heavy {
                    key: resident.clone(),
                    value: ELASTIC_TAU,
                    eviction: false,
                },
                ElasticDelta::Evicted {
                    key: resident.clone(),
                    votes: 0,
                },
            ],
            "a slot seated from vacant ships an unflagged counter, and the takeover \
             right after it hands the just-promoted resident over empty"
        );

        worker.flush(&mut |d: ElasticDelta| shipped.push(d));
        for delta in shipped {
            parent.apply(delta);
        }

        let exact = ELASTIC_TAU as i32 + SPILLS as i32;
        let bucket = parent
            .sketch
            .heavy
            .iter()
            .find(|b| b.flow_id == resident)
            .expect("the resident must still hold the parent's bucket");
        assert!(
            bucket.eviction,
            "a flow evicted at a worker must be flagged at the parent"
        );
        assert_eq!(bucket.vote_pos, ELASTIC_TAU as i32, "the promoted window");
        assert_eq!(
            parent.sketch.query(resident.clone()),
            exact,
            "the spilled arrivals must be read back through the light layer"
        );
    }

    // -----------------------------------------------------------------------
    // Runtime: the multi-threaded pipeline over both families
    // -----------------------------------------------------------------------

    #[cfg(feature = "octo-runtime")]
    mod runtime {
        use super::*;
        use asap_sketchlib::{OctoConfig, run_octo};

        fn config(num_workers: usize) -> OctoConfig {
            OctoConfig {
                num_workers,
                pin_cores: false,
                queue_capacity: 4096,
                ..OctoConfig::default()
            }
        }

        /// Neither Coco's election nor Elastic's contest is commutative, so the
        /// parent these produce is *not* a function of the partition alone -
        /// unlike every other family in this file, a runtime run cannot be
        /// compared to a replay. What survives every interleaving is the mass:
        /// a promoted window is cleared exactly once and absorbed exactly once,
        /// whichever bucket the parent chooses for it. `run_octo` flushes at the
        /// end of the stream, so the parent holds all of it.
        #[test]
        fn run_octo_coco_conserves_the_stream_mass_at_every_worker_count() {
            let stream = flow_stream(60_000, 2_048, 23_001);
            let inputs = inputs_from(&stream);
            let distinct: HashSet<String> = stream.iter().map(|raw| key_of(*raw)).collect();

            for workers in [1usize, 2, 3, 4, 8] {
                let parent = run_octo(&inputs, &config(workers), CocoOctoPlan::new(512, 2), || {
                    CocoOctoAggregator::new(512, 2)
                })
                .parent
                .sketch;

                assert_eq!(
                    coco_mass(&parent),
                    stream.len() as u64,
                    "{workers} workers: the parent table lost mass"
                );
                let attributed: u64 = distinct.iter().map(|k| parent.estimate_key(k)).sum();
                assert_eq!(
                    attributed,
                    stream.len() as u64,
                    "{workers} workers: point queries must partition the stream"
                );
            }
        }

        /// The Elastic identity, run through real threads: for every light row
        /// the parent's heavy votes plus that row account for the whole stream,
        /// and the sketch is still one-sided for every flow in it.
        #[test]
        fn run_octo_elastic_conserves_the_stream_mass_whatever_the_interleaving() {
            let stream = flow_stream(60_000, 4_096, 23_002);
            let inputs = inputs_from(&stream);
            let truth = truth_of(&stream);
            let (rows, cols) = (3usize, 2_048usize);

            for workers in [1usize, 2, 3, 4, 8] {
                let parent = run_octo(
                    &inputs,
                    &config(workers),
                    ElasticOctoPlan::new(128, rows, cols),
                    || ElasticOctoAggregator::new(128, rows, cols),
                )
                .parent
                .sketch;

                let heavy = elastic_heavy_mass(&parent);
                for row in 0..rows {
                    assert_eq!(
                        heavy + elastic_light_row(&parent, row),
                        stream.len() as i64,
                        "{workers} workers, row {row}: mass leaked"
                    );
                }
                for (key, exact) in &truth {
                    assert!(
                        parent.query(key.clone()) >= *exact as i32,
                        "{workers} workers: {key} underestimated"
                    );
                }
            }
        }

        /// One worker is the only configuration where the keyed families are
        /// reproducible: a single queue delivers in order, so the aggregator
        /// sees the same contest sequence as a sequential replay. Elastic draws
        /// no random numbers, so that makes it bit-exact. This pins the
        /// plumbing - payload rendering, delta routing, end-of-stream flush -
        /// with an equality the multi-worker cases cannot offer.
        #[test]
        fn run_octo_elastic_matches_a_single_threaded_replay_at_one_worker() {
            let stream = flow_stream(40_000, 2_048, 23_003);
            let inputs = inputs_from(&stream);
            let (rows, cols) = (3usize, 1_024usize);

            let mut replay = OctoElastic::new(1, 128, rows, cols, Route::HashByKey);
            for (i, raw) in stream.iter().enumerate() {
                replay.insert(i, *raw);
            }
            replay.flush();

            for attempt in 0..3 {
                let parent = run_octo(
                    &inputs,
                    &config(1),
                    ElasticOctoPlan::new(128, rows, cols),
                    || ElasticOctoAggregator::new(128, rows, cols),
                )
                .parent
                .sketch;

                for (bucket, (got, want)) in parent
                    .heavy
                    .iter()
                    .zip(replay.sketch().heavy.iter())
                    .enumerate()
                {
                    assert_eq!(
                        (&got.flow_id, got.vote_pos, got.vote_neg, got.eviction),
                        (&want.flow_id, want.vote_pos, want.vote_neg, want.eviction),
                        "attempt {attempt}, heavy bucket {bucket} diverged"
                    );
                }
                for row in 0..rows {
                    for col in 0..cols {
                        assert_eq!(
                            parent.light.as_storage().query_one_counter(row, col),
                            replay
                                .sketch()
                                .light
                                .as_storage()
                                .query_one_counter(row, col),
                            "attempt {attempt}, light cell ({row},{col}) diverged"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Paper conformance: delta promotion vs the sketch-merge baseline
    // -----------------------------------------------------------------------

    /// Merge periods that spend roughly the given counter budget, the sizing
    /// `merge_periods` does for the counter families.
    fn merge_period(octo_counters: usize, sketch_counters: usize, per_worker: usize) -> usize {
        let merges = (octo_counters as f64 / sketch_counters as f64)
            .ceil()
            .max(1.0) as usize;
        (per_worker / (merges + 1)).max(1)
    }

    /// Figures 20(a) and 21(a): CocoSketch's F1 and relative error against the
    /// same sketch queried *while the stream runs*, under OctoSketch and under
    /// periodic sketch-merge, at a merge baseline funded past parity. Asserts
    /// the direction the figures show, not Table 1's undefined accuracy ratio.
    ///
    /// The stream is sized so `ALPHA * N` sits well above τ: below that a
    /// threshold-sized flow barely promotes and Octo's F1 trails merge's.
    #[test]
    fn coco_online_accuracy_beats_sketch_merge() {
        let (w, d, workers, n) = (512usize, 2usize, 4usize, 200_000usize);
        let mut octo_are = 0.0;
        let mut merge_are = 0.0;
        let mut ideal_are = 0.0;
        let mut octo_f1 = 0.0;
        let mut merge_f1 = 0.0;
        let mut samples = 0.0;
        let (mut octo_hh_are, mut merge_hh_are) = (0.0, 0.0);
        let mut budget = (0usize, 0usize);

        for seed in 0..3u64 {
            let stream = flow_stream(n, 5_000, 24_001 + seed);
            let truth = truth_of(&stream);
            let mut ranked: Vec<(&String, &u64)> = truth.iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let watched: Vec<String> = ranked.iter().take(32).map(|(k, _)| (*k).clone()).collect();

            let mut probe = OctoCoco::new(workers, w, d, Route::HashByKey);
            for (i, raw) in stream.iter().enumerate() {
                probe.insert(i, *raw);
            }
            let period = merge_period(probe.sent_counters, w * d, n / workers);

            let mut octo = OctoCoco::new(workers, w, d, Route::HashByKey);
            let mut merge = MergeCoco::new(workers, w, d, period, Route::HashByKey);
            let mut ideal: Coco = Coco::init_with_size(w, d);
            let mut running: HashMap<String, u64> = HashMap::new();
            let mut seen = 0u64;
            let stride = n / 8;

            for (i, raw) in stream.iter().enumerate() {
                *running.entry(key_of(*raw)).or_insert(0) += 1;
                seen += 1;
                octo.insert(i, *raw);
                merge.insert(i, *raw);
                ideal.insert(&key_of(*raw), 1);
                if i > 0 && i % stride == stride / 3 {
                    let threshold = (seen as f64 * HH_ALPHA) as u64;
                    for key in &watched {
                        let exact = running[key] as f64;
                        octo_are += (octo.sketch().estimate_key(key) as f64 - exact).abs() / exact;
                        merge_are += (merge.parent.estimate_key(key) as f64 - exact).abs() / exact;
                        ideal_are += (ideal.estimate_key(key) as f64 - exact).abs() / exact;
                        samples += 1.0;
                    }
                    let octo_hh = score(
                        &coco_reported(octo.sketch(), threshold),
                        &running,
                        threshold,
                    );
                    let merge_hh = score(
                        &coco_reported(&merge.parent, threshold),
                        &running,
                        threshold,
                    );
                    octo_f1 += octo_hh.f1();
                    merge_f1 += merge_hh.f1();
                    octo_hh_are += octo_hh.are;
                    merge_hh_are += merge_hh.are;
                }
            }
            budget = (
                budget.0 + probe.sent_counters,
                budget.1 + merge.sent_counters,
            );
        }

        let (octo_are, merge_are, ideal_are) =
            (octo_are / samples, merge_are / samples, ideal_are / samples);
        let sweeps = samples / 32.0;
        let (octo_f1, merge_f1) = (octo_f1 / sweeps, merge_f1 / sweeps);
        println!(
            "Coco online: top-32 ARE ideal={ideal_are:.4} octo={octo_are:.4} merge={merge_are:.4} | \
             heavy hitters F1 octo={octo_f1:.4} merge={merge_f1:.4}, ARE octo={:.4} merge={:.4} \
             (merge spent {:.1}x octo's counters)",
            octo_hh_are / sweeps,
            merge_hh_are / sweeps,
            budget.1 as f64 / budget.0 as f64
        );

        assert!(
            budget.1 >= budget.0,
            "the baseline was starved: {} counters against {}",
            budget.1,
            budget.0
        );
        assert!(
            octo_are < merge_are / 2.0,
            "delta promotion should halve the online relative error: {octo_are:.4} vs {merge_are:.4}"
        );
        assert!(
            octo_f1 > merge_f1,
            "F1 should favour delta promotion: {octo_f1:.4} vs {merge_f1:.4}"
        );
        assert!(
            ideal_are < octo_are,
            "a single-threaded pass should lead: ideal {ideal_are:.4} vs octo {octo_are:.4}"
        );
        assert!(
            octo_are < 0.10,
            "octo relative error {octo_are:.4} left the measured band"
        );
    }

    /// Figures 20(c) and 21(c), the Elastic counterparts. Deterministic end to
    /// end: one seed is the whole population here, not a sample of it, so the
    /// relative errors below are the numbers rather than a band around them -
    /// 0.0000 for a single-threaded pass, 0.0466 for Octo, 0.6287 for merge.
    /// The gap between the first two is the mass still sitting in the workers,
    /// which §7.2 calls out as OctoSketch's systematic under-read.
    #[test]
    fn elastic_online_accuracy_beats_sketch_merge() {
        let (buckets, rows, cols, workers, n) =
            (1_024i32, 3usize, 4_096usize, 4usize, 300_000usize);
        let stream = flow_stream(n, 5_000, 24_101);
        let truth = truth_of(&stream);
        let mut ranked: Vec<(&String, &u64)> = truth.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let watched: Vec<String> = ranked.iter().take(32).map(|(k, _)| (*k).clone()).collect();

        let mut probe = OctoElastic::new(workers, buckets, rows, cols, Route::HashByKey);
        for (i, raw) in stream.iter().enumerate() {
            probe.insert(i, *raw);
        }
        let sketch_counters = buckets as usize + rows * cols;
        let period = merge_period(probe.sent_counters, sketch_counters, n / workers);

        let mut octo = OctoElastic::new(workers, buckets, rows, cols, Route::HashByKey);
        let mut merge = MergeElastic::new(workers, buckets, rows, cols, period, Route::HashByKey);
        let mut ideal: Elastic = Elastic::init_with_dimensions(buckets, rows, cols);
        let mut running: HashMap<String, u64> = HashMap::new();
        let mut seen = 0u64;
        let stride = n / 8;
        let (mut octo_are, mut merge_are, mut ideal_are) = (0.0, 0.0, 0.0);
        let (mut octo_f1, mut merge_f1) = (0.0, 0.0);
        let (mut octo_hh_are, mut merge_hh_are) = (0.0, 0.0);
        let (mut samples, mut sweeps) = (0.0, 0.0);

        for (i, raw) in stream.iter().enumerate() {
            let key = key_of(*raw);
            *running.entry(key.clone()).or_insert(0) += 1;
            seen += 1;
            octo.insert(i, *raw);
            merge.insert(i, *raw);
            ideal.insert(key);
            if i > 0 && i % stride == stride / 3 {
                let threshold = (seen as f64 * HH_ALPHA) as u64;
                for key in &watched {
                    let exact = running[key] as f64;
                    octo_are += (octo.sketch().query(key.clone()) as f64 - exact).abs() / exact;
                    merge_are += (merge.parent.query(key.clone()) as f64 - exact).abs() / exact;
                    ideal_are += (ideal.query(key.clone()) as f64 - exact).abs() / exact;
                    samples += 1.0;
                }
                let octo_hh = score(
                    &elastic_reported(octo.sketch(), threshold),
                    &running,
                    threshold,
                );
                let merge_hh = score(
                    &elastic_reported(&merge.parent, threshold),
                    &running,
                    threshold,
                );
                octo_f1 += octo_hh.f1();
                merge_f1 += merge_hh.f1();
                octo_hh_are += octo_hh.are;
                merge_hh_are += merge_hh.are;
                sweeps += 1.0;
            }
        }

        let (octo_are, merge_are, ideal_are) =
            (octo_are / samples, merge_are / samples, ideal_are / samples);
        let (octo_f1, merge_f1) = (octo_f1 / sweeps, merge_f1 / sweeps);
        println!(
            "Elastic online: top-32 ARE ideal={ideal_are:.4} octo={octo_are:.4} merge={merge_are:.4} \
             | heavy hitters F1 octo={octo_f1:.4} merge={merge_f1:.4}, ARE octo={:.4} merge={:.4} \
             (merge spent {:.1}x octo's counters)",
            octo_hh_are / sweeps,
            merge_hh_are / sweeps,
            merge.sent_counters as f64 / probe.sent_counters as f64
        );

        assert!(
            merge.sent_counters >= probe.sent_counters,
            "the baseline was starved: {} counters against {}",
            merge.sent_counters,
            probe.sent_counters
        );
        assert!(
            octo_are < merge_are / 2.0,
            "delta promotion should halve the online relative error: {octo_are:.4} vs {merge_are:.4}"
        );
        assert!(
            octo_f1 > merge_f1 * 1.2,
            "F1 should favour delta promotion: {octo_f1:.4} vs {merge_f1:.4}"
        );
        assert!(
            ideal_are < octo_are,
            "a single-threaded pass should lead: ideal {ideal_are:.4} vs octo {octo_are:.4}"
        );
        assert!(
            octo_are < 0.06,
            "octo relative error {octo_are:.4} left the measured band"
        );
    }
}

// ---------------------------------------------------------------------------
// The top-k plans: CmTopKOctoPlan and CountTopKOctoPlan
// ---------------------------------------------------------------------------

/// `CmTopKOctoPlan` / `CountTopKOctoPlan` are the plans in which the *aggregator*
/// owns the pipeline's only heavy-hitter heap: workers keep no flow keys, and
/// instead ship the key alongside every promoted counter (§3.2, Idea 3).
///
/// That design choice is what these tests are about. Because workers hold no
/// key storage, they also have **no flush**: a residual counter cannot be
/// attributed back to a key, so a flow that never crossed τ on any worker is
/// absent from the aggregator's heap entirely and estimates zero — not merely
/// undercounted. Every bound below is stated with that in mind.
mod top_k_plans {
    use super::*;
    use asap_sketchlib::{
        CMSHeap, CSHeap, CmTopKOctoAggregator, CmTopKOctoPlan, CountTopKOctoAggregator,
        CountTopKOctoPlan, HeapItem, KeyedHashes,
    };
    use common::specs::{CountMinSpec, CountSketchSpec, Tally};

    const WORKERS: usize = 4;
    const TOP_K: usize = 32;
    const N: usize = 200_000;
    const DOMAIN: usize = 8_192;
    const STREAM_SEED: u64 = 0x0C70_1101;

    /// The per-counter gap the paper's deterministic half allows between the
    /// aggregator and a single-threaded sketch: every one of the `k` workers
    /// may hold back up to `τ` of its own share of a shared counter.
    fn counter_ceiling() -> f64 {
        (WORKERS as i32 * TAU) as f64
    }

    fn stream() -> Vec<u64> {
        zipf_u64(N, DOMAIN, 1.1, STREAM_SEED)
    }

    fn truth_of(stream: &[u64]) -> FreqTruth {
        let mut t = FreqTruth::default();
        for k in stream {
            t.observe(*k as i64);
        }
        t
    }

    /// Drives a top-k plan across `WORKERS` workers under one partition rule
    /// and returns the aggregator.
    fn drive_cm(route: Route, stream: &[u64]) -> CmTopKOctoAggregator {
        let plan = CmTopKOctoPlan::new(ROWS, COLS);
        let mut workers: Vec<_> = (0..WORKERS).map(|i| plan.worker(i)).collect();
        let mut parent = plan.aggregator(TOP_K);
        for (i, k) in stream.iter().enumerate() {
            let input = DataInput::U64(*k);
            let w = route.worker(i, &input, WORKERS);
            let payload: KeyedHashes = plan.prepare(&input);
            let mut shipped = Vec::new();
            workers[w].process(&payload, &mut |d| shipped.push(d));
            for d in shipped {
                parent.apply(d);
            }
        }
        parent
    }

    fn drive_count(route: Route, stream: &[u64]) -> CountTopKOctoAggregator {
        let plan = CountTopKOctoPlan::new(ROWS, COLS);
        let mut workers: Vec<_> = (0..WORKERS).map(|i| plan.worker(i)).collect();
        let mut parent = plan.aggregator(TOP_K);
        for (i, k) in stream.iter().enumerate() {
            let input = DataInput::U64(*k);
            let w = route.worker(i, &input, WORKERS);
            let payload: KeyedHashes = plan.prepare(&input);
            let mut shipped = Vec::new();
            workers[w].process(&payload, &mut |d| shipped.push(d));
            for d in shipped {
                parent.apply(d);
            }
        }
        parent
    }

    /// Single-threaded reference: the same sketch fed the same stream with no
    /// promotion protocol in between.
    fn reference_cm(stream: &[u64]) -> CMSHeap<Vector2D<i32>, RegularPath> {
        let mut sk: CMSHeap<Vector2D<i32>, RegularPath> = CMSHeap::new(ROWS, COLS, TOP_K);
        for k in stream {
            sk.insert(&DataInput::U64(*k));
        }
        sk
    }

    fn reference_cs(stream: &[u64]) -> CSHeap<Vector2D<i32>, RegularPath> {
        let mut sk: CSHeap<Vector2D<i32>, RegularPath> = CSHeap::new(ROWS, COLS, TOP_K);
        for k in stream {
            sk.insert(&DataInput::U64(*k));
        }
        sk
    }

    /// `(key, count)` pairs out of an aggregator's heap.
    fn heap_entries(items: &[asap_sketchlib::HHItem]) -> Vec<(i64, i64)> {
        items
            .iter()
            .map(|it| match it.key {
                HeapItem::U64(v) => (v as i64, it.count),
                ref other => panic!("unexpected heap key {other:?}"),
            })
            .collect()
    }

    /// Point estimates from the Count-Min top-k aggregator, against the
    /// single-threaded reference and against exact truth.
    ///
    /// Two bounds apply and both are asserted:
    ///
    /// - **against the reference**, deterministically: Count-Min never
    ///   underestimates and each of the `k` workers withholds at most `τ` per
    ///   counter, so `ref - k*τ <= octo <= ref` for every key, under either
    ///   partition;
    /// - **against exact truth**, probabilistically: Count-Min's own additive
    ///   bound, widened by exactly that same `k*τ` promotion residual and by
    ///   nothing else.
    #[test]
    fn cm_top_k_point_estimates_trail_the_single_thread_reference_by_at_most_k_tau() {
        let stream = stream();
        let truth = truth_of(&stream);
        let reference = reference_cm(&stream);
        let ceiling = counter_ceiling();

        for route in [Route::HashByKey, Route::RoundRobin] {
            let parent = drive_cm(route, &stream);
            let context = format!(
                "{route:?} workers={WORKERS} rows={ROWS} cols={COLS} tau={TAU} top_k={TOP_K} \
                 zipf(1.1) domain={DOMAIN} n={N} seed={STREAM_SEED:#x}"
            );

            let mut gap = Tally::default();
            for (k, _) in truth.pairs() {
                let d = DataInput::U64(k as u64);
                let octo = parent.sketch.estimate(&d) as f64;
                let single = reference.estimate(&d) as f64;
                gap.record(octo <= single && single - octo <= ceiling, || {
                    format!(
                        "key {k}: octo {octo}, single-thread {single}, gap {} outside \
                         [0, k*tau = {ceiling}]",
                        single - octo
                    )
                });
            }
            gap.assert_none(
                &format!("CmTopK octo vs single-thread ({route:?})"),
                &context,
            );

            // Count-Min's own bound, widened only by the promotion residual.
            //
            // Two acceptance rules, because the two halves are different kinds
            // of statement. The simultaneous bound is union-bounded over every
            // probed key, so it needs no independence between them and
            // tolerates zero violations; the marginal `e*(N-f)/w` is pinned as
            // a violation *rate* against the theorem's own `e^-d`, because the
            // keys of one sketch share one hash and are a single realisation
            // rather than a battery of trials.
            let spec = CountMinSpec::new(ROWS, COLS);
            let total = truth.total() as f64;
            let distinct = truth.distinct();
            let mut simultaneous = Tally::default();
            let mut marginal = Tally::default();
            let mut one_sided = Tally::default();
            for (k, c) in truth.pairs() {
                let est = parent.sketch.estimate(&DataInput::U64(k as u64)) as f64;
                let f = c as f64;
                // One-sided modulo the residual: the parent can only read low
                // by what the workers are still holding.
                one_sided.record(est >= f - ceiling, || {
                    format!("key {k}: true {f}, octo {est}, deficit beyond k*tau = {ceiling}")
                });
                let hi = f + spec.simultaneous_bound(
                    total,
                    f,
                    distinct,
                    common::specs::SIMULTANEOUS_LEVEL,
                );
                simultaneous.record(est <= hi, || {
                    format!("key {k}: true {f}, octo {est} above simultaneous ceiling {hi:.1}")
                });
                let marginal_hi = f + spec.marginal_bound(total, f);
                marginal.record(est <= marginal_hi, || {
                    format!(
                        "key {k}: true {f}, octo {est} above e*(N-f)/w ceiling {marginal_hi:.1}"
                    )
                });
            }
            one_sided.assert_none(
                &format!("CmTopK one-sided modulo k*tau ({route:?})"),
                &context,
            );
            simultaneous.assert_none(
                &format!("CmTopK simultaneous additive bound ({route:?})"),
                &context,
            );
            marginal.assert_rate_at_most(
                &format!("CmTopK marginal additive bound ({route:?})"),
                spec.marginal_failure(),
                &context,
            );
        }
    }

    /// The same two bounds for the signed plan: Count Sketch's L2 bound, and a
    /// two-sided `k*τ` gap against the single-threaded reference (signed
    /// counters can be withheld in either direction).
    #[test]
    fn count_top_k_point_estimates_satisfy_the_l2_bound_within_the_promotion_residual() {
        let stream = stream();
        let truth = truth_of(&stream);
        let reference = reference_cs(&stream);
        let ceiling = counter_ceiling();
        let spec = CountSketchSpec::new(ROWS, COLS);

        for route in [Route::HashByKey, Route::RoundRobin] {
            let parent = drive_count(route, &stream);
            let context = format!(
                "{route:?} workers={WORKERS} rows={ROWS} cols={COLS} tau={TAU} top_k={TOP_K} \
                 zipf(1.1) domain={DOMAIN} n={N} seed={STREAM_SEED:#x}"
            );

            let mut gap = Tally::default();
            for (k, _) in truth.pairs() {
                let d = DataInput::U64(k as u64);
                let octo = parent.sketch.estimate(&d);
                let single = reference.estimate(&d);
                gap.record((octo - single).abs() <= ceiling, || {
                    format!(
                        "key {k}: octo {octo}, single-thread {single}, |gap| {} above \
                         k*tau = {ceiling}",
                        (octo - single).abs()
                    )
                });
            }
            gap.assert_none(
                &format!("CountTopK octo vs single-thread ({route:?})"),
                &context,
            );

            // The L2 bound with the promotion residual added; nothing else.
            // Simultaneous half is union-bounded over the key set and tolerates
            // nothing; marginal half is a rate pin at kappa = 3.
            let f2 = truth.f2();
            let kappa =
                spec.simultaneous_kappa(truth.distinct(), common::specs::SIMULTANEOUS_LEVEL);
            let mut simultaneous = Tally::default();
            let mut marginal = Tally::default();
            for (k, c) in truth.pairs() {
                let f = c as f64;
                let est = parent.sketch.estimate(&DataInput::U64(k as u64));
                let residual_l2 = (f2 - f * f).max(0.0).sqrt();
                let simul = spec.scale_at(kappa, residual_l2) + ceiling;
                simultaneous.record((est - f).abs() <= simul, || {
                    format!(
                        "key {k}: true {f}, octo {est}, |error| {:.1} above the simultaneous \
                         sqrt(kappa/w)*||f_-i||_2 + k*tau = {simul:.1} (kappa={kappa:.1})",
                        (est - f).abs()
                    )
                });
                let marg = spec.marginal_scale(residual_l2) + ceiling;
                marginal.record((est - f).abs() <= marg, || {
                    format!(
                        "key {k}: true {f}, octo {est}, |error| {:.1} above \
                         sqrt(3/w)*||f_-i||_2 + k*tau = {marg:.1}",
                        (est - f).abs()
                    )
                });
            }
            simultaneous.assert_none(
                &format!("CountTopK simultaneous L2 bound + promotion residual ({route:?})"),
                &context,
            );
            marginal.assert_rate_at_most(
                &format!("CountTopK marginal L2 bound + promotion residual ({route:?})"),
                spec.marginal_failure(),
                &context,
            );
        }
    }

    /// Heavy-hitter recall and precision for both top-k plans.
    ///
    /// Recall is stated over the keys the protocol can actually deliver: a
    /// flow must cross τ on some worker before its key ever reaches the
    /// aggregator, so the guaranteed set is the flows whose count clears the
    /// worst-case promotion floor, `k*τ`. Precision is stated over the reported
    /// set: every key the heap holds must genuinely be heavy, which for
    /// Count-Min means at or above the true k-th count minus the residual.
    ///
    /// The `HashByKey` premise matters and is named in the assertion: under it
    /// a flow's whole count lands on one worker, so a flow of count `f`
    /// promotes `floor(f/τ)` times. Under `RoundRobin` the same flow is split
    /// `k` ways and may promote less, which is why recall is measured
    /// separately per partition instead of averaged.
    #[test]
    fn top_k_plans_recover_the_heavy_hitters_their_promotion_floor_guarantees() {
        let stream = stream();
        let truth = truth_of(&stream);
        let floor = counter_ceiling();
        let true_top: Vec<(i64, i64)> = truth.top_k(TOP_K);
        let kth = true_top[TOP_K - 1].1 as f64;

        for route in [Route::HashByKey, Route::RoundRobin] {
            let context = format!(
                "{route:?} workers={WORKERS} tau={TAU} top_k={TOP_K} promotion floor k*tau={floor} \
                 zipf(1.1) domain={DOMAIN} n={N} seed={STREAM_SEED:#x}, true k-th count {kth}"
            );

            // Each entry is (key, count held in the heap, estimate the
            // aggregator's own sketch returns for that key).
            for (label, entries) in [
                {
                    let parent = drive_cm(route, &stream);
                    let e: Vec<(i64, i64, f64)> = heap_entries(parent.sketch.heap().heap())
                        .into_iter()
                        .map(|(k, held)| {
                            (
                                k,
                                held,
                                parent.sketch.estimate(&DataInput::U64(k as u64)) as f64,
                            )
                        })
                        .collect();
                    ("CmTopK", e)
                },
                {
                    let parent = drive_count(route, &stream);
                    let e: Vec<(i64, i64, f64)> = heap_entries(parent.sketch.heap().heap())
                        .into_iter()
                        .map(|(k, held)| {
                            (k, held, parent.sketch.estimate(&DataInput::U64(k as u64)))
                        })
                        .collect();
                    ("CountTopK", e)
                },
            ] {
                let keys: Vec<i64> = entries.iter().map(|(k, _, _)| *k).collect();
                assert!(
                    keys.len() <= TOP_K,
                    "{label} {route:?}: heap holds {} entries, capacity {TOP_K}",
                    keys.len()
                );
                let unique: std::collections::HashSet<i64> = keys.iter().copied().collect();
                assert_eq!(
                    unique.len(),
                    keys.len(),
                    "{label} {route:?}: the heap holds a duplicate key"
                );

                // Recall over the guaranteed set: every true top-k key whose
                // count clears the promotion floor must be present.
                let guaranteed: Vec<i64> = true_top
                    .iter()
                    .filter(|(_, c)| *c as f64 > floor)
                    .map(|(k, _)| *k)
                    .collect();
                assert!(
                    guaranteed.len() >= TOP_K / 2,
                    "test premise: only {} of the true top {TOP_K} clear the promotion floor \
                     {floor}; pick a heavier stream. {context}",
                    guaranteed.len()
                );
                let found = guaranteed.iter().filter(|k| keys.contains(k)).count();
                assert_eq!(
                    found,
                    guaranteed.len(),
                    "{label} {route:?}: recovered {found} of {} keys above the promotion \
                     floor. {context}",
                    guaranteed.len()
                );

                // Precision over the reported set: nothing in the heap may be
                // lighter than the true k-th count, allowing for the residual.
                let mut precision = Tally::default();
                for (k, _, _) in &entries {
                    let exact = truth.get(*k) as f64;
                    precision.record(exact >= kth - floor, || {
                        format!(
                            "key {k} is in the top-{TOP_K} heap with a true count of {exact}, \
                             below the true k-th count {kth} minus the residual {floor}"
                        )
                    });
                }
                precision.assert_none(&format!("{label} {route:?} precision"), &context);

                // Heap/sketch consistency is structural: the aggregator writes
                // the estimate into the heap as it applies each delta, so an
                // entry that disagrees with the sketch means one of the two
                // was updated and the other was not.
                let mut consistency = Tally::default();
                for (k, held, est) in &entries {
                    consistency.record(*held == *est as i64, || {
                        format!("key {k}: heap holds {held} but the sketch estimates {est}")
                    });
                }
                consistency.assert_none(&format!("{label} {route:?} heap consistency"), &context);
            }
        }
    }

    /// The top-k workers have no `flush`, and that is a contract, not an
    /// omission: with no key storage a residual counter cannot be attributed
    /// back to a flow. A key that never crosses τ therefore never reaches the
    /// aggregator at all and estimates zero — which is what makes the `k*τ`
    /// deficit above a *deficit* rather than an unbounded miss.
    #[test]
    fn a_key_below_the_promotion_threshold_never_reaches_the_top_k_aggregator() {
        let plan = CmTopKOctoPlan::new(ROWS, COLS);
        let mut worker = plan.worker(0);
        let mut parent = plan.aggregator(TOP_K);

        // One key, fewer arrivals than τ: nothing may be shipped.
        let cold = DataInput::U64(0xC01D);
        let payload = plan.prepare(&cold);
        let mut shipped = 0usize;
        for _ in 0..(TAU - 1) {
            worker.process(&payload, &mut |_| shipped += 1);
        }
        assert_eq!(
            shipped, 0,
            "a flow below tau={TAU} must not promote a single counter"
        );
        assert_eq!(
            parent.sketch.estimate(&cold),
            0,
            "an unpromoted flow must estimate zero at the aggregator, not a partial count"
        );
        assert!(
            parent.sketch.heap().heap().is_empty(),
            "an unpromoted flow must be absent from the heap entirely"
        );

        // One more arrival crosses tau and the key appears, carrying exactly
        // the promoted window.
        let mut promoted = Vec::new();
        worker.process(&payload, &mut |d| promoted.push(d));
        assert!(
            !promoted.is_empty(),
            "crossing tau={TAU} must promote at least one counter"
        );
        for d in promoted {
            assert_eq!(
                d.key,
                asap_sketchlib::input_to_owned(&cold),
                "every top-k delta must carry the flow key that produced it"
            );
            parent.apply(d);
        }
        assert_eq!(
            parent.sketch.estimate(&cold),
            TAU,
            "the first promotion must hand over exactly the tau-sized window"
        );
        assert_eq!(
            parent.sketch.heap().heap().len(),
            1,
            "the promoted key must now hold a heap slot"
        );
    }
}

// ---------------------------------------------------------------------------
// Partition accuracy: the five plans that were only ever driven under HashByKey
// ---------------------------------------------------------------------------

/// Accuracy — not just mass conservation — for the HLL, DDSketch, CocoSketch,
/// Elastic and UnivMon Octo plans under **both** partitions.
///
/// The rest of this file drives those five plans under `HashByKey` only, and
/// checks invariants that are partition-independent by construction (mass
/// conservation, flush completeness, delta well-formedness). Those invariants
/// would hold on an implementation that routed every key to worker 0, so on
/// their own they say nothing about whether `RoundRobin` is *accurate*.
///
/// The difference the partition makes is real and family-specific:
///
/// - under `HashByKey` a flow's whole count lands on one worker, so it crosses
///   the promotion threshold `floor(f / tau)` times and at most `tau - 1` of its
///   mass is ever in flight;
/// - under `RoundRobin` the same flow is split across all `k` workers, so up to
///   `k * (tau - 1)` of its mass can be held back at once, and any structure
///   that evicts per worker (CocoSketch, Elastic) sees `k` independent eviction
///   contests instead of one.
///
/// So each test below compares the aggregator against **exact truth** under the
/// family's own error metric, with the residual the route actually permits, and
/// against a single-threaded reference where the protocol promises equality.
mod partition_accuracy {
    use super::*;
    use asap_sketchlib::{
        Coco, CocoDelta, CocoOctoPlan, DdOctoPlan, Elastic, ElasticDelta, ElasticOctoPlan,
        HllOctoPlan, UNIVMON_PROMASK, flow_key_string,
    };
    use common::specs::{CardinalityConfidenceSpec, Tally};

    const WORKERS: usize = 4;
    const N: usize = 60_000;
    const DOMAIN: usize = 1_024;
    const STREAM_SEED: u64 = 0x0C70_ACC0;

    fn routes() -> [Route; 2] {
        [Route::HashByKey, Route::RoundRobin]
    }

    fn stream() -> Vec<u64> {
        keys(N, DOMAIN, STREAM_SEED)
    }

    fn truth_of(stream: &[u64]) -> FreqTruth {
        let mut truth = FreqTruth::default();
        for k in stream {
            truth.observe(*k as i64);
        }
        truth
    }

    /// The rendered flow key `CocoOctoPlan` and `ElasticOctoPlan` ship, so a
    /// query uses exactly the key the plan produced.
    fn key_of(raw: u64) -> String {
        flow_key_string(&DataInput::U64(raw))
    }

    fn context(route: Route, extra: &str) -> String {
        format!(
            "{route:?} workers={WORKERS} zipf(1.1) domain={DOMAIN} n={N} \
             seed={STREAM_SEED:#x}; {extra}"
        )
    }

    /// HLL promotion carries **no partition penalty at all**.
    ///
    /// A register only ever moves to a larger leading-zero count, and
    /// `HLL_PROMASK` promotes every improvement, so the parent's registers are
    /// the elementwise maximum of what any worker saw — which is the
    /// single-threaded sketch, byte for byte, under any routing. That is an
    /// equality, and stating it as one is strictly stronger than a band: a
    /// tolerance would pass on an implementation that dropped a slice of the
    /// improvements under `RoundRobin`.
    ///
    /// The cardinality band is then checked against exact truth, so the test
    /// also fails if the shared reference is itself wrong.
    #[test]
    fn hll_octo_plan_is_register_exact_and_in_band_under_both_partitions() {
        let stream = stream();
        let distinct: std::collections::HashSet<u64> = stream.iter().copied().collect();

        let mut reference = HyperLogLog::<Classic>::default();
        for k in &stream {
            reference.insert(&DataInput::U64(*k));
        }

        let spec = CardinalityConfidenceSpec::hll(14, 4.0);
        for route in routes() {
            let plan = HllOctoPlan::new();
            let mut workers: Vec<_> = (0..WORKERS).map(|id| plan.worker(id)).collect();
            let mut parent = plan.aggregator();
            for (i, k) in stream.iter().enumerate() {
                let input = DataInput::U64(*k);
                let w = route.worker(i, &input, WORKERS);
                let payload = plan.prepare(&input);
                workers[w].process(&payload, &mut |d| parent.apply(d));
            }
            for w in workers.iter_mut() {
                w.flush(&mut |d| parent.apply(d));
            }

            let ctx = context(route, "HllOctoPlan, p14 Classic");
            assert_eq!(
                parent.sketch.registers_as_slice(),
                reference.registers_as_slice(),
                "HLL promotion must be register-exact under {route:?}: max-merge is \
                 order- and partition-independent. {ctx}"
            );
            if let Err(detail) = spec.check(parent.sketch.estimate() as f64, distinct.len()) {
                panic!("HllOctoPlan {route:?}: {detail}. {ctx}");
            }
        }
    }

    /// DDSketch promotion is **mass-exact after a flush**, under either route.
    ///
    /// Bucket deltas are additive `(index, count)` pairs and a flush ships every
    /// counter a worker still holds, so the parent's store is the single-pass
    /// store bucket for bucket however the stream was partitioned. The relative
    /// value error against the exact order statistics is then checked with
    /// zero tolerance, because DDSketch's guarantee is deterministic.
    #[test]
    fn dd_octo_plan_is_bucket_exact_after_flush_and_holds_alpha_under_both_partitions() {
        use common::specs::RelativeQuantileSpec;

        const ALPHA: f64 = 0.01;
        const QS: [f64; 7] = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0];

        // Positive values spanning three decades, so the bucket store is wide.
        let values: Vec<f64> = stream().iter().map(|k| 1.0 + *k as f64).collect();
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);

        let mut reference = DDSketch::new(ALPHA);
        for v in &values {
            reference.add(v);
        }

        let spec = RelativeQuantileSpec::core(ALPHA);
        for route in routes() {
            let plan = DdOctoPlan::new(ALPHA);
            let mut workers: Vec<_> = (0..WORKERS).map(|id| plan.worker(id)).collect();
            let mut parent = plan.aggregator();
            for (i, v) in values.iter().enumerate() {
                let input = DataInput::F64(*v);
                let w = route.worker(i, &input, WORKERS);
                let payload = plan.prepare(&input);
                workers[w].process(&payload, &mut |d| parent.apply(d));
            }
            for w in workers.iter_mut() {
                w.flush(&mut |d| parent.apply(d));
            }

            let ctx = context(route, &format!("DdOctoPlan alpha={ALPHA}"));
            assert_eq!(
                parent.sketch.get_count(),
                reference.get_count(),
                "a flushed DDSketch parent must hold every observation. {ctx}"
            );
            // Bucket-for-bucket equality, allowing for the two stores having
            // grown to different offsets: compare the shared index range.
            let (p_off, r_off) = (parent.sketch.store_offset(), reference.store_offset());
            let (p, r) = (parent.sketch.store_counts(), reference.store_counts());
            let lo = p_off.min(r_off);
            let hi = (p_off + p.len() as i32).max(r_off + r.len() as i32);
            let at = |counts: &[u64], off: i32, k: i32| -> u64 {
                let idx = k - off;
                if idx < 0 || idx as usize >= counts.len() {
                    0
                } else {
                    counts[idx as usize]
                }
            };
            let mut buckets = Tally::default();
            for k in lo..hi {
                let (a, b) = (at(p, p_off, k), at(r, r_off, k));
                buckets.record(a == b, || {
                    format!("bucket {k}: octo {a}, single-thread {b}")
                });
            }
            buckets.assert_none(
                &format!("DdOctoPlan bucket equality after flush ({route:?})"),
                &ctx,
            );

            let mut tally = Tally::default();
            spec.tally_into(&mut tally, &sorted, &QS, |q| {
                parent.sketch.get_value_at_quantile(q)
            });
            tally.assert_none(
                &format!("DdOctoPlan relative value error ({route:?})"),
                &ctx,
            );
        }
    }

    /// UnivMon promotion under both partitions, judged on the quantity the
    /// aggregator actually reports.
    ///
    /// # What `calc_l1` means for the Octo plan, and why it is not `N`
    ///
    /// `UnivMon::calc_l1` returns the sketch's `bucket_size`, which for an Octo
    /// parent is restored from the `weight_total` each worker stamps on the
    /// deltas it emits. `UnivMonOctoWorker` is the one shipped worker with **no
    /// `flush`**, and it cannot have one: `LayeredCountDelta` carries the flow
    /// key that triggered the promotion, because `apply_layered_delta` needs it
    /// to update the aggregator's heavy-hitter heap — and a counter flushed at
    /// end-of-stream has no single key to name. So a worker's updates after its
    /// last emitted delta never reach the parent, and the parent's L1 is a
    /// **lower bound** on `N`, not an equality.
    ///
    /// That is asserted exactly rather than with a tolerance: the test records
    /// what each worker last reported, and the parent must equal precisely the
    /// sum of those. `N` minus that sum is the mass still in flight, printed in
    /// the failure message so the size of the lag is visible. See the coverage
    /// matrix's findings section.
    ///
    /// `calc_l2` returns the L2 **norm** — `sqrt(F2)` — over the delivered
    /// counters, so it is squared before being held to the AMS second-moment
    /// bound the estimator actually satisfies.
    #[test]
    fn univmon_octo_plan_reports_the_delivered_l1_and_holds_its_f2_bound_under_both_partitions() {
        use common::specs::SecondMomentSpec;

        const ROWS: usize = 5;
        const COLS: usize = 4_096;
        const LAYERS: usize = 8;
        const HEAP: usize = 64;

        let stream = stream();
        let truth = truth_of(&stream);
        let f2_spec = SecondMomentSpec::new(ROWS, COLS);

        for route in routes() {
            let plan = UnivMonOctoPlan::new(ROWS, COLS, LAYERS);
            let mut workers: Vec<_> = (0..WORKERS).map(|id| plan.worker(id)).collect();
            let mut parent = plan.aggregator(HEAP);
            // The last `weight_total` each worker stamped on a delta. The
            // parent's L1 is restored from exactly these, so this is the
            // predicted value rather than a band around it.
            let mut last_reported = vec![0u64; WORKERS];
            let mut processed = vec![0u64; WORKERS];
            for (i, k) in stream.iter().enumerate() {
                let input = DataInput::U64(*k);
                let w = route.worker(i, &input, WORKERS);
                processed[w] += 1;
                let payload = plan.prepare(&input);
                let mut reported = last_reported[w];
                workers[w].process(&payload, &mut |d: LayeredCountDelta| {
                    reported = reported.max(d.weight_total);
                    parent.apply(d);
                });
                last_reported[w] = reported;
            }
            // The default `OctoWorker::flush` is a no-op here, which is the
            // whole point: nothing further arrives.
            for w in workers.iter_mut() {
                w.flush(&mut |d: LayeredCountDelta| parent.apply(d));
            }

            let delivered: u64 = last_reported.iter().sum();
            let in_flight = truth.total() as u64 - delivered;
            let ctx = context(
                route,
                &format!(
                    "UnivMonOctoPlan rows={ROWS} cols={COLS} layers={LAYERS} heap={HEAP} \
                     tau={UNIVMON_PROMASK}; per-worker processed {processed:?}, last \
                     reported {last_reported:?}, so {in_flight} updates are still held by \
                     workers that emitted no further delta"
                ),
            );

            assert!(
                delivered <= truth.total() as u64,
                "a worker cannot report more weight than it processed. {ctx}"
            );
            assert_eq!(
                parent.sketch.calc_l1(),
                delivered as f64,
                "the parent's L1 is restored from the last weight_total each worker \
                 stamped on a delta, so it must equal their sum exactly. {ctx}"
            );

            // `calc_l2()` is the norm, so square it to get the F2 the AMS
            // row-median bound is stated over.
            let f2_estimate = parent.sketch.calc_l2().powi(2);
            if let Err(detail) = f2_spec.check(f2_estimate, truth.f2()) {
                panic!("UnivMonOctoPlan {route:?} L2/F2: {detail}. {ctx}");
            }
        }
    }

    /// CocoSketch and Elastic under both partitions, against exact truth.
    ///
    /// Both are eviction-based heavy-hitter sketches, so their guarantee is
    /// one-sided over-attribution: an estimate may credit a key with mass that
    /// belonged to a colliding one, but a key that survives eviction is never
    /// read *low*. That is what the route can change — under `RoundRobin` each
    /// of the `k` workers runs its own eviction contest, so a flow can be
    /// evicted `k` times independently — and it is checked here rather than
    /// assumed.
    ///
    /// The comparison is against exact truth for the true top flows, and
    /// against a single-threaded sketch for the total mass the protocol
    /// delivered.
    #[test]
    fn coco_and_elastic_octo_plans_stay_one_sided_on_heavy_keys_under_both_partitions() {
        const COCO_W: usize = 4_096;
        const COCO_D: usize = 2;
        const ELASTIC_BUCKETS: i32 = 512;
        const ELASTIC_ROWS: usize = 3;
        const ELASTIC_COLS: usize = 4_096;
        const HEAVY: usize = 32;

        let stream = stream();
        let truth = truth_of(&stream);
        let heavy = truth.top_k(HEAVY);

        for route in routes() {
            // --- CocoSketch ------------------------------------------------
            let plan = CocoOctoPlan::new(COCO_W, COCO_D);
            let mut workers: Vec<_> = (0..WORKERS).map(|id| plan.worker(id)).collect();
            let mut parent = plan.aggregator();
            for (i, k) in stream.iter().enumerate() {
                let input = DataInput::U64(*k);
                let w = route.worker(i, &input, WORKERS);
                let payload = key_of(*k);
                workers[w].process(&payload, &mut |d: CocoDelta| parent.apply(d));
            }
            for w in workers.iter_mut() {
                w.flush(&mut |d: CocoDelta| parent.apply(d));
            }

            let mut reference: Coco = Coco::init_with_size(COCO_W, COCO_D);
            for k in &stream {
                reference.insert(&key_of(*k), 1);
            }

            let ctx = context(route, &format!("CocoOctoPlan w={COCO_W} d={COCO_D}"));
            let mut one_sided = Tally::default();
            for (k, c) in &heavy {
                let est = parent.sketch.estimate_key(&key_of(*k as u64)) as i64;
                one_sided.record(est >= *c, || {
                    format!(
                        "key {k}: true {c}, octo {est} — CocoSketch must not read a \
                         surviving heavy key low"
                    )
                });
            }
            one_sided.assert_none(
                &format!("CocoOctoPlan one-sided on the true top {HEAVY} ({route:?})"),
                &ctx,
            );
            // Over-attribution has a ceiling: no estimate may exceed the whole
            // stream mass, because that is more than was ever inserted. The
            // single-threaded reference is checked against the same two rules,
            // so a failure separates "the partition broke it" from "CocoSketch
            // at these dimensions does this anyway".
            let mut ceiling = Tally::default();
            for (k, c) in &heavy {
                let key = key_of(*k as u64);
                let est = parent.sketch.estimate_key(&key) as i64;
                let single = reference.estimate_key(&key) as i64;
                ceiling.record(est <= truth.total(), || {
                    format!(
                        "key {k}: octo {est} exceeds the whole stream mass {}",
                        truth.total()
                    )
                });
                ceiling.record(single >= *c && single <= truth.total(), || {
                    format!(
                        "key {k}: the single-threaded reference itself reads {single} for a \
                         true count of {c}, so the partition is not what broke this"
                    )
                });
            }
            ceiling.assert_none(
                &format!("CocoOctoPlan estimates stay inside the stream mass ({route:?})"),
                &ctx,
            );

            // --- Elastic ---------------------------------------------------
            let plan = ElasticOctoPlan::new(ELASTIC_BUCKETS, ELASTIC_ROWS, ELASTIC_COLS);
            let mut workers: Vec<_> = (0..WORKERS).map(|id| plan.worker(id)).collect();
            let mut parent = plan.aggregator();
            for (i, k) in stream.iter().enumerate() {
                let input = DataInput::U64(*k);
                let w = route.worker(i, &input, WORKERS);
                let payload = key_of(*k);
                workers[w].process(&payload, &mut |d: ElasticDelta| parent.apply(d));
            }
            for w in workers.iter_mut() {
                w.flush(&mut |d: ElasticDelta| parent.apply(d));
            }

            let mut reference: Elastic =
                Elastic::init_with_dimensions(ELASTIC_BUCKETS, ELASTIC_ROWS, ELASTIC_COLS);
            for k in &stream {
                reference.insert(key_of(*k));
            }

            let ctx = context(
                route,
                &format!(
                    "ElasticOctoPlan buckets={ELASTIC_BUCKETS} rows={ELASTIC_ROWS} \
                     cols={ELASTIC_COLS}"
                ),
            );
            // Elastic's light layer is a Count-Min, so a key evicted from the
            // heavy table is read through a one-sided estimator: the answer can
            // sit above the truth but never above the whole stream mass, and
            // never below zero. As above, the single-threaded reference is held
            // to the same rule so the two can be told apart.
            let mut elastic_ceiling = Tally::default();
            for (k, _) in &heavy {
                let key = key_of(*k as u64);
                let est = parent.sketch.query(key.clone()) as i64;
                let single = reference.query(key) as i64;
                elastic_ceiling.record(est >= 0 && est <= truth.total(), || {
                    format!(
                        "key {k}: octo {est} outside [0, stream mass {}]",
                        truth.total()
                    )
                });
                elastic_ceiling.record(single >= 0 && single <= truth.total(), || {
                    format!(
                        "key {k}: the single-threaded reference itself reads {single}, \
                         outside [0, stream mass {}]",
                        truth.total()
                    )
                });
            }
            elastic_ceiling.assert_none(
                &format!("ElasticOctoPlan estimates stay inside the stream mass ({route:?})"),
                &ctx,
            );
        }
    }
}
