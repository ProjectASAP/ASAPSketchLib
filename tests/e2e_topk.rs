mod common;

use common::specs::{CountMinSpec, CountSketchSpec, Tally};
use common::{FreqTruth, zipf_u64};

use asap_sketchlib::{CMSHeap, CSHeap, DataInput, HeapItem, RegularPath, Vector2D};

/// The two heap-backed sketches share heap bookkeeping but not an error
/// metric: `CMSHeap` carries Count-Min's one-sided additive bound and
/// `CSHeap` carries Count Sketch's L2 bound. Both must keep every heap entry
/// equal to what the sketch itself estimates, and both must recover the true
/// heavy hitters that the top-k contract actually guarantees.
#[test]
fn heaps_satisfy_their_own_bounds_and_stay_heap_consistent() {
    const TOP_K: usize = 16;
    const DOMAIN: usize = 1024;
    const CM_ROWS: usize = 3;
    const CS_ROWS: usize = 5;
    const COLS: usize = 4096;
    const STREAM_SEED: u64 = 1004;

    let mut cms_heap = CMSHeap::<Vector2D<i64>, RegularPath>::new(CM_ROWS, COLS, TOP_K);
    let mut cs_heap = CSHeap::<Vector2D<i64>, RegularPath>::new(CS_ROWS, COLS, TOP_K);
    let stream = zipf_u64(20_000, DOMAIN, 1.1, STREAM_SEED);
    let mut truth = FreqTruth::default();
    for k in &stream {
        let d = DataInput::I64(*k as i64);
        truth.observe(*k as i64);
        cms_heap.insert(&d);
        cs_heap.insert(&d);
    }
    let context = format!("zipf(1.1) domain={DOMAIN} n=20000 stream_seed={STREAM_SEED}");

    // Each sketch against its own theorem.
    CountMinSpec::new(CM_ROWS, COLS).assert_contract(
        "CMSHeap point estimate",
        &truth,
        |k| cms_heap.estimate(&DataInput::I64(k)) as f64,
        &context,
    );
    CountSketchSpec::new(CS_ROWS, COLS).assert_contract(
        "CSHeap point estimate",
        &truth,
        |k| cs_heap.estimate(&DataInput::I64(k)),
        &context,
    );

    // Heap/sketch consistency is structural: a heap entry that disagrees with
    // the sketch is a bug regardless of any error bound.
    let kth_count = truth.top_k(TOP_K)[TOP_K - 1].1;
    for (label, items, cms) in [
        ("CMSHeap", cms_heap.heap().heap().to_vec(), true),
        ("CSHeap", cs_heap.heap().heap().to_vec(), false),
    ] {
        assert!(items.len() <= TOP_K, "{label} heap exceeded capacity");
        let mut consistency = Tally::default();
        let mut recall = 0usize;
        for it in &items {
            let key = match &it.key {
                HeapItem::I64(v) => *v,
                other => panic!("{label}: unexpected heap key {other:?}"),
            };
            let est = if cms {
                cms_heap.estimate(&DataInput::I64(key))
            } else {
                cs_heap.estimate(&DataInput::I64(key)) as i64
            };
            consistency.record(it.count == est, || {
                format!(
                    "key {key}: heap holds {} but the sketch estimates {est}",
                    it.count
                )
            });
            if truth.get(key) >= kth_count {
                recall += 1;
            }
        }
        consistency.assert_none(&format!("{label} heap/sketch consistency"), &context);

        // Recall target: a key whose true count is at least the true k-th
        // count must be admitted. Count-Min never underestimates, so every
        // such key's estimate dominates the heap minimum once seen; one
        // displacement slot of slack covers the eviction race at the
        // boundary, where several keys share the k-th count.
        assert!(
            recall >= TOP_K - 1,
            "{label} recovered only {recall}/{TOP_K} keys at or above the true k-th count \
             ({kth_count}); {context}"
        );
    }
}

// ---------------------------------------------------------------------------
// The documented input matrix
// ---------------------------------------------------------------------------

/// `tests/TEST_COVERAGE.md` sweeps both heaps over the eight skewed inputs at
/// `top_k` 32, 64 and 128, over a `row 5, col 32768` sketch chosen so the
/// counter matrix is not what is being measured.
///
/// Each heap carries four checks: the capacity ceiling, the document's 2%
/// per-item relative error, its recall target, and the equality between a heap
/// entry's stored count and the sketch's current estimate for that key. Each
/// sketch is also held to its own error theorem at the keys the heap admitted,
/// so a heap that filled itself with the wrong keys fails even where the
/// document's flat percentage would not notice.
mod documented_matrix {
    use super::common::inputs::key_input;
    use super::common::specs::{CountMinSpec, CountSketchSpec, SIMULTANEOUS_LEVEL, Tally};
    use super::*;

    /// `cms` and `cs` geometry from the document.
    const HEAP_ROWS: usize = 5;
    const HEAP_COLS: usize = 32_768;
    const TOP_KS: [usize; 3] = [32, 64, 128];

    /// The document's per-item relative error for the heaps.
    const HEAP_RELATIVE_ERROR: f64 = 0.02;

    fn key_of(item: &HeapItem) -> i64 {
        match item {
            HeapItem::I64(v) => *v,
            HeapItem::F64(v) => v.to_bits() as i64,
            other => panic!("unexpected heap key form {other:?}"),
        }
    }

    fn heap_documented_matrix(id: u8) {
        let input = key_input(id);
        let truth = input.truth();

        for top_k in TOP_KS {
            let mut cms_heap =
                CMSHeap::<Vector2D<i64>, RegularPath>::new(HEAP_ROWS, HEAP_COLS, top_k);
            let mut cs_heap =
                CSHeap::<Vector2D<i64>, RegularPath>::new(HEAP_ROWS, HEAP_COLS, top_k);
            for key in &input.keys {
                let d = input.data(*key);
                cms_heap.insert(&d);
                cs_heap.insert(&d);
            }
            let context = format!(
                "{} top_k={top_k} rows={HEAP_ROWS} cols={HEAP_COLS}",
                input.context()
            );

            let kth_count = truth.top_k(top_k)[top_k - 1].1;
            for (label, items, cms) in [
                ("CMSHeap", cms_heap.heap().heap().to_vec(), true),
                ("CSHeap", cs_heap.heap().heap().to_vec(), false),
            ] {
                assert!(
                    items.len() <= top_k,
                    "{label} heap holds {} entries, past its {top_k} capacity. {context}",
                    items.len()
                );

                let mut consistency = Tally::default();
                let mut documented = Tally::default();
                let mut recall = 0usize;
                let mut heavy: Vec<(i64, i64)> = Vec::with_capacity(items.len());
                for it in &items {
                    let key = key_of(&it.key);
                    let d = input.data(key);
                    let est = if cms {
                        cms_heap.estimate(&d)
                    } else {
                        cs_heap.estimate(&d) as i64
                    };
                    consistency.record(it.count == est, || {
                        format!(
                            "key {key}: heap holds {} but the sketch estimates {est}",
                            it.count
                        )
                    });
                    let t = truth.get(key);
                    heavy.push((key, t));
                    if t >= kth_count {
                        recall += 1;
                    }
                    let rel = ((it.count - t) as f64 / t as f64).abs();
                    documented.record(rel <= HEAP_RELATIVE_ERROR, || {
                        format!(
                            "key {key}: heap count {} vs true {t} is {:.2}% off, past the \
                             documented {:.0}%",
                            it.count,
                            rel * 100.0,
                            HEAP_RELATIVE_ERROR * 100.0
                        )
                    });
                }
                // Each sketch against its own theorem, evaluated at the keys the
                // heap actually admitted.
                let total = truth.total() as f64;
                let probed = heavy.len();
                if cms {
                    let spec = CountMinSpec::new(HEAP_ROWS, HEAP_COLS);
                    let mut bound = Tally::default();
                    for (key, count) in &heavy {
                        let f = *count as f64;
                        let est = cms_heap.estimate(&input.data(*key)) as f64;
                        let simul = spec.simultaneous_bound(total, f, probed, SIMULTANEOUS_LEVEL);
                        bound.record(est >= f && est - f <= simul, || {
                            format!(
                                "key {key}: est {est} against true {f} leaves the one-sided \
                                 additive band of {simul:.1}"
                            )
                        });
                    }
                    bound.assert_none("CMSHeap / Count-Min bound at the heap's keys", &context);
                } else {
                    let spec = CountSketchSpec::new(HEAP_ROWS, HEAP_COLS);
                    let f2 = truth.f2();
                    let kappa = spec.simultaneous_kappa(probed, SIMULTANEOUS_LEVEL);
                    let mut bound = Tally::default();
                    for (key, count) in &heavy {
                        let f = *count as f64;
                        let est = cs_heap.estimate(&input.data(*key));
                        let residual_l2 = (f2 - f * f).max(0.0).sqrt();
                        let scale = spec.scale_at(kappa, residual_l2);
                        bound.record((est - f).abs() <= scale, || {
                            format!(
                                "key {key}: |{est:.1} - {f}| exceeds sqrt(kappa/w)*||f_-i||_2 \
                                 = {scale:.1} at kappa={kappa:.1}"
                            )
                        });
                    }
                    bound.assert_none("CSHeap / L2 bound at the heap's keys", &context);
                }

                consistency.assert_none(&format!("{label} heap/sketch consistency"), &context);
                documented.assert_none(
                    &format!("{label} / documented per-item relative error"),
                    &context,
                );
                assert!(
                    recall >= top_k - 1,
                    "{label} recovered only {recall}/{top_k} keys at or above the true \
                     k-th count ({kth_count}). {context}"
                );
            }
        }
    }

    macro_rules! documented_topk_matrix {
        ($($name:ident => $id:literal;)*) => {
            $(
                #[test]
                fn $name() {
                    heap_documented_matrix($id);
                }
            )*
        };
    }

    documented_topk_matrix! {
        heaps_on_input_3_hold_their_bounds => 3;
        heaps_on_input_4_hold_their_bounds => 4;
        heaps_on_input_5_hold_their_bounds => 5;
        heaps_on_input_6_hold_their_bounds => 6;
        heaps_on_input_9_hold_their_bounds => 9;
        heaps_on_input_10_hold_their_bounds => 10;
        heaps_on_input_11_hold_their_bounds => 11;
        heaps_on_input_12_hold_their_bounds => 12;
    }
}
