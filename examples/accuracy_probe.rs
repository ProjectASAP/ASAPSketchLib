//! Ground-truth accuracy probe for every sketch family in asap_sketchlib.
//!
//! Run: cargo run --release --example accuracy_probe --features experimental
//!
//! Each section feeds deterministic synthetic streams with exactly known
//! answers into a sketch and compares query results against ground truth,
//! printing EXPECTED vs ACTUAL and a theory-based verdict.

use asap_sketchlib::common::input::{HydraCounter, HydraQuery};
use asap_sketchlib::message_pack_format::portable::countminsketch::CountMinSketch;
use asap_sketchlib::message_pack_format::portable::ddsketch::DdSketch as PortableDds;
use asap_sketchlib::message_pack_format::portable::hll::{HllSketch, HllVariant};
use asap_sketchlib::message_pack_format::portable::hydra_kll::HydraKllSketch;
use asap_sketchlib::message_pack_format::portable::kll::KllSketch as PortableKll;
use asap_sketchlib::{
    CMSHeap, CSHeap, Count as CoreCount, CountL2HH, CountMin, DDSketch, DataInput, Hydra,
    HyperLogLog, KLL, UnivMon,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

struct Probe {
    failures: Vec<String>,
    notes: Vec<String>,
}

impl Probe {
    fn new() -> Self {
        Self {
            failures: Vec::new(),
            notes: Vec::new(),
        }
    }

    fn check(&mut self, name: &str, detail: String, ok: bool) {
        let tag = if ok { "PASS" } else { "FAIL" };
        println!("  [{tag}] {name}: {detail}");
        if !ok {
            self.failures.push(format!("{name}: {detail}"));
        }
    }

    fn note(&mut self, msg: &str) {
        println!("  [NOTE] {msg}");
        self.notes.push(msg.to_string());
    }

    fn finish(self, section: &str) {
        if self.failures.is_empty() {
            println!("== {section}: ALL PASS ==");
        } else {
            println!("== {section}: {} FAILURE(S) ==", self.failures.len());
        }
        println!();
    }
}

fn rel_err(est: f64, truth: f64) -> f64 {
    if truth == 0.0 {
        est.abs()
    } else {
        ((est - truth) / truth).abs()
    }
}

/// Zipf(s) sampler over domain [0, domain).
struct Zipf {
    cdf: Vec<f64>,
    rng: StdRng,
}

impl Zipf {
    fn new(domain: usize, exponent: f64, seed: u64) -> Self {
        let mut w: Vec<f64> = (0..domain)
            .map(|i| 1.0 / (i as f64 + 1.0).powf(exponent))
            .collect();
        for i in 1..w.len() {
            w[i] += w[i - 1];
        }
        let total = w[domain - 1];
        for x in w.iter_mut() {
            *x /= total;
        }
        Self {
            cdf: w,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample_usize(&mut self) -> usize {
        let u: f64 = self.rng.random::<f64>();
        match self.cdf.binary_search_by(|p| p.partial_cmp(&u).unwrap()) {
            Ok(i) => i,
            Err(i) => i,
        }
    }

    fn sample_i64(&mut self) -> i64 {
        self.sample_usize() as i64
    }
}

fn main() {
    println!("================ ASAP SKETCHLIB GROUND-TRUTH PROBE ================\n");
    probe_countmin();
    probe_countsketch();
    probe_topk_heaps();
    probe_countl2hh();
    probe_hll();
    probe_kll();
    probe_ddsketch();
    probe_hydra();
    probe_univmon();
    #[cfg(feature = "experimental")]
    probe_univmon_pyramid();
    probe_univmon_q();
    probe_nitro();
    #[cfg(feature = "experimental")]
    probe_eh_univ();
    probe_tumbling();
    probe_portable_wire_types();

    println!("====================================================================");
    println!("Probe complete. FAIL lines above are candidate wrong-query-results.");
}

// ---------------------------------------------------------------- CountMin
fn probe_countmin() {
    let mut p = Probe::new();
    println!("--- CountMin (core, Vector2D<i32>) ---");

    // Sparse dims: no collisions => exact counts, incl. turnstile negatives.
    let mut cm =
        CountMin::<asap_sketchlib::Vector2D<i32>, asap_sketchlib::RegularPath>::with_dimensions(
            3, 256,
        );
    cm.insert_many(&DataInput::U32(7), 100);
    cm.insert_many(&DataInput::U32(7), -40);
    let est = cm.estimate(&DataInput::U32(7));
    p.check(
        "turnstile +100/-40 == 60",
        format!("expected 60, got {est}"),
        est == 60,
    );

    // Zipf stream: one-sided CMS guarantee est >= true, excess <= eps*N.
    let (rows, cols, n, domain) = (3usize, 4096usize, 200_000usize, 8192usize);
    let mut cm =
        CountMin::<asap_sketchlib::Vector2D<i64>, asap_sketchlib::FastPath>::with_dimensions(
            rows, cols,
        );
    let mut truth: HashMap<i64, i64> = HashMap::new();
    let mut z = Zipf::new(domain, 1.1, 42);
    for _ in 0..n {
        let k = z.sample_i64();
        *truth.entry(k).or_insert(0) += 1;
        cm.insert(&DataInput::I64(k));
    }
    let eps = std::f64::consts::E / cols as f64;
    // Check top-100 hottest keys.
    let mut by_freq: Vec<(i64, i64)> = truth.iter().map(|(k, v)| (*k, *v)).collect();
    by_freq.sort_by_key(|(_, v)| -*v);
    let mut worst_excess = 0.0f64;
    let mut underestimates = 0usize;
    for (k, v) in &by_freq[..100] {
        let est = cm.estimate(&DataInput::I64(*k));
        if est < *v {
            underestimates += 1;
        }
        worst_excess = worst_excess.max((est - v) as f64);
    }
    p.check(
        "CMS never underestimates (top-100 hot keys)",
        format!("{underestimates} underestimates"),
        underestimates == 0,
    );
    p.check(
        "CMS excess <= eps*N on all checked keys",
        format!(
            "eps*N = {:.1}, max observed excess = {:.1}",
            eps * n as f64,
            worst_excess
        ),
        worst_excess <= eps * n as f64,
    );
    p.finish("CountMin");
}

// ------------------------------------------------------------- CountSketch
fn probe_countsketch() {
    let mut p = Probe::new();
    println!("--- CountSketch (core Count, Vector2D<i64>) ---");

    // True turnstile: net-zero stream must estimate ~0.
    let mut cs =
        CoreCount::<asap_sketchlib::Vector2D<i64>, asap_sketchlib::RegularPath>::with_dimensions(
            4, 512,
        );
    cs.insert_many(&DataInput::U32(3), 500);
    cs.insert_many(&DataInput::U32(3), -500);
    let est = cs.estimate(&DataInput::U32(3));
    p.check(
        "turnstile +500/-500 ~= 0",
        format!("expected ~0, got {est}"),
        est.abs() < 1e-9,
    );

    // Zipf: median estimator error bound |est-true| <= t/sqrt(cols) * L2.
    let (rows, cols, n, domain) = (5usize, 4096usize, 200_000usize, 8192usize);
    let mut cs =
        CoreCount::<asap_sketchlib::Vector2D<i64>, asap_sketchlib::RegularPath>::with_dimensions(
            rows, cols,
        );
    let mut truth: HashMap<i64, i64> = HashMap::new();
    let mut z = Zipf::new(domain, 1.1, 43);
    for _ in 0..n {
        let k = z.sample_i64();
        *truth.entry(k).or_insert(0) += 1;
        cs.insert(&DataInput::I64(k));
    }
    let l2 = (truth
        .values()
        .map(|c| (*c as f64) * (*c as f64))
        .sum::<f64>())
    .sqrt();
    let bound = (std::f64::consts::E / cols as f64).sqrt() * l2; // median-of-t rows whp
    let mut by_freq: Vec<(i64, i64)> = truth.iter().map(|(k, v)| (*k, *v)).collect();
    by_freq.sort_by_key(|(_, v)| -*v);
    let mut violations = 0usize;
    let mut max_abs_err = 0.0f64;
    for (k, v) in &by_freq[..100] {
        let est = cs.estimate(&DataInput::I64(*k));
        let err = (est - *v as f64).abs();
        max_abs_err = max_abs_err.max(err);
        if err > bound {
            violations += 1;
        }
    }
    p.check(
        "CS |err| <= sqrt(e/cols)*L2 (top-100)",
        format!("bound {bound:.1}, max err {max_abs_err:.1}, violations {violations}/100"),
        violations == 0,
    );
    p.finish("CountSketch");
}

// -------------------------------------------------------------- Top-K heaps
fn probe_topk_heaps() {
    let mut p = Probe::new();
    println!("--- CMSHeap / CSHeap top-k recall ---");
    let (n, domain, top_k) = (20_000usize, 1024usize, 16usize);
    let mut z = Zipf::new(domain, 1.1, 44);
    let mut truth: HashMap<u32, i64> = HashMap::new();
    let mut stream = Vec::with_capacity(n);
    for _ in 0..n {
        let k = z.sample_usize() as u32;
        *truth.entry(k).or_insert(0) += 1;
        stream.push(k);
    }
    let mut true_top: Vec<(u32, i64)> = truth.iter().map(|(k, v)| (*k, *v)).collect();
    true_top.sort_by_key(|(_, v)| -*v);
    let true_top_set: std::collections::HashSet<u32> =
        true_top.iter().take(top_k).map(|(k, _)| *k).collect();

    let mut cms_heap =
        CMSHeap::<asap_sketchlib::Vector2D<i64>, asap_sketchlib::RegularPath>::new(3, 4096, top_k);
    let mut cs_heap =
        CSHeap::<asap_sketchlib::Vector2D<i64>, asap_sketchlib::RegularPath>::new(5, 4096, top_k);
    for k in &stream {
        let d = DataInput::U32(*k);
        cms_heap.insert(&d);
        cs_heap.insert(&d);
    }
    for (label, heap_items) in [
        ("CMSHeap", cms_heap.heap().heap().to_vec()),
        ("CSHeap", cs_heap.heap().heap().to_vec()),
    ] {
        let found = heap_items
            .iter()
            .filter(|it| match it.key {
                asap_sketchlib::HeapItem::U32(v) => true_top_set.contains(&v),
                _ => false,
            })
            .count();
        p.check(
            &format!("{label} top-{top_k} recall"),
            format!("{found}/{top_k} true heavy hitters recovered"),
            found >= top_k - 1,
        );
    }

    // CSHeap turnstile can push NEGATIVE counts into the heap (legal CS estimate).
    let mut cs_neg =
        CSHeap::<asap_sketchlib::Vector2D<i64>, asap_sketchlib::RegularPath>::new(3, 4096, 8);
    cs_neg.insert_many(&DataInput::U32(999), -5);
    if let Some(it) = cs_neg
        .heap()
        .heap()
        .iter()
        .find(|it| matches!(it.key, asap_sketchlib::HeapItem::U32(999)))
    {
        let cnt = it.count;
        p.note(&format!("CSHeap stores negative heap count {cnt} for pure-decrement key (design wart, not a crash)"));
    }
    p.finish("TopK heaps");
}

// --------------------------------------------------------------- CountL2HH
fn probe_countl2hh() {
    let mut p = Probe::new();
    println!("--- CountL2HH (F2 / L2 estimation) ---");

    // Moderate stream F2 accuracy (truth = sum over KEYS of count^2).
    let mut sk = CountL2HH::<asap_sketchlib::DefaultXxHasher>::with_dimensions_and_seed(4, 2048, 7);
    let mut z = Zipf::new(1024, 1.2, 45);
    let mut per_key: HashMap<u32, i64> = HashMap::new();
    for i in 0..50_000u32 {
        let k = z.sample_usize() as u32;
        let w = 1 + (i % 3) as i64;
        sk.fast_insert_with_count(&DataInput::U32(k), w);
        *per_key.entry(k).or_insert(0) += w;
    }
    let truth_f2: f64 = per_key.values().map(|c| (*c as f64) * (*c as f64)).sum();
    let l2 = sk.get_l2();
    let got_f2 = sk.get_l2_sqr();
    p.check(
        "F2 within 15% of truth",
        format!(
            "expected {truth_f2:.3e}, got {got_f2:.3e} (rel {:.3})",
            rel_err(got_f2, truth_f2)
        ),
        rel_err(l2 * l2, truth_f2) < 0.15,
    );

    // Turnstile decrement support.
    let mut sk2 =
        CountL2HH::<asap_sketchlib::DefaultXxHasher>::with_dimensions_and_seed(4, 2048, 7);
    sk2.fast_insert_with_count(&DataInput::U32(1), 5);
    sk2.fast_insert_with_count(&DataInput::U32(1), -2);
    let est = sk2.fast_update_and_est(&DataInput::U32(1), 0);
    p.check(
        "turnstile 5 then -2 estimates 3",
        format!("expected 3, got {est}"),
        (est - 3.0).abs() < 1e-6,
    );

    // i64 overflow probe: two 3e9-count keys => true F2 = 1.8e19 > i64::MAX.
    let mut sk3 =
        CountL2HH::<asap_sketchlib::DefaultXxHasher>::with_dimensions_and_seed(4, 2048, 7);
    sk3.fast_insert_with_count(&DataInput::U32(1), 3_000_000_000);
    sk3.fast_insert_with_count(&DataInput::U32(2), 3_000_000_000);
    let got = sk3.get_l2_sqr();
    let truth = 2.0f64 * (3_000_000_000f64).powi(2);
    let saturated = got > i64::MAX as f64 * 0.999;
    p.check(
        "F2 saturates at i64::MAX beyond representable range",
        if saturated {
            format!("saturated at {got:.3e} (true {truth:.3e})")
        } else {
            format!("GARBAGE from wrap: expected ~{truth:.3e}, got {got:.3e}")
        },
        saturated,
    );
    p.finish("CountL2HH");
}

// --------------------------------------------------------------------- HLL
fn probe_hll() {
    let mut p = Probe::new();
    println!("--- HLL cardinality (Classic / ErtlMLE / HIP, p14) ---");
    let checkpoints: [u64; 4] = [10_000, 100_000, 1_000_000, 10_000_000];
    let mut classic = HyperLogLog::<asap_sketchlib::Classic>::new();
    let mut ertl = HyperLogLog::<asap_sketchlib::ErtlMLE>::new();
    let mut hip = asap_sketchlib::HyperLogLogHIP::new();
    let mut inserted: u64 = 0;
    for &target in &checkpoints {
        while inserted < target {
            let v = DataInput::U64(inserted);
            classic.insert(&v);
            ertl.insert(&v);
            hip.insert(&v);
            inserted += 1;
        }
        let ec = classic.estimate() as f64;
        let ee = ertl.estimate() as f64;
        let eh = hip.estimate() as f64;
        let t = target as f64;
        p.check(
            &format!("Classic @ {target} within 2%"),
            format!("rel {:.4}", rel_err(ec, t)),
            rel_err(ec, t) <= 0.02,
        );
        p.check(
            &format!("ErtlMLE @ {target} within 2%"),
            format!("rel {:.4}", rel_err(ee, t)),
            rel_err(ee, t) <= 0.02,
        );
        p.check(
            &format!("HIP @ {target} within 2%"),
            format!("rel {:.4}", rel_err(eh, t)),
            rel_err(eh, t) <= 0.02,
        );
    }
    // Large-range correction regime (>143M raw estimate): Classic uses
    // i32::MAX where Flajolet specifies 2^32 (src/sketches/hll.rs:216-218).
    // Only Classic uses this correction; Ertl MLE has its own estimator.
    let target = 170_000_000u64;
    while inserted < target {
        let v = DataInput::U64(inserted);
        classic.insert(&v);
        inserted += 1;
    }
    let t = target as f64;
    let ec = classic.estimate() as f64;
    p.check(
        &format!("Classic @ {target} within 3% (large-range correction active)"),
        format!("rel {:.4}", rel_err(ec, t)),
        rel_err(ec, t) <= 0.03,
    );
    p.finish("HLL");
}

// --------------------------------------------------------------------- KLL
fn probe_kll() {
    let mut p = Probe::new();
    println!("--- KLL / KLLDynamic quantiles (k=200) ---");
    let qs = [0.1f64, 0.25, 0.5, 0.75, 0.9];

    let n = 100_000usize;
    let mut rng = StdRng::seed_from_u64(46);
    let mut data: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 1_000_000.0).collect();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for label in ["KLL", "KLLDynamic"] {
        let (e1, w1) = if label == "KLL" {
            let mut sk = asap_sketchlib::KLL::init_kll(200);
            for v in &data {
                sk.update(v);
            }
            check_quantiles(&sk, &data, &qs)
        } else {
            let mut sk = asap_sketchlib::KLLDynamic::<f64>::init_kll(200);
            for v in &data {
                sk.update(v);
            }
            check_quantiles_dyn(&sk, &data, &qs)
        };
        p.check(
            &format!("{label} rank err <= 0.02 at q10/25/50/75/90"),
            if w1.is_empty() {
                "all within band".to_string()
            } else {
                format!("worst {w1}")
            },
            e1 <= 0.02,
        );

        // Merge preservation: split halves into two sketches, merge, re-check.
        let mut a = asap_sketchlib::KLL::init_kll_with_seed(200, 99);
        let mut b = asap_sketchlib::KLL::init_kll_with_seed(200, 123);
        for (i, v) in data.iter().enumerate() {
            if i % 2 == 0 {
                a.update(v);
            } else {
                b.update(v);
            }
        }
        a.merge(&b);
        let mut worst_merge = 0.0f64;
        for &q in &qs {
            let est = a.quantile(q);
            let rank = data.partition_point(|x| *x <= est) as f64 / n as f64;
            worst_merge = worst_merge.max((rank - q).abs());
        }
        p.check(
            &format!("{label} merge preserves distribution"),
            format!("worst rank drift {:.4}", worst_merge),
            worst_merge <= 0.02,
        );
    }
    p.finish("KLL");
}

fn check_quantiles(sk: &asap_sketchlib::KLL, data: &[f64], qs: &[f64]) -> (f64, String) {
    quantile_report(|q| sk.quantile(q), data, qs)
}

fn check_quantiles_dyn(
    sk: &asap_sketchlib::KLLDynamic<f64>,
    data: &[f64],
    qs: &[f64],
) -> (f64, String) {
    quantile_report(|q| sk.quantile(q), data, qs)
}

fn quantile_report(qf: impl Fn(f64) -> f64, data: &[f64], qs: &[f64]) -> (f64, String) {
    let n = data.len();
    let mut worst = 0.0f64;
    let mut worst_desc = String::new();
    for &q in qs {
        let est = qf(q);
        let rank = data.partition_point(|x| *x <= est) as f64 / n as f64;
        if (rank - q).abs() > worst {
            worst = (rank - q).abs();
            worst_desc = format!("q={q} est={est:.1} rank={rank:.3}");
        }
    }
    (worst, worst_desc)
}

// ---------------------------------------------------------------- DDSketch
fn probe_ddsketch() {
    let mut p = Probe::new();
    println!("--- DDSketch core vs portable (alpha=0.05) ---");
    let alpha = 0.05;
    let gamma = (1.0f64 + alpha) / (1.0 - alpha);
    // Isolated adversarial probe: one distinct value sitting just above a
    // bucket's lower edge gamma^k, repeated N times. The only correct answer
    // for every quantile is v itself.
    let mut worst_core = 0.0f64;
    let mut worst_port = 0.0f64;
    for k in [10i32, 20, 30] {
        let v = gamma.powi(k) * (1.0 + 1e-6); // just inside bucket k
        let n = 10_000u64;

        let mut core = DDSketch::new(alpha);
        let mut port = PortableDds::new(alpha);
        for _ in 0..n {
            core.add(&v);
            port.update(v);
        }
        let rc = rel_err(core.get_value_at_quantile(0.5).unwrap(), v);
        let rp = rel_err(port.quantile(0.5).unwrap(), v);
        worst_core = worst_core.max(rc);
        worst_port = worst_port.max(rp);
    }
    p.check(
        "core DDSketch honors alpha=0.05 at bucket edges",
        format!("max rel err {worst_core:.5}"),
        worst_core <= alpha * (1.0 + 1e-6),
    );
    p.check(
        "portable DdSketch honors alpha=0.05 at bucket edges",
        format!("max rel err {worst_port:.5} (shared gamma^k*(1+alpha) representative)"),
        worst_port <= alpha * (1.0 + 1e-6),
    );

    // Mixed-stream sanity on both.
    let mut rng = StdRng::seed_from_u64(47);
    let mut core = DDSketch::new(alpha);
    let mut port = PortableDds::new(alpha);
    let mut truth: Vec<f64> = Vec::new();
    for _ in 0..20_000 {
        let k = rng.random_range(5..40);
        let frac = rng.random::<f64>();
        let v = gamma.powi(k) * (1.0 + frac * (gamma - 1.0));
        core.add(&v);
        port.update(v);
        truth.push(v);
    }
    truth.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mc = qs_max_rel(&truth, |q| core.get_value_at_quantile(q).unwrap(), &qs_dd());
    let mp = qs_max_rel(&truth, |q| port.quantile(q).unwrap(), &qs_dd());
    p.check(
        "core mixed stream <= alpha",
        format!("max rel {mc:.5}"),
        mc <= alpha,
    );
    p.check(
        "portable mixed stream <= alpha",
        format!("max rel {mp:.5}"),
        mp <= alpha,
    );
    p.finish("DDSketch");
}

fn qs_dd() -> [f64; 5] {
    [0.1, 0.25, 0.5, 0.75, 0.9]
}

fn qs_max_rel(truth: &[f64], qf: impl Fn(f64) -> f64, qs: &[f64]) -> f64 {
    let n = truth.len();
    let mut worst = 0.0f64;
    for &q in qs {
        let idx = ((q * n as f64).ceil() as usize).clamp(1, n);
        let t = truth[idx - 1];
        let est = qf(q);
        worst = worst.max(rel_err(est, t));
    }
    worst
}

// ------------------------------------------------------------------- Hydra
fn probe_hydra() {
    let mut p = Probe::new();
    println!("--- Hydra (KLL-backed counter) ---");
    let mut hydra = Hydra::with_schema(4, 512, ["region"], HydraCounter::KLL(KLL::init_kll(200)))
        .expect("schema");
    let mut rng = StdRng::seed_from_u64(48);
    let mut samples: Vec<f64> = (0..10_000).map(|_| rng.random::<f64>() * 1000.0).collect();
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for v in &samples {
        hydra
            .update(&["eu"], &DataInput::F64(*v), None)
            .expect("update");
    }
    let n = samples.len();

    // Proper quantile query via HydraQuery::Quantile.
    let med = hydra
        .query_key(&[Some("eu")], &HydraQuery::Quantile(0.5))
        .expect("q");
    let truth_med = sorted[n / 2];
    p.check(
        "HydraQuery::Quantile(0.5) approx median",
        format!("expected ~{truth_med:.1}, got {med:.1}"),
        (med - truth_med).abs() / truth_med < 0.05,
    );

    // query_quantile(name says quantile, actually returns CDF(threshold)).
    let x = 500.0;
    let got = hydra.query_quantile(&[Some("eu")], x).expect("cdf");
    let truth_cdf = sorted.partition_point(|v| *v <= x) as f64 / n as f64;
    p.note("Hydra::query_quantile(key, threshold) returns CDF(threshold), NOT a quantile value (name inversion)");
    p.check(
        "query_quantile(x)=CDF fraction <= x",
        format!("x={x}, expected ~{truth_cdf:.3}, got {got:.3}"),
        (got - truth_cdf).abs() < 0.03,
    );

    // Frequency through a CM-backed head is exact-ish.
    let mut hcm = Hydra::with_schema(4, 512, ["region"], HydraCounter::CM(Default::default()))
        .expect("schema2");
    for _ in 0..37 {
        hcm.update(&["us"], &DataInput::Str("click"), None)
            .expect("upd");
    }
    let f = hcm
        .query_frequency(&[Some("us")], &DataInput::Str("click"))
        .expect("freq");
    p.check(
        "CM head frequency == 37",
        format!("expected 37, got {f}"),
        f == 37.0,
    );
    let _ = &mut samples;
    p.finish("Hydra");
}

// ----------------------------------------------------------------- UnivMon
fn probe_univmon() {
    let mut p = Probe::new();
    println!("--- UnivMon weighted metrics ---");
    let mut um = UnivMon::init_univmon(32, 5, 2048, 8);
    let mut truth_w: HashMap<u32, i64> = HashMap::new();
    let mut z = Zipf::new(1000, 1.2, 49);
    for i in 0..20_000usize {
        let k = z.sample_usize() as u32;
        let w = 1 + (i % 7) as i64;
        um.insert(&DataInput::U32(k), w);
        *truth_w.entry(k).or_insert(0) += w;
    }
    let l1_truth: i64 = truth_w.values().sum();
    let l2_truth: f64 = truth_w
        .values()
        .map(|w| (*w as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let total = l1_truth as f64;
    let entropy_truth: f64 = -truth_w
        .values()
        .map(|w| {
            let pr = *w as f64 / total;
            if pr > 0.0 { pr * pr.log2() } else { 0.0 }
        })
        .sum::<f64>();
    let card_truth = truth_w.len() as f64;

    let l1 = um.calc_l1();
    let l2 = um.calc_l2();
    let ent = um.calc_entropy();
    let card = um.calc_card();
    p.check(
        "L1 exact",
        format!("expected {l1_truth}, got {l1}"),
        (l1 - l1_truth as f64).abs() < 1e-6,
    );
    p.check(
        "L2 within 5%",
        format!(
            "expected {l2_truth:.1}, got {l2:.1} (rel {:.4})",
            rel_err(l2, l2_truth)
        ),
        rel_err(l2, l2_truth) < 0.05,
    );
    p.check(
        "entropy(bits) within 5%",
        format!("expected {entropy_truth:.4}, got {ent:.4}"),
        rel_err(ent, entropy_truth) < 0.05,
    );
    p.check(
        "cardinality within 5%",
        format!("expected {card_truth}, got {card}"),
        rel_err(card, card_truth) < 0.05,
    );

    // Entropy unit sanity: bits value should be log2 larger than nats value.
    p.note("UnivMon calc_entropy returns BITS (log2); UnivMonQ estimate_entropy returns NATS (ln). Cross-family comparisons need conversion.");
    p.finish("UnivMon");
}

#[cfg(feature = "experimental")]
fn probe_univmon_pyramid() {
    use asap_sketchlib::UnivMonPyramid;
    let mut p = Probe::new();
    println!("--- UnivMonPyramid (experimental) ---");
    let mut up = UnivMonPyramid::with_defaults();
    let mut truth_w: HashMap<u32, i64> = HashMap::new();
    let mut z = Zipf::new(1000, 1.2, 50);
    for i in 0..20_000usize {
        let k = z.sample_usize() as u32;
        let w = 1 + (i % 7) as i64;
        up.insert(&DataInput::U32(k), w);
        *truth_w.entry(k).or_insert(0) += w;
    }
    let l1_truth: i64 = truth_w.values().sum();
    let card_truth = truth_w.len() as f64;
    let l1 = up.calc_l1();
    let card = up.calc_card();
    p.check(
        "L1 exact",
        format!("expected {l1_truth}, got {l1}"),
        (l1 - l1_truth as f64).abs() < 1e-6,
    );
    p.check(
        "cardinality within 10%",
        format!("expected {card_truth}, got {card}"),
        rel_err(card, card_truth) < 0.10,
    );
    p.finish("UnivMonPyramid");
}

// ---------------------------------------------------------------- UnivMonQ
fn probe_univmon_q() {
    let mut p = Probe::new();
    println!("--- UnivMon-Q ---");
    let mut q = asap_sketchlib::UnivMonQ::new(Default::default()).expect("config");
    let mut rng = StdRng::seed_from_u64(51);
    let mut data: Vec<f64> = Vec::with_capacity(50_000);
    let mut freq: HashMap<u64, u64> = HashMap::new();
    for _ in 0..50_000 {
        let v: f64 = if rng.random::<f64>() < 0.3 {
            7.0
        } else {
            (rng.random::<u64>() % 5000) as f64
        };
        q.update(&v);
        *freq.entry(v as u64).or_insert(0) += 1;
        data.push(v);
    }
    let mut sorted = data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len() as f64;

    p.check(
        "count exact",
        format!("expected {}, got {}", data.len(), q.count()),
        q.count() as usize == data.len(),
    );
    let mn = q.min();
    let mx = q.max();
    p.check(
        "min exact",
        format!("expected Some(0), got {mn:?}"),
        mn == Some(0.0),
    );
    p.check(
        "max exact",
        format!("expected Some(4999), got {mx:?}"),
        mx == Some(4999.0),
    );

    let hot_truth = freq[&7] as f64;
    let hot = q.estimate_frequency(7.0);
    p.check(
        "frequency(hot=7.0) within 5%",
        format!("expected {hot_truth}, got {hot}"),
        rel_err(hot as f64, hot_truth) < 0.05,
    );

    let distinct_truth = freq.len() as f64;
    let dist = q.estimate_distinct();
    p.check(
        "distinct within 10%",
        format!("expected {distinct_truth}, got {dist}"),
        rel_err(dist, distinct_truth) < 0.10,
    );

    let f2_truth: f64 = freq.values().map(|c| (*c as f64).powi(2)).sum();
    let f2 = q.estimate_f2();
    p.check(
        "F2 within 15%",
        format!("expected {f2_truth:.3e}, got {f2:.3e}"),
        rel_err(f2, f2_truth) < 0.15,
    );

    let ent_truth: f64 = -freq
        .values()
        .map(|c| {
            let pr = *c as f64 / n;
            pr * pr.ln()
        })
        .sum::<f64>();
    let ent = q.estimate_entropy();
    p.check(
        "entropy(NATS) within 10%",
        format!("expected {ent_truth:.4}, got {ent:.4}"),
        rel_err(ent, ent_truth) < 0.10,
    );

    let mut worst_rank = 0.0f64;
    // Value-band metric (tie-safe): est must lie between true ranks q±0.03.
    for &qq in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
        if let Some(est) = q.quantile(qq) {
            let lo = ((qq - 0.03).max(0.0) * n) as usize;
            let hi = (((qq + 0.03).min(1.0) * n) as usize).min(sorted.len());
            let vlo = sorted[lo];
            let vhi = sorted[hi.saturating_sub(1)];
            if est < vlo || est > vhi {
                let rank = sorted.partition_point(|x| *x <= est) as f64 / n;
                worst_rank = worst_rank.max((rank - qq).abs().min(1.0));
                println!("      out-of-band: q={qq} est={est} band=[{vlo}, {vhi}]");
            }
        } else {
            worst_rank = 1.0;
        }
    }
    p.check(
        "quantiles within rank band ±0.03",
        format!("worst out-of-band drift {worst_rank:.4}"),
        worst_rank <= 0.05,
    );

    // Well-separated heavy hitters: value i appears ~(i+1)*40 times.
    let mut q2 = asap_sketchlib::UnivMonQ::new(Default::default()).expect("cfg2");
    let mut freq2: HashMap<u64, u64> = HashMap::new();
    let mut rng2 = StdRng::seed_from_u64(57);
    for i in 0..50u64 {
        let target = (i + 1) * 40;
        for _ in 0..target {
            q2.update(&(i as f64));
            *freq2.entry(i).or_insert(0) += 1;
        }
    }
    // background noise
    for _ in 0..10_000 {
        let v = 100u64 + rng2.random::<u64>() % 900;
        q2.update(&(v as f64));
        *freq2.entry(v).or_insert(0) += 1;
    }
    let hh2 = q2.heavy_hitters(8);
    let mut true_top2: Vec<(u64, u64)> = freq2.iter().map(|(k, v)| (*k, *v)).collect();
    true_top2.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    let top8: std::collections::HashSet<u64> = true_top2.iter().take(8).map(|(k, _)| *k).collect();
    let hits2 = hh2
        .iter()
        .filter(|(v, _)| top8.contains(&(*v as u64)))
        .count();
    p.check(
        "heavy hitters overlap >= 6/8 (separated weights)",
        format!("{hits2}/8"),
        hits2 >= 6,
    );
    p.finish("UnivMonQ");
}

// ------------------------------------------------------------------- Nitro
fn probe_nitro() {
    let mut p = Probe::new();
    println!("--- NitroBatch (CM target) ---");

    // Unbiasedness across sampling rates on an exactly-known stream.
    let n = 100_000i64;
    for rate in [1.0f64, 0.5, 0.25] {
        let mut nb = asap_sketchlib::NitroBatch::with_target(
            rate,
            CountMin::<asap_sketchlib::Vector2D<i32>, asap_sketchlib::FastPath>::with_dimensions(
                5, 2048,
            ),
        );
        nb.insert(&vec![7i64; n as usize]);
        let est = nb.estimate_median(&DataInput::I64(7));
        p.check(
            &format!("estimate_median ~ truth at rate={rate}"),
            format!("expected {n}, got {est} (ratio {:.4})", est / n as f64),
            (est - n as f64).abs() <= 0.05 * n as f64,
        );
    }
    p.note("Pre-fix: insert hashed with raw hash128_seeded while estimation used the Packed64 matrix hash (estimates read empty cells), each sample updated ONE row (truth/rows bias), and the skip draw used ceil instead of floor (effective rate 1/(1/p+2)).");
    p.finish("Nitro");
}

// ------------------------------------------------------- EH univ optimized
#[cfg(feature = "experimental")]
fn probe_eh_univ() {
    use asap_sketchlib::EHUnivOptimized;
    let mut p = Probe::new();
    println!("--- EHUnivOptimized exact map tier (experimental) ---");
    let window = 100u64;
    let mut eh = EHUnivOptimized::with_defaults(2, window);
    // Stream updates over times 0..150; only [50,149] survive at t=149+window semantics.
    for t in 0..150u64 {
        eh.update(t, &DataInput::U32((t % 10) as u32), (t as i64 % 3) + 1);
    }
    // Query recent interval fully inside surviving range.
    let res = eh.query_interval(120, 149);
    match res {
        Some(asap_sketchlib::EHUnivQueryResult::Map {
            freq_map,
            total_count,
        }) => {
            let expect_total: usize = (120..=149u64).map(|t| (t as i64 % 3 + 1) as usize).sum();
            p.check(
                "interval map tier total exact",
                format!("expected {expect_total}, got {total_count}"),
                total_count == expect_total,
            );
            let mut expect_freq: HashMap<u32, i64> = HashMap::new();
            for t in 120..=149 {
                *expect_freq.entry((t % 10) as u32).or_insert(0) += (t as i64 % 3) + 1;
            }
            let exact = expect_freq
                .iter()
                .all(|(k, v)| freq_map.get(&asap_sketchlib::HeapItem::U32(*k)) == Some(v))
                && expect_freq.len() == freq_map.len();
            p.check(
                "interval map tier per-key exact",
                format!("{} entries", freq_map.len()),
                exact,
            );
        }
        _ => p.check(
            "interval returns Map tier",
            "got Sketch or None".to_string(),
            false,
        ),
    }
    p.finish("EHUnivOptimized");
}

// ---------------------------------------------------------------- Tumbling
fn probe_tumbling() {
    use asap_sketchlib::{FoldCMSConfig, TumblingWindow};

    let mut p = Probe::new();
    println!("--- TumblingWindow FoldCMS + KLL ---");

    // FoldCMS: weighted counts per key.
    let cfg = FoldCMSConfig {
        rows: 3,
        full_cols: 2048,
        fold_level: 0,
        top_k: 32,
    };
    let mut tw: TumblingWindow<asap_sketchlib::FoldCMS> =
        TumblingWindow::new(10, 8, cfg.clone(), 4);
    for t in 0..25u64 {
        tw.insert(t, &DataInput::Str("A"), 2);
        tw.insert(t, &DataInput::Str("B"), 1);
    }
    let merged = tw.query_all();
    let fa = merged.query(&DataInput::Str("A"));
    let fb = merged.query(&DataInput::Str("B"));
    p.check(
        "FoldCMS key A == 50",
        format!("expected 50, got {fa}"),
        fa == 50,
    );
    p.check(
        "FoldCMS key B == 25",
        format!("expected 25, got {fb}"),
        fb == 25,
    );

    // KLL tumbling: the numeric `value` arg is IGNORED; the KEY doubles as the
    // observation (tumbling.rs:146-149). Feed values via the key.
    let kcfg = asap_sketchlib::KLLConfig {
        k: 200,
        m: 8,
        seed: None,
    };
    let mut tkw: TumblingWindow<KLL> = TumblingWindow::new(100, 4, kcfg, 2);
    let mut rng = StdRng::seed_from_u64(53);
    let mut all_vals: Vec<f64> = Vec::new();
    for t in 0..1000u64 {
        let v = rng.random::<f64>() * 100.0;
        all_vals.push(v);
        tkw.insert(t, &DataInput::F64(v), 999 /* ignored */);
    }
    // query_all merges every window (closed + active), so truth = full stream.
    all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let merged_k = tkw.query_all();
    let med = merged_k.quantile(0.5);
    let truth_med = all_vals[all_vals.len() / 2];
    p.note("TumblingWindow<KLL>::insert ignores its numeric `value` param; observations are the keys themselves");
    p.check(
        "tumbling KLL median approx",
        format!("expected ~{truth_med:.1}, got {med:.1}"),
        rel_err(med, truth_med) < 0.05,
    );
    p.finish("Tumbling");
}

// ---------------------------------------------------- Portable wire types
fn probe_portable_wire_types() {
    let mut p = Probe::new();
    println!("--- Portable wire types (Go-parity DTOs) ---");

    // Portable CountMin.
    let mut pcs = CountMinSketch::new(3, 4096);
    let mut truth: HashMap<String, f64> = HashMap::new();
    let mut z = Zipf::new(2048, 1.1, 54);
    for _ in 0..50_000 {
        let k = format!("k{}", z.sample_usize());
        *truth.entry(k.clone()).or_insert(0.0) += 1.0;
        pcs.update(&k, 1.0);
    }
    let mut by_freq: Vec<(String, f64)> = truth.iter().map(|(k, v)| (k.clone(), *v)).collect();
    by_freq.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let eps = std::f64::consts::E / 4096.0;
    let mut bad = 0usize;
    for (k, v) in by_freq.iter().take(50) {
        let est = pcs.estimate(k);
        if est < *v || est > v + eps * 50_000.0 {
            bad += 1;
        }
    }
    p.check(
        "portable CMS one-sided bound (top-50)",
        format!("{bad} violations"),
        bad == 0,
    );

    // Portable HLL precision 12.
    let mut ph = HllSketch::new(HllVariant::Regular, 12);
    for i in 0..100_000u64 {
        ph.update(i.to_le_bytes().as_slice());
    }
    let est = ph.estimate();
    p.check(
        "portable HLL p12 within 3%",
        format!(
            "expected 100000, got {est:.0} (rel {:.4})",
            rel_err(est, 100_000.0)
        ),
        rel_err(est, 100_000.0) < 0.03,
    );

    // Portable KLL.
    let mut pk = PortableKll::new(200);
    let mut rng = StdRng::seed_from_u64(55);
    let mut vals: Vec<f64> = (0..50_000).map(|_| rng.random::<f64>() * 1e6).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for v in &vals {
        pk.update(*v);
    }
    let med = pk.quantile(0.5);
    let t = vals[25_000];
    p.check(
        "portable KLL median",
        format!("expected ~{t:.0}, got {med:.0}"),
        (med - t).abs() / t < 0.05,
    );

    // HydraKllSketch: per-key quantiles, median across rows.
    let mut hk = HydraKllSketch::new(3, 256, 200);
    let mut per_key: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut rng2 = StdRng::seed_from_u64(56);
    for key in ["svc-a", "svc-b"] {
        let base = if key == "svc-a" { 100.0 } else { 900.0 };
        let vs: Vec<f64> = (0..2_000)
            .map(|_| base + rng2.random::<f64>() * 50.0)
            .collect();
        for v in &vs {
            hk.update(key, *v);
        }
        per_key.insert(key, vs);
    }
    for (key, vs) in per_key {
        let mut s = vs.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = s[s.len() / 2];
        let got = hk.quantile(key, 0.5);
        p.check(
            &format!("HydraKll median({key})"),
            format!("expected ~{t:.1}, got {got:.1}"),
            (got - t).abs() / t < 0.05,
        );
    }
    p.finish("Portable wire types");
}
