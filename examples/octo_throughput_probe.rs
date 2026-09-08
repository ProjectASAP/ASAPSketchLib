//! Release-only throughput probe for the OctoSketch pipeline on UnivMon.
//!
//! Measures the three stages separately so the bottleneck is visible:
//! a single-threaded `UnivMon::insert`, an Octo worker's per-insert work, and
//! the aggregator's per-delta work. The sustainable pipeline rate is the
//! aggregator's rate divided by how many deltas each insert produces, because
//! one aggregator serves every worker.
//!
//! ```text
//! cargo run --release --example octo_throughput_probe
//! ```

use std::time::Instant;

use asap_sketchlib::{
    DataInput, MAX_PROMASK, OctoAggregator, OctoPlan, OctoThreshold, OctoWorker, UNIVMON_PROMASK,
    UnivMon, UnivMonOctoAggregator, UnivMonOctoPlan, UnivMonOctoWorker, univmon_layer_threshold,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `n` Zipf(exponent) draws over `[0, domain)`; skew is what puts the hot keys
/// in the heavy-hitter heap, which is where the per-insert cost lives.
fn zipf(n: usize, domain: usize, exponent: f64, seed: u64) -> Vec<u64> {
    let mut cdf: Vec<f64> = (0..domain)
        .map(|i| 1.0 / (i as f64 + 1.0).powf(exponent))
        .collect();
    for i in 1..cdf.len() {
        cdf[i] += cdf[i - 1];
    }
    let total = cdf[domain - 1];
    for x in cdf.iter_mut() {
        *x /= total;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let u: f64 = rng.random();
            match cdf.binary_search_by(|p| p.partial_cmp(&u).unwrap()) {
                Ok(i) | Err(i) => (i as u64).min(domain as u64 - 1),
            }
        })
        .collect()
}

fn main() {
    let (heap, rows, cols, layers) = (64usize, 5usize, 1_024usize, 12usize);
    let n = 2_000_000usize;
    let keys = zipf(n, 4_096, 1.1, 13_001);
    let inputs: Vec<DataInput<'static>> = keys.iter().map(|k| DataInput::U64(*k)).collect();

    // Absolute rates below belong to this host. Record it, or the numbers are
    // not comparable with anyone else's - or with the same box under load.
    println!(
        "host: {} {} | {} logical cores | core pinning: {}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        // Probe from a throwaway thread: `set_for_current` pins whoever calls
        // it, and pinning the thread about to run the measurements would be a
        // side effect of reporting.
        std::thread::spawn(|| {
            match core_affinity::get_core_ids().and_then(|ids| ids.into_iter().next()) {
                Some(id) if core_affinity::set_for_current(id) => "available",
                Some(_) => "REJECTED by this platform (measurements are unpinned)",
                None => "no core ids reported",
            }
        })
        .join()
        .unwrap_or("could not probe")
    );
    println!(
        "profile: {}",
        if cfg!(debug_assertions) {
            "debug - rates are meaningless, rebuild with --release"
        } else {
            "release"
        }
    );

    let mut ideal = UnivMon::init_univmon(heap, rows, cols, layers);
    let started = Instant::now();
    for input in &inputs {
        ideal.insert(input, 1);
    }
    let single_rate = n as f64 / started.elapsed().as_secs_f64() / 1e6;
    let (ideal_card, ideal_entropy) = (ideal.calc_card(), ideal.calc_entropy());
    println!(
        "single-threaded UnivMon: {single_rate:.2} Mops/s  \
         (heap={heap} rows={rows} cols={cols} layers={layers}, n={n} Zipf 1.1 over 4096)"
    );
    println!(
        "{:<6} {:>12} {:>9} {:>12} {:>14} {:>12} {:>10}",
        "tau", "worker", "speedup", "deltas/ins", "agg Mdelta/s", "pipeline", "gap"
    );

    // A geometric sweep of the usable range. Powers of two are the natural
    // grid because `univmon_layer_threshold` halves tau per layer, so a base of
    // 2^k walks down to 1 exactly at layer k. 127 is MAX_PROMASK, the widest a
    // signed one-byte worker counter holds; 64 is UNIVMON_PROMASK.
    for tau in [1u32, 2, 4, 8, 16, 32, 64, 127] {
        let plan = UnivMonOctoPlan::with_threshold(rows, cols, layers, OctoThreshold::new(tau));
        // Preparation runs on the dispatching thread in the real pipeline, so
        // it is measured with the worker rather than folded into it.
        let payloads: Vec<_> = inputs.iter().map(|i| plan.prepare(i)).collect();
        let mut worker =
            UnivMonOctoWorker::with_threshold(0, rows, cols, layers, OctoThreshold::new(tau));
        let mut emitted = 0usize;
        let started = Instant::now();
        for payload in &payloads {
            worker.process(payload, &mut |_| emitted += 1);
        }
        let worker_rate = n as f64 / started.elapsed().as_secs_f64() / 1e6;

        let mut replay =
            UnivMonOctoWorker::with_threshold(0, rows, cols, layers, OctoThreshold::new(tau));
        let mut deltas = Vec::with_capacity(emitted);
        for payload in &payloads {
            replay.process(payload, &mut |d| deltas.push(d));
        }
        let mut aggregator = UnivMonOctoAggregator::new(heap, rows, cols, layers, tau);
        let started = Instant::now();
        for delta in deltas {
            aggregator.apply(delta);
        }
        let aggregator_rate = emitted as f64 / started.elapsed().as_secs_f64() / 1e6;

        let per_insert = emitted as f64 / n as f64;
        let parent = &aggregator.sketch;
        let gap = ((parent.calc_card() - ideal_card).abs() / ideal_card.abs().max(1.0)
            + (parent.calc_entropy() - ideal_entropy).abs() / ideal_entropy.abs().max(1e-9))
            / 2.0;
        let note = match tau {
            UNIVMON_PROMASK => "  <- UnivMon default",
            MAX_PROMASK => "  <- ceiling",
            _ => "",
        };
        println!(
            "{tau:<6} {worker_rate:>9.1} M/s {:>8.0}x {per_insert:>12.3} {aggregator_rate:>14.2} \
             {:>9.2} M/s {gap:>10.4}{note}",
            worker_rate / single_rate,
            aggregator_rate / per_insert,
        );
    }

    println!(
        "\nper-layer tau at base {MAX_PROMASK}: {:?}",
        (0..layers)
            .map(|l| univmon_layer_threshold(MAX_PROMASK, l))
            .collect::<Vec<_>>()
    );

    // The aggregator is the pipeline's ceiling, and the heap sets the
    // aggregator.
    println!("\naggregator rate vs heap capacity (tau=64):");
    for capacity in [4usize, 16, 64, 256] {
        let plan = UnivMonOctoPlan::with_threshold(rows, cols, layers, OctoThreshold::new(64));
        let mut worker =
            UnivMonOctoWorker::with_threshold(0, rows, cols, layers, OctoThreshold::new(64));
        let mut deltas = Vec::new();
        for input in &inputs {
            worker.process(&plan.prepare(input), &mut |d| deltas.push(d));
        }
        let count = deltas.len();
        let mut aggregator = UnivMonOctoAggregator::new(capacity, rows, cols, layers, 64);
        let started = Instant::now();
        for delta in deltas {
            aggregator.apply(delta);
        }
        println!(
            "  heap={capacity:<4} {:.2} Mdelta/s",
            count as f64 / started.elapsed().as_secs_f64() / 1e6
        );
    }
}
