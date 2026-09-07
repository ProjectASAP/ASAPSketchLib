//! Observational probe for when a key reaches the `*TopK*` aggregator's heap.
//!
//! A `*TopK*` worker keeps no flow-key storage, so a key enters the parent's
//! heap only when an increment it caused takes some row to τ on its worker.
//! That is the key's τ-th occurrence only while its cells are collision-free;
//! collisions move it either way. These two sweeps print the heap size that
//! results, and neither has a bound worth asserting — that is why they are a
//! probe and not a test.
//!
//! ```text
//! cargo run --release --features octo-runtime --example octo_topk_promotion_probe
//! ```

#[cfg(feature = "octo-runtime")]
fn main() {
    use asap_sketchlib::{
        CM_PROMASK, CmTopKOctoAggregator, CmTopKOctoPlan, DataInput, OctoConfig, run_octo,
    };

    fn config(num_workers: usize) -> OctoConfig {
        OctoConfig {
            num_workers,
            // A host may have fewer cores than the widest configuration here.
            pin_cores: false,
            queue_capacity: 8192,
            ..OctoConfig::default()
        }
    }

    let tau = CM_PROMASK;

    // Interleaved order, tight columns: can a key need MORE than tau?
    println!("== interleaved arrival, one worker ==");
    for &(rows, cols, nkeys) in &[(3usize, 16usize, 20u64), (3, 8, 20), (1, 32, 20), (3, 4, 8)] {
        for off in [0u64, 1000, 7777] {
            let mut inputs: Vec<DataInput<'_>> = Vec::new();
            for _ in 0..tau {
                for key in off..off + nkeys {
                    inputs.push(DataInput::U64(key));
                }
            }
            let sk = run_octo(&inputs, &config(1), CmTopKOctoPlan::new(rows, cols), || {
                CmTopKOctoAggregator::new(rows, cols, 1024)
            })
            .parent
            .sketch;
            println!(
                "interleaved rows={rows} cols={cols} keys={nkeys} off={off}: heap={} (want {nkeys})",
                sk.heap().len()
            );
        }
    }

    // How fragile is the τ-th-occurrence boundary as collisions get likelier?
    println!("\n== one occurrence below tau, then at tau ==");
    for &(rows, cols, nkeys, workers) in &[
        (3usize, 1024usize, 20u64, 4usize),
        (3, 1024, 20, 1),
        (3, 256, 20, 4),
        (3, 128, 20, 4),
        (3, 64, 20, 4),
        (3, 1024, 60, 4),
        (3, 1024, 100, 4),
        (3, 1024, 200, 4),
    ] {
        let run = |occ: u32, off: u64| {
            let mut inputs: Vec<DataInput<'_>> = Vec::new();
            for key in off..off + nkeys {
                for _ in 0..occ {
                    inputs.push(DataInput::U64(key));
                }
            }
            run_octo(
                &inputs,
                &config(workers),
                CmTopKOctoPlan::new(rows, cols),
                || CmTopKOctoAggregator::new(rows, cols, 1024),
            )
            .parent
            .sketch
        };
        for off in [0u64, 1000, 7777, 123456] {
            let b = run(tau - 1, off).heap().len();
            let a = run(tau, off).heap().len();
            println!(
                "rows={rows} cols={cols} keys={nkeys} workers={workers} off={off}: below={b} at={a} (want 0/{nkeys})"
            );
        }
    }
}

#[cfg(not(feature = "octo-runtime"))]
fn main() {
    println!(
        "this probe needs the `octo-runtime` feature: cargo run --release --features octo-runtime --example octo_topk_promotion_probe"
    );
}
