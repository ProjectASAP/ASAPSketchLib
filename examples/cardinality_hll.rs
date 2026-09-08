// Cardinality estimation with HyperLogLog.
//
// Shows the exact HashSet baseline alongside the HLL sketch so the
// accuracy/memory tradeoff is explicit. Run with:
//
//   cargo run --example cardinality_hll
use std::collections::HashSet;

use asap_sketchlib::{DataInput, ErtlMLE, HyperLogLog};

fn main() {
    // 50_000 events drawn from 1_000 distinct user IDs.
    let user_ids: Vec<u64> = (0..50_000).map(|i| i % 1_000).collect();

    // Exact baseline: HashSet stores every distinct ID.
    let exact: HashSet<u64> = user_ids.iter().copied().collect();
    let exact_count = exact.len();

    // Sketch: HyperLogLog<ErtlMLE> estimates distinct count in fixed memory.
    // ErtlMLE is more accurate than Classic at very low or very high cardinalities.
    let mut hll = HyperLogLog::<ErtlMLE>::default();
    for &id in &user_ids {
        hll.insert(&DataInput::U64(id));
    }
    let estimated_count = hll.estimate();

    println!("distinct user IDs (exact):     {exact_count}");
    println!("distinct user IDs (HLL est.):  {estimated_count}");

    let error_pct =
        (estimated_count as f64 - exact_count as f64).abs() / exact_count as f64 * 100.0;
    println!("relative error: {error_pct:.2}%");
}
