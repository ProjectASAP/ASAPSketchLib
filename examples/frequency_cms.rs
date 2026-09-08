// Frequency estimation with Count-Min Sketch.
//
// Shows the exact HashMap baseline alongside the CMS sketch so the
// accuracy/memory tradeoff is explicit. Run with:
//
//   cargo run --example frequency_cms
use std::collections::HashMap;

use asap_sketchlib::{CountMin, DataInput, FastPath, FixedMatrix};

fn main() {
    // 10_000 events from 100 distinct users (each appears 100 times),
    // except user 42 who gets 500 extra occurrences.
    let target: u64 = 42;
    let mut user_ids: Vec<u64> = (0..10_000).map(|i| i % 100).collect();
    user_ids.extend(std::iter::repeat_n(target, 500));

    // Exact baseline: HashMap stores one counter per distinct key.
    let mut counts: HashMap<u64, u64> = HashMap::new();
    for &id in &user_ids {
        *counts.entry(id).or_insert(0) += 1;
    }
    let exact_count = counts[&target];

    // Sketch: CountMin<FixedMatrix, FastPath> estimates frequencies in bounded memory.
    // FixedMatrix uses a statically-sized backing array.
    // FastPath selects the optimized hashing route for throughput-critical workloads.
    let mut cms = CountMin::<FixedMatrix, FastPath>::default();
    for &id in &user_ids {
        cms.insert(&DataInput::U64(id));
    }
    let estimated_count = cms.estimate(&DataInput::U64(target));

    println!("frequency of user {target} (exact):    {exact_count}");
    println!("frequency of user {target} (CMS est.): {estimated_count}");

    let error = estimated_count as i64 - exact_count as i64;
    println!("overcount: {error}");
}
