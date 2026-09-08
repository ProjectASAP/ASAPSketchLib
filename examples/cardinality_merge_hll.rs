// Union cardinality across shards with HyperLogLog merge.
//
// Each shard processes a disjoint partition of user IDs. HLL sketches can be
// merged to get the cardinality of the union without re-processing the data —
// a fundamental property for distributed or multi-partition workloads.
//
// Run with:
//
//   cargo run --example cardinality_merge_hll
use asap_sketchlib::{DataInput, ErtlMLE, HyperLogLog};

fn main() {
    // Three shards, each with 10_000 distinct user IDs. IDs do not overlap.
    let shard_size = 10_000u64;
    let num_shards = 3u64;

    let mut shards: Vec<HyperLogLog<ErtlMLE>> = (0..num_shards)
        .map(|shard| {
            let mut hll = HyperLogLog::<ErtlMLE>::default();
            let start = shard * shard_size;
            for id in start..start + shard_size {
                hll.insert(&DataInput::U64(id));
            }
            hll
        })
        .collect();

    // Merge all shards into the first sketch.
    let (first, rest) = shards.split_first_mut().unwrap();
    for shard in rest.iter() {
        first.merge(shard);
    }

    let exact = (shard_size * num_shards) as usize;
    let estimated = first.estimate();
    let error_pct = (estimated as f64 - exact as f64).abs() / exact as f64 * 100.0;

    println!("shards: {num_shards}  ×  {shard_size} distinct IDs each");
    println!("exact total distinct: {exact}");
    println!("HLL merged estimate:  {estimated}");
    println!("relative error: {error_pct:.2}%");
}
