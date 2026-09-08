// Serializing and deserializing sketches.
//
// Sketches can be serialized to bytes (MessagePack) and reconstructed later.
// This is the key primitive for shipping sketch state across the network,
// persisting checkpoints, or reading results from another service.
//
// Run with:
//
//   cargo run --example serialize_deserialize
use asap_sketchlib::{CountMin, DataInput, ErtlMLE, HyperLogLog, KLL, RegularPath, Vector2D};

fn main() {
    // --- Count-Min Sketch ---
    let mut cms: CountMin<Vector2D<i64>, RegularPath> = CountMin::with_dimensions(5, 1024);
    for id in 0u64..1_000 {
        cms.insert(&DataInput::U64(id % 100)); // 100 distinct IDs
    }
    let before = cms.estimate(&DataInput::U64(42));

    let bytes = cms.serialize_to_bytes().expect("CMS serialize");
    let cms2 = CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&bytes)
        .expect("CMS deserialize");
    let after = cms2.estimate(&DataInput::U64(42));

    println!(
        "CMS  — before: {before}, after round-trip: {after}  (equal: {})",
        before == after
    );
    println!("CMS  — serialized size: {} bytes", bytes.len());

    // --- HyperLogLog ---
    let mut hll = HyperLogLog::<ErtlMLE>::default();
    for id in 0u64..10_000 {
        hll.insert(&DataInput::U64(id));
    }
    let card_before = hll.estimate();

    let bytes = hll.serialize_to_bytes().expect("HLL serialize");
    let hll2 = HyperLogLog::<ErtlMLE>::deserialize_from_bytes(&bytes).expect("HLL deserialize");
    let card_after = hll2.estimate();

    println!(
        "HLL  — before: {card_before}, after round-trip: {card_after}  (equal: {})",
        card_before == card_after
    );
    println!("HLL  — serialized size: {} bytes", bytes.len());

    // --- KLL ---
    let mut kll = KLL::<i64>::init_kll(200);
    for v in 1i64..=10_000 {
        kll.update(&v);
    }
    let p99_before = kll.cdf().query(0.99);

    let bytes = kll.serialize_to_bytes().expect("KLL serialize");
    let kll2 = KLL::<i64>::deserialize_from_bytes(&bytes).expect("KLL deserialize");
    let p99_after = kll2.cdf().query(0.99);

    println!(
        "KLL  — p99 before: {p99_before:.1}, after round-trip: {p99_after:.1}  (equal: {})",
        (p99_before - p99_after).abs() < 1.0
    );
    println!("KLL  — serialized size: {} bytes", bytes.len());
}
