// Merging two Count-Min Sketches.
//
// A common pattern in distributed pipelines: each worker maintains its own
// CMS, then sketches are merged at query time. The merged sketch estimates
// frequencies as if all items had been inserted into a single sketch.
//
// Run with:
//
//   cargo run --example merge_cms
use asap_sketchlib::{CountMin, DataInput, RegularPath, Vector2D};

fn main() {
    // Worker A sees even user IDs.
    let mut cms_a: CountMin<Vector2D<i32>, RegularPath> = CountMin::with_dimensions(5, 1024);
    for id in (0u64..10_000).step_by(2) {
        cms_a.insert(&DataInput::U64(id));
    }

    // Worker B sees odd user IDs, plus extra occurrences of user 42.
    let mut cms_b: CountMin<Vector2D<i32>, RegularPath> = CountMin::with_dimensions(5, 1024);
    for id in (1u64..10_000).step_by(2) {
        cms_b.insert(&DataInput::U64(id));
    }
    for _ in 0..99 {
        cms_b.insert(&DataInput::U64(42));
    }

    // Merge B into A — merge requires identical dimensions.
    cms_a.merge(&cms_b);

    // User 42 appeared once in A (even) and 1 + 99 = 100 times in B → total ≥ 101.
    let est = cms_a.estimate(&DataInput::U64(42));
    println!("estimated count for user 42 after merge: {est}  (exact: 101)");
}
