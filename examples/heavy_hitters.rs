// Top-K heavy hitter detection with CMSHeap.
//
// CMSHeap combines a Count-Min Sketch with a min-heap to track the top-K most
// frequent items automatically. Each insert updates both the frequency estimate
// and the heap — no separate query loop needed.
//
// Run with:
//
//   cargo run --example heavy_hitters
use asap_sketchlib::{CMSHeap, DataInput, RegularPath, Vector2D};

fn main() {
    // Track the top-5 most frequent user IDs across 10_000 events.
    // Users 0-4 each appear ~1_000 times; users 5-99 share the remaining ~5_000.
    let top_k = 5;
    let mut sketch = CMSHeap::<Vector2D<i64>, RegularPath>::new(5, 1024, top_k);

    for i in 0u64..10_000 {
        let user_id = if i % 2 == 0 { i % 5 } else { 5 + (i % 95) };
        sketch.insert(&DataInput::U64(user_id));
    }

    println!("top-{top_k} heavy hitters:");
    let mut heap_items: Vec<_> = sketch.heap().heap().to_vec();
    heap_items.sort_by_key(|b| std::cmp::Reverse(b.count));
    for item in &heap_items {
        println!("  {:?}  count ≈ {}", item.key, item.count);
    }

    // Spot-check: user 0 should appear ~1_000 times.
    let est = sketch.estimate(&DataInput::U64(0));
    println!("\nestimated frequency of user 0: {est}  (expect ~1000)");
}
