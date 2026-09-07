// Quantile estimation with KLL.
//
// Inserts 10_000 sequential integers and queries the p50, p90, and p99
// quantiles. On a streaming workload the sketch avoids the sort-over-buffer
// phase that exact engines (e.g., Polars) must pay at query time.
//
// Run with:
//
//   cargo run --example quantile_kll
use asap_sketchlib::KLL;
use asap_sketchlib::sketches::kll::Cdf;

fn main() {
    // Insert 1..=10_000 as i64 values (exact p50 ≈ 5000, p90 ≈ 9000, p99 ≈ 9900).
    let values: Vec<i64> = (1..=10_000).collect();

    let mut sketch = KLL::<i64>::init_kll(200);
    // Single-item path:
    for v in &values {
        sketch.update(v);
    }

    // Build the CDF once after all insertions; subsequent queries are O(log n).
    let cdf: Cdf = sketch.cdf();

    let p50 = cdf.query(0.50);
    let p90 = cdf.query(0.90);
    let p99 = cdf.query(0.99);

    println!("p50: {p50:.1}  (exact ~5000)");
    println!("p90: {p90:.1}  (exact ~9000)");
    println!("p99: {p99:.1}  (exact ~9900)");
}
