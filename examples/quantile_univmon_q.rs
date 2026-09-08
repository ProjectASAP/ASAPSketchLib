// Universal frequency measurements plus quantiles with UnivMon-Q.
//
// Run with:
//
//   cargo run --example quantile_univmon_q

use asap_sketchlib::{UnivMonQ, UnivMonQConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = UnivMonQConfig::default().with_window_bound(100_000, 1e-6)?;
    let mut sketch = UnivMonQ::new(config)?;

    for value in 1..=100_000 {
        sketch.add(&value);
    }

    let query = sketch.prepare_queries();
    let quantiles = query.quantiles(&[0.50, 0.90, 0.99]);
    println!("p50: {:.1}", quantiles[0].unwrap());
    println!("p90: {:.1}", quantiles[1].unwrap());
    println!("p99: {:.1}", quantiles[2].unwrap());
    println!("distinct: {:.0}", query.estimate_distinct());
    println!("F2: {:.0}", query.estimate_f2());
    println!("memory: {} KiB", sketch.estimated_memory_bytes() / 1024);
    Ok(())
}
