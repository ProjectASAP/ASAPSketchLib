# Examples

Runnable end-to-end examples live in [`examples/`](../examples/). Each one pairs an exact-data-structure baseline with its sketch equivalent so the accuracy/memory tradeoff is explicit.

```
cargo run --example cardinality_hll
cargo run --example frequency_cms
cargo run --example quantile_kll
cargo run --example quantile_univmon_q
```

---

## Cardinality estimation — [`examples/cardinality_hll.rs`](../examples/cardinality_hll.rs)

Count how many distinct user IDs appear in a stream.

**Exact baseline** — `HashSet` stores every unique ID; memory grows with cardinality.

```rust
use std::collections::HashSet;

let user_ids: Vec<u64> = get_user_ids();

let mut unique_user_ids = HashSet::new();
for &user_id in &user_ids {
    unique_user_ids.insert(user_id);
}
let unique_users = unique_user_ids.len();
```

**Sketch version** — `HyperLogLog<ErtlMLE>` estimates the count in fixed, bounded memory.

```rust
use asap_sketchlib::{ErtlMLE, HyperLogLog, DataInput};

let user_ids: Vec<u64> = get_user_ids();

// ErtlMLE is more accurate than Classic at very low or very high cardinalities.
let mut hll = HyperLogLog::<ErtlMLE>::default();
for &user_id in &user_ids {
    hll.insert(&DataInput::U64(user_id));
}
let unique_users = hll.estimate();
```

API reference: [`docs/api/api_hyperloglog.md`](./api/api_hyperloglog.md)

---

## Frequency estimation — [`examples/frequency_cms.rs`](../examples/frequency_cms.rs)

Count how many times a specific user ID appears in a stream.

**Exact baseline** — `HashMap` stores one counter per distinct key; memory grows with distinct items.

```rust
use std::collections::HashMap;

let user_ids: Vec<u64> = get_user_ids();

let mut user_counts: HashMap<u64, u64> = HashMap::new();
for &user_id in &user_ids {
    *user_counts.entry(user_id).or_insert(0) += 1;
}
let exact_count = user_counts.get(&101).copied().unwrap_or(0);
```

**Sketch version** — `CountMin<FixedMatrix, FastPath>` estimates frequencies in a compact, fixed-size matrix.

```rust
use asap_sketchlib::{CountMin, DataInput, FixedMatrix, FastPath};

let user_ids: Vec<u64> = get_user_ids();

// FixedMatrix: statically-sized backing array.
// FastPath: optimized hashing route for throughput-critical workloads.
let mut cms = CountMin::<FixedMatrix, FastPath>::default();
for &user_id in &user_ids {
    cms.insert(&DataInput::U64(user_id));
}
let estimated_count = cms.estimate(&DataInput::U64(101));
```

API reference: [`docs/api/api_countmin.md`](./api/api_countmin.md)

---

## Quantile estimation — [`examples/quantile_kll.rs`](../examples/quantile_kll.rs)

Compute quantiles (p50, p90, p99, …) over a large stream of values.

On a 10M-item Zipf workload (101 quantiles), KLL completes in ~214 ms versus ~7,980 ms for Polars. The difference is the sort phase: Polars must sort all buffered values at query time, while KLL does the maintenance work incrementally during insertion.

**Sketch version** — `KLL` maintains a compact sketch during insertion; the CDF is built once and queries are O(log n).

```rust
use asap_sketchlib::sketches::kll::Cdf;
use asap_sketchlib::KLL;

let values: Vec<i64> = get_values();

let mut sketch = KLL::<i64>::init_kll(200);
for v in &values {
    sketch.update(v);
}

// Build the CDF once after all insertions.
let cdf: Cdf = sketch.cdf();

// Query any quantile in O(log n) against the cached CDF.
let p50 = cdf.query(0.50);
let p99 = cdf.query(0.99);
```

API reference: [`docs/api/api_kll.md`](./api/api_kll.md)

---

## Experimental universal quantiles — [`examples/quantile_univmon_q.rs`](../examples/quantile_univmon_q.rs)

The experimental `UnivMonQ` sketch is intended for cases where quantiles are
required together with frequency, distinct count, F2, compatible generic
g-sums, entropy, or heavy-hitter queries. It shares a terminal-stratum
CountSketch pyramid across the universal metrics and adds an adaptively
assisted bottom-k occurrence sample for entropy and ordered ranks.

```rust
use asap_sketchlib::{UnivMonQ, UnivMonQConfig};

let config = UnivMonQConfig::default()
    .with_window_bound(100_000, 1e-6)?;
let mut sketch = UnivMonQ::new(config)?;
for value in 1..=100_000 {
    sketch.add(&value);
}

let query = sketch.prepare_queries();
let percentiles = query.quantiles(&[0.50, 0.90, 0.99]);
let distinct = query.estimate_distinct();
let f2 = query.estimate_f2();
# Ok::<(), Box<dyn std::error::Error>>(())
```

API reference: [`docs/api/api_univmon_q.md`](./api/api_univmon_q.md)
