# asap_sketchlib

[![Crates.io](https://img.shields.io/crates/v/asap_sketchlib.svg)](https://crates.io/crates/asap_sketchlib)
[![docs.rs](https://docs.rs/asap_sketchlib/badge.svg)](https://docs.rs/asap_sketchlib)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ProjectASAP/asap_sketchlib/blob/main/LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

A Rust library for **streaming data sketches** — compact data structures that give approximate answers (counts, distinct counts, percentiles) over data streams too large to store exactly.

## Why asap_sketchlib

- **Fast.** Up to 8–14× higher insertion throughput than comparable libraries on frequency sketches, 2–3× on cardinality sketches, and 2–4× on quantile sketches. Rust-native with no language-boundary overhead. See [benchmarks](#performance).
- **High coverage.** Supports frequency, cardinality, quantile, distribution, heavy-hitter, and set-membership sketches (`CountMin`, `Count`, `HyperLogLog`, `KLL`, `DDSketch`, `SpaceSaving`, `Bloom`). Also includes algorithms not found in other libraries: `UnivMon` for estimating a broad class of streaming statistics (L1/L2 norms, entropy) in a single pass, `Hydra` for answering sketch queries over arbitrary subpopulations without per-group sketches, and `NitroBatch` for accelerating sketch updates through batching. Unique sketch frameworks for sliding windows (`ExponentialHistogram`) and subpopulation queries (`Hydra`).
- **Easy to use.** Most sketches provide a unified API style, while some (such as `KLL`) use `update`/`quantile`; the crate also offers typed inputs via `DataInput`, pluggable hashing via `SketchHasher`, and multi-sketch composition with shared hashing (`HashSketchEnsemble`).

## Supported Sketches

| Goal | Sketch | When to pick it | What it does | Polars equivalent |
| --- | --- | --- | --- | --- |
| Frequency estimation | `CountMin`, `Count` | Fast approximate counts for high-volume keys | Estimates how often each key appears in a stream | `df.group_by("key").agg(pl.len())` |
| Heavy hitters / frequent items | `SpaceSaving`, `CMSHeap`, `CSHeap` | The top-k keys and their counts under a fixed memory budget | Tracks the most frequent keys of a stream, each with a per-key error bound | `df["key"].value_counts().top_k(10, by="count")` |
| Approximate set membership | `Bloom` | Cheap "have I seen this key?" checks at a chosen false-positive rate | Answers membership with no false negatives and a bounded false-positive rate | `df["key"].is_in(seen)` — exact, but stores every key |
| Cardinality estimation | `HyperLogLog`, `HyperLogLogHIP` | Approximate distinct counts with bounded memory | Estimates the number of unique elements | `df["col"].n_unique()` |
| Quantiles / distribution | `KLL`, `DDSketch`, `UnivMonQ` (experimental) | Percentiles alone, or percentiles sharing state with universal frequency metrics | Approximates arbitrary quantiles (e.g. p50, p99) of a value distribution | `df["col"].quantile(0.99)` |
| Subpopulation queries | `Hydra` | Filtered sketch queries | Answers sketch queries over arbitrary subpopulations without maintaining per-group sketches | No direct equivalent — requires per-group aggregation |
| Universal monitoring | `UnivMon` | G-sum queries (L1/L2 norms, cardinality, entropy) | Estimates a broad class of streaming statistics in a single pass | No direct equivalent — requires custom multi-pass pipelines |
| Universal monitoring + quantiles | `UnivMonQ` (experimental) | One mergeable structure for frequencies, F0/F2/compatible g-sums, entropy, heavy hitters, ranks, and quantiles | Extends a terminal-stratum UnivMon core with an adaptively assisted occurrence sample | No direct equivalent — requires multiple aggregations |
| Update acceleration | `NitroBatch` | Batch-accelerated sketch updates | Speeds up sketch insertions by batching updates | No direct equivalent |

Full sketch status and API details: [APIs Index](./docs/apis.md).

## Quick Start

**Minimum Supported Rust Version (MSRV): 1.85** (Rust 2024 edition)

Install from [crates.io](https://crates.io/crates/asap_sketchlib):

```bash
cargo add asap_sketchlib
```

```toml
[dependencies]
asap_sketchlib = "0.3"
```

API docs are hosted on [docs.rs](https://docs.rs/asap_sketchlib).

Alternatively, pin to a tagged revision from GitHub:

```toml
[dependencies]
asap_sketchlib = { git = "https://github.com/ProjectASAP/asap_sketchlib", tag = "v0.3.0" }
```

### Count distinct users with HyperLogLog

```rust
use asap_sketchlib::{ErtlMLE, HyperLogLog, DataInput};

// HyperLogLog estimates the number of distinct items in a stream using fixed memory.
// ErtlMLE is one of the HLL variants we offer — it tends to be more accurate than
// the `Classic` variant, especially at very low or very high cardinalities.
let mut hll = HyperLogLog::<ErtlMLE>::default();

// Insert some user IDs — HLL handles distinct counting and deduplicates items.
for user_id in [101, 202, 303, 101, 404, 202, 505, 101] {
    hll.insert(&DataInput::U64(user_id));
}

let unique_users = hll.estimate();
println!("estimated unique users: {unique_users}"); // ≈ 5
```

### Estimate frequency of items with Count-Min Sketch

```rust
use asap_sketchlib::{CountMin, FastPath, Vector2D, DataInput};

// Count-Min Sketch estimates how often each item appears in a stream.
// It may over-count but never under-counts.
//
// Vector2D<i32> is the backing storage (a 2D array of 32-bit counters).
// FastPath uses a single hash with bit-masking to pick row indices — faster
// than the default RegularPath which hashes once per row.
let mut cms = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 2048);

// Simulate an event stream with known frequencies.
let events = [
    ("page_view", 1000),
    ("click",      500),
    ("signup",     100),
    ("purchase",    50),
];
for &(event, count) in &events {
    for _ in 0..count {
        cms.insert(&DataInput::Str(event));
    }
}

// Estimates are close to the true counts (CMS may over-count, but never under-counts).
for &(event, true_count) in &events {
    let est = cms.estimate(&DataInput::Str(event));
    println!("{event:>10}: estimate = {est}, true = {true_count}");
}
```

### Track latency percentiles with KLL

```rust
use asap_sketchlib::KLL;

// KLL is a quantile sketch — it tracks the distribution of values so you can
// ask questions like "what is the median?" without storing every data point.
let mut sketch = KLL::<f64>::default();

// Simulate 1000 latency samples in milliseconds
for i in 0..1000 {
    let ms = (i as f64) * 0.5 + 1.0;
    sketch.update(&ms);
}

let p50 = sketch.quantile(0.50);
let p99 = sketch.quantile(0.99);
println!("median ≈ {p50:.1} ms, p99 ≈ {p99:.1} ms");
```

### Experiment with one sketch for universal metrics and quantiles

`UnivMonQ` is experimental: its API, estimators, and guarantees may change as
the construction is evaluated further. See the
[large synthetic comparison with UnivMon](./docs/univmon_q_evaluation.md).

```rust
use asap_sketchlib::{UnivMonQ, UnivMonQConfig};

let config = UnivMonQConfig::default()
    .with_window_bound(100_000, 1e-6)?;
let mut sketch = UnivMonQ::new(config)?;

for value in [8.0, 3.0, 5.0, 5.0, 10.0] {
    sketch.update(&value);
}

let query = sketch.prepare_queries();
println!("p50 ≈ {:?}", query.quantile(0.5));
println!("distinct ≈ {}", query.estimate_distinct());
println!("F2 ≈ {}", query.estimate_f2());
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Merge multiple sketch instances

```rust
use asap_sketchlib::{ErtlMLE, HyperLogLog, DataInput};

// Sketches are mergeable — you can build one per node and combine them later
// to get a global answer without shipping raw data.
let mut node_a = HyperLogLog::<ErtlMLE>::default();
let mut node_b = HyperLogLog::<ErtlMLE>::default();

// Each node sees different (and some overlapping) users
for id in [1, 2, 3, 4, 5]  { node_a.insert(&DataInput::U64(id)); }
for id in [4, 5, 6, 7, 8]  { node_b.insert(&DataInput::U64(id)); }

node_a.merge(&node_b);
println!("total unique users ≈ {}", node_a.estimate()); // ≈ 8
```

## Choosing Between Sketches

Several sketches address the same goal with different trade-offs — for example, `CountMin` vs `Count` for frequency, or `KLL` vs `DDSketch` for quantiles.

We are building **SketchPlan**, a profiler that analyzes a representative sample of your data and recommends the best sketch configuration (algorithm, memory budget, error tolerance) for your workload. Until SketchPlan is ready, the [APIs Index](./docs/apis.md) lists guarantees, error bounds, and caveats for each sketch to help you decide.

## Performance

Insertion throughput on 10M Zipf-distributed values, averaged over 10 runs:

- Frequency sketches: up to 8-14x higher insertion throughput than comparable libraries
- Cardinality sketches: roughly 2-3x higher insertion throughput
- Quantile sketches: roughly 2-4x higher insertion throughput

Benchmark methodology, tuning notes, and performance details (including cache-friendly layouts and `FastPath` single-hash mode) are in [Performance Notes](./docs/features.md).

## Documentation

| Doc | Contents |
| --- | --- |
| [APIs Index](./docs/apis.md) | Per-sketch API reference with status and error guarantees |
| [Advanced Use Cases](./docs/advanced_use_cases.md) | Subpopulation queries, windowed sketching, multi-sketch coordination |
| [Docs Index](./docs/index.md) | Full documentation index |

If you are evaluating the crate for production use, start with the API index first. It calls out which APIs are stable today and which are still feature-gated or experimental.

## Dev Commands

```bash
cargo build --all-targets
cargo test --all-features
```

- `--all-targets` builds everything: the library, binaries, and tests.
- `--all-features` enables every Cargo feature, so all feature-gated code is compiled and tested. The features include:
  - `experimental` — enables sketches and APIs that are still under development and may change without notice.
  - `octo-runtime` — enables the Octo multi-threaded runtime (pulls in `core_affinity` and `crossbeam-channel`).

To build or test with a specific feature:

```bash
cargo build --features experimental
cargo test --features "experimental octo-runtime"
```

## Protobuf code generation

`asap_sketchlib` is a pure-Rust crate: building it does **not** require
`protoc` or any build script. The Rust types generated from
`proto/**/*.proto` are vendored into `src/proto/generated/` and refreshed
manually by maintainers using the in-repo tool at `tools/gen-proto/`.

Downstream users can therefore simply add the crate to their `Cargo.toml` and
build it like any other pure-Rust dependency.

### For maintainers

After editing any `.proto` file, regenerate the vendored output and commit
the result:

```bash
cargo run --manifest-path tools/gen-proto/Cargo.toml
git add src/proto/generated
```

The tool uses `prost-build` together with the `protoc-bin-vendored` binary,
so no system `protoc` is required to regenerate either. CI rejects any pull
request whose committed `src/proto/generated/` does not match the result of
running this command on its current `.proto` sources.

## FAQ

### When is Apache DataSketches a better fit?

- You need its broader algorithm catalog (CPC, Theta/Tuple with set operators, REQ, VarOpt/Reservoir, FM85).
- You need cross-language binary compatibility with existing DataSketches deployments in Java, C++, or Python.
- You need long-running production maturity and an Apache-governed release cycle.

## Contributors

### Major Contributors

- [Yancheng Yuan](https://github.com/GordonYuanyc)
- [Zeying Zhu](https://github.com/zzylol)

### Other Contributors

- [Sie Deta Dirganjaya](https://github.com/SieDeta)
- [Gnanesh Gnani](https://github.com/GnaneshGnani)

## License

Copyright 2025 - present ProjectASAP

Licensed under the MIT License. See [LICENSE](https://github.com/ProjectASAP/asap_sketchlib/blob/main/LICENSE).
