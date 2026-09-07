# Advanced Use Cases (Frameworks)

This page covers three distinct categories of advanced usage in `asap_sketchlib`. These are separate problems that happen to share a common answer: composing multiple sketches together.

---

## 1. Subpopulation Queries

**Goal**: Query a stream broken down by one or more categorical dimensions (e.g., "frequency of errors, grouped by region and service").

**The problem with plain sketches**: A single flat CMS can answer "how often does key X appear?" but it cannot answer "how often does key X appear *within region=us*?" without maintaining a separate sketch per dimension value — which blows up memory for high-cardinality dimensions.

**Solution: `Hydra`**

`Hydra` maintains an `r x c` grid of sketches over a fixed schema of labelled key columns. A single `update` supplies one value per column and fans out into all `2^D - 1` non-empty subpopulations that row belongs to. A query then constrains one or more columns.

```rust
use asap_sketchlib::{DataInput, Hydra, input::HydraCounter};

let counter = HydraCounter::CM(Default::default());
let mut hydra = Hydra::with_schema(4, 4096, ["region", "service"], counter).unwrap();
hydra.update(&["us", "api"], &DataInput::Str("err"), None).unwrap();
hydra.update(&["eu", "db"], &DataInput::Str("err"), None).unwrap();
// A subpopulation query with one key and one `None`: errors within region=us.
let est = hydra.query_frequency(&[Some("us"), None], &DataInput::Str("err")).unwrap();
assert!(est >= 1.0);
```

**`HydraCounter`** selects which inner sketch backs each grid cell. **`HydraQuery`** selects the query type: `Frequency`, `Quantile`, `Cdf`, `Cardinality`, `L1Norm`, `L2Norm` or `Entropy`.

API reference: [`docs/api/api_hydra.md`](./api/api_hydra.md)

---

## 2. Sketch Coordination (Hash-Once-Use-Many)

**Goal**: Maintain several different sketch statistics over the same stream while minimizing redundant hash computation.

**The problem**: If you need frequency counts and cardinality simultaneously over the same stream, naively inserting into two separate sketches computes the hash of each element twice.

**Solution: `HashSketchEnsemble` and `UnivMon`**

`HashSketchEnsemble` computes the element hash once and distributes the pre-computed hash value to all member sketches. This enables correlated multi-sketch inserts with a single hash call.

`UnivMon` goes further: it implements the Universal Monitoring framework (Liu et al., SIGCOMM 2016), which answers L1, L2, entropy, and cardinality queries from a single data structure by organizing Count Sketches in a geometric sampling hierarchy.

`NitroBatch` wraps a CMS or Count Sketch in a batch-sampling mode: elements are Nitro-sampled before insertion, reducing the effective insertion rate while preserving accuracy guarantees.

`OctoSketch` provides an alternative sketch-serving framework for high-throughput coordination.

API references: [`docs/api/api_hashlayer.md`](./api/api_hashlayer.md), [`docs/api/api_univmon.md`](./api/api_univmon.md), [`docs/api/api_nitrobatch.md`](./api/api_nitrobatch.md), [`docs/api/api_octo.md`](./api/api_octo.md)

---

## 3. Sliding Windows

**Goal**: Answer frequency or quantile queries over a recent time window (e.g., "top-K IPs in the last 5 minutes") rather than over the entire stream.

**The problem**: A single sketch accumulates all history. To answer windowed queries, you need to expire old data without replaying the stream.

**Solution: `ExponentialHistogram`**

`ExponentialHistogram` (EH) provides sliding-window semantics. It maintains a sequence of sketch "buckets" of geometrically increasing age. When two same-size buckets accumulate, they are merged pairwise. This keeps the total number of buckets logarithmic in the window size. `EHSketchList` provides a unified enum for inserting into and querying across heterogeneous bucket types.

`EHUnivOptimized` is an experimental two-tier EH that integrates `UnivMon` with sketch memory reuse (currently `Unstable`).

**When to use which**:

| Use Case | Recommended |
| --- | --- |
| Sliding window, standard memory | `ExponentialHistogram` + any mergeable sketch |
| Sliding window + universal monitoring | `EHUnivOptimized` (Unstable) |

API references: [`docs/api/api_exponential_histogram.md`](./api/api_exponential_histogram.md), [`docs/api/api_ehsketchlist.md`](./api/api_ehsketchlist.md)
