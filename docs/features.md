# Feature Status

This document provides a high-level overview of implemented and planned features in asap_sketchlib. For detailed API documentation, see [apis.md](apis.md) and [api_common.md](./api/api_common.md).

---

## Table of Contents

1. [Completed Features](#completed-features)
2. [In Progress](#in-progress)
3. [Planned Features](#planned-features)

---

## Completed Features

### Core Infrastructure

**Common API** ([api_common.md](./api/api_common.md))

- `DataInput` - Unified type system for all sketches
- `Vector1D`, `Vector2D` - Flat storage structures for sketch counters
- `BitMatrix` - Packed one-bit-per-cell grid behind the same `MatrixStorage` interface as the counter matrices, for membership sketches
- `impl_fixed_matrix!` macro - Define compile-time fixed-size matrix types with any counter type and dimensions
- `impl_hll_bucket_list!` macro - Define compile-time fixed-size HLL register storage types at any precision (the crate ships `lg_k` 12/14/16)
- `CommonHeap` & `HHHeap` - Generic and specialized heaps for heavy hitter tracking
- Deterministic hashing with seed management
- Pluggable hash via the `SketchHasher` trait — swap the hash function without changing sketch code
- `DigestHasher` / `DigestBuildHasher` - Single-avalanche hasher for maps keyed by a value that is already a digest
- `RegularPath` / `FastPath` modes - Type-level pairing of insert/estimate paths

**Sketch APIs** — Frequency estimation, cardinality, quantiles, heavy hitters, set membership, sampling, and more. See [apis.md](apis.md) for the full list with per-sketch status, error guarantees, and references.

### Frameworks

**Hydra** - Subpopulation queries over multi-dimensional keys ([apis.md](apis.md))

**UnivMon** - Universal monitoring (L1, L2, entropy, cardinality from single structure) ([apis.md](apis.md))

**UnivMonPyramid** - Optimized two-tier UnivMon with `UnivSketchPool` for insert and memory management ([apis.md](apis.md))

**UnivMonQ (Experimental)** - Terminal-stratum UnivMon with exact insertion-only L1, F0/F2/compatible g-sum queries, an adaptively assisted occurrence sample for entropy, ranks, and quantiles, and reusable prepared query views. Its API, estimators, and guarantees may change.

**HashSketchEnsemble** - Hash-once-use-many pattern for coordinating multiple sketches with single hash computation

**NitroBatch** - Batch-mode sampling wrapper for CMS/Count FastPath

**EHSketchList** - Unified sketch enum for insert/merge/query across sketch types, that can be integrated into `ExponentialHistogram`

**ExponentialHistogram** - Sliding window coordinator for mergeable sketches

**TumblingWindow** - Non-overlapping windows over pooled mergeable sketches, with built-in support for FoldCMS, FoldCS, KLL and the experimental UnivMon-Q

**EHUnivOptimized** - Hybrid two-tier ExponentialHistogram for UnivMon with sketch memory reuse (currently `Unstable`)

**OctoSketch** - Multi-threaded sketch-serving framework with worker/aggregator architecture ([apis.md](apis.md))

### Performance Optimizations

**Reduced hashing overhead** — Hashing is often the bottleneck when updating sketches at high throughput. `FastPath` mode computes a single hash per insert and derives all row indices from it via bit-masking, avoiding redundant hash calls. `HashSketchEnsemble` extends this across multiple sketches, so inserting one item into several sketches still costs only one hash.

**Cache-friendly memory layout** — Sketch counters are stored in flat, row-major arrays (`Vector1D`, `Vector2D`) so that sequential access patterns hit L1/L2 cache instead of chasing pointers.

**Zero-copy access** — Query and merge operations work directly on borrowed slices, avoiding unnecessary allocation and copying on the hot path.

**Fixed-size storage and monomorphization** — The `impl_fixed_matrix!` macro generates matrix types with compile-time-known dimensions and counter types. This lets the compiler inline size computations, eliminate bounds checks, and fully monomorphize hot loops — removing the overhead of dynamically-sized storage on the critical path.

### Serialization

**MessagePack (rmp-serde)** - Serialization support across most sketch types

### Sampling

**Nitro sampling** - Streaming Nitro (`Vector2D`) and batch Nitro (`NitroBatch`)

---

## In Progress

### Performance

Insertion throughput measured on 10,000,000 Zipf-distributed `int64` values (s=1.1, support=100k), averaged over 10 seeded runs.

### Testing

- Current test coverage is documented in [tests.md](tests.md). Additional unit tests and strict correctness tests are in progress.

### Serialization

MessagePack (`rmp-serde`) support. **serde support** means the type derives `Serialize`/`Deserialize` and can be used with any serde-compatible serializer. **Built-in helpers** (`serialize_to_bytes` / `deserialize_from_bytes`) provide one-call round-tripping without requiring users to depend on `rmp-serde` directly. These helpers emit the self-describing **ASAPv1** wire envelope (see the [ASAPv1 wire format spec](./asapv1_wire_format.md)) for every sketch the spec's `kind_id` registry marks *implemented*; NitroBatch, FoldCMS, FoldCS, HashSketchEnsemble, EHUnivOptimized and OctoSketch are not converted.

| Component | serde support | Built-in helpers |
| --- | --- | --- |
| CountMin | Yes | Yes |
| Count / CountL2HH | Yes | Yes |
| HyperLogLog (all variants) | Yes | Yes |
| DDSketch | Yes | Yes |
| KLL / KLLDynamic | Yes | Yes |
| KMV | Yes | Yes |
| Elastic | Yes | Yes |
| Coco | Yes | Yes |
| UniformSampling | Yes | Yes |
| FoldCMS / FoldCS | Yes | In Progress |
| CMSHeap / CSHeap | In Progress | Yes |
| SpaceSaving | Yes | Yes |
| Bloom | Yes | Yes |
| Hydra | Yes | Yes |
| UnivMon | Yes | Yes |
| UnivMon Optimized | No | Yes |
| UnivMonQ (experimental) | Internal wire DTO | Yes |
| NitroBatch | Yes | In Progress |
| EHSketchList | Yes | Yes |
| ExponentialHistogram | No | Yes |

Protobuf (prost): `.proto` definitions exist for CountMin, Count, HLL, DDSketch, KLL, Elastic, Coco, Hydra, and UnivMon. Rust conversion code is in progress.

### API Stability

- Public APIs are stabilizing but may still change in naming and structure

---

## Planned Features

### Performance Optimization

**SIMD support**

- Vector operations for counter updates (AVX2/NEON)

### Algorithm Improvements

**Custom RNG for KLL**

- Fast coin-flipping random number generator optimized for KLL compactor operations
