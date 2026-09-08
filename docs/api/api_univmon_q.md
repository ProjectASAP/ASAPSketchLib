# API: UnivMon-Q

Status: `Experimental`

## Purpose

The experimental `UnivMonQ` API combines universal frequency-vector measurements with additive-rank
quantile estimation. Choose it when one stream must answer point frequency,
distinct count, F2, entropy, heavy-hitter, rank, CDF, and quantile queries from
shared mergeable state. For quantiles alone, KLL or DDSketch is normally much
smaller. Its API, estimators, and guarantees may change as the construction is
evaluated further.

The update path uses one key hash and one physical CountSketch layer per
observation. A second hash of `(source_id, local_sequence)` supplies an
independent coordinated bottom-k priority for occurrence sampling.

Each terminal stratum keeps the identities with the largest observed
CountSketch frequency estimates. A separate `ever_evicted` bit records whether
that bounded identity set has ever become incomplete; merely filling the table
does not count as eviction. Query reconstruction applies the L2 threshold only
to incomplete logical candidate sets.

## Types

- `UnivMonQ<H = DefaultXxHasher>`
- `UnivMonQQuery<'a, H = DefaultXxHasher>`
- `UnivMonQConfig`
- `UnivMonQPoint`
- `UnivMonQError`

## Constructors

```rust
fn default() -> Self
fn new(config: UnivMonQConfig) -> Result<Self, UnivMonQError>
fn new_with_source_id(config: UnivMonQConfig, source_id: u64)
    -> Result<Self, UnivMonQError>
fn with_hasher(config: UnivMonQConfig) -> Result<Self, UnivMonQError>
fn with_hasher_and_source_id(config: UnivMonQConfig, source_id: u64)
    -> Result<Self, UnivMonQError>
```

`new` selects `DefaultXxHasher` and allocates a process-local source ID. Use
`new_with_source_id` with a stable, globally unique partition or shard ID for
distributed or cross-process merging. The `with_hasher` variants provide the
same choices for a custom hash implementation.

`UnivMonQConfig::with_window_bound(max_updates, failure_probability)` chooses
the smallest level count whose deepest sample fits the candidate table under a
Bernstein tail bound. `max_updates` is the aggregate logical window size across
all merged shards.

## Update

```rust
fn update(&mut self, value: &f64)
fn add<T: NumericalValue>(&mut self, value: &T)
fn update_data_input(&mut self, value: &DataInput) -> Result<(), &'static str>
fn clear(&mut self)
```

`DataInput` strings and byte arrays are rejected. Integer `DataInput` variants
are projected to `f64`, matching the type-erased KLL path; integers above the
exact `f64` range should use an explicitly typed application-level encoding if
distinct identity must be preserved.

Values follow `f64::total_cmp`, so `-0.0` and `0.0` remain distinct and NaNs
have deterministic positions.

## Query

```rust
fn count(&self) -> u64
fn len(&self) -> u64
fn is_empty(&self) -> bool
fn min(&self) -> Option<f64>
fn max(&self) -> Option<f64>
fn source_id(&self) -> u64

fn estimate_frequency(&self, value: f64) -> u64
fn estimate_distinct(&self) -> f64
fn estimate_f2(&self) -> f64
fn estimate_l1(&self) -> f64
fn estimate_g_sum<F>(&self, g: F) -> f64 where F: Fn(f64) -> f64
fn estimate_entropy(&self) -> f64
fn estimate_entropy_universal(&self) -> f64
fn estimate_entropy_occurrence(&self) -> Option<f64>
fn heavy_hitters(&self, k: usize) -> Vec<(f64, u64)>

fn estimate_rank_universal(&self, value: f64) -> Option<u64>
fn rank(&self, value: f64) -> Option<u64>
fn quantile(&self, q: f64) -> Option<f64>
fn quantiles(&self, quantiles: &[f64]) -> Vec<Option<f64>>
fn get_value_at_quantile(&self, q: f64) -> Option<f64>
fn cdf(&self) -> Vec<UnivMonQPoint>
fn prepare_queries(&self) -> UnivMonQQuery<'_, H>
fn estimated_memory_bytes(&self) -> usize
```

The public generic g-sum API preserves UnivMon's ability to evaluate compatible
frequency functions. Super-quadratic moments such as F3 are deliberately not
advertised: the current memory profile does not provide their stronger space
guarantees.

`estimate_l1()` returns the exact observation count for the supported
insertion-only stream. `estimate_entropy()` adaptively selects between the
original UnivMon recurrence and the coordinated occurrence-sample identity
`H = E[ln(N/f_X)]`. The recurrence is used for diffuse streams and complete
candidate recovery; concentrated streams with incomplete recovery use the
occurrence path. The two component estimators remain exposed for experiments.

`prepare_queries` reconstructs candidate frequencies, logical sampled levels,
F2 thresholds, and the ordered CDF once. Its returned immutable view exposes
count/min/max/frequency, F0/F2/g-sum/entropy/heavy hitters, rank, batched
quantiles, and a borrowed CDF slice. Prefer it when a scrape or report asks for
multiple metrics from one snapshot:

```rust
let query = sketch.prepare_queries();
let f0 = query.estimate_distinct();
let f2 = query.estimate_f2();
let entropy = query.estimate_entropy();
let percentiles = query.quantiles(&[0.5, 0.9, 0.99]);
let cdf = query.cdf();
```

The direct methods remain convenient for isolated queries. `quantiles` is the
batched alternative to repeated `quantile` calls and constructs the CDF once.

`estimate_rank_universal(x)` is the most direct extension of the original
UnivMon construction. It evaluates the key-dependent separable function
`g_x(v, f_v) = f_v * I[v <= x]` through the usual sampled-level recurrence and
does not require `ordered_samples`. It is intended for a small, predetermined
set of fixed thresholds. Separate calls are not constrained to be monotone, so
this experimental method is not used to implement `quantile` or `cdf`.

Setting `ordered_samples = 0` removes ordered-sample memory and disables
non-endpoint rank/quantile/CDF queries. Entropy then always uses the universal
recurrence; the other universal metrics are unchanged.

### Entropy guarantee boundary

For recovered heavy set `A`, residual mass `P_R`, and an occurrence-uniform
residual value `X_R`, Shannon entropy decomposes as

```text
H = sum_(h in A) (f_h/N) ln(N/f_h) + P_R E[ln(N/f_X_R)].
```

If retained residual occurrences used exact frequencies, Hoeffding's
inequality gives residual sampling error at most

```text
P_R ln(N) * sqrt(ln(2 / delta) / (2 m_R))
```

with probability `1 - delta`, where `m_R` is the number of retained residual
occurrences. Replacing `f_X` by CountSketch estimate `f_hat_X` adds
`P_R mean |ln(f_X/f_hat_X)|`; if every sampled residual frequency has relative
error at most `rho < 1`, that term is at most `-P_R ln(1-rho)`. Heavy-frequency
and residual-mass estimation add their corresponding plug-in error. UnivMon-Q's
occurrence priorities use a separate hash domain from its key-frequency hashes.

The adaptive selector keeps the original recurrence when recovery is complete
or estimated concentration is below `1/k`; otherwise it uses this occurrence
identity. The component bounds explain the estimator, but selection at that
empirical gate is not yet a new end-to-end theorem. UnivMon-Q remains
experimental.

### Ordered-query guarantee boundary

There are two rank mechanisms with different guarantee boundaries:

1. `estimate_rank_universal(x)` stays within the original UnivMon hierarchy.
   Subject to the standard L2-heavy recovery assumptions, it estimates one
   fixed threshold using the same recurrence as other separable sums. In a
   diffuse stream its additive error has the sampling scale `N / sqrt(k)`, so
   obtaining normalized rank error `epsilon` requires `k` on the order of
   `1 / epsilon^2` (before confidence amplification). A uniform guarantee over
   the full CDF also needs simultaneous control over all thresholds.
2. `rank`, `cdf`, and `quantile` expose one adaptive assisted estimator. The
   implementation always retains a uniform bottom-k sample of occurrences. It
   uses UnivMon's F2 estimate to detect concentration and, only then, replaces
   reliable heavy-value sample mass with recovered CountSketch frequencies.
   The occurrence sample estimates the remaining residual distribution. When
   no heavy value qualifies, the same algorithm has an empty heavy set and
   reduces internally to the ordinary empirical occurrence CDF; this is not a
   separate public mode.

Let `H` be the recovered heavy set, `f_h` and `f_hat_h` its exact and estimated
frequencies, `P_hat_R = 1 - sum_h f_hat_h/N` the estimated residual mass, and
`m_R` the retained occurrences outside `H`. Define

```text
E_H = sum_h |f_hat_h - f_h| / N
epsilon_R = sqrt(log(2 / delta) / (2 m_R)).
```

Conditioned on the recovered heavy set and sufficiently independent occurrence
priorities, the monotone assisted CDF obeys

```text
sup_x |F_hat(x) - F(x)| <= 2 E_H + P_hat_R * epsilon_R.
```

The adaptive gate is `F2_hat / N^2 >= 1 / ordered_samples`; admitted heavy
values also satisfy `f_hat_h >= sqrt(F2_hat / width)`. Thus diffuse streams use
the distribution-independent occurrence bound directly, while concentrated
streams can reduce residual sampling error at the cost of the explicit
UnivMon heavy-frequency error term. This remains `O(1/epsilon^2)` sample space;
use KLL when optimal quantile-only space is the primary requirement.

## Merge

```rust
fn merge(&mut self, other: &Self) -> Result<(), UnivMonQError>
```

Merges require identical dimensions and `hash_seed`, the same compile-time
hasher type, and globally unique occurrence source IDs. Terminal counters add,
candidate identities are unioned and rescored from the merged CountSketch,
eviction history is ORed with any merge-time truncation, and occurrence samples
are unioned and pruned by their common priorities. Source-ID uniqueness is a
caller contract rather than an unbounded registry stored in the sketch. With
32-bit counters, callers must size the aggregate window so no signed cell
saturates; use `counter_bits = 64` otherwise.

`UnivMonQ` also implements `TumblingWindowSketch`, so it can be used directly
with `TumblingWindow<UnivMonQ>`. The window adapter treats the `DataInput` key
as the numeric observation, like the existing KLL adapter.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope (kind `0x1a 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). They exist for any
`H: HashProfile`, so a decode of one profile's bytes into another profile's
sketch is rejected.

The whole `UnivMonQConfig` is construction config and lives in the metadata
(`seed_index`, `levels`, `width`, `width_halving_period`, `depth`,
`counter_type`, `candidates`, `ordered_samples`), so every per-level width, the
hash layout and each level's candidate capacity are derived and none is stored.
The payload is `[counters, candidate_lens, candidate_keys, candidate_scores,
ever_evicted, count, min, max, source_id, next_sequence,
occurrence_priority_high, occurrence_priority_low, occurrence_keys]`.

Both heaps — a level's candidate min-heap and the ordered sample — are rebuilt
on decode, so the payload pins an order instead: candidates ascending by key,
occurrences ascending by `(priority_high, priority_low, key)`. `source_id` and
`next_sequence` are carried so a decoded sketch continues the same occurrence
draw sequence; `min` and `max` travel as msgpack nil when the sketch is empty.
`ever_evicted` is carried too — a full candidate table does not say whether it
ever displaced anything.

## Example

```rust
use asap_sketchlib::{UnivMonQ, UnivMonQConfig};

let config = UnivMonQConfig::default()
    .with_window_bound(100_000, 1e-6)?;
let mut left = UnivMonQ::new_with_source_id(config, 10)?;
let mut right = UnivMonQ::new_with_source_id(config, 20)?;

for value in 0..50_000 {
    left.add(&value);
}
for value in 50_000..100_000 {
    right.add(&value);
}
left.merge(&right)?;

assert!(left.quantile(0.5).is_some());
assert!(left.estimate_distinct() > 0.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Accuracy caveat

This is an empirically validated UnivMon extension, not a replacement for
KLL's formal `O(1/epsilon)` additive-rank guarantee. Its occurrence sample has
the natural `O(1/epsilon^2)` space behavior, and the assisted theorem includes
the recovered-heavy frequency error explicitly.

## Empirical comparison with UnivMon

The reproducible [large synthetic evaluation](../univmon_q_evaluation.md)
compares UnivMon-Q with terminal-only UnivMon over seven skew levels and five
trials. At the tested compact candidate budget, UnivMon was generally more
accurate for shared universal metrics, while UnivMon-Q added ordered queries
and used substantially less update time, merge time, query time, and memory.
The document also separates construction differences from the current
UnivMon heavy-hitter metadata bottleneck.

## References

- [Liu et al., “One Sketch to Rule Them All,” SIGCOMM 2016](https://doi.org/10.1145/2934872.2934906).
- [Yang et al., “Joltik,” MobiCom 2020](https://doi.org/10.1145/3372224.3419191).
- [Braverman, Krauthgamer, and Yang, “Universal Streaming of Subset Norms”](https://arxiv.org/abs/1812.00241).
- [Karnin, Lang, and Liberty, “Optimal Quantile Approximation in Streams”](https://arxiv.org/abs/1603.05346).
