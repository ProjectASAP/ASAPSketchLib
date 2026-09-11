# E2E Testing Harness & Conformance Kit

Every sketch in `asap_sketchlib` is validated end-to-end against exact ground
truth from seeded synthetic streams, and every approximate answer is judged
against **its own family's** error bound.

New sketches go through the conformance batteries — copy an adapter from
`tests/conformance_kit.rs` — and then assert their actual guarantee with the
matching spec from `tests/common/specs.rs`. The batteries are a floor, not the
guarantee: passing them says a sketch is not grossly broken, not that it meets
its theorem.

## Layout

| Path | Purpose |
| --- | --- |
| `common/mod.rs` | Seeded stream generators (`zipf_u64`, `uniform_u64`, `normal_f64`, `exponential_f64`, adversarial `log_uniform_f64`, `duplicate_heavy_f64`, `monotonic_f64`, `outside_in_ordering`), exact truth trackers (`FreqTruth`, `NumericTruth`), assertion helpers |
| `common/specs.rs` | One error model per metric — `CountMinSpec`, `CountSketchSpec`, `SecondMomentSpec`, `KllRankSpec`, `RelativeQuantileSpec` (+ `DdRankConvention`), `CardinalityConfidenceSpec`, `SamplingConfidenceSpec`, `PrioritySampleSpec` — plus the three acceptance rules and the trial-unit rules that say which one applies |
| `common/conformance.rs` | Capability traits + standard conformance batteries (the floor) |
| `conformance_kit.rs` | Reference adapters: established sketches running through the kit (copy these) |
| `e2e_frequency.rs` / `e2e_cardinality.rs` / `e2e_quantiles.rs` | Deep per-family suites, each against its own theorem |
| `e2e_matrix_instances.rs` | Every built-in `(storage, hashing path)` instance of CountMin / Count / CMSHeap / CSHeap, plus counter-width edges |
| `e2e_numeric_types.rs` | Every `NumericalValue` type through `KLL<T>`, `KLLDynamic<T>` and `DDSketch::add<T>` |
| `e2e_windows.rs` | Every `EHSketchList` variant in an `ExponentialHistogram`, every `TumblingWindow` payload, and `MicroCM`'s in-cell sliding window |
| `e2e_composition.rs` | `HashSketchEnsemble`, `UnivMonQ`'s config surface, and the portable sketch+heap facade |
| `e2e_nitro.rs` | Every Nitro ingestion path — row-level `CountMin`/`Count`, `NitroBatch::insert` / `insert_cached_step`, and the bare `Vector2D<u32>` target — under one sampling model, plus merge, seeding, saturation and serde/context continuation. **All Nitro E2E behaviour lives here**; the sampler's own structural tests are unit tests in `src/common/structure_utils.rs` and `src/sketch_framework/nitro.rs`, where the cursor and skip counter are visible without a public accessor |
| `e2e_frameworks.rs` | Hydra's subpopulation lattice and UnivMon composition |
| `e2e_heavy_hitters.rs` | Space-Saving's error sandwich, `min_count` ceiling and Stream-Summary lists under sustained eviction, plus CocoSketch and Elastic through the batteries and the heavy-hitter properties no battery models |
| `e2e_membership.rs` | Bloom: no false negative, exact union, and the delivered false-positive rate against its own sizing |
| `e2e_octo.rs` | OctoSketch delta-promotion invariants, the `octo-runtime` pipeline, and conformance to the paper's Theorems 1-4 and its sketch-merge baseline |
| `e2e_experimental.rs` | The remaining `feature = "experimental"` sketches: KMV, UniformSampling, EHUnivOptimized, CountMinHll |
| `bug_verification.rs` | Regression tests for fixed wrong-query-results bugs |
| `spec_self_tests.rs` | The specs checking themselves — the median's bad-row threshold, the binomial tails it selects, the four-row counter-example behind it, and the simultaneous `kappa` search. Its own binary because a `#[test]` in `common/specs.rs` runs once per suite that says `mod common;` |

## Onboarding a new sketch

1. **Pick the batteries that match what your sketch promises.**

   | Your sketch estimates… | Battery | Trait to impl |
   | --- | --- | --- |
   | Per-key frequency | `frequency_battery` | `FrequencyOps<K>` |
   | …never underestimates (CMS-style) | same, `one_sided: true` in `FrequencySpec` | — |
   | …with signed/turnstile updates | `turnstile_battery` | `SignedFrequencyOps<K>` |
   | Set membership | `membership_battery` | `MembershipOps<K>` |
   | Distinct count | `cardinality_battery` | `CardinalityOps` |
   | Quantiles of a numeric stream | `quantile_battery` | `QuantileOps` |
   | Mergeable shards | `merge_equivalence_battery` | add `MergeOps` |

2. **Write one adapter struct** implementing the trait(s) by delegating to
   your public API. See any adapter in `tests/conformance_kit.rs`.

3. **Run the battery** with production-shaped dimensions, as a smoke test:

   ```rust
   conformance::quantile_battery(
       "MyNewSketch",
       || MyAdapter(MySketch::with_defaults()),
       &values,
       QuantileSpec::default(),
   )
   .assert_ok();
   ```

4. **Assert your sketch's actual bound** with the matching spec from
   `common/specs.rs`, evaluated at the parameters the instance reports:

   ```rust
   let spec = CountSketchSpec::new(sketch.rows(), sketch.cols());
   spec.assert_contract("MyNewSketch", &truth, |k| sketch.estimate(k), &context);
   ```

   If no spec fits, add one — one spec per metric, with the formula, the
   per-check failure probability, and the citation in its doc comment. Do not
   reuse another family's.

5. **Add deeper, sketch-specific checks** (merge semantics, windowing,
   serialization) as a test in the matching `e2e_*.rs` suite.

## Rules

- All randomness flows through seeded generators; no `rand::rng()` in tests,
  and no sketch constructed by a wall-clock-seeded constructor. Stream seeds
  and sketch seeds are separate and both are pinned.
- Ground truth is tracked exactly while generating — never re-derived from
  another approximation.
- Tolerances are **computed from the instance's own parameters**, not written
  down. A tolerance that does not move with `k`, `w` or the precision is not
  that family's bound.
- Theory-backed tests are named `*_satisfies_<theorem_or_bound>`; measured ones
  are named `*_stays_within_the_documented_empirical_band` and record the
  configuration, seed and measured value they came from.
- Never widen a bound to make a failure pass. If the correct bound exposes a
  defect, keep the reproducing test and fix the implementation.
- Batteries accumulate failures into a `BatteryReport`; call `.assert_ok()`
  once at the end so a single run reports every violation, not just the first.

## Running

```bash
cargo test --test conformance_kit        # kit + reference adapters
cargo test --test e2e_matrix_instances   # every storage x path instance
cargo test --test e2e_numeric_types      # every NumericalValue type
cargo test --all-features               # everything, incl. e2e_experimental
cargo run --release --example accuracy_probe --features experimental
                                         # heavy release-only probes
```
