# The E2E Testing Harness and Conformance Kit

This document explains how the end-to-end testing harness is designed and how
to use it when adding or changing a sketch. The short version, in two parts:

1. **Every public instance is driven by seeded synthetic data and compared
   against exactly-known ground truth.** Which instances those are, and which
   test covers each, is enumerated in
   [`docs/e2e_coverage_matrix.md`](./e2e_coverage_matrix.md) — that document,
   not this one, is the authority on what is and is not covered.
2. **Every approximate answer is judged against its own family's bound.** The
   conformance batteries are a shared floor for capabilities a sketch has in
   common with others; they are not where a family's guarantee is asserted.
   That lives in the per-metric specs in
   [`tests/common/specs.rs`](../tests/common/specs.rs).

Ad-hoc assertions alone are not an acceptable bar for a new sketch, and neither
is a battery pass on its own.

Related reading:

- [`docs/e2e_coverage_matrix.md`](./e2e_coverage_matrix.md) — instance → test → bound
- [`tests/README.md`](../tests/README.md) — quick onboarding recipe
- [`tests/common/specs.rs`](../tests/common/specs.rs) — the error-model specs
- [`tests/common/conformance.rs`](../tests/common/conformance.rs) — the capability kit
- [`tests/conformance_kit.rs`](../tests/conformance_kit.rs) — reference adapters

## Why it exists

Sketches are approximate by design, which makes them easy to get subtly wrong.
Three properties make this harness effective where ordinary unit tests are not:

1. **Ground truth is exact.** Generators track the true frequency vector,
   order statistics, entropy, and cardinality while they emit data. Expected
   values are never derived from another approximation.
2. **Everything is deterministic.** All randomness flows through seeded RNGs.
   A failure today reproduces tomorrow, on any machine.
3. **Tolerances are computed, not written down.** Where theory exists the
   bound is evaluated from the sketch's own configuration and the exact truth —
   never a hand-picked percentage. Where it does not, the tolerance is a
   documented empirical band, the test is *named* as one, and the measurement
   it came from is recorded next to it.

Point 3 is the one that is easy to get wrong in a way that looks fine. Having a
test is not the same as having an accuracy test, and an accuracy test is not
the same as correctly verifying a theoretical guarantee. A Count Sketch checked
against Count-Min's `ε·N` passes comfortably and verifies almost nothing,
because on a skewed stream that bound is enormously looser than the one Count
Sketch actually promises.

The harness earned its keep immediately: its first run surfaced four real
defects — a Nitro estimator that returned zero because inserts and queries
hashed keys differently, a structural under-count from single-row updates, a
skip-draw rounding bug that halved the effective sampling rate, and a portable
DDSketch representative that violated the advertised α bound at bucket edges.
None of these were caught by the existing unit tests.

Tightening the bounds to the ones each family actually promises surfaced more:
a missing `ELASTIC` arm in `EHSketchList::merge` that silently discarded every
bucket merge in an exponential histogram, HLL Classic's accuracy cliff at the
linear-counting switchover, and a structural coupling in `EHUnivOptimized` that
makes its sketch-tier cardinality unrecoverable. See the findings section of
the coverage matrix.

## Architecture

```sh
tests/
├── common/
│   ├── mod.rs               # generators + truth trackers + assertion helpers
│   ├── specs.rs             # per-metric error models + acceptance rules  ← the bounds
│   └── conformance.rs       # capability traits + reusable batteries      ← the kit
├── conformance_kit.rs       # reference adapters (copy these)
├── e2e_frequency.rs         # deep per-family suites…
├── e2e_cardinality.rs
├── e2e_quantiles.rs
├── e2e_matrix_instances.rs  # every (storage, path) instance of the matrix families
├── e2e_numeric_types.rs     # every NumericalValue type through KLL / DDSketch
├── e2e_windows.rs           # every EHSketchList variant + TumblingWindow payloads
├── e2e_composition.rs       # HashSketchEnsemble, NitroBatch, UnivMonQ config, portable facade
├── e2e_frameworks.rs        # Hydra and UnivMon composition
├── e2e_octo.rs              # …the OctoSketch promotion protocol
├── e2e_heavy_hitters.rs     # …Space-Saving, CocoSketch and Elastic
├── e2e_membership.rs        # …the Bloom filter's membership guarantees
├── e2e_experimental.rs      # …the remaining feature-gated sketches
└── bug_verification.rs      # regression tests for fixed defects
```

There are four layers:

**Layer 1 — primitives (`common/mod.rs`).**
Seeded generators produce streams with known shape: `zipf_u64` (heavy heads),
`uniform_u64`, `normal_f64`, `exponential_f64`, and `log_uniform_f64`, whose
values deliberately land on bucket edges of logarithmic mappings. As each
stream is produced, truth accumulates in `FreqTruth` (per-key counts with
exact F0/F1/F2/L2/entropy/top-k) or `NumericTruth` (sorted values exposing
nearest-rank quantiles, CDF, and rank-tolerance value bands). Assertion
helpers turn comparisons into readable failures that print expected vs actual.

**Layer 2 — the error-model specs (`common/specs.rs`).**
One spec per *metric*, not per sketch, because the metric is what differs:

| Spec | Bounds | Formula |
| --- | --- | --- |
| `CountMinSpec` | one-sided additive excess | marginal `e·(N − f)/w` at failure `e^-d`; simultaneous `b·(N − f)/w` with `b = (D/δ)^{1/d}` |
| `CountSketchSpec` | two-sided L2, rank-independent | marginal `sqrt(3/w)·‖f₋ᵢ‖₂` at `P[Bin(d, 1/3) ≥ ⌈d/2⌉]`; simultaneous at the smallest κ with `P[Bin(d,1/κ) ≥ ⌈d/2⌉] ≤ δ/D` |
| `SecondMomentSpec` | F2 from a Count Sketch matrix | `sqrt(2κ/w)`, same median amplification |
| `KllRankSpec` | KLL **maximum** rank error over a q grid | `ε(k) = 2.446 / k^0.9433` — an Apache DataSketches *characterization fit*, not a theorem about this code |
| `RelativeQuantileSpec` + `DdRankConvention` | DDSketch relative value error | `α + ULP slack` vs the exact order statistic **of that implementation's own rank convention** |
| `CardinalityConfidenceSpec` | HLL / KMV | `z · σ_rel`, σ derived exactly, the tail a normal approximation |
| `SamplingConfidenceSpec` | Nitro | `z · sqrt(f(p·r(1−r) + (1−p)/p))`, `r = frac(1/p)` |
| `PrioritySampleSpec` | `UniformSampling` | `len = ⌈n·rate⌉` exactly; `Var[mean] = (σ_N²/m)(N−m)/(N−1)` |

### The statistical unit is part of the spec

A bound `P[error > B] ≤ p` speaks about **one** draw of the randomness the
estimator is built on. Turning `n` observed checks into a binomial tail at `p`
requires `n` independent draws of *that* randomness — and most natural
batteries are not:

- several `q` off one KLL share one compaction history;
- several keys off one Count-Min or Count Sketch share one hash;
- rising checkpoints on one HLL or KMV are nested;
- for HLL, KMV and counter matrices a shard merge is *exact*, so the merged
  reading is literally the same number as the single pass.

So `Tally` offers three acceptance rules and every call site must pick the one
matching what it collected:

| Rule | Valid when |
| --- | --- |
| `assert_none` | structural facts, and **simultaneous** bounds already union-bounded over the whole battery |
| `assert_independent_binomial` | each recorded check is a fresh seed — sketch, hash or sampling |
| `assert_rate_at_most` | one fixed realisation, pinned at the guarantee's own marginal probability. A regression pin, not a tail test |

`TEST_LEVEL = 1e-6` and `SIMULTANEOUS_LEVEL = 1e-3` are both fixed before the
run, so the number of tolerated violations cannot be adjusted after seeing the
result.

The usual way to make a battery over one sketch legitimate is to **reduce it to
one outcome first**: the sketch's maximum rank error over the whole `q` grid, or
whether any probed key broke a simultaneous bound. That single outcome, repeated
over independent seeds, is a binomial.

Keeping these apart is deliberate. There is no shared
`QuantileSpec { rank_tol }` that KLL and DDSketch both use, because they do not
promise the same thing: a correct KLL can return a value 100× off on a
heavy-tailed stream and still be within its rank guarantee, and a correct
DDSketch has no rank guarantee at all. For the same reason the two DDSketch
implementations do not share a truth helper: `DDSketch::get_value_at_quantile`
answers `sorted[ceil(q·n) − 1]` while the portable `DdSketch::quantile` answers
`sorted[floor(q·(n−1))]`, so `DdRankConvention` carries the choice and each is
scored on the question it actually answers.

### Six statuses, not two

The coverage matrix labels every row `theorem`, `asymptotic model`, `empirical`,
`structural`, `regression`, or `gap`. Two of those distinctions are easy to
lose:

- an **exactly derived standard deviation** is not a tail bound. `z · σ` for HLL
  or KMV becomes a failure probability only under a normal approximation, so
  those rows are `asymptotic model`;
- an **empirical fit imported from another implementation** is not a theorem.
  KLL's `ε(k)` is DataSketches' characterization constant, so its tests are
  named `..._characterization`, never `..._theorem`.

**Layer 3 — the conformance kit (`common/conformance.rs`).**
A sketch describes *what it guarantees* by implementing small capability
traits; batteries translate those guarantees into checks. Traits:

| Trait | Promise |
| --- | --- |
| `FrequencyOps<K>` | point frequency estimates per key |
| `SignedFrequencyOps<K>` | weighted/turnstile ingestion (+/− updates) |
| `MembershipOps<K>` | set membership per key |
| `CardinalityOps` | distinct count over opaque byte keys |
| `QuantileOps` | quantiles of a numeric stream |
| `MergeOps` | in-place merge from a same-config instance |

Batteries consume traits and a tolerance spec, accumulate every violation
into a `BatteryReport`, and report all failures at once via `.assert_ok()`:

| Battery | Checks |
| --- | --- |
| `frequency_battery` | dense-key accuracy; one-sided sketches must never underestimate and stay within `(1 + rel_tol)` |
| `merge_equivalence_battery` | shard-merged sketch ≡ single-pass sketch within slack |
| `turnstile_battery` | `+500 / −200 → ~300`; full cancellation → 0 |
| `membership_battery` | a fresh sketch rejects every probe; every inserted key reports present; the measured false-positive rate over disjoint probes stays at or under `max_fpp` |
| `cardinality_battery` | unique-stream accuracy at a checkpoint; re-ingesting seen keys must not move the estimate |
| `quantile_battery` | estimates land inside rank-tolerance value bands across the standard q grid |

The kit's own specs (`FrequencySpec`, `MembershipSpec`, `CardinalitySpec`,
`QuantileSpec`) carry loose smoke-test tolerances. They exist so a new sketch
can be wired up in one adapter and immediately checked for gross breakage —
they are a floor, not the guarantee. A sketch's actual bound is asserted in its
suite with the matching spec from Layer 2.

**Layer 4 — suites.**
`conformance_kit.rs` wires established sketches through the kit as reference
adapters. The `e2e_*.rs` suites add depth the kit deliberately does not
attempt: serialization round trips, window semantics, framework composition
(Hydra fan-out, tumbling windows, sliding histograms), and cross-implementation
parity between core types and their portable wire twins.

A composition framework can also *be* a battery subject, in the same suite that
already covers it. `e2e_frameworks.rs` runs a single-column Hydra — where the
grid does the keying and each cell holds one counter — through the standard
batteries once per counter family it can host (CM, Count Sketch, HLL, KLL),
alongside the depth no battery models: the `2^D - 1` fan-out across the
subpopulation lattice checked against exact per-subpopulation truth, wildcard
marginals reconciled against the cells beneath them, exact shard-merge
equality, wire round trips for every counter variant, and subkey injectivity
for delimiter-laden values. Suites grow with their family; a framework that
outgrows a smoke test deepens in place rather than splitting off.

What does split is a suite held together by something other than its subject.
`e2e_experimental.rs` groups sketches by cargo feature; Space-Saving, CocoSketch
and the Elastic sketch sit in `e2e_heavy_hitters.rs` instead, organised by what
they are: heavy-hitter sketches, which answer *which flows are big* from a flow
key kept beside every counter. None of the three is feature-gated, so the
default-feature run gets all of them. `e2e_octo.rs` holds the multi-threaded
Octo variants of Coco and Elastic, in its own `heavy_hitters` module, beside
the rest of the promotion protocol.

## Onboarding a new sketch

Worked example. Suppose you added `MyHeavyHitter` that tracks per-key counts,
supports decrements, and merges.

### 1. Choose batteries matching your documented promises

If your docs say "returns per-key counts", you implement `FrequencyOps`. If
they promise "counts never underestimate" (Count-Min style), set
`one_sided: true`. Only claim what you document — the battery enforces the
spec you select, not what the sketch happens to do.

### 2. Write one adapter struct

```rust
mod common;

use common::conformance::{self, FrequencyOps, SignedFrequencyOps};

struct MyAdapter(MyHeavyHitter);

impl FrequencyOps<i64> for MyAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.estimate(&DataInput::I64(*key)) as f64
    }
}

impl SignedFrequencyOps<i64> for MyAdapter {
    fn ingest_weighted(&mut self, key: &i64, weight: i64) {
        self.0.insert_many(&DataInput::I64(*key), weight);
    }
}
```

Adapters exist so the kit stays stable while sketch constructors vary. Keep
them thin — no logic beyond translating types.

### 3. Run production-shaped batteries

```rust
#[test]
fn my_heavy_hitter_passes_conformance() {
    let stream = common::zipf_u64(60_000, 2048, 1.1, /* seed */ 9001);
    let mut truth = common::FreqTruth::default();
    for k in &stream { truth.observe(*k as i64); }

    let spec = conformance::FrequencySpec { one_sided: false, rel_tol: 0.05, abs_tol: 20.0 };
    conformance::frequency_battery(
        "MyHeavyHitter",
        || MyAdapter(MyHeavyHitter::with_dimensions(5, 4096)),
        // ^ construct at dimensions users actually deploy, not toy sizes
        &stream.iter().map(|v| *v as i64).collect::<Vec<_>>(),
        &truth,
        spec,
    )
    .assert_ok();

    conformance::turnstile_battery("MyHeavyHitter", || MyAdapter(MyHeavyHitter::with_dimensions(5, 4096)), 42)
        .assert_ok();
}
```

Use realistic parameters. A sketch tuned to pass a 16-column toy grid tells
you nothing about behavior at 4096 columns under Zipf traffic; several real
bugs only appeared at deployment-shaped dimensions.

### 4. Add depth where the sketch is unusual

Anything not covered by a battery belongs in the matching `e2e_*.rs` suite:
window semantics, merge order-independence, wire-format parity with the
portable type, heavy-hitter recall targets, and so on.

## Tolerance policy

**Compute the bound; do not write it down.** Evaluate it from the sketch's own
configuration and the exact truth, through the matching spec. A tolerance that
does not move when `k`, `w` or the precision moves is not that family's bound:
a hard-coded 0.02 rank tolerance passes identically at `k = 64` and `k = 800`,
so it cannot notice `k` failing to reach the compactors at all.

**Name empirical bands as empirical.** Theory-backed tests are named
`*_satisfies_<theorem_or_bound>`; measured ones are named
`*_stays_within_the_documented_empirical_band`, and the doc comment records the
configuration, the stream seed, and the number actually measured. An empirical
band is never described as a theoretical bound.

**Never widen a bound to make a failure go away.** If the correct bound exposes
an implementation defect, keep the reproducing test and fix the implementation —
or, when the behaviour is a documented limitation rather than a bug, pin it
with *both* an upper and a lower guard so that fixing it later fails the test
and forces the documentation to be updated instead of going stale. Two tests
in this suite do exactly that: `hll_classic_switchover_band_stays_within_the_documented_empirical_band`
and `eh_univ_optimized_sketch_tier_cardinality_is_documented_as_unrecoverable`.

**Do not use a fudge factor.** `1.5 * bound` and `alpha * 1.05` are not bounds;
the second one accepts results that break the advertised guarantee by 5%. If a
looser constant is genuinely needed, change the constant *inside* the derivation
(e.g. Chebyshev's κ) and state the failure probability it now implies.

**Fixed seeds are part of the contract.** Stream seeds and sketch seeds are
separate things and both are pinned. A sketch whose only constructor seeds from
the wall clock cannot be used in an accuracy test — add a seeded constructor.
Every failure message prints the seeds, the configuration, and the measured
error against its bound.

## Rules and anti-patterns

- No unseeded randomness anywhere in tests — no `rand::rng()`, no wall-clock
  seeding, no implicit RNG inside a constructor. `KLL::init_kll_with_seed`,
  `KLLDynamic::init_kll_with_seed`, `KllSketch::with_seed`,
  `NitroBatch::with_target_and_seed` and `UniformSampling::with_seed` exist for
  this. Coco-style nondeterministic implementations are tested statistically
  over bounded ranges instead.
- A probabilistic guarantee is about randomness the test cannot resample: the
  library fixes its hash seed table, so counting how many keys clear a bound
  under *one* hash is not a measurement of the theorem's failure probability.
  Where independent trials are needed, resample the *key population* and say so;
  where the API genuinely cannot expose the dimension, say that instead of
  claiming the theorem was verified.
- Never derive expected values from another approximation. Truth comes from
  `FreqTruth`/`NumericTruth`, full stop.
- Do not bypass public APIs in tests to "fix" them. The Nitro estimator
  shipped broken precisely because its tests re-implemented estimation with
  matching hashes instead of calling the public method.
- Beware generator bugs masquerading as sketch bugs: an early draft drew a
  fresh random number inside a `binary_search_by` comparator, corrupting the
  whole distribution. When accuracy looks wrong, first verify the stream's
  empirical shape matches its name.
- Batteries report everything; call `.assert_ok()` once rather than asserting
  inline so a single CI run shows the complete picture.

## Running

```bash
cargo test --all-features --locked           # full matrix incl. experimental
cargo test --test conformance_kit            # kit + reference adapters only
cargo test --test e2e_matrix_instances       # every storage x path instance
cargo test --test e2e_numeric_types          # every NumericalValue type
cargo run --release --example accuracy_probe --features experimental
                                             # heavy release-only probes
```

CI enforces format, clippy (`-D warnings`, all features), and the full test
matrix on every PR.
