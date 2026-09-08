# API: NitroBatch

Status: `Ready`

## Purpose

Batch-mode geometric sampling wrapper that updates a sketch target.

## Type/Struct

- `NitroBatch<S: NitroTarget>`
- Traits: `NitroTarget`, `NitroMerge`, `NitroEstimate`

## Constructors

```rust
fn init_nitro(rate: f64) -> Self
fn with_target(rate: f64, sk: S) -> Self
fn init_nitro_with_seed(rate: f64, seed: u64) -> Self
fn with_target_and_seed(rate: f64, sk: S, seed: u64) -> Self
```

Sampling is where all of Nitro's randomness lives: which updates reach the
target sketch is drawn from the geometric skip distribution. `init_nitro` and
`with_target` seed that RNG from the OS, so the admitted subset — and therefore
every estimate — differs between runs. The `*_with_seed` forms make it a
deterministic function of the input, which is what lets an accuracy bound be
asserted reproducibly.

## Insert/Update

```rust
fn insert(&mut self, data: &[i64])
fn insert_cached_step(&mut self, data: &[i64])
fn draw_geometric(&mut self)
fn reduce_to_skip(&mut self)
```

## Query

```rust
fn target(&self) -> &S
fn target_mut(&mut self) -> &mut S
fn into_target(self) -> S
fn get_sampling_rate(&self) -> f64
fn get_ctx(&self) -> (usize, f64, usize, usize)
fn estimate_median(&self, value: &DataInput) -> f64
```

## Merge

```rust
fn merge(&mut self, other: &Self)
```

## Serialization

No dedicated serialization API.

## Examples

```rust
use asap_sketchlib::{CountMin, FastPath, NitroBatch, Vector2D};

let base = CountMin::<Vector2D<i32>, FastPath>::default();
let mut nitro = NitroBatch::with_target(0.1, base);
nitro.insert(&[1, 2, 3, 4]);
```

## Accuracy

Each update is admitted with probability `p` and the admitted update writes a
compensating weight of `1/p`. Counters are integers, so that weight is rounded
**stochastically** — `floor(1/p)` plus a `Bernoulli(frac(1/p))` drawn per
admitted update — which makes the estimator unbiased at *every* rate the
constructor accepts, not only at rates whose reciprocal is an integer:

```text
  E[est]   = f
  Var[est] = f ( p·r(1−r) + (1−p)/p ),   r = frac(1/p)
```

Rounding the same way every time would be a bias rather than noise: at
`p = 0.3`, `ceil(1/p) = 4` puts every estimate 20% high. When `1/p` *is* an
integer the fraction is zero, no draw is made, and the emitted weights are
identical to the unrounded ones.

## Caveats

- Works with targets implementing `NitroTarget`.
- Sampling introduces intentional approximation; the variance above is the
  price, and it is on top of whatever the wrapped sketch contributes.
- `with_target` seeds from the OS. Use `with_target_and_seed` /
  `init_nitro_with_seed` when a result has to reproduce.

## Status

Ready for batch sampling workflows.
