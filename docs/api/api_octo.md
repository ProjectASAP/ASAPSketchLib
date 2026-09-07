# API: OctoSketch

Status: `Ready`

## Purpose

Delta-promotion primitives for multi-threaded sketch updates, based on
the OctoSketch architecture (NSDI 2024). Child sketches maintain small
counters and emit compact deltas when a promotion threshold τ is
reached; a parent sketch absorbs those deltas at full precision.

The paper's three ideas map onto this module as:

| Paper | Here |
| --- | --- |
| Idea 1 — change-based updates (§3.2) | `OctoWorker` emits one counter at a time |
| Idea 2 — adaptive resource allocation (§4.3) | `OctoAdaptiveThreshold` drives a shared `OctoThreshold` |
| Idea 3 — reconstructed data structures (§3.2) | `CmWorkerSketch` / `CountWorkerSketch` drop key storage and use one-byte counters; only the `*TopK*` aggregators hold a heap |

### Two Usage Levels

- **Low-level (this page)**: Call `insert_emit_delta` / `apply_delta`
  directly on sketch structs. You manage threads, channels, and
  scheduling yourself.
- **Turnkey runtime**: Use `OctoRuntime` or `run_octo` for a ready-made
  multi-threaded pipeline. See [Runtime API](#runtime-api) at the bottom.

## Delta Types

Defined in `src/sketches/octo_delta.rs`.

```rust
pub struct CmDelta    { pub row: u32, pub col: u32, pub value: u32 }
pub struct CountDelta { pub row: u32, pub col: u32, pub value: i32 }
pub struct HllDelta   { pub pos: u32, pub value: u8 }
pub struct DdDelta    { pub index: i32, pub value: u64 }

pub struct KeyedCmDelta    { pub key: HeapItem, pub delta: CmDelta }
pub struct KeyedCountDelta { pub key: HeapItem, pub delta: CountDelta }

pub struct CocoDelta { pub key: String, pub value: u64 }
pub enum ElasticDelta {
    Heavy { key: String, value: u32, eviction: bool },
    Evicted { key: String, votes: u32 },
    Light(CmDelta),
}
```

Indices are `u32`, so any geometry the sketch itself supports can be
addressed. The keyed forms carry the flow key the paper's 4-tuple
message includes, which is what lets an aggregator keep a heavy-hitter
heap the workers no longer maintain.

`CocoDelta` and `ElasticDelta::Heavy` are §4.4's *"Handling counters with
flow keys"* message: a sketch that stores a flow key beside every counter
ships the pair, and the aggregator replays it through the parent's own
insertion logic. They carry no cell index, unlike every other delta here,
because the parent re-derives one — CocoSketch's victim choice and
Elastic's bucket contest depend on what the *parent's* buckets hold, not
on where the worker put the key.

`ElasticDelta`'s extra field and extra variant are **not** in the paper.
See [The Elastic eviction flag](#the-elastic-eviction-flag) for what they
are, why the paper does not reach this case, and where the mechanics come
from.

### Promotion Thresholds

τ is a runtime value, not a constant. The constants below are only the
defaults used when a worker is built without an explicit threshold.

| Sketch | Default | Rule |
| --- | --- | --- |
| CountMin | `CM_PROMASK` = 31 | Emit and clear when a counter reaches τ |
| Count | `COUNT_PROMASK` = 31 | Emit and clear when `\|counter\|` reaches τ |
| DDSketch | `DD_PROMASK` = 4 | Emit and clear when a bucket reaches τ |
| HyperLogLog | `HLL_PROMASK` = 0 | Emit when `\|2^C' - 2^C\| >= 2^τ`; never cleared |
| UnivMon | `UNIVMON_PROMASK` = 64 | As Count, with τ halved per layer |
| CocoSketch | `COCO_PROMASK` = 31 | Emit the bucket's `<key, counter>` and clear when the counter reaches τ |
| Elastic | `ELASTIC_PROMASK` = 31 | One τ for both halves: heavy votes and light counters alike |

`DD_PROMASK` is much lower on purpose: a bucket that never reaches τ
never reaches the aggregator at all, so DDSketch loses its sparse tail
rather than merely lagging. Size it against your samples-per-bucket, and
read `DdWorkerSketch::held_back` to bound the rank error it costs.

`HLL_PROMASK = 0` makes the parent's registers bit-identical to a
single-threaded sketch at any cardinality — stronger than the paper's
Theorem 4, which only guarantees equality above `2·α_m·m²·2^(τ-2)`. The
cost is that every register improvement is sent.

In practice 0 is the only HLL threshold worth using below that
precondition. A register the worker has held back reads at the parent as
an *empty* bucket rather than a low one, so the harmonic mean the
estimator is built on collapses: over 50k distinct keys, τ=4 estimates
about 3.2k. `HllOctoWorker::with_threshold` panics outright on a τ no
register gain could ever reach (`>= max_hll_threshold(precision)`),
because such a worker promotes nothing at all and leaves the parent
empty rather than merely lagging.

### Choosing τ from an accuracy target

```rust
pub fn threshold_for_error(epsilon: f64, l1: f64, k_prime: usize) -> u32
```

Equation 4 of the paper, `τ = εL1/k'`. `k_prime` is the number of
workers one flow may reach: 1 under `OctoPartition::HashByKey`, the
worker count under `OctoPartition::RoundRobin`.

### Shared, adjustable τ

```rust
let tau = OctoThreshold::new(31);
tau.get(); tau.set(64); tau.increase(1); tau.decrease(1);
```

`OctoThreshold` is an `Arc<AtomicU32>` clamped to `1..=MAX_PROMASK`
(127). The ceiling is a *signed* one-byte worker counter, and it is the
same for every sketch on purpose: one shared threshold serves workers of
different kinds, so a τ that meant 200 in one and 127 in another would
leave the aggregator's controller a band in which raising τ changes
nothing. Clone it into every worker and into `OctoConfig::threshold` so
the controller and the workers refer to the same value.

## CountMin Delta API

Available on `CountMin<S, RegularPath, H>` and `CountMin<S, FastPath, H>`
where `S::Counter = i32`.

```rust
fn insert_emit_delta(&mut self, value: &DataInput, emit: &mut impl FnMut(CmDelta))
fn insert_emit_delta_with_threshold(&mut self, value: &DataInput, threshold: u32, emit: &mut impl FnMut(CmDelta))
fn insert_emit_keyed_delta_with_threshold(&mut self, value: &DataInput, threshold: u32, emit: &mut impl FnMut(KeyedCmDelta))
fn apply_delta(&mut self, delta: CmDelta)
```

`insert_emit_delta` inserts a key and, for each row counter that reaches
τ, emits a delta carrying that counter's value and clears it — Algorithm
1 of the paper. What the child still holds plus what the parent received
always reconstructs a single-threaded pass.

### CountMin Delta Example

```rust
use asap_sketchlib::{CountMin, RegularPath, DataInput, Vector2D};
use asap_sketchlib::sketches::octo_delta::CmDelta;

let mut child = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(3, 4096);
let mut parent = CountMin::<Vector2D<i32>, RegularPath>::with_dimensions(3, 4096);

let key = DataInput::U64(42);
child.insert_emit_delta(&key, &mut |delta: CmDelta| {
    parent.apply_delta(delta);
});
```

## Count Sketch Delta API

Same shape as CountMin, on `Count<S, RegularPath, H>` and
`Count<S, FastPath, H>` where `S::Counter = i32`. Counters are signed, so
the threshold is applied to `|counter|` (§4.4).

```rust
fn insert_emit_delta(&mut self, value: &DataInput, emit: &mut impl FnMut(CountDelta))
fn insert_emit_delta_with_threshold(&mut self, value: &DataInput, threshold: u32, emit: &mut impl FnMut(CountDelta))
fn insert_emit_keyed_delta_with_threshold(&mut self, value: &DataInput, threshold: u32, emit: &mut impl FnMut(KeyedCountDelta))
fn apply_delta(&mut self, delta: CountDelta)
```

## HyperLogLog Delta API

Available on all `HyperLogLogImpl<Variant, Registers, H>` variants.

```rust
fn insert_emit_delta(&mut self, obj: &DataInput, emit: &mut impl FnMut(HllDelta))
fn insert_emit_delta_with_threshold(&mut self, obj: &DataInput, threshold: u8, emit: &mut impl FnMut(HllDelta))
fn insert_emit_delta_with_hash(&mut self, hashed_val: u64, emit: &mut impl FnMut(HllDelta))
fn insert_emit_delta_with_hash_and_threshold(&mut self, hashed_val: u64, threshold: u8, emit: &mut impl FnMut(HllDelta))
fn apply_delta(&mut self, delta: HllDelta)
```

`apply_delta` applies a max-update: the parent register is set to
`max(current, delta.value)`.

## DDSketch Delta API

```rust
fn bucket_index_for(&self, value: f64) -> Option<i32>
fn apply_delta(&mut self, delta: DdDelta)
```

`bucket_index_for` exposes the logarithmic mapping so a worker can hold
one-byte counters over the same bucket space; it returns `None` for any
value `add` would itself drop. `apply_delta` advances `count` exactly and
advances `sum`/`min`/`max` with the bucket's representative value — the
same α-bounded estimate a deserialize-and-recompute produces.

## UnivMon Delta API

```rust
fn bottom_layer_for(&self, key: &DataInput) -> usize
fn apply_layered_delta(&mut self, delta: &LayeredCountDelta, fidelity: UnivMonDeltaFidelity)
fn set_total_weight(&mut self, weight: usize)
fn mark_candidates_incomplete(&mut self)
fn candidates_complete(&self) -> &[bool]
```

```rust
pub struct LayeredCountDelta {
    pub layer: u32,
    pub key: HeapItem,
    pub delta: CountDelta,
    pub worker_id: u32,
    pub weight_total: u64,
}
```

An insert reaches layers `0..=bottom_layer_for(key)`, which is a pure
function of the key's hash, so a worker selects exactly the layers a
single-threaded insert would. `apply_layered_delta` writes the counter,
re-reads the layer's estimate and updates that layer's heap — Algorithm
2, per layer.

**Where the speed comes from, and where it stops.** A worker's per-insert
work is a hash and a few one-byte increments; the L2 accumulator, the
median estimate and the heavy-hitter heap all move to the aggregator,
which touches them once per *promotion* rather than once per insert.

One aggregator serves every worker, so the pipeline's ceiling is the
aggregator's rate divided by how many deltas each insert produces — which
makes τ the parameter that decides throughput, and is why §4.3 has the
aggregator drive it. At a low τ an insert produces several deltas, the
aggregator does more heap work than a single-threaded `UnivMon::insert`
would, and the pipeline is slower than one core; the accuracy cost of
raising τ stays negligible well past the default, so the trade is
throughput against messages rather than against error.

`cargo run --release --example octo_throughput_probe` measures the three
stages separately on your hardware and prints the host it ran on.

The aggregator's own ceiling is `HHHeap`, not the delta protocol: its
rate falls off as `1/heap_size`, because the heap rebuilds its whole
position index on every accepted update. Single-threaded UnivMon pays the
same cost on every insert rather than every promotion.

**The threshold is scaled per layer.** Layer L only receives the keys that
survive L coin flips, so it carries roughly `n / 2^L` of the stream. One
threshold across the whole pyramid is sized for layer 0 and starves the
deep layers outright — and those are exactly the layers the recursive
estimator leans on for cardinality. `univmon_layer_threshold(base, layer)`
halves τ per layer with a floor of 1; measured on a 60k Zipf stream with
12 layers, a flat τ=31 emptied layers 10 and 11 and collapsed the
cardinality estimate from 3387 to 187, while the scaled rule reproduced
the single-core estimate of 3387. Measured by
`a_flat_threshold_starves_the_deep_univmon_layers`.

Three pieces of state a delta stream would otherwise lose are handled
explicitly:

- **L2 per row.** `CountL2HH` carries `l2[row] += new² − old²` on every
  counter write, so `CountL2HH::apply_delta` does the same fix-up and a
  parent fed only deltas reports the same `get_l2` as one fed the stream.
- **Total weight.** `bucket_size` is what g-sum queries divide by and it
  cannot be recovered from thresholded counters, so each delta carries the
  emitting worker's running total. The aggregator keeps the newest report
  per worker and sums them — no extra messages, and the total trails by
  only what arrived after each worker's last promotion.
- **Candidate completeness.** See the caveats below.

The worker mirrors `CountL2HH`'s cell mapping through the shared
`l2hh_cell_for_row`, not a copy of it: a UnivMon layer slices one
128-bit hash into per-row columns and takes each row's sign from a
different high bit, under a per-layer seed, so a hand-written copy would
be easy to drift.

## Compact Worker Sketches

```rust
pub struct CmWorkerSketch { /* Vec<u8> */ }
pub struct CountWorkerSketch { /* Vec<i8> */ }
pub struct DdWorkerSketch { /* HashMap<i32, u8> */ }
pub struct L2hhWorkerSketch { /* Vec<i8>, one per UnivMon layer */ }

pub struct CocoWorkerSketch { /* Vec<Option<String>> + Vec<u8> */ }
pub struct ElasticWorkerSketch { /* heavy buckets + a CmWorkerSketch light layer */ }
```

A worker counter is cleared the moment it reaches τ, so it never exceeds
one byte. Against the 32-bit counters a full `CountMin` uses that is the
paper's 4× memory saving, and the workers keep no flow-key storage at
all. Use these when you drive the protocol yourself; `CmOctoWorker` and
friends already do.

The two keyed-bucket workers are the exception to "no flow-key storage".
Idea 3 removes a *redundant* key store — a heap the aggregator can rebuild
— but a Coco bucket or an Elastic heavy bucket *is* the key store, and its
key is the only record of what its counter counts. Both workers keep the
keys and shrink only the counters. Because they hold the key, both can
also flush, which the `*TopK*` workers cannot.

---

## Runtime API

> **Feature gate:** The runtime API below requires the `octo-runtime` Cargo feature.
> Enable it with `features = ["octo-runtime"]` in your `Cargo.toml`.
> This pulls in `core_affinity` and `crossbeam-channel` as dependencies.

### OctoConfig

```rust
pub struct OctoConfig {
    pub num_workers: usize,                          // default: 4
    pub pin_cores: bool,                             // default: true
    pub queue_capacity: usize,                       // default: 65536
    pub threshold: OctoThreshold,                    // default: CM_PROMASK
    pub partition: OctoPartition,                    // default: HashByKey
    pub adaptive: Option<OctoAdaptiveThreshold>,     // default: None
}
```

Build one with `..OctoConfig::default()` so later fields stay additive.

### OctoPartition

```rust
pub enum OctoPartition { HashByKey, RoundRobin }
```

`HashByKey` — the default and the paper's setting — sends one flow to one
worker, so `k'` is 1 and the additive `k'τ` term in the error bounds is
as small as it can be. `RoundRobin` spreads load perfectly even under a
skewed key distribution, at the cost of `k' = k`.

Note that either way a *counter* is shared by whatever flows hash into
it, and each worker may hold back up to τ of its own share, so the
provable per-counter gap to a single-threaded sketch is `workers · (τ - 1)`. `k'τ`
bounds only the queried flow's own held-back count.

`RoundRobin` also costs `Elastic` its one-sided guarantee: a flow that is
a stable, unflagged resident on one worker while losing the bucket contest
on another reaches the parent unflagged, over light-layer mass an
unflagged bucket never reads, and `Elastic::query` comes back low. See
[The Elastic eviction flag](#the-elastic-eviction-flag). The mode is a
whole-runtime setting, so no individual plan opts out of it.

### OctoAdaptiveThreshold

```rust
pub struct OctoAdaptiveThreshold {
    pub target_queue_len: usize,   // Q, default 10
    pub alpha: f64,                // dead band, default 0.25
    pub interval: Duration,        // default 100µs
    pub min_threshold: u32,
    pub max_threshold: u32,
}
```

The aggregator samples total queue occupancy every `interval`, predicts
the next window with Equation 1 (`Q̂ₜ₊₁ = Qₜ + (Qₜ − Qₜ₋₁)`), and moves τ
by one per Equation 2: down when the prediction falls below
`(1−α)·Q`, up when it rises above `(1+α)·Q`, unchanged in between. Set
`min_threshold` from `threshold_for_error` to hold an accuracy floor.

### Plans: nothing borrowed crosses a thread

A `DataInput` may borrow, and a worker runs on another thread, so the
borrow has to end before the hand-off. An `OctoPlan` is the object that
makes that possible: it stays on the dispatching thread and holds the
geometry, which both of its jobs need — building workers, and converting
an input into the borrow-free payload that actually crosses.

```rust
pub trait OctoPlan: Send + 'static {
    type Worker: OctoWorker;
    fn worker(&self, worker_id: usize) -> Self::Worker;
    fn prepare(&self, input: &DataInput<'_>) -> <Self::Worker as OctoWorker>::Payload;
}
```

`insert` hashes for partitioning and calls `prepare`, both on the calling
thread, so a borrowed key is finished with by the time it returns — the
key's owner never has to outlive the runtime.

What crosses follows from what the worker needs. A worker that only
hashes gets hashes and no copy; one that must store the key gets an owned
copy, made once at the hand-off.

| Worker | Payload | Copies the key |
| --- | --- | --- |
| CountMin, Count | `RowHashes` (`SmallVec<[u64; 8]>`) | no — inline for any realistic row count |
| HyperLogLog | `u64` | no |
| DDSketch | `Option<f64>` | no |
| CountMin/Count top-k | `KeyedHashes` | yes, unavoidably |
| UnivMon | `UnivMonInput` | yes, unavoidably |
| CocoSketch, Elastic | `String` | yes, unavoidably |

CocoSketch and Elastic key on a `String`, so their plans render the input
with `flow_key_string`, which renders every `DataInput` variant: numbers by
`to_string`, bytes as lowercase hex. `EHSketchList` converts differently —
its `ELASTIC` arm renders bytes with `String::from_utf8_lossy`, and its
`COCO` arm drops every input that is not `Str` or `String`. The
`flow_key_string` rendering is what the aggregator stores, so it is what
`Coco::estimate_key` or `Elastic::query` must be asked for. Their payloads
carry no hashes: the heavy half of an Elastic insert can evict a key the
caller never sent, so the worker has to be able to hash for itself anyway,
and hashing there keeps the work off the dispatching thread.

`UnivMonInput` hashes only the layers a key actually reaches. Layer depth
is geometric, so that is nearly always one or two.

The shipped plans are `CmOctoPlan`, `CountOctoPlan`, `CmTopKOctoPlan`,
`CountTopKOctoPlan`, `HllOctoPlan`, `DdOctoPlan`, `UnivMonOctoPlan`,
`CocoOctoPlan` and `ElasticOctoPlan`.
Each takes the same dimensions its worker did, plus an optional shared
`OctoThreshold`:

```rust
let plan = CmOctoPlan::new(4, 4096);
let result = run_octo(&inputs, &config, plan.clone(), || plan.aggregator());
```

Build the parent from the plan. Worker and parent geometry are two
independent arguments and a mismatch is silent: every delta names a row
the parent has, the rows past the worker's stay zero, and Count-Min's
min-over-rows estimate is zero for every key. Every plan has an
`aggregator()`.

### OctoRuntime (Streaming)

```rust
fn new<PF>(config: &OctoConfig, plan: L, parent_factory: PF) -> Self
fn insert(&mut self, input: DataInput<'_>)
fn insert_batch(&mut self, inputs: &[DataInput<'_>])
fn flush(&mut self)
fn read_handle(&self) -> OctoReadHandle<P>
fn close(&self)
fn finish(self) -> OctoResult<P>
```

### Flushing before a query

Between promotions a worker holds every counter still under τ. For
Count-Min and Count that only leaves the parent low - by up to
`workers · (τ - 1)` per cell, since each worker holds its own residue.
For DDSketch, or a HyperLogLog running a positive threshold, an
un-promoted cell is *absent* from the parent rather than lagging — and a
quantile or a cardinality is exactly a statement about which cells
exist, so those queries are wrong without bound rather than within α.

`flush` hands over every residual counter and waits for the aggregator to
apply it, so the parent answers against every input accepted so far. It
is the point at which a stream is handed over for querying. `finish`
flushes too, so a completed run always answers against the whole stream —
a flushed Count-Min parent equals a single-threaded pass cell for cell.

It costs one message per non-empty cell, which is as much as shipping the
sketch, so call it when a query needs to be right rather than on a timer.
Inserting afterwards is fine; a flush does not seal the runtime. After
`close` it returns immediately without draining — `close` has already
asked every worker to flush, but nothing waits for that to land, so read
the result through `finish` rather than a live handle.

`OctoWorker::flush` defaults to doing nothing. The `*TopK*` and UnivMon
workers keep that default on purpose: every delta they send carries the
key that produced it, and a worker keeps no key storage, so a residual
cell cannot be attributed back to a key. Their parents stay low by under
`workers · (τ - 1)` per cell, and - the sharper consequence - a key that
has never promoted is absent from the heap entirely rather than merely
undercounted. A key promotes the first time an increment it caused takes
some row to τ on its worker, which is its τ-th occurrence only while its
cells are collision-free; collisions move it either way.

At the low level, `CmWorkerSketch`, `CountWorkerSketch`, `DdWorkerSketch`
and `L2hhWorkerSketch` each expose `flush` directly.

### run_octo (Batch)

```rust
pub fn run_octo<L, P>(
    inputs: &[DataInput<'_>],
    config: &OctoConfig,
    plan: L,
    parent_factory: impl FnOnce() -> P,
) -> OctoResult<P>
```

### Concrete Worker / Aggregator Pairs

| Sketch | Worker | Aggregator | Delta Type |
| --- | --- | --- | --- |
| CountMin | `CmOctoWorker` | `CmOctoAggregator` | `CmDelta` |
| CountMin + top-k | `CmTopKOctoWorker` | `CmTopKOctoAggregator` | `KeyedCmDelta` |
| Count | `CountOctoWorker` | `CountOctoAggregator` | `CountDelta` |
| Count + top-k | `CountTopKOctoWorker` | `CountTopKOctoAggregator` | `KeyedCountDelta` |
| DDSketch | `DdOctoWorker` | `DdOctoAggregator` | `DdDelta` |
| UnivMon | `UnivMonOctoWorker` | `UnivMonOctoAggregator` | `LayeredCountDelta` |
| HyperLogLog | `HllOctoWorker` | `HllOctoAggregator` | `HllDelta` |
| CocoSketch | `CocoOctoWorker` | `CocoOctoAggregator` | `CocoDelta` |
| Elastic | `ElasticOctoWorker` | `ElasticOctoAggregator` | `ElasticDelta` |

The `*TopK*` pairs hold the pipeline's only heavy-hitter heap, in the
aggregator: each keyed delta updates the parent counter and then the
heap, which is Algorithm 2.

The two keyed-bucket pairs instead replay the key, per §4.4.
`CocoOctoAggregator` calls the weighted `Coco::insert`, so the promoted
mass contests the parent's own buckets and runs the parent's own `v/val`
election. `ElasticOctoAggregator` splits the halves: a `Heavy` message
goes through `Elastic::merge_heavy`, a `Light` message is an ordinary
Count-Min cell delta applied to `sketch.light`, and an `Evicted` message
goes through `Elastic::absorb_evicted`. Neither aggregator keeps a heap:
Appendix C notes both sketches already carry their own heavy-key storage,
which is also why Table 1 gives them a throughput ratio near 1 (1.01× and
0.93×) while still reporting 37.25× and 14.03× better accuracy.

### The Elastic eviction flag

§4.4, *"Handling counters with flow keys"*: for a sketch with a flow key
beside every counter, OctoSketch "will send both the key and the counter
to the aggregator and set the counter to zero if the counter is large
enough. For each `<key, counter>` pair, the aggregator inserts the key
into the sketch using the same insertion logic as the original sketch."
That is the whole rule. It says nothing about an eviction flag, and what
follows is this crate's.

An `Elastic` heavy bucket carries a flag meaning *part of this flow's mass
is in the light layer*, and `Elastic::query` reads the light layer only
when the flag is set. The delta protocol carries it in two places.

1. *The flag rides with the counter.* `ElasticDelta::Heavy` carries the
   worker bucket's flag, and the worker maintains one with the parent's
   semantics — set on takeover by eviction, cleared when seating a
   previously unoccupied slot. The aggregator ORs it into the parent
   bucket only when the arriving key ends up resident there.
2. *The eviction spill travels keyed.* An evicted resident goes to the
   light part under its own key, which a worker cannot express through an
   unkeyed cell delta, so it ships `ElasticDelta::Evicted`. The aggregator
   adds `votes` to the parent's light layer under that key and flags the
   key's bucket if it is resident. The message goes out on *every* worker
   eviction, `votes` of zero included — a promotion zeroes the counter
   while the flow stays resident, so an eviction right after one carries
   nothing but is still the only way the parent hears that this flow's
   remaining mass will now arrive through the light layer.

The ordinary "an arrival that loses a bucket contest spills 1" path is
unkeyed, batched through the light `CmWorkerSketch`, and promoted at τ
like any other Count-Min cell. The keyed spill costs +0.1–0.6% more
worker→aggregator messages at 4 and 8 workers, +7% at the degenerate
1-worker setting.

Measured on a Zipf-1.2 stream (20k keys, 200k packets, 1024 heavy
buckets, 3×4096 light, τ=31, hash-by-key, ARE over the true top-200,
seed 7; seeds 11 and 23 agree): 0.0167 at 1 worker, 0.0071 at 4 and
0.0055 at 8, against 0.0067 for a single-threaded `Elastic`, with no flow
under-estimated at any width.

**Where the flag stops closing the loop.** It closes it only while a flow
visits one worker. Under `OctoPartition::RoundRobin` a flow can be a
stable, unflagged resident on one worker and a perpetual loser of the
bucket contest on another: the second worker's share of its mass goes out
unkeyed as `ElasticDelta::Light`, the first worker's `Heavy` messages
carry `eviction: false`, and the parent seats the flow unflagged over
light counters it will never read. `Elastic::query` then returns less than
the true count, which is the one thing the sketch otherwise never does.
Same stream and geometry as above, seed 7, over all 11,144 flows it
contains rather than the top 200:

| routing | workers | flows under-estimated | worst deficit |
| --- | --- | --- | --- |
| `HashByKey` | 1, 4, 8 | 0 | 0 |
| `RoundRobin` | 1 | 0 | 0 |
| `RoundRobin` | 4 | 384 | 55 |
| `RoundRobin` | 8 | 406 | 75 |

Seeds 11 and 23 put the `RoundRobin` counts at 390 and 401, and 400 and
402, for 4 and 8 workers, and leave `HashByKey` at zero. `HashByKey`, the
default, sends a flow to one worker and keeps the guarantee;
`OctoConfig::partition` is set for the whole runtime, so a plan cannot ask
for it on its own.

### Sharing a threshold across the fleet

```rust
let plan = CmOctoPlan::with_threshold(4, 4096, OctoThreshold::new(31));
let config = OctoConfig {
    num_workers: 8,
    threshold: plan.threshold().clone(),
    adaptive: Some(OctoAdaptiveThreshold::default()),
    ..OctoConfig::default()
};
let result = run_octo(&inputs, &config, plan, || CmOctoAggregator::new(4, 4096));
```

## Caveats

- Counts below τ reach the parent only at a flush. `finish` flushes, and
  `OctoRuntime::flush` does it mid-stream; a worker driven directly at the
  low level must be flushed by its caller. The keyed workers cannot flush
  at all — see above.
- DDSketch loses whole buckets rather than lagging on them, so read it
  only after a flush. Between flushes, keep τ small and check
  `held_back()`; the measured cost of each τ is in the DDSketch caveat
  below.
- Core pinning silently falls back when the platform has fewer cores than
  `num_workers + 1`, and is a no-op outright on Apple Silicon: macOS
  exposes `thread_policy_set(THREAD_AFFINITY_POLICY)` but arm64 rejects
  it, while `core_affinity::get_core_ids` reports a count without testing
  whether pinning works. The throughput probe prints which case you are
  in.
- The runtime's `insert` dispatches from the calling thread, so that
  thread is a serialization point. The paper instead has each worker pull
  from its own NIC queue.
- `insert` after `close` panics.
- A UnivMon layer that thresholds cannot call its candidate set complete,
  so `candidates_complete()` reads false on exactly those layers. Queries
  then take the conservative branch of `heavy_threshold` instead of
  overcounting. Deep layers, whose scaled threshold floors at 1, keep the
  heap's own verdict. At a base threshold of 1 a single worker reproduces
  a single-threaded UnivMon exactly - counters, L2, heaps, total weight
  and g-sum alike.
- DDSketch is the one integrated sketch where delta promotion does not pay
  off. A lagging Count-Min counter is still counted, just low; a DDSketch
  bucket that never reaches τ is absent, and a quantile is a statement
  about where mass sits. On a skewed 200k stream the measured trade is
  τ=1 → exact at 200k messages, τ=2 → 7x ideal error, τ=4 → 19x, τ=8 →
  41x, while periodic merge stays at ideal for 3x the traffic. Prefer
  merge for DDSketch unless τ can be 1.
- The Elastic aggregator ORs the sending worker's flag into a parent bucket
  the arriving key ends up resident in, and flags outright the bucket a
  keyed eviction spill names. That carries the guarantee under
  `HashByKey`. Under `RoundRobin` it does not: a flow unflagged on the
  worker that holds it and spilled unkeyed by the workers that do not
  reads back low — see above.
- A Coco or Elastic worker promotes a key it may not currently hold. Coco's
  losing arrival promotes the *incumbent* of the bucket it hit, and an
  Elastic eviction spills the *evicted* resident, so a message can name a
  key that never appeared in this batch. That is why their payload is the
  key rather than a set of hashes.
- Coco promotion moves mass in batches of τ rather than one at a time, and
  the parent's replacement probability is `v/val`, so a batch is that much
  likelier to take a bucket from its resident. Mass is conserved exactly,
  but which key holds it churns faster than in a single-threaded pass.
  Lower τ if a specific flow's residency matters more than message count.

## Status

Core multi-threaded insertion framework; actively used and tested.
