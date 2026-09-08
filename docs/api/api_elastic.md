# API: Elastic

Status: `Ready`

> Warning: This API is kept for backward compatibility. It does not follow the full structured API parity used by Ready sketches.

## Purpose

Heavy/light split frequency estimator with a heavy bucket plus Count-Min backing sketch.

## Type/Struct

- `Elastic<H = DefaultXxHasher>`
- `HeavyBucket`
- `LAMBDA` — the paper's eviction threshold (8).
- `DEFAULT_LIGHT_ROWS` / `DEFAULT_LIGHT_COLS` — the default light layer (3 x 4096).

## Constructors

```rust
fn new() -> Self
fn init_with_length(l: i32) -> Self
fn init_with_dimensions(bucket_count: i32, light_rows: usize, light_cols: usize) -> Self
```

`new()` builds 8 heavy buckets over a 3 x 4096 light layer; `init_with_length`
takes the bucket count and keeps that same light layer. Both panic on a
non-positive bucket count or an empty light layer.

**Footprint.** The light layer is `rows * cols * 4` bytes of `i32` counters —
48 KiB at the default. Each heavy bucket is 40 bytes plus one heap allocation
for its resident flow id, so the heavy part is `bucket_count * 40` bytes and
change.

**Sizing the heavy part.** Bucket count is set by how many elephant flows you
expect to separate, not by the stream length. Section 3.1.2 gives the elephant
collision rate — the fraction of buckets holding more than one elephant flow —
in closed form:

```
P_hc = 1 - (H/w + 1) * e^(-H/w)
```

for `H` elephant flows over `w` buckets. Ten buckets per expected elephant
(`H/w = 0.1`) puts collisions at 0.46%; a hundred (`H/w = 0.01`) puts them at
0.005%. Collisions are what cost accuracy here, since a colliding elephant gets
evicted into the light layer and can inflate the mouse flows sharing its
counters.

**Sizing the light part.** These dimensions set the error on every flow that is
not resident in the heavy part — evicted elephants and all mice. Section 4.1
recommends depth 1 for speed, on the grounds that the heavy part already
carries the accuracy; the default here is 3, which is more accurate and slower.
Widening matters more than deepening: on a 20k-flow stream that evicted a
50-packet flow, a 1 x 64 light estimated it at 379, a 1 x 512 at 98, and the
default 3 x 4096 at exactly 50.

## Insert/Update

```rust
fn insert(&mut self, id: String)
```

Follows the SIGCOMM '18 insertion rule. A vacant bucket seats the flow. A
matching bucket takes a positive vote. Otherwise the bucket takes a negative
vote, and either the arriving flow goes to the light layer, or — once
`vote_neg >= LAMBDA * vote_pos` — the **resident** flow is evicted into the
light layer with its whole positive vote and the arrival takes the bucket with
`(vote_pos, vote_neg, eviction) = (1, 1, true)`.

### Weighted and merge-time insertion

```rust
fn insert_many(&mut self, id: String, count: i32)
fn merge_heavy(&mut self, id: String, votes: i32, eviction: bool)
fn absorb_evicted(&mut self, id: String, votes: i32)
```

`insert_many` is `insert` with a weight: a matching bucket takes `count`
positive votes, a non-matching one takes `count` negative votes, and a takeover
seats the arrival with `count` of each. `count` of 1 is `insert` exactly.

`merge_heavy` absorbs a `<flow, votes, eviction>` message from another sketch's
heavy part: `insert_many`, plus the sender's flag OR-ed in when the arrival ends
up resident here. The flag travels with the counter it qualifies: the counter
word handed between the two parts *is* the flag. A sender whose bucket is
unflagged holds
that flow's whole mass in its heavy part, so flagging it here would make the
estimate read Count-Min mass belonging to other flows.

`absorb_evicted` absorbs a resident the sender's heavy part evicted, under its
own key. The votes go to the light layer under `id`, and
`id`'s bucket here is flagged if it still holds it. `votes` of zero still flags:
the sender can no longer speak for this flow's heavy part, so whatever it sees
of the flow next arrives through the light layer.

### Overload mode

```rust
fn insert_heavy_only(&mut self, id: String)
```

Section 3.3 adapts to packet rate by letting arrivals "only access the heavy
part, so as to record the information of elephant flows only and discard mouse
flows". Call this instead of `insert` while your input queue is backed up, and
go back to `insert` when it drains; the sketch holds no mode of its own, since
only the caller can see the queue.

Seating and matching behave exactly as in `insert`. The two other cases differ:

- A non-matching arrival takes its negative vote and is then **dropped**. It
  does not reach the light layer.
- On takeover the arrival **inherits the evicted flow's positive vote** rather
  than starting at 1 — the paper's "the flow size of `f'` is set to the flow
  size of `f`" — and the evicted flow's size is **discarded**, not spilled.

The light layer is still read by `query`; it is only never written.

The arrival also **inherits the bucket's eviction flag**, and the negative vote
resets to `0`. The paper's prose names neither: this path writes only the new
key and a zeroed guard, leaving the counter — and so the flag living in its high
bit — untouched, so the arrival takes over the size and the flag together. The
normal path instead sets the flag and resets the size to 1, which is the paper's
case 4 `(f, 1, T, 1)`.

> **This path breaks the one-sided guarantee.** Every other operation on this
> sketch never returns less than the true count. Overload mode drops mouse
> flows outright and loses each evicted flow's accumulated size, so its
> estimates can and do come back low. Mixing it into a stream permanently
> lowers the answers for the flows it dropped.

Measured on a 400k-arrival Zipf(1.1) stream over 20k flows, single core:
33.3 vs 24.4 Mops/s at 64 buckets (1.37x), and 35.9 vs 30.2 Mops/s at 1024
buckets (1.19x). The gain comes from skipping the light layer's 3 hashes and
3 counter writes, so it is largest when collisions are frequent enough to send
arrivals there often.

## Query

```rust
fn query(&self, id: String) -> i32
```

Returns `vote_pos` for a resident flow whose bucket carries no eviction flag,
`vote_pos + light.estimate(id)` when it does, and the light estimate otherwise.
The estimator is one-sided: it never returns less than the true count. Under
the OctoSketch runtime that holds for `OctoPartition::HashByKey` only — see
Multi-core, below.

## Merge

```rust
fn merge(&mut self, other: &Elastic<H>)
fn merge_max(&mut self, other: &Elastic<H>)
```

Both combine the heavy parts bucket by bucket and then combine the light
layers. Elephants stay elephants: a flow holding a bucket on either side keeps
one in the merged sketch, and only the flow that loses a contested bucket is
spilled into the light layer. Both require the same heavy bucket count and the
same light dimensions — merging across different widths appears only in the
technical report's appendix A.8 and is not implemented.

The heavy half follows two sources. The technical report's B.5 combines two
sketches by combining their heavy parts, not by dissolving them: "The heavy
parts are easy to combine: we combine all the heavy parts one by one... For all
light parts, we merge them into one using the merging algorithms, by choosing
the maximum of the corresponding counters." B.5 is describing sharded inputs,
where each flow reaches exactly one sketch, so its heavy parts concatenate and
the table grows. Holding the table at its original width instead needs a rule
for two flows landing on one bucket, and §3.4 has one, stated for compression:
"for the two keys in the buckets, we query their frequencies in the Elastic
sketch, and keep the larger one, and evict the other one into the light part."
Applied across sketches rather than within one, that is the merge below.

Per bucket:

| both sides | result |
| --- | --- |
| both vacant | vacant |
| one vacant | the occupied bucket carries over |
| the same flow | one bucket, `vote_pos` summed |
| different flows | each queried against its own sketch, larger keeps the bucket, loser's `vote_pos` spills to the light layer |

Two choices the paper does not make. `vote_neg` takes the larger of the pair,
which evicts sooner. And every surviving bucket ends up flagged: the flow that
keeps a bucket may have been a mouse on the other side and left mass in that
light layer, and nothing short of trusting the peer's Count-Min rules it out.
Flagging overestimates rather than underestimates, the direction the guarantee
allows.

`merge` combines the light layers with the paper's **Sum merging**, adding them
counter by counter. It is correct whatever the two sketches saw.

`merge_max` uses **Maximum merging**, keeping the larger of each counter pair,
which is what B.5 prescribes. It is tighter, but the light half is only correct
when the two sketches observed **disjoint flow sets**.

> A *mouse* flow both sides saw reads back under `merge_max` as the larger side
> rather than the sum, which underestimates it — exactly what Elastic's
> one-sided guarantee otherwise rules out. Elephants held by both heavy parts
> are summed either way; it is the light half that carries the restriction.
> Reach for `merge` whenever a mouse flow can repeat across sketches.

### What keeping the heavy part buys

Not exact elephants — a merged resident is flagged, so its estimate still
carries Count-Min error. The gain is that elephant mass never enters the light
layer at all, which lowers the collision floor for **every** flow. Measured
over 60k Zipf(1.2) arrivals split across two 64-bucket sketches, mean relative
error per flow, flushing everything into the light layer versus contesting
buckets:

| light layer | elephants, flush / contest | mice, flush / contest |
| --- | --- | --- |
| 1x64 | 70.74 / 15.72 | 447.06 / 148.53 |
| 1x256 | 5.66 / 3.12 | 105.24 / 37.27 |
| 1x1024 | 1.26 / 1.10 | 28.37 / 9.13 |
| 3x1024 | 0.18 / 0.18 | 2.02 / 1.85 |
| 3x4096 | 0.006 / 0.006 | 0.096 / 0.095 |

Neither underestimates anywhere. The mice column moves most, which is the
mechanism showing itself: the win comes from what is kept *out* of the
Count-Min. A generously sized light layer has room for the elephant mass
anyway, and the two converge.

## Growing and shrinking the heavy part

```rust
fn expand_heavy(&mut self)
fn full_bucket_count(&self, t2: i32) -> usize
fn compress_heavy(&mut self, ratio: i32)
```

Section 3.4 starts the heavy part small and doubles it when elephants fill it:
"just copy the heavy part and combine the heavy part with the copy into one",
changing `h(.) % w` to `h(.) % 2w`. Lemma 3.2 — `(i % w) % w' = i % w'` — is
what makes that safe: every resident still hashes to a half that holds it, so
no estimate moves.

The paper's trigger is two thresholds: a bucket is full when its flows all
exceed `T2`, and the table is full when more than `T1` buckets are. Deciding
that is the caller's — `full_bucket_count(t2)` reports the numerator. Call
`expand_heavy` when your own `T1` is crossed.

After a doubling each flow sits in both halves, and the copy in the half it no
longer hashes to is stale. Cleanup is incremental, as the paper describes: an
insert landing on a stale copy drops it and seats the arrival. Buckets nobody
lands on keep their copy, which the paper notes "does not negatively impact the
algorithm".

Stale copies are skipped when a merge flushes the heavy part, so a flow is
spilled once rather than once per half — on both sides of the merge. Two
sketches expanded a different number of times have different bucket counts, so
`merge` and `merge_max` assert rather than silently misalign.

Expansion follows the paper text; the heavy part of a fixed-size implementation
has no equivalent.

`compress_heavy(ratio)` is the reverse, and gives the sketch section 3.4's
"ability to actively release memory when needed". New bucket `j` absorbs old
buckets `j`, `j + w'`, `j + 2w'`, … — the same equal-division grouping as the
Maximum Compression of section 3.2.1, and sound for the same reason, Lemma 3.2.
What differs is what gets merged: "for the heavy part, we merge buckets (key,
vote+, flag, vote−) rather than counters". Each group is resolved by querying
every resident's whole size and keeping the largest; the rest spill their
`vote_pos` into the light layer, where they read back as ordinary non-resident
flows.

`ratio` must divide the bucket count — Lemma 3.2 needs `w' | w` — and
`compress_heavy` asserts rather than quietly misplacing flows.

The paper does not say what happens to the votes of the buckets that lose.
The winner's bucket is carried over untouched, votes included: the pair records
the contests that flow actually fought, and folding in a stranger's negative
votes would distort `vote_neg / vote_pos` and so the timing of its next
eviction. The losers' votes go with them into the light layer as size.

Compressing a table that was expanded first drops the stale copies before
grouping. Without that, a copy whose live twin falls in a different group wins
its own group and becomes a second live entry for the same flow — reachable
whenever the bucket count is not a power of two, for instance 12 buckets
doubled to 24 and compressed by 3.

## Measurement tasks

```rust
fn heavy_hitters(&self, threshold: i32) -> Vec<(String, i32)>
fn heavy_changes(&self, other: &Elastic<H>, threshold: i32) -> Vec<(String, i32, i32)>
```

Section 5 lists six tasks the sketch is meant to serve. Two are implemented
here, both sorted by flow id and both skipping the stale copies an expansion
leaves behind, so no flow is reported twice.

**Heavy hitter detection.** "For this task, we query the size of each flow in
the heavy part. If one's size is larger than the predefined threshold, then we
report this flow as a heavy hitter." Every resident is put through `query`
rather than read straight off `vote_pos`, so a flow that was once evicted picks
up its light-layer remainder. The reference reports on `>=`, and so does this.

**Heavy change detection.** "For two adjacent time windows, we build two
Elastic sketches, respectively... we check all flows in the heavy parts of the
two sketches, and if the size difference of a flow in the two windows is larger
than T, we report it as a heavy change." Residents of both heavy parts are
taken together, so a flow that appears in only one window is still a change,
measured against whatever the other window's light layer holds for it. Each
entry is `(flow, size in self, size in other)`.

### The other four tasks

- **Flow size estimation** is `query`.
- **Cardinality**, **entropy**, and **flow size distribution** are not
  implemented. The paper's estimators for these read the light layer's counter
  value distribution, and the reference builds them on a `d = 1` flat array of
  8-bit counters: `get_cardinality` is linear counting over that array,
  `get_entropy` accumulates `mice_dist[i] * i * log2(i)`, and
  `get_distribution` runs ten EM epochs over the raw counters. This crate's
  light layer is a `rows x cols` Count-Min of `i32`, so none of the three
  carries over without picking a different estimator.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error>
```

ASAPv1 MessagePack, kind_id `0x0b 0x00`. The metadata carries the hash spec plus
both geometries: `heavy_buckets`, and the inlined light layer's `light_rows`,
`light_cols`, `light_counter_type` (`"i32"`) and `light_mode` (`"regular"`). The
payload is `[flow_ids, vote_pos, vote_neg, evictions, stale_copies,
light_counts]`; the four heavy arrays are dense in bucket index order and
`light_counts` is row-major. The light Count-Min is inlined, not nested in its
own envelope. A free bucket is msgpack `nil`, so an inserted `""` flow id stays
distinct from a bucket that holds nothing. Both methods require
`H: HashProfile`.

The light layer is a Count-Min matrix, so it carries that sketch's row bound:
`1 <= light_rows <= 20` (`MATRIX_MAX_ROWS`, the seed list length), refused on
both sides past it.

A zero dimension, a `bktlen` disagreeing with the heavy table, a free bucket
that still names a flow, a payload length that disagrees with the declared
geometry, and a `nil` flow id whose `vote_pos` is non-zero are all rejected on
both sides. `Elastic` also derives serde.

## Examples

```rust
use asap_sketchlib::Elastic;

let mut sk = Elastic::init_with_length(8);
sk.insert("flow".to_string());
let _ = sk.query("flow".to_string());
```

## Caveats

- Light-layer counters are fixed at `i32`; the paper's 8-bit counters are not
  selectable.
- `insert_heavy_only` is lossy by design and does not keep the one-sided
  guarantee the rest of the API holds.
- Under the OctoSketch runtime the one-sided guarantee needs
  `OctoPartition::HashByKey`, the default. `RoundRobin` breaks it; see
  Multi-core, below.
- String-centric API (`String` in insert/query).
- Lifecycle and parity differ from structured sketches.

## Multi-core (OctoSketch)

> **Feature gate:** requires `octo-runtime` for the runtime itself.

```rust
let plan = ElasticOctoPlan::new(256, 3, 4096);
let config = OctoConfig { threshold: plan.threshold().clone(), ..OctoConfig::default() };
let out = run_octo(&inputs, &config, plan.clone(), || plan.aggregator());
out.parent.sketch.query("flow::7".to_string());
```

Appendix C of the OctoSketch paper keeps both halves in the worker and promotes
them differently. `ElasticOctoWorker` holds a heavy table with one-byte vote
counters and a one-byte Count-Min light layer: the heavy part ships
`<flow, votes, eviction>`, the light part ships an ordinary unkeyed cell delta
for arrivals that lose a bucket contest, and an eviction ships the *evicted*
flow keyed, as `ElasticDelta::Evicted`. `ElasticOctoAggregator` routes those to
`merge_heavy` — the parent runs its own bucket contest, and may evict a different
flow than the worker did — `sketch.light` directly, and `absorb_evicted`.

The flag and the keyed eviction go beyond §4.4's `<key, counter>` rule.
`docs/api/api_octo.md`, "The Elastic eviction flag", describes the protocol.

Route with `OctoPartition::HashByKey`, the default. The flag only reaches the
parent alongside a heavy counter, so it closes the loop while a flow visits one
worker. Under `RoundRobin` a flow reaches every worker, and one that is a
stable unflagged resident on one while losing the bucket contest on another is
seated at the parent unflagged over light-layer mass an unflagged bucket never
reads: `query` comes back low. Measured counts are in `docs/api/api_octo.md`,
"The Elastic eviction flag".

Keys are rendered with `flow_key_string`, and that rendering is what `query`
must be asked for. See `docs/api/api_octo.md`.

## Relation to the paper

- **Eviction threshold.** Section 3.1.1 evicts once `vote-/vote+ >= lambda`,
  and this module does.
- **Bucket shape.** This module is the basic version of section 3.1, one flow
  per bucket, not the software version of section 4.3.
- **Takeover.** The normal-path swap is the paper's case 4, `(f, 1, T, 1)`.

## Status

Ready.
