# API: Coco

Status: `Ready`

> Warning: This API is kept for backward compatibility. It uses specialized substring semantics and does not match structured sketch interfaces.

## Purpose

Flow-size estimator over arbitrary key spaces, extended with substring-based
subset aggregation.

## Type/Struct

- `Coco<H = DefaultXxHasher>`
- `CocoBucket`

## Constructors

```rust
fn new() -> Self
fn init_with_size(w: usize, d: usize) -> Self
```

`new()` builds a `1024 x 4` table: 4096 buckets, 128 KiB, plus a heap allocation
per occupied bucket for its stored key.

Sizing: `d` is the number of arrays an insert scans, and the paper recommends
2 to 4 — larger `d` spreads small flows better at the cost of a longer scan per
packet. `w` is the buckets per array, and it is the parameter to raise: bucket
count drives the collision rate, and the sketch can attribute mass to at most
`w * d` distinct keys at once, so size it against the number of heavy flows you
expect to resolve.

## Insert/Update

```rust
fn insert(&mut self, key: &str, v: u64)
```

The SIGCOMM '21 update: the `d` mapped buckets are scanned for `key` first and a
match absorbs `v` directly; otherwise the whole increment lands in the smallest
of them and that bucket's key is replaced with `key` with probability `v / val`.
When several buckets share that smallest value the paper draws one of them
uniformly at random, which matters most on a fresh table where every mapped
bucket still holds `0`.

## Query

```rust
fn recorded_flows(&self) -> impl Iterator<Item = (&str, u64)>
fn group_by<F>(&self, project: F) -> HashMap<String, u64>

fn estimate_key(&self, key: &str) -> u64
fn estimate_projected<F>(&self, partial_key: &str, project: F) -> u64
fn estimate_with_udf<F>(&self, partial_key: &str, udf: F) -> u64
fn estimate_substring(&self, partial_key: &str) -> u64
```

Section 4.3 answers a partial-key query in two steps, and `recorded_flows` plus
`group_by` are those two steps. Step 3 builds a `(Full Key, Size)` table over
the recorded full-key flows, which is what `recorded_flows` yields; an insert
leaves a key in at most one bucket, so no key comes back twice. Step 4 is
`SELECT g(k_F), SUM(Size) ... GROUP BY g(k_F)`, which is `group_by(g)`.

Reach for `group_by` when you want the whole result table: it folds every
recorded flow onto its partial key in one `O(w * d)` pass, where asking
`estimate_projected` for each partial key separately costs one such pass per
group. A single known partial key is still cheaper through `estimate_projected`,
which allocates nothing.

`estimate_key` is the paper's point query: the sum of the `d` mapped buckets
that currently hold `key`, in `O(d)`. Because every increment is attributed to
exactly one bucket, summing it over the observed keys returns the total inserted
mass.

`estimate_projected` is the paper's partial-key query. Section 4.3 defines it as
`SELECT g(k_F), SUM(Size) ... GROUP BY g(k_F)`, where `g` maps a full key to a
partial key; `project` is that `g`. The table is scanned in `O(w * d)` and a
bucket counts only when its stored full key projects exactly to `partial_key`.

`estimate_with_udf` is the same scan behind a general predicate over the
(full key, partial key) pair, for matches a projection cannot express.

`estimate_substring` sums every bucket whose stored key *contains*
`partial_key`. Containment is not a key projection, so `estimate_substring("k1")`
also collects `k10`, `k11`, and `k100`. Reach for `estimate_key` or
`estimate_projected` unless substring matching is what you actually want.

## Merge

```rust
fn merge(&mut self, other: &Coco<H>)
```

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error>
```

ASAPv1 MessagePack, kind_id `0x0c 0x00`. The metadata carries the hash spec plus
the table geometry as `rows` (`d`) and `cols` (`w`); the payload is
`[keys, values]`, two dense row-major arrays of `rows * cols` entries. A free
bucket is msgpack `nil`, so an inserted `""` key stays distinct from a bucket
that holds nothing. Both methods require `H: HashProfile`.

A zero dimension, a table whose bucket count disagrees with `w`/`d`, a payload
length that disagrees with the declared geometry, and a `nil` key carrying a
non-zero value are all rejected on both sides. `Coco` also derives serde.

Row `i` hashes at seed index `i`, so the wire covers `1 <= d <= 20`
(`MATRIX_MAX_ROWS`, the seed list length): past that, row `i` and row `i + 20`
map every key to the same bucket. A wider table is refused on both sides.

## Examples

```rust
use asap_sketchlib::Coco;

let mut sk = Coco::init_with_size(1024, 4);
sk.insert("region=us|id=1", 3);
let _ = sk.estimate_key("region=us|id=1");
let _ = sk.estimate_projected("region=us", |full| {
    full.split('|').next().unwrap_or(full)
});
```

## Caveats

- `estimate_substring` matches by containment, which over-attributes across keys that prefix one another.
- Replacement behavior is probabilistic.

## Multi-core (OctoSketch)

> **Feature gate:** requires `octo-runtime` for the runtime itself.

```rust
let plan = CocoOctoPlan::new(1024, 2);
let config = OctoConfig { threshold: plan.threshold().clone(), ..OctoConfig::default() };
let out = run_octo(&inputs, &config, plan.clone(), || plan.aggregator());
out.parent.sketch.estimate_key("flow::7");
```

`CocoOctoWorker` keeps a table of the same shape with one-byte counters and
ships a `<key, counter>` message when a bucket reaches τ — the bucket's key
*after* the election, so a losing arrival promotes the incumbent instead of
itself. `CocoOctoAggregator` replays each message through `insert(key, value)`,
this sketch's own insertion logic, which is what appendix C of the OctoSketch
paper prescribes. Mass is conserved exactly; residency churns faster than a
single-threaded pass, because a promoted batch of τ takes a bucket with
probability `τ/val` rather than `1/val`. Keys are rendered with
`flow_key_string`, and that rendering is what `estimate_key` must be asked for.
See `docs/api/api_octo.md`.

## Relation to the paper

This module is the basic sketch of section 4.1: the match-then-victim scan, the
whole increment landing in one bucket, and the replacement probability taken
against the post-increment value.

- **Tie-breaking.** Section 4.1 ends the victim rule with "If multiple buckets
  share the same smallest size value, randomly select one to update", and this
  module does.
- **Depth.** This module defaults to 4 arrays, the top of the range section 3.2
  recommends.

## Status

Ready.
