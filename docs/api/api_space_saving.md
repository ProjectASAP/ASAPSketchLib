# API: SpaceSaving

Status: `Ready`

## Purpose

Tracks the most frequent keys of a stream in a fixed number of counters. A
monitored key's recorded count never falls below its true count, and exceeds it
by at most that key's own error allowance.

## Type/Struct

- `SpaceSaving<H = DefaultXxHasher>`

The Stream-Summary of the paper's section 3.1: buckets carrying a count, ordered
by count in a doubly linked list, each owning a doubly linked list of the
counters at that count, plus a key index into those counters. A unit arrival
moves one counter to the neighbouring bucket and an eviction takes the head of
the lowest bucket, so `insert` touches a constant number of links whatever the
capacity. A weighted `insert_many` lands further along the bucket list and walks
it to reach its destination, one step per bucket it passes, so only the unit
path is constant-work.

Both lists are arenas of indices rather than pointers. `counters` is allocated
once up to `capacity` and reused in place — an eviction overwrites the victim's
slot — and `buckets` recycles through a free list. The key index is keyed by an
xxh3 digest and hashes it through `DigestBuildHasher` rather than a second
time.

## Constructors

```rust
fn with_capacity(capacity: usize) -> Self
```

`capacity` is the number of keys monitored at once, and it is the only sizing
parameter. `Default` is 1024. Capacity floors at 1.

Size it against the ceiling you can tolerate: with `capacity` counters over a
stream of total weight `N`, the smallest monitored count is at most `N /
capacity`, and no key below that is guaranteed to survive. Choosing a capacity
at or above the number of distinct keys makes the summary exact.

## Insert/Update

```rust
fn insert(&mut self, value: &DataInput)
fn insert_many(&mut self, value: &DataInput, count: u64)
fn bulk_insert(&mut self, values: &[DataInput])
```

A monitored key rises by `count`. An unmonitored key seats itself at
`min_count + count` and carries `min_count` as its error, taking a free counter
while one is left and displacing the smallest monitored key after that. On a
summary that has never merged, `min_count` is zero until every counter is in
use, so a fresh key starts at `count` with no error. `insert_many` with a weight
is the same as repeating `insert` that many times. A `count` of zero is a no-op,
and counts saturate at `u64::MAX` rather than wrapping.

## Query

```rust
fn estimate(&self, value: &DataInput) -> u64
fn upper_bound(&self, value: &DataInput) -> u64
fn error(&self, value: &DataInput) -> u64
fn is_guaranteed(&self, value: &DataInput) -> bool
fn top_k(&self, k: usize) -> Vec<(HeapItem, u64, u64)>
fn entries(&self) -> Vec<(HeapItem, u64, u64)>
fn min_count(&self) -> u64
fn total(&self) -> u64
fn len(&self) -> usize
fn capacity(&self) -> usize
```

For a monitored key, `estimate - error <= truth <= estimate`. For an unmonitored
key `estimate` is zero and the truth is at most `min_count`, which is what
`upper_bound` reports instead; `upper_bound` therefore never reads below the
truth for any key in the stream, and `estimate` does so only for keys the
summary has dropped.

`min_count` is the ceiling on anything unmonitored: the larger of the smallest
count still held, once every counter is in use, and the largest count that has
left the summary through an eviction or a merge. Residency alone does not settle
it — a merge can leave a summary holding fewer keys than its capacity and still
missing keys the other side had already dropped.

`is_guaranteed` holds when `estimate - error > min_count`, so the key outranks
every key the summary discarded — a claim about the stream, not about the
summary.

`top_k` walks the bucket list from the high end and returns `(key, count,
error)` in non-increasing count order.

## Merge

```rust
fn merge(&mut self, other: &Self)
```

Counts for a shared key add. A key held by only one side takes the other's
`min_count` as both extra count and extra error, since that is the most the
other side can say about it. The union is then trimmed back to `capacity`, and
the merged ceiling rises to at least the sum of the two `min_count`s — a key
both sides dropped can have reached that much between them.

The union is ordered by count and broken out of ties by digest and then by key,
so merging the same two summaries always yields the same survivors.

This is **not** equivalent to running one summary over the concatenated streams:
a key both sides evicted cannot be recovered. `merge_equivalence_battery` does
not apply for that reason.

The `min_count` a one-sided key picks up is weight the stream never carried, so
after a merge the counts sum to more than `total` and `estimate / total` is no
longer a frequency.

## Serialization

```rust
fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError>
fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError>
```

These produce/consume the **ASAPv1** wire envelope (kind `0x18 0x00`) — see the
[ASAPv1 wire format spec](../asapv1_wire_format.md). The impl is bounded on
`H: HashProfile`, so a summary built with an unprofiled hasher cannot serialize
at all. `SpaceSaving` also implements `serde::Serialize` / `Deserialize` for
in-Rust use; both forms carry the same state.

A summary travels as `capacity`, `total`, the unmonitored ceiling, and one
`(key, count, error)` triple per monitored key. The bucket and counter lists and
the key index all follow from the triples and are rebuilt on load, so no arena
index reaches the wire and no crafted state can point one out of bounds or into
a loop. `capacity` and the key type are ASAPv1 metadata; the payload is
`[keys, counts, errors, total, floor]`.

Keys are `HeapItem`s, so the key type is a runtime property: the metadata's
`key_type` names the **exact** variant (`"i32"` stays `"i32"`, never widened to
`"i64"`) and the payload's key array is homogeneous in it, because a decoded key
of a different variant would stop matching the caller's `DataInput` and read
zero. A summary whose monitored keys mix variants refuses to serialize, as does
one holding a 128-bit key, which has no MessagePack integer form. An empty
summary serializes with `key_type` `"u64"`.

The triples are emitted in descending count, ties broken by a total order over
the key, so the same triples always produce the same bytes and re-serializing a
decoded summary reproduces them exactly.

Decoding rejects any state the algorithm could not have produced — a foreign
`kind_id`, a hash profile other than the target's, a `key_type` the payload does
not carry, arrays of unequal length, a zero capacity, more entries than the
capacity allows, a counter at zero, an error above its count, or the same key
twice — with an error rather than a panic. A declared capacity is not allocated
up front, so a large one costs nothing until the counters are actually filled.

## Examples

```rust
use asap_sketchlib::{DataInput, SpaceSaving};

let mut top: SpaceSaving = SpaceSaving::with_capacity(1024);
for flow in &flows {
    top.insert(&DataInput::Str(flow));
}

for (key, count, error) in top.top_k(10) {
    println!("{key:?}: {count} (+0/-{error})");
}
```

## Caveats

- No decrement path. Counts only rise, so the summary does not model a
  turnstile stream.
- A key can be evicted and re-enter later. Its count then restarts from the
  minimum rather than from what it had, and its error grows accordingly.
- `estimate` returns zero for a key that is not monitored. That reads below the
  truth for an evicted key; use `upper_bound` when the caller needs a value
  that never does.
- `top_k(k)` allocates and clones `k` keys.
- A weighted `insert_many` is not constant work: both a fresh counter and a
  raise walk the bucket list to their destination.

## Relation to the paper

- **Structure.** Section 3.1's Stream-Summary, with the bucket and counter
  lists as index arenas.
- **Guarantee.** Section 3.2's `count - error <= truth <= count`, and the
  `min_count` ceiling on anything unmonitored.
- **Merge.** Not in the paper; the rule above follows Cormode and
  Hadjieleftheriou's survey treatment and is approximate.

## Status

Ready.
