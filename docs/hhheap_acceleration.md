# HHHeap acceleration

`HHHeap` is the top-k structure behind `CMSHeap`, `CSHeap`, `FoldCMS`, `FoldCS`,
`UnivMon`, and the Octo aggregator. It sits on the insert path of every one of
them, so its cost per update is theirs. This page describes what it does per
update and what that measures at.

Reproduce with `cargo run --release --example hhheap_probe`. Numbers below are
from an Apple M-series laptop, 200k updates drawn from a Zipf(1.1) stream over
20k distinct keys.

## The structure

A capacity-bounded min-heap of `HHItem`, and an index from key digest to heap
position.

- `positions: HashMap<u64, SmallVec<[usize; 2]>>` holds heap indices only.
  Identity is re-checked against `heap[idx].key`, so no key is cloned into the
  index. Two residents colliding on a 64-bit digest is rare enough that the
  inline pair is effectively never spilled.
- `slots: Vec<u64>` runs parallel to the heap array and holds each resident's
  digest, so a sift that moves an element needs no re-hash of its key.
- The map is keyed by a value that is already an xxh3 digest, so it uses
  `DigestHasher` rather than running SipHash over it again.

`CommonHeap` reports movement through `push_back_with`, `replace_root_with` and
`update_at_with`, each taking `&mut impl FnMut(usize, usize)` called once per
swap the sift performs. `HHHeap` patches only the entries a sift touched;
residents sharing a digest sit in the same bucket, so exchanging their two
positions leaves that bucket unchanged and writes nothing.

Per update this is one key hash, one map probe, and `O(log k)` index patches.

The index is derived, so it is not serialized. A decoded heap rebuilds it before
first use.

## What it measures at

`HHHeap::update`, string keys and `u64` keys:

| capacity | string Mups/s | u64 Mups/s |
| --- | --- | --- |
| 8 | 45.0 | 76.0 |
| 32 | 43.6 | 54.5 |
| 128 | 53.0 | 57.4 |
| 512 | 55.5 | 70.0 |
| 2048 | 52.9 | 71.2 |

The rate is flat in `k`, which is the property that matters: capacity is an
accuracy parameter, not a throughput one.

At the sketch level, against a bare `CountMin` at 210.1 Minsert/s:

| `CMSHeap` top_k | Minsert/s | share of time spent in the heap |
| --- | --- | --- |
| 8 | 31.2 | 85.1% |
| 32 | 28.4 | 86.5% |
| 128 | 28.1 | 86.6% |
| 512 | 29.1 | 86.1% |
| 2048 | 27.0 | 87.2% |

The heap dominates the insert - it does a hash, a map probe and a sift against
the sketch's four counter writes - but its share is flat in capacity.

## Why not Space-Saving

A `SpaceSaving` summary gives `O(1)` amortized updates, and this crate ships
one. It is not a substitute here. Its `O(1)` rests on an arrival moving a
counter to the *neighbouring* count bucket, while `HHHeap` takes its counts from
the sketch it is attached to, where an estimate can jump by any amount. The two
also retain different keys: Space-Saving owns its counters and its eviction
rule, so it answers a different question. See
[SpaceSaving](./api/api_space_saving.md).

## Equivalence

`HHHeap` is compared element for element against a reference that rebuilds the
whole index after every accepted update, at capacities 0 through 257, over 30k
updates, on both key forms, and with counts that fall as well as rise - see
`differential` in `src/common/heap.rs`. Which key is seated, which is evicted,
and where either lands are identical. `index_invariants` in the same file pins
the index itself: every resident is reachable at its own position and the index
carries nothing else.

The map is keyed by `DigestHasher` rather than the randomly seeded default, so
the structure carries no run-to-run state at all.

## Compatibility

`HHHeap`'s serialized form is the heap and its capacity. The key index is
derived rather than carried, and a decoded heap rebuilds it before first use.
A named-map encoding that carries an index field still decodes, since the extra
key is skipped; a positional encoding of the three-field form does not. Nothing
in-crate writes the positional form: the portable MessagePack wire for the top-k
sketches carries a `(key, value)` list and rebuilds through `update`, and no
golden covers a top-k envelope.
