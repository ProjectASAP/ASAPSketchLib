# API: Common Heap Utilities

Status: `Shared`

## Purpose

Common heavy-hitter heap helper used by multiple sketches/frameworks.

## Type/Struct

- `HHHeap`

A capacity-bounded min-heap of `HHItem` beside an index from key digest to heap
position, so a keyed update is a lookup and a sift rather than a scan. The index
is patched through each sift: `update` costs one hash, one map probe and
`O(log k)` index writes, and its rate does not fall as `k` grows. See
[HHHeap acceleration](../hhheap_acceleration.md) for the measurements.

## Constructors

```rust
fn new(k: usize) -> Self
fn from_heap(other: &HHHeap) -> Self
```

## Insert/Update

```rust
fn update(&mut self, key: &DataInput, count: i64) -> bool
fn update_heap_item(&mut self, key: &HeapItem, count: i64) -> bool
fn clear(&mut self)
```

## Query

```rust
fn find(&self, key: &DataInput) -> Option<usize>
fn find_heap_item(&self, key: &HeapItem) -> Option<usize>
fn heap(&self) -> &[HHItem]
fn len(&self) -> usize
fn is_empty(&self) -> bool
fn capacity(&self) -> usize
```

## Merge

No dedicated merge method on `HHHeap`; reconciliation is sketch-specific.

## Serialization

No dedicated byte API helpers. Derived serde carries the heap and its capacity;
the key index is derived data and is rebuilt on load rather than stored.

## Examples

```rust
use asap_sketchlib::{HHHeap, DataInput};

let mut hh = HHHeap::new(8);
hh.update(&DataInput::Str("u1"), 10);
assert!(hh.find(&DataInput::Str("u1")).is_some());
```

## Caveats

- Key ownership conversion follows `DataInput`/`HeapItem` behavior from [Common Input Types](./api_common_input.md).
- `find` returns a heap position, which the next `update` may move. Read it
  before mutating, not after.
- Counts come from the caller, so `update` accepts a count that falls as well as
  one that rises; the resident re-sifts either way.

## See Also

- [Common Module (Canonical)](./api_common.md)
- [Common Input Types](./api_common_input.md)

## Status

Canonical shared heap utility.
