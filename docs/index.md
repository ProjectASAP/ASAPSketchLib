# Documentation Index

This page is the docs home for `asap_sketchlib`.

## Start Here

- [Project Overview](./project_overview.md)
- [Examples](./examples.md) — HyperLogLog, Count-Min Sketch, and KLL end-to-end
- [Library Map](./library_map.md)
- [Development Guide](./development_guide.md)

## Core References

- [APIs Index](./apis.md)
- [API Pages Directory](./api/)
- [Common API](./api/api_common.md)
- [ASAPv1 Wire Format](./asapv1_wire_format.md) - authoritative byte-level spec for the self-describing sketch envelope mirrored by `sketchlib-go`
- [Message Pack Format](./message_pack_format.md) - how the serialization code is organized (shared envelope + per-sketch `wire.rs`; `portable`/`native` retiring)
- [Advanced Use Cases](./advanced_use_cases.md)
- [HHHeap Acceleration](./hhheap_acceleration.md) - what the top-k heap costs per update, how its digest-keyed position index is maintained through the sift, and the throughput it measures at
- [Test Coverage Map](./tests.md)
- [E2E Testing Harness](./e2e_testing_harness.md) - how the end-to-end suites are built and what a new sketch must satisfy
- [Feature Status](./features.md)
