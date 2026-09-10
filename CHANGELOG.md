# Changelog

All notable changes to `asap_sketchlib` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
For pre-1.0 (`0.y.z`) releases we follow the Cargo convention of treating `y`
as the major component: bumping `y` signals a breaking change, bumping `z`
signals a backwards-compatible change.

## [Unreleased]

### Added

- **`Vector3D<T>`**, filled in from a stub. A `rows x cols` grid whose every
  cell is a contiguous `depth`-length record, addressed by the same
  `MatrixFastHash` machinery as `Vector2D`. Reach for it when a cell is made
  of several mutually coupled fields — when the estimator or the update reads
  or writes them as a unit. `Vector2D<[T; N]>` has the identical memory
  layout; what `Vector3D` adds is a `depth` chosen at run time, which is what
  a HyperLogLog precision or a sub-window count needs and what a const generic
  cannot express. See
  [Common Structures](docs/api/api_common_structures.md#when-to-reach-for-vector3d-instead-of-vector2d).
- **`CountMinHll`** (`experimental` feature). Grouped distinct counting: given
  a stream of `(key, distinct_value)` pairs, how many distinct values has this
  key seen? A Count-Min-shaped grid whose every cell is a small HyperLogLog,
  held in one contiguous `Vector3D<u8>` register plane. `Hydra` with
  `HydraCounter::HLL` answers the same query on a stable API, with one
  heap-allocated HLL per cell.
- **`MicroCM`** (`experimental` feature), a sliding-window frequency sketch
  following MicroscopeSketch (Zhao et al., KDD 2023). Each cell holds `T + 2`
  byte-wide pixels and one shared zoom exponent, so the counters stay narrow
  and the exponent buys the range instead of the counters being widened for
  the largest count they might ever hold. Count-based and time-based windows
  are one `SubWindowClock` type. Not yet checked against the paper's published
  measurements.

### Changed

- **`Vector3D`'s dimensions changed meaning, and so did its `serde` shape.**
  The stub took `init(layer, row, col)` and serialized `layer`/`row`/`col`;
  it now takes `init(rows, cols, depth)` and serializes `rows`/`cols`/`depth`.
  The arity is unchanged, so existing calls still compile while meaning
  something different, and previously serialized bytes will not decode. The
  stub had no callers in this crate and no sketch used it, but the type was
  publicly exported, so this is a breaking change to a public API.
- `Vector3D::init` now rejects a zero dimension instead of building a
  container whose bucket accessors panic on first use, and deserialization
  rejects a payload whose data length disagrees with `rows * cols * depth`.
- `HyperLogLog`'s classic estimator body is now a shared function that
  `CountMinHll` also uses, so the two cannot drift. HyperLogLog's own numbers
  are unchanged.

## [0.3.0] - 2026-09-07

Breaking release. Adds `Bloom` and `SpaceSaving`, completes ASAPv1 wire
coverage for every implemented sketch, and corrects the Nitro row sampler.
The wire format and the `serde` shape of several types changed; see
**Changed** below for the payloads and types affected.

### Added

- **`UnivMonQQuery::ordered_query_diagnostics`**, returning
  `OrderedQueryDiagnostics`. A read-only view of the heavy set the ordered CDF
  actually used, the mass it credited to it, and the number of retained
  occurrence samples backing the residual — the `E_H`, `P_hat_R` and `m_R` of
  the documented bound `sup_x |F_hat(x) - F(x)| <= 2 E_H + P_hat_R * eps_R`.
  None of the three was reachable from outside the crate, so the bound could
  only be checked in the diffuse special case where the heavy set is empty; it
  is now verified in full. Purely observational: it reports state the CDF
  construction already computes, and changes no answer and no wire format.
- **`cs_heap_count`**, naming the `f64 -> i64` conversion `CSHeap` uses when it
  writes a Count Sketch estimate into its top-k heap. The conversion saturates
  at `i64::MAX` / `i64::MIN` (and maps `NaN` to `0`) rather than wrapping, which
  matters for the `i128`-backed instances: they accept `insert_many(key, i128)`
  and really can hold counts past `i64::MAX`. Making it a named, documented
  function rather than an inline cast means the choice cannot change silently.

- **`Bloom`, a partitioned Bloom filter.** `rows` slices of `cols` bits, one
  slice per hash function, over a new packed `BitMatrix` that implements
  `MatrixStorage` — so the filter probes the same `rows x cols` shape
  `CountMin` does and a membership query is the row-wise minimum, which over
  single bits is their AND. `with_capacity(n, p)` caps the slice count at
  `BLOOM_MAX_SLICES` (the seed list's length, since two slices seeded from the
  same entry are identical) and solves the slice width for that count, so
  `predicted_fpp` never claims a rate the filter cannot deliver; each slice is
  rounded up to a power of two, which removes the column fold's modulo bias.
  Every door holds the slice bound: `with_dimensions` panics past it and both
  serialized forms reject it, so a filter that cannot be stored cannot be built
  either. Total bits are capped at `BLOOM_MAX_BITS`, and a non-finite target
  rate is rejected. Union is exact, so the filter shards without loss. The
  `BloomMode` marker tags the hash path on the wire, so bytes written by one
  path do not decode into the other.
- **`SpaceSaving`, a fixed-counter heavy-hitter summary.** The paper's
  Stream-Summary: count-ordered buckets in a doubly linked list, each owning a
  doubly linked list of its counters, plus a key index — so a unit arrival moves
  one counter to the neighbouring bucket and an eviction takes the head of the
  lowest bucket, constant work at any capacity. Both lists are index arenas
  rather than pointers. A monitored key is sandwiched by its own error, and
  `upper_bound` never reads below the truth for any key in the stream: the
  summary carries the largest count known to have left it, so the ceiling
  survives a merge that leaves it holding fewer keys than its capacity. Counts
  saturate rather than wrap, and `merge` picks the same survivors on every
  run. Only the monitored `(key, count, error)` triples reach the wire; the
  arena is rebuilt on load and a payload that does not describe a valid summary
  is rejected.
- **`BitMatrix` in `common::structures`.** A packed one-bit-per-cell grid behind
  the `MatrixStorage` interface. It carries `words`, `rows` and `cols` on the
  wire and recomputes the word stride and column mask on load; out-of-range
  coordinates panic rather than aliasing into a neighbouring row.
- **`DigestHasher` in `common::hash`.** A `Hasher` for `u64` keys that are
  already digests: it replaces the full byte-wise hash with a finalizing mix,
  which is what a table index still needs once the digests come from a fixed
  seed list. `DigestBuildHasher` is its `BuildHasher`.
- **ASAPv1 wire serialization for `Bloom` (`0x17 0x00`) and `SpaceSaving`
  (`0x18 0x00`).** `serialize_to_bytes` / `deserialize_from_bytes` on each,
  through the shared envelope, with the metadata derived from the hasher's
  `HashProfile` as Count-Min's and HLL's are. Bloom's payload is the packed
  words and the insert count; its wire covers the geometries `with_capacity`
  produces, on both sides, so it never emits bytes it would refuse to read.
  Space-Saving's payload is the monitored `(key, count, error)` triples plus
  `total` and the dropped-count ceiling — the bucket list, counter arena and key
  index are rebuilt on load, so no crafted payload can point an arena index out
  of bounds or into a cycle. Its `key_type` names the exact `HeapItem` variant
  and is never widened, since the variant is part of a key's identity while the
  digest is blind to it; a mixed-variant or 128-bit-keyed summary refuses to
  serialize. Entries are emitted in a defined order, so equal summaries encode
  to equal bytes. Both payloads are specified in `docs/asapv1_wire_format.md`
  §3.4 and §3.5.
- **`membership_battery` in the conformance kit.** A new `MembershipOps`
  capability with the exact no-false-negative check, a false-positive-rate
  ceiling and a band around `predicted_fpp`, since the existing batteries are
  all frequency- or numeric-shaped.
- **ASAPv1 wire payloads for every remaining implemented sketch.** `CMSHeap`
  (`0x03 0x00`), `DDSketch` (`0x05 0x00`), `Hydra` over each of its five
  counters (`0x07 0x00`-`0x07 0x04`), `CSHeap` (`0x0a 0x00`), `Elastic`
  (`0x0b 0x00`), `Coco` (`0x0c 0x00`), `UniformSampling` (`0x0d 0x00`), `KMV`
  (`0x0e 0x00`), `UnivMon` (`0x10 0x00`), `UnivMon Optimized` (`0x11 0x00`),
  `ExponentialHistogram` (`0x13 0x00`), `EHSketchList` (`0x14 0x00`),
  `CountL2HH` (`0x19 0x00`) and `UnivMon-Q` (`0x1a 0x00`) each serialize through
  the shared envelope, with a closed metadata schema and a positional payload
  specified in `docs/asapv1_wire_format.md` Section 3. A nested sketch is
  inlined rather than wrapped in an envelope of its own, except where the nested
  algorithm is data rather than a type: an `EHSketchList` carries the variant's
  own `kind_id`, metadata block and payload block with the framing stripped, so
  that variant's own decoder validates it. Every decoder fails closed — a
  declared capacity never sizes an allocation, every geometry is bounds- and
  overflow-checked before anything is sized from it, and a container whose
  order does not survive a rebuild has its emitted order pinned, so a decoded
  sketch re-serializes byte-identically.
- **Seeded constructors for the sketches whose randomness was wall-clock or
  OS-drawn.** `KLLDynamic::init_with_seed` / `init_kll_with_seed`,
  `NitroBatch::with_target_and_seed` / `init_nitro_with_seed`,
  `KllSketch::with_seed` (and the free `new_sketchlib_kll_with_seed`) and
  `HydraKllSketch::with_seed`, each mirroring the unseeded constructor beside
  it. A KLL's compaction coin and a Nitro batch's geometric skip draw are the
  only randomness those types hold, so fixing them makes the sketch a
  deterministic function of its input — which is what reproducible replay,
  cross-process parity and any assertion of an accuracy bound require.
  `KLLDynamic` stores its seed so `clear()` re-seeds from it, as `KLL` already
  did; the seed is not part of the wire format, since it describes how a sketch
  was built rather than what it holds, so encoded bytes are unchanged.
  `HydraKllSketch::with_seed` shares one seed across every cell.

### Changed

- **BREAKING (wire format):** `serialize_to_bytes` / `deserialize_from_bytes`
  emit and read the ASAPv1 envelope for every sketch the `kind_id` registry
  marks implemented. Bytes produced by the earlier serde-derived form of those
  types **do not decode**; there is no legacy read path, since a new encoding
  takes a new `kind_id` rather than a payload version field. Golden byte-vector
  fixtures exist for HLL, Count-Min, Count Sketch and compact KLL only, so the
  other kinds have no cross-language drift guard yet and `portable` stays in
  place until they do.
- **BREAKING (ASAPv1 envelopes for matrix sketches past 20 rows):** every
  payload whose rows are seeded per row now rejects a matrix with more rows than
  `SEEDLIST` has seeds, on both the encode and the decode side. Seed indices
  wrap at the seed list, so row `r` and row `r + 20` draw the same seed, fold
  every key into the same column and hold identical cells — a wider matrix buys
  storage and a hash, not independence. The new public `MATRIX_MAX_ROWS` names
  the bound and `BLOOM_MAX_SLICES` is now defined as it, so the two names share
  one definition. Covered: Count-Min (`0x02 0x00`), CMSHeap (`0x03 0x00`),
  Count Sketch (`0x04 0x00`), Hydra's grid and matrix counters (`0x07 0x01` /
  `0x07 0x02`), CSHeap (`0x0a 0x00`), Elastic's light layer (`0x0b 0x00`),
  Coco's table (`0x0c 0x00`), UnivMon (`0x10 0x00`) and UnivMon Optimized
  (`0x11 0x00`) layers, and CountL2HH (`0x19 0x00`) — Bloom (`0x17 0x00`)
  already drew this line. UnivMon-Q (`0x1a 0x00`) is unaffected: its rows are
  bit fields of one 128-bit hash, bounded by that budget instead. A wider
  counter matrix is still buildable in memory; it no longer serializes, and
  Bloom refuses one at construction.
- **Made `HHHeap::update` independent of capacity.** The key index was rebuilt
  in full after every accepted update, cloning each resident's key, so the top-k
  structure behind `CMSHeap`, `CSHeap`, `FoldCMS`, `FoldCS`, `UnivMon` and the
  Octo aggregator cost `O(k)` per insert. It now holds heap indices only,
  re-checking identity against the heap, and patches them through each sift:
  `CommonHeap` gained `push_back_with`, `replace_root_with` and
  `update_at_with`, which report every swap to a caller-supplied closure, while
  `push` and `update_at` delegate with a no-op and are unchanged for every other
  caller. A parallel `slots` vector carries each resident's digest so a sift
  re-hashes nothing, and the index is keyed by `DigestHasher` rather than
  running SipHash over a value that is already an xxh3 digest. Measured on a
  Zipf(1.1) stream, `HHHeap::update` goes from 4.75 to 45.0 Mups/s at capacity
  8 and from 0.01 to 52.9 at capacity 2048 — flat in `k` rather than halving
  with each doubling — and `CMSHeap::insert` at top_k=2048 goes from 0.009 to
  27.0 Minsert/s. Retention is unchanged: a differential test compares the heap
  element for element against the rebuild implementation at capacities 0 through
  257 over 30k updates on both key forms.
- **BREAKING (positionally encoded `HHHeap` and `UnivMon` state):** `HHHeap` no
  longer serializes its key index, which is derived data rebuilt on load. The
  serialized form went from `{heap, positions, k}` to `{heap, k}`. A named-map
  encoding written by an earlier version still decodes, since the extra key is
  skipped; a positional encoding of the three-field form does not. Nothing
  in-crate writes the positional form — the portable MessagePack wire for the
  top-k sketches carries a `(key, value)` list and rebuilds through `update`,
  and no golden covers a top-k envelope.
- **BREAKING (`serde` shape of `Nitro`, and therefore of `Vector2D` and any
  sketch embedding one):** `Nitro::rounding_state` is now a serialized field,
  appended after `mask`, so a sketch resumed from a decode continues its weight
  sequence instead of restarting the rounding stream from a constant. It
  carries `#[serde(default)]`, so a payload written before this field existed
  decodes unchanged in both map and array encodings. A *new* payload read by an
  *older* build decodes only in a map encoding (`rmp_serde::to_vec_named`,
  `serde_json`), where the unknown key is skipped; a positional encoding gains
  one trailing element. **The ASAPv1 wire format is unaffected**: no ASAPv1
  payload has ever carried Nitro state — `CountMin` and `Count` envelopes
  serialize the counter slice and the metadata block only — so every
  `serialize_to_bytes` / `deserialize_from_bytes` byte string and every golden
  fixture is unchanged.
- **BREAKING (`Nitro::admit_rows`, `Nitro::table_cursor`,
  `Nitro::skip_table_len`, `NitroBatch::table_cursor`,
  `NitroBatch::skip_table_len`):** removed from the public API.
  `admit_rows` is now `pub(crate)` and takes a
  `SmallVec<[(usize, u64); MATRIX_MAX_ROWS]>`; the cursor and table-length
  accessors existed only so an integration test could read private state, and
  the tests that used them are now unit tests in the modules that own it. The
  seeded constructors — `enable_nitro_with_seed`, `init_nitro_seeded`,
  `with_target_and_seed`, `init_nitro_with_seed` — are unaffected: they answer
  a real need to reproduce a sampling run.
- **`CountMin::fast_insert_nitro` and `Count::fast_insert_nitro` no longer
  allocate per observation.** Each call built a `Vec` for the admitted rows.
  They now collect into a `SmallVec` inlined to `MATRIX_MAX_ROWS`, so no matrix
  the hash family supports reaches the heap, at any sampling rate. The hash is
  computed once per observation and both paths share the one admission walk in
  `Nitro::admit_rows`.
- **BREAKING (external implementors of `MatrixFastHash`):** the trait gained a
  required `row_hash(row, mask_bits, mask)` method, split out of
  `col_for_row` so storages can decode against precomputed parameters and
  skip `% cols` for power-of-two column counts. The three in-crate impls
  (`MatrixHashType`, `u64`, `u128`) are updated; downstream code implementing
  this trait directly must add the method. Ship in the next `0.y` bump
  (Cargo convention: `y` is the major component pre-1.0).

### Fixed

- **The row-level Nitro sampler never advanced its skip-table cursor, and its
  schedule ignored the configured rate.** `Nitro::draw_geometric` advanced the
  cursor with `(idx + 1) & mask` where `mask` was the table's *length*
  (`0x10000`) rather than `length - 1`; `x & 0x10000` is zero for every `x`
  below `0xFFFF`, so `idx` stayed at 0 for the life of the sketch. Every skip
  was the same distance, and the stochastic-rounding draw derived from that
  cursor was a constant — which meant the `p = 0.3` weight bias the previous
  release claimed to fix was still `ceil(1/p)` on this path. The same function
  also read a table pre-divided by `ln(0.99)`, so every sampling rate got the
  schedule for `p = 0.01`: at `p = 0.5` it admitted about 1% of the stream while
  weighting each admission as if it were 50%. The cursor now wraps on the
  table's real length (and folds an out-of-range value decoded from an old
  payload back into range), the schedule multiplies the unscaled `ln(1 - u)`
  table by `1 / ln(1 - p)`, and the rounding draw comes from a separate
  splitmix64 stream (`Nitro::rounding_state`) so it cannot phase-lock to the
  skip schedule. `NitroBatch::insert_cached_step` had the identical mask and
  rate bugs and is fixed the same way.
- **Nitro's insert and query derived cells from different hash forms, so every
  per-key estimate read 0.** `CountMin::fast_insert_nitro`,
  `Count::fast_insert_nitro` and `NitroBatch`'s own insert hashed with
  `hash128_seeded(0, value)` and sliced columns out of that raw `u128`, while
  `nitro_estimate`, `Count::estimate` and `NitroBatch::estimate_median` query
  through `FastPathHasher::hash_for_matrix` and `MatrixFastHash::col_for_row`.
  Those agree only when the matrix hash is the identity on the raw hash, which
  it is not — the packing mode is chosen from `rows` and `cols`. Every insert
  path now derives cell and sign exactly as a plain `insert` does. Separately,
  `NitroBatch` had assigned each sampled record to a single row
  (`position % rows`), which divided every per-row counter by the depth; a
  sampled record now reaches every row, which is what makes the estimator
  unbiased. Update weights saturate into the counter's domain
  (`nitro_delta_saturated_i32` / `_u32`) instead of wrapping into a decrement,
  reachable at rates below ~4.7e-10 or by writing the public `delta` field.
- **The row-level Nitro admission walk underflowed once skips became small.**
  The carry `(r + to_skip + 1) - rows` is negative on `usize` whenever the next
  admitted slot falls inside the same update — the common case at any rate above
  roughly `1/rows`, reachable only once the schedule above was rate-correct. The
  Count-Min path also admitted at most one row per update and dropped any
  further landings. Both now walk the update's slots through
  `Nitro::admit_rows`, whose arithmetic saturates so an outstanding skip near
  `usize::MAX` — reachable from `commit_ctx` or a decoded payload — cannot
  overflow the walk.
- **The row-level Nitro sampler admitted its first row slot unconditionally.**
  `Nitro` was built with `to_skip = 0`, so the first call to `admit_rows`
  admitted row 0 at every rate. For a one-row sketch that puts the estimate at
  `1/p` instead of 1 — 100× at `p = 0.01` — and for a `d`-row sketch it biases
  the first `d` slots of every fresh sketch. Both constructors now draw the
  first skip at construction; `init_nitro_seeded` places the cursor at the
  seed's table offset first, so the initial draw is the seed's own. At `p = 1`
  every row is still admitted with weight 1.
- **A decoded `Nitro` restarted its stochastic-rounding stream from a fixed
  constant.** `rounding_state` was `#[serde(skip)]`, so a sketch serialized
  mid-stream and resumed emitted a different weight sequence than the
  uninterrupted run at any rate whose reciprocal is not an integer. It is now
  serialized. See *Changed* for what that does to the encoded shape.

### Added

- **`CountMin::enable_nitro_with_seed`, `Count::enable_nitro_with_seed`,
  `Vector2D::enable_nitro_with_seed`, `Nitro::init_nitro_seeded`.** The
  unseeded path starts every sketch at the same point in the shared skip table,
  so two sketches at the same rate admit exactly the same subset — there was no
  way to run a Nitro accuracy battery over independent trials. The seed moves
  both the table cursor and the rounding stream. `NitroBatch::with_target_and_seed`
  now seeds the cursor too, so `insert_cached_step` is seed-dependent as
  `insert` already was.
- **`NitroContext`, `Nitro::context` and `Nitro::restore_context`.** The
  complete row-level sampling state — table cursor, outstanding skip, and the
  stochastic-rounding stream — so a snapshot restored onto another sketch
  continues the admission and weight sequence exactly. The older
  `Nitro::get_ctx` / `commit_ctx` pair carries the first two only; it is kept
  for compatibility and its documentation now says so rather than claiming to
  save the sampling state. `NitroBatch`'s pair covers the *cached* schedule
  only — its live `SmallRng` is not serializable — and says that too.

- **NitroSketch's compensating weight is no longer biased at rates whose
  reciprocal is not an integer.** `NitroBatch` and the row-level `Nitro` behind
  `CountMin::fast_insert_nitro` / `Count::fast_insert_nitro` both wrote
  `ceil(1 / p)` into the target on every admitted update. That is only the
  right compensation when `1 / p` is an integer: the public constructors accept
  any `0 < p <= 1`, and at `p = 0.3` every estimate came back
  `f * 0.3 * ceil(3.33) = 1.2 f` — a flat +20% on every key, in the shipped
  estimator, invisible to a test grid of `{1, 1/2, 1/10, 1/100}`. The weight is
  now rounded **stochastically**, `floor(1/p) + Bernoulli(frac(1/p))` drawn per
  admitted update, so `E[W] = 1/p` and hence `E[est] = f` at every rate, at the
  cost of `frac(1/p)(1 - frac(1/p)) <= 1/4` extra variance per admission.
  `NitroBatch::admitted_weight` draws from the sampling RNG; `Nitro::
  admitted_delta` draws from its own splitmix64 stream, advanced once per
  admitted slot and independent of the skip cursor, so the dither cannot become
  a fixed function of position in the skip table. When `1 / p` is an integer the
  fraction is zero, no draw is consumed, and the emitted weights are exactly
  what they were before — so the common rates (1, 1/2, 1/10, 1/100) are
  unchanged.

- **`ExponentialHistogram` no longer discards its payload when the bucket
  sketch is an `Elastic`.** `EHSketchList::merge` had no `ELASTIC` arm, so the
  catch-all returned `Cannot merge sketches of different types` for two Elastic
  payloads — and `EHBucket::to_merge` drops that `Result`, so the histogram
  went on summing bucket sizes while keeping only the counters of whichever
  bucket it merged into. Every interval query then answered from a single
  bucket: on a 10k-update Zipf stream each of the true top flows read 0 against
  counts in the thousands. The arm is now present and delegates to
  `Elastic::merge` like every other variant. No wire or API change.

- **`HeapItem` now has an owned byte-array key, so `DataInput::Bytes` no longer
  goes through `String`.** `input_to_owned` decoded every byte array as UTF-8
  and panicked on any that was not, and the ones that were became
  `HeapItem::String`, which made `Bytes(b"abc")` and `Str("abc")` the same key.
  `HeapItem::Bytes(Vec<u8>)` carries the bytes unchanged from insert through
  query, merge and the wire: `hash_item64_seeded` / `hash_item128_seeded` hash
  them exactly as `hash64_seeded` / `hash128_seeded` hash the borrowed
  `DataInput::Bytes`, equality against a `DataInput` is variant-exact so a byte
  key and a string key stay distinct, and `key_order` orders the two families
  apart. On the ASAPv1 wire this is a new `key_type` of `"bytes"`, written as
  msgpack `bin` and read back from `bin` alone (Space-Saving `0x18 0x00`,
  CMSHeap `0x03 0x00`, CSHeap `0x0a 0x00`, the UnivMon pyramids and Hydra's
  UnivMon cells). Existing integer and `"string"` encodings are unchanged and
  bytes written before this release still decode.

- **Restored the α relative-accuracy guarantee in portable `DdSketch`.**
  Quantile representatives changed from the log-midpoint γ^(k+0.5), whose edge
  error √γ−1 exceeds α, to γ^k·(1+α) — matching core `DDSketch` and DataDog's
  mapping. No wire/state migration; query outputs shift by at most α.
- **Portable `DdSketch` now rejects untrackable inputs like core.** Non-finite
  values (a NaN previously floor-cast into bucket 0) and finite-but-extreme
  values beyond the indexable range are dropped silently, mirroring core
  `DDSketch`'s min/max-indexable guards and DataDog's mapping (#70). Unguarded,
  one `f64::MAX` sample mapped ~35k buckets away at α=0.01 (~277 KiB of dense
  store per sample), scaling with 1/ln γ.
- **Portable `DdSketch::apply_delta` bounds hostile wire input.** Deltas now
  pre-validate their bucket span and return `Err` instead of padding the
  dense store: a corrupt delta carrying an index near `i32::MAX` would
  previously have attempted a ~2·10⁹-bucket (~17 GiB) allocation in one call.
  The limit (4M buckets) dwarfs any legitimate span; the `apply_delta*`
  family propagates the error. `merge`/`merge_refs` apply the same span cap
  to decoded snapshots, with a pass-through for stores that already
  legitimately exceed it. Portable `DdSketch::new` now asserts α ∈ (0,1)
  like core, and the indexable-range formulas live in one shared helper
  (`ddsketch_indexable_bounds`) used by both implementations, so their input
  guards cannot drift apart.
- Removed the raw-pointer write from `DDSketch`'s per-sample insertion path;
  bucket increments now go through safe indexing with the same bounds check
  they already performed (#70 item 6).
- **CountL2HH hot-path L2 accumulation saturates instead of wrapping.**
  Per-row Σcount² updates run through a saturating i128 intermediate, so
  extreme turnstile counts no longer wrap silently in release builds (or
  panic on overflow in debug builds).
- **Heap-allocate HLL register storage.** `impl_hll_bucket_list!`'s `Default`
  and `Deserialize` no longer build an `[u8; NUM_REGISTERS]` value on the
  stack before boxing it, which overflowed the stack in debug builds at
  larger precisions. The MessagePack encoding is unchanged.
- Corrected UnivMon and UnivMonPyramid merge semantics: aggregate stream
  weight, recomputed CountL2HH row norms, and merged-counter candidate
  re-estimation.
- Restored per-layer L2 maintenance for standard UnivMon updates and completed
  Joltik-style terminal-only updates with query-time logical reconstruction.
- Replaced UnivMon's fixed cardinality cutoff with a heap-sized L2-heavy-hitter
  threshold, tracked whether bounded candidate sets are complete, and made
  queries refresh candidate frequencies from current CountSketch counters.
- Removed duplicate stream-weight accounting in experimental UnivMon windows.
- Made L1 exact for UnivMon, UnivMonPyramid, and experimental UnivMon-Q by
  returning their tracked non-negative stream weight instead of evaluating a
  noisy generic recurrence.

### Added
- **Synthetic-data E2E testing harness with a reusable conformance kit.** A
  shared `tests/common` module (seeded stream generators, exact ground-truth
  trackers, tolerance-based assertion helpers) plus themed integration suites
  covering frequency, cardinality, quantile, framework/composition, and
  experimental sketches — every public sketch family is exercised end to end
  against exact ground truth. `tests/common/conformance.rs` defines standard
  batteries (`frequency`, `turnstile`, `cardinality`, `quantile`,
  `merge_equivalence`) that new sketches must pass via small adapter impls;
  `tests/README.md` documents the onboarding recipe and
  `tests/conformance_kit.rs` shows reference adapters. Also includes
  `tests/bug_verification.rs` regression tests for the fixes above and
  `examples/accuracy_probe.rs`, a release-mode ground-truth probe.
- **Deep Hydra coverage in `tests/e2e_frameworks.rs`.** A single-column Hydra —
  where the grid does the keying and each cell holds one counter — now runs the
  standard conformance batteries once per counter family it can host (CM,
  Count Sketch, HLL, KLL), joined by what no battery models: the `2^D - 1`
  fan-out across the subpopulation lattice checked against exact
  per-subpopulation truth, wildcard marginals reconciled against the cells
  beneath them, exact shard-merge equality, MessagePack round trips for every
  counter variant, and subkey injectivity for delimiter-laden key values. A
  Theorem 2 check (Manousis et al., VLDB 2022) asserts the additive `eps * G_s`
  bound against the exact binomial median-failure rate over 314 subpopulations
  at `G_s = 840k`, in both the sparse deployment regime (5x4096) and a
  deliberately overloaded grid (5x256) where the in-bound fraction is actually
  exercised. The mixed `HashSketchEnsemble` test moved here from
  `tests/e2e_cardinality.rs`.
- **Export the `impl_hll_bucket_list!` macro.** Downstream crates can now
  generate an `HllRegisterStorage` type at any precision (e.g. `lg_k = 18`)
  instead of being limited to the built-in `HllBucketListP12/P14/P16`.
- **`ErtlMLE` estimation at any precision.** `HyperLogLogImpl::<ErtlMLE, _>::estimate`
  is now a generic impl instead of one generated per storage type, so it works
  with custom `impl_hll_bucket_list!` precisions rather than only
  `HllBucketListP12/P14/P16`.
- **Experimental UnivMon-Q core sketch.** Adds `UnivMonQ<H>`, a Joltik-style
  terminal-stratum UnivMon implementation extended with an adaptively assisted
  occurrence sample for entropy, rank, CDF, and quantile estimates. The sketch
  supports pluggable `SketchHasher`, compatible merges, native MessagePack round-trips,
  exact extrema, point frequency, F0, F2, compatible generic g-sums, entropy,
  and recovered heavy-hitter queries. Reusable prepared query views share
  logical-hierarchy and CDF reconstruction across a metric batch. The API,
  estimators, and guarantees are experimental; an ASAPv1 cross-language kind
  is not yet assigned.
- **KLL `bulk_update` API for batch ingestion.** Adds `KLL::bulk_update(&[T])`
  and `KLL<f64>::bulk_update_data_input(&[DataInput])` (plus `KLLDynamic` and
  `HydraCounter::bulk_insert` delegating) — exactly equivalent to looping
  `update` (including `Coin` order, `count`, and byte-identical
  `serialize_to_bytes` for same seed), with empty slices as a no-op that
  preserves the memoized CDF. Fixes #88.

### Changed (breaking — wire format)
- **Drop the DataPoint-level METRIC scalars from `DDSketchState`.** Removed
  `count` (4), `sum` (5), `min` (6) and `max` (7) from
  `proto/ddsketch/ddsketch.proto` (tags now `reserved`), regenerated the prost
  bindings, and dropped the matching `count`/`sum`/`min`/`max` fields from the
  portable msgpack DTO (`message_pack_format::portable::ddsketch::DdSketch`) so
  its rmp-compact array shrinks from 7 to 3 elements
  (`[alpha, store_counts, store_offset]`). The total count is recovered by
  summing `store_counts` (`DdSketch::total_count()`); quantile estimation now
  derives the high-end estimate from the highest non-empty bucket instead of the
  removed `max`. The in-memory `sketches::DDSketch` keeps these as runtime-only
  state (`#[serde(skip)]`) and rebuilds them from the bucket store on decode.
  Byte-parity twin of the parallel `sketchlib-go` change
  (ProjectASAP/sketchlib-go#243).

## [0.2.2] - 2026-05-18

Performance patch release. No API changes; all public sketch APIs behave
identically to `0.2.1`.

### Performance
- **Restore `FixedMatrix` fast-path specialization in non-PGO builds**
  ([#50](https://github.com/ProjectASAP/asap_sketchlib/pull/50)). Added
  `#[inline(always)]` to `hash_for_matrix_seeded` and
  `hash_for_matrix_seeded_generic` in `src/common/hash.rs` so LLVM can
  propagate compile-time `rows` / `cols` literals from `FixedMatrix` call
  sites, fold the matrix-hash-mode dispatch, and unroll the inner loop.
  Measured on 1M `i64` inserts, single-threaded:
  - `CountMin<QuickMatrixI32, FastPath>` 5×2048: 89.7 → 173.9 M/s (+94%)
  - `Count<QuickMatrixI32, FastPath>` 5×2048: 71.5 → 115.1 M/s (+61%)

  Both within 3% of a PGO build. Downstream benchmarks no longer need PGO
  machinery or `-C llvm-args=-inline-threshold=...` overrides to hit the
  specialized path. KLL / HLL throughput unchanged.

## [0.2.1] - 2026-05-14

Maintenance release. No source-level changes to sketch algorithms; all public
sketch APIs (`CountMin`, `Count`, `HyperLogLog`, `KLL`, `DDSketch`, …)
behave identically to `0.2.0`.

### Changed
- **Precompute tables now build lazily at runtime via `std::sync::LazyLock`**
  instead of being shipped as multi-megabyte literal arrays
  ([`fba7a5b`](https://github.com/ProjectASAP/asap_sketchlib/commit/fba7a5b)).
  This drops ~147k lines from the published crate and brings it under the
  crates.io size limit. Index/iter/len access patterns are preserved through
  `Deref`, so typical usage is unaffected:

  ```rust
  let h = asap_sketchlib::PRECOMPUTED_HASH[42]; // still works
  for x in asap_sketchlib::PRECOMPUTED_SAMPLE.iter() { /* ... */ }
  ```

  Strictly per Rust's [SemVer guide](https://doc.rust-lang.org/cargo/reference/semver.html),
  changing the type of a `pub static` is classified as a breaking change. The
  affected items are:

  | Item | Old type | New type |
  | --- | --- | --- |
  | `PRECOMPUTED_HASH` | `[u128; 0x4000]` | `LazyLock<Box<[u128]>>` |
  | `PRECOMPUTED_SAMPLE` | `[f64; 0x10000]` | `LazyLock<Box<[f64]>>` |
  | `PRECOMPUTED_SAMPLE_RATE_1PERCENT` | `[f64; 0x10000]` | `LazyLock<Box<[f64]>>` |

  We are shipping this under a patch bump because (a) these tables are
  internal precompute artifacts that no known downstream binds by type, and
  (b) the value at every index is bit-for-bit identical to `0.2.0`. If you
  do depend on the concrete array type, please open an issue.

### Removed
- **Cargo feature `internal-bins`** and the three maintainer binaries it
  gated (`generate_precomputed_hash`, `generate_precomputed_sample`,
  `generate_precomputed_sample2`). These were intended only for regenerating
  the precompute tables, which now build lazily and require no codegen step.

### Build
- **Vendored prost output; `build.rs` removed**
  ([`bfcf906`](https://github.com/ProjectASAP/asap_sketchlib/commit/bfcf906)).
  Generated Rust types from `proto/**/*.proto` are now checked in under
  `src/proto/generated/` and refreshed manually by maintainers via
  `tools/gen-proto/`. Downstream users no longer need `protoc` and the crate
  no longer has any `[build-dependencies]`.
- CI now enforces that the committed `src/proto/generated/` matches the
  result of running `tools/gen-proto` against the current `.proto` sources
  ([`a13f353`](https://github.com/ProjectASAP/asap_sketchlib/commit/a13f353)).

### Docs
- README install instructions now point to crates.io with version, docs.rs,
  license, and MSRV badges
  ([`1a20eba`](https://github.com/ProjectASAP/asap_sketchlib/commit/1a20eba)).
- README's git-tag pin example updated to `v0.2.1`.

## [0.2.0] - 2026-05-12

Initial crates.io release.

- First publication of `asap_sketchlib` to [crates.io](https://crates.io/crates/asap_sketchlib).
- Public sketch APIs: `CountMin`, `Count`, `HyperLogLog` (`Classic`, `ErtlMLE`,
  HIP variants), `KLL`, `DDSketch`, plus framework layers (`Hydra`, `UnivMon`,
  `NitroBatch`, `ExponentialHistogram`) and shared primitives (`DataInput`,
  `SketchHasher`, `HashSketchEnsemble`).
- MessagePack wire format shared with `sketchlib-go`.
- MSRV: Rust 1.85 (Rust 2024 edition).

## [0.1.0] - 2026-04-24

Pre-release tag. Not published to crates.io.

[Unreleased]: https://github.com/ProjectASAP/asap_sketchlib/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ProjectASAP/asap_sketchlib/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/ProjectASAP/asap_sketchlib/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ProjectASAP/asap_sketchlib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ProjectASAP/asap_sketchlib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ProjectASAP/asap_sketchlib/releases/tag/v0.1.0
