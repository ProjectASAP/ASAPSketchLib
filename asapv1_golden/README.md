# ASAPv1 cross-language golden byte-vectors

These `.hex` files pin the exact bytes the ASAPv1 wire format (see
`docs/asapv1_wire_format.md`) emits for a set of fixed, known sketch states. They
are the machine-checked proof that the Rust (`asap_sketchlib`) and Go
(`sketchlib-go`) implementations serialize **byte-identically**.

**Coverage.** The fixtures below cover six `kind_id`s: HLL's three estimators,
Count-Min, Count Sketch and compact KLL. Every other `kind_id` the wire-format
registry marks *implemented* — Bloom, Space-Saving, CMSHeap, CSHeap, DDSketch,
Hydra's five counter variants, Elastic, Coco, UniformSampling, KMV, the UnivMon
family, CountL2HH, ExponentialHistogram and EHSketchList — has **no fixture and
therefore no cross-language drift guard**. `docs/asapv1_wire_format.md` fixes
their bytes; nothing here checks them.

**The copy here and the copy in `sketchlib-go/asapv1_golden/` MUST stay
byte-identical.** They are the same fixtures, checked into both repos so each
side's test suite is self-contained. The bytes are authored by the Rust side
(rmp_serde is the reference encoder); Go conforms to them, never the reverse.

Each file is one line of lowercase hex (no `0x`, no whitespace) = the complete
ASAPv1 envelope `[ magic | version | kind_id | metadata_len | payload_len |
metadata | payload ]`.

## Design principle: state is fixed, not hashed

Every fixture is built from a **known raw sketch state** (specific register
bytes / matrix values set directly), never by hashing input values. So the
golden tests the **wire encoding**, isolated from the hash functions.

## Fixtures

| File | Sketch | kind_id | State |
| ---- | ------ | ------- | ----- |
| `hll_classic_p12` | HLL Classic, P12 | `01 01` | 4096 registers, set: `[0]=1, [1]=7, [100]=42, [4095]=3` |
| `hll_ertl_mle_p12` | HLL Ertl-MLE, P12 | `01 02` | same register pattern |
| `hll_hip_p12` | HLL HIP, P12 | `01 03` | same registers + `hip_kxq0=1.5, hip_kxq1=2.5, hip_est=3.0` |
| `cms_i64_regular_2x3` | Count-Min i64, RegularPath | `02 00` | 2×3 row-major `[[0,1,127],[128,300,65536]]` |
| `cms_f64_fast_2x3` | Count-Min f64, FastPath | `02 00` | 2×3 row-major `[[0.0,1.5,2.25],[3.75,4.125,5.0625]]` |
| `cs_i64_regular_2x4` | Count Sketch i64, RegularPath | `04 00` | 2×4 row-major `[[0,127,128,65536],[-1,-33,-32768,-2147483648]]` |
| `cs_i64_fast_2x4` | Count Sketch i64, FastPath | `04 00` | same matrix — differs from the above only by `mode` |
| `cs_i32_regular_2x4` | Count Sketch i32, RegularPath | `04 00` | same matrix — differs from the first only by `counter_type` |
| `kll_f64_k200` | KLL f64, k=200 | `06 00` | integers `1..=50`, compaction seed 42 (recorded in metadata as `seed`) |
| `kll_i64_k200` | KLL i64, k=200 | `06 00` | integers `1..=50`, compaction seed 42 (recorded in metadata as `seed`) |

The CMS i64 fixture deliberately spans the msgpack integer width boundaries
(positive fixint / uint8 / uint16 / uint32) to lock the "non-negative integer →
uint family, minimal width" rule (`docs/asapv1_wire_format.md` Section 4).

The Count Sketch fixtures cover the **negative** side, which no other fixture
reaches, because Count Sketch cells are signed — it adds `±weight`: negative
fixint / int8 / int16 / int32, alongside positive fixint / uint8 / uint32.

All three Count Sketch files hold the same matrix, so each pair isolates one
metadata key: the two i64 files differ only by `mode`, and `cs_i32_regular_2x4`
differs from `cs_i64_regular_2x4` only in the `counter_type` value, `"i64"`
against `"i32"`. The payloads are byte-identical, because msgpack encodes an
integer at its minimal width whatever the source type is. So the i32 fixture
pins that the counter type reaches the bytes, and that nothing else does.

The KLL fixtures are a special case of "state is fixed, not hashed": KLL never
hashes — it orders raw numeric values — so inserting `1..=50` places exactly
those retained samples. `k=200` keeps the input below the level-0 capacity, so no
compaction fires (`num_levels = 1`, one level `[1..50]`) and the state is fully
deterministic. The fixed compaction seed (42) pins the carried coin state, which
must match `sketchlib-go`'s coin for the same input. Only the compact KLL
(`06 00`) has a golden; the dynamic variant (`06 01`) shares the payload shape but
lacks a seeded constructor, so its fixture waits on one.

## Tests that consume these

- Rust: `tests/asapv1_golden.rs` — builds each fixture from known state,
  serializes, asserts `== golden`; and asserts `deserialize(golden)` round-trips.
- Go: `wire/asapmsgpack/golden_test.go` — `Unmarshal(golden)`→re-`Marshal` ==
  golden, **and** `Marshal(equivalent known state) == golden` (the cross-language
  parity proof).

A `kind_id` that gains a fixture needs one on both sides at once: a new `.hex`
here and in `sketchlib-go/asapv1_golden/`, plus the case in each test file.

## Regenerating

If the wire format intentionally changes, regenerate from the Rust side (the
reference encoder) and copy the files into both repos. Both test suites must then
pass unchanged.
