# Test Matrix

One component per section. Each section contains a table with test name, description, and what the test validates.

## How To Run

```bash
cargo test
```

## Sketches

### CountMin

Test file: [`src/sketches/countminsketch.rs`](../src/sketches/countminsketch.rs)

Wire tests: [`src/sketches/countminsketch/wire.rs`](../src/sketches/countminsketch/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `dimension_test` | Default/custom dimensions initialize zeroed counters. | Verifies default dimensions (`rows=3`, `cols=4096`), custom dimensions (`3x17`), and zero-initialized counters after construction. |
| `fast_insert_same_estimate` | Fast and regular insert paths produce identical estimates. | Inserts five string keys once into both `RegularPath` and `FastPath` sketches (`3x64`) and asserts equal estimates for every key. |
| `merge_adds_counters_element_wise` | Merge sums counters element-wise for matching dimensions. | Merges two `2x32` sketches after inserting the same key (`1` on left, `2` on right) and checks merged per-row target counters equal `3`. |
| `countmin_insert_emit_delta_emits_at_threshold_and_resets_period` | Worker-path delta emission fires at promotion threshold and resets. | Inserts key into `3x64` CMS via `insert_emit_delta`; verifies no delta emitted before `CM_PROMASK` inserts, then exactly `3` deltas (one per row) with `value == CM_PROMASK` at threshold, no extra deltas in next sub-threshold window, and another batch of `3` at the next threshold. |
| `countmin_apply_delta_increments_parent_counter` | Apply delta increments parent counter. | Constructs a `CmDelta{row=1, col=5, value=CM_PROMASK}`, applies it to a `3x64` parent CMS, and verifies the target counter at `(1,5)` equals `CM_PROMASK`. |
| `cm_regular_path_correctness` | Regular-path hashing, counters, and estimates are exact on a deterministic stream. | Recomputes expected counter indices for `I32(0..9)` using per-row hashing, asserts exact full-matrix equality after one pass, doubled counters after second pass, and estimate `== 2` for each inserted key. |
| `cm_fast_path_correctness` | Fast-path counter placement matches bit-sliced hash mapping. | Recomputes expected fast-path indices for `I32(0..9)` from one hash plus row bit-slices/mask bits and asserts exact full-matrix equality. |
| `count_min_round_trip_serialization` | Serialization round trip preserves full sketch state. | Serializes/deserializes a populated `3x8` regular-path sketch and verifies dimensions plus the full counter array are unchanged. |
| `count_min_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a `3x8` sketch over a hasher declaring its own `HashProfile`, verifies the counter array round-trips, the bytes differ from the standard-profile sketch's over the same inserts, and a standard-profile decode rejects them. |
| `count_min_f64_and_mode_in_metadata_round_trip` | The `f64` counter type and the `mode` both travel in the metadata. | Round-trips a `4x16` fast-path `Vector2D<f64>` sketch fed fractional weights, verifies the counter array is preserved, then that the same bytes fail to decode as an `i64` regular-path sketch. |
| `count_min_rejects_zero_dimension_payload` | A zero dimension is a decode error, not a `Vector2D::from_fn` panic. | Verifies a crafted `4x0` envelope with an empty `counts` payload fails rather than panicking in the `cols.ilog2()` mask derivation. |
| `cms_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the eleven `CmsMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `CmsMetadata`. |
| `count_min_i32_round_trips_and_is_pinned_by_counter_type` | The `i32` wire config round-trips and its width is identity, not a detail. | Round-trips a `2x4` `Vector2D<i32>` sketch, then verifies its bytes differ from the numerically equal `i64` sketch's, that `i32` bytes fail to decode as `i64`, and that `i64` bytes fail to decode as `i32`. |
| `count_min_counter_types_reject_each_other` | Each wire counter type refuses the others' bytes. | Verifies a `2x4` `f64` envelope fails to decode as both an `i32` and an `i64` sketch, and that an `i32` envelope fails to decode as an `f64` one. |
| `count_min_rejects_too_many_rows` | More rows than the seed list has seeds is refused on both sides. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### Count

Test file: [`src/sketches/countsketch.rs`](../src/sketches/countsketch.rs)

Wire tests: [`src/sketches/countsketch/wire.rs`](../src/sketches/countsketch/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `default_initializes_expected_dimensions` | Default dimensions initialize zeroed counters. | Verifies default `Count` dimensions (`rows=3`, `cols=4096`) and that all counters are zero after construction. |
| `with_dimensions_uses_custom_sizes` | Custom dimensions initialize zeroed rows. | Verifies `with_dimensions(3, 17)` applies requested shape and each row slice is zero-initialized. |
| `insert_updates_signed_counters_per_row` | Regular insert applies per-row signed updates. | After one insert of key `"alpha"` into a `3x64` sketch, checks each row’s hashed counter equals that row’s expected sign (`+1` or `-1`). |
| `fast_insert_produces_consistent_estimates` | Fast-path single inserts return unit estimates. | Inserts five string keys once into a fast-path sketch (`4x128`) and asserts estimate `== 1.0` for each key. |
| `insert_produces_consistent_estimates` | Regular-path single inserts return unit estimates. | Inserts five string keys once into a regular-path sketch (`3x64`) and asserts estimate `== 1.0` for each key. |
| `estimate_recovers_frequency_for_repeated_key` | Regular path recovers repeated-key frequency. | Inserts key `"theta"` 37 times into a regular-path sketch (`3x64`) and asserts estimate `== 37.0`. |
| `fast_path_recovers_repeated_insertions` | Fast path recovers repeated insertions across keys. | Inserts five keys for 5 rounds into a fast-path sketch (`4x256`) and asserts estimate `== 5.0` for each key. |
| `merge_adds_counters_element_wise` | Merge sums signed counters for matching dimensions. | Merges two regular-path `2x32` sketches after inserting the same key (`1` on left, `2` on right) and checks per-row target counters equal `sign(row,key) * 3`. |
| `count_child_insert_emits_at_threshold` | Worker-path delta emission fires after sufficient inserts. | Inserts key into `3x64` `Count` via `insert_emit_delta` for `200` iterations and verifies at least `3` deltas (one per row) are emitted. |
| `zipf_stream_stays_within_twenty_percent_for_most_keys` | Zipf stream keeps relative error under 20% for most keys. | On Zipf stream (`rows=5`, `cols=8192`, `domain=8192`, `exponent=1.1`, `N=200_000`), computes per-key relative error and requires at least 70% of keys with error `< 0.20`. |
| `cs_regular_path_correctness` | Regular-path counter/sign mapping and estimates are exact on deterministic inserts. | Recomputes expected signed counter updates for `I32(0..9)` using regular hashing/sign logic, asserts exact matrix match after one pass, doubled counters after second pass, and estimate `== 2.0` for each inserted key. |
| `cs_fast_path_correctness` | Fast-path row-hash/sign mapping matches expected counters. | Recomputes expected fast-path updates for `I32(0..9)` using matrix hash row slices and row signs, then asserts exact full-matrix equality. |
| `count_sketch_round_trip_serialization` | The envelope frames a sketch under kind_id `0x04 0x00`, and the state survives a round trip. | Serializes a populated `3x8` regular-path `Vector2D<i64>` sketch and verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x04 0x00`, then that the decode preserves rows, cols, and the full counter array. |
| `count_sketch_negative_counters_round_trip` | Signed cells reach the wire and come back unchanged. | Round-trips a `2x4` matrix holding alternating positive and negative counters and verifies the decoded slice equals the source and still holds a negative cell. |
| `count_sketch_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a `3x8` sketch over a hasher declaring its own `HashProfile`, verifies the bytes round-trip, differ from the standard-profile sketch's bytes over the same inserts, and are rejected by a standard-profile decode. |
| `count_sketch_mode_in_metadata_round_trips` | The metadata `mode` pins which column derivation built the sketch. | Round-trips a `4x16` fast-path sketch and verifies the counter array is preserved, then that the same bytes fail to decode as a `RegularPath` sketch. |
| `count_sketch_rejects_foreign_kind_id` | Count-Min's kind_id is refused although the payload shape is identical. | Verifies a `3x8` `CountMin<Vector2D<i64>, RegularPath>` envelope fails to decode as a `Count`. |
| `count_sketch_rejects_zero_dimension_payload` | A zero dimension is a decode error, not a `Vector2D::from_fn` panic. | Verifies a crafted `4x0` envelope with an empty `counts` payload fails rather than panicking in the `cols.ilog2()` mask derivation. |
| `count_sketch_rejects_dimension_length_mismatch` | The length check fires from the declared dimensions, before any allocation is sized from them. | Verifies a crafted envelope declaring `MATRIX_MAX_ROWS x 2^24` while carrying three counters fails. |
| `count_sketch_rejects_serializing_an_unfilled_matrix` | The encode side refuses a matrix its own decoder would reject. | Verifies `Vector2D::init(2, 4)`, which reserves eight cells without filling them, fails to serialize rather than emitting a `2x4` envelope carrying an empty `counts` array. |
| `count_sketch_rejects_too_many_rows` | More rows than the seed list has seeds is refused on both sides. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |
| `cs_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the eleven `CsMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `CsMetadata`. |
| `cs_metadata_rejects_a_missing_counter_type_key` | `counter_type` is required and can never be silently defaulted. | Encodes the other ten `CsMetadata` fields as a named map with `counter_type` omitted and verifies it does not decode as `CsMetadata`. |
| `cs_metadata_rejects_a_foreign_counter_type_name` | A Count-Min-only counter type name is not a Count Sketch one. | Wraps metadata naming `counter_type: "f64"` around a valid 2x4 payload and verifies both the `i64` and the `i32` sketch reject it. |
| `count_sketch_i32_round_trips_and_is_pinned_by_counter_type` | The `i32` wire config round-trips and its width is identity, not a detail. | Round-trips a `2x4` `Vector2D<i32>` sketch, then verifies its bytes differ from the numerically equal `i64` sketch's, that `i32` bytes fail to decode as `i64`, and that `i64` bytes fail to decode as `i32`. |

### HyperLogLog

Test file: [`src/sketches/hll.rs`](../src/sketches/hll.rs)

Wire tests: [`src/sketches/hll/wire.rs`](../src/sketches/hll/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `hll_child_insert_emits_on_improvement` | Child insert emits delta only on register improvement. | Inserts a key via `insert_emit_delta` into `HyperLogLog<Classic>`; verifies exactly `1` delta emitted on the first insert and `0` additional deltas on a duplicate insert. |
| `hyperloglog_p12_accuracy_within_two_percent` | P12 Classic HyperLogLog stays within P12 error tolerance across scale checkpoints. | Applies the same checkpointed unique-stream accuracy test to `HyperLogLogP12<Classic>`, requiring relative error `<= P12_ERROR_TOLERANCE` at each target cardinality. |
| `hll_ertl_p12_accuracy_within_two_percent` | P12 ErtlMLE HyperLogLog stays within P12 error tolerance across scale checkpoints. | Applies the same checkpointed accuracy test to `HyperLogLogP12<ErtlMLE>`, requiring relative error `<= P12_ERROR_TOLERANCE`. |
| `hllds_p12_accuracy_within_two_percent` | P12 HIP HyperLogLog stays within P12 error tolerance across scale checkpoints. | Applies the same checkpointed accuracy test to `HyperLogLogHIPP12`, requiring relative error `<= P12_ERROR_TOLERANCE`. |
| `hyperloglog_p12_merge_within_two_percent` | P12 Classic HyperLogLog merge remains within P12 error tolerance. | Applies the same even/odd split merge scenario to `HyperLogLogP12<Classic>`, requiring merged relative error `<= P12_ERROR_TOLERANCE`. |
| `hll_ertl_p12_merge_within_two_percent` | P12 ErtlMLE HyperLogLog merge remains within P12 error tolerance. | Applies the same even/odd split merge scenario to `HyperLogLogP12<ErtlMLE>`, requiring merged relative error `<= P12_ERROR_TOLERANCE`. |
| `hyperloglog_round_trip_serialization` | Classic HyperLogLog round trip preserves bytes and estimate stability. | After inserting `100_000` unique values, verifies serialized payload is non-empty, `deserialize -> reserialize` bytes are identical, and estimate drift is within `0.02 * max(original_est, 1.0)`. |
| `hll_ertl_round_trip_serialization` | ErtlMLE HyperLogLog round trip preserves bytes and estimate stability. | Applies the same `100_000`-value serialization round-trip checks: non-empty bytes, byte-for-byte reserialization equality, and bounded estimate drift. |
| `hllds_round_trip_serialization` | HIP HyperLogLog round trip preserves bytes and estimate stability. | Applies the same `100_000`-value serialization round-trip checks for `HyperLogLogHIP`: non-empty bytes, byte-for-byte reserialization equality, and bounded estimate drift. |
| `hyperloglog_p12_round_trip_serialization` | P12 Classic HyperLogLog round trip preserves bytes and estimate stability. | Applies the same `100_000`-value serialization round-trip checks for `HyperLogLogP12<Classic>`: non-empty bytes, byte-for-byte reserialization equality, and bounded estimate drift. |
| `hll_ertl_p12_round_trip_serialization` | P12 ErtlMLE HyperLogLog round trip preserves bytes and estimate stability. | Applies the same `100_000`-value serialization round-trip checks for `HyperLogLogP12<ErtlMLE>`: non-empty bytes, byte-for-byte reserialization equality, and bounded estimate drift. |
| `hllds_p12_round_trip_serialization` | P12 HIP HyperLogLog round trip preserves bytes and estimate stability. | Applies the same `100_000`-value serialization round-trip checks for `HyperLogLogHIPP12`: non-empty bytes, byte-for-byte reserialization equality, and bounded estimate drift. |
| `hll_correctness_test` | Register update logic matches expected bucket/index behavior for all HLL variants. | Runs fixed hashed inserts against Classic, ErtlMLE, and HIP variants; asserts exact expected register values at specific bucket indices and confirms an untouched bucket remains zero. |
| `hll_envelope_structure_and_kind_id_guard` | The envelope frames an Ertl-MLE sketch under kind_id `0x01 0x02`, and a Classic decoder refuses it. | For 1,000 inserts into `HyperLogLog<ErtlMLE>`, verifies the bytes open with the ASAPv1 magic, `envelope::VERSION`, a `kind_id_len` of `2`, and `0x01 0x02`, that the decoded registers match the source, and that a `HyperLogLog<Classic>` decode fails. |
| `hll_hip_round_trip_preserves_state` | The HIP running scalars travel beside the registers. | For 1,000 inserts into `HyperLogLogHIP`, verifies the bytes carry kind_id `0x01 0x03` and that the decoded sketch matches the source on the registers and on `kxq0`, `kxq1`, and `est`. |
| `native_and_portable_hll_bytes_match` | The native and portable encoders emit the same envelope. | For 1,000 inserts each, verifies an `ErtlMLE` sketch's bytes equal the portable `HllSketch` `Datafusion` encoding of its registers, and a `HyperLogLogHIP` sketch's bytes equal the portable `Hip` encoding of its registers plus `kxq0`, `kxq1`, and `est`. |
| `hll_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a P14 `ErtlMLE` sketch over a hasher declaring its own `HashProfile`, verifies the registers round-trip, the bytes differ from the standard-profile sketch's over the same 1,000 inserts, and a standard-profile decode rejects them. |
| `hll_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the eight `HllMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `HllMetadata`. |
| `hll_precision_cross_rejection` | The metadata `precision` pins the register storage the bytes belong to. | Verifies a populated `HyperLogLogP12<Classic>` envelope fails to decode as the P14 `HyperLogLog<Classic>`. |
| `hll_hip_kind_id_rejected_by_classic` | The HIP kind_id is refused by the Classic decoder. | Verifies a populated `HyperLogLogHIP` envelope fails to decode as `HyperLogLog<Classic>`. |

### KLL

Test file: [`src/sketches/kll.rs`](../src/sketches/kll.rs)

Wire tests: [`src/sketches/kll/wire.rs`](../src/sketches/kll/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `coin_bit_cache_behavior` | Coin consumes cached random bits in deterministic bit order. | From a fixed seed, validates 3 successive 64-bit RNG blocks are consumed bit-by-bit (`0..63`) before refill, matching expected xorshift-derived bits exactly. |
| `coin_state_never_zero` | Coin state is never zero, including zero-seed initialization. | Verifies `Coin::from_seed(0)` normalizes to non-zero state and remains non-zero across 128 tosses. |
| `test_data_input_api` | DataInput numeric API is accepted and non-numeric input is rejected. | Inserts `I32`, `I64`, `F64`, `F32`, and `U32` values, checks median query lies between `20.0` and `40.2`, and verifies string input returns error `KLL sketch only accepts numeric inputs`. |
| `test_forced_compact` | Small-capacity KLL triggers compaction and keeps median in valid compacted outcomes. | With `KLL::init(3,3)` and inserts `[10,20,30,40,50]`, asserts median query is one of `{30.0, 40.0}` under forced compaction. |
| `test_no_compact` | Larger-capacity KLL avoids compaction for small stream and returns exact median. | With `KLL::init_kll(8)` and inserts `[10,20,30,40,50]`, asserts median query equals `30.0`. |
| `merge_preserves_quantiles_within_tolerance` | Merging two KLL sketches preserves quantiles within 2% rank tolerance. | Splits 10,000 uniform samples (`1,000,000..10,000,000`, seed `0xC0FFEE`) across two `k=200` sketches by index parity, merges, and checks quantiles `{0,0.1,0.25,0.5,0.75,0.9,1}` remain within `q +/- 0.02` truth bounds. |
| `cdf_handles_empty_sketch` | Empty KLL CDF queries return zero-valued defaults. | For empty `KLL::init_kll(64)`, asserts `cdf.quantile(123.0) == 0.0`, `cdf.query(0.5) == 0.0`, and `cdf.query_li(0.5) == 0.0`. |
| `kll_round_trip_rmp` | RMP round trip preserves KLL structure, packed data, and queried quantiles. | Serializes/deserializes `KLL::init_kll(256)` after 5,000 uniform updates (`0..1,000,000`, seed `0xDEAD_BEEF`), verifies non-empty bytes, core fields and packed arrays (`levels`, `items`) are identical, and CDF queries at `{0,0.1,0.25,0.5,0.75,0.9,1}` match within `f64::EPSILON`. |
| `generic_kll_i64_sanity` | Generic `KLL<T>` path works for non-`f64` numeric types. | Builds `KLL<i64>`, inserts `1..=20_000` through the typed `update(&T)` API, checks approximate count and p50/p90 quantiles, verifies merge on two `KLL<i64>` instances, and confirms MessagePack round-trip preserves weighted count. |
| `kll_envelope_structure_and_round_trip` | The envelope frames a compact sketch under kind_id `0x06 0x00`, and the bytes are stable across a round trip. | For `k=200` seeded at `42` over 200,000 updates, verifies the bytes open with the ASAPv1 magic, `envelope::VERSION`, a `kind_id_len` of `2`, and `0x06 0x00`, that the decoded sketch re-serializes to the same bytes, and that quantiles at `{0, 0.01, 0.25, 0.5, 0.75, 0.99, 1}` match exactly. |
| `kll_empty_round_trip` | A sketch that saw nothing round-trips. | Verifies `KLL::<f64>::init_kll_with_seed(200, 7)` decodes back with a `count` of `0` and identical re-serialized bytes. |
| `kll_i64_round_trip` | The generic `KLL<T>` path reaches the wire under the same kind_id. | For `KLL<i64>` seeded at `5` over 50,000 updates, verifies the bytes carry `0x06 0x00`, re-serialize identically after a decode, and preserve `count`. |
| `kll_item_type_cross_rejection` | The metadata `item_type` pins the element type. | Verifies an `f64` `KLL` envelope fails to decode as a `KLL<i64>`. |
| `kll_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the four required `KllMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `KllMetadata`. |
| `kll_rejects_inconsistent_levels` | A level layout the items do not fill is refused rather than panicking. | Verifies a crafted envelope whose `levels` end at `3` while `items` carries two entries fails to decode. |
| `kll_rejects_out_of_range_k_m` | A crafted `k`/`m` is refused before `compute_max_capacity` sizes anything. | Verifies crafted envelopes at `(u32::MAX, u32::MAX)`, `(MAX_CACHEABLE_K + 1, 8)`, and `(200, 1)` each fail to decode rather than driving a giant allocation. |
| `kll_seed_present_when_seeded_omitted_when_unseeded` | The seed key is present exactly when the sketch has one. | Verifies `init_kll_with_seed(200, 42)` emits metadata carrying `seed` `Some(42)`, while an `init_kll(200)` sketch omits the key entirely. |
| `kll_unseeded_round_trip_byte_stable` | The absent-seed key decodes as well as it encodes. | For an unseeded `k=200` sketch over 2,000 updates, verifies the decoded sketch re-serializes to the same bytes and keeps `count`. |
| `kll_seed_survives_round_trip_so_clear_stays_deterministic` | Carrying the seed keeps `clear` deterministic after a decode. | Decodes a `k=200` sketch seeded at `42` over 5,000 updates, calls `clear` on it and on a freshly seeded twin, feeds both 3,000 updates, and verifies their bytes are identical. |
| `kll_rejects_weighted_count_overflow` | A level layout whose weighted count overflows is refused rather than handed back. | Verifies a crafted envelope parking 16 items at compactor level 60, so that `16 * 2^60` overflows `usize` in `count()`, fails to decode. |
| `kll_dynamic_kind_id_rejected_by_compact` | The dynamic kind_id is refused by the compact decoder. | Verifies a crafted envelope carrying valid compact metadata and payload under `0x06 0x01` fails to decode as a `KLL<f64>`. |

### KLLDynamic

Test file: [`src/sketches/kll_dynamic.rs`](../src/sketches/kll_dynamic.rs)

Wire tests: [`src/sketches/kll_dynamic/wire.rs`](../src/sketches/kll_dynamic/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `test_data_input_api` | `KLLDynamic<f64>` accepts numeric `DataInput` and rejects non-numeric input. | Inserts `I32`, `I64`, `F64`, `F32`, and `U32` values through `update_data_input`, checks median query lies between `20.0` and `40.2`, and verifies string input returns error `KLL sketch only accepts numeric inputs`. |
| `test_forced_compact` | Small-capacity KLLDynamic triggers compaction and keeps median in valid compacted outcomes. | With `KLLDynamic::init(3,3)` and typed inserts `[10,20,30,40,50]`, asserts median query is one of `{30.0, 40.0}` under forced compaction. |
| `test_no_compact` | Larger-capacity KLLDynamic avoids compaction for small stream and returns exact median. | With `KLLDynamic::init_kll(8)` and typed inserts `[10,20,30,40,50]`, asserts median query equals `30.0`. |
| `merge_preserves_quantiles_within_tolerance` | Merging two KLLDynamic sketches preserves quantiles within 2% rank tolerance. | Splits 10,000 uniform samples (`1,000,000..10,000,000`, seed `0xC0FFEE`) across two `k=200` sketches by index parity, merges, and checks quantiles `{0,0.1,0.25,0.5,0.75,0.9,1}` remain within `q +/- 0.02` truth bounds. |
| `cdf_handles_empty_sketch` | Empty KLLDynamic CDF queries return zero-valued defaults. | For empty `KLLDynamic::<f64>::init_kll(64)`, asserts `cdf.quantile(123.0) == 0.0`, `cdf.query(0.5) == 0.0`, and `cdf.query_li(0.5) == 0.0`. |
| `kll_dynamic_round_trip_rmp` | RMP round trip preserves KLLDynamic structure, packed data, and queried quantiles. | Serializes/deserializes `KLLDynamic::init_kll(256)` after 5,000 uniform updates (`0..1,000,000`, seed `0xDEAD_BEEF`), verifies non-empty bytes, core fields and packed arrays (`levels`, `items`) are identical, and CDF queries at `{0,0.1,0.25,0.5,0.75,0.9,1}` match within `f64::EPSILON`. |
| `generic_kll_dynamic_i64_sanity` | Generic `KLLDynamic<T>` path works for non-`f64` numeric types. | Builds `KLLDynamic<i64>`, inserts `1..=20_000` through the typed `update(&T)` API, checks approximate count and p50/p90 quantiles, and confirms MessagePack round-trip preserves weighted count. |
| `kll_dynamic_envelope_structure_and_round_trip` | The envelope frames a dynamic sketch under kind_id `0x06 0x01`, and the bytes are stable across a round trip. | For `k=200` over 200,000 updates, verifies the bytes open with the ASAPv1 magic, `envelope::VERSION`, a `kind_id_len` of `2`, and `0x06 0x01`, that the decoded sketch re-serializes to the same bytes, and that quantiles at `{0, 0.01, 0.25, 0.5, 0.75, 0.99, 1}` match exactly. |
| `kll_dynamic_empty_round_trip` | A sketch that saw nothing round-trips. | Verifies `KLLDynamic::<f64>::init_kll(200)` decodes back and re-serializes to identical bytes. |
| `kll_dynamic_i64_round_trip` | The generic `KLLDynamic<T>` path reaches the wire under the same kind_id. | For `KLLDynamic<i64>` over 50,000 updates, verifies the bytes carry `0x06 0x01` and re-serialize identically after a decode. |
| `kll_dynamic_item_type_cross_rejection` | The metadata `item_type` pins the element type. | Verifies an `f64` `KLLDynamic` envelope fails to decode as a `KLLDynamic<i64>`. |

### DDSketch

Test file: [`src/sketches/ddsketch.rs`](../src/sketches/ddsketch.rs)

Wire tests: [`src/sketches/ddsketch/wire.rs`](../src/sketches/ddsketch/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `insert_and_query_basic` | Basic insert/query preserves count semantics and quantile monotonicity. | Inserts mixed values `[0.0, -5.0, 1.0, 2.0, 3.0, 10.0, 50.0, 100.0, 1000.0]`, verifies non-positive values are ignored (`count == 7`), and checks queried quantiles at `{0.0, 0.5, 0.9, 0.99, 1.0}` are monotone and bounded by sketch min/max. |
| `empty_quantile_returns_none` | Empty sketch returns no quantiles and zero count. | For a new `DDSketch(alpha=0.01)`, asserts `get_value_at_quantile` returns `None` for `p in {0.0, 0.5, 1.0}` and `get_count() == 0`. |
| `merge_two_sketches_combines_counts_and_bounds` | Merge combines counts and preserves quantile boundary invariants. | Merges sketches built from `[1,2,3,4]` and `[5,10,20]`, then verifies merged `count=7`, `min=1`, `max=20`, exact boundary quantiles (`q0=1`, `q1=20`), and median lies within `[1,20]`. |
| `dds_serialization_round_trip` | Serialization round trip preserves count, sum, bounds, and selected quantiles. | Serializes/deserializes a populated sketch (`alpha=0.01`), verifies non-empty bytes, equal `count/sum/min/max`, and exact quantile matches at `{0.0, 0.1, 0.5, 0.9, 1.0}`. |
| `ddsketch_envelope_structure_and_round_trip` | The envelope frames a sketch under kind_id `0x05 0x00`, and the bytes are stable across a round trip. | For a sketch at `alpha = 0.01` over eight values, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x05 0x00`, that the metadata carries `metadata_version` `1` and that alpha, that the decoded store counts and offset match, that re-serializing reproduces the bytes, and that quantiles at `{0, 0.1, 0.5, 0.9, 1}` are unchanged. |
| `ddsketch_scalars_survive_exactly` | `sum`, `min`, and `max` are carried, not recomputed from the buckets. | Verifies the decoded sketch reports the source's `sum` and `alpha`, a `min` of exactly `0.25`, and a `max` of exactly `1000.0`, rather than the alpha-bounded bucket representatives a recomputation would give. |
| `ddsketch_count_is_recovered_from_the_buckets` | The payload carries no `count` field. | Verifies the payload reads back as a five-element array whose bucket counts sum to `get_count()`, that a six-element read fails, and that the decoded sketch reports the same count. |
| `ddsketch_empty_round_trip` | A sketch that saw nothing round-trips. | Verifies a fresh `DDSketch(alpha=0.01)` decodes back with a `count` of `0`, `min` and `max` of `None`, an empty store, and identical re-serialized bytes. |
| `ddsketch_merged_round_trip` | A store grown on both sides of a merge round-trips. | Merges 200 values against the same 200 scaled by `0.001`, then verifies the decoded sketch matches the source on `count`, `sum`, `min`, and `max` and re-serializes to the same bytes. |
| `ddsketch_rejects_foreign_kind_id` | Another sketch's kind_id is refused even when the rest parses cleanly. | Verifies a `3x8` Count-Min envelope fails to decode as a `DDSketch`, and that well-formed DDSketch metadata and payload wrapped under kind_id `0x02 0x00` fail too. |
| `dd_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the two `DdMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `DdMetadata`. |
| `dd_metadata_rejects_a_missing_alpha_key` | `alpha` is required and can never be silently defaulted. | Encodes a named map holding only `metadata_version` and verifies it does not decode as `DdMetadata`. |
| `ddsketch_rejects_alpha_outside_the_unit_interval` | An `alpha` outside `(0, 1)` makes every bucket index meaningless and is refused. | Verifies crafted envelopes at `alpha` of `0.0`, `1.0`, `-0.5`, `2.0`, `NaN`, and infinity each fail with an alpha or metadata-mismatch complaint, while the same payload at `0.01` decodes. |
| `ddsketch_rejects_a_store_span_past_i32` | The span rule fires from the offset and the array length, before the store is rebuilt. | Verifies a crafted store of ten buckets at offset `i32::MAX - 2` fails with a "store span past i32" complaint, while the same payload three buckets long decodes. |
| `ddsketch_rejects_a_nonzero_offset_on_an_empty_store` | An empty store has exactly one encoding. | Verifies a crafted empty store at offset `42` fails with an "empty store must be at offset 0" complaint. |
| `ddsketch_rejects_a_total_count_overflow` | Bucket counts that overflow the recovered total are refused rather than wrapped. | Verifies a crafted payload of `[u64::MAX, 1]` fails with an "overflow the total sample count" complaint. |
| `ddsketch_rejects_inconsistent_scalars` | The scalars the buckets do not determine are still bounded by them. | Verifies six crafted payloads each fail on a scalar rule: a populated store carrying the empty-store sentinels, `min` above `max`, a non-positive `min`, a `sum` below the smallest ingested value, an empty store carrying a non-zero `sum`, and all-zero buckets carrying populated-store scalars. |
| `ddsketch_rejects_serializing_a_store_span_past_i32` | The encode side refuses a store its own decoder would reject. | Verifies a sketch whose store holds ten buckets at offset `i32::MAX - 2` fails to serialize. |
| `ddsketch_rejects_serializing_an_overflowing_bucket_total` | The encode side enforces the total-count rule too. | Verifies a sketch whose store holds `[u64::MAX, 1]` fails to serialize. |

### CMSHeap

Test file: [`src/sketches/countminsketch_topk.rs`](../src/sketches/countminsketch_topk.rs)

Wire tests: [`src/sketches/countminsketch_topk/wire.rs`](../src/sketches/countminsketch_topk/wire.rs)

Shared heap wire tests: [`src/sketches/countminsketch_topk/heap_wire.rs`](../src/sketches/countminsketch_topk/heap_wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `insert_and_estimate` | Repeated inserts increment Count-Min estimate for one key. | Inserts `"hello"` 5 times into `CMSHeap::new(3,64,10)` and verifies `estimate("hello") == 5`. |
| `heap_tracks_top_k` | Heap keeps highest-frequency keys within top-k capacity. | Inserts keys `1..5` with frequencies `10,20,30,40,50` into `top_k=3` sketch and verifies heap counts are exactly `[30,40,50]`. |
| `merge_reconciles_heaps` | Merge combines counters and refreshes heap counts from merged sketch. | Merges two sketches containing `"merge_key"` counts `10` and `20`, then verifies merged estimate and heap count are both `30`. |
| `insert_many_updates_estimate_and_heap` | `insert_many` updates estimate and heap entry consistently. | Calls `insert_many("many", 11)` and verifies `estimate == 11` plus heap entry count `11`. |
| `bulk_insert_updates_multiple_keys` | `bulk_insert` updates multiple keys and heavy-hitter counts correctly. | Inserts stream `[7,8,7,9,7]` and verifies estimates `7->3`, `8->1`, `9->1`, with heap count for key `7` equal to `3`. |
| `clear_heap_keeps_cms_counters` | Clearing heap does not clear CMS counters. | After `insert_many("persist",5)`, calls `clear_heap()`, verifies estimate remains `5`, then one more insert rebuilds heap entry to `6`. |
| `from_storage_uses_storage_dimensions` | `from_storage` preserves backend dimensions and requested heap capacity. | Builds from `Vector2D::init(4,128)` with `top_k=9` and verifies `rows=4`, `cols=128`, `heap.capacity=9`. |
| `merge_refreshes_existing_self_heap_entries` | Merge refreshes pre-existing self heap keys to merged estimates. | After merging sketches with `a-key` counts `10` and `5`, verifies merged `a-key` estimate `15` and heap entry count `15`. |
| `fast_path_insert_and_estimate` | Fast path repeated inserts keep estimate exact for single key. | Inserts `"fast"` 7 times into fast-path sketch and verifies estimate `7`. |
| `fast_path_insert_many_and_bulk_insert` | Fast path batched APIs keep heap and estimate in sync. | Applies `insert_many("fast-many",6)` plus bulk inserts adding 2 more hits, then verifies estimate and heap count are `8`. |
| `fast_path_heap_tracks_top_k` | Fast path heap still preserves top-k ordering under weighted updates. | Inserts keys `1..5` with counts `10,20,30,40,50` via `insert_many` and verifies heap counts `[30,40,50]`. |
| `fast_path_merge_refreshes_existing_self_heap_entries` | Fast path merge refreshes self heap entries using merged totals. | Merges sketches where `"a-fast"` contributes `10` and `5` across sides, then verifies estimate and heap count are `15`. |
| `default_construction` | Default CMSHeap constructor uses expected dimensions and heap capacity. | Verifies `CMSHeap::<Vector2D<i64>, RegularPath>::default()` has `rows=3`, `cols=4096`, and `heap.capacity=DEFAULT_TOP_K`. |
| `default_construction_fixed_backends_parity` | Default constructors across storage backends keep intended size/capacity contracts. | Verifies defaults for Fixed/Quick backends are `5x2048`, DefaultMatrix backends are `3x4096`, and all regular/fast variants use `DEFAULT_TOP_K`. |
| `merge_requires_matching_dimensions_panics` | Merge panics on incompatible sketch dimensions. | Verifies merging `CMSHeap::new(3,256,4)` with `CMSHeap::new(4,256,4)` panics with dimension-mismatch message. |
| `heap_entries_match_cms_estimates_after_mutations` | Every heap entry count matches current CMS estimate after updates and merge. | Checks heap-entry equality to `estimate(key)` both before and after merging another mutated sketch. |
| `bulk_insert_equivalent_to_repeated_insert` | Bulk insert is equivalent to repeated single inserts. | Compares bulk vs repeated insertion on same stream and verifies identical per-key estimates and heap counts for keys `1..5`. |
| `regular_vs_fast_equivalence_on_same_stream` | Regular and fast wrappers agree on identical deterministic stream. | Feeds same 10-item string stream to both paths and verifies per-key estimates and heap counts match for `{alpha,beta,gamma,delta,epsilon}`. |
| `merge_with_empty_other_and_empty_self` | Merge behavior is stable when one side is empty. | Verifies merging non-empty with empty leaves counts unchanged and merging empty-self with non-empty copies counts/heap visibility correctly. |
| `duplicate_candidate_keys_during_merge_do_not_corrupt_heap` | Duplicate merge candidates do not duplicate heap entries. | Merges sketches both containing `"dup"`; verifies merged count `19`, heap size within capacity, and exactly one heap entry for `"dup"`. |
| `zipf_stream_top_k_recall_regular_fast_budget` | Regular path heap achieves high top-k recall on Zipf stream. | On Zipf stream (`rows=3`, `cols=4096`, `top_k=16`, `domain=1024`, `exponent=1.1`, `N=20_000`), verifies heap size bound, entry-count consistency, and recall hits `>= 15` vs truth top-16. |
| `zipf_stream_top_k_recall_fast_path_fast_budget` | Fast path heap achieves high top-k recall on Zipf stream. | Runs same Zipf setup in fast mode and verifies heap size bound, entry-count consistency, and recall hits `>= 15`. |
| `zipf_stream_regular_fast_heap_overlap` | Regular and fast heaps substantially overlap on Zipf heavy hitters. | On shared Zipf stream (`top_k=16`), verifies key overlap ratio between regular and fast top-k heaps is at least `0.8`. |
| `cms_heap_round_trip_serialization` | The envelope frames a sketch under kind_id `0x03 0x00`, and the bytes are stable across a round trip. | For a `3x8` regular-path `i64` sketch with a heap of `4` holding three weighted `u64` keys, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x03 0x00`, that the metadata carries `3x8`, `counter_type` `i64`, `mode` `regular`, `k` `4`, and `key_type` `u64`, that dimensions, counters, heap length and capacity, and per-key estimates and heap counts all match, and that re-serializing reproduces the bytes. |
| `cms_heap_byte_keys_round_trip` | A raw byte-array heap key reaches the wire as msgpack `bin`. | For a `2x4` `i64` sketch whose heap holds `[0xff, 0x00, 0xfe]` at `11`, verifies the metadata names `key_type` `bytes`, the decoded heap still finds the key by its own `DataInput::Bytes` at that count, and the bytes re-serialize identically. |
| `cms_heap_f64_counters_round_trip` | The `f64` base counter reaches the wire beside a string-keyed heap. | For a `2x4` `Vector2D<f64>` sketch holding one `Str` heap key, verifies the metadata names `counter_type` `f64` and `key_type` `string`, that the counters and the heap key survive the decode, and that the bytes re-serialize identically. |
| `cms_heap_counter_type_is_pinned_by_the_target` | The base counter type separates two otherwise equal sketches. | For `2x4` sketches holding numerically equal `i64` and `f64` cells, verifies the bytes differ, that `i64` bytes fail to decode as `f64`, and that `f64` bytes fail to decode as `i64`. |
| `cms_heap_mode_in_metadata_round_trips` | The metadata `mode` pins which column derivation built the sketch. | Round-trips a `4x16` fast-path sketch, verifies the metadata names `mode` `fast` and the counters are preserved, then that the same bytes fail to decode as a `RegularPath` sketch. |
| `cms_heap_rejects_foreign_kind_ids` | The neighbouring sketches' kind_ids are refused. | Verifies `CSHeap`, `CountMin`, and `Count` envelopes at `3x8` each fail to decode as a `CMSHeap`. |
| `cms_heap_rejects_zero_dimension_payload` | A zero dimension is a decode error, not a `Vector2D::from_fn` panic. | Verifies a crafted `4x0` envelope fails with a "must be non-zero" complaint. |
| `cms_heap_rejects_dimension_length_mismatch` | The length check fires from the declared dimensions, before any allocation is sized from them. | Verifies a crafted envelope declaring `MATRIX_MAX_ROWS x 2^24` while carrying three counters fails with a "!= rows*cols" complaint. |
| `cms_heap_rejects_serializing_an_unfilled_matrix` | The encode side refuses a matrix its own decoder would reject. | Verifies `Vector2D::<i64>::init(2, 4)`, which reserves eight cells without filling them, fails to serialize. |
| `cms_heap_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the thirteen `TopKMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `TopKMetadata`. |
| `cms_heap_metadata_rejects_a_missing_k_key` | `k` is required, so the heap capacity can never be silently defaulted. | Encodes the other twelve `TopKMetadata` fields as a named map with `k` omitted and verifies it does not decode as `TopKMetadata`. |
| `cms_heap_rejects_a_foreign_counter_type_name` | A Count-Sketch-only counter type name is not a Count-Min one. | Wraps metadata naming `counter_type: "i32"` around a valid `2x4` payload and verifies an `i64` `CMSHeap` rejects it. |
| `cms_heap_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a `3x8` sketch with a heap of `4` over a hasher declaring its own `HashProfile`, verifies the counters and both heap entries round-trip, the bytes differ from the standard-profile sketch's over the same keys, and a standard-profile decode rejects them. |
| `cms_heap_refuses_a_k_the_metadata_cannot_carry` | A `k` past the metadata's `u32` field fails the encode rather than truncating. | Verifies a `2x4` sketch built at a heap capacity of `1 << 40` fails to serialize with a message naming the "exceeds the u32 metadata field" rule. |
| `heap_every_key_type_round_trips_and_keeps_its_variant` | The `key_type` names the exact `HeapItem` variant and is never widened. | Across all 13 wire key types - `i8`, `i16`, `i32`, `i64`, `isize`, `u8`, `u16`, `u32`, `u64`, `usize`, `f32`, `f64`, and `string` - verifies the metadata carries the expected name, the bytes re-serialize identically, and every decoded key is still found by its original `DataInput` at the weight it was given. |
| `heap_refuses_mixed_and_128_bit_keys` | Keys the wire cannot carry refuse to serialize rather than being coerced. | Verifies a heap holding both an `I32` and an `I64` key fails with a "keys mix variants" complaint naming `key_type is i32`, that a lone `I128` or `U128` key fails with a "128-bit" complaint, and that a 128-bit key seated behind a `U64` one fails too. |
| `heap_empty_emits_the_pinned_key_type_and_round_trips` | A heap monitoring nothing has one encoding. | For an empty `2x8` sketch with a heap of `8`, verifies the metadata carries the pinned `EMPTY_KEY_TYPE` and a `k` of `8`, and that the decode holds no entries at capacity `8` with identical re-serialized bytes. |
| `heap_emitted_order_is_independent_of_seat_order` | The emitted order is descending count with ties broken by the key, not the sift path. | Seats four entries carrying two count ties in one order and in reverse, verifies both serialize to the same bytes, that the payload reads `heap_counts` `[9, 9, 5, 5]` and `keys` `[1, 4, 2, 3]`, and that a decoded heap re-serializes identically. |
| `heap_rejects_a_key_type_the_payload_does_not_carry` | A payload relabelled with another `key_type` is refused. | Re-frames a string-keyed payload under a `u64` `key_type` and a `u64`-keyed payload under `string`, and verifies neither decodes. |
| `heap_index_is_rebuilt_so_updates_still_move_entries` | `slots` and `positions` are rebuilt on decode, so later updates land the same way. | Round-trips a four-entry heap, applies the same rescore and new-key update to the source and to the decoded copy, and verifies both serialize to the same bytes and agree on every key's count. |
| `heap_does_not_allocate_a_declared_k` | A declared `k` is metadata, never an allocation size. | Verifies an envelope declaring `k` `u32::MAX` with two entries decodes, reports that capacity, and holds two entries. |
| `heap_rejects_crafted_entry_sets` | Entry sets the heap could not have reached are refused. | Verifies three crafted envelopes each fail with their own complaint: two entries over a `k` of `1`, the same key twice, and two keys against one count. |
| `cms_heap_rejects_too_many_rows` | The base matrix carries the seed list's row bound. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### CSHeap

Test file: [`src/sketches/countsketch_topk.rs`](../src/sketches/countsketch_topk.rs)

Wire tests: [`src/sketches/countsketch_topk/wire.rs`](../src/sketches/countsketch_topk/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `insert_and_estimate` | Repeated inserts increment `Count` estimate for one key. | Inserts `"hello"` 5 times into `CSHeap::new(5,256,10)` and verifies estimate is `5.0` within `1e-9`. |
| `heap_tracks_top_k` | Heap keeps highest-frequency keys within top-k capacity. | Inserts keys `1..5` with frequencies `100,200,300,400,500` into `top_k=3` sketch and verifies heap counts are exactly `[300,400,500]`. |
| `merge_reconciles_heaps` | Merge combines counters and refreshes heap counts from merged sketch. | Merges two sketches containing `"merge_key"` counts `10` and `20`, then verifies estimate is `30.0` and heap count is `30`. |
| `insert_many_updates_estimate_and_heap` | `insert_many` updates estimate and heap entry consistently. | Calls `insert_many("many", 17)` and verifies estimate `17.0` plus heap entry count equals estimated count. |
| `bulk_insert_updates_multiple_keys` | `bulk_insert` updates multiple keys and heavy-hitter counts correctly. | Inserts stream `[7,8,7,9,7]`, verifies estimate for key `7` is `3.0`, and heap count for key `7` matches estimate cast to integer. |
| `clear_heap_keeps_cs_counters` | Clearing heap does not clear `Count` counters. | After `insert_many("persist",5)`, calls `clear_heap()`, verifies estimate remains `5.0`, then one more insert repopulates heap with updated estimate count. |
| `from_storage_uses_storage_dimensions` | `from_storage` preserves backend dimensions and requested heap capacity. | Builds from `Vector2D::init(4,128)` with `top_k=9` and verifies `rows=4`, `cols=128`, `heap.capacity=9`. |
| `merge_refreshes_existing_self_heap_entries` | Merge refreshes pre-existing self heap keys to merged estimates. | Merges sketches where `"a-key"` is updated on both sides (`120` and `40`), then verifies heap count for `"a-key"` equals merged estimate. |
| `fast_path_insert_and_estimate` | Fast path repeated inserts keep estimate exact for single key. | Inserts `"fast"` 7 times into fast-path sketch and verifies estimate is `7.0` within `1e-9`. |
| `fast_path_insert_many_and_bulk_insert` | Fast path batched APIs keep heap and estimate in sync. | Applies `insert_many("fast-many",6)` plus bulk inserts adding 2 hits, then verifies estimate is `8.0` and heap count matches it. |
| `fast_path_heap_tracks_top_k` | Fast path heap preserves top-k ordering under weighted updates. | Inserts keys `1..5` with counts `100,200,300,400,500` via `insert_many` and verifies heap counts `[300,400,500]`. |
| `fast_path_merge_refreshes_existing_self_heap_entries` | Fast path merge refreshes self heap entries using merged totals. | Merges fast sketches where `"a-fast"` is updated on both sides (`120` and `40`) and verifies heap count equals merged estimate. |
| `default_construction` | Default CSHeap constructor uses expected dimensions and heap capacity. | Verifies `CSHeap::<Vector2D<i64>, RegularPath>::default()` has `rows=3`, `cols=4096`, and `heap.capacity=DEFAULT_TOP_K`. |
| `default_construction_fixed_backends_parity` | Default constructors across storage backends keep intended size/capacity contracts. | Verifies defaults for Fixed/Quick backends are `5x2048`, DefaultMatrix backends are `3x4096`, and all regular/fast variants use `DEFAULT_TOP_K`. |
| `merge_requires_matching_dimensions_panics` | Merge panics on incompatible sketch dimensions. | Verifies merging `CSHeap::new(5,256,4)` with `CSHeap::new(6,256,4)` panics with dimension-mismatch message. |
| `heap_entries_match_cs_estimates_after_mutations` | Every heap entry count matches current sketch estimate after updates and merge. | Checks heap-entry equality to `estimate(key)` both before and after merging another mutated sketch. |
| `bulk_insert_equivalent_to_repeated_insert` | Bulk insert is equivalent to repeated single inserts. | Compares bulk vs repeated insertion on same stream and verifies per-key estimates match within `1e-9` plus identical heap counts for keys `1..5`. |
| `regular_vs_fast_equivalence_on_same_stream` | Regular and fast wrappers agree on identical deterministic stream. | Feeds same 10-item string stream to both paths and verifies per-key estimates match within `1e-9` and heap counts match for `{alpha,beta,gamma,delta,epsilon}`. |
| `merge_with_empty_other_and_empty_self` | Merge behavior is stable when one side is empty. | Verifies merging non-empty with empty leaves estimates/heap size unchanged and merging empty-self with non-empty reproduces estimates and heap visibility. |
| `duplicate_candidate_keys_during_merge_do_not_corrupt_heap` | Duplicate merge candidates do not duplicate heap entries. | Merges sketches both containing `"dup"`; verifies heap count equals merged estimate, heap size stays within capacity, and only one heap entry exists for `"dup"`. |
| `zipf_stream_top_k_recall_regular_fast_budget` | Regular path heap achieves high top-k recall on Zipf stream. | On Zipf stream (`rows=5`, `cols=4096`, `top_k=16`, `domain=1024`, `exponent=1.1`, `N=20_000`), verifies heap size bound, entry-count consistency, and recall hits `>= 15` vs truth top-16. |
| `zipf_stream_top_k_recall_fast_path_fast_budget` | Fast path heap achieves high top-k recall on Zipf stream. | Runs same Zipf setup in fast mode and verifies heap size bound, entry-count consistency, and recall hits `>= 15`. |
| `zipf_stream_regular_fast_heap_overlap` | Regular and fast heaps substantially overlap on Zipf heavy hitters. | On shared Zipf stream (`top_k=16`), verifies key overlap ratio between regular and fast top-k heaps is at least `0.8`. |
| `cs_heap_round_trip_serialization` | The envelope frames a sketch under kind_id `0x0a 0x00`, and the bytes are stable across a round trip. | For a `3x8` regular-path `i64` sketch with a heap of `4` holding three weighted `u64` keys, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x0a 0x00`, that the metadata carries `3x8`, `counter_type` `i64`, `mode` `regular`, `k` `4`, and `key_type` `u64`, that dimensions, counters, heap length and capacity, and per-key estimates and heap counts all match, and that re-serializing reproduces the bytes. |
| `cs_heap_negative_counters_round_trip` | Signed cells reach the wire and come back unchanged. | Round-trips a `2x4` matrix of alternating positive and negative counters beside a `Str` heap key, and verifies the decoded slice equals the source, still holds a negative cell, keeps the heap key, and re-serializes identically. |
| `cs_heap_i32_round_trips_and_is_pinned_by_counter_type` | The `i32` wire config round-trips and its width is identity, not a detail. | Round-trips a `2x4` `Vector2D<i32>` sketch, verifies the metadata names `counter_type` `i32`, then that its bytes differ from the numerically equal `i64` sketch's, that `i32` bytes fail to decode as `i64`, and that `i64` bytes fail to decode as `i32`. |
| `cs_heap_mode_in_metadata_round_trips` | The metadata `mode` pins which column derivation built the sketch. | Round-trips a `4x16` fast-path sketch, verifies the metadata names `mode` `fast` and the counters are preserved, then that the same bytes fail to decode as a `RegularPath` sketch. |
| `cs_heap_rejects_foreign_kind_ids` | The neighbouring sketches' kind_ids are refused. | Verifies `CMSHeap`, `CountMin`, and `Count` envelopes at `3x8` each fail to decode as a `CSHeap`. |
| `cs_heap_rejects_zero_dimension_payload` | A zero dimension is a decode error, not a `Vector2D::from_fn` panic. | Verifies a crafted `4x0` envelope fails with a "must be non-zero" complaint. |
| `cs_heap_rejects_dimension_length_mismatch` | The length check fires from the declared dimensions, before any allocation is sized from them. | Verifies a crafted envelope declaring `MATRIX_MAX_ROWS x 2^24` while carrying three counters fails with a "!= rows*cols" complaint. |
| `cs_heap_rejects_serializing_an_unfilled_matrix` | The encode side refuses a matrix its own decoder would reject. | Verifies `Vector2D::<i64>::init(2, 4)`, which reserves eight cells without filling them, fails to serialize. |
| `cs_heap_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the thirteen `TopKMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `TopKMetadata`. |
| `cs_heap_metadata_rejects_a_missing_key_type_key` | `key_type` is required, so the heap's key variant can never be silently defaulted. | Encodes the other twelve `TopKMetadata` fields as a named map with `key_type` omitted and verifies it does not decode as `TopKMetadata`. |
| `cs_heap_rejects_a_foreign_counter_type_name` | A Count-Min-only counter type name is not a Count Sketch one. | Wraps metadata naming `counter_type: "f64"` around a valid `2x4` payload and verifies both the `i64` and the `i32` sketch reject it. |
| `cs_heap_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a `3x8` sketch with a heap of `4` over a hasher declaring its own `HashProfile`, verifies the counters and both heap entries round-trip, the bytes differ from the standard-profile sketch's over the same keys, and a standard-profile decode rejects them. |
| `cs_heap_refuses_a_k_the_metadata_cannot_carry` | A `k` past the metadata's `u32` field fails the encode rather than truncating. | Verifies a `2x4` sketch built at a heap capacity of `1 << 40` fails to serialize with a message naming the "exceeds the u32 metadata field" rule. |
| `cs_heap_rejects_too_many_rows` | The base matrix carries the seed list's row bound. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### CountL2HH

Test file: [`src/sketches/countsketch_topk.rs`](../src/sketches/countsketch_topk.rs)

Wire tests: [`src/sketches/countsketch_topk/l2hh_wire.rs`](../src/sketches/countsketch_topk/l2hh_wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `countl2hh_estimates_and_l2_are_consistent` | CountL2HH updates keep estimate and L2 consistent. | For `CountL2HH(3x32)`, applies `+5` then `-2` to one key, verifies estimates `5.0` then `3.0`, and asserts non-trivial L2 (`>= 3.0`). |
| `countl2hh_merge_combines_frequency_vectors` | CountL2HH merge combines per-key frequencies. | Merges two `CountL2HH(3x32)` sketches with same key counts `4` and `9`, then verifies merged estimate `== 13.0`. |
| `countl2hh_round_trip_serialization` | CountL2HH serialization round trip preserves estimate and L2. | Serializes/deserializes `CountL2HH::with_dimensions_and_seed(3,32,7)` after updates, verifying rows/cols and that both estimate and L2 remain unchanged (within `f64::EPSILON`). |
| `count_l2hh_round_trip_serialization` | The envelope frames a sketch under kind_id `0x19 0x00`, and the state survives a round trip. | For a `3x32` sketch seeded at `7` over three signed weighted inserts, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x19 0x00`, that the metadata carries `seed_index` `7` and `3x32`, and that the decode matches on dimensions, seed index, the counter array, the `l2` accumulators, and `get_l2`. |
| `count_l2hh_decoded_re_serializes_identically` | A decoded sketch re-serializes byte for byte. | Verifies the populated `3x32` sketch's decode re-encodes to the bytes it came from. |
| `count_l2hh_negative_counters_round_trip` | Signed cells reach the wire and come back unchanged. | Round-trips a `2x8` sketch fed two negative weights and verifies the decoded slice equals the source, still holds a negative cell, and re-serializes identically. |
| `count_l2hh_seed_index_travels_with_the_sketch` | The seed index is state, not a profile constant. | Verifies `2x8` sketches seeded at `0` and at `9` do not share bytes, and that the decoded seed-`9` sketch reports a `seed_idx` of `9`. |
| `count_l2hh_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a `3x32` sketch over a hasher declaring its own `HashProfile`, verifies the counter array round-trips, the bytes differ from the standard-profile sketch's over the same insert, and a standard-profile decode rejects them. |
| `count_l2hh_rejects_foreign_kind_ids` | The neighbouring universal-sketch kind_ids are refused. | Verifies `Count`, `UnivMon`, `UnivMonPyramid`, and `UnivMonQ` envelopes each fail to decode as a `CountL2HH`. |
| `count_l2hh_rejects_crafted_geometry` | Every geometry rule fires before an allocation is sized from it. | Verifies crafted envelopes carrying a zero `cols`, a `MATRIX_MAX_ROWS x 2^24` declaration against three counters, a five-entry `l2` array against two rows, and a negative `l2` accumulator each fail to decode. |
| `count_l2hh_rejects_serializing_an_unfilled_matrix` | The encode side refuses a matrix its own decoder would reject. | Verifies a `2x4` sketch whose counts are replaced by `Vector2D::init(2, 4)`, which reserves eight cells without filling them, fails to serialize. |
| `l2hh_metadata_rejects_unknown_and_missing_keys` | An unexpected metadata key and a missing required one both fail closed. | Encodes the nine `L2hhMetadata` fields plus a `bogus_field`, and the same fields with `cols` omitted, and verifies neither decodes as `L2hhMetadata`. |
| `count_l2hh_empty_has_one_encoding` | A sketch holding nothing has exactly one encoding. | Verifies a fresh `3x32` sketch and one cleared after an insert serialize to identical bytes. |
| `countl2hh_rejects_too_many_rows` | CountL2HH carries the seed list's row bound on both sides. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### SpaceSaving

Test file: [`tests/e2e_heavy_hitters.rs`](../tests/e2e_heavy_hitters.rs)

Unit tests: [`src/sketches/space_saving.rs`](../src/sketches/space_saving.rs)

Wire tests: [`src/sketches/space_saving/wire.rs`](../src/sketches/space_saving/wire.rs)

Conformance: [`tests/conformance_kit.rs`](../tests/conformance_kit.rs) runs `space_saving_passes_frequency_conformance`, the shared one-sided `frequency_battery` over 1,024 counters.

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `a_monitored_key_is_sandwiched_by_its_error` | Every monitored key's estimate brackets its true count. | Over a seeded Zipf stream, verifies each monitored key reads at or above its true count and no further above it than that key's own `error`, and that the summary is saturated at its capacity. |
| `an_unmonitored_key_never_exceeds_the_minimum_count` | Dropped keys stay under the `min_count` ceiling. | Verifies every key the summary evicted has a true count at or below `min_count`, and that `upper_bound` and `error` for such a key both report that ceiling. |
| `the_true_heavy_hitters_are_reported_exactly_and_in_order` | `top_k` returns the real heaviest keys in descending order. | Compares `top_k` against exact `FreqTruth` ordering and verifies the reported counts are exact and carry zero error, a true heavy hitter never having been displaced. |
| `a_guaranteed_key_really_outranks_everything_dropped` | `is_guaranteed` is a claim about the real stream. | Verifies every key accepted by `is_guaranteed` has a true count strictly above the largest true count among the keys the summary dropped. |
| `an_arrival_displaces_the_minimum_and_inherits_its_count` | Eviction hands the victim's count to the arrival as its error. | Drives a hand-built sequence through a two-counter summary and verifies the new key takes the minimum slot at `min + 1` with the old minimum as its error allowance, while `total` still counts every arrival. |
| `the_stream_summary_lists_stay_well_formed_under_eviction` | The bucket and counter lists stay traversable under sustained eviction. | Across capacities `1`, `2`, `17`, `256`, and `1_024`, verifies residency equals `min(capacity, distinct)`, the bucket walk reaches every counter exactly once in descending count order, and `entries` agrees with the walk. |
| `a_weighted_arrival_matches_repeating_it` | `insert_many` matches that many single inserts. | Builds two summaries over the same stream, one key-at-a-time and one weighted per key, and verifies equal totals, equal residency, and equal per-key estimates. |
| `a_summary_larger_than_the_domain_is_exact` | A summary with room for the whole domain is exact. | With capacity above the distinct count, verifies every estimate equals the true count, every error is `0`, and `min_count` stays `0`. |
| `a_merge_keeps_the_total_and_never_reads_low` | Merge conserves the inserted mass and stays one-sided. | Partitions the stream into two shards, merges their summaries, and verifies the merged total equals the whole stream length, residency stays within capacity, and no monitored key reads below either side's truth. |
| `a_serde_round_trip_preserves_every_answer` | Serde round trip preserves state and the rebuilt key index. | Round-trips through `rmp-serde` and verifies residency, capacity, total, `min_count`, and per-key estimate/error all match, then confirms the decoded summary still takes inserts against its rebuilt index. |
| `an_empty_summary_answers` | An empty summary answers every query without panicking. | Verifies `len`/`total`/`min_count` are zero, point queries return `0`, `is_guaranteed` is false, and `top_k`/`entries` are empty. |
| `a_weighted_arrival_matches_repeating_it_under_eviction` | A weighted arrival matches repeating it once the summary is evicting. | Feeds the same Zipf stream key by key into two 64-counter summaries, one arrival at a time and one `insert_many` per key, verifies the run saturates all 64 counters, and that `total`, `min_count`, and the whole sorted key/count/error entry list agree. |
| `bulk_insert_matches_repeated_inserts_and_a_zero_weight_is_inert` | `bulk_insert` is the loop it replaces, and a zero weight records nothing. | Verifies a 128-counter summary filled by `bulk_insert` matches one filled one at a time on `total` and on `top_k(usize::MAX)`, then that `insert_many` at weight `0` for a resident key and for an absent one leaves `total`, residency, and `top_k` untouched. |
| `a_zero_capacity_summary_still_answers` | A summary asked for no counters still holds one. | Verifies `with_capacity(0)` reports a capacity of `1`, and that after ten distinct keys residency is `1`, `total` is `10`, and the single held counter reads `10` with an error of `9`. |
| `a_merge_into_an_under_full_summary_keeps_the_ceiling_honest` | A merge carries the peer's ceiling even when there are counters to spare. | Merges a one-counter summary that saw key 7 ten times and key 8 twenty times into an empty 33-counter summary, then verifies residency is `1` and below capacity, `upper_bound` for the dropped key 7 still covers its true `10`, and `is_guaranteed` is false for the key that survived. |
| `a_merge_chain_stays_one_sided_against_the_truth` | A chain of merges over asymmetric capacities stays one-sided. | Shards the stream three ways by position into summaries of capacity 64, 512, and 7, merges them in turn, and verifies the merged `total` equals the stream length, residency stays within capacity, every true key's `upper_bound` covers its true count, every monitored key reads at or above its truth, and every key `is_guaranteed` accepts truly outranks the largest dropped key. |
| `a_merge_picks_the_same_survivors_every_time` | The merge depends on what the summaries hold, not on arrival order. | Merges two 200-counter summaries over keys `0..150` and `100..250`, verifies the union fills the capacity at 200 with exactly the 50 shared keys at count `2`, that five repeats give an identical key/count/error list, and that reversing both arrival orders gives the same list. |
| `a_round_trip_carries_the_merged_ceiling` | The ceiling a merge established survives the wire. | Merges a 4-counter summary of the stream's second half into a 32-counter summary of its first half, round-trips through `rmp-serde`, and verifies `min_count`, `total`, and residency are unchanged, every key's estimate matches, and every true key's decoded `upper_bound` still covers its true count. |
| `crafted_state_fails_closed` | Serialized state no run could have produced is refused. | Verifies hand-built payloads with a zero capacity, entries over the capacity, a counter at `0`, an error above its count, a repeated key, a ceiling of `200` over the lowest count of `3`, and a `total` of `2` under a counter's `9` all fail to decode, while a well-formed two-counter state answers `top_k` and `estimate`. |
| `a_fresh_summary_is_well_formed` | A fresh summary passes the structural check. | For `with_capacity(4)`, verifies `validate()` succeeds, `min_count` is `0`, and `capacity` is `4`. |
| `a_capacity_of_zero_floors_at_one` | A requested capacity of zero floors at one counter. | Verifies `with_capacity(0)` reports a capacity of `1`, and that after two distinct keys `validate()` passes, residency is `1`, and the survivor reads `2`. |
| `a_weighted_arrival_displaces_the_minimum_and_starts_above_it` | A weighted eviction starts the arrival above the count it displaced. | In a two-counter summary holding `1` at 5 and `2` at 2, `insert_many(3, 4)` is verified to evict key 2, leaving key 3 at `6` with error `2`, key 2 at `0`, key 1 at `5`, `min_count` at `5`, `total` at `11`, and `validate()` passing. |
| `a_weighted_raise_passes_every_bucket_below_its_destination` | A multi-hop raise walks the bucket list rather than skipping it. | Over counters at 1, 2, 3, and 4, `insert_many(1, 9)` lifts key 1 from the bottom to the top; verifies the descending walk becomes `[(1,10), (4,4), (3,3), (2,2)]` and `validate()` passes. |
| `counts_saturate_and_keep_the_bucket_order` | Counts saturate at `u64::MAX` instead of wrapping. | Raises a counter seeded at `u64::MAX - 2` twice past the ceiling, verifies `validate()` after each raise, both saturated keys read `u64::MAX`, `total` is `u64::MAX`, and the three-counter walk stays descending, ending at the small counter's `7`. |
| `an_eviction_from_a_saturated_counter_stays_sound` | Evicting a saturated counter keeps the structure sound. | A one-counter summary holding `u64::MAX` is evicted by a new key; verifies `validate()` passes, residency is `1`, the arrival reads `u64::MAX` with an error of `u64::MAX`, and the evicted key's `upper_bound` is `u64::MAX`. |
| `a_merge_saturates_instead_of_wrapping` | Merge saturates rather than wrapping past the ceiling. | Merges a three-counter peer into a two-counter summary already at `u64::MAX` and `u64::MAX - 1`; verifies `validate()` passes, residency stays `2`, `total` is `u64::MAX`, both survivors read `u64::MAX`, and all four keys' `upper_bound` is `u64::MAX`. |
| `a_merge_carries_the_ceiling_into_an_under_full_summary` | A merge into an empty summary still raises its ceiling. | Merging a one-counter peer that saw key 7 ten times and key 8 twenty times into an empty 33-counter summary is verified to leave residency `1` below capacity, `min_count` at or above `10`, `upper_bound(7)` at or above `10`, `is_guaranteed(8)` false, and `validate()` passing. |
| `a_truncating_merge_keeps_the_keys_the_encoder_emits_first` | A merge that must truncate a count tie cuts it by key order alone. | Merges a 3-counter peer holding keys 40, 50, and 60 at 2 into a 4-counter summary holding key 1 at 9 and keys 10, 20, and 30 at 3; verifies `validate()` passes, the six tied keys each reach an `upper_bound` of `5`, the survivors are keys 1, 10, 20, and 30 while the swapped merge keeps 1, 10, and 20, and that the encoder emits those survivors in the same order. |
| `a_chain_of_merges_keeps_the_ceiling_above_everything_dropped` | Two merges compound their ceilings. | Merges two one-counter peers (7 at 10 against 8 at 20, then 9 at 5 against 10 at 7) into a 5-counter summary; verifies `validate()` after each merge, residency below capacity, `upper_bound` covering keys 7 and 9 at their true 10 and 5, and `min_count` at or above `15`. |
| `a_key_that_re_enters_after_a_merge_never_reads_low` | A key re-entering a merged summary comes back above the ceiling. | After a merge drops key 7 at its true 12, one further insert is verified to seat it at `min_count + 1` with `min_count` as its error, at or above its true `13`, with `validate()` passing. |
| `randomized_operations_keep_the_structure_sound` | Randomized weighted inserts keep the structure sound at every step. | Drives 4,000 `insert_many` calls of weight 1, 3, 11, or 97 over a 96-key domain at capacities 1, 2, 7, 64, and 257, calling `validate()` after every step, then checks residency equals `min(capacity, distinct)`, `total` equals the inserted mass, the walk is the right length and descending, and every truth is bracketed by its counter's estimate and error. |
| `randomized_merges_keep_the_structure_sound` | Randomized merges keep the structure sound and one-sided. | For five capacity/domain pairs, merges two independently fuzzed summaries into a fuzzed one and then inserts 200 more keys, calling `validate()` after each merge and after the inserts, and checks the error sandwich against the combined truth, `total` equal to the combined mass, and residency within capacity. |
| `a_decoded_summary_rebuilds_both_link_directions` | Decoding rebuilds the bucket and counter links, not just the counts. | Round-trips a fuzzed 48-counter summary through `rmp-serde` and verifies the decoded summary passes `validate()`, matches residency, `min_count`, and `total`, still brackets every truth, and walks the same key/count set. |
| `a_crafted_state_fails_closed` | `rebuild` refuses state the algorithm could not have produced. | Verifies a zero capacity, entries over the capacity, a counter at `0`, an error above its count, a repeated key, a `discarded_max` of `u64::MAX` over the lowest count of `3`, and a `total` of `0` under a counter's `9` are each rejected with the matching complaint. |
| `a_declared_capacity_is_not_allocated_on_decode` | A huge declared capacity is not allocated on decode. | A state declaring a capacity of `1 << 40` with a single entry is verified to rebuild, pass `validate()`, report that capacity, and hold one counter. |
| `a_non_utf8_byte_key_is_monitored_and_queried` | A byte array that is not UTF-8 is a key like any other. | Inserts `[0xff, 0x00, 0xfe]` three times and `[0x00, 0x80]` at weight 2 into a 4-counter summary, and verifies `validate()` passes, the repeats share one counter, both keys are held as their raw bytes, each estimates its weight, and an unseen byte key reads `0`. |
| `a_byte_key_and_a_string_key_are_separate_counters` | `Bytes(b"abc")` and `Str("abc")` are two keys, not one. | Seats both in a 4-counter summary at weights 5 and 2 and verifies residency is `2`, the byte key reads `5`, and the string key reads `2` through `Str` and `String` alike. |
| `an_evicted_byte_key_keeps_its_bound` | An evicted byte key still reads under the ceiling. | In a 2-counter summary holding a byte key at 3 and another at 2, a third byte key is verified to displace the smallest, leaving the evicted key at `estimate` `0` with an `upper_bound` at or above its true `2`, a `min_count` of `3`, and the newcomer at `3` carrying an error of `2`. |
| `a_merge_pairs_byte_keys_by_their_bytes` | A merge pairs byte keys by their bytes. | Merges a peer holding the shared byte key at 2 and its own at 7 into a summary holding the shared key at 5 and its own at 3, and verifies `validate()` passes, all three byte keys survive as their raw bytes, and the shared key reads `7`. |
| `a_serde_round_trip_keeps_a_byte_key` | A serde round trip carries raw bytes and re-hashes them to the digest a query reaches. | Round-trips a summary holding `[0xff, 0x00, 0xfe]` at 9 and five `0x00` bytes at 4 through `rmp-serde`, and verifies the decoded summary passes `validate()`, holds the same raw byte keys, answers both `DataInput::Bytes` queries at their weights, and reports the same `total`. |
| `space_saving_envelope_structure_and_round_trip` | The envelope frames a summary under kind_id `0x18 0x00`, and the bytes are stable across a round trip. | For a 48-counter summary saturated by 30,000 weighted draws over 400 distinct keys, verifies the bytes open with the ASAPv1 magic, `envelope::VERSION`, a `kind_id_len` of `2`, and `0x18 0x00`, that re-serializing the decoded summary reproduces them exactly, and that residency, capacity, `total`, `min_count`, and `estimate`/`error`/`upper_bound`/`is_guaranteed` agree across all 400 keys. |
| `space_saving_merged_ceiling_survives_the_wire` | The ceiling a merge leaves behind travels in `floor`, which the triples do not determine. | Merges a one-counter summary that saw key 7 ten times and key 8 twenty times into an empty 33-counter summary, then verifies the decoded `min_count` and the `upper_bound` on the dropped key 7 both match the source and still cover its true `10`, with identical re-serialized bytes; a payload carrying only the triples decodes to a `min_count` of `0` here. |
| `space_saving_empty_round_trip` | A summary monitoring nothing has one encoding. | Verifies an empty 16-counter summary reports the pinned `EMPTY_KEY_TYPE` of `u64` in its metadata and decodes back to residency `0`, capacity `16`, `min_count` `0`, `total` `0`, and identical re-serialized bytes. |
| `space_saving_every_key_type_round_trips_and_keeps_its_variant` | The `key_type` names the exact `HeapItem` variant and is never widened. | Across all 14 wire key types - `i8`, `i16`, `i32`, `i64`, `isize`, `u8`, `u16`, `u32`, `u64`, `usize`, `f32`, `f64`, `string` reached from `Str` and `String` alike, and `bytes` - verifies the metadata carries the expected name, the bytes re-serialize identically, and every decoded key still answers its original `DataInput` with the weight it was given; a decoder widening `i32` to `i64` keeps the digest but stops comparing equal, so `estimate` would read zero. |
| `space_saving_emitted_order_is_independent_of_seat_order` | The emitted order is descending count with ties broken by `key_order`, not the arena's seat order. | Rebuilds the same four triples - carrying two count ties that only the key separates - in one order and in reverse, and verifies both serialize to the same bytes. |
| `space_saving_refuses_mixed_and_128_bit_keys` | Keys the wire cannot carry refuse to serialize rather than being coerced. | Verifies a summary holding both an `I32` and an `I64` key fails to serialize, that a lone `I128` or `U128` key fails, and that a 128-bit key seated behind a wire-eligible one is caught on the way into the payload rather than by the first-key check. |
| `space_saving_rejects_a_key_type_the_payload_does_not_carry` | A payload relabelled with another `key_type` is refused. | Re-frames a string-keyed payload under `u64` and under `bytes`, a `u64`-keyed payload under `string` and under `bytes`, and a byte-keyed payload under `u64` and under `string`, and verifies none of them decodes. |
| `space_saving_byte_keys_round_trip_arbitrary_bytes` | A byte-array key reaches the wire as msgpack `bin`, so any byte string survives. | For an 8-counter summary holding `[0xff, 0x00, 0xfe]`, forty `0x80` bytes, and the empty byte string at weights 3, 6, and 9, verifies the metadata names `key_type` `bytes`, the emitted `keys` carry exactly those bytes, every decoded key still answers its original `DataInput::Bytes` at its weight, and the bytes re-serialize identically. |
| `space_saving_refuses_byte_keys_mixed_with_string_keys` | A `Bytes` key and a `String` key are different key types, so a summary holding both has no `key_type`. | Seats `Bytes(b"abc")` and `Str("abc")` in one summary, verifies they take two counters, and that serializing fails with a "mix variants" complaint. |
| `space_saving_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the summary rather than a hardcoded profile. | For a four-counter summary over a hasher declaring its own `HashProfile`, verifies the bytes round-trip with `estimate` intact and re-serialize identically, differ from the standard-profile summary's bytes over the same keys, and are rejected by a standard-profile decode. |
| `space_saving_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the eight `SpaceSavingMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `SpaceSavingMetadata`. |
| `space_saving_rejects_a_crafted_envelope` | Every structural rule is pinned by the complaint it fails with. | Verifies eleven crafted envelopes are each refused for their own reason: a zero capacity, entries over the capacity, `keys` longer than `counts`, `counts` longer than `errors`, a counter at zero, an error of `4` against a count of `3`, the same key twice, an unknown `key_type` of `u256`, a `discarded_max` of `u64::MAX` over the lowest count of `3`, a `total` of `0` under a counter's `9`, and a foreign kind_id. |
| `space_saving_refuses_a_capacity_the_metadata_cannot_carry` | A capacity past the metadata's `u32` field fails the encode rather than truncating. | Rebuilds a one-counter summary declaring a capacity of `1 << 40`, verifies it reports that capacity, and that serializing fails with a message naming the "exceeds the u32 metadata field" rule. |
| `space_saving_does_not_allocate_a_declared_capacity` | A declared capacity is metadata, never an allocation size. | Verifies an envelope declaring `u32::MAX` counters with two entries decodes, reports that capacity, holds two counters with key 7 reading `9`, a `min_count` of `0`, and a two-entry `top_k`. |

### Bloom

Test file: [`tests/e2e_membership.rs`](../tests/e2e_membership.rs)

Unit tests: [`src/sketches/bloom.rs`](../src/sketches/bloom.rs)

Wire tests: [`src/sketches/bloom/wire.rs`](../src/sketches/bloom/wire.rs)

Conformance: [`tests/conformance_kit.rs`](../tests/conformance_kit.rs) runs the shared `membership_battery` on both hash paths against a filter sized for 20,000 keys at a 1% target: `bloom_passes_membership_conformance` over `Bloom<FastPath>` and `bloom_regular_path_passes_membership_conformance` over `Bloom<RegularPath>`. The battery also checks a `predicted-false-positive-rate` band, holding the measured rate to within a quarter either side of the filter's own `predicted_fpp`.

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `an_inserted_key_is_never_reported_absent` | No false negative, on either hash path. | Fills a filter sized for 20,000 keys at a 1% target and verifies every inserted key reports present on both the `RegularPath` and `FastPath` filters, which decode columns differently. |
| `an_empty_filter_rejects_every_probe` | A fresh filter reports nothing present. | Verifies `is_empty` and a zero insert count, and that probes disjoint from the member set are rejected on both hash paths. |
| `the_measured_false_positive_rate_matches_what_the_sizing_predicts` | The delivered rate honours the target and matches `predicted_fpp`. | Across targets `0.1`, `0.01`, and `0.001`, measures the rate over 200,000 disjoint probes and verifies predicted and measured both stay at or under the target and agree within five binomial standard errors. |
| `the_fill_based_estimate_tracks_the_measured_rate` | `estimated_fpp` reads the bits set, not the insert count. | Inserts the member set three times and verifies `estimated_fpp` still tracks the measured rate within a quarter either way although the insert count has tripled. |
| `repeated_inserts_leave_the_bits_unchanged` | Fill is a function of the distinct keys alone. | Builds one filter with single inserts and one with duplicated inserts over the same keys and verifies identical `count_ones` and `fill_ratio`. |
| `a_union_equals_the_filter_of_the_concatenated_stream` | Merge is an exact union, which is what makes the filter shardable. | Splits a stream across two same-geometry filters, merges them, and verifies the same set bits and insert count as a single-pass filter, plus identical answers on members and non-members alike. |
| `sizing_meets_the_target_and_is_one_power_of_two_from_missing_it` | `dimensions_for` is pinned to its contract rather than to its own expression. | For `(n, p)` pairs from `(1, 0.5)` through `(10_000, 1e-12)`, verifies every slice is a power of two and the row count stays within `BLOOM_MAX_SLICES`, that the chosen geometry's `predicted_fpp(n)` meets the target, and that halving `cols` misses it. |
| `degenerate_geometries_still_answer` | Degenerate geometries answer rather than panic. | Verifies a one-bit filter answers and reports an `estimated_fpp` of `1.0` once set, and that a single-row filter still holds every member it was given. |
| `clearing_restores_an_empty_filter` | `clear` restores the constructed state and keeps the dimensions. | After clearing a filled filter, verifies it is empty with a zero insert count, unchanged rows and columns, and no member reported present. |
| `a_serde_round_trip_preserves_every_answer` | Serde round trip preserves every answer. | Round-trips through `rmp-serde` and verifies matching dimensions, insert count, and set bits, plus identical answers on members and non-members. |
| `storage_is_one_bit_per_cell` | Packed storage is one bit per cell, not one byte. | For an `8x4096` filter, verifies `bit_capacity` equals `rows * cols` and `size_in_bytes` is one eighth of it. |
| `sizing_never_asks_for_more_slices_than_the_seed_list_has` | The slice count stops at the number of seeds the hasher has. | For 10,000 keys at targets `1e-7`, `1e-9`, and `1e-12`, verifies the row count stays within `BLOOM_MAX_SLICES` and that no two rows of the filled filter hold identical bits. |
| `sizing_stays_inside_the_allocation_ceiling` | A target the seed list cannot reach yields the widest geometry that fits. | For `expected_items` up to `usize::MAX` and targets `f64::MIN_POSITIVE`, `0.0`, and `2.0`, verifies `cols` is at least one and a power of two, rows fall in `1..=BLOOM_MAX_SLICES`, and `rows * cols` stays within `BLOOM_MAX_BITS`. |
| `a_nan_target_rate_is_rejected` | A NaN target rate panics. | Verifies `with_capacity(1_000, f64::NAN)` panics with "target false-positive rate must be finite". |
| `an_infinite_target_rate_is_rejected` | An infinite target rate panics. | Verifies `with_capacity(1_000, f64::INFINITY)` panics with "target false-positive rate must be finite". |
| `a_negative_infinite_target_rate_is_rejected` | A negative infinite target rate panics. | Verifies `with_capacity(1_000, f64::NEG_INFINITY)` panics with "target false-positive rate must be finite". |
| `the_two_hash_paths_agree_only_where_the_geometry_gives_each_row_its_own_hash` | Whether the paths land on the same bits is a property of the geometry. | Verifies `8 x 65_536` needs 136 hash bits, so both paths fall back to one seeded hash per row and set identical bits, while `7 x 65_536` fits a packed hash and the paths differ; also verifies that second shape is `BLOOM_DEFAULT_ROWS` by `BLOOM_DEFAULT_COLS`. |
| `a_filter_cannot_be_decoded_into_the_other_hash_path` | The serialized form carries which path built the filter. | Verifies regular-path bytes fail to decode as `Bloom<FastPath>` with an error naming both paths, fast-path bytes fail as `Bloom<RegularPath>` and as the unannotated `Bloom`, and fast-path bytes decoded back into `Bloom<FastPath>` still report every member present. |
| `a_fast_path_serde_round_trip_preserves_every_answer` | The fast path round-trips like the regular one, tag and all. | Round-trips a fast-path filter sized for 20,000 keys at a 1% target and verifies matching dimensions, insert count, and set bits, plus identical answers on 20,000 non-member probes. |
| `a_fast_path_union_equals_the_filter_of_the_concatenated_stream` | Merge is an exact union on the fast path too. | Splits 10,000 keys by parity across two `7 x 2^14` fast-path filters, merges them, and verifies the same set bits and insert count as a single-pass filter, with every key reported present. |
| `merging_filters_of_different_widths_panics` | A merge across slice widths panics. | Verifies merging a `7 x 2^14` filter with a `7 x 2^13` one panics with "bit matrices must have the same dimensions". |
| `merging_filters_of_different_slice_counts_panics` | A merge across slice counts panics. | Verifies merging a `7 x 2^14` filter with an `8 x 2^14` one panics with "bit matrices must have the same dimensions". |
| `bulk_insert_matches_inserting_one_at_a_time` | `bulk_insert` is the loop, not a different filter. | On both hash paths, verifies a filter filled by `bulk_insert` over 20,000 keys has the same set bits and insert count as one filled key by key. |
| `with_capacity_builds_exactly_the_geometry_dimensions_for_reports` | The sizing helper describes the filter `with_capacity` actually builds. | Over `expected_items` `0`, `1`, `17`, `1_000` and `250_000` against targets `0.5`, `0.1`, `0.01`, `1e-4` and `1e-6`, verifies the built filter's rows and columns equal the pair `dimensions_for` reported for the same arguments. |
| `the_slice_count_comes_from_the_bit_budget_not_the_target_alone` | The slice count tracks `expected_items` as well as the target rate. | Verifies `dimensions_for` chooses `5` slices at `(1, 0.05)`, `8` at `(1, 1/128)` and `14` at `(2, 1e-4)`, each exactly one over `round(log2(1/p))`; that the two agree for `expected_items` `1_000`, `100_000` and `10_000_000` across six targets; and that `(1_000_000, 1e-12)` saturates at `BLOOM_MAX_SLICES`. |
| `a_target_outside_the_open_unit_interval_is_clamped_to_the_endpoints` | A target outside `(0, 1)` clamps onto the interval's own endpoints. | Verifies `0.0`, `-0.0`, `-1.0` and `f64::MIN` all size as `f64::MIN_POSITIVE` does, that `1.0`, `1.5` and `f64::MAX` all size as `1.0 - f64::EPSILON` does, and that both endpoint geometries are non-zero with a power-of-two width. |
| `a_target_that_is_not_a_number_panics_instead_of_sizing_a_degenerate_filter` | A NaN target panics rather than clamping into a degenerate filter. | Verifies `dimensions_for(1_000, f64::NAN)` panics naming the finite-rate requirement. |
| `an_infinite_target_panics_instead_of_sizing_a_degenerate_filter` | The same panic reaches through `with_capacity`, not only the helper it delegates to. | Verifies `with_capacity(1_000, f64::INFINITY)` panics naming the finite-rate requirement. |
| `sizing_for_zero_expected_items_still_yields_a_filter_that_answers` | Sizing for no items still yields a filter that holds a key. | Verifies `dimensions_for(0, 0.01)` has both dimensions at least one and equals `dimensions_for(1, 0.01)`, and that the filter built from it reports its only member present. |
| `the_default_geometry_is_the_documented_pair` | `Default` is the two public constants, and they name a filter the hash family can serve. | Verifies a default filter is `BLOOM_DEFAULT_ROWS x BLOOM_DEFAULT_COLS`, that its rows are within `BLOOM_MAX_SLICES`, its width a power of two, its bit capacity under `BLOOM_MAX_BITS`, and that it starts empty. |
| `more_slices_than_the_seed_list_has_is_rejected_at_construction` | More slices than the seed list has is refused at construction, not at serialization. | Verifies `with_dimensions(BLOOM_MAX_SLICES + 1, 1024)` panics with a message naming `BLOOM_MAX_SLICES`. |
| `the_seed_list_length_itself_is_a_legal_slice_count` | The bound is inclusive: the assert fires past it, not at it. | Verifies `with_dimensions(BLOOM_MAX_SLICES, 1024)` builds and reports that many rows. |
| `a_serde_payload_past_the_slice_cap_is_rejected` | The plain serde form is a second door into the filter and fails closed on the same bound. | Encodes the serialized form over a `BLOOM_MAX_SLICES + 1` row grid and verifies the `rmp-serde` decode returns an error naming `BLOOM_MAX_SLICES` rather than panicking inside the decoder. |
| `both_rate_reporters_raise_the_per_slice_rate_to_the_slice_count` | Both rate reporters use the slice count as the exponent. | For a `BLOOM_MAX_SLICES x 1024` filter holding 400 keys at a fill strictly inside `(0, 1)`, verifies `estimated_fpp` equals `fill_ratio^rows` and `predicted_fpp(400)` equals `(1 - e^(-400/cols))^rows`. |
| `an_empty_filter_predicts_nothing_and_a_saturated_one_predicts_everything` | The two ends of the rate scale are reported exactly. | Verifies an empty filter reports a zero fill ratio, a zero `estimated_fpp` and `predicted_fpp(0) == 0`, then fills a `3x8` filter until its fill ratio is `1.0` and verifies `estimated_fpp` is exactly `1.0`. |
| `a_single_insert_sets_exactly_one_bit_in_every_slice` | Partitioning: one key touches exactly one bit per slice, whatever the width. | Across widths `1`, `2`, `7`, `64`, `100` and `1024` by row counts `1`, `5` and `BLOOM_MAX_SLICES`, verifies one insert sets exactly `rows` bits on both hash paths, that `fill_ratio` equals `rows / (rows * cols)`, and that the key reads present. |
| `the_insert_counter_counts_calls_and_clear_returns_it_to_zero` | `inserted` counts calls rather than distinct keys, and `clear` resets it with the bits. | Verifies five inserts of one key count `5`, a `bulk_insert` of three more leaves `8`, and that `clear` returns the count, the bits and the fill ratio to zero. |
| `merging_identical_geometries_sums_the_insert_counts` | A merge adds the counts and keeps both streams' members. | Merges two `5x512` filters holding 30 and 40 distinct keys and verifies the merged count is the sum, all 70 members read present, and the geometry is unchanged. |
| `merging_mismatched_geometries_panics_instead_of_unioning_a_prefix` | A cross-geometry union panics rather than answering no about its own members. | Verifies merging a `5x512` filter with a `5x256` one panics in `BitMatrix::union_from`. |
| `a_zero_slice_width_is_rejected_at_construction` | A zero dimension is refused at construction. | Verifies `with_dimensions(4, 0)` panics rather than handing back a filter every query would panic on. |
| `bit_capacity_and_packed_size_describe_the_same_grid` | The addressable grid and the packed storage behind it describe one filter. | Across `1x1`, `3x7`, `5x64`, `7x65` and `BLOOM_MAX_SLICES x 1024`, verifies `bit_capacity` is `rows * cols`, `size_in_bytes` is `rows * ceil(cols / 64) * 8`, and that the bytes cover the addressable bits. |
| `bloom_envelope_structure_and_round_trip` | The envelope frames a filter under kind_id `0x17 0x00`, and the bytes are stable across a round trip. | For a `7 x 2^14` regular-path filter holding 5,000 members, verifies the bytes open with the ASAPv1 magic, `envelope::VERSION`, a `kind_id_len` of `2`, and `0x17 0x00`, then that re-serializing the decoded filter reproduces them exactly and that `inserted`, `fill_ratio`, every member, and 20,000 disjoint probes all answer as the source does. |
| `bloom_fast_path_round_trip` | The fast path round-trips byte for byte. | For a `7 x 2^14` fast-path filter holding 5,000 members, verifies the decoded filter re-serializes to the same bytes, keeps `inserted`, and answers identically on the members and on 20,000 disjoint probes. |
| `bloom_empty_round_trip` | A filter with no inserts round-trips. | Verifies `Bloom::<RegularPath>::default()` decodes back empty with a zero insert count, unchanged rows and columns, and identical re-serialized bytes. |
| `bloom_cross_mode_rejection` | The metadata `mode` turns a cross-mode decode into an error rather than a filter that denies its own members. | For `7 x 2^12` filters on both paths over the same 5,000 members, verifies the two byte strings differ, regular-path bytes fail to decode as `Bloom<FastPath>`, and fast-path bytes fail as `Bloom<RegularPath>`. |
| `bloom_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the filter rather than a hardcoded profile. | For a `7 x 2^12` filter over a hasher declaring its own `HashProfile`, verifies the bytes round-trip and preserve every answer, differ from the standard-profile filter's bytes over the same members, and are rejected by a standard-profile decode. |
| `bloom_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the ten `BloomMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `BloomMetadata`. |
| `bloom_rejects_foreign_kind_id` | Another sketch's kind_id is refused even when the rest parses cleanly. | Wraps well-formed Bloom metadata and payload in an envelope carrying kind_id `0x02 0x00` and verifies the decode fails. |
| `bloom_rejects_zero_dimension` | A zero dimension is a decode error, not a `BitMatrix::new` panic. | For crafted `0 x 64` and `4 x 0` envelopes, verifies the decode fails with a message naming the "dimensions must be non-zero" rule. |
| `bloom_rejects_non_power_of_two_cols` | A modulo-folded width is outside the wire-eligible subset. | Verifies a crafted `2 x 96` envelope fails with "not a power of two", while the same four-word payload at `2 x 128` decodes, so the geometry rule is what rejects it and not the word count. |
| `bloom_rejects_too_many_rows` | More slices than the seed list has entries is outside the wire-eligible subset. | Verifies a crafted `BLOOM_MAX_SLICES + 1` envelope fails with a message naming `BLOOM_MAX_SLICES`, and that the boundary row count itself decodes. |
| `bloom_rejects_oversized_bit_capacity` | The capacity rule fires from the declared dimensions, before any allocation is sized from them. | Verifies a crafted envelope declaring `4 x (BLOOM_MAX_BITS / 2)` while carrying two words fails with a message naming `BLOOM_MAX_BITS`. |
| `bloom_rejects_word_count_mismatch` | The word stride is derived from `cols`, so a payload that disagrees is refused in either direction. | Verifies 5-word and 7-word payloads for a `3x128` grid both fail with a "words length" complaint, while the correct 6 words decode. |
| `bloom_rejects_padding_bits_set` | Bits parked in a row's trailing slack are rejected. | For a `2x8` grid whose bits `8..64` are padding, verifies payloads setting the first padding bit, the last, and one mid-row are each refused with a "row padding" complaint - such bits are unreachable by `get` but counted by `count_ones`, so they would skew `fill_ratio` and `estimated_fpp` while every membership answer looked fine - and that the same words with the padding clear decode to a `fill_ratio` of `0.5`. |
| `bloom_with_capacity_round_trip_meets_predicted_fpp` | A filter sized for a realistic target still delivers it after a round trip. | For 50,000 keys at a `0.001` target, verifies the decoded filter keeps the source's dimensions, reports every member present, and delivers a measured rate over 200,000 disjoint probes within five binomial standard errors of its own `predicted_fpp`. |

### Elastic

Test file: [`src/sketches/elastic.rs`](../src/sketches/elastic.rs)

Wire tests: [`src/sketches/elastic/wire.rs`](../src/sketches/elastic/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `init_with_dimensions_sizes_both_parts` | Both parts take the requested dimensions. | `init_with_dimensions(12, 2, 256)` is verified to yield 12 heavy buckets and a 2x256 light layer. |
| `init_with_length_keeps_the_default_light_layer` | The old constructor is unchanged. | `init_with_length(8)` is verified to yield 8 heavy buckets over `DEFAULT_LIGHT_ROWS` x `DEFAULT_LIGHT_COLS`. |
| `an_empty_heavy_table_is_rejected` | A zero bucket count panics. | `init_with_length(0)` is verified to panic with "at least one heavy bucket" rather than divide by zero in the bucket index. |
| `a_negative_heavy_table_is_rejected` | A negative bucket count panics. | `init_with_length(-1)` is verified to panic rather than widen to a huge `usize`. |
| `an_empty_light_layer_is_rejected` | A zero-row light layer panics. | `init_with_dimensions(8, 0, 4096)` is verified to panic with "non-empty light layer". |
| `a_zero_width_light_layer_is_rejected` | A zero-column light layer panics. | `init_with_dimensions(8, 3, 0)` is verified to panic with "non-empty light layer". |
| `heavy_bucket_tracks_repeated_flow_exactly` | Heavy bucket tracks repeated flow exactly. | Top-K/heavy-hitter tracking and updates behave as expected. |
| `light_sketch_counts_colliding_flows` | Light sketch counts colliding flows. | Core functional behavior for this component path is validated. |
| `eviction_moves_the_resident_flow_into_the_light_layer` | Takeover evicts the resident flow, not the arriving one. | After 10 inserts of a resident and `LAMBDA * 10` inserts of a colliding flow, verifies the bucket holds the arrival with `(vote_pos, vote_neg, eviction) = (1, 1, true)`, `query(resident) == 10`, and `query(arrival) == 80`. |
| `expansion_doubles_the_heavy_table` | Copy operation doubles bucket count. | After `expand_heavy()` on an 8-bucket table, verifies `bktlen` and `heavy.len()` are both `16`. |
| `expansion_preserves_every_existing_estimate` | Lemma 3.2 keeps estimates put across a doubling. | Records `query` for 24 flows over 8 buckets, doubles, and verifies every estimate is unchanged. |
| `repeated_expansion_keeps_estimates_intact` | Two doublings in a row stay correct. | Same 24 flows through two `expand_heavy()` calls to `bktlen = 32`, all estimates unchanged. |
| `an_insert_onto_a_stale_copy_replaces_it` | Incremental cleanup drops stale copies. | Finds a stale bucket after a doubling, inserts a key that hashes to it, and verifies the arrival takes the slot at `vote_pos = 1` while the displaced flow keeps its mass elsewhere. |
| `merge_after_expansion_does_not_double_count` | Flushing an expanded table spills each flow once. | Doubles a 24-flow table, merges an empty peer, and verifies every estimate equals its true count rather than twice it. |
| `merge_does_not_double_count_an_expanded_peer` | The peer's stale copies are skipped too. | Merges an expanded 24-flow peer into an unexpanded sketch and verifies no estimate doubles. |
| `maximum_merging_does_not_double_count_an_expanded_peer` | Same for max merging. | As above through `merge_max`. |
| `full_bucket_count_counts_residents_above_the_threshold` | Full-bucket count matches the threshold. | With 6 flows at 10 votes each, verifies `full_bucket_count(9) == 6` and `full_bucket_count(10) == 0`. |
| `compression_shrinks_the_heavy_table` | Active compression divides the bucket count. | After `compress_heavy(4)` on a 16-bucket table, verifies `bktlen` and `heavy.len()` are both `4`. |
| `compression_keeps_the_larger_flow_and_spills_the_smaller` | The bigger resident wins its group. | Puts a 30-vote flow in bucket 0 and a 3-vote flow in bucket 4, halves the table, and verifies the big flow still reads `30` from the heavy part while the small one has left it and reads back at least `3` from the light layer. |
| `compression_neither_loses_nor_doubles_mass` | Compression only ever adds error. | Compresses 40 flows over 16 buckets by 4 and verifies no flow underestimates and the summed estimate rises without doubling. |
| `a_ratio_that_does_not_divide_the_table_is_rejected` | Lemma 3.2 needs `w' \| w`. | `compress_heavy(3)` on an 8-bucket table panics with "must divide the bucket count". |
| `compression_after_expansion_does_not_double_count` | Stale copies are dropped before grouping. | Expands 12 buckets to 24 — putting each twin 12 apart — compresses by 3 so twins land in different groups, verifies no flow is resident twice, then merges an empty peer and verifies every estimate equals its true count. |
| `expand_then_compress_returns_to_the_original_size` | Doubling and halving round-trips. | `expand_heavy()` then `compress_heavy(2)` returns an 8-bucket table to 8 buckets with no flow underestimated. |
| `heavy_only_insert_never_touches_the_light_layer` | Overload mode leaves every light counter alone. | Seeds the light layer through `insert`, snapshots all `2x64` counters, then runs `insert_heavy_only` across a vacant seat, a match, discarded arrivals, and a takeover, and verifies the snapshot is unchanged. |
| `heavy_only_takeover_inherits_the_evicted_flow_size` | Takeover carries the evicted flow's size to the arrival. | After 10 resident votes and `LAMBDA * 10` colliding arrivals, verifies the bucket becomes `(arrival, vote_pos=10, vote_neg=0)` rather than starting at `1`. |
| `heavy_only_takeover_inherits_the_eviction_flag` | Takeover carries the bucket's flag to the arrival. | Seeds a resident bucket's `eviction` to `false` and to `true` in turn, drives a takeover through `insert_heavy_only`, and verifies the arrival reads back the seeded value rather than a forced `true`. |
| `heavy_only_takeover_discards_the_evicted_flow_as_designed` | The evicted flow's size is dropped, not spilled. | In the same scenario, verifies `query` on the evicted flow returns `0` against a true count of `10`. |
| `heavy_only_matches_insert_while_buckets_seat_and_match` | Seating and matching agree with the normal path. | Feeds 6 flows into a 16-bucket table through both paths and verifies every bucket field and every `query` result matches. |
| `merge_keeps_uncontested_flows_in_the_heavy_part` | Merge leaves elephants in the heavy part. | Verifies post-merge `query` returns exactly `30` and `18`, that both flows are still resident with those vote counts, and that every bucket carries the eviction flag. |
| `merge_keeps_the_larger_flow_on_a_contested_bucket` | A contested bucket goes to the larger flow. | With a 20-count and a 9-count flow on one bucket, verifies the 20 keeps the bucket and the loser reads back at `>= 9` from the light layer. |
| `merge_keeps_the_peers_flow_when_it_is_the_larger` | Each side is sized against its own sketch. | A 3-count local flow loses its bucket to the peer's 50-count flow; querying the peer's flow against the local sketch would read near 0 and flip the outcome. |
| `merge_sums_the_votes_of_a_flow_both_sides_held` | A shared elephant is summed, not replaced. | A flow with 30 left and 20 right comes back resident with `vote_pos == 50`. |
| `merge_never_underestimates_across_a_large_flow_set` | Merging preserves the one-sided guarantee. | 60 flows, half of them shared, through a `2x64` light layer; every flow reads back at or above its true count. |
| `merge_does_not_leave_a_stale_copy_as_a_resident` | Expansion copies do not survive a merge. | After a doubling and a merge, `heavy_hitters` reports each flow exactly once; the merge clears the stale flag, so a kept copy would look live. |
| `merge_preserves_colliding_flow_mass` | Merge preserves mass for bucket-colliding flows. | Merges two sketches whose flows share a heavy bucket and verifies both estimates stay at or above their true counts. |
| `a_bucket_reoccupied_after_merge_still_reads_the_light_layer` | A post-merge resident keeps its flushed mass. | After merging a 30-count flow away and re-inserting it once, verifies `query` returns `31` rather than `1`. |
| `maximum_merging_never_underestimates_disjoint_flows` | Maximum merging keeps Elastic's one-sided guarantee. | Merges two sketches over 80 disjoint flows through a `2x64` light layer and verifies every per-flow estimate is at or above its true count. |
| `maximum_merging_is_tighter_than_sum_merging` | Maximum merging beats sum merging on disjoint flows. | Runs the same 80-flow disjoint input through `merge` and `merge_max` and verifies no flow is looser under max and at least one is strictly tighter; measured totals are 434 against 359 for a truth of 275. |
| `maximum_merging_underestimates_a_mouse_flow_both_sides_saw` | Maximum merging's precondition, pinned as behavior. | A mouse flow kept out of the heavy part by a hot flow, inserted 30 times left and 20 times right, reads back as `30` after `merge_max` and `50` after `merge`. |
| `maximum_merging_sums_a_flow_both_heavy_parts_held` | The restriction is on the light half only. | A shared flow resident on both sides comes back with `vote_pos == 50` under `merge_max`, since the heavy parts are combined bucket by bucket either way. |
| `maximum_merging_keeps_the_larger_flow_on_a_contested_bucket` | Maximum merging contests buckets the same way. | The 20-count flow keeps the bucket against a 9-count peer, and the loser reads back at `>= 9`. |
| `heavy_hitters_reports_every_resident_above_the_threshold` | Heavy hitter detection reports the right set. | Over four residents of a 256-bucket table (50/30/12/3), verifies `heavy_hitters(20)` is exactly the 50 and 30 flows, `heavy_hitters(100)` is empty, and `heavy_hitters(1)` has all four. |
| `heavy_hitters_includes_a_flow_sitting_exactly_on_the_threshold` | The threshold is inclusive. | A flow of exactly 20 is reported at `threshold = 20` and one of 19 is not, matching the reference's `val >= threshold`. |
| `heavy_hitters_does_not_report_a_flow_twice_after_expansion` | Expansion does not duplicate hitters. | After `expand_heavy()` leaves every resident a stale copy, verifies three flows come back once each rather than twice. |
| `heavy_changes_reports_only_moves_past_the_threshold` | Heavy change detection filters by size of move. | Over rising (10->55), falling (60->8), and steady (40->42) flows, verifies only the first two are reported at `threshold = 20`. |
| `heavy_changes_covers_a_flow_present_in_only_one_window` | A flow in one window only is a change. | Verifies a flow of 40 that vanishes reports `(40, 0)` and one of 45 that appears reports `(0, 45)`. |
| `heavy_changes_reports_each_flow_once` | Each flow appears once in the change list. | A flow resident in both windows, each expanded so it also has a stale copy, reaches the id list four times and is reported once. |
| `elastic_round_trip_serialization` | The envelope frames a sketch under kind_id `0x0b 0x00`, and the state survives a round trip. | For an 8-bucket heavy table over a `2x256` light layer fed 12 hits of one flow and one of another, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x0b 0x00`, and that the decode keeps `bktlen`, the light dimensions, every heavy bucket, every light counter, and a `query` of `12`. |
| `elastic_rejects_foreign_kind_id` | The inlined Count-Min's own envelope is not an Elastic one. | Verifies a stand-alone `2x256` `CountMin<Vector2D<i32>, RegularPath>` envelope fails to decode as an `Elastic`. |
| `elastic_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the twelve `ElasticMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `ElasticMetadata`. |
| `elastic_metadata_rejects_a_missing_light_cols_key` | `light_cols` is required and can never be silently defaulted. | Encodes the other eleven `ElasticMetadata` fields as a named map with `light_cols` omitted and verifies it does not decode as `ElasticMetadata`. |
| `elastic_rejects_zero_dimension_payload` | A zero dimension in either part is a decode error, not a panic. | Verifies crafted envelopes declaring `0` heavy buckets, `0` light rows, and `0` light columns each fail rather than panicking in the `cols.ilog2()` mask derivation. |
| `elastic_rejects_dimension_length_mismatch` | Both length checks run before the heavy table and the light matrix are built. | Verifies a crafted envelope declaring `1024` buckets over `MATRIX_MAX_ROWS x 2^24` while carrying two buckets and three light counters fails, and that a payload whose heavy arrays agree with each other but whose `light_counts` holds eight entries against a declared `2x256` fails too. |
| `elastic_rejects_a_flow_and_vote_that_disagree_on_occupancy` | A free bucket is `nil` with no vote, and a payload where the two disagree is refused. | Verifies a crafted payload naming a flow whose `vote_pos` is `0`, and one carrying a vote under a `nil` flow, both fail to decode. |
| `elastic_rejects_serializing_an_inconsistent_sketch` | The encode side refuses states its own decoder would reject. | Verifies an 8-bucket sketch whose `bktlen` is set to `16` fails to serialize, and that one whose free bucket names a flow fails too. |
| `elastic_free_buckets_are_nil_and_never_collide_with_an_empty_flow_id` | A free bucket is `nil`, not an empty string. | For an all-free 4-bucket table, verifies the payload holds one `0xc0` per bucket and no `0xa0` and that the decode matches bucket for bucket; then that a table holding an inserted empty flow id emits an `0xa0`, decodes identically, and answers that flow's `query` with `1`. |
| `elastic_mixed_occupancy_round_trips` | A table mixing free, occupied and evicted buckets round-trips bucket for bucket. | For a 16-bucket table over `3x512` fed a resident, a colliding flow past the `LAMBDA` threshold, and six more flows, verifies the fixture holds both vacant and flagged buckets and that the decode matches on every bucket, every light counter, and both flows' `query` results. |
| `elastic_negative_votes_and_light_counters_round_trip` | Votes and light counters are signed on the wire. | Sets a bucket's `vote_neg` to `-300` and `vote_pos` to `-7` and inserts a light weight of `-9` into a 4-bucket `2x8` sketch, then verifies the decode matches on every bucket and every light counter. |
| `elastic_stale_copies_round_trips_in_both_states` | `stale_copies` is carried, since the buckets do not determine it. | Verifies an 8-bucket sketch decodes with the flag `false`, then that after `expand_heavy()` it decodes with the flag `true`, a `bktlen` of `16`, matching buckets, `heavy_hitters` agreeing with the source, and exactly twice as many occupied buckets as reported flows. |
| `elastic_decoded_sketch_reserializes_byte_identically` | The emitted order is bucket index order for the heavy table and row-major for the light layer. | For a 16-bucket `3x512` sketch fed 40 flows and then expanded, verifies the decoded sketch re-serializes to the bytes it came from. |
| `elastic_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For an 8-bucket `2x256` sketch over a hasher declaring its own `HashProfile`, verifies the buckets and light counters round-trip, the bytes differ from the standard-profile sketch's over the same flows, and a standard-profile decode rejects them. |
| `elastic_rejects_too_many_light_rows` | The light Count-Min layer carries the seed list's row bound. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### Coco

Test file: [`src/sketches/coco.rs`](../src/sketches/coco.rs)

Wire tests: [`src/sketches/coco/wire.rs`](../src/sketches/coco/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `insert_then_estimate_matches_full_value_for_partial_key` | Insert then estimate matches full value for partial key. | Core behavior for insert/query/update and deterministic semantics is validated; the substring query and `estimate_key` both return `5`. |
| `estimate_with_udf_allows_custom_partial_matching` | Estimate with udf allows custom partial matching. | Core behavior for insert/query/update and deterministic semantics is validated. |
| `tied_minimum_buckets_are_chosen_uniformly_at_random` | Buckets tied at the smallest value are picked uniformly. | Inserts one key into 2,000 fresh `32x4` tables, where all four mapped buckets tie at `0`, and verifies each row takes between 200 and 800 of the landings rather than row 0 taking all of them. |
| `the_three_queries_disagree_on_a_key_that_prefixes_another` | The three query shapes are distinguished on prefixing keys. | With `k1=7` and `k10=5` inserted, verifies `estimate_substring("k1")` returns `12` while `estimate_key` and `estimate_projected` both return `7`. |
| `estimate_projected_aggregates_full_keys_sharing_a_partial_key` | Projection aggregates full keys onto one partial key. | Reproduces the paper's figure 7: two full keys on srcip `19.98.10.26` sum to `1041`, and the lone `34.52.73.17` key returns `856`. |
| `recorded_flows_yields_each_occupied_bucket_once` | The query front-end lists every recorded flow exactly once. | After 20 weighted inserts into a `32x4` table, verifies the iterator yields one entry per occupied bucket and that no key appears twice. |
| `group_by_agrees_with_per_key_projected_queries` | One-pass grouping matches the per-key scan. | Over 60 inserts across 5 families, verifies every `group_by` entry equals `estimate_projected` for the same partial key. |
| `group_by_preserves_the_inserted_mass` | Grouping conserves the inserted mass under eviction. | Drives 400 inserts of weight 3 through an `8x2` table and verifies the grouped totals still sum to `1200`. |
| `group_by_reproduces_the_papers_figure_seven` | Grouping reproduces the paper's worked example. | Groups `19.98.10.26\|80=521`, `19.98.10.26\|443=520`, and `34.52.73.17\|118=856` by srcip and verifies exactly two entries, `1041` and `856`. |
| `a_key_occupies_at_most_one_bucket_per_row` | A key never gains a second home in the table. | After 64 inserts of one key into a `32x4` table, verifies exactly one bucket holds it and `estimate_key` returns `64`. |
| `estimate_key_never_exceeds_the_inserted_mass` | Point queries stay inside the table mass. | Over 500 weighted inserts across 40 keys in an `8x2` table, verifies the table mass equals the inserted mass and no per-key estimate exceeds it. |
| `merge_combines_tables_without_losing_counts` | Merge combines tables without losing counts. | Merge behavior preserves expected aggregate semantics and internal invariants. |
| `coco_round_trip_serialization` | The envelope frames a sketch under kind_id `0x0c 0x00`, and the state survives a round trip. | For an `init_with_size(8, 4)` table holding two weighted keys, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x0c 0x00`, and that the decode keeps `w` `8`, `d` `4`, every bucket's key and value, and an `estimate_key` of `521`. |
| `coco_rejects_foreign_kind_id` | Another sketch carrying `rows`/`cols` metadata is refused on its kind_id. | Verifies a `4x8` Count-Min envelope fails to decode as a `Coco`. |
| `coco_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the eight `CocoMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `CocoMetadata`. |
| `coco_metadata_rejects_a_missing_cols_key` | `cols` is required and can never be silently defaulted. | Encodes the other seven `CocoMetadata` fields as a named map with `cols` omitted and verifies it does not decode as `CocoMetadata`. |
| `coco_rejects_zero_dimension_payload` | A zero dimension is a decode error, not a `Vector2D` panic. | Verifies a crafted `4x0` envelope with an empty payload fails rather than panicking in the `cols.ilog2()` mask derivation. |
| `coco_rejects_dimension_length_mismatch` | The length check fires from the declared dimensions, before any allocation is sized from them. | Verifies a crafted envelope declaring `MATRIX_MAX_ROWS x 2^24` while carrying three buckets fails. |
| `coco_rejects_mass_under_an_unoccupied_bucket` | An unoccupied bucket holds no mass. | Verifies a crafted `1x2` payload pairing a `nil` key with a value of `9` fails to decode. |
| `coco_rejects_serializing_a_geometry_mismatch` | The encode side refuses geometries its own decoder would reject. | Verifies a sketch whose `w` is set to `16` against an 8-wide table fails to serialize, and that an `init_with_size(8, 0)` sketch fails too. |
| `coco_empty_buckets_are_nil_and_never_collide_with_an_empty_key` | An unoccupied bucket is `nil`, not an empty string. | For an all-empty `init_with_size(4, 2)` table, verifies the payload holds eight `0xc0` and no `0xa0`, the decode matches cell for cell, and `recorded_flows` is empty; then that a table holding an inserted empty key emits an `0xa0`, decodes identically, lists one flow, and answers that key's `estimate_key` with `5`. |
| `coco_mixed_occupancy_round_trips` | A table mixing occupied and free buckets round-trips bucket for bucket. | For an `init_with_size(16, 3)` table fed ten weighted keys, verifies the fixture leaves both bucket states present and that the decode matches on every cell and on `estimate_key` for all ten keys. |
| `coco_decoded_sketch_reserializes_byte_identically` | The emitted order is the table's own index order, so a decode re-encodes exactly. | For an `init_with_size(16, 3)` table fed twelve weighted keys, verifies the decoded sketch re-serializes to the bytes it came from. |
| `coco_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For an `init_with_size(8, 4)` table over a hasher declaring its own `HashProfile`, verifies every cell round-trips, the bytes differ from the standard-profile sketch's over the same keys, and a standard-profile decode rejects them. |
| `coco_rejects_too_many_rows` | Row `i` hashes at seed index `i`, so the table carries the seed list's row bound. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### KMV

Test file: [`src/sketches/kmv.rs`](../src/sketches/kmv.rs)

Wire tests: [`src/sketches/kmv/wire.rs`](../src/sketches/kmv/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `assert_serialization_round_trip` | Assert serialization round trip. | Serialization/deserialization preserves component state and behavior after round trip. |
| `kmv_round_trip_serialization` | The envelope frames a sketch under kind_id `0x0e 0x00`, and the bytes are stable across a round trip. | For `k = 64` filled by 5,000 distinct keys, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x0e 0x00`, that the metadata carries `k` `64` and the canonical seed index, that the 64 emitted hashes are strictly ascending, and that the decode matches on `k`, capacity, retained hashes, heap root, and `estimate`, re-serializing identically. |
| `kmv_emitted_order_is_independent_of_insertion_order` | The emitted order follows the hash value, not the arrival order. | Fills two `k = 32` sketches with the same 2,000 keys forward and backward, verifies their heap arrays differ, and that both serialize to the same bytes, which a decode re-encodes exactly. |
| `kmv_empty_round_trip` | A sketch that retains nothing has exactly one encoding. | Verifies a fresh `k = 16` sketch emits `k` `16` beside an empty `hashes` array and decodes back to length `0` at capacity `16` with an `estimate` of `0.0` and identical re-serialized bytes. |
| `kmv_carries_hashes_at_full_u64_width` | The retained digests travel at 64 bits. | Inserts `0`, `1`, `u64::MAX / 2`, and `u64::MAX` by hash into a `k = 4` sketch and verifies the payload and the decoded sketch both carry those four values, re-serializing identically. |
| `kmv_rejects_foreign_kind_id` | Another sketch's kind_id is refused. | Verifies a `3x8` Count-Min envelope fails to decode as a `KMV` with a "kind_id mismatch" complaint. |
| `kmv_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the eight `KmvMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `KmvMetadata`. |
| `kmv_metadata_rejects_a_missing_k_key` | `k` is required and can never be silently defaulted. | Encodes the other seven `KmvMetadata` fields as a named map with `k` omitted and verifies it does not decode as `KmvMetadata`. |
| `kmv_rejects_a_crafted_envelope` | Every structural rule is pinned by the complaint it fails with. | Verifies four crafted envelopes are each refused for their own reason: a `k` of `0`, three hashes over a `k` of `2`, an unsorted hash array, and one holding a duplicate. |
| `kmv_rejects_a_payload_declaring_more_hashes_than_it_carries` | An over-declared array header is refused on the read. | Verifies a payload whose `array32` header declares `2^30` elements while carrying two of them fails rather than being allocated. |
| `kmv_does_not_allocate_a_declared_k` | A declared `k` is metadata, never an allocation size. | Verifies an envelope declaring `k` `u32::MAX` with two hashes decodes, reports that bound and an `estimate` of `2.0`, and that a further hash is appended rather than evicting one. |
| `kmv_refuses_a_k_the_metadata_cannot_carry` | A `k` past the metadata's `u32` field fails the encode rather than truncating. | Verifies a sketch built at `k = 1 << 40` fails to serialize with a message naming the "exceeds the u32 metadata field" rule. |
| `kmv_refuses_to_serialize_a_state_decode_would_reject` | The encode side refuses the states decode refuses. | Verifies a `k = 0` sketch fails to serialize, and that one retaining two hashes at `k = 1` fails with a "2 hashes over a k of 1" complaint. |
| `kmv_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a `k = 16` sketch over 500 keys through a hasher declaring its own `HashProfile`, verifies the retained hashes round-trip and re-serialize identically, that the two profiles retain the same hashes yet emit different bytes, and that a standard-profile decode rejects them. |

### UniformSampling

Test file: [`src/sketches/uniform.rs`](../src/sketches/uniform.rs)

Wire tests: [`src/sketches/uniform/wire.rs`](../src/sketches/uniform/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `sample_count_tracks_rate` | Sample count tracks rate. | Core behavior for insert/query/update and deterministic semantics is validated. |
| `samples_are_drawn_from_input_stream` | Samples are drawn from input stream. | Core behavior for insert/query/update and deterministic semantics is validated. |
| `merge_combines_samples_using_rate_based_target` | Merge combines samples using rate based target. | Merge behavior preserves expected aggregate semantics and internal invariants. |
| `merge_rejects_different_rates` | Merge rejects different rates. | Merge behavior preserves expected aggregate semantics and internal invariants. |
| `sample_access_is_stable` | Sample access is stable. | Core behavior for insert/query/update and deterministic semantics is validated. |
| `uniform_sampling_round_trip_serialization` | The envelope frames a sampler under kind_id `0x0d 0x00`, and the state survives a round trip. | For a sampler at rate `0.25` seeded at `0xBEEF_FACE` over 40 updates, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x0d 0x00`, and that the decode matches on `sample_rate`, `total_seen`, `len`, and the retained samples. |
| `uniform_sampling_re_encodes_byte_identically` | The priorities are payload state, not derived. | For a sampler at rate `0.5` seeded at `0xFACE_FACE` over 33 updates, verifies the decoded sampler re-serializes to the bytes it came from. |
| `uniform_sampling_rng_state_resumes_the_same_sequence` | The payload carries the RNG position, so a decode resumes the same draws. | Feeds the same 40 further updates to a rate-`0.3` sampler and to its decoded copy, and verifies they agree on the samples, on `total_seen`, and on their re-serialized bytes. |
| `uniform_sampling_empty_round_trip_has_exactly_one_encoding` | An empty sampler has one encoding for a given rate and RNG position. | Verifies a rate-`0.1` sampler seeded at `0xABC1` decodes back empty with `total_seen` `0` and its rate intact, and that its bytes equal both a fresh twin's and the decoded sampler's re-serialization. |
| `uniform_sampling_rejects_foreign_kind_id` | Another sketch's kind_id is refused. | Verifies a `3x8` Count-Min envelope fails to decode as a `UniformSampling`. |
| `us_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the three `UsMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `UsMetadata`. |
| `us_metadata_rejects_a_missing_item_type_key` | `item_type` is required and can never be silently defaulted. | Encodes a named map holding only `metadata_version` and `sample_rate` and verifies it does not decode as `UsMetadata`. |
| `us_metadata_rejects_a_foreign_item_type_name` | Samples are `f64`, and no other `item_type` name decodes. | Wraps an `i64`-labelled metadata around a one-sample payload and verifies it does not decode. |
| `uniform_sampling_rejects_an_out_of_range_sample_rate` | A rate outside `(0, 1]` is refused before it reaches `target_size`. | Verifies crafted envelopes at rates `0.0`, `-0.5`, `1.5`, `NaN`, and infinity each fail rather than panic. |
| `uniform_sampling_huge_declared_stream_costs_two_samples` | The declared stream length never sizes an allocation. | Verifies a payload declaring `total_seen` `u64::MAX` beside two samples decodes to a sampler holding exactly those two. |
| `uniform_sampling_rejects_more_samples_than_the_rate_allows` | Retaining more than the rate allows is not a state the algorithm reaches. | Verifies a payload of three samples at `total_seen` `2` and rate `0.5` fails to decode. |
| `uniform_sampling_rejects_parallel_array_length_mismatch` | The two payload arrays are parallel. | Verifies a payload of three priorities against two values fails to decode. |
| `uniform_sampling_rejects_unordered_priorities` | Entries are held in ascending priority, with ties broken by the value. | Verifies a payload of priorities `[9, 2, 5]` fails to decode, and that one with equal priorities whose values run `[8.0, 1.0]` fails too. |
| `uniform_sampling_rejects_crafted_bytes_without_panicking` | Truncated, foreign and garbage bytes are errors, never panics. | Verifies six truncations of a valid envelope, 64 bytes of `0xff`, and a valid envelope carrying a garbage payload all fail to decode. |
| `uniform_sampling_rejects_serializing_an_over_full_sampler` | The encode side refuses a sampler its own decoder would reject. | Verifies a sampler holding two samples whose `total_seen` is set to `1` fails to serialize. |

## Sketch Frameworks

### Hydra

Test file: [`src/sketch_framework/hydra.rs`](../src/sketch_framework/hydra.rs)

Wire tests: [`src/sketch_framework/hydra/wire.rs`](../src/sketch_framework/hydra/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `hydra_updates_countmin_frequency` | Hydra updates countmin frequency. | Updates `"user;session"` with value `"event"` 5 times and verifies combined query `>= 5` while an unrelated key query is exactly `0.0`. |
| `hydra_updates_countmin_frequency_multiple_values` | Hydra updates countmin frequency multiple values. | Inserts values `I64(0..4)` with multiplicity `i` under one key, verifies per-value fan-out query `>= i`, and checks unrelated-key query returns `0.0`. |
| `hydra_round_trip_serialization` | Hydra round trip serialization. | After mixed inserts, verifies MessagePack round trip keeps non-empty payload, preserves dimensions/template type, and keeps queried frequencies exactly unchanged. |
| `hydra_subpopulation_frequency_test` | Hydra subpopulation frequency test. | On a fixed labeled dataset, asserts exact subpopulation frequencies for single-label, multi-label, full-key, and disjoint cross-population queries (including zero-result case). |
| `hydra_subpopulation_cardinality_test` | Hydra subpopulation cardinality test. | Using HLL-backed counters, checks single/multi/full-key cardinalities are approximately `3.0` (within `EPSILON`) and disjoint/unknown keys return `0.0`. |
| `hydra_tracks_kll_quantiles` | Hydra tracks KLL quantiles. | For inserted samples `[10,20,30,40,50]`, verifies CDF query at `30.0` is `0.6` (within `1e-9`) and empty-bucket query returns `0.0`. |
| `hydra_kll_single_label_cdfs` | Hydra KLL single label cdfs. | For each label group, verifies exact expected CDF levels `{1/3, 2/3, 1}` at chosen thresholds using `EPSILON` tolerance. |
| `hydra_kll_multi_label_cdfs` | Hydra KLL multi label cdfs. | Verifies exact CDF values for multi-label combinations and confirms a non-overlapping key pair returns CDF `0.0`. |
| `hydra_kll_extreme_queries` | Hydra KLL extreme queries. | Confirms CDF boundary behavior (`0` below range, `1` above range) for known keys and `0` for unknown keys. |
| `test_count_min_frequency_query` | Test count min frequency query. | Inserts one key three times into `HydraCounter::CM`, then verifies `Frequency` query succeeds and returns exactly `3.0`. |
| `test_count_min_invalid_query_types` | Test count min invalid query types. | Verifies unsupported CM queries return errors, including exact message for `Quantile` (`"Count-Min Sketch Counter does not support Quantile Query"`). |
| `test_hll_cardinality_query` | Test HLL cardinality query. | Inserts `100` unique items plus one duplicate and verifies `Cardinality` query succeeds with estimate constrained to `(90.0, 110.0)`. |
| `test_kll_quantile_query` | Test KLL quantile query. | Inserts values `1..=100` and verifies median query succeeds with estimate within `+/-5` of `50.0`. |
| `test_univmon_universal_queries` | Test univmon universal queries. | Inserts `A` 10 times and `B` 20 times, then checks `L1=30.0`, cardinality is approximately `2.0` (`abs err < 0.5`), and entropy is positive. |
| `test_merge_counters` | Test merge counters. | Merges two CM counters and verifies frequency sum (`2.0`) for shared key, then confirms merging with mismatched counter type (`HLL`) returns error. |
| `test_count_frequency_query` | Test count frequency query. | Inserts one `Count` key four times and verifies `Frequency` query succeeds with exact result `4.0`. |
| `test_count_invalid_query_types` | Test count invalid query types. | Verifies unsupported `Count` queries fail, including exact `Quantile` error message and error on `Cardinality`. |
| `hydra_count_min_round_trip_serialization` | A Count-Min grid round-trips under kind_id `0x07 0x01`. | For a `3x8` grid of `2x16` fast-path Count-Min counters over a two-column schema fed 30 records, verifies the envelope names that kind_id, that the grid and counter dimensions and the schema survive, that three frequency probes answer identically, and that the decode re-serializes to the same bytes. |
| `hydra_count_sketch_round_trip_serialization` | A Count Sketch grid round-trips under kind_id `0x07 0x02`. | Runs the same round trip over `2x16` fast-path Count Sketch counters and the same three frequency probes. |
| `hydra_hyperloglog_round_trip_serialization` | An HLL grid round-trips under kind_id `0x07 0x03`. | Runs the same round trip over `HyperLogLog<ErtlMLE>` counters fed 300 records, probing two cardinality queries. |
| `hydra_kll_round_trip_serialization` | A KLL grid round-trips under kind_id `0x07 0x00`. | Runs the same round trip over default `KLL` counters fed 300 records, probing two quantiles and a CDF. |
| `hydra_univmon_round_trip_serialization` | A UnivMon grid round-trips under kind_id `0x07 0x04`. | Runs the same round trip over `UnivMon::init_univmon(4, 2, 16, 3)` counters fed 120 records, probing L1, L2, entropy, and cardinality. |
| `hydra_count_sketch_negative_cells_round_trip` | Signed cells reach the wire and come back unchanged. | Verifies the Count Sketch grid holds a negative counter and that every cell of every counter matches after a decode. |
| `hydra_schema_round_trips_exactly` | The key columns round-trip exactly, escaping included. | For a grid whose two labels carry a semicolon, a colon, and a backslash and whose subkeys carry a semicolon, verifies the round trip preserves both frequency probes' answers and that the decoded schema equals the labels. |
| `hydra_variants_reject_each_others_envelopes` | Each variant's decoder owns exactly one kind_id. | Runs all five per-variant decoders against all five variants' envelopes and verifies each succeeds only on its own, and that a plain `2x16` Count-Min envelope is refused by every decoder and by `deserialize_from_bytes`. |
| `hydra_rejects_a_mixed_variant_grid` | A grid mixing counter variants has no encoding. | Replaces one cell of a Count-Min grid with an HLL counter and verifies serialization fails with a complaint naming `cell (1, 1)` and both counter types. |
| `hydra_rejects_serializing_an_inconsistent_grid` | The encode side refuses states its own decoder would reject. | Verifies four Count-Min grids each fail to serialize: one whose `row_num` is `4` against its storage, one whose grid is an unfilled `Vector2D::init(3, 8)`, one holding a `2x8` cell against a `2x16` prototype, and one whose `type_to_clone` holds data. |
| `hydra_univmon_rejects_cells_mixing_key_variants` | A UnivMon grid whose cells hold different `HeapItem` variants has no single `counter_key_type`. | Seats a string-keyed UnivMon in a `u64`-keyed grid and verifies serialization fails with a "mix key variants" complaint. |
| `hydra_rejects_crafted_geometry` | Every declared count is measured against the payload before anything is sized from it. | Verifies nine crafted Count-Min metadata shapes each fail with their own complaint - a `MATRIX_MAX_ROWS x 2^20` grid, an overflowing product, a grid and a counter each one row past `MATRIX_MAX_ROWS`, a zero grid row or column, a zero counter row or column, and a `4x16` counter against the payload - and that an empty schema fails too. |
| `hydra_rejects_crafted_geometry_for_the_variable_counters` | The variable-length counters are cut by the same rule. | Verifies a `MATRIX_MAX_ROWS x 4096` KLL grid declaration fails, and that a `MATRIX_MAX_ROWS x 4096` UnivMon declaration and shapes carrying a zero `sketch_col`, `layer_size`, or `heap_size` each fail too. |
| `hydra_metadata_rejects_unknown_and_missing_keys` | An unexpected metadata key and a missing required one both fail closed. | Encodes the fourteen `HydraMatrixMetadata` fields plus a `bogus_field`, and the same fields with `schema` omitted, and verifies neither decodes as `HydraMatrixMetadata`. |
| `hydra_cell_envelopes_mirror_the_counters_own_bytes` | A cell's inlined state is exactly that counter's own payload. | Verifies the HLL, KLL, and UnivMon cell envelopes built from a cell's state equal the bytes those counters serialize to on their own. |
| `hydra_pins_its_hash_profile` | Hydra hashes its subkeys through the crate default, so it has one truthful profile. | Verifies a Count-Min grid emits `DefaultXxHasher`'s profile id and seed list, and that the same payload re-framed under a custom profile is different bytes that fail to decode. |
| `hydra_rejects_too_many_rows` | The grid and the matrix counters each carry the seed list's row bound. | Verifies a grid past `MATRIX_MAX_ROWS` and a counter past it both fail to serialize with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### HashSketchEnsemble

Test file: [`src/sketch_framework/hashlayer.rs`](../src/sketch_framework/hashlayer.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `test_insert_and_estimate` | Insert and frequency estimate accuracy on Zipf stream. | On Zipf stream (`N=10_000`, `domain=1000`, `exp=1.5`), builds default 2-sketch ensemble (CMS + Count, `3x4096`) and verifies average relative error for both CMS (index `0`) and Count (index `1`) frequency estimates over sampled keys is below `0.1`. |
| `test_insert_at` | Targeted insert updates only specified indices. | Inserts only at index `[0]` via `insert_at`, then verifies CMS at index `0` has a positive estimate while Count at index `1` returns `0.0`. |
| `test_insert_with_hash_matches_insert` | Pre-computed hash insert matches regular insert. | Builds two identical ensembles; one uses `insert()`, the other uses `hash_input()` + `insert_with_hash()`. Verifies CMS estimates at index `0` are identical for a probe key. |
| `test_hll_cardinality` | HLL-only ensemble cardinality accuracy. | Builds HLL-only ensemble (`HyperLogLog<ErtlMLE>`), inserts Zipf stream, and verifies cardinality relative error vs true distinct count is `< 0.02`. |
| `test_estimate_on_hll_returns_error` | Frequency query on HLL sketch returns error. | Calls `estimate()` on an HLL-only ensemble and verifies it returns `Err`. |
| `test_cardinality_on_cms_returns_error` | Cardinality query on CMS sketch returns error. | Calls `cardinality()` on a CMS+Count ensemble and verifies it returns `Err`. |
| `test_direct_access` | Index-based get/get_mut and sketch type reporting. | Verifies `get(0)` and `get(1)` return `Some`, `get(2)` returns `None`, and `get_mut(0)` reports `sketch_type() == "CountMin"`. |
| `test_bounds_checking` | Out-of-bounds queries return errors. | Confirms `estimate(999, ...)`, `cardinality(999)`, and `estimate_with_hash(999, ...)` all return `Err`. |
| `test_custom_dimensions` | Custom-dimension ensemble insert and estimate. | Builds 2-sketch ensemble (CMS + Count, `5x2048`), verifies `len=2`/non-empty, inserts Zipf stream, and confirms both indices return positive estimates. |
| `test_mixed_matrix_and_hll` | Mixed CMS + HLL ensemble queries. | Builds ensemble with one CMS and one `HyperLogLog<ErtlMLE>`, inserts Zipf stream, verifies CMS estimate at index `0` is positive and HLL cardinality error at index `1` is `< 0.05`. |
| `test_push_compatible` | Push compatible sketch succeeds. | Creates single-CMS ensemble (`3x4096`), pushes a Count sketch with matching dimensions, verifies `push` returns `Ok` and `len=2`. |
| `test_push_incompatible_rejected` | Push incompatible sketch is rejected. | Creates single-CMS ensemble (`3x4096`), pushes a Count sketch with different dimensions (`5x2048`), verifies `push` returns `Err`. |

### UnivMon

Test file: [`src/sketch_framework/univmon.rs`](../src/sketch_framework/univmon.rs)

Wire tests: [`src/sketch_framework/univmon/wire.rs`](../src/sketch_framework/univmon/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `univmon_round_trip_serialization` | Univmon round trip serialization. | After weighted inserts, verifies non-empty serialization and round-trip preservation of configuration fields, `bucket_size`, `L1/L2/entropy` (`<1e-6` drift), and cardinality (`< EPSILON` drift). |
| `update_populates_bucket_size_and_heavy_hitters` | Update populates bucket size and heavy hitters. | Inserting one hot key `40` times sets `bucket_size=40`, tracks key in heavy-hitter heap with count `>=20`, and yields exact `L1=40` and `cardinality=1`. |
| `merge_with_combines_heavy_hitters` | Merge with combines heavy hitters. | Merging sketches with disjoint heavy keys verifies merged left heap contains both contributions (`left=25`, `right=30`) while right heap retains `right=30`. |
| `univmon_layers_use_different_seeds` | Univmon layers use different seeds. | Verifies hash outputs for the same key with seed indices `0..3` are all pairwise different. |
| `univmon_cardinality_is_positive` | Univmon cardinality is positive. | After inserting `20` distinct flow keys, cardinality estimate is exactly `20.0`. |
| `univmon_bucket_size_tracked_correctly` | Univmon bucket size tracked correctly. | Inserts counts `100`, `200`, `150` for three flows and verifies `bucket_size` equals total `450`. |
| `univmon_basic_operation` | Univmon basic operation. | On fixed mixed workload, verifies exact aggregate metrics `cardinality=10.0` and `L1=131.0`. |
| `test_statistical_accuracy` | Test statistical accuracy. | On heavy/medium/noise synthetic distribution, verifies relative error for both `L2` and `entropy` is below `0.15`. |
| `univmon_random_data_matches_ground_truth_within_configured_tolerance` | Random weighted updates keep every metric inside its own tolerance. | Over `10_000` seeded random weighted updates across 5,000 keys into `init_univmon(256, 6, 8192, 16)`, requires relative error against the exact truth map at or under `0.07` for cardinality and at or under `0.05` for `L1`, `L2`, and entropy. |
| `univmon_layers_with_different_heap_loads_round_trip` | Each layer's heap contents and emitted order survive the round trip. | For an `init_univmon(8, 2, 16, 4)` pyramid fed 40 weighted keys, verifies the layers carry different heap loads, that every layer's length, capacity of `8`, and per-key counts match after a decode, and that the bytes re-serialize identically. |
| `univmon_carries_update_mode_and_candidate_flags` | `update_mode` and `candidate_complete` are state, not derived. | For a terminal-only `init_univmon(2, 2, 8, 3)` pyramid fed 20 keys through `fast_insert`, verifies the payload carries `update_mode` `2`, that the decode preserves the candidate flags, cardinality, and entropy and re-serializes identically, and that forcing every flag true changes what the pyramid reports. |
| `univmon_empty_has_one_encoding` | An empty pyramid has exactly one encoding. | Verifies a fresh `init_univmon(4, 2, 16, 3)` pyramid and one freed after an insert serialize to identical bytes carrying the pinned `EMPTY_KEY_TYPE`, and that the decode holds empty heaps and re-serializes identically. |
| `univmon_pins_its_hash_profile` | UnivMon hashes through the crate default, so it has one truthful profile. | Verifies a populated pyramid emits `DefaultXxHasher`'s profile id and seed list, and that the same payload re-framed under a custom profile is different bytes that fail to decode. |
| `univmon_rejects_foreign_kind_ids` | The neighbouring universal-sketch kind_ids are refused. | Verifies `Count`, `CountL2HH`, `UnivMonPyramid`, and `UnivMonQ` envelopes each fail to decode as a `UnivMon`. |
| `univmon_rejects_crafted_shapes` | Every shape rule fires before an allocation is sized from it. | Verifies six crafted metadata shapes fail - a `layer_size` of `u32::MAX` or `0`, a zero `sketch_col` or `heap_size`, a `MATRIX_MAX_ROWS x 4096` sketch, and a `heap_size` of `1` - and that a short `heap_counts`, a short `candidate_complete`, and an `update_mode` of `7` each fail too. |
| `univmon_rejects_serializing_an_inconsistent_pyramid` | The encode side refuses states its own decoder would reject. | Verifies a layer resized to `2x32` and a layer hashing at another layer's seed index each fail to serialize while the matching `2x16` layer at its own index serializes, and that a pyramid whose heap mixes key variants fails with a "keys mix variants" complaint while a 128-bit key fails outright. |
| `univmon_metadata_rejects_unknown_and_missing_keys` | An unexpected metadata key and a missing required one both fail closed. | Encodes the eleven `UnivMonMetadata` fields plus a `bogus_field`, and the same fields with `key_type` omitted, and verifies neither decodes as `UnivMonMetadata`. |
| `univmon_rejects_too_many_layer_rows` | Every layer is a CountL2HH matrix, so `sketch_row` carries the seed list's row bound. | Verifies a sketch past `MATRIX_MAX_ROWS` fails to serialize, that a crafted envelope of that geometry fails to decode with a message naming `MATRIX_MAX_ROWS`, and that the boundary row count still serializes. |

### UnivMon Optimized

Test file: [`src/sketch_framework/univmon_optimized.rs`](../src/sketch_framework/univmon_optimized.rs)

Wire tests: [`src/sketch_framework/univmon_optimized/wire.rs`](../src/sketch_framework/univmon_optimized/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `pool_basic_take_put` | Pool basic take put. | Validates pool accounting: initial preallocation (`available=2`, `allocated=2`), on-demand allocation when empty (`allocated=3`), and reuse on put/take without further allocation. |
| `pool_free_resets_sketch` | Pool free resets sketch. | Confirms returning a used sketch to pool resets state so retaken sketch has `bucket_size=0` and near-zero `L2` in layer `0`. |
| `pyramid_basic_insert_and_query` | Pyramid basic insert and query. | For simple inserts, verifies exact aggregate state with `bucket_size=65`, `L1` approximately `65` (`<1e-6`), and `cardinality=3`. |
| `pyramid_fast_insert_matches_standard` | Pyramid fast insert matches standard. | On identical 500-item stream, verifies standard vs fast paths keep identical `bucket_size`, with `L1` deviation `<10%` and cardinality deviation `<15%`. |
| `pyramid_two_tier_dimensions` | Pyramid two tier dimensions. | Verifies two-tier layout metadata for configured pyramid (`layer_size=8`, `elephant_layers=4`). |
| `pyramid_free_resets_state` | Pyramid free resets state. | After bulk inserts, `free()` resets sketch to empty baseline (`bucket_size=0`, layer-0 `L2` approximately `0`). |
| `pyramid_merge_combines_data` | Pyramid merge combines data. | Merging disjoint halves verifies merged `L1` stays within `10%` of the sum of pre-merge `L1` values. |
| `pyramid_accuracy_zipf` | Pyramid accuracy Zipf. | On heavy/medium/light Zipf-like workload, requires relative error `<15%` for `L1`, `L2`, cardinality, and entropy. |
| `pyramid_fast_insert_accuracy` | Pyramid fast insert accuracy. | Using `fast_insert` only, requires relative error `<15%` for `L1`, `L2`, cardinality, and entropy versus exact frequency map. |
| `pyramid_memory_savings_vs_uniform` | Pyramid memory savings vs uniform. | Verifies pyramid column budget is smaller than uniform baseline and computed memory savings exceed `30%`. |
| `pyramid_round_trip_serialization` | The envelope frames a pyramid under kind_id `0x11 0x00`, and the bytes are stable across a round trip. | For `UnivMonPyramid::new(4, 2, 3, 16, 2, 8, 4)` fed four weighted string keys, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x11 0x00`, that the metadata carries that layout and `key_type` `string`, and that the decode matches on `bucket_size`, L1, L2, cardinality, entropy, and the candidate flags, re-serializing identically. |
| `pyramid_two_tier_geometry_survives` | The two tiers are derived from the layer's position. | Verifies every decoded layer keeps its tier's dimensions - `3x16` for the two elephant layers and `2x8` for the mouse layers - its own seed index, and its counter array. |
| `pyramid_layers_with_different_heap_loads_round_trip` | Each layer's heap contents survive the round trip. | For `UnivMonPyramid::new(8, 2, 3, 32, 2, 16, 4)` fed 40 weighted keys, verifies the layers carry different heap loads, that every layer's length and per-key counts match after a decode, and that the bytes re-serialize identically. |
| `pyramid_without_mouse_layers_round_trips` | The one-tier case still round-trips. | For a pyramid whose four layers are all elephants, verifies the decoded layers all read `2x16` and the bytes re-serialize identically. |
| `pyramid_empty_has_one_encoding` | An empty pyramid has exactly one encoding. | Verifies a fresh pyramid and one freed after an insert serialize to identical bytes carrying a `key_type` of `u64`, and that the decode re-serializes identically. |
| `pyramid_carries_update_mode_and_candidate_flags` | `update_mode` and `candidate_complete` are state, not derived. | For a terminal-only `UnivMonPyramid::new(2, 1, 2, 8, 2, 8, 3)` fed 20 keys through `fast_insert`, verifies the payload carries `update_mode` `2` and that the decode preserves the candidate flags and cardinality and re-serializes identically. |
| `pyramid_pins_its_hash_profile` | The pyramid hashes through the crate default, so it has one truthful profile. | Verifies a populated pyramid emits `DefaultXxHasher`'s profile id and seed list, and that the same payload re-framed under a custom profile is different bytes that fail to decode. |
| `pyramid_rejects_foreign_kind_ids` | The neighbouring universal-sketch kind_ids are refused. | Verifies `Count`, `CountL2HH`, `UnivMon`, and `UnivMonQ` envelopes each fail to decode as a `UnivMonPyramid`. |
| `pyramid_rejects_crafted_shapes` | Every layout rule fires before an allocation is sized from it. | Verifies six crafted metadata layouts fail - a `layer_size` of `u32::MAX` or `0`, a `heap_size` of `0` or `1`, a zero `elephant_col`, and a `MATRIX_MAX_ROWS x 4096` mouse tier - and that a short `heap_lens` and an `update_mode` of `9` each fail too. |
| `pyramid_rejects_serializing_an_inconsistent_layout` | The encode side refuses states its own decoder would reject. | Verifies a mouse layer holding the elephant tier's `3x16` dimensions fails to serialize, and that a layer hashing at another layer's seed index fails too. |
| `pyramid_metadata_rejects_unknown_and_missing_keys` | An unexpected metadata key and a missing required one both fail closed. | Encodes the fourteen `PyramidMetadata` fields plus a `bogus_field`, and the same fields with `mouse_col` omitted, and verifies neither decodes as `PyramidMetadata`. |

### UnivMon-Q

Test file: [`src/sketch_framework/univmon_q.rs`](../src/sketch_framework/univmon_q.rs)

Wire tests: [`src/sketch_framework/univmon_q/wire.rs`](../src/sketch_framework/univmon_q/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `univmon_q_round_trip_serialization` | The envelope frames a sketch under kind_id `0x1a 0x00`, and the bytes are stable across a round trip. | For a 4-level, 64-wide, depth-3 sketch seeded at `5` with source id `7` over 200 updates, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x1a 0x00`, that the metadata carries that shape, `counter_type` `i64`, and `seed_index` `5`, and that the decode matches on config, source id, `count`, `min`, `max`, `cdf`, and `estimate_f2`, re-serializing identically. |
| `univmon_q_ordering_state_continues_after_a_round_trip` | The ordering state survives, so a decoded sketch draws the same occurrence priorities. | Feeds the same 400 further updates to a sketch built over 300 updates and to its decoded copy, and verifies they agree on `next_sequence`, on the sorted ordered heap, on `cdf`, and on their serialized bytes. |
| `univmon_q_empty_has_one_encoding` | An empty sketch has exactly one encoding, and `min` and `max` travel as msgpack nil. | Verifies two fresh sketches at the same config and source id serialize identically with a payload `count` of `0` and no extrema, that a cleared sketch is deliberately different bytes since it keeps its occurrence sequence, and that the decode is empty and re-serializes identically. |
| `univmon_q_counter_type_is_pinned` | The counter width is pinned by `counter_type`. | Verifies a 32-bit-counter sketch emits `counter_type` `i32` and decodes with its `estimate_f2` intact, that its bytes differ from the 64-bit sketch's over the same 50 updates, and that relabelling the metadata `f64` makes the decode fail. |
| `univmon_q_custom_hasher_profile_round_trips_and_is_self_describing` | The metadata describes the hasher that built the sketch rather than a hardcoded profile. | For a sketch over a hasher declaring its own `HashProfile`, verifies `estimate_f2` round-trips, the bytes differ from the standard-profile sketch's over the same 100 updates, and a standard-profile decode rejects them. |
| `univmon_q_rejects_foreign_kind_ids` | The neighbouring universal-sketch kind_ids are refused. | Verifies `Count`, `CountL2HH`, `UnivMon`, and `UnivMonPyramid` envelopes each fail to decode as a `UnivMonQ`. |
| `univmon_q_rejects_crafted_shapes` | Every shape rule fires before an allocation is sized from it. | Verifies seven crafted configs fail - `levels` of `63` or `1`, a `width` of `u32::MAX`, a `depth` of `4`, zero `candidates` or `ordered_samples`, and another `hash_seed` - and that a short `candidate_scores`, a short `ever_evicted`, a missing `min`, swapped extrema, a short `occurrence_keys`, and reversed `candidate_keys` each fail too. |
| `univmon_q_rejects_serializing_an_inconsistent_state` | The encode side refuses states its own decoder would reject. | Verifies a level whose `PackedCountSketch` is not the config's size, a `count`/`min`/`max` triple the algorithm cannot reach, and a candidate table over its declared capacity each fail to serialize. |
| `univmon_q_metadata_rejects_unknown_and_missing_keys` | An unexpected metadata key and a missing required one both fail closed. | Encodes the fourteen `UnivMonQMetadata` fields plus a `bogus_field`, and the same fields with `counter_type` omitted, and verifies neither decodes as `UnivMonQMetadata`. |

### NitroBatch

Test file: [`src/sketch_framework/nitro.rs`](../src/sketch_framework/nitro.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `nitro_batch_countmin_error_bound_zipf` | Nitro batch countmin error bound Zipf. | On Zipf stream (`rows=3`, `cols=4096`, `N=200_000`), verifies CountMin estimates satisfy in-bound key count `> (1-delta)*distinct` using `epsilon=e/cols`, `delta=e^-rows`, and bound `epsilon*N`. |
| `nitro_batch_count_error_bound_zipf` | Nitro batch Count Sketch L2 error bound on a Zipf stream. | On the same stream, checks `Count` median estimates against **Count Sketch's own** bound `sqrt(kappa/cols) * \|\|f_-i\|\|_2` (kappa = 3, residual L2 recomputed per key from the exact frequency vector), requiring the in-bound share to exceed `1 - P[Bin(rows, 1/3) >= ceil(rows/2)]`. Count-Min's `epsilon*N` does not apply to this sketch and is far looser on a skewed stream. Sampling RNG is seeded. |

### ExponentialHistogram

Test file: [`src/sketch_framework/eh.rs`](../src/sketch_framework/eh.rs)

Wire tests: [`src/sketch_framework/eh/wire.rs`](../src/sketch_framework/eh/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `constructor_infers_merge_norm` | Constructor infers merge norm. | Verifies constructor infers `SketchNorm::L1` for CM payload and `SketchNorm::L2` for `COUNTL2HH` payload. |
| `l1_merge_invariant_same_size` | L1 merge invariant same size. | Under repeated updates with `k=2`, verifies L1 merge policy compacts buckets so `bucket_count < 10`. |
| `l2_merge_invariant_sum_l22` | L2 merge invariant sum l22. | With `k=1` and weighted updates, verifies L2 merge rule keeps bucket count bounded (`bucket_count <= 2`). |
| `merge_recomputes_l2_mass` | Merge recomputes L2 mass. | After L2 merges, verifies bounded bucket count (`<=2`) and non-negative recomputed `l2_mass` for every payload bucket. |
| `test_basic_insertion_and_query` | Test basic insertion and query. | After one update at `t=100`, verifies single bucket presence, exact min/max timestamps (`100`), and successful interval merge query for `[100,100]`. |
| `eh_round_trip_serialization` | The envelope frames a histogram under kind_id `0x13 0x00`, and the state survives a round trip. | For `k = 2` over a 1,000-tick window of Count-Min buckets fed six timestamped updates, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x13 0x00`, and that the decode matches on `window`, `k`, `merge_norm`, every bucket's size, time range, and bit-exact `l2_mass`, and the bucket count. |
| `eh_every_variant_round_trips_as_a_bucket` | Every variant this build carries round-trips as a bucket and as the prototype. | For each populated variant, builds a `k = 3` histogram over four timestamped updates and verifies the decoded prototype keeps its `sketch_type`, the bucket ranges match, and the bytes are stable across a re-serialization. |
| `eh_empty_has_one_encoding_and_round_trips` | A histogram with no buckets has exactly one encoding. | Verifies two `k = 2` histograms over the same Count-Min prototype serialize identically and that the decode holds a bucket count of `0` and re-serializes to the same bytes. |
| `eh_carries_a_non_empty_prototype` | A prototype carrying state keeps it, so later buckets start from it. | For a prototype fed seven inserts of one key, verifies the decoded prototype answers that key as the source does and at or above `7.0`. |
| `eh_decoded_re_serializes_byte_identically_and_queries_agree` | A decoded histogram re-encodes exactly and answers the same interval query. | Verifies the populated histogram's decode re-serializes to the bytes it came from and that `query_interval_merge(0, 50)` answers the same key identically on both. |
| `eh_rejects_foreign_kind_ids` | An `EHSketchList` envelope and a Count-Min envelope are not histogram envelopes. | Verifies both fail to decode as an `ExponentialHistogram`. |
| `eh_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes the three `EhMetadata` fields plus a `bogus_field` as a named map and verifies it does not decode as `EhMetadata`. |
| `eh_metadata_rejects_a_missing_key` | `k` is required and can never be silently defaulted. | Encodes a named map holding only `metadata_version` and `window` and verifies it does not decode as `EhMetadata`. |
| `eh_rejects_a_zero_k` | `k` is at least `1` on both sides. | Verifies a histogram whose `k` is set to `0` fails to serialize, and that a valid payload re-framed under metadata declaring `k` `0` fails to decode. |
| `eh_rejects_parallel_arrays_of_unequal_length` | A declared array far longer than the buckets carried is refused before anything is sized from it. | Verifies a payload of one bucket against a `sizes` array of a million entries fails to decode. |
| `eh_rejects_impossible_bucket_state` | A zero size and an inverted time range are states the algorithm never reaches. | Verifies a crafted bucket of size `0` and one spanning `[9, 4]` each fail to decode, and that a histogram whose first bucket's size is set to `0` fails to serialize. |
| `eh_rejects_derived_fields_that_disagree` | A cached field that disagrees with the state it derives from has no encoding. | Verifies a histogram whose first bucket's `l2_mass` is set to `42.0` fails to serialize, and that one whose `merge_norm` is switched to `L2` over Count-Min buckets fails too. |
| `eh_rejects_an_experimental_kind_id_in_a_bucket` | An experimental variant's kind_id in a bucket is refused without the feature. | Verifies crafted buckets relabelled with the `Coco`, `Elastic`, and `UniformSampling` kind_ids each fail to decode, with a message naming the variant and "experimental" in builds without the feature. |
| `eh_rejects_a_custom_hash_profile_bucket` | A bucket naming a custom hash profile is refused, since the variant's decoder pins the profile of the type it rebuilds. | Verifies a bucket whose descriptor comes from a custom-profile Count-Min fails to decode. |
| `eh_rejects_an_unknown_kind_id_in_a_bucket` | An unknown kind_id in a bucket is refused. | Verifies a bucket relabelled `0xff 0xff` fails with a "not a wire variant" complaint. |

### EHSketchList

Test file: [`src/sketch_framework/eh_sketch_list.rs`](../src/sketch_framework/eh_sketch_list.rs)

Wire tests: [`src/sketch_framework/eh_sketch_list/wire.rs`](../src/sketch_framework/eh_sketch_list/wire.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `insert_routes_to_countl2hh_and_univmon` | Insert routes to countl2hh and univmon. | Verifies variant routing by checking `COUNTL2HH` estimate `>=9` after 9 inserts and `UNIVMON` `bucket_size=6` after 6 inserts. |
| `count_sketch_insert_and_query_round_trip` | Count insert and query round trip. | Confirms the `Count` variant updates/query path by inserting one key and verifying returned estimate is at least `1.0`. |
| `ddsketch_insert_and_quantile_query_round_trip` | DDSketch insert and quantile query round trip. | Inserts `10,20,30` into DDSketch variant and verifies queried median (`q=0.5`) lies within `[10.0, 30.0]`. |
| `supports_norm_whitelist_is_enforced` | Supports norm whitelist is enforced. | Validates norm capability matrix: `CM/CS/DDS` support `L1` only, while `COUNTL2HH/UNIVMON` support `L2` only. |
| `eh_sketch_list_round_trip_serialization` | The envelope frames a union under kind_id `0x14 0x00`, and the state survives a round trip. | For a `3x8` fast-path Count-Min variant holding one key, verifies the bytes open with the ASAPv1 magic, a `kind_id_len` of `2`, and `0x14 0x00`, and that the decode keeps a `sketch_type` of `CountMin` and answers that key as the source does. |
| `eh_sketch_list_every_variant_round_trips` | Every variant this build carries round-trips and keeps its type. | For each populated variant, verifies the decode reports the same `sketch_type` and re-serializes to the bytes it came from. |
| `eh_sketch_list_kind_ids_are_build_independent` | The ten nested kind_ids and their registry names are the same in every build. | Verifies `variant_name` maps each of the ten ids to its name and `0xff 0xff` to `None`, and that every populated variant emits the id the table gives it. |
| `eh_sketch_list_rejects_an_experimental_kind_id` | An experimental variant's kind_id is refused without the feature. | Verifies crafted triples carrying the `Coco`, `Elastic`, and `UniformSampling` kind_ids each fail to decode, with a message naming the variant and "experimental" in builds without the feature. |
| `eh_sketch_list_rejects_an_unknown_kind_id` | An unknown kind_id is refused. | Verifies a Count-Min triple relabelled `0xff 0xff` fails with a "not a wire variant" complaint. |
| `eh_sketch_list_rejects_sibling_algorithm_kind_ids` | The nested ids are pinned to one algorithm each. | Verifies an HLL triple relabelled `0x01 0x01` or `0x01 0x03` fails to decode, and that a KLL triple relabelled `0x06 0x01` fails too. |
| `eh_sketch_list_rejects_a_mismatched_kind_id_and_descriptor` | A kind_id that does not match the blocks it carries is refused by the variant's own decoder. | Verifies an HLL triple relabelled with the Count-Min kind_id fails to decode. |
| `eh_sketch_list_rejects_foreign_kind_ids` | A Count-Min envelope and an `ExponentialHistogram` envelope are not union envelopes. | Verifies both fail to decode as an `EHSketchList`. |
| `eh_sketch_list_metadata_rejects_unknown_keys` | An unexpected metadata key fails closed. | Encodes `metadata_version` plus a `bogus_field` as a named map and verifies it does not decode as `EhSketchListMetadata`. |
| `eh_sketch_list_metadata_rejects_a_missing_key` | `metadata_version` is required. | Encodes an empty named map and verifies it does not decode as `EhSketchListMetadata`. |
| `eh_sketch_list_rejects_crafted_blocks` | Crafted blocks fail closed with an error, never a panic. | Verifies a Count-Min triple whose descriptor is cut in half, and one whose state is three `0xc1` bytes, each fail to decode. |
| `eh_sketch_list_rejects_a_custom_hash_profile_descriptor` | A descriptor naming a custom hash profile is refused, since the variant's decoder pins the profile of the type it rebuilds. | Verifies a triple built from a custom-profile Count-Min fails to decode. |
| `eh_sketch_list_query_agrees_after_decode` | A decoded union answers a query the way the original did. | For every populated variant, verifies the source and the decode return the same answer for that variant's sample input. |

### EHUnivOptimized

Test file: [`src/sketch_framework/eh_univ_optimized.rs`](../src/sketch_framework/eh_univ_optimized.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `basic_insertion_and_query` | Basic insertion and query. | For updates `{(1,5),(2,3),(1,2)}` across `[100,102]`, verifies map-tier result with exact counts (`1->7`, `2->3`, `total=10`) plus `L1=10` and `cardinality=2`. |
| `map_merge_bounds_volume` | Map merge bounds volume. | With `k=1` and 50 one-count updates, verifies merge policy bounds growth so `bucket_count < 50`. |
| `promotion_creates_sketch_buckets` | Promotion creates sketch buckets. | Under small promotion thresholds and many distinct updates, verifies at least one map bucket is promoted (`um_buckets` becomes non-empty). |
| `window_expiration` | Window expiration. | With `window=100`, advancing to `t=200` after earlier inserts confirms expiration by forcing oldest surviving `min_time` to recent range (`>=100` or `==200`). |
| `hybrid_query_returns_sketch` | Hybrid query returns sketch. | After forcing both map and sketch tiers to coexist, verifies interval query spanning both returns `EHUnivQueryResult::Sketch` (not map-only). |
| `cover_check` | Cover check. | Verifies coverage logic transitions from false (empty) to true for contained intervals and remains false when query extends outside observed range. |
| `accuracy_known_distribution` | Accuracy known distribution. | On fixed known histogram, verifies query estimates for `L1`, `L2`, cardinality, and entropy each stay within `10%` relative error. |
| `pool_used_during_promotion` | Pool used during promotion. | With bounded preallocated pool, promotion workload verifies sketch-tier creation and confirms pool allocation accounting remains active (`total_allocated >= 2`). |
| `correctness_map_only_exact` | Correctness map only exact. | For map-only regime, verifies `L1/L2/cardinality/entropy` each match exact truth within `1%` tolerance. |
| `correctness_subinterval_query` | Correctness subinterval query. | For two-phase stream, verifies full-interval query recovers `L1` approximately `200` and `cardinality` approximately `2` within `5%` tolerance. |
| `correctness_expired_data_excluded` | Correctness expired data excluded. | After sliding beyond window cutoff, verifies very old segment is excluded by checking earliest retained bucket time is at least `50`. |
| `correctness_volume_bounded_long_stream` | Correctness volume bounded long stream. | Over `20_000` updates with `k=4`, verifies EH volume bound by requiring maximum observed bucket count `< 200`. |
| `correctness_pool_recycling_across_cycles` | Correctness pool recycling across cycles. | Long-run expiration/promotion cycling keeps pool bounded (`total_allocated < 50`) and still returns valid interval query results. |
| `correctness_sketch_merge_preserves_metrics` | Correctness sketch merge preserves metrics. | After repeated promotions/merges, verifies each sketch bucket has positive `L2^2` and stored `l22` stays within `1%` relative difference of recomputed value. |
| `accuracy_zipf_distribution_sketch_tier` | Accuracy Zipf distribution sketch tier. | On heavy/medium/light Zipf-like stream in sketch tier, requires `L1/L2/cardinality/entropy` relative errors each `<= 15%`. |
| `accuracy_uniform_distribution` | Accuracy uniform distribution. | On uniform stream, requires `L1/L2/cardinality/entropy` relative errors each `<= 10%`. |
| `accuracy_sliding_window` | Accuracy sliding window. | Across suffix and periodic sliding-window queries, verifies average relative error for `L1`, `L2`, cardinality, and entropy is each below `15%`. |
| `accuracy_varies_with_k` | Accuracy varies with K. | For `k in {2,8,32}`, verifies per-k average of `L1/L2` relative errors remains under `15%` on same fixed stream/window. |
| `accuracy_suffix_queries` | Accuracy suffix queries. | Across suffix lengths `[1000,2000,5000,8000]`, verifies worst observed `L2` relative error remains below `20%`. |
| `accuracy_distribution_shift` | Accuracy distribution shift. | For two-phase distribution shift stream, verifies full-span `L1/L2/cardinality/entropy` estimates each stay within `15%` relative error. |

## Common

### Common Hash Utilities

Test file: [`src/common/hash.rs`](../src/common/hash.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `hash128_seeded_preserves_cardinality` | Hash128 seeded preserves cardinality. | With `SEED_IDX=0` and `SAMPLE_SIZE=5000`, verifies uniform and Zipf sample unique-input counts exactly match unique-hash counts (no observed collisions). |
| `hash128_seeded_is_deterministic_for_repeated_inputs` | Hash128 seeded is deterministic for repeated inputs. | For fixed key `"deterministic-key"` and seed `3`, verifies 100 repeated `hash128_seeded` calls always equal the first hash value. |
| `digest_hasher_spreads_digests_that_share_their_low_bits` | `DigestHasher` mixes rather than passing a digest through. | Hashes `0..1024` shifted left by 16, so every digest shares its ten low bits, and verifies more than 550 of the 1,024 low-bit buckets are occupied rather than the single bucket a pass-through hash would fill; the unshifted range is held to the same bound. |
| `owned_byte_keys_hash_like_the_borrowed_input` | A `HeapItem::Bytes` key hashes exactly like the `DataInput::Bytes` a caller queries with. | Across the empty byte string, `b"projectasap"`, `[0xff, 0x00, 0xfe]`, and 64 `0x80` bytes, at seed indices `0`, `1`, `CANONICAL_HASH_SEED`, and one past the seed list, verifies `hash64_seeded` equals `hash_item64_seeded` and `hash128_seeded` equals `hash_item128_seeded`. |

### Common Input Types

Test file: [`src/common/input.rs`](../src/common/input.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `a_byte_array_owns_and_borrows_back_unchanged` | A borrowed byte array owns as a byte array, not as a string. | Verifies `input_to_owned(&DataInput::Bytes([0xff, 0x00, 0xfe]))` is `HeapItem::Bytes` of those bytes, that `heap_item_to_sketch_input` borrows the same bytes back, and that the owned key compares equal to the original input. |
| `a_byte_key_is_not_the_string_of_the_same_bytes` | `Bytes` and `Str` are different keys even where the bytes are valid UTF-8. | Verifies the owned forms of `Bytes(b"abc")` and `Str("abc")` are unequal, that the byte key matches only `DataInput::Bytes`, and that the string key matches `Str` and `String` but not `Bytes`. |

### Common Heap Utilities

Test file: [`src/common/heap.rs`](../src/common/heap.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `heap_retains_top_k_items_by_count` | Heap retains top K items by count. | For `HHHeap::new(3)` updated with counts `1..5`, verifies heap size is `3` and retained counts are exactly `[3,4,5]`. |
| `update_count_increments_existing_entry` | Update count increments existing entry. | Repeatedly updates key `alpha` with counts `1,2,3` and verifies stored heap entry count is `3` (incremental update, not replacement). |
| `clean_resets_heap_state` | Clean resets heap state. | After inserting two items into `HHHeap::new(2)`, `clear()` is verified to leave the heap empty. |
| `test_min_heap_basic` | Test min heap basic. | For `CommonHeap::<i32, KeepSmallest>::new_min(5)`, verifies `peek=1` and pop order `1,3,5,7`, then `None`. |
| `test_max_heap_basic` | Test max heap basic. | For `CommonHeap::<i32, KeepLargest>::new_max(5)`, verifies `peek=7` and pop order `7,5,3,1`, then `None`. |
| `test_bounded_heap_capacity` | Test bounded heap capacity. | With min-heap capacity `3`, verifies length never exceeds 3 and final retained values are `[5,7,10]` after pushing `5,3,7,1,10`. |
| `test_update_at` | Test update at. | After mutating an internal element (`heap[1]=3`) and calling `update_at(1)`, verifies heap root updates so `peek()` becomes `3`. |
| `test_custom_struct_with_ord` | Test custom struct with ord. | Uses `HHItem` values with counts `5,3,7` and verifies min-heap ordering by checking root count is `3`. |
| `test_topk_use_case` | Test topk use case. | Simulated top-k flow keeps only counts `[3,4,5]` at capacity 3 and verifies lookup for `key-4` succeeds with count `4`. |
| `test_heap_size` | Test heap size. | Verifies both `CommonHeap<u64, KeepSmallest>` and `CommonHeap<u64, KeepLargest>` sizes equal `size_of::<Vec<u64>>() + size_of::<usize>()`. |
| `test_topk_with_custom_comparator` | Test topk with custom comparator. | With custom comparator and capacity 3, verifies low-count insert is rejected/replaced as expected so heap size is 3 and root count is `5`. |
| `test_exact_topk_heap_replacement` | Test exact topk heap replacement. | Reproduces TopK-style find/update flow for keys `1..5`, verifies retained counts `[3,4,5]`, finds `key-4` with count `4`, then verifies `clear()` makes heap empty. |
| `the_index_survives_a_long_churning_stream` | The index stays exact through eviction, promotion, and re-entry. | At capacities 1, 2, 7, 64, and 512, streams 20,000 skewed draws over a 2,048-key domain and every 97 steps verifies `slots` carries each resident's digest, every position is listed in its own bucket, `find_heap_item` returns that position, and the index holds no entry beyond the residents; residency ends at `min(capacity, distinct)`. |
| `the_residents_are_the_k_largest_counts` | The residents are exactly the `k` largest counts. | Over 20,000 skewed draws across a 512-key domain into a 32-slot heap, verifies against a brute-force ranking that every resident's count is current and at or above the 32nd largest count, and that the heap is full. |
| `re_scoring_a_resident_never_duplicates_it` | A resident is re-scored in place rather than seated twice. | Updates `hot` and `warm` 50 rounds each into a 4-slot heap and verifies the index stays consistent, residency is `2`, `find` locates `hot`, and its count reads `50`. |
| `a_zero_capacity_heap_turns_everything_away` | A zero-capacity heap accepts nothing and stays consistent. | Verifies 32 updates into `HHHeap::new(0)` all return `false`, the heap stays empty, `find` returns `None`, and the index is consistent. |
| `clearing_drops_the_index_with_the_heap` | `clear` drops the index along with the entries. | After filling an 8-slot heap from 32 keys, verifies `clear()` leaves it empty with `find` returning `None` and a consistent index, then refills from 40 fresh keys and verifies the index is consistent again at residency `8`. |
| `a_decoded_heap_rebuilds_its_index` | The index is derived, so decoding rebuilds it rather than carrying it. | Round-trips a 16-slot heap fed 5,000 skewed draws through `rmp-serde` and verifies the decoded index is consistent, residency and capacity match, every resident is still found, and a further update re-scores a resident to `1_000_000` with the index still consistent. |
| `the_indexed_heap_matches_the_rebuild_implementation` | Retention does not depend on how the index is maintained. | At capacities 0, 1, 2, 3, 7, 8, 64, and 257, runs 30,000 skewed draws over a 4,096-key domain through both the shipped heap and a reference that rebuilds its key index in full after every accepted update, and verifies at every step that the completeness flag agrees and that both arrays hold the same key and count at every position. |
| `the_indexed_heap_matches_the_rebuild_implementation_on_string_keys` | The same agreement on the owned-key path and its different hash. | Runs 20,000 `flow::<n>` string keys over a 2,048-key domain into a 32-slot heap and verifies the shipped and rebuild implementations agree on the completeness flag and on every position's key and count at every step. |
| `the_two_agree_when_counts_move_in_both_directions` | Falling counts sink a resident back down and the two still agree. | Feeds a 16-slot heap 10,000 updates whose counts oscillate over `-30..=30` rather than only climbing, and verifies the shipped and rebuild implementations agree on the completeness flag and on every position at every step. |
| `the_heap_is_reproducible_across_runs` | The same stream gives the same heap, array position included. | Runs 20,000 skewed draws over a 1,024-key domain into a 64-slot heap five times over and verifies all five key/count snapshots are identical, so neither the index nor the map's iteration order reaches the output. |
| `a_bucket_holding_both_sides_of_a_swap_is_unchanged` | Swapping two entries of one bucket leaves the bucket alone. | With heap positions `0` and `1` both carrying digest `7`, `swap_entry(0, 1)` is verified to leave the slots as `[7, 7]` and the bucket listing positions `[0, 1]`. |
| `a_swap_moves_only_its_own_entry_out_of_a_shared_bucket` | A swap patches the matching entry, not the bucket head. | With digest `7` listed at positions `[2, 0]` and digest `9` at `[1]`, `swap_entry(0, 1)` is verified to leave slots `[9, 7, 7]`, digest `7` listing `[1, 2]`, and digest `9` listing `[0]`; patching the head instead would corrupt position 2. |
| `dropping_one_entry_keeps_the_rest_of_its_bucket` | Dropping one position leaves the bucket's other positions. | With digest `7` listed at `[0, 3, 5]`, `drop_entry(7, 3)` is verified to leave `[0, 5]`. |
| `dropping_the_last_entry_removes_the_bucket` | The bucket goes when its last position does. | With digest `7` listed at `[4]`, `drop_entry(7, 4)` is verified to remove the key from the index entirely. |
| `two_shared_buckets_follow_the_sift` | The index tracks the slots through a long run of sifts. | Seeds a 16-item `HHHeap` so residents alternate between two digests, then runs 200 `rescore` calls and verifies after each that every position is listed under the digest its slot names and that the buckets together name every heap position exactly once. |

### Common Structure Utilities

Test file: [`src/common/structure_utils.rs`](../src/common/structure_utils.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `median_test` | Median test. | For 1,000 seeded random arrays of lengths 3, 4, and 5, verifies `compute_median_inline_f64` exactly matches sort-based median for every case. |

### Vector2D (Common Structure)

Test file: [`src/common/structures/vector2d.rs`](../src/common/structures/vector2d.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `required_bits_match_expected_thresholds` | Required bits match expected thresholds. | Verifies `Vector2D::get_required_bits()` returns `64` for `(3,4096)`, `32` for `(3,64)`, and `128` for `(5,1_048_576)`. |

### BitMatrix (Common Structure)

Test file: [`src/common/structures/bit_matrix.rs`](../src/common/structures/bit_matrix.rs)

| test_name | test_description | what_is_tested |
| --- | --- | --- |
| `every_cell_owns_exactly_one_bit` | Each cell owns one bit and no other cell's. | Across 33 geometries (`1`, `2`, and `5` rows by columns `1`, `2`, `7`, `63`, `64`, `65`, `100`, `127`, `128`, `129`, and `255`), sets every cell in turn and verifies it read clear beforehand, reads set afterwards, and that `count_ones` rises by exactly one each time, ending at `rows * cols` with a `fill_ratio` of `1.0`. |
| `row_padding_is_never_reachable_or_counted` | Rows are padded out to whole words and the padding never counts. | Over the same 33 geometries, fills every addressable cell and verifies `size_in_bytes` is `rows * cols.div_ceil(64) * 8` while `count_ones` is exactly `rows * cols`. |
| `put_and_clear_reset_individual_bits` | `put(.., false)` and `clear` both return the matrix to empty. | Over the same 33 geometries, sets every cell through `put(.., true)`, clears them one at a time through `put(.., false)` and verifies `count_ones` is `0` and `fill_ratio` is `0.0`, then sets the last cell and verifies `clear()` empties it again. |
| `a_column_past_the_last_one_panics_rather_than_aliasing` | A column past `cols` panics instead of landing in row padding. | Verifies `set(0, 100)` on a `3x100` matrix panics with "(0, 100) is outside a 3x100 bit matrix". |
| `a_column_reaching_the_next_row_panics` | A column that reaches the next row's words panics. | Verifies `set(0, 128)` on a `3x100` matrix panics with "(0, 128) is outside a 3x100 bit matrix" rather than aliasing row 1. |
| `a_row_past_the_last_one_panics` | A row past the last one panics. | Verifies `get(3, 0)` on a `3x100` matrix panics with "(3, 0) is outside a 3x100 bit matrix". |
| `put_checks_bounds_too` | `put` is bounds-checked like `set` and `get`. | Verifies `put(0, 120, true)` on a `2x100` matrix panics with "(0, 120) is outside a 2x100 bit matrix". |
| `union_takes_the_bitwise_or` | `union_from` is the bitwise or of the two grids. | Unions a `3x100` matrix holding `(0,5)` and `(1,99)` with one holding `(1,99)` and `(2,0)`, then verifies `count_ones` is `3` and all three cells read set. |
| `union_across_geometries_panics` | A union across geometries panics. | Verifies `union_from` between a `3x100` and a `3x101` matrix panics with "bit matrices must have the same dimensions". |
| `a_zero_dimension_is_rejected` | A zero dimension panics at construction. | Verifies `BitMatrix::new(0, 64)` panics with "a bit matrix needs both dimensions". |
| `a_round_trip_recomputes_the_derived_fields` | A decoded matrix folds hashes exactly as the original does. | Over the same 33 geometries, round-trips through `rmp-serde` and verifies the dimensions and every cell match, then drives one `hash_for_matrix` digest through `fast_insert` on both the original and the decoded matrix and verifies they set the same cells. |
| `a_payload_that_does_not_fit_its_dimensions_is_rejected` | A word count that disagrees with the dimensions fails at decode. | Verifies a `3x100` payload carrying 2 words is refused with "needs 6 words, got 2", one carrying 9 words is refused, and a zero-row payload is refused with "needs both dimensions". |
| `the_wire_form_carries_only_the_stored_fields` | The wire carries `words`, `rows`, and `cols` and nothing derived from them. | Verifies a fresh `3x100` matrix encodes byte for byte identically to a struct of just those three fields. |
