//! Cross-language ASAPv1 golden byte-vector tests.
//!
//! Each fixture is built from a fixed, KNOWN raw sketch state (register bytes /
//! matrix values set directly, never hashed) and asserted to serialize to the
//! exact bytes checked into `asapv1_golden/*.hex`. The same `.hex` files live
//! byte-identically in `sketchlib-go/asapv1_golden/`, where the Go test suite
//! proves its encoder emits the same bytes. Together they machine-prove the Rust
//! and Go ASAPv1 wire encodings are byte-identical for these configs.
//!
//! See `asapv1_golden/README.md`.

use asap_sketchlib::{
    Classic, Count, CountMin, ErtlMLE, FastPath, HllSketch, HllVariant, HyperLogLogHIPP12,
    HyperLogLogP12, KLL, MessagePackCodec, RegularPath, Vector2D,
};

fn decode_hex(s: &str) -> Vec<u8> {
    let s = s.trim();
    assert!(s.len() % 2 == 0, "golden hex must have even length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

const GOLDEN_CLASSIC: &str = include_str!("../asapv1_golden/hll_classic_p12.hex");
const GOLDEN_ERTL: &str = include_str!("../asapv1_golden/hll_ertl_mle_p12.hex");
const GOLDEN_HIP: &str = include_str!("../asapv1_golden/hll_hip_p12.hex");
const GOLDEN_CMS_I64: &str = include_str!("../asapv1_golden/cms_i64_regular_2x3.hex");
const GOLDEN_CMS_F64: &str = include_str!("../asapv1_golden/cms_f64_fast_2x3.hex");
const GOLDEN_CS_REGULAR: &str = include_str!("../asapv1_golden/cs_i64_regular_2x4.hex");
const GOLDEN_CS_FAST: &str = include_str!("../asapv1_golden/cs_i64_fast_2x4.hex");
const GOLDEN_CS_I32: &str = include_str!("../asapv1_golden/cs_i32_regular_2x4.hex");
const GOLDEN_KLL_F64: &str = include_str!("../asapv1_golden/kll_f64_k200.hex");
const GOLDEN_KLL_I64: &str = include_str!("../asapv1_golden/kll_i64_k200.hex");

/// The known P12 register pattern shared by all three HLL fixtures.
fn p12_registers() -> Vec<u8> {
    let mut r = vec![0u8; 4096];
    r[0] = 1;
    r[1] = 7;
    r[100] = 42;
    r[4095] = 3;
    r
}

const I64_VALS: [[i64; 3]; 2] = [[0, 1, 127], [128, 300, 65536]];
const F64_VALS: [[f64; 3]; 2] = [[0.0, 1.5, 2.25], [3.75, 4.125, 5.0625]];

/// Count Sketch cells are **signed** (the sketch adds `±weight`), so this
/// fixture sweeps the msgpack integer widths in both directions: row 0 is
/// positive fixint / positive fixint max / uint8 / uint32, row 1 is negative
/// fixint / int8 / int16 / int32.
const CS_VALS: [[i64; 4]; 2] = [[0, 127, 128, 65536], [-1, -33, -32768, -2147483648]];

// ---------------------------------------------------------------------------
// HLL: build known state -> serialize == golden, and golden round-trips.
// ---------------------------------------------------------------------------

#[test]
fn hll_classic_p12_matches_golden() {
    let want = decode_hex(GOLDEN_CLASSIC);
    let regs = p12_registers();

    // Build known state and serialize.
    let got = HllSketch::from_raw(HllVariant::Regular, 12, regs.clone(), 0.0, 0.0, 0.0)
        .to_msgpack()
        .expect("serialize");
    assert_eq!(got, want, "Classic P12 bytes diverge from golden");

    // Native wire path (src/sketches/hll.rs) round-trips the golden identically.
    let native = HyperLogLogP12::<Classic>::deserialize_from_bytes(&want).expect("native decode");
    assert_eq!(native.registers_as_slice(), regs.as_slice());
    assert_eq!(native.serialize_to_bytes().expect("re-serialize"), want);

    // Portable decode round-trips to the same known state.
    let decoded = HllSketch::from_msgpack(&want).expect("decode");
    assert_eq!(decoded.registers, regs);
    assert_eq!(decoded.precision, 12);
    assert_eq!(decoded.variant, HllVariant::Regular);
}

#[test]
fn hll_ertl_mle_p12_matches_golden() {
    let want = decode_hex(GOLDEN_ERTL);
    let regs = p12_registers();

    let got = HllSketch::from_raw(HllVariant::Datafusion, 12, regs.clone(), 0.0, 0.0, 0.0)
        .to_msgpack()
        .expect("serialize");
    assert_eq!(got, want, "Ertl-MLE P12 bytes diverge from golden");

    let native = HyperLogLogP12::<ErtlMLE>::deserialize_from_bytes(&want).expect("native decode");
    assert_eq!(native.registers_as_slice(), regs.as_slice());
    assert_eq!(native.serialize_to_bytes().expect("re-serialize"), want);

    let decoded = HllSketch::from_msgpack(&want).expect("decode");
    assert_eq!(decoded.registers, regs);
    assert_eq!(decoded.variant, HllVariant::Datafusion);
}

#[test]
fn hll_hip_p12_matches_golden() {
    let want = decode_hex(GOLDEN_HIP);
    let regs = p12_registers();

    let got = HllSketch::from_raw(HllVariant::Hip, 12, regs.clone(), 1.5, 2.5, 3.0)
        .to_msgpack()
        .expect("serialize");
    assert_eq!(got, want, "HIP P12 bytes diverge from golden");

    // Native HIP wire path round-trips the golden identically.
    let native = HyperLogLogHIPP12::deserialize_from_bytes(&want).expect("native decode");
    assert_eq!(native.serialize_to_bytes().expect("re-serialize"), want);

    let decoded = HllSketch::from_msgpack(&want).expect("decode");
    assert_eq!(decoded.registers, regs);
    assert_eq!(decoded.variant, HllVariant::Hip);
    assert_eq!(decoded.hip_kxq0, 1.5);
    assert_eq!(decoded.hip_kxq1, 2.5);
    assert_eq!(decoded.hip_est, 3.0);
}

// ---------------------------------------------------------------------------
// Count-Min: build known matrix state -> serialize == golden, round-trips.
// ---------------------------------------------------------------------------

#[test]
fn cms_i64_regular_2x3_matches_golden() {
    let want = decode_hex(GOLDEN_CMS_I64);

    let sketch =
        CountMin::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::from_fn(2, 3, |r, c| {
            I64_VALS[r][c]
        }));
    let got = sketch.serialize_to_bytes().expect("serialize");
    assert_eq!(got, want, "CMS i64/regular bytes diverge from golden");

    let decoded =
        CountMin::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&want).expect("decode");
    let flat: Vec<i64> = I64_VALS.iter().flatten().copied().collect();
    assert_eq!(decoded.as_storage().as_slice(), flat.as_slice());
    assert_eq!(decoded.rows(), 2);
    assert_eq!(decoded.cols(), 3);
    assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), want);
}

#[test]
fn cms_f64_fast_2x3_matches_golden() {
    let want = decode_hex(GOLDEN_CMS_F64);

    let sketch =
        CountMin::<Vector2D<f64>, FastPath>::from_storage(Vector2D::from_fn(2, 3, |r, c| {
            F64_VALS[r][c]
        }));
    let got = sketch.serialize_to_bytes().expect("serialize");
    assert_eq!(got, want, "CMS f64/fast bytes diverge from golden");

    let decoded =
        CountMin::<Vector2D<f64>, FastPath>::deserialize_from_bytes(&want).expect("decode");
    let flat: Vec<f64> = F64_VALS.iter().flatten().copied().collect();
    assert_eq!(decoded.as_storage().as_slice(), flat.as_slice());
    assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), want);
}

// ---------------------------------------------------------------------------
// Count Sketch: build known matrix state -> serialize == golden, round-trips.
// All three fixtures hold the SAME matrix: the i64 pair differs only by `mode`
// and the i32 fixture only by `counter_type`, so the set pins that both
// metadata keys reach the bytes and that nothing else does.
// ---------------------------------------------------------------------------

#[test]
fn cs_i64_regular_2x4_matches_golden() {
    let want = decode_hex(GOLDEN_CS_REGULAR);

    let sketch =
        Count::<Vector2D<i64>, RegularPath>::from_storage(Vector2D::from_fn(2, 4, |r, c| {
            CS_VALS[r][c]
        }));
    let got = sketch.serialize_to_bytes().expect("serialize");
    assert_eq!(
        got, want,
        "Count Sketch i64/regular bytes diverge from golden"
    );

    let decoded =
        Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&want).expect("decode");
    let flat: Vec<i64> = CS_VALS.iter().flatten().copied().collect();
    assert_eq!(decoded.as_storage().as_slice(), flat.as_slice());
    assert_eq!(decoded.rows(), 2);
    assert_eq!(decoded.cols(), 4);
    assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), want);
}

#[test]
fn cs_i64_fast_2x4_matches_golden() {
    let want = decode_hex(GOLDEN_CS_FAST);

    let sketch = Count::<Vector2D<i64>, FastPath>::from_storage(Vector2D::from_fn(2, 4, |r, c| {
        CS_VALS[r][c]
    }));
    let got = sketch.serialize_to_bytes().expect("serialize");
    assert_eq!(got, want, "Count Sketch i64/fast bytes diverge from golden");

    let decoded = Count::<Vector2D<i64>, FastPath>::deserialize_from_bytes(&want).expect("decode");
    let flat: Vec<i64> = CS_VALS.iter().flatten().copied().collect();
    assert_eq!(decoded.as_storage().as_slice(), flat.as_slice());
    assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), want);

    // Same matrix, different mode -> different bytes.
    assert_ne!(want, decode_hex(GOLDEN_CS_REGULAR));
}

#[test]
fn cs_i32_regular_2x4_matches_golden() {
    let want = decode_hex(GOLDEN_CS_I32);

    let sketch =
        Count::<Vector2D<i32>, RegularPath>::from_storage(Vector2D::from_fn(2, 4, |r, c| {
            CS_VALS[r][c] as i32
        }));
    let got = sketch.serialize_to_bytes().expect("serialize");
    assert_eq!(
        got, want,
        "Count Sketch i32/regular bytes diverge from golden"
    );

    let decoded =
        Count::<Vector2D<i32>, RegularPath>::deserialize_from_bytes(&want).expect("decode");
    let flat: Vec<i32> = CS_VALS.iter().flatten().map(|&v| v as i32).collect();
    assert_eq!(decoded.as_storage().as_slice(), flat.as_slice());
    assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), want);

    // Same matrix, same mode, narrower counter: the fixtures differ only in the
    // metadata `counter_type`, so the payload bytes are identical.
    let wide = decode_hex(GOLDEN_CS_REGULAR);
    assert_eq!(want.len(), wide.len());
    assert_ne!(want, wide);
    assert!(
        Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&want).is_err(),
        "the i32 golden must not decode as an i64 sketch"
    );
}

// ---------------------------------------------------------------------------
// KLL: build known state (k=200, seed 42, integers 1..=50 — below the level-0
// capacity, so no compaction fires and the retained set is deterministic) ->
// serialize == golden, and golden round-trips. Matches the deterministic
// scenario the proto parity test uses (sketchlib-go's KLLSketch over the same
// input), so the coin state (42) lines up cross-language.
// ---------------------------------------------------------------------------

/// A k=200 KLL over `1..=50` with a fixed compaction seed: fully deterministic.
fn kll_1to50<T: From<u8> + asap_sketchlib::NumericalValue>(seed: u64) -> KLL<T> {
    let mut sketch = KLL::<T>::init_kll_with_seed(200, seed);
    for v in 1..=50u8 {
        sketch.update(&T::from(v));
    }
    sketch
}

#[test]
fn kll_f64_k200_matches_golden() {
    let want = decode_hex(GOLDEN_KLL_F64);

    let sketch = kll_1to50::<f64>(42);
    let got = sketch.serialize_to_bytes().expect("serialize");
    assert_eq!(got, want, "KLL f64 bytes diverge from golden");

    // Golden round-trips: decode, and re-encode is byte-identical.
    let decoded = KLL::<f64>::deserialize_from_bytes(&want).expect("decode");
    assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), want);
    assert_eq!(decoded.quantile(0.0), 1.0);
    assert_eq!(decoded.quantile(1.0), 50.0);
}

#[test]
fn kll_i64_k200_matches_golden() {
    let want = decode_hex(GOLDEN_KLL_I64);

    let sketch = kll_1to50::<i64>(42);
    let got = sketch.serialize_to_bytes().expect("serialize");
    assert_eq!(got, want, "KLL i64 bytes diverge from golden");

    let decoded = KLL::<i64>::deserialize_from_bytes(&want).expect("decode");
    assert_eq!(decoded.serialize_to_bytes().expect("re-serialize"), want);
    assert_eq!(decoded.quantile(0.0), 1.0);
    assert_eq!(decoded.quantile(1.0), 50.0);
}
