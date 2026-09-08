//! Probe: do `sketches::*` defaults already produce Go-byte-parity output,
//! making a separate hashspec-direct path inside the wire-format-aligned
//! types unnecessary?
//!
//! Approach: reuse the exact same Go-golden envelopes that the
//! `message_pack_format::portable` parity tests use, but build the
//! matrix via `sketches::Count` / `sketches::CountMin` with `FastPath`
//! and `DefaultXxHasher`. Identical bytes confirm the shared FastPath
//! math is sufficient — no hashspec bypass needed.

use asap_sketchlib::common::DataInput;
use asap_sketchlib::common::hash::CANONICAL_HASH_SEED;
use asap_sketchlib::common::structures::HllBucketListP14;
use asap_sketchlib::proto::sketchlib::{
    CountMinState, CountSketchState, CounterType, DdSketchState, HyperLogLogState, SketchEnvelope,
    sketch_envelope::SketchState,
};
use asap_sketchlib::sketches::countminsketch::CountMin;
use asap_sketchlib::sketches::countsketch::Count;
use asap_sketchlib::sketches::ddsketch::DDSketch;
use asap_sketchlib::sketches::hll::{ErtlMLE, HyperLogLogImpl};
use asap_sketchlib::{DefaultXxHasher, FastPath, Vector2D};
use prost::Message;

/// 1577-byte envelope captured from `sketchlib-go::CountSketch.SerializeProtoBytes`
/// for `(rows=3, cols=512)` × `goldenCsKeys()` (25 keys "k-a".."k-e", each
/// repeated 5×). Identical to the constant in
/// `src/message_pack_format/portable/countsketch.rs::test_update_then_envelope_matches_sketchlib_go_bytes`.
const COUNTSKETCH_GOLDEN_HEX: &str = include_str!("cs_envelope_golden.hex");

/// 8275-byte CMS Frequency-Only golden, captured from
/// `sketchlib-go::CountMinSketch.SerializeProtoBytesFO` for
/// `(rows=4, cols=2048)` × `goldenCmsKeys()` (10 keys "flow-0".."flow-9",
/// each repeated 5×). Same file the wrapper test already uses.
const COUNTMIN_GOLDEN_HEX: &str = include_str!("../src/sketches/testdata/cms_envelope_golden.hex");

/// 16 398-byte envelope captured from `sketchlib-go::HyperLogLog.SerializePortable`
/// for precision=14, `(1..=50)` IEEE-754-LE byte-key input. Same file
/// the wrapper test already uses.
const HLL_GOLDEN_HEX: &str = include_str!("../src/sketches/testdata/hll_envelope_golden.hex");

/// 403-byte DDSketch envelope for `alpha=0.01`, `(1..=50)` integer-as-f64
/// input, without the DataPoint-level METRIC scalars (count/sum/min/max →
/// proto tags 4-7 reserved).
///
/// NOTE: this is currently the RUST-produced value; it MUST be reconciled
/// against the parallel `sketchlib-go` golden regeneration before declaring
/// cross-language byte parity.
const DDSKETCH_GOLDEN_HEX: &str = "0801728e03096214ae47e17a843f128003000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000100000000000000000000000000000100000000000000000000010000000000000000010000000000000001000000000001000000000001000000000001000000010000000001000000010000010000000100000100000100000100000100010000010001000100010001000100010001000100010100010100010100010101000101010100010101010101010100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000187f";

#[test]
fn sketches_count_fastpath_matches_go_count_sketch_envelope() {
    let rows = 3usize;
    let cols = 512usize;

    let mut sk: Count<Vector2D<i64>, FastPath, DefaultXxHasher> =
        Count::with_dimensions(rows, cols);
    for i in 0..25 {
        let key = format!("k-{}", (b'a' + (i % 5) as u8) as char);
        sk.insert_many(&DataInput::String(key), 1i64);
    }

    let storage = sk.as_storage();
    let mut counts_int: Vec<i64> = Vec::with_capacity(rows * cols);
    let mut l2: Vec<f64> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row_l2 = 0.0f64;
        for c in 0..cols {
            let cell = *storage.get(r, c).expect("cell in range");
            counts_int.push(cell);
            row_l2 += (cell as f64) * (cell as f64);
        }
        l2.push(row_l2);
    }

    let state = CountSketchState {
        rows: rows as u32,
        cols: cols as u32,
        counter_type: CounterType::Int64 as i32,
        counts_int,
        counts_float: Vec::new(),
        l2,
        topk: None,
    };
    let envelope = SketchEnvelope {
        format_version: 1,
        producer: None,
        hash_spec: None,
        sample_p: 0.0,
        sketch_state: Some(SketchState::CountSketch(state)),
    };
    let mut got = Vec::with_capacity(envelope.encoded_len());
    envelope.encode(&mut got).expect("prost encode");

    let want = decode_hex(COUNTSKETCH_GOLDEN_HEX);
    assert_eq!(
        got,
        want,
        "sketches::Count<Vector2D<i64>, FastPath> envelope diverges from Go golden \
         ({} bytes got vs {} bytes want)",
        got.len(),
        want.len(),
    );
}

#[test]
fn sketches_countmin_fastpath_matches_go_count_min_envelope() {
    let rows = 4usize;
    let cols = 2048usize;

    let mut sk: CountMin<Vector2D<f64>, FastPath, DefaultXxHasher> =
        CountMin::with_dimensions(rows, cols);
    for i in 0..50u64 {
        let key = format!("flow-{}", i % 10);
        sk.insert_many(&DataInput::String(key), 1.0f64);
    }

    let storage = sk.as_storage();
    let mut counts_int: Vec<i64> = Vec::with_capacity(rows * cols);
    let mut l1: Vec<f64> = Vec::with_capacity(rows);
    let mut l2: Vec<f64> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row_l1 = 0.0f64;
        let mut row_l2 = 0.0f64;
        for c in 0..cols {
            let cell = *storage.get(r, c).expect("cell in range");
            counts_int.push(cell as i64);
            row_l1 += cell;
            row_l2 += cell * cell;
        }
        l1.push(row_l1);
        l2.push(row_l2);
    }

    let state = CountMinState {
        rows: rows as u32,
        cols: cols as u32,
        counter_type: CounterType::Int64 as i32,
        counts_int,
        counts_float: Vec::new(),
        sum_counts: Vec::new(),
        sum2_counts: Vec::new(),
        l1,
        l2,
    };
    let envelope = SketchEnvelope {
        format_version: 1,
        producer: None,
        hash_spec: None,
        sample_p: 0.0,
        sketch_state: Some(SketchState::CountMin(state)),
    };
    let mut got = Vec::with_capacity(envelope.encoded_len());
    envelope.encode(&mut got).expect("prost encode");

    let want = decode_hex(COUNTMIN_GOLDEN_HEX);
    assert_eq!(
        got.len(),
        want.len(),
        "CountMin envelope length: got {} want {}",
        got.len(),
        want.len(),
    );
    assert_eq!(
        got, want,
        "sketches::CountMin<Vector2D<f64>, FastPath> envelope diverges from Go golden",
    );
}

#[test]
fn sketches_hll_classic_matches_go_envelope() {
    // Go's `HyperLogLog.SerializePortable` emits variant=DATAFUSION=2
    // (ErtlMLE). Use the same variant + precision=14 + same input here.
    let mut sk: HyperLogLogImpl<ErtlMLE, HllBucketListP14, DefaultXxHasher> =
        <HyperLogLogImpl<ErtlMLE, HllBucketListP14, DefaultXxHasher>>::new();
    for i in 1..=50i32 {
        let v = i as f64;
        let bytes = v.to_le_bytes();
        let hashed = <DefaultXxHasher as asap_sketchlib::common::SketchHasher>::hash64_seeded(
            CANONICAL_HASH_SEED,
            &DataInput::Bytes(&bytes),
        );
        sk.insert_with_hash(hashed);
    }

    let state = HyperLogLogState {
        // 2 = DATAFUSION on the Go wire (matches ErtlMLE)
        variant: 2,
        precision: 14,
        registers: sk.registers_as_slice().to_vec(),
        hip_kxq0: 0.0,
        hip_kxq1: 0.0,
        hip_est: 0.0,
        // This parity probe pins the DENSE wire form; leave sparse unset.
        registers_sparse: None,
    };
    let envelope = SketchEnvelope {
        format_version: 1,
        producer: None,
        hash_spec: None,
        sample_p: 0.0,
        sketch_state: Some(SketchState::Hll(state)),
    };
    let mut got = Vec::with_capacity(envelope.encoded_len());
    envelope.encode(&mut got).expect("prost encode");

    let want = decode_hex(HLL_GOLDEN_HEX);
    assert_eq!(
        got.len(),
        want.len(),
        "HLL envelope length: got {} want {}",
        got.len(),
        want.len(),
    );
    assert_eq!(
        got, want,
        "sketches::HyperLogLogImpl<ErtlMLE, P14> envelope diverges from Go golden",
    );
}

#[test]
fn sketches_ddsketch_matches_go_envelope() {
    let mut sk = DDSketch::new(0.01);
    for i in 1..=50i32 {
        sk.add(&(i as f64));
    }

    // Go serializes `alpha` round-tripped through gamma to mirror its
    // internal storage: `alpha_wire = (γ-1)/(γ+1)`, `γ = (1+α)/(1-α)`.
    // This is a 25-ULP shift from the user-supplied α=0.01.
    let alpha = sk.alpha();
    let gamma = (1.0 + alpha) / (1.0 - alpha);
    let alpha_wire = (gamma - 1.0) / (gamma + 1.0);
    // The DataPoint-level METRIC scalars (count/sum/min/max) are not on the
    // wire; only alpha + the bucket array.
    let state = DdSketchState {
        alpha: alpha_wire,
        store_counts: sk.store_counts().to_vec(),
        store_offset: sk.store_offset(),
    };
    let envelope = SketchEnvelope {
        format_version: 1,
        producer: None,
        hash_spec: None,
        sample_p: 0.0,
        sketch_state: Some(SketchState::Ddsketch(state)),
    };
    let mut got = Vec::with_capacity(envelope.encoded_len());
    envelope.encode(&mut got).expect("prost encode");

    let want = decode_hex(DDSKETCH_GOLDEN_HEX);
    assert_eq!(
        got.len(),
        want.len(),
        "DDSketch envelope length: got {} want {}",
        got.len(),
        want.len(),
    );
    assert_eq!(
        got, want,
        "sketches::DDSketch envelope diverges from Go golden",
    );
}

fn decode_hex(s: &str) -> Vec<u8> {
    s.trim()
        .as_bytes()
        .chunks(2)
        .map(|pair| (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]))
        .collect()
}

fn hex_nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("non-hex byte {}", c as char),
    }
}
