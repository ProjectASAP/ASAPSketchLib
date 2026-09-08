use asap_sketchlib::{
    Bloom, Classic, Count, CountMin, DDSketch, DataInput, FastPath, HyperLogLog, KLL, RegularPath,
    SpaceSaving, Vector2D, hash64_seeded, hash128_seeded,
};

const SEEDS: [usize; 4] = [0, 1, 3, 7];

fn every_variant() -> Vec<(&'static str, DataInput<'static>)> {
    vec![
        ("I8", DataInput::I8(-8)),
        ("I16", DataInput::I16(-16)),
        ("I32", DataInput::I32(-32)),
        ("I64", DataInput::I64(-64)),
        ("I128", DataInput::I128(-128)),
        ("ISIZE", DataInput::ISIZE(-1_000)),
        ("U8", DataInput::U8(8)),
        ("U16", DataInput::U16(16)),
        ("U32", DataInput::U32(32)),
        ("U64", DataInput::U64(64)),
        ("U128", DataInput::U128(128)),
        ("USIZE", DataInput::USIZE(1_000)),
        ("F32", DataInput::F32(1.5)),
        ("F64", DataInput::F64(2.5)),
        ("Str", DataInput::Str("borrowed")),
        ("String", DataInput::String("owned".to_string())),
        ("Bytes", DataInput::Bytes(b"raw-bytes")),
    ]
}

fn assert_same_digest(left: &DataInput, right: &DataInput, why: &str) {
    for seed in SEEDS {
        assert_eq!(
            hash64_seeded(seed, left),
            hash64_seeded(seed, right),
            "seed {seed}: 64-bit digests differ, {why}"
        );
        assert_eq!(
            hash128_seeded(seed, left),
            hash128_seeded(seed, right),
            "seed {seed}: 128-bit digests differ, {why}"
        );
    }
}

fn assert_different_digest(left: &DataInput, right: &DataInput, why: &str) {
    assert_ne!(
        hash64_seeded(0, left),
        hash64_seeded(0, right),
        "64-bit digests collided, {why}"
    );
    assert_ne!(
        hash128_seeded(0, left),
        hash128_seeded(0, right),
        "128-bit digests collided, {why}"
    );
}

#[test]
fn every_data_input_variant_reaches_the_hash() {
    let variants = every_variant();
    assert_eq!(
        variants.len(),
        17,
        "the variant table must cover every DataInput variant"
    );
    let mut digests = std::collections::HashSet::new();
    for (name, value) in &variants {
        let d = hash64_seeded(0, value);
        assert!(
            digests.insert(d),
            "{name} produced a digest another variant already produced"
        );
    }
}

#[test]
fn signed_integer_variants_sign_extend_to_one_canonical_digest() {
    for probe in [-1i8, -8, -128, 0, 1, 42, 127] {
        let widened = DataInput::I64(probe as i64);
        assert_same_digest(
            &DataInput::I8(probe),
            &widened,
            &format!("I8({probe}) must sign-extend onto I64"),
        );
        assert_same_digest(
            &DataInput::I16(probe as i16),
            &widened,
            &format!("I16({probe}) must sign-extend onto I64"),
        );
        assert_same_digest(
            &DataInput::I32(probe as i32),
            &widened,
            &format!("I32({probe}) must sign-extend onto I64"),
        );
        assert_same_digest(
            &DataInput::ISIZE(probe as isize),
            &widened,
            &format!("ISIZE({probe}) must sign-extend onto I64"),
        );
    }
}

#[test]
fn unsigned_integer_variants_zero_extend_to_one_canonical_digest() {
    for probe in [0u8, 1, 7, 255] {
        let widened = DataInput::U64(probe as u64);
        assert_same_digest(
            &DataInput::U8(probe),
            &widened,
            &format!("U8({probe}) must zero-extend onto U64"),
        );
        assert_same_digest(
            &DataInput::U16(probe as u16),
            &widened,
            &format!("U16({probe}) must zero-extend onto U64"),
        );
        assert_same_digest(
            &DataInput::U32(probe as u32),
            &widened,
            &format!("U32({probe}) must zero-extend onto U64"),
        );
        assert_same_digest(
            &DataInput::USIZE(probe as usize),
            &widened,
            &format!("USIZE({probe}) must zero-extend onto U64"),
        );
    }
}

#[test]
fn a_non_negative_value_hashes_the_same_whether_it_arrives_signed_or_unsigned() {
    for probe in [0u8, 1, 7, 100, 127] {
        assert_same_digest(
            &DataInput::I8(probe as i8),
            &DataInput::U8(probe),
            &format!("I8({probe}) and U8({probe}) are the same integer"),
        );
        assert_same_digest(
            &DataInput::I64(probe as i64),
            &DataInput::U64(probe as u64),
            &format!("I64({probe}) and U64({probe}) are the same integer"),
        );
    }
}

#[test]
fn a_negative_value_and_its_twos_complement_unsigned_reading_share_a_digest() {
    assert_same_digest(
        &DataInput::I8(-1),
        &DataInput::U64(u64::MAX),
        "sign extension makes I8(-1) the all-ones word",
    );
    assert_same_digest(
        &DataInput::I64(-1),
        &DataInput::U64(u64::MAX),
        "sign extension makes I64(-1) the all-ones word",
    );
    assert_different_digest(
        &DataInput::I8(-1),
        &DataInput::U8(255),
        "U8(255) zero-extends to 255, not to the all-ones word",
    );
}

#[test]
fn the_128_bit_integer_variants_hash_a_wider_word_than_the_64_bit_ones() {
    assert_same_digest(
        &DataInput::I128(7),
        &DataInput::U128(7),
        "a non-negative 128-bit value reads the same either way",
    );
    assert_different_digest(
        &DataInput::I128(7),
        &DataInput::I64(7),
        "a sixteen-byte encoding cannot equal an eight-byte one",
    );
    assert_same_digest(
        &DataInput::I128(-1),
        &DataInput::U128(u128::MAX),
        "sign extension makes I128(-1) the all-ones double word",
    );
}

#[test]
fn the_float_variants_hash_their_own_width_and_do_not_alias_the_integers() {
    assert_different_digest(
        &DataInput::F32(1.0),
        &DataInput::F64(1.0),
        "a four-byte float cannot hash as an eight-byte one",
    );
    assert_different_digest(
        &DataInput::F64(1.0),
        &DataInput::U64(1),
        "the bit pattern of 1.0f64 is not the integer 1",
    );
    assert_same_digest(
        &DataInput::F64(0.0),
        &DataInput::U64(0),
        "positive zero and the zero word share a bit pattern",
    );
    assert_different_digest(
        &DataInput::F64(0.0),
        &DataInput::F64(-0.0),
        "the two signed zeroes are distinct bit patterns",
    );
}

#[test]
fn the_string_and_byte_variants_share_one_digest_for_one_encoding() {
    assert_same_digest(
        &DataInput::Str("hello"),
        &DataInput::String("hello".to_string()),
        "borrowed and owned strings are the same bytes",
    );
    assert_same_digest(
        &DataInput::Str("hello"),
        &DataInput::Bytes(b"hello"),
        "a string hashes as its UTF-8 bytes",
    );
    assert_different_digest(
        &DataInput::Str(""),
        &DataInput::Str("\0"),
        "an empty string is not a NUL byte",
    );
}

#[test]
fn count_min_answers_every_variant_and_folds_the_aliasing_ones_together() {
    let mut regular = CountMin::<Vector2D<i64>, RegularPath>::with_dimensions(4, 4_096);
    let mut fast = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(4, 4_096);
    for (name, value) in every_variant() {
        for _ in 0..10 {
            regular.insert(&value);
            fast.insert(&value);
        }
        assert!(
            regular.estimate(&value) >= 10,
            "{name}: the regular path lost its own inserts"
        );
        assert!(
            fast.estimate(&value) >= 10,
            "{name}: the fast path lost its own inserts"
        );
    }

    let mut aliasing = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(4, 4_096);
    for _ in 0..5 {
        aliasing.insert(&DataInput::I8(7));
        aliasing.insert(&DataInput::U16(7));
        aliasing.insert(&DataInput::I64(7));
    }
    assert_eq!(
        aliasing.estimate(&DataInput::U64(7)),
        15,
        "three spellings of the integer 7 must land in one cell"
    );
    assert_eq!(
        aliasing.estimate(&DataInput::I128(7)),
        0,
        "the 128-bit spelling of 7 is a different key"
    );
}

#[test]
fn count_sketch_answers_every_variant() {
    let mut sketch = Count::<Vector2D<i64>, RegularPath>::with_dimensions(5, 4_096);
    for (_, value) in every_variant() {
        for _ in 0..200 {
            sketch.insert(&value);
        }
    }
    for (name, value) in every_variant() {
        let est = sketch.estimate(&value);
        assert!(
            (est - 200.0).abs() <= 60.0,
            "{name}: estimate {est} is nowhere near the 200 inserts it received"
        );
    }
}

#[test]
fn a_bloom_filter_answers_every_variant_without_a_false_negative() {
    let mut filter = Bloom::<RegularPath>::with_capacity(64, 0.01);
    let variants = every_variant();
    for (_, value) in &variants {
        filter.insert(value);
    }
    for (name, value) in &variants {
        assert!(
            filter.contains(value),
            "{name}: an inserted key was reported absent"
        );
    }

    let mut fast = Bloom::<FastPath>::with_capacity(64, 0.01);
    for (_, value) in &variants {
        fast.insert(value);
    }
    for (name, value) in &variants {
        assert!(
            fast.contains(value),
            "{name}: the fast path reported an inserted key absent"
        );
    }
}

#[test]
fn hyperloglog_counts_the_variants_as_distinct_identities() {
    let mut hll = HyperLogLog::<Classic>::new();
    let variants = every_variant();
    for (_, value) in &variants {
        hll.insert(value);
    }
    let after_first_pass = hll.estimate();
    for (_, value) in &variants {
        hll.insert(value);
    }
    assert_eq!(
        hll.estimate(),
        after_first_pass,
        "replaying the same variants must not move the estimate"
    );
    assert!(
        after_first_pass > 0,
        "a populated sketch must report a positive cardinality"
    );
}

#[test]
fn space_saving_tracks_every_variant() {
    let mut summary: SpaceSaving = SpaceSaving::with_capacity(64);
    for (_, value) in every_variant() {
        for _ in 0..12 {
            summary.insert(&value);
        }
    }
    for (name, value) in every_variant() {
        assert!(
            summary.estimate(&value) >= 12,
            "{name}: a monitored key must never read below its own count"
        );
    }
}

#[test]
fn the_numeric_variants_reach_the_quantile_sketches_and_the_others_are_refused() {
    let numeric: Vec<(&str, DataInput)> = vec![
        ("I8", DataInput::I8(1)),
        ("I16", DataInput::I16(2)),
        ("I32", DataInput::I32(3)),
        ("I64", DataInput::I64(4)),
        ("I128", DataInput::I128(5)),
        ("ISIZE", DataInput::ISIZE(6)),
        ("U8", DataInput::U8(7)),
        ("U16", DataInput::U16(8)),
        ("U32", DataInput::U32(9)),
        ("U64", DataInput::U64(10)),
        ("U128", DataInput::U128(11)),
        ("USIZE", DataInput::USIZE(12)),
        ("F32", DataInput::F32(13.0)),
        ("F64", DataInput::F64(14.0)),
    ];

    let mut kll: KLL<f64> = KLL::init_kll_with_seed(200, 0xDA7A_0001);
    let mut dds = DDSketch::new(0.01);
    for (name, value) in &numeric {
        kll.update_data_input(value)
            .unwrap_or_else(|e| panic!("{name}: KLL refused a numeric variant: {e}"));
        dds.add_input(value)
            .unwrap_or_else(|e| panic!("{name}: DDSketch refused a numeric variant: {e}"));
    }
    assert_eq!(
        kll.count(),
        numeric.len(),
        "every numeric variant must reach the KLL buffer"
    );
    assert_eq!(
        dds.get_count() as usize,
        numeric.len(),
        "every numeric variant must reach a DDSketch bucket"
    );
    assert_eq!(
        kll.quantile(0.0),
        1.0,
        "the smallest projected value must be the minimum"
    );
    assert_eq!(
        kll.quantile(1.0),
        14.0,
        "the largest projected value must be the maximum"
    );

    for (name, value) in [
        ("Str", DataInput::Str("x")),
        ("String", DataInput::String("x".to_string())),
        ("Bytes", DataInput::Bytes(b"x")),
    ] {
        assert!(
            kll.update_data_input(&value).is_err(),
            "{name}: KLL must refuse a non-numeric variant"
        );
        assert!(
            dds.add_input(&value).is_err(),
            "{name}: DDSketch must refuse a non-numeric variant"
        );
    }
    assert_eq!(
        kll.count(),
        numeric.len(),
        "a refused variant must not be counted"
    );
    assert_eq!(
        dds.get_count() as usize,
        numeric.len(),
        "a refused variant must not be counted"
    );
}
