mod common;

use common::{uniform_u64, zipf_u64};

use asap_sketchlib::input::{HydraCounter, HydraQuery};
use asap_sketchlib::{
    Bloom, CMSHeap, CSHeap, Coco, Count, CountMin, DDSketch, DataInput, DefaultXxHasher,
    EHSketchList, Elastic, ErtlMLE, ExponentialHistogram, FastPath, Hydra, HyperLogLog, KLL,
    KLLDynamic, RegularPath, SpaceSaving, UnivMon, UnivMonPyramid, UnivMonQ, Vector2D,
};

fn assert_kind(bytes: &[u8], family: u8, variant: u8, label: &str) {
    assert!(
        bytes.len() > 9,
        "{label}: envelope is too short to carry a kind_id"
    );
    assert_eq!(bytes[7], 2, "{label}: kind_id_len");
    assert_eq!(
        (bytes[8], bytes[9]),
        (family, variant),
        "{label}: kind_id bytes"
    );
}

fn frequency_stream() -> Vec<u64> {
    zipf_u64(20_000, 1_024, 1.1, 0x_C1EA_0001)
}

#[test]
fn a_ddsketch_envelope_round_trips_and_names_its_kind() {
    let mut sketch = DDSketch::new(0.01);
    for v in uniform_u64(20_000, 1_000_000, 0x_1DE7_0001) {
        sketch.add(&(1.0 + v as f64));
    }
    let bytes = sketch.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x05, 0x00, "DDSketch");
    let decoded = DDSketch::deserialize_from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.get_count(), sketch.get_count(), "count");
    assert_eq!(decoded.alpha(), sketch.alpha(), "alpha");
    for q in [0.0f64, 0.1, 0.5, 0.9, 1.0] {
        assert_eq!(
            decoded.get_value_at_quantile(q),
            sketch.get_value_at_quantile(q),
            "q={q}"
        );
    }
}

#[test]
fn a_bloom_envelope_round_trips_on_both_paths_and_names_its_kind() {
    let mut regular = Bloom::<RegularPath>::with_capacity(2_000, 0.01);
    let mut fast = Bloom::<FastPath>::with_capacity(2_000, 0.01);
    for i in 0..2_000i64 {
        regular.insert(&DataInput::I64(i));
        fast.insert(&DataInput::I64(i));
    }

    let regular_bytes = regular.serialize_to_bytes().expect("encode regular");
    assert_kind(&regular_bytes, 0x17, 0x00, "Bloom<RegularPath>");
    let decoded_regular =
        Bloom::<RegularPath>::deserialize_from_bytes(&regular_bytes).expect("decode regular");

    let fast_bytes = fast.serialize_to_bytes().expect("encode fast");
    assert_kind(&fast_bytes, 0x17, 0x00, "Bloom<FastPath>");
    let decoded_fast = Bloom::<FastPath>::deserialize_from_bytes(&fast_bytes).expect("decode fast");

    for i in 0..2_000i64 {
        assert!(
            decoded_regular.contains(&DataInput::I64(i)),
            "regular path lost member {i} across the wire"
        );
        assert!(
            decoded_fast.contains(&DataInput::I64(i)),
            "fast path lost member {i} across the wire"
        );
    }
    for i in 10_000..12_000i64 {
        assert_eq!(
            decoded_regular.contains(&DataInput::I64(i)),
            regular.contains(&DataInput::I64(i)),
            "regular path changed its answer for a non-member {i}"
        );
    }
    assert_eq!(
        decoded_regular.inserted(),
        regular.inserted(),
        "insert count"
    );
}

#[test]
fn a_coco_envelope_round_trips_and_names_its_kind() {
    let mut coco = Coco::<DefaultXxHasher>::init_with_size(512, 4);
    for k in frequency_stream() {
        coco.insert(&format!("flow-{k}"), 1);
    }
    let bytes = coco.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x0c, 0x00, "Coco");
    let decoded = Coco::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
    for k in 0..1_024u64 {
        let key = format!("flow-{k}");
        assert_eq!(
            decoded.estimate_key(&key),
            coco.estimate_key(&key),
            "flow {k} changed across the wire"
        );
    }
}

#[test]
fn an_elastic_envelope_round_trips_and_names_its_kind() {
    let mut elastic = Elastic::<DefaultXxHasher>::init_with_dimensions(64, 3, 1_024);
    for k in frequency_stream() {
        elastic.insert(format!("flow-{k}"));
    }
    let bytes = elastic.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x0b, 0x00, "Elastic");
    let decoded = Elastic::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
    for k in 0..1_024u64 {
        let key = format!("flow-{k}");
        assert_eq!(
            decoded.query(key.clone()),
            elastic.query(key),
            "flow {k} changed across the wire"
        );
    }
}

#[test]
fn a_space_saving_envelope_round_trips_and_names_its_kind() {
    let mut summary = SpaceSaving::<DefaultXxHasher>::with_capacity(256);
    for k in frequency_stream() {
        summary.insert(&DataInput::U64(k));
    }
    let bytes = summary.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x18, 0x00, "SpaceSaving");
    let decoded = SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&bytes).expect("decode");
    for k in 0..1_024u64 {
        assert_eq!(
            decoded.estimate(&DataInput::U64(k)),
            summary.estimate(&DataInput::U64(k)),
            "key {k} changed across the wire"
        );
    }
}

#[test]
fn a_kll_dynamic_envelope_round_trips_and_names_its_kind() {
    let mut sketch = KLLDynamic::<f64>::init_kll_with_seed(200, 0x5EED_9001);
    for v in uniform_u64(20_000, 1_000_000, 0x_C0DE_0001) {
        sketch.update(&(v as f64));
    }
    let bytes = sketch.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x06, 0x01, "KLLDynamic");
    let decoded = KLLDynamic::<f64>::deserialize_from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.count(), sketch.count(), "retained mass");
    for q in [0.0f64, 0.1, 0.5, 0.9, 1.0] {
        assert_eq!(
            decoded.quantile(q).to_bits(),
            sketch.quantile(q).to_bits(),
            "q={q}"
        );
    }
    assert_eq!(
        decoded.serialize_to_bytes().expect("re-encode"),
        bytes,
        "a decoded sketch must re-encode byte-identically"
    );
}

#[test]
fn the_heap_backed_matrix_envelopes_round_trip_and_name_their_kinds() {
    let stream = frequency_stream();

    let mut cms = CMSHeap::<Vector2D<i64>, FastPath>::new(4, 2_048, 32);
    let mut cs = CSHeap::<Vector2D<i64>, RegularPath>::new(5, 2_048, 32);
    for k in &stream {
        cms.insert(&DataInput::U64(*k));
        cs.insert(&DataInput::U64(*k));
    }

    let cms_bytes = cms.serialize_to_bytes().expect("encode CMSHeap");
    assert_kind(&cms_bytes, 0x03, 0x00, "CMSHeap");
    let decoded_cms =
        CMSHeap::<Vector2D<i64>, FastPath>::deserialize_from_bytes(&cms_bytes).expect("decode");

    let cs_bytes = cs.serialize_to_bytes().expect("encode CSHeap");
    assert_kind(&cs_bytes, 0x0a, 0x00, "CSHeap");
    let decoded_cs =
        CSHeap::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&cs_bytes).expect("decode");

    for k in 0..1_024u64 {
        assert_eq!(
            decoded_cms.estimate(&DataInput::U64(k)),
            cms.estimate(&DataInput::U64(k)),
            "CMSHeap key {k} changed across the wire"
        );
        assert_eq!(
            decoded_cs.estimate(&DataInput::U64(k)),
            cs.estimate(&DataInput::U64(k)),
            "CSHeap key {k} changed across the wire"
        );
    }
    assert_eq!(
        decoded_cms.heap().len(),
        cms.heap().len(),
        "CMSHeap heap size changed across the wire"
    );
    assert_eq!(
        decoded_cs.heap().len(),
        cs.heap().len(),
        "CSHeap heap size changed across the wire"
    );
}

fn eh_prototype() -> EHSketchList {
    EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 512))
}

#[test]
fn an_exponential_histogram_envelope_round_trips_and_names_its_kind() {
    let mut eh = ExponentialHistogram::new(8, 1_000_000, eh_prototype());
    for (t, k) in frequency_stream().iter().take(3_000).enumerate() {
        eh.update(t as u64, &DataInput::U64(*k));
    }
    let bytes = eh.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x13, 0x00, "ExponentialHistogram");
    let decoded = ExponentialHistogram::deserialize_from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.bucket_count(), eh.bucket_count(), "bucket count");
    assert_eq!(decoded.get_min_time(), eh.get_min_time(), "min time");
    assert_eq!(decoded.get_max_time(), eh.get_max_time(), "max time");

    let (lo, hi) = (
        eh.get_min_time().expect("populated"),
        eh.get_max_time().expect("populated"),
    );
    let before = eh.query_interval_merge(lo, hi).expect("merge");
    let after = decoded.query_interval_merge(lo, hi).expect("merge");
    for k in 0..256u64 {
        assert_eq!(
            after.query(&DataInput::U64(k)).expect("answer"),
            before.query(&DataInput::U64(k)).expect("answer"),
            "key {k} changed across the wire"
        );
    }
}

#[test]
fn an_eh_sketch_list_envelope_round_trips_and_names_its_kind() {
    let mut payload = eh_prototype();
    for k in frequency_stream().iter().take(5_000) {
        payload.insert(&DataInput::U64(*k));
    }
    let bytes = payload.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x14, 0x00, "EHSketchList");
    let decoded = EHSketchList::deserialize_from_bytes(&bytes).expect("decode");
    assert_eq!(
        decoded.sketch_type(),
        payload.sketch_type(),
        "the decoded payload must name the same sketch"
    );
    for k in 0..512u64 {
        assert_eq!(
            decoded.query(&DataInput::U64(k)).expect("answer"),
            payload.query(&DataInput::U64(k)).expect("answer"),
            "key {k} changed across the wire"
        );
    }
}

#[test]
fn a_hydra_envelope_round_trips_and_names_the_counter_it_carries() {
    let mut hydra = Hydra::with_schema(
        4,
        512,
        ["region", "user"],
        HydraCounter::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(3, 512)),
    )
    .expect("schema");
    for r in ["eu", "us", "apac"] {
        for u in ["alice", "bob", "carol"] {
            for _ in 0..40 {
                hydra
                    .update(&[r, u], &DataInput::Str("event"), None)
                    .expect("update");
            }
        }
    }
    let bytes = hydra.serialize_to_bytes().expect("encode");
    assert_kind(&bytes, 0x07, 0x01, "Hydra<CountMin>");
    let decoded = Hydra::deserialize_from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.schema(), hydra.schema(), "schema");
    for r in ["eu", "us", "apac"] {
        for u in ["alice", "bob", "carol"] {
            let key = [Some(r), Some(u)];
            assert_eq!(
                decoded
                    .query_key(&key, &HydraQuery::Frequency(DataInput::Str("event")))
                    .expect("query"),
                hydra
                    .query_key(&key, &HydraQuery::Frequency(DataInput::Str("event")))
                    .expect("query"),
                "cell {r}/{u} changed across the wire"
            );
        }
    }
}

#[test]
fn the_univmon_family_envelopes_round_trip_and_name_their_kinds() {
    let stream = frequency_stream();

    let mut um = UnivMon::init_univmon(32, 5, 512, 8);
    let mut pyramid = UnivMonPyramid::with_defaults();
    let mut q = UnivMonQ::new(Default::default()).expect("default config valid");
    for k in &stream {
        um.insert(&DataInput::U32(*k as u32), 1);
        pyramid.insert(&DataInput::U32(*k as u32), 1);
        q.update(&(*k as f64));
    }

    let um_bytes = um.serialize_to_bytes().expect("encode UnivMon");
    assert_kind(&um_bytes, 0x10, 0x00, "UnivMon");
    let decoded_um = UnivMon::deserialize_from_bytes(&um_bytes).expect("decode UnivMon");
    assert_eq!(decoded_um.calc_l1(), um.calc_l1(), "UnivMon L1");
    assert_eq!(decoded_um.calc_l2(), um.calc_l2(), "UnivMon L2");
    assert_eq!(
        decoded_um.calc_card(),
        um.calc_card(),
        "UnivMon cardinality"
    );

    let pyramid_bytes = pyramid.serialize_to_bytes().expect("encode UnivMonPyramid");
    assert_kind(&pyramid_bytes, 0x11, 0x00, "UnivMonPyramid");
    let decoded_pyramid =
        UnivMonPyramid::deserialize_from_bytes(&pyramid_bytes).expect("decode UnivMonPyramid");
    assert_eq!(
        decoded_pyramid.calc_l1(),
        pyramid.calc_l1(),
        "UnivMonPyramid L1"
    );
    assert_eq!(
        decoded_pyramid.calc_l2(),
        pyramid.calc_l2(),
        "UnivMonPyramid L2"
    );

    let q_bytes = q.serialize_to_bytes().expect("encode UnivMonQ");
    assert_kind(&q_bytes, 0x1a, 0x00, "UnivMonQ");
    let decoded_q =
        UnivMonQ::<DefaultXxHasher>::deserialize_from_bytes(&q_bytes).expect("decode UnivMonQ");
    assert_eq!(decoded_q.count(), q.count(), "UnivMonQ count");
    assert_eq!(decoded_q.estimate_f2(), q.estimate_f2(), "UnivMonQ F2");
    for probe in [0.1f64, 0.5, 0.9] {
        assert_eq!(
            decoded_q.quantile(probe),
            q.quantile(probe),
            "UnivMonQ quantile q={probe}"
        );
    }
}

#[test]
fn an_envelope_is_refused_by_every_decoder_but_its_own() {
    let mut dds = DDSketch::new(0.01);
    for v in 1..1_000u64 {
        dds.add(&(v as f64));
    }
    let dds_bytes = dds.serialize_to_bytes().expect("encode");

    let mut bloom = Bloom::<RegularPath>::with_capacity(100, 0.01);
    bloom.insert(&DataInput::I64(1));
    let bloom_bytes = bloom.serialize_to_bytes().expect("encode");

    assert!(
        Bloom::<RegularPath>::deserialize_from_bytes(&dds_bytes).is_err(),
        "a DDSketch envelope must not decode as a Bloom filter"
    );
    assert!(
        DDSketch::deserialize_from_bytes(&bloom_bytes).is_err(),
        "a Bloom envelope must not decode as a DDSketch"
    );
    assert!(
        SpaceSaving::<DefaultXxHasher>::deserialize_from_bytes(&dds_bytes).is_err(),
        "a DDSketch envelope must not decode as a Space-Saving summary"
    );
    assert!(
        DDSketch::deserialize_from_bytes(&[]).is_err(),
        "an empty buffer is not an envelope"
    );
    assert!(
        DDSketch::deserialize_from_bytes(&dds_bytes[..dds_bytes.len() / 2]).is_err(),
        "a truncated envelope must be refused"
    );
}

#[test]
fn the_matrix_and_quantile_envelopes_carry_their_answers_unchanged() {
    let stream = frequency_stream();
    let mut cm = CountMin::<Vector2D<i64>, FastPath>::with_dimensions(4, 2_048);
    let mut cs = Count::<Vector2D<i64>, RegularPath>::with_dimensions(5, 2_048);
    let mut kll: KLL<f64> = KLL::init_kll_with_seed(200, 0x5EED_9002);
    let mut hll = HyperLogLog::<ErtlMLE>::new();
    for k in &stream {
        cm.insert(&DataInput::U64(*k));
        cs.insert(&DataInput::U64(*k));
        kll.update(&(*k as f64));
        hll.insert(&DataInput::U64(*k));
    }

    let cm_bytes = cm.serialize_to_bytes().expect("encode CountMin");
    assert_kind(&cm_bytes, 0x02, 0x00, "CountMin");
    let cs_bytes = cs.serialize_to_bytes().expect("encode Count");
    assert_kind(&cs_bytes, 0x04, 0x00, "Count");
    let kll_bytes = kll.serialize_to_bytes().expect("encode KLL");
    assert_kind(&kll_bytes, 0x06, 0x00, "KLL");
    let hll_bytes = hll.serialize_to_bytes().expect("encode HyperLogLog");
    assert_kind(&hll_bytes, 0x01, 0x02, "HyperLogLog");

    let decoded_cm =
        CountMin::<Vector2D<i64>, FastPath>::deserialize_from_bytes(&cm_bytes).expect("decode");
    let decoded_cs =
        Count::<Vector2D<i64>, RegularPath>::deserialize_from_bytes(&cs_bytes).expect("decode");
    let decoded_kll = KLL::<f64>::deserialize_from_bytes(&kll_bytes).expect("decode");
    let decoded_hll = HyperLogLog::<ErtlMLE>::deserialize_from_bytes(&hll_bytes).expect("decode");

    for k in 0..1_024u64 {
        assert_eq!(
            decoded_cm.estimate(&DataInput::U64(k)),
            cm.estimate(&DataInput::U64(k)),
            "CountMin key {k}"
        );
        assert_eq!(
            decoded_cs.estimate(&DataInput::U64(k)),
            cs.estimate(&DataInput::U64(k)),
            "Count key {k}"
        );
    }
    for q in [0.0f64, 0.5, 1.0] {
        assert_eq!(
            decoded_kll.quantile(q).to_bits(),
            kll.quantile(q).to_bits(),
            "KLL q={q}"
        );
    }
    assert_eq!(
        decoded_hll.estimate(),
        hll.estimate(),
        "HyperLogLog estimate"
    );
}

#[cfg(feature = "experimental")]
#[test]
fn the_experimental_envelopes_round_trip_and_name_their_kinds() {
    use asap_sketchlib::{KMV, UniformSampling};

    let mut kmv = KMV::<DefaultXxHasher>::new(1_024);
    for k in uniform_u64(50_000, 10_000_000, 0x_5EED_1001) {
        kmv.insert(&DataInput::U64(k));
    }
    let kmv_bytes = kmv.serialize_to_bytes().expect("encode KMV");
    assert_kind(&kmv_bytes, 0x0e, 0x00, "KMV");
    let mut decoded_kmv =
        KMV::<DefaultXxHasher>::deserialize_from_bytes(&kmv_bytes).expect("decode KMV");
    assert_eq!(decoded_kmv.estimate(), kmv.estimate(), "KMV estimate");

    let mut sampler = UniformSampling::with_seed(0.2, 0x_5EED_1002);
    for v in uniform_u64(20_000, 1_000_000, 0x_5EED_1003) {
        sampler.update(v as f64);
    }
    let sampler_bytes = sampler
        .serialize_to_bytes()
        .expect("encode UniformSampling");
    assert_kind(&sampler_bytes, 0x0d, 0x00, "UniformSampling");
    let decoded_sampler =
        UniformSampling::deserialize_from_bytes(&sampler_bytes).expect("decode UniformSampling");
    assert_eq!(decoded_sampler.len(), sampler.len(), "retained count");
    assert_eq!(
        decoded_sampler.total_seen(),
        sampler.total_seen(),
        "total seen"
    );
    assert_eq!(
        decoded_sampler.serialize_to_bytes().expect("re-encode"),
        sampler_bytes,
        "a decoded sampler must re-encode byte-identically"
    );
}
