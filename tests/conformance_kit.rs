//! Reference wiring for the conformance kit: established sketches run
//! through the same standard batteries every new sketch must pass. Copy an
//! adapter from here when onboarding a new sketch (see tests/README.md).

mod common;

use common::conformance::{
    self, CardinalityOps, CardinalitySpec, FrequencyOps, MembershipOps, MembershipSpec, MergeOps,
    QuantileOps, QuantileSpec, SignedFrequencyOps,
};
use common::{FreqTruth, zipf_u64};

use asap_sketchlib::message_pack_format::portable::ddsketch::DdSketch as PortableDds;
use asap_sketchlib::message_pack_format::portable::hll::{HllSketch, HllVariant};
use asap_sketchlib::{
    Bloom, CMSHeap, CSHeap, CountL2HH, CountMin, DDSketch, DataInput, FastPath, FoldCMS, FoldCS,
    HyperLogLog, HyperLogLogHIP, KLL, KLLDynamic, RegularPath, SpaceSaving, UnivMonQ, Vector2D,
};
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Adapters: one small impl block per sketch is all it takes to onboard.
// ---------------------------------------------------------------------------

struct CountMinAdapter(CountMin<Vector2D<i64>, FastPath>);

impl FrequencyOps<i64> for CountMinAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.estimate(&DataInput::I64(*key)) as f64
    }
}

impl MergeOps for CountMinAdapter {
    fn merge_from(&mut self, other: &Self) {
        self.0.merge(&other.0);
    }
}

struct BloomAdapter(Bloom<FastPath>);

impl MembershipOps<i64> for BloomAdapter {
    fn add(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn contains(&self, key: &i64) -> bool {
        self.0.contains(&DataInput::I64(*key))
    }
}

struct RegularBloomAdapter(Bloom<RegularPath>);

impl MembershipOps<i64> for RegularBloomAdapter {
    fn add(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn contains(&self, key: &i64) -> bool {
        self.0.contains(&DataInput::I64(*key))
    }
}

struct SpaceSavingAdapter(SpaceSaving);

impl FrequencyOps<i64> for SpaceSavingAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.estimate(&DataInput::I64(*key)) as f64
    }
}

struct CountSketchAdapter(asap_sketchlib::Count<Vector2D<i64>, RegularPath>);

impl FrequencyOps<i64> for CountSketchAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.estimate(&DataInput::I64(*key))
    }
}

impl SignedFrequencyOps<i64> for CountSketchAdapter {
    fn ingest_weighted(&mut self, key: &i64, weight: i64) {
        self.0.insert_many(&DataInput::I64(*key), weight);
    }
}

impl MergeOps for CountSketchAdapter {
    fn merge_from(&mut self, other: &Self) {
        self.0.merge(&other.0);
    }
}

struct HllClassicAdapter(HyperLogLog<asap_sketchlib::Classic>);

impl CardinalityOps for HllClassicAdapter {
    fn ingest(&mut self, key: &[u8]) {
        self.0
            .insert(&DataInput::U64(u64::from_be_bytes(key.try_into().unwrap())));
    }
    fn estimate(&self) -> f64 {
        self.0.estimate() as f64
    }
}

struct HllErtlAdapter(HyperLogLog<asap_sketchlib::ErtlMLE>);

impl CardinalityOps for HllErtlAdapter {
    fn ingest(&mut self, key: &[u8]) {
        self.0
            .insert(&DataInput::U64(u64::from_be_bytes(key.try_into().unwrap())));
    }
    fn estimate(&self) -> f64 {
        self.0.estimate() as f64
    }
}

struct HllHipAdapter(HyperLogLogHIP);

impl CardinalityOps for HllHipAdapter {
    fn ingest(&mut self, key: &[u8]) {
        self.0
            .insert(&DataInput::U64(u64::from_be_bytes(key.try_into().unwrap())));
    }
    fn estimate(&self) -> f64 {
        self.0.estimate() as f64
    }
}

struct PortableHllAdapter(HllSketch);

impl CardinalityOps for PortableHllAdapter {
    fn ingest(&mut self, key: &[u8]) {
        self.0.update(key);
    }
    fn estimate(&self) -> f64 {
        self.0.estimate()
    }
}

struct KllAdapter(KLL);

impl QuantileOps for KllAdapter {
    fn update(&mut self, value: f64) {
        KLL::update(&mut self.0, &value);
    }
    fn quantile(&self, q: f64) -> f64 {
        self.0.quantile(q)
    }
}

struct KllCachedAdapter(RefCell<KLL>);

impl QuantileOps for KllCachedAdapter {
    fn update(&mut self, value: f64) {
        self.0.borrow_mut().update(&value);
    }
    fn quantile(&self, q: f64) -> f64 {
        self.0.borrow_mut().quantile_cached(q)
    }
}

struct KllDynamicAdapter(KLLDynamic<f64>);

impl QuantileOps for KllDynamicAdapter {
    fn update(&mut self, value: f64) {
        KLLDynamic::update(&mut self.0, &value);
    }
    fn quantile(&self, q: f64) -> f64 {
        self.0.quantile(q)
    }
}

struct PortableDdsAdapter(PortableDds);

impl QuantileOps for PortableDdsAdapter {
    fn update(&mut self, value: f64) {
        self.0.update(value);
    }
    fn quantile(&self, q: f64) -> f64 {
        self.0.quantile(q).expect("non-empty sketch")
    }
}

struct UnivMonQAdapter(UnivMonQ);

impl QuantileOps for UnivMonQAdapter {
    fn update(&mut self, value: f64) {
        UnivMonQ::update(&mut self.0, &value);
    }
    fn quantile(&self, q: f64) -> f64 {
        self.0.quantile(q).expect("ordered samples enabled")
    }
}

// ---------------------------------------------------------------------------
// Battery runs
// ---------------------------------------------------------------------------

/// Space-Saving's over-estimate ceiling on `zipf_stream`. At 1024 counters the
/// summary's minimum count settles at 12, below the battery's dense-key
/// threshold of 25, so no dense key is ever evicted and the measured worst
/// over-estimate is 1.
const SPACE_SAVING_ABS_TOL: f64 = 2.0;

fn zipf_stream() -> Vec<i64> {
    zipf_u64(60_000, 2048, 1.1, 9001)
        .iter()
        .map(|v| *v as i64)
        .collect()
}

fn stream_truth(stream: &[i64]) -> FreqTruth {
    let mut truth = FreqTruth::default();
    for k in stream {
        truth.observe(*k);
    }
    truth
}

#[test]
fn countmin_passes_frequency_and_merge_conformance() {
    let stream = zipf_stream();
    let truth = stream_truth(&stream);
    let spec = conformance::FrequencySpec {
        one_sided: true,
        rel_tol: 0.01,
        abs_tol: 4.0,
    };

    conformance::frequency_battery(
        "CountMin<FastPath>",
        || {
            CountMinAdapter(CountMin::<Vector2D<i64>, FastPath>::with_dimensions(
                4, 4096,
            ))
        },
        &stream,
        &truth,
        spec,
    )
    .assert_ok();

    conformance::merge_equivalence_battery(
        "CountMin<FastPath>",
        || {
            CountMinAdapter(CountMin::<Vector2D<i64>, FastPath>::with_dimensions(
                4, 4096,
            ))
        },
        &stream,
        spec,
    )
    .assert_ok();
}

#[test]
fn countsketch_passes_signed_frequency_conformance() {
    let stream = zipf_stream();
    let truth = stream_truth(&stream);
    let spec = conformance::FrequencySpec {
        one_sided: false,
        rel_tol: 0.06,
        abs_tol: 25.0,
    };
    let make = || {
        CountSketchAdapter(
            asap_sketchlib::Count::<Vector2D<i64>, RegularPath>::with_dimensions(5, 4096),
        )
    };

    conformance::frequency_battery("Count<Regular>", make, &stream, &truth, spec).assert_ok();
    conformance::turnstile_battery("Count<Regular>", make, 42i64).assert_ok();
    conformance::merge_equivalence_battery("Count<Regular>", make, &stream, spec).assert_ok();
}

#[test]
fn hll_variants_pass_cardinality_conformance() {
    // Distinct u64 keys; encodings are byte-identical for every adapter.
    let unique: Vec<u64> = (0..100_000u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let spec = CardinalitySpec { rel_tol: 0.03 };

    conformance::cardinality_battery(
        "HyperLogLog<Classic>",
        || HllClassicAdapter(HyperLogLog::<asap_sketchlib::Classic>::new()),
        &unique,
        100_000,
        spec,
    )
    .assert_ok();

    conformance::cardinality_battery(
        "HyperLogLog<ErtlMLE>",
        || HllErtlAdapter(HyperLogLog::<asap_sketchlib::ErtlMLE>::new()),
        &unique,
        100_000,
        spec,
    )
    .assert_ok();

    conformance::cardinality_battery(
        "HyperLogLogHIP",
        || HllHipAdapter(HyperLogLogHIP::new()),
        &unique,
        100_000,
        spec,
    )
    .assert_ok();

    conformance::cardinality_battery(
        "portable HllSketch<p14>",
        || PortableHllAdapter(HllSketch::new(HllVariant::Regular, 14)),
        &unique,
        100_000,
        spec,
    )
    .assert_ok();
}

/// Compaction-coin seed for the KLL adapters. The unseeded `init_kll`
/// constructors draw from the wall clock, so a failure under them would not
/// reproduce.
const KIT_KLL_SEED: u64 = 0x4017_0001;

/// The KLL family through the **rank-error** battery.
///
/// DDSketch is deliberately absent: it promises relative *value* error and has
/// no rank guarantee, so it goes through `relative_quantile_battery` below.
#[test]
fn kll_family_passes_quantile_conformance() {
    let values: Vec<f64> = common::normal_f64(40_000, 500.0, 80.0, 7001)
        .into_iter()
        .filter(|v| *v > 0.0)
        .collect();

    conformance::quantile_battery(
        "KLL",
        || KllAdapter(KLL::init_kll_with_seed(200, KIT_KLL_SEED)),
        &values,
        QuantileSpec::default(),
    )
    .assert_ok();

    conformance::quantile_battery(
        "KLLDynamic",
        || KllDynamicAdapter(KLLDynamic::<f64>::init_kll_with_seed(200, KIT_KLL_SEED)),
        &values,
        QuantileSpec::default(),
    )
    .assert_ok();

    conformance::quantile_battery(
        "KLL-cached",
        || KllCachedAdapter(RefCell::new(KLL::init_kll_with_seed(200, KIT_KLL_SEED))),
        &values,
        QuantileSpec::default(),
    )
    .assert_ok();
}

/// DDSketch through the **relative-value-error** battery, against exact
/// nearest-rank order statistics — the guarantee it actually makes.
#[test]
fn ddsketch_passes_relative_quantile_conformance() {
    const ALPHA: f64 = 0.01;
    let values: Vec<f64> = common::normal_f64(40_000, 500.0, 80.0, 7001)
        .into_iter()
        .filter(|v| *v > 0.0)
        .collect();

    conformance::relative_quantile_battery(
        "PortableDds",
        || PortableDdsAdapter(PortableDds::new(ALPHA)),
        &values,
        common::specs::RelativeQuantileSpec::portable(ALPHA),
        &conformance::DEFAULT_QUANTILE_QS,
    )
    .assert_ok();
}

#[test]
fn univmonq_passes_quantile_conformance() {
    let values: Vec<f64> = common::uniform_u64(30_000, 50_000, 7002)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    conformance::quantile_battery(
        "UnivMonQ",
        || UnivMonQAdapter(UnivMonQ::new(Default::default()).expect("config")),
        &values,
        QuantileSpec {
            rank_tol: 0.04,
            ..QuantileSpec::default()
        },
    )
    .assert_ok();
}

/// A Bloom filter sized for 20k keys at 1% must show no false negative and a
/// measured rate that matches its own prediction. Sizing rounds each slice up
/// to a power of two, so the delivered rate sits well below the 1% target; the
/// target alone would pass an implementation off by 2x, so the battery is held
/// to `predicted_fpp` instead.
#[test]
fn bloom_passes_membership_conformance() {
    let members: Vec<i64> = (0..20_000).collect();
    let non_members: Vec<i64> = (1_000_000..1_100_000).collect();
    let predicted = Bloom::<FastPath>::with_capacity(20_000, 0.01).predicted_fpp(20_000);

    conformance::membership_battery(
        "Bloom<FastPath>",
        || BloomAdapter(Bloom::<FastPath>::with_capacity(20_000, 0.01)),
        &members,
        &non_members,
        MembershipSpec {
            max_fpp: 0.01,
            predicted_fpp: Some(predicted),
            fpp_band: 0.25,
        },
    )
    .assert_ok();
}

/// The regular path is sized by the same formula and must land in the same
/// place. It is the default type parameter and the shape `predicted_fpp`
/// models, so the kit covers it rather than only the packed-hash path.
#[test]
fn bloom_regular_path_passes_membership_conformance() {
    let members: Vec<i64> = (0..20_000).collect();
    let non_members: Vec<i64> = (1_000_000..1_100_000).collect();
    let predicted = Bloom::<RegularPath>::with_capacity(20_000, 0.01).predicted_fpp(20_000);

    conformance::membership_battery(
        "Bloom<RegularPath>",
        || RegularBloomAdapter(Bloom::<RegularPath>::with_capacity(20_000, 0.01)),
        &members,
        &non_members,
        MembershipSpec {
            max_fpp: 0.01,
            predicted_fpp: Some(predicted),
            fpp_band: 0.25,
        },
    )
    .assert_ok();
}

/// Space-Saving over 1024 counters holds every dense key of the shared Zipf
/// stream, so each reports at or above its true count. The stream carries 1996
/// distinct keys, so eviction is live rather than the summary simply fitting
/// the domain.
///
/// `merge_equivalence_battery` does not fit: merging two summaries cannot
/// recover a key both sides evicted, so the union is not the summary the
/// concatenated stream would have built.
#[test]
fn space_saving_passes_frequency_conformance() {
    let stream = zipf_stream();
    let truth = stream_truth(&stream);
    let spec = conformance::FrequencySpec {
        one_sided: true,
        rel_tol: 0.0,
        abs_tol: SPACE_SAVING_ABS_TOL,
    };

    conformance::frequency_battery(
        "SpaceSaving",
        || SpaceSavingAdapter(SpaceSaving::with_capacity(1024)),
        &stream,
        &truth,
        spec,
    )
    .assert_ok();
}

struct CoreDdSketchAdapter(DDSketch);

impl QuantileOps for CoreDdSketchAdapter {
    fn update(&mut self, value: f64) {
        self.0.add(&value);
    }
    fn quantile(&self, q: f64) -> f64 {
        self.0.get_value_at_quantile(q).unwrap_or(f64::NAN)
    }
}

struct CmsHeapAdapter(CMSHeap<Vector2D<i64>, FastPath>);

impl FrequencyOps<i64> for CmsHeapAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.estimate(&DataInput::I64(*key)) as f64
    }
}

impl MergeOps for CmsHeapAdapter {
    fn merge_from(&mut self, other: &Self) {
        self.0.merge(&other.0);
    }
}

struct CsHeapAdapter(CSHeap<Vector2D<i64>, RegularPath>);

impl FrequencyOps<i64> for CsHeapAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.estimate(&DataInput::I64(*key))
    }
}

impl MergeOps for CsHeapAdapter {
    fn merge_from(&mut self, other: &Self) {
        self.0.merge(&other.0);
    }
}

struct CountL2hhAdapter(CountL2HH);

impl FrequencyOps<i64> for CountL2hhAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.fast_insert_with_count(&DataInput::I64(*key), 1);
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.fast_get_est(&DataInput::I64(*key))
    }
}

impl SignedFrequencyOps<i64> for CountL2hhAdapter {
    fn ingest_weighted(&mut self, key: &i64, weight: i64) {
        self.0.fast_insert_with_count(&DataInput::I64(*key), weight);
    }
}

impl MergeOps for CountL2hhAdapter {
    fn merge_from(&mut self, other: &Self) {
        self.0.merge(&other.0);
    }
}

struct FoldCmsAdapter(FoldCMS);

impl FrequencyOps<i64> for FoldCmsAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert_one(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.query(&DataInput::I64(*key)) as f64
    }
}

impl MergeOps for FoldCmsAdapter {
    fn merge_from(&mut self, other: &Self) {
        self.0.merge_same_level(&other.0);
    }
}

struct FoldCsAdapter(FoldCS);

impl FrequencyOps<i64> for FoldCsAdapter {
    fn ingest(&mut self, key: &i64) {
        self.0.insert_one(&DataInput::I64(*key));
    }
    fn estimate(&self, key: &i64) -> f64 {
        self.0.query(&DataInput::I64(*key)) as f64
    }
}

impl SignedFrequencyOps<i64> for FoldCsAdapter {
    fn ingest_weighted(&mut self, key: &i64, weight: i64) {
        self.0.insert(&DataInput::I64(*key), weight);
    }
}

impl MergeOps for FoldCsAdapter {
    fn merge_from(&mut self, other: &Self) {
        self.0.merge_same_level(&other.0);
    }
}

#[cfg(feature = "experimental")]
struct KmvAdapter(RefCell<asap_sketchlib::KMV>);

#[cfg(feature = "experimental")]
impl CardinalityOps for KmvAdapter {
    fn ingest(&mut self, key: &[u8]) {
        self.0.borrow_mut().insert(&DataInput::Bytes(key));
    }
    fn estimate(&self) -> f64 {
        self.0.borrow_mut().estimate()
    }
}

#[cfg(feature = "experimental")]
struct UniformSamplingAdapter(asap_sketchlib::UniformSampling);

#[cfg(feature = "experimental")]
impl QuantileOps for UniformSamplingAdapter {
    fn update(&mut self, value: f64) {
        self.0.update(value);
    }
    fn quantile(&self, q: f64) -> f64 {
        let mut retained = self.0.samples();
        if retained.is_empty() {
            return f64::NAN;
        }
        retained.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in the fixture"));
        let idx =
            ((q.clamp(0.0, 1.0) * retained.len() as f64).ceil() as usize).clamp(1, retained.len());
        retained[idx - 1]
    }
}

fn relative_quantile_values() -> Vec<f64> {
    let mut values: Vec<f64> = zipf_u64(20_000, 4_096, 1.1, 7_701)
        .into_iter()
        .map(|v| 1_000.0 + v as f64)
        .collect();
    values.extend(
        zipf_u64(10_000, 512, 1.3, 7_702)
            .into_iter()
            .map(|v| 1e6 + v as f64 * 1e3),
    );
    values
}

#[test]
fn core_ddsketch_passes_relative_quantile_conformance() {
    let values = relative_quantile_values();
    for alpha in [0.001f64, 0.01, 0.05] {
        conformance::relative_quantile_battery(
            "DDSketch",
            || CoreDdSketchAdapter(DDSketch::new(alpha)),
            &values,
            common::specs::RelativeQuantileSpec::core(alpha),
            &conformance::DEFAULT_QUANTILE_QS,
        )
        .assert_ok();
    }
}

#[test]
fn heap_backed_frequency_sketches_pass_frequency_and_merge_conformance() {
    let stream = zipf_stream();
    let truth = stream_truth(&stream);

    let one_sided = conformance::FrequencySpec {
        one_sided: true,
        rel_tol: 0.01,
        abs_tol: 8.0,
    };
    let make_cms = || CmsHeapAdapter(CMSHeap::<Vector2D<i64>, FastPath>::new(4, 4_096, 64));
    conformance::frequency_battery("CMSHeap<FastPath>", make_cms, &stream, &truth, one_sided)
        .assert_ok();
    conformance::merge_equivalence_battery("CMSHeap<FastPath>", make_cms, &stream, one_sided)
        .assert_ok();

    let two_sided = conformance::FrequencySpec {
        one_sided: false,
        rel_tol: 0.06,
        abs_tol: 25.0,
    };
    let make_cs = || CsHeapAdapter(CSHeap::<Vector2D<i64>, RegularPath>::new(5, 4_096, 64));
    conformance::frequency_battery("CSHeap<Regular>", make_cs, &stream, &truth, two_sided)
        .assert_ok();
    conformance::merge_equivalence_battery("CSHeap<Regular>", make_cs, &stream, two_sided)
        .assert_ok();
}

#[test]
fn count_l2hh_passes_signed_frequency_and_merge_conformance() {
    let stream = zipf_stream();
    let truth = stream_truth(&stream);
    let spec = conformance::FrequencySpec {
        one_sided: false,
        rel_tol: 0.06,
        abs_tol: 25.0,
    };
    let make = || CountL2hhAdapter(CountL2HH::with_dimensions(5, 4_096));

    conformance::frequency_battery("CountL2HH", make, &stream, &truth, spec).assert_ok();
    conformance::turnstile_battery("CountL2HH", make, 42i64).assert_ok();
    conformance::merge_equivalence_battery("CountL2HH", make, &stream, spec).assert_ok();
}

#[test]
fn folded_frequency_sketches_pass_frequency_and_merge_conformance() {
    let stream = zipf_stream();
    let truth = stream_truth(&stream);

    let one_sided = conformance::FrequencySpec {
        one_sided: true,
        rel_tol: 0.02,
        abs_tol: 12.0,
    };
    let make_cms = || FoldCmsAdapter(FoldCMS::new(4, 8_192, 1, 64));
    conformance::frequency_battery("FoldCMS", make_cms, &stream, &truth, one_sided).assert_ok();
    conformance::merge_equivalence_battery("FoldCMS", make_cms, &stream, one_sided).assert_ok();

    let two_sided = conformance::FrequencySpec {
        one_sided: false,
        rel_tol: 0.08,
        abs_tol: 30.0,
    };
    let make_cs = || FoldCsAdapter(FoldCS::new(5, 8_192, 1, 64));
    conformance::frequency_battery("FoldCS", make_cs, &stream, &truth, two_sided).assert_ok();
    conformance::turnstile_battery("FoldCS", make_cs, 42i64).assert_ok();
    conformance::merge_equivalence_battery("FoldCS", make_cs, &stream, two_sided).assert_ok();
}

#[cfg(feature = "experimental")]
#[test]
fn kmv_passes_cardinality_conformance() {
    let unique: Vec<u64> = (0..100_000u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let spec = CardinalitySpec { rel_tol: 0.08 };

    conformance::cardinality_battery(
        "KMV",
        || KmvAdapter(RefCell::new(asap_sketchlib::KMV::new(4_096))),
        &unique,
        100_000,
        spec,
    )
    .assert_ok();
}

#[cfg(feature = "experimental")]
#[test]
fn uniform_sampling_at_full_rate_passes_quantile_conformance() {
    let values: Vec<f64> = zipf_u64(20_000, 4_096, 1.1, 7_703)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    let spec = QuantileSpec {
        rank_tol: 0.0,
        ..Default::default()
    };

    conformance::quantile_battery(
        "UniformSampling(rate=1)",
        || UniformSamplingAdapter(asap_sketchlib::UniformSampling::with_seed(1.0, 7_704)),
        &values,
        spec,
    )
    .assert_ok();
}
