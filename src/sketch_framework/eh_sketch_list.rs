//! Sketch enums used by the exponential histogram framework.

use crate::{DataInput, Vector2D};
use serde::{Deserialize, Serialize};

use super::super::sketches::*;
use super::UnivMon;
use crate::sketches::countsketch_topk::CountL2HH;

pub(crate) mod wire;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Norm choices used by EH merge policies.
pub enum SketchNorm {
    /// L1-based merge policy.
    L1,
    /// L2-based merge policy.
    L2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Erased sketch variants supported by exponential histograms.
pub enum EHSketchList {
    /// Count-Min sketch.
    CM(CountMin<Vector2D<i32>, FastPath>),
    /// CocoSketch.
    COCO(Coco),
    /// CountL2HH sketch.
    COUNTL2HH(CountL2HH),
    /// Count Sketch.
    CS(Count<Vector2D<i32>, FastPath>),
    /// DDSketch.
    DDS(DDSketch),
    /// Elastic sketch.
    ELASTIC(Elastic),
    /// HyperLogLog.
    HLL(HyperLogLog<ErtlMLE>),
    /// KLL sketch.
    KLL(KLL),
    #[cfg(feature = "experimental")]
    UNIFORM(UniformSampling),
    // LOCHER(LocherSketch),
    /// UnivMon sketch.
    UNIVMON(UnivMon),
}

impl EHSketchList {
    /// Returns whether this sketch supports the requested merge norm.
    pub fn supports_norm(&self, norm: SketchNorm) -> bool {
        match self {
            EHSketchList::COUNTL2HH(_) | EHSketchList::UNIVMON(_) => norm == SketchNorm::L2,
            EHSketchList::CM(_)
            | EHSketchList::COCO(_)
            | EHSketchList::CS(_)
            | EHSketchList::DDS(_)
            | EHSketchList::ELASTIC(_)
            | EHSketchList::HLL(_)
            | EHSketchList::KLL(_) => norm == SketchNorm::L1,
            #[cfg(feature = "experimental")]
            EHSketchList::UNIFORM(_) => norm == SketchNorm::L1,
        }
    }

    pub(crate) fn eh_l2_mass(&self) -> Option<f64> {
        match self {
            EHSketchList::COUNTL2HH(sketch) => Some(sketch.get_l2_sqr()),
            EHSketchList::UNIVMON(sketch) => {
                let l2 = sketch.calc_l2();
                Some(l2 * l2)
            }
            _ => None,
        }
    }

    /// Insert a value into the sketch
    pub fn insert(&mut self, val: &DataInput) {
        match self {
            EHSketchList::CM(sketch) => sketch.insert(val),
            EHSketchList::COCO(sketch) => match val {
                DataInput::Str(s) => sketch.insert(s, 1),
                DataInput::String(s) => sketch.insert(s.as_str(), 1),
                _ => {}
            },
            EHSketchList::COUNTL2HH(sketch) => sketch.fast_insert_with_count(val, 1),
            EHSketchList::CS(sketch) => sketch.insert(val),
            EHSketchList::DDS(sketch) => {
                let _ = sketch.add_input(val);
            }
            EHSketchList::ELASTIC(sketch) => match val {
                DataInput::String(s) => sketch.insert(s.to_string()),
                DataInput::I32(i) => sketch.insert(i.to_string()),
                DataInput::I64(i) => sketch.insert(i.to_string()),
                DataInput::U32(u) => sketch.insert(u.to_string()),
                DataInput::U64(u) => sketch.insert(u.to_string()),
                DataInput::F32(f) => sketch.insert(f.to_string()),
                DataInput::F64(f) => sketch.insert(f.to_string()),
                DataInput::Str(s) => sketch.insert(s.to_string()),
                DataInput::Bytes(items) => {
                    let s = String::from_utf8_lossy(items).to_string();
                    sketch.insert(s)
                }
                DataInput::I8(i) => sketch.insert(i.to_string()),
                DataInput::I16(i) => sketch.insert(i.to_string()),
                DataInput::I128(i) => sketch.insert(i.to_string()),
                DataInput::ISIZE(i) => sketch.insert(i.to_string()),
                DataInput::U8(u) => sketch.insert(u.to_string()),
                DataInput::U16(u) => sketch.insert(u.to_string()),
                DataInput::U128(u) => sketch.insert(u.to_string()),
                DataInput::USIZE(u) => sketch.insert(u.to_string()),
            },
            EHSketchList::HLL(sketch) => sketch.insert(val),
            EHSketchList::KLL(sketch) => {
                let _ = sketch.update_data_input(val);
            }
            #[cfg(feature = "experimental")]
            EHSketchList::UNIFORM(sketch) => {
                let _ = sketch.update_input(val);
            }
            EHSketchList::UNIVMON(sketch) => sketch.insert(val, 1),
        }
    }

    /// Merge another sketch of the same type into this one
    pub fn merge(&mut self, other: &EHSketchList) -> Result<(), &'static str> {
        match (self, other) {
            (EHSketchList::CM(s), EHSketchList::CM(o)) => {
                s.merge(o);
                Ok(())
            }
            (EHSketchList::COCO(s), EHSketchList::COCO(o)) => {
                s.merge(o);
                Ok(())
            }
            (EHSketchList::COUNTL2HH(s), EHSketchList::COUNTL2HH(o)) => {
                s.merge(o);
                Ok(())
            }
            (EHSketchList::CS(s), EHSketchList::CS(o)) => {
                s.merge(o);
                Ok(())
            }
            (EHSketchList::DDS(s), EHSketchList::DDS(o)) => s
                .merge(o)
                .map_err(|_| "Cannot merge DDSketches with different index mappings"),
            (EHSketchList::ELASTIC(s), EHSketchList::ELASTIC(o)) => {
                s.merge(o);
                Ok(())
            }
            (EHSketchList::HLL(s), EHSketchList::HLL(o)) => {
                s.merge(o);
                Ok(())
            }
            (EHSketchList::KLL(s), EHSketchList::KLL(o)) => {
                s.merge(o);
                Ok(())
            }
            #[cfg(feature = "experimental")]
            (EHSketchList::UNIFORM(s), EHSketchList::UNIFORM(o)) => s.merge(o),
            (EHSketchList::UNIVMON(s), EHSketchList::UNIVMON(o)) => {
                s.merge(o);
                Ok(())
            }
            _ => Err("Cannot merge sketches of different types"),
        }
    }

    /// Queries the sketch using a sketch-specific interpretation of `key`.
    pub fn query(&self, key: &DataInput) -> Result<f64, &'static str> {
        match (self, key) {
            (EHSketchList::CM(count_min), _) => Ok(count_min.estimate(key) as f64),
            (EHSketchList::COCO(coco), DataInput::Str(s)) => Ok(coco.estimate_key(s) as f64),
            (EHSketchList::COCO(coco), DataInput::String(s)) => {
                Ok(coco.estimate_key(s.as_str()) as f64)
            }
            (EHSketchList::COUNTL2HH(count_univ), _) => Ok(count_univ.fast_get_est(key)),
            (EHSketchList::CS(count_sketch), _) => Ok(count_sketch.estimate(key)),
            (EHSketchList::DDS(dd), DataInput::I32(i)) => dd
                .get_value_at_quantile(*i as f64)
                .ok_or("DDSketch has no samples"),
            (EHSketchList::DDS(dd), DataInput::I64(i)) => dd
                .get_value_at_quantile(*i as f64)
                .ok_or("DDSketch has no samples"),
            (EHSketchList::DDS(dd), DataInput::U32(u)) => dd
                .get_value_at_quantile(*u as f64)
                .ok_or("DDSketch has no samples"),
            (EHSketchList::DDS(dd), DataInput::U64(u)) => dd
                .get_value_at_quantile(*u as f64)
                .ok_or("DDSketch has no samples"),
            (EHSketchList::DDS(dd), DataInput::F32(f)) => dd
                .get_value_at_quantile(*f as f64)
                .ok_or("DDSketch has no samples"),
            (EHSketchList::DDS(dd), DataInput::F64(f)) => dd
                .get_value_at_quantile(*f)
                .ok_or("DDSketch has no samples"),
            (EHSketchList::DDS(dd), DataInput::Str(cmd)) => match *cmd {
                "count" => Ok(dd.get_count() as f64),
                "min" => dd.min().ok_or("DDSketch has no samples"),
                "max" => dd.max().ok_or("DDSketch has no samples"),
                _ => Err("Unsupported command for DDSketch"),
            },
            (EHSketchList::DDS(dd), DataInput::String(cmd)) => match cmd.as_str() {
                "count" => Ok(dd.get_count() as f64),
                "min" => dd.min().ok_or("DDSketch has no samples"),
                "max" => dd.max().ok_or("DDSketch has no samples"),
                _ => Err("Unsupported command for DDSketch"),
            },
            (EHSketchList::ELASTIC(elastic), DataInput::String(s)) => {
                Ok(elastic.query(s.clone()) as f64)
            }
            (EHSketchList::HLL(hll_df_modified), _) => Ok(hll_df_modified.estimate() as f64),
            (EHSketchList::KLL(kll), DataInput::I32(i)) => Ok(kll.quantile(*i as f64)),
            (EHSketchList::KLL(kll), DataInput::I64(i)) => Ok(kll.quantile(*i as f64)),
            (EHSketchList::KLL(kll), DataInput::U32(u)) => Ok(kll.quantile(*u as f64)),
            (EHSketchList::KLL(kll), DataInput::U64(u)) => Ok(kll.quantile(*u as f64)),
            (EHSketchList::KLL(kll), DataInput::F32(f)) => Ok(kll.quantile(*f as f64)),
            (EHSketchList::KLL(kll), DataInput::F64(f)) => Ok(kll.quantile(*f)),
            #[cfg(feature = "experimental")]
            (EHSketchList::UNIFORM(sampler), DataInput::U64(idx)) => sampler
                .sample_at(*idx as usize)
                .ok_or("Sample index out of bounds"),
            #[cfg(feature = "experimental")]
            (EHSketchList::UNIFORM(sampler), DataInput::U32(idx)) => sampler
                .sample_at(*idx as usize)
                .ok_or("Sample index out of bounds"),
            #[cfg(feature = "experimental")]
            (EHSketchList::UNIFORM(sampler), DataInput::I64(idx)) if *idx >= 0 => sampler
                .sample_at(*idx as usize)
                .ok_or("Sample index out of bounds"),
            #[cfg(feature = "experimental")]
            (EHSketchList::UNIFORM(sampler), DataInput::I32(idx)) if *idx >= 0 => sampler
                .sample_at(*idx as usize)
                .ok_or("Sample index out of bounds"),
            #[cfg(feature = "experimental")]
            (EHSketchList::UNIFORM(sampler), DataInput::Str(cmd)) => match *cmd {
                "len" => Ok(sampler.len() as f64),
                "total_seen" => Ok(sampler.total_seen() as f64),
                _ => Err("Unsupported command for UniformSampling"),
            },
            #[cfg(feature = "experimental")]
            (EHSketchList::UNIFORM(sampler), DataInput::String(cmd)) => match cmd.as_str() {
                "len" => Ok(sampler.len() as f64),
                "total_seen" => Ok(sampler.total_seen() as f64),
                _ => Err("Unsupported command for UniformSampling"),
            },
            (EHSketchList::UNIVMON(um), DataInput::Str(cmd)) => match *cmd {
                "cardinality" | "card" => Ok(um.calc_card()),
                "l1" => Ok(um.calc_l1()),
                "l2" => Ok(um.calc_l2()),
                "entropy" => Ok(um.calc_entropy()),
                _ => Err("Unsupported command for UnivMon"),
            },
            (EHSketchList::UNIVMON(um), DataInput::String(cmd)) => match cmd.as_str() {
                "cardinality" | "card" => Ok(um.calc_card()),
                "l1" => Ok(um.calc_l1()),
                "l2" => Ok(um.calc_l2()),
                "entropy" => Ok(um.calc_entropy()),
                _ => Err("Unsupported command for UnivMon"),
            },
            _ => Err("Parameter type and Sketch Type Mismatched"),
        }
    }

    /// Get the type of sketch as a string
    pub fn sketch_type(&self) -> &'static str {
        match self {
            EHSketchList::CM(_) => "CountMin",
            EHSketchList::COCO(_) => "Coco",
            EHSketchList::COUNTL2HH(_) => "CountL2HH",
            EHSketchList::CS(_) => "CountSketch",
            EHSketchList::DDS(_) => "DDSketch",
            EHSketchList::ELASTIC(_) => "Elastic",
            EHSketchList::HLL(_) => "HLL",
            EHSketchList::KLL(_) => "KLL",
            #[cfg(feature = "experimental")]
            EHSketchList::UNIFORM(_) => "UniformSampling",
            EHSketchList::UNIVMON(_) => "UnivMon",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_routes_to_countl2hh_and_univmon() {
        let key = DataInput::I64(7);

        let mut count_l2hh = EHSketchList::COUNTL2HH(CountL2HH::with_dimensions(5, 1024));
        for _ in 0..9 {
            count_l2hh.insert(&key);
        }
        let l2hh_est = count_l2hh.query(&key).expect("query COUNTL2HH");
        assert!(
            l2hh_est >= 9.0,
            "expected COUNTL2HH estimate >= 9, got {l2hh_est}"
        );

        let mut um = EHSketchList::UNIVMON(UnivMon::default());
        for _ in 0..6 {
            um.insert(&key);
        }
        match um {
            EHSketchList::UNIVMON(ref u) => assert_eq!(u.bucket_size, 6),
            _ => panic!("expected UnivMon chapter variant"),
        }
    }

    #[test]
    fn count_sketch_insert_and_query_round_trip() {
        let mut cs = EHSketchList::CS(Count::<Vector2D<i32>, FastPath>::default());
        let key = DataInput::I64(11);
        cs.insert(&key);
        let est = cs.query(&key).expect("query CountSketch");
        assert!(est >= 1.0, "expected CountSketch estimate >= 1, got {est}");
    }

    #[test]
    fn ddsketch_insert_and_quantile_query_round_trip() {
        let mut dd = EHSketchList::DDS(DDSketch::new(0.01));
        dd.insert(&DataInput::F64(10.0));
        dd.insert(&DataInput::F64(20.0));
        dd.insert(&DataInput::F64(30.0));

        let q50 = dd.query(&DataInput::F64(0.5)).expect("query DDSketch q50");
        assert!((10.0..=30.0).contains(&q50), "unexpected q50 {q50}");
    }

    /// `EHSketchList` derives its serde form from the variants it holds, so a
    /// round trip of a nested DDSketch must carry the sketch's whole state.
    #[test]
    fn nested_ddsketch_survives_a_serde_round_trip() {
        let mut dd = EHSketchList::DDS(DDSketch::new(0.01));
        for v in [10.0f64, 20.0, 30.0, 40.0] {
            dd.insert(&DataInput::F64(v));
        }
        let before = match &dd {
            EHSketchList::DDS(inner) => (inner.get_count(), inner.sum(), inner.min(), inner.max()),
            _ => panic!("expected the DDSketch variant"),
        };

        let bytes = rmp_serde::to_vec(&dd).expect("encode");
        let restored: EHSketchList = rmp_serde::from_slice(&bytes).expect("decode");
        let after = match &restored {
            EHSketchList::DDS(inner) => (inner.get_count(), inner.sum(), inner.min(), inner.max()),
            _ => panic!("expected the DDSketch variant"),
        };
        assert_eq!(after, before, "the nested sketch lost its running state");

        let q50 = restored
            .query(&DataInput::F64(0.5))
            .expect("query the restored DDSketch");
        assert!((10.0..=40.0).contains(&q50), "unexpected q50 {q50}");
    }

    #[test]
    fn supports_norm_whitelist_is_enforced() {
        let cm = EHSketchList::CM(CountMin::<Vector2D<i32>, FastPath>::default());
        assert!(cm.supports_norm(SketchNorm::L1));
        assert!(!cm.supports_norm(SketchNorm::L2));

        let count_l2hh = EHSketchList::COUNTL2HH(CountL2HH::with_dimensions(5, 1024));
        assert!(count_l2hh.supports_norm(SketchNorm::L2));
        assert!(!count_l2hh.supports_norm(SketchNorm::L1));

        let cs = EHSketchList::CS(Count::<Vector2D<i32>, FastPath>::default());
        assert!(cs.supports_norm(SketchNorm::L1));
        assert!(!cs.supports_norm(SketchNorm::L2));

        let dd = EHSketchList::DDS(DDSketch::new(0.01));
        assert!(dd.supports_norm(SketchNorm::L1));
        assert!(!dd.supports_norm(SketchNorm::L2));

        let um = EHSketchList::UNIVMON(UnivMon::default());
        assert!(um.supports_norm(SketchNorm::L2));
        assert!(!um.supports_norm(SketchNorm::L1));
    }
}
