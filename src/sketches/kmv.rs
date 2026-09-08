//! KMV distinct-count sketch.
//!
//! Reference:
//! - "On synopses for distinct-value estimation under multiset operations"
//!   <https://dl.acm.org/doi/10.1145/1247480.1247504>

use crate::{
    CANONICAL_HASH_SEED, CommonHeap, DataInput, DefaultXxHasher, KeepLargest, SketchHasher,
};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

mod wire;

// expect error bound to be less than 2%
const KMV_DEFAULT_LENGTH: usize = 4096_usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct KMV<H: SketchHasher = DefaultXxHasher> {
    pub k: usize,
    pub k_vals: CommonHeap<u64, KeepLargest>,
    #[serde(skip)]
    _hasher: PhantomData<H>,
}

impl Default for KMV {
    fn default() -> Self {
        Self::new(KMV_DEFAULT_LENGTH)
    }
}

impl<H: SketchHasher> KMV<H> {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            k_vals: CommonHeap::new_max(k),
            _hasher: PhantomData,
        }
    }

    pub fn insert(&mut self, item: &DataInput) {
        let hashed = H::hash64_seeded(CANONICAL_HASH_SEED, item);
        self.insert_by_hash(hashed);
    }

    pub fn insert_by_hash(&mut self, hash_value: u64) {
        if self.k_vals.iter().any(|value| *value == hash_value) {
            return;
        }
        self.k_vals.push(hash_value);
    }

    pub fn estimate(&mut self) -> f64 {
        if self.k_vals.len() < self.k {
            return self.k_vals.len() as f64;
        }
        let largest = *self
            .k_vals
            .peek()
            .expect("k_vals should be non-empty when len >= k");
        const DIVISOR: f64 = 1.0 / (1u64 << 53) as f64;
        let mapped: f64 = (largest >> 11) as f64 * DIVISOR;
        (self.k - 1) as f64 / mapped
    }

    pub fn merge(&mut self, other: &mut KMV<H>) {
        assert_eq!(
            self.k, other.k,
            "Two KMV sketch have different k size, not mergeable"
        );
        for &value in other.k_vals.iter() {
            self.insert_by_hash(value);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::DataInput;

    const ERROR_TOLERANCE: f64 = 0.02;
    const SERDE_SAMPLE: usize = 100_000;

    #[test]
    fn assert_serialization_round_trip() {
        let mut sketch: KMV = KMV::default();
        for value in 0..SERDE_SAMPLE {
            let input = DataInput::U64(value as u64);
            sketch.insert(&input);
        }

        let encoded = sketch
            .serialize_to_bytes()
            .unwrap_or_else(|err| panic!("KMV serialize_to_bytes failed: {err}"));
        assert!(
            !encoded.is_empty(),
            "KMV serialization output should not be empty"
        );

        let mut decoded: KMV = KMV::deserialize_from_bytes(&encoded)
            .unwrap_or_else(|err| panic!("KMV deserialize_from_bytes failed: {err}"));

        let reencoded = decoded
            .serialize_to_bytes()
            .unwrap_or_else(|err| panic!("KMV re-serialize failed: {err}"));

        assert_eq!(
            encoded, reencoded,
            "KMV serialized bytes differed after round trip"
        );

        let original_est = sketch.estimate();
        let decoded_est = decoded.estimate();
        assert!(
            (original_est - decoded_est).abs() <= ERROR_TOLERANCE * original_est.max(1.0),
            "KMV estimate mismatch after round trip: before {original_est}, after {decoded_est}"
        );
    }
}
