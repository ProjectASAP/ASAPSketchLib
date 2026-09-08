use serde::{Deserialize, Serialize};

use crate::DataInput;

mod wire;

const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const GAMMA: u64 = 0xBF58_476D_1CE4_E5B9;
const DELTA: u64 = 0x94D0_49BB_1331_11EB;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SampleEntry {
    priority: u64,
    value: f64,
}

impl SampleEntry {
    fn new(priority: u64, value: f64) -> Self {
        Self { priority, value }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniformSampling {
    sample_rate: f64,
    total_seen: u64,
    rng_state: u64,
    entries: Vec<SampleEntry>,
}

impl UniformSampling {
    pub fn new(sample_rate: f64) -> Self {
        Self::with_seed(sample_rate, DEFAULT_SEED)
    }

    pub fn with_seed(sample_rate: f64, seed: u64) -> Self {
        assert!(
            (0.0..=1.0).contains(&sample_rate) && sample_rate > 0.0,
            "uniform sampling rate must be within (0, 1]"
        );
        let init_state = if seed == 0 { DEFAULT_SEED } else { seed };
        Self {
            sample_rate,
            total_seen: 0,
            rng_state: init_state,
            entries: Vec::new(),
        }
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    pub fn samples(&self) -> Vec<f64> {
        self.entries.iter().map(|entry| entry.value).collect()
    }

    pub fn sample_at(&self, idx: usize) -> Option<f64> {
        self.entries.get(idx).map(|entry| entry.value)
    }

    pub fn update_input(&mut self, value: &DataInput<'_>) -> Result<(), &'static str> {
        match value {
            DataInput::I32(v) => {
                self.update(*v as f64);
                Ok(())
            }
            DataInput::I64(v) => {
                self.update(*v as f64);
                Ok(())
            }
            DataInput::U32(v) => {
                self.update(*v as f64);
                Ok(())
            }
            DataInput::U64(v) => {
                self.update(*v as f64);
                Ok(())
            }
            DataInput::F32(v) => {
                self.update(*v as f64);
                Ok(())
            }
            DataInput::F64(v) => {
                self.update(*v);
                Ok(())
            }
            _ => Err("UniformSampling only supports numeric inputs"),
        }
    }

    pub fn update(&mut self, value: f64) {
        self.total_seen = self.total_seen.saturating_add(1);
        let target_size = Self::target_size(self.total_seen, self.sample_rate);
        let priority = self.next_random();
        self.insert_entry(SampleEntry::new(priority, value));
        self.truncate_to(target_size);
    }

    pub fn merge(&mut self, other: &UniformSampling) -> Result<(), &'static str> {
        if (self.sample_rate - other.sample_rate).abs() > f64::EPSILON {
            return Err("Cannot merge uniform samplers with different sampling rates");
        }
        let combined_seen = self.total_seen.saturating_add(other.total_seen);
        let mut merged = Vec::with_capacity(self.entries.len() + other.entries.len());
        merged.extend(self.entries.iter().cloned());
        merged.extend(other.entries.iter().cloned());
        merged.sort_by_key(|a| a.priority);
        let target_size = Self::target_size(combined_seen, self.sample_rate);
        merged.truncate(target_size);
        self.entries = merged;
        self.total_seen = combined_seen;
        self.mix_state(other.rng_state);
        Ok(())
    }

    fn target_size(total_seen: u64, rate: f64) -> usize {
        if total_seen == 0 {
            0
        } else {
            ((total_seen as f64) * rate).ceil() as usize
        }
    }

    fn insert_entry(&mut self, entry: SampleEntry) {
        let idx = match self
            .entries
            .binary_search_by(|probe| probe.priority.cmp(&entry.priority))
        {
            Ok(position) | Err(position) => position,
        };
        self.entries.insert(idx, entry);
    }

    fn truncate_to(&mut self, target_size: usize) {
        while self.entries.len() > target_size {
            self.entries.pop();
        }
    }

    fn next_random(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(DEFAULT_SEED);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(GAMMA);
        z = (z ^ (z >> 27)).wrapping_mul(DELTA);
        z ^ (z >> 31)
    }

    fn mix_state(&mut self, other: u64) {
        let mixed = self.rng_state ^ other.rotate_left(19);
        self.rng_state = if mixed == 0 { DEFAULT_SEED } else { mixed };
    }
}

#[cfg(test)]
mod tests {
    use super::UniformSampling;

    fn expected_size(rate: f64, total_seen: u64) -> usize {
        if total_seen == 0 {
            0
        } else {
            ((total_seen as f64) * rate).ceil() as usize
        }
    }

    #[test]
    fn sample_count_tracks_rate() {
        let mut sampler = UniformSampling::with_seed(0.4, 0xABC1);
        for (idx, value) in (0..10).enumerate() {
            sampler.update(value as f64);
            let seen = (idx + 1) as u64;
            assert_eq!(sampler.total_seen(), seen);
            assert_eq!(sampler.len(), expected_size(0.4, seen));
        }
    }

    #[test]
    fn samples_are_drawn_from_input_stream() {
        let mut sampler = UniformSampling::with_seed(0.25, 0xBEEFFACE);
        for value in 0..128 {
            sampler.update(value as f64);
        }
        assert_eq!(sampler.total_seen(), 128);
        assert_eq!(sampler.len(), expected_size(0.25, sampler.total_seen()));
        for value in sampler.samples() {
            assert!(value.floor() == value);
            assert!((0.0..128.0).contains(&value));
        }
    }

    #[test]
    fn merge_combines_samples_using_rate_based_target() {
        let mut left = UniformSampling::with_seed(0.2, 0xDEAD);
        for value in 0..64 {
            left.update(value as f64);
        }
        let mut right = UniformSampling::with_seed(0.2, 0xBEEF);
        for value in 100..200 {
            right.update(value as f64);
        }
        let mut combined = left.clone();
        combined.merge(&right).unwrap();
        assert_eq!(
            combined.total_seen(),
            left.total_seen() + right.total_seen()
        );
        assert_eq!(combined.len(), expected_size(0.2, combined.total_seen()));
        for value in combined.samples() {
            assert!(
                (0.0..64.0).contains(&value) || (100.0..200.0).contains(&value),
                "unexpected sample {value}"
            );
        }
    }

    #[test]
    fn merge_rejects_different_rates() {
        let mut left = UniformSampling::with_seed(0.1, 0x1);
        left.update(1.0);
        let mut right = UniformSampling::with_seed(0.2, 0x2);
        right.update(2.0);
        assert!(left.merge(&right).is_err());
    }

    #[test]
    fn sample_access_is_stable() {
        let mut sampler = UniformSampling::with_seed(0.5, 0xFACEFACE);
        for value in 0..20 {
            sampler.update(value as f64);
        }
        let snapshot = sampler.samples();
        for (idx, expected) in snapshot.iter().enumerate() {
            let value = sampler.sample_at(idx).expect("sample exists");
            assert_eq!(value, *expected);
        }
    }
}
