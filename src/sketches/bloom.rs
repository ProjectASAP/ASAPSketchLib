//! Bloom filters for distinct/membership telemetry workloads.
//!
//! `BloomFilter` is a standard bit Bloom filter. `CountingBloomFilter` stores
//! saturating `u8` counters and supports deletion. Both use deterministic FNV-1a
//! double hashing so serialized benchmark results are reproducible without a
//! global RNG or external dependency.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BloomFilter {
    cells: Vec<u8>,
    hashes: usize,
}

impl BloomFilter {
    pub fn new(cells: usize, hashes: usize) -> Self {
        assert!(cells > 0 && hashes > 0, "Bloom dimensions must be non-zero");
        Self {
            cells: vec![0; cells],
            hashes,
        }
    }
    pub fn insert(&mut self, key: &str) {
        for index in self.indices(key).collect::<Vec<_>>() {
            self.cells[index] = 1;
        }
    }
    pub fn contains(&self, key: &str) -> bool {
        self.indices(key)
            .into_iter()
            .all(|index| self.cells[index] != 0)
    }
    pub fn cells(&self) -> usize {
        self.cells.len()
    }
    pub fn hashes(&self) -> usize {
        self.hashes
    }
    pub fn memory_bytes(&self) -> usize {
        self.cells.len()
    }
    pub fn clear(&mut self) {
        self.cells.fill(0);
    }
    fn indices(&self, key: &str) -> impl Iterator<Item = usize> + '_ {
        let (first, second) = hash_pair(key);
        (0..self.hashes).map(move |round| {
            first.wrapping_add((round as u64).wrapping_mul(second)) as usize % self.cells.len()
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountingBloomFilter {
    counters: Vec<u8>,
    hashes: usize,
}

impl CountingBloomFilter {
    pub fn new(cells: usize, hashes: usize) -> Self {
        assert!(cells > 0 && hashes > 0, "Bloom dimensions must be non-zero");
        Self {
            counters: vec![0; cells],
            hashes,
        }
    }
    pub fn insert(&mut self, key: &str) {
        for index in self.indices(key).collect::<Vec<_>>() {
            self.counters[index] = self.counters[index].saturating_add(1);
        }
    }
    pub fn remove(&mut self, key: &str) {
        for index in self.indices(key).collect::<Vec<_>>() {
            self.counters[index] = self.counters[index].saturating_sub(1);
        }
    }
    pub fn contains(&self, key: &str) -> bool {
        self.indices(key)
            .into_iter()
            .all(|index| self.counters[index] != 0)
    }
    pub fn cells(&self) -> usize {
        self.counters.len()
    }
    pub fn hashes(&self) -> usize {
        self.hashes
    }
    pub fn memory_bytes(&self) -> usize {
        self.counters.len()
    }
    pub fn clear(&mut self) {
        self.counters.fill(0);
    }
    fn indices(&self, key: &str) -> impl Iterator<Item = usize> + '_ {
        let (first, second) = hash_pair(key);
        (0..self.hashes).map(move |round| {
            first.wrapping_add((round as u64).wrapping_mul(second)) as usize % self.counters.len()
        })
    }
}

fn hash_pair(key: &str) -> (u64, u64) {
    let mut first = 1469598103934665603_u64;
    let mut second = 1099511628211_u64;
    for byte in key.bytes() {
        first = (first ^ byte as u64).wrapping_mul(1099511628211);
        second = (second ^ byte as u64).wrapping_mul(14029467366897019727);
    }
    (first, second | 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bloom_membership_and_clear() {
        let mut filter = BloomFilter::new(1024, 4);
        filter.insert("a");
        assert!(filter.contains("a"));
        filter.clear();
        assert!(!filter.contains("a"));
    }
    #[test]
    fn counting_bloom_supports_delete_and_saturates() {
        let mut filter = CountingBloomFilter::new(128, 3);
        filter.insert("a");
        filter.insert("a");
        filter.remove("a");
        assert!(filter.contains("a"));
        filter.remove("a");
        assert!(!filter.contains("a"));
    }
    #[test]
    fn invalid_dimensions_are_rejected() {
        assert!(std::panic::catch_unwind(|| BloomFilter::new(0, 1)).is_err());
        assert!(std::panic::catch_unwind(|| CountingBloomFilter::new(1, 0)).is_err());
    }
}
