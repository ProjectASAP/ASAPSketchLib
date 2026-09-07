//! KLL quantile sketch (compact / insert-optimized variant).
//!
//! Insertion and compaction follow the compact KLL layout from:
//! "Insert-optimized implementation of streaming data sketches" (Pfeil et al., 2025).
//! CDF construction follows the pattern described in dgryski/go-kll, based on the
//! weighted CDF approach from the original KLL paper (Karnin, Lang & Liberty, FOCS 2016).
//!
//! References:
//! - Karnin, Lang & Liberty, "Optimal Quantile Approximation in Streams," FOCS 2016.
//!   <https://arxiv.org/abs/1603.05346>
//! - <https://www.amazon.science/publications/insert-optimized-implementation-of-streaming-data-sketches>

use rand::{Rng, rng};
use serde::{Deserialize, Serialize};

use crate::common::input::data_input_to_f64;
use crate::common::numerical::NumericalValue;
use crate::{DataInput, Vector1D};

mod wire;
pub(crate) use wire::{
    KLL_KIND_DYNAMIC, KllCoinWire, KllPayload, KllWireItem, kll_metadata, split_and_validate_meta,
    validate_kll_payload,
};

const MAX_LEVELS: usize = 61;

const CAPACITY_CACHE_LEN: usize = 20;
const MAX_CACHEABLE_K: usize = 26_602;
const CAPACITY_DECAY: f64 = 2.0 / 3.0;
const DEFAULT_K: i32 = 200;

/// Coin generates deterministic pseudo-random coin flips while amortizing
/// calls to the RNG by consuming one bit at a time from a 64-bit buffer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Coin {
    state: u64,
    bit_cache: u64,
    #[serde(default)]
    remaining_bits: u8,
}

impl Coin {
    pub fn new() -> Self {
        let mut rng = rng();
        Self::from_seed(rng.random::<u64>())
    }

    pub fn xorshift_mult64(mut x: u64) -> u64 {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(2685821657736338717)
    }

    pub(crate) fn from_seed(seed: u64) -> Self {
        Self {
            state: Self::normalize_seed(seed),
            bit_cache: 0,
            remaining_bits: 0,
        }
    }

    #[inline]
    fn normalize_seed(seed: u64) -> u64 {
        const FALLBACK: u64 = 0x9e37_79b9_7f4a_7c15;
        if seed == 0 { FALLBACK } else { seed }
    }

    #[inline]
    fn refill(&mut self) {
        self.state = Self::normalize_seed(Self::xorshift_mult64(self.state));
        self.bit_cache = self.state;
        self.remaining_bits = u64::BITS as u8;
    }

    pub fn toss(&mut self) -> bool {
        if self.remaining_bits == 0 {
            self.refill();
        }
        let bit = (self.bit_cache & 1) != 0;
        self.bit_cache >>= 1;
        self.remaining_bits -= 1;
        bit
    }

    /// The coin's raw state in `sketchlib-go::CoinState` shape:
    /// `(state, bit_cache, remaining_bits)`. Used by the ASAPv1 wire payload
    /// (both KLL variants) to carry the compaction RNG so a decoded sketch can
    /// continue compacting deterministically.
    #[inline]
    pub(crate) fn to_wire(&self) -> (u64, u64, u32) {
        (self.state, self.bit_cache, self.remaining_bits as u32)
    }

    /// Rebuilds a coin from its raw wire state. `remaining_bits` is validated by
    /// the caller (it must be `<= 64`) so decode fails closed on crafted bytes
    /// rather than silently truncating into the `u8` field.
    #[inline]
    pub(crate) fn from_wire(state: u64, bit_cache: u64, remaining_bits: u8) -> Self {
        Self {
            state,
            bit_cache,
            remaining_bits,
        }
    }
}

/// A single (value, cumulative-quantile) pair in a CDF table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdfEntry {
    value: f64,
    quantile: f64,
}

/// Computes the maximum number of items the sketch can hold across all
/// levels for the given `k` and `m`. The buffer is pre-allocated to this
/// size so that no dynamic reallocation ever occurs.
fn compute_max_capacity(k: usize, m: usize) -> usize {
    let mut total = 0;
    let mut scale = 1.0_f64;
    for _ in 0..MAX_LEVELS {
        total += ((k as f64) * scale).ceil().max(m as f64) as usize;
        scale *= CAPACITY_DECAY;
    }
    total
}

/// Checked total weighted item count for compactor-level sizes given
/// **bottom-first** (`sizes[h]` is the item count at compactor level `h`, whose
/// weight is `2^h`). Returns `None` on `usize` overflow.
///
/// A live sketch's weighted count equals the number of ingested items, so it
/// always fits. Decoders use this to reject a crafted-but-structurally-valid
/// level layout (e.g. many items parked at a high level) that would otherwise
/// overflow `count()` / `rank()` / `cdf()` at query time — a fail-closed guard
/// so decode never yields a sketch that panics on a later query.
pub(crate) fn checked_weighted_count(sizes_bottom_first: &[usize]) -> Option<usize> {
    let mut total = 0usize;
    for (h, &size) in sizes_bottom_first.iter().enumerate() {
        // `h < MAX_LEVELS (61) < 64`, so the shift amount is always valid.
        let weight = 1usize.checked_shl(h as u32)?;
        total = total.checked_add(size.checked_mul(weight)?)?;
    }
    Some(total)
}

/// Halves a sorted run, placing survivors in the **upper** (right) half of
/// `items[begin..begin+pop]` so they are contiguous with the level above.
/// Traverses backwards to avoid overwriting unread source elements.
#[inline]
pub(crate) fn randomly_halve_up<T: Copy>(
    items: &mut [T],
    begin: usize,
    pop: usize,
    offset: usize,
) -> usize {
    let num_survivors = (pop - offset).div_ceil(2);
    let dest = begin + pop - num_survivors;
    for d in (0..num_survivors).rev() {
        items[dest + d] = items[begin + offset + 2 * d];
    }
    num_survivors
}

/// Merges two contiguous sorted runs in `slice` using `NumericalValue::total_cmp`.
/// `slice[..left_len]` is the first sorted run, `slice[left_len..]` is the
/// second.  `buf` is a reusable scratch buffer.
#[inline]
pub(crate) fn merge_sorted_runs<T: NumericalValue>(
    slice: &mut [T],
    left_len: usize,
    buf: &mut Vec<T>,
) {
    let total = slice.len();
    if left_len == 0 || left_len >= total {
        return;
    }
    if slice[left_len - 1].total_cmp(&slice[left_len]).is_le() {
        return;
    }

    let right_len = total - left_len;
    buf.clear();

    if left_len <= right_len {
        buf.extend_from_slice(&slice[..left_len]);
        let mut i = 0;
        let mut j = left_len;
        let mut k = 0;
        while i < buf.len() && j < total {
            if buf[i].total_cmp(&slice[j]).is_le() {
                slice[k] = buf[i];
                i += 1;
            } else {
                slice[k] = slice[j];
                j += 1;
            }
            k += 1;
        }
        if i < buf.len() {
            slice[k..k + (buf.len() - i)].copy_from_slice(&buf[i..]);
        }
    } else {
        buf.extend_from_slice(&slice[left_len..]);
        let mut i = left_len;
        let mut j = buf.len();
        let mut k = total;
        while i > 0 && j > 0 {
            k -= 1;
            if buf[j - 1].total_cmp(&slice[i - 1]).is_ge() {
                slice[k] = buf[j - 1];
                j -= 1;
            } else {
                slice[k] = slice[i - 1];
                i -= 1;
            }
        }
        if j > 0 {
            slice[..j].copy_from_slice(&buf[..j]);
        }
    }
}

// ---------------------------------------------------------------------------
// KLL sketch
// ---------------------------------------------------------------------------

/// Compact, insert-optimized KLL quantile sketch.
///
/// Memory layout (grows leftward):
/// ```text
/// items: [ free ← | L0 (unsorted) | L1 | L2 | … | L_top ]
///         0        levels[0]                        levels[num_levels]
/// ```
///
/// `levels[h]` = start of level h.  `levels[h+1] - levels[h]` = size of level h.
#[derive(Clone, Debug)]
pub struct KLL<T: NumericalValue = f64> {
    items: Box<[T]>,
    levels: Box<[usize]>,
    k: usize,
    m: usize,
    num_levels: usize,
    max_capacity: usize,
    co: Coin,
    /// Explicit seed for the compaction RNG, when present. When `Some`, the
    /// sketch's bytes are reproducible across runs and `clear()` re-seeds
    /// from this value. When `None`, the legacy time-based path is used and
    /// `clear()` re-seeds from the wall clock.
    seed: Option<u64>,
    capacity_cache: [u32; CAPACITY_CACHE_LEN],
    top_height: usize,
    level0_capacity: usize,
    merge_buf: Vec<T>,
    /// Memoized CDF, valid only while no mutation has occurred since it was
    /// built. Every mutating entry point (`push_value`, `merge`, `clear`,
    /// `ensure_levels_sorted`) drops it; query-side accessors never touch it.
    /// Rebuilding the CDF sorts every retained item, which made repeated
    /// quantile queries O(n log n) each — the dashboard fill-then-query
    /// pattern paid that cost per query.
    cdf_cache: Option<Cdf>,
}

impl<T: NumericalValue> Default for KLL<T> {
    fn default() -> Self {
        Self::init_kll(DEFAULT_K)
    }
}

impl<T: NumericalValue> KLL<T> {
    /// Creates a new KLL sketch with accuracy parameter `k` and minimum level capacity `m`.
    pub fn init(k: usize, m: usize) -> Self {
        Self::init_internal(k, m, Coin::new(), None)
    }

    /// Creates a new KLL sketch with an explicit RNG seed for the compaction
    /// coin. Two sketches built with the same seed and fed the same input
    /// sequence produce byte-identical serialized state — required for
    /// reproducible-replay scenarios and cross-process parity tests. The seed
    /// is also stored so `clear()` re-seeds deterministically across window
    /// rotations rather than silently jumping back to the wall clock.
    ///
    /// Callers that don't care about determinism should keep using `init` /
    /// `init_kll`.
    pub fn init_with_seed(k: usize, m: usize, seed: u64) -> Self {
        Self::init_internal(k, m, Coin::from_seed(seed), Some(seed))
    }

    /// Creates a new KLL sketch with the given `k` and a default minimum level capacity of 8.
    pub fn init_kll(k: i32) -> Self {
        Self::init(k as usize, 8)
    }

    /// `init_kll` with an explicit RNG seed. See `init_with_seed`.
    pub fn init_kll_with_seed(k: i32, seed: u64) -> Self {
        Self::init_with_seed(k as usize, 8, seed)
    }

    fn init_internal(k: usize, m: usize, coin: Coin, seed: Option<u64>) -> Self {
        let mut norm_m = m.min(MAX_CACHEABLE_K);
        norm_m = norm_m.max(2);
        let mut norm_k = k.max(norm_m);
        if norm_k > MAX_CACHEABLE_K {
            norm_k = MAX_CACHEABLE_K;
        }
        let max_cap = compute_max_capacity(norm_k, norm_m);
        let mut s = Self {
            items: vec![T::default(); max_cap].into_boxed_slice(),
            levels: {
                let mut v = vec![0usize; MAX_LEVELS + 1];
                v[0] = max_cap;
                v[1] = max_cap;
                v.into_boxed_slice()
            },
            k: norm_k,
            m: norm_m,
            num_levels: 1,
            max_capacity: max_cap,
            co: coin,
            seed,
            capacity_cache: [0; CAPACITY_CACHE_LEN],
            top_height: 0,
            level0_capacity: 0,
            merge_buf: Vec::with_capacity(norm_k),
            cdf_cache: None,
        };
        s.rebuild_capacity_cache();
        s
    }

    /// Reconstruct a KLL directly from portable wire state — the retained
    /// `items` in level order plus the `levels` boundary array (proto contract:
    /// `levels[0] == 0`, `levels[num_levels] == items.len()`). Avoids replaying
    /// every item through `update()`: it places the items straight into the
    /// internal buffer and fixes up the level boundaries, so the result is
    /// bit-identical to the source sketch (identical quantiles) at a fraction
    /// of the cost. Errors if the supplied state is inconsistent.
    pub fn from_portable_state(
        k: usize,
        items: &[T],
        levels: &[usize],
        num_levels: usize,
    ) -> Result<Self, String> {
        let mut s = Self::init_kll(k as i32);
        if items.is_empty() {
            return Ok(s);
        }
        if num_levels == 0 || num_levels >= s.levels.len() {
            return Err(format!(
                "from_portable_state: invalid num_levels {num_levels}"
            ));
        }
        if levels.len() != num_levels + 1 {
            return Err(format!(
                "from_portable_state: levels.len()={} != num_levels+1={}",
                levels.len(),
                num_levels + 1
            ));
        }
        if levels[0] != 0 || levels[num_levels] != items.len() {
            return Err(format!(
                "from_portable_state: level bounds [{}..{}] inconsistent with items.len()={}",
                levels[0],
                levels[num_levels],
                items.len()
            ));
        }
        if items.len() > s.max_capacity {
            return Err(format!(
                "from_portable_state: {} items exceed max_capacity {} for k={}",
                items.len(),
                s.max_capacity,
                k
            ));
        }
        // Live data occupies the high end of the buffer; free space at the front.
        let offset = s.max_capacity - items.len();
        let max_cap = s.max_capacity;
        s.items[offset..].clone_from_slice(items);
        for (dst, &src) in s.levels.iter_mut().zip(levels[..=num_levels].iter()) {
            *dst = src + offset;
        }
        for dst in s.levels.iter_mut().skip(num_levels + 1) {
            *dst = max_cap;
        }
        s.num_levels = num_levels;
        s.rebuild_capacity_cache(); // recomputes top_height + level0_capacity for num_levels
        Ok(s)
    }

    /// Hot-path insert: decrement `levels[0]`, write item, check capacity.
    #[inline]
    fn push_value(&mut self, value: T) {
        self.invalidate_cdf_cache();
        if self.levels[0] == 0 {
            self.compress_while_updating();
        }
        self.levels[0] -= 1;
        self.items[self.levels[0]] = value;

        if self.levels[1] - self.levels[0] > self.level0_capacity {
            self.compress_while_updating();
        }
    }

    /// Inserts a typed numeric value into the sketch.
    pub fn update(&mut self, val: &T) {
        self.push_value(*val);
    }

    /// Inserts a batch of values. Exactly equivalent to `for v in values { update(v) }`
    /// but exposed as a single call for callers that already have a slice.
    /// Empty slices are a no-op and do not disturb the CDF cache or coin.
    #[inline]
    pub fn bulk_update(&mut self, values: &[T]) {
        for v in values {
            self.push_value(*v);
        }
    }

    // -- Compaction ----------------------------------------------------------

    fn compress_while_updating(&mut self) {
        let mut h = 0;
        loop {
            let pop = self.level_size(h);
            let cap = self.capacity_for_level(h);
            if pop <= cap {
                break;
            }
            if h + 1 == self.num_levels {
                self.add_new_top_level();
            }
            self.compact(h);
            h += 1;
        }
    }

    fn compact(&mut self, h: usize) {
        // Redundant today (compact is only reachable via push_value, which
        // already drops the cache) but cheap insurance against future
        // call-graph drift silently serving stale quantiles.
        self.invalidate_cdf_cache();
        let beg = self.levels[h];
        let end = self.levels[h + 1];
        let pop = end - beg;

        if h == 0 {
            self.items[beg..end].sort_unstable_by(T::total_cmp);
        }

        let offset = usize::from(self.co.toss());
        let num_survivors = randomly_halve_up(&mut self.items, beg, pop, offset);
        let surv_start = beg + pop - num_survivors;

        let pop_above = self.levels[h + 2] - end;
        if pop_above > 0 {
            merge_sorted_runs(
                &mut self.items[surv_start..end + pop_above],
                num_survivors,
                &mut self.merge_buf,
            );
        }

        let delta = surv_start - beg;
        if delta > 0 && h > 0 {
            let lo = self.levels[0];
            let hi = beg;
            if hi > lo {
                self.items.copy_within(lo..hi, lo + delta);
            }
            for lvl in self.levels[..h].iter_mut() {
                *lvl += delta;
            }
        }

        self.levels[h] = surv_start;
        self.levels[h + 1] = surv_start;
    }

    fn add_new_top_level(&mut self) {
        let sentinel = self.levels[self.num_levels];
        self.num_levels += 1;
        self.levels[self.num_levels] = sentinel;
        self.top_height = self.num_levels - 1;
        self.level0_capacity = self.capacity_for_level(0);
    }

    // -- Capacity helpers ----------------------------------------------------

    fn capacity_for_level(&self, level: usize) -> usize {
        if self.num_levels == 0 {
            return self.m;
        }
        let height_from_top = self.top_height.saturating_sub(level);
        let idx = height_from_top.min(CAPACITY_CACHE_LEN - 1);
        self.capacity_cache[idx] as usize
    }

    fn rebuild_capacity_cache(&mut self) {
        self.top_height = self.num_levels.saturating_sub(1);
        let mut scale = 1.0_f64;
        for idx in 0..CAPACITY_CACHE_LEN {
            let scaled = ((self.k as f64) * scale).ceil() as usize;
            let cap = scaled.max(self.m);
            self.capacity_cache[idx] = cap as u32;
            scale *= CAPACITY_DECAY;
        }
        self.level0_capacity = self.capacity_for_level(0);
    }

    #[inline]
    fn level_size(&self, h: usize) -> usize {
        self.levels[h + 1] - self.levels[h]
    }

    #[inline(always)]
    fn invalidate_cdf_cache(&mut self) {
        self.cdf_cache = None;
    }

    // -- Query-side ----------------------------------------------------------

    /// Builds and returns the cumulative distribution function (CDF) from the current sketch state.
    pub fn cdf(&self) -> Cdf {
        let mut cdf = Cdf {
            entries: Vector1D::init(self.buffer_size()),
        };
        let mut total_w = 0usize;

        for h in 0..self.num_levels {
            let start = self.levels[h];
            let end = self.levels[h + 1];
            let weight = 1 << h;
            for &value in &self.items[start..end] {
                cdf.entries.push(CdfEntry {
                    value: value.to_f64(),
                    quantile: weight as f64,
                });
            }
            total_w += (end - start) * weight;
        }

        if total_w == 0 {
            return cdf;
        }

        // Stability is irrelevant: entries sharing a value contribute the same
        // weight either way, so the unstable sort's reordering cannot change
        // any prefix sum. `total_cmp` also avoids `partial_cmp`'s NaN branch.
        cdf.entries
            .as_mut_slice()
            .sort_unstable_by(|a, b| a.value.total_cmp(&b.value));

        let mut cur_w = 0.0;
        for entry in cdf.entries.as_mut_slice() {
            cur_w += entry.quantile;
            entry.quantile = cur_w / total_w as f64;
        }

        cdf
    }

    /// Returns the CDF, rebuilding it only if the sketch changed since the
    /// last call. Identical result to [`Self::cdf`]; the point is that the
    /// O(n log n) sort is paid once per mutation instead of once per query.
    ///
    /// Takes `&mut self` so the cache can live in the sketch without
    /// interior mutability (keeping `KLL: Send`). The uncached [`Self::cdf`]
    /// stays available for shared-reference call sites.
    pub fn cdf_cached(&mut self) -> &Cdf {
        if self.cdf_cache.is_none() {
            self.cdf_cache = Some(self.cdf());
        }
        self.cdf_cache.as_ref().expect("cache just populated")
    }

    /// Cached variant of [`Self::quantile`]: see [`Self::cdf_cached`].
    pub fn quantile_cached(&mut self, q: f64) -> f64 {
        self.cdf_cached().query(q)
    }

    /// Merges another KLL sketch's retained items into this one,
    /// **preserving each retained item's level weight** (`2^level`)
    /// instead of discarding it.
    ///
    /// This is the standard weight-preserving KLL merge: for every
    /// compactor level `h` present in either sketch, the two sketches'
    /// retained items at that level are combined (level 0 is unsorted, so
    /// it's concatenated and re-sorted; levels ≥ 1 are each already sorted
    /// in a valid KLL, so the two runs are combined in place with
    /// `merge_sorted_runs`). Any level left over its own capacity by the
    /// merge is then compacted with the exact same randomized
    /// halve-and-promote step ordinary inserts use, via `randomly_halve_up`
    /// and `merge_sorted_runs` again, looping at that level until it
    /// settles back within bounds. A merge can leave a level arbitrarily far
    /// over capacity, unlike a single `push_value`, which can only overshoot
    /// by one element, so unlike `compress_while_updating` this cascade can
    /// revisit the same level more than once.
    pub fn merge(&mut self, other: &KLL<T>) {
        let other_start = other.levels[0];
        let other_end = other.levels[other.num_levels];
        if other_end == other_start {
            return; // `other` is empty: nothing to merge.
        }
        self.invalidate_cdf_cache();

        let target_num_levels = self.num_levels.max(other.num_levels);

        // work[h] holds level h's combined (not-yet-compacted) retained
        // items. +1 slack so `work[h + 1]` is always valid while cascading.
        let mut work: Vec<Vec<T>> = vec![Vec::new(); target_num_levels + 1];
        #[allow(clippy::needless_range_loop)] // `h` also indexes self.levels/self.items
        for h in 0..self.num_levels {
            let s = self.levels[h];
            let e = self.levels[h + 1];
            work[h].extend_from_slice(&self.items[s..e]);
        }
        let mut merge_buf: Vec<T> = Vec::new();
        #[allow(clippy::needless_range_loop)] // `h` also indexes other.levels/other.items
        for h in 0..other.num_levels {
            let s = other.levels[h];
            let e = other.levels[h + 1];
            let self_len = work[h].len();
            work[h].extend_from_slice(&other.items[s..e]);
            // Levels >= 1 are individually sorted in both operands already;
            // normalize the concatenation into one sorted run up front so
            // the cascade below can treat every level >= 1 as sorted, same
            // as the invariant a settled KLL always maintains.
            if h > 0 {
                merge_sorted_runs(work[h].as_mut_slice(), self_len, &mut merge_buf);
            }
        }

        // Grow self's level bookkeeping to cover the merged height.
        self.num_levels = target_num_levels;
        self.rebuild_capacity_cache();

        // Cascade-compact exactly like `compress_while_updating`/`compact`,
        // except a level may need more than one halving pass here (a merge
        // can leave a level far over capacity, not just one element over).
        let mut h = 0;
        while h < self.num_levels {
            if h == 0 {
                work[0].sort_unstable_by(T::total_cmp);
            }
            while work[h].len() > self.capacity_for_level(h) {
                if h + 1 == self.num_levels {
                    self.num_levels += 1;
                    self.rebuild_capacity_cache();
                    work.resize(self.num_levels + 1, Vec::new());
                }
                let pop = work[h].len();
                let offset = usize::from(self.co.toss());
                let num_survivors = randomly_halve_up(work[h].as_mut_slice(), 0, pop, offset);
                let discard = pop - num_survivors;
                work[h].drain(0..discard);

                // Promote the (sorted) survivors into level h+1, merging
                // with its existing (already-sorted) content.
                let mut promoted = std::mem::take(&mut work[h]);
                let left_len = promoted.len();
                promoted.append(&mut work[h + 1]);
                merge_sorted_runs(promoted.as_mut_slice(), left_len, &mut merge_buf);
                work[h + 1] = promoted;
            }
            h += 1;
        }

        // Write the compacted per-level vectors back into the fixed
        // buffer, top level first (matches the buffer's grow-leftward
        // layout). `total <= max_capacity` is guaranteed here because
        // every level 0..self.num_levels now satisfies its own capacity
        // bound — the same invariant a live KLL always maintains — and
        // max_capacity is exactly the sum of all per-level capacities.
        let mc = self.max_capacity;
        let mut cursor = mc;
        for h in (0..self.num_levels).rev() {
            let len = work[h].len();
            cursor -= len;
            self.items[cursor..cursor + len].copy_from_slice(&work[h]);
            self.levels[h] = cursor;
        }
        self.levels[self.num_levels] = mc;
        for lvl in self.levels[self.num_levels + 1..].iter_mut() {
            *lvl = mc;
        }
    }

    /// Returns the estimated value at quantile `q` (in `[0, 1]`).
    pub fn quantile(&self, q: f64) -> f64 {
        let cdf = self.cdf();
        cdf.query(q)
    }

    /// Returns the estimated (weighted) rank of value `x`.
    pub fn rank(&self, x: f64) -> usize {
        let mut r = 0;
        for h in 0..self.num_levels {
            let start = self.levels[h];
            let end = self.levels[h + 1];
            let weight = 1 << h;
            for &val in &self.items[start..end] {
                if val.to_f64() <= x {
                    r += weight;
                }
            }
        }
        r
    }

    /// Returns the configured compactor capacity `k`.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the total (weighted) number of items ingested by the sketch.
    pub fn count(&self) -> usize {
        let mut total = 0;
        for h in 0..self.num_levels {
            total += self.level_size(h) * (1 << h);
        }
        total
    }

    fn buffer_size(&self) -> usize {
        self.levels[self.num_levels] - self.levels[0]
    }

    // -- Lifecycle -----------------------------------------------------------

    /// Resets the sketch to its empty initial state, keeping the same `k` and `m` parameters.
    /// If the sketch was constructed with an explicit seed (`init_with_seed` /
    /// `init_kll_with_seed`), the coin is re-seeded from that seed so determinism
    /// survives `clear()` (and therefore window rotation in stateful aggregators).
    pub fn clear(&mut self) {
        let mc = self.max_capacity;
        self.invalidate_cdf_cache();
        self.levels[0] = mc;
        self.levels[1] = mc;
        self.num_levels = 1;
        self.co = match self.seed {
            Some(s) => Coin::from_seed(s),
            None => Coin::new(),
        };
        self.rebuild_capacity_cache();
    }

    /// Prints compactor contents for debugging.
    pub fn print_compactors(&self)
    where
        T: std::fmt::Debug,
    {
        println!(
            "KLL Packed (k={}, levels={}, items={})",
            self.k,
            self.num_levels,
            self.buffer_size()
        );
        for h in (0..self.num_levels).rev() {
            let start = self.levels[h];
            let end = self.levels[h + 1];
            println!("  L{}: {:?}", h, &self.items[start..end]);
        }
    }

    // -- Wire-format-aligned accessors --------------------------------------

    /// Returns the configured `k` parameter.
    #[inline]
    pub fn wire_k(&self) -> u32 {
        self.k as u32
    }

    /// Returns the configured `m` parameter.
    #[inline]
    pub fn wire_m(&self) -> u32 {
        self.m as u32
    }

    /// Returns the number of currently populated compactor levels.
    #[inline]
    pub fn wire_num_levels(&self) -> u32 {
        self.num_levels as u32
    }

    /// Returns the compaction-coin state in `sketchlib-go::CoinState`
    /// shape: `(state, bit_cache, remaining_bits)`. Wire-format wrappers
    /// pack this directly into the `KLLState.coin` proto field; without
    /// it the coin state would have to be poked from private internals
    /// or reconstructed by a serde round-trip, neither of which is
    /// stable across `asap_sketchlib` releases.
    #[inline]
    pub fn wire_coin(&self) -> (u64, u64, u32) {
        (
            self.co.state,
            self.co.bit_cache,
            self.co.remaining_bits as u32,
        )
    }

    /// Returns the level-boundary array in `sketchlib-go`'s wire shape:
    /// length `num_levels + 1`, starting at 0 and ending at the total
    /// number of retained items. The chunk
    /// `wire_items()[wire_levels()[i] .. wire_levels()[i+1]]` is the
    /// **top-most-first** run for the proto's `KLLState.levels`/`items`
    /// fields, matching `sketchlib-go::KLLSketch.SerializePortable`.
    /// See the `KLLState` docstring in `proto/kll/kll.proto`: index `i`
    /// in `levels` maps to compactor level `num_levels - 1 - i` in the
    /// in-memory representation.
    pub fn wire_levels(&self) -> Vec<u32> {
        // Walk from top compactor-level downward, accumulating sizes.
        let n = self.num_levels;
        let mut out = Vec::with_capacity(n + 1);
        out.push(0u32);
        let mut acc = 0u32;
        for h in (0..n).rev() {
            let size = (self.levels[h + 1] - self.levels[h]) as u32;
            acc += size;
            out.push(acc);
        }
        out
    }

    /// Returns the retained items in `sketchlib-go`'s wire shape:
    /// concatenated top-most-level-first. Sketchlib-go pushes inputs
    /// into the unsorted L0 in input order; `asap_sketchlib`'s compact
    /// layout instead grows L0 leftward, so the buffer reads
    /// reverse-input-order. This accessor reverses the unsorted L0 run
    /// so the emitted byte sequence is identical to Go's for the same
    /// input stream (when no compaction has yet occurred). Higher
    /// levels are sorted in both producers and emitted as-is.
    ///
    /// Caveat: after L0 → L1 compaction the two producers' L1 content
    /// orderings diverge (Go's `compact` leaves L1 as two concatenated
    /// sorted runs; `asap_sketchlib` merge-sorts on the way up). The
    /// retained set is identical and quantile semantics agree, but
    /// strict byte parity past the first compaction is not guaranteed.
    /// The cross-language byte-parity test in
    /// `ASAPCollector::cross_language_parity::kll_byte_parity_with_go`
    /// uses `(1..=50)` with `k=200`, well below the L0 capacity, so
    /// this caveat does not affect the parity guard.
    pub fn wire_items(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.buffer_size());
        for h in (0..self.num_levels).rev() {
            let start = self.levels[h];
            let end = self.levels[h + 1];
            if h == 0 {
                // Unsorted L0 in `asap_sketchlib` reads reverse of the
                // input order because `push_value` decrements
                // `levels[0]` before each write. Reverse here so the
                // emitted bytes match Go's input-order L0 layout.
                out.extend(self.items[start..end].iter().rev().copied());
            } else {
                out.extend_from_slice(&self.items[start..end]);
            }
        }
        out
    }

    // -- Serialization -------------------------------------------------------
    //
    // The ASAPv1 wire methods (`serialize_to_bytes` / `deserialize_from_bytes`)
    // live in the `wire` submodule, which authors the envelope + metadata +
    // payload. The `serde::{Serialize, Deserialize}` impls below stay: they are
    // the *nested* codec used when a `KLL` is embedded in a larger serde value
    // (e.g. `HydraCounter::KLL`), which is a different concern from the
    // standalone ASAPv1 envelope.

    fn ensure_levels_sorted(&mut self) {
        if self.num_levels <= 1 {
            return;
        }
        self.invalidate_cdf_cache();
        for h in 1..self.num_levels {
            let s = self.levels[h];
            let e = self.levels[h + 1];
            if s < e {
                self.items[s..e].sort_unstable_by(T::total_cmp);
            }
        }
    }
}

/// Wire format for serialization (only the used portion of the buffer).
#[derive(Serialize, Deserialize)]
struct KLLWire<T> {
    items: Vec<T>,
    levels: Vec<usize>,
    k: usize,
    m: usize,
    num_levels: usize,
    co: Coin,
}

impl<T: NumericalValue + Serialize> Serialize for KLL<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let used_start = self.levels[0];
        let used_end = self.levels[self.num_levels];
        let wire = KLLWire {
            items: self.items[used_start..used_end].to_vec(),
            levels: self.levels[..=self.num_levels]
                .iter()
                .map(|&l| l - used_start)
                .collect(),
            k: self.k,
            m: self.m,
            num_levels: self.num_levels,
            co: self.co.clone(),
        };
        wire.serialize(serializer)
    }
}

impl KLL<f64> {
    /// Inserts a value from a [`DataInput`] into a `KLL<f64>` sketch.
    ///
    /// This adapter exists for the `HydraCounter` dispatch path, which stores a
    /// type-erased `DataInput`. Non-numeric variants return an error.
    pub fn update_data_input(&mut self, val: &DataInput) -> Result<(), &'static str> {
        let value = data_input_to_f64(val)?;
        self.push_value(value);
        Ok(())
    }

    /// Batch variant of `update_data_input`. Stops on the first non-numeric
    /// input and returns the same error as the per-element path, so
    /// `bulk_update_data_input` is exactly equivalent to looping `update_data_input`.
    /// On error `count()` is the successful prefix before the error.
    pub fn bulk_update_data_input(&mut self, values: &[DataInput]) -> Result<(), &'static str> {
        for v in values {
            let value = data_input_to_f64(v)?;
            self.push_value(value);
        }
        Ok(())
    }
}

impl<'de, T: NumericalValue + Deserialize<'de>> Deserialize<'de> for KLL<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let wire = KLLWire::<T>::deserialize(deserializer)?;

        // Fail closed on crafted bytes. This nested serde path is reachable with
        // untrusted input via `HydraCounter::KLL` (`Hydra::deserialize_from_bytes`)
        // and direct `rmp_serde::from_slice::<KLL<_>>`, so it must guard the same
        // way the ASAPv1 decoder does: bound `k`/`m` to the range the constructor
        // clamps to (else `compute_max_capacity` / `Vec::with_capacity` blow up),
        // and validate the level layout (else the buffer math below underflows or
        // indexes out of bounds, and `count()` overflows on a later query).
        if wire.m < 2 || wire.m > wire.k || wire.k > MAX_CACHEABLE_K {
            return Err(D::Error::custom(format!(
                "KLL: k={}, m={} outside valid range (2 <= m <= k <= {MAX_CACHEABLE_K})",
                wire.k, wire.m
            )));
        }
        if wire.num_levels == 0
            || wire.num_levels > MAX_LEVELS
            || wire.levels.len() != wire.num_levels + 1
            || wire.levels.first() != Some(&0)
            || wire.levels.windows(2).any(|w| w[0] > w[1])
            || wire.levels.last() != Some(&wire.items.len())
        {
            return Err(D::Error::custom("KLL: inconsistent level layout"));
        }
        let sizes: Vec<usize> = wire.levels.windows(2).map(|w| w[1] - w[0]).collect();
        if checked_weighted_count(&sizes).is_none() {
            return Err(D::Error::custom(
                "KLL: level layout overflows weighted count",
            ));
        }

        let max_cap = compute_max_capacity(wire.k, wire.m);
        let used_len = wire.items.len();
        if used_len > max_cap {
            return Err(D::Error::custom(format!(
                "KLL: {used_len} items exceed max_capacity {max_cap}"
            )));
        }
        let offset = max_cap - used_len;

        let mut items = vec![T::default(); max_cap].into_boxed_slice();
        items[offset..offset + used_len].copy_from_slice(&wire.items);

        let mut levels = vec![0usize; MAX_LEVELS + 1].into_boxed_slice();
        for (i, &l) in wire.levels.iter().enumerate() {
            levels[i] = l + offset;
        }

        let mut sketch = KLL {
            items,
            levels,
            k: wire.k,
            m: wire.m,
            num_levels: wire.num_levels,
            max_capacity: max_cap,
            co: wire.co,
            // Wire format does not carry the explicit-seed flag — a
            // round-tripped sketch keeps its current coin state but
            // would re-randomize from time on a subsequent clear().
            // Callers that need clear()-determinism after a deserialize
            // should rebuild via init_with_seed.
            seed: None,
            capacity_cache: [0; CAPACITY_CACHE_LEN],
            top_height: 0,
            level0_capacity: 0,
            merge_buf: Vec::with_capacity(wire.k),
            cdf_cache: None,
        };
        sketch.rebuild_capacity_cache();
        sketch.ensure_levels_sorted();
        Ok(sketch)
    }
}

/// The CDF for quantile queries.
#[derive(Clone, Debug)]
pub struct Cdf {
    entries: Vector1D<CdfEntry>,
}

impl Cdf {
    /// Returns the quantile for value `x` using the CDF table.
    pub fn quantile(&self, x: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let slice = self.entries.as_slice();
        match slice
            .binary_search_by(|e| e.value.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(idx) => slice[idx].quantile,
            Err(0) => 0.0,
            Err(idx) => slice[idx - 1].quantile,
        }
    }

    /// Prints the CDF entries for debugging.
    pub fn print_entries(&self) {
        println!("entries: {:?}", self.entries);
    }

    /// Returns the estimated value corresponding to quantile `p`.
    pub fn query(&self, p: f64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let slice = self.entries.as_slice();
        match slice.binary_search_by(|e| {
            e.quantile
                .partial_cmp(&p)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(idx) => slice[idx].value,
            Err(idx) if idx == slice.len() => slice[slice.len() - 1].value,
            Err(idx) => slice[idx].value,
        }
    }

    /// Quantile estimation of value `x` using linear interpolation.
    pub fn quantile_li(&self, x: f64) -> f64 {
        let slice = self.entries.as_slice();
        if slice.is_empty() {
            return 0.0;
        }
        let idx = slice.partition_point(|e| e.value < x);
        if idx == slice.len() {
            return 1.0;
        }
        if idx == 0 {
            return 0.0;
        }
        let a = slice[idx - 1].value;
        let aq = slice[idx - 1].quantile;
        let b = slice[idx].value;
        let bq = slice[idx].quantile;
        ((a - x) * bq + (x - b) * aq) / (a - b)
    }

    /// Value estimation given quantile `p`, using linear interpolation.
    pub fn query_li(&self, p: f64) -> f64 {
        let slice = self.entries.as_slice();
        if slice.is_empty() {
            return 0.0;
        }
        let idx = slice.partition_point(|e| e.quantile < p);
        if idx == slice.len() {
            return slice[slice.len() - 1].value;
        }
        if idx == 0 {
            return slice[0].value;
        }
        let a = slice[idx - 1].value;
        let aq = slice[idx - 1].quantile;
        let b = slice[idx].value;
        let bq = slice[idx].quantile;
        ((aq - p) * b + (p - bq) * a) / (aq - bq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_uniform_f64, sample_zipf_f64};

    /// The memoized CDF must agree with the from-scratch rebuild at every
    /// lifecycle stage: fresh, mid-stream, post-merge, and post-clear.
    #[test]
    fn cdf_cached_matches_uncached_across_lifecycle() {
        let mut sk = KLL::<f64>::init_with_seed(200, 8, 42);

        // Empty sketch.
        for q in [0.1f64, 0.5, 0.9] {
            assert_eq!(sk.quantile_cached(q), sk.quantile(q), "empty q={q}");
        }

        // Mid-stream.
        let values = sample_uniform_f64(0.0, 1000.0, 20_000, 0xACAC_0001);
        for &v in &values {
            sk.update(&v);
        }
        for q in [0.01f64, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let cached = sk.quantile_cached(q);
            let uncached = sk.quantile(q);
            assert_eq!(cached, uncached, "mid-stream q={q}");
        }

        // Post-merge (merge must invalidate).
        let mut other = KLL::<f64>::init_with_seed(200, 8, 43);
        for &v in &sample_uniform_f64(-5000.0, -4000.0, 5_000, 0xACAC_0002) {
            other.update(&v);
        }
        sk.merge(&other);
        for q in [0.1f64, 0.5, 0.9] {
            assert_eq!(sk.quantile_cached(q), sk.quantile(q), "post-merge q={q}");
        }

        // Post-clear.
        sk.clear();
        assert_eq!(sk.quantile_cached(0.5), 0.0, "cleared sketch is empty");
        assert_eq!(sk.count(), 0);
    }

    /// A cached query followed by more inserts must NOT serve the stale CDF:
    /// after pushing a heavy batch of large values, the cached median has to
    /// move up.
    #[test]
    fn cdf_cache_invalidated_by_subsequent_updates() {
        let mut sk = KLL::<f64>::init_with_seed(200, 8, 42);
        for &v in &sample_uniform_f64(0.0, 100.0, 10_000, 0xACAC_0003) {
            sk.update(&v);
        }
        let before = sk.quantile_cached(0.5);

        for _ in 0..10_000 {
            sk.update(&999_999.0);
        }
        let after = sk.quantile_cached(0.5);

        assert!(
            after > before + 50.0,
            "median {before} barely moved after 10k huge inserts -> stale cache served"
        );
    }

    // Crafted nested-serde bytes (the `HydraCounter::KLL` path, reachable with
    // untrusted input) with an out-of-range `k` must fail closed, not overflow
    // `compute_max_capacity` / drive a huge allocation.
    #[test]
    fn kllwire_serde_rejects_crafted_dimensions() {
        let wire = KLLWire::<f64> {
            items: vec![],
            levels: vec![0, 0],
            k: usize::MAX,
            m: 8,
            num_levels: 1,
            co: Coin::from_seed(1),
        };
        let bytes = rmp_serde::to_vec(&wire).expect("encode crafted wire");
        assert!(
            rmp_serde::from_slice::<KLL<f64>>(&bytes).is_err(),
            "an out-of-range k must be rejected by the nested serde decoder"
        );
    }

    // Direct reconstruction from portable wire state must be BIT-EXACT:
    // identical quantiles to the source sketch (unlike a lossy
    // replay-through-`update()` reconstruction).
    #[test]
    fn from_portable_state_reproduces_source_exactly() {
        let k = 200usize;
        let mut src = KLL::<f64>::init_kll(k as i32);
        for i in 0..200_000u32 {
            src.update(&((i as f64) * 0.0007 + 3.0));
        }
        // Extract the compacted portable form (proto contract: levels[0]==0,
        // levels[num_levels]==items.len()).
        let off = src.levels[0];
        let items: Vec<f64> = src.items[off..src.max_capacity].to_vec();
        let levels: Vec<usize> = (0..=src.num_levels).map(|h| src.levels[h] - off).collect();
        let n = src.num_levels;

        // (A) bit-exact: direct reconstruction reproduces the source exactly.
        let rebuilt = KLL::<f64>::from_portable_state(k, &items, &levels, n).unwrap();
        for &q in &[0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
            assert_eq!(
                rebuilt.quantile(q),
                src.quantile(q),
                "direct reconstruction quantile mismatch at q={q}"
            );
        }

        // An empty sketch round-trips to an empty sketch.
        let empty = KLL::<f64>::from_portable_state(k, &[], &[], 0).unwrap();
        assert_eq!(empty.num_levels, 1);
        assert_eq!(empty.levels[0], empty.max_capacity);
    }

    // Ensure each 64-bit chunk is consumed bit-by-bit before refilling.
    #[test]
    fn coin_bit_cache_behavior() {
        let seed = 0x0123_4567_89ab_cdef;
        let mut coin = Coin::from_seed(seed);
        let mut expected_state = Coin::normalize_seed(seed);

        for block in 0..3 {
            expected_state = Coin::normalize_seed(Coin::xorshift_mult64(expected_state));
            for bit in 0..64 {
                let expected = ((expected_state >> bit) & 1) != 0;
                assert_eq!(
                    coin.toss(),
                    expected,
                    "mismatch at block {block}, bit {bit}"
                );
            }
        }
    }

    // Zero seeds must map to a valid state and never fall back to zero.
    #[test]
    fn coin_state_never_zero() {
        let mut coin = Coin::from_seed(0);
        assert_ne!(coin.state, 0);

        for _ in 0..128 {
            coin.toss();
            assert_ne!(coin.state, 0);
        }
    }

    // Two sketches built with the same seed and fed the same input
    // sequence must produce byte-identical serialized state. Without
    // the seedable constructor this is impossible because Coin::new()
    // pulls from a non-deterministic source.
    #[test]
    fn seeded_sketches_are_byte_identical() {
        const SEED: u64 = 42;
        let values = sample_uniform_f64(0.0, 1_000_000.0, 5000, 7);

        let mut a: KLL<f64> = KLL::init_kll_with_seed(SKETCH_K, SEED);
        let mut b: KLL<f64> = KLL::init_kll_with_seed(SKETCH_K, SEED);
        for v in &values {
            a.update(v);
            b.update(v);
        }
        let bytes_a = a.serialize_to_bytes().expect("serialize a");
        let bytes_b = b.serialize_to_bytes().expect("serialize b");
        assert_eq!(
            bytes_a, bytes_b,
            "seeded KLL sketches with identical inputs produced different bytes"
        );
    }

    // Different seeds must (with high probability) drive different
    // compaction outcomes — proves the seed actually propagates into
    // toss() rather than being a no-op stored on the struct.
    #[test]
    fn different_seeds_produce_different_bytes() {
        let values = sample_uniform_f64(0.0, 1_000_000.0, 5000, 11);

        let mut a: KLL<f64> = KLL::init_kll_with_seed(SKETCH_K, 1);
        let mut b: KLL<f64> = KLL::init_kll_with_seed(SKETCH_K, 2);
        for v in &values {
            a.update(v);
            b.update(v);
        }
        let bytes_a = a.serialize_to_bytes().expect("serialize a");
        let bytes_b = b.serialize_to_bytes().expect("serialize b");
        assert_ne!(
            bytes_a, bytes_b,
            "seeds 1 and 2 should not produce identical sketch bytes"
        );
    }

    // clear() on a seeded sketch must re-seed from the stored seed,
    // not from the wall clock. Otherwise determinism evaporates after
    // the first window rotation in stateful aggregators that reuse a
    // KLL across windows.
    #[test]
    fn clear_preserves_seed_determinism() {
        const SEED: u64 = 1234;
        let values = sample_uniform_f64(0.0, 1_000_000.0, 3000, 13);

        let mut a: KLL<f64> = KLL::init_kll_with_seed(SKETCH_K, SEED);
        // Burn a partial window with unrelated data, then clear.
        let noise = sample_uniform_f64(0.0, 1_000_000.0, 1500, 99);
        for v in &noise {
            a.update(v);
        }
        a.clear();

        let mut b: KLL<f64> = KLL::init_kll_with_seed(SKETCH_K, SEED);
        for v in &values {
            a.update(v);
            b.update(v);
        }
        let bytes_a = a.serialize_to_bytes().expect("serialize a");
        let bytes_b = b.serialize_to_bytes().expect("serialize b");
        assert_eq!(
            bytes_a, bytes_b,
            "clear() lost determinism: post-clear sketch diverges from a fresh seeded sketch"
        );
    }

    const SKETCH_K: i32 = 200;

    // return element from input with given quantile
    fn quantile_from_sorted(data: &[f64], quantile: f64) -> f64 {
        assert!(!data.is_empty(), "data set must not be empty");
        if quantile <= 0.0 {
            return data[0];
        }
        if quantile >= 1.0 {
            return data[data.len() - 1];
        }
        let n = data.len();
        let idx = ((quantile * n as f64).ceil() as isize - 1).clamp(0, (n - 1) as isize) as usize;
        data[idx]
    }

    fn assert_quantiles_within_error(
        sketch: &KLL,
        sorted_truth: &[f64],
        quantiles: &[(f64, &str)],
        tolerance: f64,
        context: &str,
        sample_size: usize,
        seed: u64,
    ) {
        let cdf = sketch.cdf();
        for &(quantile, label) in quantiles {
            let lower_q = (quantile - tolerance).max(0.0);
            let upper_q = (quantile + tolerance).min(1.0);
            let truth_min = quantile_from_sorted(sorted_truth, lower_q);
            let truth_max = quantile_from_sorted(sorted_truth, upper_q);
            let estimate = cdf.query(quantile);
            assert!(
                (truth_min..=truth_max).contains(&estimate),
                "{label} exceeded tolerance: context={context}, sample_size={sample_size}, seed=0x{seed:08x}, \
                quantile={quantile:.4}, truth_min={truth_min:.4}, truth_max={truth_max:.4}, \
                estimate={estimate:.4}, tolerance={tolerance:.4}, total_length={}",
                sorted_truth.len()
            );
        }
    }

    #[test]
    fn test_data_input_api() {
        let mut kll = KLL::init_kll(128);

        // Test with different numeric types
        kll.update_data_input(&DataInput::I32(10)).unwrap();
        kll.update_data_input(&DataInput::I64(20)).unwrap();
        kll.update_data_input(&DataInput::F64(30.5)).unwrap();
        kll.update_data_input(&DataInput::F32(40.2)).unwrap();
        kll.update_data_input(&DataInput::U32(50)).unwrap();

        // Query quantiles
        let cdf = kll.cdf();
        // kll.print_compactors();
        let median = cdf.query(0.5);

        // Median should be 30.5
        assert!(median > 20.0 && median < 40.2, "Median = {median}");

        // Test error handling for non-numeric input
        let result = kll.update_data_input(&DataInput::String("not a number".to_string()));
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "KLL sketch only accepts numeric inputs"
        );
    }

    #[test]
    fn test_forced_compact() {
        // force compaction to happen with small k/m
        let mut kll = KLL::init(3, 3);
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(10.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(20.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(30.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(40.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(50.0)).unwrap();
        // kll.print_compactors();
        let cdf = kll.cdf();
        // cdf.print_entries();
        let median = cdf.query(0.5);
        // only 30 and 40 is possible
        assert!(median == 30.0 || median == 40.0, "Median = {median}");
    }

    #[test]
    fn test_no_compact() {
        // no compaction should happen
        let mut kll = KLL::init_kll(8);
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(10.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(20.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(30.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(40.0)).unwrap();
        // kll.print_compactors();
        kll.update_data_input(&DataInput::F64(50.0)).unwrap();
        // kll.print_compactors();

        // Query quantiles
        let cdf = kll.cdf();
        // cdf.print_entries();
        // kll.print_compactors();
        let median = cdf.query(0.5);
        // Median should be 30
        assert!(median == 30.0, "Median = {median}");
    }

    #[test]
    fn merge_preserves_quantiles_within_tolerance() {
        const TOLERANCE: f64 = 0.02;
        const QUANTILES: &[(f64, &str)] = &[
            (0.0, "min"),
            (0.10, "p10"),
            (0.25, "p25"),
            (0.50, "p50"),
            (0.75, "p75"),
            (0.90, "p90"),
            (1.0, "max"),
        ];

        let values = sample_uniform_f64(1_000_000.0, 10_000_000.0, 10_000, 0xC0FFEE);
        let mut sketch_a = KLL::init_kll(SKETCH_K);
        let mut sketch_b = KLL::init_kll(SKETCH_K);

        for (idx, value) in values.iter().copied().enumerate() {
            if idx % 2 == 0 {
                sketch_a.update_data_input(&DataInput::F64(value)).unwrap();
            } else {
                sketch_b.update_data_input(&DataInput::F64(value)).unwrap();
            }
        }

        sketch_a.merge(&sketch_b);

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_quantiles_within_error(
            &sketch_a,
            &sorted,
            QUANTILES,
            TOLERANCE,
            "merge",
            values.len(),
            0x00C0_FFEE,
        );
    }

    // Note on the count assertion: `count()` is only *approximately* N even
    // for a sketch built by plain `update()` calls with no merge involved —
    // odd-sized levels can resolve to `2*ceil(pop/2)` or `2*floor(pop/2)`
    // depending on the compaction coin, off by +/-1 per affected level (see
    // `generic_kll_i64_sanity` above, which already documents this and
    // budgets a 5% tolerance). That's an inherent property of
    // randomized-halving KLL. Merging into an *empty* target, though, is a
    // pure structural no-op — `other`'s levels already each satisfy their
    // own capacity, so folding them into an empty self triggers no further
    // compaction at all — so `dst.count()` must come out **exactly equal**
    // to `src.count()`, whatever that value is.
    #[test]
    fn merge_into_empty_target_preserves_weight_issue_68_repro() {
        let mut src = KLL::<f64>::init_kll(SKETCH_K);
        for i in 1..=1000u32 {
            src.update(&(i as f64));
        }

        // Sanity: plain inserts (no merge) track N closely; confirms the
        // source sketch itself is healthy before we exercise merge.
        let src_count = src.count();
        assert!(
            (980..=1020).contains(&src_count),
            "source sketch count before merge should track N=1000 closely, got {src_count}"
        );

        let mut dst = KLL::<f64>::init_kll(SKETCH_K);
        assert_eq!(dst.count(), 0, "target must start empty for this repro");

        dst.merge(&src);

        assert_eq!(
            dst.count(),
            src_count,
            "merge into an empty target must preserve total weight EXACTLY \
             (the old item-replay merge rescaled this down to src's \
             retained-item count, e.g. 1000 -> ~350)"
        );

        let median = dst.quantile(0.5);
        // True median of 1..=1000 is 500/500.5. KLL's guaranteed rank error
        // at k=200 is well under 5% of the domain for this small, orderly
        // input; allow a generous +/-5% band.
        assert!(
            (475.0..=525.0).contains(&median),
            "merged median drifted outside tolerance: median={median} \
             (pre-fix this drifted to ~700)"
        );
    }

    // General case: merging two NON-empty KLLs must still preserve total
    // count (within the same inherent +/-1-per-compacted-level rounding
    // budget as ordinary inserts — see the note above) and produce
    // quantile estimates consistent with a reference built from the union
    // of both inputs' raw data (within KLL's accuracy bound). This guards
    // against a fix that only special-cases the empty-target repro above
    // without being a genuinely correct weighted merge for the general
    // case.
    #[test]
    fn merge_two_nonempty_sketches_preserves_weight_and_quantiles() {
        const TOLERANCE: f64 = 0.03;
        const QUANTILES: &[(f64, &str)] = &[
            (0.0, "min"),
            (0.10, "p10"),
            (0.25, "p25"),
            (0.50, "p50"),
            (0.75, "p75"),
            (0.90, "p90"),
            (1.0, "max"),
        ];

        // Each side gets enough volume (and a different distribution) to
        // force real compaction, i.e. non-trivial retained-item weights at
        // multiple levels on BOTH operands before they're merged.
        let values_a = sample_uniform_f64(0.0, 1_000_000.0, 50_000, 0xA11CE);
        let values_b = sample_zipf_f64(0.0, 1_000_000.0, 8_192, 1.1, 50_000, 0xB0B);

        let mut a = KLL::<f64>::init_kll(SKETCH_K);
        for v in &values_a {
            a.update(v);
        }
        let mut b = KLL::<f64>::init_kll(SKETCH_K);
        for v in &values_b {
            b.update(v);
        }

        let count_a = a.count();
        let count_b = b.count();
        // Sanity: plain-insert counts track N closely (see note above).
        assert!(
            (count_a as f64 - values_a.len() as f64).abs() / (values_a.len() as f64) < 0.03,
            "sketch a count before merge diverged from N: count={count_a}, n={}",
            values_a.len()
        );
        assert!(
            (count_b as f64 - values_b.len() as f64).abs() / (values_b.len() as f64) < 0.03,
            "sketch b count before merge diverged from N: count={count_b}, n={}",
            values_b.len()
        );

        a.merge(&b);

        let merged_count = a.count() as f64;
        let expected_count = (count_a + count_b) as f64;
        assert!(
            (merged_count - expected_count).abs() / expected_count < 0.03,
            "merging two non-empty sketches must preserve total weight (within the \
             same rounding budget as ordinary inserts): merged={merged_count}, \
             expected~={expected_count} (count_a={count_a}, count_b={count_b})"
        );

        let mut union: Vec<f64> = values_a.iter().chain(values_b.iter()).copied().collect();
        union.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_quantiles_within_error(
            &a,
            &union,
            QUANTILES,
            TOLERANCE,
            "merge_two_nonempty",
            union.len(),
            0xA11C_E0B0,
        );
    }

    #[test]
    fn cdf_handles_empty_sketch() {
        let sketch = KLL::<f64>::init_kll(64);
        let cdf = sketch.cdf();
        assert_eq!(cdf.quantile(123.0), 0.0);
        assert_eq!(cdf.query(0.5), 0.0);
        assert_eq!(cdf.query_li(0.5), 0.0);
    }

    #[test]
    fn kll_round_trip_rmp() {
        let mut sketch = KLL::init_kll(256);
        let samples = sample_uniform_f64(0.0, 1_000_000.0, 5_000, 0xDEAD_BEEF);
        for value in &samples {
            sketch.update_data_input(&DataInput::F64(*value)).unwrap();
        }

        let bytes = sketch.serialize_to_bytes().expect("serialize KLL with rmp");
        assert!(!bytes.is_empty(), "serialized bytes should not be empty");

        let restored = KLL::deserialize_from_bytes(&bytes).expect("deserialize KLL with rmp");
        assert_eq!(sketch.k, restored.k);
        assert_eq!(sketch.m, restored.m);
        assert_eq!(sketch.num_levels, restored.num_levels);
        assert_eq!(sketch.top_height, restored.top_height);
        assert_eq!(sketch.level0_capacity, restored.level0_capacity);
        assert_eq!(
            sketch.levels, restored.levels,
            "level boundaries changed after round-trip"
        );

        let s_start = sketch.levels[0];
        let s_end = sketch.levels[sketch.num_levels];
        let r_start = restored.levels[0];
        let r_end = restored.levels[restored.num_levels];
        assert_eq!(
            &sketch.items[s_start..s_end],
            &restored.items[r_start..r_end],
            "packed items changed after round-trip"
        );

        let quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
        let original_cdf = sketch.cdf();
        let restored_cdf = restored.cdf();
        for &q in &quantiles {
            assert!(
                (original_cdf.query(q) - restored_cdf.query(q)).abs() < f64::EPSILON,
                "quantile mismatch at p={q}: original={}, restored={}",
                original_cdf.query(q),
                restored_cdf.query(q)
            );
        }
    }

    // Sanity pass for the generic KLL<T> specialization: exercise update/cdf/merge
    // on KLL<i64> to confirm the non-default T path compiles and produces sensible
    // quantiles. Tolerance is loose — this is a smoke test, not an accuracy bound.
    #[test]
    fn generic_kll_i64_sanity() {
        let mut sketch = KLL::<i64>::init_kll(200);
        let n: i64 = 20_000;
        for v in 1..=n {
            sketch.update(&v);
        }

        // Weighted count after random compactions is approximately n but not exact.
        let count = sketch.count() as f64;
        assert!(
            (count - n as f64).abs() / (n as f64) < 0.05,
            "count={count} diverged from n={n}"
        );

        let cdf = sketch.cdf();
        let p50 = cdf.query(0.5);
        let p90 = cdf.query(0.9);
        let tol = n as f64 * 0.02;
        assert!(
            (p50 - (n as f64 * 0.5)).abs() < tol,
            "p50={p50} out of range for n={n}"
        );
        assert!(
            (p90 - (n as f64 * 0.9)).abs() < tol,
            "p90={p90} out of range for n={n}"
        );

        // Merge between KLL<i64> sketches should work and preserve roughly the
        // same quantiles.
        let mut a = KLL::<i64>::init_kll(200);
        let mut b = KLL::<i64>::init_kll(200);
        for v in 1..=n {
            if v % 2 == 0 {
                a.update(&v);
            } else {
                b.update(&v);
            }
        }
        a.merge(&b);
        let merged_p50 = a.cdf().query(0.5);
        assert!(
            (merged_p50 - (n as f64 * 0.5)).abs() < tol,
            "merged p50={merged_p50} out of range"
        );

        // Serialization round-trip for the generic specialization.
        let bytes = a.serialize_to_bytes().expect("serialize KLL<i64>");
        let restored = KLL::<i64>::deserialize_from_bytes(&bytes).expect("deserialize KLL<i64>");
        assert_eq!(a.count(), restored.count());
    }

    #[test]
    fn bulk_update_equivalent_to_loop_and_empty_is_noop() {
        // Empty must be no-op and keep cache/coin.
        let mut sk = KLL::<f64>::init_with_seed(200, 8, 42);
        for &v in &[1.0, 2.0, 3.0] {
            sk.update(&v);
        }
        let cnt_before = sk.count();
        let q_before = sk.quantile_cached(0.5);
        let bytes_before = sk.serialize_to_bytes().unwrap();
        sk.bulk_update(&[]);
        assert_eq!(sk.count(), cnt_before);
        assert_eq!(sk.quantile(0.5), q_before);
        assert_eq!(sk.serialize_to_bytes().unwrap(), bytes_before);

        // Bulk vs loop equivalence for generic and seeded determinism.
        let vals = sample_uniform_f64(0.0, 1000.0, 20_000, 0xBEEF_1234);
        let mut a = KLL::<f64>::init_with_seed(200, 8, 99);
        let mut b = KLL::<f64>::init_with_seed(200, 8, 99);
        for v in &vals {
            a.update(v);
        }
        b.bulk_update(&vals);
        assert_eq!(a.count(), b.count());
        assert_eq!(
            a.serialize_to_bytes().unwrap(),
            b.serialize_to_bytes().unwrap()
        );
        for q in [0.1, 0.5, 0.9] {
            assert_eq!(a.quantile(q), b.quantile(q), "q={q}");
        }

        // DataInput batch equivalence.
        let di_vals: Vec<DataInput> = vals[..100].iter().map(|v| DataInput::F64(*v)).collect();
        let mut c = KLL::<f64>::init_with_seed(200, 8, 77);
        let mut d = KLL::<f64>::init_with_seed(200, 8, 77);
        for v in &di_vals {
            c.update_data_input(v).unwrap();
        }
        d.bulk_update_data_input(&di_vals).unwrap();
        assert_eq!(
            c.serialize_to_bytes().unwrap(),
            d.serialize_to_bytes().unwrap()
        );

        // Non-numeric stops on first error, same as per-element.
        let bad = vec![
            DataInput::F64(1.0),
            DataInput::String("x".into()),
            DataInput::F64(2.0),
        ];
        let mut e = KLL::<f64>::init_with_seed(200, 8, 11);
        assert!(e.bulk_update_data_input(&bad).is_err());
        assert_eq!(e.count(), 1);
    }
}
