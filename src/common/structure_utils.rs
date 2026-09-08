//! Common data structure that is served as basic building block
//! Vector1D:
//! Vector2D:
//! Vector3D:
//! CommonHeap:
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::PRECOMPUTED_SAMPLE;
use crate::common::hash::MATRIX_MAX_ROWS;
use crate::common::precompute_sample::PRECOMPUTED_SAMPLE_LEN;

/// Helper trait for converting sketch counter types to f64 for median calculation.
pub trait ToF64 {
    /// Converts the value into `f64`.
    fn to_f64(self) -> f64;
}

impl ToF64 for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToF64 for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToF64 for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToF64 for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Row-level NitroSketch sampling state.
///
/// NitroSketch samples **per row slot**: a `d`-row sketch turns an `n`-update
/// stream into `n * d` slots, each admitted independently with probability
/// `p`, and an admitted slot is written with weight `1/p` (stochastically
/// rounded, see [`Nitro::admitted_delta`]). Skip distances are drawn from a
/// shared table of `ln(1 - u)` values; `to_skip` carries across update
/// boundaries so the schedule is one continuous geometric stream.
///
/// Reference: <https://dl.acm.org/doi/10.1145/3341302.3342076>.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Nitro {
    /// Whether Nitro sampling is enabled.
    pub is_nitro_mode: bool,
    sampling_rate: f64,
    /// Remaining row slots to skip before the next admission.
    pub to_skip: usize,
    /// Precomputed `1 / ln(1 - sampling_rate)`, the inverse-CDF scale that
    /// turns a table entry into a `Geometric(p)` draw.
    inv_ln_one_minus_p: f64,
    /// Integer part of the weight one admitted slot carries.
    pub delta: u64,
    idx: usize,
    /// Unused; retained so the serialized field order is unchanged.
    mask: usize,
    /// splitmix64 state for the stochastic-rounding draw, advanced once per
    /// admitted slot whose `1/p` is not an integer.
    ///
    /// Serialized (with a default for payloads written before this field
    /// existed), because it is part of what the sketch will do next: a decoded
    /// sketch that restarted this stream from a fixed constant would emit a
    /// different weight sequence than the uninterrupted run.
    #[serde(default = "default_rounding_state")]
    rounding_state: u64,
}

/// Seed the rounding stream starts from when it is not seeded explicitly, and
/// the value a payload written before `rounding_state` existed decodes to.
const ROUNDING_SEED: u64 = 0x2545_F491_4F6C_DD1D;

fn default_rounding_state() -> u64 {
    ROUNDING_SEED
}

/// The complete sampling state of a [`Nitro`]: everything that decides which
/// slots it admits next and at what weight.
///
/// Restoring one reproduces the uninterrupted run exactly. The legacy
/// [`Nitro::get_ctx`] / [`Nitro::commit_ctx`] pair carries only the first two
/// fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NitroContext {
    /// Cursor into the shared skip table.
    pub table_cursor: usize,
    /// Row slots still to skip before the next admission.
    pub to_skip: usize,
    /// splitmix64 state of the stochastic-rounding stream.
    pub rounding_state: u64,
}

/// Rows one update's admission walk can report without allocating.
/// [`MATRIX_MAX_ROWS`] is the widest matrix the hash family supports, so the
/// fast path never touches the heap.
pub(crate) type AdmittedRows = SmallVec<[(usize, u64); MATRIX_MAX_ROWS]>;

impl Default for Nitro {
    fn default() -> Self {
        Self {
            is_nitro_mode: false,
            sampling_rate: 0.0,
            to_skip: 0,
            inv_ln_one_minus_p: 0.0, // not used unless Nitro mode is enabled
            delta: 0,
            idx: 0,
            mask: PRECOMPUTED_SAMPLE_LEN - 1,
            rounding_state: ROUNDING_SEED,
        }
    }
}

impl Nitro {
    /// Creates an enabled Nitro state sampling at `rate`.
    ///
    /// The first skip is drawn immediately, from table entry 0. Without it
    /// `to_skip` would be 0 for the first row slot the sketch ever sees, so
    /// that slot would be admitted unconditionally — which is not
    /// `Bernoulli(p)` for any `p < 1`.
    ///
    /// # Panics
    ///
    /// If `rate` is NaN or outside `(0.0, 1.0]`.
    pub fn init_nitro(rate: f64) -> Self {
        Self::seeded_from(rate, None)
    }

    /// [`Nitro::init_nitro`] with both sources of randomness selected by
    /// `seed`: the skip schedule starts at table offset `seed % table_len`,
    /// and the rounding stream starts at `seed`.
    ///
    /// The unseeded constructor always starts at offset 0, so every sketch at
    /// a given rate admits the same subset; seeding is what makes a battery
    /// over independent trials possible while still reproducing exactly.
    ///
    /// # Panics
    ///
    /// If `rate` is NaN or outside `(0.0, 1.0]`.
    pub fn init_nitro_seeded(rate: f64, seed: u64) -> Self {
        Self::seeded_from(rate, Some(seed))
    }

    fn seeded_from(rate: f64, seed: Option<u64>) -> Self {
        assert!(
            !rate.is_nan() && rate > 0.0 && rate <= 1.0,
            "sample_rate must be within (0.0, 1.0]"
        );
        let inv_ln = if (rate - 1.0).abs() <= f64::EPSILON {
            0.0 // Not used for full sampling
        } else {
            1.0 / (1.0 - rate).ln()
        };
        let mut nitro = Self {
            is_nitro_mode: true,
            sampling_rate: rate,
            to_skip: 0,
            inv_ln_one_minus_p: inv_ln,
            delta: 0,
            idx: seed.map_or(0, |s| (s % PRECOMPUTED_SAMPLE_LEN as u64) as usize),
            mask: PRECOMPUTED_SAMPLE_LEN - 1,
            rounding_state: match seed {
                Some(0) | None => ROUNDING_SEED,
                Some(s) => s,
            },
        };
        nitro.delta = nitro.scaled_increment(1);
        // The cursor is in place, so this reads the seed's own stretch of the
        // table rather than entry 0's.
        nitro.draw_geometric();
        nitro
    }

    /// The cursor actually used to index the skip table, folded into range so
    /// a value decoded from an older payload cannot index out of bounds.
    #[inline(always)]
    fn cursor(&self) -> usize {
        self.idx % PRECOMPUTED_SAMPLE_LEN
    }

    /// Advances the cursor by one, wrapping at the table's length.
    #[inline(always)]
    fn next_cursor(&mut self) {
        self.idx = (self.cursor() + 1) % PRECOMPUTED_SAMPLE_LEN;
    }

    /// The weight one admitted row slot carries: stochastic rounding of `1/p`.
    ///
    /// With `q = floor(1/p)` and `r = frac(1/p)`,
    ///
    /// ```text
    ///   W = q + Bernoulli(r)      E[W] = 1/p      Var[W] = r (1 - r)
    /// ```
    ///
    /// so the estimator is unbiased at every rate, not only at rates whose
    /// reciprocal is an integer. Rounding the same way every time is a bias
    /// rather than noise: `ceil(1/0.3) = 4` puts every estimate 20% high.
    ///
    /// The Bernoulli is drawn from `rounding_state`, advanced once per
    /// admitted slot and independent of the skip cursor. Drawing it per slot
    /// rather than as a function of position is what makes `E[W] = 1/p` hold
    /// for each key separately.
    ///
    /// When `1/p` is an integer `r` is zero, **no draw is consumed**, and this
    /// returns `delta` unchanged — so `p ∈ {1, 1/2, 1/10, 1/100}` emit exactly
    /// the weights they would without the correction and their rounding stream
    /// never advances.
    ///
    /// Unbiasedness is exact given a uniform draw; that the splitmix64 stream
    /// supplies one is the same modelling assumption the shared skip table
    /// already rests on.
    #[inline(always)]
    pub fn admitted_delta(&mut self) -> u64 {
        if self.is_full_sampling() {
            return self.delta;
        }
        let exact = 1.0 / self.sampling_rate;
        let frac = exact - exact.floor();
        if frac <= 0.0 {
            return self.delta;
        }
        self.delta + u64::from(self.next_rounding_uniform() < frac)
    }

    /// Next uniform in `[0, 1)` from the rounding stream (splitmix64).
    #[inline(always)]
    fn next_rounding_uniform(&mut self) -> f64 {
        self.rounding_state = self.rounding_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rounding_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Draws the next geometric skip distance at the configured rate.
    ///
    /// The table holds `ln(1 - u)` for a fixed stream of uniforms; multiplying
    /// by `inv_ln_one_minus_p = 1 / ln(1 - p)` makes each entry an inverse-CDF
    /// draw of `Geometric(p)` on `{0, 1, ...}`, so `E[skip] = (1-p)/p`. The
    /// `floor` is load-bearing: the caller's `+1` stride supplies the admitted
    /// slot itself, and `ceil` would add one to every distance.
    ///
    /// At full sampling the distance is always 0 and the cursor does not move.
    #[inline]
    pub fn draw_geometric(&mut self) {
        if self.is_full_sampling() {
            self.to_skip = 0;
            return;
        }
        self.to_skip =
            (PRECOMPUTED_SAMPLE[self.cursor()] * self.inv_ln_one_minus_p).floor() as usize;
        self.next_cursor();
    }

    /// Consumes one update's worth of row slots from the sampling schedule and
    /// appends the `(row, weight)` pairs that were admitted to `out`.
    ///
    /// `out` is cleared first. `to_skip` is the number of slots still to skip
    /// and is carried across update boundaries, so an update can admit several
    /// rows or none, and the remainder of a long skip survives into the next
    /// call. At `p = 1` every row is admitted with weight 1.
    ///
    /// The walk is saturating: at a rate small enough for one skip to exceed
    /// `usize::MAX` the addition would otherwise overflow.
    pub(crate) fn admit_rows(&mut self, rows: usize, out: &mut AdmittedRows) {
        out.clear();
        if rows == 0 {
            return;
        }
        if self.is_full_sampling() {
            out.extend((0..rows).map(|r| (r, 1u64)));
            return;
        }
        let mut row = self.to_skip;
        while row < rows {
            let weight = self.admitted_delta();
            out.push((row, weight));
            self.draw_geometric();
            // `+ 1` for the admitted slot itself, then the freshly drawn skip.
            row = row.saturating_add(1).saturating_add(self.to_skip);
        }
        // Whatever is left of the skip once this update's window is exhausted.
        self.to_skip = row - rows;
    }

    #[inline(always)]
    /// Decrements the current skip counter by one.
    pub fn reduce_to_skip(&mut self) {
        self.to_skip -= 1;
    }

    #[inline(always)]
    /// Decrements the current skip counter by `c`.
    pub fn reduce_to_skip_by_count(&mut self, c: usize) {
        self.to_skip -= c;
    }

    #[inline(always)]
    /// Returns the configured sampling rate.
    pub fn get_sampling_rate(&self) -> f64 {
        self.sampling_rate
    }

    #[inline(always)]
    /// The integer part of the weight one admitted slot carries. The
    /// fractional remainder is paid per slot by [`Nitro::admitted_delta`].
    pub fn scaled_increment(&self, weight: u64) -> u64 {
        if self.is_full_sampling() {
            weight
        } else {
            ((weight as f64) / self.sampling_rate).floor() as u64
        }
    }

    #[inline(always)]
    /// Whether every slot is admitted, i.e. `p == 1`.
    pub fn is_full_sampling(&self) -> bool {
        (self.sampling_rate - 1.0).abs() <= f64::EPSILON
    }

    /// The complete sampling state. Restoring it with
    /// [`Nitro::restore_context`] continues the admission and weight sequence
    /// exactly where it was taken.
    #[inline(always)]
    pub fn context(&self) -> NitroContext {
        NitroContext {
            table_cursor: self.cursor(),
            to_skip: self.to_skip,
            rounding_state: self.rounding_state,
        }
    }

    /// Restores a state captured by [`Nitro::context`].
    #[inline(always)]
    pub fn restore_context(&mut self, ctx: NitroContext) {
        self.idx = ctx.table_cursor;
        self.to_skip = ctx.to_skip;
        self.rounding_state = ctx.rounding_state;
    }

    #[inline(always)]
    /// Legacy skip-state snapshot: `(cursor, 1/ln(1-p), to_skip, unused)`.
    ///
    /// This is **not** the full sampling state — it omits the
    /// stochastic-rounding stream, so a sketch restored from it emits a
    /// different weight sequence at any rate whose reciprocal is not an
    /// integer. Use [`Nitro::context`] instead.
    pub fn get_ctx(&self) -> (usize, f64, usize, usize) {
        (self.idx, self.inv_ln_one_minus_p, self.to_skip, self.mask)
    }

    #[inline(always)]
    /// Restores the legacy skip state captured by [`Nitro::get_ctx`], leaving
    /// the rounding stream where it is. See [`Nitro::restore_context`].
    pub fn commit_ctx(&mut self, idx: usize, to_skip: usize) {
        self.idx = idx;
        self.to_skip = to_skip;
    }
}

/// Compute median from a mutable slice of f64 values (inline helper)
/// This is used by query_median_with_custom_hash for HydraCounter queries
#[inline(always)]
pub fn compute_median_inline_f64(values: &mut [f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        1 => values[0],
        2 => (values[0] + values[1]) / 2.0,
        // starting here is an assumption that LLVM and compiler
        // will load var into register and perform simple register swap
        // no heavy sort or memory swap
        3 => {
            let (mut v0, mut v1, v2) = (values[0], values[1], values[2]);
            // ensure v0 is smaller than v1
            if v0 > v1 {
                std::mem::swap(&mut v0, &mut v1);
            }
            // ensure v1 is smaller than v2, and ignore the actual v2 value
            if v1 > v2 {
                v1 = v2;
            }
            // ensure v1 is still greater than v0
            if v0 > v1 {
                v1 = v0;
            }
            v1
        }
        4 => {
            let (mut v0, mut v1, mut v2, mut v3) = (values[0], values[1], values[2], values[3]);
            // ensure the order of v0 and v1
            if v0 > v1 {
                std::mem::swap(&mut v0, &mut v1);
            }
            // ensure the order of v2 and v3
            if v2 > v3 {
                std::mem::swap(&mut v2, &mut v3);
            }
            // the smaller of v0 and v2 will be smaller than v1 anyway
            // ignore the smaller one, which will be min (dropped)
            if v0 > v2 {
                v2 = v0;
            }
            // the greater of v1 and v3 will be greater than v2 anyway
            // ignore the greeater one, which will be max (dropped)
            if v1 > v3 {
                v1 = v3;
            }
            (v1 + v2) / 2.0
        }
        5 => {
            let (mut v0, mut v1, mut v2, mut v3, mut v4) =
                (values[0], values[1], values[2], values[3], values[4]);
            // ensure the order of v0 and v1
            if v0 > v1 {
                std::mem::swap(&mut v0, &mut v1);
            }
            // ensure the order of v3 and v4
            if v3 > v4 {
                std::mem::swap(&mut v3, &mut v4);
            }
            // the smaller of v0 v3 will be smaller than v1 v4 and the other
            // smaller than 3 value, so not median of 5
            if v0 > v3 {
                v3 = v0;
            }
            // the greater of v1 v4 will be greater than v0 v3 and the other
            // greater than 3 value, so not median of 5
            if v1 > v4 {
                v1 = v4;
            }
            // median of 5 is reduced to median of v1 v2 v3
            // v0 and v4 will not change the order
            // v0 will be one of the two smallest
            // v4 will be one of the two greatest
            // safely ignored
            if v1 > v2 {
                std::mem::swap(&mut v1, &mut v2);
            }
            if v2 > v3 {
                v2 = v3;
            }
            if v1 > v2 {
                v2 = v1;
            }
            v2
        }
        _ => {
            values.sort_unstable_by(f64::total_cmp);
            let mid = values.len() / 2;
            if values.len() % 2 == 1 {
                values[mid]
            } else {
                (values[mid - 1] + values[mid]) / 2.0
            }
        }
    }
}

/// Structural and distributional tests for the row-level sampler.
///
/// These live here rather than in `tests/e2e_nitro.rs` because they read the
/// cursor and the skip counter, which are private: exposing them as public
/// accessors would put a test-only surface in the crate's API.
#[cfg(test)]
mod nitro_tests {
    use super::*;

    /// Gaussian quantile for the distributional bands below. Two-sided failure
    /// 6.3e-5 per assertion.
    const Z: f64 = 4.0;

    /// `Var[W * 1{admitted}]` for one row slot at rate `p`, with the weight
    /// stochastically rounded:
    ///
    /// ```text
    ///   Var = (1-p)/p + p r (1-r),   r = frac(1/p)
    /// ```
    fn slot_variance(p: f64) -> f64 {
        let r = (1.0 / p).fract();
        (1.0 - p) / p + p * r * (1.0 - r)
    }

    /// The cursor advances by one per draw and wraps at the table's length.
    #[test]
    fn the_skip_cursor_advances_by_one_per_draw_and_wraps_at_the_table_length() {
        let len = PRECOMPUTED_SAMPLE_LEN;
        assert!(len > 1, "the skip table must be non-trivial");

        // Construction draws once, so an unseeded sampler is already at 1.
        let mut nitro = Nitro::init_nitro(0.5);
        assert_eq!(
            nitro.cursor(),
            1,
            "init_nitro must draw the first skip from entry 0 and leave the cursor at 1"
        );

        let mut seen = std::collections::HashSet::new();
        for _ in 0..4_096 {
            seen.insert(nitro.cursor());
            nitro.draw_geometric();
        }
        assert_eq!(
            seen.len(),
            4_096,
            "4096 draws must touch 4096 distinct table entries; the cursor is stuck \
             or aliasing"
        );

        let mut wrapper = Nitro::init_nitro(0.5);
        for _ in 0..len - 1 {
            wrapper.draw_geometric();
        }
        assert_eq!(
            wrapper.cursor(),
            0,
            "after exactly {len} draws (one at construction) the cursor must be back \
             at entry 0"
        );

        // A cursor decoded from an older payload can be anywhere, including
        // past the table. Reading must stay in bounds and rejoin the cycle.
        let mut hostile = Nitro::init_nitro(0.5);
        hostile.commit_ctx(usize::MAX, 0);
        assert!(
            hostile.cursor() < len,
            "an out-of-range cursor must be folded back into the table"
        );
        hostile.draw_geometric(); // must not panic

        // Full sampling never consumes an entry.
        let mut full = Nitro::init_nitro(1.0);
        for _ in 0..64 {
            full.draw_geometric();
            assert_eq!(full.to_skip, 0, "full sampling must never skip anything");
        }
        assert_eq!(full.cursor(), 0, "full sampling must not move the cursor");
    }

    /// The skip distances must be `Geometric(p)` at the **configured** rate.
    ///
    /// `floor(ln(1-u) / ln(1-p))` is exactly `Geometric(p)` on `{0, 1, ...}`,
    /// so `E[skip] = (1-p)/p` and `Var[skip] = (1-p)/p^2`. The table is fixed,
    /// so the measurement is deterministic; the threshold is not. It is `z`
    /// standard errors of the mean of `n` such draws, which is what the table
    /// would have to satisfy if it really were such a sample.
    #[test]
    fn the_geometric_skip_mean_matches_the_configured_rate() {
        const DRAWS: usize = 20_000;

        for &rate in &[0.5f64, 0.3, 0.1, 0.07, 0.01] {
            let mut nitro = Nitro::init_nitro(rate);
            let mut total = 0f64;
            for _ in 0..DRAWS {
                total += nitro.to_skip as f64;
                nitro.draw_geometric();
            }
            let observed = total / DRAWS as f64;
            let expected = (1.0 - rate) / rate;
            let standard_error = ((1.0 - rate) / (rate * rate) / DRAWS as f64).sqrt();
            assert!(
                (observed - expected).abs() <= Z * standard_error,
                "rate={rate}: mean skip {observed:.4} vs (1-p)/p = {expected:.4}, off by \
                 {:.4} > z*se = {Z}*{standard_error:.4}. A schedule locked to a different \
                 rate shows up here first.",
                (observed - expected).abs()
            );
        }
    }

    /// A reciprocal-integer rate must not consume a rounding draw; a
    /// non-integer one must, and the weights it emits must average `1/p`.
    #[test]
    fn stochastic_rounding_only_draws_where_the_reciprocal_is_not_an_integer() {
        for &rate in &[1.0f64, 0.5, 0.1, 0.01] {
            let mut nitro = Nitro::init_nitro(rate);
            let before = nitro.rounding_state;
            let expected = if rate >= 1.0 {
                1
            } else {
                (1.0 / rate).round() as u64
            };
            let weights: Vec<u64> = (0..64).map(|_| nitro.admitted_delta()).collect();
            assert!(
                weights.iter().all(|w| *w == expected),
                "rate={rate}: 1/p is an integer, so every admitted weight must be \
                 exactly {expected}; got {:?}",
                &weights[..8]
            );
            assert_eq!(
                nitro.rounding_state, before,
                "rate={rate}: an integer 1/p must not advance the rounding stream"
            );
        }

        for &rate in &[0.3f64, 0.07] {
            let mut nitro = Nitro::init_nitro(rate);
            let exact = 1.0 / rate;
            let floor = exact.floor() as u64;
            let frac = exact.fract();
            const DRAWS: usize = 50_000;
            let weights: Vec<u64> = (0..DRAWS).map(|_| nitro.admitted_delta()).collect();
            assert!(
                weights.contains(&floor) && weights.contains(&(floor + 1)),
                "rate={rate}: both {floor} and {} must occur; a frozen dither emits one \
                 of them forever, which is the +20% bias at p=0.3",
                floor + 1
            );
            let mean = weights.iter().map(|w| *w as f64).sum::<f64>() / DRAWS as f64;
            let se = (frac * (1.0 - frac) / DRAWS as f64).sqrt();
            assert!(
                (mean - exact).abs() <= Z * se,
                "rate={rate}: mean weight {mean:.6} vs 1/p = {exact:.6}, off by {:.6} > \
                 z*se = {:.6}",
                (mean - exact).abs(),
                Z * se
            );
        }
    }

    /// The **first** row slot a sampler ever sees must be admitted with
    /// probability `p`, not unconditionally.
    ///
    /// One update into a one-row sketch is a single slot, so the whole
    /// estimate is `W * 1{admitted}`: mean `1`, variance `slot_variance(p)`.
    /// Each seed is one independent draw, and the acceptance band is
    /// `z * sqrt(Var / T)` on the mean over `T` seeds.
    ///
    /// A sampler that starts at `to_skip = 0` admits every first slot, so its
    /// mean is `1/p` — 2 at `p = 0.5`, 100 at `p = 0.01`. Both are hundreds of
    /// standard errors outside the band.
    #[test]
    fn the_first_row_slot_is_admitted_with_probability_p() {
        const TRIALS: usize = 4_000;
        // Each trial reads at most two table entries, so a stride of 16 keeps
        // every trial on its own disjoint stretch.
        const SEED_STRIDE: u64 = 16;

        for &rate in &[0.5f64, 0.3, 0.1, 0.07, 0.01] {
            let mut admitted_mass = 0f64;
            let mut admissions = 0usize;
            let mut out = AdmittedRows::new();
            for trial in 0..TRIALS {
                let mut nitro = Nitro::init_nitro_seeded(rate, 1 + trial as u64 * SEED_STRIDE);
                nitro.admit_rows(1, &mut out);
                assert!(out.len() <= 1, "a one-row update cannot admit twice");
                if let Some((row, weight)) = out.first() {
                    assert_eq!(*row, 0, "the only slot is row 0");
                    admitted_mass += *weight as f64;
                    admissions += 1;
                }
            }

            let mean = admitted_mass / TRIALS as f64;
            let allowed = Z * (slot_variance(rate) / TRIALS as f64).sqrt();
            assert!(
                (mean - 1.0).abs() <= allowed,
                "rate={rate}: the mean of W*1{{admitted}} over {TRIALS} independent seeds \
                 is {mean:.4} against a true 1 — off by {:.4} > z*sqrt(Var/T) = {allowed:.4} \
                 with Var = (1-p)/p + p r(1-r) = {:.4}. A sampler that admits the first \
                 slot unconditionally lands at 1/p = {:.1}. ({admissions} of {TRIALS} \
                 trials admitted.)",
                (mean - 1.0).abs(),
                slot_variance(rate),
                1.0 / rate,
            );
        }
    }

    /// Replays the schedule from the shared table by hand and compares it to
    /// what `admit_rows` reports, across a run of updates.
    ///
    /// The rate's reciprocal is an integer, so no rounding draw is consumed
    /// and the whole walk is a deterministic function of the table.
    fn reference_walk(
        rate: f64,
        start_cursor: usize,
        rows: usize,
        updates: usize,
    ) -> Vec<Vec<(usize, u64)>> {
        let inv_ln = 1.0 / (1.0 - rate).ln();
        let delta = (1.0 / rate).floor() as u64;
        let mut cursor = start_cursor % PRECOMPUTED_SAMPLE_LEN;
        let draw = |cursor: &mut usize| {
            let skip = (PRECOMPUTED_SAMPLE[*cursor] * inv_ln).floor() as usize;
            *cursor = (*cursor + 1) % PRECOMPUTED_SAMPLE_LEN;
            skip
        };
        let mut to_skip = draw(&mut cursor);
        let mut walk = Vec::with_capacity(updates);
        for _ in 0..updates {
            let mut admitted = Vec::new();
            let mut row = to_skip;
            while row < rows {
                admitted.push((row, delta));
                to_skip = draw(&mut cursor);
                row = row + 1 + to_skip;
            }
            to_skip = row - rows;
            walk.push(admitted);
        }
        walk
    }

    /// The skip carries across update boundaries, one update can admit several
    /// rows, and the cursor wraps mid-run without disturbing either.
    #[test]
    fn admit_rows_carries_the_skip_across_updates_and_admits_every_landing() {
        const ROWS: usize = 8;
        const UPDATES: usize = 4_000;

        for (rate, start_cursor) in [
            (0.5f64, 0usize),
            (0.25, 0),
            // Starts close enough to the end that the run wraps the cursor.
            (0.5, PRECOMPUTED_SAMPLE_LEN - 4),
        ] {
            let seed = start_cursor as u64;
            let mut nitro = Nitro::init_nitro_seeded(rate, seed);
            assert_eq!(
                nitro.cursor(),
                (start_cursor + 1) % PRECOMPUTED_SAMPLE_LEN,
                "the seed must place the cursor before the first draw"
            );

            let expected = reference_walk(rate, start_cursor, ROWS, UPDATES);
            let mut out = AdmittedRows::new();
            let mut multi_admission_updates = 0usize;
            let mut empty_updates = 0usize;
            for (u, want) in expected.iter().enumerate() {
                nitro.admit_rows(ROWS, &mut out);
                assert_eq!(
                    out.as_slice(),
                    want.as_slice(),
                    "rate={rate} start={start_cursor} update {u}: admissions disagree with \
                     the schedule replayed from the table"
                );
                if out.len() > 1 {
                    multi_admission_updates += 1;
                }
                if out.is_empty() {
                    empty_updates += 1;
                }
            }
            assert!(
                multi_admission_updates > 0,
                "rate={rate}: no update admitted more than one row over {UPDATES} updates \
                 of {ROWS} rows; a walk that stops after the first landing would pass \
                 every other assertion here"
            );
            assert!(
                empty_updates > 0,
                "rate={rate}: every update admitted something, so the cross-update carry \
                 is never exercised"
            );
            // The run must have wrapped for the third case to mean anything.
            if start_cursor > 0 {
                assert!(
                    nitro.cursor() < start_cursor,
                    "the cursor must have wrapped past the end of the table"
                );
            }
        }
    }

    /// At `p = 1` every row is admitted with weight exactly 1, and nothing is
    /// carried between updates.
    #[test]
    fn full_sampling_admits_every_row_at_unit_weight() {
        let mut nitro = Nitro::init_nitro(1.0);
        let mut out = AdmittedRows::new();
        for _ in 0..16 {
            nitro.admit_rows(5, &mut out);
            assert_eq!(
                out.as_slice(),
                &[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
                "full sampling must admit every row at unit weight"
            );
            assert_eq!(nitro.to_skip, 0, "full sampling must carry no skip");
        }
    }

    /// The admission buffer must stay inline for every matrix the hash family
    /// supports, at every rate — including `p = 1`, where every row is
    /// admitted at once.
    ///
    /// `SmallVec::spilled()` is the direct question: it is true exactly when
    /// the buffer has moved to the heap.
    #[test]
    fn the_admission_buffer_never_reaches_the_heap_within_the_supported_row_count() {
        let mut out = AdmittedRows::new();
        for &rate in &[1.0f64, 0.9, 0.5, 0.3, 0.07, 0.01] {
            let mut nitro = Nitro::init_nitro_seeded(rate, 0x0117_0042);
            for _ in 0..2_000 {
                nitro.admit_rows(MATRIX_MAX_ROWS, &mut out);
                assert!(
                    !out.spilled(),
                    "rate={rate}: the admission buffer spilled to the heap at \
                     {MATRIX_MAX_ROWS} rows with {} admissions",
                    out.len()
                );
            }
        }
    }

    /// A skip counter large enough to overflow the walk's arithmetic — which a
    /// decoded payload or `commit_ctx` can supply — must be absorbed, not
    /// panic.
    #[test]
    fn an_oversized_skip_counter_does_not_overflow_the_admission_walk() {
        let mut nitro = Nitro::init_nitro(0.5);
        nitro.commit_ctx(0, usize::MAX);
        let mut out = AdmittedRows::new();
        nitro.admit_rows(5, &mut out);
        assert!(out.is_empty(), "a skip of usize::MAX admits nothing");
        assert_eq!(
            nitro.to_skip,
            usize::MAX - 5,
            "the walk must consume exactly `rows` slots of the outstanding skip"
        );

        // And the same through the smallest rates the constructor accepts.
        for &rate in &[1e-12f64, 1e-15, f64::MIN_POSITIVE] {
            let mut tiny = Nitro::init_nitro(rate);
            for _ in 0..64 {
                tiny.admit_rows(20, &mut out);
            }
        }
    }

    /// The full context restores the admission **and** weight sequence; the
    /// legacy pair restores only the skip state.
    #[test]
    fn the_context_restores_the_rounding_stream_and_the_legacy_pair_does_not() {
        const ROWS: usize = 5;
        for &rate in &[0.3f64, 0.07] {
            let mut source = Nitro::init_nitro_seeded(rate, 0x0117_5EED);
            let mut out = AdmittedRows::new();
            for _ in 0..37 {
                source.admit_rows(ROWS, &mut out);
            }
            let ctx = source.context();
            let legacy = source.get_ctx();

            let tail = |nitro: &mut Nitro| {
                let mut buf = AdmittedRows::new();
                let mut seen = Vec::new();
                for _ in 0..64 {
                    nitro.admit_rows(ROWS, &mut buf);
                    seen.extend(buf.iter().copied());
                }
                seen
            };

            let mut continued = source.clone();
            let expected = tail(&mut continued);

            let mut restored = Nitro::init_nitro_seeded(rate, 0x0117_5EED);
            restored.restore_context(ctx);
            assert_eq!(
                tail(&mut restored),
                expected,
                "rate={rate}: a full context restore must continue the uninterrupted run"
            );

            let mut legacy_restored = Nitro::init_nitro_seeded(rate, 0x0117_5EED);
            legacy_restored.commit_ctx(legacy.0, legacy.2);
            assert_ne!(
                tail(&mut legacy_restored),
                expected,
                "rate={rate}: `commit_ctx` carries no rounding state, so it must not be \
                 documented or used as a full restore"
            );
        }
    }
}

/// Trait defining heap ordering behavior.
#[cfg(test)]
mod tests {

    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn build_three() -> Vec<[f64; 3]> {
        let mut rng = StdRng::seed_from_u64(0x5eed_c0de_1234_5678);
        (0..1_000)
            .map(|_| {
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ]
            })
            .collect()
    }

    fn build_four() -> Vec<[f64; 4]> {
        let mut rng = StdRng::seed_from_u64(0x5eed_c0de_1234_5678);
        (0..1_000)
            .map(|_| {
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ]
            })
            .collect()
    }

    fn build_five() -> Vec<[f64; 5]> {
        let mut rng = StdRng::seed_from_u64(0x5eed_c0de_1234_5678);
        (0..1_000)
            .map(|_| {
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ]
            })
            .collect()
    }

    fn median_three_sort(values: &mut [f64; 3]) -> f64 {
        values.sort_unstable_by(f64::total_cmp);
        let mid = values.len() / 2;
        if values.len() % 2 == 1 {
            values[mid]
        } else {
            (values[mid - 1] + values[mid]) / 2.0
        }
    }

    fn median_four_sort(values: &mut [f64; 4]) -> f64 {
        values.sort_unstable_by(f64::total_cmp);
        let mid = values.len() / 2;
        if values.len() % 2 == 1 {
            values[mid]
        } else {
            (values[mid - 1] + values[mid]) / 2.0
        }
    }

    fn median_five_sort(values: &mut [f64; 5]) -> f64 {
        values.sort_unstable_by(f64::total_cmp);
        let mid = values.len() / 2;
        if values.len() % 2 == 1 {
            values[mid]
        } else {
            (values[mid - 1] + values[mid]) / 2.0
        }
    }

    #[test]
    fn median_test() {
        let mut three_vec = build_three();
        let mut four_vec = build_four();
        let mut five_vec = build_five();
        for v in &mut three_vec {
            let fast_median = compute_median_inline_f64(v);
            let sort_median = median_three_sort(v);
            assert_eq!(
                fast_median, sort_median,
                "median for sort is {sort_median} but fast gives {fast_median}, input is {v:?}"
            );
        }
        for v in &mut four_vec {
            let fast_median = compute_median_inline_f64(v);
            let sort_median = median_four_sort(v);
            assert_eq!(
                fast_median, sort_median,
                "median for sort is {sort_median} but fast gives {fast_median}, input is {v:?}"
            );
        }
        for v in &mut five_vec {
            let fast_median = compute_median_inline_f64(v);
            let sort_median = median_five_sort(v);
            assert_eq!(
                fast_median, sort_median,
                "median for sort is {sort_median} but fast gives {fast_median}, input is {v:?}"
            );
        }
    }
}
