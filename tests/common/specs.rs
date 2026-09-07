//! Error-model specifications, one per *error metric*, plus the statistical
//! machinery that turns a guarantee into an acceptance rule.
//!
//! The specs in [`super::conformance`] carry loose smoke-test tolerances,
//! which is the right thing for wiring up a new adapter but wrong as a
//! statement of theory: KLL promises *rank* error while DDSketch promises
//! *relative value* error, and Count-Min's one-sided additive `eps*N` is not
//! Count Sketch's two-sided `L2/sqrt(w)`. Each spec here models exactly one
//! guarantee and nothing else, so no two families can be judged by the same
//! number by accident.
//!
//! # The statistical unit
//!
//! A bound of the form `P[error > B] <= p` is a statement about **one** draw of
//! the randomness the estimator is built on — one hash choice, one compaction
//! coin sequence, one sampling seed. Turning `n` observed checks into a
//! binomial tail at `p` requires those `n` checks to be `n` *independent draws
//! of that randomness*. Almost none of the natural check batteries are:
//!
//! - several quantiles `q` read off **one** KLL sketch share one compaction
//!   history;
//! - several keys read off **one** Count-Min or Count Sketch share one hash;
//! - increasing checkpoints on **one** HLL or KMV are nested — the later
//!   estimate contains the earlier one's state;
//! - a single-pass sketch and its merged twin share most of their state.
//!
//! So this file offers three acceptance rules and each caller must pick the one
//! that matches what it actually collected:
//!
//! | Rule | When it is valid |
//! | --- | --- |
//! | [`Tally::assert_none`] | structural facts, and *simultaneous* bounds whose failure probability was already union-bounded down over every check in the battery |
//! | [`Tally::assert_independent_binomial`] | every recorded check is a separate draw of the estimator's randomness (a fresh sketch seed, a fresh hash, a fresh sampling seed) |
//! | [`Tally::assert_rate_at_most`] | one fixed realisation, pinned against the guarantee's own marginal probability — a regression pin, not a probability statement |
//!
//! The way a battery over one sketch becomes a legitimate trial is to reduce it
//! to a single outcome first: the sketch's *maximum* rank error over the whole
//! `q` grid, or whether *any* probed key broke a simultaneous bound. That one
//! outcome, repeated over independent seeds, is a binomial.
//!
//! Where a guarantee has no closed form, or the public API cannot expose the
//! dimension it quantifies over, the test says so and is named
//! `*_stays_within_the_documented_empirical_band` rather than being dressed up
//! as theory.

#![allow(dead_code)]

use super::FreqTruth;

// ---------------------------------------------------------------------------
// Binomial machinery
// ---------------------------------------------------------------------------

/// `ln(n choose k)` via log-gamma, stable for the counts used here.
fn ln_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// `ln(n!)` — exact summation below 256, Stirling/Lanczos above.
fn ln_factorial(n: usize) -> f64 {
    if n < 256 {
        (1..=n).map(|i| (i as f64).ln()).sum()
    } else {
        let x = n as f64 + 1.0;
        // Stirling series; accurate to well past the precision this needs.
        (x - 0.5) * x.ln() - x
            + 0.5 * (std::f64::consts::TAU).ln()
            + 1.0 / (12.0 * x)
            + -1.0 / (360.0 * x * x * x)
    }
}

/// `P(Binomial(n, p) >= k)`.
pub fn binomial_tail_ge(n: usize, k: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (lp, l1p) = (p.ln(), (1.0 - p).ln());
    let mut acc = 0.0;
    for i in k..=n {
        acc += (ln_choose(n, i) + i as f64 * lp + (n - i) as f64 * l1p).exp();
    }
    acc.min(1.0)
}

/// The largest violation count a run of **independent** trials may show and
/// still be consistent with a per-trial failure probability of `p`, at test
/// level `test_level`.
///
/// Returns the smallest `c` with `P(Binomial(trials, p) > c) <= test_level`.
/// Fixing `test_level` up front is what stops a tolerance from being tuned to
/// whatever the current run happens to produce.
pub fn max_allowed_failures(trials: usize, p: f64, test_level: f64) -> usize {
    if trials == 0 {
        return 0;
    }
    for c in 0..=trials {
        if binomial_tail_ge(trials, c + 1, p) <= test_level {
            return c;
        }
    }
    trials
}

/// Standard test level for every acceptance rule in this file.
///
/// Every stream, every sketch seed and every trial grid in these suites is
/// fixed, so a run is a deterministic function of the code — re-running
/// cannot produce a different verdict and there is no flake to buy insurance
/// against. The level therefore only has to be small enough that a *correct*
/// implementation clears it with room to spare, and large enough that the
/// acceptance rule still bites.
pub const TEST_LEVEL: f64 = 1e-6;

/// Total failure probability a *simultaneous* (union-bounded) battery is sized
/// at, before it is spread over the checks in the battery.
///
/// A simultaneous bound holds for **every** probed key at once with probability
/// `1 - SIMULTANEOUS_LEVEL`, so the battery tolerates zero violations and needs
/// no independence assumption anywhere. It buys that soundness by being wider
/// than the marginal bound by a factor that grows with the number of keys —
/// slowly, because both families' per-key failure probabilities decay
/// polynomially or exponentially in the bound's scale factor.
pub const SIMULTANEOUS_LEVEL: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Violation tallies
// ---------------------------------------------------------------------------

/// Accumulates violations across checks so a single acceptance rule is applied
/// once, at the end, to the whole population.
///
/// A `Tally` is only a counter. Which acceptance rule is legitimate depends
/// entirely on what the caller recorded into it — see the module docs.
#[derive(Debug, Default)]
pub struct Tally {
    pub checks: usize,
    pub violations: usize,
    /// Up to `SAMPLE_LIMIT` rendered violations, for the failure message.
    pub samples: Vec<String>,
}

const SAMPLE_LIMIT: usize = 12;

impl Tally {
    pub fn record(&mut self, ok: bool, detail: impl FnOnce() -> String) {
        self.checks += 1;
        if !ok {
            self.violations += 1;
            if self.samples.len() < SAMPLE_LIMIT {
                self.samples.push(detail());
            }
        }
    }

    /// Observed violation rate over the recorded checks.
    pub fn rate(&self) -> f64 {
        if self.checks == 0 {
            0.0
        } else {
            self.violations as f64 / self.checks as f64
        }
    }

    fn rendered(&self) -> String {
        self.samples
            .iter()
            .map(|s| format!("    {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Binomial acceptance at per-trial failure probability `p`.
    ///
    /// **Only valid when every recorded check is an independent draw of the
    /// randomness the bound quantifies over** — a distinct sketch seed, hash
    /// seed or sampling seed. Several queries against one sketch are not
    /// independent trials; reduce them to one outcome per sketch first.
    pub fn assert_independent_binomial(self, label: &str, p: f64, context: &str) {
        let allowed = max_allowed_failures(self.checks, p, TEST_LEVEL);
        assert!(
            self.violations <= allowed,
            "{label}: {} of {} independent trials violated the bound; a per-trial \
             failure probability p={p:.3e} allows at most {allowed} at test level \
             {TEST_LEVEL:.0e}.\n  context: {context}\n  first violations:\n{}",
            self.violations,
            self.checks,
            self.rendered(),
        );
    }

    /// Applies a rule that tolerates no violations at all.
    ///
    /// For structural guarantees (Count-Min never underestimates, Bloom never
    /// has a false negative, a merge is exact addition) and for *simultaneous*
    /// bounds, whose per-key failure probability has already been union-bounded
    /// down over the whole battery so that a single violation is significant.
    pub fn assert_none(self, label: &str, context: &str) {
        assert!(
            self.violations == 0,
            "{label}: {} of {} checks violated a guarantee that tolerates none.\n  \
             context: {context}\n  first violations:\n{}",
            self.violations,
            self.checks,
            self.rendered(),
        );
    }

    /// Pins the observed violation *rate* of one fixed realisation against the
    /// guarantee's own marginal per-check failure probability.
    ///
    /// This is **not** a probability statement: the sketch, its hash and its
    /// stream are all fixed, so the rate is a deterministic number and there is
    /// exactly one draw. It is a regression pin at a threshold the guarantee
    /// supplies (never a hand-picked percentage), and it is what catches an
    /// estimator whose error grew by an order of magnitude while still landing
    /// inside a union-bounded simultaneous band.
    pub fn assert_rate_at_most(self, label: &str, marginal_p: f64, context: &str) {
        let rate = self.rate();
        assert!(
            rate <= marginal_p,
            "{label}: {} of {} keys ({rate:.4}) exceeded the marginal bound, above the \
             guarantee's own marginal failure probability {marginal_p:.4}. This is a \
             single fixed realisation, so it is a regression pin rather than a tail \
             test.\n  context: {context}\n  first violations:\n{}",
            self.violations,
            self.checks,
            self.rendered(),
        );
    }
}

// ---------------------------------------------------------------------------
// Count-Min: one-sided additive error
// ---------------------------------------------------------------------------

/// Count-Min Sketch, Cormode & Muthukrishnan (2005), Theorem 1.
///
/// For a sketch of `d` rows and `w` columns, with `f` the exact frequency
/// vector and `N = ||f||_1`:
///
/// - **structural:** `est(i) >= f(i)` for every key, always, no probability;
/// - **marginal:** `P[ est(i) - f(i) > b * (N - f(i)) / w ] <= b^-d` for any
///   `b > 1`, one key at a time.
///
/// The row estimator is `f(i) + X_r` with `X_r >= 0` and
/// `E[X_r] = (N - f(i)) / w` (only the *other* keys can collide into `i`'s
/// cell). Markov gives `P[X_r >= b * E[X_r]] <= 1/b` per row; the `d` rows use
/// independent hashes and the minimum exceeds the bound only if every row
/// does, hence `b^-d`. The familiar form is `b = e`, giving `e^-d`.
///
/// `N - f(i)` rather than `N` is the tighter and correct tail mass: it is what
/// the expectation actually is, and using `N` would silently hand a hot key a
/// larger budget than the theorem grants it.
///
/// # Simultaneous form
///
/// Because the per-key failure probability decays as `b^-d`, a union bound over
/// `D` probed keys costs only a factor `(D / delta)^(1/d)` in the budget. That
/// gives a genuine statement about *all* `D` keys at once, with no independence
/// assumption between keys, which is what makes a zero-tolerance assertion over
/// a whole key battery legitimate.
///
/// The `d` rows being independent is exact for `RegularPath` (one hash call per
/// row with a distinct seed) and a modelling assumption for `FastPath`, which
/// slices row indices out of one 128-bit hash.
#[derive(Clone, Copy, Debug)]
pub struct CountMinSpec {
    pub rows: usize,
    pub cols: usize,
}

impl CountMinSpec {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// `e * (N - f_i) / w` — the marginal additive excess budget for one key.
    pub fn marginal_bound(&self, total_mass: f64, key_mass: f64) -> f64 {
        std::f64::consts::E * (total_mass - key_mass) / self.cols as f64
    }

    /// `e^-d` — the marginal probability a single key's minimum exceeds the
    /// marginal budget.
    pub fn marginal_failure(&self) -> f64 {
        (-(self.rows as f64)).exp()
    }

    /// `b = (D / delta)^(1/d)`: the Markov factor that makes the additive bound
    /// hold for all `D` keys at once with probability `1 - delta`.
    pub fn simultaneous_factor(&self, distinct: usize, delta: f64) -> f64 {
        (distinct.max(1) as f64 / delta).powf(1.0 / self.rows as f64)
    }

    /// `b * (N - f_i) / w` at the simultaneous `b`.
    pub fn simultaneous_bound(
        &self,
        total_mass: f64,
        key_mass: f64,
        distinct: usize,
        delta: f64,
    ) -> f64 {
        self.simultaneous_factor(distinct, delta) * (total_mass - key_mass) / self.cols as f64
    }

    /// Runs the full Count-Min contract over every key in `truth`.
    ///
    /// Three assertions, each with the acceptance rule its statement earns:
    ///
    /// 1. one-sidedness — structural, zero tolerated;
    /// 2. the simultaneous additive bound — union-bounded to
    ///    `SIMULTANEOUS_LEVEL` over the whole key set, zero tolerated;
    /// 3. the marginal additive bound — a rate pin on one fixed realisation at
    ///    the theorem's own `e^-d`, because the keys of one sketch share one
    ///    hash and are not independent trials.
    pub fn assert_contract<F>(&self, label: &str, truth: &FreqTruth, estimate: F, context: &str)
    where
        F: Fn(i64) -> f64,
    {
        let total = truth.total() as f64;
        let distinct = truth.distinct();
        let factor = self.simultaneous_factor(distinct, SIMULTANEOUS_LEVEL);
        let mut one_sided = Tally::default();
        let mut simultaneous = Tally::default();
        let mut marginal = Tally::default();
        for (key, count) in truth.pairs() {
            let est = estimate(key);
            let f = count as f64;
            one_sided.record(est >= f, || {
                format!("key {key}: est {est} < true {f} (Count-Min must never underestimate)")
            });
            let simul = self.simultaneous_bound(total, f, distinct, SIMULTANEOUS_LEVEL);
            simultaneous.record(est - f <= simul, || {
                format!(
                    "key {key}: excess {:.1} > b*(N-f)/w = {simul:.1} with b={factor:.2} \
                     (true {f}, est {est})",
                    est - f
                )
            });
            let marg = self.marginal_bound(total, f);
            marginal.record(est - f <= marg, || {
                format!(
                    "key {key}: excess {:.1} > e*(N-f)/w = {marg:.1} (true {f}, est {est})",
                    est - f
                )
            });
        }
        let ctx = format!(
            "{context}; rows={} cols={} N={total:.0} distinct={distinct} \
             simultaneous b={factor:.3} at delta={SIMULTANEOUS_LEVEL:.0e}",
            self.rows, self.cols,
        );
        one_sided.assert_none(&format!("{label} / one-sided"), &ctx);
        simultaneous.assert_none(&format!("{label} / simultaneous b*(N-f)/w"), &ctx);
        marginal.assert_rate_at_most(
            &format!("{label} / marginal e*(N-f)/w"),
            self.marginal_failure(),
            &ctx,
        );
    }
}

// ---------------------------------------------------------------------------
// Count Sketch: two-sided L2 error
// ---------------------------------------------------------------------------

/// Count Sketch, Charikar, Chen & Farach-Colton (2002).
///
/// This is **not** Count-Min's bound and must not reuse `eps * N`. Count
/// Sketch's error is driven by the L2 norm of the *residual* frequency vector,
/// is two-sided (signed), and is rank-independent — a cold key gets the same
/// absolute error band as the hottest one.
///
/// Per row `r`, the estimator is `X_r = s_r(i) * C[r][h_r(i)]`, which is
/// unbiased with
///
/// ```text
///   E[X_r] = f(i)
///   Var[X_r] <= || f_{-i} ||_2^2 / w
/// ```
///
/// where `f_{-i}` is `f` with coordinate `i` removed and `w` is the row width.
/// Chebyshev at `t = sqrt(kappa / w) * || f_{-i} ||_2` gives a per-row failure
/// probability of at most `1 / kappa`. The reported estimate is the median of
/// the `d` rows, so it exceeds `t` only when at least `ceil(d/2)` independent
/// rows do:
///
/// ```text
///   P[ |est(i) - f(i)| > sqrt(kappa / w) * ||f_{-i}||_2 ]
///        <= P[ Binomial(d, 1/kappa) >= ceil(d/2) ]
/// ```
///
/// `kappa = 3` is the usual marginal choice (per-row failure 1/3, so the median
/// is a genuine amplification); it is a declared constant of the spec, not a
/// fudge factor multiplied onto a bound after the fact.
///
/// # Even `d`, and why the threshold is `ceil(d/2)`
///
/// `compute_median_inline_f64` returns the *average of the two middle order
/// statistics* when `d` is even, which is the standard even-length median and
/// is what every even-depth sketch in this crate reports. That changes how many
/// bad rows it takes to break the bound.
///
/// Write `t` for the error scale and call a row *bad* if its estimate leaves
/// `[f - t, f + t]`. For odd `d` the reported value is the middle order
/// statistic, and it is bad only if at least `ceil(d/2)` rows are — the
/// classic majority argument. For even `d` the reported value is
/// `(X_(d/2) + X_(d/2+1)) / 2`:
///
/// - with `d/2 - 1` or fewer bad rows, both middle order statistics are good
///   and their average lies in `[f - t, f + t]`, which is convex — safe;
/// - with `d/2` bad rows **all on the same side**, one of the two middle order
///   statistics is bad and unbounded, so the average can leave the interval by
///   an arbitrary amount.
///
/// At `d = 4` that means **two** bad rows suffice, not three. The threshold is
/// therefore `ceil(d/2) = (d + 1) / 2` in integer arithmetic, which agrees with
/// `d/2 + 1` for odd `d` and is strictly smaller for even `d`.
///
/// This matters because it is the direction that *overstates* the guarantee:
/// `rows / 2 + 1` at `d = 4, kappa = 3` reports a per-key failure probability
/// of `P[Bin(4, 1/3) >= 3] = 0.111` where the estimator only earns
/// `P[Bin(4, 1/3) >= 2] = 0.407`, and it makes the simultaneous `kappa` search
/// stop early — a bound narrower than the theorem supports. `CountL2HH` runs at
/// four rows, so this was live.
///
/// # Simultaneous form
///
/// The keys of one sketch share one hash, so their outcomes are dependent and a
/// binomial over keys is not available. Raising `kappa` until the per-key
/// failure probability is below `delta / D` makes the bound hold for all `D`
/// probed keys at once by a union bound, which needs no independence at all —
/// and the required `kappa` grows only polynomially, because the median tail
/// falls off like `kappa^-ceil(d/2)`.
#[derive(Clone, Copy, Debug)]
pub struct CountSketchSpec {
    pub rows: usize,
    pub cols: usize,
    /// Chebyshev slack for the marginal bound. Per-row failure is `1 / kappa`.
    pub kappa: f64,
}

impl CountSketchSpec {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            kappa: 3.0,
        }
    }

    /// `sqrt(kappa / w) * ||f_{-i}||_2` at an explicit `kappa`.
    pub fn scale_at(&self, kappa: f64, residual_l2: f64) -> f64 {
        (kappa / self.cols as f64).sqrt() * residual_l2
    }

    /// `sqrt(3 / w) * ||f_{-i}||_2` — the marginal error scale.
    pub fn marginal_scale(&self, residual_l2: f64) -> f64 {
        self.scale_at(self.kappa, residual_l2)
    }

    /// `ceil(d/2)` — the smallest number of bad rows that can push the
    /// reported estimate outside the band. See the type docs for the even-`d`
    /// derivation.
    pub fn bad_row_threshold(&self) -> usize {
        self.rows.div_ceil(2)
    }

    /// `P[Binomial(d, 1/kappa) >= ceil(d/2)]` at an explicit `kappa`.
    pub fn key_failure_at(&self, kappa: f64) -> f64 {
        binomial_tail_ge(self.rows, self.bad_row_threshold(), 1.0 / kappa)
    }

    /// Marginal per-key failure probability at `kappa = 3`.
    pub fn marginal_failure(&self) -> f64 {
        self.key_failure_at(self.kappa)
    }

    /// Smallest `kappa` whose per-key failure probability is at most
    /// `delta / D`, so a union bound over `D` keys leaves `delta` overall.
    ///
    /// The tail is monotone decreasing in `kappa`, so a bisection on
    /// `log(kappa)` converges; the search starts from the marginal `kappa` and
    /// doubles until the target is met.
    pub fn simultaneous_kappa(&self, distinct: usize, delta: f64) -> f64 {
        let target = delta / distinct.max(1) as f64;
        let mut lo = self.kappa;
        if self.key_failure_at(lo) <= target {
            return lo;
        }
        let mut hi = lo;
        for _ in 0..80 {
            hi *= 2.0;
            if self.key_failure_at(hi) <= target {
                break;
            }
            lo = hi;
        }
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if self.key_failure_at(mid) <= target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// Runs the L2 contract over every key in `truth`.
    ///
    /// The residual norm is recomputed per key from the exact frequency
    /// vector: `||f_{-i}||_2 = sqrt(F2 - f(i)^2)`. The simultaneous bound is
    /// asserted with zero tolerance; the marginal one is a rate pin, because
    /// the keys of a single sketch are one realisation, not `D` trials.
    pub fn assert_contract<F>(&self, label: &str, truth: &FreqTruth, estimate: F, context: &str)
    where
        F: Fn(i64) -> f64,
    {
        let distinct = truth.distinct();
        let kappa = self.simultaneous_kappa(distinct, SIMULTANEOUS_LEVEL);
        let mut simultaneous = Tally::default();
        let mut marginal = Tally::default();
        self.tally_into(&mut simultaneous, &mut marginal, truth, estimate);
        let ctx = format!(
            "{context}; rows={} cols={} marginal kappa={} simultaneous kappa={kappa:.1} \
             at delta={SIMULTANEOUS_LEVEL:.0e} over {distinct} keys, F2={:.3e}",
            self.rows,
            self.cols,
            self.kappa,
            truth.f2(),
        );
        simultaneous.assert_none(&format!("{label} / simultaneous L2"), &ctx);
        marginal.assert_rate_at_most(
            &format!("{label} / marginal sqrt(3/w)*||f_-i||_2"),
            self.marginal_failure(),
            &ctx,
        );
    }

    /// Same checks, accumulated into caller-owned tallies so several instances
    /// share one acceptance rule.
    pub fn tally_into<F>(
        &self,
        simultaneous: &mut Tally,
        marginal: &mut Tally,
        truth: &FreqTruth,
        estimate: F,
    ) where
        F: Fn(i64) -> f64,
    {
        let f2 = truth.f2();
        let kappa = self.simultaneous_kappa(truth.distinct(), SIMULTANEOUS_LEVEL);
        for (key, count) in truth.pairs() {
            let f = count as f64;
            let residual_l2 = (f2 - f * f).max(0.0).sqrt();
            let est = estimate(key);
            let err = (est - f).abs();

            let simul = self.scale_at(kappa, residual_l2);
            simultaneous.record(err <= simul, || {
                format!(
                    "key {key}: |{est:.1} - {f}| = {err:.1} > sqrt(kappa/w)*||f_-i||_2 = \
                     {simul:.1} at the simultaneous kappa={kappa:.1}"
                )
            });

            let marg = self.marginal_scale(residual_l2);
            marginal.record(err <= marg, || {
                format!("key {key}: |{est:.1} - {f}| = {err:.1} > sqrt(3/w)*||f_-i||_2 = {marg:.1}")
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Second moment (F2) from a Count Sketch row
// ---------------------------------------------------------------------------

/// F2 estimation from the counter matrix of a Count Sketch, as `CountL2HH`
/// and UnivMon do.
///
/// Each row's `Y_r = sum_j C[r][j]^2` is the AMS tug-of-war estimator (Alon,
/// Matias & Szegedy 1996; Charikar, Chen & Farach-Colton 2002):
///
/// ```text
///   E[Y_r]   = F2
///   Var[Y_r] = 2 (F2^2 - sum_i f_i^4) / w  <=  2 F2^2 / w
/// ```
///
/// Chebyshev at `t = sqrt(2 kappa / w) * F2` gives a per-row failure
/// probability of `1 / kappa`, and the reported value is the median over the
/// `d` rows, so the query fails only when at least `ceil(d/2)` rows do.
///
/// **Trial unit:** one F2 query is one sketch's single answer, so a battery of
/// F2 checks over distinct sketches *is* a battery of independent trials; two
/// F2 reads from the same sketch are not.
///
/// This is *not* an exactly maintained quantity: the accumulator tracks the
/// sketch's own counters, which carry collisions, so the answer is an estimate
/// with a real error bound and must be checked as one.
#[derive(Clone, Copy, Debug)]
pub struct SecondMomentSpec {
    pub rows: usize,
    pub cols: usize,
    pub kappa: f64,
}

impl SecondMomentSpec {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            kappa: 3.0,
        }
    }

    /// `sqrt(2 kappa / w)` — the relative half-width around the exact F2.
    pub fn relative_bound(&self) -> f64 {
        (2.0 * self.kappa / self.cols as f64).sqrt()
    }

    pub fn per_row_failure(&self) -> f64 {
        1.0 / self.kappa
    }

    /// `ceil(d/2)` bad rows, for the same reason as `CountSketchSpec`: an
    /// even-depth sketch reports the average of the two middle order
    /// statistics, so half the rows going bad in one direction is enough.
    pub fn bad_row_threshold(&self) -> usize {
        self.rows.div_ceil(2)
    }

    pub fn per_query_failure(&self) -> f64 {
        binomial_tail_ge(self.rows, self.bad_row_threshold(), self.per_row_failure())
    }

    pub fn check(&self, estimate: f64, exact_f2: f64) -> Result<(), String> {
        let rel = ((estimate - exact_f2) / exact_f2).abs();
        let bound = self.relative_bound();
        if rel <= bound {
            Ok(())
        } else {
            Err(format!(
                "F2 estimate {estimate:.6e} vs exact {exact_f2:.6e}: relative error {rel:.5} > \
                 sqrt(2*kappa/w) = {bound:.5} (rows={} cols={} kappa={})",
                self.rows, self.cols, self.kappa
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// KLL: rank error
// ---------------------------------------------------------------------------

/// Normalized-rank-error contract for a KLL sketch of parameter `k`.
///
/// KLL (Karnin, Lang & Liberty, FOCS 2016) promises *rank* error, not value
/// error: for a query at normalized rank `q` the returned value `v` must
/// satisfy
///
/// ```text
///   rank_incl(v) >= q - eps   and   rank_excl(v) <= q + eps
/// ```
///
/// where `rank_excl(v) = |{x < v}| / n` and `rank_incl(v) = |{x <= v}| / n`.
/// Stating it on the *interval* of ranks that `v` occupies is what makes the
/// check correct on streams with many repeated values, where a single value
/// legitimately spans a wide rank range.
///
/// # Where the constant comes from, and what it is not
///
/// The KLL paper's guarantee is asymptotic: a sketch of `O(k)` retained items
/// answers every rank query within `eps = O(1/k)` with constant probability,
/// with the constants left unresolved and no closed form for a specific
/// implementation's compactor schedule.
///
/// The number used here,
///
/// ```text
///   eps(k) = 2.446 / k^0.9433
/// ```
///
/// is **not** that theorem. It is Apache DataSketches' published
/// characterization result: a least-squares fit, over a large body of
/// characterization runs of *their* implementation, to the 99th percentile of
/// the **maximum** rank error observed across a whole quantile grid
/// (`getNormalizedRankError(k, false)`). This crate's KLL implements the same
/// compact layout (capacity decay 2/3, `m = 8`, randomized halving with a
/// shared coin per compaction), so the fit is the right *external empirical
/// contract to hold it to* — a characterization target, not a proof about this
/// code.
///
/// Two consequences for how it must be tested:
///
/// - the quantity the constant bounds is the **maximum** rank error over the
///   grid, so one sketch produces exactly **one** pass/fail outcome — the
///   individual `q` values are not separate Bernoulli draws;
/// - the quoted 1% is the failure probability of that single per-sketch
///   outcome, so a binomial over **independent compaction seeds** is the only
///   legitimate aggregation.
#[derive(Clone, Copy, Debug)]
pub struct KllRankSpec {
    pub k: usize,
    /// Failure probability of one *sketch trial* — of the maximum rank error
    /// over a whole `q` grid, not of a single `q`.
    pub trial_failure_probability: f64,
}

impl KllRankSpec {
    /// The Apache DataSketches characterization target at 99% confidence.
    pub fn datasketches(k: usize) -> Self {
        Self {
            k,
            trial_failure_probability: 0.01,
        }
    }

    /// `eps(k) = 2.446 / k^0.9433`.
    pub fn epsilon(&self) -> f64 {
        2.446 / (self.k as f64).powf(0.9433)
    }

    /// The maximum normalized rank error one sketch shows over `qs`, together
    /// with the `q` that attained it.
    ///
    /// This is the reduction that turns a whole grid of dependent queries into
    /// the single number the contract is quoted for.
    pub fn max_rank_error<F>(&self, sorted: &[f64], qs: &[f64], quantile: F) -> (f64, String)
    where
        F: Fn(f64) -> f64,
    {
        let mut worst = 0.0f64;
        let mut detail = String::from("(no query exceeded rank 0)");
        for &q in qs {
            let v = quantile(q);
            let e = rank_error(sorted, q, v);
            if e >= worst {
                worst = e;
                let (excl, incl) = rank_interval(sorted, v);
                detail = format!(
                    "q={q}: value {v} occupies ranks [{excl:.6}, {incl:.6}], rank error {e:.6}"
                );
            }
        }
        (worst, detail)
    }

    /// Records one sketch trial: pass iff its maximum rank error over `qs` is
    /// within `eps(k)`.
    pub fn record_trial<F>(
        &self,
        tally: &mut Tally,
        label: &str,
        sorted: &[f64],
        qs: &[f64],
        quantile: F,
    ) where
        F: Fn(f64) -> f64,
    {
        let eps = self.epsilon();
        let (worst, detail) = self.max_rank_error(sorted, qs, quantile);
        tally.record(worst <= eps, || {
            format!(
                "{label}: max rank error {worst:.6} > eps(k={}) = {eps:.6}; worst {detail}",
                self.k
            )
        });
    }

    /// Checks one `(q, value)` answer against the rank band of `sorted`.
    ///
    /// Useful where a single query is the whole answer (an exact-window check,
    /// a single-key probe). Aggregating many of these from one sketch under a
    /// binomial is not.
    pub fn check(&self, sorted: &[f64], q: f64, value: f64) -> Result<(), String> {
        rank_violation(sorted, q, value, self.epsilon())
            .map(|d| Err(format!("{d} (k={})", self.k)))
            .unwrap_or(Ok(()))
    }
}

/// The rank interval `[rank_excl(v), rank_incl(v)]` that `value` occupies.
pub fn rank_interval(sorted: &[f64], value: f64) -> (f64, f64) {
    let n = sorted.len();
    assert!(n > 0, "rank error is undefined on an empty stream");
    let excl = sorted.partition_point(|x| *x < value) as f64 / n as f64;
    let incl = sorted.partition_point(|x| *x <= value) as f64 / n as f64;
    (excl, incl)
}

/// Normalized rank error of answering `value` for a query at rank `q`:
/// the distance from `q` to the interval of ranks `value` occupies, and `0`
/// when `q` falls inside it.
///
/// Measuring the distance to the *interval* rather than to a single rank is
/// what keeps the quantity correct on streams with repeated values, where one
/// value legitimately spans a wide rank range.
pub fn rank_error(sorted: &[f64], q: f64, value: f64) -> f64 {
    let (excl, incl) = rank_interval(sorted, value);
    (q - incl).max(excl - q).max(0.0)
}

/// The rank-error predicate itself, independent of where `eps` came from.
///
/// A quantile answer `value` for a query at normalized rank `q` is correct
/// within `eps` when the interval of ranks `value` occupies overlaps
/// `[q - eps, q + eps]`.
///
/// Returns `None` when the answer is within the band, or a rendered violation.
/// Shared so that estimators deriving `eps` from a different argument — KLL
/// from `k`, UnivMon-Q from its occurrence-sample bound — check the same
/// predicate rather than each rolling their own.
pub fn rank_violation(sorted: &[f64], q: f64, value: f64, eps: f64) -> Option<String> {
    let (excl, incl) = rank_interval(sorted, value);
    if incl >= q - eps && excl <= q + eps {
        None
    } else {
        Some(format!(
            "q={q}: value {value} occupies ranks [{excl:.5}, {incl:.5}], outside \
             [q-eps, q+eps] = [{:.5}, {:.5}] for eps={eps:.5}",
            q - eps,
            q + eps
        ))
    }
}

/// Two-sided Dvoretzky-Kiefer-Wolfowitz / Hoeffding band for an empirical CDF
/// built from `m` uniform samples of the stream:
///
/// ```text
///   P[ sup_x |F_m(x) - F(x)| > sqrt(ln(2/delta) / (2m)) ] <= delta
/// ```
///
/// This is the residual term `epsilon_R` in UnivMon-Q's documented
/// ordered-query bound, and it is the whole bound in the diffuse regime where
/// the heavy set is empty. Because it is a `sup_x` statement it already covers
/// every query against one sketch at once: one sketch is one trial.
pub fn occurrence_sample_epsilon(samples: usize, delta: f64) -> f64 {
    ((2.0 / delta).ln() / (2.0 * samples as f64)).sqrt()
}

// ---------------------------------------------------------------------------
// Kolmogorov distance between a step CDF and an exact empirical CDF
// ---------------------------------------------------------------------------

/// `sup_x |F_hat(x) - F(x)|`, computed exactly, plus the `x` that attains it.
///
/// `estimated` is a step CDF as `(value, cumulative rank)` pairs in ascending
/// value order — the shape `UnivMonQ::cdf` returns. `sorted_truth` is the exact
/// stream, ascending, with multiplicity.
///
/// # Both CDFs are right-continuous, and that makes the sweep exact
///
/// ```text
///   F(x)     = #{ i : x_i <= x } / n
///   F_hat(x) = rank of the last breakpoint whose value is <= x   (0 if none)
/// ```
///
/// Both are step functions that jump only at points of
/// `S = {distinct truth values} ∪ {breakpoint values}`, and both are constant
/// on every interval between consecutive elements of `S`. So `|F_hat - F|` is
/// constant there too, and its value on `[s_k, s_{k+1})` is its value **at**
/// `s_k`. Below `min S` both functions are 0. Evaluating at every element of
/// `S` therefore returns the true supremum over all of `R` — not a sample of
/// it.
///
/// Both inputs are already ascending, so the evaluation is a single linear
/// merge: `O(n + m)` with one pass over each, rather than re-sorting the union
/// and binary-searching both sides at every point.
///
/// # Why this is not the same as a rank-interval check
///
/// Scoring each *estimated breakpoint* against the true rank interval of its
/// own value only ever looks at `x` values the estimate itself chose. An
/// estimate that is right wherever it has a breakpoint and simply *has no
/// breakpoint* across a region carrying a lot of mass scores zero error under
/// that check and arbitrarily large error under this one.
/// `cdf_sup_distance_detects_a_gap_a_breakpoint_scan_misses` in
/// `tests/e2e_quantiles.rs` is exactly that fixture.
pub fn cdf_sup_distance(estimated: &[(f64, f64)], sorted_truth: &[f64]) -> (f64, f64) {
    let n = sorted_truth.len();
    assert!(
        n > 0,
        "the Kolmogorov distance is undefined on an empty stream"
    );

    let mut i = 0usize; // truth observations at or below the current x
    let mut j = 0usize; // breakpoints at or below the current x
    let mut hat = 0.0f64; // F_hat at the current x
    let mut worst = 0.0f64;
    let mut worst_at = f64::NAN;

    while i < n || j < estimated.len() {
        // The next jump point of either function.
        let x = match (sorted_truth.get(i), estimated.get(j)) {
            (Some(t), Some((v, _))) => {
                if t.total_cmp(v) == std::cmp::Ordering::Greater {
                    *v
                } else {
                    *t
                }
            }
            (Some(t), None) => *t,
            (None, Some((v, _))) => *v,
            (None, None) => unreachable!("the loop condition guarantees one side is live"),
        };
        while i < n && sorted_truth[i].total_cmp(&x) != std::cmp::Ordering::Greater {
            i += 1;
        }
        while j < estimated.len() && estimated[j].0.total_cmp(&x) != std::cmp::Ordering::Greater {
            hat = estimated[j].1;
            j += 1;
        }
        let d = (hat - i as f64 / n as f64).abs();
        if d > worst {
            worst = d;
            worst_at = x;
        }
    }
    (worst, worst_at)
}

/// For each estimated breakpoint, the distance from its reported rank to the
/// true rank interval of its own value.
///
/// Kept **only** so a fixture can show that it reports zero where
/// [`cdf_sup_distance`] reports a large error. It is not a Kolmogorov distance
/// and must not be used as one.
pub fn breakpoint_rank_interval_distance(estimated: &[(f64, f64)], sorted_truth: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for &(value, rank) in estimated {
        let (excl, incl) = rank_interval(sorted_truth, value);
        worst = worst.max((rank - incl).max(excl - rank).max(0.0));
    }
    worst
}

// ---------------------------------------------------------------------------
// DDSketch: relative value error
// ---------------------------------------------------------------------------

/// Which order statistic a DDSketch implementation answers a quantile query
/// with. The two shipped implementations do not agree, and the truth a test
/// compares against has to follow the implementation it is testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DdRankConvention {
    /// `DDSketch::get_value_at_quantile`: `rank = ceil(q * n)`, 1-based, so the
    /// answer is `sorted[ceil(q*n) - 1]`. `q = 0` and `q = 1` short-circuit to
    /// the exact stored minimum and maximum.
    CeilNearestRank,
    /// Portable `DdSketch::quantile`: `target = floor(q * (n - 1))`, 0-based,
    /// so the answer is `sorted[floor(q*(n-1))]` — the lower-quantile
    /// convention of the DDSketch paper and of DataDog's reference
    /// implementation. No exact minimum or maximum is retained, so the
    /// endpoints are bucket representatives like any other rank.
    LowerFloor,
}

impl DdRankConvention {
    /// The zero-based index into an ascending `sorted` slice this convention
    /// answers `q` with.
    pub fn index(self, n: usize, q: f64) -> usize {
        assert!(
            n > 0,
            "a quantile convention is undefined on an empty stream"
        );
        let q = q.clamp(0.0, 1.0);
        match self {
            DdRankConvention::CeilNearestRank => (((q * n as f64).ceil() as usize).clamp(1, n)) - 1,
            DdRankConvention::LowerFloor => ((q * (n - 1) as f64).floor() as usize).min(n - 1),
        }
    }

    /// The exact order statistic this convention answers `q` with.
    pub fn order_statistic(self, sorted: &[f64], q: f64) -> f64 {
        sorted[self.index(sorted.len(), q)]
    }

    pub fn name(self) -> &'static str {
        match self {
            DdRankConvention::CeilNearestRank => "ceil(q*n) nearest-rank",
            DdRankConvention::LowerFloor => "floor(q*(n-1)) lower-quantile",
        }
    }
}

/// Relative-value-error contract for a DDSketch of accuracy parameter `alpha`.
///
/// DDSketch's guarantee is on the *value*, not the rank: the returned estimate
/// for the `q`-quantile must be within `alpha` relative error of the exact
/// order statistic **at the same `q` under that implementation's own rank
/// convention**:
///
/// ```text
///   |est - true| / |true| <= alpha + numerical_slack(true)
/// ```
///
/// With the logarithmic mapping `k = floor(ln v / ln gamma)`,
/// `gamma = (1+alpha)/(1-alpha)`, and bucket representative
/// `gamma^k * (1+alpha)`, a value at the bucket's lower edge is over-estimated
/// by exactly `alpha` and one approaching the upper edge is under-estimated by
/// almost `alpha` — so `alpha` is attained but never exceeded in exact
/// arithmetic. The guarantee is deterministic: there is no hash and no
/// sampling, so it tolerates **zero** violations and no statistical model
/// applies at all.
///
/// `numerical_slack` is therefore a *floating-point* term only, on the order of
/// a few ULP: `ln`, `floor` and `powf` compose to a relative error of roughly
/// `|ln v| * f64::EPSILON`. It is emphatically not a percentage of `alpha` —
/// `alpha * 1.05` would accept results that break the advertised guarantee by
/// 5%, which is the entire thing the guarantee exists to forbid.
#[derive(Clone, Copy, Debug)]
pub struct RelativeQuantileSpec {
    pub alpha: f64,
    pub convention: DdRankConvention,
}

impl RelativeQuantileSpec {
    /// A spec for the core `DDSketch`.
    pub fn core(alpha: f64) -> Self {
        Self {
            alpha,
            convention: DdRankConvention::CeilNearestRank,
        }
    }

    /// A spec for the portable `DdSketch`.
    pub fn portable(alpha: f64) -> Self {
        Self {
            alpha,
            convention: DdRankConvention::LowerFloor,
        }
    }

    /// A few ULP of headroom, scaled by the magnitude of the logarithm because
    /// that is what `powf`/`ln` round-off actually tracks.
    ///
    /// The estimate is `gamma^k * (1 + alpha)` with `k = floor(ln v / ln
    /// gamma)`. Three roundings compose: `ln v` (relative error `eps`, absolute
    /// error `|ln v| * eps`), the division by `ln gamma`, and `powf`, whose
    /// result carries a relative error of about `|k * ln gamma| * eps =
    /// |ln v| * eps`. Summing the three and rounding the constant up gives
    /// `8 * eps * (1 + |ln v|)`, which is a few ULP at every magnitude these
    /// suites reach and never a fraction of `alpha`.
    pub fn numerical_slack(&self, true_value: f64) -> f64 {
        8.0 * f64::EPSILON * (1.0 + true_value.abs().ln().abs())
    }

    pub fn tolerance(&self, true_value: f64) -> f64 {
        self.alpha + self.numerical_slack(true_value)
    }

    /// Checks one estimate against an exact value supplied by the caller.
    pub fn check(&self, q: f64, estimate: f64, true_value: f64) -> Result<(), String> {
        let tol = self.tolerance(true_value);
        let rel = ((estimate - true_value) / true_value.abs()).abs();
        if rel <= tol {
            Ok(())
        } else {
            Err(format!(
                "q={q}: est {estimate:.10e} vs exact {true_value:.10e} \
                 -> relative error {rel:.3e} > alpha + slack = {tol:.3e} (alpha={}, {})",
                self.alpha,
                self.convention.name(),
            ))
        }
    }

    /// Tallies a sketch's answers over a q grid against the exact order
    /// statistics **of this spec's own rank convention**.
    ///
    /// `sorted` must be ascending. Using one truth helper for both
    /// implementations would compare each against the other's question.
    pub fn tally_into<F>(&self, tally: &mut Tally, sorted: &[f64], qs: &[f64], quantile: F)
    where
        F: Fn(f64) -> Option<f64>,
    {
        for &q in qs {
            let truth = self.convention.order_statistic(sorted, q);
            match quantile(q) {
                None => tally.record(false, || format!("q={q}: sketch returned None")),
                Some(est) => {
                    let outcome = self.check(q, est, truth);
                    tally.record(outcome.is_ok(), || outcome.unwrap_err());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HLL / KMV: cardinality confidence bands
// ---------------------------------------------------------------------------

/// Which estimator's error model a cardinality band is derived from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardinalityModel {
    /// Flajolet, Fusy, Gandouet & Meunier (2007): relative standard error
    /// `1.04 / sqrt(m)` for `m = 2^precision` registers. Ertl's maximum
    /// likelihood estimator (2017) attains the same Cramer-Rao bound
    /// `sqrt(3 ln 2 - 1) / sqrt(m) = 1.0389 / sqrt(m)`, so both share this
    /// constant.
    HyperLogLogRegisters,
    /// Historic Inverse Probability (Cohen 2015; Ting 2014): relative standard
    /// error `sqrt(ln 2 / m) = 0.8326 / sqrt(m)`. HIP is strictly tighter than
    /// the register-only estimators because it integrates the insertion
    /// history, so it must not be checked against their constant.
    HyperLogLogHip,
    /// K-Minimum-Values, the `(k-1)/U_(k)` estimator over the `k` smallest
    /// normalized hashes.
    ///
    /// With `n` distinct uniform hashes, `U_(k) ~ Beta(k, n-k+1)`, so
    /// `E[(k-1)/U_(k)] = n` (the estimator is unbiased for `k > 1`) and
    ///
    /// ```text
    ///   Var[(k-1)/U_(k)] = n (n - k + 1) / (k - 2)
    ///   RSE(n, k)        = sqrt( (n - k + 1) / (n (k - 2)) )
    ///                    -> 1 / sqrt(k - 2)   as n -> infinity
    /// ```
    ///
    /// The finite-`n` form is used, because it is exact and because it is
    /// *smaller* than the asymptote — pretending otherwise would hand the
    /// estimator a band it has not earned. `1 / sqrt(k - 1)`, which an earlier
    /// revision of this file called "marginally conservative", is smaller than
    /// `1 / sqrt(k - 2)` and therefore stricter, not looser.
    KMinimumValues,
}

/// A cardinality band `n * (1 +- z * sigma_rel)` with an explicit `z`.
///
/// Nothing here is a flat percentage: `sigma_rel` comes from the estimator's
/// own error model and `z` is stated as a Gaussian quantile.
///
/// # What the `z` band is and is not
///
/// `sigma_rel` is a **standard deviation**, derived exactly. Turning it into a
/// tail probability via `2 (1 - Phi(z))` additionally assumes the estimator is
/// approximately normal, which is an *asymptotic model*, not a theorem:
///
/// - HLL's register-based estimators are asymptotically normal in `m` and `n`,
///   with known deviations near the linear-counting switchover;
/// - KMV's `(k-1)/U_(k)` is a reciprocal Beta variate, right-skewed at small
///   `k`, and only asymptotically normal as `k` grows.
///
/// So a band built from `z` is an **asymptotic, model-based confidence band**.
/// It is quoted as such in the coverage matrix and is never labelled a theorem.
///
/// # Trial unit
///
/// One sketch answering at one cardinality is one draw. Increasing checkpoints
/// on the *same* sketch are nested — the state at `n = 10^5` contains the state
/// at `n = 10^4` — so a battery of checkpoints is one trial, not many. Batteries
/// that need a binomial must vary the hash seed or the identity namespace.
#[derive(Clone, Copy, Debug)]
pub struct CardinalityConfidenceSpec {
    pub model: CardinalityModel,
    /// `m` registers for HLL, `k` retained minima for KMV.
    pub size: usize,
    /// Gaussian quantile. `z = 4` is a two-sided failure probability of
    /// 6.3e-5 per check under the normal model.
    pub z: f64,
}

impl CardinalityConfidenceSpec {
    pub fn hll(precision: u32, z: f64) -> Self {
        Self {
            model: CardinalityModel::HyperLogLogRegisters,
            size: 1usize << precision,
            z,
        }
    }

    pub fn hll_hip(precision: u32, z: f64) -> Self {
        Self {
            model: CardinalityModel::HyperLogLogHip,
            size: 1usize << precision,
            z,
        }
    }

    pub fn kmv(k: usize, z: f64) -> Self {
        Self {
            model: CardinalityModel::KMinimumValues,
            size: k,
            z,
        }
    }

    /// The estimator's relative standard error at `true_distinct` distinct
    /// elements.
    ///
    /// Only KMV's depends on `n`; the HLL constants are asymptotic in `n` by
    /// construction.
    pub fn sigma_rel_at(&self, true_distinct: usize) -> f64 {
        let s = self.size as f64;
        match self.model {
            CardinalityModel::HyperLogLogRegisters => 1.04 / s.sqrt(),
            CardinalityModel::HyperLogLogHip => (std::f64::consts::LN_2 / s).sqrt(),
            CardinalityModel::KMinimumValues => {
                let n = true_distinct as f64;
                let k = s;
                if k <= 2.0 || n <= 0.0 {
                    return f64::INFINITY;
                }
                (((n - k + 1.0).max(0.0)) / (n * (k - 2.0))).sqrt()
            }
        }
    }

    /// The estimator's asymptotic relative standard error.
    pub fn sigma_rel(&self) -> f64 {
        let s = self.size as f64;
        match self.model {
            CardinalityModel::HyperLogLogRegisters => 1.04 / s.sqrt(),
            CardinalityModel::HyperLogLogHip => (std::f64::consts::LN_2 / s).sqrt(),
            CardinalityModel::KMinimumValues => 1.0 / (s - 2.0).sqrt(),
        }
    }

    /// `z * sigma_rel` — the half-width of the accepted relative band, using
    /// the asymptotic RSE.
    pub fn tolerance(&self) -> f64 {
        self.z * self.sigma_rel()
    }

    /// `z * sigma_rel(n)` — the half-width at a specific cardinality.
    pub fn tolerance_at(&self, true_distinct: usize) -> f64 {
        self.z * self.sigma_rel_at(true_distinct)
    }

    /// Two-sided failure probability at `z` under the normal model.
    pub fn per_check_failure(&self) -> f64 {
        2.0 * (1.0 - standard_normal_cdf(self.z))
    }

    /// KMV is exact only while it has seen **fewer** than `k` distinct
    /// elements: `KMV::estimate` returns the retained count verbatim when
    /// `k_vals.len() < k`, and switches to `(k-1)/U_(k)` at `len == k`.
    ///
    /// The `n == k` boundary is therefore *estimated*, not exact — the earlier
    /// `n <= k` form silently demanded exactness of a number whose standard
    /// deviation there is about `sqrt(k / (k-2)) ~ 1`.
    pub fn is_exact_regime(&self, true_distinct: usize) -> bool {
        self.model == CardinalityModel::KMinimumValues && true_distinct < self.size
    }

    pub fn check(&self, estimate: f64, true_distinct: usize) -> Result<(), String> {
        let t = true_distinct as f64;
        if self.is_exact_regime(true_distinct) {
            // Exact regime: the sketch holds every distinct hash it has seen.
            return if (estimate - t).abs() <= 0.5 {
                Ok(())
            } else {
                Err(format!(
                    "n={true_distinct} < k={}: estimate {estimate} must be exact",
                    self.size
                ))
            };
        }
        let tol = self.tolerance_at(true_distinct);
        let rel = ((estimate - t) / t).abs();
        if rel <= tol {
            Ok(())
        } else {
            Err(format!(
                "n={true_distinct}: estimate {estimate:.1} -> relative error {rel:.5} > \
                 z*sigma(n) = {:.2}*{:.5} = {tol:.5} (model {:?}, size {})",
                self.z,
                self.sigma_rel_at(true_distinct),
                self.model,
                self.size
            ))
        }
    }

    pub fn tally_into(&self, tally: &mut Tally, estimate: f64, true_distinct: usize) {
        let outcome = self.check(estimate, true_distinct);
        tally.record(outcome.is_ok(), || outcome.unwrap_err());
    }
}

// ---------------------------------------------------------------------------
// Nitro: sampling confidence
// ---------------------------------------------------------------------------

/// Confidence band for a Bernoulli-sampled counter, as NitroSketch uses.
///
/// Nitro admits each update with probability `p` and writes an integer weight
/// `W` per admitted update. For a key of true count `f` the number of admitted
/// updates is `X ~ Binomial(f, p)` and the estimate is `sum of W over the X
/// admissions`.
///
/// The implementation draws `W` by **stochastic rounding** of `1/p`: with
/// `q = floor(1/p)` and `r = 1/p - q`, `W = q + Bernoulli(r)`, drawn
/// independently per admitted update. That makes `E[W] = 1/p` for *every*
/// rate rather than only for reciprocal-integer ones, so
///
/// ```text
///   E[est] = f * p * (1/p) = f
///   Var[W] = r (1 - r)
///   Var[est] = E[X] Var[W] + Var[X] E[W]^2
///            = f p r(1-r) + f p (1-p) / p^2
///            = f ( p r (1-r) + (1-p)/p )
/// ```
///
/// so the accepted band is `f +- z * sqrt(f (p r (1-r) + (1-p)/p))`, plus
/// whatever the wrapped sketch's own error contributes. At `p = 1` both terms
/// vanish and the band collapses to zero width, which is correct — full
/// sampling is exact. At a reciprocal-integer rate `r = 0` and the formula
/// reduces to the familiar `sqrt(f (1-p) / p)`.
///
/// # Trial unit
///
/// The randomness is the sampling RNG. One `(seed, key)` probe is one draw;
/// two keys of the *same* sketch share one skip sequence and are dependent, so
/// a binomial must be taken over **sampling seeds**, with each seed's battery
/// of keys reduced to a single outcome first.
///
/// `z` is a Gaussian quantile, so this is an asymptotic normal band on a sum of
/// `f` i.i.d. terms — accurate for the counts these suites use (`f p` in the
/// hundreds) but a model, not an exact tail.
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfidenceSpec {
    pub rate: f64,
    pub z: f64,
}

impl SamplingConfidenceSpec {
    pub fn new(rate: f64, z: f64) -> Self {
        Self { rate, z }
    }

    /// `floor(1/p)` and the fractional part `1/p - floor(1/p)` the
    /// implementation's stochastic rounding uses.
    pub fn weight_parts(&self) -> (f64, f64) {
        if self.rate >= 1.0 {
            return (1.0, 0.0);
        }
        let inv = 1.0 / self.rate;
        let floor = inv.floor();
        (floor, inv - floor)
    }

    /// Expected weight per admitted update — `1/p` exactly, by construction.
    pub fn expected_weight(&self) -> f64 {
        if self.rate >= 1.0 {
            1.0
        } else {
            1.0 / self.rate
        }
    }

    /// Standard deviation of the scaled estimate for a key of count `f`.
    pub fn sigma(&self, true_count: f64) -> f64 {
        if self.rate >= 1.0 {
            return 0.0;
        }
        let p = self.rate;
        let (_, r) = self.weight_parts();
        (true_count * (p * r * (1.0 - r) + (1.0 - p) / p)).sqrt()
    }

    /// Half-width of the accepted band, before any sketch-side error.
    pub fn half_width(&self, true_count: f64) -> f64 {
        self.z * self.sigma(true_count)
    }

    pub fn per_check_failure(&self) -> f64 {
        2.0 * (1.0 - standard_normal_cdf(self.z))
    }

    /// `sketch_slack` is added to the sampling band for the wrapped sketch's
    /// own error (zero when the sketch is exact for the probed key).
    pub fn check(&self, estimate: f64, true_count: f64, sketch_slack: f64) -> Result<(), String> {
        let half = self.half_width(true_count) + sketch_slack;
        if (estimate - true_count).abs() <= half {
            Ok(())
        } else {
            let (q, r) = self.weight_parts();
            Err(format!(
                "rate={} (weight {q}+Bernoulli({r:.4}), E[W]={:.4}): estimate {estimate:.1} vs \
                 true {true_count:.0}, |error| {:.1} > z*sigma + slack = {:.2}*{:.2} + \
                 {sketch_slack:.1} = {half:.1}",
                self.rate,
                self.expected_weight(),
                (estimate - true_count).abs(),
                self.z,
                self.sigma(true_count),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Priority (bottom-k) sampling
// ---------------------------------------------------------------------------

/// `UniformSampling`'s retention and sample-correctness model.
///
/// The sampler draws an independent uniform 64-bit priority per update and
/// retains the `m = ceil(n * rate)` entries of *smallest* priority. Two things
/// follow, and they need very different acceptance rules:
///
/// - **Structural, exact:** `len() == ceil(total_seen * rate)` once anything has
///   been seen (never a band — the truncation target is computed, not sampled),
///   every retained value is one of the values fed in, and no value is retained
///   more times than it was fed.
/// - **Statistical:** because the priorities are i.i.d. and independent of the
///   values, the retained set is a uniform sample of size `m` drawn **without
///   replacement** from the `n` values seen. The sample mean of a numeric
///   stream therefore has
///
///   ```text
///     E[mean_hat]   = mu
///     Var[mean_hat] = (sigma_N^2 / m) * (N - m) / (N - 1)
///   ```
///
///   where `sigma_N^2` is the population variance with divisor `N` and the
///   second factor is the finite-population correction.
///
/// # Trial unit
///
/// One seed is one draw of the whole priority sequence, so one sampler is one
/// trial. Several statistics read off the same sampler are not independent.
#[derive(Clone, Copy, Debug)]
pub struct PrioritySampleSpec {
    pub rate: f64,
    pub z: f64,
}

impl PrioritySampleSpec {
    pub fn new(rate: f64, z: f64) -> Self {
        Self { rate, z }
    }

    /// `ceil(n * rate)` — the retained size, exactly.
    pub fn retained(&self, total_seen: u64) -> usize {
        if total_seen == 0 {
            0
        } else {
            ((total_seen as f64) * self.rate).ceil() as usize
        }
    }

    /// Standard deviation of the sample mean under sampling without
    /// replacement of `m` out of `population` items with population variance
    /// `population_variance` (divisor `N`).
    pub fn mean_sigma(&self, population: usize, sample: usize, population_variance: f64) -> f64 {
        if sample == 0 || population <= 1 {
            return f64::INFINITY;
        }
        let n = population as f64;
        let m = sample as f64;
        (population_variance / m * ((n - m) / (n - 1.0)).max(0.0)).sqrt()
    }

    pub fn per_check_failure(&self) -> f64 {
        2.0 * (1.0 - standard_normal_cdf(self.z))
    }
}

// ---------------------------------------------------------------------------
// Shared numerics
// ---------------------------------------------------------------------------

/// `Phi(z)` via the Abramowitz & Stegun 7.1.26 error-function approximation
/// (absolute error < 1.5e-7), which is far finer than any band it sizes.
pub fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}
