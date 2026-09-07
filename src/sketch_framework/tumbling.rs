//! Tumbling (non-overlapping, fixed-duration) window support for sketches.
//!
//! A tumbling window collects items for a fixed duration, then closes. Closed
//! windows are retained up to a configurable limit and can be merged for
//! aggregate queries. A [`SketchPool`] recycles sketch instances when windows
//! expire, avoiding allocation churn.
//!
//! # Supported sketch types
//!
//! Any type implementing [`TumblingWindowSketch`] can be used. Built-in
//! implementations are provided for [`FoldCMS`], [`FoldCS`], [`KLL`], and
//! experimental [`UnivMonQ`].

use crate::DataInput;
use crate::fold_cms::FoldCMS;
use crate::fold_cs::FoldCS;
use crate::kll::KLL;
use crate::{UnivMonQ, UnivMonQConfig};

// ---------------------------------------------------------------------------
// TumblingWindowSketch trait
// ---------------------------------------------------------------------------

/// Trait unifying insert / merge / clear / construct for pool and tumbling
/// window manager use.
///
/// The `tumbling_` prefix avoids collision with existing method names on the
/// underlying sketch types.
pub trait TumblingWindowSketch: Clone + Sized {
    /// Configuration sufficient to construct a fresh instance.
    type Config: Clone;

    /// Create a new sketch from its configuration.
    fn from_config(config: &Self::Config) -> Self;

    /// Insert one observation.
    fn tumbling_insert(&mut self, key: &DataInput, value: i64);

    /// Merge `other` into `self`.
    fn tumbling_merge(&mut self, other: &Self);

    /// Reset to the empty state, preserving allocations where possible.
    fn tumbling_clear(&mut self);
}

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

/// Configuration for constructing a [`FoldCMS`] via [`TumblingWindowSketch`].
#[derive(Clone, Debug)]
pub struct FoldCMSConfig {
    /// Number of sketch rows.
    pub rows: usize,
    /// Full-width column count before folding.
    pub full_cols: usize,
    /// Folding depth applied to the sketch.
    pub fold_level: u32,
    /// Capacity of the heavy-hitter heap.
    pub top_k: usize,
}

/// Configuration for constructing a [`FoldCS`] via [`TumblingWindowSketch`].
#[derive(Clone, Debug)]
pub struct FoldCSConfig {
    /// Number of sketch rows.
    pub rows: usize,
    /// Full-width column count before folding.
    pub full_cols: usize,
    /// Folding depth applied to the sketch.
    pub fold_level: u32,
    /// Capacity of the heavy-hitter heap.
    pub top_k: usize,
}

/// Configuration for constructing a [`KLL`] via [`TumblingWindowSketch`].
#[derive(Clone, Debug)]
pub struct KLLConfig {
    /// Main KLL capacity parameter.
    pub k: usize,
    /// Minimum compactor size parameter.
    pub m: usize,
    /// Optional seed for the compaction coin. When `Some`, every pool sketch
    /// is built with [`KLL::init_with_seed`], so two processes running the
    /// same tumbling pipeline produce byte-identical sketches — including
    /// across window rotations, because `tumbling_clear` re-seeds from the
    /// stored seed. When `None`, the legacy wall-clock-seeded [`KLL::init`]
    /// path is used and cross-process agreement is not achievable.
    pub seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// TumblingWindowSketch impls
// ---------------------------------------------------------------------------

impl TumblingWindowSketch for FoldCMS {
    type Config = FoldCMSConfig;

    fn from_config(config: &Self::Config) -> Self {
        FoldCMS::new(
            config.rows,
            config.full_cols,
            config.fold_level,
            config.top_k,
        )
    }

    fn tumbling_insert(&mut self, key: &DataInput, value: i64) {
        self.insert(key, value);
    }

    fn tumbling_merge(&mut self, other: &Self) {
        self.merge_same_level(other);
    }

    fn tumbling_clear(&mut self) {
        self.clear();
    }
}

impl TumblingWindowSketch for FoldCS {
    type Config = FoldCSConfig;

    fn from_config(config: &Self::Config) -> Self {
        FoldCS::new(
            config.rows,
            config.full_cols,
            config.fold_level,
            config.top_k,
        )
    }

    fn tumbling_insert(&mut self, key: &DataInput, value: i64) {
        self.insert(key, value);
    }

    fn tumbling_merge(&mut self, other: &Self) {
        self.merge_same_level(other);
    }

    fn tumbling_clear(&mut self) {
        self.clear();
    }
}

impl TumblingWindowSketch for KLL {
    type Config = KLLConfig;

    fn from_config(config: &Self::Config) -> Self {
        match config.seed {
            Some(seed) => KLL::init_with_seed(config.k, config.m, seed),
            None => KLL::init(config.k, config.m),
        }
    }

    fn tumbling_insert(&mut self, key: &DataInput, _value: i64) {
        // KLL is a quantile sketch — each call is one observation.
        let _ = self.update_data_input(key);
    }

    fn tumbling_merge(&mut self, other: &Self) {
        self.merge(other);
    }

    fn tumbling_clear(&mut self) {
        self.clear();
    }
}

impl TumblingWindowSketch for UnivMonQ {
    type Config = UnivMonQConfig;

    fn from_config(config: &Self::Config) -> Self {
        UnivMonQ::new(*config).expect("TumblingWindow received an invalid UnivMon-Q config")
    }

    fn tumbling_insert(&mut self, key: &DataInput, _value: i64) {
        let _ = self.update_data_input(key);
    }

    fn tumbling_merge(&mut self, other: &Self) {
        self.merge(other)
            .expect("TumblingWindow UnivMon-Q configurations must match");
    }

    fn tumbling_clear(&mut self) {
        self.clear();
    }
}

// ---------------------------------------------------------------------------
// SketchPool
// ---------------------------------------------------------------------------

/// Generic object pool that recycles sketch instances via [`TumblingWindowSketch::tumbling_clear`].
pub struct SketchPool<S: TumblingWindowSketch> {
    free_list: Vec<S>,
    total_allocated: usize,
    config: S::Config,
}

impl<S: TumblingWindowSketch> SketchPool<S> {
    /// Create a pool and pre-allocate `cap` sketches.
    pub fn new(cap: usize, config: S::Config) -> Self {
        let mut free_list = Vec::with_capacity(cap);
        for _ in 0..cap {
            free_list.push(S::from_config(&config));
        }
        SketchPool {
            free_list,
            total_allocated: cap,
            config,
        }
    }

    /// Take a sketch from the free-list, or allocate a fresh one.
    pub fn take(&mut self) -> S {
        if let Some(s) = self.free_list.pop() {
            s
        } else {
            self.total_allocated += 1;
            S::from_config(&self.config)
        }
    }

    /// Return a sketch to the pool after clearing it.
    pub fn put(&mut self, mut sketch: S) {
        sketch.tumbling_clear();
        self.free_list.push(sketch);
    }

    /// Number of sketches currently available in the free-list.
    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of sketches ever allocated by this pool.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }
}

// ---------------------------------------------------------------------------
// ClosedWindow
// ---------------------------------------------------------------------------

/// A closed (immutable) tumbling window with its sketch and metadata.
struct ClosedWindow<S: TumblingWindowSketch> {
    sketch: S,
    _window_id: u64,
}

// ---------------------------------------------------------------------------
// TumblingWindow
// ---------------------------------------------------------------------------

/// Manages a sequence of tumbling (non-overlapping) windows over a sketch
/// type `S`. Each window collects items for `window_size` time units, then
/// closes. At most `max_windows` closed windows are retained; older ones
/// are evicted and their sketches returned to the pool.
pub struct TumblingWindow<S: TumblingWindowSketch> {
    /// Currently open window's sketch.
    active: S,
    /// Sequential counter for window IDs.
    active_window_id: u64,
    /// Timestamp when the active window opened.
    active_start: u64,
    /// Duration of each window.
    window_size: u64,
    /// Maximum number of closed windows to retain.
    max_windows: usize,
    /// Closed windows, ordered oldest-to-newest.
    closed: Vec<ClosedWindow<S>>,
    /// Pool for recycling sketch instances.
    pool: SketchPool<S>,
}

impl<S: TumblingWindowSketch> TumblingWindow<S> {
    /// Create a new tumbling window manager.
    ///
    /// * `window_size` — duration of each window in abstract time units.
    /// * `max_windows` — maximum number of closed windows to retain.
    /// * `config`      — sketch configuration for constructing fresh sketches.
    /// * `pool_cap`    — initial number of pre-allocated pool sketches.
    pub fn new(window_size: u64, max_windows: usize, config: S::Config, pool_cap: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        let mut pool = SketchPool::new(pool_cap, config.clone());
        let active = pool.take();
        TumblingWindow {
            active,
            active_window_id: 0,
            active_start: 0,
            window_size,
            max_windows,
            closed: Vec::with_capacity(max_windows),
            pool,
        }
    }

    /// Insert an observation at the given timestamp.
    ///
    /// If `time` falls beyond the current window boundary, the active window
    /// is closed and new windows are opened as needed (empty intermediate
    /// windows are skipped).
    pub fn insert(&mut self, time: u64, key: &DataInput, value: i64) {
        // Advance windows as needed.
        while time >= self.active_start + self.window_size {
            self.close_active();
        }
        self.active.tumbling_insert(key, value);
    }

    /// Force-close the active window at `current_time` and open a fresh one.
    pub fn flush(&mut self, current_time: u64) {
        while current_time >= self.active_start + self.window_size {
            self.close_active();
        }
        // If the active window still contains the current_time, close it anyway.
        self.close_active();
    }

    /// Close the active window, push it to closed, evict if over limit.
    fn close_active(&mut self) {
        let old_active = std::mem::replace(&mut self.active, self.pool.take());
        self.closed.push(ClosedWindow {
            sketch: old_active,
            _window_id: self.active_window_id,
        });
        self.active_window_id += 1;
        self.active_start += self.window_size;

        // Evict oldest if over limit.
        while self.closed.len() > self.max_windows {
            let evicted = self.closed.remove(0);
            self.pool.put(evicted.sketch);
        }
    }

    /// Merge all closed windows + the active window into a single sketch.
    pub fn query_all(&self) -> S {
        let mut merged = self.active.clone();
        for cw in &self.closed {
            merged.tumbling_merge(&cw.sketch);
        }
        merged
    }

    /// Merge the `n` most recent closed windows + the active window.
    ///
    /// If `n >= closed_count()`, this is equivalent to `query_all()`.
    pub fn query_recent(&self, n: usize) -> S {
        let mut merged = self.active.clone();
        let start = self.closed.len().saturating_sub(n);
        for cw in &self.closed[start..] {
            merged.tumbling_merge(&cw.sketch);
        }
        merged
    }

    /// Reference to the active (currently open) sketch.
    pub fn active_sketch(&self) -> &S {
        &self.active
    }

    /// Number of closed windows currently retained.
    pub fn closed_count(&self) -> usize {
        self.closed.len()
    }

    /// Number of sketches available in the pool.
    pub fn pool_available(&self) -> usize {
        self.pool.available()
    }

    /// Total sketches ever allocated by the pool.
    pub fn pool_total_allocated(&self) -> usize {
        self.pool.total_allocated()
    }
}

// ---------------------------------------------------------------------------
// Specialized hierarchical merge queries for FoldCMS / FoldCS
// ---------------------------------------------------------------------------

impl TumblingWindow<FoldCMS> {
    /// Hierarchically merges all windows into one FoldCMS.
    ///
    /// This produces a progressively unfolded result — more accurate than
    /// repeated `merge_same_level` when the fold level is > 0.
    pub fn query_all_hierarchical(&self) -> FoldCMS {
        let sketches: Vec<FoldCMS> = self
            .closed
            .iter()
            .map(|cw| cw.sketch.clone())
            .chain(std::iter::once(self.active.clone()))
            .collect();
        FoldCMS::hierarchical_merge(&sketches)
    }
}

impl TumblingWindow<FoldCS> {
    /// Hierarchically merges all windows into one FoldCS.
    pub fn query_all_hierarchical(&self) -> FoldCS {
        let sketches: Vec<FoldCS> = self
            .closed
            .iter()
            .map(|cw| cw.sketch.clone())
            .chain(std::iter::once(self.active.clone()))
            .collect();
        FoldCS::hierarchical_merge(&sketches)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_normal_f64, sample_uniform_f64, sample_zipf_u64};
    use std::collections::HashMap;

    // -- Pool tests ----------------------------------------------------------

    #[test]
    fn pool_take_returns_preallocated() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut pool = SketchPool::<FoldCMS>::new(4, config);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.total_allocated(), 4);

        let _s1 = pool.take();
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.total_allocated(), 4);
    }

    #[test]
    fn pool_take_allocates_when_empty() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut pool = SketchPool::<FoldCMS>::new(0, config);
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.total_allocated(), 0);

        let _s = pool.take();
        assert_eq!(pool.total_allocated(), 1);
    }

    #[test]
    fn pool_put_recycles() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut pool = SketchPool::<FoldCMS>::new(1, config);
        let s = pool.take();
        assert_eq!(pool.available(), 0);

        pool.put(s);
        assert_eq!(pool.available(), 1);
        assert_eq!(pool.total_allocated(), 1);
    }

    // -- Clear method tests --------------------------------------------------

    #[test]
    fn fold_cms_clear_resets_to_empty() {
        let mut sk: FoldCMS = FoldCMS::new(3, 1024, 3, 10);
        for i in 0..50u64 {
            sk.insert(&DataInput::U64(i), 1);
        }
        assert!(sk.query(&DataInput::U64(0)) > 0);

        sk.clear();

        for i in 0..50u64 {
            assert_eq!(
                sk.query(&DataInput::U64(i)),
                0,
                "key {i} should be 0 after clear"
            );
        }
        assert!(sk.heap().is_empty(), "heap should be empty after clear");
    }

    #[test]
    fn fold_cs_clear_resets_to_empty() {
        let mut sk: FoldCS = FoldCS::new(3, 1024, 3, 10);
        for i in 0..50u64 {
            sk.insert(&DataInput::U64(i), 1);
        }
        assert_ne!(sk.query(&DataInput::U64(0)), 0);

        sk.clear();

        for i in 0..50u64 {
            assert_eq!(
                sk.query(&DataInput::U64(i)),
                0,
                "key {i} should be 0 after clear"
            );
        }
        assert!(sk.heap().is_empty(), "heap should be empty after clear");
    }

    #[test]
    fn kll_clear_resets_to_empty() {
        let mut sk = KLL::init(200, 8);
        for i in 0..1000 {
            sk.update_data_input(&DataInput::F64(i as f64)).unwrap();
        }
        assert!(sk.count() > 0);

        sk.clear();

        assert_eq!(sk.count(), 0, "count should be 0 after clear");
        let cdf = sk.cdf();
        assert_eq!(cdf.query(0.5), 0.0, "empty sketch should return 0.0");
    }

    #[test]
    fn univmon_q_tumbling_merges_quantiles_and_universal_metrics() {
        let config = UnivMonQConfig {
            levels: 8,
            width: 256,
            width_halving_period: 0,
            depth: 5,
            counter_bits: 64,
            candidates: 64,
            ordered_samples: 64,
            hash_seed: 5,
        };
        let mut windows: TumblingWindow<UnivMonQ> = TumblingWindow::new(100, 4, config, 2);
        for value in 0..300_u64 {
            windows.insert(value, &DataInput::F64((value % 50) as f64), 1);
        }
        let merged = windows.query_all();
        assert_eq!(merged.count(), 300);
        assert_eq!(merged.quantile(0.5), Some(24.0));
        assert_eq!(merged.estimate_distinct(), 50.0);
    }

    // -- Construction guard tests --------------------------------------------

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn zero_window_size_panics() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let _tw = TumblingWindow::<FoldCMS>::new(0, 5, config, 4);
    }

    // -- Window mechanics tests ----------------------------------------------

    #[test]
    fn window_closes_on_time_advance() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(100, 10, config, 4);

        // Insert into window 0 (time 0..99).
        tw.insert(0, &DataInput::Str("a"), 1);
        tw.insert(50, &DataInput::Str("a"), 1);
        assert_eq!(tw.closed_count(), 0);

        // Time 100 → window 0 closes, window 1 opens.
        tw.insert(100, &DataInput::Str("b"), 1);
        assert_eq!(tw.closed_count(), 1);

        // Time 200 → window 1 closes.
        tw.insert(200, &DataInput::Str("c"), 1);
        assert_eq!(tw.closed_count(), 2);
    }

    #[test]
    fn window_evicts_oldest_beyond_max() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(100, 3, config, 4);

        // One key per window: five windows open, four of them close, and the
        // fourth close pushes the retained count past max_windows.
        for w in 0..5 {
            tw.insert(w * 100, &DataInput::U64(w), 1);
        }

        assert_eq!(
            tw.closed_count(),
            3,
            "max_windows 3 retains exactly three closed windows"
        );

        // Windows 1, 2 and 3 are the retained ones and window 4 is active, so
        // the merge sees their keys and not window 0's.
        let merged = tw.query_all();
        for w in 1..5u64 {
            assert_eq!(
                merged.query(&DataInput::U64(w)),
                1,
                "window {w}'s key is still in the merge"
            );
        }
        assert_eq!(
            merged.query(&DataInput::U64(0)),
            0,
            "the evicted window's key is gone from the merge, not merely capped"
        );
    }

    #[test]
    fn window_pool_recycles_on_eviction() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(100, 2, config, 4);

        let initial_total = tw.pool_total_allocated();

        // Six keys open six windows: five close and three are evicted, so
        // three sketches go back to the pool and every close after the first
        // eviction is served from the free list.
        for w in 0..6 {
            tw.insert(w * 100, &DataInput::U64(w), 1);
        }

        assert_eq!(tw.closed_count(), 2, "max_windows 2 retains two windows");
        assert_eq!(
            tw.pool_available(),
            1,
            "the last evicted sketch is waiting in the pool"
        );
        assert_eq!(
            tw.pool_total_allocated(),
            initial_total,
            "recycling covers every window after the pool was primed, so nothing new is allocated"
        );

        // The free list is a stack, so the active window is running on the
        // sketch window 2 was using. `put` clears on the way in, so window 2's
        // key must not read back through it.
        assert_eq!(
            tw.active_sketch().query(&DataInput::U64(5)),
            1,
            "the recycled sketch holds the active window's own key"
        );
        assert_eq!(
            tw.active_sketch().query(&DataInput::U64(2)),
            0,
            "a recycled sketch is cleared, so its previous window's key is gone"
        );
    }

    // -- Merge correctness tests ---------------------------------------------

    #[test]
    fn query_all_matches_manual_merge() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(100, 10, config.clone(), 4);

        let mut manual: FoldCMS = FoldCMS::new(
            config.rows,
            config.full_cols,
            config.fold_level,
            config.top_k,
        );

        let keys: Vec<DataInput> = (0..20u64).map(DataInput::U64).collect();
        for (i, key) in keys.iter().enumerate() {
            let time = (i as u64) * 30; // spread across windows
            tw.insert(time, key, 1);
            manual.insert(key, 1);
        }

        let merged = tw.query_all();
        for key in &keys {
            assert_eq!(
                merged.query(key),
                manual.query(key),
                "query_all mismatch for {key:?}"
            );
        }
    }

    #[test]
    fn query_recent_selects_subset() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(100, 10, config, 4);

        // Window 0: key "old"
        tw.insert(0, &DataInput::Str("old"), 5);
        // Window 1: key "new"
        tw.insert(100, &DataInput::Str("new"), 10);
        // Window 2 (active): key "active"
        tw.insert(200, &DataInput::Str("active"), 7);

        // query_recent(1) should include 1 most recent closed + active.
        let recent = tw.query_recent(1);
        assert_eq!(recent.query(&DataInput::Str("new")), 10);
        assert_eq!(recent.query(&DataInput::Str("active")), 7);
        // "old" is in window 0 which is not in the recent 1.
        assert_eq!(recent.query(&DataInput::Str("old")), 0);
    }

    // -- FoldCMS hierarchical merge via tumbling windows ----------------------

    #[test]
    fn fold_cms_tumbling_hierarchical_merge() {
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 5000;
        let exponent = 1.1;
        let samples_per_window = 10_000;
        let num_windows = 8;

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let mut truth = HashMap::<u64, i64>::new();
        let stream = sample_zipf_u64(
            domain,
            exponent,
            samples_per_window * num_windows,
            0xCAFE_BABE,
        );

        for (i, &value) in stream.iter().enumerate() {
            tw.insert(i as u64, &DataInput::U64(value), 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged = tw.query_all_hierarchical();

        // Verify most estimates are within CMS error bound.
        let epsilon = std::f64::consts::E / full_cols as f64;
        let l1_norm: f64 = truth.values().map(|&c| c as f64).sum();
        let error_bound = epsilon * l1_norm;

        let mut within = 0usize;
        for (&key, &true_count) in &truth {
            let est = merged.query(&DataInput::U64(key));
            if ((est - true_count).abs() as f64) < error_bound {
                within += 1;
            }
        }

        let pct = within as f64 / truth.len() as f64 * 100.0;
        assert!(
            pct > 90.0,
            "only {pct:.1}% within CMS error bound (expected > 90%)"
        );
    }

    // -- KLL quantile via tumbling windows ------------------------------------

    #[test]
    fn kll_tumbling_quantile_accuracy() {
        let config = KLLConfig {
            k: 200,
            m: 8,
            seed: None,
        };
        let samples_per_window = 5000;
        let num_windows = 4;

        let mut tw: TumblingWindow<KLL> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let values = sample_uniform_f64(
            0.0,
            1_000_000.0,
            samples_per_window * num_windows,
            0xDEAD_BEEF,
        );

        for (i, &v) in values.iter().enumerate() {
            tw.insert(i as u64, &DataInput::F64(v), 1);
        }

        let merged = tw.query_all();
        let cdf = merged.cdf();

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Check median is roughly correct (within 2% rank tolerance).
        let est_median = cdf.query(0.5);
        let n = sorted.len();
        let lower_idx = ((0.48 * n as f64).ceil() as usize).min(n - 1);
        let upper_idx = ((0.52 * n as f64).ceil() as usize).min(n - 1);
        assert!(
            est_median >= sorted[lower_idx] && est_median <= sorted[upper_idx],
            "median estimate {est_median} outside [{}, {}]",
            sorted[lower_idx],
            sorted[upper_idx]
        );
    }

    // -- Flush test ----------------------------------------------------------

    #[test]
    fn flush_closes_active_window() {
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(100, 10, config, 4);

        tw.insert(10, &DataInput::Str("x"), 5);
        assert_eq!(tw.closed_count(), 0);

        tw.flush(50);
        assert_eq!(tw.closed_count(), 1);

        // Active should now be empty.
        assert_eq!(tw.active_sketch().query(&DataInput::Str("x")), 0);

        // But query_all should still find the data.
        let all = tw.query_all();
        assert_eq!(all.query(&DataInput::Str("x")), 5);
    }

    // -- FoldCS tumbling test ------------------------------------------------

    #[test]
    fn fold_cs_tumbling_basic() {
        let config = FoldCSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCS> = TumblingWindow::new(100, 10, config, 4);

        tw.insert(0, &DataInput::Str("hello"), 5);
        tw.insert(100, &DataInput::Str("hello"), 3);
        tw.insert(200, &DataInput::Str("hello"), 2);

        let merged = tw.query_all();
        assert_eq!(merged.query(&DataInput::Str("hello")), 10);
    }

    #[test]
    fn fold_cs_tumbling_hierarchical_merge() {
        let config = FoldCSConfig {
            rows: 5,
            full_cols: 4096,
            fold_level: 2,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCS> = TumblingWindow::new(100, 10, config, 6);

        for w in 0..4u64 {
            for i in (w * 10)..((w + 1) * 10) {
                tw.insert(w * 100 + i, &DataInput::U64(i), 1);
            }
        }

        let merged = tw.query_all_hierarchical();
        assert_eq!(merged.fold_level(), 0);

        // Count Sketch uses signed counters + median; allow small error.
        let mut errors = 0;
        for i in 0..40u64 {
            let est = merged.query(&DataInput::U64(i));
            if (est - 1).abs() > 1 {
                errors += 1;
            }
        }
        assert!(
            errors == 0,
            "{errors}/40 keys had error > 1 (expected 0 with wide sketch)"
        );
    }

    // -- Accuracy & correctness tests ----------------------------------------

    #[test]
    fn fold_cms_tumbling_accuracy_zipf() {
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 10_000;
        let exponent = 1.1;
        let total_samples = 500_000;
        let num_windows = 16;
        let samples_per_window = total_samples / num_windows;

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let stream = sample_zipf_u64(domain, exponent, total_samples, 0xACC0_BAC1);
        let mut truth = HashMap::<u64, i64>::new();

        for (i, &value) in stream.iter().enumerate() {
            tw.insert(i as u64, &DataInput::U64(value), 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged = tw.query_all();

        // CMS error bound: epsilon = e / full_cols, bound = epsilon * L1
        let epsilon = std::f64::consts::E / full_cols as f64;
        let l1_norm: f64 = truth.values().map(|&c| c as f64).sum();
        let error_bound = epsilon * l1_norm;
        // delta = 1 / e^rows → expected fraction within bound = 1 - delta
        let delta = (-(rows as f64)).exp();
        let required_pct = (1.0 - delta) * 100.0;

        let mut within = 0usize;
        let mut total_abs_error = 0i64;
        let mut max_abs_error = 0i64;
        for (&key, &true_count) in &truth {
            let est = merged.query(&DataInput::U64(key));
            let err = (est - true_count).abs();
            total_abs_error += err;
            max_abs_error = max_abs_error.max(err);
            if (err as f64) <= error_bound {
                within += 1;
            }
        }

        let pct = within as f64 / truth.len() as f64 * 100.0;
        let mean_abs_error = total_abs_error as f64 / truth.len() as f64;
        eprintln!(
            "[fold_cms_tumbling_accuracy_zipf] mean_abs_error={mean_abs_error:.2}, \
             max_abs_error={max_abs_error}, pct_within_bound={pct:.1}% \
             (required>{required_pct:.1}%), error_bound={error_bound:.2}"
        );

        assert!(
            pct >= required_pct,
            "only {pct:.1}% within CMS error bound (expected >= {required_pct:.1}%)"
        );
    }

    #[test]
    fn fold_cms_hierarchical_vs_flat_merge() {
        let rows = 3;
        let full_cols = 4096;
        // fold_level = 3 so hierarchical merge with 8 windows (log2(8)=3 rounds)
        // reaches fold_level 0.
        let fold_level = 3;
        let top_k = 20;
        let domain = 5000;
        let exponent = 1.1;
        let total_samples = 100_000;
        let num_windows = 8;
        let samples_per_window = total_samples / num_windows;

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };

        let stream = sample_zipf_u64(domain, exponent, total_samples, 0xF1A7_CAFE);

        // Build two identical tumbling window instances.
        let mut tumbling_flat: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config.clone(),
            num_windows + 2,
        );
        let mut tumbling_hier: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let mut truth = HashMap::<u64, i64>::new();
        for (i, &value) in stream.iter().enumerate() {
            let t = i as u64;
            let key = DataInput::U64(value);
            tumbling_flat.insert(t, &key, 1);
            tumbling_hier.insert(t, &key, 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged_flat = tumbling_flat.query_all();
        let merged_hier = tumbling_hier.query_all_hierarchical();

        // Hierarchical merge should reach fold_level 0.
        assert_eq!(
            merged_hier.fold_level(),
            0,
            "hierarchical merge should reach fold_level 0"
        );

        // Both should produce identical estimates for all keys.
        // Note: flat merge stays at the original fold_level, so we unfold it
        // for a fair comparison.
        let merged_flat_unfolded = merged_flat.unfold_full();

        for &key in truth.keys() {
            let est_flat = merged_flat_unfolded.query(&DataInput::U64(key));
            let est_hier = merged_hier.query(&DataInput::U64(key));
            assert_eq!(
                est_flat, est_hier,
                "flat vs hierarchical mismatch for key {key}: flat={est_flat}, hier={est_hier}"
            );
        }
    }

    #[test]
    fn fold_cs_tumbling_accuracy_zipf() {
        let rows = 5;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 10_000;
        let exponent = 1.1;
        let total_samples = 500_000;
        let num_windows = 16;
        let samples_per_window = total_samples / num_windows;

        let config = FoldCSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let stream = sample_zipf_u64(domain, exponent, total_samples, 0xC5_ACCA);
        let mut truth = HashMap::<u64, i64>::new();

        for (i, &value) in stream.iter().enumerate() {
            tw.insert(i as u64, &DataInput::U64(value), 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged = tw.query_all();

        // CS error bound: epsilon = sqrt(e / cols), bound = epsilon * L2
        let epsilon = (std::f64::consts::E / full_cols as f64).sqrt();
        let l2_norm: f64 = truth
            .values()
            .map(|&c| (c as f64) * (c as f64))
            .sum::<f64>()
            .sqrt();
        let error_bound = epsilon * l2_norm;
        let delta = (-(rows as f64)).exp();
        let required_pct = (1.0 - delta) * 100.0;

        let mut within = 0usize;
        let mut total_abs_error = 0i64;
        let mut max_abs_error = 0i64;
        for (&key, &true_count) in &truth {
            let est = merged.query(&DataInput::U64(key));
            let err = (est - true_count).abs();
            total_abs_error += err;
            max_abs_error = max_abs_error.max(err);
            if (err as f64) <= error_bound {
                within += 1;
            }
        }

        let pct = within as f64 / truth.len() as f64 * 100.0;
        let mean_abs_error = total_abs_error as f64 / truth.len() as f64;
        eprintln!(
            "[fold_cs_tumbling_accuracy_zipf] mean_abs_error={mean_abs_error:.2}, \
             max_abs_error={max_abs_error}, pct_within_bound={pct:.1}% \
             (required>{required_pct:.1}%), error_bound={error_bound:.2}"
        );

        assert!(
            pct >= required_pct,
            "only {pct:.1}% within CS error bound (expected >= {required_pct:.1}%)"
        );
    }

    #[test]
    fn kll_tumbling_multi_quantile_accuracy() {
        let config = KLLConfig {
            k: 200,
            m: 8,
            seed: None,
        };
        let total_samples = 100_000;
        let num_windows = 8;
        let samples_per_window = total_samples / num_windows;

        let mut tw: TumblingWindow<KLL> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let values = sample_uniform_f64(0.0, 10_000_000.0, total_samples, 0x411_00171);

        for (i, &v) in values.iter().enumerate() {
            tw.insert(i as u64, &DataInput::F64(v), 1);
        }

        let merged = tw.query_all();
        let cdf = merged.cdf();

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();

        let quantiles = [0.10, 0.25, 0.50, 0.75, 0.90];
        let tolerance = 0.02;

        for &q in &quantiles {
            let est = cdf.query(q);
            let lo_idx = (((q - tolerance).max(0.0) * n as f64).ceil() as usize).min(n - 1);
            let hi_idx = (((q + tolerance).min(1.0) * n as f64).ceil() as usize).min(n - 1);
            assert!(
                est >= sorted[lo_idx] && est <= sorted[hi_idx],
                "quantile {q} estimate {est} outside [{}, {}] (rank tolerance {tolerance})",
                sorted[lo_idx],
                sorted[hi_idx]
            );
        }
    }

    #[test]
    fn kll_tumbling_distribution_shift() {
        let config = KLLConfig {
            k: 400,
            m: 8,
            seed: None,
        };
        let samples_per_phase = 50_000;
        let windows_per_phase = 4;
        let samples_per_window = samples_per_phase / windows_per_phase;
        let num_windows = windows_per_phase * 2;

        let mut tw: TumblingWindow<KLL> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        // Phase 1: normal(100, 10)
        let phase1 = sample_normal_f64(100.0, 10.0, samples_per_phase, 0xFA_ACE1);
        // Phase 2: normal(500, 50)
        let phase2 = sample_normal_f64(500.0, 50.0, samples_per_phase, 0xFA_ACE2);

        for (i, &v) in phase1.iter().enumerate() {
            tw.insert(i as u64, &DataInput::F64(v), 1);
        }
        for (i, &v) in phase2.iter().enumerate() {
            let t = (samples_per_phase + i) as u64;
            tw.insert(t, &DataInput::F64(v), 1);
        }

        let merged = tw.query_all();
        let cdf = merged.cdf();

        // Check distributional shape rather than exact quantile bounds.
        // With equal halves from normal(100,10) and normal(500,50):
        // - p10 should be solidly in the first distribution (< 200)
        // - p50 should be between the two modes (50 .. 600)
        // - p90 should be solidly in the second distribution (> 350)
        let p10 = cdf.query(0.10);
        let p50 = cdf.query(0.50);
        let p90 = cdf.query(0.90);

        eprintln!("[kll_tumbling_distribution_shift] p10={p10:.1}, p50={p50:.1}, p90={p90:.1}");

        assert!(
            p10 < 200.0,
            "p10={p10:.1} should be in the first distribution (< 200)"
        );
        assert!(
            p50 > 50.0 && p50 < 600.0,
            "p50={p50:.1} should be between modes (50..600)"
        );
        assert!(
            p90 > 350.0,
            "p90={p90:.1} should be in the second distribution (> 350)"
        );

        // Verify ordering is monotonic.
        assert!(p10 < p50, "p10 ({p10:.1}) should be < p50 ({p50:.1})");
        assert!(p50 < p90, "p50 ({p50:.1}) should be < p90 ({p90:.1})");
    }

    // -- Seeded determinism ---------------------------------------------------

    #[test]
    fn kll_tumbling_seeded_is_byte_identical_across_instances() {
        let make = || {
            TumblingWindow::<KLL>::new(
                1000,
                4,
                KLLConfig {
                    k: 200,
                    m: 8,
                    seed: Some(0x5EED_CAFE),
                },
                6,
            )
        };
        // Two independent managers, as if on two hosts/processes.
        let mut tw_a = make();
        let mut tw_b = make();

        let values = sample_uniform_f64(0.0, 1_000_000.0, 20_000, 0x5EED_0001);
        for (i, &v) in values.iter().enumerate() {
            let t = i as u64;
            tw_a.insert(t, &DataInput::F64(v), 1);
            tw_b.insert(t, &DataInput::F64(v), 1);
        }

        let bytes_a = tw_a.query_all().serialize_to_bytes().unwrap();
        let bytes_b = tw_b.query_all().serialize_to_bytes().unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "same KLLConfig seed must give byte-identical tumbling state"
        );

        // A different seed must not silently alias onto the same coin stream.
        let mut tw_c: TumblingWindow<KLL> = TumblingWindow::new(
            1000,
            4,
            KLLConfig {
                k: 200,
                m: 8,
                seed: Some(0x5EED_CAFE + 1),
            },
            6,
        );
        for (i, &v) in values.iter().enumerate() {
            tw_c.insert(i as u64, &DataInput::F64(v), 1);
        }
        let bytes_c = tw_c.query_all().serialize_to_bytes().unwrap();
        assert_ne!(
            bytes_a, bytes_c,
            "different KLLConfig seeds must produce different state"
        );
    }

    #[test]
    fn tumbling_eviction_correctness() {
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 5000;
        let exponent = 1.1;
        let max_windows = 4;
        let total_windows = 8;
        let samples_per_window = 10_000;

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            max_windows,
            config,
            max_windows + 2,
        );

        let stream = sample_zipf_u64(
            domain,
            exponent,
            samples_per_window * total_windows,
            0xE01C_0100,
        );

        // Track per-window truth to identify retained windows.
        let mut per_window_truth: Vec<HashMap<u64, i64>> = Vec::new();
        for w in 0..total_windows {
            let mut window_truth = HashMap::<u64, i64>::new();
            let start = w * samples_per_window;
            let end = start + samples_per_window;
            for (i, &value) in stream.iter().enumerate().take(end).skip(start) {
                tw.insert(i as u64, &DataInput::U64(value), 1);
                *window_truth.entry(value).or_insert(0) += 1;
            }
            per_window_truth.push(window_truth);
        }

        // The last insert is at time (total_windows * samples_per_window - 1),
        // which falls in window (total_windows - 1). After all inserts:
        // - Windows 0..(total_windows-2) are closed.
        // - Window (total_windows-1) is active.
        // - With max_windows=4, only the 4 most recent closed windows are kept
        //   plus the active window.
        //
        // Retained: the most recent max_windows closed + active.
        // Closed windows are windows 0..6 (7 closed). Retain the last 4: 3,4,5,6.
        // Active window: 7.
        // So retained = windows 3,4,5,6,7.
        let retained_start = total_windows.saturating_sub(max_windows + 1);

        // Build ground truth for retained windows only.
        let mut retained_truth = HashMap::<u64, i64>::new();
        for w in per_window_truth
            .iter()
            .take(total_windows)
            .skip(retained_start)
        {
            for (&key, &count) in w {
                *retained_truth.entry(key).or_insert(0) += count;
            }
        }

        // Build ground truth for evicted windows only.
        let mut evicted_only = HashMap::<u64, i64>::new();
        for w in per_window_truth.iter().take(retained_start) {
            for (&key, &count) in w {
                *evicted_only.entry(key).or_insert(0) += count;
            }
        }

        let merged = tw.query_all();

        // CMS error bound for the retained data.
        let epsilon = std::f64::consts::E / full_cols as f64;
        let retained_l1: f64 = retained_truth.values().map(|&c| c as f64).sum();
        let error_bound = epsilon * retained_l1;

        // Retained keys: estimates should be close to retained truth.
        let mut within = 0usize;
        for (&key, &true_count) in &retained_truth {
            let est = merged.query(&DataInput::U64(key));
            if ((est - true_count).abs() as f64) <= error_bound {
                within += 1;
            }
        }
        let pct = within as f64 / retained_truth.len() as f64 * 100.0;
        assert!(
            pct > 90.0,
            "only {pct:.1}% of retained keys within bound (expected > 90%)"
        );

        // Keys that appear ONLY in evicted windows should have small estimates.
        // CMS has inherent false positives (hash collisions), so we check that
        // evicted-only key estimates are within the CMS error bound rather than
        // demanding exact zero.
        for &key in evicted_only.keys() {
            if retained_truth.contains_key(&key) {
                continue; // key also appears in retained windows
            }
            let est = merged.query(&DataInput::U64(key));
            assert!(
                (est as f64) <= error_bound,
                "evicted-only key {key} estimate {est} exceeds error bound {error_bound:.2}"
            );
        }
    }

    #[test]
    fn tumbling_query_recent_accuracy() {
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 5000;
        let exponent = 1.1;
        let total_windows = 6;
        let recent_n = 3;
        let samples_per_window = 10_000;

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            total_windows, // keep all closed
            config,
            total_windows + 2,
        );

        let stream = sample_zipf_u64(
            domain,
            exponent,
            samples_per_window * total_windows,
            0xBEC3_0A00,
        );

        let mut per_window_truth: Vec<HashMap<u64, i64>> = Vec::new();
        for w in 0..total_windows {
            let mut window_truth = HashMap::<u64, i64>::new();
            let start = w * samples_per_window;
            let end = start + samples_per_window;
            for (i, &value) in stream.iter().enumerate().take(end).skip(start) {
                tw.insert(i as u64, &DataInput::U64(value), 1);
                *window_truth.entry(value).or_insert(0) += 1;
            }
            per_window_truth.push(window_truth);
        }

        // query_recent(3) returns: 3 most recent closed + active.
        // After 6 windows of data: windows 0..4 are closed, window 5 is active.
        // recent(3) → closed[2,3,4] + active(5) = windows 2,3,4,5.
        let recent_start = total_windows - recent_n - 1;
        let mut recent_truth = HashMap::<u64, i64>::new();
        for w in per_window_truth
            .iter()
            .take(total_windows)
            .skip(recent_start)
        {
            for (&key, &count) in w {
                *recent_truth.entry(key).or_insert(0) += count;
            }
        }

        let merged = tw.query_recent(recent_n);

        let epsilon = std::f64::consts::E / full_cols as f64;
        let recent_l1: f64 = recent_truth.values().map(|&c| c as f64).sum();
        let error_bound = epsilon * recent_l1;

        let mut within = 0usize;
        for (&key, &true_count) in &recent_truth {
            let est = merged.query(&DataInput::U64(key));
            if ((est - true_count).abs() as f64) <= error_bound {
                within += 1;
            }
        }
        let pct = within as f64 / recent_truth.len() as f64 * 100.0;
        assert!(
            pct > 90.0,
            "only {pct:.1}% of recent keys within bound (expected > 90%)"
        );

        // Keys only in excluded windows (0, 1) should have small estimates.
        // CMS has inherent false positives from hash collisions, so we check
        // estimates are within the error bound rather than demanding exact zero.
        let mut excluded_truth = HashMap::<u64, i64>::new();
        for w in per_window_truth.iter().take(recent_start) {
            for (&key, &count) in w {
                *excluded_truth.entry(key).or_insert(0) += count;
            }
        }
        for &key in excluded_truth.keys() {
            if recent_truth.contains_key(&key) {
                continue;
            }
            let est = merged.query(&DataInput::U64(key));
            assert!(
                (est as f64) <= error_bound,
                "excluded key {key} estimate {est} exceeds error bound {error_bound:.2}"
            );
        }
    }

    #[test]
    fn fold_cms_tumbling_heap_correctness() {
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 10_000;
        let exponent = 1.3; // heavier skew → clearer heavy hitters
        let total_samples = 200_000;
        let num_windows = 8;
        let samples_per_window = total_samples / num_windows;

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let stream = sample_zipf_u64(domain, exponent, total_samples, 0xBEAF_C0DE);
        let mut truth = HashMap::<u64, i64>::new();

        for (i, &value) in stream.iter().enumerate() {
            tw.insert(i as u64, &DataInput::U64(value), 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged = tw.query_all();

        // Ground truth top-k sorted by frequency (descending).
        let mut truth_sorted: Vec<(u64, i64)> = truth.iter().map(|(k, v)| (*k, *v)).collect();
        truth_sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
        let true_top_k: Vec<u64> = truth_sorted.iter().take(top_k).map(|&(k, _)| k).collect();

        // Heap entries from merged sketch.
        let heap_entries = merged.heap().heap();
        assert_eq!(
            heap_entries.len(),
            top_k,
            "merging {num_windows} windows fills the heap"
        );

        // The counts the heap reports, not just the keys it holds. A resident's
        // count is the merged estimate taken when it was reconciled in, so it
        // trails the final estimate but never exceeds it, and Count-Min's
        // one-sided error puts that estimate at or above the truth.
        let mut smallest = i64::MAX;
        for item in heap_entries {
            let crate::HeapItem::U64(key) = item.key else {
                panic!("a U64 stream seats U64 keys, got {:?}", item.key);
            };
            let estimate = merged.query(&DataInput::U64(key));
            assert!(item.count > 0, "resident {key} carries a count of zero");
            assert!(
                item.count <= estimate,
                "resident {key}'s count {} exceeds the merged estimate {estimate}",
                item.count
            );
            if let Some(&exact) = truth.get(&key) {
                assert!(
                    estimate >= exact,
                    "Count-Min undercounted {key}: {estimate} < {exact}"
                );
            }
            smallest = smallest.min(item.count);
        }

        // The bounded min-heap keeps its smallest resident at the root; that is
        // what the next eviction compares against.
        assert_eq!(
            heap_entries[0].count, smallest,
            "the root is not the smallest resident"
        );

        // Every key in the true top-k should appear in the heap.
        let mut found_in_heap = 0usize;
        for &true_key in &true_top_k {
            let in_heap = heap_entries
                .iter()
                .any(|item| item.key == crate::HeapItem::U64(true_key));
            if in_heap {
                found_in_heap += 1;
            }
        }

        // With Zipf(1.3) and 200k samples, the top-k should be well-separated.
        // Require at least 80% of the true top-k to be present in the heap.
        let recall_pct = found_in_heap as f64 / top_k as f64 * 100.0;
        eprintln!(
            "[fold_cms_tumbling_heap_correctness] heap_size={}, \
             true_top_k_recall={found_in_heap}/{top_k} ({recall_pct:.0}%)",
            heap_entries.len()
        );
        assert!(
            recall_pct >= 80.0,
            "only {recall_pct:.0}% of true top-{top_k} found in heap (expected >= 80%)"
        );
    }

    // -- Gap 1: FoldCS query_recent accuracy ---------------------------------

    #[test]
    fn fold_cs_tumbling_query_recent_accuracy() {
        let rows = 5;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 5000;
        let exponent = 1.1;
        let total_windows = 6;
        let recent_n = 3;
        let samples_per_window = 10_000;

        let config = FoldCSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCS> = TumblingWindow::new(
            samples_per_window as u64,
            total_windows,
            config,
            total_windows + 2,
        );

        let stream = sample_zipf_u64(
            domain,
            exponent,
            samples_per_window * total_windows,
            0xC5_0A01,
        );

        let mut per_window_truth: Vec<HashMap<u64, i64>> = Vec::new();
        for w in 0..total_windows {
            let mut window_truth = HashMap::<u64, i64>::new();
            let start = w * samples_per_window;
            let end = start + samples_per_window;
            for (i, &value) in stream.iter().enumerate().take(end).skip(start) {
                tw.insert(i as u64, &DataInput::U64(value), 1);
                *window_truth.entry(value).or_insert(0) += 1;
            }
            per_window_truth.push(window_truth);
        }

        // query_recent(3) → 3 most recent closed + active.
        // After 6 windows: windows 0..4 closed, window 5 active.
        // recent(3) → closed[2,3,4] + active(5) = windows 2,3,4,5.
        let recent_start = total_windows - recent_n - 1;
        let mut recent_truth = HashMap::<u64, i64>::new();
        for w in per_window_truth
            .iter()
            .take(total_windows)
            .skip(recent_start)
        {
            for (&key, &count) in w {
                *recent_truth.entry(key).or_insert(0) += count;
            }
        }

        let merged = tw.query_recent(recent_n);

        // CS error bound: epsilon = sqrt(e / cols), bound = epsilon * L2
        let epsilon = (std::f64::consts::E / full_cols as f64).sqrt();
        let l2_norm: f64 = recent_truth
            .values()
            .map(|&c| (c as f64) * (c as f64))
            .sum::<f64>()
            .sqrt();
        let error_bound = epsilon * l2_norm;

        let mut within = 0usize;
        for (&key, &true_count) in &recent_truth {
            let est = merged.query(&DataInput::U64(key));
            if ((est - true_count).abs() as f64) <= error_bound {
                within += 1;
            }
        }
        let pct = within as f64 / recent_truth.len() as f64 * 100.0;
        eprintln!(
            "[fold_cs_tumbling_query_recent_accuracy] pct_within_bound={pct:.1}%, \
             error_bound={error_bound:.2}"
        );
        assert!(
            pct > 90.0,
            "only {pct:.1}% of recent keys within CS bound (expected > 90%)"
        );

        // Keys only in excluded windows should have small estimates.
        let mut excluded_truth = HashMap::<u64, i64>::new();
        for w in per_window_truth.iter().take(recent_start) {
            for (&key, &count) in w {
                *excluded_truth.entry(key).or_insert(0) += count;
            }
        }
        for &key in excluded_truth.keys() {
            if recent_truth.contains_key(&key) {
                continue;
            }
            let est = merged.query(&DataInput::U64(key));
            assert!(
                (est.abs() as f64) <= error_bound,
                "excluded key {key} estimate {est} exceeds error bound {error_bound:.2}"
            );
        }
    }

    // -- Gap 2: KLL query_recent accuracy ------------------------------------

    #[test]
    fn kll_tumbling_query_recent_accuracy() {
        let config = KLLConfig {
            k: 200,
            m: 8,
            seed: None,
        };
        let total_samples = 100_000;
        let num_windows = 8;
        let recent_n = 3;
        let samples_per_window = total_samples / num_windows;

        let mut tw: TumblingWindow<KLL> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let values = sample_uniform_f64(0.0, 10_000_000.0, total_samples, 0xA11_0072);

        let mut per_window: Vec<Vec<f64>> = Vec::new();
        for w in 0..num_windows {
            let start = w * samples_per_window;
            let end = start + samples_per_window;
            let window_values: Vec<f64> = values[start..end].to_vec();
            for (j, &v) in window_values.iter().enumerate() {
                tw.insert((start + j) as u64, &DataInput::F64(v), 1);
            }
            per_window.push(window_values);
        }

        // query_recent(3) → 3 most recent closed + active.
        // After 8 windows: windows 0..6 closed, window 7 active.
        // recent(3) → closed[4,5,6] + active(7) = windows 4,5,6,7.
        let recent_start = num_windows - recent_n - 1;
        let mut recent_values: Vec<f64> = Vec::new();
        for w in per_window.iter().take(num_windows).skip(recent_start) {
            recent_values.extend_from_slice(w);
        }
        recent_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = recent_values.len();

        let merged = tw.query_recent(recent_n);
        let cdf = merged.cdf();

        let quantiles = [0.10, 0.25, 0.50, 0.75, 0.90];
        let tolerance = 0.03;

        for &q in &quantiles {
            let est = cdf.query(q);
            let lo_idx = (((q - tolerance).max(0.0) * n as f64).ceil() as usize).min(n - 1);
            let hi_idx = (((q + tolerance).min(1.0) * n as f64).ceil() as usize).min(n - 1);
            assert!(
                est >= recent_values[lo_idx] && est <= recent_values[hi_idx],
                "quantile {q} estimate {est} outside [{}, {}] (rank tolerance {tolerance})",
                recent_values[lo_idx],
                recent_values[hi_idx]
            );
        }
    }

    // -- Gap 3: FoldCS heap correctness --------------------------------------

    #[test]
    fn fold_cs_tumbling_heap_correctness() {
        let rows = 5;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 10_000;
        let exponent = 1.3;
        let total_samples = 200_000;
        let num_windows = 8;
        let samples_per_window = total_samples / num_windows;

        let config = FoldCSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let stream = sample_zipf_u64(domain, exponent, total_samples, 0xC5_4EA9);
        let mut truth = HashMap::<u64, i64>::new();

        for (i, &value) in stream.iter().enumerate() {
            tw.insert(i as u64, &DataInput::U64(value), 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged = tw.query_all();

        let mut truth_sorted: Vec<(u64, i64)> = truth.iter().map(|(k, v)| (*k, *v)).collect();
        truth_sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
        let true_top_k: Vec<u64> = truth_sorted.iter().take(top_k).map(|&(k, _)| k).collect();

        let heap_entries = merged.heap().heap();
        assert_eq!(
            heap_entries.len(),
            top_k,
            "merging {num_windows} windows fills the heap"
        );

        // The counts the heap reports, not just the keys it holds. Count
        // Sketch's error is two-sided, so a resident's count is checked
        // against the truth as a relative band rather than as a floor.
        let mut smallest = i64::MAX;
        for item in heap_entries {
            let crate::HeapItem::U64(key) = item.key else {
                panic!("a U64 stream seats U64 keys, got {:?}", item.key);
            };
            assert!(item.count > 0, "resident {key} carries a count of zero");
            if let Some(&exact) = truth.get(&key) {
                let estimate = merged.query(&DataInput::U64(key));
                let error = (estimate - exact).abs() as f64 / exact as f64;
                assert!(
                    error < 0.02,
                    "resident {key}: estimate {estimate} is {:.0}% off the exact {exact}",
                    error * 100.0
                );
            }
            smallest = smallest.min(item.count);
        }

        // The bounded min-heap keeps its smallest resident at the root; that is
        // what the next eviction compares against.
        assert_eq!(
            heap_entries[0].count, smallest,
            "the root is not the smallest resident"
        );

        let mut found_in_heap = 0usize;
        for &true_key in &true_top_k {
            let in_heap = heap_entries
                .iter()
                .any(|item| item.key == crate::HeapItem::U64(true_key));
            if in_heap {
                found_in_heap += 1;
            }
        }

        let recall_pct = found_in_heap as f64 / top_k as f64 * 100.0;
        eprintln!(
            "[fold_cs_tumbling_heap_correctness] heap_size={}, \
             true_top_k_recall={found_in_heap}/{top_k} ({recall_pct:.0}%)",
            heap_entries.len()
        );
        assert!(
            recall_pct >= 80.0,
            "only {recall_pct:.0}% of true top-{top_k} found in heap (expected >= 80%)"
        );
    }

    // -- Gap 4: Tumbling vs monolithic degradation ---------------------------

    #[test]
    fn fold_cms_tumbling_vs_monolithic() {
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 10_000;
        let exponent = 1.1;
        let total_samples = 200_000;
        let num_windows = 8;
        let samples_per_window = total_samples / num_windows;

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config.clone(),
            num_windows + 2,
        );
        let mut mono: FoldCMS = FoldCMS::new(rows, full_cols, fold_level, top_k);

        let stream = sample_zipf_u64(domain, exponent, total_samples, 0xDE_AD01);
        let mut truth = HashMap::<u64, i64>::new();

        for (i, &value) in stream.iter().enumerate() {
            let key = DataInput::U64(value);
            tw.insert(i as u64, &key, 1);
            mono.insert(&key, 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged = tw.query_all();

        let mut tumbling_total_err = 0i64;
        let mut mono_total_err = 0i64;
        for (&key, &true_count) in &truth {
            let sk = DataInput::U64(key);
            tumbling_total_err += (merged.query(&sk) - true_count).abs();
            mono_total_err += (mono.query(&sk) - true_count).abs();
        }

        let tumbling_mae = tumbling_total_err as f64 / truth.len() as f64;
        let mono_mae = mono_total_err as f64 / truth.len() as f64;
        eprintln!(
            "[fold_cms_tumbling_vs_monolithic] tumbling_mae={tumbling_mae:.2}, \
             mono_mae={mono_mae:.2}, ratio={:.2}",
            tumbling_mae / mono_mae.max(1.0)
        );
        assert!(
            tumbling_mae <= mono_mae * 1.5,
            "tumbling MAE ({tumbling_mae:.2}) > 1.5x monolithic ({mono_mae:.2})"
        );
    }

    #[test]
    fn fold_cs_tumbling_vs_monolithic() {
        let rows = 5;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let domain = 10_000;
        let exponent = 1.1;
        let total_samples = 200_000;
        let num_windows = 8;
        let samples_per_window = total_samples / num_windows;

        let config = FoldCSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config.clone(),
            num_windows + 2,
        );
        let mut mono: FoldCS = FoldCS::new(rows, full_cols, fold_level, top_k);

        let stream = sample_zipf_u64(domain, exponent, total_samples, 0xDE_AD02);
        let mut truth = HashMap::<u64, i64>::new();

        for (i, &value) in stream.iter().enumerate() {
            let key = DataInput::U64(value);
            tw.insert(i as u64, &key, 1);
            mono.insert(&key, 1);
            *truth.entry(value).or_insert(0) += 1;
        }

        let merged = tw.query_all();

        let mut tumbling_total_err = 0i64;
        let mut mono_total_err = 0i64;
        for (&key, &true_count) in &truth {
            let sk = DataInput::U64(key);
            tumbling_total_err += (merged.query(&sk) - true_count).abs();
            mono_total_err += (mono.query(&sk) - true_count).abs();
        }

        let tumbling_mae = tumbling_total_err as f64 / truth.len() as f64;
        let mono_mae = mono_total_err as f64 / truth.len() as f64;
        eprintln!(
            "[fold_cs_tumbling_vs_monolithic] tumbling_mae={tumbling_mae:.2}, \
             mono_mae={mono_mae:.2}, ratio={:.2}",
            tumbling_mae / mono_mae.max(1.0)
        );
        assert!(
            tumbling_mae <= mono_mae * 1.5,
            "tumbling MAE ({tumbling_mae:.2}) > 1.5x monolithic ({mono_mae:.2})"
        );
    }

    #[test]
    fn kll_tumbling_vs_monolithic() {
        let config = KLLConfig {
            k: 200,
            m: 8,
            seed: None,
        };
        let total_samples = 100_000;
        let num_windows = 8;
        let samples_per_window = total_samples / num_windows;

        let mut tw: TumblingWindow<KLL> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config.clone(),
            num_windows + 2,
        );
        let mut mono = KLL::init(config.k, config.m);

        let values = sample_uniform_f64(0.0, 10_000_000.0, total_samples, 0xDE_AD03);

        for (i, &v) in values.iter().enumerate() {
            tw.insert(i as u64, &DataInput::F64(v), 1);
            mono.update_data_input(&DataInput::F64(v)).unwrap();
        }

        let merged = tw.query_all();
        let tw_cdf = merged.cdf();
        let mono_cdf = mono.cdf();

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();

        let quantiles = [0.10, 0.25, 0.50, 0.75, 0.90];
        let mut tw_max_rank_err = 0.0f64;
        let mut mono_max_rank_err = 0.0f64;

        for &q in &quantiles {
            let tw_est = tw_cdf.query(q);
            let mono_est = mono_cdf.query(q);

            // Compute rank error: how far off is the rank of the estimated value?
            let tw_rank = sorted.partition_point(|&x| x <= tw_est) as f64 / n as f64;
            let mono_rank = sorted.partition_point(|&x| x <= mono_est) as f64 / n as f64;

            tw_max_rank_err = tw_max_rank_err.max((tw_rank - q).abs());
            mono_max_rank_err = mono_max_rank_err.max((mono_rank - q).abs());
        }

        eprintln!(
            "[kll_tumbling_vs_monolithic] tw_max_rank_err={tw_max_rank_err:.4}, \
             mono_max_rank_err={mono_max_rank_err:.4}"
        );
        assert!(
            tw_max_rank_err <= mono_max_rank_err + 0.02,
            "tumbling rank error ({tw_max_rank_err:.4}) > monolithic ({mono_max_rank_err:.4}) + 0.02"
        );
    }

    // -- Gap 5: Edge cases ---------------------------------------------------

    #[test]
    fn tumbling_single_window_accuracy() {
        // All data fits in one window (no window closes).
        let total_samples = 50_000;

        // --- FoldCMS subsection ---
        {
            let config = FoldCMSConfig {
                rows: 3,
                full_cols: 4096,
                fold_level: 4,
                top_k: 20,
            };
            let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
                total_samples as u64 + 1, // window_size > total_samples → never closes
                10,
                config,
                4,
            );

            let stream = sample_zipf_u64(5000, 1.1, total_samples, 0x51_0001);
            let mut truth = HashMap::<u64, i64>::new();
            for (i, &value) in stream.iter().enumerate() {
                tw.insert(i as u64, &DataInput::U64(value), 1);
                *truth.entry(value).or_insert(0) += 1;
            }

            assert_eq!(tw.closed_count(), 0, "no windows should have closed");

            let merged = tw.query_all();
            let epsilon = std::f64::consts::E / 4096.0;
            let l1: f64 = truth.values().map(|&c| c as f64).sum();
            let bound = epsilon * l1;

            let mut within = 0usize;
            for (&key, &true_count) in &truth {
                let est = merged.query(&DataInput::U64(key));
                if ((est - true_count).abs() as f64) <= bound {
                    within += 1;
                }
            }
            let pct = within as f64 / truth.len() as f64 * 100.0;
            assert!(
                pct > 90.0,
                "FoldCMS single-window: only {pct:.1}% within bound"
            );
        }

        // --- FoldCS subsection ---
        {
            let config = FoldCSConfig {
                rows: 5,
                full_cols: 4096,
                fold_level: 4,
                top_k: 20,
            };
            let mut tw: TumblingWindow<FoldCS> =
                TumblingWindow::new(total_samples as u64 + 1, 10, config, 4);

            let stream = sample_zipf_u64(5000, 1.1, total_samples, 0x51_0002);
            let mut truth = HashMap::<u64, i64>::new();
            for (i, &value) in stream.iter().enumerate() {
                tw.insert(i as u64, &DataInput::U64(value), 1);
                *truth.entry(value).or_insert(0) += 1;
            }

            assert_eq!(tw.closed_count(), 0, "no windows should have closed");

            let merged = tw.query_all();
            let epsilon = (std::f64::consts::E / 4096.0f64).sqrt();
            let l2: f64 = truth
                .values()
                .map(|&c| (c as f64) * (c as f64))
                .sum::<f64>()
                .sqrt();
            let bound = epsilon * l2;

            let mut within = 0usize;
            for (&key, &true_count) in &truth {
                let est = merged.query(&DataInput::U64(key));
                if ((est - true_count).abs() as f64) <= bound {
                    within += 1;
                }
            }
            let pct = within as f64 / truth.len() as f64 * 100.0;
            assert!(
                pct > 90.0,
                "FoldCS single-window: only {pct:.1}% within bound"
            );
        }

        // --- KLL subsection ---
        {
            let config = KLLConfig {
                k: 200,
                m: 8,
                seed: None,
            };
            let mut tw: TumblingWindow<KLL> =
                TumblingWindow::new(total_samples as u64 + 1, 10, config, 4);

            let values = sample_uniform_f64(0.0, 1_000_000.0, total_samples, 0x51_0003);
            for (i, &v) in values.iter().enumerate() {
                tw.insert(i as u64, &DataInput::F64(v), 1);
            }

            assert_eq!(tw.closed_count(), 0, "no windows should have closed");

            let merged = tw.query_all();
            let cdf = merged.cdf();

            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();

            for &q in &[0.25, 0.50, 0.75] {
                let est = cdf.query(q);
                let lo = (((q - 0.02).max(0.0) * n as f64).ceil() as usize).min(n - 1);
                let hi = (((q + 0.02).min(1.0) * n as f64).ceil() as usize).min(n - 1);
                assert!(
                    est >= sorted[lo] && est <= sorted[hi],
                    "KLL single-window: quantile {q} est {est} outside [{}, {}]",
                    sorted[lo],
                    sorted[hi]
                );
            }
        }
    }

    #[test]
    fn tumbling_very_small_windows() {
        let window_size = 10u64;
        let total_samples = 5000;
        let max_windows = 50;

        // FoldCMS with tiny windows — many rotations.
        let config = FoldCMSConfig {
            rows: 3,
            full_cols: 1024,
            fold_level: 3,
            top_k: 10,
        };
        let mut tw: TumblingWindow<FoldCMS> =
            TumblingWindow::new(window_size, max_windows, config, max_windows + 2);

        let stream = sample_zipf_u64(500, 1.2, total_samples, 0x77_1100);

        for (i, &value) in stream.iter().enumerate() {
            tw.insert(i as u64, &DataInput::U64(value), 1);
        }

        // With 5000 samples and window_size=10, there are 500 windows total.
        // max_windows=50, so only the last ~50 closed + active are retained.
        // Just verify the system didn't crash and produces reasonable estimates.
        let merged = tw.query_all();

        // Build truth for what the tumbling window actually retains.
        // The retained windows cover the most recent portion of the stream.
        let total_windows_created = total_samples / window_size as usize;
        let retained_start_window = total_windows_created.saturating_sub(max_windows);
        let retained_start_sample = retained_start_window * window_size as usize;

        let mut retained_truth = HashMap::<u64, i64>::new();
        for &k in stream
            .iter()
            .take(total_samples)
            .skip(retained_start_sample)
        {
            *retained_truth.entry(k).or_insert(0) += 1;
        }

        let epsilon = std::f64::consts::E / 1024.0;
        let l1: f64 = retained_truth.values().map(|&c| c as f64).sum();
        let bound = epsilon * l1;

        let mut within = 0usize;
        for (&key, &true_count) in &retained_truth {
            let est = merged.query(&DataInput::U64(key));
            if ((est - true_count).abs() as f64) <= bound {
                within += 1;
            }
        }
        let pct = within as f64 / retained_truth.len() as f64 * 100.0;
        eprintln!(
            "[tumbling_very_small_windows] retained_windows={max_windows}, \
             pct_within_bound={pct:.1}%, bound={bound:.2}"
        );
        assert!(
            pct > 85.0,
            "only {pct:.1}% within CMS bound with tiny windows (expected > 85%)"
        );
    }

    #[test]
    fn tumbling_skewed_load() {
        // 5 windows where window 2 gets 90% of data.
        let rows = 3;
        let full_cols = 4096;
        let fold_level = 4;
        let top_k = 20;
        let num_windows = 5;
        let light_samples = 1_000;
        let heavy_samples = 36_000; // window 2 gets ~90% of 40k total
        let samples_per_window = 10_000; // large enough that light samples fit in one window

        let config = FoldCMSConfig {
            rows,
            full_cols,
            fold_level,
            top_k,
        };
        let mut tw: TumblingWindow<FoldCMS> = TumblingWindow::new(
            samples_per_window as u64,
            num_windows,
            config,
            num_windows + 2,
        );

        let domain = 5000;
        let exponent = 1.1;
        let mut truth = HashMap::<u64, i64>::new();
        let mut time = 0u64;

        for w in 0..num_windows {
            let n_samples = if w == 2 { heavy_samples } else { light_samples };
            let stream = sample_zipf_u64(domain, exponent, n_samples, 0xBE_EF00 + w as u64);
            for &value in &stream {
                tw.insert(time, &DataInput::U64(value), 1);
                *truth.entry(value).or_insert(0) += 1;
                time += 1;
            }
            // Advance time to next window boundary.
            let next_boundary =
                ((time / samples_per_window as u64) + 1) * samples_per_window as u64;
            time = next_boundary;
        }

        let merged = tw.query_all();

        let epsilon = std::f64::consts::E / full_cols as f64;
        let l1: f64 = truth.values().map(|&c| c as f64).sum();
        let bound = epsilon * l1;

        let mut within = 0usize;
        for (&key, &true_count) in &truth {
            let est = merged.query(&DataInput::U64(key));
            if ((est - true_count).abs() as f64) <= bound {
                within += 1;
            }
        }
        let pct = within as f64 / truth.len() as f64 * 100.0;
        eprintln!("[tumbling_skewed_load] pct_within_bound={pct:.1}%, bound={bound:.2}");
        assert!(
            pct > 90.0,
            "only {pct:.1}% within CMS bound with skewed load (expected > 90%)"
        );
    }
}
