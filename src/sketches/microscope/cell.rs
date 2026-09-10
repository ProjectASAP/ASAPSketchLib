//! One MicroscopeSketch cell: the record layout and the operations on it.
//!
//! A cell is a byte record, so this module works on `&[u8]` / `&mut [u8]` and
//! knows nothing about the matrix it lives in. That keeps the algorithm
//! testable on its own, which matters because everything above it — the
//! `rows x cols` grid, the sub-window clock — is bookkeeping around these few
//! operations.
//!
//! # The camera
//!
//! A cell counts the items of a sliding window of `W` items, split into `T`
//! equal sub-windows. It holds:
//!
//! - **pixels**: `T + 2` counters of `l` bits each, a ring indexed by
//!   sub-window number. Pixel `i` holds the count of sub-window `i`, scaled.
//! - **zoom** `Z`: one shared exponent. A pixel's unit is `c^Z` items, so the
//!   whole cell zooms together and one exponent serves every pixel.
//! - **shutter** `S`: items counted since the current pixel last ticked, i.e.
//!   the remainder below one `c^Z` unit.
//!
//! `Z` is what makes the fields *coupled*: a pixel value is meaningless
//! without it, and a zoom rescales every pixel at once. That coupling is the
//! reason the cell is one contiguous record rather than `T + 2` independent
//! counters spread across separate matrices.
//!
//! # Why `T + 2` pixels for a `T` sub-window window
//!
//! At sub-window `n` the window covers all of sub-windows `n-T+1 ..= n` plus
//! a shrinking fraction of `n-T`, which is `T + 1` live pixels. The extra
//! slot is the one being cleared ahead of its reuse at `n+1`.
//!
//! # This implementation fixes `l = 8`
//!
//! One pixel is one byte, so a pixel saturates at 255 and a zoom-out is
//! triggered by the increment that would reach 256. Packing two 4-bit pixels
//! per byte halves the record and is a later change; the layout type already
//! hides every offset from callers, so it is contained here.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::Cell;

/// The largest value an 8-bit pixel can hold.
const PIXEL_MAX: u8 = u8::MAX;

/// Zoom exponents are capped so `c^Z` always fits a `u32`.
///
/// A cell at the cap has pixels worth `c^Z` items each; for `c = 2` that is
/// `2^31` items per pixel unit, which no sliding window this sketch is meant
/// for will reach. At the cap a pixel saturates instead of zooming further,
/// so the estimate stops growing rather than wrapping.
const MAX_SHUTTER: u32 = u32::MAX;

/// Which end of the partially expired oldest sub-window to charge to the
/// estimate.
///
/// At sub-window `n` the window has left some fraction of sub-window `n-T`
/// behind. The cell cannot know how much of *its* count sat in that fraction,
/// so the choice is a policy:
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeltaStrategy {
    /// Count all of the oldest sub-window. Never underestimates.
    Over,
    /// Count none of it. Never overestimates.
    Under,
    /// Count the fraction `p` of it that is still inside the window,
    /// assuming the sub-window's items are spread evenly over its span.
    Linear(f64),
}

/// Configuration of a MicroscopeSketch cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MicroParams {
    /// Number of sub-windows the sliding window is split into.
    pub t: usize,
    /// Zoom base. Each zoom-out divides every pixel by `c` and raises `Z`.
    pub c: u32,
}

impl MicroParams {
    /// Builds parameters, rejecting the values the layout cannot represent.
    ///
    /// Panics if `t` is zero or if `c` is below 2.
    pub fn new(t: usize, c: u32) -> Self {
        assert!(t > 0, "t (sub-windows per window) must be at least 1");
        assert!(c >= 2, "zoom base c must be at least 2, got {c}");
        Self { t, c }
    }
}

impl Default for MicroParams {
    /// `T = 12`, `c = 2`.
    fn default() -> Self {
        Self { t: 12, c: 2 }
    }
}

/// Byte offsets of a cell's fields, derived once from [`MicroParams`].
///
/// Layout: `[ pixels: T+2 bytes ][ Z: 1 byte ][ S: 4 bytes LE ][ pad ]`,
/// with the record length rounded up to a multiple of 4 so the shutter of
/// every cell sits at the same alignment within the backing storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MicroLayout {
    params: MicroParams,
    pixels: usize,
    zoom_at: usize,
    shutter_at: usize,
    depth: usize,
    /// The largest pixel value that survives a zoom-in, i.e. `256 / c`.
    zoom_in_ceiling: u32,
}

impl MicroLayout {
    /// Derives the layout for `params`.
    pub fn new(params: MicroParams) -> Self {
        let pixels = params.t + 2;
        let zoom_at = pixels;
        let shutter_at = pixels + 1;
        let unpadded = shutter_at + 4;
        let depth = unpadded.div_ceil(4) * 4;
        Self {
            params,
            pixels,
            zoom_at,
            shutter_at,
            depth,
            zoom_in_ceiling: (PIXEL_MAX as u32 + 1) / params.c,
        }
    }

    /// The parameters this layout came from.
    #[inline(always)]
    pub fn params(&self) -> MicroParams {
        self.params
    }

    /// Bytes per cell.
    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of pixels, i.e. `T + 2`.
    #[inline(always)]
    pub fn pixels(&self) -> usize {
        self.pixels
    }

    /// Byte offset of the pixel holding sub-window `n`.
    #[inline(always)]
    fn pixel_at(&self, n: u64) -> usize {
        (n % self.pixels as u64) as usize
    }

    /// The largest zoom exponent for which `c^Z` still fits a `u32`.
    fn max_zoom(&self) -> u8 {
        let mut z: u8 = 0;
        let mut value: u64 = 1;
        while value.saturating_mul(self.params.c as u64) <= MAX_SHUTTER as u64 {
            value *= self.params.c as u64;
            z += 1;
        }
        z
    }
}

/// A cell's zoom exponent.
#[inline(always)]
pub fn zoom(cell: &[u8], layout: &MicroLayout) -> u8 {
    cell[layout.zoom_at]
}

/// A cell's shutter.
///
/// Read and written through `to_le_bytes`/`from_le_bytes` rather than a cast,
/// so the record needs no alignment guarantee beyond being a byte slice.
#[inline(always)]
pub fn shutter(cell: &[u8], layout: &MicroLayout) -> u32 {
    let at = layout.shutter_at;
    u32::from_le_bytes([cell[at], cell[at + 1], cell[at + 2], cell[at + 3]])
}

#[inline(always)]
fn set_shutter(cell: &mut [u8], layout: &MicroLayout, value: u32) {
    let at = layout.shutter_at;
    cell[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// The pixel value for sub-window `n`.
#[inline(always)]
pub fn pixel(cell: &[u8], layout: &MicroLayout, n: u64) -> u8 {
    cell[layout.pixel_at(n)]
}

/// A tiny xorshift64* used for the unbiased rounding of a zoom-out.
///
/// It advances through a `Cell`, so a `&Rounding` can be captured by a
/// non-`mut` closure — which is what lets a cell insert run inside
/// `Vector3D::fast_insert`, whose closure is `Fn`. The state is a plain
/// `u64`, so a sketch holding one can serialize it or reseed on load.
#[derive(Clone, Debug)]
pub struct Rounding {
    state: Cell<u64>,
}

impl Serialize for Rounding {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.state.get())
    }
}

impl<'de> Deserialize<'de> for Rounding {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(u64::deserialize(deserializer)?))
    }
}

impl Rounding {
    /// Seeds the generator. A zero seed is replaced, since xorshift is stuck
    /// at zero.
    pub fn new(seed: u64) -> Self {
        Self {
            state: Cell::new(if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            }),
        }
    }

    /// The current state, for serialization.
    #[inline(always)]
    pub fn state(&self) -> u64 {
        self.state.get()
    }

    #[inline(always)]
    fn next_u64(&self) -> u64 {
        let mut x = self.state.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state.set(x);
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Divides `value` by `c`, rounding the remainder up with probability `r / c`.
///
/// Truncating every zoom-out would bias a cell's estimate down by up to
/// `(c-1)/c` of a pixel unit per zoom, compounding across zooms and across
/// pixels. Rounding probabilistically makes each division unbiased, so the
/// estimate stays unbiased however many times the cell has zoomed.
#[inline]
pub fn unbiased_div(value: u32, c: u32, rounding: &Rounding) -> u32 {
    let quotient = value / c;
    let remainder = value % c;
    if remainder == 0 {
        return quotient;
    }
    if (rounding.next_u64() % c as u64) < remainder as u64 {
        quotient + 1
    } else {
        quotient
    }
}

/// Halves the cell: `Z += 1` and every pixel is divided by `c`.
///
/// At the zoom cap this is a no-op, and the caller saturates the pixel
/// instead.
fn zoom_out(cell: &mut [u8], layout: &MicroLayout, rounding: &Rounding) -> bool {
    let z = cell[layout.zoom_at];
    if z >= layout.max_zoom() {
        return false;
    }
    cell[layout.zoom_at] = z + 1;
    let c = layout.params.c;
    for slot in cell.iter_mut().take(layout.pixels) {
        *slot = unbiased_div(*slot as u32, c, rounding) as u8;
    }
    true
}

/// Records one item in sub-window `n`.
///
/// Two things happen:
///
/// 1. The shutter advances. It only rolls into the current pixel once it
///    reaches `c^Z`, which is what makes a pixel worth `c^Z` items.
/// 2. If that roll would take the pixel past what 8 bits hold, the cell
///    zooms out first and the increment lands on the halved value.
///
/// Step 2 differs from a literal reading of "increment, then zoom if the
/// pixel reached `2^l`": an 8-bit pixel cannot hold 256 even transiently, so
/// the overflow is caught on the increment that would cause it.
///
/// Expiry is *not* done here — see [`enter_sub_window`].
pub fn insert(cell: &mut [u8], layout: &MicroLayout, n: u64, rounding: &Rounding) {
    // Advance the shutter.
    let z = cell[layout.zoom_at];
    let unit = (layout.params.c as u64).pow(z as u32);
    let next_shutter = shutter(cell, layout) as u64 + 1;
    if next_shutter < unit {
        set_shutter(cell, layout, next_shutter as u32);
        return;
    }

    // The shutter closed: roll it into the current pixel.
    set_shutter(cell, layout, 0);
    let current = layout.pixel_at(n);
    if cell[current] == PIXEL_MAX && !zoom_out(cell, layout, rounding) {
        // At the zoom cap; saturate rather than wrap.
        return;
    }
    cell[current] += 1;
}

/// Moves the cell into sub-window `new_n`, having crossed `crossed`
/// boundaries to get there.
///
/// This is where a cell forgets. The pixel ring is indexed by sub-window
/// number, so the slot a new sub-window lands on still holds the count of
/// the sub-window `T + 2` before it; unless it is cleared on the way in,
/// that stale count is read back as if it belonged to the new sub-window.
///
/// Clearing has to happen for **every** cell at a boundary, not for the cell
/// an insert happens to touch. A cell whose key stops appearing receives no
/// inserts, and is exactly the cell whose stale pixels would otherwise be
/// re-read as recent ones — so a key that went quiet would keep reporting
/// its old count forever. Since a boundary already walks the whole table to
/// offer each cell a zoom-in, the clearing rides along at no extra pass.
///
/// A jump of `T + 2` or more sub-windows has left nothing behind, so the
/// cell is reset outright rather than cleared slot by slot.
pub fn enter_sub_window(cell: &mut [u8], layout: &MicroLayout, new_n: u64, crossed: u64) {
    if crossed == 0 {
        return;
    }
    if crossed >= layout.pixels as u64 {
        for slot in cell.iter_mut().take(layout.pixels) {
            *slot = 0;
        }
        // The shutter held a partial unit of a sub-window that is now gone.
        set_shutter(cell, layout, 0);
    } else {
        // Clear each slot entered on the way, oldest of them first.
        for step in (0..crossed).rev() {
            let entered = new_n - step;
            cell[layout.pixel_at(entered)] = 0;
        }
    }
    // One chance to reclaim resolution per boundary crossed.
    for _ in 0..crossed.min(layout.pixels as u64) {
        if !try_zoom_in(cell, layout) {
            break;
        }
    }
}

/// Restores resolution when the cell has room for it: if every pixel would
/// still fit after multiplying by `c`, multiply them and drop `Z` by one.
///
/// Meant to be called once per sub-window boundary, not per item. Returns
/// whether it zoomed.
pub fn try_zoom_in(cell: &mut [u8], layout: &MicroLayout) -> bool {
    let z = cell[layout.zoom_at];
    if z == 0 {
        return false;
    }
    let ceiling = layout.zoom_in_ceiling;
    if cell
        .iter()
        .take(layout.pixels)
        .any(|&p| p as u32 >= ceiling)
    {
        return false;
    }
    let c = layout.params.c;
    for slot in cell.iter_mut().take(layout.pixels) {
        *slot = (*slot as u32 * c) as u8;
    }
    cell[layout.zoom_at] = z - 1;
    true
}

/// Estimates the cell's count over the sliding window ending inside
/// sub-window `n`.
///
/// The sum is the shutter, plus the `T` sub-windows `n-T+1 ..= n` that lie
/// wholly inside the window, plus whatever `strategy` charges for the
/// partially expired sub-window `n-T`.
pub fn estimate(cell: &[u8], layout: &MicroLayout, n: u64, strategy: DeltaStrategy) -> f64 {
    let z = cell[layout.zoom_at];
    let unit = (layout.params.c as f64).powi(z as i32);
    let t = layout.params.t as u64;

    let mut total = shutter(cell, layout) as f64;
    for back in 0..t {
        // Sub-windows before the stream began contributed nothing, and their
        // pixels are still zero, so a missing index is simply skipped.
        if let Some(sub_window) = n.checked_sub(back) {
            total += pixel(cell, layout, sub_window) as f64 * unit;
        }
    }

    let oldest = n
        .checked_sub(t)
        .map(|sub_window| pixel(cell, layout, sub_window) as f64 * unit)
        .unwrap_or(0.0);
    total += match strategy {
        DeltaStrategy::Over => oldest,
        DeltaStrategy::Under => 0.0,
        DeltaStrategy::Linear(fraction) => fraction.clamp(0.0, 1.0) * oldest,
    };
    total
}

/// Divides `value` by `c` `steps` times, unbiased at each step.
#[inline]
fn scale_down(mut value: u32, steps: u8, c: u32, rounding: &Rounding) -> u32 {
    for _ in 0..steps {
        value = unbiased_div(value, c, rounding);
    }
    value
}

/// Merges `src` into `dst`, both positioned at sub-window `n`.
///
/// The paper does not define a merge, so this one is chosen to preserve the
/// only thing a merge can be asked to preserve here: the estimate of the
/// union. Three steps.
///
/// 1. **Agree on a zoom.** Pixels are only comparable at the same `Z`, so
///    both sides are brought to `max(Z_dst, Z_src)` — never to the smaller
///    one, which would have to invent resolution neither side has. The
///    rescaling is the same unbiased division a zoom-out uses.
/// 2. **Add pixel-wise**, in `u32` so a sum cannot wrap, then zoom the result
///    out until every pixel fits a byte again.
/// 3. **Add the shutters.** They are raw item counts, not scaled by `Z`, so
///    they add directly; whole `c^Z` units of the sum carry into the current
///    sub-window's pixel and the remainder stays in the shutter.
///
/// Both cells must sit at the same sub-window: the pixel ring is indexed by
/// sub-window number, so merging cells at different `n` would add unrelated
/// sub-windows together. The grid enforces that; this function assumes it.
pub fn merge_cells(dst: &mut [u8], src: &[u8], layout: &MicroLayout, n: u64, rounding: &Rounding) {
    let c = layout.params.c;
    let z_dst = zoom(dst, layout);
    let z_src = zoom(src, layout);
    let mut target = z_dst.max(z_src);

    let mut sums: Vec<u32> = (0..layout.pixels)
        .map(|i| {
            scale_down(dst[i] as u32, target - z_dst, c, rounding)
                + scale_down(src[i] as u32, target - z_src, c, rounding)
        })
        .collect();

    // Shutters are item counts below one unit each, so their sum is below two
    // units and carries at most once — but compute it generally.
    let combined = shutter(dst, layout) as u64 + shutter(src, layout) as u64;
    let unit = (c as u64).pow(target as u32);
    sums[layout.pixel_at(n)] += (combined / unit) as u32;
    let mut remainder = combined % unit;

    let cap = layout.max_zoom();
    while sums.iter().any(|&v| v > PIXEL_MAX as u32) {
        if target >= cap {
            for value in sums.iter_mut() {
                *value = (*value).min(PIXEL_MAX as u32);
            }
            break;
        }
        target += 1;
        for value in sums.iter_mut() {
            *value = unbiased_div(*value, c, rounding);
        }
        // The remainder was below the old unit, so it is below the new one
        // too and needs no adjustment.
    }

    for (slot, value) in dst.iter_mut().take(layout.pixels).zip(sums) {
        *slot = value as u8;
    }
    dst[layout.zoom_at] = target;
    remainder = remainder.min(u32::MAX as u64);
    set_shutter(dst, layout, remainder as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a cell through `items` inserts spread over sub-windows of
    /// `per_sub_window` items each, calling `try_zoom_in` at every boundary
    /// the way a sketch would, and returns the cell plus the final `n`.
    fn run(layout: &MicroLayout, items: u64, per_sub_window: u64) -> (Vec<u8>, u64) {
        let mut cell = vec![0u8; layout.depth()];
        let rounding = Rounding::new(0x5EED);
        let mut n = 0u64;
        for i in 0..items {
            if i > 0 && i % per_sub_window == 0 {
                n += 1;
                enter_sub_window(&mut cell, layout, n, 1);
            }
            insert(&mut cell, layout, n, &rounding);
        }
        (cell, n)
    }

    /// How many of the last `items` inserts are still inside a window of
    /// `t` sub-windows of `per_sub_window` items, counting the partially
    /// expired oldest sub-window in full.
    fn truth_over(items: u64, t: u64, per_sub_window: u64, n: u64) -> u64 {
        let oldest_kept = (n + 1).saturating_sub(t + 1);
        items - oldest_kept * per_sub_window
    }

    #[test]
    fn layout_places_every_field_and_pads_to_four() {
        // T=12: 14 pixels + 1 zoom + 4 shutter = 19, padded to 20.
        let l = MicroLayout::new(MicroParams::new(12, 2));
        assert_eq!(l.pixels(), 14);
        assert_eq!(l.depth(), 20);
        // T=1: 3 pixels + 1 + 4 = 8, already a multiple of 4.
        let l = MicroLayout::new(MicroParams::new(1, 2));
        assert_eq!(l.pixels(), 3);
        assert_eq!(l.depth(), 8);
    }

    #[test]
    fn fields_round_trip_independently() {
        let layout = MicroLayout::new(MicroParams::new(4, 2));
        let mut cell = vec![0u8; layout.depth()];
        set_shutter(&mut cell, &layout, 0xDEAD_BEEF);
        cell[layout.zoom_at] = 7;
        cell[layout.pixel_at(3)] = 42;
        assert_eq!(shutter(&cell, &layout), 0xDEAD_BEEF);
        assert_eq!(zoom(&cell, &layout), 7);
        assert_eq!(pixel(&cell, &layout, 3), 42);
        // Writing the shutter did not spill into a pixel or the zoom byte.
        assert_eq!(pixel(&cell, &layout, 0), 0);
    }

    /// While no pixel has overflowed, the cell has not zoomed and every item
    /// is counted individually, so the window count is exact — no estimator
    /// error at all.
    #[test]
    fn is_exact_before_the_first_zoom() {
        for t in [1usize, 4, 12] {
            let layout = MicroLayout::new(MicroParams::new(t, 2));
            // 200 items per sub-window keeps every pixel under 255.
            let (cell, n) = run(&layout, 200 * (t as u64 + 1), 200);
            assert_eq!(zoom(&cell, &layout), 0, "t={t}: should not have zoomed");
            let est = estimate(&cell, &layout, n, DeltaStrategy::Over);
            let truth = truth_over(200 * (t as u64 + 1), t as u64, 200, n);
            assert_eq!(est, truth as f64, "t={t}: exact regime must be exact");
        }
    }

    /// The two one-sided strategies bracket the truth, whatever the cell has
    /// been through. Driven past several zoom-outs so the bracketing is
    /// tested with rounding error in play, over a range of shapes.
    #[test]
    fn over_and_under_bracket_the_window_count() {
        for t in [1usize, 4, 12] {
            for per_sub_window in [1u64, 7, 500, 9_000] {
                let layout = MicroLayout::new(MicroParams::new(t, 2));
                let items = per_sub_window * (t as u64 + 3);
                let (cell, n) = run(&layout, items, per_sub_window);
                let over = estimate(&cell, &layout, n, DeltaStrategy::Over);
                let under = estimate(&cell, &layout, n, DeltaStrategy::Under);
                let linear = estimate(&cell, &layout, n, DeltaStrategy::Linear(0.5));
                assert!(
                    under <= linear && linear <= over,
                    "t={t} w={per_sub_window}: Under {under} <= Linear {linear} <= Over {over}"
                );
                // Over counts a superset of the window's sub-windows and
                // Under a subset, so the true window count sits between them
                // up to the rounding a zoom introduces.
                let truth = truth_over(items, t as u64, per_sub_window, n) as f64;
                let unit = 2f64.powi(zoom(&cell, &layout) as i32);
                let slack = unit * (t as f64 + 2.0);
                assert!(
                    over + slack >= truth,
                    "t={t} w={per_sub_window}: Over {over} under-reported {truth} \
                     by more than the {slack} a zoom can round away"
                );
            }
        }
    }

    /// Everything ages out: after `T + 1` empty sub-windows nothing of an
    /// earlier burst is left, because each insert clears the slot ahead of it.
    #[test]
    fn a_burst_expires_after_t_plus_one_sub_windows() {
        let layout = MicroLayout::new(MicroParams::new(4, 2));
        let rounding = Rounding::new(1);
        let mut cell = vec![0u8; layout.depth()];
        let mut n = 0u64;
        for _ in 0..300 {
            insert(&mut cell, &layout, n, &rounding);
        }
        assert!(estimate(&cell, &layout, n, DeltaStrategy::Under) > 0.0);
        // Advance past the burst with a single item per sub-window.
        for _ in 0..=(layout.params().t + 1) {
            n += 1;
            enter_sub_window(&mut cell, &layout, n, 1);
            insert(&mut cell, &layout, n, &rounding);
        }
        let est = estimate(&cell, &layout, n, DeltaStrategy::Over);
        assert!(
            est <= (layout.params().t + 2) as f64,
            "only the trickle should remain, got {est}"
        );

        // And a cell that receives nothing at all still forgets, because the
        // boundary is what clears it.
        let mut idle = vec![0u8; layout.depth];
        let mut m = 0u64;
        for _ in 0..300 {
            insert(&mut idle, &layout, m, &rounding);
        }
        assert!(estimate(&idle, &layout, m, DeltaStrategy::Under) > 0.0);
        for _ in 0..=(layout.params().t + 1) {
            m += 1;
            enter_sub_window(&mut idle, &layout, m, 1);
        }
        assert_eq!(
            estimate(&idle, &layout, m, DeltaStrategy::Over),
            0.0,
            "an idle cell must be emptied by the boundary sweep alone"
        );
    }

    /// A zoom-out followed by a zoom-in is lossless when every pixel is a
    /// multiple of `c`, and `Z` never goes below zero.
    #[test]
    fn zoom_out_then_in_restores_multiples_and_z_never_underflows() {
        let layout = MicroLayout::new(MicroParams::new(6, 2));
        let rounding = Rounding::new(7);
        let mut cell = vec![0u8; layout.depth()];
        for (i, slot) in cell.iter_mut().take(layout.pixels()).enumerate() {
            *slot = (2 * (i as u8 + 1)) & 0x7E; // even, and under the ceiling
        }
        let before: Vec<u8> = cell[..layout.pixels()].to_vec();
        assert!(zoom_out(&mut cell, &layout, &rounding));
        assert_eq!(zoom(&cell, &layout), 1);
        assert!(try_zoom_in(&mut cell, &layout));
        assert_eq!(zoom(&cell, &layout), 0);
        assert_eq!(&cell[..layout.pixels()], before.as_slice());
        // Already at Z=0: zooming in again is refused, not an underflow.
        assert!(!try_zoom_in(&mut cell, &layout));
        assert_eq!(zoom(&cell, &layout), 0);
    }

    /// A zoom-in is refused while any pixel would overflow, so it can never
    /// corrupt a cell.
    #[test]
    fn zoom_in_is_refused_when_a_pixel_would_overflow() {
        let layout = MicroLayout::new(MicroParams::new(3, 2));
        let mut cell = vec![0u8; layout.depth()];
        cell[layout.zoom_at] = 2;
        cell[layout.pixel_at(0)] = 10;
        cell[layout.pixel_at(1)] = 200; // 200 * 2 > 255
        assert!(!try_zoom_in(&mut cell, &layout));
        assert_eq!(zoom(&cell, &layout), 2);
        assert_eq!(pixel(&cell, &layout, 1), 200);
    }

    /// `unbiased_div` is unbiased: the mean over many draws lands on the true
    /// quotient, where truncation would sit a fixed distance below it.
    #[test]
    fn unbiased_div_has_the_exact_quotient_as_its_mean() {
        const DRAWS: u32 = 200_000;
        for (value, c) in [(7u32, 2u32), (255, 2), (10, 3), (1, 4), (123, 7)] {
            let rounding = Rounding::new(0xC0FFEE ^ value as u64);
            let mut total: u64 = 0;
            for _ in 0..DRAWS {
                total += unbiased_div(value, c, &rounding) as u64;
            }
            let mean = total as f64 / DRAWS as f64;
            let exact = value as f64 / c as f64;
            assert!(
                (mean - exact).abs() < 0.01,
                "value={value} c={c}: mean {mean} is not the exact quotient {exact}"
            );
            // And truncation would have been visibly biased, unless it
            // divides evenly.
            if value % c != 0 {
                assert!((mean - (value / c) as f64).abs() > 0.05);
            }
        }
    }

    /// An exact multiple divides deterministically — no randomness is
    /// consumed and no rounding is introduced.
    #[test]
    fn unbiased_div_is_deterministic_on_exact_multiples() {
        let rounding = Rounding::new(99);
        let before = rounding.state();
        assert_eq!(unbiased_div(64, 2, &rounding), 32);
        assert_eq!(rounding.state(), before, "no draw should have been taken");
    }

    /// A cell that has zoomed still tracks the truth: the estimate stays
    /// within the rounding a zoom can introduce, rather than drifting.
    #[test]
    fn a_zoomed_cell_tracks_the_window_count() {
        let layout = MicroLayout::new(MicroParams::new(8, 2));
        let per_sub_window = 20_000u64;
        let items = per_sub_window * 12;
        let (cell, n) = run(&layout, items, per_sub_window);
        assert!(
            zoom(&cell, &layout) > 0,
            "this load should have forced a zoom"
        );
        let over = estimate(&cell, &layout, n, DeltaStrategy::Over);
        let truth = truth_over(items, 8, per_sub_window, n) as f64;
        let rel = (over - truth).abs() / truth;
        assert!(
            rel < 0.05,
            "zoomed estimate {over} against {truth} is {:.2}% off",
            rel * 100.0
        );
    }
}
