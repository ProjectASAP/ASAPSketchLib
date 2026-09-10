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
/// so the choice is a policy.
///
/// The one-sidedness these describe is about the **span**: `Over` charges a
/// superset of the window and `Under` a subset. It is exact only while the
/// cell has not zoomed. Once `Z > 0` a pixel has been through probabilistic
/// rounding, which errs in both directions by up to `c^Z` per pixel, so the
/// two bracket the true count only up to that rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeltaStrategy {
    /// Count all of the oldest sub-window: the widest span, so it does not
    /// under-report the window.
    Over,
    /// Count none of it: the narrowest span, so it does not over-report the
    /// window.
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

/// Upper bound on the zoom base.
///
/// A pixel is one byte, so a single zoom-out at `c > 256` collapses every
/// pixel to 0 or 1 and the cell carries no information below the exponent.
pub const MAX_ZOOM_BASE: u32 = 256;

/// Upper bound on `T`, so a record stays a sane size and `T + 2` cannot
/// overflow while a layout is being derived from untrusted bytes.
///
/// Re-exported as [`crate::sketches::microscope::MAX_SUB_WINDOWS`].
pub const MAX_SUB_WINDOWS: usize = 4096;

impl MicroParams {
    /// Builds parameters, rejecting the values the layout cannot represent.
    ///
    /// Panics if `t` is zero or above
    /// [`MAX_SUB_WINDOWS`](crate::sketches::microscope::MAX_SUB_WINDOWS), or
    /// if `c` is outside
    /// `2..=`[`MAX_ZOOM_BASE`](crate::sketches::microscope::MAX_ZOOM_BASE).
    pub fn new(t: usize, c: u32) -> Self {
        Self::checked(t, c).unwrap_or_else(|detail| panic!("{detail}"))
    }

    /// [`Self::new`] as a `Result`, for validating deserialized values.
    pub fn checked(t: usize, c: u32) -> Result<Self, String> {
        if t == 0 || t > MAX_SUB_WINDOWS {
            return Err(format!(
                "t (sub-windows per window) must be in 1..={MAX_SUB_WINDOWS}, got {t}"
            ));
        }
        if !(2..=MAX_ZOOM_BASE).contains(&c) {
            return Err(format!(
                "zoom base c must be in 2..={MAX_ZOOM_BASE}, got {c}"
            ));
        }
        Ok(Self { t, c })
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
/// with the record length rounded up to a multiple of 4.
///
/// Every field is read and written a byte at a time, so nothing here needs
/// an alignment guarantee; the padding only keeps the record size a round
/// number. Packing the zoom and the shutter into the bits they actually
/// need would shrink a record further, at the cost of byte-addressable
/// fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MicroLayout {
    params: MicroParams,
    pixels: usize,
    zoom_at: usize,
    shutter_at: usize,
    depth: usize,
    /// A zoom-in is refused when any pixel is `>= 256 / c`, since multiplying
    /// it by `c` would not fit a byte. The test is conservative for a `c`
    /// that does not divide 256: at `c = 3` it refuses 85, whose product 255
    /// would in fact have fitted.
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
            zoom_in_ceiling: PIXEL_MAX as u32 / params.c,
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

/// Checks the two invariants a cell's own bytes must satisfy, for use on
/// deserialized records.
///
/// Neither is expressible in the layout, so a crafted payload can carry a
/// record that decodes cleanly and then misbehaves on first use:
///
/// - `Z` above the cap makes `c^Z` overflow, which panics in a debug build
///   and silently wraps to a nonsense unit in a release one.
/// - a shutter at or above `c^Z` breaks the assumption that a shutter holds
///   less than one unit, which is what bounds the carry a merge computes.
pub fn validate(cell: &[u8], layout: &MicroLayout) -> Result<(), String> {
    let z = cell[layout.zoom_at];
    let cap = layout.max_zoom();
    if z > cap {
        return Err(format!("zoom exponent {z} exceeds the maximum {cap}"));
    }
    let unit = (layout.params.c as u64).pow(z as u32);
    let s = shutter(cell, layout) as u64;
    if s >= unit {
        return Err(format!(
            "shutter {s} is not below one unit (c^Z = {unit}) at zoom {z}"
        ));
    }
    Ok(())
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

/// The count sub-window `n` holds in this cell, in items rather than pixel
/// units — that is, the pixel scaled by `c^Z`.
#[inline(always)]
pub fn scaled_pixel(cell: &[u8], layout: &MicroLayout, n: u64) -> f64 {
    let unit = (layout.params.c as f64).powi(cell[layout.zoom_at] as i32);
    cell[layout.pixel_at(n)] as f64 * unit
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
    unbiased_div_u64(value as u64, c as u64, rounding) as u32
}

/// [`unbiased_div`] over `u64`, for dividing by a `c^Z` unit rather than by
/// `c` itself.
#[inline]
pub fn unbiased_div_u64(value: u64, divisor: u64, rounding: &Rounding) -> u64 {
    debug_assert!(divisor > 0, "divisor must be non-zero");
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder == 0 {
        return quotient;
    }
    if (rounding.next_u64() % divisor) < remainder {
        quotient + 1
    } else {
        quotient
    }
}

/// Halves the cell: `Z += 1` and every pixel is divided by `c`.
///
/// At the zoom cap this is a no-op, and the caller saturates the pixel
/// instead.
/// Adds one unit to the pixel of sub-window `at`, zooming the cell out if
/// that pixel has no room left.
///
/// The zoom is not "divide, then increment". A pixel at `2^l - 1` conceptually
/// takes the increment to `2^l` and *then* halves, which lands exactly on
/// `2^l / c`; dividing first and adding one afterwards credits up to a whole
/// extra unit every time a cell zooms, a systematic upward bias. So the
/// crediting pixel is **set** to `2^l / c` and only the other pixels are
/// divided.
///
/// At the zoom cap the pixel saturates rather than wrapping, and the unit is
/// left alone.
fn credit_pixel(cell: &mut [u8], layout: &MicroLayout, at: usize, rounding: &Rounding) {
    if cell[at] < PIXEL_MAX {
        cell[at] += 1;
        return;
    }
    let z = cell[layout.zoom_at];
    if z >= layout.max_zoom() {
        return;
    }
    cell[layout.zoom_at] = z + 1;
    let c = layout.params.c;
    for (index, slot) in cell.iter_mut().take(layout.pixels).enumerate() {
        *slot = if index == at {
            ((PIXEL_MAX as u32 + 1) / c) as u8
        } else {
            unbiased_div(*slot as u32, c, rounding) as u8
        };
    }
}

/// Records one item in sub-window `n`.
///
/// The shutter advances by one and rolls a whole `c^Z` unit into the current
/// pixel when it reaches one, carrying the remainder rather than discarding
/// it.
///
/// The loop is a guard, not a path: `shutter < c^Z` holds everywhere, so
/// `pending` starts at no more than one unit and the body runs at most
/// once. It is written as a loop because crediting a pixel may raise `Z`,
/// and a remainder measured against the old unit would then be wrong — the
/// shape stays correct if a future change ever lets the shutter grow.
///
/// Expiry is not done here — see [`enter_sub_window`].
pub fn insert(cell: &mut [u8], layout: &MicroLayout, n: u64, rounding: &Rounding) {
    let current = layout.pixel_at(n);
    let c = layout.params.c as u64;
    let mut pending = shutter(cell, layout) as u64 + 1;
    loop {
        let unit = c.pow(cell[layout.zoom_at] as u32);
        if pending < unit {
            break;
        }
        pending -= unit;
        // May raise Z, which is why `unit` is re-read each time round.
        credit_pixel(cell, layout, current, rounding);
    }
    set_shutter(cell, layout, pending as u32);
}

/// Moves the cell into sub-window `new_n`, having crossed `crossed`
/// boundaries to get there.
///
/// Three things happen, and every cell gets all three — not just the cell an
/// insert happens to touch.
///
/// 1. **The shutter is closed out.** It holds a partial `c^Z` unit belonging
///    to the sub-window that just ended, so it is rounded into that
///    sub-window's pixel — up with probability `S / c^Z` — and reset. A
///    shutter that survived the boundary would be re-reported for as long as
///    the cell lived, since nothing else ever clears it.
/// 2. **The slots the new sub-windows land on are cleared.** The pixel ring
///    is indexed by sub-window number, so the slot a new sub-window takes
///    still holds the count of the sub-window `T + 2` earlier; unless it is
///    cleared on the way in, that stale count is read back as if it were
///    recent. A cell whose key stops appearing receives no inserts, and is
///    exactly the cell whose stale pixels would otherwise be re-read — so a
///    key that went quiet would keep reporting its old count forever.
/// 3. **Resolution is reclaimed**, as far as the pixels allow.
///
/// A jump of `T + 2` or more sub-windows has left nothing behind, so the
/// cell is reset outright rather than cleared slot by slot.
pub fn enter_sub_window(
    cell: &mut [u8],
    layout: &MicroLayout,
    new_n: u64,
    crossed: u64,
    rounding: &Rounding,
) {
    if crossed == 0 {
        return;
    }
    if crossed >= layout.pixels as u64 {
        for slot in cell.iter_mut().take(layout.pixels) {
            *slot = 0;
        }
        set_shutter(cell, layout, 0);
    } else {
        // 1. Close out the shutter into the sub-window that just ended.
        let ended = new_n - crossed;
        let unit = (layout.params.c as u64).pow(cell[layout.zoom_at] as u32);
        let pending = shutter(cell, layout) as u64;
        set_shutter(cell, layout, 0);
        if pending > 0 && unit > 0 {
            // `pending < unit`, so this is a coin at odds `pending : unit`.
            let carry = unbiased_div_u64(pending, unit, rounding);
            if carry > 0 {
                credit_pixel(cell, layout, layout.pixel_at(ended), rounding);
            }
        }

        // 2. Clear each slot entered on the way, oldest of them first.
        for step in (0..crossed).rev() {
            cell[layout.pixel_at(new_n - step)] = 0;
        }
    }

    // 3. Reclaim as much resolution as the pixels now allow.
    while try_zoom_in(cell, layout) {}
}

/// Restores resolution when the cell has room for it: if every pixel would
/// still fit after multiplying by `c`, multiply them and drop `Z` by one.
///
/// [`enter_sub_window`] drives this to a fixed point, so a cell that emptied
/// out recovers full resolution in one boundary rather than one step per
/// boundary. Returns whether it zoomed.
///
/// The shutter is left alone, which is safe only because the boundary has
/// just reset it: at any other moment lowering `Z` could leave the shutter at
/// or above the new unit, and the next insert would then credit a whole
/// pixel for less than a unit of pending items.
pub fn try_zoom_in(cell: &mut [u8], layout: &MicroLayout) -> bool {
    let z = cell[layout.zoom_at];
    if z == 0 {
        return false;
    }
    // `p * c <= 255` is the condition; comparing against `255 / c` says the
    // same thing without a multiply, and — unlike `256 / c` — never rounds
    // to zero, which would have made the test vacuously true for every cell
    // and left a zoomed-out cell unable to ever zoom back in.
    let ceiling = layout.zoom_in_ceiling;
    if cell.iter().take(layout.pixels).any(|&p| p as u32 > ceiling) {
        return false;
    }
    let c = layout.params.c;
    for slot in cell.iter_mut().take(layout.pixels) {
        *slot = (*slot as u32 * c) as u8;
    }
    cell[layout.zoom_at] = z - 1;
    true
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

    /// The pixel value for sub-window `n`.
    fn pixel(cell: &[u8], layout: &MicroLayout, n: u64) -> u8 {
        cell[layout.pixel_at(n)]
    }

    /// A single cell's window count, which is what a one-row sketch would
    /// answer.
    ///
    /// The grid does not go through this: it reduces across rows per
    /// sub-window and sums the minima, which is tighter than summing a cell
    /// and then reducing. This is the per-cell reference the property tests
    /// below are written against.
    fn estimate(cell: &[u8], layout: &MicroLayout, n: u64, strategy: DeltaStrategy) -> f64 {
        let t = layout.params.t as u64;
        let mut total = shutter(cell, layout) as f64;
        for back in 0..t {
            if let Some(sub_window) = n.checked_sub(back) {
                total += scaled_pixel(cell, layout, sub_window);
            }
        }
        let oldest = n
            .checked_sub(t)
            .map(|sub_window| scaled_pixel(cell, layout, sub_window))
            .unwrap_or(0.0);
        total += match strategy {
            DeltaStrategy::Over => oldest,
            DeltaStrategy::Under => 0.0,
            DeltaStrategy::Linear(fraction) => fraction.clamp(0.0, 1.0) * oldest,
        };
        total
    }

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
                enter_sub_window(&mut cell, layout, n, 1, &rounding);
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
            enter_sub_window(&mut cell, &layout, n, 1, &rounding);
            insert(&mut cell, &layout, n, &rounding);
        }
        let est = estimate(&cell, &layout, n, DeltaStrategy::Over);
        assert!(
            est <= (layout.params().t + 2) as f64,
            "only the trickle should remain, got {est}"
        );
    }

    /// A cell that receives nothing is emptied by the boundary sweep alone,
    /// **at every burst size** — including the ones that leave a partial
    /// shutter behind.
    ///
    /// The shutter is the trap here. It holds a fraction of a `c^Z` unit, and
    /// nothing but the boundary ever resets it, so a version that clears only
    /// the pixels leaves a floor that is reported for the rest of the cell's
    /// life. Sweeping the burst size is what makes that visible: a single
    /// size can land on `shutter == 0` and pass a broken implementation.
    #[test]
    fn an_idle_cell_is_emptied_whatever_it_was_holding() {
        let layout = MicroLayout::new(MicroParams::new(4, 2));
        // Sizes chosen to span several zoom levels and every shutter residue.
        for burst in [1u64, 2, 3, 255, 256, 257, 299, 300, 301, 4_095, 100_000] {
            let rounding = Rounding::new(0xB0_1234 ^ burst);
            let mut cell = vec![0u8; layout.depth()];
            let mut n = 0u64;
            for _ in 0..burst {
                insert(&mut cell, &layout, n, &rounding);
            }
            assert!(
                estimate(&cell, &layout, n, DeltaStrategy::Under) > 0.0 || burst < (1 << 0),
                "burst={burst}: the cell should be holding something"
            );
            for _ in 0..=(layout.params().t + 1) {
                n += 1;
                enter_sub_window(&mut cell, &layout, n, 1, &rounding);
            }
            assert_eq!(
                shutter(&cell, &layout),
                0,
                "burst={burst}: the shutter must not survive a boundary"
            );
            assert_eq!(
                estimate(&cell, &layout, n, DeltaStrategy::Over),
                0.0,
                "burst={burst}: an idle cell must be emptied by the sweep alone"
            );
            assert_eq!(
                zoom(&cell, &layout),
                0,
                "burst={burst}: an emptied cell must be back at full resolution"
            );
        }
    }

    /// A zoom-out followed by a zoom-in is lossless for the pixels that were
    /// merely divided, and `Z` never goes below zero.
    #[test]
    fn zoom_out_then_in_restores_multiples_and_z_never_underflows() {
        let layout = MicroLayout::new(MicroParams::new(6, 2));
        let rounding = Rounding::new(7);
        let mut cell = vec![0u8; layout.depth()];
        for (i, slot) in cell.iter_mut().take(layout.pixels()).enumerate() {
            *slot = (2 * (i as u8 + 1)) & 0x7E; // even, and under the ceiling
        }
        // Pixel 0 is the one that overflows and triggers the zoom.
        cell[0] = PIXEL_MAX;
        let others: Vec<u8> = cell[1..layout.pixels()].to_vec();

        credit_pixel(&mut cell, &layout, 0, &rounding);
        assert_eq!(zoom(&cell, &layout), 1);
        // The crediting pixel lands exactly on 2^l / c, not on a divided-then-
        // incremented value.
        assert_eq!(cell[0], 128);
        // Every other pixel was even, so its division is exact.
        for (slot, before) in cell[1..layout.pixels()].iter().zip(&others) {
            assert_eq!(*slot as u32, *before as u32 / 2);
        }

        // 128 is exactly the zoom-in ceiling, so the cell that just zoomed
        // out cannot immediately zoom back in and thrash.
        assert!(!try_zoom_in(&mut cell, &layout));
        assert_eq!(zoom(&cell, &layout), 1);

        // Once that pixel drains, the zoom-in is lossless for the pixels
        // that were merely divided.
        cell[0] = 4;
        assert!(try_zoom_in(&mut cell, &layout));
        assert_eq!(zoom(&cell, &layout), 0);
        assert_eq!(cell[0], 8);
        assert_eq!(&cell[1..layout.pixels()], others.as_slice());
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

    /// The shutter a boundary closes out is credited to the sub-window that
    /// just ended, and credited at the right odds: `S / c^Z`.
    ///
    /// This is the mechanism that stops a partial unit being reported for
    /// the rest of a cell's life, and dropping it is invisible to a test
    /// that only checks the shutter reaches zero — resetting it without
    /// crediting does that too, while biasing every zoomed cell down by
    /// half a unit per boundary. Measuring the rate catches both halves.
    #[test]
    fn the_boundary_credits_the_shutter_it_closes_out_at_the_right_odds() {
        const SEEDS: u64 = 4_000;
        let layout = MicroLayout::new(MicroParams::new(4, 2));
        // Z = 3, so one unit is 8 items and a shutter of S must be carried
        // S times in 8.
        for pending in [0u32, 1, 6, 7] {
            let mut credited = 0u64;
            for seed in 0..SEEDS {
                let rounding = Rounding::new(0xA11CE ^ seed);
                let mut cell = vec![0u8; layout.depth()];
                cell[layout.zoom_at] = 3;
                cell[layout.pixel_at(0)] = 10;
                set_shutter(&mut cell, &layout, pending);
                // `scaled_pixel` is invariant under the zoom-in the boundary
                // then performs, so it measures the credit and nothing else.
                let before = scaled_pixel(&cell, &layout, 0);
                enter_sub_window(&mut cell, &layout, 1, 1, &rounding);
                let gained = scaled_pixel(&cell, &layout, 0) - before;
                assert!(
                    gained == 0.0 || gained == 8.0,
                    "S={pending}: a boundary credits nothing or one unit, got {gained}"
                );
                if gained > 0.0 {
                    credited += 1;
                }
                assert_eq!(shutter(&cell, &layout), 0, "S={pending}: not reset");
            }
            let rate = credited as f64 / SEEDS as f64;
            let expected = pending as f64 / 8.0;
            assert!(
                (rate - expected).abs() < 0.03,
                "S={pending}: carried {rate} of the time, expected {expected}"
            );
        }
    }

    /// A jump of more than one but fewer than `T + 2` sub-windows clears
    /// exactly the slots it entered, and leaves the rest alone.
    ///
    /// The only multi-boundary path otherwise exercised is a jump past the
    /// whole ring, which takes the full-reset branch instead.
    #[test]
    fn a_multi_sub_window_jump_clears_only_what_it_entered() {
        let layout = MicroLayout::new(MicroParams::new(4, 2));
        let rounding = Rounding::new(4_242);
        let mut cell = vec![0u8; layout.depth()];
        // Distinct marks so a wrong slot is visible, in sub-windows 0..=5.
        for n in 0..6u64 {
            cell[layout.pixel_at(n)] = (n as u8 + 1) * 10;
        }
        let before: Vec<u8> = cell[..layout.pixels()].to_vec();

        // Standing in sub-window 5, jump three to sub-window 8: that enters
        // 6, 7 and 8, and nothing else may change.
        enter_sub_window(&mut cell, &layout, 8, 3, &rounding);

        for n in 0..9u64 {
            let entered = (6..=8).contains(&n);
            let slot = layout.pixel_at(n);
            if entered {
                assert_eq!(
                    cell[slot], 0,
                    "sub-window {n} was entered and must have been cleared"
                );
            }
        }
        // The three slots the jump did not touch keep their marks, scaled by
        // whatever zoom-in the boundary performed (Z was 0, so unchanged).
        assert_eq!(zoom(&cell, &layout), 0);
        for n in 3..=5u64 {
            let slot = layout.pixel_at(n);
            assert_eq!(
                cell[slot], before[slot],
                "sub-window {n} was not entered and must be untouched"
            );
        }
    }

    /// The boundary drives zoom-in to a fixed point, so a cell that empties
    /// out recovers full resolution at the next boundary rather than one
    /// step per boundary.
    #[test]
    fn a_boundary_reclaims_all_the_resolution_the_pixels_allow() {
        let layout = MicroLayout::new(MicroParams::new(4, 2));
        let rounding = Rounding::new(31_337);
        let mut cell = vec![0u8; layout.depth()];
        // Load one sub-window hard enough to force several zoom-outs.
        for _ in 0..300_000u64 {
            insert(&mut cell, &layout, 0, &rounding);
        }
        let zoomed = zoom(&cell, &layout);
        assert!(zoomed >= 3, "expected several zoom-outs, got Z={zoomed}");

        // One quiet boundary clears the entered slot; the burst's own pixel
        // still holds a large value, so resolution cannot come back yet.
        enter_sub_window(&mut cell, &layout, 1, 1, &rounding);
        assert_eq!(
            zoom(&cell, &layout),
            zoomed,
            "the loaded pixel still needs the range"
        );

        // Once the burst's sub-window ages out, the very next boundary must
        // recover every level at once, not one per boundary.
        let mut n = 1u64;
        while pixel(&cell, &layout, 0) != 0 {
            n += 1;
            enter_sub_window(&mut cell, &layout, n, 1, &rounding);
        }
        assert_eq!(
            zoom(&cell, &layout),
            0,
            "the boundary that empties the last loaded pixel must reclaim \
             every zoom level in one pass"
        );
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
        // A zoom rounds each pixel by less than one unit, and the estimate
        // reads T+2 of them, so that product is the slack the arithmetic
        // itself allows. It moves with the load and the parameters, unlike a
        // percentage.
        let unit = 2f64.powi(zoom(&cell, &layout) as i32);
        let slack = unit * (layout.pixels() as f64);
        assert!(
            (over - truth).abs() <= slack,
            "zoomed estimate {over} against {truth} is off by more than the \
             {slack} that {} pixels of unit {unit} can round away",
            layout.pixels()
        );
    }
}
