//! MicroscopeSketch: sliding-window frequency estimation with adaptive zoom.
//!
//! A sliding-window counter normally pays for its window in memory: keep one
//! counter per sub-window, and a window of `T` sub-windows costs `T` counters
//! per cell, each wide enough for the largest count it might ever hold. Most
//! of those bits are idle most of the time.
//!
//! MicroscopeSketch instead gives each cell `T + 2` narrow **pixels** and one
//! shared **zoom** exponent `Z`, so a pixel is worth `c^Z` items and the cell
//! rescales itself when a pixel is about to overflow. The counters stay
//! narrow; the exponent buys the range.
//!
//! [`cell`] holds the record layout and the per-cell algorithm, on `&[u8]`
//! and nothing else. This module puts a grid of those cells behind a hash.
//!
//! # Why the fields share a record
//!
//! `Z` is shared by every pixel of a cell: a pixel value means nothing
//! without it, and a zoom rescales all of them together. An insert reads the
//! shutter, may roll it into a pixel, and may rescale every pixel — one
//! cell's worth of coupled state per item. That is what
//! [`Vector3D`](crate::Vector3D) is for, and why `T + 2` separate
//! [`Vector2D`](crate::Vector2D)s would not be the same structure.
//!
//! # Status
//!
//! Experimental: behind the `experimental` cargo feature. The algorithm
//! follows Zhao et al., but this implementation has not been checked against
//! the paper's published measurements, so the constants and the accuracy it
//! achieves are not yet corroborated by anything outside this repository.
//!
//! # References
//!
//! - Zhao, Wang, Li, Dong, Yang, Chen, Zhang, Uhlig, "MicroscopeSketch:
//!   Accurate Sliding Estimation Using Adaptive Zooming," KDD 2023.

/// The per-cell record layout and algorithm.
pub mod cell;

pub use cell::{DeltaStrategy, MicroLayout, MicroParams, Rounding};
