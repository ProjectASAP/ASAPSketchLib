//! Shared synthetic data generation, ground-truth tracking, and assertion
//! helpers for the E2E sketch test suites.
//!
//! All generators are seeded and deterministic so failures reproduce exactly.
//! Ground truth is tracked exactly while the stream is generated, then used
//! to assert sketch outputs against theory-based tolerances.

// Which helpers are "used" varies by feature flags and per-suite coverage.
#![allow(dead_code)]

pub mod conformance;
pub mod inputs;
pub mod prefix_structure;
pub mod specs;
pub mod streamgen;
pub mod truth;

// Each test binary compiles this module separately, so a re-export used by
// one suite is dead in another.
#[allow(unused_imports)]
pub use inputs::*;

#[allow(unused_imports)]
pub use prefix_structure::*;

#[allow(unused_imports)]
pub use streamgen::*;

#[allow(unused_imports)]
pub use truth::*;
