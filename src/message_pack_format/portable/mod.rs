//! Older per-sketch MessagePack wire types, being retired.
//!
//! Each per-algorithm submodule holds one sketch's pre-envelope wire type.
//! These carry a cross-language contract of their own, on goldens apart
//! from ASAPv1's `asapv1_golden/`: HLL, Count-Min, Count Sketch and KLL pin
//! byte parity against Go's serializers, KLL in both directions; DDSketch's
//! golden is a Rust self-pin, not yet reconciled. New code uses ASAPv1:
//! prefer a sketch's own `serialize_to_bytes`; do not add callers here.

pub mod countminsketch;
pub mod countminsketch_topk;
pub mod countsketch;
pub mod countsketch_topk;
pub mod ddsketch;
pub mod delta_set_aggregator;
pub mod hll;
pub mod hydra_kll;
pub mod kll;
pub mod sampling;
pub mod set_aggregator;
