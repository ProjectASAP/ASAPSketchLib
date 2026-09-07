//! Older per-sketch MessagePack wire types, being retired.
//!
//! Each per-algorithm submodule holds one sketch's pre-envelope wire type.
//! What `sketchlib-go` mirrors is ASAPv1 — the framing in `envelope.rs`
//! plus the per-sketch `wire.rs` beside each sketch — not these types.
//! [`hll::HllSketch`] is the exception: it encodes through the HLL sketch's
//! own ASAPv1 framing, metadata and payload, so the two agree byte for
//! byte. Prefer a sketch's own `serialize_to_bytes`; do not add callers here.

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
