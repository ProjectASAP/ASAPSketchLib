//! Older `MessagePackCodec` shims over the pure-Rust sketch types in
//! [`crate::sketches`], being retired.
//!
//! Each impl forwards to that sketch's own `serialize_to_bytes` /
//! `deserialize_from_bytes`, so the bytes are ASAPv1 — the framing in
//! `envelope.rs` plus the per-sketch `wire.rs` — and the shim adds nothing
//! to them. Call those methods directly; `docs/asapv1_wire_format.md`
//! specifies the format `sketchlib-go` mirrors.

pub mod countminsketch;
pub mod countsketch;
pub mod countsketch_topk;
pub mod ddsketch;
pub mod hll;
pub mod kll;
pub mod kll_dynamic;
#[cfg(feature = "experimental")]
pub mod kmv;
