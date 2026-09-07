//! Rust-side MessagePack serialization plumbing.
//!
//! The current format is **ASAPv1**: each sketch serializes into one
//! self-delimiting envelope — a sketch-agnostic frame (magic, version,
//! `kind_id`, two length prefixes) around a metadata map and a payload
//! array. The framing lives in `envelope.rs`; the `kind_id`, metadata and
//! payload are per-sketch, in the `wire.rs` beside each sketch under
//! [`crate::sketches`] and [`crate::sketch_framework`]. `sketchlib-go`
//! mirrors ASAPv1, `asapv1_golden/` guards against drift, and
//! `docs/asapv1_wire_format.md` is the spec.
//!
//! Two sub-modules are the older path, being retired:
//!
//! - [`portable`] — per-sketch wire types predating the envelope.
//! - [`native`] — thin shims over the sketches' `serialize_to_bytes` /
//!   `deserialize_from_bytes`, each a pass-through to ASAPv1.
//!
//! [`MessagePackCodec`] and [`Error`] are the contract both share.

pub mod codec;
pub(crate) mod envelope;
pub mod error;
pub mod native;
pub mod portable;
pub(crate) mod wire_key;

pub use codec::MessagePackCodec;
pub use error::Error;
