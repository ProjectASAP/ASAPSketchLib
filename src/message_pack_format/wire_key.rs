//! The msgpack form of a byte-array key and of a string key.
//!
//! ASAPv1's `key_type` names the exact key variant and the decoder reads the
//! `keys` array **as** that type (`docs/asapv1_wire_format.md` §3.5). msgpack's
//! own decode does not separate `bin` from `str` in either direction:
//! rmp-serde forwards `deserialize_bytes`, `deserialize_str` and
//! `deserialize_string` to `deserialize_any`, which dispatches on the marker
//! alone, and serde's built-in visitors then take the other family — `Vec<u8>`
//! accepts a `str`, and `String` accepts any `bin` holding valid UTF-8. A
//! `Bytes` key and a `String` key are different keys, so a relabelled payload
//! would decode into the wrong variant and then answer `0` for every key it
//! holds.
//!
//! [`WireBytes`] and [`WireString`] close both halves: each writes its own
//! msgpack family and its visitor refuses the other, so neither a `str`-keyed
//! payload relabelled `"bytes"` nor a UTF-8 `bin`-keyed payload relabelled
//! `"string"` decodes.

use std::fmt;

use serde::de::{Deserialize, Deserializer, Error, Visitor};
use serde::ser::{Serialize, Serializer};

/// A byte-array key on the wire: msgpack `bin`, and `bin` only.
#[derive(Debug)]
pub(crate) struct WireBytes(pub(crate) Vec<u8>);

impl WireBytes {
    /// Takes the bytes back out.
    pub(crate) fn into_vec(self) -> Vec<u8> {
        self.0
    }
}

impl Serialize for WireBytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for WireBytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BytesOnly;

        impl<'de> Visitor<'de> for BytesOnly {
            type Value = WireBytes;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a msgpack bin value")
            }

            fn visit_bytes<E: Error>(self, value: &[u8]) -> Result<Self::Value, E> {
                Ok(WireBytes(value.to_vec()))
            }

            fn visit_byte_buf<E: Error>(self, value: Vec<u8>) -> Result<Self::Value, E> {
                Ok(WireBytes(value))
            }

            fn visit_str<E: Error>(self, _: &str) -> Result<Self::Value, E> {
                Err(E::custom(
                    "ASAPv1: a bytes key must be msgpack bin, not str",
                ))
            }
        }

        deserializer.deserialize_bytes(BytesOnly)
    }
}

/// A text key on the wire: msgpack `str`, and `str` only.
#[derive(Debug)]
pub(crate) struct WireString(pub(crate) String);

impl WireString {
    /// Takes the string back out.
    pub(crate) fn into_string(self) -> String {
        self.0
    }
}

impl Serialize for WireString {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for WireString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrOnly;

        impl<'de> Visitor<'de> for StrOnly {
            type Value = WireString;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a msgpack str value")
            }

            fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(WireString(value.to_string()))
            }

            fn visit_string<E: Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(WireString(value))
            }

            fn visit_bytes<E: Error>(self, _: &[u8]) -> Result<Self::Value, E> {
                Err(E::custom(
                    "ASAPv1: a string key must be msgpack str, not bin",
                ))
            }

            fn visit_byte_buf<E: Error>(self, _: Vec<u8>) -> Result<Self::Value, E> {
                Err(E::custom(
                    "ASAPv1: a string key must be msgpack str, not bin",
                ))
            }
        }

        deserializer.deserialize_str(StrOnly)
    }
}
