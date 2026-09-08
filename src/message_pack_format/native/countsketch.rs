//! Native MessagePack codec impl for [`crate::sketches::countsketch::Count`].
//!
//! Only the canonical wire configs — `i32` or `i64` counters (`CsWireCounter`)
//! with a fast/regular mode (`CsWireMode`) — are serializable. There is no `f64`
//! counterpart to Count-Min's, because `CountSketchCounter` is implemented only
//! for `i32`, `i64` and `i128`; `i128` and non-`Vector2D` storage must be
//! converted first.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error, MessagePackCodec};
use crate::sketches::countsketch::{Count, CountSketchCounter, CsWireCounter, CsWireMode};
use crate::{HashProfile, SketchHasher, Vector2D};

impl<T, Mode, H> MessagePackCodec for Count<Vector2D<T>, Mode, H>
where
    T: CsWireCounter
        + CountSketchCounter
        + std::ops::AddAssign
        + Serialize
        + for<'de> Deserialize<'de>,
    Mode: CsWireMode,
    H: SketchHasher + HashProfile,
{
    fn to_msgpack(&self) -> Result<Vec<u8>, Error> {
        Ok(self.serialize_to_bytes()?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self::deserialize_from_bytes(bytes)?)
    }
}
