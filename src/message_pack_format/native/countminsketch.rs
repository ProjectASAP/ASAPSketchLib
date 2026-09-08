//! Native MessagePack codec impl for [`crate::sketches::countminsketch::CountMin`].
//!
//! Only the canonical wire configs — i32/i64/f64 counters (`CmsWireCounter`)
//! with a fast/regular mode (`CmsWireMode`) — are serializable. Exotic in-memory
//! counters (i128/…) must be converted to a wire type first.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error, MessagePackCodec};
use crate::sketches::countminsketch::{CmsWireCounter, CmsWireMode, CountMin};
use crate::{HashProfile, SketchHasher, Vector2D};

impl<T, Mode, H> MessagePackCodec for CountMin<Vector2D<T>, Mode, H>
where
    // `AddAssign` is required for `Vector2D<T>: MatrixStorage`.
    T: CmsWireCounter + std::ops::AddAssign + Serialize + for<'de> Deserialize<'de>,
    Mode: CmsWireMode,
    H: SketchHasher + HashProfile,
{
    fn to_msgpack(&self) -> Result<Vec<u8>, Error> {
        Ok(self.serialize_to_bytes()?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self::deserialize_from_bytes(bytes)?)
    }
}
