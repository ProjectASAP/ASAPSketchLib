//! Native MessagePack codec impl for [`crate::sketches::ddsketch::DDSketch`].
//!
//! Produces and consumes the ASAPv1 envelope (kind_id `0x05 0x00`): `alpha` in
//! the metadata, `[counts, offset, sum, min, max]` in the payload.

use crate::message_pack_format::{Error, MessagePackCodec};
use crate::sketches::ddsketch::DDSketch;

impl MessagePackCodec for DDSketch {
    fn to_msgpack(&self) -> Result<Vec<u8>, Error> {
        Ok(self.serialize_to_bytes()?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self::deserialize_from_bytes(bytes)?)
    }
}
