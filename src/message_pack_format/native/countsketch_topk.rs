//! Native MessagePack codec impl for [`crate::sketches::countsketch_topk::CountL2HH`].

use crate::message_pack_format::{Error, MessagePackCodec};
use crate::sketches::countsketch_topk::CountL2HH;
use crate::{HashProfile, SketchHasher};

impl<H: SketchHasher + HashProfile> MessagePackCodec for CountL2HH<H> {
    fn to_msgpack(&self) -> Result<Vec<u8>, Error> {
        Ok(self.serialize_to_bytes()?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self::deserialize_from_bytes(bytes)?)
    }
}
