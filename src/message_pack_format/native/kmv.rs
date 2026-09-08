//! Native MessagePack codec impl for [`crate::sketches::kmv::KMV`].
//!
//! The bytes are the ASAPv1 wire envelope (kind `0x0e 0x00`); the impl is
//! bounded on `H: HashProfile`, so a sketch built with an unprofiled hasher has
//! no codec.

use crate::message_pack_format::{Error, MessagePackCodec};
use crate::sketches::kmv::KMV;
use crate::{HashProfile, SketchHasher};

impl<H: SketchHasher + HashProfile> MessagePackCodec for KMV<H> {
    fn to_msgpack(&self) -> Result<Vec<u8>, Error> {
        Ok(self.serialize_to_bytes()?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self::deserialize_from_bytes(bytes)?)
    }
}
