//! Native MessagePack codec impl for [`crate::sketches::kll_dynamic::KLLDynamic`].

use serde::{Deserialize, Serialize};

use crate::common::numerical::NumericalValue;
use crate::message_pack_format::{Error, MessagePackCodec};
use crate::sketches::kll::KllWireItem;
use crate::sketches::kll_dynamic::KLLDynamic;

impl<T> MessagePackCodec for KLLDynamic<T>
where
    T: NumericalValue + KllWireItem + Serialize + for<'de> Deserialize<'de>,
{
    fn to_msgpack(&self) -> Result<Vec<u8>, Error> {
        Ok(self.serialize_to_bytes()?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self::deserialize_from_bytes(bytes)?)
    }
}
