//! Shared ASAPv1 envelope framing (sketch-agnostic).
//!
//! Every serialized ASAP sketch binary is wrapped in this envelope:
//!
//! ```text
//! [ magic:6 | version:u8 | kind_id_len:u8 | kind_id:bytes
//!           | metadata_len:u32_be | payload_len:u32_be
//!           | metadata:msgpack | payload:msgpack ]
//! ```
//!
//! This module owns only the parts that are identical for every sketch: the
//! magic sentinel, the layout version, and the byte framing ([`encode`] /
//! [`split`]). The hash-profile values each sketch's metadata is built from live
//! with the hasher itself (see [`crate::HashProfile`]). The `kind_id`, the
//! metadata *contents*, and the payload are per-sketch and defined alongside
//! each sketch — see `docs/asapv1_wire_format.md`.
//!
//! [`split`] validates only the magic, version, and framing; it does **not**
//! check `kind_id` against a registry. Each sketch decoder checks that the
//! `kind_id` is one it owns.

/// 6-byte ASCII sentinel opening every ASAP sketch binary (`b"ASAPv1"`).
pub(crate) const MAGIC: &[u8; 6] = b"ASAPv1";

/// Envelope layout version. Bumped only when the framing itself changes.
pub(crate) const VERSION: u8 = 0x01;

/// Assemble the envelope around an already-encoded metadata block and payload.
pub(crate) fn encode(kind_id: &[u8], metadata: &[u8], payload: &[u8]) -> Vec<u8> {
    let kind_id_len = u8::try_from(kind_id.len()).expect("ASAPv1 kind_id too long (>255 bytes)");
    let metadata_len = u32::try_from(metadata.len()).expect("ASAPv1 metadata too large");
    let payload_len = u32::try_from(payload.len()).expect("ASAPv1 payload too large");
    let mut out = Vec::with_capacity(
        MAGIC.len() + 1 + 1 + kind_id.len() + 4 + 4 + metadata.len() + payload.len(),
    );
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(kind_id_len);
    out.extend_from_slice(kind_id);
    out.extend_from_slice(&metadata_len.to_be_bytes());
    out.extend_from_slice(&payload_len.to_be_bytes());
    out.extend_from_slice(metadata);
    out.extend_from_slice(payload);
    out
}

/// The three envelope slices returned by [`split`]: `(kind_id, metadata, payload)`.
pub(crate) type Parts<'a> = (&'a [u8], &'a [u8], &'a [u8]);

/// Split an envelope into `(kind_id, metadata, payload)`, validating the magic,
/// version, and that the two length-framed blocks fit. Returns an error string
/// on any structural mismatch. Does **not** validate `kind_id` — the caller does.
pub(crate) fn split(bytes: &[u8]) -> Result<Parts<'_>, String> {
    let magic_len = MAGIC.len();
    let kind_id_len_offset = magic_len + 1;
    let kind_id_offset = magic_len + 2;
    let header_min = kind_id_offset + 4 + 4;

    if bytes.len() < header_min {
        return Err(format!(
            "ASAPv1 envelope: too short ({} bytes, need at least {header_min})",
            bytes.len()
        ));
    }
    if &bytes[..magic_len] != MAGIC {
        return Err("ASAPv1 envelope: bad magic".to_string());
    }
    if bytes[magic_len] != VERSION {
        return Err(format!(
            "ASAPv1 envelope: unsupported version 0x{:02x}",
            bytes[magic_len]
        ));
    }
    let kind_id_len = bytes[kind_id_len_offset] as usize;
    let lengths_offset = kind_id_offset + kind_id_len;
    if bytes.len() < lengths_offset + 8 {
        return Err("ASAPv1 envelope: truncated kind_id / length fields".to_string());
    }
    let kind_id = &bytes[kind_id_offset..lengths_offset];

    let metadata_len = u32::from_be_bytes(
        bytes[lengths_offset..lengths_offset + 4]
            .try_into()
            .expect("four-byte length slice"),
    ) as usize;
    let payload_len = u32::from_be_bytes(
        bytes[lengths_offset + 4..lengths_offset + 8]
            .try_into()
            .expect("four-byte length slice"),
    ) as usize;
    let metadata_start = lengths_offset + 8;
    // Use checked arithmetic so crafted lengths near usize::MAX (reachable on
    // 32-bit targets) fail closed with an error instead of overflowing the
    // bounds check and panicking on the slice index.
    let payload_start = metadata_start
        .checked_add(metadata_len)
        .ok_or_else(|| "ASAPv1 envelope: length overflow".to_string())?;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| "ASAPv1 envelope: length overflow".to_string())?;
    if bytes.len() < payload_end {
        return Err("ASAPv1 envelope: truncated metadata / payload".to_string());
    }
    Ok((
        kind_id,
        &bytes[metadata_start..payload_start],
        &bytes[payload_start..payload_end],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_frames_metadata_and_payload() {
        let kind_id = [0x01u8, 0x03];
        let metadata = b"meta-block";
        let payload = b"payload-bytes";
        let bytes = encode(&kind_id, metadata, payload);

        assert!(bytes.starts_with(MAGIC));
        assert_eq!(bytes[6], VERSION);
        assert_eq!(bytes[7], 2); // kind_id_len

        let (k, m, p) = split(&bytes).expect("split");
        assert_eq!(k, kind_id);
        assert_eq!(m, metadata);
        assert_eq!(p, payload);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut bytes = encode(&[0x01, 0x01], b"m", b"p");
        assert!(split(&bytes).is_ok());
        bytes[0] = b'X';
        assert!(split(&bytes).is_err());

        let mut bytes = encode(&[0x01, 0x01], b"m", b"p");
        bytes[6] = 0xFF; // version
        assert!(split(&bytes).is_err());
    }

    #[test]
    fn rejects_truncation() {
        let bytes = encode(&[0x01, 0x01], b"metadata", b"payload");
        assert!(split(&bytes[..bytes.len() - 1]).is_err());
        assert!(split(&bytes[..3]).is_err());
    }
}
