//! Wire-format-aligned HyperLogLog types.
//!
//! Self-contained: maintains its own `Vec<u8>` register array. Parity
//! with Go's `HyperLogLog.SerializePortable` (and equivalence with
//! `sketches::HyperLogLogImpl<.., ErtlMLE, P14>`) is locked in by the
//! `test_update_then_envelope_matches_sketchlib_go_bytes` test below and
//! by `tests/sketches_go_parity_probe.rs`.

use serde::{Deserialize, Serialize};

use crate::message_pack_format::{Error as MsgPackError, MessagePackCodec};
use crate::{CANONICAL_HASH_SEED, DataInput, hash64_seeded};

// ---------------------------------------------------------------------------
// DEPRECATED — temporary compatibility shim, slated for removal.
//
// A few call sites still depend on this portable `HllSketch`. Its ASAPv1 codec
// now reuses the shared `envelope` framing plus the HLL metadata/payload types
// in `crate::sketches::hll`, so it stays byte-identical with the native
// `HyperLogLogImpl` / `HyperLogLogHIPImpl` serialization. Do NOT add new callers.
//
// Prefer instead the native path:
// `sketches::hll::HyperLogLogImpl::serialize_to_bytes` /
// `HyperLogLogHIPImpl::serialize_to_bytes` (+ `deserialize_from_bytes`).
// ---------------------------------------------------------------------------

use crate::message_pack_format::envelope;
use crate::sketches::hll::{
    HLL_KIND_CLASSIC, HLL_KIND_ERTL_MLE, HLL_KIND_HIP, HllMetadata, HllPayloadHip, HllPayloadPlain,
    standard_hll_metadata,
};

/// Retained only for `test_msgpack_round_trip`'s magic-prefix assertion.
#[cfg(test)]
const HLL_WRAPPER_MAGIC: &[u8; 6] = b"ASAPv1";

/// Map a portable [`HllVariant`] to its ASAPv1 wire `kind_id`. `Unspecified` is
/// a placeholder with no wire form.
fn wire_kind_id(variant: HllVariant) -> Result<&'static [u8], MsgPackError> {
    match variant {
        HllVariant::Regular => Ok(HLL_KIND_CLASSIC),
        HllVariant::Datafusion => Ok(HLL_KIND_ERTL_MLE),
        HllVariant::Hip => Ok(HLL_KIND_HIP),
        HllVariant::Unspecified => Err(MsgPackError::Decode(
            rmp_serde::decode::Error::Uncategorized(
                "ASAPv1 HLL: the Unspecified variant is not serializable".to_string(),
            ),
        )),
    }
}

/// HLL estimator variant. Mirrors `asap_sketchlib::proto::sketchlib::HllVariant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HllVariant {
    Unspecified,
    Regular,
    Datafusion,
    Hip,
}

/// Sparse delta between two consecutive HLL snapshots — input shape for
/// [`HllSketch::apply_delta`]. Mirrors the `HLLDelta` proto in
/// `sketchlib-go/proto/hll/hll.proto`.
#[derive(Debug, Clone, Default)]
pub struct HllSketchDelta {
    pub updates: Vec<(u32, u8)>,
}

/// Minimal HLL state — registers + variant + precision. Register-wise
/// mergeable (max over aligned cells).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HllSketch {
    pub variant: HllVariant,
    pub precision: u32,
    pub registers: Vec<u8>,
    pub hip_kxq0: f64,
    pub hip_kxq1: f64,
    pub hip_est: f64,
}

impl HllSketch {
    pub fn new(variant: HllVariant, precision: u32) -> Self {
        let n = 1usize << precision;
        Self {
            variant,
            precision,
            registers: vec![0u8; n],
            hip_kxq0: 0.0,
            hip_kxq1: 0.0,
            hip_est: 0.0,
        }
    }

    pub fn from_raw(
        variant: HllVariant,
        precision: u32,
        registers: Vec<u8>,
        hip_kxq0: f64,
        hip_kxq1: f64,
        hip_est: f64,
    ) -> Self {
        Self {
            variant,
            precision,
            registers,
            hip_kxq0,
            hip_kxq1,
            hip_est,
        }
    }

    pub fn merge(
        &mut self,
        other: &HllSketch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.variant != other.variant {
            return Err(format!(
                "HllSketch variant mismatch: self={:?}, other={:?}",
                self.variant, other.variant
            )
            .into());
        }
        if self.precision != other.precision {
            return Err(format!(
                "HllSketch precision mismatch: self={}, other={}",
                self.precision, other.precision
            )
            .into());
        }
        if self.registers.len() != other.registers.len() {
            return Err(format!(
                "HllSketch register-length mismatch: self={}, other={}",
                self.registers.len(),
                other.registers.len()
            )
            .into());
        }
        for (s, o) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *o > *s {
                *s = *o;
            }
        }
        if self.variant == HllVariant::Hip {
            self.hip_kxq0 += other.hip_kxq0;
            self.hip_kxq1 += other.hip_kxq1;
            self.hip_est += other.hip_est;
        }
        Ok(())
    }

    pub fn apply_delta(
        &mut self,
        delta: &HllSketchDelta,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let n = self.registers.len();
        for (idx, new_val) in &delta.updates {
            let i = *idx as usize;
            if i >= n {
                return Err(format!(
                    "HllSketchDelta index {i} out of range (precision={} → {n} registers)",
                    self.precision
                )
                .into());
            }
            if *new_val > self.registers[i] {
                self.registers[i] = *new_val;
            }
        }
        Ok(())
    }

    /// Compute a sparse, proto-marshalled `HLLDelta` of `self` against a
    /// `snapshot`. A register update is included when `self`'s register
    /// value increased over the snapshot's (`self[i] > snapshot[i]`); its
    /// carried value is the new (larger) register value.
    ///
    /// This is the Rust twin of the Go reference implementation's
    /// `ComputeRegisterDelta` + `SerializeRegisterDelta`. HLL uses max
    /// semantics, so only increases are meaningful and the delta is
    /// **lossless** — every increased register is carried, regardless of
    /// `threshold` (the parameter is accepted for a uniform delta API and
    /// has no effect here; an HLL register update is never dropped). The
    /// returned bytes are a `prost`-encoded
    /// [`crate::proto::sketchlib::HllDelta`], byte-identical to the Go
    /// `proto.Marshal(HLLDelta)` output for the same inputs (cross-language
    /// byte parity).
    ///
    /// Delta-against-empty: when `snapshot` is the all-zero sketch, every
    /// non-zero register of `self` is carried, so the result is this
    /// window's full register state encoded as a delta (no cross-window
    /// subtraction). HLL deltas carry only register max-updates — there are
    /// no DataPoint-level metric scalars to drop.
    pub fn compute_delta(&self, snapshot: &HllSketch, _threshold: u64) -> Vec<u8> {
        use crate::proto::sketchlib::HllDelta as ProtoDelta;
        use prost::Message;

        let cur = &self.registers;
        let snap = &snapshot.registers;
        let n = cur.len().min(snap.len());

        // Varint-pack the increased registers as (index_delta, value) pairs in
        // ascending index order — the same layout `encode_sparse_registers`
        // uses for the full sparse state. Walking registers in index order
        // yields a sorted sequence, so `prev` only ever advances.
        let mut packed: Vec<u8> = Vec::new();
        let mut prev: usize = 0;
        for i in 0..n {
            if cur[i] > snap[i] {
                write_uvarint(&mut packed, (i - prev) as u64);
                write_uvarint(&mut packed, cur[i] as u64);
                prev = i;
            }
        }
        // Guard: if `self` has more registers than the snapshot (should not
        // happen at a fixed precision), carry all non-zero extras. Matches
        // the Go reference's trailing-register guard.
        for (i, &v) in cur.iter().enumerate().take(cur.len()).skip(n) {
            if v > 0 {
                write_uvarint(&mut packed, (i - prev) as u64);
                write_uvarint(&mut packed, v as u64);
                prev = i;
            }
        }

        ProtoDelta {
            packed_updates: packed,
        }
        .encode_to_vec()
    }

    /// Apply a `prost`-encoded [`crate::proto::sketchlib::HllDelta`] to this
    /// sketch in place (register max-merge). The Rust twin of the Go
    /// reference implementation's `DeserializeRegisterDelta` +
    /// `ApplyRegisterDelta`: each update sets
    /// `register[index] = max(register[index], value)`.
    ///
    /// Returns `Err` if `bytes` is not a valid `HLLDelta` proto or a
    /// register index is out of range for this sketch's precision.
    pub fn apply_delta_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::proto::sketchlib::HllDelta as ProtoDelta;
        use prost::Message;

        let proto = ProtoDelta::decode(bytes)?;

        // Unpack the (index_delta, value) blob — inverse of `compute_delta`,
        // same layout as `decode_sparse_registers` but bounded by the blob's
        // own contents rather than a fixed register count.
        let packed = &proto.packed_updates;
        let mut updates: Vec<(u32, u8)> = Vec::new();
        let mut off = 0usize;
        let mut prev: u32 = 0;
        let mut first = true;
        while off < packed.len() {
            let (delta, used) =
                read_uvarint(&packed[off..]).ok_or("hll: delta index-delta varint corrupt")?;
            off += used;
            let (val, used) =
                read_uvarint(&packed[off..]).ok_or("hll: delta value varint corrupt")?;
            off += used;

            let idx = prev + delta as u32;
            // First delta is an absolute index (prev=0); later deltas must be
            // > 0 so indices stay strictly increasing and unique.
            if !first && delta == 0 {
                return Err(format!("hll: delta non-increasing index at idx {idx}").into());
            }
            first = false;
            if val > 0xff {
                return Err(format!("hll: delta value {val} exceeds u8 at idx {idx}").into());
            }
            updates.push((idx, val as u8));
            prev = idx;
        }

        let delta = HllSketchDelta { updates };
        self.apply_delta(&delta)
    }

    /// MessagePack twin of [`Self::compute_delta`]. Carries every register
    /// that increased over `snapshot` (`self[i] > snapshot[i]`) as an
    /// `(index, new_value)` pair, serialized as the parallel-array
    /// MessagePack layout via [`HllSketchDelta`]'s [`MessagePackCodec`]
    /// instead of proto. HLL uses max semantics, so the delta is lossless
    /// regardless of `_threshold` (accepted for a uniform delta API).
    /// Delta-against-empty carries every non-zero register — this window's
    /// full register state encoded as a delta.
    pub fn compute_delta_msgpack(&self, snapshot: &HllSketch, _threshold: u64) -> Vec<u8> {
        let cur = &self.registers;
        let snap = &snapshot.registers;
        let n = cur.len().min(snap.len());
        let mut updates: Vec<(u32, u8)> = Vec::new();
        for i in 0..n {
            if cur[i] > snap[i] {
                updates.push((i as u32, cur[i]));
            }
        }
        // Guard: registers present in `self` beyond the snapshot length
        // (should not happen at a fixed precision) carry their non-zero
        // values. Matches the proto path's trailing-register guard.
        for (i, &v) in cur.iter().enumerate().skip(n) {
            if v > 0 {
                updates.push((i as u32, v));
            }
        }
        HllSketchDelta { updates }
            .to_msgpack()
            .expect("HllSketchDelta msgpack encode is infallible for owned update arrays")
    }

    /// MessagePack twin of [`Self::apply_delta_bytes`]. Decodes the
    /// parallel-array MessagePack [`HllSketchDelta`] and applies it in
    /// place (register max-merge).
    ///
    /// Returns `Err` if `bytes` is not a valid MessagePack `HllSketchDelta`
    /// or a register index is out of range for this sketch's precision.
    pub fn apply_delta_msgpack_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let delta = HllSketchDelta::from_msgpack(bytes)?;
        self.apply_delta(&delta)
    }

    pub fn merge_refs(
        inputs: &[&HllSketch],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let first = inputs
            .first()
            .ok_or("HllSketch::merge_refs called with empty input")?;
        let mut merged = HllSketch::new(first.variant, first.precision);
        for hll in inputs {
            merged.merge(hll)?;
        }
        Ok(merged)
    }

    /// Insert a value. Math mirrors `sketches::HyperLogLogImpl::insert_with_hash`
    /// (CANONICAL_HASH_SEED, packed bucket/leading-zero); locked in by
    /// `tests/sketches_go_parity_probe.rs`.
    pub fn update(&mut self, value: &[u8]) {
        let hashed_val = hash64_seeded(CANONICAL_HASH_SEED, &DataInput::Bytes(value));
        let p = self.precision as usize;
        let register_bits = (u64::BITS as usize) - p;
        let p_mask: u64 = (1u64 << p) - 1;
        let bucket_num = ((hashed_val >> register_bits) & p_mask) as usize;
        let leading_zero = ((hashed_val << p) + p_mask).leading_zeros() as u8 + 1;
        if bucket_num < self.registers.len() && leading_zero > self.registers[bucket_num] {
            self.registers[bucket_num] = leading_zero;
        }
    }

    /// Estimate cardinality (Classic HLL estimator with the small-range
    /// correction). Ranks are drawn from a 64-bit hash, so the estimate needs
    /// no large-range correction.
    pub fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        if m == 0.0 {
            return 0.0;
        }
        let mut z = 0.0_f64;
        let mut zero_count = 0usize;
        for &reg_val in &self.registers {
            if reg_val == 0 {
                zero_count += 1;
            }
            z += 2f64.powi(-(reg_val as i32));
        }
        let indicator = 1.0 / z;

        let alpha_m = 0.7213 / (1.0 + 1.079 / m);
        let mut est = alpha_m * m * m * indicator;

        if est <= m * 5.0 / 2.0 && zero_count != 0 {
            est = m * (m / zero_count as f64).ln();
        }
        est
    }

    /// Return the proto-enum value Go's `HyperLogLog.SerializePortable`
    /// emits for this sketch's [`HllVariant`].
    #[inline]
    pub fn wire_proto_variant(&self) -> i32 {
        match self.variant {
            HllVariant::Unspecified => 0,
            HllVariant::Regular => 1,
            HllVariant::Datafusion => 2,
            HllVariant::Hip => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Proto dual-read decode (DENSE tag 3 + SPARSE tag 7)
// ---------------------------------------------------------------------------

use crate::proto::sketchlib::{HllSparseRegisters, HyperLogLogState};

/// Decode the full dense register array from a `HyperLogLogState`, accepting
/// BOTH wire encodings emitted by sketchlib-go:
///   - DENSE (`registers`, tag 3): raw 1-byte-per-register array, or
///   - SPARSE (`registers_sparse`, tag 7): HLL++ style varint-packed
///     (index_delta, value) pairs for non-zero registers only.
///
/// The SPARSE field takes priority when present; otherwise the DENSE field is
/// used. Both reconstruct the identical `2^precision`-byte register array, so
/// downstream estimation / merge is encoding-agnostic. This is the Rust half of
/// the backend-decode-first rollout: it must ship before any producer emits
/// sparse. Mirrors the Go decoder in
/// `sketchlib-go/sketches/HLL/portable.go::registersFromState` +
/// `sparse.go::decodeSparseRegisters`.
pub fn registers_from_state(
    state: &HyperLogLogState,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(sp) = &state.registers_sparse {
        return decode_sparse_registers(sp);
    }
    Ok(state.registers.clone())
}

/// Reconstruct the dense register array from a sparse message. Inverse of the
/// Go encoder `sparse.go::encodeSparseRegisters`.
pub fn decode_sparse_registers(
    sp: &HllSparseRegisters,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let n = sp.num_registers as usize;
    let mut regs = vec![0u8; n];

    let packed = &sp.packed;
    let mut off = 0usize;
    let mut prev: usize = 0;
    let mut first = true;
    while off < packed.len() {
        let (delta, used) =
            read_uvarint(&packed[off..]).ok_or("hll: sparse index-delta varint corrupt")?;
        off += used;
        let (val, used) = read_uvarint(&packed[off..]).ok_or("hll: sparse value varint corrupt")?;
        off += used;

        let idx = prev + delta as usize;
        // First delta is an absolute index (prev=0); later deltas must be > 0
        // so indices stay strictly increasing and unique.
        if !first && delta == 0 {
            return Err(format!("hll: sparse non-increasing index at idx {idx}").into());
        }
        first = false;
        if idx >= n {
            return Err(format!("hll: sparse index {idx} out of range [0,{n})").into());
        }
        if val > 0xff {
            return Err(format!("hll: sparse value {val} exceeds u8 at idx {idx}").into());
        }
        regs[idx] = val as u8;
        prev = idx;
    }
    Ok(regs)
}

/// Encode the non-zero registers of `regs` into an [`HllSparseRegisters`]
/// message, byte-identical to the Go encoder `sparse.go::encodeSparseRegisters`.
/// Provided so a Rust producer (or the reverse-direction golden test) can emit
/// the same sparse wire form Go does.
pub fn encode_sparse_registers(regs: &[u8]) -> HllSparseRegisters {
    let mut packed: Vec<u8> = Vec::new();
    let mut prev: usize = 0;
    for (i, &v) in regs.iter().enumerate() {
        if v == 0 {
            continue;
        }
        write_uvarint(&mut packed, (i - prev) as u64);
        write_uvarint(&mut packed, v as u64);
        prev = i;
    }
    HllSparseRegisters {
        num_registers: regs.len() as u32,
        packed,
    }
}

/// Append one unsigned LEB128 varint to `out`. Matches Go's
/// `encoding/binary.PutUvarint`.
fn write_uvarint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Read one unsigned LEB128 varint, returning (value, bytes_consumed). Matches
/// Go's `encoding/binary.Uvarint`. Returns None on truncation / overflow.
fn read_uvarint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    for (i, &b) in buf.iter().enumerate() {
        if i >= 10 {
            return None; // varint too long for u64
        }
        if shift == 63 && b > 1 {
            return None; // overflow
        }
        result |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            return Some((result, i + 1));
        }
        shift += 7;
    }
    None
}

impl MessagePackCodec for HllSketch {
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let kind_id = wire_kind_id(self.variant)?;
        let metadata = rmp_serde::to_vec_named(&standard_hll_metadata(self.precision))?;
        let payload = if self.variant == HllVariant::Hip {
            rmp_serde::to_vec(&HllPayloadHip {
                registers: self.registers.clone(),
                hip_kxq0: self.hip_kxq0,
                hip_kxq1: self.hip_kxq1,
                hip_est: self.hip_est,
            })?
        } else {
            rmp_serde::to_vec(&HllPayloadPlain {
                registers: self.registers.clone(),
            })?
        };
        Ok(envelope::encode(kind_id, &metadata, &payload))
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        let (kind_id, metadata, payload) = envelope::split(bytes)
            .map_err(|msg| MsgPackError::Decode(rmp_serde::decode::Error::Uncategorized(msg)))?;

        // Validate the hash spec is the standard profile (precision is read from
        // the metadata, so compare against a standard block built with it).
        let meta: HllMetadata = rmp_serde::from_slice(metadata)?;
        if meta != standard_hll_metadata(meta.precision) {
            return Err(MsgPackError::Decode(
                rmp_serde::decode::Error::Uncategorized(
                    "ASAPv1 HLL envelope: hash metadata mismatch".to_string(),
                ),
            ));
        }

        let variant = if kind_id == HLL_KIND_CLASSIC {
            HllVariant::Regular
        } else if kind_id == HLL_KIND_ERTL_MLE {
            HllVariant::Datafusion
        } else if kind_id == HLL_KIND_HIP {
            HllVariant::Hip
        } else {
            return Err(MsgPackError::Decode(
                rmp_serde::decode::Error::Uncategorized(format!(
                    "ASAPv1 HLL envelope: unsupported kind_id {kind_id:?}"
                )),
            ));
        };

        let (registers, hip_kxq0, hip_kxq1, hip_est) = if variant == HllVariant::Hip {
            let p: HllPayloadHip = rmp_serde::from_slice(payload)?;
            (p.registers, p.hip_kxq0, p.hip_kxq1, p.hip_est)
        } else {
            let p: HllPayloadPlain = rmp_serde::from_slice(payload)?;
            (p.registers, 0.0, 0.0, 0.0)
        };

        // Fail closed if the register bin does not match `2^precision`
        // (doc §2 validation rule 3). `checked_shl` avoids a shift-overflow
        // panic on a crafted out-of-range `precision`.
        let expected = 1usize.checked_shl(meta.precision).ok_or_else(|| {
            MsgPackError::Decode(rmp_serde::decode::Error::Uncategorized(format!(
                "ASAPv1 HLL envelope: precision {} out of range",
                meta.precision
            )))
        })?;
        if registers.len() != expected {
            return Err(MsgPackError::Decode(
                rmp_serde::decode::Error::Uncategorized(format!(
                    "ASAPv1 HLL envelope: {} registers, expected 2^{} = {expected}",
                    registers.len(),
                    meta.precision
                )),
            ));
        }

        Ok(HllSketch::from_raw(
            variant,
            meta.precision,
            registers,
            hip_kxq0,
            hip_kxq1,
            hip_est,
        ))
    }
}

impl MessagePackCodec for HllSketchDelta {
    /// Encodes the register delta as the 2-element MessagePack array
    /// `[ reg_idx:[]u32, reg_val:[]u8 ]` — parallel arrays of register
    /// index → new (max) value — byte-identical to Rust
    /// `rmp_serde::to_vec(&(Vec<u32>, Vec<u8>))` (compact mode). The
    /// `Vec<u8>` serializes as a msgpack array-of-int (not `bin`), matching
    /// the Go `asapmsgpack` register-array convention.
    fn to_msgpack(&self) -> Result<Vec<u8>, MsgPackError> {
        let reg_idx: Vec<u32> = self.updates.iter().map(|(i, _)| *i).collect();
        let reg_val: Vec<u8> = self.updates.iter().map(|(_, v)| *v).collect();
        Ok(rmp_serde::to_vec(&(reg_idx, reg_val))?)
    }

    fn from_msgpack(bytes: &[u8]) -> Result<Self, MsgPackError> {
        let (reg_idx, reg_val): (Vec<u32>, Vec<u8>) = rmp_serde::from_slice(bytes)?;
        Ok(HllSketchDelta {
            updates: reg_idx.into_iter().zip(reg_val).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranks are drawn from a 64-bit hash, so a register state whose estimate
    /// runs past the 32-bit range must still follow the HyperLogLog formula
    /// rather than saturate or turn into a NaN.
    #[test]
    fn estimate_holds_past_the_32_bit_range() {
        for rank in [12u8, 16, 18, 20] {
            let mut sketch = HllSketch::new(HllVariant::Regular, 14);
            sketch.registers.fill(rank);
            let m = sketch.registers.len() as f64;
            let alpha_m = 0.7213 / (1.0 + 1.079 / m);
            let expected = alpha_m * m * 2f64.powi(rank as i32);
            let estimate = sketch.estimate();
            let error = (estimate - expected).abs() / expected;
            assert!(
                error <= 1e-9,
                "rank {rank} estimate {estimate} deviates from {expected} by {error:.6}"
            );
        }
    }

    #[test]
    fn msgpack_delta_against_empty_round_trips() {
        let mut w = HllSketch::new(HllVariant::Regular, 12);
        for i in 0..2000u64 {
            w.update(&i.to_le_bytes());
        }
        let empty = HllSketch::new(HllVariant::Regular, 12);
        let bytes = w.compute_delta_msgpack(&empty, 1);
        let mut recon = HllSketch::new(HllVariant::Regular, 12);
        recon.apply_delta_msgpack_bytes(&bytes).unwrap();
        // Register delta over an empty base carries every non-zero register,
        // so reconstructed registers are identical.
        assert_eq!(recon.registers, w.registers);
        let (got, want) = (recon.estimate(), w.estimate());
        assert!((got - want).abs() <= want * 0.001, "got={got} want={want}");
    }

    #[test]
    fn msgpack_delta_wire_layout_is_parallel_arrays() {
        use crate::message_pack_format::MessagePackCodec;
        let mut w = HllSketch::new(HllVariant::Regular, 4);
        w.update(b"a");
        w.update(b"b");
        let empty = HllSketch::new(HllVariant::Regular, 4);
        let bytes = w.compute_delta_msgpack(&empty, 1);
        let mut reg_idx: Vec<u32> = Vec::new();
        let mut reg_val: Vec<u8> = Vec::new();
        for (i, &v) in w.registers.iter().enumerate() {
            if v > 0 {
                reg_idx.push(i as u32);
                reg_val.push(v);
            }
        }
        let expected = rmp_serde::to_vec(&(reg_idx, reg_val)).unwrap();
        assert_eq!(bytes, expected);
        let d = HllSketchDelta::from_msgpack(&bytes).unwrap();
        assert_eq!(
            d.updates.len(),
            w.registers.iter().filter(|&&v| v > 0).count()
        );
    }

    #[test]
    fn test_new_empty() {
        let h = HllSketch::new(HllVariant::Regular, 4);
        assert_eq!(h.registers.len(), 16);
        assert!(h.registers.iter().all(|&r| r == 0));
    }

    #[test]
    fn test_merge_register_wise_max() {
        let mut a = HllSketch::from_raw(HllVariant::Regular, 2, vec![1, 5, 3, 7], 0.0, 0.0, 0.0);
        let b = HllSketch::from_raw(HllVariant::Regular, 2, vec![4, 2, 6, 0], 0.0, 0.0, 0.0);
        a.merge(&b).unwrap();
        assert_eq!(a.registers, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_apply_delta_max_semantics() {
        let mut h = HllSketch::from_raw(HllVariant::Regular, 2, vec![1, 5, 3, 7], 0.0, 0.0, 0.0);
        let delta = HllSketchDelta {
            updates: vec![(0, 4), (1, 2), (2, 6), (3, 0)],
        };
        h.apply_delta(&delta).unwrap();
        assert_eq!(h.registers, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_apply_delta_out_of_range() {
        let mut h = HllSketch::new(HllVariant::Regular, 2);
        let delta = HllSketchDelta {
            updates: vec![(7, 3)],
        };
        assert!(h.apply_delta(&delta).is_err());
    }

    #[test]
    fn test_apply_delta_matches_full_merge() {
        let base = HllSketch::from_raw(HllVariant::Regular, 2, vec![1, 5, 3, 7], 0.0, 0.0, 0.0);
        let addition = HllSketch::from_raw(HllVariant::Regular, 2, vec![4, 0, 6, 0], 0.0, 0.0, 0.0);
        let mut via_merge = base.clone();
        via_merge.merge(&addition).unwrap();

        let delta = HllSketchDelta {
            updates: vec![(0, 4), (2, 6)],
        };
        let mut via_delta = base;
        via_delta.apply_delta(&delta).unwrap();
        assert_eq!(via_delta.registers, via_merge.registers);
    }

    #[test]
    fn test_merge_variant_mismatch() {
        let mut a = HllSketch::new(HllVariant::Regular, 4);
        let b = HllSketch::new(HllVariant::Datafusion, 4);
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn test_merge_precision_mismatch() {
        let mut a = HllSketch::new(HllVariant::Regular, 4);
        let b = HllSketch::new(HllVariant::Regular, 5);
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn test_merge_refs() {
        let a = HllSketch::from_raw(HllVariant::Regular, 1, vec![1, 0], 0.0, 0.0, 0.0);
        let b = HllSketch::from_raw(HllVariant::Regular, 1, vec![0, 3], 0.0, 0.0, 0.0);
        let c = HllSketch::from_raw(HllVariant::Regular, 1, vec![2, 2], 0.0, 0.0, 0.0);
        let merged = HllSketch::merge_refs(&[&a, &b, &c]).unwrap();
        assert_eq!(merged.registers, vec![2, 3]);
    }

    #[test]
    fn test_update_then_estimate_within_2pct() {
        let n: usize = 50_000;
        let mut h = HllSketch::new(HllVariant::Regular, 12);
        for i in 0..n {
            let key = format!("key-{i}");
            h.update(key.as_bytes());
        }
        let est = h.estimate();
        let rel_err = (est - n as f64).abs() / n as f64;
        assert!(
            rel_err < 0.02,
            "HLL estimate {est} not within 2% of {n} (rel_err {rel_err:.4})",
        );
    }

    #[test]
    fn test_estimate_empty_is_zero() {
        let h = HllSketch::new(HllVariant::Regular, 4);
        assert_eq!(h.estimate(), 0.0);
    }

    #[test]
    fn test_msgpack_round_trip() {
        let original = HllSketch::from_raw(
            HllVariant::Hip,
            3,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            1.0,
            2.0,
            3.0,
        );
        let bytes = original.to_msgpack().unwrap();
        assert!(bytes.starts_with(HLL_WRAPPER_MAGIC));
        let decoded = HllSketch::from_msgpack(&bytes).unwrap();
        assert_eq!(decoded.registers, original.registers);
        assert_eq!(decoded.precision, original.precision);
        assert_eq!(decoded.hip_kxq0, 1.0);
    }

    /// Cross-language byte-parity guard against
    /// `sketchlib-go::HyperLogLog.SerializePortable` for the
    /// deterministic input stream `(1..=50i32).map(|i| (i as f64)
    /// .to_le_bytes())`.
    #[test]
    fn test_update_then_envelope_matches_sketchlib_go_bytes() {
        use crate::proto::sketchlib::{
            HyperLogLogState, SketchEnvelope, sketch_envelope::SketchState,
        };
        use prost::Message;

        let mut sk = HllSketch::new(HllVariant::Datafusion, 14);
        for i in 1..=50i32 {
            let v = i as f64;
            sk.update(&v.to_le_bytes());
        }

        let state = HyperLogLogState {
            variant: sk.wire_proto_variant(),
            precision: sk.precision,
            registers: sk.registers.clone(),
            hip_kxq0: sk.hip_kxq0,
            hip_kxq1: sk.hip_kxq1,
            hip_est: sk.hip_est,
            // This parity golden predates the sparse field and intentionally
            // pins the DENSE wire form, so leave the sparse field unset.
            registers_sparse: None,
        };
        let envelope = SketchEnvelope {
            format_version: 1,
            producer: None,
            hash_spec: None,
            // Golden pins the pre-sampling wire form; 0.0 is the proto3 default
            // (dual-read as 1.0) so the encoded bytes are unchanged.
            sample_p: 0.0,
            sketch_state: Some(SketchState::Hll(state)),
        };
        let mut got = Vec::with_capacity(envelope.encoded_len());
        envelope.encode(&mut got).expect("prost encode");

        const GOLDEN_HEX: &str = include_str!("../../sketches/testdata/hll_envelope_golden.hex");
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got.len(),
            want.len(),
            "HLL envelope length differs: got {} bytes, want {} bytes",
            got.len(),
            want.len(),
        );
        assert_eq!(
            got, want,
            "HLL envelope bytes diverge from sketchlib-go golden"
        );
    }

    fn decode_hex(s: &str) -> Vec<u8> {
        let s = s.trim();
        s.as_bytes()
            .chunks(2)
            .map(|pair| {
                let high = hex_nibble(pair[0]);
                let low = hex_nibble(pair[1]);
                (high << 4) | low
            })
            .collect()
    }

    fn hex_nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("non-hex byte {}", c as char),
        }
    }

    // ----- sparse full-state encoding -----

    use crate::proto::sketchlib::HyperLogLogState;

    /// A handful of (index, value) pairs that exercise small and large index
    /// deltas plus the maximum register value.
    fn sample_regs(n: usize) -> Vec<u8> {
        let mut regs = vec![0u8; n];
        regs[0] = 1; // first delta is absolute index 0
        regs[1] = 7; // adjacent index (delta 1)
        regs[300] = 51; // max p=14 register value (Q+1)
        regs[8191] = 12; // mid-range, multi-byte index delta
        regs[n - 1] = 3; // last register
        regs
    }

    #[test]
    fn sparse_round_trip_exact() {
        let regs = sample_regs(16384);
        let sp = encode_sparse_registers(&regs);
        assert_eq!(sp.num_registers, 16384);
        let decoded = decode_sparse_registers(&sp).unwrap();
        assert_eq!(decoded, regs, "sparse round trip must be exact");
    }

    #[test]
    fn registers_from_state_reads_both() {
        let regs = sample_regs(16384);

        // SPARSE: registers_sparse set, registers empty.
        let sparse_state = HyperLogLogState {
            variant: 2,
            precision: 14,
            registers: Vec::new(),
            hip_kxq0: 0.0,
            hip_kxq1: 0.0,
            hip_est: 0.0,
            registers_sparse: Some(encode_sparse_registers(&regs)),
        };
        assert_eq!(registers_from_state(&sparse_state).unwrap(), regs);

        // DENSE: registers set, registers_sparse None.
        let dense_state = HyperLogLogState {
            variant: 2,
            precision: 14,
            registers: regs.clone(),
            hip_kxq0: 0.0,
            hip_kxq1: 0.0,
            hip_est: 0.0,
            registers_sparse: None,
        };
        assert_eq!(registers_from_state(&dense_state).unwrap(), regs);
    }

    #[test]
    fn sparse_empty_is_all_zero() {
        let regs = vec![0u8; 16384];
        let sp = encode_sparse_registers(&regs);
        assert!(sp.packed.is_empty());
        assert_eq!(decode_sparse_registers(&sp).unwrap(), regs);
    }

    #[test]
    fn sparse_rejects_out_of_range_index() {
        // packed = uvarint(delta=5) , uvarint(value=3) but num_registers=4.
        let sp = HllSparseRegisters {
            num_registers: 4,
            packed: vec![5, 3],
        };
        assert!(decode_sparse_registers(&sp).is_err());
    }

    /// `compute_delta` against an EMPTY snapshot reconstructs the window's
    /// full register state when its bytes are applied to a fresh empty
    /// sketch (register max-merge round-trip). HLL deltas are lossless —
    /// every increased register is carried.
    #[test]
    fn test_compute_delta_against_empty_round_trips() {
        let mut window = HllSketch::new(HllVariant::Datafusion, 14);
        for i in 0..5000u64 {
            window.update(&i.to_le_bytes());
        }
        let empty = HllSketch::new(HllVariant::Datafusion, 14);

        let delta_bytes = window.compute_delta(&empty, 0);

        let mut reconstructed = HllSketch::new(HllVariant::Datafusion, 14);
        reconstructed.apply_delta_bytes(&delta_bytes).unwrap();

        assert_eq!(reconstructed.registers, window.registers);
    }

    /// A delta computed between two non-empty snapshots reconstructs the
    /// current sketch when applied to the base (max-merge of the registers
    /// that increased).
    #[test]
    fn test_compute_delta_then_apply_matches_current() {
        let mut base = HllSketch::new(HllVariant::Datafusion, 12);
        for i in 0..2000u64 {
            base.update(&i.to_le_bytes());
        }
        let mut current = base.clone();
        for i in 2000..5000u64 {
            current.update(&i.to_le_bytes());
        }

        let delta_bytes = current.compute_delta(&base, 0);
        let mut reconstructed = base.clone();
        reconstructed.apply_delta_bytes(&delta_bytes).unwrap();
        assert_eq!(reconstructed.registers, current.registers);
    }

    /// Cross-language byte-parity guard: `compute_delta` against an empty
    /// snapshot must emit bytes identical to the Go reference
    /// implementation's `SerializeRegisterDelta(ComputeRegisterDelta(empty,
    /// current))` for the same precision-14 sketch fed `(1..=50)` as f64
    /// little-endian byte values. A delta-against-empty carries every
    /// non-zero register as an `(index, value)` update; the golden hex was
    /// captured from a `proto.Marshal` of the Go reference's `HLLDelta`.
    #[test]
    fn test_compute_delta_matches_go_golden_bytes() {
        let mut current = HllSketch::new(HllVariant::Datafusion, 14);
        for i in 1..=50i32 {
            let v = i as f64;
            current.update(&v.to_le_bytes());
        }
        let empty = HllSketch::new(HllVariant::Datafusion, 14);
        let got = current.compute_delta(&empty, 0);

        // Captured from the Go reference implementation's
        // SerializeRegisterDelta(ComputeRegisterDelta(empty, current)) for the
        // same input, with the packed (index_delta, value) HLLDelta encoding.
        const GOLDEN_HEX: &str = "0a81015801bb0402800c015001970401dd0301b10101b2080132016a01a901022f018402020902a10a029a03027501c7020428042b05ec0702a60102850205c2030258021101fe030128022002a60403cd0102e7060211060b0334020b069c0101ff06012a015e0581020170041201f80401a70404a60101bc03019e0103cc0201880203";
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "HLL delta bytes diverge from the Go reference golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );
    }

    /// Cross-language byte-parity guard for a SUB-WINDOW delta (not against
    /// empty): a precision-14 sketch is fed `(0..2000)` as f64 little-endian
    /// bytes (the base/snapshot), then `(2000..5000)`; the delta carries only
    /// the registers that grew between the two, packed as (index_delta, value)
    /// pairs. Must be byte-identical to the Go reference implementation's
    /// `SerializeRegisterDelta(ComputeRegisterDelta(base, current))`.
    #[test]
    fn test_compute_subwindow_delta_matches_go_golden_bytes() {
        let mut base = HllSketch::new(HllVariant::Datafusion, 14);
        for i in 0..2000i32 {
            base.update(&(i as f64).to_le_bytes());
        }
        let mut current = base.clone();
        for i in 2000..5000i32 {
            current.update(&(i as f64).to_le_bytes());
        }
        let got = current.compute_delta(&base, 0);

        // Captured from the Go reference implementation's
        // SerializeRegisterDelta(ComputeRegisterDelta(base, current)) for the
        // same input, with the packed (index_delta, value) HLLDelta encoding.
        const GOLDEN_HEX: &str = "0ae627000402030401020302030c0109020502040201011108100202021304030101010b010e02040102030b0407020201050405021404060105010301010101020101030306020201040102011101010407010106080104030101010103020c0204021105040108010101180201030f010b01010103010e01120201030101010303030301030107030a0108010a0108012106060309010302070203010202030101010e0102020a020102060101020a01040101011702020203021003060103030e02110206010102040604010d010a050c011104010104050a0103020f0203010305050209010301020401020805010101010201020104010201060501040601020203020601020304010401020101010601010419010e010503050107020302010102010d010501110105050503020104050201020215050401030101020d041a010601190209020301020404070e01090202010b01030107040a0103010403010117050602040601010301040101030203030105010401020108030b0517020c01010506010402030103040704010102020d010a011201060304020c010c020e010b020702070101040101010104021504070109040804060502021802030104021605010107020603010101010201100104010605020205010e010406040102020f0203020a02040201030201010303010401030302010a030201060514020d02050104010201050103020802040105020602080213020d02020105020301140204010a0509060d03070206020902070101080a0202010a010e0102020503080201010e0101010403080104020d0102010501040403060401050405011203010101040e010202040303020c010203090201030d010a0306020201030208020e0103010701030113020202030501020c010b0101040502010105010203060105020a0315050b030c0101010f060405040204030b03010502020101080302020c010b02060403010404030102010f020f0107030601040109060301040301010703040303020801010104010b011101070102011101010102030201070106010601030201020301030302020e030501190108030201040205020503030106030a01050112020102010107030b02060209050201010304010502030106060601050108020602080201010905080205010204010203020303060205010102070201010308160304040f02010105020605070407030b0104010701010205020c02060103010b040c0102030202060102010e030102070108030301030205010101040301030303010203020802190105020601120101020701010101010302080303010f0104020202060402020403010103010503060203030701010105030903030508010101020103020305060605030e0203020603010102060f010201040202030504030103020f01130204010201020104010102010401010403050201030502040306060d0302010a010a02080103020101020208030301030102010403030305011003030206010f01010609030302020102030402020102010103020408010502130306010303060102010c01030202030401030101020101070202010901150104010f010401080107010601010206020b0201010d040303070202011403090210010302090403010403050201010b05010401010301090102010c010b04040206010d0102010b05040301020702020302010d0204030102110106010803020108030101070104020104040201030605050102040101010302010b01020103030e031101030101020103070112010403010101010b011001060313030301070203030803050101030601050207030a0104030302020202030c010101080103010c0102020404020102030c01110102030901140307010602020303020401070102010b020402030707010101010201070705130102011b0201060501010101010201020109010601020205010b0103040504070202010501080108020703030104030401010201030a010201030206010101020104020d02040102010102040301040605100201020801030306020e010402110106030602080105011101060106020d0216010702050301020101040409010403010201040102020205010401070207010d0205040704050202021201030510040303020602020703090204060102070102010301020101010401200101030202040103021301010107050a010702030311010303020103020b010505100302060102010104020a010402070210010704030304050a02010409010101010106030901030110031002020103010d0301020202040101020806080105010401060126020201080109020e0102030503120106010901010301010b01040201020402020115020d0108030f010c0303020c0101010602080504030802050202030f030202050404010603030208010a0204010d02020202030a0209030e030304030101060d0104020702010209020a0406010101070204020b0201030d021201020106031a0504050202040102010101020103030601080102020f021a020201120112020d02030405020403090304020102010108010203080101010704060104010902010401020203040107010e02070102060601080105010101060204010e010c020d010601050108010801020508010102070302030201030101010b020b02060107010c020b01020104020302060203040c02020105010301030202020902090103030503070606010103040216020401050205020404020208010b0207010502050108010502030102040f0206021203050105010a010a010b04090101030401040c03010102040104020c04090203020301040202020101060205020d010303040202020b010c010f0205030d010301050202030a02030309020203110103010102090106020402010202010303020206010a0104010601030208020803080110020702030205050201130102022102060204021504040102010a040c0401010503030207010701050103010d0104040601100201010201020302010a010501110102080504010106030101060601040a01050202080203020509010c010102090203010d0101020102120713021104070204010201010208020502060408010309020101020d0104010206030102010201010202050202040101010102050102030201060209020b01020105040202060401020f010103090504030101030111020e031001110204020602010201011101050204010a01040402030501010202010602070503011002020406010401030103020f0403050e0104020201010101040b0105030103010106020803080201010101010202050404010103011c0306030201010105010202060206010102020101010703010502010c010303020206030c021601130121020f03020307030e0109010602040102011f01090506010b01050402020e01020303020a040201020202010301030204030a010201080404030304110104010403070203060301020107020c050502020109030901020102010801030203090d0104021502010104040504090105030a010503110102020a010203050303040e030502030101030105020108010103020102030103010209030202010205010301010202020804010207010d020d020a010605080306010c0202021101090202030f010101040109020f0301020501010103020501040109021202080601070f0204020105070119030101110208040a0203020c010201070503030b0108030201040204020402010203010c022603040505010e04010103010702010109050b020801020405020e01140204020201050406020a020301060404010c010102050106020207010102010801050202020105040103010108010113050d0109020703070205010f0703050b0112010f02010103070b010201050101030201010203010705090605020f0103040b010801020210020501060204040c010104070105020a020104040201010403040109010c0101050a0305010301090108010b030b0201010601030202030e020203030101050202010106010d010d010601020101010c0304010901010103010401030201010902010101050b01010107010105020105020c01030405011404020104030a01030105010801050202020a0107010c04150105030301050401020603020203040b01100101010301030103070101010215020302010101010101080203020a010a021d031302030104020201030305030b01030203030b03010212020302030202050d010703040202040501040101030203060405031802010206020402020109020202070305021c01010105031c010503050305030d010e01090105030401070104010302010103010201100103020b0107050301090104010203030205030103010105010504080104020302020111050601050404022202070102010201090705080b01030109012903090206030104040107010102070201040f020201030103030b020804020207012d0105010301080103020201030202050501070105010202010304020303020401030802020103020102010501010c0106021201050317010a0304010101020207090102030205010b0204010b010801080312010f01150102020802040102011102080101040a0102010c020403020203010603050203010301040111031505040202030e01010102050607050101010502020301010c0210010d0407010903020202010e01050106020401070402010c010602090311060202100206020202020316020103010101010a030301060101010203150201050e04030204021c020c020c030102030101010502060102010d02010411040c0208010101010101040201010205050101060214010901010407020206160101050407050104010101010205050501060103010101060411020302070211010201090203021a0602010f0106030a04010106010301090101070501020107060102010404020b0102010202040110010201050106020901090309020d0105020601080101010501010205030403180401030d0105040802240209010a02020305020502040102010301090303020802040108010405020201040d0403010501020102020503040215010d030902030116010f010d0107030204070204040a0109011002090119031e020502090104031b050605020201010c0207020503010104010c0102010a01060502010c020d020202050101011102050107040f0102010201010106020305020101010402010302010c0306051505010104020802010102070c010803010506011101050207020302040103010301080103080d030c0103040d01010209040501030101010402020301010f01020102010301010401010202050408011c050904080106040401040105010402120102020103030201010a030103020402021303020305020403030201030901010504010902010305010301030203020a0108010702050107020a01160205020a060201030106030e02030103031001070102030d0119020202070505021202030108010101060808010202050108010f0102010701030404020802020214030c0211010b020301080303020105080105010401100106011309040203010103010208030b020102010301010e0107021501080203010402010101010606030204010d0108010e0102010401010105030903020302030c0103010104080101010405030201020c020101070201020102010401010301080102041f01070110030101060107010201060108010d010a030a0114041003020104010302020115011a02070802010301010106020b0101010201050109020703020205010a05050402030b0108010901070201020101020105040401110101020101040103010301010107010201180101020a030201030211011a02110103010102050103040e02160106010b02050a0e0202010203030101010402120305010301030305010805070304010402050102020402040106040105060105010b01110104010c0202020b01020104020e0201020704070206010c02030105010601030106011303070103010c0601010e021a01010704020801020201030c0506010a010203010202030e0311010201050105010b010504040203040a0102010302030302010102090301010403010104010e010501050a04070201060304030b0203010e01030201011b0102020c010701040303020d0604011001130202021304020505020201010105010a010f0106020801030113010802020104040b020104080104010e03090205020905020205010e040306020304010204100106010101150103050501050101010c0102010401050201030203010315010f0111020e0403010702030203030c0101020d02030205030a01040605050a02020101011201060502020201020101010a030f010706040105010101080502020301010608040b050101010104010d0103032001030201010f01040103011003010102060401070107010f01030116030301030205030902020611030102010101010101010201020202050403011b0208010b02030202010e0108010b010e01010508040701010103050604040515020202030104020703060203010601060106020f0101020302040102010c0302010101030105010a03020311020301030201010d03040107020502010109030b0302090801070304010301010402020501010201010501050202040302030305010101040201020802020101020f0107010b020302070313010b06050201030c010b04010106050302030102030401090208010901050307020d030b0309030e01090106010302020202010501040708010e0102010401010101020304010303010505050104020201070107020101040102010e0104011a0105050a030d030f010a02130103030b0107050e0203040403010208030d0701030101080206010a010e0203020501070303011102030203010101050117010e040d040701180201010e0201010402090204012c031c010e0204020502080113010201";
        let want = decode_hex(GOLDEN_HEX);
        assert_eq!(
            got,
            want,
            "HLL sub-window delta bytes diverge from the Go reference golden \
             ({} bytes got vs {} bytes want)",
            got.len(),
            want.len(),
        );

        // Sanity: applying the packed sub-window delta to the base
        // reconstructs the current register state (register max-merge).
        let mut reconstructed = base.clone();
        reconstructed.apply_delta_bytes(&got).unwrap();
        assert_eq!(reconstructed.registers, current.registers);
    }
}
