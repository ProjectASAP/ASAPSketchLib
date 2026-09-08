//! ASAPv1 wire serialization for [`Hydra`].
//!
//! Child submodule of [`crate::sketch_framework::hydra`]: it holds ALL of
//! Hydra's serialization (the metadata/payload DTOs, the five kind_id
//! constants, and the `serialize_to_bytes` / `deserialize_from_bytes` impls)
//! while the algorithm lives in the parent module file. Being a descendant
//! module, it reads the private `schema` field directly without widening any
//! field visibility. See `docs/asapv1_wire_format.md`.
//!
//! ## One kind_id per counter variant
//!
//! `0x07 0x00` KLL, `0x07 0x01` Count-Min, `0x07 0x02` Count Sketch,
//! `0x07 0x03` HyperLogLog, `0x07 0x04` UnivMon. The kind_id names the counter,
//! so the payload carries no per-cell variant tag and a grid mixing variants
//! has no encoding.
//!
//! ## Counters are inlined, not nested
//!
//! Each cell's raw state goes straight into Hydra's positional array in the
//! shape that counter's own spec section fixes (§3.1 `[registers]`, §3.2 /
//! §3.6 `[counts]`, §3.3 `[levels, items, coin]`, UnivMon's pyramid array). No
//! cell carries an envelope, a magic or a metadata map of its own.
//!
//! ## The grid, the schema and the counter geometry live in the metadata
//!
//! Every cell is a clone of `type_to_clone`, so the counter geometry is carried
//! once rather than per cell, and the prototype itself reaches the wire only
//! through that geometry.
//!
//! ## Hashing
//!
//! Hydra places subkeys with the free-function matrix hash at
//! [`crate::HYDRA_SEED`], so its metadata carries [`DefaultXxHasher`]'s hash
//! spec. That index is fixed by the algorithm rather than by the profile, so no
//! seed-index key names it.
//!
//! ## Emitted order (cross-language contract)
//!
//! Cells are emitted row-major over the grid and, within a cell, in that
//! counter's own payload order, so a decoded Hydra re-serializes
//! byte-identically.

use rmp_serde::{decode::Error as RmpDecodeError, encode::Error as RmpEncodeError, from_slice};
use serde::de::IgnoredAny;
use serde::{Deserialize, Serialize};

use crate::common::hash::check_matrix_rows;
use crate::input::HydraCounter;
use crate::message_pack_format::envelope;
use crate::sketch_framework::univmon::wire::{PyramidPayload, UnivMonMetadata, univmon_metadata};
use crate::{
    Count, CountMin, DefaultXxHasher, ErtlMLE, FastPath, HashProfile, HllBucketListP14,
    HyperLogLog, KLL, UnivMon, Vector2D,
};

use super::{Hydra, KeySchema};

/// Hydra kind_ids: family `0x07`, one variant per counter.
const HYDRA_KIND_KLL: &[u8] = &[0x07, 0x00];
const HYDRA_KIND_CM: &[u8] = &[0x07, 0x01];
const HYDRA_KIND_CS: &[u8] = &[0x07, 0x02];
const HYDRA_KIND_HLL: &[u8] = &[0x07, 0x03];
const HYDRA_KIND_UNIVMON: &[u8] = &[0x07, 0x04];

/// Metadata `counter_type`: both matrix counters are `Vector2D<i32>`, carried
/// at their own width.
const COUNTER_TYPE: &str = "i32";

/// Metadata `counter_mode`: both matrix counters take the fast path.
const COUNTER_MODE: &str = "fast";

/// Metadata `counter_item_type`: the KLL counter retains `f64` samples.
const COUNTER_ITEM_TYPE: &str = "f64";

/// The counters' own kind_ids. A cell is rebuilt by handing its inlined state
/// to that counter's decoder; neither id reaches the Hydra wire.
const HLL_CELL_KIND: &[u8] = &[0x01, 0x02];
const KLL_CELL_KIND: &[u8] = &[0x06, 0x00];
const UNIVMON_CELL_KIND: &[u8] = &[0x10, 0x00];

/// The grid geometry and key columns every Hydra metadata carries.
struct HydraGrid {
    rows: u32,
    cols: u32,
    schema: Vec<String>,
}

/// Hydra descriptor metadata for the two matrix counters (`0x07 0x01`
/// Count-Min, `0x07 0x02` Count Sketch), a msgpack **map** (`to_vec_named`)
/// with keys in this declaration order — the canonical order the wire spec
/// fixes (Go must mirror it).
///
/// Hash-spec fields first, then the grid (`rows`, `cols`), the key columns, and
/// the counter's own structural params. The two kind_ids share this schema the
/// way the two KLL kind_ids share one.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HydraMatrixMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    matrix_seed_index: u32,
    rows: u32,
    cols: u32,
    schema: Vec<String>,
    counter_rows: u32,
    counter_cols: u32,
    counter_type: String,
    counter_mode: String,
}

/// Hydra descriptor metadata for the HyperLogLog counter (`0x07 0x03`).
/// `counter_precision` fixes the register run each cell contributes.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HydraHllMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    canonical_seed_index: u32,
    rows: u32,
    cols: u32,
    schema: Vec<String>,
    counter_precision: u32,
}

/// Hydra descriptor metadata for the KLL counter (`0x07 0x00`).
///
/// The hash-spec group is Hydra's own: the grid hashes its subkeys even though
/// the counters never hash. It carries no seed index, since neither Hydra nor
/// KLL reads one from the profile.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HydraKllMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    rows: u32,
    cols: u32,
    schema: Vec<String>,
    counter_k: u32,
    counter_m: u32,
    counter_item_type: String,
}

/// Hydra descriptor metadata for the UnivMon counter (`0x07 0x04`). The four
/// pyramid dimensions and the heaps' `counter_key_type` are shared by every
/// cell.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HydraUnivMonMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    rows: u32,
    cols: u32,
    schema: Vec<String>,
    counter_layer_size: u32,
    counter_sketch_row: u32,
    counter_sketch_col: u32,
    counter_heap_size: u32,
    counter_key_type: String,
}

/// Builds the matrix-counter metadata from the hasher's [`HashProfile`], so the
/// wire bytes truthfully describe how the grid was hashed.
fn hydra_matrix_metadata<H: HashProfile>(
    grid: &HydraGrid,
    counter_rows: u32,
    counter_cols: u32,
) -> HydraMatrixMetadata {
    HydraMatrixMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        matrix_seed_index: H::MATRIX_SEED_INDEX,
        rows: grid.rows,
        cols: grid.cols,
        schema: grid.schema.clone(),
        counter_rows,
        counter_cols,
        counter_type: COUNTER_TYPE.to_string(),
        counter_mode: COUNTER_MODE.to_string(),
    }
}

/// Builds the HyperLogLog-counter metadata from the hasher's [`HashProfile`].
fn hydra_hll_metadata<H: HashProfile>(
    grid: &HydraGrid,
    counter_precision: u32,
) -> HydraHllMetadata {
    HydraHllMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        canonical_seed_index: H::CANONICAL_SEED_INDEX,
        rows: grid.rows,
        cols: grid.cols,
        schema: grid.schema.clone(),
        counter_precision,
    }
}

/// Builds the KLL-counter metadata from the hasher's [`HashProfile`].
fn hydra_kll_metadata<H: HashProfile>(
    grid: &HydraGrid,
    counter_k: u32,
    counter_m: u32,
) -> HydraKllMetadata {
    HydraKllMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        rows: grid.rows,
        cols: grid.cols,
        schema: grid.schema.clone(),
        counter_k,
        counter_m,
        counter_item_type: COUNTER_ITEM_TYPE.to_string(),
    }
}

/// Builds the UnivMon-counter metadata from the hasher's [`HashProfile`].
fn hydra_univmon_metadata<H: HashProfile>(
    grid: &HydraGrid,
    shape: &UnivMonShape,
    counter_key_type: &str,
) -> HydraUnivMonMetadata {
    HydraUnivMonMetadata {
        metadata_version: 1,
        hash_profile_id: H::PROFILE_ID.to_string(),
        hash_algorithm: H::ALGORITHM.to_string(),
        seed_derivation: H::SEED_DERIVATION.to_string(),
        input_encoding: H::INPUT_ENCODING.to_string(),
        seed_list: H::seed_list(),
        rows: grid.rows,
        cols: grid.cols,
        schema: grid.schema.clone(),
        counter_layer_size: shape.layer_size,
        counter_sketch_row: shape.sketch_row,
        counter_sketch_col: shape.sketch_col,
        counter_heap_size: shape.heap_size,
        counter_key_type: counter_key_type.to_string(),
    }
}

/// Hydra payload for the two matrix counters, a msgpack **array** (`to_vec`,
/// positional): `[counts]`. The cells tile one array in grid row-major order,
/// each cell's own counters row-major inside its run.
#[derive(Debug, Serialize, Deserialize)]
struct HydraMatrixPayload {
    counts: Vec<i32>,
}

/// Hydra payload for the HyperLogLog counter: `[registers]`, one msgpack `bin`
/// holding the cells' register runs in grid row-major order.
#[derive(Debug, Serialize, Deserialize)]
struct HydraHllPayload {
    #[serde(with = "serde_bytes")]
    registers: Vec<u8>,
}

/// One KLL cell's retained state, the §3.3 payload shape `[levels, items,
/// coin]`. Serialized positionally, so it lands as a nested 3-element array.
#[derive(Debug, Serialize, Deserialize)]
struct HydraKllCell {
    levels: Vec<u32>,
    items: Vec<f64>,
    coin: HydraKllCoin,
}

/// A KLL cell's compaction-RNG state, mirroring `sketchlib-go`'s `CoinState`.
#[derive(Debug, Serialize, Deserialize)]
struct HydraKllCoin {
    state: u64,
    bit_cache: u64,
    remaining_bits: u32,
}

/// Hydra payload for the KLL counter: `[cells]`, one element per grid cell in
/// row-major order. A KLL cell retains a variable number of samples, so the
/// cells are carried one element each rather than tiled.
#[derive(Debug, Serialize, Deserialize)]
struct HydraKllPayload {
    cells: Vec<HydraKllCell>,
}

/// Hydra payload for the UnivMon counter: `[cells]`, one pyramid array per grid
/// cell in row-major order, each in UnivMon's own payload shape.
#[derive(Debug, Serialize, Deserialize)]
struct HydraUnivMonPayload<K> {
    cells: Vec<PyramidPayload<K>>,
}

/// The HyperLogLog counter's own descriptor, rebuilt to hand a cell's registers
/// to the HLL decoder.
#[derive(Serialize)]
struct HllCellMetadata {
    metadata_version: u8,
    hash_profile_id: String,
    hash_algorithm: String,
    seed_derivation: String,
    input_encoding: String,
    seed_list: Vec<u64>,
    canonical_seed_index: u32,
    precision: u32,
}

/// The HyperLogLog counter's own payload, `[registers]`.
#[derive(Serialize)]
struct HllCellPayload<'a> {
    #[serde(with = "serde_bytes")]
    registers: &'a [u8],
}

/// The KLL counter's own descriptor. KLL never hashes, so it is structural
/// only; the optional compaction `seed` key is absent.
#[derive(Serialize)]
struct KllCellMetadata {
    metadata_version: u8,
    k: u32,
    m: u32,
    item_type: String,
}

/// The pyramid dimensions every UnivMon cell shares.
struct UnivMonShape {
    layer_size: u32,
    sketch_row: u32,
    sketch_col: u32,
    heap_size: u32,
}

/// Register count of one HyperLogLog cell.
const HLL_CELL_REGISTERS: usize = HllBucketListP14::NUM_REGISTERS;

/// Precision of one HyperLogLog cell.
const HLL_CELL_PRECISION: u32 = HllBucketListP14::PRECISION as u32;

fn encode_error(problem: String) -> RmpEncodeError {
    RmpEncodeError::Syntax(problem)
}

fn decode_error(problem: String) -> RmpDecodeError {
    RmpDecodeError::Uncategorized(problem)
}

/// Narrows a structural parameter into its `u32` metadata field.
fn to_u32(name: &str, value: usize) -> Result<u32, RmpEncodeError> {
    u32::try_from(value).map_err(|_| {
        encode_error(format!(
            "ASAPv1 Hydra envelope: {name} {value} exceeds the u32 metadata field"
        ))
    })
}

/// The cell count a declared geometry names, rejecting a zero dimension and an
/// overflowing product before anything is sized from it.
fn checked_cells(what: &str, rows: usize, cols: usize) -> Result<usize, String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "Hydra {what} dimensions must be non-zero: rows={rows}, cols={cols}"
        ));
    }
    check_matrix_rows(&format!("Hydra {what}"), rows)?;
    rows.checked_mul(cols)
        .ok_or_else(|| format!("Hydra {what} {rows}x{cols} overflows a cell count"))
}

/// The counter run a declared geometry names, rejecting a zero dimension and an
/// overflowing `cells * per_cell` product before anything is sized from it.
fn tiled_len(cells: usize, per_cell: usize, what: &str) -> Result<usize, String> {
    if per_cell == 0 {
        return Err(format!("Hydra counter {what} must be non-zero"));
    }
    cells
        .checked_mul(per_cell)
        .ok_or_else(|| format!("Hydra {cells} cells of {per_cell} {what} overflow a length"))
}

/// Rebuilds the key columns, re-validating the labels the metadata carries.
fn schema_of(labels: &[String]) -> Result<KeySchema, RmpDecodeError> {
    KeySchema::try_from(labels.to_vec())
        .map_err(|problem| decode_error(format!("ASAPv1 Hydra envelope: {problem}")))
}

/// Splits the envelope and rejects any kind_id but this variant's.
fn split_for<'a>(
    bytes: &'a [u8],
    expected_kind_id: &[u8],
) -> Result<(&'a [u8], &'a [u8]), RmpDecodeError> {
    let (kind_id, metadata, payload) = envelope::split(bytes).map_err(decode_error)?;
    if kind_id != expected_kind_id {
        return Err(decode_error(format!(
            "Hydra kind_id mismatch: stored {kind_id:?}, expected {expected_kind_id:?}"
        )));
    }
    Ok((metadata, payload))
}

/// The envelope a HyperLogLog cell's registers decode from.
fn hll_cell_envelope(registers: &[u8]) -> Result<Vec<u8>, RmpEncodeError> {
    let metadata = rmp_serde::to_vec_named(&HllCellMetadata {
        metadata_version: 1,
        hash_profile_id: DefaultXxHasher::PROFILE_ID.to_string(),
        hash_algorithm: DefaultXxHasher::ALGORITHM.to_string(),
        seed_derivation: DefaultXxHasher::SEED_DERIVATION.to_string(),
        input_encoding: DefaultXxHasher::INPUT_ENCODING.to_string(),
        seed_list: DefaultXxHasher::seed_list(),
        canonical_seed_index: DefaultXxHasher::CANONICAL_SEED_INDEX,
        precision: HLL_CELL_PRECISION,
    })?;
    let payload = rmp_serde::to_vec(&HllCellPayload { registers })?;
    Ok(envelope::encode(HLL_CELL_KIND, &metadata, &payload))
}

/// The envelope a KLL cell's retained state decodes from.
fn kll_cell_envelope(k: u32, m: u32, cell: &HydraKllCell) -> Result<Vec<u8>, RmpEncodeError> {
    let metadata = rmp_serde::to_vec_named(&KllCellMetadata {
        metadata_version: 1,
        k,
        m,
        item_type: COUNTER_ITEM_TYPE.to_string(),
    })?;
    let payload = rmp_serde::to_vec(cell)?;
    Ok(envelope::encode(KLL_CELL_KIND, &metadata, &payload))
}

/// The envelope a UnivMon cell's pyramid state decodes from.
fn univmon_cell_envelope(
    shape: &UnivMonShape,
    key_type: &str,
    payload: &[u8],
) -> Result<Vec<u8>, RmpEncodeError> {
    let metadata = rmp_serde::to_vec_named(&univmon_metadata::<DefaultXxHasher>(
        shape.layer_size,
        shape.sketch_row,
        shape.sketch_col,
        shape.heap_size,
        key_type,
    ))?;
    Ok(envelope::encode(UNIVMON_CELL_KIND, &metadata, payload))
}

/// Reads a cell's `keys` run length without knowing the key type.
fn univmon_cell_holds_keys(payload: &[u8]) -> Result<bool, RmpEncodeError> {
    let probe: PyramidPayload<IgnoredAny> =
        from_slice(payload).map_err(|e| encode_error(e.to_string()))?;
    Ok(!probe.keys.is_empty())
}

// Wire serialization for Hydra. `wire` is a descendant of the sketch module, so
// these impls read the private `schema` field and construct the struct
// directly.
impl Hydra {
    /// Serializes the grid into an ASAPv1 MessagePack envelope, one kind_id per
    /// counter variant (`0x07 0x00`-`0x07 0x04`). The metadata is derived from
    /// [`DefaultXxHasher`]'s [`HashProfile`], the profile the subkey hash at
    /// [`crate::HYDRA_SEED`] uses.
    ///
    /// A grid mixing counter variants, a cell whose geometry differs from the
    /// prototype's, a prototype holding data, and a declared geometry the
    /// storage does not match are errors rather than bytes that would be
    /// refused on decode.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>, RmpEncodeError> {
        let grid = self.wire_grid()?;
        match &self.type_to_clone {
            HydraCounter::KLL(prototype) => self.encode_kll(&grid, prototype),
            HydraCounter::CM(prototype) => self.encode_cm(&grid, prototype),
            HydraCounter::CS(prototype) => self.encode_cs(&grid, prototype),
            HydraCounter::HLL(prototype) => self.encode_hll(&grid, prototype),
            HydraCounter::UNIVERSAL(prototype) => self.encode_univmon(&grid, prototype),
        }
    }

    /// Deserializes a grid from an ASAPv1 MessagePack envelope, routing on the
    /// kind_id to the counter variant it names. Any other kind_id is rejected.
    ///
    /// Every declared count is measured against what the payload actually
    /// carries before anything is sized from it, so crafted geometry fails
    /// closed with an error rather than a panic or an allocation.
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (kind_id, _, _) = envelope::split(bytes).map_err(decode_error)?;
        match kind_id {
            id if id == HYDRA_KIND_KLL => Self::decode_kll(bytes),
            id if id == HYDRA_KIND_CM => Self::decode_cm(bytes),
            id if id == HYDRA_KIND_CS => Self::decode_cs(bytes),
            id if id == HYDRA_KIND_HLL => Self::decode_hll(bytes),
            id if id == HYDRA_KIND_UNIVMON => Self::decode_univmon(bytes),
            other => Err(decode_error(format!(
                "Hydra kind_id mismatch: stored {other:?}, expected one of {HYDRA_KIND_KLL:?}, {HYDRA_KIND_CM:?}, {HYDRA_KIND_CS:?}, {HYDRA_KIND_HLL:?}, {HYDRA_KIND_UNIVMON:?}"
            ))),
        }
    }

    /// The declared grid, rejecting a geometry the storage does not match.
    fn wire_grid(&self) -> Result<HydraGrid, RmpEncodeError> {
        let (rows, cols) = (self.sketches.rows(), self.sketches.cols());
        if (self.row_num, self.col_num) != (rows, cols) {
            return Err(encode_error(format!(
                "ASAPv1 Hydra envelope: declared grid {}x{} != the storage's {rows}x{cols}",
                self.row_num, self.col_num
            )));
        }
        let cells = checked_cells("grid", rows, cols).map_err(encode_error)?;
        if self.sketches.as_slice().len() != cells {
            return Err(encode_error(format!(
                "ASAPv1 Hydra envelope: storage holds {} cells != rows*cols {cells}",
                self.sketches.as_slice().len()
            )));
        }
        Ok(HydraGrid {
            rows: to_u32("rows", rows)?,
            cols: to_u32("cols", cols)?,
            schema: self.schema.labels().to_vec(),
        })
    }

    /// The grid's cells in row-major order, projected by `cell`. Fails on the
    /// first cell that is not the prototype's counter variant.
    fn cells<'a, T>(
        &'a self,
        cell: impl Fn(&'a HydraCounter) -> Option<T>,
    ) -> Result<Vec<T>, RmpEncodeError> {
        let counters = self.sketches.as_slice();
        let mut out = Vec::with_capacity(counters.len());
        for (i, counter) in counters.iter().enumerate() {
            match cell(counter) {
                Some(state) => out.push(state),
                None => {
                    return Err(encode_error(format!(
                        "ASAPv1 Hydra envelope: cell ({}, {}) holds a {counter}, the grid holds a {}",
                        i / self.col_num,
                        i % self.col_num,
                        self.type_to_clone
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Tiles the cells' counters into one row-major array, rejecting a cell
    /// whose geometry or cell count differs from the prototype's.
    fn tile_counters(
        &self,
        cells: &[&Vector2D<i32>],
        counter_rows: usize,
        counter_cols: usize,
    ) -> Result<Vec<i32>, RmpEncodeError> {
        let per_cell =
            checked_cells("counter", counter_rows, counter_cols).map_err(encode_error)?;
        let total = tiled_len(cells.len(), per_cell, "counters").map_err(encode_error)?;
        let mut counts = Vec::with_capacity(total);
        for (i, storage) in cells.iter().enumerate() {
            if (storage.rows(), storage.cols()) != (counter_rows, counter_cols) {
                return Err(encode_error(format!(
                    "ASAPv1 Hydra envelope: cell ({}, {}) is {}x{} against the prototype's {counter_rows}x{counter_cols}",
                    i / self.col_num,
                    i % self.col_num,
                    storage.rows(),
                    storage.cols()
                )));
            }
            let cell = storage.as_slice();
            if cell.len() != per_cell {
                return Err(encode_error(format!(
                    "ASAPv1 Hydra envelope: cell ({}, {}) holds {} counters != {per_cell}",
                    i / self.col_num,
                    i % self.col_num,
                    cell.len()
                )));
            }
            counts.extend_from_slice(cell);
        }
        Ok(counts)
    }

    /// Rejects a prototype holding data: it reaches the wire only through the
    /// metadata, so a fresh counter is what a decode rebuilds.
    fn check_fresh_prototype(&self, empty: bool) -> Result<(), RmpEncodeError> {
        if empty {
            return Ok(());
        }
        Err(encode_error(format!(
            "ASAPv1 Hydra envelope: type_to_clone holds data; the {} prototype carries only its geometry",
            self.type_to_clone
        )))
    }

    fn encode_cm(
        &self,
        grid: &HydraGrid,
        prototype: &CountMin<Vector2D<i32>, FastPath>,
    ) -> Result<Vec<u8>, RmpEncodeError> {
        self.check_fresh_prototype(prototype.as_storage().as_slice().iter().all(|&c| c == 0))?;
        let cells = self.cells(|counter| match counter {
            HydraCounter::CM(cm) => Some(cm.as_storage()),
            _ => None,
        })?;
        self.encode_matrix(
            grid,
            HYDRA_KIND_CM,
            &cells,
            prototype.rows(),
            prototype.cols(),
        )
    }

    fn encode_cs(
        &self,
        grid: &HydraGrid,
        prototype: &Count<Vector2D<i32>, FastPath>,
    ) -> Result<Vec<u8>, RmpEncodeError> {
        self.check_fresh_prototype(prototype.as_storage().as_slice().iter().all(|&c| c == 0))?;
        let cells = self.cells(|counter| match counter {
            HydraCounter::CS(cs) => Some(cs.as_storage()),
            _ => None,
        })?;
        self.encode_matrix(
            grid,
            HYDRA_KIND_CS,
            &cells,
            prototype.rows(),
            prototype.cols(),
        )
    }

    /// The shared body of the two matrix variants: the counters tile the
    /// payload and the prototype's geometry is a structural param.
    fn encode_matrix(
        &self,
        grid: &HydraGrid,
        kind_id: &[u8],
        cells: &[&Vector2D<i32>],
        counter_rows: usize,
        counter_cols: usize,
    ) -> Result<Vec<u8>, RmpEncodeError> {
        let counts = self.tile_counters(cells, counter_rows, counter_cols)?;
        let metadata = rmp_serde::to_vec_named(&hydra_matrix_metadata::<DefaultXxHasher>(
            grid,
            to_u32("counter_rows", counter_rows)?,
            to_u32("counter_cols", counter_cols)?,
        ))?;
        let payload = rmp_serde::to_vec(&HydraMatrixPayload { counts })?;
        Ok(envelope::encode(kind_id, &metadata, &payload))
    }

    fn encode_hll(
        &self,
        grid: &HydraGrid,
        prototype: &HyperLogLog<ErtlMLE>,
    ) -> Result<Vec<u8>, RmpEncodeError> {
        self.check_fresh_prototype(prototype.registers_as_slice().iter().all(|&r| r == 0))?;
        let cells = self.cells(|counter| match counter {
            HydraCounter::HLL(hll) => Some(hll.registers_as_slice()),
            _ => None,
        })?;
        let total =
            tiled_len(cells.len(), HLL_CELL_REGISTERS, "registers").map_err(encode_error)?;
        let mut registers = Vec::with_capacity(total);
        for cell in &cells {
            registers.extend_from_slice(cell);
        }
        let metadata = rmp_serde::to_vec_named(&hydra_hll_metadata::<DefaultXxHasher>(
            grid,
            HLL_CELL_PRECISION,
        ))?;
        let payload = rmp_serde::to_vec(&HydraHllPayload { registers })?;
        Ok(envelope::encode(HYDRA_KIND_HLL, &metadata, &payload))
    }

    fn encode_kll(&self, grid: &HydraGrid, prototype: &KLL) -> Result<Vec<u8>, RmpEncodeError> {
        self.check_fresh_prototype(
            prototype.wire_num_levels() == 1 && prototype.wire_items().is_empty(),
        )?;
        let (k, m) = (prototype.wire_k(), prototype.wire_m());
        let counters = self.cells(|counter| match counter {
            HydraCounter::KLL(kll) => Some(kll),
            _ => None,
        })?;
        let mut cells = Vec::with_capacity(counters.len());
        for (i, kll) in counters.iter().enumerate() {
            if (kll.wire_k(), kll.wire_m()) != (k, m) {
                return Err(encode_error(format!(
                    "ASAPv1 Hydra envelope: cell ({}, {}) has k={}, m={} against the prototype's k={k}, m={m}",
                    i / self.col_num,
                    i % self.col_num,
                    kll.wire_k(),
                    kll.wire_m()
                )));
            }
            let (state, bit_cache, remaining_bits) = kll.wire_coin();
            cells.push(HydraKllCell {
                levels: kll.wire_levels(),
                items: kll.wire_items(),
                coin: HydraKllCoin {
                    state,
                    bit_cache,
                    remaining_bits,
                },
            });
        }
        let metadata = rmp_serde::to_vec_named(&hydra_kll_metadata::<DefaultXxHasher>(grid, k, m))?;
        let payload = rmp_serde::to_vec(&HydraKllPayload { cells })?;
        Ok(envelope::encode(HYDRA_KIND_KLL, &metadata, &payload))
    }

    fn encode_univmon(
        &self,
        grid: &HydraGrid,
        prototype: &UnivMon,
    ) -> Result<Vec<u8>, RmpEncodeError> {
        if prototype.layer_size == 0
            || prototype.sketch_row == 0
            || prototype.sketch_col == 0
            || prototype.heap_size == 0
        {
            return Err(encode_error(format!(
                "ASAPv1 Hydra envelope: UnivMon counter dimensions must be non-zero: layers={}, {}x{}, heap={}",
                prototype.layer_size,
                prototype.sketch_row,
                prototype.sketch_col,
                prototype.heap_size
            )));
        }
        let fresh = UnivMon::init_univmon(
            prototype.heap_size,
            prototype.sketch_row,
            prototype.sketch_col,
            prototype.layer_size,
        );
        self.check_fresh_prototype(prototype.serialize_to_bytes()? == fresh.serialize_to_bytes()?)?;
        let shape = UnivMonShape {
            layer_size: to_u32("counter_layer_size", prototype.layer_size)?,
            sketch_row: to_u32("counter_sketch_row", prototype.sketch_row)?,
            sketch_col: to_u32("counter_sketch_col", prototype.sketch_col)?,
            heap_size: to_u32("counter_heap_size", prototype.heap_size)?,
        };
        let counters = self.cells(|counter| match counter {
            HydraCounter::UNIVERSAL(um) => Some(um),
            _ => None,
        })?;

        // Each cell is encoded through UnivMon's own codec, so every rule that
        // codec enforces holds for a Hydra cell too.
        let encoded: Vec<Vec<u8>> = counters
            .iter()
            .map(|um| um.serialize_to_bytes())
            .collect::<Result<_, _>>()?;
        let mut payloads = Vec::with_capacity(encoded.len());
        let mut key_type: Option<&str> = None;
        let mut metas = Vec::with_capacity(encoded.len());
        for bytes in &encoded {
            let (_, metadata, payload) = envelope::split(bytes).map_err(encode_error)?;
            metas.push(
                from_slice::<UnivMonMetadata>(metadata).map_err(|e| encode_error(e.to_string()))?,
            );
            payloads.push(payload);
        }
        for (i, meta) in metas.iter().enumerate() {
            if (
                meta.layer_size,
                meta.sketch_row,
                meta.sketch_col,
                meta.heap_size,
            ) != (
                shape.layer_size,
                shape.sketch_row,
                shape.sketch_col,
                shape.heap_size,
            ) {
                return Err(encode_error(format!(
                    "ASAPv1 Hydra envelope: cell ({}, {}) is a {}x{}x{} pyramid against the prototype's {}x{}x{}",
                    i / self.col_num,
                    i % self.col_num,
                    meta.layer_size,
                    meta.sketch_row,
                    meta.sketch_col,
                    shape.layer_size,
                    shape.sketch_row,
                    shape.sketch_col
                )));
            }
            // A cell holding no keys carries no variant, so only the cells that
            // hold one pin the grid's key type.
            if univmon_cell_holds_keys(payloads[i])? {
                match key_type {
                    None => key_type = Some(&meta.key_type),
                    Some(pinned) if pinned != meta.key_type => {
                        return Err(encode_error(format!(
                            "ASAPv1 Hydra envelope: cells mix key variants — key_type is {pinned}, but cell ({}, {}) holds {} keys",
                            i / self.col_num,
                            i % self.col_num,
                            meta.key_type
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
        let key_type = key_type.unwrap_or(match metas.first() {
            Some(meta) => &meta.key_type,
            None => {
                return Err(encode_error(
                    "ASAPv1 Hydra envelope: the grid holds no cells".to_string(),
                ));
            }
        });
        let metadata = rmp_serde::to_vec_named(&hydra_univmon_metadata::<DefaultXxHasher>(
            grid, &shape, key_type,
        ))?;
        let payload = pack_univmon_cells(key_type, &payloads)?;
        Ok(envelope::encode(HYDRA_KIND_UNIVMON, &metadata, &payload))
    }

    fn decode_cm(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (counts, meta, per_cell) = Self::decode_matrix(bytes, HYDRA_KIND_CM)?;
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        let (counter_rows, counter_cols) = (meta.counter_rows as usize, meta.counter_cols as usize);
        let sketches = Vector2D::from_fn(rows, cols, |r, c| {
            let base = (r * cols + c) * per_cell;
            HydraCounter::CM(CountMin::from_storage(Vector2D::from_fn(
                counter_rows,
                counter_cols,
                |i, j| counts[base + i * counter_cols + j],
            )))
        });
        Ok(Hydra {
            row_num: rows,
            col_num: cols,
            sketches,
            type_to_clone: HydraCounter::CM(CountMin::with_dimensions(counter_rows, counter_cols)),
            schema: schema_of(&meta.schema)?,
        })
    }

    fn decode_cs(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (counts, meta, per_cell) = Self::decode_matrix(bytes, HYDRA_KIND_CS)?;
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        let (counter_rows, counter_cols) = (meta.counter_rows as usize, meta.counter_cols as usize);
        let sketches = Vector2D::from_fn(rows, cols, |r, c| {
            let base = (r * cols + c) * per_cell;
            HydraCounter::CS(Count::from_storage(Vector2D::from_fn(
                counter_rows,
                counter_cols,
                |i, j| counts[base + i * counter_cols + j],
            )))
        });
        Ok(Hydra {
            row_num: rows,
            col_num: cols,
            sketches,
            type_to_clone: HydraCounter::CS(Count::with_dimensions(counter_rows, counter_cols)),
            schema: schema_of(&meta.schema)?,
        })
    }

    /// The shared body of the two matrix variants' decoders: validate the
    /// metadata, then measure the tiled counters against the declared geometry
    /// before any matrix is built.
    fn decode_matrix(
        bytes: &[u8],
        kind_id: &[u8],
    ) -> Result<(Vec<i32>, HydraMatrixMetadata, usize), RmpDecodeError> {
        let (metadata, payload) = split_for(bytes, kind_id)?;
        let meta: HydraMatrixMetadata = from_slice(metadata)?;
        if meta
            != hydra_matrix_metadata::<DefaultXxHasher>(
                &HydraGrid {
                    rows: meta.rows,
                    cols: meta.cols,
                    schema: meta.schema.clone(),
                },
                meta.counter_rows,
                meta.counter_cols,
            )
        {
            return Err(decode_error(
                "ASAPv1 Hydra envelope: metadata mismatch".to_string(),
            ));
        }
        let cells =
            checked_cells("grid", meta.rows as usize, meta.cols as usize).map_err(decode_error)?;
        let per_cell = checked_cells(
            "counter",
            meta.counter_rows as usize,
            meta.counter_cols as usize,
        )
        .map_err(decode_error)?;
        let total = tiled_len(cells, per_cell, "counters").map_err(decode_error)?;
        let p: HydraMatrixPayload = from_slice(payload)?;
        if p.counts.len() != total {
            return Err(decode_error(format!(
                "Hydra counts length {} != rows*cols*counter_rows*counter_cols {total}",
                p.counts.len()
            )));
        }
        Ok((p.counts, meta, per_cell))
    }

    fn decode_hll(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (metadata, payload) = split_for(bytes, HYDRA_KIND_HLL)?;
        let meta: HydraHllMetadata = from_slice(metadata)?;
        if meta
            != hydra_hll_metadata::<DefaultXxHasher>(
                &HydraGrid {
                    rows: meta.rows,
                    cols: meta.cols,
                    schema: meta.schema.clone(),
                },
                HLL_CELL_PRECISION,
            )
        {
            return Err(decode_error(
                "ASAPv1 Hydra envelope: metadata mismatch".to_string(),
            ));
        }
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        let cells = checked_cells("grid", rows, cols).map_err(decode_error)?;
        let total = tiled_len(cells, HLL_CELL_REGISTERS, "registers").map_err(decode_error)?;
        let p: HydraHllPayload = from_slice(payload)?;
        if p.registers.len() != total {
            return Err(decode_error(format!(
                "Hydra registers length {} != rows*cols*2^precision {total}",
                p.registers.len()
            )));
        }
        let mut counters = Vec::with_capacity(cells);
        for cell in p.registers.chunks_exact(HLL_CELL_REGISTERS) {
            let envelope_bytes =
                hll_cell_envelope(cell).map_err(|e| decode_error(e.to_string()))?;
            counters.push(HydraCounter::HLL(
                HyperLogLog::<ErtlMLE>::deserialize_from_bytes(&envelope_bytes)?,
            ));
        }
        let mut counters = counters.into_iter();
        let sketches = Vector2D::from_fn(rows, cols, |_, _| {
            counters.next().expect("one counter per grid cell")
        });
        Ok(Hydra {
            row_num: rows,
            col_num: cols,
            sketches,
            type_to_clone: HydraCounter::HLL(HyperLogLog::<ErtlMLE>::default()),
            schema: schema_of(&meta.schema)?,
        })
    }

    fn decode_kll(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (metadata, payload) = split_for(bytes, HYDRA_KIND_KLL)?;
        let meta: HydraKllMetadata = from_slice(metadata)?;
        if meta
            != hydra_kll_metadata::<DefaultXxHasher>(
                &HydraGrid {
                    rows: meta.rows,
                    cols: meta.cols,
                    schema: meta.schema.clone(),
                },
                meta.counter_k,
                meta.counter_m,
            )
        {
            return Err(decode_error(
                "ASAPv1 Hydra envelope: metadata mismatch".to_string(),
            ));
        }
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        let cells = checked_cells("grid", rows, cols).map_err(decode_error)?;
        let p: HydraKllPayload = from_slice(payload)?;
        if p.cells.len() != cells {
            return Err(decode_error(format!(
                "Hydra carries {} KLL cells != rows*cols {cells}",
                p.cells.len()
            )));
        }
        let mut counters = Vec::with_capacity(cells);
        for cell in &p.cells {
            let envelope_bytes = kll_cell_envelope(meta.counter_k, meta.counter_m, cell)
                .map_err(|e| decode_error(e.to_string()))?;
            counters.push(HydraCounter::KLL(KLL::deserialize_from_bytes(
                &envelope_bytes,
            )?));
        }
        let mut counters = counters.into_iter();
        let sketches = Vector2D::from_fn(rows, cols, |_, _| {
            counters.next().expect("one counter per grid cell")
        });
        Ok(Hydra {
            row_num: rows,
            col_num: cols,
            sketches,
            type_to_clone: HydraCounter::KLL(KLL::init(
                meta.counter_k as usize,
                meta.counter_m as usize,
            )),
            schema: schema_of(&meta.schema)?,
        })
    }

    fn decode_univmon(bytes: &[u8]) -> Result<Self, RmpDecodeError> {
        let (metadata, payload) = split_for(bytes, HYDRA_KIND_UNIVMON)?;
        let meta: HydraUnivMonMetadata = from_slice(metadata)?;
        let shape = UnivMonShape {
            layer_size: meta.counter_layer_size,
            sketch_row: meta.counter_sketch_row,
            sketch_col: meta.counter_sketch_col,
            heap_size: meta.counter_heap_size,
        };
        if meta
            != hydra_univmon_metadata::<DefaultXxHasher>(
                &HydraGrid {
                    rows: meta.rows,
                    cols: meta.cols,
                    schema: meta.schema.clone(),
                },
                &shape,
                &meta.counter_key_type,
            )
        {
            return Err(decode_error(
                "ASAPv1 Hydra envelope: metadata mismatch".to_string(),
            ));
        }
        let (rows, cols) = (meta.rows as usize, meta.cols as usize);
        let cells = checked_cells("grid", rows, cols).map_err(decode_error)?;
        if shape.layer_size == 0
            || shape.sketch_row == 0
            || shape.sketch_col == 0
            || shape.heap_size == 0
        {
            return Err(decode_error(format!(
                "Hydra UnivMon counter dimensions must be non-zero: layers={}, {}x{}, heap={}",
                shape.layer_size, shape.sketch_row, shape.sketch_col, shape.heap_size
            )));
        }
        let payloads = unpack_univmon_cells(&meta.counter_key_type, payload)?;
        if payloads.len() != cells {
            return Err(decode_error(format!(
                "Hydra carries {} UnivMon cells != rows*cols {cells}",
                payloads.len()
            )));
        }
        let mut counters = Vec::with_capacity(cells);
        for cell in &payloads {
            let envelope_bytes = univmon_cell_envelope(&shape, &meta.counter_key_type, cell)
                .map_err(|e| decode_error(e.to_string()))?;
            counters.push(HydraCounter::UNIVERSAL(UnivMon::deserialize_from_bytes(
                &envelope_bytes,
            )?));
        }
        let mut counters = counters.into_iter();
        let sketches = Vector2D::from_fn(rows, cols, |_, _| {
            counters.next().expect("one counter per grid cell")
        });
        Ok(Hydra {
            row_num: rows,
            col_num: cols,
            sketches,
            type_to_clone: HydraCounter::UNIVERSAL(UnivMon::init_univmon(
                shape.heap_size as usize,
                shape.sketch_row as usize,
                shape.sketch_col as usize,
                shape.layer_size as usize,
            )),
            schema: schema_of(&meta.schema)?,
        })
    }
}

/// Packs the cells' pyramid payloads with `keys` typed by `key_type`. A cell
/// whose keys are not of that variant fails the encode.
fn pack_univmon_cells(key_type: &str, payloads: &[&[u8]]) -> Result<Vec<u8>, RmpEncodeError> {
    macro_rules! pack {
        ($ty:ty) => {{
            let mut cells = Vec::with_capacity(payloads.len());
            for payload in payloads {
                cells.push(
                    from_slice::<PyramidPayload<$ty>>(payload)
                        .map_err(|e| encode_error(e.to_string()))?,
                );
            }
            rmp_serde::to_vec(&HydraUnivMonPayload { cells })
        }};
    }
    match key_type {
        "i8" => pack!(i8),
        "i16" => pack!(i16),
        "i32" => pack!(i32),
        "i64" => pack!(i64),
        "isize" => pack!(isize),
        "u8" => pack!(u8),
        "u16" => pack!(u16),
        "u32" => pack!(u32),
        "u64" => pack!(u64),
        "usize" => pack!(usize),
        "f32" => pack!(f32),
        "f64" => pack!(f64),
        "string" => pack!(String),
        "bytes" => pack!(crate::message_pack_format::wire_key::WireBytes),
        other => Err(encode_error(format!(
            "ASAPv1 Hydra: key_type {other:?} is not a wire key type"
        ))),
    }
}

/// Reads the cells with `keys` typed by the metadata `counter_key_type`,
/// returning each cell's pyramid payload bytes. An unknown key type is
/// rejected, and keys whose msgpack types disagree fail in `from_slice`.
fn unpack_univmon_cells(key_type: &str, payload: &[u8]) -> Result<Vec<Vec<u8>>, RmpDecodeError> {
    macro_rules! unpack {
        ($ty:ty) => {{
            let decoded: HydraUnivMonPayload<$ty> = from_slice(payload)?;
            decoded
                .cells
                .iter()
                .map(|cell| rmp_serde::to_vec(cell).map_err(|e| decode_error(e.to_string())))
                .collect()
        }};
    }
    match key_type {
        "i8" => unpack!(i8),
        "i16" => unpack!(i16),
        "i32" => unpack!(i32),
        "i64" => unpack!(i64),
        "isize" => unpack!(isize),
        "u8" => unpack!(u8),
        "u16" => unpack!(u16),
        "u32" => unpack!(u32),
        "u64" => unpack!(u64),
        "usize" => unpack!(usize),
        "f32" => unpack!(f32),
        "f64" => unpack!(f64),
        "string" => unpack!(String),
        "bytes" => unpack!(crate::message_pack_format::wire_key::WireBytes),
        other => Err(decode_error(format!(
            "ASAPv1 Hydra: key_type {other:?} is not a wire key type"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::HydraQuery;
    use crate::{CANONICAL_HASH_SEED, DataInput, HeapItem, MATRIX_MAX_ROWS, SketchHasher};

    /// The row bound as a metadata field width.
    const MAX_ROWS: u32 = MATRIX_MAX_ROWS as u32;

    /// Two key columns, and the records every fixture grid is fed.
    const SCHEMA: [&str; 2] = ["region", "service"];
    const KEYS: [[&str; 2]; 3] = [["eu", "auth"], ["eu", "cart"], ["us", "auth"]];

    type Probe<'a> = (&'a [Option<&'a str>], HydraQuery<'a>);

    fn grid(counter: HydraCounter) -> Hydra {
        Hydra::with_schema(3, 8, SCHEMA, counter).expect("valid schema")
    }

    fn feed(hydra: &mut Hydra, values: impl Iterator<Item = DataInput<'static>>) {
        for (i, value) in values.enumerate() {
            hydra
                .update(&KEYS[i % KEYS.len()], &value, None)
                .expect("arity 2");
        }
    }

    fn cm_counter() -> HydraCounter {
        HydraCounter::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(2, 16))
    }

    fn cs_counter() -> HydraCounter {
        HydraCounter::CS(Count::<Vector2D<i32>, FastPath>::with_dimensions(2, 16))
    }

    fn cm_grid() -> Hydra {
        let mut hydra = grid(cm_counter());
        feed(&mut hydra, (0..30u64).map(|i| DataInput::U64(i % 7)));
        hydra
    }

    fn cs_grid() -> Hydra {
        let mut hydra = grid(cs_counter());
        feed(&mut hydra, (0..30u64).map(|i| DataInput::U64(i % 7)));
        hydra
    }

    fn hll_grid() -> Hydra {
        let mut hydra = grid(HydraCounter::HLL(HyperLogLog::<ErtlMLE>::default()));
        feed(&mut hydra, (0..300u64).map(DataInput::U64));
        hydra
    }

    fn kll_grid() -> Hydra {
        let mut hydra = grid(HydraCounter::KLL(KLL::default()));
        feed(&mut hydra, (0..300u64).map(|i| DataInput::F64(i as f64)));
        hydra
    }

    fn univmon_grid() -> Hydra {
        let mut hydra = grid(HydraCounter::UNIVERSAL(UnivMon::init_univmon(4, 2, 16, 3)));
        feed(&mut hydra, (0..120u64).map(|i| DataInput::U64(i % 20)));
        hydra
    }

    fn frequency_probes() -> Vec<Probe<'static>> {
        vec![
            (
                &[Some("eu"), Some("auth")],
                HydraQuery::Frequency(DataInput::U64(3)),
            ),
            (
                &[Some("eu"), None],
                HydraQuery::Frequency(DataInput::U64(5)),
            ),
            (
                &[None, Some("auth")],
                HydraQuery::Frequency(DataInput::U64(0)),
            ),
        ]
    }

    fn answers(hydra: &Hydra, probes: &[Probe<'_>]) -> Vec<f64> {
        probes
            .iter()
            .map(|(key, query)| hydra.query_key(key, query).expect("supported query"))
            .collect()
    }

    fn assert_envelope(bytes: &[u8], kind_id: &[u8]) {
        assert!(bytes.starts_with(b"ASAPv1"));
        assert_eq!(bytes[7], 2, "kind_id_len");
        assert_eq!(&bytes[8..10], kind_id);
    }

    /// One round trip: the envelope names the variant, the schema and every
    /// answer survive, and the decoded grid re-serializes byte-identically.
    fn assert_round_trip(hydra: &Hydra, kind_id: &[u8], probes: &[Probe<'_>]) -> Hydra {
        let encoded = hydra.serialize_to_bytes().expect("serialize Hydra");
        assert_envelope(&encoded, kind_id);

        let decoded = Hydra::deserialize_from_bytes(&encoded).expect("deserialize Hydra");
        assert_eq!(decoded.row_num, hydra.row_num);
        assert_eq!(decoded.col_num, hydra.col_num);
        assert_eq!(decoded.sketches.rows(), hydra.sketches.rows());
        assert_eq!(decoded.sketches.cols(), hydra.sketches.cols());
        assert_eq!(decoded.schema(), hydra.schema(), "schema lost");
        assert_eq!(
            answers(&decoded, probes),
            answers(hydra, probes),
            "answers changed across the wire"
        );
        assert_eq!(
            decoded.serialize_to_bytes().expect("re-serialize"),
            encoded,
            "a decoded Hydra re-serialized to different bytes"
        );
        decoded
    }

    #[test]
    fn hydra_count_min_round_trip_serialization() {
        assert_round_trip(&cm_grid(), HYDRA_KIND_CM, &frequency_probes());
    }

    #[test]
    fn hydra_count_sketch_round_trip_serialization() {
        assert_round_trip(&cs_grid(), HYDRA_KIND_CS, &frequency_probes());
    }

    #[test]
    fn hydra_hyperloglog_round_trip_serialization() {
        let probes: Vec<Probe<'_>> = vec![
            (&[Some("eu"), Some("auth")], HydraQuery::Cardinality),
            (&[Some("eu"), None], HydraQuery::Cardinality),
        ];
        assert_round_trip(&hll_grid(), HYDRA_KIND_HLL, &probes);
    }

    #[test]
    fn hydra_kll_round_trip_serialization() {
        let probes: Vec<Probe<'_>> = vec![
            (&[Some("eu"), Some("auth")], HydraQuery::Quantile(0.5)),
            (&[Some("eu"), None], HydraQuery::Quantile(0.9)),
            (&[None, Some("auth")], HydraQuery::Cdf(100.0)),
        ];
        assert_round_trip(&kll_grid(), HYDRA_KIND_KLL, &probes);
    }

    #[test]
    fn hydra_univmon_round_trip_serialization() {
        let probes: Vec<Probe<'_>> = vec![
            (&[Some("eu"), Some("auth")], HydraQuery::L1Norm),
            (&[Some("eu"), Some("auth")], HydraQuery::L2Norm),
            (&[Some("eu"), None], HydraQuery::Entropy),
            (&[None, Some("auth")], HydraQuery::Cardinality),
        ];
        assert_round_trip(&univmon_grid(), HYDRA_KIND_UNIVMON, &probes);
    }

    /// Count Sketch cells are signed: the payload must carry negatives through
    /// unchanged.
    #[test]
    fn hydra_count_sketch_negative_cells_round_trip() {
        let hydra = cs_grid();
        let cells = |h: &Hydra| -> Vec<i32> {
            h.sketches
                .as_slice()
                .iter()
                .flat_map(|counter| match counter {
                    HydraCounter::CS(cs) => cs.as_storage().as_slice().to_vec(),
                    other => panic!("expected a Count Sketch cell, got {other}"),
                })
                .collect()
        };
        assert!(
            cells(&hydra).iter().any(|&c| c < 0),
            "expected a negative counter in the grid"
        );
        let encoded = hydra.serialize_to_bytes().expect("serialize");
        let decoded = Hydra::deserialize_from_bytes(&encoded).expect("decode");
        assert_eq!(cells(&decoded), cells(&hydra));
    }

    /// The key columns round-trip exactly, escaping included, so a decoded grid
    /// reproduces every subkey.
    #[test]
    fn hydra_schema_round_trips_exactly() {
        let labels = ["a;b", "c:d\\e"];
        let mut hydra = Hydra::with_schema(3, 8, labels, cm_counter()).expect("valid schema");
        for _ in 0..5 {
            hydra
                .update(&["x;y", "z"], &DataInput::U64(1), None)
                .expect("arity 2");
        }
        let probes: Vec<Probe<'_>> = vec![
            (
                &[Some("x;y"), Some("z")],
                HydraQuery::Frequency(DataInput::U64(1)),
            ),
            (
                &[Some("x;y"), None],
                HydraQuery::Frequency(DataInput::U64(1)),
            ),
        ];
        let decoded = assert_round_trip(&hydra, HYDRA_KIND_CM, &probes);
        assert_eq!(decoded.schema(), labels);
    }

    /// Each variant's decoder owns exactly one kind_id, and none of them reads
    /// a plain Count-Min envelope.
    #[test]
    fn hydra_variants_reject_each_others_envelopes() {
        type Decoder = fn(&[u8]) -> Result<Hydra, RmpDecodeError>;
        let envelopes: [(&[u8], Vec<u8>); 5] = [
            (
                HYDRA_KIND_KLL,
                kll_grid().serialize_to_bytes().expect("serialize KLL"),
            ),
            (
                HYDRA_KIND_CM,
                cm_grid().serialize_to_bytes().expect("serialize CM"),
            ),
            (
                HYDRA_KIND_CS,
                cs_grid().serialize_to_bytes().expect("serialize CS"),
            ),
            (
                HYDRA_KIND_HLL,
                hll_grid().serialize_to_bytes().expect("serialize HLL"),
            ),
            (
                HYDRA_KIND_UNIVMON,
                univmon_grid()
                    .serialize_to_bytes()
                    .expect("serialize UnivMon"),
            ),
        ];
        let decoders: [(&[u8], Decoder); 5] = [
            (HYDRA_KIND_KLL, Hydra::decode_kll),
            (HYDRA_KIND_CM, Hydra::decode_cm),
            (HYDRA_KIND_CS, Hydra::decode_cs),
            (HYDRA_KIND_HLL, Hydra::decode_hll),
            (HYDRA_KIND_UNIVMON, Hydra::decode_univmon),
        ];
        let foreign = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(2, 16)
            .serialize_to_bytes()
            .expect("serialize Count-Min");
        for (owned, decode) in decoders {
            for (kind_id, bytes) in &envelopes {
                assert_eq!(
                    decode(bytes).is_ok(),
                    owned == *kind_id,
                    "decoder {owned:?} against envelope {kind_id:?}"
                );
            }
            assert!(
                decode(&foreign).is_err(),
                "a Count-Min envelope must not decode as a Hydra"
            );
        }
        assert!(Hydra::deserialize_from_bytes(&foreign).is_err());
    }

    /// A grid mixing counter variants has no encoding: the kind_id names one
    /// counter and the payload carries no per-cell tag.
    #[test]
    fn hydra_rejects_a_mixed_variant_grid() {
        let mut hydra = cm_grid();
        hydra.sketches.as_mut_slice()[9] = HydraCounter::HLL(HyperLogLog::<ErtlMLE>::default());
        let problem = hydra
            .serialize_to_bytes()
            .expect_err("a mixed grid must not serialize")
            .to_string();
        assert!(problem.contains("cell (1, 1)"), "got {problem}");
        assert!(problem.contains("HyperLogLog Counter"), "got {problem}");
        assert!(
            problem.contains("Count-Min Sketch Counter"),
            "got {problem}"
        );
    }

    /// A state the decoder would refuse must not serialize.
    #[test]
    fn hydra_rejects_serializing_an_inconsistent_grid() {
        let mut declared = cm_grid();
        declared.row_num = 4;
        assert!(
            declared.serialize_to_bytes().is_err(),
            "a declared grid the storage does not match must not serialize"
        );

        let mut unfilled = cm_grid();
        unfilled.sketches = Vector2D::init(3, 8);
        assert!(
            unfilled.serialize_to_bytes().is_err(),
            "an unfilled grid must not serialize"
        );

        let mut resized = cm_grid();
        resized.sketches.as_mut_slice()[2] = HydraCounter::CM(CountMin::with_dimensions(2, 8));
        assert!(
            resized.serialize_to_bytes().is_err(),
            "a cell that is not the prototype's size must not serialize"
        );

        let mut primed = cm_grid();
        let mut counter = CountMin::<Vector2D<i32>, FastPath>::with_dimensions(2, 16);
        counter.insert(&DataInput::U64(1));
        primed.type_to_clone = HydraCounter::CM(counter);
        let problem = primed
            .serialize_to_bytes()
            .expect_err("a prototype holding data must not serialize")
            .to_string();
        assert!(
            problem.contains("type_to_clone holds data"),
            "got {problem}"
        );
    }

    /// A UnivMon grid whose cells hold different `HeapItem` variants has no
    /// single `counter_key_type`.
    #[test]
    fn hydra_univmon_rejects_cells_mixing_key_variants() {
        let mut hydra = univmon_grid();
        let mut strings = UnivMon::init_univmon(4, 2, 16, 3);
        strings.insert(&DataInput::Str("alpha"), 5);
        hydra.sketches.as_mut_slice()[0] = HydraCounter::UNIVERSAL(strings);
        let problem = hydra
            .serialize_to_bytes()
            .expect_err("a grid mixing key variants must not serialize")
            .to_string();
        assert!(problem.contains("mix key variants"), "got {problem}");
    }

    /// Fail closed (not panic) on crafted geometry: every declared count is
    /// measured against the payload before anything is sized from it.
    #[test]
    fn hydra_rejects_crafted_geometry() {
        let hydra = cm_grid();
        let encoded = hydra.serialize_to_bytes().expect("serialize");
        let (_, metadata, payload) = envelope::split(&encoded).expect("split");
        let base: HydraMatrixMetadata = from_slice(metadata).expect("metadata");
        let blank = || HydraGrid {
            rows: 0,
            cols: 0,
            schema: base.schema.clone(),
        };
        let shaped = |rows, cols, counter_rows, counter_cols| HydraMatrixMetadata {
            rows,
            cols,
            counter_rows,
            counter_cols,
            ..hydra_matrix_metadata::<DefaultXxHasher>(&blank(), 0, 0)
        };
        let cases = [
            (shaped(MAX_ROWS, 1 << 20, 2, 16), "counts length"),
            (shaped(MAX_ROWS, u32::MAX, MAX_ROWS, u32::MAX), "overflow"),
            (shaped(MAX_ROWS + 1, 8, 2, 16), "MATRIX_MAX_ROWS"),
            (shaped(3, 8, MAX_ROWS + 1, 16), "MATRIX_MAX_ROWS"),
            (shaped(0, 8, 2, 16), "grid dimensions must be non-zero"),
            (shaped(3, 0, 2, 16), "grid dimensions must be non-zero"),
            (shaped(3, 8, 0, 16), "counter dimensions must be non-zero"),
            (shaped(3, 8, 2, 0), "counter dimensions must be non-zero"),
            (shaped(3, 8, 4, 16), "counts length"),
        ];
        for (meta, expected) in cases {
            let bytes = envelope::encode(
                HYDRA_KIND_CM,
                &rmp_serde::to_vec_named(&meta).expect("metadata"),
                payload,
            );
            let problem = Hydra::deserialize_from_bytes(&bytes)
                .expect_err("a crafted geometry must be rejected, not decoded")
                .to_string();
            assert!(problem.contains(expected), "got {problem}");
        }

        // A schema no Hydra could have been built with.
        let mut empty = shaped(3, 8, 2, 16);
        empty.schema = Vec::new();
        let bytes = envelope::encode(
            HYDRA_KIND_CM,
            &rmp_serde::to_vec_named(&empty).expect("metadata"),
            payload,
        );
        assert!(Hydra::deserialize_from_bytes(&bytes).is_err());
    }

    /// The grid and the counters are both matrices, so each carries the seed
    /// list's row bound; the encode side refuses what the decoder would.
    #[test]
    fn hydra_rejects_too_many_rows() {
        let rows = MATRIX_MAX_ROWS + 1;
        let wide_grid = Hydra::with_schema(rows, 8, SCHEMA, cm_counter()).expect("valid schema");
        let problem = wide_grid
            .serialize_to_bytes()
            .expect_err("a grid past MATRIX_MAX_ROWS must not serialize")
            .to_string();
        assert!(problem.contains("MATRIX_MAX_ROWS"), "got {problem}");

        let wide_counter = HydraCounter::CM(CountMin::<Vector2D<i32>, FastPath>::with_dimensions(
            rows, 16,
        ));
        let problem = grid(wide_counter)
            .serialize_to_bytes()
            .expect_err("a counter past MATRIX_MAX_ROWS must not serialize")
            .to_string();
        assert!(problem.contains("MATRIX_MAX_ROWS"), "got {problem}");

        // The boundary itself is eligible.
        assert!(
            Hydra::with_schema(MATRIX_MAX_ROWS, 8, SCHEMA, cm_counter())
                .expect("valid schema")
                .serialize_to_bytes()
                .is_ok()
        );
    }

    /// The variable-length counters are cut by the same rule: a grid larger
    /// than the payload carries, and a zero counter dimension, are rejected.
    #[test]
    fn hydra_rejects_crafted_geometry_for_the_variable_counters() {
        let encoded = kll_grid().serialize_to_bytes().expect("serialize");
        let (_, metadata, payload) = envelope::split(&encoded).expect("split");
        let base: HydraKllMetadata = from_slice(metadata).expect("metadata");
        let huge = HydraKllMetadata {
            rows: MAX_ROWS,
            cols: 4096,
            ..hydra_kll_metadata::<DefaultXxHasher>(
                &HydraGrid {
                    rows: 0,
                    cols: 0,
                    schema: base.schema.clone(),
                },
                base.counter_k,
                base.counter_m,
            )
        };
        let bytes = envelope::encode(
            HYDRA_KIND_KLL,
            &rmp_serde::to_vec_named(&huge).expect("metadata"),
            payload,
        );
        assert!(Hydra::deserialize_from_bytes(&bytes).is_err());

        let encoded = univmon_grid().serialize_to_bytes().expect("serialize");
        let (_, metadata, payload) = envelope::split(&encoded).expect("split");
        let base: HydraUnivMonMetadata = from_slice(metadata).expect("metadata");
        let shaped = |rows, cols, shape: UnivMonShape| HydraUnivMonMetadata {
            rows,
            cols,
            ..hydra_univmon_metadata::<DefaultXxHasher>(
                &HydraGrid {
                    rows: 0,
                    cols: 0,
                    schema: base.schema.clone(),
                },
                &shape,
                &base.counter_key_type,
            )
        };
        let shape = || UnivMonShape {
            layer_size: base.counter_layer_size,
            sketch_row: base.counter_sketch_row,
            sketch_col: base.counter_sketch_col,
            heap_size: base.counter_heap_size,
        };
        let cases = [
            shaped(MAX_ROWS, 4096, shape()),
            shaped(
                3,
                8,
                UnivMonShape {
                    sketch_col: 0,
                    ..shape()
                },
            ),
            shaped(
                3,
                8,
                UnivMonShape {
                    layer_size: 0,
                    ..shape()
                },
            ),
            shaped(
                3,
                8,
                UnivMonShape {
                    heap_size: 0,
                    ..shape()
                },
            ),
        ];
        for meta in cases {
            let bytes = envelope::encode(
                HYDRA_KIND_UNIVMON,
                &rmp_serde::to_vec_named(&meta).expect("metadata"),
                payload,
            );
            assert!(
                Hydra::deserialize_from_bytes(&bytes).is_err(),
                "a crafted UnivMon shape must be rejected, not decoded"
            );
        }
    }

    /// Fail closed on an unexpected metadata key, and on a missing required
    /// one.
    #[test]
    fn hydra_metadata_rejects_unknown_and_missing_keys() {
        #[derive(Serialize)]
        struct WithExtra {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            matrix_seed_index: u32,
            rows: u32,
            cols: u32,
            schema: Vec<String>,
            counter_rows: u32,
            counter_cols: u32,
            counter_type: String,
            counter_mode: String,
            bogus_field: u8, // key not in HydraMatrixMetadata
        }
        #[derive(Serialize)]
        struct WithoutSchema {
            metadata_version: u8,
            hash_profile_id: String,
            hash_algorithm: String,
            seed_derivation: String,
            input_encoding: String,
            seed_list: Vec<u64>,
            matrix_seed_index: u32,
            rows: u32,
            cols: u32,
            counter_rows: u32,
            counter_cols: u32,
            counter_type: String,
            counter_mode: String,
        }
        let m = hydra_matrix_metadata::<DefaultXxHasher>(
            &HydraGrid {
                rows: 3,
                cols: 8,
                schema: SCHEMA.iter().map(|s| s.to_string()).collect(),
            },
            2,
            16,
        );
        let extra = WithExtra {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            matrix_seed_index: m.matrix_seed_index,
            rows: m.rows,
            cols: m.cols,
            schema: m.schema.clone(),
            counter_rows: m.counter_rows,
            counter_cols: m.counter_cols,
            counter_type: m.counter_type.clone(),
            counter_mode: m.counter_mode.clone(),
            bogus_field: 7,
        };
        let without = WithoutSchema {
            metadata_version: m.metadata_version,
            hash_profile_id: m.hash_profile_id.clone(),
            hash_algorithm: m.hash_algorithm.clone(),
            seed_derivation: m.seed_derivation.clone(),
            input_encoding: m.input_encoding.clone(),
            seed_list: m.seed_list.clone(),
            matrix_seed_index: m.matrix_seed_index,
            rows: m.rows,
            cols: m.cols,
            counter_rows: m.counter_rows,
            counter_cols: m.counter_cols,
            counter_type: m.counter_type.clone(),
            counter_mode: m.counter_mode.clone(),
        };
        assert!(
            from_slice::<HydraMatrixMetadata>(&rmp_serde::to_vec_named(&extra).unwrap()).is_err(),
            "an unknown metadata key must be rejected"
        );
        assert!(
            from_slice::<HydraMatrixMetadata>(&rmp_serde::to_vec_named(&without).unwrap()).is_err(),
            "a missing required key must be rejected"
        );
    }

    /// A cell's inlined state is exactly the bytes that counter's own payload
    /// carries, so the envelope a decode rebuilds it from is the counter's own.
    #[test]
    fn hydra_cell_envelopes_mirror_the_counters_own_bytes() {
        let mut hll = HyperLogLog::<ErtlMLE>::default();
        hll.insert(&DataInput::U64(7));
        assert_eq!(
            hll_cell_envelope(hll.registers_as_slice()).expect("HLL cell"),
            hll.serialize_to_bytes().expect("HLL envelope")
        );

        let mut kll = KLL::default();
        for v in 0..500u64 {
            kll.update(&(v as f64));
        }
        let (state, bit_cache, remaining_bits) = kll.wire_coin();
        let cell = HydraKllCell {
            levels: kll.wire_levels(),
            items: kll.wire_items(),
            coin: HydraKllCoin {
                state,
                bit_cache,
                remaining_bits,
            },
        };
        assert_eq!(
            kll_cell_envelope(kll.wire_k(), kll.wire_m(), &cell).expect("KLL cell"),
            kll.serialize_to_bytes().expect("KLL envelope")
        );

        let mut um = UnivMon::init_univmon(4, 2, 16, 3);
        um.insert(&DataInput::U64(9), 3);
        let bytes = um.serialize_to_bytes().expect("UnivMon envelope");
        let (_, metadata, payload) = envelope::split(&bytes).expect("split");
        let meta: UnivMonMetadata = from_slice(metadata).expect("metadata");
        let shape = UnivMonShape {
            layer_size: meta.layer_size,
            sketch_row: meta.sketch_row,
            sketch_col: meta.sketch_col,
            heap_size: meta.heap_size,
        };
        assert_eq!(
            univmon_cell_envelope(&shape, &meta.key_type, payload).expect("UnivMon cell"),
            bytes
        );
    }

    // A test-only custom hasher: hashes exactly like `DefaultXxHasher` but
    // declares a DIFFERENT `HashProfile`.
    #[derive(Clone, Debug)]
    struct AltHasher;

    impl SketchHasher for AltHasher {
        type HashType = <DefaultXxHasher as SketchHasher>::HashType;

        fn hash64_seeded(d: usize, key: &DataInput) -> u64 {
            DefaultXxHasher::hash64_seeded(d, key)
        }
        fn hash128_seeded(d: usize, key: &DataInput) -> u128 {
            DefaultXxHasher::hash128_seeded(d, key)
        }
        fn hash_item64_seeded(d: usize, key: &HeapItem) -> u64 {
            DefaultXxHasher::hash_item64_seeded(d, key)
        }
        fn hash_item128_seeded(d: usize, key: &HeapItem) -> u128 {
            DefaultXxHasher::hash_item128_seeded(d, key)
        }
        fn hash_for_matrix_seeded(
            seed_idx: usize,
            rows: usize,
            cols: usize,
            key: &DataInput,
        ) -> Self::HashType {
            DefaultXxHasher::hash_for_matrix_seeded(seed_idx, rows, cols, key)
        }
    }

    impl HashProfile for AltHasher {
        const PROFILE_ID: &'static str = "test.alt.profile.v1";
        const ALGORITHM: &'static str = "xxh3_64_128";
        const SEED_DERIVATION: &'static str = "seed_list_index_wrap";
        const INPUT_ENCODING: &'static str = "projectasap.input.v1";
        fn seed_list() -> Vec<u64> {
            vec![1, 2, 3, 4, 5]
        }
        const CANONICAL_SEED_INDEX: u32 = CANONICAL_HASH_SEED as u32;
        const MATRIX_SEED_INDEX: u32 = 0;
    }

    /// Hydra hashes its subkeys through the crate default, so it has one
    /// truthful profile: (a) it emits that profile, (b) a custom-profile
    /// envelope is different bytes, and (c) decode fails closed on it.
    #[test]
    fn hydra_pins_its_hash_profile() {
        let hydra = cm_grid();
        let encoded = hydra.serialize_to_bytes().expect("serialize");
        let (_, metadata, payload) = envelope::split(&encoded).expect("split");
        let meta: HydraMatrixMetadata = from_slice(metadata).expect("metadata");
        assert_eq!(meta.hash_profile_id, DefaultXxHasher::PROFILE_ID);
        assert_eq!(meta.seed_list, DefaultXxHasher::seed_list());

        let alt = hydra_matrix_metadata::<AltHasher>(
            &HydraGrid {
                rows: meta.rows,
                cols: meta.cols,
                schema: meta.schema.clone(),
            },
            meta.counter_rows,
            meta.counter_cols,
        );
        let forged = envelope::encode(
            HYDRA_KIND_CM,
            &rmp_serde::to_vec_named(&alt).expect("metadata"),
            payload,
        );
        assert_ne!(forged, encoded);
        assert!(
            Hydra::deserialize_from_bytes(&forged).is_err(),
            "a custom-profile envelope must be rejected"
        );
    }
}
