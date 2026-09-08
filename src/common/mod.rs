//! Shared building blocks used across the library.
//!
//! This module groups the reusable primitives that power multiple sketch
//! implementations:
//!
//! - input and key types such as [`DataInput`], [`HeapItem`], and [`HHItem`]
//! - hashing abstractions such as [`SketchHasher`] and [`DefaultXxHasher`]
//! - storage backends such as [`Vector2D`] and [`MatrixStorage`]
//! - shared utilities such as heaps, numerical traits, and sampling helpers
//!
//! Most users only need [`DataInput`] and, for matrix-backed frequency
//! sketches, [`Vector2D`]. Advanced users can use this module to customize
//! hashing and storage layout or to integrate directly with lower-level shared
//! infrastructure.
//!
//! The precomputed tables and low-level helpers exposed here are public for
//! transparency and advanced use cases, but they are not required for ordinary
//! sketch usage.

/// Hashing utilities and seed definitions shared across sketches.
pub mod hash;
/// Cross-language hash spec (`HashSpec`/`derive_index`/`derive_sign`)
/// shared by matrix-backed sketches that need byte parity with
/// `sketchlib-go`.
pub mod hashspec;
pub mod heap;
pub mod input;
pub mod numerical;
pub mod precompute_hash;
pub mod precompute_sample;
pub mod precompute_sample2;
pub mod structure_utils;
/// Reusable storage backends and low-level containers.
pub mod structures;

pub use hash::{
    BOTTOM_LAYER_FINDER, CANONICAL_HASH_SEED, DefaultXxHasher, DigestBuildHasher, DigestHasher,
    HYDRA_SEED, HashProfile, MATRIX_MAX_ROWS, MatrixHashMode, SEEDLIST, SketchHasher,
    hash_for_matrix, hash_for_matrix_generic, hash_for_matrix_seeded,
    hash_for_matrix_seeded_generic, hash_for_matrix_seeded_with_mode,
    hash_for_matrix_seeded_with_mode_generic, hash_item64_seeded, hash_item128_seeded,
    hash_mode_for_matrix, hash64_seeded, hash128_seeded,
};
pub use heap::HHHeap;
pub use input::{DataInput, HHItem, HeapItem, L2HH, heap_item_to_sketch_input, input_to_owned};
pub use numerical::NumericalValue;
pub use precompute_hash::PRECOMPUTED_HASH;
pub use precompute_sample::PRECOMPUTED_SAMPLE;
pub use precompute_sample2::PRECOMPUTED_SAMPLE_RATE_1PERCENT;
pub use structure_utils::{Nitro, NitroContext, compute_median_inline_f64};
pub use structures::{
    BitMatrix, CommonHeap, CommonHeapOrder, DefaultMatrixI32, DefaultMatrixI64, DefaultMatrixI128,
    FastPathHasher, FixedMatrix, HllBucketList, HllBucketListP12, HllBucketListP14,
    HllBucketListP16, HllRegisterStorage, KeepLargest, KeepSmallest, MatrixFastHash,
    MatrixHashType, MatrixStorage, QuickMatrixI32, QuickMatrixI64, QuickMatrixI128, Vector1D,
    Vector2D, Vector3D,
};
