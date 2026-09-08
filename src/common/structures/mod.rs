/// One-dimensional vector wrapper used by sketches.
pub mod vector1d;
pub use vector1d::Vector1D;

/// Two-dimensional matrix wrapper used by sketches.
pub mod vector2d;
pub use vector2d::Vector2D;

/// Three-dimensional storage wrapper used by layered sketches.
pub mod vector3d;
pub use vector3d::Vector3D;

/// Generic heap implementations and order policies.
pub mod heap;
pub use heap::{CommonHeap, CommonHeapOrder, KeepLargest, KeepSmallest};

pub mod matrix_storage;
pub use matrix_storage::{FastPathHasher, MatrixFastHash, MatrixHashType, MatrixStorage};

/// Packed single-bit grid behind the `MatrixStorage` interface.
pub mod bit_matrix;
pub use bit_matrix::BitMatrix;

pub mod fixed_structure;
pub use fixed_structure::{
    DefaultMatrixI32, DefaultMatrixI64, DefaultMatrixI128, FixedMatrix, HllBucketList,
    HllBucketListP12, HllBucketListP14, HllBucketListP16, HllRegisterStorage, QuickMatrixI32,
    QuickMatrixI64, QuickMatrixI128,
};
