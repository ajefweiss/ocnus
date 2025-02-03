//! A collection of internal type aliases.

use crate::Fp;
use nalgebra::{Dyn, Matrix, VecStorage, ViewStorage, ViewStorageMut};

/// A dynamically sized parameter matrix representing an ensemble of P-dimensional parameter vectors.
pub type PMatrix<P> = Matrix<Fp, P, Dyn, VecStorage<Fp, P, Dyn>>;

/// A matrix view into a [`PMatrix`].
pub type PMatrixView<'a, R, C, DStride, CStride> =
    Matrix<Fp, R, C, ViewStorage<'a, Fp, R, C, DStride, CStride>>;

/// A mutable matrix view into a [`PMatrix`].
pub type PMatrixViewMut<'a, R, C, DStride, CStride> =
    Matrix<Fp, R, C, ViewStorageMut<'a, Fp, R, C, DStride, CStride>>;
