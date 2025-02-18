use std::cmp::Ordering;

use derive_more::Deref;
use itertools::zip_eq;
use log::error;
use nalgebra::{
    allocator::Allocator, DMatrix, DMatrixView, DVector, DefaultAllocator, Dim, MatrixView,
    OMatrix, SMatrix, SMatrixView, SVector,
};
use serde::{Deserialize, Serialize};

/// A trait that must be implemented for any type acting as a covariance matrix.
pub trait CovMatrix {
    type MatrixType;

    /// Returns a reference to the lower triangular matrix from the Cholesky decomposition.
    fn cholesky_ltm(&self) -> &Self::MatrixType;

    /// Returns a reference to the the inverse of the covariance matrix.
    fn inverse_matrix(&self) -> &Self::MatrixType;

    /// Returns a reference to the underlying covariance matrix.
    fn matrix(&self) -> &Self::MatrixType;

    /// Returns the number of dimensions of the covariance matrix.
    fn ndim(&self) -> usize;

    /// Returns the pseudo-determinant of the covariance matrix.
    fn pseudo_determinant(&self) -> f32;
}

/// A dynamically sized covariance matrix.
#[derive(Clone, Debug, Deref, Deserialize, Serialize)]
pub struct CovDMatrix {
    /// The  lower triangular matrix from the Cholesky decomposition.
    cholesky_ltm: DMatrix<f32>,

    /// The inverse of the covariance matrix.
    inverse_matrix: DMatrix<f32>,

    /// The underlying dynamically sized covariance matrix.
    #[deref]
    matrix: DMatrix<f32>,

    /// The pseudo-determinant of the covariance matrix.
    pseudo_determinant: f32,
}

impl CovDMatrix {
    /// Create a [`CovDMatrix`] from a positive-definite square matrix.
    fn from_matrix(matrix: &DMatrixView<f32>) -> Option<Self> {
        match from_matrix(matrix) {
            Some((cholesky_ltm, inverse_matrix, matrix, determinant)) => Some(Self {
                cholesky_ltm,
                inverse_matrix,
                matrix: matrix,
                pseudo_determinant: determinant,
            }),
            None => None,
        }
    }

    /// Compute the multivariate likelihood from two iterators over `f32`.
    /// The length of both iterators must be equal and also a multiple
    /// of the dimension of the covariance matrix (panic).
    pub fn multivarate_likelihood(
        &self,
        x: impl IntoIterator<Item = f32>,
        mu: impl IntoIterator<Item = f32>,
    ) -> f32 {
        let delta = DVector::from(zip_eq(x, mu).map(|(i, j)| i - j).collect::<Vec<f32>>());

        let ndim = delta.len() / self.ndim();

        if delta.len() % self.ndim() != 0 {
            panic!("iterator length must be a multiple of the covariance matrix dimension")
        }

        let mut lh =
            (self.pseudo_determinant.ln() + ndim as f32 * std::f32::consts::PI * 2.0) / 2.0;

        for idx in 0..ndim {
            let view = delta.rows_with_step(idx, self.ndim(), ndim - 1);

            lh -= (&view.transpose()
                * self
                    .inverse_matrix
                    .view((0, 0), (view.nrows(), view.nrows()))
                * view)[(0, 0)]
                / 2.0;
        }

        lh
    }
}

impl CovMatrix for CovDMatrix {
    type MatrixType = DMatrix<f32>;

    fn cholesky_ltm(&self) -> &Self::MatrixType {
        &self.cholesky_ltm
    }

    fn inverse_matrix(&self) -> &Self::MatrixType {
        &self.inverse_matrix
    }

    fn matrix(&self) -> &Self::MatrixType {
        &self.matrix
    }

    fn ndim(&self) -> usize {
        self.matrix.nrows()
    }

    fn pseudo_determinant(&self) -> f32 {
        self.pseudo_determinant
    }
}

impl TryFrom<&[f32]> for CovDMatrix {
    type Error = ();

    fn try_from(value: &[f32]) -> Result<Self, Self::Error> {
        match Self::from_matrix(
            &DMatrix::from_diagonal(&DVector::from_iterator(value.len(), value.iter().copied()))
                .as_view(),
        ) {
            Some(result) => Ok(result),
            None => Err(()),
        }
    }
}

impl<'a> TryFrom<&DMatrixView<'a, f32>> for CovDMatrix {
    type Error = ();

    fn try_from(value: &DMatrixView<'a, f32>) -> Result<Self, Self::Error> {
        match Self::from_matrix(value) {
            Some(result) => Ok(result),
            None => Err(()),
        }
    }
}

/// A statically sized covariance matrix.
#[derive(Clone, Debug, Deref, Deserialize, Serialize)]
pub struct CovSMatrix<const D: usize> {
    /// The  lower triangular matrix from the Cholesky decomposition.
    cholesky_ltm: SMatrix<f32, D, D>,

    /// The inverse of the covariance matrix.
    inverse_matrix: SMatrix<f32, D, D>,

    /// The underlying dynamically sized covariance matrix.
    #[deref]
    matrix: SMatrix<f32, D, D>,

    /// The pseudo-determinant of the covariance matrix.
    pseudo_determinant: f32,
}

impl<const D: usize> CovMatrix for CovSMatrix<D> {
    type MatrixType = SMatrix<f32, D, D>;

    fn cholesky_ltm(&self) -> &Self::MatrixType {
        &self.cholesky_ltm
    }

    fn inverse_matrix(&self) -> &Self::MatrixType {
        &self.inverse_matrix
    }

    fn matrix(&self) -> &Self::MatrixType {
        &self.matrix
    }

    fn ndim(&self) -> usize {
        self.matrix.nrows()
    }

    fn pseudo_determinant(&self) -> f32 {
        self.pseudo_determinant
    }
}

impl<const D: usize> CovSMatrix<D> {
    /// Create a [`CovSMatrix`] from a positive-definite square matrix.
    fn from_matrix(matrix: &SMatrixView<f32, D, D>) -> Option<Self> {
        match from_matrix(matrix) {
            Some((cholesky_ltm, inverse_matrix, matrix, determinant)) => Some(Self {
                cholesky_ltm,
                inverse_matrix,
                matrix: matrix,
                pseudo_determinant: determinant,
            }),
            None => None,
        }
    }
}

impl<const D: usize> TryFrom<f32> for CovSMatrix<D> {
    type Error = ();

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        match Self::from_matrix(&SMatrix::<f32, D, D>::from_diagonal_element(value).as_view()) {
            Some(result) => Ok(result),
            None => Err(()),
        }
    }
}

impl<const D: usize> TryFrom<[f32; D]> for CovSMatrix<D> {
    type Error = ();

    fn try_from(value: [f32; D]) -> Result<Self, Self::Error> {
        match Self::from_matrix(&SMatrix::from_diagonal(&SVector::from(value)).as_view()) {
            Some(result) => Ok(result),
            None => Err(()),
        }
    }
}

impl<'a, const D: usize> TryFrom<&SMatrixView<'a, f32, D, D>> for CovSMatrix<D> {
    type Error = ();

    fn try_from(value: &SMatrixView<'a, f32, D, D>) -> Result<Self, Self::Error> {
        match Self::from_matrix(value) {
            Some(result) => Ok(result),
            None => Err(()),
        }
    }
}

/// Compute the cholesky decomposition, the determinant and the inverse matrix from an owned square matrix view.
fn from_matrix<D>(
    matrix: &MatrixView<f32, D, D>,
) -> Option<(
    OMatrix<f32, D, D>,
    OMatrix<f32, D, D>,
    OMatrix<f32, D, D>,
    f32,
)>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
{
    let (cholesky_ltm, determinant) = match matrix.clone().cholesky() {
        Some(result) => (result.l(), result.determinant()),
        None => {
            error!("failed to perform the cholesky decomposition: {}", matrix);
            return None;
        }
    };

    if matches!(
        determinant.abs().partial_cmp(&f32::EPSILON).unwrap(),
        Ordering::Less
    ) {
        error!(
            "input matrix determinant is below precision threshold: {}",
            matrix
        );
        return None;
    }

    let inverse_matrix = match matrix.clone().try_inverse() {
        Some(result) => result,
        None => {
            error!("failed to invert matrix: {}", matrix);
            return None;
        }
    };

    Some((
        cholesky_ltm,
        inverse_matrix,
        matrix.into_owned(),
        determinant,
    ))
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use nalgebra::{SMatrix, U5};

//     #[test]
//     fn test_from_matrix() {
//         let sq_matrix_dynamic = DMatrix::<f32>::from_diagonal_element(5, 5, 1.0);
//         let sq_matrix_static = SMatrix::<f32, 5, 5>::from_diagonal_element(1.0);

//         let cov_dynamic = CovMatrix::<U5>::from_matrix(sq_matrix_dynamic);
//     }
// }
