use crate::{Fp, PVectorsView, FP_EPSILON};
use derive_more::From;
use itertools::zip_eq;
use log::error;
use nalgebra::{
    allocator::{Allocator, Reallocator},
    Const, DVector, DefaultAllocator, Dim, DimAdd, Dyn, OMatrix,
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    ops::{Mul, MulAssign},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CovMatrixError {
    #[error("array or matrix dimensions are mismatched")]
    DimensionMismatch((usize, usize)),
    #[error("matrix determinant is  zero or its absolute value is below the precision limit")]
    SingularMatrix(Fp),
}

/// A data structure for holding a dynamically or statically sized covariance matrix, its inverse, the cholesky decomposition (L matrix) and the determinant.
#[derive(Deserialize, Serialize)]
pub struct CovMatrix<const D: usize>
where
    Const<D>: Dim + DimAdd<Const<1>>,
    DefaultAllocator: Allocator<Const<D>>
        + Allocator<Const<1>, Const<D>>
        + Allocator<Const<D>, Const<D>>
        + Reallocator<Fp, Const<D>, Const<D>, Const<D>, Dyn>,
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
        for<'a> Deserialize<'a> + Serialize,
{
    /// The lower triangular matrix result the Cholesky decomposition.
    pub cholesky: OMatrix<Fp, Const<D>, Const<D>>,

    /// The matrix determinant.
    pub determinant: Fp,

    /// The inverse of the covariance matrix.
    pub inverse: OMatrix<Fp, Const<D>, Const<D>>,

    /// The covariance matrix itself.
    pub matrix: OMatrix<Fp, Const<D>, Const<D>>,
}

impl<const D: usize> CovMatrix<D>
where
    Const<D>: Dim + DimAdd<Const<1>>,
    DefaultAllocator: Allocator<Const<D>>
        + Allocator<Const<1>, Const<D>>
        + Allocator<Const<D>, Const<D>>
        + Reallocator<Fp, Const<D>, Const<D>, Const<D>, Dyn>,
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
        for<'a> Deserialize<'a> + Serialize,
{
    /// Compute the multivariate likelihood from two vectors with of length k x Const<D>.
    pub fn multivarate_likelihood<const N: usize>(
        &self,
        x: impl IntoIterator<Item = Fp>,
        mu: impl IntoIterator<Item = Fp>,
    ) -> Result<Fp, CovMatrixError> {
        let xminmu = DVector::from(zip_eq(x, mu).map(|(i, j)| i - j).collect::<Vec<Fp>>());
        let ndim = xminmu.len() / self.ndim();

        if xminmu.len() % self.ndim() != 0 {
            return Err(CovMatrixError::DimensionMismatch((
                xminmu.len(),
                self.ndim(),
            )));
        }

        let mut result =
            (self.determinant.ln() + ndim as Fp * std::f64::consts::PI as Fp * 2.0) / 2.0;

        for idx in 0..ndim {
            let view = xminmu.rows_with_step(idx, self.inverse.nrows(), ndim - 1);

            result -= (&view.transpose()
                * self.inverse.view((0, 0), (view.nrows(), view.nrows()))
                * view)[(0, 0)]
                / 2.0;
        }

        Ok(result)
    }

    /// Returns the dimensionality of the covariance matrix.
    pub fn ndim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Create a new [`Covariance`] object from a positive-definite matrix.
    fn new_from_matrix(matrix: OMatrix<Fp, Const<D>, Const<D>>) -> Result<Self, CovMatrixError> {
        let (cholesky, determinant) = match matrix.clone().cholesky() {
            Some(value) => (value.l(), value.determinant()),
            None => {
                error!("failed to perform a cholesky decomposition: {}", matrix);
                return Err(CovMatrixError::SingularMatrix(0.0));
            }
        };

        if matches!(
            determinant.abs().partial_cmp(&FP_EPSILON).unwrap(),
            Ordering::Less
        ) {
            error!(
                "matrix determinant is below precision threshold: {}",
                matrix
            );
            return Err(CovMatrixError::SingularMatrix(determinant));
        }

        let inverse = match matrix.clone().try_inverse() {
            Some(value) => value,
            None => {
                error!("failed to invert matrix: {}", matrix);
                return Err(CovMatrixError::SingularMatrix(determinant));
            }
        };

        Ok(CovMatrix {
            cholesky,
            determinant,
            inverse,
            matrix,
        })
    }

    /// Create a new [`Covariance`] object from an array of D-dimensional parameter vectors.
    ///
    /// This function properly handles constant parameters (resulting in empty columns/rows) and can also optionally use weighted vectors.
    fn new_from_particles(
        particles: PVectorsView<D, Dyn>,
        optional_weights: Option<&[Fp]>,
    ) -> Result<CovMatrix<D>, CovMatrixError> {
        let mut matrix =
            OMatrix::<Fp, Const<D>, Const<D>>::from_iterator((0..(D * D)).map(|idx| {
                let jdx = idx / D;
                let kdx = idx % D;

                if jdx <= kdx {
                    let x = particles.row(jdx);
                    let y = particles.row(kdx);

                    match optional_weights {
                        Some(w) => covariance_with_weights(x, y, w),
                        None => covariance(x, y),
                    }
                } else {
                    0.0
                }
            }));

        matrix += matrix.transpose()
            - OMatrix::<Fp, Const<D>, Const<D>>::from_diagonal(&matrix.diagonal());

        // Remove vanishing columns/rows.
        let remove_indices = (0..matrix.nrows())
            .filter(|idx| {
                matches!(
                    matrix[(*idx, *idx)]
                        .partial_cmp(&(512.0 * f32::EPSILON))
                        .expect("matrix has NaN values"),
                    Ordering::Less
                )
            })
            .collect::<Vec<usize>>();

        let matrix_copy = matrix.clone();

        // Remove the empty columns and rows.
        let matrix_red = matrix_copy
            .remove_columns_at(&remove_indices)
            .remove_rows_at(&remove_indices);

        let (cholesky_red, determinant) = match matrix_red.clone().cholesky() {
            Some(value) => (value.l(), value.determinant()),
            None => {
                error!("failed to perform a cholesky decomposition: {}", matrix_red);
                return Err(CovMatrixError::SingularMatrix(0.0));
            }
        };

        if matches!(
            determinant.abs().partial_cmp(&FP_EPSILON).unwrap(),
            Ordering::Less
        ) {
            error!(
                "matrix determinant is below precision threshold: {}",
                matrix_red
            );
            return Err(CovMatrixError::SingularMatrix(determinant));
        }

        let inverse_red = match matrix_red.clone().try_inverse() {
            Some(value) => value,
            None => {
                error!("failed to invert matrix: {}", matrix_red);
                return Err(CovMatrixError::SingularMatrix(determinant));
            }
        };

        let mut inverse_rec = inverse_red.clone();
        let mut cholesky_rec = cholesky_red.clone();

        // Re-insert empty columns/rows.
        for idx in remove_indices.iter() {
            inverse_rec = inverse_rec.insert_column(*idx, 0.0).insert_row(*idx, 0.0);
            cholesky_rec = cholesky_rec.insert_column(*idx, 0.0).insert_row(*idx, 0.0);
        }

        let inverse_proper =
            OMatrix::<f32, Const<D>, Const<D>>::from_iterator(inverse_rec.iter().cloned());
        let cholesky_proper =
            OMatrix::<f32, Const<D>, Const<D>>::from_iterator(cholesky_rec.iter().cloned());

        Ok(Self {
            matrix,
            inverse: inverse_proper,
            cholesky: cholesky_proper,
            determinant,
        })
    }
}

impl<const D: usize> Mul<Fp> for CovMatrix<D>
where
    Const<D>: Dim + DimAdd<Const<1>>,
    DefaultAllocator: Allocator<Const<D>>
        + Allocator<Const<1>, Const<D>>
        + Allocator<Const<D>, Const<D>>
        + Reallocator<f32, Const<D>, Const<D>, Const<D>, Dyn>,
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
        for<'a> Deserialize<'a> + Serialize,
{
    type Output = CovMatrix<D>;

    fn mul(self, rhs: Fp) -> Self::Output {
        let dim = self.matrix.ncols() as i32;

        Self::Output {
            cholesky: self.cholesky * rhs.sqrt(),
            determinant: self.determinant * rhs.powi(dim),
            inverse: self.inverse / rhs,
            matrix: self.matrix * rhs,
        }
    }
}

impl<const D: usize> Mul<CovMatrix<D>> for Fp
where
    Const<D>: Dim + DimAdd<Const<1>>,
    DefaultAllocator: Allocator<Const<D>>
        + Allocator<Const<1>, Const<D>>
        + Allocator<Const<D>, Const<D>>
        + Reallocator<Fp, Const<D>, Const<D>, Const<D>, Dyn>,
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
        for<'a> Deserialize<'a> + Serialize,
{
    type Output = CovMatrix<D>;
    fn mul(self, rhs: CovMatrix<D>) -> Self::Output {
        let dim = rhs.matrix.ncols() as i32;

        Self::Output {
            cholesky: rhs.cholesky * self.sqrt(),
            determinant: rhs.determinant * self.powi(dim),
            inverse: rhs.inverse / self,
            matrix: rhs.matrix * self,
        }
    }
}

impl<const D: usize> MulAssign<Fp> for CovMatrix<D>
where
    Const<D>: Dim + DimAdd<Const<1>>,
    DefaultAllocator: Allocator<Const<D>>
        + Allocator<Const<1>, Const<D>>
        + Allocator<Const<D>, Const<D>>
        + Reallocator<Fp, Const<D>, Const<D>, Const<D>, Dyn>,
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
        for<'a> Deserialize<'a> + Serialize,
{
    fn mul_assign(&mut self, rhs: Fp) {
        let dim = self.matrix.ncols() as i32;

        self.cholesky *= rhs.sqrt();
        self.determinant *= rhs.powi(dim);
        self.inverse /= rhs;
        self.matrix *= rhs;
    }
}

impl<const D: usize> TryFrom<OMatrix<Fp, Const<D>, Const<D>>> for CovMatrix<D>
where
    Const<D>: Dim + DimAdd<Const<1>>,
    DefaultAllocator: Allocator<Const<D>>
        + Allocator<Const<1>, Const<D>>
        + Allocator<Const<D>, Const<D>>
        + Reallocator<Fp, Const<D>, Const<D>, Const<D>, Dyn>,
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
        for<'a> Deserialize<'a> + Serialize,
{
    type Error = CovMatrixError;

    fn try_from(value: OMatrix<Fp, Const<D>, Const<D>>) -> Result<Self, Self::Error> {
        Self::new_from_matrix(value)
    }
}

/// Computes the unbiased covariance over two slices.
pub fn covariance<'a>(
    x: impl Clone + IntoIterator<Item = &'a Fp>,
    y: impl Clone + IntoIterator<Item = &'a Fp>,
) -> Fp {
    let mu_x = x.clone().into_iter().sum::<Fp>();
    let mu_y = y.clone().into_iter().sum::<Fp>();

    let length = x.clone().into_iter().fold(0, |acc, _| acc + 1);

    zip_eq(x, y)
        .map(|(val_x, val_y)| (mu_x - val_x) * (mu_y - val_y))
        .sum::<Fp>()
        / (length - 1) as Fp
}

/// Computes the unbiased covariance over two slices.
pub fn covariance_with_weights<'a>(
    x: impl Clone + IntoIterator<Item = &'a Fp>,
    y: impl Clone + IntoIterator<Item = &'a Fp>,
    w: impl Clone + IntoIterator<Item = &'a Fp>,
) -> Fp {
    let wsum = w.clone().into_iter().sum::<Fp>();
    let wsumsq = w.clone().into_iter().map(|val_w| val_w.powi(2)).sum::<Fp>();

    let wfac = wsum - wsumsq / wsum;

    let mu_x = zip_eq(x.clone(), w.clone())
        .map(|(val_x, val_w)| val_x * val_w)
        .sum::<Fp>()
        / wsum;
    let mu_y = zip_eq(y.clone(), w.clone())
        .map(|(val_y, val_w)| val_y * val_w)
        .sum::<Fp>()
        / wsum;

    zip_eq(x.clone(), zip_eq(y.clone(), w))
        .map(|(val_x, (val_y, val_w))| val_w * (mu_x - val_x) * (mu_y - val_y))
        .sum::<Fp>()
        / wfac
}
