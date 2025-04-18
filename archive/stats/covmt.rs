use crate::{alias::PMatrixView, stats::StatsError, Fp, FP_EPSILON};
use derive_more::From;
use itertools::zip_eq;
use log::error;
use nalgebra::{
    allocator::{Allocator, Reallocator},
    DVector, DefaultAllocator, Dim, DimName, Dyn, Matrix, OMatrix, StorageMut, VecStorage,
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    ops::{Mul, MulAssign},
};

/// A data structure for holding a dynamically or statically sized covariance matrix.
///
/// This struct not only stores the covariance matrix itself, but also holds the lower triangular
/// matrix result from the Cholesky decomposition, the determinant and the covariance inverse.
/// These associated matrices are computed from a modified covariance matrix where any
/// zero-columns/rows are removed to handle constant model parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CovMatrix<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
    <DefaultAllocator as Allocator<D, D>>::Buffer<Fp>: for<'a> Deserialize<'a> + Serialize,
{
    /// The lower triangular matrix result the Cholesky decomposition.
    cholesky: OMatrix<Fp, D, D>,

    /// The matrix determinant.
    determinant: Fp,

    /// The inverse of the covariance matrix.
    inverse: OMatrix<Fp, D, D>,

    /// The covariance matrix itself.
    matrix: OMatrix<Fp, D, D>,
}

impl<D> CovMatrix<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
    <DefaultAllocator as Allocator<D, D>>::Buffer<Fp>: for<'a> Deserialize<'a> + Serialize,
{
    /// Compute the multivariate likelihood from two vectors with of length k x D.
    pub fn multivarate_likelihood(
        &self,
        x: impl IntoIterator<Item = Fp>,
        mu: impl IntoIterator<Item = Fp>,
    ) -> Result<Fp, StatsError> {
        let xminmu = DVector::from(zip_eq(x, mu).map(|(i, j)| i - j).collect::<Vec<Fp>>());
        let ndim = xminmu.len() / self.ndim();

        if xminmu.len() % self.ndim() != 0 {
            return Err(StatsError::DimensionMismatch((xminmu.len(), self.ndim())));
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

    /// Create a new [`CovMatrix`] object from a positive-definite matrix.
    pub fn from_matrix(matrix: OMatrix<Fp, D, D>) -> Result<Self, StatsError> {
        let (cholesky, determinant) = match matrix.clone().cholesky() {
            Some(value) => (value.l(), value.determinant()),
            None => {
                error!("failed to perform a cholesky decomposition: {}", matrix);
                return Err(StatsError::SingularMatrix(0.0));
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
            return Err(StatsError::SingularMatrix(determinant));
        }

        let inverse = match matrix.clone().try_inverse() {
            Some(value) => value,
            None => {
                error!("failed to invert matrix: {}", matrix);
                return Err(StatsError::SingularMatrix(determinant));
            }
        };

        Ok(CovMatrix {
            cholesky,
            determinant,
            inverse,
            matrix,
        })
    }

    /// Create a new [`CovMatrix`] object from an array of D-dimensional parameter vectors.
    ///
    /// This function properly handles constant parameters (resulting in empty columns/rows) and can also optionally use weights.
    pub fn from_particles<RStride: Dim, CStride: Dim>(
        particles: &PMatrixView<D, Dyn, RStride, CStride>,
        optional_weights: Option<&[Fp]>,
    ) -> Result<CovMatrix<D>, StatsError>
    where
        D: DimName,
        DefaultAllocator: Allocator<D>
            + Allocator<D, D, Buffer<Fp> = VecStorage<Fp, D, D>>
            + Reallocator<Fp, D, D, D, Dyn>,
        VecStorage<Fp, D, D>: StorageMut<Fp, D, D>,
    {
        let mut matrix = Matrix::<Fp, D, D, VecStorage<Fp, D, D>>::from_iterator(
            (0..(D::USIZE.pow(2))).map(|idx| {
                let jdx = idx / D::USIZE;
                let kdx = idx % D::USIZE;

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
            }),
        );

        matrix += matrix.transpose()
            - Matrix::<Fp, D, D, VecStorage<Fp, D, D>>::from_diagonal(&matrix.diagonal());

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
                return Err(StatsError::SingularMatrix(0.0));
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
            return Err(StatsError::SingularMatrix(determinant));
        }

        let inverse_red = match matrix_red.clone().try_inverse() {
            Some(value) => value,
            None => {
                error!("failed to invert matrix: {}", matrix_red);
                return Err(StatsError::SingularMatrix(determinant));
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
            Matrix::<Fp, D, D, VecStorage<Fp, D, D>>::from_iterator(inverse_rec.iter().cloned());

        let cholesky_proper: Matrix<f32, D, D, VecStorage<f32, D, D>> =
            Matrix::<Fp, D, D, VecStorage<Fp, D, D>>::from_iterator(cholesky_rec.iter().cloned());

        Ok(Self {
            matrix,
            inverse: inverse_proper,
            cholesky: cholesky_proper,
            determinant,
        })
    }
}

impl<D> Mul<Fp> for CovMatrix<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
    <DefaultAllocator as Allocator<D, D>>::Buffer<Fp>: for<'a> Deserialize<'a> + Serialize,
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

impl<D> Mul<CovMatrix<D>> for Fp
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
    <DefaultAllocator as Allocator<D, D>>::Buffer<Fp>: for<'a> Deserialize<'a> + Serialize,
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

impl<D> MulAssign<Fp> for CovMatrix<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
    <DefaultAllocator as Allocator<D, D>>::Buffer<Fp>: for<'a> Deserialize<'a> + Serialize,
{
    fn mul_assign(&mut self, rhs: Fp) {
        let dim = self.matrix.ncols() as i32;

        self.cholesky *= rhs.sqrt();
        self.determinant *= rhs.powi(dim);
        self.inverse /= rhs;
        self.matrix *= rhs;
    }
}

impl<D> TryFrom<OMatrix<Fp, D, D>> for CovMatrix<D>
where
    D: Dim,
    DefaultAllocator: Allocator<D, D>,
    <DefaultAllocator as Allocator<D, D>>::Buffer<Fp>: for<'a> Deserialize<'a> + Serialize,
{
    type Error = StatsError;

    fn try_from(value: OMatrix<Fp, D, D>) -> Result<Self, Self::Error> {
        Self::from_matrix(value)
    }
}

/// Computes the unbiased covariance over two slices.
pub fn covariance<'a, I>(x: I, y: I) -> Fp
where
    I: IntoIterator<Item = &'a Fp>,
    <I as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();

    let mu_x = x_iter.clone().sum::<Fp>();
    let mu_y = x_iter.clone().sum::<Fp>();

    let length = x_iter.clone().fold(0, |acc, _| acc + 1);

    zip_eq(x_iter, y_iter)
        .map(|(val_x, val_y)| (mu_x - val_x) * (mu_y - val_y))
        .sum::<Fp>()
        / (length - 1) as Fp
}

/// Computes the unbiased covariance over two slices.
pub fn covariance_with_weights<'a, I1, I2>(x: I1, y: I1, w: I2) -> Fp
where
    I1: IntoIterator<Item = &'a Fp>,
    I2: IntoIterator<Item = &'a Fp>,
    <I1 as IntoIterator>::IntoIter: Clone,
    <I2 as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();
    let w_iter = w.into_iter();

    let wsum = w_iter.clone().sum::<Fp>();
    let wsumsq = w_iter.clone().map(|val_w| val_w.powi(2)).sum::<Fp>();

    let wfac = wsum - wsumsq / wsum;

    let mu_x = zip_eq(x_iter.clone(), w_iter.clone())
        .map(|(val_x, val_w)| val_x * val_w)
        .sum::<Fp>()
        / wsum;
    let mu_y = zip_eq(y_iter.clone(), w_iter.clone())
        .map(|(val_y, val_w)| val_y * val_w)
        .sum::<Fp>()
        / wsum;

    zip_eq(x_iter, zip_eq(y_iter, w_iter))
        .map(|(val_x, (val_y, val_w))| val_w * (mu_x - val_x) * (mu_y - val_y))
        .sum::<Fp>()
        / wfac
}
