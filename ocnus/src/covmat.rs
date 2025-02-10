use core::f32;
use itertools::zip_eq;
use log::error;
use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix, MatrixView};
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
/// zero-columns/rows are modified to handle constant model parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CovMatrix {
    /// The lower triangular matrix result the Cholesky decomposition.
    cholesky_ltm: DMatrix<f32>,

    /// The matrix determinant.
    determinant: f32,

    /// The inverse of the covariance matrix.
    inverse: DMatrix<f32>,

    /// The covariance matrix itself.
    matrix: DMatrix<f32>,
}

impl CovMatrix {
    /// Returns the lower triangular matrix of the Cholesky decomposition.
    pub fn cholesky_ltm(&self) -> &DMatrix<f32> {
        &self.cholesky_ltm
    }

    /// Returns the determinant of the covariance matrix.
    pub fn determinant(&self) -> f32 {
        self.determinant
    }

    /// Returns the inverse of the covariance matrix.
    pub fn inverse(&self) -> &DMatrix<f32> {
        &self.inverse
    }

    /// Compute the multivariate likelihood from two vectors with of length k x D.
    pub fn multivarate_likelihood(
        &self,
        x: impl IntoIterator<Item = f32>,
        mu: impl IntoIterator<Item = f32>,
    ) -> f32 {
        let xminmu = DVector::from(zip_eq(x, mu).map(|(i, j)| i - j).collect::<Vec<f32>>());
        let ndim = xminmu.len() / self.ndim();

        let mut lh =
            (self.determinant.ln() + ndim as f32 * std::f64::consts::PI as f32 * 2.0) / 2.0;

        for idx in 0..ndim {
            let view = xminmu.rows_with_step(idx, self.inverse.nrows(), ndim - 1);

            lh -= (&view.transpose()
                * self.inverse.view((0, 0), (view.nrows(), view.nrows()))
                * view)[(0, 0)]
                / 2.0;
        }

        lh
    }

    /// Returns the dimensionality of the covariance matrix.
    pub fn ndim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Create a [`CovMatrix`] from a positive-definite matrix.
    pub fn from_matrix(matrix: DMatrix<f32>) -> Option<Self> {
        let (cholesky_ltm, determinant) = match matrix.clone().cholesky() {
            Some(value) => (value.l(), value.determinant()),
            None => {
                error!("failed to perform a cholesky decomposition: {}", matrix);
                return None;
            }
        };

        if matches!(
            determinant.abs().partial_cmp(&f32::EPSILON).unwrap(),
            Ordering::Less
        ) {
            error!(
                "matrix determinant is below precision threshold: {}",
                matrix
            );
            return None;
        }

        let inverse = match matrix.clone().try_inverse() {
            Some(value) => value,
            None => {
                error!("failed to invert matrix: {}", matrix);
                return None;
            }
        };

        Some(CovMatrix {
            cholesky_ltm,
            determinant,
            inverse,
            matrix,
        })
    }

    /// Create a [`CovMatrix`] from an array of P-dimensional parameter vectors.
    ///
    /// This function properly handles constant parameters (resulting in empty columns/rows) and can also optionally use weights.
    pub fn from_particles<const P: usize>(
        particles: &Matrix<f32, Const<P>, Dyn, nalgebra::VecStorage<f32, Const<P>, Dyn>>,
        optional_weights: Option<&[f32]>,
    ) -> Option<CovMatrix> {
        let mut matrix = DMatrix::from_iterator(
            P,
            P,
            (0..(P.pow(2))).map(|idx| {
                let jdx = idx / P;
                let kdx = idx % P;

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

        matrix += matrix.transpose() - DMatrix::<f32>::from_diagonal(&matrix.diagonal());

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
                return None;
            }
        };

        if matches!(
            determinant.abs().partial_cmp(&f32::EPSILON).unwrap(),
            Ordering::Less
        ) {
            error!(
                "matrix determinant is below precision threshold: {}",
                matrix_red
            );
            return None;
        }

        let inverse_red = match matrix_red.clone().try_inverse() {
            Some(value) => value,
            None => {
                error!("failed to invert matrix: {}", matrix_red);
                return None;
            }
        };

        let mut inverse_rec = inverse_red.clone();
        let mut cholesky_rec = cholesky_red.clone();

        // Re-insert empty columns/rows.
        for idx in remove_indices.iter() {
            inverse_rec = inverse_rec.insert_column(*idx, 0.0).insert_row(*idx, 0.0);
            cholesky_rec = cholesky_rec.insert_column(*idx, 0.0).insert_row(*idx, 0.0);
        }

        Some(Self {
            matrix,
            inverse: inverse_rec,
            cholesky_ltm: cholesky_rec,
            determinant,
        })
    }
}

impl Mul<f32> for CovMatrix {
    type Output = CovMatrix;

    fn mul(self, rhs: f32) -> Self::Output {
        let dim = self.matrix.ncols() as i32;

        Self::Output {
            cholesky: self.cholesky_ltm * rhs.sqrt(),
            determinant: self.determinant * rhs.powi(dim),
            inverse: self.inverse / rhs,
            matrix: self.matrix * rhs,
        }
    }
}

impl Mul<CovMatrix> for f32 {
    type Output = CovMatrix;
    fn mul(self, rhs: CovMatrix) -> Self::Output {
        let dim = rhs.matrix.ncols() as i32;

        Self::Output {
            cholesky: rhs.cholesky * self.sqrt(),
            determinant: rhs.determinant * self.powi(dim),
            inverse: rhs.inverse / self,
            matrix: rhs.matrix * self,
        }
    }
}

impl MulAssign<f32> for CovMatrix {
    fn mul_assign(&mut self, rhs: f32) {
        let dim = self.matrix.ncols() as i32;

        self.cholesky_ltm *= rhs.sqrt();
        self.determinant *= rhs.powi(dim);
        self.inverse /= rhs;
        self.matrix *= rhs;
    }
}

impl TryFrom<DMatrix<f32>> for CovMatrix {
    type Error = ();

    fn try_from(value: DMatrix<f32>) -> Result<Self, Self::Error> {
        match Self::from_matrix(value) {
            Some(covm) => Ok(covm),
            None => Err(()),
        }
    }
}

/// Computes the unbiased covariance over two slices.
pub fn covariance<'a, I>(x: I, y: I) -> f32
where
    I: IntoIterator<Item = &'a f32>,
    <I as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();

    let mu_x = x_iter.clone().sum::<f32>();
    let mu_y = x_iter.clone().sum::<f32>();

    let length = x_iter.clone().fold(0, |acc, _| acc + 1);

    zip_eq(x_iter, y_iter)
        .map(|(val_x, val_y)| (mu_x - val_x) * (mu_y - val_y))
        .sum::<f32>()
        / (length - 1) as f32
}

/// Computes the unbiased covariance over two slices.
pub fn covariance_with_weights<'a, I1, I2>(x: I1, y: I1, w: I2) -> f32
where
    I1: IntoIterator<Item = &'a f32>,
    I2: IntoIterator<Item = &'a f32>,
    <I1 as IntoIterator>::IntoIter: Clone,
    <I2 as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();
    let w_iter = w.into_iter();

    let wsum = w_iter.clone().sum::<f32>();
    let wsumsq = w_iter.clone().map(|val_w| val_w.powi(2)).sum::<f32>();

    let wfac = wsum - wsumsq / wsum;

    let mu_x = zip_eq(x_iter.clone(), w_iter.clone())
        .map(|(val_x, val_w)| val_x * val_w)
        .sum::<f32>()
        / wsum;
    let mu_y = zip_eq(y_iter.clone(), w_iter.clone())
        .map(|(val_y, val_w)| val_y * val_w)
        .sum::<f32>()
        / wsum;

    zip_eq(x_iter, zip_eq(y_iter, w_iter))
        .map(|(val_x, (val_y, val_w))| val_w * (mu_x - val_x) * (mu_y - val_y))
        .sum::<f32>()
        / wfac
}
