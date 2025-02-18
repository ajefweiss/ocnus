use crate::fevms::PMatrix;
use core::f32;
use derive_more::Deref;
use itertools::zip_eq;
use log::error;
use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix, SquareMatrix, ViewStorage};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    ops::{Mul, MulAssign},
};

/// A dynamically sized covariance matrix.
///
/// For convenience, this data structure also holds the inverse matrix `imat`, the matrix determinant `detv` and
/// lower triangular matrix from the Cholesky decomposition `cltm` which can be accessed using the
/// appropriate getter functions. These associated values are not directly computed from
/// the covariance matrix but a modification thereof in which 0-rows/columns are taken into
/// account so that one can properly handle constant model parameters.
#[derive(Clone, Debug, Deref, Deserialize, Serialize)]
pub struct CovMatrix {
    /// The lower triangular matrix result from the Cholesky decomposition.
    cltm: DMatrix<f32>,

    /// The matrix detv.
    detv: f32,

    /// The inverse of the covariance matrix.
    imat: DMatrix<f32>,

    /// The covariance matrix itself.
    #[deref]
    cmat: DMatrix<f32>,
}

impl CovMatrix {
    /// Returns the lower triangular matrix L from the Cholesky decomposition.
    ///
    /// This function is a getter function and performs no calculations.
    pub fn ltm(&self) -> &DMatrix<f32> {
        &self.cltm
    }

    /// Returns the determinant of the covariance matrix.
    ///
    /// This function is a getter function and performs no calculations.
    pub fn determinant(&self) -> f32 {
        self.detv
    }

    /// Returns the inverse of the covariance matrix.
    ///
    /// This function is a getter function and performs no calculations.
    pub fn inverse(&self) -> &DMatrix<f32> {
        &self.imat
    }

    /// Compute the multivariate likelihood from two iterators over `f32`.
    /// The length of both iterators must be equal and also a multiple
    /// of the dimension of the covariance matrix (panic).
    pub fn multivarate_likelihood(
        &self,
        x: impl IntoIterator<Item = f32>,
        mu: impl IntoIterator<Item = f32>,
    ) -> Option<f32> {
        let xminmu = DVector::from(zip_eq(x, mu).map(|(i, j)| i - j).collect::<Vec<f32>>());
        let ndim = xminmu.len() / self.ndim();

        if xminmu.len() % self.ndim() != 0 {
            panic!("iterator length must be a multiple of the covariance matrix dimension")
        }

        let mut lh = (self.detv.ln() + ndim as f32 * std::f32::consts::PI * 2.0) / 2.0;

        for idx in 0..ndim {
            let view = xminmu.rows_with_step(idx, self.imat.nrows(), ndim - 1);

            lh -= (&view.transpose() * self.imat.view((0, 0), (view.nrows(), view.nrows())) * view)
                [(0, 0)]
                / 2.0;
        }

        Some(lh)
    }

    /// Returns the number of dimensions of the covariance matrix.
    pub fn ndim(&self) -> usize {
        self.cmat.nrows()
    }

    /// Create a [`CovMatrix`] from a positive-definite square matrix.
    pub fn from_matrix(matrix: SquareMatrix<f32, D, S>) -> Option<Self> {
        let (cltm, detv) = match matrix.clone().cholesky() {
            Some(result) => (result.l(), result.determinant()),
            None => {
                error!("failed to perform a cholesky decomposition: {}", matrix);
                return None;
            }
        };

        if matches!(
            detv.abs().partial_cmp(&f32::EPSILON).unwrap(),
            Ordering::Less
        ) {
            error!(
                "input matrix determinant is below precision threshold: {}",
                matrix
            );
            return None;
        }

        let imat = match matrix.clone().try_inverse() {
            Some(result) => result,
            None => {
                error!("failed to invert matrix: {}", matrix);
                return None;
            }
        };

        Some(CovMatrix {
            cltm,
            detv,
            imat,
            cmat: matrix,
        })
    }

    /// Create a [`CovMatrix`] from an array of P-dimensional parameter vectors.
    ///
    /// This function properly handles constant parameters (resulting in empty columns/rows) and can optionally also use weights.
    pub fn from_particles<const P: usize>(
        particles: &PMatrix<P>,
        opt_weights: Option<&[f32]>,
    ) -> Option<CovMatrix> {
        let mut cmat = DMatrix::from_iterator(
            P,
            P,
            (0..(P.pow(2))).map(|idx| {
                let jdx = idx / P;
                let kdx = idx % P;

                if jdx <= kdx {
                    let x = particles.row(jdx);
                    let y = particles.row(kdx);

                    match opt_weights {
                        Some(w) => covariance_with_weights(x, y, w),
                        None => covariance(x, y),
                    }
                } else {
                    0.0
                }
            }),
        );

        // Fill up the other side of the matrix.
        cmat += cmat.transpose() - DMatrix::<f32>::from_diagonal(&cmat.diagonal());

        // Detect columns/rows that need to be modified using alternate sums.
        // Normally we would select any columns/rows with zero variance
        // but floating-point arithmetic makes this unreliable.
        let zero_variance_indices = (0..cmat.nrows())
            .filter(|idx| {
                let row = particles.row(*idx);

                let pos = &row.columns_with_step(0, particles.ncols() / 2, 1);
                let neg: &Matrix<
                    f32,
                    Const<1>,
                    Dyn,
                    ViewStorage<'_, f32, Const<1>, Dyn, Const<1>, Dyn>,
                > = &row.columns_with_step(1, particles.ncols() / 2, 1);

                let altsum = (pos - neg).iter().sum::<f32>() / (particles.ncols() / 2) as f32;

                if altsum.abs() < f32::EPSILON {
                    cmat.column_mut(*idx).iter_mut().for_each(|cov| *cov = 0.0);
                    cmat.row_mut(*idx).iter_mut().for_each(|cov| *cov = 0.0);

                    cmat[(*idx, *idx)] = 1.0;

                    true
                } else {
                    false
                }
            })
            .collect::<Vec<usize>>();

        let (mut cltm, detv) = match cmat.clone().cholesky() {
            Some(result) => (result.l(), result.determinant()),
            None => {
                error!("failed to perform a cholesky decomposition: {}", cmat);
                return None;
            }
        };

        if matches!(
            detv.abs().partial_cmp(&f32::EPSILON).unwrap(),
            Ordering::Less
        ) {
            error!("matrix detv is below precision threshold: {}", cmat);
            return None;
        }

        let mut imat = match cmat.clone().try_inverse() {
            Some(result) => result,
            None => {
                error!("failed to invert matrix: {}", cmat);
                return None;
            }
        };

        // Reset zero variance columns/rows to zero.
        for idx in zero_variance_indices.iter() {
            cmat[(*idx, *idx)] = 0.0;
            imat[(*idx, *idx)] = 0.0;
            cltm[(*idx, *idx)] = 0.0;
        }

        Some(Self {
            cmat,
            imat,
            cltm,
            detv,
        })
    }
}

impl Mul<f32> for CovMatrix {
    type Output = CovMatrix;

    fn mul(self, rhs: f32) -> Self::Output {
        let dim = self.cmat.ncols() as i32;

        Self::Output {
            cltm: self.cltm * rhs.sqrt(),
            detv: self.detv * rhs.powi(dim),
            imat: self.imat / rhs,
            cmat: self.cmat * rhs,
        }
    }
}

impl Mul<CovMatrix> for f32 {
    type Output = CovMatrix;
    fn mul(self, rhs: CovMatrix) -> Self::Output {
        let dim = rhs.cmat.ncols() as i32;

        Self::Output {
            cltm: rhs.cltm * self.sqrt(),
            detv: rhs.detv * self.powi(dim),
            imat: rhs.imat / self,
            cmat: rhs.cmat * self,
        }
    }
}

impl MulAssign<f32> for CovMatrix {
    fn mul_assign(&mut self, rhs: f32) {
        let dim = self.cmat.ncols() as i32;

        self.cltm *= rhs.sqrt();
        self.detv *= rhs.powi(dim);
        self.imat /= rhs;
        self.cmat *= rhs;
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
///
/// The length of both iterators must be equal (panic).
pub fn covariance<'a, I>(x: I, y: I) -> f32
where
    I: IntoIterator<Item = &'a f32>,
    <I as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();

    let length = x_iter.clone().fold(0, |acc, _| acc + 1);

    let mu_x = x_iter.clone().sum::<f32>() / length as f32;
    let mu_y = x_iter.clone().sum::<f32>() / length as f32;

    zip_eq(x_iter, y_iter)
        .map(|(val_x, val_y)| (mu_x - val_x) * (mu_y - val_y))
        .sum::<f32>()
        / (length - 1) as f32
}

/// Computes the unbiased covariance over two slices with weights.
///
/// The length of all three iterators must be equal (panic).
pub fn covariance_with_weights<'a, IV, IW>(x: IV, y: IV, w: IW) -> f32
where
    IV: IntoIterator<Item = &'a f32>,
    IW: IntoIterator<Item = &'a f32>,
    <IV as IntoIterator>::IntoIter: Clone,
    <IW as IntoIterator>::IntoIter: Clone,
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

#[cfg(test)]
mod tests {
    use nalgebra::{VecStorage, U3};
    use rand::{Rng, SeedableRng};
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::*;

    #[test]
    fn test_covariance() {
        assert!(
            covariance(
                &[10.0, 34.0, 23.0, 54.0, 9.0],
                &[4.0, 5.0, 11.0, 15.0, 20.0]
            ) == 5.75
        );

        assert!(
            (covariance_with_weights(
                &[10.0, 34.0, 23.0, 54.0, 9.0],
                &[4.0, 5.0, 11.0, 15.0, 20.0],
                &[1.0, 0.8, 1.1, 1.3, 0.9]
            ) - 19.523237)
                .abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn test_covmat() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = Uniform::new(-0.5, 0.5).unwrap();

        let array = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10,
            (0..30).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    (idx % 3) as f32 + rng.sample(uniform)
                }
            }),
        );

        let covm = CovMatrix::from_particles(&array, None).unwrap();

        assert!((covm.cltm[(0, 0)] - 0.40718567).abs() < f32::EPSILON);
        assert!((covm.cltm[(2, 0)] - 0.07841061).abs() < f32::EPSILON);

        assert!((covm.imat[(0, 0)] - 6.915894).abs() < f32::EPSILON);
        assert!((covm.imat[(2, 0)] + 4.5933948).abs() < f32::EPSILON);

        assert!((covm.detv() - 0.0069507807).abs() < f32::EPSILON);
    }
}
