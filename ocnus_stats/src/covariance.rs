use derive_more::Deref;
use itertools::{Itertools, zip_eq};
use log::error;
use nalgebra::{Const, DMatrix, DMatrixView, DVector, Dyn, MatrixView, RealField, Scalar};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Mul, MulAssign},
};

/// A dynamically sized covariance matrix.
#[derive(Clone, Debug, Deref, Deserialize, Serialize)]
pub struct CovMatrix<T>
where
    T: Scalar,
{
    /// The lower triangular matrix from the Cholesky decomposition.
    cholesky_ltm: DMatrix<T>,

    /// The inverse of the covariance matrix.
    matrix_inverse: DMatrix<T>,

    /// The covariance matrix.
    #[deref]
    matrix: DMatrix<T>,

    /// The pseudo-determinant of the covariance matrix.
    determinant: T,
}

impl<T> CovMatrix<T>
where
    T: Copy + RealField + Scalar,
{
    /// Create a [`CovMatrix`] from a semi positive definite square matrix.
    pub fn from_matrix(matrix: &DMatrixView<T>) -> Option<Self> {
        let mut matrix_owned = matrix.into_owned();

        // Detect zero'd columns/rows that need to be modified.
        let zero_variance_indices = (0..matrix_owned.nrows())
            .filter(|idx| {
                match matrix_owned[(*idx, *idx)].eq(&T::zero()) {
                    true => {
                        matrix_owned[(*idx, *idx)] = T::one();

                        // Set off diagonals to zero.
                        for jdx in 0..matrix.ncols() {
                            if jdx != *idx {
                                matrix_owned[(*idx, jdx)] = T::zero();
                                matrix_owned[(jdx, *idx)] = T::zero();
                            }
                        }

                        true
                    }
                    false => false,
                }
            })
            .collect::<Vec<usize>>();

        let (mut cholesky_ltm, determinant) = match matrix_owned.clone().cholesky() {
            Some(result) => (result.l(), result.determinant()),
            None => {
                error!(
                    "CovMatrix::from_matrix: failed to perform the cholesky decomposition: {}",
                    matrix
                );
                return None;
            }
        };

        let mut matrix_inverse = match matrix_owned.clone().try_inverse() {
            Some(result) => result,
            None => {
                error!(
                    "CovMatrix::from_matrix input matrix is singular: {}",
                    matrix
                );
                return None;
            }
        };

        // Reset zero variance columns/rows to zero.
        for idx in zero_variance_indices.iter() {
            cholesky_ltm[(*idx, *idx)] = T::zero();
            matrix_inverse[(*idx, *idx)] = T::zero();
            matrix_owned[(*idx, *idx)] = T::zero();
        }

        Some(Self {
            cholesky_ltm,
            matrix_inverse,
            matrix: matrix_owned,
            determinant,
        })
    }

    /// Create a [`CovMatrix`] from an ensemble of particles.
    pub fn from_particles<const N: usize>(
        particles: &MatrixView<T, Const<N>, Dyn>,
        opt_weights: Option<&[T]>,
    ) -> Option<Self>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let mut matrix = DMatrix::from_iterator(
            N,
            N,
            (0..(N.pow(2))).map(|idx| {
                let jdx = idx / N;
                let kdx = idx % N;

                if jdx <= kdx {
                    let x = particles.row(jdx);
                    let y = particles.row(kdx);

                    match opt_weights {
                        Some(w) => covariance_with_weights(x, y, w),
                        None => covariance(x, y),
                    }
                } else {
                    T::zero()
                }
            }),
        );

        // Fill up the other side of the matrix.
        matrix += matrix.transpose() - DMatrix::<T>::from_diagonal(&matrix.diagonal());

        // Detect zero'd columns/rows that need to be modified.
        let zero_variance_indices = (0..particles.nrows())
            .filter(|idx| {
                let row = particles.row(*idx);

                match row.iter().all_equal() {
                    true => {
                        matrix[(*idx, *idx)] = T::one();

                        // Set off diagonals to zero.
                        for jdx in 0..matrix.ncols() {
                            if jdx != *idx {
                                matrix[(*idx, jdx)] = T::zero();
                                matrix[(jdx, *idx)] = T::zero();
                            }
                        }

                        true
                    }
                    false => false,
                }
            })
            .collect::<Vec<usize>>();

        let mut result = Self::from_matrix(&matrix.as_view())?;

        // Reset zero variance columns/rows to zero.
        for idx in zero_variance_indices.iter() {
            result.cholesky_ltm[(*idx, *idx)] = T::zero();
            result.matrix_inverse[(*idx, *idx)] = T::zero();
            result.matrix[(*idx, *idx)] = T::zero();
        }

        Some(result)
    }

    /// Compute the multivariate likelihood from two iterators over `T`.
    /// The length of both iterators must be equal and also a multiple
    /// of the dimension of the covariance matrix (panic otherwise).
    pub fn multivariate_likelihood(
        &self,
        x: impl IntoIterator<Item = T>,
        mu: impl IntoIterator<Item = T>,
    ) -> T {
        let delta = DVector::from(zip_eq(x, mu).map(|(i, j)| i - j).collect::<Vec<T>>());

        let ndim = delta.len() / self.ndim();

        if delta.len() % self.ndim() != 0 {
            panic!("iterator length must be a multiple of the covariance matrix dimension")
        }

        let mut lh = -(self.determinant.ln() + T::from_usize(ndim).unwrap() * (T::two_pi()))
            / T::from_usize(2).unwrap();

        for idx in 0..ndim {
            let view = delta.rows_with_step(idx, self.ndim(), ndim - 1);

            lh -= (&view.transpose()
                * self
                    .matrix_inverse
                    .view((0, 0), (view.nrows(), view.nrows()))
                * view)[(0, 0)]
                / T::from_usize(2).unwrap();
        }

        lh
    }

    /// Returns the number of dimensions of the covariance matrix.
    pub fn ndim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Returns the (pseudo-)determinant of the covariance matrix.
    pub fn determinant(&self) -> T {
        self.determinant
    }

    /// Returns a reference to the lower triangular matrix L from the Cholesky decomposition.
    pub fn ref_cholesky_ltm(&self) -> &DMatrix<T> {
        &self.cholesky_ltm
    }

    /// Returns a reference to the inverse of the covariance matrix.
    pub fn ref_matrix_inverse(&self) -> &DMatrix<T> {
        &self.matrix_inverse
    }

    /// Returns a reference to the covariance matrix.
    pub fn ref_matrix(&self) -> &DMatrix<T> {
        &self.matrix
    }
}

impl<T> TryFrom<&[T]> for CovMatrix<T>
where
    T: Copy + RealField + Scalar,
{
    type Error = ();

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        if let Some(result) = Self::from_matrix(
            &DMatrix::from_diagonal(&DVector::from_iterator(value.len(), value.iter().cloned()))
                .as_view(),
        ) {
            Ok(result)
        } else {
            Err(())
        }
    }
}

impl<'a, T> TryFrom<&DMatrixView<'a, T>> for CovMatrix<T>
where
    T: Copy + RealField + Scalar,
{
    type Error = ();

    fn try_from(value: &DMatrixView<'a, T>) -> Result<Self, Self::Error> {
        if let Some(result) = Self::from_matrix(value) {
            Ok(result)
        } else {
            Err(())
        }
    }
}

impl<T> Mul<T> for CovMatrix<T>
where
    T: Copy + RealField,
{
    type Output = CovMatrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let dim = self.matrix.ncols() as i32;

        Self::Output {
            cholesky_ltm: self.cholesky_ltm * rhs.sqrt(),
            matrix_inverse: self.matrix_inverse / rhs,
            matrix: self.matrix * rhs,
            determinant: self.determinant * rhs.powi(dim),
        }
    }
}

impl<T> MulAssign<T> for CovMatrix<T>
where
    T: Copy + RealField,
{
    fn mul_assign(&mut self, rhs: T) {
        let dim = self.matrix.ncols() as i32;

        self.cholesky_ltm *= rhs.sqrt();
        self.matrix_inverse /= rhs;
        self.matrix *= rhs;
        self.determinant *= rhs.powi(dim);
    }
}

/// Computes the unbiased covariance over two slices.
///
/// The length of both iterators must be equal (panic).
pub fn covariance<'a, T, I>(x: I, y: I) -> T
where
    T: Copy + RealField + Sum + for<'x> Sum<&'x T>,
    I: IntoIterator<Item = &'a T>,
    <I as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();

    let length = x_iter.clone().fold(0, |acc, _| acc + 1);

    let mu_x = x_iter.clone().sum::<T>() / T::from_usize(length).unwrap();
    let mu_y = x_iter.clone().sum::<T>() / T::from_usize(length).unwrap();

    zip_eq(x_iter, y_iter)
        .map(|(val_x, val_y)| (mu_x - *val_x) * (mu_y - *val_y))
        .sum::<T>()
        / T::from_usize(length - 1).unwrap()
}

/// Computes the unbiased covariance over two slices with weights.
///
/// The length of all three iterators must be equal (panic).
pub fn covariance_with_weights<'a, T, IV, IW>(x: IV, y: IV, w: IW) -> T
where
    T: Copy + RealField + Sum + for<'x> Sum<&'x T>,
    IV: IntoIterator<Item = &'a T>,
    IW: IntoIterator<Item = &'a T>,
    <IV as IntoIterator>::IntoIter: Clone,
    <IW as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();
    let w_iter = w.into_iter();

    let wsum = w_iter.clone().sum::<T>();
    let wsumsq = w_iter.clone().map(|val_w| val_w.powi(2)).sum::<T>();

    let wfac = wsum - wsumsq / wsum;

    let mu_x = zip_eq(x_iter.clone(), w_iter.clone())
        .map(|(val_x, val_w)| *val_x * *val_w)
        .sum::<T>()
        / wsum;

    let mu_y = zip_eq(y_iter.clone(), w_iter.clone())
        .map(|(val_y, val_w)| *val_y * *val_w)
        .sum::<T>()
        / wsum;

    zip_eq(x_iter, zip_eq(y_iter, w_iter))
        .map(|(val_x, (val_y, val_w))| *val_w * (mu_x - *val_x) * (mu_y - *val_y))
        .sum::<T>()
        / wfac
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix, U3, VecStorage};
    use rand::{Rng, SeedableRng};
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_covariance() {
        assert!(
            covariance(
                &[10.0_f32, 34.0, 23.0, 54.0, 9.0],
                &[4.0, 5.0, 11.0, 15.0, 20.0]
            ) == 5.75
        );

        assert!(
            (covariance_with_weights(
                &[10.0_f32, 34.0, 23.0, 54.0, 9.0],
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

        let array_view = &array.as_view();

        let covmat = CovMatrix::from_particles(array_view, None).unwrap();

        assert!((covmat.cholesky_ltm[(0, 0)] - 0.40718567).abs() < f32::EPSILON);
        assert!((covmat.cholesky_ltm[(2, 0)] - 0.07841061).abs() < f32::EPSILON);

        assert!((covmat.matrix_inverse[(0, 0)] - 6.915894).abs() < f32::EPSILON);
        assert!((covmat.matrix_inverse[(2, 0)] + 4.5933948).abs() < f32::EPSILON);

        assert!((covmat.determinant() - 0.0069507807).abs() < f32::EPSILON);
    }
}
