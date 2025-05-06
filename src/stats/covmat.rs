use derive_more::Deref;
use itertools::{Itertools, zip_eq};
use log::{error, warn};
use nalgebra::{
    Const, DMatrix, DMatrixView, DVector, DVectorView, Dim, Dyn, MatrixView, RealField, Scalar,
};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Div, DivAssign, Mul, MulAssign},
};

/// A dynamically sized covariance matrix, implemented as a wrapper around `DMatrix<T>`.
///
/// This type can be constructed directly from a positive semi-definite matrix or from an ensemble of vectors.
#[derive(Clone, Debug, Deref, Deserialize, Serialize)]
pub struct CovMatrix<T>
where
    T: Scalar,
{
    /// The lower triangular matrix from the Cholesky decomposition.
    cholesky_ltm: DMatrix<T>,

    /// The covariance matrix itself.
    #[deref]
    matrix: DMatrix<T>,

    /// The pseudo-determinant of the covariance matrix.
    pseudo_determinant: T,

    /// The pseudo inverse of the covariance matrix.
    pseudo_inverse: DMatrix<T>,
}

impl<T> CovMatrix<T>
where
    T: Copy + RealField + Scalar,
{
    /// Draw a random sample from the covariance matrix.
    pub fn draw_sample(&self, rng: &mut impl Rng) -> DVector<T>
    where
        StandardNormal: Distribution<T>,
    {
        self.ref_cholesky_ltm()
            * DVector::<T>::from_iterator(
                self.ndim(),
                (0..self.ndim()).map(|_| rng.sample(StandardNormal)),
            )
    }

    /// Create a [`CovMatrix`] from a positive semi-definite square matrix.
    pub fn from_matrix(matrix: &DMatrixView<T>) -> Option<Self> {
        let svd = matrix.svd(true, false);

        println!("{}", &svd.clone().u.unwrap());
        let qr_r = svd.u.as_ref().unwrap().clone();

        // println!("{}", &sym_eigen.eigenvectors);
        // println!("{}", &sym_eigen.eigenvectors.clone().qr().r().transpose());

        let cholesky_ltm = qr_r
            * DMatrix::from_diagonal(&DVector::from_iterator(
                matrix.nrows(),
                svd.singular_values.iter().map(|value| value.sqrt()),
            ));

        // Compute pseudo determinant by multiplying all non-zero eigenvalues.
        let pseudo_determinant = svd.singular_values.iter().fold(T::one(), |acc, next| {
            if !matches!(next.partial_cmp(&T::zero()).unwrap(), Ordering::Equal) {
                acc * *next
            } else {
                acc
            }
        });

        let pseudo_inverse = match matrix.try_inverse() {
            Some(result) => result,
            None => {
                error!(
                    "CovMatrix::from_matrix input matrix is singular: {}",
                    matrix
                );

                return None;
            }
        };

        Some(Self {
            cholesky_ltm,
            matrix: matrix.clone_owned(),
            pseudo_determinant,
            pseudo_inverse,
        })
    }

    /// Create a [`CovMatrix`] from an ensemble of vectors.
    pub fn from_vectors<const N: usize>(
        vectors: &MatrixView<T, Const<N>, Dyn>,
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
                    let x = vectors.row(jdx);
                    let y = vectors.row(kdx);

                    if !x.iter().all_equal() && !y.iter().all_equal() {
                        match opt_weights {
                            Some(w) => covariance_with_weights(x, y, w),
                            None => covariance(x, y),
                        }
                    } else {
                        T::zero()
                    }
                } else {
                    T::zero()
                }
            }),
        );

        // Fill up the other side of the matrix.
        matrix += matrix.transpose() - DMatrix::<T>::from_diagonal(&matrix.diagonal());

        Some(Self::from_matrix(&matrix.as_view())?)
    }

    /// Compute the squared Mahalanobis distance.
    pub fn mahalanobis_distance_squared<RStride, CStride>(
        &self,
        x: &DVectorView<T, RStride, CStride>,
    ) -> T
    where
        RStride: Dim,
        CStride: Dim,
    {
        (&x.transpose() * self.pseudo_inverse.view((0, 0), (x.nrows(), x.nrows())) * x)[(0, 0)]
    }

    /// Compute the multivariate likelihood from two slices over `T`.
    /// The length of both slices must be equal and also a multiple
    /// of the dimension of the covariance matrix.
    pub fn multivariate_likelihood(&self, x: &[T], mu: &[T]) -> Option<T> {
        let ndim = x.len() / self.ndim();

        if (x.len() != mu.len()) || (x.len() % self.ndim()) != 0 {
            warn!(
                "failed to compute the multivariate likelihood, slice 
                lengths either do not match each other or are not a 
                multiple of the dimension of the covariance matrix"
            );

            return None;
        }

        let delta = DVector::from(x.iter().zip(mu).map(|(i, j)| *i - *j).collect::<Vec<T>>());

        let mut lh = -(self.pseudo_determinant.ln() + T::from_usize(ndim).unwrap() * T::two_pi())
            / T::from_usize(2).unwrap();

        for idx in 0..ndim {
            let view: nalgebra::Matrix<
                T,
                Dyn,
                Const<1>,
                nalgebra::ViewStorage<'_, T, Dyn, Const<1>, Dyn, Dyn>,
            > = delta.rows_with_step(idx, self.ndim(), ndim - 1);

            lh -= self.mahalanobis_distance_squared(&view) / T::from_usize(2).unwrap();
        }

        Some(lh)
    }

    /// Returns the number of dimensions of the covariance matrix.
    pub fn ndim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Returns the (pseudo-)determinant of the covariance matrix.
    pub fn pseudo_determinant(&self) -> T {
        self.pseudo_determinant
    }

    /// Returns a reference to the lower triangular matrix L from the Cholesky decomposition.
    pub fn ref_cholesky_ltm(&self) -> &DMatrix<T> {
        &self.cholesky_ltm
    }

    /// Returns a reference to the inverse of the covariance matrix.
    pub fn ref_pseudo_inverse(&self) -> &DMatrix<T> {
        &self.pseudo_inverse
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

impl<T> Div<T> for CovMatrix<T>
where
    T: Copy + RealField,
{
    type Output = CovMatrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        let dim = self.matrix.ncols() as i32;

        Self::Output {
            cholesky_ltm: self.cholesky_ltm / rhs.sqrt(),
            matrix: self.matrix / rhs,
            pseudo_determinant: self.pseudo_determinant / rhs.powi(dim),
            pseudo_inverse: self.pseudo_inverse * rhs,
        }
    }
}

impl<T> DivAssign<T> for CovMatrix<T>
where
    T: Copy + RealField,
{
    fn div_assign(&mut self, rhs: T) {
        let dim = self.matrix.ncols() as i32;

        self.cholesky_ltm /= rhs.sqrt();
        self.matrix /= rhs;
        self.pseudo_determinant /= rhs.powi(dim);
        self.pseudo_inverse *= rhs;
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
            matrix: self.matrix * rhs,
            pseudo_determinant: self.pseudo_determinant * rhs.powi(dim),
            pseudo_inverse: self.pseudo_inverse / rhs,
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
        self.matrix *= rhs;
        self.pseudo_determinant *= rhs.powi(dim);
        self.pseudo_inverse /= rhs;
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

        let covmat = CovMatrix::from_vectors(array_view, None).unwrap();

        println!("{}", &covmat.matrix);
        println!("{}", &covmat.cholesky_ltm);

        assert!((covmat.cholesky_ltm[(0, 0)] - 0.40718567).abs() < f32::EPSILON);
        assert!((covmat.cholesky_ltm[(2, 0)] - 0.07841061).abs() < f32::EPSILON);

        assert!((covmat.pseudo_inverse[(0, 0)] - 6.915894).abs() < f32::EPSILON);
        assert!((covmat.pseudo_inverse[(2, 0)] + 4.5933948).abs() < f32::EPSILON);

        assert!((covmat.determinant() - 0.0069507807).abs() < f32::EPSILON);
    }
}
