use itertools::{Itertools, zip_eq};
use nalgebra::{
    Const, DVector, DVectorView, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimName,
    DimSub, Dyn, LDL, Matrix, MatrixView, OMatrix, OVector, Owned, RealField, Scalar, U1,
    allocator::Allocator,
};
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Div, DivAssign, Mul, MulAssign},
};

/// A covariance matrix, i.e. a positive semi-definite square matrix.
///
/// This type can be constructed directly from a positive semi-definite square matrix or from an ensemble of vectors.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "T: Serialize, OMatrix<T, D, D>: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>, OMatrix<T, D, D>: Deserialize<'de>"))]
pub struct CovarianceMatrix<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    /// The lower triangular matrix from the Cholesky / LDL decomposition.
    chol_ltm: OMatrix<T, D, D>,

    /// The covariance matrix itself.
    matrix: OMatrix<T, D, D>,

    /// The pseudo-determinant of the covariance matrix.
    pseudo_determinant: T,

    /// The pseudo-inverse of the covariance matrix.
    pseudo_inverse: OMatrix<T, D, D>,
}

impl<T, D> CovarianceMatrix<T, D>
where
    T: Copy + RealField + Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
    usize: AsPrimitive<T>,
{
    /// Draw a random sample vector using the covariance matrix.
    pub fn draw_sample(&self, rng: &mut impl Rng) -> OVector<T, D>
    where
        StandardNormal: Distribution<T>,
    {
        let n = self.matrix.shape_generic().0;

        let random_vector = OVector::<T, D>::from_iterator_generic(
            n,
            Const::<1>,
            (0..self.ndim()).map(|_| rng.sample(StandardNormal)),
        );

        &self.chol_ltm * random_vector
    }

    /// Create a [`CovarianceMatrix`] from a positive semi-definite square matrix.
    pub fn from_matrix(matrix: OMatrix<T, D, D>) -> Option<Self>
    where
        DimMinimum<D, D>: DimSub<U1>,
        D: DimMin<D>,
        DefaultAllocator: Allocator<DimDiff<DimMinimum<D, D>, U1>>
            + Allocator<DimMinimum<D, D>, D>
            + Allocator<D, DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
    {
        let ldl = LDL::new(matrix.clone_owned()).expect("ldl factorization failed");

        // Compute pseudo-determinant by multiplying all non-zero eigenvalues.
        let pseudo_determinant = ldl.d.iter().fold(T::one(), |acc, next| {
            if !matches!(
                next.partial_cmp(&T::zero())
                    .expect("ldl factorization diagonal value is NaN"),
                Ordering::Equal
            ) {
                acc * *next
            } else {
                acc
            }
        });

        let mut pseudo_inverse = matrix
            .clone_owned()
            .pseudo_inverse(T::default_epsilon())
            .expect("failed to construct pseudo inverse");

        // Zero out rows/columns with zero eigenvalues.
        let n_dim = matrix.shape_generic().0;

        matrix
            .diagonal()
            .iter()
            .enumerate()
            .for_each(|(idx, value)| {
                if matches!(value.partial_cmp(&T::zero()).unwrap(), Ordering::Equal) {
                    pseudo_inverse
                        .set_column(idx, &OVector::<T, D>::zeros_generic(n_dim, Const::<1>));
                    pseudo_inverse.set_row(
                        idx,
                        &Matrix::<T, U1, D, Owned<T, U1, D>>::zeros_generic(Const::<1>, n_dim),
                    );
                }
            });

        Some(Self {
            chol_ltm: ldl.cholesky_l(),
            matrix,
            pseudo_determinant,
            pseudo_inverse,
        })
    }

    /// Create a [`CovarianceMatrix`] from an ensemble of vectors.
    pub fn from_vectors(vectors: &MatrixView<T, D, Dyn>, opt_weights: Option<&[T]>) -> Option<Self>
    where
        T: Sum + for<'x> Sum<&'x T>,
        D: DimName + DimMin<D>,
        DimMinimum<D, D>: DimSub<U1>,
        DefaultAllocator: Allocator<DimDiff<DimMinimum<D, D>, U1>>
            + Allocator<DimMinimum<D, D>, D>
            + Allocator<D, DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimMinimum<D, D>>
            + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
    {
        let n = D::USIZE;

        let mut matrix = OMatrix::<T, D, D>::from_iterator((0..(n.pow(2))).map(|idx| {
            let jdx = idx / n;
            let kdx = idx % n;

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
        }));

        // Fill up the other side of the matrix.
        matrix += matrix.transpose() - OMatrix::from_diagonal(&matrix.diagonal());

        Self::from_matrix(matrix)
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

    /// Compute the multivariate likelihood using an observation `x` and mean `mu`.
    /// The length of both input slices must be the same, and their length must also be a multiple
    /// of the dimension of the covariance matrix.
    pub fn multivariate_likelihood(&self, x: &[T], mu: &[T]) -> T {
        let ndim = x.len() / self.ndim();

        if (x.len() != mu.len()) || (x.len() % self.ndim()) != 0 {
            panic!("failed to compute the multivariate likelihood, slice lengths are invalid");
        }

        let delta = DVector::from(x.iter().zip(mu).map(|(i, j)| *i - *j).collect::<Vec<T>>());

        let mut lh = -(self.pseudo_determinant.ln() + ndim.as_() * T::two_pi()) / 2.as_();

        for idx in 0..ndim {
            let view: nalgebra::Matrix<
                T,
                Dyn,
                Const<1>,
                nalgebra::ViewStorage<'_, T, Dyn, Const<1>, Dyn, Dyn>,
            > = delta.rows_with_step(idx, self.ndim(), ndim - 1);

            lh -= self.mahalanobis_distance_squared(&view) / 2.as_();
        }

        lh
    }

    /// Returns the number of dimensions of the covariance matrix.
    pub fn ndim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Returns the pseudo-determinant of the covariance matrix.
    pub fn pseudo_determinant(&self) -> T {
        self.pseudo_determinant
    }

    /// Returns a reference to the lower triangular matrix L from the Cholesky decomposition.
    pub fn chol_ltm(&self) -> &OMatrix<T, D, D> {
        &self.chol_ltm
    }

    /// Returns a reference to the inverse of the covariance matrix.
    pub fn pseudo_inverse(&self) -> &OMatrix<T, D, D> {
        &self.pseudo_inverse
    }

    /// Returns a reference to the covariance matrix.
    pub fn matrix(&self) -> &OMatrix<T, D, D> {
        &self.matrix
    }
}

impl<T, D> TryFrom<&[T]> for CovarianceMatrix<T, D>
where
    T: Copy + RealField + Scalar,
    D: Dim + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
    usize: AsPrimitive<T>,
{
    type Error = ();

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        let matrix = OMatrix::from_diagonal(&OVector::from_iterator_generic(
            D::from_usize(value.len()),
            Const::<1>,
            value.iter().cloned(),
        ));

        if let Some(result) = Self::from_matrix(matrix) {
            Ok(result)
        } else {
            Err(())
        }
    }
}

impl<'a, T, D> TryFrom<&MatrixView<'a, T, D, D>> for CovarianceMatrix<T, D>
where
    T: Copy + RealField + Scalar,
    D: Dim + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>,
    usize: AsPrimitive<T>,
{
    type Error = ();

    fn try_from(value: &MatrixView<'a, T, D, D>) -> Result<Self, Self::Error> {
        if let Some(result) = Self::from_matrix(value.into_owned()) {
            Ok(result)
        } else {
            Err(())
        }
    }
}

impl<T, D> Div<T> for CovarianceMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = CovarianceMatrix<T, D>;

    fn div(self, rhs: T) -> Self::Output {
        let dim = self.matrix.ncols() as i32;

        Self::Output {
            chol_ltm: self.chol_ltm / rhs.sqrt(),
            matrix: self.matrix / rhs,
            pseudo_determinant: self.pseudo_determinant / rhs.powi(dim),
            pseudo_inverse: self.pseudo_inverse * rhs,
        }
    }
}

impl<T, D> DivAssign<T> for CovarianceMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn div_assign(&mut self, rhs: T) {
        let dim = self.matrix.ncols() as i32;

        self.chol_ltm /= rhs.sqrt();
        self.matrix /= rhs;
        self.pseudo_determinant /= rhs.powi(dim);
        self.pseudo_inverse *= rhs;
    }
}

impl<T, D> Mul<T> for CovarianceMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = CovarianceMatrix<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        let dim = self.matrix.ncols() as i32;

        Self::Output {
            chol_ltm: self.chol_ltm * rhs.sqrt(),
            matrix: self.matrix * rhs,
            pseudo_determinant: self.pseudo_determinant * rhs.powi(dim),
            pseudo_inverse: self.pseudo_inverse / rhs,
        }
    }
}

impl<T, D> MulAssign<T> for CovarianceMatrix<T, D>
where
    T: Copy + RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn mul_assign(&mut self, rhs: T) {
        let dim = self.matrix.ncols() as i32;

        self.chol_ltm *= rhs.sqrt();
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
    usize: AsPrimitive<T>,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();

    let length = x_iter.clone().fold(0, |acc, _| acc + 1);

    let mu_x = x_iter.clone().sum::<T>() / length.as_();
    let mu_y = x_iter.clone().sum::<T>() / length.as_();

    zip_eq(x_iter, y_iter)
        .map(|(val_x, val_y)| (mu_x - *val_x) * (mu_y - *val_y))
        .sum::<T>()
        / (length - 1).as_()
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
    use approx::ulps_eq;
    use nalgebra::{Matrix, U3, VecStorage};
    use rand::{Rng, SeedableRng};
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_covariance() {
        assert!(ulps_eq!(
            covariance(
                &[10.0_f32, 34.0, 23.0, 54.0, 9.0],
                &[4.0, 5.0, 11.0, 15.0, 20.0]
            ),
            5.75
        ));

        assert!(ulps_eq!(
            covariance_with_weights(
                &[10.0_f32, 34.0, 23.0, 54.0, 9.0],
                &[4.0, 5.0, 11.0, 15.0, 20.0],
                &[1.0, 0.8, 1.1, 1.3, 0.9]
            ),
            19.523237
        ));
    }

    #[test]
    fn test_covariance_matrix() {
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

        let covmat = CovarianceMatrix::<_, Const<3>>::from_vectors(array_view, None).unwrap();

        assert!(ulps_eq!(covmat.chol_ltm[(0, 0)], 0.40718567));
        assert!(ulps_eq!(covmat.chol_ltm[(2, 0)], 0.07841061));

        assert!(ulps_eq!(covmat.pseudo_inverse[(0, 0)], 6.915894));
        assert!(ulps_eq!(covmat.pseudo_inverse[(2, 0)], -4.5933948));

        assert!(ulps_eq!(covmat.pseudo_determinant(), 0.0069507807));
    }
}
