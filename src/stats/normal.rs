use crate::stats::{CovarianceMatrix, Density, DensityRange};
use log::error;
use nalgebra::{
    DMatrix, DefaultAllocator, DimDiff, DimMin, DimMinimum, DimName, DimSub, Dyn, MatrixView,
    OMatrix, OVector, RealField, SVector, SVectorView, Scalar, U1, VectorView,
    allocator::Allocator,
};
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A multivariate normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(
    serialize = "T: Serialize, OVector<T, D>: Serialize, OVector<DensityRange<T>, D>: Serialize, OMatrix<T, D, D>: Serialize"
))]
#[serde(bound(
    deserialize = "T: Deserialize<'de>, OVector<T, D>: Deserialize<'de>, OVector<DensityRange<T>, D>: Deserialize<'de>, OMatrix<T, D, D>: Deserialize<'de>"
))]
pub struct MultivariateNormalDensity<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    /// A [`CovarianceMatrix`] that describes the underlying multivariate normal probability density function.
    covmat: CovarianceMatrix<T, D>,

    /// The mean parameter of the multivariate normal distribution.
    mean: OVector<T, D>,

    /// Valid parameter range.
    range: OVector<DensityRange<T>, D>,
}

impl<T, D> MultivariateNormalDensity<T, D>
where
    T: Copy + RealField + Scalar,
    D: DimName + DimMin<D>,
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
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// Calculates the exact density at a specific position `x`.
    pub fn density(&self, x: &VectorView<T, D>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.pseudo_inverse() * diff)[(0, 0)];

        let p_nonzero: usize = self.covmat.matrix().diagonal().iter().fold(0, |acc, next| {
            if next.abs() > T::zero() { acc + 1 } else { acc }
        });

        (-value / T::from_usize(2).unwrap()).exp()
            / ((T::two_pi()).powi(p_nonzero as i32) * self.covmat.pseudo_determinant()).sqrt()
    }

    /// Returns the pseudo-determinant of the covariance matrix.
    pub fn determinant(&self) -> T {
        self.covmat.pseudo_determinant()
    }

    /// Create a [`MultivariateNormalDensity`] from a covariance matrix.
    pub fn from_covmat(
        covmat: CovarianceMatrix<T, D>,
        mean: OVector<T, D>,
        range: OVector<DensityRange<T>, D>,
    ) -> Self {
        Self {
            covmat,
            mean,
            range,
        }
    }

    /// Create a [`MultivariateNormalDensity`] from an ensemble of particles.
    pub fn from_vectors<'a>(
        particles: &MatrixView<'a, T, D, Dyn>,
        range: OVector<DensityRange<T>, D>,
        weights: Option<&[T]>,
    ) -> Option<Self>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let covmat = CovarianceMatrix::from_vectors(&particles.as_view(), weights)?;
        let mut mean = particles.column_mean();

        // Set mean to first particle value if covariance is zero.
        // This fixes numerical issues where taking the mean over many
        // particles does not equal the constant value.
        covmat
            .matrix()
            .diagonal()
            .iter()
            .zip(mean.iter_mut())
            .enumerate()
            .for_each(|(idx, (cov, value))| {
                if cov.eq(&T::zero()) {
                    *value = particles[(idx, 0)];
                }
            });

        Some(Self {
            covmat,
            mean,
            range,
        })
    }

    /// Compute the Kullback-Leibler divergence between two [`MultivariateNormalDensity`].
    pub fn kullback_leibler_divergence(&self, other: &MultivariateNormalDensity<T, D>) -> Option<T>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let mut l_0 = self.covmat.chol_ltm().clone();
        let mu_0 = self.mean;

        let mut l_1 = other.covmat.chol_ltm().clone();
        let mu_1 = other.mean;

        let mut p_nonzero = D::USIZE;

        // Detect zero'd columns/rows that need to be modified.
        (0..l_0.nrows()).for_each(|idx| {
            if l_0[(idx, idx)].eq(&T::zero()) {
                l_0[(idx, idx)] = T::one() / T::zero();

                p_nonzero -= 1;

                // Set off diagonals to zero.
                for jdx in 0..l_0.ncols() {
                    if jdx != idx {
                        l_0[(idx, jdx)] = T::zero();
                        l_0[(jdx, idx)] = T::zero();
                    }
                }
            };
        });

        let m = l_0.clone().solve_lower_triangular(&l_1).unwrap();

        // Detect zero'd columns/rows that need to be modified.
        (0..l_1.nrows()).for_each(|idx| {
            if l_1[(idx, idx)].eq(&T::zero()) {
                l_1[(idx, idx)] = T::one() / T::zero();

                // Set off diagonals to zero.
                for jdx in 0..l_1.ncols() {
                    if jdx != idx {
                        l_1[(idx, jdx)] = T::zero();
                        l_1[(jdx, idx)] = T::zero();
                    }
                }
            };
        });

        let y = l_1.clone().solve_lower_triangular(&(mu_1 - mu_0)).unwrap();

        Some(
            (m.iter().sum::<T>() - T::from_usize(p_nonzero).unwrap()
                + y.norm()
                + T::from_usize(2).unwrap()
                    * l_1
                        .diagonal()
                        .iter()
                        .zip(l_0.diagonal().iter())
                        .map(|(a, b)| {
                            if a.is_finite() && b.is_finite() {
                                (*a / *b).ln()
                            } else {
                                T::zero()
                            }
                        })
                        .sum::<T>())
                / T::from_usize(2).unwrap(),
        )
    }

    /// Returns a reference to the lower triangular matrix L from the Cholesky decomposition
    /// of the covariance matrix.
    pub fn chol_ltm(&self) -> &OMatrix<T, D, D> {
        self.covmat.chol_ltm()
    }

    /// Returns a reference to the inverse of the underlying covariance matrix.
    pub fn pseudo_inverse(&self) -> &OMatrix<T, D, D> {
        self.covmat.pseudo_inverse()
    }

    /// Returns a reference to the underlying covariance matrix.
    pub fn matrix(&self) -> &OMatrix<T, D, D> {
        self.covmat.matrix()
    }

    /// Returns the center of the multivariate normal density.
    pub fn mean(&self) -> &OVector<T, D> {
        &self.mean
    }
}

impl<T, D> Density<T, D> for &MultivariateNormalDensity<T, D>
where
    T: Copy + RealField,
    D: DimName,
    StandardNormal: Distribution<T>,
{
    fn constant_values(&self) -> [Option<T>; N] {
        self.covmat
            .ref_matrix()
            .diagonal()
            .iter()
            .zip(self.mean.iter())
            .map(|(cov, mean)| {
                if cov.eq(&T::zero()) {
                    Some(*mean)
                } else {
                    None
                }
            })
            .collect::<Vec<Option<T>>>()
            .try_into()
            .unwrap()
    }

    fn density_rel(&self, x: &VectorView<T, D>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_matrix_inverse() * diff)[(0, 0)];

        (-value / T::from_usize(2).unwrap()).exp()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<OVector<T, D>, StatsError<T>> {
        let normal = StandardNormal;

        let mut proposal = self.mean
            + self.covmat.ref_chol_ltm()
                * SVector::<T, N>::from_iterator((0..N).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = self.mean
                + self.covmat.ref_chol_ltm()
                    * SVector::<T, N>::from_iterator((0..N).map(|_| rng.sample(normal)));

            attempts += 1;

            if attempts > 499 {
                error!(
                    "MultivariateNormalDensity::draw_sample has failed to draw a valid sample after {} tries",
                    attempts
                );

                return Err(StatsError::InefficientSampling {
                    name: "Normal1D",
                    count: 500,
                });
            }
        }

        Ok(proposal)
    }

    fn get_valid_range(&self) -> OVector<DensityRange<T>, D> {
        self.range
    }
}

impl<T, D> Add<&OVector<T, D>> for MultivariateNormalDensity<T, N>
where
    T: Clone + RealField + Scalar,
{
    type Output = MultivariateNormalDensity<T, N>;

    fn add(self, rhs: &OVector<T, D>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<T, D> AddAssign<&OVector<T, D>> for MultivariateNormalDensity<T, N>
where
    T: Clone + RealField + Scalar,
{
    fn add_assign(&mut self, rhs: &OVector<T, D>) {
        self.mean += rhs
    }
}

impl<T, D> Mul<T> for MultivariateNormalDensity<T, N>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateNormalDensity<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            covmat: self.covmat * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<T, D> MulAssign<T> for MultivariateNormalDensity<T, N>
where
    T: Copy + RealField + Scalar,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covmat *= rhs;
    }
}

impl<T, D> Sub<&OVector<T, D>> for MultivariateNormalDensity<T, N>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateNormalDensity<T, N>;

    fn sub(self, rhs: &OVector<T, D>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<T, D> SubAssign<&OVector<T, D>> for MultivariateNormalDensity<T, N>
where
    T: Copy + RealField + Scalar,
{
    fn sub_assign(&mut self, rhs: &OVector<T, D>) {
        self.mean -= rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix, U3, VecStorage};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_multivariatend() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let array_view = &array.as_view();

        let covmat = CovarianceMatrix::from_vectors(array_view, None).unwrap();

        let mvpdf = &MultivariateNormalDensity::from_covmat(
            covmat,
            SVector::from([0.1, 0.0, 0.25]),
            [(-0.75, 0.75); 3],
        );

        assert!(
            (mvpdf.density(&SVector::from([0.2, 0.0, 0.35]).as_view()) - 0.1611928).abs() < 1e-6
        );

        assert!(
            mvpdf
                .density_rel(&SVector::from([0.2, 0.1, 0.35]).as_view())
                .is_nan()
        );

        assert!(
            (mvpdf.draw_sample(&mut rng).unwrap() - SVector::from([-0.4150916, 0.0, 0.4898513]))
                .norm()
                < 1e-6
        );

        assert!(mvpdf.validate_sample(&mvpdf.draw_sample(&mut rng).unwrap().as_view()));

        let mvpdf_ensbl =
            MultivariateNormalDensity::from_vectors(array_view, [(-0.75, 0.75); 3], None).unwrap();

        assert!(mvpdf.covmat.ref_matrix() == mvpdf_ensbl.covmat.ref_matrix());
    }

    #[test]
    fn test_multivariate_kld() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array_1 = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let array_2 = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    0.25 + rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let mvpdf_1 =
            MultivariateNormalDensity::from_vectors(&array_1.as_view(), [(-2.0, 2.0); 3], None)
                .unwrap();
        let mvpdf_2 =
            MultivariateNormalDensity::from_vectors(&array_2.as_view(), [(-2.0, 2.0); 3], None)
                .unwrap();

        assert!((mvpdf_1.kullback_leibler_divergence(&mvpdf_2).unwrap() - 0.20391017).abs() < 1e-6);
    }
}
