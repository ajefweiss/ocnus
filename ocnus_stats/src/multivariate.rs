use crate::{CovMatrix, Density, StatsError};
use log::error;
use nalgebra::{Const, DMatrix, Dyn, MatrixView, RealField, SVector, SVectorView, Scalar};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A multivariate normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultivariateND<T, const N: usize>
where
    T: Scalar,
{
    /// A [`CovMatrix`] that describes the underlying multivariate normal probability density function.
    covmat: CovMatrix<T>,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<T, N>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(T, T); N],
}

impl<T, const N: usize> MultivariateND<T, N>
where
    T: Copy + RealField + Scalar,
    StandardNormal: Distribution<T>,
{
    /// Calculates the exact density  at a specific position `x`.
    pub fn density(&self, x: &SVectorView<T, N>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_matrix_inverse() * diff)[(0, 0)];

        let p_nonzero: usize = self
            .covmat
            .ref_matrix()
            .diagonal()
            .iter()
            .fold(
                0,
                |acc, next| {
                    if next.abs() > T::zero() { acc + 1 } else { acc }
                },
            );

        (-value / T::from_usize(2).unwrap()).exp()
            / ((T::two_pi()).powi(p_nonzero as i32) * self.covmat.determinant()).sqrt()
    }

    /// Returns the pseudo-determinant of the covariance matrix.
    pub fn determinant(&self) -> T {
        self.covmat.determinant()
    }

    /// Create a [`MultivariateND`] from a covariance matrix.
    pub fn from_covmat(covmat: CovMatrix<T>, mean: SVector<T, N>, range: [(T, T); N]) -> Self {
        Self {
            covmat,
            mean,
            range,
        }
    }

    /// Create a [`MultivariateND`] from an ensemble of particles.
    pub fn from_particles<'a>(
        particles: &MatrixView<'a, T, Const<N>, Dyn>,
        range: [(T, T); N],
        weights: Option<&[T]>,
    ) -> Option<Self>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let covmat = CovMatrix::from_particles(&particles.as_view(), weights)?;
        let mut mean = particles.column_mean();

        // Set mean to first particle value if covariance is zero.
        // This fixes numerical issues where taking the mean over many
        // particles does not equal the constant value.
        covmat
            .ref_matrix()
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

    /// Compute the Kullback-Leibler divergence between two [`MultivariateND`].
    pub fn kullback_leibler_divergence(
        &self,
        other: &MultivariateND<T, N>,
    ) -> Result<T, StatsError<T>>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        if self.covmat.ndim() != other.covmat.ndim() {
            return Err(StatsError::InvalidCovMatrix {
                msg: "covariance matrix dimensions are mismatched",
                covm_ndim: self.covmat.ndim(),
                expd_ndim: other.covmat.ndim(),
            });
        }

        let mut l_0 = self.covmat.ref_cholesky_ltm().clone();
        let mu_0 = self.mean;

        let mut l_1 = other.covmat.ref_cholesky_ltm().clone();
        let mu_1 = other.mean;

        let mut p_nonzero = N;

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

        let y = l_1
            .clone()
            .solve_lower_triangular(&(&mu_1 - &mu_0))
            .unwrap();

        Ok((m.iter().sum::<T>() - T::from_usize(p_nonzero).unwrap()
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
            / T::from_usize(2).unwrap())
    }

    /// Returns a reference to the lower triangular matrix L from the Cholesky decomposition
    /// of the covariance matrix.
    pub fn ref_cholesky_ltm(&self) -> &DMatrix<T> {
        &self.covmat.ref_cholesky_ltm()
    }

    /// Returns a reference to the inverse of the underlying covariance matrix.
    pub fn ref_matrix_inverse(&self) -> &DMatrix<T> {
        &self.covmat.ref_matrix_inverse()
    }

    /// Returns a reference to the underlying covariance matrix.
    pub fn ref_matrix(&self) -> &DMatrix<T> {
        &self.covmat.ref_matrix()
    }

    /// Returns the center of the multivariate normal density.
    pub fn ref_mean(&self) -> &SVector<T, N> {
        &self.mean
    }
}

impl<T, const N: usize> Density<T, N> for &MultivariateND<T, N>
where
    T: Copy + RealField,
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

    fn density_rel(&self, x: &SVectorView<T, N>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_matrix_inverse() * diff)[(0, 0)];

        (-value / T::from_usize(2).unwrap()).exp()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, N>, StatsError<T>> {
        let normal = StandardNormal;

        let mut proposal = self.mean
            + self.covmat.ref_cholesky_ltm()
                * SVector::<T, N>::from_iterator((0..N).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = self.mean
                + self.covmat.ref_cholesky_ltm()
                    * SVector::<T, N>::from_iterator((0..N).map(|_| rng.sample(normal)));

            attempts += 1;

            if attempts > 499 {
                error!(
                    "MultivariateND::draw_sample has failed to draw a valid sample after {} tries",
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

    fn get_valid_range(&self) -> [(T, T); N] {
        self.range
    }
}

impl<T, const N: usize> Add<&SVector<T, N>> for MultivariateND<T, N>
where
    T: Clone + RealField + Scalar,
{
    type Output = MultivariateND<T, N>;

    fn add(self, rhs: &SVector<T, N>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<T, const N: usize> AddAssign<&SVector<T, N>> for MultivariateND<T, N>
where
    T: Clone + RealField + Scalar,
{
    fn add_assign(&mut self, rhs: &SVector<T, N>) {
        self.mean += rhs
    }
}

impl<T, const N: usize> Mul<T> for MultivariateND<T, N>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateND<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            covmat: self.covmat * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<T, const N: usize> MulAssign<T> for MultivariateND<T, N>
where
    T: Copy + RealField + Scalar,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covmat *= rhs;
    }
}

impl<T, const N: usize> Sub<&SVector<T, N>> for MultivariateND<T, N>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateND<T, N>;

    fn sub(self, rhs: &SVector<T, N>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<T, const N: usize> SubAssign<&SVector<T, N>> for MultivariateND<T, N>
where
    T: Copy + RealField + Scalar,
{
    fn sub_assign(&mut self, rhs: &SVector<T, N>) {
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

        let covmat = CovMatrix::from_particles(array_view, None).unwrap();

        let mvpdf = MultivariateND::from_covmat(
            covmat,
            SVector::from([0.1, 0.0, 0.25]),
            [(-0.75, 0.75); 3],
        );

        assert!(
            (mvpdf.density(&SVector::from([0.2, 0.0, 0.35]).as_view()) - 0.1611928).abs() < 1e-6
        );

        assert!(
            (&mvpdf)
                .density_rel(&SVector::from([0.2, 0.1, 0.35]).as_view())
                .is_nan()
        );

        assert!(
            ((&mvpdf).draw_sample(&mut rng).unwrap() - SVector::from([-0.4150916, 0.0, 0.4898513]))
                .norm()
                < 1e-6
        );

        assert!((&mvpdf).validate_sample(&(&mvpdf).draw_sample(&mut rng).unwrap().as_view()));

        let mvpdf_ensbl =
            MultivariateND::from_particles(array_view, [(-0.75, 0.75); 3], None).unwrap();

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
            MultivariateND::from_particles(&array_1.as_view(), [(-2.0, 2.0); 3], None).unwrap();
        let mvpdf_2 =
            MultivariateND::from_particles(&array_2.as_view(), [(-2.0, 2.0); 3], None).unwrap();

        assert!((mvpdf_1.kullback_leibler_divergence(&mvpdf_2).unwrap() - 0.20391017).abs() < 1e-6);
    }
}
