use log::error;
use nalgebra::{Const, DMatrix, Dyn, MatrixView, RealField, SVector, SVectorView, Scalar};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{CovMatrix, OcnusPDF, StatsError};

/// A multivariate normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultivariateND<T, const P: usize>
where
    T: Scalar,
{
    /// A [`CovMatrix`] that describes the underlying multivariate normal probability density function.
    covmat: CovMatrix<T>,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<T, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(T, T); P],
}

impl<T, const P: usize> MultivariateND<T, P>
where
    T: Copy + RealField + Scalar,
{
    /// Estimates the exact density  at a specific position `x`.
    pub fn density(&self, x: &SVectorView<T, P>) -> T {
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

    /// Create a [`MultivariateND`] from a covariance matrix.
    pub fn from_covmat(covmat: CovMatrix<T>, mean: SVector<T, P>, range: [(T, T); P]) -> Self {
        Self {
            covmat,
            mean,
            range,
        }
    }

    /// Create a [`MultivariateND`] from an ensemble of particles.
    pub fn from_particles<'a>(
        particles: &MatrixView<'a, T, Const<P>, Dyn>,
        range: [(T, T); P],
        weights: Option<&[T]>,
    ) -> Option<Self>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let covmat = CovMatrix::from_particles(&particles.as_view(), weights)?;

        Some(Self {
            covmat,
            mean: particles.column_mean(),
            range,
        })
    }

    /// Compute the Kullback-Leibler divergence between two [`MultivariateND`].
    pub fn kullback_leibler_divergence(&self, other: &MultivariateND<T, P>) -> T
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let l_0 = self.covmat.ref_cholesky_ltm();
        let mu_0 = self.mean;

        let l_1 = other.covmat.ref_cholesky_ltm();
        let mu_1 = other.mean;

        let m = l_0.clone().cholesky().unwrap().solve(&l_1);
        let y = l_1.clone().cholesky().unwrap().solve(&(&mu_1 - &mu_0));

        (m.iter().sum::<T>() - T::from_usize(P).unwrap()
            + y.norm()
            + T::from_usize(2).unwrap()
                * l_1
                    .diagonal()
                    .iter()
                    .zip(l_0.diagonal().iter())
                    .map(|(a, b)| (*a / *b).ln())
                    .sum::<T>())
            / T::from_usize(2).unwrap()
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
}

impl<T, const P: usize> OcnusPDF<T, P> for &MultivariateND<T, P>
where
    T: Copy + RealField,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &SVectorView<T, P>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_matrix_inverse() * diff)[(0, 0)];

        (-value / T::from_usize(2).unwrap()).exp()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>> {
        let normal = StandardNormal;

        let mut proposal = self.mean
            + self.covmat.ref_cholesky_ltm()
                * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = self.mean
                + self.covmat.ref_cholesky_ltm()
                    * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

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

    fn get_valid_range(&self) -> [(T, T); P] {
        self.range
    }
}

impl<T, const P: usize> Add<&SVector<T, P>> for MultivariateND<T, P>
where
    T: Clone + RealField + Scalar,
{
    type Output = MultivariateND<T, P>;

    fn add(self, rhs: &SVector<T, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<T, const P: usize> AddAssign<&SVector<T, P>> for MultivariateND<T, P>
where
    T: Clone + RealField + Scalar,
{
    fn add_assign(&mut self, rhs: &SVector<T, P>) {
        self.mean += rhs
    }
}

impl<T, const P: usize> Mul<T> for MultivariateND<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateND<T, P>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            covmat: self.covmat * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<T, const P: usize> MulAssign<T> for MultivariateND<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covmat *= rhs;
    }
}

impl<T, const P: usize> Sub<&SVector<T, P>> for MultivariateND<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateND<T, P>;

    fn sub(self, rhs: &SVector<T, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<T, const P: usize> SubAssign<&SVector<T, P>> for MultivariateND<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn sub_assign(&mut self, rhs: &SVector<T, P>) {
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
                    (idx % 3) as f32 + rng.sample::<f32, StandardNormal>(uniform)
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
                .density_rel(&SVector::from([2.2, 0.0, 0.35]).as_view())
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
}
