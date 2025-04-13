use crate::{
    fXX,
    math::{CovMatrix, MathError, T, exp, ln, powi, sqrt},
    prodef::{OcnusProDeF, ProDeFError},
};
use log::warn;
use nalgebra::{Const, DMatrix, Dyn, MatrixView, SVector, SVectorView};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A multivariate normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultivariateND<T, const P: usize>
where
    T: fXX,
{
    /// A [`CovMatrix<T>`] that describes the underlying multivariate normal PDF.
    covmat: CovMatrix<T>,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<T, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(T, T); P],
}

impl<T, const P: usize> MultivariateND<T, P>
where
    T: fXX,
{
    /// Estimates the density  at a specific position `x`.
    pub fn density(&self, x: &SVectorView<T, P>) -> T {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_inverse_matrix() * diff)[(0, 0)];

        exp!(T!(-0.5) * value) / sqrt!(powi!(T::two_pi(), P as i32) * self.covmat.determinant())
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
    ) -> Result<Self, MathError<T>> {
        let covmat = CovMatrix::from_particles(&particles.as_view(), weights)?;

        Ok(Self {
            covmat,
            mean: particles.column_mean(),
            range,
        })
    }

    /// Compute the Kullback-Leibler divergence between two [`MultivariateND`].
    pub fn kullback_leibler_divergence(&self, other: &MultivariateND<T, P>) -> T {
        let l_0 = self.covmat.ref_cholesky_ltm();
        let mu_0 = self.mean;

        let l_1 = other.covmat.ref_cholesky_ltm();
        let mu_1 = other.mean;

        let m = l_0.clone().cholesky().unwrap().solve(&l_1);
        let y = l_1.clone().cholesky().unwrap().solve(&(&mu_1 - &mu_0));

        T!(0.5)
            * (m.iter().sum::<T>() - T::from_usize(P).unwrap()
                + y.norm()
                + T!(2.0)
                    * l_1
                        .diagonal()
                        .iter()
                        .zip(l_0.diagonal().iter())
                        .map(|(a, b)| ln!(*a / *b))
                        .sum::<T>())
    }

    /// Returns a reference to the lower triangular matrix L from the Cholesky decomposition
    /// of the covariance matrix.
    pub fn ref_cholesky_ltm(&self) -> &DMatrix<T> {
        &self.covmat.ref_cholesky_ltm()
    }

    /// Returns a reference to the inverse of the underlying covariance matrix.
    pub fn ref_inverse_matrix(&self) -> &DMatrix<T> {
        &self.covmat.ref_inverse_matrix()
    }

    /// Returns a reference to the underlying covariance matrix.
    pub fn ref_matrix(&self) -> &DMatrix<T> {
        &self.covmat.ref_matrix()
    }
}

impl<T, const P: usize> OcnusProDeF<T, P> for &MultivariateND<T, P>
where
    T: fXX,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &SVectorView<T, P>) -> T {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_inverse_matrix() * diff)[(0, 0)];

        exp!(T!(-0.5) * value)
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, ProDeFError<T>> {
        let normal = Normal::new(T::zero(), T::one()).unwrap();

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

            if (attempts > 150) && (attempts % 150 == 0) {
                warn!(
                    "MultivariateND::draw_sample has failed to draw a valid sample after {} tries",
                    attempts
                );
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
    T: fXX,
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
    T: fXX,
{
    fn add_assign(&mut self, rhs: &SVector<T, P>) {
        self.mean += rhs
    }
}

impl<T, const P: usize> Mul<T> for MultivariateND<T, P>
where
    T: fXX,
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
    T: fXX,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covmat *= rhs;
    }
}

impl<T, const P: usize> Sub<&SVector<T, P>> for MultivariateND<T, P>
where
    T: fXX,
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
    T: fXX,
{
    fn sub_assign(&mut self, rhs: &SVector<T, P>) {
        self.mean -= rhs
    }
}
