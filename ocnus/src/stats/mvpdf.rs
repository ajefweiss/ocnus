use crate::stats::{CovMatrix, PDF, PDFExactDensity, StatsError};
use log::warn;
use nalgebra::{RealField, SVector, SVectorView, Scalar};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A multivariate normal PDF .
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFMultivariate<T, const P: usize>
where
    T: Copy + Scalar,
{
    /// A [`CovMatrix<T>`] that describes the underlying multivariate normal PDF.
    covmat: CovMatrix<T>,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<T, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(T, T); P],
}

impl<T, const P: usize> PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    /// Create a [`PDFMultivariate`] from a covariance matrix.
    pub fn from_covmat(covmat: CovMatrix<T>, mean: SVector<T, P>, range: [(T, T); P]) -> Self {
        Self {
            covmat,
            mean,
            range,
        }
    }

    /// Compute the Kullback-Leibler divergence between two [`PDFMultivariate`].
    pub fn kullback_leibler_divergence(&self, other: &PDFMultivariate<T, P>) -> T {
        let _l_0 = self.covmat.cholesky_ltm();
        let _l_1 = other.covmat.cholesky_ltm();

        unimplemented!();
    }
}

impl<T, const P: usize> PDF<T, P> for &PDFMultivariate<T, P>
where
    T: Copy + Float + RealField + Scalar,
    StandardNormal: Distribution<T>,
{
    fn relative_density(&self, x: &SVectorView<T, P>) -> T {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.inverse_matrix() * diff)[(0, 0)];

        Float::exp(T::from_f64(-0.5).unwrap() * value)
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>> {
        let normal = Normal::new(T::zero(), T::one()).unwrap();

        let mut proposal = self.mean
            + self.covmat.cholesky_ltm()
                * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = self.mean
                + self.covmat.cholesky_ltm()
                    * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            attempts += 1;

            if (attempts > 150) && (attempts % 150 == 0) {
                warn!(
                    "PDFMultivariate::draw_sample has failed to draw a valid sample after {} tries",
                    attempts
                );
            }
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(T, T); P] {
        self.range
    }
}

impl<T, const P: usize> PDFExactDensity<T, P> for &PDFMultivariate<T, P>
where
    T: Copy + Float + RealField + Scalar,
    StandardNormal: Distribution<T>,
{
    fn exact_density(&self, x: &SVectorView<T, P>) -> T {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.inverse_matrix() * diff)[(0, 0)];

        Float::exp(T::from_f64(-0.5).unwrap() * value)
            / Float::sqrt(
                Float::powi(T::from_f64(2.0).unwrap() * T::pi(), P as i32)
                    * self.covmat.determinant(),
            )
    }
}

impl<T, const P: usize> Add<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = PDFMultivariate<T, P>;

    fn add(self, rhs: &SVector<T, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<T, const P: usize> AddAssign<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn add_assign(&mut self, rhs: &SVector<T, P>) {
        self.mean += rhs
    }
}

impl<T, const P: usize> Mul<T> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = PDFMultivariate<T, P>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            covmat: self.covmat * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<const P: usize> Mul<PDFMultivariate<f32, P>> for f32 {
    type Output = PDFMultivariate<f32, P>;

    fn mul(self, rhs: PDFMultivariate<f32, P>) -> Self::Output {
        Self::Output {
            covmat: self * rhs.covmat,
            mean: rhs.mean,
            range: rhs.range,
        }
    }
}

impl<const P: usize> Mul<PDFMultivariate<f64, P>> for f64 {
    type Output = PDFMultivariate<f64, P>;

    fn mul(self, rhs: PDFMultivariate<f64, P>) -> Self::Output {
        Self::Output {
            covmat: self * rhs.covmat,
            mean: rhs.mean,
            range: rhs.range,
        }
    }
}

impl<T, const P: usize> MulAssign<T> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covmat *= rhs;
    }
}

impl<T, const P: usize> Sub<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = PDFMultivariate<T, P>;

    fn sub(self, rhs: &SVector<T, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<T, const P: usize> SubAssign<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn sub_assign(&mut self, rhs: &SVector<T, P>) {
        self.mean -= rhs
    }
}
