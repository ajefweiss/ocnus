use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    stats::CovMatrix,
    stats::{PDF, PDFExactDensity, StatsError},
};
use log::warn;
use nalgebra::{SVector, SVectorView};
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// A multivariate normal PDF .
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFMultivariate<const P: usize> {
    /// A [`CovMatrix`] that describes the underlying multivariate normal PDF.
    covmat: CovMatrix,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<f64, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(f64, f64); P],
}

impl<const P: usize> PDFMultivariate<P> {
    /// Create a [`PDFMultivariate`] from a covariance matrix.
    pub fn from_covmat(covmat: CovMatrix, mean: SVector<f64, P>, range: [(f64, f64); P]) -> Self {
        Self {
            covmat,
            mean,
            range,
        }
    }
}

impl<const P: usize> PDF<P> for &PDFMultivariate<P> {
    fn relative_density(&self, x: &SVectorView<f64, P>) -> f64 {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.inverse_matrix() * diff)[(0, 0)];

        (-0.5 * value).exp()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f64, P>, StatsError> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut proposal = self.mean
            + self.covmat.cholesky_ltm()
                * SVector::<f64, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = self.mean
                + self.covmat.cholesky_ltm()
                    * SVector::<f64, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

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

    fn valid_range(&self) -> [(f64, f64); P] {
        self.range
    }
}

impl<const P: usize> PDFExactDensity<P> for &PDFMultivariate<P> {
    fn exact_density(&self, x: &SVectorView<f64, P>) -> f64 {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.inverse_matrix() * diff)[(0, 0)];

        (-0.5 * value).exp()
            / ((2.0 * std::f64::consts::PI).powi(P as i32) * self.covmat.determinant()).sqrt()
    }
}

impl<const P: usize> Add<&SVector<f64, P>> for PDFMultivariate<P> {
    type Output = PDFMultivariate<P>;

    fn add(self, rhs: &SVector<f64, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<const P: usize> AddAssign<&SVector<f64, P>> for PDFMultivariate<P> {
    fn add_assign(&mut self, rhs: &SVector<f64, P>) {
        self.mean += rhs
    }
}

impl<const P: usize> Mul<f64> for PDFMultivariate<P> {
    type Output = PDFMultivariate<P>;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            covmat: self.covmat * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<const P: usize> Mul<PDFMultivariate<P>> for f64 {
    type Output = PDFMultivariate<P>;

    fn mul(self, rhs: PDFMultivariate<P>) -> Self::Output {
        Self::Output {
            covmat: self * rhs.covmat,
            mean: rhs.mean,
            range: rhs.range,
        }
    }
}

impl<const P: usize> MulAssign<f64> for PDFMultivariate<P> {
    fn mul_assign(&mut self, rhs: f64) {
        self.covmat *= rhs;
    }
}

impl<const P: usize> Sub<&SVector<f64, P>> for PDFMultivariate<P> {
    type Output = PDFMultivariate<P>;

    fn sub(self, rhs: &SVector<f64, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<const P: usize> SubAssign<&SVector<f64, P>> for PDFMultivariate<P> {
    fn sub_assign(&mut self, rhs: &SVector<f64, P>) {
        self.mean -= rhs
    }
}
