use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    cmat::{OcnusStatsError, PDFExactDensity, PDF},
    covmat::CovMatrix,
};
use log::warn;
use nalgebra::SVector;
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// A multivariate normal PDF .
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultivariatePDF<const P: usize> {
    /// A [`CovMatrix`] that describes the underlying multivariate normal PDF.
    covm: CovMatrix,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<f32, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(f32, f32); P],
}

impl<const P: usize> MultivariatePDF<P> {
    /// Create a new [`MultivariatePDF`].
    pub fn new(covm: CovMatrix, mean: SVector<f32, P>, range: [(f32, f32); P]) -> Self {
        Self { covm, mean, range }
    }
}

impl<const P: usize> PDF<P> for &MultivariatePDF<P> {
    fn relative_density(&self, x: &nalgebra::SVectorView<f32, P>) -> f32 {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covm.inverse() * diff)[(0, 0)];

        (-0.5 * value).exp()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, P>, OcnusStatsError> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut proposal = self.mean
            + self.covm.cholesky_ltm()
                * SVector::<f32, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = self.mean
                + self.covm.cholesky_ltm()
                    * SVector::<f32, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            attempts += 1;

            if (attempts > 150) && (attempts % 150 == 0) {
                warn!(
                    "MultivariatePDF::draw_sample has failed to draw a valid sample after {} tries",
                    attempts
                );
            }
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(f32, f32); P] {
        self.range
    }
}

impl<const P: usize> PDFExactDensity<P> for &MultivariatePDF<P> {
    fn exact_density(&self, x: &nalgebra::SVectorView<f32, P>) -> f32 {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covm.inverse() * diff)[(0, 0)];

        (-0.5 * value).exp()
            / ((2.0 * std::f32::consts::PI).powi(P as i32) * self.covm.determinant()).sqrt()
    }
}

impl<const P: usize> Add<SVector<f32, P>> for MultivariatePDF<P> {
    type Output = MultivariatePDF<P>;

    fn add(self, rhs: SVector<f32, P>) -> Self::Output {
        Self {
            covm: self.covm,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<const P: usize> AddAssign<SVector<f32, P>> for MultivariatePDF<P> {
    fn add_assign(&mut self, rhs: SVector<f32, P>) {
        self.mean += rhs
    }
}

impl<const P: usize> Mul<f32> for MultivariatePDF<P> {
    type Output = MultivariatePDF<P>;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            covm: self.covm * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<const P: usize> Mul<MultivariatePDF<P>> for f32 {
    type Output = MultivariatePDF<P>;

    fn mul(self, rhs: MultivariatePDF<P>) -> Self::Output {
        Self::Output {
            covm: self * rhs.covm,
            mean: rhs.mean,
            range: rhs.range,
        }
    }
}

impl<const P: usize> MulAssign<f32> for MultivariatePDF<P> {
    fn mul_assign(&mut self, rhs: f32) {
        self.covm *= rhs;
    }
}

impl<const P: usize> Sub<SVector<f32, P>> for MultivariatePDF<P> {
    type Output = MultivariatePDF<P>;

    fn sub(self, rhs: SVector<f32, P>) -> Self::Output {
        Self {
            covm: self.covm,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<const P: usize> SubAssign<SVector<f32, P>> for MultivariatePDF<P> {
    fn sub_assign(&mut self, rhs: SVector<f32, P>) {
        self.mean -= rhs
    }
}
