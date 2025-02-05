use crate::{
    stats::{CovMatrix, StatsError, PDF},
    Fp,
};
use nalgebra::{Const, SVector};
use rand::Rng;
use rand_distr::Normal;
use serde::Serialize;

/// A multivariate normal PDF .
#[derive(Clone, Debug, Serialize)]
pub struct MultivariatePDF<const P: usize> {
    /// A [`CovMatrix`] object that describes the multivariate normal PDF.
    covm: CovMatrix<Const<P>>,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<Fp, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],
}

impl<const P: usize> MultivariatePDF<P> {
    /// Create a new [`MultivariatePDF`] object.
    pub fn new(covm: CovMatrix<Const<P>>, mean: SVector<Fp, P>, range: [(Fp, Fp); P]) -> Self {
        Self { covm, mean, range }
    }
}

impl<const P: usize> PDF<P> for &MultivariatePDF<P> {
    fn relative_likelihood(&self, x: &nalgebra::SVectorView<Fp, P>) -> Fp {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covm.inverse() * diff)[(0, 0)];

        (-0.5 * value).exp()
            / ((2.0 * std::f64::consts::PI as Fp).powi(P as i32) * self.covm.determinant()).sqrt()
    }

    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut proposal = self.mean
            + self.covm.cholesky_ltm()
                * SVector::<Fp, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut limit = 0;

        while !self.validate_sample(&proposal.as_view()) {
            if limit > 100 {
                return Err(StatsError::SamplerLimit(100));
            }

            proposal = self.mean
                + self.covm.cholesky_ltm()
                    * SVector::<Fp, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            limit += 1;
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.range
    }
}
