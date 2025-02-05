//! Bayesian inference, probability density functions and statistics.

mod covm;
mod mvpdf;
mod ptpdf;
mod uvpdf;

pub use covm::{covariance, covariance_with_weights, CovMatrix};
use itertools::zip_eq;
pub use mvpdf::MultivariatePDF;
pub use ptpdf::{ParticleMutPDF, ParticlePDF, ParticleRefPDF};
pub use uvpdf::{ConstantPDF, PUnivariatePDF, UniformPDF, UnivariatePDF};

use nalgebra::{SVector, SVectorView};
use rand::Rng;

use crate::Fp;
use thiserror::Error;

/// Errors associated with the [`crate::stats`] module.
#[derive(Debug, Error)]
pub enum StatsError {
    #[error("array or matrix dimensions are mismatched")]
    BadDimensions((usize, usize)),
    #[error("invalid parameter range")]
    InvalidParamRange((Fp, Fp)),
    #[error("reached the maximum number of allowed sampler iterations")]
    SamplerLimit(usize),
    #[error("matrix determinant is zero or its absolute value is below the precision limit")]
    SingularMatrix(Fp),
}

/// A trait that provides sampling functionality for a P-dimensional PDF.
pub trait PDF<const P: usize>: Sync {
    /// Estimates the relative likelihood value at a specific position `x`.
    /// The result is not necessarily normalized depending on the specific PDF.
    fn relative_likelihood(&self, x: &SVectorView<Fp, P>) -> Fp;

    /// Draw a single params vector from the underlying PDF.
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError>;

    /// Validate a sample by checking for out of bounds.
    fn validate_sample(&self, sample: &SVectorView<Fp, P>) -> bool {
        zip_eq(sample.iter(), self.valid_range().iter()).fold(true, |acc, (c, range)| {
            acc & ((&range.0 <= c) & (c <= &range.1))
        })
    }

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(Fp, Fp); P];
}
