//! Bayesian inference, statistics and probability density functions.

mod covmt;
mod mvpdf;
mod ptpdf;
mod uvpdf;

pub use covmt::CovMatrix;
pub use mvpdf::MultivariatePDF;
pub use ptpdf::ParticlePDF;
pub use uvpdf::{
    ConstantPDF, CosinePDF, NormalPDF, ReciprocalPDF, UniformPDF, UnivariatePDF, UnivariatePDFs,
};

use crate::{
    alias::{PMatrixView, PMatrixViewMut},
    Fp,
};
use itertools::zip_eq;
use nalgebra::{Const, Dim, Dyn, U1};
use rand::Rng;
use thiserror::Error;

/// Errors associated with the [`crate::stats`] module.
#[derive(Debug, Error)]
pub enum StatsError {
    #[error("array or matrix dimensions must be static and cannot be dynamic")]
    ConstantDimensionExpected,
    #[error("array or matrix dimensions are mismatched")]
    DimensionMismatch((usize, usize)),
    #[error("invalid parameter range")]
    InvalidParamRange((Fp, Fp)),
    #[error("missing covariance matrix")]
    MissingCovMat,
    #[error("reached the maximum number of iterations for a random sampler")]
    ReachedSamplerLimit(usize),
    #[error("matrix determinant is  zero or its absolute value is below the precision limit")]
    SingularMatrix(Fp),
}

/// A trait that provides sampling functionality for a probability density function (PDF) with `P` dimensions.
pub trait ProbabilityDensityFunctionSampling<const P: usize>: Send + Sync {
    /// Fill a contiguous parameter matrix view with random samples from the underlying probability density function.
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError>;

    /// Validate a single parameter vector by checking for out of bounds.
    fn validate_pvector<RStride: Dim, CStride: Dim>(
        &self,
        pvector: &PMatrixView<Const<P>, U1, RStride, CStride>,
    ) -> bool {
        zip_eq(pvector.iter(), self.valid_range().iter()).fold(true, |acc, (c, range)| {
            acc & ((&range.0 <= c) & (c <= &range.1))
        })
    }

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(Fp, Fp); P];
}

/// A generic probability density function (PDF) with `P` dimensions of sub-type `T`.
pub struct ProbabilityDensityFunction<T, const P: usize>(T);

impl<T, const P: usize> ProbabilityDensityFunction<T, P>
where
    for<'a> &'a T: ProbabilityDensityFunctionSampling<P>,
{
    // /// Computes the density p(x) for each position in `xs` using a kernel density estimation algorithm with `size` particles.
    // pub fn kernel_density_estimate<'a, RStride: Dim, CStride: Dim>(
    //     &self,
    //     xs: &PMatrixView<Const<P>, Dyn, RStride, CStride>,
    //     size: usize,
    //     rng: &mut (impl Rng + Clone),
    // ) -> Result<Fp, StatsError> {
    //     let sample = self.0.sample_array(size, rng);

    //     Ok(0.0)
    // }

    // /// Computes the Kullback-Leibler divergence P(self | other) with respect to another density.
    // pub fn kullback_leibler_div(
    //     &self,
    //     other: &Self,
    //     size: usize,
    //     rng: &mut (impl Rng + Clone),
    // ) -> Result<Fp, StatsError> {
    //     let sample_p = self.0.sample_array(size, rng);

    //     Ok(0.0)
    // }

    /// Fill a contiguous parameter matrix view with random samples from the underlying probability density function.
    pub fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        (&self.0).sample_fill(pmatrix, rng)
    }

    /// Returns the valid range for parameter vector samples.
    pub fn valid_range(&self) -> [(Fp, Fp); P] {
        (&self.0).valid_range()
    }
}
