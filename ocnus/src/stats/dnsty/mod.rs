mod mvpdf;
mod ptpdf;
mod uvpdf;

pub use crate::stats::{CovMatrix, CovarianceError, ParRng};
use crate::{
    alias::{PMatrix, PMatrixView, PMatrixViewMut},
    Fp,
};
use itertools::zip_eq;
pub use mvpdf::MultivariatePDF;
use nalgebra::{Const, Dim, Dyn, SVector, U1};
pub use ptpdf::ParticlePDF;
use rand::Rng;
use thiserror::Error;
pub use uvpdf::{
    ConstantPDF, CosinePDF, NormalPDF, ReciprocalPDF, UniformPDF, UnivariatePDF, UnivariatePDFs,
};

/// Collection of PDF related errors.
#[derive(Debug, Error)]
pub enum DensityError {
    #[error("covariance matrix error")]
    CovarianceError(CovarianceError),
    #[error("invalid parameter range")]
    InvalidParamRange((Fp, Fp)),
    #[error("missing covariance matrix")]
    MissingCovariance,
    #[error("reached the maximum number of iterations for a random sampler")]
    ReachedSamplerLimit(usize),
}

/// A trait that provides sampling functionality for a probability density function (PDF) with `P` dimensions.
pub trait ProbabilityDensityFunctionSampling<const P: usize>: Send + Sync {
    /// Draw a single parameter vector sample from the underlying probability density function
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, DensityError> {
        let mut sample = SVector::<Fp, P>::zeros();

        self.sample_fill(
            &mut sample.as_view_mut::<Const<P>, Dyn, U1, Const<P>>(),
            rng,
        )?;

        Ok(sample)
    }

    /// Draw a fixed number of parameter vector samples from the underlying probability density function using parallel iterators.
    fn sample_array(
        &self,
        size: usize,
        par_rng: &ParRng,
    ) -> Result<PMatrix<Const<P>>, DensityError> {
        let mut samples = PMatrix::<Const<P>>::zeros(size);

        self.par_sample_fill(
            &mut samples.as_view_mut::<Const<P>, Dyn, U1, Const<P>>(),
            par_rng,
        )?;

        Ok(samples)
    }

    /// Fill a contiguous parameter matrix view with random samples from the underlying probability density function.
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        rng: &mut impl Rng,
    ) -> Result<(), DensityError>;

    /// Fill a contiguous parameter matrix view with random samples from the underlying probability density function using parallel iterators.
    fn par_sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        par_rng: &ParRng,
    ) -> Result<(), DensityError>;

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
    //     rng: &mut impl Rng,
    // ) -> Result<Fp, DensityError> {
    //     let sample = self.0.sample_array(size, rng);

    //     Ok(0.0)
    // }

    // /// Computes the Kullback-Leibler divergence P(self | other) with respect to another density.
    // pub fn kullback_leibler_div(
    //     &self,
    //     other: &Self,
    //     size: usize,
    //     rng: &mut impl Rng,
    // ) -> Result<Fp, DensityError> {
    //     let sample_p = self.0.sample_array(size, rng);

    //     Ok(0.0)
    // }

    /// Draw a single parameter vector sample from the underlying probability density function
    pub fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, DensityError> {
        (&self.0).sample(rng)
    }

    // Draw a fixed number of parameter vector samples from the underlying probability density function using parallel iterators.
    pub fn sample_array(
        &self,
        size: usize,
        par_rng: &ParRng,
    ) -> Result<PMatrix<Const<P>>, DensityError> {
        (&self.0).sample_array(size, par_rng)
    }

    /// Fill a contiguous parameter matrix view with random samples from the underlying probability density function.
    pub fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        rng: &mut impl Rng,
    ) -> Result<(), DensityError> {
        (&self.0).sample_fill(pmatrix, rng)
    }

    /// Fill a contiguous parameter matrix view with random samples from the underlying probability density function using parallel iterators.
    pub fn par_sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        par_rng: &ParRng,
    ) -> Result<(), DensityError> {
        (&self.0).par_sample_fill(pmatrix, par_rng)
    }

    /// Returns the valid range for parameter vector samples.
    pub fn valid_range(&self) -> [(Fp, Fp); P] {
        (&self.0).valid_range()
    }
}
