mod cvmat;
// mod mvpdf;
// mod ptpdf;
// mod uvpdf;

use crate::{
    alias::{PMatrixView, PMatrixViewMut},
    Fp,
};
use cvmat::CovMatrixError;
use itertools::zip_eq;
use nalgebra::{Const, Dim, Dyn, Matrix, SVector, VecStorage, U1};
use rand::Rng;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DensityError {
    #[error("covariance matrix error")]
    CovMatrixError(CovMatrixError),
    #[error("invalid parameter range")]
    InvalidParamRange((Fp, Fp)),
    #[error("reached the maximum number of iterations for a random sampler")]
    ReachedSamplerLimit(usize),
}

/// A trait that defines the sampling functionality of a probability density function (PDF) with `P` dimensions.
pub trait ProbabilityDensityFunctionSampling<const P: usize> {
    /// Draw a single parameter vector sample from the underlying probability density function
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, DensityError> {
        let mut sample = SVector::<Fp, P>::zeros();

        self.sample_fill(
            &mut sample.as_view_mut::<Const<P>, Dyn, U1, Const<P>>(),
            rng,
        )?;

        Ok(sample)
    }

    /// Draw a fixed number of parameter vector samples from the underlying probability density function.
    fn sample_array(
        &self,
        size: usize,
        rng: &mut impl Rng,
    ) -> Result<Matrix<Fp, Const<P>, Dyn, VecStorage<Fp, Const<P>, Dyn>>, DensityError> {
        let mut samples = Matrix::<Fp, Const<P>, Dyn, VecStorage<Fp, Const<P>, Dyn>>::zeros(size);

        self.sample_fill(
            &mut samples.as_view_mut::<Const<P>, Dyn, U1, Const<P>>(),
            rng,
        )?;

        Ok(samples)
    }

    /// Fill a contiguous parameter matrix view with random samples from the underlying probability density function.
    fn sample_fill<'a, RStride: Dim, CStride: Dim>(
        &self,
        view: &mut PMatrixViewMut<'a, Const<P>, Dyn, RStride, CStride>,
        rng: &mut impl Rng,
    ) -> Result<(), DensityError>;

    /// Validate a single parameter vector by checking for out of bounds.
    fn validate_pvector<'a, RStride: Dim, CStride: Dim>(
        &self,
        view: &PMatrixView<'a, Const<P>, U1, RStride, CStride>,
    ) -> bool {
        zip_eq(view.iter(), self.valid_range().iter()).fold(true, |acc, (c, range)| {
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
    T: ProbabilityDensityFunctionSampling<P>,
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
        self.0.sample(rng)
    }

    /// Draw a single parameter vector sample from the underlying probability density function
    pub fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        view: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        rng: &mut impl Rng,
    ) -> Result<(), DensityError> {
        self.0.sample_fill(view, rng)
    }

    /// Returns the valid range for parameter vector samples.
    pub fn valid_range(&self) -> [(Fp, Fp); P] {
        self.0.valid_range()
    }
}
