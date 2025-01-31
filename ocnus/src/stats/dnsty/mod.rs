mod cvmat;
mod ptpdf;
mod uvpdf;

use crate::{Fp, PVector, PVectors, PVectorsView, PVectorsViewMut};
use cvmat::CovMatrixError;
use nalgebra::{Const, Dyn, U1};
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

pub trait ProbabilityDensityFunctionSampling<const D: usize> {
    /// Draw a single parameter vector sample from the underlying probability density function
    fn sample(&self, rng: &mut impl Rng) -> Result<PVector<D>, DensityError> {
        let mut sample = PVector::<D>::zeros();

        self.sample_fill(
            &mut sample.as_view_mut::<Const<D>, Dyn, U1, Const<D>>(),
            rng,
        )?;

        Ok(sample)
    }

    /// Draw a fixed number of parameter vector samples from the underlying probability density function.
    fn sample_array(&self, size: usize, rng: &mut impl Rng) -> Result<PVectors<D>, DensityError> {
        let mut samples = PVectors::<D>::zeros(size);

        self.sample_fill(
            &mut samples.as_view_mut::<Const<D>, Dyn, U1, Const<D>>(),
            rng,
        )?;

        Ok(samples)
    }

    /// Fill a contiguous parameter vector array view with random samples from the underlying probability density function.
    fn sample_fill(
        &self,
        view: &mut PVectorsViewMut<D, Dyn>,
        rng: &mut impl Rng,
    ) -> Result<(), DensityError>;

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(Fp, Fp); D];
}

/// A generic probability density function (PDF) with `D` dimensions of sub-type `T`.
pub struct ProbabilityDensityFunction<T, const D: usize>(T);

impl<T, const D: usize> ProbabilityDensityFunction<T, D>
where
    T: ProbabilityDensityFunctionSampling<D>,
{
    /// Computes the density p(x) for each position in `xs` using a kernel density estimation algorithm with `size` particles.
    pub fn kernel_density_estimate(
        &self,
        xs: PVectorsView<D, Dyn>,
        size: usize,
        rng: &mut impl Rng,
    ) -> Result<Fp, DensityError> {
        let sample = self.0.sample_array(size, rng);

        Ok(0.0)
    }

    /// Computes the Kullback-Leibler divergence D(self | other) with respect to another density.
    pub fn kullback_leibler_div(
        &self,
        other: &Self,
        size: usize,
        rng: &mut impl Rng,
    ) -> Result<Fp, DensityError> {
        let sample_p = self.0.sample_array(size, rng);

        Ok(0.0)
    }

    /// Draw a single parameter vector sample from the underlying probability density function
    pub fn sample(&self, rng: &mut impl Rng) -> Result<PVector<D>, DensityError> {
        self.0.sample(rng)
    }

    /// Draw a single parameter vector sample from the underlying probability density function
    pub fn sample_fill(
        &self,
        view: &mut PVectorsViewMut<D, Dyn>,
        rng: &mut impl Rng,
    ) -> Result<(), DensityError> {
        self.0.sample_fill(view, rng)
    }

    /// Returns the valid range for parameter vector samples.
    pub fn valid_range(&self) -> [(Fp, Fp); D] {
        self.0.valid_range()
    }
}
