//! Noise generators.

use std::marker::PhantomData;

use nalgebra::DVector;
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    OFloat,
    obser::{ObserVec, ScObsSeries},
    stats::CovMatrix,
};

pub struct FEVMNoise<T, const N: usize, F>(F, u64, PhantomData<T>)
where
    T: OFloat,
    F: FEVMNoiseGenerator<T, N>;

impl<T, const N: usize, F> FEVMNoise<T, N, F>
where
    T: OFloat,
    F: FEVMNoiseGenerator<T, N>,
{
    /// Generate a random noise time-series.
    pub fn generate_noise(
        &mut self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        rng: &mut impl Rng,
    ) -> DVector<ObserVec<T, N>> {
        self.0.generate_noise(series, rng)
    }

    /// Return the random number generator seed.
    pub fn seed(&self) -> u64 {
        self.1
    }
}

/// A trait that must be implemented for any type that acts as a FEVM noise generator.
pub trait FEVMNoiseGenerator<T, const N: usize>: Clone + Sync
where
    T: OFloat,
{
    /// Generate a random noise time-series.
    fn generate_noise(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        rng: &mut impl Rng,
    ) -> DVector<ObserVec<T, N>>;
}

/// A univariate normal FEVM noise generator.
#[derive(Clone)]
pub struct FEVMNoiseGaussian<T>(pub T);

impl<T, const N: usize> FEVMNoiseGenerator<T, N> for FEVMNoiseGaussian<T>
where
    T: OFloat,
    StandardNormal: Distribution<T>,
{
    fn generate_noise(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        rng: &mut impl Rng,
    ) -> DVector<ObserVec<T, N>> {
        let normal = Normal::new(T::zero(), self.0).unwrap();
        let size = series.len();

        DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])))
    }
}

/// A multivariate normal FEVM noise generator.
#[derive(Clone)]
pub struct FEVMNoiseMultivariate<T>(pub CovMatrix<T>)
where
    T: OFloat;

impl<T, const N: usize> FEVMNoiseGenerator<T, N> for FEVMNoiseMultivariate<T>
where
    T: OFloat,
    StandardNormal: Distribution<T>,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    fn generate_noise(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        rng: &mut impl Rng,
    ) -> DVector<ObserVec<T, N>> {
        let normal = Normal::new(T::zero(), T::one()).unwrap();
        let size = series.len();

        let mut result =
            DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])));

        for i in 0..N {
            let values = self.0.cholesky_ltm()
                * DVector::from_iterator(size, (0..size).map(|_| rng.sample(normal)));

            result
                .iter_mut()
                .zip(values.row_iter())
                .for_each(|(res, val)| res[i] = val[(0, 0)]);
        }

        result
    }
}
