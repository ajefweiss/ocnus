use crate::obser::ObserVec;
use nalgebra::DVector;
use rand::Rng;
use rand_distr::Normal;

/// A trait that must be implemented for any type that acts as a FEVM noise generator.
pub trait FEVMNoiseGenerator<const N: usize>: Clone + Sync {
    /// Generate a random noise time-series.
    fn generate_noise(&self, size: usize, rng: &mut impl Rng) -> DVector<ObserVec<N>>;
}

/// A zero noise generator.
#[derive(Clone)]
pub struct FEVMNoiseNull;

impl<const N: usize> FEVMNoiseGenerator<N> for FEVMNoiseNull {
    fn generate_noise(&self, _size: usize, _rng: &mut impl Rng) -> DVector<ObserVec<N>> {
        DVector::zeros(N)
    }
}

/// A univariate random normal FEVM noise generator.
#[derive(Clone)]
pub struct FEVMNoiseGaussian(f64);

impl<const N: usize> FEVMNoiseGenerator<N> for FEVMNoiseGaussian {
    fn generate_noise(&self, size: usize, rng: &mut impl Rng) -> DVector<ObserVec<N>> {
        let normal = Normal::new(0.0, self.0).unwrap();

        DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])))
    }
}
