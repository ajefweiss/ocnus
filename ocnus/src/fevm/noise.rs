use crate::obser::ObserVec;
use nalgebra::DVector;
use rand::Rng;
use rand_distr::Normal;

/// A trait that must be implemented for any type that acts as a FEVM noise generator.
pub trait FEVMNoiseGen<const N: usize>: Clone + Sync {
    /// Generate a random noise time-series.
    fn generate_noise(&self, size: usize, rng: &mut impl Rng) -> DVector<ObserVec<N>>;
}

/// A zero noise generator.
#[derive(Clone)]
pub struct FEVMNoiseZero;

impl<const N: usize> FEVMNoiseGen<N> for FEVMNoiseZero {
    fn generate_noise(&self, _size: usize, _rng: &mut impl Rng) -> DVector<ObserVec<N>> {
        DVector::zeros(N)
    }
}

/// A univariate random normal FEVM noise generator.
#[derive(Clone)]
pub struct FEVMNoiseGaussian(f64);

impl<const N: usize> FEVMNoiseGen<N> for FEVMNoiseGaussian {
    fn generate_noise(&self, size: usize, rng: &mut impl Rng) -> DVector<ObserVec<N>> {
        let normal = Normal::new(0.0, self.0).unwrap();

        DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])))
    }
}
