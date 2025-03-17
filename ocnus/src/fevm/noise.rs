//! Noise generators.

use crate::{ScObsSeries, obser::ObserVec, stats::CovMatrix};
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

/// A univariate normal FEVM noise generator.
#[derive(Clone)]
pub struct FEVMNoiseGaussian(pub f32);

impl<const N: usize> FEVMNoiseGenerator<N> for FEVMNoiseGaussian {
    fn generate_noise(&self, size: usize, rng: &mut impl Rng) -> DVector<ObserVec<N>> {
        let normal = Normal::new(0.0, self.0).unwrap();

        DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])))
    }
}

/// A multivariate normal FEVM noise generator.
#[derive(Clone)]
pub struct FEVMNoiseMultivariate(pub CovMatrix);

impl FEVMNoiseMultivariate {
    /// Wrapper function for [`CovMatrix::multivariate_likelihood`]
    pub fn mvlh<const N: usize>(
        &self,
        x: &[ObserVec<N>],
        series: &ScObsSeries<ObserVec<N>>,
    ) -> f32 {
        let mut x_flat = x.iter().flat_map(|value| value.0).collect::<Vec<f32>>();
        let mut mu_flat = series
            .into_iter()
            .flat_map(|value| value.observation().unwrap_or(&ObserVec::default()).0)
            .collect::<Vec<f32>>();

        // Correct NaNs if, and only if, appropriate.
        x_flat
            .iter_mut()
            .zip(mu_flat.iter_mut())
            .for_each(|(x, y)| {
                if x.is_nan() & y.is_nan() {
                    *x = 0.0;
                    *y = 0.0;
                }
            });

        self.0.multivariate_likelihood(x_flat, mu_flat)
    }
}

impl<const N: usize> FEVMNoiseGenerator<N> for FEVMNoiseMultivariate {
    fn generate_noise(&self, size: usize, rng: &mut impl Rng) -> DVector<ObserVec<N>> {
        let normal = Normal::new(0.0, 1.0).unwrap();

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
