use crate::{
    obser::{ObserVec, ScObsSeries},
    stats::CovMatrix,
};
use nalgebra::{DMatrix, DVector, DVectorView, RealField, Scalar};
use num_traits::Float;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// A generic noise generator for [`FEVM`].
#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum FEVMNoise<T>
where
    T: Copy + Scalar,
{
    Gaussian(T, u64),
    Multivariate(CovMatrix<T>, u64),
}

impl<T> FEVMNoise<T>
where
    T: Copy + Display + Float + RealField + Scalar,
    StandardNormal: Distribution<T>,
{
    /// Generate a random noise time-series.
    pub fn generate_noise<const N: usize>(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        rng: &mut impl Rng,
    ) -> DVector<ObserVec<T, N>> {
        match self {
            FEVMNoise::Gaussian(std_dev, ..) => {
                let normal = Normal::new(T::zero(), *std_dev).unwrap();
                let size = series.len();

                DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])))
            }
            FEVMNoise::Multivariate(covmat, ..) => {
                let normal = Normal::new(T::zero(), T::one()).unwrap();
                let size = series.len();

                let mut result = DVector::from_iterator(
                    size,
                    (0..size).map(|_| ObserVec([rng.sample(normal); N])),
                );

                for i in 0..N {
                    let values = covmat.cholesky_ltm()
                        * DVector::from_iterator(size, (0..size).map(|_| rng.sample(normal)));

                    result
                        .iter_mut()
                        .zip(values.row_iter())
                        .for_each(|(res, val)| res[i] = val[(0, 0)]);
                }

                result
            }
        }
    }

    /// Increment randon number seed.
    pub fn increment_seed(&mut self) {
        match self {
            FEVMNoise::Gaussian(.., seed) => {
                *seed += 1;
            }
            FEVMNoise::Multivariate(.., seed) => {
                *seed += 1;
            }
        }
    }

    /// Initialize a new random number generator using the base seed
    pub fn initialize_rng(&self, multiplier: u64, offset: u64) -> Xoshiro256PlusPlus {
        match self {
            FEVMNoise::Gaussian(.., seed) | FEVMNoise::Multivariate(.., seed) => {
                Xoshiro256PlusPlus::seed_from_u64(*seed * multiplier + offset)
            }
        }
    }

    /// Compute the likelihood for an observation `x` with expected mean `mu`.
    pub fn multivariate_likelihood<const N: usize>(
        &self,
        x: &DVectorView<ObserVec<T, N>>,
        mu: &ScObsSeries<T, ObserVec<T, N>>,
    ) -> T {
        let x_flat = x
            .iter()
            .flat_map(|x_obs| x_obs.iter().cloned().collect::<Vec<T>>())
            .collect::<Vec<T>>();

        let mu_flat = mu
            .into_iter()
            .flat_map(|mu_scobs| mu_scobs.observation().iter().cloned().collect::<Vec<T>>())
            .collect::<Vec<T>>();

        match self {
            FEVMNoise::Gaussian(std_dev, ..) => {
                let covmat = CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(x.len(), x.len(), *std_dev).as_view(),
                )
                .unwrap();

                covmat.multivariate_likelihood(x_flat, mu_flat)
            }
            FEVMNoise::Multivariate(covmat, ..) => covmat.multivariate_likelihood(x_flat, mu_flat),
        }
    }
}

impl<T> Default for FEVMNoise<T>
where
    T: Copy + RealField,
{
    fn default() -> Self {
        FEVMNoise::Gaussian(T::one(), 0)
    }
}
