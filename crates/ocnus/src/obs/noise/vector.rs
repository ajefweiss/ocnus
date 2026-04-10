use crate::obs::data::ObsVec;
use crate::obs::{conf::ObsTime, noise::ObsNoise};
use nalgebra::allocator::Allocator;
use nalgebra::{DVectorViewMut, DefaultAllocator, Dyn, RealField, U1};
use prodef::{Density, domain::UDomain, multinormal::MultiNormalDensity};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Generic N-dimensional observation vector noise
#[derive(Clone, Debug, Deserialize, Serialize)]
#[allow(missing_docs)]
pub enum ObsVecNoise<T>
where
    T: RealField,
{
    AdditiveNormal(T, u64),
    AdditiveMultiNormal(MultiNormalDensity<T, Dyn, UDomain<Dyn>>, u64),
    MultiplicativeNormal(T, u64),
}

impl<T, const N: usize, OC> ObsNoise<T, OC, ObsVec<T, N>> for ObsVecNoise<T>
where
    T: RealField,
    OC: ObsTime<T>,
    DefaultAllocator: Allocator<Dyn> + Allocator<U1, Dyn> + Allocator<Dyn, Dyn>,
    StandardNormal: Distribution<T>,
{
    fn generate_noise(&self, data: &mut DVectorViewMut<ObsVec<T, N>>, rng: &mut impl rand::RngExt) {
        match self {
            ObsVecNoise::AdditiveNormal(std_dev, ..) => {
                let normal = StandardNormal;

                data.iter_mut().for_each(|obs_vec| {
                    obs_vec.iter_mut().for_each(|val| {
                        *val += rng.sample(normal) * std_dev.clone();
                    })
                });
            }
            ObsVecNoise::AdditiveMultiNormal(mvnpdf, ..) => {
                for i in 0..N {
                    data.iter_mut()
                        .zip(
                            mvnpdf
                                .sample(rng, &prodef::SamplingMode::UntilValid { max_attempts: 32 })
                                .expect("failed to draw multi normal noise sample")
                                .row_iter(),
                        )
                        .for_each(|(res, val)| {
                            res.set(i, res.get(i).unwrap().clone() + val[(0, 0)].clone())
                        });
                }
            }
            ObsVecNoise::MultiplicativeNormal(std_dev, ..) => {
                let normal = StandardNormal;

                data.iter_mut().for_each(|obs_vec| {
                    obs_vec.iter_mut().for_each(|val| {
                        *val += val.clone() * rng.sample(normal) * std_dev.clone();
                    })
                });
            }
        }
    }

    fn get_random_seed(&self) -> u64 {
        match self {
            ObsVecNoise::AdditiveNormal(.., seed) => *seed,
            ObsVecNoise::AdditiveMultiNormal(.., seed) => *seed,
            ObsVecNoise::MultiplicativeNormal(.., seed) => *seed,
        }
    }

    fn increment_random_seed(&mut self) {
        match self {
            ObsVecNoise::AdditiveNormal(.., seed) => {
                *seed += 1;
            }
            ObsVecNoise::AdditiveMultiNormal(.., seed) => {
                *seed += 1;
            }
            ObsVecNoise::MultiplicativeNormal(.., seed) => {
                *seed += 1;
            }
        }
    }
}
