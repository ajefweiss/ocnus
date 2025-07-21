use crate::{base::ScObs, obsty::Observable};
use nalgebra::{DVector, Scalar};
use num_traits::Zero;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// A trait that is shared by all noise models.
pub trait NoiseModel<T, OT>
where
    T: Copy + Scalar,
    OT: Observable,
{
    /// Generate a random noise time-scobs.
    fn generate_noise(&self, scobs: &ScObs<T, OT>, rng: &mut impl Rng) -> DVector<OT>;

    /// Get randon number seed.
    fn get_random_seed(&self) -> u64;

    /// Increment randon number seed.
    fn increment_random_seed(&mut self);

    /// Initialize a new random number generator using the base seed.
    fn initialize_rng(&self, multiplier: u64, offset: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(self.get_random_seed() * multiplier + offset)
    }
}

/// A noise model that does nothing.
#[derive(Clone, Deserialize, Serialize)]
pub struct NullNoise<T> {
    _data: PhantomData<T>,
}

impl<T, OT> NoiseModel<T, OT> for NullNoise<T>
where
    T: Copy + PartialOrd + Scalar,
    OT: Observable + Scalar + Zero,
{
    fn generate_noise(&self, scobs: &ScObs<T, OT>, _rng: &mut impl Rng) -> DVector<OT> {
        DVector::zeros(scobs.len())
    }

    fn get_random_seed(&self) -> u64 {
        0
    }

    fn increment_random_seed(&mut self) {}
}
