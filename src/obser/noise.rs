use crate::{base::ScObsSeries, obser::OcnusObser};
use nalgebra::{DVector, Scalar};
use num_traits::Zero;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, ops::AddAssign};

/// A trait that is shared by all observation noise models.
pub trait OcnusNoise<T, O>
where
    O: AddAssign + OcnusObser,
{
    /// Generate a random noise time-series.
    fn generate_noise(&self, series: &ScObsSeries<T>, rng: &mut impl Rng) -> DVector<O>;

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

impl<T, O> OcnusNoise<T, O> for NullNoise<T>
where
    T: PartialOrd,
    O: AddAssign + OcnusObser + Scalar + Zero,
{
    fn generate_noise(&self, series: &ScObsSeries<T>, _rng: &mut impl Rng) -> DVector<O> {
        DVector::zeros(series.len())
    }

    fn get_random_seed(&self) -> u64 {
        0
    }

    fn increment_random_seed(&mut self) {}
}
