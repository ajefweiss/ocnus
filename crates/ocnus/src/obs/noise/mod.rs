//! Observation noise types.

mod null;
mod vector;

pub use null::*;
pub use vector::*;

use crate::obs::data::ObsData;
use nalgebra::{DVectorViewMut, Scalar};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// A trait that is shared by all observation noise models.
pub trait ObsNoise<T, OC, OD>
where
    T: Scalar,
    OD: ObsData,
{
    /// Generate a random noise time-series.
    fn generate_noise(&self, data: &mut DVectorViewMut<OD>, rng: &mut impl RngExt);

    /// Get random number seed.
    fn get_random_seed(&self) -> u64;

    /// Increment random number seed.
    fn increment_random_seed(&mut self);

    /// Initialize a new random number generator using the base seed.
    fn initialize_rng(&self, multiplier: u64, offset: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(self.get_random_seed() * multiplier + offset)
    }
}
