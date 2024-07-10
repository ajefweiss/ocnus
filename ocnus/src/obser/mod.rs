//! Observable types and spacecraft observations.
//!
//! # Observables
//!
//! The core component of this module is the [`OcnusObser`] trait, which must be implemented for any
//! type acting as a model observable.
//!
//! Implemented observables types:
//! - [`ObserVec`] A generic N-dimensional vector quantity.
//!
//! # Spacecraft
//!
//! Individual spacecraft observations are represented by [`ScObs`] with an observable type `O`.
//! A series of single observations, with the same observable type, can be gathered into an
//! [`ScObsSeries`] type.
//!
//! [`ScObsSeries`] has, among others, three important implementations:
//! - [`Add / +`](`std::ops::Add`) Allows the composition of two, or more, [`ScObsSeries`] objects.
//! - [`sort_by_timestamp`](`ScObsSeries::sort_by_timestamp`) Sorts the underlying vector of
//!   [`ScObs`] objects by their timestamps.
//! - [`split`](`ScObsSeries::split`) The reciprocal of one or multiple [`Add`][`std::ops::Add`]
//!   calls. Calling this function consumes a composite [`ScObsSeries`] and returns the original
//!   individual [`ScObsSeries`] objects in a vector.
//!
//! An individual [`ScObs`] also stores a spacecraft configuration [`ScObsConf`]. The following variants are currently implemented:
//! - [`Distance`](`ScObsConf::Distance`) (x) - position of the spacecraft in a heliocentric coordinate
//!  system.
//! - [`Position`](`ScObsConf::Position`) (x, y, z) - position of the spacecraft in a heliocentric coordinate
//!   system.

mod scobs;
mod vector;

use num_traits::Zero;
pub use scobs::{ScObs, ScObsConf, ScObsSeries};
pub use vector::{ObserVec, ObserVecNoise};

use nalgebra::{DVector, Scalar};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::fXX;

/// A trait that must be implemented for any model observable noise type.
pub trait OcnusNoise<T, O>: Clone
where
    O: OcnusObser + Scalar,
{
    /// Generate a random noise time-series.
    fn generate_noise(&self, series: &ScObsSeries<T, O>, rng: &mut impl Rng) -> DVector<O>;

    /// Get randon number seed.
    fn get_random_seed(&self) -> u64;

    /// Increment randon number seed.
    fn increment_random_seed(&mut self);

    /// Initialize a new random number generator using the base seed.
    fn initialize_rng(&self, multiplier: u64, offset: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(self.get_random_seed() * multiplier + offset)
    }
}

/// A noise type that does nothing.
#[derive(Clone, Deserialize, Serialize)]
pub struct NoNoise<T> {
    _data: PhantomData<T>,
}

impl<T, O> OcnusNoise<T, O> for NoNoise<T>
where
    T: fXX,
    O: OcnusObser + Scalar + Zero,
{
    fn generate_noise(&self, series: &ScObsSeries<T, O>, _rng: &mut impl Rng) -> DVector<O> {
        DVector::zeros(series.len())
    }

    fn get_random_seed(&self) -> u64 {
        0
    }

    fn increment_random_seed(&mut self) {}
}

/// A trait that must be implemented for any type that acts as a model observable.
pub trait OcnusObser: Clone + Default + Send + Sync {
    /// Returns `true` if the observation is considered valid.
    fn is_valid(&self) -> bool;
}

impl OcnusObser for f32 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl OcnusObser for f64 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}
