//! Observable types and spacecraft observations.
//!
//! # Observables
//!
//! The core item of this module is the `OcnusObser` trait, which must be implemented
//! for any type acting as a model observable.
//! Currently implemented observables are:
//! - [`ObserVec`]: A generic N-dimensional vector quantity.
//!
//! # Spacecraft
//!
//! Spacecraft observations can be represented by either [`ScObs`] (a single observation)
//! or a [`ScObsSeries`] (a time-series).
//!
//!
//!
//! [`ScObsSeries`] has, among others, three important implementations:
//! - [`Add / +`](`std::ops::Add`) : Allows composition of two [`ScObsSeries`] objects.
//! - [`sort_by_timestamp`](`ScObsSeries::sort_by_timestamp`) : Sorts the underlying vector of [`ScObs`]
//!   objets by their timestamps.
//! - [`split`](`ScObsSeries::split`) : The reciprocal of one or multiple [`std::ops::Add`] calls.
//!   Calling this function consumes a composite [`ScObsSeries`] and returns the original
//!   [`ScObsSeries`] objects in a vector.
//!
//! An individual [`ScObs`] also stores a spacecraft configuration.
//! The following variants are currently implemented:
//! - [`Distance`](`ScObsConf::Distance`) : (x) - position of the spacecraft in a heliocentric coordinate
//!  system.
//! - [`Position`](`ScObsConf::Position`) : (x, y, z) - position of the spacecraft in a heliocentric coordinate
//!   system.

mod scobs;
mod vector;

pub use scobs::*;
pub use vector::ObserVec;

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
