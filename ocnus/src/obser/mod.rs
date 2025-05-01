//! # Observation types and spacecraft measurements.
//!
//! This module provides the [`OcnusObser`] trait, that is shared for all model observable types, and the spacecraft observation series type [`ScObsSeries`] that represents a time-series of observations.
//!
//!
//! ## Model Observables
//!
//! Any model observable type must implement [`OcnusObser::is_valid`], to check for a null observation.
//! This design choice was used instead of storing `Option<O>` types to represent a lack or invalid observation.
//!
//! Currently types that implement [`OcnusObser`]:
//! - [`f32`] / [`f64`] Floating point types.
//! - [`ObserVec`] A N-dimensional vector quantity.
//!
//! For the [`ObserVec`] type, currently implemented error metrics are the [`RMSE`](`observec_rmse`) and [`MSE`](`observec_mse`) functions.
//!
//! #### Noise Models
//!
//! The [`OcnusNoise`] trait is shared for all noise models, which are types that can generate random "noise" observations.
//! This requires the underlying observation type `O` to additionally implement the [`AddAssign`](`std::ops::AddAssign`) trait.
//! By default, the only available noise model is [`NullNoise`], which represents a noise model that does nothing.
//!
//! # Spacecraft Observations
//!
//! Individual spacecraft observations are represented by [`ScObs`] type with the generic observable type `O`.
//! A series of single observations, with the same observable type, can be gathered into a [`ScObsSeries`], which is implemented as wrapper around `Vec<ScObs>`.
//!
//! [`ScObsSeries`] has, among others, three important implementations:
//! - [`Add / +`](`std::ops::Add`) Allows the composition of two, or more, [`ScObsSeries`] objects.
//! - [`sort_by_timestamp`](`ScObsSeries::sort_by_timestamp`) Sorts the underlying vector of [`ScObs`] objects by their timestamps. This is useful for generating a continous time-series containing observations with different configurations (i.e. locations).
//! - [`split`](`ScObsSeries::split`) The reciprocal of one or multiple [`Add`][`std::ops::Add`] calls. Calling this function consumes a composite [`ScObsSeries`] and returns the original individual [`ScObsSeries`] objects in a vector.
//!
//! An individual [`ScObs`] also stores a spacecraft configuration [`ScObsConf`], representing different types of measurement configurations.
//! The following variants are currently implemented:
//! - [`Distance`](`ScObsConf::Distance`) (x) - position of the spacecraft in a heliocentric coordinate  system.
//! - [`Position`](`ScObsConf::Position`) (x, y, z) - position of the spacecraft in a heliocentric coordinate system.

mod noise;
mod scobs;
mod vector;

pub use noise::{NullNoise, OcnusNoise};
pub use scobs::{ScObs, ScObsConf, ScObsSeries};
pub use vector::{ObserVec, ObserVecNoise, observec_mse, observec_rmse};

/// A trait that is shared by all model observable types.
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
