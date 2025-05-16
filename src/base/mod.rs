//! # Core building blocks of the **ocnus** framework.
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
//! - [`Position`](`ScObsConf::Position`) (x, y, z) - position of the spacecraft in a heliocentric coordinate system.

mod ensbl;
mod model;
mod scobs;

pub use ensbl::*;
pub use model::*;
pub use scobs::*;
