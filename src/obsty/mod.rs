//! # Model observable types, measurement traits and noise models.
//!
//! This module provides the [`Observable`] trait, that is shared for all model observable types.
//!
//! Any model observable type `OT` must provide an implementation of [`Observable::is_valid`], to check for a null observation.
//! This design choice was used instead of storing `Option<OT>` to represent missing or invalid observation.
//!
//! Types that currently implement [`Observable`]:
//! - [`f32`] / [`f64`] Floating point types.
//! - [`ObserVec`] A N-dimensional vector quantity.
//! - [`ICSCoordsBasis`] A triplet of coordinates and three respective basis vectors, internally a 12-dimensional vector quantity.
//!
//! #### Noise Models
//!
//! The [`NoiseModel`] trait is shared for all noise models, which are types that can generate random "noise" that somehow
//! influences the observations.
//! By default, the only available noise model is [`NullNoise`], which represents a noise model that does nothing.
//!
//! #### Measurement Traits
//!
//! Traits that define shared observation quantities for multiple models.
//!
//! Currently implemented measurement traits:
//! - [`InSituMagnetometer`] In situ magnetic field measurements.
//! - [`InSituPlasmaBulkVelocity`] In situ plasma bulk velocity measurements.

mod image;
mod meas;
mod noise;
mod vector;

pub use image::*;
pub use meas::*;
pub use noise::*;
pub use vector::*;

use nalgebra::Scalar;
use num_traits::Zero;

/// A trait that is shared by all model observable types.
pub trait Observable: Clone + Default + Scalar + Send + Sync + Zero {
    /// Returns `true` if the observation is considered valid.
    fn is_valid(&self) -> bool;
}

impl Observable for f32 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl Observable for f64 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}
