//! # Model observable types, measurement traits and noise models.
//!
//! This module provides the [`OcnusObser`] trait, that is shared for all model observable types.
//!
//! Any model observable type must provide an implementation of [`OcnusObser::is_valid`], to check for a null observation.
//! This design choice was used instead of storing `Option<O>` types to represent a lack or invalid observation.
//!
//! Currently types that implement [`OcnusObser`]
//! - [`f32`] / [`f64`] Floating point types.
//! - [`ObserVec`] A N-dimensional vector quantity.
//!
//! #### Noise Models
//!
//! The [`OcnusNoise`] trait is shared for all noise models, which are types that can generate random "noise" observations.
//! This requires the underlying observation type `OT` to additionally implement the [`AddAssign`](`std::ops::AddAssign`).
//! By default, the only available noise model is [`NullNoise`], which represents a noise model that does nothing.

mod meas;
mod noise;
mod vector;

pub use meas::*;
pub use noise::*;
pub use vector::*;

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
