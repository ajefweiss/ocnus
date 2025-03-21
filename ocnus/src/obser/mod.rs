//! Spacecraft observations and implementations of observable types.

mod scobs;
mod vector;

pub use scobs::*;
pub use vector::ObserVec;

use serde::Serialize;

use crate::OFloat;

/// A trait that must be implemented for any type that acts as a model observable.
pub trait OcnusObser: Clone + Default + Send + Serialize + Sync {
    /// Returns `true` if the observation is considered valid.
    fn is_valid(&self) -> bool;
}

impl<T> OcnusObser for T
where
    T: OFloat,
{
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}
