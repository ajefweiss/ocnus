//! # The core building blocks of the **ocnus** framework.

mod model;
mod scobs;
mod vector;

pub use model::*;
pub use scobs::*;
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
