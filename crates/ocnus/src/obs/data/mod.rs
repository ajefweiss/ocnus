//! Observation data types.

mod basis;
mod image;
mod vector;

pub use basis::*;
pub use image::*;
pub use vector::*;

use nalgebra::Scalar;
use num_traits::Zero;

/// A trait that is shared by all model observation data types.
pub trait ObsData: Clone + Default + Scalar + Send + Sync + Zero {
    /// Returns `true` if the observation is considered valid.
    fn is_valid(&self) -> bool;
}

impl ObsData for f32 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl ObsData for f64 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}
