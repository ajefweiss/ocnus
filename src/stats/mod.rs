//! # The statistics sub-module for the **ocnus** framework.

mod covariance;
mod normal;
mod simple;

pub use covariance::*;
pub use normal::*;
pub use simple::*;

use nalgebra::{DefaultAllocator, DimName, OVector, RealField, VectorView, allocator::Allocator};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A trait that is shared by all joint probability density functions.
pub trait Density<T, D>
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    Self: Send + Sync,
{
    /// Draw a random sample vector from the underlying density.
    ///
    /// This function may fail to return a sample if the sampling attempts reach a hard-coded threshold.
    fn draw_sample(&self, rng: &mut impl Rng) -> Option<OVector<T, D>>;

    /// Returns the constant values for each dimension,
    /// returns NaN for each dimensions that is not fixed.
    fn get_constants(&self) -> OVector<T, D>;

    /// Returns the minimum and maximum valid values for each dimension.
    fn get_range(&self) -> OVector<DensityRange<T>, D>;

    /// Calculates or estimates a relative density value at a specific position `x`.
    ///
    /// If the position `x` is outside the valid range, this function returns NaN.
    fn relative_density(&self, x: &VectorView<T, D>) -> T;

    /// Validate a random sample vector by checking the sample w.r.t. to the valid range.
    fn validate_sample(&self, sample: &VectorView<T, D>) -> bool {
        sample
            .iter()
            .zip(self.get_range().iter())
            .zip(self.get_constants())
            .fold(true, |acc, ((value, range), opt_constant)| {
                if let Some(constant) = opt_constant {
                    acc & (*value == constant)
                } else {
                    acc & ((&range.min() <= value) & (value <= &range.max()))
                }
            })
    }

    /// Same as [`validate_sample`], except that a boolean array is returned where each violating dimension is flagged.
    fn validate_sample_values(&self, sample: &VectorView<T, D>) -> OVector<bool, D> {
        sample
            .iter()
            .zip(self.get_range().iter())
            .zip(self.get_constants())
            .map(|((value, range), opt_constant)| {
                if let Some(constant) = opt_constant {
                    *value == constant
                } else {
                    (&range.min() <= value) & (value <= &range.max())
                }
            })
            .collect::<Vec<bool>>()
            .try_into()
            .unwrap()
    }
}

/// A density range, wrapped around `(T, T)`.
#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DensityRange<T>((T, T));

impl<T> DensityRange<T>
where
    T: Copy + PartialOrd,
{
    /// The maximum value of the range.
    pub fn max(&self) -> T {
        self.0.1
    }

    /// The minimum value of the range.
    pub fn min(&self) -> T {
        self.0.0
    }

    /// Create a new [`DensityRange`].
    pub fn new(minamax: (T, T)) -> Self {
        assert!(
            minamax.0 <= minamax.1,
            "minimum value must be smaller or equal than the maximum value"
        );

        Self(minamax)
    }
}
