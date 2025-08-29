//! # Statistics module for the **ocnus** framework.
//!
//! This module introduces, and contains implementations of, the [`Density`] trait, which is used to describe probability density functions (PDFs).
//!
//! There are currently three implementations of the [`Density`] trait, i.e. probability density functions, which are:
//! - [`MultivariateDensity`] A joint probability density function defined by a set of independent univariate PDFs.
//! - [`MultivariateNormalDensity`] A joint normal distribution defined by a [`CovMatrix`](`crate::math::CovMatrix`).
//! - [`ParticleDensity`] A joint probability density function defined by an ensemble of particles or samples.
//!
//! All univariate density functions are summarized within the [`UnivariateDensity`] ADT, and as such these types are not intended to be used on their own.

mod normal;
mod particles;
mod univariate;

use std::ops::Sub;

pub use normal::*;
pub use particles::*;
use serde::{Deserialize, Serialize};
pub use univariate::*;

use nalgebra::{RealField, SVector, SVectorView, SVectorViewMut};
use rand::Rng;

/// A trait that is shared by all probability density functions.
pub trait Density<T, const D: usize>: Sync
where
    T: Copy + RealField,
{
    /// Clip sample to the minimum or maximum boundary values.
    fn clip_sample(&self, sample: &mut SVectorViewMut<T, D>) {
        sample
            .iter_mut()
            .zip(self.get_range().iter())
            .zip(self.get_constants().iter())
            .for_each(|((value, range), constant)| {
                if !constant.is_finite() {
                    if *value <= range.min {
                        *value = range.min
                    } else if *value >= range.max {
                        *value = range.max
                    }
                }
            })
    }

    /// Draw a random sample from the underlying density.
    ///
    /// This function is limited to `max_attempts` sampling attempts,
    /// and returns None if no valid samples are drawn.
    fn draw_sample(&self, rng: &mut impl Rng, max_attempts: usize) -> Option<SVector<T, D>>;

    /// Returns the constant values for each dimension and
    /// returns NaN for respective dimensions that are not constant.
    fn get_constants(&self) -> SVector<T, D>;

    /// Returns the valid parameter range for each dimension.
    fn get_range(&self) -> SVector<DensityRange<T>, D>;

    /// Calculates or estimates a relative density value at a specific position `x`.
    ///
    /// Returns NaN if the position `x` is outside the valid range, which can be retrieved using [`Density::get_range`].
    fn relative_density(&self, x: &SVectorView<T, D>) -> T;

    /// Validate a random sample by checking whether the parameter values are within the valid range.
    fn validate_sample(&self, sample: &SVectorView<T, D>) -> bool {
        sample
            .iter()
            .zip(self.get_range().iter())
            .zip(self.get_constants().iter())
            .fold(true, |acc, ((value, range), constant)| {
                if constant.is_finite() {
                    acc & (value == constant)
                } else {
                    acc & ((&range.min <= value) & (value <= &range.max))
                }
            })
    }
}

/// Defines the valid parameter range `[min, max]` for a probability density function.
#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DensityRange<T> {
    /// The minimum inclusive value of the range.
    pub min: T,

    /// The maximum inclusive value of the range.
    pub max: T,
}

impl<T> DensityRange<T>
where
    T: Copy + PartialOrd,
{
    /// Returns an un-bounded range.
    pub fn inf() -> Self
    where
        T: RealField,
    {
        Self {
            min: -T::one() / T::zero(),
            max: T::one() / T::zero(),
        }
    }

    /// Returns the length of the range, not be confused with the common function `len`.
    pub fn length(&self) -> T
    where
        T: Sub<T, Output = T>,
    {
        self.max - self.min
    }

    /// Create a new [`DensityRange`].
    pub fn new(min: T, max: T) -> Self {
        assert!(
            min <= max,
            "minimum value must be smaller or equal than the maximum value"
        );

        Self { min, max }
    }
}
