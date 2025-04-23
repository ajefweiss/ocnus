//! # Statistics sub-module for the **ocnus** framework.
//!
//! This module introduces the [`Density`] trait, which is the trait that is shared by all joint or univariate probability density functions.
//! The three main implemented probability density functions are:
//! - [`MultivariateDensity`] A joint probability density function defined by a set of univariate proability density functions.
//! - [`MultivariateNormalDensity`] A joint probability density function defined by a [`CovMatrix`](`covmatrix::CovMatrix`).
//! - [`ParticleDensity`] A joint probability density function defined by an ensemble of particles.
//!
//! All univariate density functions are summarized within the [`UnivariateDensity`] ADT, and these types should not be used by their own.

mod normal;
mod particles;
mod simple;

pub use normal::*;
pub use particles::*;
pub use simple::*;

use nalgebra::{RealField, SVector, SVectorView};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// A trait that is shared by all probability density functions.
pub trait Density<T, const D: usize>: Sync
where
    T: Copy + RealField,
{
    /// Draw a random sample (vector) from the underlying density.
    ///
    /// This function is limited to `A` sampling attempts, and returns None if the sampling process fails.
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, D>>
    where
        StandardNormal: Distribution<T>;

    /// Returns the constant values for each dimension,
    /// returns NaN for each dimensions that is not fixed.
    fn get_constants(&self) -> SVector<T, D>;

    /// Returns the minimum and maximum valid values for each dimension.
    fn get_range(&self) -> SVector<DensityRange<T>, D>;

    /// Calculates or estimates a relative density value at a specific position `x`.
    ///
    /// If the position `x` is outside the valid range, this function returns NaN.
    fn relative_density(&self, x: &SVectorView<T, D>) -> T;

    /// Validate a random sample vector by checking the sample w.r.t. to the valid range.
    fn validate_sample(&self, sample: &SVectorView<T, D>) -> bool {
        sample
            .iter()
            .zip(self.get_range().iter())
            .zip(self.get_constants().iter())
            .fold(true, |acc, ((value, range), constant)| {
                if constant.is_finite() {
                    acc & (value == constant)
                } else {
                    acc & ((&range.min() <= value) & (value <= &range.max()))
                }
            })
    }

    /// Same as [`validate_sample`](`crate::stats::Density::validate_sample`), except that a boolean array is returned where each violating dimension is flagged.
    fn validate_sample_flags(&self, sample: &SVectorView<T, D>) -> SVector<bool, D> {
        SVector::from_iterator(
            sample
                .iter()
                .zip(self.get_range().iter())
                .zip(self.get_constants().iter())
                .map(|((value, range), constant)| {
                    if constant.is_finite() {
                        value == constant
                    } else {
                        (&range.min() <= value) & (value <= &range.max())
                    }
                }),
        )
    }
}

/// Defines the valid parameter range for a probability density function.
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
