//! Probability density functions (PDFs).
//!
//! There are three generic probability density functions that are used extensively throughout the crate:
//! - [`ParticlesND`] A density approximated by an ensemble of *particles* with individual weights.
//! - [`MultivariateND`] A multivariate normal density described by a [`CovMatrix`](`crate::math::CovMatrix`).
//! - [`UnivariateND`] A density composed of N individual [`Univariate1D`] densities.
//!
//! There are five 1-dimensional [`Univariate1D`] type that can be used to describe a [`UnivariateND`]:
//! - [`Constant1D`] A constant density that is used for fixed values.
//! - [`Cosine1D`] A density with a cosine shape distribution defined within the range `[-π/2, π/2]`.
//! - [`Normal1D`] A normal distribution with a given mean and standard deviation.
//! - [`Reciprocal1D`] A reciprocal density defined for positive definite numbers.
//! - [`Uniform1D`] A uniformly distributed density.
//!
//! It is recommended to use [`UnivariateND`] as prior for any model as one can easily independantly fine tune parameters, e.g.:
//! ```
//! # use ocnus::prodef::{UnivariateND, Uniform1D, Constant1D, Reciprocal1D};
//! let prior = UnivariateND::new([
//!     Uniform1D::new((-1.5, 1.5)).unwrap(),
//!     Constant1D::new(1.0),
//!     Reciprocal1D::new((0.1, 0.35)).unwrap(),
//! ]);
//! ```

mod multivariate;
mod particles;
mod univariate;

pub use multivariate::MultivariateND;
pub use particles::ParticlesND;
pub use univariate::{
    Constant1D, Cosine1D, Normal1D, Reciprocal1D, Uniform1D, Univariate1D, UnivariateND,
};

use itertools::zip_eq;
use nalgebra::{SVector, SVectorView};
use rand::Rng;
use std::fmt::Debug;
use thiserror::Error;

/// Errors associated with the [`stats`](crate::prodef) module.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ProDeFError<T> {
    #[error("invalid range {name} [{minv} - {maxv}]")]
    InvalidRange {
        name: &'static str,
        maxv: T,
        minv: T,
    },
}

/// A trait that must be implemented for any type that acts as a probability density function.
pub trait OcnusProDeF<T, const P: usize>: Debug
where
    T: PartialOrd,
    Self: Send + Sync,
{
    /// Estimates the relative (not-normalized) density  at a specific position `x`.
    fn density_rel(&self, x: &SVectorView<T, P>) -> T;

    /// Draw a single parameter vector from the underlying density.
    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, ProDeFError<T>>;

    /// Validate a single sample by checking for values that are out of the valid parameter range.
    fn validate_sample(&self, sample: &SVectorView<T, P>) -> bool {
        zip_eq(sample.iter(), self.get_valid_range().iter()).fold(true, |acc, (c, range)| {
            acc & ((&range.0 <= c) & (c <= &range.1))
        })
    }

    /// Returns the valid range for parameter vector samples.
    fn get_valid_range(&self) -> [(T, T); P];
}
