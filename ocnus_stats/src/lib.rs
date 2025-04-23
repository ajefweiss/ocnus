//! # Statistics sub-module for the **ocnus** framework.
//!
//! This module primarily provides the [`OcnusPDF`] trait, and related implementations, to describes probability density functions.
//! It is important to note that the sampling function [`OcnusPDF::draw_sample`] can fail if the number of sampling attempts crosses a hard-coded limit.
//! This can be expected to occasionally occur if the sampling region is located in a far away "tail" of the underlying density distribution.
//!
//! The module also provides a number of functions to compute covariances and the [`CovMatrix`] type to work directly with covariance matrices.
//!
//! ## Covariances
//!
//! A covariance matrix is represented by the [`CovMatrix`] type that can either be constructed directly from a semi-positive definite matrix or from an ensemble of particles.
//! An important feature is that this type can handle dimensions with vanishing covariance and these dimensions are excluded from calculations of the matrix inverse, the determinant (i.e pseudo-determinant) or the Cholesky decomposition.
//!
//! The [`CovMatrix`] type can also be used to calculate the multivariate likelihood via [`CovMatrix::multivariate_likelihood`] for slices that have conforming dimensions w.r.t. to the covariance matrix.
//!
//! ## Probability Density Functions
//!
//! The three implemented joint probability density functions:
//! - [`ParticlesND`] A generic density that is approximed by an ensemble of particles.
//! - [`MultivariateND`] Multivariate normal density described by a [`CovMatrix`].
//! - [`UnivariateND`] Independent joint density described by N individual [`Univariate1D`]
//!   densities.
//!
//! Basic, single dimensional, probability density functions can be described using the
//! [`Univariate1D`] type that contains one of the following five basic densities:
//! - [`Constant1D`] A constant distribution that is used for fixed values.
//! - [`Cosine1D`] A cosine shape distribution defined within the range `[-π/2, π/2]`.
//! - [`Normal1D`] A normal distribution with a given mean and standard deviation.
//! - [`Reciprocal1D`] A reciprocal distribution defined for positive definite numbers.
//! - [`Uniform1D`] A uniform distribution.
//!
//! It is generally recommended to use [`UnivariateND`] as prior, constructed from individual [`Univariate1D`] objects, for any model as one can easily independently fine tune parameters, e.g.:
//! ```
//! # use ocnus_stats::{UnivariateND, Uniform1D, Constant1D, Reciprocal1D};
//! let prior = UnivariateND::new([
//!     Uniform1D::new((-1.5, 1.5)).unwrap(),
//!     Constant1D::new(1.0),
//!     Reciprocal1D::new((0.1, 0.35)).unwrap(),
//! ]);
//! ```

mod covariance;
mod multivariate;
mod particles;
mod univariate;

pub use covariance::{CovMatrix, covariance, covariance_with_weights};
pub use multivariate::MultivariateND;
pub use particles::ParticlesND;
pub use univariate::{
    Constant1D, Cosine1D, Normal1D, Reciprocal1D, Uniform1D, Univariate1D, UnivariateND,
};

use itertools::zip_eq;
use nalgebra::{RealField, SVector, SVectorView};
use rand::Rng;
use std::fmt::Debug;
use thiserror::Error;

/// A generic error type for the statistics module.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum StatsError<T> {
    #[error("inefficient sampling for {name}, aborted after {count} attempts")]
    InefficientSampling { name: &'static str, count: usize },
    #[error("invalid pdf range {name} [{minv} - {maxv}]")]
    InvalidRange {
        name: &'static str,
        maxv: T,
        minv: T,
    },
}

/// A trait that is shared by all joint probability density functions.
pub trait OcnusPDF<T, const N: usize>
where
    T: RealField,
    Self: Send + Sync,
{
    /// Calculates or estimates a relative density value at a specific position `x`.
    ///
    /// If the position `x` is outside the valid range, this function returns NaN.
    fn density_rel(&self, x: &SVectorView<T, N>) -> T;

    /// Draw a single parameter vector from the underlying density.
    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, N>, StatsError<T>>;

    /// Validate a single sample by checking for values that are out of the valid parameter range.
    fn validate_sample(&self, sample: &SVectorView<T, N>) -> bool {
        zip_eq(sample.iter(), self.get_valid_range().iter()).fold(true, |acc, (c, range)| {
            acc & ((&range.0 <= c) & (c <= &range.1))
        })
    }

    /// Returns the valid range for parameter vector samples.
    fn get_valid_range(&self) -> [(T, T); N];
}
