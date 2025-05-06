//! # The statistics sub-module for the **ocnus** framework.
//!
//!
//! #### Covariance Matrix
//!
//! An arbitrarily sized covariance matrix is represented by the [`CovMatrix`] type, which can either be constructed directly from a positive semi-definite matrix or from an ensemble of vectors.
//! Importantly, this type can handle dimensions with vanishing covariance and these dimensions are excluded from calculations of the covariance matrix inverse, the determinant (i.e pseudo-determinant) or the Cholesky decomposition.
//!
//! The [`CovMatrix`] type can also be used to calculate the multivariate likelihood via [`CovMatrix::multivariate_likelihood`] for slices with a conforming length.
//!
//! #### Probability Density Functions
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

//! ```
//!
//! This module provides the [`CovMatrix`] type and defines the [`Density`] trait and related implementations.
//!
//! This module provides the [`Density`] trait, and related implementations, to describes probability density functions and associated functionality.
//! It is important to note that the sampling function [`Density::draw_sample`] can fail if the number of sampling attempts crosses a hard-coded limit.
//! This can be expected to occasionally occur if the sampling region is located in a far away "tail" of the underlying density distribution.
//! A future goal is an implementation of the exponential tilting technique, to increase the efficiency of drawing random samples for bad edge cases.
//!

mod covmat;

// pub use covmat::{CovMatrix, covariance, covariance_with_weights};
// # use ocnus_stats::{UnivariateND, Uniform1D, Constant1D, Reciprocal1D};
// # let prior = UnivariateND::new([
//  #    Uniform1D::new((-1.5, 1.5)).unwrap(),
//  #    Constant1D::new(1.0),
//  #    Reciprocal1D::new((0.1, 0.35)).unwrap(),
//  #]);
