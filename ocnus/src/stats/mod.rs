//! Covariance matrices and probability density functions.
//!
//! # Covariance Matrices
//!
//! A [`CovMatrix`] can be created directly from a semi positive definite matrix or
//! an ensemble of N-dimensional particles.
//!
//! This type can handle dimensions without any variance. These dimensions
//! are zero'd out in the resulting matrix, inverse matrix, Cholesky decomposition and are
//! ignored for the calculation of the pseudo determinant. They are also ignored in the
//! likelihood calculations.
//!
//! Also stores the inverse covariance matrix ([`CovMatrix::ref_inverse_matrix`]), the lower
//! triangular matrix from the Cholesky decomposition ([`CovMatrix::ref_cholesky_ltm`]) and the
//! pseudo determinant ([`CovMatrix::pseudo_determinant`]).
//!
//! # Probability Density Functions (PDFs)
//!
//! PDFs are used as model priors and posteriors. All of them implement basic sampling
//! functionality (these sampling functions may panic if they are deemed to inefficient) and relative
//! density calculations.
//!
//! There are currently three implemented generic probability density functions
//! that are used extensively throughout the crate:
//! - [`PDFParticles`]: A PDF composed of an ensemble of particles with individual
//!   weights.
//! - [`PDFMultivariate`]: A multivariate normal PDF described by a [`CovMatrix`].
//! - [`PDFUnivariates`]: A PDF composed of a set of [`PDFUnivariate`] PDFs.
//!
//! There are five 1-dimensional [`PDFUnivariate`] PDFs that can be used to describe a [`PDFUnivariates`] PDF:
//! - [`PDFConstant`]: A constant PDF that is used for fixed values.
//! - [`PDFCosine`]: A PDF with a cosine shape distribution defined within the range `[-π/2, π/2]`.
//! - [`PDFNormal`]: A normal distribution with a given mean and standard deviation.
//! - [`PDFReciprocal`]: A reciprocal PDF defined for positive definite numbers.
//! - [`PDFUniform`]: A uniformly distributed PDF.
//!
//! It is recommended to use a [`PDFUnivariates`] as prior for any model as one can easily
//! independantly fine tune parameters:
//! ```
//! # use ocnus::stats::{PDFUnivariates, PDFUniform, PDFConstant, PDFReciprocal};
//!
//! let prior = PDFUnivariates::new([
//!     PDFUniform::new_uvpdf((-1.5, 1.5)).unwrap(),
//!     PDFConstant::new_uvpdf(1.0),
//!     PDFReciprocal::new_uvpdf((0.1, 0.35)).unwrap(),
//! ]);
//! ```
//!

mod covmat;
mod mvpdf;
mod ptpdf;
mod uvpdf;

pub use covmat::*;
pub use mvpdf::*;
pub use ptpdf::*;
pub use uvpdf::*;

use itertools::zip_eq;
use nalgebra::{DMatrix, SVector, SVectorView};
use rand::Rng;
use std::fmt::Debug;
use thiserror::Error;

/// Errors associated with the [`stats`](crate::stats) module.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum StatsError<T> {
    #[error("matrix does not obey required invariants: {msg}")]
    InvalidMatrix {
        msg: &'static str,
        matrix: DMatrix<T>,
    },
    #[error("invalid range {name} [{minv} - {maxv}]")]
    InvalidRange {
        name: &'static str,
        maxv: T,
        minv: T,
    },
}

/// A trait that must be implemented for any type that acts as a probability density function.
pub trait PDF<T, const P: usize>: Debug
where
    T: PartialOrd,
    Self: Sync,
{
    /// Estimates the relative density  at a specific position `x`.
    ///
    /// This result is not necessarily normalized,
    /// for an exact calculation or estimate of the density see the [`PDFExactDensity`] trait.
    fn relative_density(&self, x: &SVectorView<T, P>) -> T;

    /// Draw a single parameter vector from the underlying density.
    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>>;

    /// Validate a single sample by checking for values that are out of the valid parameter range.
    fn validate_sample(&self, sample: &SVectorView<T, P>) -> bool {
        zip_eq(sample.iter(), self.valid_range().iter()).fold(true, |acc, (c, range)| {
            acc & ((&range.0 <= c) & (c <= &range.1))
        })
    }

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(T, T); P];
}

/// A trait that provides a function that calculates or estimates the exact density.
pub trait PDFExactDensity<T, const P: usize>: PDF<T, P>
where
    T: PartialOrd,
{
    /// Calculates or estimates the exact density  at a specific position `x`.
    fn exact_density(&self, x: &SVectorView<T, P>) -> T;
}
