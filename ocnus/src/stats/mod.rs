//! Probability density functions (PDFs), covariance matrices, and statistics.

mod covmat;
mod mvpdf;
mod ptpdf;
mod uvpdf;

use std::fmt::Debug;

pub use covmat::*;
pub use mvpdf::*;
pub use ptpdf::*;
pub use uvpdf::*;

use itertools::zip_eq;
use nalgebra::{DMatrix, SVector, SVectorView};
use rand::Rng;
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
