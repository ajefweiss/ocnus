//! Bayesian inference, probability density functions (PDFs) and statistics.

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
use thiserror::Error;

/// Errors associated with the [`stats`](crate::stats) module.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum StatsError {
    #[error("matrix does not obey required invariants: {msg}")]
    InvalidMatrix {
        msg: &'static str,
        matrix: DMatrix<f64>,
    },
    #[error("invalid range {name} [{minv} - {maxv}]")]
    InvalidRange {
        name: &'static str,
        maxv: f64,
        minv: f64,
    },
}
/// A trait that must be implemented for any type that acts as a probability density function.
pub trait PDF<const P: usize>: std::fmt::Debug
where
    Self: Sync,
{
    /// Estimates the relative density  at a specific position `x`.
    ///
    /// This result is not necessarily normalized,
    /// for an exact calculation or estimate of the density see the [`PDFExactDensity`] trait.
    fn relative_density(&self, x: &SVectorView<f64, P>) -> f64;

    /// Draw a single parameter vector from the underlying density.
    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f64, P>, StatsError>;

    /// Validate a single sample by checking for values that are out of the valid parameter range.
    fn validate_sample(&self, sample: &SVectorView<f64, P>) -> bool {
        zip_eq(sample.iter(), self.valid_range().iter()).fold(true, |acc, (c, range)| {
            acc & ((&range.0 <= c) & (c <= &range.1))
        })
    }

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(f64, f64); P];
}

/// A trait that provides a function that calculates or estimates the exact density .
pub trait PDFExactDensity<const P: usize>: PDF<P> {
    /// Calculates or estimates the exact density  at a specific position `x`.
    fn exact_density(&self, x: &SVectorView<f64, P>) -> f64;
}
