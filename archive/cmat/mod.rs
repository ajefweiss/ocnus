//! Bayesian inference, probability density functions (PDFs) and statistics.

mod mvpdf;
mod ptpdf;
mod uvpdf;

pub use mvpdf::*;
pub use ptpdf::*;
pub use uvpdf::*;

use itertools::zip_eq;
use nalgebra::{SVector, SVectorView};
use rand::Rng;
use thiserror::Error;

/// Errors associated with the [`stats`](crate::stats) module.
#[derive(Debug, Error)]
pub enum OcnusStatsError {
    #[error("invalid range {name} [{minv} - {maxv}]")]
    InvalidRange {
        name: &'static str,
        maxv: f32,
        minv: f32,
    },
}

/// A trait that must be implemented for any type that acts as a probability density function.
pub trait PDF<const P: usize>
where
    Self: Sync,
{
    /// Estimates the relative density (or likelihood) at a specific position `x`.
    /// The result is not necessarily normalized,
    /// for an exact calculation or estimate of the density see the [`PDFExactDensity`] trait.
    fn relative_density(&self, x: &SVectorView<f32, P>) -> f32;

    /// Draw a single parameter vector from the underlying density.
    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, P>, OcnusStatsError>;

    /// Validate a single sample by checking for values that are out of the valid parameter range.
    fn validate_sample(&self, sample: &SVectorView<f32, P>) -> bool {
        zip_eq(sample.iter(), self.valid_range().iter()).fold(true, |acc, (c, range)| {
            acc & ((&range.0 <= c) & (c <= &range.1))
        })
    }

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(f32, f32); P];
}

/// A trait that provides a function that calculates or estimates the exact density (or likelihood).
pub trait PDFExactDensity<const P: usize>: PDF<P> {
    /// Calculates or estimates the exact density (or likelihood) at a specific position `x`.
    fn exact_density(&self, x: &SVectorView<f32, P>) -> f32;
}
