//! Math types and routines.

mod bessel;
mod covmatrix;
mod factorial;
mod quantile;
mod vector;

pub use bessel::bessel_jn;
pub use covmatrix::{CovMatrix, covariance, covariance_with_weights};
pub use factorial::factorial;
pub use quantile::{largest_n, quantiles};
pub use vector::normalize;
