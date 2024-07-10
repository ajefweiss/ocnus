//! Mathematical data types, functions and routines.
//!
//! # Covariance Matrices
//!
//! A [`CovMatrix`] can be created directly from a semi positive definite matrix or an ensemble
//! of N-dimensional vectors.
//!
//! [`CovMatrix`] can handle dimensions without any variance. These dimensions are zero'd out
//! in the resulting matrix, inverse matrix and the Cholesky decomposition. Also therefore, only
//! the pseudo-determinant is calculated and provided. Likelihood calculations also ignore these
//! dimensions.
//!
//! This type directly stores the inverse covariance matrix, the lower triangular matrix from the
//! Cholesky decomposition and the pseudo determinant to avoid unnecessary calculations.

mod bessel;
mod covariance;
mod factorial;
mod quaternions;

pub use bessel::bessel_jn;
pub use covariance::{CovMatrix, covariance, covariance_with_weights};
pub use factorial::factorial;
pub use quaternions::quaternion_rot;

use nalgebra::DMatrix;
use thiserror::Error;

/// Errors associated with the [`math`](crate::math) module.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum MathError<T> {
    #[error("matrix does not obey required invariants: {msg}")]
    InvalidMatrix {
        msg: &'static str,
        matrix: DMatrix<T>,
    },
}

/// A shorthand for converting constants to type `T`.
macro_rules! T {
    ($value: expr) => {
        T::from_f64($value).unwrap()
    };
}

macro_rules! abs {
    ($value: expr) => {
        num_traits::Float::abs($value)
    };
}

macro_rules! acos {
    ($value: expr) => {
        num_traits::Float::acos($value)
    };
}

macro_rules! asin {
    ($value: expr) => {
        num_traits::Float::asin($value)
    };
}

macro_rules! atan2 {
    ($value_y: expr, $value_x: expr) => {
        num_traits::Float::atan2($value_y, $value_x)
    };
}

macro_rules! cos {
    ($value: expr) => {
        num_traits::Float::cos($value)
    };
}

macro_rules! exp {
    ($value: expr) => {
        num_traits::Float::exp($value)
    };
}

macro_rules! ln {
    ($value: expr) => {
        num_traits::Float::ln($value)
    };
}

macro_rules! powf {
    ($value: expr, $float: expr) => {
        num_traits::Float::powf($value, $float)
    };
}

macro_rules! powi {
    ($value: expr, $integer: expr) => {
        num_traits::Float::powi($value, $integer)
    };
}

macro_rules! sin {
    ($value: expr) => {
        num_traits::Float::sin($value)
    };
}

macro_rules! sqrt {
    ($value: expr) => {
        num_traits::Float::sqrt($value)
    };
}

pub(crate) use T;
pub(crate) use abs;
pub(crate) use acos;
pub(crate) use asin;
pub(crate) use atan2;
pub(crate) use cos;
pub(crate) use exp;
pub(crate) use ln;
pub(crate) use powf;
pub(crate) use powi;
pub(crate) use sin;
pub(crate) use sqrt;
