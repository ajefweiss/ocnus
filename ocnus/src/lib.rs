#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

// pub mod fevm;
pub mod coords;
pub mod math;
pub mod obser;
pub mod prodef;

use math::MathError;
use nalgebra::{RealField, Scalar};
use num_traits::{Float, FromPrimitive, float::TotalOrder};
use prodef::ProDeFError;
use std::{
    fmt::{Debug, Display},
    iter::Sum,
};
use thiserror::Error;

/// Generic container type for errors.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum OcnusError<T> {
    #[error("math error")]
    Math(#[from] MathError<T>),
    #[error("stats error")]
    ProDeF(#[from] ProDeFError<T>),
}

/// A trait that describes a generic floating point numbers within the **ocnus** crate. In practical
/// terms this trait is only used for the f32/f64 types.
#[allow(non_camel_case_types)]
pub trait fXX:
    'static
    + Copy
    + Debug
    + Display
    + Float
    + FromPrimitive
    + RealField
    + Scalar
    + Send
    + Sum
    + for<'x> Sum<&'x Self>
    + Sync
    + TotalOrder
{
    /// Returns 4π.
    fn four_pi() -> Self {
        Self::two_pi() + Self::two_pi()
    }
}

impl fXX for f32 {}
impl fXX for f64 {}
