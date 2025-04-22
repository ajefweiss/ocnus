#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

pub mod coords;
pub mod forward;
pub mod math;
pub mod obser;
pub mod prodef;

use coords::CoordsError;
use forward::{FSMError, filters::ParticleFilterError};
use math::MathError;
use nalgebra::{RealField, Scalar};
use num_traits::{AsPrimitive, Float, FromPrimitive, float::TotalOrder};
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
    #[error("forward error")]
    Coords(#[from] CoordsError),
    #[error("forward error")]
    FSM(#[from] FSMError<T>),
    #[error("math error")]
    Math(#[from] MathError<T>),
    #[error("particle filter error")]
    ParticleFilter(#[from] ParticleFilterError<T>),
    #[error("stats error")]
    ProDeF(#[from] ProDeFError<T>),
}

/// A trait that describes a generic floating point numbers within the **ocnus** crate. In practical
/// terms this trait is only used for the f32/f64 types.
#[allow(non_camel_case_types)]
pub trait fXX:
    'static
    + AsPrimitive<usize>
    + Copy
    + Debug
    + Default
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

    /// Returns π/2.
    fn half_pi() -> Self {
        RealField::frac_pi_2()
    }
}

impl fXX for f32 {}
impl fXX for f64 {}
