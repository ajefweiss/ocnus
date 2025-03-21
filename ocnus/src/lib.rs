#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

use std::ops::AddAssign;

use num_traits::AsPrimitive;

pub mod fevm;
pub mod geom;
pub mod math;
pub mod obser;
pub mod stats;

/// Internal trait to generalize over f32/f64
pub trait OFloat:
    'static
    + for<'x> AddAssign<&'x Self>
    + AsPrimitive<usize>
    + Clone
    + std::fmt::Debug
    + Default
    + for<'x> serde::Deserialize<'x>
    + std::fmt::Display
    + num_traits::Float
    + std::ops::Mul<Self>
    + nalgebra::RealField
    + serde::Serialize
    + std::iter::Sum<Self>
    + for<'x> std::iter::Sum<&'x Self>
{
}

impl OFloat for f32 {}
impl OFloat for f64 {}

/// Internal trait for states.
pub trait OState:
    Clone + std::fmt::Debug + Default + for<'x> serde::Deserialize<'x> + Send + serde::Serialize
{
}
