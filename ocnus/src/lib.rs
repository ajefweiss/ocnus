#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

pub mod fevm;
pub mod geom;
pub mod math;
pub mod obser;
pub mod stats;

/// A shorthand for converting constants to type `T`.
macro_rules! t_from {
    ($value: expr) => {
        T::from_f64($value).unwrap()
    };
}

pub(crate) use t_from;
