#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

pub mod fevm;
pub mod geometry;
pub mod math;
pub mod obser;
mod scobs;
pub mod stats;

pub use scobs::*;
