#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

pub mod fevm;
pub mod math;
mod model;
pub mod models;
pub mod obser;
mod scobs;
pub mod stats;

pub use model::*;
pub use scobs::*;
