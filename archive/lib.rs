#![doc = include_str!("../../README.md")]

mod cmat;
pub mod fevms;
mod math;
mod model;
pub mod obser;
mod scobs;

pub use cmat::CovMatrix;
pub use model::*;
pub use scobs::*;
