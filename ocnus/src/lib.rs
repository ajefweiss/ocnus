#![doc = include_str!("../../README.md")]

// pub mod fevms;
mod covmat;
pub mod fevms;
mod math;
mod model;
pub mod obser;
mod scobs;
pub mod stats;

pub use fevms::*;
pub use model::*;
pub use scobs::*;
