#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

pub mod fevm;
pub mod geom;
pub mod math;
pub mod obser;
pub mod stats;

// /// Internal trait for states.
// pub trait OState:
//     Clone + std::fmt::Debug + Default + for<'x> serde::Deserialize<'x> + Send + serde::Serialize
// {
// }
