#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

pub use ocnus_stats as stats;

pub mod coords;
mod ensbl;
mod math;
pub mod methods;
mod model;
pub mod models;
pub mod obser;

pub use ensbl::OcnusEnsbl;
pub use model::OcnusModel;

use coords::CoordsError;
use methods::ParticleFilterError;
use model::ModelError;
use stats::StatsError;
use thiserror::Error;

/// Generic container for all error types in the **ocnus** framework.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum OcnusError<T> {
    #[error("coords error")]
    Coords(#[from] CoordsError),
    #[error("particle filter error")]
    FilterError(#[from] ParticleFilterError<T>),
    #[error("model error")]
    Model(#[from] ModelError<T>),
    #[error("stats error")]
    Stats(#[from] StatsError<T>),
}
