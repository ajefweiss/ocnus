mod obsvc;

use crate::{
    stats::{ProbabilityDensityFunction, ProbabilityDensityFunctionSampling, StatsError},
    EnsblData, EnsblModel, OcnusModel, OcnusState, PMatrix, ScConf,
};
use log::debug;
use nalgebra::Const;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use obsvc::ModelObserVec;

/// Errors associated with the [`crate::fevms`] module.
#[derive(Debug, Error)]
pub enum FEVMSError {
    #[error("stats error")]
    Stats(StatsError),
}

/// The trait that must be implemented for any forward ensemble model with vector observables with prior `T` and state `S`.
pub trait ForwardEnsembleVectorModel<T, S, const P: usize, const N: usize>:
    EnsblModel<T, S, P>
where
    T: ProbabilityDensityFunctionSampling<P>,
    S: OcnusState,
{
}
