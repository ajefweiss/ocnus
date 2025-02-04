use crate::{
    stats::{ProbabilityDensityFunction, ProbabilityDensityFunctionSampling},
    Fp, PMatrix, PMatrixView, ScConf,
};
use log::debug;
use nalgebra::{Const, U1};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("negative time_step")]
    NegativeTimeStep,
    #[error("state initialization failed")]
    StateInitializationFailed,
}

/// A data structure for holding an ensemble of model parameters and states.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EnsblData<S, const P: usize>(pub PMatrix<Const<P>>, pub Vec<S>);

impl<S, const P: usize> EnsblData<S, P>
where
    S: OcnusState,
{
    /// Returns `true`` if the ensemble contains no elements.
    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    /// Returns the number of elements in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Create a new [`EnsembleData`] object with zero entries.
    pub fn new(size: usize) -> Self {
        Self(PMatrix::<Const<P>>::zeros(size), vec![S::default(); size])
    }
}

/// A trait that must be implemented for any type that acts as an ensemble model.
pub trait EnsblModel<T, S, const P: usize>: OcnusModel<T, S, P>
where
    T: ProbabilityDensityFunctionSampling<P>,
    S: OcnusState,
{
    /// The rayon chunk size that is used for any parallel iterators.
    /// Specific cheap or expensive operations may use fractions or multiples of this value.
    const RCS: usize;

    /// Initialize the ensemble model with model parameters drawn from the underlying prior.
    fn initialize_ensemble(
        &self,
        scobs: &[ScConf],
        ensbl: &mut EnsblData<S, P>,
        seed: u64,
    ) -> Result<(), ModelError> {
        debug!("initializing an fevm (size={})", ensbl.len());

        ensbl
            .0
            .par_column_iter_mut()
            .zip(ensbl.1.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|(pvector, state)| {
                    let x = 0;

                    Ok(())
                })?;

                // chunks_params
                //     .iter_mut()
                //     .zip(chunks_state.iter_mut())
                //     .try_for_each(|(data, state)| {
                //         self.initialize(scobs, data, state, &mut rng)?;

                //         Ok(())
                //     })?;

                Ok(())
            })?;

        Ok(())
    }
}

/// A trait that must be implemented for any type that acts as a model with `P` parameters.
pub trait OcnusModel<T, S, const P: usize>
where
    T: ProbabilityDensityFunctionSampling<P>,
    S: OcnusState,
{
    /// Array of parameter names.
    const PARAMS: [&'static str; P];

    /// Compute adaptive step sizes for finite differences based on the valid range of the model prior.
    ///
    /// The step size for constant parameters is zero.
    fn adaptive_step_sizes(&self) -> [f32; P] {
        self.valid_range()
            .iter()
            .map(|(min, max)| 1024.0 * (max - min) * f32::EPSILON)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap()
    }

    /// Evolve the model forward in time.
    fn evolve(
        &self,
        time_step: Fp,
        pvector: &PMatrixView<Const<P>, U1, U1, Const<P>>,
        state: &mut S,
    ) -> Result<(), ModelError>;

    /// Initialize the model state from a given set of model parameters.
    fn initialize_state(
        &self,
        scconf: &[ScConf],
        pvector: &PMatrixView<Const<P>, U1, U1, Const<P>>,
        state: &mut S,
    ) -> Result<(), ModelError>;

    /// Get parameter index by name.
    fn get_param_index(name: &str) -> Option<usize> {
        Self::PARAMS.into_iter().position(|param| param == name)
    }

    /// Returns the underlying model prior.
    fn prior(&self) -> &ProbabilityDensityFunction<T, P>;

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(Fp, Fp); P];
}

/// A trait that must be implemented for any type that acts as a model observable.
pub trait OcnusObser: Clone + Default + Send + serde::Serialize + Sync {}

// Implement the [`OcnusObser`] traits for f32/f64.
impl OcnusObser for f32 {}
impl OcnusObser for f64 {}

/// A trait that must be implemented for any type that acts as a model state.
pub trait OcnusState: Clone + Default + Send + serde::Serialize + Sync {}

// Implement the [`OcnusState`] traits for f32/f64.
impl OcnusState for f32 {}
impl OcnusState for f64 {}
