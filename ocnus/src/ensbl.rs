//! Implementation of the [`EnsembleForwardModel`] trait and related structures.

use crate::{
    alias::{PMatrix, PMatrixView, PMatrixViewMut},
    scobs::ScConf,
    stats::dnsty::{DensityError, ProbabilityDensityFunctionSampling},
    Fp, OcnusModel, OcnusState,
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, DimAdd, Dyn, SVector, ToTypenum, U1};
use num_traits::AsPrimitive;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

/// Collection of model related errors.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("density error")]
    DensityError(DensityError),
}

/// A data structure for holding an ensemble of model parameters and states.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EnsembleData<S, const P: usize>(pub PMatrix<Const<P>>, pub Vec<S>);

impl<S, const P: usize> EnsembleData<S, P>
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
        EnsembleData(PMatrix::<Const<P>>::zeros(size), vec![S::default(); size])
    }
}

/// A trait that must be implemented for any type that acts as an ensemble model with `P` model parameters.
pub trait EnsembleForwardModel<S, const P: usize>: OcnusModel<S, P>
where
    S: OcnusState,

    Self: Sync,
{
    /// The rayon chunk size that is used for any parallel iterators.
    /// Specific cheap or expensive operations may use fractions or multiples of this value.
    const RCS: usize;

    /// Evolve the ensemble model states forward in time.
    fn forward_evolution(
        &self,
        time_step: f64,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, U1, Const<P>>,
        state: &mut [S],
    ) -> Result<(), ModelError>;

    /// Initialize an [`EnsembleData`] structure.
    fn initialize_ensemble(
        &self,
        scobs: &[ScConf],
        ensbl: &mut EnsembleData<S, P>,
        random_seed: u64,
    ) -> Result<(), ModelError> {
        //     ensbl
        //         .0
        //         .par_column_iter_mut()
        //         .chunks(Self::RCS)
        //         .zip(ensbl.1.par_iter_mut().chunks(Self::RCS))
        //         .enumerate()
        //         .try_for_each(|(cdx, (pmatrix, states))| {
        //             let mut rng = rng_gen!(random_seed, 0, cdx, Xoshiro256PlusPlus);

        //             chunks.iter_mut().try_for_each(|(params, state)| {
        //                 let x = 0;

        //                 Ok(())
        //             })?;

        //             Ok(())
        //         })?;

        Ok(())
    }

    //     /// Ensemble wrapper for [`OcnusModel::initialize`].
    //     fn initialize_ensemble(
    //         &self,
    //         scobs: &[ScConf],
    //         ensbl: &mut EnsblData<S, P>,
    //         seed: u64,
    //     ) -> Result<(), ModelError> {
    //         debug!("initializing an ensemble (size={})", ensbl.len());

    //         ensbl
    //             .0
    //             .par_chunks_mut(Self::RCS)
    //             .zip(ensbl.1.par_chunks_mut(Self::RCS))
    //             .enumerate()
    //             .try_for_each(|(cdx, (chunks_params, chunks_state))| {
    //                 let mut rng = ChaCha8Rng::seed_from_u64(43 + seed + (cdx as u64 * 47));

    //                 chunks_params
    //                     .iter_mut()
    //                     .zip(chunks_state.iter_mut())
    //                     .try_for_each(|(data, state)| {
    //                         self.initialize(scobs, data, state, &mut rng)?;

    //                         Ok(())
    //                     })?;

    //                 Ok(())
    //             })?;

    //         Ok(())
    //     }

    //     /// Ensemble wrapper for [`OcnusModel::initialize_from_density`].
    //     fn initialize_ensemble_from_density(
    //         &self,
    //         scobs: &[ScConf],
    //         ensbl: &mut EnsblData<S, P>,
    //         pdf: impl ProbabilityDensityFunctionSampling<P>,
    //         seed: u64,
    //     ) -> Result<(), ModelError> {
    //         use rand::SeedableRng;

    //         debug!(
    //             "initializing an ensemble from a density (size={})",
    //             ensbl.len()
    //         );

    //         ensbl
    //             .0
    //             .par_chunks_mut(Self::RCS)
    //             .zip(ensbl.1.par_chunks_mut(Self::RCS))
    //             .enumerate()
    //             .try_for_each(|(cdx, (chunks_params, chunks_state))| {
    //                 let mut rng = ChaCha8Rng::seed_from_u64(43 + seed + (cdx as u64 * 57));

    //                 chunks_params
    //                     .iter_mut()
    //                     .zip(chunks_state.iter_mut())
    //                     .try_for_each(|(data, state)| {
    //                         self.initialize_from_density(scobs, data, state, pdf, &mut rng)?;

    //                         Ok(())
    //                     })?;

    //                 Ok(())
    //             })?;

    //         Ok(())
    //     }

    //     /// Ensemble wrapper for [`OcnusModel::initialize_states`].
    //     fn initialize_states_ensemble(
    //         &self,
    //         scobs: &[ScConf],
    //         ensbl: &mut EnsblData<S, P>,
    //     ) -> Result<(), ModelError> {
    //         debug!(
    //             "initializing the states of an ensemble (size={})",
    //             ensbl.len()
    //         );

    //         ensbl
    //             .0
    //             .par_chunks(Self::RCS)
    //             .zip(ensbl.1.par_chunks_mut(Self::RCS))
    //             .into_par_iter()
    //             .try_for_each(|(chunks_params, chunks_state)| {
    //                 chunks_params
    //                     .iter()
    //                     .zip(chunks_state.iter_mut())
    //                     .try_for_each(|(data, state)| {
    //                         self.initialize_states(scobs, data, state)?;

    //                         Ok(())
    //                     })?;

    //                 Ok(())
    //             })?;

    //         Ok(())
    //     }

    //     /// Ensemble wrapper for [`OcnusModel::initialize_params`].
    //     fn initialize_params_ensemble(
    //         &self,
    //         ensbl: &mut EnsblData<S, P>,
    //         seed: u64,
    //     ) -> Result<(), ModelError> {
    //         use rand::SeedableRng;

    //         debug!(
    //             "initializing the parameters an ensemble (size={})",
    //             ensbl.len()
    //         );

    //         ensbl
    //             .0
    //             .par_chunks_mut(Self::RCS)
    //             .enumerate()
    //             .try_for_each(|(cdx, chunks)| {
    //                 let mut rng = ChaCha8Rng::seed_from_u64((11 + cdx as u64 * 17) + seed);

    //                 chunks.iter_mut().try_for_each(|data| {
    //                     self.initialize_params(data, &mut rng)?;

    //                     Ok(())
    //                 })?;

    //                 Ok(())
    //             })?;

    //         Ok(())

    //     }
}
