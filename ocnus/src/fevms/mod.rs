//! Forward ensemble vector models (FEVMs).

mod models;
mod obsvc;

use crate::{
    stats::{ProbabilityDensityFunction, ProbabilityDensityFunctionSampling, StatsError},
    Fp, OcnusState, PMatrix, ScConf,
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, SVectorView};
use ndarray::{ArrayViewMut2, Axis};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use obsvc::ModelObserVec;

/// Errors associated with the [`crate::fevms`] module.
#[derive(Debug, Error)]
pub enum FEVModelError {
    #[error("output array has incorrect dimensions")]
    InvalidOutputArray((usize, usize)),
    #[error("negative time step in simulation")]
    NegativeTimeStep(Fp),
    #[error("stats error")]
    Stats(StatsError),
}

/// A data structure for holding the model parameters and states of a FEVM.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FEVMData<S, const P: usize>(pub PMatrix<Const<P>>, pub Vec<S>);

impl<S, const P: usize> FEVMData<S, P>
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

/// The trait that must be implemented for any FEVM with N-dimensional vector observables.
pub trait ForwardEnsembleVectorModel<T, S, const P: usize, const N: usize>
where
    for<'a> &'a T: ProbabilityDensityFunctionSampling<P>,
    S: OcnusState,
    Self: Sync,
{
    /// The rayon chunk size that is used for any parallel iterators.
    /// Specific cheap or expensive operations may use fractions or multiples of this value.
    const RCS: usize;

    /// Evolve a model state forward in time (model specific).
    fn fevm_forward(
        &self,
        time_step: Fp,
        params: &SVectorView<Fp, P>,
        state: &mut S,
    ) -> Result<(), FEVModelError>;

    /// Initialize a [`FEVMData`] object. If no pdf argument is given,
    /// the underlying model prior is used to generate random model parameter values.
    fn fevm_initialize(
        &self,
        scconf: &[ScConf],
        fevmd: &mut FEVMData<S, P>,
        optional_pdf: Option<impl ProbabilityDensityFunctionSampling<P>>,
        seed: u64,
    ) -> Result<(), FEVModelError> {
        debug!(
            "initializing a forward ensemble vector model (size={})",
            fevmd.len()
        );

        fevmd
            .0
            .par_column_iter_mut()
            .zip(fevmd.1.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|(params, state)| {
                    let sample = match optional_pdf.as_ref() {
                        Some(pdf) => match pdf.sample(&mut rng) {
                            Ok(value) => value,
                            Err(err) => return Err(FEVModelError::Stats(err)),
                        },
                        None => match self.model_prior().sample(&mut rng) {
                            Ok(value) => value,
                            Err(err) => return Err(FEVModelError::Stats(err)),
                        },
                    };

                    params.set_column(0, &sample);

                    self.fevm_state(scconf, &params.as_view(), state)?;

                    Ok(())
                })?;

                Ok(())
            })?;

        Ok(())
    }

    // Initialize the model parameters within a [`FEVMData`] object. If no pdf argument is given,
    /// the underlying model prior is used to generate random model parameter values.
    fn fevm_initialize_params_only(
        &self,
        fevmd: &mut FEVMData<S, P>,
        optional_pdf: Option<impl ProbabilityDensityFunctionSampling<P>>,
        seed: u64,
    ) -> Result<(), FEVModelError> {
        debug!(
            "initializing params of a forward ensemble vector model (size={})",
            fevmd.len()
        );

        fevmd
            .0
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|params| {
                    let sample = match optional_pdf.as_ref() {
                        Some(pdf) => match pdf.sample(&mut rng) {
                            Ok(value) => value,
                            Err(err) => return Err(FEVModelError::Stats(err)),
                        },
                        None => match self.model_prior().sample(&mut rng) {
                            Ok(value) => value,
                            Err(err) => return Err(FEVModelError::Stats(err)),
                        },
                    };

                    params.set_column(0, &sample);

                    Ok(())
                })?;

                Ok(())
            })?;

        Ok(())
    }

    /// Initialize the model states within a [`FEVMData`] object.
    fn fevm_initialize_states_only(
        &self,
        scconf: &[ScConf],
        fevmd: &mut FEVMData<S, P>,
    ) -> Result<(), FEVModelError> {
        debug!(
            "initializing states of a forward ensemble vector model (size={})",
            fevmd.len()
        );

        fevmd
            .0
            .par_column_iter()
            .zip(fevmd.1.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunks| {
                chunks.iter_mut().try_for_each(|(params, state)| {
                    self.fevm_state(scconf, &params.as_view(), state)?;

                    Ok(())
                })?;

                Ok(())
            })?;

        Ok(())
    }

    /// Generate a vector observable.
    fn fevm_observe(
        &self,
        scconf: &ScConf,
        params: &SVectorView<Fp, P>,
        state: &S,
    ) -> Result<Option<ModelObserVec<N>>, FEVModelError>;

    /// Perform an ensemble forward simulation and generate synthetic vector observables for the given spacecraft observers.
    fn fevm_simulate(
        &self,
        scconf: &[ScConf],
        fevmd: &mut FEVMData<S, P>,
        output: &mut ArrayViewMut2<Option<ModelObserVec<N>>>,
    ) -> Result<(), FEVModelError> {
        let mut timer = 0.0;

        if fevmd.len() != output.shape()[0] {
            return Err(FEVModelError::InvalidOutputArray((
                output.shape()[0],
                output.shape()[1],
            )));
        }

        if scconf.len() != output.shape()[1] {
            return Err(FEVModelError::InvalidOutputArray((
                output.shape()[0],
                output.shape()[1],
            )));
        }

        debug!(
            "performing an forward ensemble vector model simulation (size={} x {})",
            scconf.len(),
            fevmd.len()
        );

        zip_eq(scconf.iter(), output.outer_iter_mut()).try_for_each(
            |(scconf_i, mut output_t)| {
                // Compute time step to next observation.
                let time_step = scconf_i.get_timestamp() - timer;
                timer = scconf_i.get_timestamp();

                if time_step < 0.0 {
                    return Err(FEVModelError::NegativeTimeStep(time_step));
                } else {
                    fevmd
                        .0
                        .par_column_iter()
                        .chunks(Self::RCS)
                        .zip(fevmd.1.par_chunks_mut(Self::RCS))
                        .zip(
                            output_t
                                .axis_chunks_iter_mut(Axis(0), Self::RCS)
                                .into_par_iter(),
                        )
                        .try_for_each(|((chunks_params, chunks_states), mut chunks_out)| {
                            chunks_params
                                .iter()
                                .zip(chunks_states.iter_mut())
                                .zip(chunks_out.iter_mut())
                                .try_for_each(|((data, state), out)| {
                                    self.fevm_forward(time_step, data, state)?;
                                    *out = self.fevm_observe(scconf_i, data, state)?;

                                    Ok(())
                                })?;

                            Ok(())
                        })?;
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    /// Initialize a model state (model specific).
    fn fevm_state(
        &self,
        scconf: &[ScConf],
        params: &SVectorView<Fp, P>,
        state: &mut S,
    ) -> Result<(), FEVModelError>;

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> &ProbabilityDensityFunction<T, P>;
}
