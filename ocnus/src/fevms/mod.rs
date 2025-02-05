//! Forward ensemble vector models (FEVMs).

mod flter;
mod models;
mod obsvc;

pub use models::{CCLFF_Model, CCUT_Model, NC18_Model};
pub use obsvc::{ModelObserArray, ModelObserVec};

use crate::{
    stats::{CovMatrix, MultivariatePDF, ParticleMutPDF, ParticleRefPDF, StatsError, PDF},
    Fp, OcnusState, PMatrix, ScConf,
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{
    allocator::{Allocator, Reallocator},
    Const, DefaultAllocator, Dim, DimName, Dyn, SVector, SVectorView, StorageMut, VecStorage,
};
use ndarray::{ArrayViewMut2, Axis};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub struct A<T>(T);

/// Errors associated with the [`crate::fevms`] module.
#[derive(Debug, Error)]
pub enum FEVModelError {
    #[error("model parameter value is invalid")]
    InvalidModelParam((&'static str, Fp)),
    #[error("output array has incorrect dimensions")]
    InvalidOutputArray((&'static str, usize, usize)),
    #[error("negative time step in simulation")]
    NegativeTimeStep(Fp),
    #[error("stats error")]
    Stats(StatsError),
}

/// A data structure for holding the parameters, states and weights of a FEVM ensemble.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FEVMEnsbl<S, const P: usize> {
    pub ensbl: PMatrix<Const<P>>,
    pub states: Vec<S>,
    pub weights: Option<Vec<Fp>>,
}

impl<S, const P: usize> FEVMEnsbl<S, P>
where
    S: OcnusState,
{
    // Creates a [`MultivariatePDF`] object from the underlying FEVM ensemble.
    pub fn as_mvpdf(&self, range: [(Fp, Fp); P]) -> Result<MultivariatePDF<P>, FEVModelError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        let covm = match CovMatrix::<Const<P>>::from_particles(&self.ensbl, self.weights.as_ref()) {
            Ok(result) => result,
            Err(err) => return Err(FEVModelError::Stats(err)),
        };

        Ok(MultivariatePDF::new(covm, self.mean(), range))
    }

    /// Creates a [`ParticleRefPDF`] object from the underlying FEVM ensemble.
    pub fn as_ptpdf_ref(&self, range: [(Fp, Fp); P]) -> Result<ParticleRefPDF<P>, FEVModelError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        match ParticleRefPDF::new(None, &self.ensbl, range, self.weights.as_ref()) {
            Ok(result) => Ok(result),
            Err(err) => Err(FEVModelError::Stats(err)),
        }
    }

    /// Creates a [`ParticleMutPDF`] object from the underlying FEVM ensemble.
    pub fn as_ptpdf_mut(&mut self, range: [(Fp, Fp); P]) -> Result<ParticleMutPDF<P>, FEVModelError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        match ParticleMutPDF::new(None, &mut self.ensbl, range, self.weights.as_mut()) {
            Ok(result) => Ok(result),
            Err(err) => Err(FEVModelError::Stats(err)),
        }
    }

    /// Returns `true`` if the ensemble contains no elements.
    pub fn is_empty(&self) -> bool {
        self.ensbl.ncols() == 0
    }

    /// Returns the number of elements in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.ensbl.ncols()
    }

    /// Calculate the mean model parameter from the underlying FEVM ensemble.
    pub fn mean(&self) -> SVector<Fp, P> {
        self.ensbl.column_mean()
    }

    /// Create a new [`EnsembleData`] object with zero'd out entries.
    pub fn new(size: usize) -> Self {
        Self {
            ensbl: PMatrix::<Const<P>>::zeros(size),
            states: vec![S::default(); size],
            weights: None,
        }
    }
}

/// The trait that must be implemented for any FEVM with N-dimensional vector observables.
pub trait ForwardEnsembleVectorModel<S, const P: usize, const N: usize>
where
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

    /// Initialize a [`FEVMEnsbl`] object. If no pdf argument is given,
    /// the underlying model prior is used to generate random model parameter values.
    fn fevm_initialize(
        &self,
        scconf: &[ScConf],
        fevme: &mut FEVMEnsbl<S, P>,
        optional_pdf: Option<impl PDF<P>>,
        seed: u64,
    ) -> Result<(), FEVModelError> {
        debug!(
            "initializing a forward ensemble vector model (size={})",
            fevme.len()
        );

        fevme
            .ensbl
            .par_column_iter_mut()
            .zip(fevme.states.par_iter_mut())
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

    // Initialize the model parameters within a [`FEVMEnsbl`] object. If no pdf argument is given,
    /// the underlying model prior is used to generate random model parameter values.
    fn fevm_initialize_params_only(
        &self,
        fevme: &mut FEVMEnsbl<S, P>,
        optional_pdf: Option<impl PDF<P>>,
        seed: u64,
    ) -> Result<(), FEVModelError> {
        debug!(
            "initializing params of a forward ensemble vector model (size={})",
            fevme.len()
        );

        fevme
            .ensbl
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + (cdx * 23) as u64);

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

    /// Initialize the model states within a [`FEVMEnsbl`] object.
    fn fevm_initialize_states_only(
        &self,
        scconf: &[ScConf],
        fevme: &mut FEVMEnsbl<S, P>,
    ) -> Result<(), FEVModelError> {
        debug!(
            "initializing states of a forward ensemble vector model (size={})",
            fevme.len()
        );

        fevme
            .ensbl
            .par_column_iter()
            .zip(fevme.states.par_iter_mut())
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
        fevme: &mut FEVMEnsbl<S, P>,
        output: &mut ArrayViewMut2<Option<ModelObserVec<N>>>,
    ) -> Result<(), FEVModelError> {
        let mut timer = 0.0;

        if scconf.len() != output.shape()[0] {
            return Err(FEVModelError::InvalidOutputArray((
                "scobs length doesnt match output array",
                output.shape()[0],
                output.shape()[1],
            )));
        }

        if fevme.len() != output.shape()[1] {
            return Err(FEVModelError::InvalidOutputArray((
                "fevme length doesnt match output array",
                output.shape()[0],
                output.shape()[1],
            )));
        }

        debug!(
            "performing an forward ensemble vector model simulation (size={} x {})",
            scconf.len(),
            fevme.len()
        );

        zip_eq(scconf.iter(), output.outer_iter_mut()).try_for_each(
            |(scconf_i, mut output_t)| {
                // Compute time step to next observation.
                let time_step = scconf_i.get_timestamp() - timer;
                timer = scconf_i.get_timestamp();

                if time_step < 0.0 {
                    return Err(FEVModelError::NegativeTimeStep(time_step));
                } else {
                    fevme
                        .ensbl
                        .par_column_iter()
                        .chunks(Self::RCS)
                        .zip(fevme.states.par_chunks_mut(Self::RCS))
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
    fn model_prior(&self) -> impl PDF<P>;
}

pub trait ForwardEnsembleVectorModelPDF {}
