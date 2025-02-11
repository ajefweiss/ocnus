//! Implementations of forward ensemble vector models (FEVMs).

mod acylm;

// pub use acylm:

use crate::{
    obser::ObserVec,
    stats::{ParticlePDF, PDF},
    OcnusError, OcnusModel, OcnusState, ScObs,
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{DMatrix, SVectorView};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::time::Instant;
use thiserror::Error;

pub struct FEVMData<S, const P: usize>
where
    S: OcnusState,
{
    pub ptpdf: ParticlePDF<P>,
    pub states: Vec<S>,
}

impl<S, const P: usize> FEVMData<S, P>
where
    S: OcnusState,
{
    /// Returns `true`` if the ensemble contains no elements.
    pub fn is_empty(&self) -> bool {
        self.ptpdf.is_empty()
    }

    /// Returns the number of particles in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.ptpdf.len()
    }
}

/// Errors associated with types that implement the [`FEVM`] trait.
#[derive(Debug, Error)]
pub enum FEVMError {
    #[error("invalid model parameter {name}={value}")]
    InvalidParameter { name: &'static str, value: f32 },
    #[error("invalid range {output_rows} x {output_cols} but expected {expected_rows} x {expected_cols}")]
    InvalidOutputShape {
        expected_cols: usize,
        expected_rows: usize,
        output_cols: usize,
        output_rows: usize,
    },
    #[error("attempted to simulate backwards in time (dt={0:.2}sec)")]
    NegativeTimeStep(f32),
}

/// The trait that must be implemented for any FEVM with N-dimensional vector observables.
pub trait FEVM<S, const P: usize, const N: usize>: OcnusModel<P>
where
    S: OcnusState,
    Self: Sync,
{
    /// The rayon chunk size that is used for any parallel iterators.
    /// Cheap or expensive operations may use fractions or multiples of this value.
    const RCS: usize;

    /// Create a new [`FEVMData`] from "valid" simulations.
    fn fevm_data(
        &self,
        series: &[ScObs<ObserVec<N>>],
        fevmd: &mut FEVMData<S, P>,
        output: &mut DMatrix<ObserVec<N>>,
    ) -> Result<(), OcnusError> {
        Ok(())
    }

    /// Evolve a model state forward in time (model specific).
    fn fevm_forward(
        &self,
        time_step: f32,
        params: &SVectorView<f32, P>,
        state: &mut S,
    ) -> Result<(), OcnusError>;

    /// Initialize parameters and states for a FEVM.
    /// If no pdf is given, the underlying model prior is used instead.
    fn fevm_initialize(
        &self,
        series: &[ScObs<ObserVec<N>>],
        fevmd: &mut FEVMData<S, P>,
        optional_pdf: Option<&impl PDF<P>>,
        seed: u64,
    ) -> Result<(), OcnusError> {
        let start = Instant::now();

        fevmd
            .ptpdf
            .particles_mut()
            .par_column_iter_mut()
            .zip(fevmd.states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|(params, state)| {
                    let sample = match optional_pdf.as_ref() {
                        Some(pdf) => pdf.draw_sample(&mut rng),
                        None => self.model_prior().draw_sample(&mut rng),
                    }?;

                    params.set_column(0, &sample);

                    self.fevm_state(series, &params.as_view(), state)?;

                    Ok::<(), OcnusError>(())
                })?;

                Ok::<(), OcnusError>(())
            })?;

        debug!(
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            fevmd.len() as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3
        );

        Ok(())
    }

    /// Initialize parameters and states for a FEVM.
    /// If no pdf is given, the underlying model prior is used instead.
    fn fevm_initialize_params_only(
        &self,
        fevmd: &mut FEVMData<S, P>,
        optional_pdf: Option<&impl PDF<P>>,
        seed: u64,
    ) -> Result<(), OcnusError> {
        let start = Instant::now();

        fevmd
            .ptpdf
            .particles_mut()
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|params| {
                    let sample = match optional_pdf.as_ref() {
                        Some(pdf) => pdf.draw_sample(&mut rng),
                        None => self.model_prior().draw_sample(&mut rng),
                    }?;

                    params.set_column(0, &sample);

                    Ok::<(), OcnusError>(())
                })?;

                Ok::<(), OcnusError>(())
            })?;

        debug!(
            "fevm_initialize_params_only: {:2.2}M evaluations in {:.2} sec",
            fevmd.len() as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3
        );

        Ok(())
    }

    /// Initialize the model states within a [`FEVMEnsbl`] object.
    fn fevm_initialize_states_only(
        &self,
        series: &[ScObs<ObserVec<N>>],
        fevmd: &mut FEVMData<S, P>,
    ) -> Result<(), OcnusError> {
        let start = Instant::now();

        fevmd
            .ptpdf
            .particles_ref()
            .par_column_iter()
            .zip(fevmd.states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunks| {
                chunks.iter_mut().try_for_each(|(params, state)| {
                    self.fevm_state(series, &params.as_view(), state)?;

                    Ok::<(), OcnusError>(())
                })?;

                Ok::<(), OcnusError>(())
            })?;

        debug!(
            "fevm_initialize_states_only: {:2.2}M evaluations in {:.2} sec",
            fevmd.len() as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3
        );

        Ok(())
    }

    /// Generate a vector observable.
    fn fevm_observe(
        &self,
        scobs: &ScObs<ObserVec<N>>,
        params: &SVectorView<f32, P>,
        state: &S,
    ) -> Result<ObserVec<N>, OcnusError>;

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers.
    fn fevm_simulate(
        &self,
        series: &[ScObs<ObserVec<N>>],
        fevmd: &mut FEVMData<S, P>,
        output: &mut DMatrix<ObserVec<N>>,
    ) -> Result<(), OcnusError> {
        let start = Instant::now();
        let mut timer = 0.0;

        if (series.len() != output.nrows()) || (fevmd.len() != output.ncols()) {
            return Err(OcnusError::FEVM(FEVMError::InvalidOutputShape {
                expected_cols: fevmd.len(),
                expected_rows: series.len(),
                output_cols: output.ncols(),
                output_rows: output.nrows(),
            }));
        }

        zip_eq(series.iter(), output.row_iter_mut()).try_for_each(|(scobs, mut output_col)| {
            // Compute time step to next observation.
            let time_step = scobs.timestamp() - timer;
            timer = scobs.timestamp();

            if time_step < 0.0 {
                return Err(OcnusError::FEVM(FEVMError::NegativeTimeStep(time_step)));
            } else {
                fevmd
                    .ptpdf
                    .particles_ref()
                    .par_column_iter()
                    .zip(fevmd.states.par_iter_mut())
                    .zip(output_col.par_column_iter_mut())
                    .chunks(Self::RCS)
                    .try_for_each(|mut chunks| {
                        chunks.iter_mut().try_for_each(|((data, state), out)| {
                            self.fevm_forward(time_step, data, state)?;
                            out[(0, 0)] = self.fevm_observe(scobs, data, state)?;

                            Ok::<(), OcnusError>(())
                        })?;

                        Ok::<(), OcnusError>(())
                    })?;
            }

            Ok::<(), OcnusError>(())
        })?;

        debug!(
            "fevm_simulate: {:2.2}M evaluations in {:.2} sec",
            (series.len() * fevmd.len()) as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3
        );

        Ok(())
    }

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers. Returns indices of runs that are valid w.r.t. the
    /// observation series.
    fn fevm_simulate_valid_indices(
        &self,
        series: &[ScObs<ObserVec<N>>],
        fevmd: &mut FEVMData<S, P>,
        output: &mut DMatrix<ObserVec<N>>,
    ) -> Result<Vec<bool>, OcnusError> {
        self.fevm_simulate(series, fevmd, output)?;

        let mut valid_indices_flags = vec![false; fevmd.len()];

        // Collect indices that produce valid results and add random noise.
        output
            .par_column_iter_mut()
            .zip(valid_indices_flags.par_iter_mut())
            .chunks(Self::RCS)
            .for_each(|mut chunks| {
                chunks.iter_mut().for_each(|(out, flag)| {
                    **flag = zip_eq(out.iter(), series.iter()).fold(true, |acc, (o, r)| {
                        let obs = r.obersvation();
                        acc & ((!o.any_nan() && obs.is_some()) || (o.any_nan() && obs.is_none()))
                    })
                });
            });

        Ok(valid_indices_flags)
    }

    /// Initialize a model state (model specific).
    fn fevm_state(
        &self,
        series: &[ScObs<ObserVec<N>>],
        params: &SVectorView<f32, P>,
        state: &mut S,
    ) -> Result<(), OcnusError>;

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl PDF<P>;

    /// Returns true if the ranges given by the model prior are within the parameter ranges
    /// of the model.
    fn validate_model_prior(&self) -> bool;
}
