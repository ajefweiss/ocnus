//! Implementations of forward ensemble vector models (FEVMs).

// mod acylm;
mod noise;

// pub use acylm:
pub use noise::*;

use crate::{
    obser::ObserVec,
    stats::{ParticlePDF, PDF},
    OcnusError, OcnusModel, OcnusState, ScObs,
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, DMatrix, DVectorView, Dyn, Matrix, SVectorView, VecStorage, U1};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::time::Instant;
use thiserror::Error;

pub type PMatrix<const P: usize> = Matrix<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>;
pub type ScObserVec<const N: usize> = ScObs<ObserVec<N>>;

/// A data structure that stores a FEVM ensemble particles, model states, and noise configuration.
pub struct FEVMData<G, S, const P: usize, const N: usize>
where
    G: FEVMNoiseGen<N>,
    S: OcnusState,
{
    pub ptpdf: ParticlePDF<P>,
    pub states: Vec<S>,
    pub opt_noise: Option<G>,
    pub seed: u64,
}

impl<G, S, const P: usize, const N: usize> FEVMData<G, S, P, N>
where
    G: FEVMNoiseGen<N>,
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
    fn fevm_data<G>(
        &self,
        series: &[ScObserVec<N>],
        output: &mut DMatrix<ObserVec<N>>,
        size: usize,
        opt_pdf: Option<&impl PDF<P>>,
        opt_noise: Option<G>,
        seed: u64,
    ) -> Result<(), OcnusError>
    where
        G: FEVMNoiseGen<N>,
    {
        let mut counter = 0;
        let mut fevmd = FEVMData {
            ptpdf: ParticlePDF::new(
                Matrix::<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>::zeros(size),
                Self::PARAM_RANGES,
                vec![1.0 / size as f32; size],
            ),
            states: vec![S::default(); size],
            opt_noise,
            seed,
        };
        let mut new_ptpdf = ParticlePDF::new(
            Matrix::<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>::zeros(size),
            Self::PARAM_RANGES,
            vec![1.0 / size as f32; size],
        );

        while counter != size {
            self.fevm_initialize(series, &mut fevmd, opt_pdf)?;

            let indices = self.fevm_simulate_filter(
                series,
                &mut fevmd,
                output,
                None::<fn(&DVectorView<ObserVec<N>>, &[ScObserVec<N>]) -> bool>,
            )?;

            let mut indices_valid = indices
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            // Remove excessive ensemble members.
            if counter + indices_valid.len() > size {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - size
                );
                indices_valid.drain((size - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                new_ptpdf
                    .particles_mut()
                    .column_mut(counter + edx)
                    .iter_mut()
                    .zip(fevmd.ptpdf.particles_ref().column(*idx).iter())
                    .for_each(|(target, value)| *target = *value);
            });

            counter += indices_valid.len();
        }

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
    fn fevm_initialize<G>(
        &self,
        series: &[ScObserVec<N>],
        fevmd: &mut FEVMData<G, S, P, N>,
        opt_pdf: Option<&impl PDF<P>>,
    ) -> Result<(), OcnusError>
    where
        G: FEVMNoiseGen<N>,
    {
        let start = Instant::now();

        fevmd
            .ptpdf
            .particles_mut()
            .par_column_iter_mut()
            .zip(fevmd.states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(fevmd.seed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|(params, state)| {
                    let sample = match opt_pdf.as_ref() {
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
    fn fevm_initialize_params_only<G>(
        &self,
        fevmd: &mut FEVMData<G, S, P, N>,
        opt_pdf: Option<&impl PDF<P>>,
    ) -> Result<(), OcnusError>
    where
        G: FEVMNoiseGen<N>,
    {
        let start = Instant::now();

        fevmd
            .ptpdf
            .particles_mut()
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(fevmd.seed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|params| {
                    let sample = match opt_pdf.as_ref() {
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
    fn fevm_initialize_states_only<G>(
        &self,
        series: &[ScObserVec<N>],
        fevmd: &mut FEVMData<G, S, P, N>,
    ) -> Result<(), OcnusError>
    where
        G: FEVMNoiseGen<N>,
    {
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
        scobs: &ScObserVec<N>,
        params: &SVectorView<f32, P>,
        state: &S,
    ) -> Result<ObserVec<N>, OcnusError>;

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers.
    fn fevm_simulate<G>(
        &self,
        series: &[ScObserVec<N>],
        fevmd: &mut FEVMData<G, S, P, N>,
        output: &mut DMatrix<ObserVec<N>>,
    ) -> Result<(), OcnusError>
    where
        G: FEVMNoiseGen<N>,
    {
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

        if let Some(noise) = fevmd.opt_noise.as_ref() {
            output
                .par_column_iter_mut()
                .chunks(Self::RCS)
                .enumerate()
                .for_each(|(cdx, mut chunks)| {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(fevmd.seed + (cdx * 73) as u64);

                    chunks.iter_mut().for_each(|col| {
                        let size = col.nrows();

                        col.iter_mut()
                            .zip(&noise.generate_noise(size, &mut rng))
                            .for_each(|(value, noisevec)| *value += noisevec.clone());
                    });
                });
        }

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
    fn fevm_simulate_filter<F, G>(
        &self,
        series: &[ScObserVec<N>],
        fevmd: &mut FEVMData<G, S, P, N>,
        output: &mut DMatrix<ObserVec<N>>,
        opt_filter: Option<F>,
    ) -> Result<Vec<bool>, OcnusError>
    where
        G: FEVMNoiseGen<N>,
        F: Send + Sync + Fn(&DVectorView<ObserVec<N>>, &[ScObserVec<N>]) -> bool,
    {
        self.fevm_simulate(series, fevmd, output)?;

        let mut valid_indices_flags = vec![false; fevmd.len()];

        // Collect indices that produce valid results and add random noise.
        if let Some(filter) = opt_filter {
            output
                .par_column_iter()
                .zip(valid_indices_flags.par_iter_mut())
                .chunks(Self::RCS)
                .for_each(|mut chunks| {
                    chunks.iter_mut().for_each(|(out, flag)| {
                        if zip_eq(out.iter(), series.iter()).fold(true, |acc, (out, obs)| {
                            let scenario_1 = !out.any_nan()
                                && !obs.obersvation().unwrap_or(&ObserVec::default()).any_nan();

                            let scenario_2 = out.is_nan()
                                && obs.obersvation().unwrap_or(&ObserVec::default()).is_nan();

                            acc & (scenario_1 || scenario_2)
                        }) {
                            **flag = filter(&out.as_view::<Dyn, U1, U1, Dyn>(), series)
                        }
                    });
                });
        } else {
            output
                .par_column_iter()
                .zip(valid_indices_flags.par_iter_mut())
                .chunks(Self::RCS)
                .for_each(|mut chunks| {
                    chunks.iter_mut().for_each(|(out, flag)| {
                        **flag = zip_eq(out.iter(), series.iter()).fold(true, |acc, (out, obs)| {
                            let scenario_1 = !out.any_nan()
                                && !obs.obersvation().unwrap_or(&ObserVec::default()).any_nan();

                            let scenario_2 = out.is_nan()
                                && obs.obersvation().unwrap_or(&ObserVec::default()).is_nan();

                            acc & (scenario_1 || scenario_2)
                        });
                    });
                });
        }

        Ok(valid_indices_flags)
    }

    /// Initialize a model state (model specific).
    fn fevm_state(
        &self,
        series: &[ScObserVec<N>],
        params: &SVectorView<f32, P>,
        state: &mut S,
    ) -> Result<(), OcnusError>;

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl PDF<P>;

    /// Returns true if the ranges given by the model prior are within the parameter ranges
    /// of the model.
    fn validate_model_prior(&self) -> bool;
}
