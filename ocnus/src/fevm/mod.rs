//! Implementations of forward ensemble vector models (FEVMs).

mod noise;

use crate::stats::{OcnusStatisticsError, PDFParticles};
use crate::ScObs;
use crate::{obser::ObserVec, stats::PDF, OcnusModel, OcnusState, ScObsSeries};
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, DMatrix, DVectorView, Dyn, Matrix, U1};
use nalgebra::{SVectorView, VecStorage};
use noise::FEVMNoiseGen;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::time::Instant;
use thiserror::Error;

/// A data structure that stores a FEVM ensemble.
pub struct FEVMData<const P: usize, S>
where
    S: OcnusState,
{
    /// FEVM ensemble parameters.
    pub params: Matrix<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>,

    /// FEVM ensemble states.
    pub states: Vec<S>,

    /// FEVM random seed.
    pub rseed: u64,
}

/// Errors associated with types that implement the [`FEVM`] trait.
#[allow(missing_docs)]
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
    #[error("stats error")]
    Stats(#[from] OcnusStatisticsError),
}

/// The trait that must be implemented for any FEVM with an N-dimensional vector observable.
pub trait FEVM<const P: usize, const N: usize, S>: OcnusModel<P, S>
where
    S: OcnusState,
    Self: Sync,
{
    /// The rayon chunk size that is used for any parallel iterators.
    /// Cheap or expensive operations may use fractions or multiples of this value.
    const RCS: usize;

    /// Create a [`FEVMData`] from valid simulations.
    fn fevm_data(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        output: &mut DMatrix<ObserVec<N>>,
        size: usize,
        opt_pdf: Option<&impl PDF<P>>,
        opt_noise: Option<&impl FEVMNoiseGen<N>>,
        rseed: u64,
    ) -> Result<FEVMData<P, S>, FEVMError> {
        let mut counter = 0;

        let mut fevmd = FEVMData {
            params: Matrix::<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>::zeros(size),
            states: vec![S::default(); size],
            rseed,
        };

        let mut new_params =
            Matrix::<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>::zeros(size);

        while counter != size {
            self.fevm_initialize(series, &mut fevmd, opt_pdf)?;

            let indices = self.fevm_simulate_filter(
                series,
                &mut fevmd,
                output,
                None::<fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> bool>,
                opt_noise,
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
                new_params
                    .column_mut(counter + edx)
                    .iter_mut()
                    .zip(fevmd.params.column(*idx).iter())
                    .for_each(|(target, value)| *target = *value);
            });

            counter += indices_valid.len();
        }

        Ok(FEVMData {
            params: new_params,
            states: vec![S::default(); size],
            rseed: rseed + 1,
        })
    }

    /// Evolve a model state forward in time.
    fn fevm_forward(
        &self,
        time_step: f32,
        params: &SVectorView<f32, P>,
        state: &mut S,
    ) -> Result<(), FEVMError>;

    /// Initialize parameters and states for a FEVM ensemble.
    /// If no `opt_pdf` is given, the underlying model prior is used instead.
    fn fevm_initialize(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &mut FEVMData<P, S>,
        opt_pdf: Option<&impl PDF<P>>,
    ) -> Result<(), FEVMError> {
        let start = Instant::now();

        fevmd
            .params
            .par_column_iter_mut()
            .zip(fevmd.states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(fevmd.rseed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|(params, state)| {
                    let sample = match opt_pdf.as_ref() {
                        Some(pdf) => pdf.draw_sample(&mut rng),
                        None => self.model_prior().draw_sample(&mut rng),
                    }?;

                    params.set_column(0, &sample);

                    self.fevm_state(series, &params.as_view(), state)?;

                    Ok::<(), FEVMError>(())
                })?;

                Ok::<(), FEVMError>(())
            })?;

        debug!(
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols() as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3
        );

        Ok(())
    }

    /// Initialize parameters and states for a FEVM.
    /// If no pdf is given, the underlying model prior is used instead.
    fn fevm_initialize_params_only(
        &self,
        fevmd: &mut FEVMData<P, S>,
        opt_pdf: Option<&impl PDF<P>>,
    ) -> Result<(), FEVMError> {
        let start = Instant::now();

        fevmd
            .params
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(fevmd.rseed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|params| {
                    let sample = match opt_pdf.as_ref() {
                        Some(pdf) => pdf.draw_sample(&mut rng),
                        None => self.model_prior().draw_sample(&mut rng),
                    }?;

                    params.set_column(0, &sample);

                    Ok::<(), FEVMError>(())
                })?;

                Ok::<(), FEVMError>(())
            })?;

        debug!(
            "fevm_initialize_params_only: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols() as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3
        );

        Ok(())
    }

    /// Initialize the model states within a [`FEVMEnsbl`] object.
    fn fevm_initialize_states_only(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &mut FEVMData<P, S>,
    ) -> Result<(), FEVMError> {
        let start = Instant::now();

        fevmd
            .params
            .par_column_iter()
            .zip(fevmd.states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunks| {
                chunks.iter_mut().try_for_each(|(params, state)| {
                    self.fevm_state(series, &params.as_view(), state)?;

                    Ok::<(), FEVMError>(())
                })?;

                Ok::<(), FEVMError>(())
            })?;

        debug!(
            "fevm_initialize_states_only: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols() as f32 / 1e6,
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
    ) -> Result<ObserVec<N>, FEVMError>;

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers.
    fn fevm_simulate(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &mut FEVMData<P, S>,
        output: &mut DMatrix<ObserVec<N>>,
        opt_noise: Option<&impl FEVMNoiseGen<N>>,
    ) -> Result<(), FEVMError> {
        let start = Instant::now();
        let mut timer = 0.0;

        if (series.len() != output.nrows()) || (fevmd.params.ncols() != output.ncols()) {
            return Err(FEVMError::InvalidOutputShape {
                expected_cols: fevmd.params.ncols(),
                expected_rows: series.len(),
                output_cols: output.ncols(),
                output_rows: output.nrows(),
            });
        }

        zip_eq(series, output.row_iter_mut()).try_for_each(|(scobs, mut output_col)| {
            // Compute time step to next observation.
            let time_step = scobs.timestamp() - timer;
            timer = scobs.timestamp();

            if time_step < 0.0 {
                return Err(FEVMError::NegativeTimeStep(time_step));
            } else {
                fevmd
                    .params
                    .par_column_iter()
                    .zip(fevmd.states.par_iter_mut())
                    .zip(output_col.par_column_iter_mut())
                    .chunks(Self::RCS)
                    .try_for_each(|mut chunks| {
                        chunks.iter_mut().try_for_each(|((data, state), out)| {
                            self.fevm_forward(time_step, data, state)?;
                            out[(0, 0)] = self.fevm_observe(scobs, data, state)?;

                            Ok::<(), FEVMError>(())
                        })?;

                        Ok::<(), FEVMError>(())
                    })?;
            }

            Ok::<(), FEVMError>(())
        })?;

        if let Some(noise) = opt_noise {
            output
                .par_column_iter_mut()
                .chunks(Self::RCS)
                .enumerate()
                .for_each(|(cdx, mut chunks)| {
                    let mut rng =
                        Xoshiro256PlusPlus::seed_from_u64(fevmd.rseed + (cdx * 73) as u64);

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
            (series.len() * fevmd.params.ncols()) as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3
        );

        Ok(())
    }

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers. Returns indices of runs that are valid w.r.t. the
    /// observation series.
    fn fevm_simulate_filter<F>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &mut FEVMData<P, S>,
        output: &mut DMatrix<ObserVec<N>>,
        opt_filter: Option<F>,
        opt_noise: Option<&impl FEVMNoiseGen<N>>,
    ) -> Result<Vec<bool>, FEVMError>
    where
        F: Send + Sync + Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> bool,
    {
        self.fevm_simulate(series, fevmd, output, opt_noise)?;

        let mut valid_indices_flags = vec![false; fevmd.params.ncols()];

        // Collect indices that produce valid results and add random noise.
        if let Some(filter) = opt_filter {
            output
                .par_column_iter()
                .zip(valid_indices_flags.par_iter_mut())
                .chunks(Self::RCS)
                .for_each(|mut chunks| {
                    chunks.iter_mut().for_each(|(out, flag)| {
                        if zip_eq(out.iter(), series).fold(true, |acc, (out, obs)| {
                            let scenario_1 = !out.any_nan()
                                && !obs.observation().unwrap_or(&ObserVec::default()).any_nan();

                            let scenario_2 = out.is_nan()
                                && obs.observation().unwrap_or(&ObserVec::default()).is_nan();

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
                        **flag = zip_eq(out.iter(), series).fold(true, |acc, (out, obs)| {
                            let scenario_1 = !out.any_nan()
                                && !obs.observation().unwrap_or(&ObserVec::default()).any_nan();

                            let scenario_2 = out.is_nan()
                                && obs.observation().unwrap_or(&ObserVec::default()).is_nan();

                            acc & (scenario_1 || scenario_2)
                        });
                    });
                });
        }

        Ok(valid_indices_flags)
    }

    /// Initialize a model state.
    fn fevm_state(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        params: &SVectorView<f32, P>,
        state: &mut S,
    ) -> Result<(), FEVMError>;

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl PDF<P>;

    /// Returns true if the ranges given by the model prior are within the parameter ranges
    /// of the model.
    fn validate_model_prior(&self) -> bool;
}
