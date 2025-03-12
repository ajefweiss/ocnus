//! Implementations of forward ensemble vector models (FEVMs).

mod aclym;
pub mod filters;
pub mod noise;

pub use aclym::*;
use filters::ParticleFilterError;
use noise::FEVMNoiseGenerator;
use serde::{Deserialize, Serialize};

use crate::{
    ScObs, ScObsSeries, geometry::OcnusGeometry, obser::ObserVec, stats::PDF, stats::StatsError,
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, DMatrix, Dyn, Matrix};
use nalgebra::{SVectorView, VecStorage};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;
use thiserror::Error;

/// A data structure that stores the parameters, states and random seed for a FEVM.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FEVMData<const P: usize, FS, GS> {
    /// FEVM ensemble parameters.
    pub params: Matrix<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>,

    /// FEVM ensemble states.
    pub fevm_states: Vec<FS>,

    /// Geometry ensemble states.
    pub geom_states: Vec<GS>,

    /// Ensemble member weights.
    pub weights: Vec<f32>,
}

impl<const P: usize, FS, GS> FEVMData<P, FS, GS>
where
    FS: Clone + Default,
    GS: Clone + Default,
    Self: Serialize,
{
    /// Create a new [`FEVMData`] filled with zeros.
    pub fn new(size: usize) -> Self {
        Self {
            params: Matrix::<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>::zeros(size),
            fevm_states: vec![FS::default(); size],
            geom_states: vec![GS::default(); size],
            weights: vec![1.0 / size as f32; size],
        }
    }

    /// Returns the size of the ensemble.
    pub fn size(&self) -> usize {
        self.params.ncols()
    }

    /// Write data to a file.
    pub fn write(&self, path: String) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}

/// Errors associated with types that implement the [`FEVM`] trait.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum FEVMError {
    #[error("invalid model parameter {name}={value}")]
    InvalidParameter { name: &'static str, value: f32 },
    #[error(
        "invalid range {output_rows} x {output_cols} but expected {expected_rows} x {expected_cols}"
    )]
    InvalidOutputShape {
        expected_cols: usize,
        expected_rows: usize,
        output_cols: usize,
        output_rows: usize,
    },
    #[error("attempted to simulate backwards in time (dt=-{0:.2}sec)")]
    NegativeTimeStep(f32),
    #[error("particle filter error")]
    ParticleFilter(#[from] ParticleFilterError),
    #[error("stats error")]
    Stats(#[from] StatsError),
}

/// The trait that must be implemented for any FEVM (forward ensemble vector model) with an N-dimensional vector observable.
pub trait FEVM<const P: usize, const N: usize, FS, GS>: OcnusGeometry<P, GS>
where
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
    Self: Sync,
{
    /// The rayon chunk size that is used for any parallel iterators.
    /// Cheap or expensive operations may use fractions or multiples of this value.
    const RCS: usize;

    /// Evolve a model state forward in time.
    fn fevm_forward(
        &self,
        time_step: f32,
        params: &SVectorView<f32, P>,
        fevm_state: &mut FS,
        geom_state: &mut GS,
    ) -> Result<(), FEVMError>;

    /// Initialize parameters and states for a FEVM ensemble.
    /// If no `opt_pdf` is given, the underlying model prior is used instead.
    fn fevm_initialize<T>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &mut FEVMData<P, FS, GS>,
        opt_pdf: Option<&T>,
        rseed: u64,
    ) -> Result<(), FEVMError>
    where
        for<'a> &'a T: PDF<P>,
    {
        let start = Instant::now();

        fevmd
            .params
            .par_column_iter_mut()
            .zip(fevmd.fevm_states.par_iter_mut())
            .zip(fevmd.geom_states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

                chunks
                    .iter_mut()
                    .try_for_each(|((params, fevm_state), geom_state)| {
                        let sample = match opt_pdf.as_ref() {
                            Some(pdf) => pdf.draw_sample(&mut rng),
                            None => self.model_prior().draw_sample(&mut rng),
                        }?;

                        params.set_column(0, &sample);

                        self.fevm_state(series, &params.as_view(), fevm_state, geom_state)?;

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
        fevmd: &mut FEVMData<P, FS, GS>,
        opt_pdf: Option<impl PDF<P>>,
        rseed: u64,
    ) -> Result<(), FEVMError> {
        let start = Instant::now();

        fevmd
            .params
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

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

    /// Initialize the model states within a [`FEVMData`] object.
    fn fevm_initialize_states_only(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &mut FEVMData<P, FS, GS>,
    ) -> Result<(), FEVMError> {
        let start = Instant::now();

        fevmd
            .params
            .par_column_iter()
            .zip(fevmd.fevm_states.par_iter_mut())
            .zip(fevmd.geom_states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunks| {
                chunks
                    .iter_mut()
                    .try_for_each(|((params, fevm_state), geom_state)| {
                        self.fevm_state(series, &params.as_view(), fevm_state, geom_state)?;

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
        fevm_state: &FS,
        geom_state: &GS,
    ) -> Result<ObserVec<N>, FEVMError>;

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers. Returns indices of runs that are valid w.r.t.
    /// to the spacecraft observation series.
    fn fevm_simulate<NG>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &mut FEVMData<P, FS, GS>,
        output: &mut DMatrix<ObserVec<N>>,
        opt_noise: Option<(&NG, u64)>,
    ) -> Result<Vec<bool>, FEVMError>
    where
        NG: FEVMNoiseGenerator<N>,
    {
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
                    .zip(fevmd.fevm_states.par_iter_mut())
                    .zip(fevmd.geom_states.par_iter_mut())
                    .zip(output_col.par_column_iter_mut())
                    .chunks(Self::RCS)
                    .try_for_each(|mut chunks| {
                        chunks.iter_mut().try_for_each(
                            |(((data, fevm_state), geom_state), out)| {
                                self.fevm_forward(time_step, data, fevm_state, geom_state)?;
                                out[(0, 0)] =
                                    self.fevm_observe(scobs, data, fevm_state, geom_state)?;

                                Ok::<(), FEVMError>(())
                            },
                        )?;

                        Ok::<(), FEVMError>(())
                    })?;
            }

            Ok::<(), FEVMError>(())
        })?;

        if let Some((noise, noise_seed)) = opt_noise {
            output
                .par_column_iter_mut()
                .chunks(Self::RCS)
                .enumerate()
                .for_each(|(cdx, mut chunks)| {
                    let mut rng: Xoshiro256PlusPlus =
                        Xoshiro256PlusPlus::seed_from_u64(noise_seed + (cdx * 73) as u64);

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

        let mut valid_indices_flags = vec![false; fevmd.params.ncols()];

        output
            .par_column_iter_mut()
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

        Ok(valid_indices_flags)
    }

    /// Initialize a model state.
    fn fevm_state(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        params: &SVectorView<f32, P>,
        fevm_state: &mut FS,
        geom_state: &mut GS,
    ) -> Result<(), FEVMError>;

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl PDF<P>;

    /// Returns true if the ranges given by the model prior are within the parameter ranges
    /// of the model.
    fn validate_model_prior(&self) -> bool;
}
