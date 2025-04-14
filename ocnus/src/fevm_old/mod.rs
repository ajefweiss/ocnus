//! Forward ensemble vector modeling framework.
//!
//! # Forward Ensemble Vector Models (FEVMs)
//!
//! This module contains the bread and butter for ensemble modeling with either
//! magnetic flux rope or solar wind models. The [`FEVM`] trait can be implemented
//! for models that are forward simulations and are numerically cheap enough to run
//! in large ensembles. It further assumes that the model output can be interpreted
//! as a N-dimensional vector observable (i.e. [`ObserVec`]). A [`FEVM`] must be built
//! ontop of a geometry model (a type that implements [`OcnusCoords`]), and also make
//! use an additional state type to handle  time-dependence on top of the geometry state.
//!
//! The basic type within this module is [`FEVMEnsbl`], which describes an ensemble, with a
//! matrix describing the model parameters, and two vectors
//! for the geometry and fevm states. Each ensemble member must also be assigned a
//! weight according to the importance within the ensemble.
//!
//! The [`FEVM`] trait provides, among others, the following important methods:
//! - [`FEVM::fevm_initialize`] Initializes the model parameters of a [`FEVMEnsbl`]
//!   using the model or a custom prior.
//! - [`FEVM::fevm_initialize_params_only`] / [`FEVM::fevm_initialize_states_only`]
//!   These two functions only initialize the model parameters or the model states respectively.
//! - [`FEVM::fevm_simulate`] Generates the model outputs for a given [`ScObsSeries`].
//! - [`FEVM::fevm_simulate_diagnostics`] Computes and returns the internal coordinates,
//!   and basis vectors for a given [`ScObsSeries`]. This function exists primarily for
//!   diagnostic purposes.
//!
//! #### Noise
//!
//! Any [`FEVM`] can make use of an additive noise model via the [`FEVMNoise`] type. The currently available
//! additive noise generators are:
//! - [`FEVMNoise::Gaussian`] Gaussian noise with a given standard deviation σ.
//! - [`FEVMNoise::Multivariate`] Multivariate normal noise with a given covariance matrix Σ.
//!
//! ## Implemented Models
//!
//! #### Magnetic Flux Rope
//!
//! - [`CCLFFModel`](`models::CCLFFModel`) A linear force-free cylindrical flux rope model.
//! - [`CCUTModel`](`models::CCUTModel`) A uniform twist force-free cylindrical flux rope model.
//! - [`ECHModel`](`models::ECHModel`) An elliptic-cylindrical hybrid (lff + ut) flux rope model.
//!
//! #### Solar Wind
//!
//! - [`WSAModel`](`models::WSAModel`) A standard implementation of the Wang-Sheeley-Arge solar wind model.
//!
//! # Particle Filters

pub mod filters;
mod fisher;
pub mod models;
mod noise;

pub use fisher::*;
pub use noise::*;

/// The trait that must be implemented for any FEVM (forward ensemble vector model) with an N-dimensional vector observable.
pub trait FEVM<T, const P: usize, const N: usize, FS, GS>: OcnusCoords<T, P, GS>
where
    T: Copy + Float + RealField + SampleUniform + Scalar,
    FS: Send,
    GS: Send,
    Self: Sync,
    StandardNormal: Distribution<T>,
{
    /// The rayon chunk size that is used for any parallel iterators.
    /// Cheap or expensive operations may use fractions or multiples of this value.
    const RCS: usize;

    /// Evolve a model state forward in time.
    fn fevm_forward(
        &self,
        time_step: T,
        params: &SVectorView<T, P>,
        fevm_state: &mut FS,
        geom_state: &mut GS,
    ) -> Result<(), FEVMError<T>>;

    /// Initialize parameters and states for a FEVM ensemble.
    /// If no `opt_pdf` is given, the underlying model prior is used instead.
    fn fevm_initialize<D>(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMEnsbl<T, P, FS, GS>,
        opt_pdf: Option<&D>,
        rseed: u64,
    ) -> Result<(), FEVMError<T>>
    where
        for<'a> &'a D: OcnusProDeF<T, P>,
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

                        Self::geom_state(&params.fixed_rows::<P>(0), geom_state);
                        self.fevm_state(series, &params.as_view(), fevm_state, geom_state)?;

                        Ok::<(), FEVMError<T>>(())
                    })?;

                Ok::<(), FEVMError<T>>(())
            })?;

        debug!(
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize parameters and states for a FEVM ensemble by
    /// resampling from a [`ParticlesND`].
    fn fevm_initialize_resample(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMEnsbl<T, P, FS, GS>,
        pdf: &ParticlesND<T, P>,
        rseed: u64,
    ) -> Result<(), FEVMError<T>>
    where
        T: Sum<T> + for<'x> Sum<&'x T>,
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
                        let sample = pdf.resample(&mut rng)?;

                        params.set_column(0, &sample);

                        Self::geom_state(&params.fixed_rows::<P>(0), geom_state);
                        self.fevm_state(series, &params.as_view(), fevm_state, geom_state)?;

                        Ok::<(), FEVMError<T>>(())
                    })?;

                Ok::<(), FEVMError<T>>(())
            })?;

        debug!(
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize parameters and states for a FEVM.
    /// If no pdf is given, the underlying model prior is used instead.
    fn fevm_initialize_params_only(
        &self,
        fevmd: &mut FEVMEnsbl<T, P, FS, GS>,
        opt_pdf: Option<impl OcnusProDeF<T, P>>,
        rseed: u64,
    ) -> Result<(), FEVMError<T>> {
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

                    Ok::<(), FEVMError<T>>(())
                })?;

                Ok::<(), FEVMError<T>>(())
            })?;

        debug!(
            "fevm_initialize_params_only: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize the model states within a [`FEVMEnsbl`] object.
    fn fevm_initialize_states_only(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMEnsbl<T, P, FS, GS>,
    ) -> Result<(), FEVMError<T>> {
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
                        Self::geom_state(&params.fixed_rows::<P>(0), geom_state);
                        self.fevm_state(series, &params.as_view(), fevm_state, geom_state)?;

                        Ok::<(), FEVMError<T>>(())
                    })?;

                Ok::<(), FEVMError<T>>(())
            })?;

        debug!(
            "fevm_initialize_states_only: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Generate a vector observable.
    fn fevm_observe(
        &self,
        scobs: &ScObs<T, ObserVec<T, N>>,
        params: &SVectorView<T, P>,
        fevm_state: &FS,
        geom_state: &GS,
    ) -> Result<ObserVec<T, N>, FEVMError<T>>;

    /// Return internal coordinates of the observation.
    fn fevm_observe_diagnostics(
        &self,
        scobs: &ScObs<T, ObserVec<T, N>>,
        params: &SVectorView<T, P>,
        fevm_state: &FS,
        geom_state: &GS,
    ) -> Result<ObserVec<T, 12>, FEVMError<T>>;

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers. Returns indices of runs that are valid w.r.T.
    /// to the spacecraft observation series.
    fn fevm_simulate(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMEnsbl<T, P, FS, GS>,
        output: &mut DMatrix<ObserVec<T, N>>,
        opt_noise: Option<&mut FEVMNoise<T>>,
    ) -> Result<Vec<bool>, FEVMError<T>>
    where
        T: for<'x> AddAssign<&'x T> + Default + Send + Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

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
            let time_step = *scobs.timestamp() - timer;
            timer = *scobs.timestamp();

            if time_step < T::zero() {
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

                                Ok::<(), FEVMError<T>>(())
                            },
                        )?;

                        Ok::<(), FEVMError<T>>(())
                    })?;
            }

            Ok::<(), FEVMError<T>>(())
        })?;

        if let Some(noise) = opt_noise {
            output
                .par_column_iter_mut()
                .chunks(Self::RCS)
                .enumerate()
                .for_each(|(cdx, mut chunks)| {
                    let mut rng = noise.initialize_rng(29 * cdx as u64, 17);

                    chunks.iter_mut().for_each(|col| {
                        col.iter_mut()
                            .zip(noise.generate_noise(series, &mut rng).iter())
                            .for_each(|(value, noisevec)| *value += noisevec.clone());
                    });
                });

            noise.increment_seed()
        }

        debug!(
            "fevm_simulate: {:2.2}M evaluations in {:.2} sec",
            (series.len() * fevmd.params.ncols()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        let mut valid_indices_flags = vec![false; fevmd.params.ncols()];

        output
            .par_column_iter_mut()
            .zip(valid_indices_flags.par_iter_mut())
            .chunks(Self::RCS)
            .for_each(|mut chunks| {
                chunks.iter_mut().for_each(|(out, flag)| {
                    **flag = zip_eq(out.iter(), series).fold(true, |acc, (out, obs)| {
                        let ss = !out.any_nan() && obs.observation().is_valid();
                        let nn = out.is_nan() && !obs.observation().is_valid();

                        acc & (ss || nn)
                    });
                });
            });

        Ok(valid_indices_flags)
    }

    /// Perform an ensemble forward simulation and return diagnostic values.
    fn fevm_simulate_diagnostics(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMEnsbl<T, P, FS, GS>,
        output: &mut DMatrix<ObserVec<T, 12>>,
    ) -> Result<(), FEVMError<T>>
    where
        T: for<'x> AddAssign<&'x T> + Default + Send + Sync,
    {
        let mut timer = T::zero();

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
            let time_step = *scobs.timestamp() - timer;
            timer = *scobs.timestamp();

            if time_step < T::zero() {
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
                                out[(0, 0)] = self.fevm_observe_diagnostics(
                                    scobs, data, fevm_state, geom_state,
                                )?;

                                Ok::<(), FEVMError<T>>(())
                            },
                        )?;

                        Ok::<(), FEVMError<T>>(())
                    })?;
            }

            Ok::<(), FEVMError<T>>(())
        })?;

        Ok(())
    }

    /// Initialize a model state.
    fn fevm_state(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        params: &SVectorView<T, P>,
        fevm_state: &mut FS,
        geom_state: &mut GS,
    ) -> Result<(), FEVMError<T>>;

    /// Step sizes for finite differences.
    fn fevm_step_sizes(&self) -> [T; P];

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl OcnusProDeF<T, P>;
}

use crate::{
    geom::OcnusCoords,
    obser::{ObserVec, OcnusObser, ScObs, ScObsSeries},
    prodef::{OcnusProDeF, ParticlesND, ProDeFError},
};
use filters::ParticleFilterError;
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, DMatrix, Dyn, Matrix, RealField, SVectorView, Scalar, VecStorage};
use num_traits::Float;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, iter::Sum};
use std::{ops::AddAssign, time::Instant};
use thiserror::Error;

/// A data structure that stores the parameters, states and random seed for a FEVM.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FEVMEnsbl<T, const P: usize, FS, GS>
where
    T: Clone + Scalar,
{
    /// FEVM ensemble parameters.
    pub params: Matrix<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>,

    /// FEVM ensemble states.
    pub fevm_states: Vec<FS>,

    /// Geometry ensemble states.
    pub geom_states: Vec<GS>,

    /// Ensemble member weights.
    pub weights: Vec<T>,
}

impl<T, const P: usize, FS, GS> FEVMEnsbl<T, P, FS, GS>
where
    T: Clone + RealField + Scalar,
    FS: Clone + Default,
    GS: Clone + Default,
    Self: Serialize,
{
    /// Create a new [`FEVMEnsbl`] filled with zeros.
    pub fn new(size: usize) -> Self {
        Self {
            params: Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::zeros(size),
            fevm_states: vec![FS::default(); size],
            geom_states: vec![GS::default(); size],
            weights: vec![T::one() / T::from_usize(size).unwrap(); size],
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
pub enum FEVMError<T> {
    #[error("invalid model parameter {name}={value}")]
    InvalidParameter { name: &'static str, value: T },
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
    NegativeTimeStep(T),
    #[error("observation cannot be a vector with any NaN valus")]
    ObservationNaN,
    #[error("particle filter error")]
    ParticleFilter(#[from] ParticleFilterError<T>),
    #[error("stats error")]
    Stats(#[from] ProDeFError<T>),
}

/// An empty FEVM state.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct FEVMNullState {}
