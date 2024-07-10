//! Forward simulation models.
//!
//! The [`OcnusFSM`] trait represent a forward simulation with a generic observable type.
//! A [`OcnusFSM`] must be built on-top of a coordiante system / geometry, i.e. it must also
//! implement [`OcnusCoords`]. Any OcnusFSM is also by extension associated with the coordinate
//! system state type `CSST` of the underlying coordinate system, and is additonally associated
//! with its own forward simulation model state type `FMST`.
//!
//! For models that are sufficiently fast, this trait also provides ensemble methods with a
//! corresponding data structure [`FSMEnsbl`] to hold an ensemble.

pub mod filters;
mod fisher;
mod models;

pub use fisher::FisherInformation;
pub use models::{CCLFFModel, CCUTModel, COREModel, COREState, ECHModel};

use crate::{
    OcnusError,
    coords::OcnusCoords,
    fXX,
    obser::{ObserVec, OcnusNoise, OcnusObser, ScObs, ScObsSeries},
    prodef::{OcnusProDeF, ParticlesND, ProDeFError},
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{
    Const, DMatrixViewMut, Dim, Dyn, Matrix, SVectorView, SVectorViewMut, Scalar, VecStorage,
};
use rand::{Rng, SeedableRng};
use rand_distr::uniform::SampleUniform;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, ops::AddAssign, time::Instant};
use thiserror::Error;

/// Errors associated with forward simulation models.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum FSMError<T> {
    #[error("observation is invalid")]
    InvalidObservation,
    #[error(
        "invalid output array shape: found {output_rows} x {output_cols} 
        but expected {expected_rows} x {expected_cols}"
    )]
    InvalidOutputShape {
        expected_cols: usize,
        expected_rows: usize,
        output_cols: usize,
        output_rows: usize,
    },
    #[error("forward simulation is backwards in time (dt=-{0:.2}sec)")]
    NegativeTimeStep(T),
}

/// The trait that must be implemented for any forward simulation model with an observable of type
/// `O` and a forward simulation model state type of `FMST`.
pub trait OcnusFSM<T, O, const P: usize, FMST, CSST>: OcnusCoords<T, P, CSST>
where
    T: fXX,
    O: OcnusObser,
{
    /// The base rayon chunk size that is used for any parallel iterators.
    ///
    /// Operations may use multiples of this value.
    const RCS: usize;

    /// Evolve a model state forward in time.
    fn fsm_forward(
        &self,
        time_step: T,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), FSMError<T>>;

    /// Initialize the model parameters, the coordinate system and forward model states.
    fn fsm_initialize<D>(
        &self,
        series: &ScObsSeries<T, O>,
        params: &mut SVectorViewMut<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        opt_pdf: Option<&D>,
        rng: &mut impl Rng,
    ) -> Result<(), OcnusError<T>>
    where
        for<'a> &'a D: OcnusProDeF<T, P>,
    {
        self.fsm_initialize_params(params, opt_pdf, rng)?;
        Self::fsm_initialize_states(series, &params.as_view(), fm_state, cs_state)?;

        Ok(())
    }

    /// Initialize the model parameters, the coordinate system and forward model states for an ensemble.
    fn fsm_initialize_ensbl<D>(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut FSMEnsbl<T, P, FMST, CSST>,
        opt_pdf: Option<&D>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        for<'a> &'a D: OcnusProDeF<T, P>,
        FMST: Send,
        CSST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        ensbl
            .params_array
            .par_column_iter_mut()
            .zip(ensbl.fm_states.par_iter_mut())
            .zip(ensbl.cs_states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

                chunks
                    .iter_mut()
                    .try_for_each(|((params, fm_state), cs_state)| {
                        self.fsm_initialize(series, params, fm_state, cs_state, opt_pdf, &mut rng)?;

                        Ok::<(), OcnusError<T>>(())
                    })?;

                Ok::<(), OcnusError<T>>(())
            })?;

        debug!(
            "fevm_initialize_ensbl: {:2.2}M evaluations in {:.2} sec",
            ensbl.params_array.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize the model parameters
    fn fsm_initialize_params(
        &self,
        params: &mut SVectorViewMut<T, P>,
        opt_pdf: Option<impl OcnusProDeF<T, P>>,
        rng: &mut impl Rng,
    ) -> Result<(), ProDeFError<T>> {
        params.set_column(
            0,
            &match opt_pdf.as_ref() {
                Some(pdf) => pdf.draw_sample(rng),
                None => self.model_prior().draw_sample(rng),
            }?,
        );

        Ok(())
    }

    /// Initialize the model parameters for an ensemble.
    fn fsm_initialize_params_ensbl<D>(
        &self,
        ensbl: &mut FSMEnsbl<T, P, FMST, CSST>,
        opt_pdf: Option<&D>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        for<'a> &'a D: OcnusProDeF<T, P>,
        FMST: Send,
        CSST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        ensbl
            .params_array
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|params| {
                    self.fsm_initialize_params(params, opt_pdf, &mut rng)?;

                    Ok::<(), OcnusError<T>>(())
                })?;

                Ok::<(), OcnusError<T>>(())
            })?;

        debug!(
            "fevm_initialize_params_ensbl: {:2.2}M evaluations in {:.2} sec",
            ensbl.params_array.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize the coordinate system and forward model states.
    fn fsm_initialize_states(
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), OcnusError<T>>;

    /// Initialize the the coordinate system and forward model states for an ensemble.
    fn fsm_initialize_states_ensbl(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut FSMEnsbl<T, P, FMST, CSST>,
    ) -> Result<(), OcnusError<T>>
    where
        FMST: Send,
        CSST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        ensbl
            .params_array
            .par_column_iter()
            .zip(ensbl.fm_states.par_iter_mut())
            .zip(ensbl.cs_states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunks| {
                chunks
                    .iter_mut()
                    .try_for_each(|((params, fm_state), cs_state)| {
                        Self::fsm_initialize_states(series, params, fm_state, cs_state)?;

                        Ok::<(), OcnusError<T>>(())
                    })?;

                Ok::<(), OcnusError<T>>(())
            })?;

        debug!(
            "fevm_initialize_ensbl: {:2.2}M evaluations in {:.2} sec",
            ensbl.params_array.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Generate a vector observable.
    fn fsm_observe(
        &self,
        scobs: &ScObs<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<O, OcnusError<T>>;

    /// Return internal coordinates and the basis vectors at the location of the observation.
    fn fsm_observe_ics_plus_basis(
        &self,
        scobs: &ScObs<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 12>, OcnusError<T>>;

    /// Resample the model parameters, and re-initialize the coordinate system and forward model states for an ensemble.
    fn fsm_resample_ensbl(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut FSMEnsbl<T, P, FMST, CSST>,
        pdf: &ParticlesND<T, P>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        T: SampleUniform,
        FMST: Send,
        CSST: Send,
    {
        let start = Instant::now();

        ensbl
            .params_array
            .par_column_iter_mut()
            .zip(ensbl.fm_states.par_iter_mut())
            .zip(ensbl.cs_states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

                chunks
                    .iter_mut()
                    .try_for_each(|((params, fm_state), cs_state)| {
                        params.set_column(0, &pdf.resample(&mut rng)?);

                        Self::fsm_initialize_states(
                            series,
                            &params.fixed_rows::<P>(0),
                            fm_state,
                            cs_state,
                        )?;

                        Ok::<(), OcnusError<T>>(())
                    })?;

                Ok::<(), OcnusError<T>>(())
            })?;

        debug!(
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            ensbl.params_array.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Perform a forward simulation and generate synthetic observables for the given spacecraft
    /// observers. Returns true/false if the observation series is valid w.r.t. the spacecraft
    /// observatin series.
    fn fsm_simulate<RStride: Dim, CStride: Dim>(
        &self,
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<O, RStride, CStride>,
    ) -> Result<bool, OcnusError<T>> {
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != 1) {
            return Err(FSMError::InvalidOutputShape {
                expected_cols: 1,
                expected_rows: series.len(),
                output_cols: out_array.ncols(),
                output_rows: out_array.nrows(),
            }
            .into());
        }

        zip_eq(series, out_array.row_iter_mut()).try_for_each(|(scobs, mut out_row)| {
            // Compute time step to next observation.
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

            if time_step < T::zero() {
                return Err(FSMError::NegativeTimeStep(time_step).into());
            } else {
                self.fsm_forward(time_step, params, fm_state, cs_state)?;
                out_row[(0, 0)] = self.fsm_observe(scobs, params, fm_state, cs_state)?;
            }

            Ok::<(), OcnusError<T>>(())
        })?;

        Ok(series
            .into_iter()
            .zip(out_array.column(0).iter())
            .fold(true, |acc, (scobs, obs)| {
                acc && !(scobs.get_observation().is_valid() ^ obs.is_valid())
            }))
    }

    /// Perform an ensemble forward simulation and generate synthetic observables for the given
    /// spacecraft observers. Returns the indices of ensemble members that are valid w.r.t.
    /// to the spacecraft observation series.
    fn fsm_simulate_ensbl<N>(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut FSMEnsbl<T, P, FMST, CSST>,
        out_array: &mut DMatrixViewMut<O>,
        opt_noise: Option<&mut N>,
    ) -> Result<Vec<bool>, OcnusError<T>>
    where
        O: AddAssign + Scalar,
        N: OcnusNoise<T, O> + Sync,
        FMST: Send,
        CSST: Send,
        Self: Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != ensbl.len()) {
            return Err(FSMError::InvalidOutputShape {
                expected_cols: ensbl.len(),
                expected_rows: series.len(),
                output_cols: out_array.ncols(),
                output_rows: out_array.nrows(),
            }
            .into());
        }

        zip_eq(series, out_array.row_iter_mut()).try_for_each(|(scobs, mut out_row)| {
            // Compute time step to next observation.
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

            if time_step < T::zero() {
                return Err(FSMError::NegativeTimeStep(time_step).into());
            } else {
                ensbl
                    .params_array
                    .par_column_iter()
                    .zip(ensbl.fm_states.par_iter_mut())
                    .zip(ensbl.cs_states.par_iter_mut())
                    .zip(out_row.par_column_iter_mut())
                    .chunks(Self::RCS)
                    .try_for_each(|mut chunks| {
                        chunks.iter_mut().try_for_each(
                            |(((params, fm_state), cs_state), out)| {
                                self.fsm_forward(time_step, params, fm_state, cs_state)?;

                                out[(0, 0)] =
                                    self.fsm_observe(scobs, params, fm_state, cs_state)?;

                                Ok::<(), OcnusError<T>>(())
                            },
                        )?;

                        Ok::<(), OcnusError<T>>(())
                    })?;
            }

            Ok::<(), OcnusError<T>>(())
        })?;

        if let Some(noise) = opt_noise {
            out_array
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

            noise.increment_random_seed()
        }

        debug!(
            "fsm_simulate_ensbl: {:2.2}M evaluations in {:.2} sec",
            (series.len() * ensbl.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        let mut valid_indices_flags = vec![false; ensbl.len()];

        out_array
            .par_column_iter_mut()
            .zip(valid_indices_flags.par_iter_mut())
            .chunks(Self::RCS)
            .for_each(|mut chunks| {
                chunks.iter_mut().for_each(|(out, flag)| {
                    **flag = zip_eq(out.iter(), series).fold(true, |acc, (out, obs)| {
                        acc & !(out.is_valid() ^ obs.get_observation().is_valid())
                    });
                });
            });

        Ok(valid_indices_flags)
    }

    /// Perform a forward simulation and return the internal coordinates and basis vectors for
    /// the given spacecraft observers.
    fn fsm_simulate_ics_plus_basis(
        &self,
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<ObserVec<T, 12>>,
    ) -> Result<(), OcnusError<T>> {
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != 1) {
            return Err(FSMError::InvalidOutputShape {
                expected_cols: 1,
                expected_rows: series.len(),
                output_cols: out_array.ncols(),
                output_rows: out_array.nrows(),
            }
            .into());
        }

        zip_eq(series, out_array.row_iter_mut()).try_for_each(|(scobs, mut out)| {
            // Compute time step to next observation.
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

            if time_step < T::zero() {
                return Err(FSMError::NegativeTimeStep(time_step).into());
            } else {
                self.fsm_forward(time_step, params, fm_state, cs_state)?;
                out[(0, 0)] = self.fsm_observe_ics_plus_basis(scobs, params, fm_state, cs_state)?;
            }

            Ok::<(), OcnusError<T>>(())
        })?;

        Ok(())
    }

    /// Perform an ensemble forward simulation and return the internal coordinates and basis
    /// vectors for  the given spacecraft observers.
    fn fsm_simulate_ics_plus_basis_ensbl(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut FSMEnsbl<T, P, FMST, CSST>,
        out_array: &mut DMatrixViewMut<ObserVec<T, 12>>,
    ) -> Result<(), OcnusError<T>>
    where
        O: Scalar,
        FMST: Send,
        CSST: Send,
        Self: Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != ensbl.len()) {
            return Err(FSMError::InvalidOutputShape {
                expected_cols: ensbl.len(),
                expected_rows: series.len(),
                output_cols: out_array.ncols(),
                output_rows: out_array.nrows(),
            }
            .into());
        }

        zip_eq(series, out_array.row_iter_mut()).try_for_each(|(scobs, mut out_col)| {
            // Compute time step to next observation.
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

            if time_step < T::zero() {
                return Err(FSMError::NegativeTimeStep(time_step).into());
            } else {
                ensbl
                    .params_array
                    .par_column_iter()
                    .zip(ensbl.fm_states.par_iter_mut())
                    .zip(ensbl.cs_states.par_iter_mut())
                    .zip(out_col.par_column_iter_mut())
                    .chunks(Self::RCS)
                    .try_for_each(|mut chunks| {
                        chunks.iter_mut().try_for_each(
                            |(((params, fm_state), cs_state), out)| {
                                self.fsm_forward(time_step, params, fm_state, cs_state)?;

                                out[(0, 0)] = self.fsm_observe_ics_plus_basis(
                                    scobs, params, fm_state, cs_state,
                                )?;

                                Ok::<(), OcnusError<T>>(())
                            },
                        )?;

                        Ok::<(), OcnusError<T>>(())
                    })?;
            }

            Ok::<(), OcnusError<T>>(())
        })?;

        debug!(
            "fsm_simulate_ics_plus_basis_ensbl: {:2.2}M evaluations in {:.2} sec",
            (series.len() * ensbl.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );
        Ok(())
    }

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl OcnusProDeF<T, P>;

    /// Step sizes for finite differences.
    fn param_step_sizes(&self) -> [T; P];
}

/// A data structure that stores an ensemble of model parameters and states for a [`OcnusFSM`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FSMEnsbl<T, const P: usize, FMST, CSST>
where
    T: fXX,
{
    /// Ensemble model parameters.
    pub params_array: Matrix<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>,

    /// Forward model states
    pub fm_states: Vec<FMST>,

    /// Coordinate system states.
    pub cs_states: Vec<CSST>,

    /// Ensemble weights.
    pub weights: Vec<T>,
}

impl<T, const P: usize, FMST, CSST> FSMEnsbl<T, P, FMST, CSST>
where
    T: fXX,
{
    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of members in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.params_array.ncols()
    }

    /// Create a new [`FSMEnsbl`] filled with zeros.
    pub fn new(size: usize) -> Self
    where
        FMST: Default + Clone,
        CSST: Default + Clone,
    {
        Self {
            params_array: Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::zeros(size),
            fm_states: vec![FMST::default(); size],
            cs_states: vec![CSST::default(); size],
            weights: vec![T::one() / T::from_usize(size).unwrap(); size],
        }
    }

    /// Serialize ensemble to a JSON file.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}
