//! Forward simulation models and specialized sub-types.
//!
//! A forward simulation model (FSM) is a forward simulation with a generic observable type.
//! The ensemble vector counterparts of FSMs, are forward ensemble vector model (FEVMs) which are
//! numerically efficient enough to be run as large ensembles and output observables of the
//! [`ObserVec`] type. A [`FSM`] must be built on-top of a coordiante system / geometry, i.e. it
//! must also implement [`OcnusCoords`]. Any FSM is also by extension associated with the
//! coordinate system state type `CSST` of the underlying coordinate system, and is additonally
//! associated with its own forward simulation model state type `FMST`.

mod fevm;

pub use fevm::*;

use crate::{
    OcnusError,
    coords::OcnusCoords,
    fXX,
    obser::{ObserVec, OcnusObser, ScObs, ScObsSeries},
    prodef::{OcnusProDeF, ProDeFError},
};
use itertools::zip_eq;
use nalgebra::{DMatrixViewMut, Dim, SVectorView, SVectorViewMut};
use rand::Rng;
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
pub trait FSM<T, O, const P: usize, FMST, CSST>: OcnusCoords<T, P, CSST>
where
    T: fXX,
    O: OcnusObser,
{
    /// Evolve a model state forward in time.
    fn fsm_forward(
        &self,
        time_step: T,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), FSMError<T>>;

    /// Initialize the model parameters, the coordinate system and forward model states.
    fn fsm_initialize(
        &self,
        series: &ScObsSeries<T, O>,
        params: &mut SVectorViewMut<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        opt_pdf: Option<impl OcnusProDeF<T, P>>,
        rng: &mut impl Rng,
    ) -> Result<(), OcnusError<T>> {
        self.fsm_initialize_params(params, opt_pdf, rng)?;
        Self::fsm_initialize_states(series, &params.as_view(), fm_state, cs_state)?;

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

    /// Initialize the coordinate system and forward model states.
    fn fsm_initialize_states(
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), FSMError<T>>;

    /// Generate a vector observable.
    fn fsm_observe(
        &self,
        scobs: &ScObs<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<O, FSMError<T>>;

    /// Return internal coordinates and the basis vectors at the location of the observation.
    fn fsm_observe_ics_plus_basis(
        &self,
        scobs: &ScObs<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 12>, FSMError<T>>;

    /// Perform a forward simulation and generate synthetic observables for the given spacecraft
    /// observers. Returns true/false if the observation series is valid w.r.t. the spacecraft
    /// observation.
    fn fsm_simulate<RStride: Dim, CStride: Dim>(
        &self,
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<O, RStride, CStride>,
    ) -> Result<bool, FSMError<T>> {
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != 1) {
            return Err(FSMError::InvalidOutputShape {
                expected_cols: 1,
                expected_rows: series.len(),
                output_cols: out_array.ncols(),
                output_rows: out_array.nrows(),
            });
        }

        zip_eq(series, out_array.row_iter_mut()).try_for_each(|(scobs, mut out)| {
            // Compute time step to next observation.
            let time_step = *scobs.timestamp() - timer;
            timer = *scobs.timestamp();

            if time_step < T::zero() {
                return Err(FSMError::NegativeTimeStep(time_step));
            } else {
                self.fsm_forward(time_step, params, fm_state, cs_state)?;
                out[(0, 0)] = self.fsm_observe(scobs, params, fm_state, cs_state)?;
            }

            Ok(())
        })?;

        Ok(series
            .into_iter()
            .zip(out_array.column(0).iter())
            .fold(true, |acc, (scobs, obs)| {
                acc && !(scobs.observation().is_valid() ^ obs.is_valid())
            }))
    }

    /// Perform a forward simulation and return the internal coordinates and basis vectors for
    /// the given spacecraft observers.
    fn fsm_simulate_ics_plus_basis<RStride: Dim, CStride: Dim>(
        &self,
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<ObserVec<T, 12>, RStride, CStride>,
    ) -> Result<(), FSMError<T>> {
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != 1) {
            return Err(FSMError::InvalidOutputShape {
                expected_cols: 1,
                expected_rows: series.len(),
                output_cols: out_array.ncols(),
                output_rows: out_array.nrows(),
            });
        }

        zip_eq(series, out_array.row_iter_mut()).try_for_each(|(scobs, mut out)| {
            // Compute time step to next observation.
            let time_step = *scobs.timestamp() - timer;
            timer = *scobs.timestamp();

            if time_step < T::zero() {
                return Err(FSMError::NegativeTimeStep(time_step));
            } else {
                self.fsm_forward(time_step, params, fm_state, cs_state)?;
                out[(0, 0)] = self.fsm_observe_ics_plus_basis(scobs, params, fm_state, cs_state)?;
            }

            Ok(())
        })?;

        Ok(())
    }

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl OcnusProDeF<T, P>;

    /// Step sizes for finite differences.
    fn parameter_step_sizes(&self) -> [T; P];
}
