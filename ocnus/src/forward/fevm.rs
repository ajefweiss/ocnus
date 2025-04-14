use std::time::Instant;

use crate::{
    OcnusError, fXX,
    forward::{FSM, FSMError},
    obser::{ObserVec, ScObs, ScObsSeries},
    prodef::{OcnusProDeF, ParticlesND, ProDeFError},
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, DMatrixViewMut, Dim, Dyn, Matrix, SVectorView, SVectorViewMut, VecStorage};
use rand::{Rng, SeedableRng};
use rand_distr::uniform::SampleUniform;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The trait that must be implemented for any forward ensemble vector model (FEVM) with an
/// N-dimensional vector observable and a forward simulation model state type of `FMST`.
pub trait FEVM<T, const P: usize, const N: usize, FMST, CSST>:
    FSM<T, ObserVec<T, N>, P, FMST, CSST>
where
    T: fXX,
    FMST: Send,
    CSST: Send,
    Self: Sync,
{
    /// The base rayon chunk size that is used for any parallel iterators.
    ///
    /// Operations may use multiples of this value.
    const RCS: usize;

    /// Initialize the model parameters, the coordinate system and forward model states.
    fn fevm_initialize<D>(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        ensbl: &mut FEVMEnsbl<T, P, FMST, CSST>,
        opt_pdf: Option<&D>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        for<'a> &'a D: OcnusProDeF<T, P>,
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
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            ensbl.params_array.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Resample the model parameters, and re-initialize the coordinate system and forward model states.
    fn fevm_resample(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        ensbl: &mut FEVMEnsbl<T, P, FMST, CSST>,
        pdf: &ParticlesND<T, P>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        T: SampleUniform,
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
            ensbl.params.ncols() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }
}

/// A data structure that stores the model parameters and states of a [`FEVM`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FEVMEnsbl<T, const P: usize, FMST, CSST>
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
