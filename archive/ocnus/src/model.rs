use crate::{
    coords::OcnusCoords,
    // obser::{ObserVec, OcnusNoise, OcnusObser, ScObs, ScObsSeries},
    stats::ParticleDensity,
};
// use itertools::zip_eq;
// use log::debug;
use nalgebra::RealField;
// use rand::{Rng, SeedableRng};
// use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
// use rand_xoshiro::Xoshiro256PlusPlus;
// use rayon::prelude::*;
// use std::{ops::AddAssign, time::Instant};
use thiserror::Error;

/// A trait that is shared by all models within the **ocnus** framework.
pub trait OcnusModel<T, const P: usize, FMST, CSST>: OcnusCoords<T, P, CSST>
where
    T: Copy + RealField,
    StandardNormal: Distribution<T>,
{
    /// The base rayon chunk size that is used for any parallel iterators.
    ///
    /// Operations may use multiples of this value.
    const RCS: usize;

    /// Evolve a model state forward in time.
    fn forward(
        &self,
        time_step: T,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    );

    /// Initialize the model parameters, the coordinate system and forward model states.
    fn initialize<D>(
        &self,
        series: &ScObsSeries<T, O>,
        params: &mut SVectorViewMut<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        opt_pdf: Option<&D>,
        rng: &mut impl Rng,
    ) -> Result<(), OcnusError<T>>
    where
        for<'x> &'x D: Density<T, P>,
    {
        self.initialize_params(params, opt_pdf, rng)?;
        self.initialize_states(series, &params.as_view(), fm_state, cs_state)?;

        Ok(())
    }

    /// Initialize the model parameters, the coordinate system and forward model states for an ensemble.
    fn initialize_ensbl<D>(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut OcnusEnsbl<T, P, FMST, CSST>,
        opt_pdf: Option<&D>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        for<'x> &'x D: Density<T, P>,
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
                        self.initialize(series, params, fm_state, cs_state, opt_pdf, &mut rng)?;

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
    fn initialize_params(
        &self,
        params: &mut SVectorViewMut<T, P>,
        opt_pdf: Option<impl Density<T, P>>,
        rng: &mut impl Rng,
    ) -> Result<(), StatsError<T>> {
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
    fn initialize_params_ensbl<D>(
        &self,
        ensbl: &mut OcnusEnsbl<T, P, FMST, CSST>,
        opt_pdf: Option<&D>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        for<'x> &'x D: Density<T, P>,
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
                    self.initialize_params(params, opt_pdf, &mut rng)?;

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
    fn initialize_states(
        &self,
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), OcnusError<T>>;

    /// Initialize the the coordinate system and forward model states for an ensemble.
    fn initialize_states_ensbl(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut OcnusEnsbl<T, P, FMST, CSST>,
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
                        self.initialize_states(series, params, fm_state, cs_state)?;

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

    /// Generate an observable, dependent on the spacecraft observation configuration.
    fn observe(
        &self,
        scobs: &ScObs<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<O, OcnusError<T>>;

    /// Return internal coordinates and the basis vectors at the location of the observation.
    fn observe_ics_plus_basis(
        &self,
        scobs: &ScObs<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 12>, OcnusError<T>>;

    /// Resample the model parameters, and re-initialize the coordinate system and forward model states for an ensemble.
    fn resample_ensbl(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut OcnusEnsbl<T, P, FMST, CSST>,
        pdf: &ParticlesND<T, P>,
        rseed: u64,
    ) -> Result<(), OcnusError<T>>
    where
        T: SampleUniform,
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
                        params.set_column(0, &pdf.resample(&mut rng));

                        self.initialize_states(
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
    fn simulate<RStride: Dim, CStride: Dim>(
        &self,
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<O, RStride, CStride>,
    ) -> Result<bool, OcnusError<T>> {
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != 1) {
            return Err(ModelError::InvalidOutputShape {
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
                return Err(ModelError::NegativeTimeStep(time_step).into());
            } else {
                self.forward(time_step, params, fm_state, cs_state)?;
                out_row[(0, 0)] = self.observe(scobs, params, fm_state, cs_state)?;
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
    fn simulate_ensbl<N>(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut OcnusEnsbl<T, P, FMST, CSST>,
        out_array: &mut DMatrixViewMut<O>,
        opt_noise: Option<&mut N>,
    ) -> Option<Vec<bool>, OcnusError<T>>
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
            return Err(ModelError::InvalidOutputShape {
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
                return Err(ModelError::NegativeTimeStep(time_step).into());
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
                                self.forward(time_step, params, fm_state, cs_state)?;

                                out[(0, 0)] = self.observe(scobs, params, fm_state, cs_state)?;

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
            "simulate_ensbl: {:2.2}M evaluations in {:.2} sec",
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
    fn simulate_ics_plus_basis(
        &self,
        series: &ScObsSeries<T, O>,
        params: &SVectorView<T, P>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<ObserVec<T, 12>>,
    ) -> Result<(), OcnusError<T>> {
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != 1) {
            return Err(ModelError::InvalidOutputShape {
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
                return Err(ModelError::NegativeTimeStep(time_step).into());
            } else {
                self.forward(time_step, params, fm_state, cs_state)?;
                out[(0, 0)] = self.observe_ics_plus_basis(scobs, params, fm_state, cs_state)?;
            }

            Ok::<(), OcnusError<T>>(())
        })?;

        Ok(())
    }

    /// Perform an ensemble forward simulation and return the internal coordinates and basis
    /// vectors for  the given spacecraft observers.
    fn simulate_ics_plus_basis_ensbl(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &mut OcnusEnsbl<T, P, FMST, CSST>,
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
            return Err(ModelError::InvalidOutputShape {
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
                return Err(ModelError::NegativeTimeStep(time_step).into());
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
                                self.forward(time_step, params, fm_state, cs_state)?;

                                out[(0, 0)] =
                                    self.observe_ics_plus_basis(scobs, params, fm_state, cs_state)?;

                                Ok::<(), OcnusError<T>>(())
                            },
                        )?;

                        Ok::<(), OcnusError<T>>(())
                    })?;
            }

            Ok::<(), OcnusError<T>>(())
        })?;

        debug!(
            "simulate_ics_plus_basis_ensbl: {:2.2}M evaluations in {:.2} sec",
            (series.len() * ensbl.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );
        Ok(())
    }

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl Density<T, P>;

    /// Step sizes for finite differences.
    fn param_step_sizes(&self) -> [T; P];
}

/// Error types associated with the [`OcnusModel`] trait.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ModelError<T> {
    #[error("observation is unexpecteedly not valid")]
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
    #[error("attempted to simulate is backwards in time (dt=-{0:.2}sec)")]
    NegativeTimeStep(T),
}
