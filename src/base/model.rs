use crate::{
    base::{OcnusEnsbl, ScObs, ScObsSeries},
    coords::OcnusCoords,
    obser::{ObserVec, OcnusNoise, OcnusObser},
    stats::{Density, DensityRange, ParticleDensity},
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{DMatrixViewMut, RealField, SVector, SVectorView, SVectorViewMut, Scalar, Vector3};
use num_traits::AsPrimitive;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{AddAssign, Mul, Sub},
    time::Instant,
};
use thiserror::Error;

/// Error types associated with the [`OcnusModel`] trait.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum OcnusModelError<T> {
    #[error("failed to convert external to internal coordinates")]
    CoordinateTransform(Vector3<T>),
    #[error(
        "invalid output array shape: found {output_rows} x {output_cols}
        but expected {expected_rows} x {expected_cols}"
    )]
    OutputShape {
        expected_cols: usize,
        expected_rows: usize,
        output_cols: usize,
        output_rows: usize,
    },
    #[error("output contains NaN values")]
    OutputNaN,
    #[error("attempted to simulate is backwards in time (dt=-{0:.2}sec)")]
    NegativeTimeStep(T),
    #[error("failed to sample model parameters")]
    Sampling,
}

/// A trait that is shared by all models within the **ocnus** framework.
pub trait OcnusModel<T, const D: usize, FMST, CSST>:
    OcnusCoords<T, D, CSST> + for<'x> Deserialize<'x> + Serialize
where
    T: Copy + RealField + SampleUniform,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
{
    /// The base rayon chunk size that is used for any parallel iterators.
    ///
    /// Operations may use multiples of this value.
    const RCS: usize;

    /// Evolve a model state forward in time.
    fn forward(
        &self,
        time_step: T,
        params: &SVectorView<T, D>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), OcnusModelError<T>>;

    /// Returns the valid parameter range.
    fn get_range(&self) -> SVector<DensityRange<T>, D>;

    /// Initialize the model parameters, the coordinate system and forward model states.
    fn initialize<const A: usize, P>(
        &self,
        params: &mut SVectorViewMut<T, D>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        opt_pdf: Option<&P>,
        rng: &mut impl Rng,
    ) -> Result<(), OcnusModelError<T>>
    where
        for<'x> &'x P: Density<T, D>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        self.initialize_params::<A>(params, opt_pdf, rng)?;
        self.initialize_states(&params.as_view(), fm_state, cs_state)?;

        Ok(())
    }

    /// Initialize the model parameters, the coordinate system and forward model states for an ensemble.
    fn initialize_ensbl<const A: usize, P>(
        &self,
        ensbl: &mut OcnusEnsbl<T, D, FMST, CSST>,
        opt_pdf: Option<&P>,
        rseed: u64,
    ) -> Result<(), OcnusModelError<T>>
    where
        for<'x> &'x P: Density<T, D>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let start = Instant::now();

        ensbl
            .ptpdf
            .particles_mut()
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
                        self.initialize::<A, P>(params, fm_state, cs_state, opt_pdf, &mut rng)?;

                        Ok::<(), OcnusModelError<T>>(())
                    })?;

                Ok::<(), OcnusModelError<T>>(())
            })?;

        debug!(
            "fevm_initialize_ensbl: {:2.2}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize the model parameters
    fn initialize_params<const A: usize>(
        &self,
        params: &mut SVectorViewMut<T, D>,
        opt_pdf: Option<impl Density<T, D>>,
        rng: &mut impl Rng,
    ) -> Result<(), OcnusModelError<T>>
    where
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let opt_col = match opt_pdf.as_ref() {
            Some(pdf) => pdf.draw_sample::<A>(rng),
            None => self.model_prior().draw_sample::<A>(rng),
        };

        if let Some(col) = opt_col {
            params.set_column(0, &col);

            Ok(())
        } else {
            Err(OcnusModelError::Sampling)
        }
    }

    /// Initialize the model parameters for an ensemble.
    fn initialize_params_ensbl<const A: usize, P>(
        &self,
        ensbl: &mut OcnusEnsbl<T, D, FMST, CSST>,
        opt_pdf: Option<&P>,
        rseed: u64,
    ) -> Result<(), OcnusModelError<T>>
    where
        for<'x> &'x P: Density<T, D>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let start = Instant::now();

        ensbl
            .ptpdf
            .particles_mut()
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|params| {
                    self.initialize_params::<A>(params, opt_pdf, &mut rng)?;

                    Ok::<(), OcnusModelError<T>>(())
                })?;

                Ok::<(), OcnusModelError<T>>(())
            })?;

        debug!(
            "fevm_initialize_params_ensbl: {:2.2}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize the coordinate system and forward model states.
    fn initialize_states(
        &self,
        params: &SVectorView<T, D>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), OcnusModelError<T>>;

    /// Initialize the the coordinate system and forward model states for an ensemble.
    fn initialize_states_ensbl(
        &self,
        ensbl: &mut OcnusEnsbl<T, D, FMST, CSST>,
    ) -> Result<(), OcnusModelError<T>> {
        let start = Instant::now();

        ensbl
            .ptpdf
            .particles()
            .par_column_iter()
            .zip(ensbl.fm_states.par_iter_mut())
            .zip(ensbl.cs_states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunks| {
                chunks
                    .iter_mut()
                    .try_for_each(|((params, fm_state), cs_state)| {
                        self.initialize_states(params, fm_state, cs_state)?;

                        Ok::<(), OcnusModelError<T>>(())
                    })?;

                Ok::<(), OcnusModelError<T>>(())
            })?;

        debug!(
            "fevm_initialize_ensbl: {:2.2}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl Density<T, D>;

    /// Return internal coordinates and the basis vectors at the location of the observation.
    fn observe_ics_basis(
        &self,
        scobs: &ScObs<T>,
        params: &SVectorView<T, D>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 12>, OcnusModelError<T>>;

    /// Resample the model parameters, and re-initialize the coordinate system and forward model states for an ensemble.
    fn resample_ensbl(
        &self,
        ensbl: &mut OcnusEnsbl<T, D, FMST, CSST>,
        ptpdf: &ParticleDensity<T, D>,
        rseed: u64,
    ) -> Result<(), OcnusModelError<T>>
    where
        T: for<'x> Mul<&'x T, Output = T>
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        usize: AsPrimitive<T>,
    {
        let start = Instant::now();

        ensbl
            .ptpdf
            .particles_mut()
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
                        params.set_column(0, &ptpdf.resample(&mut rng));

                        self.initialize_states(&params.as_view(), fm_state, cs_state)?;

                        Ok::<(), OcnusModelError<T>>(())
                    })?;

                Ok::<(), OcnusModelError<T>>(())
            })?;

        debug!(
            "fevm_resample: {:2.2}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Perform a forward simulation and generate synthetic observables `OT` for the
    /// given spacecraft observers using a generating function `OF`.
    fn simulate<OT, OF>(
        &self,
        series: &ScObsSeries<T>,
        params: &SVectorView<T, D>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        obs_func: &OF,
        obs_array: &mut DMatrixViewMut<OT>,
    ) -> Result<(), OcnusModelError<T>>
    where
        OT: OcnusObser,
        OF: Fn(&Self, &ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>,
    {
        let mut timer = T::zero();

        if (series.len() != obs_array.nrows()) || (obs_array.ncols() != 1) {
            return Err(OcnusModelError::OutputShape {
                expected_cols: 1,
                expected_rows: series.len(),
                output_cols: obs_array.ncols(),
                output_rows: obs_array.nrows(),
            });
        }

        zip_eq(series, obs_array.row_iter_mut()).try_for_each(|(scobs, mut obs_row)| {
            // Compute time step to next observation.
            let time_step = *scobs.timestamp() - timer;
            timer = *scobs.timestamp();

            if time_step < T::zero() {
                return Err(OcnusModelError::NegativeTimeStep(time_step));
            } else {
                self.forward(time_step, params, fm_state, cs_state)?;

                obs_row[(0, 0)] = obs_func(
                    self,
                    scobs,
                    &SVector::<T, D>::from_iterator(params.iter().cloned()),
                    fm_state,
                    cs_state,
                )?;
            }

            Ok::<(), OcnusModelError<T>>(())
        })?;

        Ok(())
    }

    /// Perform an ensemble forward simulation and generate synthetic observables `OT` for the
    /// given spacecraft observers using a generating function `OF` and noise model `MN`.
    fn simulate_ensbl<OT, OF, NM>(
        &self,
        series: &ScObsSeries<T>,
        ensbl: &mut OcnusEnsbl<T, D, FMST, CSST>,
        obs_func: &OF,
        obs_array: &mut DMatrixViewMut<OT>,
        opt_noise: Option<&mut NM>,
    ) -> Result<(), OcnusModelError<T>>
    where
        OT: AddAssign + OcnusObser + Scalar,
        OF: Fn(&Self, &ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>
            + Sync,
        NM: OcnusNoise<T, OT> + Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        if (series.len() != obs_array.nrows()) || (obs_array.ncols() != ensbl.len()) {
            return Err(OcnusModelError::OutputShape {
                expected_cols: ensbl.len(),
                expected_rows: series.len(),
                output_cols: obs_array.ncols(),
                output_rows: obs_array.nrows(),
            });
        }

        zip_eq(series, obs_array.row_iter_mut()).try_for_each(|(scobs, mut obs_row)| {
            // Compute time step to next observation.
            let time_step = *scobs.timestamp() - timer;
            timer = *scobs.timestamp();

            if time_step < T::zero() {
                return Err(OcnusModelError::NegativeTimeStep(time_step));
            } else {
                ensbl
                    .ptpdf
                    .particles()
                    .par_column_iter()
                    .zip(ensbl.fm_states.par_iter_mut())
                    .zip(ensbl.cs_states.par_iter_mut())
                    .zip(obs_row.par_column_iter_mut())
                    .chunks(Self::RCS)
                    .try_for_each(|mut chunks| {
                        chunks.iter_mut().try_for_each(
                            |(((params, fm_state), cs_state), obs)| {
                                self.forward(time_step, params, fm_state, cs_state)?;

                                obs[(0, 0)] = obs_func(
                                    self,
                                    scobs,
                                    &SVector::<T, D>::from_iterator(params.iter().cloned()),
                                    fm_state,
                                    cs_state,
                                )?;

                                Ok::<(), OcnusModelError<T>>(())
                            },
                        )?;

                        Ok::<(), OcnusModelError<T>>(())
                    })?;
            }

            Ok::<(), OcnusModelError<T>>(())
        })?;

        if let Some(noise) = opt_noise {
            obs_array
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

        Ok(())
    }

    /// Perform a forward simulation and return the internal coordinates and basis vectors for
    /// the given spacecraft observers.
    fn simulate_ics_basis(
        &self,
        series: &ScObsSeries<T>,
        params: &SVectorView<T, D>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<ObserVec<T, 12>>,
    ) -> Result<(), OcnusModelError<T>> {
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != 1) {
            return Err(OcnusModelError::OutputShape {
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
                return Err(OcnusModelError::NegativeTimeStep(time_step));
            } else {
                self.forward(time_step, params, fm_state, cs_state)?;
                out[(0, 0)] = self.observe_ics_basis(scobs, params, fm_state, cs_state)?;
            }

            Ok::<(), OcnusModelError<T>>(())
        })?;

        Ok(())
    }

    /// Perform an ensemble forward simulation and return the internal coordinates and basis
    /// vectors for the given spacecraft observers.
    fn simulate_ics_basis_ensbl(
        &self,
        series: &ScObsSeries<T>,
        ensbl: &mut OcnusEnsbl<T, D, FMST, CSST>,
        out_array: &mut DMatrixViewMut<ObserVec<T, 12>>,
    ) -> Result<(), OcnusModelError<T>> {
        let start = Instant::now();
        let mut timer = T::zero();

        if (series.len() != out_array.nrows()) || (out_array.ncols() != ensbl.len()) {
            return Err(OcnusModelError::OutputShape {
                expected_cols: ensbl.len(),
                expected_rows: series.len(),
                output_cols: out_array.ncols(),
                output_rows: out_array.nrows(),
            });
        }

        zip_eq(series, out_array.row_iter_mut()).try_for_each(|(scobs, mut out_col)| {
            // Compute time step to next observation.
            let time_step = *scobs.timestamp() - timer;
            timer = *scobs.timestamp();

            if time_step < T::zero() {
                return Err(OcnusModelError::NegativeTimeStep(time_step));
            } else {
                ensbl
                    .ptpdf
                    .particles()
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
                                    self.observe_ics_basis(scobs, params, fm_state, cs_state)?;

                                Ok::<(), OcnusModelError<T>>(())
                            },
                        )?;

                        Ok::<(), OcnusModelError<T>>(())
                    })?;
            }

            Ok::<(), OcnusModelError<T>>(())
        })?;

        debug!(
            "simulate_ics_plus_basis_ensbl: {:2.2}M evaluations in {:.2} sec",
            (series.len() * ensbl.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );
        Ok(())
    }
}
