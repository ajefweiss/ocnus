use crate::{
    base::{OcnusEnsbl, ScObs, ScObsSeries},
    coords::OcnusCoords,
    obser::{ObserVec, OcnusNoise, OcnusObser},
    stats::{Density, DensityRange, ParticleDensity},
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{
    DMatrixViewMut, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimName, DimSub, Dyn,
    OVector, RealField, Scalar, U1, Vector3, VectorView, VectorViewMut, allocator::Allocator,
};
use num_traits::AsPrimitive;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
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
    #[error("attempted to simulate is backwards in time (dt=-{0:.2}sec)")]
    NegativeTimeStep(T),
    #[error("failed to sample model parameters")]
    Sampling,
}

/// A trait that is shared by all models within the **ocnus** framework.
pub trait OcnusModel<T, D, FMST, CSST>: OcnusCoords<T, D, CSST>
where
    T: Copy + RealField + SampleUniform,
    D: DimName + DimMin<D>,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Sync,
    <DefaultAllocator as Allocator<D, Dyn>>::Buffer<T>: Send + Sync,
    <DefaultAllocator as Allocator<D, D>>::Buffer<T>: Sync,
    <DefaultAllocator as Allocator<D>>::Buffer<DensityRange<T>>: Sync,
    usize: AsPrimitive<T>,
{
    /// The base rayon chunk size that is used for any parallel iterators.
    ///
    /// Operations may use multiples of this value.
    const RCS: usize;

    /// Evolve a model state forward in time.
    fn forward<RStride, CStride>(
        &self,
        time_step: T,
        params: &VectorView<T, D, RStride, CStride>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), OcnusModelError<T>>
    where
        RStride: Dim,
        CStride: Dim;

    /// Initialize the model parameters, the coordinate system and forward model states.
    fn initialize<const A: usize, P, RStride, CStride>(
        &self,
        params: &mut VectorViewMut<T, D, RStride, CStride>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        opt_pdf: Option<&P>,
        rng: &mut impl Rng,
    ) -> Result<(), OcnusModelError<T>>
    where
        for<'x> &'x P: Density<T, D>,
        RStride: Dim,
        CStride: Dim,
    {
        self.initialize_params::<A, RStride, CStride>(params, opt_pdf, rng)?;
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
        P: Sync,
        for<'x> &'x P: Density<T, D>,
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
                        self.initialize::<A, P, _, _>(
                            params, fm_state, cs_state, opt_pdf, &mut rng,
                        )?;

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
    fn initialize_params<const A: usize, RStride, CStride>(
        &self,
        params: &mut VectorViewMut<T, D, RStride, CStride>,
        opt_pdf: Option<impl Density<T, D>>,
        rng: &mut impl Rng,
    ) -> Result<(), OcnusModelError<T>>
    where
        RStride: Dim,
        CStride: Dim,
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
        P: Sync,
        for<'x> &'x P: Density<T, D>,
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
                    self.initialize_params::<A, _, _>(params, opt_pdf, &mut rng)?;

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
    fn initialize_states<RStride, CStride>(
        &self,
        params: &VectorView<T, D, RStride, CStride>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
    ) -> Result<(), OcnusModelError<T>>
    where
        RStride: Dim,
        CStride: Dim;

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
    fn observe_ics_basis<RStride, CStride>(
        &self,
        scobs: &ScObs<T>,
        params: &VectorView<T, D, RStride, CStride>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 12>, OcnusModelError<T>>
    where
        RStride: Dim,
        CStride: Dim;

    /// Step sizes for finite differences.
    fn param_step_sizes(&self) -> OVector<T, D>;

    /// Resample the model parameters, and re-initialize the coOFordinate system and forward model states for an ensemble.
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
    ///  given spacecraft observers using a generating function `OF`.
    fn simulate<OT, OF, RStride, CStride>(
        &self,
        series: &ScObsSeries<T>,
        params: &VectorView<T, D, RStride, CStride>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        obs_func: &OF,
        obs_array: &mut DMatrixViewMut<OT>,
    ) -> Result<(), OcnusModelError<T>>
    where
        OT: OcnusObser,
        OF: Fn(&ScObs<T>, &OVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>,
        RStride: Dim,
        CStride: Dim,
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
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

            if time_step < T::zero() {
                return Err(OcnusModelError::NegativeTimeStep(time_step));
            } else {
                self.forward(time_step, params, fm_state, cs_state)?;

                obs_row[(0, 0)] = obs_func(
                    scobs,
                    &OVector::<T, D>::from_iterator(params.iter().cloned()),
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
        OF: Fn(&ScObs<T>, &OVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>> + Sync,
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
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

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
                                    scobs,
                                    &OVector::<T, D>::from_iterator(params.iter().cloned()),
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
    fn simulate_ics_basis<RStride, CStride>(
        &self,
        series: &ScObsSeries<T>,
        params: &VectorView<T, D, RStride, CStride>,
        fm_state: &mut FMST,
        cs_state: &mut CSST,
        out_array: &mut DMatrixViewMut<ObserVec<T, 12>>,
    ) -> Result<(), OcnusModelError<T>>
    where
        RStride: Dim,
        CStride: Dim,
    {
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
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

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
            let time_step = *scobs.get_timestamp() - timer;
            timer = *scobs.get_timestamp();

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
