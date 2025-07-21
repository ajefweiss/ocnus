use crate::{
    base::{ModelEnsbl, Obser, ScConf, ScObs},
    coords::Coordinates,
    obsty::{ICSCoordsBasis, NoiseModel, Observable},
    stats::{Density, DensityRange, ParticleDensity},
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{DVector, RealField, SVector, SVectorView, SVectorViewMut, Scalar, Vector3};
use num_traits::AsPrimitive;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{iter::Sum, ops::AddAssign, time::Instant};
use thiserror::Error;

/// Error types associated with the [`Model`] trait.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ModelError<T> {
    #[error("failed to convert external to internal coords")]
    CoordinateTransform(Vector3<T>),
    #[error("output contains NaN values")]
    OutputNaN,
    #[error("attempted to simulate is backwards in time (dt=-{0:.2}sec)")]
    NegativeTimeStep(T),
    #[error("failed to sample model parameters")]
    Sampling,
}

/// A trait that is shared by all models within the **ocnus** framework.
pub trait Model<T, const D: usize>: Coordinates<T, D>
where
    T: Copy + RealField,
{
    /// The base rayon chunk size that is used for any parallel iterators.
    ///
    /// Operations may use multiples of this value.
    const RCS: usize;

    /// Forward modeling state type.
    type FMST;

    /// Evolve a model state forward in time.
    fn forward(
        &self,
        time_step: T,
        params: &SVectorView<T, D>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>>;

    /// Returns the valid model parameter range.
    fn get_range(&self) -> SVector<DensityRange<T>, D>;

    /// Initialize the model parameters, the coordinate system and forward model states.
    fn initialize<P>(
        &self,
        params: &mut SVectorViewMut<T, D>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
        opt_pdf: Option<&P>,
        max_attempts: usize,
        rng: &mut impl Rng,
    ) -> Result<(), ModelError<T>>
    where
        for<'x> &'x P: Density<T, D>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        self.initialize_params(params, opt_pdf, max_attempts, rng)?;
        self.initialize_states(&params.as_view(), fm_state, cs_state)?;

        Ok(())
    }

    /// Initialize the model parameters, the coordinate system and forward model states for an ensemble.
    fn initialize_ensbl<P>(
        &self,
        ensbl: &mut ModelEnsbl<T, Self, D>,
        opt_pdf: Option<&P>,
        max_attempts: usize,
        rseed: u64,
    ) -> Result<(), ModelError<T>>
    where
        for<'x> &'x P: Density<T, D>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        ensbl
            .ptpdf
            .par_iter_mut()
            .zip(ensbl.fm_states.par_iter_mut())
            .zip(ensbl.cs_states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

                chunks
                    .iter_mut()
                    .try_for_each(|((params, fm_state), cs_state)| {
                        self.initialize::<P>(
                            params,
                            fm_state,
                            cs_state,
                            opt_pdf,
                            max_attempts,
                            &mut rng,
                        )?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "fevm_initialize_ensbl: {:2.3}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize the model parameters
    fn initialize_params(
        &self,
        params: &mut SVectorViewMut<T, D>,
        opt_pdf: Option<impl Density<T, D>>,
        max_attempts: usize,
        rng: &mut impl Rng,
    ) -> Result<(), ModelError<T>>
    where
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let opt_col = match opt_pdf.as_ref() {
            Some(pdf) => pdf.draw_sample(rng, max_attempts),
            None => self.model_prior().draw_sample(rng, max_attempts),
        };

        if let Some(col) = opt_col {
            params.set_column(0, &col);

            Ok(())
        } else {
            Err(ModelError::Sampling)
        }
    }

    /// Initialize the model parameters for an ensemble.
    fn initialize_params_ensbl<P>(
        &self,
        ensbl: &mut ModelEnsbl<T, Self, D>,
        opt_pdf: Option<&P>,
        max_attempts: usize,
        rseed: u64,
    ) -> Result<(), ModelError<T>>
    where
        for<'x> &'x P: Density<T, D>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        ensbl
            .ptpdf
            .par_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 17) as u64);

                chunks.iter_mut().try_for_each(|params| {
                    self.initialize_params(params, opt_pdf, max_attempts, &mut rng)?;

                    Ok::<(), ModelError<T>>(())
                })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "fevm_initialize_params_ensbl: {:2.3}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Initialize the coordinate system and forward model states.
    fn initialize_states(
        &self,
        params: &SVectorView<T, D>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>>;

    /// Initialize the the coordinate system and forward model states for an ensemble.
    fn initialize_states_ensbl(
        &self,
        ensbl: &mut ModelEnsbl<T, Self, D>,
    ) -> Result<(), ModelError<T>>
    where
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        ensbl
            .ptpdf
            .par_iter()
            .zip(ensbl.fm_states.par_iter_mut())
            .zip(ensbl.cs_states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunks| {
                chunks
                    .iter_mut()
                    .try_for_each(|((params, fm_state), cs_state)| {
                        self.initialize_states(params, fm_state, cs_state)?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "fevm_initialize_ensbl: {:2.3}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Returns a reference to the underlying model prior.
    fn model_prior(&self) -> impl Density<T, D>;

    /// Return internal coords and the basis vectors at the location of the observation.
    fn observe_ics_basis(
        &self,
        scconf: &ScConf<T>,
        params: &SVectorView<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ICSCoordsBasis<T>, ModelError<T>>;

    /// Resample the model parameters, and re-initialize the coordinate system and forward model states for an ensemble.
    fn resample_ensbl(
        &self,
        ensbl: &mut ModelEnsbl<T, Self, D>,
        ptpdf: &ParticleDensity<T, D>,
        rseed: u64,
    ) -> Result<(), ModelError<T>>
    where
        T: SampleUniform + Sum,
        usize: AsPrimitive<T>,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        ensbl
            .ptpdf
            .par_iter_mut()
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

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "fevm_resample: {:2.3}M evaluations in {:.2} sec",
            ensbl.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Perform a forward simulation and generate synthetic observables `OT` for the
    /// given spacecraft observers using a generating function `OF`.
    fn simulate<OT, OF>(
        &self,
        scobs: &ScObs<T, OT>,
        params: &SVectorView<T, D>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
        obs_func: &OF,
    ) -> Result<DVector<OT>, ModelError<T>>
    where
        OT: Observable,
        OF: Fn(
            &Self,
            &ScConf<T>,
            &SVector<T, D>,
            &Self::FMST,
            &Self::CSST,
        ) -> Result<OT, ModelError<T>>,
    {
        let mut timer = T::zero();

        let mut obser_vector = DVector::zeros(scobs.len());

        zip_eq(scobs, obser_vector.iter_mut()).try_for_each(|((timestamp, scconf), obs)| {
            // Compute time step to next observation.
            let time_step = *timestamp - timer;
            timer = *timestamp;

            if time_step < T::zero() {
                return Err(ModelError::NegativeTimeStep(time_step));
            } else {
                self.forward(time_step, params, fm_state, cs_state)?;

                *obs = obs_func(
                    self,
                    scconf,
                    &SVector::<T, D>::from_iterator(params.iter().copied()),
                    fm_state,
                    cs_state,
                )?;
            }

            Ok::<(), ModelError<T>>(())
        })?;

        Ok(obser_vector)
    }

    /// Perform an ensemble forward simulation and generate synthetic observables `OT` for the
    /// given spacecraft observers using a generating function `OF` and noise model `NM`.
    fn simulate_ensbl<OT, OF, NM>(
        &self,
        ensbl: &mut ModelEnsbl<T, Self, D>,
        obser: &mut Obser<T, OT>,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OT: AddAssign + Observable + Scalar,
        OF: Fn(
                &Self,
                &ScConf<T>,
                &SVector<T, D>,
                &Self::FMST,
                &Self::CSST,
            ) -> Result<OT, ModelError<T>>
            + Sync,
        NM: NoiseModel<T, OT> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        obser
            .time_iter_mut()
            .try_for_each(|(timestamp, scconf, mut obs_row)| {
                // Compute time step to next observation.
                let time_step = timestamp - timer;
                timer = timestamp;

                if time_step < T::zero() {
                    return Err(ModelError::NegativeTimeStep(time_step));
                } else {
                    ensbl
                        .ptpdf
                        .par_iter()
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
                                        scconf,
                                        &SVector::<T, D>::from_iterator(params.iter().copied()),
                                        fm_state,
                                        cs_state,
                                    )?;

                                    Ok::<(), ModelError<T>>(())
                                },
                            )?;

                            Ok::<(), ModelError<T>>(())
                        })?;
                }

                Ok::<(), ModelError<T>>(())
            })?;

        if let Some(noise) = opt_noise {
            obser
                .par_ensbl_iter_mut()
                .chunks(Self::RCS)
                .enumerate()
                .for_each(|(cdx, mut chunks)| {
                    let mut rng = noise.initialize_rng(29 * cdx as u64, 17);

                    chunks.iter_mut().for_each(|(scobs, col)| {
                        col.iter_mut()
                            .zip(noise.generate_noise(scobs, &mut rng).iter())
                            .for_each(|(value, noisevec)| *value += noisevec.clone());
                    });
                });

            noise.increment_random_seed()
        }

        debug!(
            "simulate_ensbl: {:2.3}M evaluations in {:.2} sec",
            (obser.len() * ensbl.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        Ok(())
    }

    /// Perform a forward simulation and return the internal coords and basis vectors for
    /// the given spacecraft observers.
    fn simulate_ics_basis<OT>(
        &self,
        scobs: &ScObs<T, OT>,
        params: &SVectorView<T, D>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<DVector<ICSCoordsBasis<T>>, ModelError<T>>
    where
        OT: Clone + Scalar,
    {
        let mut timer = T::zero();

        let mut obser_vector = DVector::zeros(scobs.len());

        zip_eq(scobs, obser_vector.iter_mut()).try_for_each(|((timestamp, scconf), obs)| {
            // Compute time step to next observation.
            let time_step = *timestamp - timer;
            timer = *timestamp;

            if time_step < T::zero() {
                return Err(ModelError::NegativeTimeStep(time_step));
            } else {
                self.forward(time_step, params, fm_state, cs_state)?;
                *obs = self.observe_ics_basis(scconf, params, fm_state, cs_state)?;
            }

            Ok::<(), ModelError<T>>(())
        })?;

        Ok(obser_vector)
    }

    /// Perform an ensemble forward simulation and return the internal coords and basis
    /// vectors for the given spacecraft observers.
    fn simulate_ics_basis_ensbl(
        &self,
        ensbl: &mut ModelEnsbl<T, Self, D>,
        obser: &mut Obser<T, ICSCoordsBasis<T>>,
    ) -> Result<(), ModelError<T>>
    where
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        obser
            .time_iter_mut()
            .try_for_each(|(timestamp, scconf, mut out_row)| {
                // Compute time step to next observation.
                let time_step = timestamp - timer;
                timer = timestamp;

                if time_step < T::zero() {
                    return Err(ModelError::NegativeTimeStep(time_step));
                } else {
                    ensbl
                        .ptpdf
                        .par_iter()
                        .zip(ensbl.fm_states.par_iter_mut())
                        .zip(ensbl.cs_states.par_iter_mut())
                        .zip(out_row.par_column_iter_mut())
                        .chunks(Self::RCS)
                        .try_for_each(|mut chunks| {
                            chunks.iter_mut().try_for_each(
                                |(((params, fm_state), cs_state), out)| {
                                    self.forward(time_step, params, fm_state, cs_state)?;

                                    out[(0, 0)] =
                                        self.observe_ics_basis(scconf, params, fm_state, cs_state)?;

                                    Ok::<(), ModelError<T>>(())
                                },
                            )?;

                            Ok::<(), ModelError<T>>(())
                        })?;
                }

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "simulate_ics_plus_basis_ensbl: {:2.3}M evaluations in {:.2} sec",
            (obser.len() * ensbl.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );
        Ok(())
    }
}
