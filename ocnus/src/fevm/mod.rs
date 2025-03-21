//! Implementations of forward ensemble vector models (FEVMs).

// mod cylm;
pub mod filters;
mod fisher;

// pub use cylm::*;
pub use fisher::*;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{
    OFloat, OState,
    geom::OcnusGeometry,
    obser::{ObserVec, OcnusObser, ScObs, ScObsSeries},
    stats::{CovMatrix, PDF, PDFParticles, StatsError},
};
use filters::ParticleFilterError;
use itertools::zip_eq;
use log::debug;
use nalgebra::{Const, DMatrix, DVector, Dyn, Matrix, Scalar};
use nalgebra::{SVectorView, VecStorage};
use num_traits::AsPrimitive;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::Instant;
use thiserror::Error;

/// A data structure that stores the parameters, states and random seed for a FEVM.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FEVMData<T, const P: usize, FS, GS>
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

impl<T, const P: usize, FS, GS> FEVMData<T, P, FS, GS>
where
    T: OFloat,
    FS: Clone + Default,
    GS: Clone + Default,
    Self: Serialize,
    usize: AsPrimitive<T>,
{
    /// Create a new [`FEVMData`] filled with zeros.
    pub fn new(size: usize) -> Self {
        Self {
            params: Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::zeros(size),
            fevm_states: vec![FS::default(); size],
            geom_states: vec![GS::default(); size],
            weights: vec![T::one() / size.as_(); size],
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
    Stats(#[from] StatsError<T>),
}

/// The trait that must be implemented for any FEVM (forward ensemble vector model) with an N-dimensional vector observable.
pub trait FEVM<T, const P: usize, const N: usize, FS, GS>: OcnusGeometry<T, P, GS>
where
    T: OFloat,
    FS: OState,
    GS: OState,
    Self: Sync,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
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
        fevmd: &mut FEVMData<T, P, FS, GS>,
        opt_pdf: Option<&D>,
        rseed: u64,
    ) -> Result<(), FEVMError<T>>
    where
        for<'a> &'a D: PDF<T, P>,
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

                        Ok::<(), FEVMError<T>>(())
                    })?;

                Ok::<(), FEVMError<T>>(())
            })?;

        debug!(
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols().as_() / 1e6.as_(),
            (start.elapsed().as_millis() as f64 / 1e3).as_()
        );

        Ok(())
    }

    /// Initialize parameters and states for a FEVM ensemble by
    /// resampling from a [`PDFParticles`].
    fn fevm_initialize_resample(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMData<T, P, FS, GS>,
        pdf: &PDFParticles<T, P>,
        rseed: u64,
    ) -> Result<(), FEVMError<T>> {
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

                        self.fevm_state(series, &params.as_view(), fevm_state, geom_state)?;

                        Ok::<(), FEVMError<T>>(())
                    })?;

                Ok::<(), FEVMError<T>>(())
            })?;

        debug!(
            "fevm_initialize: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols().as_() / 1e6.as_(),
            (start.elapsed().as_millis() as f64 / 1e3).as_()
        );

        Ok(())
    }

    /// Initialize parameters and states for a FEVM.
    /// If no pdf is given, the underlying model prior is used instead.
    fn fevm_initialize_params_only(
        &self,
        fevmd: &mut FEVMData<T, P, FS, GS>,
        opt_pdf: Option<impl PDF<T, P>>,
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
            fevmd.params.ncols().as_() / 1e6.as_(),
            (start.elapsed().as_millis() as f64 / 1e3).as_()
        );

        Ok(())
    }

    /// Initialize the model states within a [`FEVMData`] object.
    fn fevm_initialize_states_only(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMData<T, P, FS, GS>,
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
                        self.fevm_state(series, &params.as_view(), fevm_state, geom_state)?;

                        Ok::<(), FEVMError<T>>(())
                    })?;

                Ok::<(), FEVMError<T>>(())
            })?;

        debug!(
            "fevm_initialize_states_only: {:2.2}M evaluations in {:.2} sec",
            fevmd.params.ncols().as_() / 1e6.as_(),
            (start.elapsed().as_millis() as f64 / 1e3).as_()
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

    /// Perform an ensemble forward simulation and generate synthetic vector observables
    /// for the given spacecraft observers. Returns indices of runs that are valid w.r.t.
    /// to the spacecraft observation series.
    fn fevm_simulate(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &mut FEVMData<T, P, FS, GS>,
        output: &mut DMatrix<ObserVec<T, N>>,
        opt_noise: Option<&mut FEVMNoise<T>>,
    ) -> Result<Vec<bool>, FEVMError<T>> {
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
            (series.len() * fevmd.params.ncols()).as_() / 1e6.as_(),
            (start.elapsed().as_millis() as f64 / 1e3).as_()
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
    fn model_prior(&self) -> impl PDF<T, P>;

    /// Returns true if the ranges given by the model prior are within the parameter ranges
    /// of the model.
    fn validate_model_prior(&self) -> bool;
}

/// A generic noise generator for [`FEVM`].
#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum FEVMNoise<T>
where
    T: Clone + Scalar,
{
    Gaussian(T, u64),
    Multivariate(CovMatrix<T>, u64),
}

impl<T> FEVMNoise<T>
where
    T: OFloat,
    StandardNormal: Distribution<T>,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    /// Generate a random noise time-series.
    pub fn generate_noise<const N: usize>(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        rng: &mut impl Rng,
    ) -> DVector<ObserVec<T, N>> {
        match self {
            FEVMNoise::Gaussian(std_dev, ..) => {
                let normal = Normal::new(T::zero(), *std_dev).unwrap();
                let size = series.len();

                DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])))
            }
            FEVMNoise::Multivariate(covmat, ..) => {
                let normal = Normal::new(T::zero(), T::one()).unwrap();
                let size = series.len();

                let mut result = DVector::from_iterator(
                    size,
                    (0..size).map(|_| ObserVec([rng.sample(normal); N])),
                );

                for i in 0..N {
                    let values = covmat.cholesky_ltm()
                        * DVector::from_iterator(size, (0..size).map(|_| rng.sample(normal)));

                    result
                        .iter_mut()
                        .zip(values.row_iter())
                        .for_each(|(res, val)| res[i] = val[(0, 0)]);
                }

                result
            }
        }
    }

    /// Increment randon number seed.
    pub fn increment_seed(&mut self) {
        match self {
            FEVMNoise::Gaussian(.., seed) => {
                *seed += 1;
            }
            FEVMNoise::Multivariate(.., seed) => {
                *seed += 1;
            }
        }
    }

    /// Initialize a new random number generator using the base seed
    pub fn initialize_rng(&self, multiplier: u64, offset: u64) -> Xoshiro256PlusPlus {
        match self {
            FEVMNoise::Gaussian(.., seed) | FEVMNoise::Multivariate(.., seed) => {
                Xoshiro256PlusPlus::seed_from_u64(*seed * multiplier + offset)
            }
        }
    }
}

impl<T> Default for FEVMNoise<T>
where
    T: OFloat,
{
    fn default() -> Self {
        FEVMNoise::Gaussian(T::one(), 0)
    }
}
