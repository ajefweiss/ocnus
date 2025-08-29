//! Particle filtering methods.

mod abc;
mod dev;
mod sir;

use crate::{
    base::{Model, ModelEnsbl, ModelError, Obser, ScConf, ScObs},
    math::quantiles,
    obsty::{NoiseModel, NullNoise, Observable},
    stats::{Density, MultivariateDensity},
};
use derive_builder::Builder;
use log::{debug, info};
use nalgebra::{Const, Dyn, Matrix, VecStorage};
use nalgebra::{DVector, RealField, SVector, Scalar};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, io::Write, iter::Sum, ops::AddAssign, time::Instant};
use thiserror::Error;

/// Errors associated with particle filters methods.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ParticleFilterError<T> {
    #[error("effective particle number too small")]
    InsufficientParticles(T),
    #[error("generic model error")]
    Model(#[from] ModelError<T>),
    #[error("nothing was done")]
    Nothing,
    #[error("simulations exceeded time limit {elapsed:.1} / {limit:.1} sec")]
    TimeLimitExceeded { elapsed: f64, limit: f64 },
}

/// A particle filter data structure.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize, 
    M: Serialize,
    M::FMST: Serialize,
    M::CSST: Serialize,
    OT: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    M: Deserialize<'de>,
    M::FMST: Deserialize<'de>,
    M::CSST: Deserialize<'de>,
    OT: Deserialize<'de>"))]
pub struct ParticleFilter<T, M, const D: usize, OT>
where
    T: Copy + RealField + SampleUniform,
    M: Model<T, D>,
    M::CSST: std::fmt::Debug + Clone,
    M::FMST: std::fmt::Debug + Clone,
    OT: Clone + Scalar,
{
    /// The model ensemble.
    pub ensbl: ModelEnsbl<T, M, D>,

    /// The model output errors.
    pub errors: Vec<T>,

    /// Iteration counter.
    pub iter: usize,

    /// Underlying model.
    pub model: M,

    /// The model obs.
    pub obser: Obser<T, OT>,

    /// Random seed (initial & running).
    pub rseed: u64,

    /// Total simulation runs counter,
    pub truns: usize,
}

impl<T, M, const D: usize, OT> ParticleFilter<T, M, D, OT>
where
    T: Copy + RealField + SampleUniform + Sum,
    M: Model<T, D>,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
    OT: Observable,
{
    /// Create a new [`ParticleFilter`] from a given set of particles.
    pub fn from_particles(
        scobs: ScObs<T, OT>,
        model: M,
        particles: Matrix<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>>,
        opt_weights: Option<DVector<T>>,
        initial_seed: u64,
    ) -> Self {
        let size = particles.ncols();

        Self {
            ensbl: ModelEnsbl::from_particles(particles, Some(&model.get_range()), opt_weights),
            errors: Vec::with_capacity(size),
            iter: 0,
            model,
            obser: Obser::new(scobs, size),
            rseed: initial_seed,
            truns: 0,
        }
    }

    /// Create a new [`ParticleFilter`].
    pub fn new(scobs: ScObs<T, OT>, model: M, size: usize, initial_seed: u64) -> Self {
        Self {
            ensbl: ModelEnsbl::new(size, Some(&model.get_range())),
            errors: Vec::with_capacity(size),
            iter: 0,
            model,
            obser: Obser::new(scobs, size),
            rseed: initial_seed,
            truns: 0,
        }
    }
}

impl<T, M, const D: usize, OT> ParticleFilter<T, M, D, OT>
where
    T: Copy + RealField + SampleUniform + Sum,
    M: Model<T, D> + Sync,
    M::FMST: std::fmt::Debug + Default + Clone + Send,
    M::CSST: std::fmt::Debug + Default + Clone + Send,
    OT: AddAssign + Observable,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// Compute a a quantile of the field `errors`.
    pub fn error_quantile(&self, quantile: T) -> Option<T>
    where
        T: AsPrimitive<usize>,
    {
        assert!(
            (T::zero()..T::one()).contains(&quantile),
            "quantile must be within [0, 1]"
        );

        if self.errors.is_empty() {
            None
        } else {
            let mut errors_sorted = self.errors.clone();

            errors_sorted.sort_by(|a, b| a.partial_cmp(b).expect("errors cannot contain any NaN"));

            Some(errors_sorted[(quantile * T::from_usize(self.errors.len()).unwrap()).as_()])
        }
    }

    /// A generic particle filtering algorithm with a filtering function `FF`.
    /// Returns the filter values and the number of sub-iterations required to reach the target number of new particles.
    #[allow(clippy::too_many_arguments)]
    pub fn pf_filter<FF, OF, P, NM>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        flt_func: &FF,
        obs_func: &OF,
        opt_pdf: Option<&P>,
        opt_noise: &mut Option<&mut NM>,
        max_attempts: usize,
        seed_multiplier: u64,
    ) -> Result<(Vec<T>, usize), ParticleFilterError<T>>
    where
        FF: Fn(&[OT], &[OT]) -> (bool, T) + Send + Sync,
        OF: Fn(&M, &ScConf<T>, &SVector<T, D>, &M::FMST, &M::CSST) -> Result<OT, ModelError<T>>
            + Sync,
        for<'x> &'x P: Density<T, D>,
        NM: NoiseModel<T, OT> + Sync,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration: usize = 0;

        let mut target_filter_values = Vec::<T>::with_capacity(self.ensbl.len());

        // Temporary ensemble and output array.
        let (mut temp_ensbl, mut temp_obser) = (
            ModelEnsbl::new(
                self.ensbl.len() * settings.simulation_ensemble_size_factor,
                Some(&self.model.model_prior().get_range()),
            ),
            Obser::new(
                self.obser.scobs().clone(),
                self.ensbl.len() * settings.simulation_ensemble_size_factor,
            ),
        );

        // Iterate until we have enough new particles.
        while counter != self.ensbl.len() {
            self.model.initialize_ensbl(
                &mut temp_ensbl,
                opt_pdf,
                max_attempts,
                self.rseed + seed_multiplier * iteration as u64,
            )?;

            self.model
                .simulate_ensbl(&mut temp_ensbl, &mut temp_obser, obs_func, opt_noise)?;

            let mut filter_flags = vec![true; temp_ensbl.len()];

            let filter_values = temp_obser
                .par_ensbl_iter()
                .zip(filter_flags.par_iter_mut())
                .chunks(M::RCS)
                .map(|mut chunks| {
                    chunks
                        .iter_mut()
                        .map(|((_, out), flag)| {
                            let (result, value) = flt_func(self.obser.refdt(), out.as_slice());

                            **flag = result;

                            value
                        })
                        .collect::<Vec<T>>()
                })
                .flatten()
                .collect::<Vec<T>>();

            // Transform the filter flags to a list of valid indices.
            let mut indices = filter_flags
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            // Remove excessive ensemble members.
            if counter + indices.len() > self.ensbl.len() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices.len() - self.ensbl.len()
                );
                indices.drain((self.ensbl.len() - counter)..indices.len());
            }

            // Copy over results.
            indices.iter().enumerate().for_each(|(edx, idx)| {
                self.ensbl
                    .ptpdf
                    .set_particle(counter + edx, &temp_ensbl.ptpdf.get_particle(*idx));

                self.obser
                    .set_output(counter + edx, &temp_obser.get_output(*idx));

                target_filter_values.push(filter_values[*idx]);
            });

            counter += indices.len();

            self.rseed += 1;
            iteration += 1;

            // Abort if simulation time is above the given limit (or is estimated to be above).
            if start.elapsed().as_millis() as f64 / 1e3 > settings.simulation_time_limit {
                info!(
                    "pf_filter aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tcollected samples = {:.1} / {}",
                    (iteration
                        * self.ensbl.len()
                        * settings.simulation_ensemble_size_factor
                        * self.obser.len()) as f64
                        / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    self.ensbl.len(),
                );

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    limit: settings.simulation_time_limit,
                });
            } else if (counter == 0)
                || (settings.simulation_time_prediction
                    && ((self.ensbl.len() / counter) as f64 * start.elapsed().as_millis() as f64
                        / 1e3
                        > settings.simulation_time_limit))
            {
                let estimated_time = match counter.cmp(&0) {
                    Ordering::Equal => f64::INFINITY,
                    _ => {
                        (self.ensbl.len() / counter) as f64 * start.elapsed().as_millis() as f64
                            / 1e3
                    }
                };

                info!(
                    "pf_filter pre-emptively aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\ttotal predicted duration: {:.2}\n\tcollected samples = {:.1} / {}",
                    (iteration
                        * self.ensbl.len()
                        * settings.simulation_ensemble_size_factor
                        * self.obser.len()) as f64
                        / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    estimated_time,
                    counter,
                    self.ensbl.len(),
                );

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: estimated_time,
                    limit: settings.simulation_time_limit,
                });
            }
        }

        Ok((target_filter_values, iteration))
    }

    /// Initialize the ensemble data with an optional filtering function `FF`.
    pub fn pf_initialize_ensbl<FF, OF>(
        &mut self,
        flt_func: &FF,
        obs_func: &OF,
        settings: &ParticleFilterSettings<T>,
    ) -> Result<(), ParticleFilterError<T>>
    where
        T: AsPrimitive<f64>,
        FF: Fn(&[OT], &[OT]) -> (bool, T) + Send + Sync,
        OF: Fn(&M, &ScConf<T>, &SVector<T, D>, &M::FMST, &M::CSST) -> Result<OT, ModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        let (filter_values, iterations) = self.pf_filter(
            settings,
            flt_func,
            obs_func,
            None::<&MultivariateDensity<T, D>>,
            &mut None::<&mut NullNoise<T>>,
            settings.max_attempts,
            7573,
        )?;

        // Compute quantiles for logging purposes.
        let quantiles = quantiles(
            &filter_values,
            &[
                T::from_f64(0.34).unwrap(),
                T::from_f64(0.50).unwrap(),
                T::from_f64(0.68).unwrap(),
            ],
        );

        let (q_low, q_mid, q_hgh) = (quantiles[0], quantiles[1], quantiles[2]);

        info!(
            "pf_initialize_data\n\tKL delta: n/a | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
            q_low,
            q_mid,
            q_hgh,
            T::from_f64(
                (iterations
                    * self.ensbl.len()
                    * settings.simulation_ensemble_size_factor
                    * self.obser.len()) as f64
                    / 1e6
            )
            .unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
        );

        // Re-estimate the covariance matrix for the ensemble of particles.
        self.ensbl.ptpdf.update_mvpdf();

        self.errors = filter_values;

        // Set diagnostic fields.
        self.iter = 1;
        self.truns = iterations * self.ensbl.len() * settings.simulation_ensemble_size_factor;

        Ok(())
    }

    /// Generate observations and fill the internal [`Obser`] field.
    #[allow(clippy::too_many_arguments)]
    pub fn pf_simulate<OF, NM>(
        &mut self,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ParticleFilterError<T>>
    where
        OF: Fn(&M, &ScConf<T>, &SVector<T, D>, &M::FMST, &M::CSST) -> Result<OT, ModelError<T>>
            + Sync,
        NM: NoiseModel<T, OT> + Sync,
    {
        self.model
            .simulate_ensbl(&mut self.ensbl, &mut self.obser, obs_func, opt_noise)?;

        Ok(())
    }

    /// Serialize result to a JSON file.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}

/// A data structure for holding particle filter settings.
///
/// Various settings may have different meanings depending on the specific filtering algorithm that is used.
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<T>
where
    T: Copy + RealField,
{
    /// Multiplier for the transition kernel (covariance matrix), a higher value leads to a better
    /// exploration of the parameter space but slower convergence. For ABC the "optimal" value is 2.0
    /// (see Filippi et al. 2013), although lower values can also be used.
    #[builder(default = T::from_usize(2).unwrap())]
    pub expl_factor: T,

    /// Maximum number of attempted sampling draws.
    pub max_attempts: usize,

    /// Maximum number of iterations.
    #[builder(default = 10)]
    pub max_iterations: usize,

    /// Effecetive particle threshold factor.
    #[builder(default = T::from_f64(0.05).unwrap())]
    pub eff_particle_threshold_factor: T,

    /// Simulation ensemble size used for each sub-iteration, this value should be a multiple of `ensemble_size`.
    #[builder(default = 4)]
    pub simulation_ensemble_size_factor: usize,

    /// Maximum simulation time limit (in seconds) for each iteration.
    #[builder(default = 5.0)]
    pub simulation_time_limit: f64,

    /// Attempt to predict the simulation time from a single sub-iteration instead of using the full time limit.
    #[builder(default = true)]
    pub simulation_time_prediction: bool,
}
