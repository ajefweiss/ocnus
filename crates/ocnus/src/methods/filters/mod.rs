//! Particle filtering algorithms & methods.

mod abc;
mod dev;
mod sir;

use crate::{
    base::{Model, ModelEnsbl, ModelError},
    math::quantiles,
    obs::{
        Obs, ObsEnsbl,
        conf::ObsTime,
        data::ObsData,
        noise::{NullNoise, ObsNoise},
    },
};
use derive_builder::Builder;
use log::{debug, info};
use nalgebra::{Const, DVector, Dyn, OMatrix, RealField, SVectorView, Scalar};
use num_traits::AsPrimitive;
use prodef::Density;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, fmt::Debug, io::Write, iter::Sum, ops::AddAssign, time::Instant};
use thiserror::Error;

/// Errors associated with particle filters methods.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum FilterError<T> {
    #[error("effective particle number too small")]
    EffectiveParticles(T),
    #[error("generic model error")]
    Model(#[from] ModelError<T>),
    #[error("nothing was done")]
    Nothing,
    #[error("simulations exceeded time limit {elapsed:.1} / {limit:.1} sec")]
    TimeLimit { elapsed: f64, limit: f64 },
    #[error(
        "simulations predicted to exceed time limit {elapsed:.1} / {predicted:.1} / {limit:.1} sec"
    )]
    TimeLimitPredicted {
        elapsed: f64,
        predicted: f64,
        limit: f64,
    },
}

/// A particle filter object for a given model ensemble.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize, 
    OC: Serialize,
    OD: Serialize,
    M: Serialize,
    M::FMST: Serialize,
    M::CSST: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    OC: Deserialize<'de>, 
    OD: Deserialize<'de>,
    M: Deserialize<'de>,
    M::FMST: Deserialize<'de>,
    M::CSST: Deserialize<'de>"))]
pub struct FilterObject<T, OC, OD, M, const D: usize, const P: usize>
where
    T: Copy + RealField,
    OC: ObsTime<T>,
    OD: Scalar,
    M: Model<T, D, P>,
    M::CSST: std::fmt::Debug + Clone,
    M::FMST: std::fmt::Debug + Clone,
{
    model_ensbl: ModelEnsbl<T, M, D, P>,

    errors: Vec<T>,

    iterations: usize,

    model: M,

    obs_ensbl: ObsEnsbl<T, OC, OD>,

    random_seed: u64,

    /// Particle filter settings.
    pub settings: FilterSettings<T>,

    total_runs: usize,
}

impl<T, OC, OD, M, const D: usize, const P: usize> FilterObject<T, OC, OD, M, D, P>
where
    T: Copy + RealField + Sum,
    OC: ObsTime<T> + Sync,
    OD: AddAssign + ObsData,
    M: Model<T, D, P> + Sized + Sync,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// Return the stored errors.
    pub fn errors(&self) -> &[T] {
        &self.errors
    }

    /// Re-evaluate errors using a given error function `EF`.
    pub fn errors_func<EF>(&self, func: &EF) -> Vec<T>
    where
        EF: Fn(&[OD], &[OD]) -> T + Sync,
    {
        self.obs_ensbl.errors_func(func)
    }

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

    /// Filtering step from a prior distribution `pdf` using a filter function `FF`.
    /// Returns the filter values and the number of sub-iterations required to reach the target number of new particles.
    pub fn filter<FF, OF, G, NM>(
        &mut self,
        flt_func: &FF,
        obs_func: &OF,
        pdf: G,
        opt_noise: &mut Option<&mut NM>,
        seed_multiplier: u64,
    ) -> Result<(Vec<T>, usize), FilterError<T>>
    where
        FF: Fn(&[OD], &[OD]) -> (bool, T) + Sync,
        OF: Fn(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
        G: Density<T, Const<P>> + Sync,
        NM: ObsNoise<T, OC, OD> + Sync,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration: usize = 0;

        let mut target_filter_values = Vec::<T>::with_capacity(self.model_ensbl.len());

        // Temporary ensemble and output array.
        let (mut temp_ensbl, mut temp_obs_ensbl) = (
            ModelEnsbl::new(
                OMatrix::<T, Const<P>, Dyn>::zeros(
                    self.model_ensbl.len() * self.settings.simulation_ensemble_size_factor,
                ),
                None,
                None,
            ),
            ObsEnsbl::new(
                self.obs_ensbl.obs().clone(),
                self.model_ensbl.len() * self.settings.simulation_ensemble_size_factor,
                None,
            )
            .unwrap(),
        );

        // Iterate until we have enough new particles.
        while counter != self.model_ensbl.len() {
            self.model.initialize_ensbl(
                &mut temp_ensbl,
                pdf.clone(),
                self.settings.max_sample_attempts,
                self.random_seed + seed_multiplier * iteration as u64,
            )?;

            self.model
                .simulate_ensbl(&mut temp_ensbl, &mut temp_obs_ensbl, obs_func, opt_noise)?;

            let mut filter_flags = vec![true; temp_ensbl.len()];

            let filter_values = temp_obs_ensbl
                .par_ensbl_iter()
                .zip(filter_flags.par_iter_mut())
                .chunks(M::RCS)
                .map(|mut chunk| {
                    chunk
                        .iter_mut()
                        .map(|((_, out), flag)| {
                            let (result, value) = flt_func(
                                self.obs_ensbl
                                    .ref_data()
                                    .expect("missing reference data")
                                    .as_slice(),
                                out.as_slice(),
                            );

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
            if counter + indices.len() > self.model_ensbl.len() {
                debug!(
                    "removing excessive ensemble members(n={})",
                    counter + indices.len() - self.model_ensbl.len()
                );
                indices.drain((self.model_ensbl.len() - counter)..indices.len());
            }

            // Copy over the results.
            indices.iter().enumerate().for_each(|(edx, idx)| {
                self.model_ensbl
                    .input
                    .set_column(counter + edx, &temp_ensbl.input.column(*idx));

                self.obs_ensbl
                    .set_output(counter + edx, &temp_obs_ensbl.output(*idx));

                target_filter_values.push(filter_values[*idx]);
            });

            counter += indices.len();

            self.random_seed += 1;
            iteration += 1;

            // Abort if simulation time is above the given limit (or is estimated to be above).
            if start.elapsed().as_millis() as f64 / 1e3 > self.settings.simulation_time_limit {
                info!(
                    "filter aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tcollected samples = {:.1} / {}",
                    (iteration
                        * self.model_ensbl.len()
                        * self.settings.simulation_ensemble_size_factor
                        * self.obs_ensbl.len()) as f64
                        / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    self.model_ensbl.len(),
                );

                return Err(FilterError::TimeLimit {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    limit: self.settings.simulation_time_limit,
                });
            } else if (counter == 0)
                || (self.settings.simulation_time_prediction
                    && ((self.model_ensbl.len() / counter) as f64
                        * start.elapsed().as_millis() as f64
                        / 1e3
                        > self.settings.simulation_time_limit))
            {
                let estimated_time = match counter.cmp(&0) {
                    Ordering::Equal => f64::INFINITY,
                    _ => {
                        (self.model_ensbl.len() / counter) as f64
                            * start.elapsed().as_millis() as f64
                            / 1e3
                    }
                };

                info!(
                    "filter pre-emptively aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\ttotal predicted duration: {:.2}\n\tcollected samples = {:.1} / {}",
                    (iteration
                        * self.model_ensbl.len()
                        * self.settings.simulation_ensemble_size_factor
                        * self.obs_ensbl.len()) as f64
                        / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    estimated_time,
                    counter,
                    self.model_ensbl.len(),
                );

                return Err(FilterError::TimeLimitPredicted {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    predicted: estimated_time,
                    limit: self.settings.simulation_time_limit,
                });
            }
        }

        Ok((target_filter_values, iteration))
    }

    /// Initialize the ensemble from a prior distribution `pdf` with a filtering function `FF`.
    pub fn initialize<G, FF, OF>(
        &mut self,
        flt_func: &FF,
        obs_func: &OF,
        pdf: G,
    ) -> Result<(), FilterError<T>>
    where
        T: AsPrimitive<f64>,
        G: Density<T, Const<P>> + Sync,
        FF: Fn(&[OD], &[OD]) -> (bool, T) + Sync,
        OF: Fn(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
    {
        let start = Instant::now();

        let (filter_values, iterations) = self.filter(
            flt_func,
            obs_func,
            pdf,
            &mut None::<&mut NullNoise<T>>,
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

        debug!(
            "initialize(n={})\n\teps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
            self.model_ensbl.len(),
            q_low,
            q_mid,
            q_hgh,
            T::from_f64(
                (iterations
                    * self.model_ensbl.len()
                    * self.settings.simulation_ensemble_size_factor
                    * self.obs_ensbl.len()) as f64
                    / 1e6
            )
            .unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
        );

        self.errors = filter_values;

        // Set diagnostic fields.
        self.iterations = 1;
        self.total_runs =
            iterations * self.model_ensbl.len() * self.settings.simulation_ensemble_size_factor;

        Ok(())
    }

    /// Check if the ensemble is empty.
    pub fn is_empty(&self) -> bool {
        self.model_ensbl.is_empty()
    }

    /// Return the ensemble size.
    pub fn len(&self) -> usize {
        self.model_ensbl.len()
    }

    /// Create a new [`FilterObject`].
    pub fn new(
        model: M,
        model_ensbl: ModelEnsbl<T, M, D, P>,
        obs_ensbl: ObsEnsbl<T, OC, OD>,
        random_seed: u64,
        opt_settings: Option<FilterSettings<T>>,
    ) -> Self
    where
        T: Default,
    {
        Self {
            model_ensbl,
            errors: Vec::new(),
            iterations: 0,
            model,
            obs_ensbl,
            random_seed,
            settings: opt_settings.unwrap_or(FilterSettingsBuilder::default().build().unwrap()),
            total_runs: 0,
        }
    }

    /// Return the model ensemble particles.
    pub fn particles(&self) -> &OMatrix<T, Const<P>, Dyn> {
        &self.model_ensbl.input
    }

    /// Return the prior distribution of the model.
    pub fn prior(&self) -> impl Density<T, Const<P>> {
        self.model.prior()
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

    /// Simulate the model ensemble and return the results in an observation ensemble.
    pub fn simulate<OF, NM>(
        &mut self,
        opt_obs: Option<&Obs<T, OC>>,
        opt_ref_data: Option<&DVector<OD>>,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<ObsEnsbl<T, OC, OD>, FilterError<T>>
    where
        OF: Fn(&M, &OC, &SVectorView<T, P>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
        NM: ObsNoise<T, OC, OD> + Sync,
        M: Debug,
    {
        let obs_ensbl = match opt_obs {
            Some(obs) => {
                let mut obs_ensbl =
                    ObsEnsbl::new(obs.clone(), self.model_ensbl.len(), opt_ref_data.cloned())
                        .unwrap();

                self.model.initialize_states_ensbl(&mut self.model_ensbl)?;

                self.model.simulate_ensbl(
                    &mut self.model_ensbl,
                    &mut obs_ensbl,
                    obs_func,
                    opt_noise,
                )?;

                obs_ensbl
            }
            None => {
                self.model.initialize_states_ensbl(&mut self.model_ensbl)?;

                self.model.simulate_ensbl(
                    &mut self.model_ensbl,
                    &mut self.obs_ensbl,
                    obs_func,
                    opt_noise,
                )?;

                self.obs_ensbl.clone()
            }
        };

        Ok(obs_ensbl)
    }

    /// Return the ensemble member weights (optional).
    pub fn weights(&self) -> Option<&Vec<T>> {
        self.model_ensbl.opt_weights.as_ref()
    }
}

/// A data structure for holding particle filter settings.
///
/// Various settings may have different meanings depending on the specific filtering algorithm that is used.
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
pub struct FilterSettings<T>
where
    T: Copy + RealField,
{
    /// Multiplier for the transition kernel, a higher value leads to a better
    /// exploration of the parameter space but slower convergence. For ABC the "optimal" value is 2.0
    /// (see Filippi et al. 2013), although lower values can also be used.
    #[builder(default = T::from_usize(2).unwrap())]
    pub exploration_factor: T,

    /// Maximum number of attempted sampling draws., by default 32.
    #[builder(default = 512)]
    pub max_sample_attempts: usize,

    /// Maximum number of iterations, by default 10.
    #[builder(default = 10)]
    pub max_iterations: usize,

    /// Effective particle threshold factor, by default 5%.
    #[builder(default = T::from_f64(0.05).unwrap())]
    pub effective_particle_threshold_factor: T,

    /// Simulation ensemble size used for each sub-iteration, this value should be a multiple of `ensemble_size`.
    /// By default this value is 8.
    #[builder(default = 8)]
    pub simulation_ensemble_size_factor: usize,

    /// Maximum simulation time limit (in seconds) for each iteration, by default 5.0 sec.
    #[builder(default = 5.0)]
    pub simulation_time_limit: f64,

    /// Attempt to predict the simulation time from a single sub-iteration instead of using the full time limit, by default true.
    #[builder(default = true)]
    pub simulation_time_prediction: bool,
}
