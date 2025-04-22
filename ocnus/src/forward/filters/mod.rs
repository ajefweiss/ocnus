//! Particle filter algorithms.

mod abc;
mod sir;

pub use abc::{ABCParticleFilter, ABCParticleFilterSettings};
pub use sir::SIRParticleFilter;

use crate::{
    OcnusError, fXX,
    forward::{FSMEnsbl, OcnusFSM},
    math::{T, sqrt},
    obser::{NoNoise, ObserVec, OcnusObser, ScObsSeries},
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, Scalar, U1};
use num_traits::Zero;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, ops::AddAssign, time::Instant};
use thiserror::Error;

/// Errors associated with particle filters.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ParticleFilterError<T> {
    #[error("insufficiently large sample size {effective_sample_size} / {ensemble_size}")]
    SmallSampleSize {
        effective_sample_size: T,
        ensemble_size: usize,
    },
    #[error("iterations exceeded given time limit {elapsed} sec - {limit} sec")]
    TimeLimitExceeded { elapsed: T, limit: T },
}

/// A trait that enables the use of generic particle filter methods for a [`OcnusFSM`].
pub trait ParticleFilter<T, O, const P: usize, FMST, CSST>: OcnusFSM<T, O, P, FMST, CSST>
where
    T: fXX,
    O: AddAssign + OcnusObser + Scalar + Zero,
    FMST: Default + Clone + Send,
    CSST: Default + Clone + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
    /// Creates a new [`FSMEnsbl`] using only valid ensemble members or optionally with an error
    /// metric (recommended). This function is intended to be used as the initialization step in
    /// any particle filtering algorithm.
    fn pf_initialize_ensemble<E>(
        &self,
        series: &ScObsSeries<T, O>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        opt_metric: Option<(&E, T)>,
        settings: &mut ParticleFilterSettings<T>,
    ) -> Result<ParticleFilterResults<T, O, P, FMST, CSST>, OcnusError<T>>
    where
        E: Fn(&DVectorView<O>, &ScObsSeries<T, O>) -> T + Send + Sync,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration: usize = 0;

        let mut target_ensbl = FSMEnsbl::new(ensemble_size);
        let mut target_output = DMatrix::<O>::zeros(series.len(), ensemble_size);
        let mut target_filter_values = Vec::<T>::with_capacity(ensemble_size);

        let mut temp_ensbl = FSMEnsbl::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<O>::zeros(series.len(), sim_ensemble_size);

        while counter != target_ensbl.len() {
            self.fsm_initialize_ensbl(
                series,
                &mut temp_ensbl,
                None,
                settings.rseed + 17 * iteration as u64,
            )?;

            let mut flags = self.fsm_simulate_ensbl(
                series,
                &mut temp_ensbl,
                &mut temp_output.as_view_mut(),
                None::<&mut NoNoise<T>>,
            )?;

            let opt_filter_values = if let Some((filter, epsilon)) = opt_metric {
                Some(
                    temp_output
                        .par_column_iter()
                        .zip(flags.par_iter_mut())
                        .chunks(Self::RCS)
                        .map(|mut chunks| {
                            chunks
                                .iter_mut()
                                .map(|(out, flag)| {
                                    if **flag {
                                        let value =
                                            filter(&out.as_view::<Dyn, U1, U1, Dyn>(), series);
                                        **flag = value < epsilon;

                                        value
                                    } else {
                                        T::nan()
                                    }
                                })
                                .collect::<Vec<T>>()
                        })
                        .flatten()
                        .collect::<Vec<T>>(),
                )
            } else {
                None
            };

            let mut indices_valid = flags
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            debug!("valid: {}", indices_valid.len());

            // Remove excessive ensemble members.
            if counter + indices_valid.len() > target_ensbl.len() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - target_ensbl.len()
                );
                indices_valid.drain((target_ensbl.len() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                target_ensbl
                    .params_array
                    .set_column(counter + edx, &temp_ensbl.params_array.column(*idx));

                if let Some(filter_values) = &opt_filter_values {
                    target_filter_values.push(filter_values[*idx]);
                }
            });

            counter += indices_valid.len();

            iteration += 1;

            if T!(start.elapsed().as_millis() as f64 / 1e3) > settings.sim_time_limit {
                info!(
                    "pf_initialize aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tsamples = {:.1} / {}",
                    (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    ensemble_size,
                );

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: T!(start.elapsed().as_millis() as f64 / 1e3),
                    limit: settings.sim_time_limit,
                }
                .into());
            }
        }

        if !target_filter_values.is_empty() {
            let filter_values_sorted = target_filter_values
                .iter()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .collect::<Vec<T>>();

            // Compute quantiles for logging purposes.
            let eps_1 =
                filter_values_sorted[(T!(ensemble_size as f64) * settings.quantiles[0]).as_()];
            let eps_2 =
                filter_values_sorted[(T!(ensemble_size as f64) * settings.quantiles[1]).as_()];
            let eps_3 = if settings.quantiles[2] >= T::one() {
                filter_values_sorted[ensemble_size - 1]
            } else {
                filter_values_sorted[(T!(ensemble_size as f64) * settings.quantiles[2]).as_()]
            };

            info!(
                "pf_initialize_data\n\tKL delta: n/a | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                eps_1,
                eps_2,
                eps_3,
                T!((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6),
                T!(start.elapsed().as_millis() as f64 / 1e3),
            );

            settings.rseed += 1;

            self.fsm_initialize_states_ensbl(series, &mut target_ensbl)?;

            self.fsm_simulate_ensbl(
                series,
                &mut target_ensbl,
                &mut target_output.as_view_mut(),
                None::<&mut NoNoise<T>>,
            )?;

            Ok(ParticleFilterResults::ABC(
                target_ensbl,
                target_output,
                target_filter_values,
                [eps_1, eps_2, eps_3],
            ))
        } else {
            info!(
                "pf_initialize_data\n\tKL delta: {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                T!((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6),
                T!(start.elapsed().as_millis() as f64 / 1e3),
            );

            settings.rseed += 1;

            Ok(ParticleFilterResults::Initialize(
                target_ensbl,
                target_output,
            ))
        }
    }
}

/// An algebraic data type holding the results of any particle filtering method.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound = "
    T: for<'x> Deserialize<'x> + Serialize, 
    O: for<'x> Deserialize<'x> + Serialize, 
    FMST: for<'x> Deserialize<'x> + Serialize, 
    CSST: for<'x> Deserialize<'x> + Serialize
")]
pub enum ParticleFilterResults<T, O, const P: usize, FMST, CSST>
where
    T: fXX,
    O: Scalar,
{
    /// Result from pf_initialize
    Initialize(FSMEnsbl<T, P, FMST, CSST>, DMatrix<O>),
    /// Advanced with results with additional values.
    SIR(FSMEnsbl<T, P, FMST, CSST>, DMatrix<O>, Vec<T>),
    ///Advanced with results with additional values and respective quantiles
    ABC(FSMEnsbl<T, P, FMST, CSST>, DMatrix<O>, Vec<T>, [T; 3]),
}

impl<T, O, const P: usize, FMST, CSST> ParticleFilterResults<T, O, P, FMST, CSST>
where
    T: fXX,
    O: Scalar,
{
    /// Return a reference to the underlying [`FSMEnsbl`].
    pub fn get_ensbl(&self) -> &FSMEnsbl<T, P, FMST, CSST> {
        match self {
            ParticleFilterResults::Initialize(ensbl, ..) => ensbl,
            ParticleFilterResults::SIR(ensbl, ..) => ensbl,
            ParticleFilterResults::ABC(ensbl, ..) => ensbl,
        }
    }

    /// Return a reference to the underlying quantile valies.
    pub fn get_quantiles(&self) -> Option<&[T]> {
        match self {
            ParticleFilterResults::Initialize(..) => None,
            ParticleFilterResults::SIR(..) => None,
            ParticleFilterResults::ABC(.., quantiles) => Some(quantiles),
        }
    }

    /// Serialize result to a JSON file.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<T>
where
    T: fXX,
{
    /// Effective sample size threshold factor.
    ///
    /// If the ESS is below this required limit (fraction), the algorithm returns an error.
    #[builder(default = T!(0.175))]
    pub ess_factor: T,

    /// Multiplier for the transition kernel (covariance matrix), a higher value leads to a better
    /// exploration of the parameter  space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = T!(2.0))]
    pub expl_factor: T,

    /// Iteration counter,
    #[builder(default = 0, setter(skip))]
    pub iter: usize,

    /// Random seed (initial & running).
    #[builder(default = 42)]
    pub rseed: u64,

    /// Maximum simulation time limit (in seconds).
    #[builder(default = T!(5.0))]
    pub sim_time_limit: T,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,

    /// Quantile evaluations.
    #[builder(default = [T!(0.25),T!(0.5),T!(0.75)])]
    pub quantiles: [T; 3],
}

/// Mean square error filter for particle filtering methods.
pub fn mean_square_metric<T, const N: usize>(
    out: &DVectorView<ObserVec<T, N>>,
    series: &ScObsSeries<T, ObserVec<T, N>>,
) -> T
where
    T: fXX,
{
    out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mean_square_error(scobs.get_observation()))
        .sum::<T>()
        / T::from_usize(series.count_observations()).unwrap()
}

/// Root mean square error filter for particle filtering methods.
pub fn root_mean_square_metric<T, const N: usize>(
    out: &DVectorView<ObserVec<T, N>>,
    series: &ScObsSeries<T, ObserVec<T, N>>,
) -> T
where
    T: fXX,
{
    sqrt!(
        out.into_iter()
            .zip(series)
            .map(|(out_vec, scobs)| out_vec.mean_square_error(scobs.get_observation()))
            .sum::<T>()
            / T::from_usize(series.count_observations()).unwrap()
    )
}
