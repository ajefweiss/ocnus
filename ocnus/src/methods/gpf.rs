use crate::{
    OcnusEnsbl, OcnusError, OcnusModel,
    obser::{NullNoise, OcnusObser, ScObsSeries},
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, Scalar, U1};
use num_traits::{AsPrimitive, Zero};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, ops::AddAssign, time::Instant};
use thiserror::Error;

/// Errors associated with particle filters.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ParticleFilterError<T> {
    #[error("small sample size {effective_sample_size} / {ensemble_size}")]
    SmallSampleSize {
        effective_sample_size: T,
        ensemble_size: usize,
    },
    #[error("simulations exceeded time limit {elapsed:.1} / {limit:.1} sec")]
    TimeLimitExceeded { elapsed: f64, limit: f64 },
}

/// A trait that is shared by all models that implement generic particle filter algorithms.
pub trait ParticleFilter<T, O, const P: usize, FMST, CSST>:
    OcnusModel<T, O, P, FMST, CSST>
where
    T: AsPrimitive<usize> + Copy + RealField,
    O: AddAssign + OcnusObser + Scalar + Zero,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
    /// Creates a new [`OcnusEnsbl`] using only valid ensemble members or optionally with an error
    /// metric (recommended). This function is intended to be used as the initialization step in
    /// any particle filtering algorithm.
    fn pf_initialize_ensemble<E>(
        &self,
        series: &ScObsSeries<T, O>,
        ensemble_sizes: (usize, usize),
        opt_metric: Option<(&E, T)>,
        settings: &mut ParticleFilterSettings<T>,
    ) -> Result<ParticleFilterResults<T, O, P, FMST, CSST>, OcnusError<T>>
    where
        E: Fn(&DVectorView<O>, &ScObsSeries<T, O>) -> T + Send + Sync,
    {
        let start = Instant::now();

        let (ensemble_size, sim_ensemble_size) = ensemble_sizes;

        let mut counter = 0;
        let mut iteration: usize = 0;

        let mut target_ensbl = OcnusEnsbl::new(ensemble_size);
        let mut target_output = DMatrix::<O>::zeros(series.len(), ensemble_size);
        let mut target_filter_values = Vec::<T>::with_capacity(ensemble_size);

        let mut temp_ensbl = OcnusEnsbl::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<O>::zeros(series.len(), sim_ensemble_size);

        while counter != target_ensbl.len() {
            self.initialize_ensbl(
                series,
                &mut temp_ensbl,
                None,
                settings.rseed + 17 * iteration as u64,
            )?;

            let mut flags = self.simulate_ensbl(
                series,
                &mut temp_ensbl,
                &mut temp_output.as_view_mut(),
                None::<&mut NullNoise<T>>,
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
                                        (-T::one()).sqrt()
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

            if start.elapsed().as_millis() as f64 / 1e3 > settings.sim_time_limit {
                info!(
                    "pf_initialize aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tsamples = {:.1} / {}",
                    (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    ensemble_size,
                );

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
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
            let eps_1 = filter_values_sorted
                [(T::from_usize(ensemble_size).unwrap() * settings.quantiles[0]).as_()];
            let eps_2 = filter_values_sorted
                [(T::from_usize(ensemble_size).unwrap() * settings.quantiles[1]).as_()];
            let eps_3 = if settings.quantiles[2] >= T::one() {
                filter_values_sorted[ensemble_size - 1]
            } else {
                filter_values_sorted
                    [(T::from_usize(ensemble_size).unwrap() * settings.quantiles[2]).as_()]
            };

            info!(
                "pf_initialize_data\n\tKL delta: n/a | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                eps_1,
                eps_2,
                eps_3,
                T::from_f64((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6)
                    .unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            );

            settings.rseed += 1;

            self.initialize_states_ensbl(series, &mut target_ensbl)?;

            self.simulate_ensbl(
                series,
                &mut target_ensbl,
                &mut target_output.as_view_mut(),
                None::<&mut NullNoise<T>>,
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
                T::from_f64((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6)
                    .unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
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
    T: RealField,
    O: Scalar,
{
    /// Result from pf_initialize
    Initialize(OcnusEnsbl<T, P, FMST, CSST>, DMatrix<O>),
    /// Advanced with results with additional values.
    SIR(OcnusEnsbl<T, P, FMST, CSST>, DMatrix<O>, Vec<T>),
    ///Advanced with results with additional values and respective quantiles
    ABC(OcnusEnsbl<T, P, FMST, CSST>, DMatrix<O>, Vec<T>, [T; 3]),
}

impl<T, O, const P: usize, FMST, CSST> ParticleFilterResults<T, O, P, FMST, CSST>
where
    T: RealField,
    O: Scalar,
{
    /// Return a reference to the underlying [`OcnusEnsbl`].
    pub fn get_ensbl(&self) -> &OcnusEnsbl<T, P, FMST, CSST> {
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

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<T>
where
    T: RealField,
{
    /// Effective sample size threshold factor.
    ///
    /// If the ESS is below this required limit (fraction), the algorithm returns an error.
    #[builder(default = T::from_f64(0.175).unwrap())]
    pub ess_factor: T,

    /// Multiplier for the transition kernel (covariance matrix), a higher value leads to a better
    /// exploration of the parameter  space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = T::from_usize(2).unwrap())]
    pub expl_factor: T,

    /// Iteration counter,
    #[builder(default = 0, setter(skip))]
    pub iter: usize,

    /// Random seed (initial & running).
    #[builder(default = 42)]
    pub rseed: u64,

    /// Maximum simulation time limit (in seconds).
    #[builder(default = 5.0)]
    pub sim_time_limit: f64,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,

    /// Quantile evaluations.
    #[builder(default = [
        T::from_f64(0.25).unwrap(),
        T::from_f64(0.50).unwrap(),
        T::from_f64(0.75).unwrap()
    ])]
    pub quantiles: [T; 3],
}
