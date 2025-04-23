//! Particle filtering methods.

mod abc;
mod dev;
mod sir;

use crate::{
    base::{OcnusEnsbl, OcnusModel, OcnusModelError, ScObs, ScObsSeries},
    obser::{NullNoise, OcnusObser},
    stats::{Density, MultivariateDensity},
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, SVector, Scalar, U1};
use num_traits::{AsPrimitive, Zero};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    iter::Sum,
    ops::{AddAssign, Mul, Sub},
    time::Instant,
};
use thiserror::Error;

/// Errors associated with particle filters methods.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ParticleFilterError<T> {
    #[error("self.model.as_ref().unwrap() error")]
    Model(#[from] OcnusModelError<T>),
    #[error("nothing was done")]
    Nothing,
    #[error("simulations exceeded time limit {elapsed:.1} / {limit:.1} sec")]
    TimeLimitExceeded { elapsed: f64, limit: f64 },
}

/// A particle filter data structure.
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize, 
    FMST: Serialize,
    CSST: Serialize,
    OT: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    FMST: Deserialize<'de>,
    CSST: Deserialize<'de>,
    OT: Deserialize<'de>"))]
pub struct ParticleFilter<M, T, const D: usize, FMST, CSST, OT>
where
    M: OcnusModel<T, D, FMST, CSST>,
    T: Copy + RealField + SampleUniform,
    FMST: Default + Clone + Send,
    CSST: Default + Clone + Send,
    OT: OcnusObser + Scalar,
{
    /// The self.model.as_ref().unwrap() ensemble.
    #[builder(default = None)]
    pub ensbl: Option<OcnusEnsbl<T, D, FMST, CSST>>,

    /// The self.model.as_ref().unwrap() ensemble output array.
    #[builder(default = None)]
    pub ensbl_output: Option<DMatrix<OT>>,

    /// The self.model.as_ref().unwrap() ensemble output errors.
    #[builder(default = None)]
    pub ensbl_errors: Option<Vec<T>>,

    /// Iteration counter,
    #[builder(default = 0, setter(skip))]
    pub iter: usize,

    /// Underlying model.
    #[builder(setter(strip_option))]
    pub model: Option<M>,

    /// Random seed (initial & running).
    #[builder(default = 42)]
    pub rseed: u64,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,
}

/// Particle filter loop settings, the settings may have different meanings depending on the specific filtering algorithm that is used.
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<T>
where
    T: Copy + RealField + SampleUniform,
{
    /// Target ensemble size.
    #[builder(default = 1024)]
    pub ensemble_size: usize,

    /// Quantile of error values to use for next iteration cut-off (ABC only).
    #[builder(default = T::from_f64(0.2).unwrap())]
    pub error_quantile: T,

    /// Multiplier for the transition kernel (covariance matrix), a higher value leads to a better
    /// exploration of the parameter  space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013), although lower values can also be used.
    #[builder(default = T::from_usize(2).unwrap())]
    pub expl_factor: T,

    /// Maximum number of iterations.
    #[builder(default = 10)]
    pub max_iterations: usize,

    /// Maximum number of (failed) iterations per iteration.
    #[builder(default = 2)]
    pub max_failed_iterations: usize,

    /// Observation series for simulations.
    pub series: ScObsSeries<T>,

    /// Simulation size used for each sub-iteration.
    #[builder(default = 4096)]
    pub simulation_ensemble_size: usize,

    /// Maximum simulation time limit (in seconds).
    #[builder(default = 5.0)]
    pub simulation_time_limit: f64,
}

impl<M, T, const D: usize, FMST, CSST, OT> ParticleFilter<M, T, D, FMST, CSST, OT>
where
    M: OcnusModel<T, D, FMST, CSST>,
    T: Copy
        + for<'x> Mul<&'x T, Output = T>
        + RealField
        + SampleUniform
        + for<'x> Sub<&'x T, Output = T>
        + Sum
        + for<'x> Sum<&'x T>,
    for<'x> &'x T: Mul<&'x T, Output = T>,
    FMST: Default + Clone + Send,
    CSST: Default + Clone + Send,
    OT: AddAssign + OcnusObser + Scalar + Zero,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// Compute a a quantile of the field `ensbl_errors`.
    pub fn error_quantile(&self, quantile: T) -> Option<T>
    where
        T: AsPrimitive<usize>,
    {
        assert!(
            (T::zero()..T::one()).contains(&quantile),
            "quantile must be within [0, 1]"
        );

        self.ensbl_errors.as_ref().map(|errors| {
            let mut errors_sorted = errors.clone();

            errors_sorted.sort_by(|a, b| {
                a.partial_cmp(b)
                    .expect("ensbl_errors cannot contain any NaN")
            });

            errors_sorted[(quantile * T::from_usize(errors.len()).unwrap()).as_()]
        })
    }

    /// Initialize the ensemble data using a error metric function `EF` and a threshold value.
    pub fn pf_initialize_ensbl<EF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        obs_func: &OF,
        err_func: (&EF, T),
    ) -> Result<(), ParticleFilterError<T>>
    where
        EF: Fn(&DVectorView<OT>) -> T + Send + Sync,
        OF: Fn(&M, &ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        let ensemble_size = settings.ensemble_size;
        let series = &settings.series;
        let sim_ensemble_size = settings.simulation_ensemble_size;

        let mut counter = 0;
        let mut iteration: usize = 0;

        let mut target_ensbl = self.ensbl.take().unwrap_or(OcnusEnsbl::new(
            ensemble_size,
            self.model.as_ref().unwrap().model_prior().get_range(),
        ));
        let mut target_output = self
            .ensbl_output
            .take()
            .unwrap_or(DMatrix::<OT>::zeros(series.len(), ensemble_size));
        let mut target_filter_values = Vec::<T>::with_capacity(ensemble_size);

        let mut temp_ensbl = OcnusEnsbl::new(
            sim_ensemble_size,
            self.model.as_ref().unwrap().model_prior().get_range(),
        );
        let mut temp_output = DMatrix::<OT>::zeros(series.len(), sim_ensemble_size);

        while counter != target_ensbl.len() {
            self.model
                .as_ref()
                .unwrap()
                .initialize_ensbl::<100, MultivariateDensity<T, D>>(
                    &mut temp_ensbl,
                    None,
                    self.rseed + 17 * iteration as u64,
                )?;

            self.model.as_ref().unwrap().simulate_ensbl(
                series,
                &mut temp_ensbl,
                obs_func,
                &mut temp_output.as_view_mut(),
                None::<&mut NullNoise<T>>,
            )?;

            let mut flags = vec![true; sim_ensemble_size];
            let filter_values = temp_output
                .par_column_iter()
                .zip(flags.par_iter_mut())
                .chunks(M::RCS)
                .map(|mut chunks| {
                    chunks
                        .iter_mut()
                        .map(|(out, flag)| {
                            let value = err_func.0(&out.as_view::<Dyn, U1, U1, Dyn>());

                            **flag = value < err_func.1;

                            value
                        })
                        .collect::<Vec<T>>()
                })
                .flatten()
                .collect::<Vec<T>>();

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
                    .ptpdf
                    .particles_mut()
                    .set_column(counter + edx, &temp_ensbl.ptpdf.particles().column(*idx));

                target_filter_values.push(filter_values[*idx]);
            });

            counter += indices_valid.len();

            iteration += 1;

            if start.elapsed().as_millis() as f64 / 1e3 > settings.simulation_time_limit {
                info!(
                    "pf_initialize aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tsamples = {:.1} / {}",
                    (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    ensemble_size,
                );

                self.ensbl = Some(target_ensbl);
                self.ensbl_output = Some(target_output);
                self.ensbl_errors = Some(target_filter_values);

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    limit: settings.simulation_time_limit,
                });
            }
        }

        if !target_filter_values.is_empty() {
            let filter_values_sorted = target_filter_values
                .iter()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .collect::<Vec<T>>();

            // Compute quantiles for logging purposes.
            let eps_1 = filter_values_sorted[(ensemble_size as f64 * 0.34) as usize];
            let eps_2 = filter_values_sorted[(ensemble_size as f64 * 0.50) as usize];
            let eps_3 = filter_values_sorted[(ensemble_size as f64 * 0.68) as usize];

            info!(
                "pf_initialize_data\n\tKL delta: n/a | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                eps_1,
                eps_2,
                eps_3,
                T::from_f64((iteration * sim_ensemble_size * series.len()) as f64 / 1e6).unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            );

            self.rseed += 1;

            self.model
                .as_ref()
                .unwrap()
                .initialize_states_ensbl(&mut target_ensbl)?;

            self.model.as_ref().unwrap().simulate_ensbl(
                series,
                &mut target_ensbl,
                obs_func,
                &mut target_output.as_view_mut(),
                None::<&mut NullNoise<T>>,
            )?;

            // Update covariance matrix.
            target_ensbl.ptpdf.update_mvpdf();

            self.ensbl = Some(target_ensbl);
            self.ensbl_output = Some(target_output);
            self.ensbl_errors = Some(target_filter_values);

            self.iter = 1;
            self.truns = iteration * sim_ensemble_size;

            Ok(())
        } else {
            info!(
                "pf_initialize_data\n\tKL delta: {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                T::from_f64((iteration * sim_ensemble_size * series.len()) as f64 / 1e6).unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            );

            self.rseed += 1;

            self.ensbl = Some(target_ensbl);
            self.ensbl_output = Some(target_output);

            Ok(())
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
