//! Implementations of various particle filters.

mod abc;
mod bootstrap;

pub use abc::*;
pub use bootstrap::*;

use crate::{
    ScObsSeries,
    fevm::{
        FEVM, FEVMData, FEVMError,
        noise::{FEVMNoiseGenerator, FEVMNoiseNull},
    },
    obser::ObserVec,
    stats::PDFUnivariates,
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVector, DVectorView, Dyn, U1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, time::Instant};
use thiserror::Error;

/// Errors associated with particle filters.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ParticleFilterError {
    #[error("sampling from the distribution is too inefficient")]
    InefficientSampling,
    #[error("insufficiently large sample size {effective_sample_size} / {ensemble_size}")]
    SmallSampleSize {
        effective_sample_size: f32,
        ensemble_size: usize,
    },
    #[error("iterations exceeded given time limit {elapsed} sec - {limit} sec")]
    TimeLimitExceeded { elapsed: f32, limit: f32 },
}

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<const N: usize, NG>
where
    NG: FEVMNoiseGenerator<N>,
{
    /// Percentange of effective samples required for each iteration.
    #[builder(default = 0.175)]
    pub effective_sample_size_threshold_factor: f32,

    /// Multiplier for the transition kernel (covariance matrix),
    /// a higher value leads to a better exploration of the parameter
    /// space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = 2.0)]
    pub exploration_factor: f32,

    /// Iteration counter,
    #[builder(default = 0, setter(skip))]
    pub iteration: usize,

    /// Noise generator.
    pub noise: NG,

    /// Random seed (initial & running).
    #[builder(default = 42)]
    pub rseed: u64,

    /// Time limit (in seconds).
    #[builder(default = 5.0)]
    pub time_limit: f32,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,

    /// Quantile evaluations.
    #[builder(default = [0.25, 0.5, 0.75])]
    pub quantiles: [f32; 3],
}

/// A data structure holding the results of any particle filtering method.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticleFilterResults<const P: usize, const N: usize, FS, GS> {
    /// [`FEVMData`] object.
    pub fevmd: FEVMData<P, FS, GS>,

    /// Output array with noise.
    pub output: DMatrix<ObserVec<N>>,

    /// Error values.
    pub errors: Option<Vec<f32>>,

    /// Error quantiles values.
    pub error_quantiles: Option<[f32; 3]>,
}

impl<const P: usize, const N: usize, FS, GS> ParticleFilterResults<P, N, FS, GS>
where
    Self: Serialize,
{
    /// Write data to a file.
    pub fn write(&self, path: String) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}

/// A trait that enables the use of generic particle filter methods for a [`FEVM`].
pub trait ParticleFilter<const P: usize, const N: usize, FS, GS>: FEVM<P, N, FS, GS>
where
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Serialize + Send,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Serialize + Send,
{
    /// Creates a new [`FEVMData`], optionally using filter (recommended).
    ///
    /// This function is intended to be used as the initialization step in any particle filtering algorithm.
    fn pf_initialize_data<F, NG>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        opt_filter: Option<(&F, f32)>,
        settings: &mut ParticleFilterSettings<N, NG>,
    ) -> Result<ParticleFilterResults<P, N, FS, GS>, FEVMError>
    where
        F: Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f32 + Send + Sync,
        NG: FEVMNoiseGenerator<N>,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;

        let mut target = FEVMData::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<N>>::zeros(series.len(), ensemble_size);
        let mut target_filter_values = Vec::<f32>::with_capacity(ensemble_size);

        let mut temp_data = FEVMData::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        while counter != target.size() {
            self.fevm_initialize(
                series,
                &mut temp_data,
                None::<&PDFUnivariates<P>>,
                settings.rseed + 17 * iteration,
            )?;

            let mut flags = self.fevm_simulate(
                series,
                &mut temp_data,
                &mut temp_output,
                None::<(&FEVMNoiseNull, u64)>,
            )?;

            let opt_filter_values = if let Some((filter, epsilon)) = opt_filter {
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
                                        f32::NAN
                                    }
                                })
                                .collect::<Vec<f32>>()
                        })
                        .flatten()
                        .collect::<Vec<f32>>(),
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
            if counter + indices_valid.len() > target.size() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - target.size()
                );
                indices_valid.drain((target.size() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                target
                    .params
                    .set_column(counter + edx, &temp_data.params.column(*idx));

                if let Some(filter_values) = &opt_filter_values {
                    target_filter_values.push(filter_values[*idx]);
                }
            });

            counter += indices_valid.len();

            iteration += 1;

            if start.elapsed().as_millis() as f32 / 1e3 > settings.time_limit {
                return Err(FEVMError::ParticleFilter(
                    ParticleFilterError::TimeLimitExceeded {
                        elapsed: start.elapsed().as_millis() as f32 / 1e3,
                        limit: settings.time_limit,
                    },
                ));
            }
        }

        let (errors, error_quantiles) = if !target_filter_values.is_empty() {
            let filter_values_sorted = target_filter_values
                .iter()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .collect::<Vec<f32>>();

            // Compute quantiles for logging purposes.
            let eps_1 =
                filter_values_sorted[(ensemble_size as f32 * settings.quantiles[0]) as usize];
            let eps_2 =
                filter_values_sorted[(ensemble_size as f32 * settings.quantiles[1]) as usize];
            let eps_3 = if settings.quantiles[2] >= 1.0 {
                filter_values_sorted[ensemble_size - 1]
            } else {
                filter_values_sorted[(ensemble_size as f32 * settings.quantiles[2]) as usize]
            };

            info!(
                "pf_initialize_data\n\tKL delta: {:.3} | ln det {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                2.0,
                eps_1,
                eps_2,
                eps_3,
                (iteration as usize * sim_ensemble_size * series.len()) as f32 / 1e6,
                start.elapsed().as_millis() as f32 / 1e3,
            );

            settings.rseed += 1;

            (Some(target_filter_values), Some([eps_1, eps_2, eps_3]))
        } else {
            info!(
                "pf_initialize_data\n\tKL delta: {:.3} | ln det {:.3} \n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                2.0,
                (iteration as usize * sim_ensemble_size * series.len()) as f32 / 1e6,
                start.elapsed().as_millis() as f32 / 1e3,
            );

            settings.rseed += 1;

            (None, None)
        };

        self.fevm_simulate(
            series,
            &mut target,
            &mut target_output,
            None::<(&FEVMNoiseNull, u64)>,
        )?;

        Ok(ParticleFilterResults {
            fevmd: target,
            output: target_output,
            errors,
            error_quantiles,
        })
    }
}

/// Mean square error filter for particle filtering methods.
pub fn mean_square_filter<const N: usize>(
    out: &DVectorView<ObserVec<N>>,
    series: &ScObsSeries<ObserVec<N>>,
) -> f32 {
    out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
        .sum::<f32>()
        / series.len() as f32
}

/// Normalized mean square error filter for particle filtering methods.
pub fn mean_square_normalized_filter<const N: usize>(
    out: &DVectorView<ObserVec<N>>,
    series: &ScObsSeries<ObserVec<N>>,
) -> f32 {
    out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
        .sum::<f32>()
        / DVector::<ObserVec<N>>::zeros(out.len())
            .into_iter()
            .zip(series)
            .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
            .sum::<f32>()
}

/// Root mean square error filter for particle filtering methods.
pub fn root_mean_square_filter<const N: usize>(
    out: &DVectorView<ObserVec<N>>,
    series: &ScObsSeries<ObserVec<N>>,
) -> f32 {
    (out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
        .sum::<f32>()
        / series.len() as f32)
        .sqrt()
}

/// Normalized root mean square error filter for particle filtering methods.
pub fn root_mean_square_normalized_filter<const N: usize>(
    out: &DVectorView<ObserVec<N>>,
    series: &ScObsSeries<ObserVec<N>>,
) -> f32 {
    (out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
        .sum::<f32>()
        / DVector::<ObserVec<N>>::zeros(out.len())
            .into_iter()
            .zip(series)
            .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
            .sum::<f32>())
    .sqrt()
}
