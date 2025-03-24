//! Particle filters (or sequential Monte-Carlo algorithms).

mod abc;
mod sir;

pub use abc::*;
pub use sir::*;

use crate::{
    fevm::{FEVM, FEVMData, FEVMError, FEVMNoise},
    obser::{ObserVec, ScObsSeries},
    stats::PDFUnivariates,
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, Scalar, U1};
use num_traits::{AsPrimitive, Float, FromPrimitive};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, iter::Sum, ops::AddAssign, time::Instant};
use thiserror::Error;

/// Errors associated with particle filters.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ParticleFilterError<T> {
    #[error("sampling from the distribution is too inefficient")]
    InefficientSampling,
    #[error("insufficiently large sample size {effective_sample_size} / {ensemble_size}")]
    SmallSampleSize {
        effective_sample_size: T,
        ensemble_size: usize,
    },
    #[error("iterations exceeded given time limit {elapsed} sec - {limit} sec")]
    TimeLimitExceeded { elapsed: T, limit: T },
}

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<T, const N: usize>
where
    T: Copy + FromPrimitive + RealField + Scalar,
{
    /// Percentange of effective samples required for each iteration.
    #[builder(default = T::from_f64(0.175).unwrap())]
    pub effective_sample_size_threshold_factor: T,

    /// Multiplier for the transition kernel (covariance matrix),
    /// a higher value leads to a better exploration of the parameter
    /// space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = T::from_f64(2.0).unwrap())]
    pub exploration_factor: T,

    /// Iteration counter,
    #[builder(default = 0, setter(skip))]
    pub iteration: usize,

    /// Noise generator.
    pub noise: FEVMNoise<T>,

    /// Random seed (initial & running).
    #[builder(default = 42)]
    pub rseed: u64,

    /// Time limit (in seconds).
    #[builder(default = T::from_f64(5.0).unwrap())]
    pub time_limit: T,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,

    /// Quantile evaluations.
    #[builder(default = [T::from_f64(0.25).unwrap(),T::from_f64(0.5).unwrap(),T::from_f64(0.75).unwrap()])]
    pub quantiles: [T; 3],
}

/// A data structure holding the results of any particle filtering method.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound = "
    T: for<'x> Deserialize<'x> + Serialize, 
    FS: for<'x> Deserialize<'x> + Serialize, 
    GS: for<'x> Deserialize<'x> + Serialize
")]
pub struct ParticleFilterResults<T, const P: usize, const N: usize, FS, GS>
where
    T: Clone + Float + Scalar,
{
    /// [`FEVMData`] object.
    pub fevmd: FEVMData<T, P, FS, GS>,

    /// Output array with noise.
    pub output: DMatrix<ObserVec<T, N>>,

    /// Error values.
    pub errors: Option<Vec<T>>,

    /// Error quantiles values.
    pub error_quantiles: Option<[T; 3]>,
}

impl<T, const P: usize, const N: usize, FS, GS> ParticleFilterResults<T, P, N, FS, GS>
where
    T: Clone + Float + Scalar,
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
pub trait ParticleFilter<T, const P: usize, const N: usize, FS, GS>: FEVM<T, P, N, FS, GS>
where
    T: AsPrimitive<usize> + Copy + Float + RealField + SampleUniform + Scalar + Serialize,
    FS: Clone + Default + Serialize + Send,
    GS: Clone + Default + Serialize + Send,
    StandardNormal: Distribution<T>,
{
    /// Creates a new [`FEVMData`], optionally using filter (recommended).
    ///
    /// This function is intended to be used.as_()he initialization step in any particle filtering algorithm.
    fn pf_initialize_ensemble<E>(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        opt_error_metric: Option<(&E, T)>,
        settings: &mut ParticleFilterSettings<T, N>,
    ) -> Result<ParticleFilterResults<T, P, N, FS, GS>, FEVMError<T>>
    where
        E: Fn(&DVectorView<ObserVec<T, N>>, &ScObsSeries<T, ObserVec<T, N>>) -> T + Send + Sync,
        T: for<'x> AddAssign<&'x T> + Default + SampleUniform + Send + Sync,
        StandardNormal: Distribution<T>,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;

        let mut target = FEVMData::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), ensemble_size);
        let mut target_filter_values = Vec::<T>::with_capacity(ensemble_size);

        let mut temp_data = FEVMData::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), sim_ensemble_size);

        while counter != target.size() {
            self.fevm_initialize(
                series,
                &mut temp_data,
                None::<&PDFUnivariates<T, P>>,
                settings.rseed + 17 * iteration,
            )?;

            let mut flags = self.fevm_simulate(series, &mut temp_data, &mut temp_output, None)?;

            let opt_filter_values = if let Some((filter, epsilon)) = opt_error_metric {
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

            if T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap() > settings.time_limit
            {
                return Err(FEVMError::ParticleFilter(
                    ParticleFilterError::TimeLimitExceeded {
                        elapsed: T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
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
                "pf_initialize_data\n\tKL delta: {:.3} | ln det {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                2.0,
                eps_1,
                eps_2,
                eps_3,
                T::from_f64((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6)
                    .unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            );

            settings.rseed += 1;

            (Some(target_filter_values), Some([eps_1, eps_2, eps_3]))
        } else {
            info!(
                "pf_initialize_data\n\tKL delta: {:.3} | ln det {:.3} \n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                2.0,
                T::from_f64((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6)
                    .unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            );

            settings.rseed += 1;

            (None, None)
        };

        self.fevm_simulate(series, &mut target, &mut target_output, None)?;

        Ok(ParticleFilterResults {
            fevmd: target,
            output: target_output,
            errors,
            error_quantiles,
        })
    }
}

/// Mean square error filter for particle filtering methods.
pub fn mean_square_filter<T, const N: usize>(
    out: &DVectorView<ObserVec<T, N>>,
    series: &ScObsSeries<T, ObserVec<T, N>>,
) -> T
where
    T: Default + Float + FromPrimitive + Scalar + Send + Sum + Sync,
{
    out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mean_square_error(scobs.observation()))
        .sum::<T>()
        / T::from_usize(series.count_observations()).unwrap()
}

/// Root mean square error filter for particle filtering methods.
pub fn root_mean_square_filter<T, const N: usize>(
    out: &DVectorView<ObserVec<T, N>>,
    series: &ScObsSeries<T, ObserVec<T, N>>,
) -> T
where
    T: Default + Float + FromPrimitive + Scalar + Send + Sum + Sync,
{
    Float::sqrt(
        out.into_iter()
            .zip(series)
            .map(|(out_vec, scobs)| out_vec.mean_square_error(scobs.observation()))
            .sum::<T>()
            / T::from_usize(series.count_observations()).unwrap(),
    )
}
