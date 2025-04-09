//! Particle filters (or sequential Monte-Carlo algorithms).

mod abc;
mod sir;

pub use abc::*;
pub use sir::*;

use crate::{
    fevm::{FEVM, FEVMData, FEVMError, FEVMNoise},
    obser::{ObserVec, ScObsSeries},
    stats::PDFUnivariates,
    t_from,
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, Scalar, U1};
use num_traits::{AsPrimitive, Float, FromPrimitive};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, iter::Sum, ops::AddAssign, path::Path, time::Instant};
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

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<T, const N: usize>
where
    T: Copy + FromPrimitive + RealField + Scalar,
{
    /// Percentange of effective samples required for each iteration.
    #[builder(default = t_from!(0.175))]
    pub effective_sample_size_threshold_factor: T,

    /// Multiplier for the transition kernel (covariance matrix),
    /// a higher value leads to a better exploration of the parameter
    /// space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = t_from!(2.0))]
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
    #[builder(default = t_from!(5.0))]
    pub simulation_time_limit: T,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,

    /// Quantile evaluations.
    #[builder(default = [t_from!(0.25),t_from!(0.5),t_from!(0.75)])]
    pub quantiles: [T; 3],
}

/// An algebraic data type holding the results of any particle filtering method.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound = "
    T: for<'x> Deserialize<'x> + Serialize, 
    FS: for<'x> Deserialize<'x> + Serialize, 
    GS: for<'x> Deserialize<'x> + Serialize
")]
pub enum ParticleFilterResults<T, const P: usize, const N: usize, FS, GS>
where
    T: Clone + Float + Scalar,
{
    /// Default result holding the particles and output.
    Default(FEVMData<T, P, FS, GS>, DMatrix<ObserVec<T, N>>),
    /// Result holding additional likelihood values.
    ByLikelihood(FEVMData<T, P, FS, GS>, DMatrix<ObserVec<T, N>>, Vec<T>),
    /// Result holding additional error values and quantile statistics.
    ByMetric(
        FEVMData<T, P, FS, GS>,
        DMatrix<ObserVec<T, N>>,
        Vec<T>,
        [T; 3],
    ),
}

impl<T, const P: usize, const N: usize, FS, GS> ParticleFilterResults<T, P, N, FS, GS>
where
    T: Clone + Float + Scalar,
    Self: Serialize,
{
    /// Get the errors (for [`ParticleFilterResults::ByMetric`] only).
    pub fn get_errors(&self) -> Option<&Vec<T>> {
        match self {
            ParticleFilterResults::Default(..) => None,
            ParticleFilterResults::ByLikelihood(..) => None,
            ParticleFilterResults::ByMetric(_, _, errors, _) => Some(errors),
        }
    }

    /// Get the error quantiles (for [`ParticleFilterResults::ByMetric`] only).
    pub fn get_error_quantiles(&self) -> Option<&[T; 3]> {
        match self {
            ParticleFilterResults::Default(..) => None,
            ParticleFilterResults::ByLikelihood(..) => None,
            ParticleFilterResults::ByMetric(_, _, _, error_quantiles) => Some(error_quantiles),
        }
    }

    /// Access the underlying [`FEVMData`].
    pub fn get_fevmd(&self) -> &FEVMData<T, P, FS, GS> {
        match self {
            ParticleFilterResults::Default(fevmd, ..) => fevmd,
            ParticleFilterResults::ByLikelihood(fevmd, ..) => fevmd,
            ParticleFilterResults::ByMetric(fevmd, ..) => fevmd,
        }
    }

    /// Get the likelihoods (for [`ParticleFilterResults::ByLikelihood`] only).
    pub fn get_likelihoods(&self) -> Option<&Vec<T>> {
        match self {
            ParticleFilterResults::Default(..) => None,
            ParticleFilterResults::ByLikelihood(_, _, likelihoods) => Some(likelihoods),
            ParticleFilterResults::ByMetric(..) => None,
        }
    }

    /// Write data to a file.
    pub fn write<S>(&self, path: S) -> std::io::Result<()>
    where
        S: AsRef<Path>,
    {
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

            if t_from!(start.elapsed().as_millis() as f64 / 1e3) > settings.simulation_time_limit {
                return Err(FEVMError::ParticleFilter(
                    ParticleFilterError::TimeLimitExceeded {
                        elapsed: t_from!(start.elapsed().as_millis() as f64 / 1e3),
                        limit: settings.simulation_time_limit,
                    },
                ));
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
                t_from!((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6),
                t_from!(start.elapsed().as_millis() as f64 / 1e3),
            );

            settings.rseed += 1;

            self.fevm_simulate(series, &mut target, &mut target_output, None)?;

            Ok(ParticleFilterResults::ByMetric(
                target,
                target_output,
                target_filter_values,
                [eps_1, eps_2, eps_3],
            ))
        } else {
            info!(
                "pf_initialize_data\n\tKL delta: {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                t_from!((iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6),
                t_from!(start.elapsed().as_millis() as f64 / 1e3),
            );

            settings.rseed += 1;

            Ok(ParticleFilterResults::Default(target, target_output))
        }
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
