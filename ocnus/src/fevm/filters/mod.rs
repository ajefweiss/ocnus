//! Implementations of various particle filters.

mod abc;
mod mvll;

pub use abc::*;
pub use mvll::*;

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
use log::debug;
use nalgebra::{DMatrix, DVector, DVectorView, Dyn, U1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors associated with particle filters.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ParticleFilterError {
    #[error("sampling from the distribution is too inefficient")]
    InefficientSampling,
    #[error("insufficiently large sample size {effective_sample_size} / {ensemble_size}")]
    SmallSampleSize {
        effective_sample_size: f64,
        ensemble_size: usize,
    },
}

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<const N: usize, NG>
where
    NG: FEVMNoiseGenerator<N>,
{
    /// Percentange of effective samples required for each iteration.
    #[builder(default = 0.2)]
    pub effective_sample_size_threshold_factor: f64,

    /// Multiplier for the transition kernel (covariance matrix),
    /// a higher value leads to a better exploration of the parameter
    /// space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = 2.0)]
    pub exploration_factor: f64,

    /// Iteration counter,
    #[builder(default = 0, setter(skip))]
    pub iteration: usize,

    /// Noise generator.
    pub noise: NG,

    /// Random seed (initial & running).
    #[builder(default = 42)]
    pub rseed: u64,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,
}

/// A data structure holding the results of any particle filtering method.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticleFilterResults<const P: usize, const N: usize, FS, GS> {
    /// [`FEVMData`] object.
    pub fevmd: FEVMData<P, FS, GS>,

    /// Output array with noise.
    pub output: DMatrix<ObserVec<N>>,

    /// Error values.
    pub errors: Vec<f64>,

    /// Error quantiles values (25%, 50% = mean, 75%).
    pub error_quantiles: [f64; 3],
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
    fn pf_initialize_data<F>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        ensemble_size: usize,
        sim_ensemble_size: usize,

        opt_filter: Option<(&F, f64)>,
        rseed: u64,
    ) -> Result<FEVMData<P, FS, GS>, FEVMError>
    where
        F: Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f64 + Send + Sync,
    {
        let mut counter = 0;
        let mut iteration = 0;

        let mut target = FEVMData::new(ensemble_size);
        let mut temp_data = FEVMData::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        while counter != target.size() {
            self.fevm_initialize(
                series,
                &mut temp_data,
                None::<&PDFUnivariates<P>>,
                rseed + 17 * iteration,
            )?;

            let mut indices = self.fevm_simulate(
                series,
                &mut temp_data,
                &mut temp_output,
                None::<(&FEVMNoiseNull, u64)>,
            )?;

            if let Some((filter, theshold)) = opt_filter {
                temp_output
                    .par_column_iter()
                    .zip(indices.par_iter_mut())
                    .chunks(Self::RCS)
                    .map(|mut chunks| {
                        chunks
                            .iter_mut()
                            .map(|(out, flag)| {
                                if **flag {
                                    let value = filter(&out.as_view::<Dyn, U1, U1, Dyn>(), series);
                                    **flag = value < theshold;

                                    value
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect::<Vec<f64>>()
                    })
                    .flatten()
                    .collect::<Vec<f64>>();
            }

            let mut indices_valid = indices
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
            });

            counter += indices_valid.len();

            iteration += 1;
        }

        Ok(target)
    }
}

/// Mean square error filter for particle filtering methods.
pub fn mean_square_filter<const N: usize>(
    out: &DVectorView<ObserVec<N>>,
    series: &ScObsSeries<ObserVec<N>>,
) -> f64 {
    out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
        .sum::<f64>()
        / series.len() as f64
}

/// Normalized mean square error filter for particle filtering methods.
pub fn mean_square_normalized_filter<const N: usize>(
    out: &DVectorView<ObserVec<N>>,
    series: &ScObsSeries<ObserVec<N>>,
) -> f64 {
    out.into_iter()
        .zip(series)
        .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
        .sum::<f64>()
        / DVector::<ObserVec<N>>::zeros(out.len())
            .into_iter()
            .zip(series)
            .map(|(out_vec, scobs)| out_vec.mse(scobs.observation().unwrap()))
            .sum::<f64>()
}
