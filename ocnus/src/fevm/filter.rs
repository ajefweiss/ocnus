use crate::{
    ScObsSeries,
    fevm::{FEVM, FEVMData, FEVMError, FEVMNoiseGenerator, FEVMNoiseNull},
    obser::ObserVec,
    stats::{PDF, PDFParticles, ptpdf_importance_weighting},
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVector, DVectorView, Dyn, U1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<const N: usize, NG>
where
    NG: FEVMNoiseGenerator<N>,
{
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

/// ABC-SMC algorithm mode.
#[derive(Clone, Debug)]
pub enum ABCParticleFilterMode<'a, F, const N: usize>
where
    F: Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f64 + Send + Sync,
{
    /// ABC-SMC runs are filtered by a threshold value.
    Threshold((&'a F, f64)),

    /// ABC-SMC runs are filtered by a fixed acceptance rate.
    AcceptanceRate((&'a F, f64)),
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
    fn pf_initialize_data<F, T>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        opt_pdf: Option<&T>,
        opt_filter: Option<(&F, f64)>,
        rseed: u64,
    ) -> Result<FEVMData<P, FS, GS>, FEVMError>
    where
        for<'a> &'a T: PDF<P>,
        F: Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f64 + Send + Sync,
    {
        let mut counter = 0;
        let mut iteration = 0;

        let mut target = FEVMData::new(ensemble_size);
        let mut temp_data = FEVMData::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        while counter != target.size() {
            self.fevm_initialize(series, &mut temp_data, opt_pdf, rseed + 17 * iteration)?;

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

/// A trait that enables the use of approximate Bayesian computation (ABC) particle filter methods
/// for a [`FEVM`].
pub trait ABCParticleFilter<const P: usize, const N: usize, FS, GS>:
    ParticleFilter<P, N, FS, GS>
where
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
{
    /// Basic ABC-SMC algorithm (single iteration) with fixed acceptance ratio.
    fn abcpf_run_iteration<F, NG>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        mut fevmd: FEVMData<P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        mode: ABCParticleFilterMode<F, N>,
        settings: &mut ParticleFilterSettings<N, NG>,
    ) -> Result<ParticleFilterResults<P, N, FS, GS>, FEVMError>
    where
        F: Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f64 + Send + Sync,
        NG: FEVMNoiseGenerator<N>,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;

        let mut target_data = FEVMData::<P, FS, GS>::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);
        let mut target_filter_values = Vec::<f64>::with_capacity(ensemble_size);

        let mut temp_data = FEVMData::<P, FS, GS>::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle PDF from input and multiply by the exploration factor.
        let mut density_old = PDFParticles::from_particles(
            fevmd.params.as_view_mut(),
            self.model_prior().valid_range(),
            &mut fevmd.weights,
        )? * settings.exploration_factor;

        while counter != ensemble_size {
            self.fevm_initialize(
                series,
                &mut temp_data,
                Some(&density_old),
                settings.rseed + 23 * iteration,
            )?;

            let mut flags = self.fevm_simulate(
                series,
                &mut temp_data,
                &mut temp_output,
                Some((&settings.noise, settings.rseed + iteration * 17)),
            )?;

            let filter_values = match mode {
                ABCParticleFilterMode::AcceptanceRate((filter, accrate)) => {
                    if !(0.01..0.99).contains(&accrate) {
                        return Err(FEVMError::InvalidParameter {
                            name: "acceptance rate",
                            value: accrate,
                        });
                    }

                    let values = temp_output
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

                                        value
                                    } else {
                                        f64::NAN
                                    }
                                })
                                .collect::<Vec<f64>>()
                        })
                        .flatten()
                        .collect::<Vec<f64>>();

                    let values_sorted = values
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .copied()
                        .collect::<Vec<f64>>();

                    let mut epsilon = values_sorted[ensemble_size * accrate as usize];

                    if !epsilon.is_finite() {
                        epsilon = f64::MAX;
                    }

                    flags
                        .par_iter_mut()
                        .zip(values.par_iter())
                        .chunks(Self::RCS)
                        .for_each(|mut chunks| {
                            chunks.iter_mut().for_each(|(flag, value)| {
                                if **flag {
                                    **flag = **value < epsilon;
                                }
                            });
                        });

                    values
                }
                ABCParticleFilterMode::Threshold((filter, epsilon)) => temp_output
                    .par_column_iter()
                    .zip(flags.par_iter_mut())
                    .chunks(Self::RCS)
                    .map(|mut chunks| {
                        chunks
                            .iter_mut()
                            .map(|(out, flag)| {
                                if **flag {
                                    let value = filter(&out.as_view::<Dyn, U1, U1, Dyn>(), series);
                                    **flag = value < epsilon;

                                    value
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect::<Vec<f64>>()
                    })
                    .flatten()
                    .collect::<Vec<f64>>(),
            };

            let mut indices_valid = flags
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            debug!("valid: {}", indices_valid.len());

            // Remove excessive ensemble members.
            if counter + indices_valid.len() > target_data.size() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - target_data.size()
                );
                indices_valid.drain((target_data.size() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                target_data
                    .params
                    .set_column(counter + edx, &temp_data.params.column(*idx));

                target_output.set_column(counter + edx, &temp_output.column(*idx));

                target_filter_values.push(filter_values[*idx]);
            });

            counter += indices_valid.len();

            iteration += 1;
        }

        // Create a Particle PDF from our result.
        let mut density_new = PDFParticles::from_particles(
            target_data.params.as_view_mut(),
            self.model_prior().valid_range(),
            &mut target_data.weights,
        )?;

        ptpdf_importance_weighting(&mut density_new, &density_old, &self.model_prior());

        // Reset the covariance matrix in the old density.
        density_old *= 1.0 / settings.exploration_factor;

        // Compute the effective sample size.
        let effective_sample_size = 1.0
            / density_new
                .weights()
                .iter()
                .map(|value| value.powi(2))
                .sum::<f64>();

        let filter_values_sorted = target_filter_values
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .collect::<Vec<f64>>();

        // Compute quantiles for logging purposes.
        let eps_25 = filter_values_sorted[ensemble_size / 4];
        let eps_50 = filter_values_sorted[ensemble_size / 2];
        let eps_75 = filter_values_sorted[3 * ensemble_size / 4];

        info!(
            "abcpf_run_iteration\n\tKL delta: {:.3} | ln det {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M simulations in {:.2} sec\n\teffective sample size = {:.0} / {}",
            0.0,
            2.0,
            eps_25,
            eps_50,
            eps_75,
            (iteration as usize * sim_ensemble_size) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            effective_sample_size,
            ensemble_size,
        );

        Ok(ParticleFilterResults {
            fevmd: target_data,
            output: target_output,
            errors: target_filter_values,
            error_quantiles: [eps_25, eps_50, eps_75],
        })
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
