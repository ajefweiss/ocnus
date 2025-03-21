use crate::{
    ScObsSeries,
    fevm::{
        FEVMData, FEVMError, FEVMNoiseGenerator, ParticleFilterError,
        filters::{ParticleFilter, ParticleFilterResults, ParticleFilterSettings},
    },
    obser::ObserVec,
    stats::{PDF, PDFParticles, ptpdf_importance_weighting},
};
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, U1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

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

/// A trait that enables the use of approximate Bayesian computation (ABC) particle filter methods
/// for a [`FEVM`].
pub trait ABCParticleFilter<const P: usize, const N: usize, FS, GS>:
    ParticleFilter<P, N, FS, GS>
where
    FS: OState
    GS: OState
{
    /// Basic ABC-SMC algorithm (single iteration) with fixed acceptance ratio.
    fn abcpf_run<F, NG>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &FEVMData<P, FS, GS>,
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
        let mut target_output = DMatrix::<ObserVec<N>>::zeros(series.len(), ensemble_size);
        let mut target_filter_values = Vec::<f64>::with_capacity(ensemble_size);

        let mut temp_data = FEVMData::<P, FS, GS>::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle PDF from input and multiply by the exploration factor.
        let mut density_old = PDFParticles::from_particles(
            fevmd.params.as_view(),
            self.model_prior().valid_range(),
            &fevmd.weights,
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
                    if !(0.001..0.5).contains(&accrate) {
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
                                        filter(&out.as_view::<Dyn, U1, U1, Dyn>(), series)
                                    } else {
                                        f64::INFINITY
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

                    let mut epsilon = values_sorted[(sim_ensemble_size as f64 * accrate) as usize];

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

            if start.elapsed().as_millis() as f64 / 1e3 > settings.time_limit {
                return Err(FEVMError::ParticleFilter(
                    ParticleFilterError::TimeLimitExceeded {
                        elapsed: start.elapsed().as_millis() as f64 / 1e3,
                        limit: settings.time_limit,
                    },
                ));
            }
        }

        // Create a Particle PDF from our result.
        let density_new = PDFParticles::from_particles(
            target_data.params.as_view(),
            self.model_prior().valid_range(),
            &target_data.weights,
        )?;

        debug!("weights update");
        // Update weights here.
        target_data.weights =
            ptpdf_importance_weighting(&density_new, &density_old, &self.model_prior());
        debug!("done weights");

        // Reset the covariance matrix in the old density.
        density_old *= 1.0 / settings.exploration_factor;

        // Compute the effective sample size.
        let effective_sample_size = 1.0
            / target_data
                .weights
                .iter()
                .map(|value| value.powi(2))
                .sum::<f64>();

        if effective_sample_size
            < ensemble_size as f64 * settings.effective_sample_size_threshold_factor
        {
            return Err(FEVMError::ParticleFilter(
                ParticleFilterError::SmallSampleSize {
                    effective_sample_size,
                    ensemble_size,
                },
            ));
        }

        let filter_values_sorted = target_filter_values
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .collect::<Vec<f64>>();

        // Compute quantiles for logging purposes.
        let eps_1 = filter_values_sorted[(ensemble_size as f64 * settings.quantiles[0]) as usize];
        let eps_2 = filter_values_sorted[(ensemble_size as f64 * settings.quantiles[1]) as usize];
        let eps_3 = if settings.quantiles[2] >= 1.0 {
            filter_values_sorted[ensemble_size - 1]
        } else {
            filter_values_sorted[(ensemble_size as f64 * settings.quantiles[2]) as usize]
        };

        let mode_string = match mode {
            ABCParticleFilterMode::AcceptanceRate((_, accrate)) => {
                format!("accept rate: {:.3}", accrate)
            }
            ABCParticleFilterMode::Threshold((_, epsilon)) => format!("epsilon: {:.3}", epsilon),
        };

        info!(
            "abcpf_run ({})\n\tKL delta: {:.3} | ln det {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
            mode_string,
            0.0,
            2.0,
            eps_1,
            eps_2,
            eps_3,
            (iteration as usize * sim_ensemble_size * series.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            effective_sample_size,
            ensemble_size,
        );

        settings.rseed += 1;

        Ok(ParticleFilterResults {
            fevmd: target_data,
            output: target_output,
            errors: Some(target_filter_values),
            error_quantiles: Some([eps_1, eps_2, eps_3]),
        })
    }
}
