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
    F: Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f32 + Send + Sync,
{
    /// ABC-SMC runs are filtered by a threshold value.
    Threshold((&'a F, f32)),

    /// ABC-SMC runs are filtered by a fixed acceptance rate.
    AcceptanceRate((&'a F, f32)),
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
    fn abcpf_run<F, NG>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        mut fevmd: FEVMData<P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        mode: ABCParticleFilterMode<F, N>,
        settings: &mut ParticleFilterSettings<N, NG>,
    ) -> Result<ParticleFilterResults<P, N, FS, GS>, FEVMError>
    where
        F: Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f32 + Send + Sync,
        NG: FEVMNoiseGenerator<N>,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;

        let mut target_data = FEVMData::<P, FS, GS>::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);
        let mut target_filter_values = Vec::<f32>::with_capacity(ensemble_size);

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
                                        filter(&out.as_view::<Dyn, U1, U1, Dyn>(), series)
                                    } else {
                                        f32::INFINITY
                                    }
                                })
                                .collect::<Vec<f32>>()
                        })
                        .flatten()
                        .collect::<Vec<f32>>();

                    let values_sorted = values
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .copied()
                        .collect::<Vec<f32>>();

                    let mut epsilon = values_sorted[(sim_ensemble_size as f32 * accrate) as usize];

                    if !epsilon.is_finite() {
                        epsilon = f32::MAX;
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
                                    f32::NAN
                                }
                            })
                            .collect::<Vec<f32>>()
                    })
                    .flatten()
                    .collect::<Vec<f32>>(),
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
                .sum::<f32>();

        if effective_sample_size
            < ensemble_size as f32 * settings.effective_sample_size_threshold_factor
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
            .collect::<Vec<f32>>();

        // Compute quantiles for logging purposes.
        let eps_25 = filter_values_sorted[ensemble_size / 4];
        let eps_50 = filter_values_sorted[ensemble_size / 2];
        let eps_75 = filter_values_sorted[3 * ensemble_size / 4];

        info!(
            "abcpf_run\n\tKL delta: {:.3} | ln det {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M simulations in {:.2} sec\n\teffective sample size = {:.0} / {}",
            0.0,
            2.0,
            eps_25,
            eps_50,
            eps_75,
            (iteration as usize * sim_ensemble_size) as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3,
            effective_sample_size,
            ensemble_size,
        );

        settings.rseed += 1;

        Ok(ParticleFilterResults {
            fevmd: target_data,
            output: target_output,
            errors: target_filter_values,
            error_quantiles: [eps_25, eps_50, eps_75],
        })
    }
}
