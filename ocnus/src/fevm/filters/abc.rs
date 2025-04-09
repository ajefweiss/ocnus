use crate::{
    fevm::{
        FEVMData, FEVMError, ParticleFilterError,
        filters::{ParticleFilter, ParticleFilterResults, ParticleFilterSettings},
    },
    obser::{ObserVec, ScObsSeries},
    stats::{PDF, PDFParticles},
    t_from,
};
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, Scalar, U1};
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use serde::Serialize;
use std::{iter::Sum, ops::AddAssign, time::Instant};

/// ABC-SMC algorithm mode.
#[derive(Clone, Debug)]
pub enum ABCParticleFilterMode<'a, T, E, const N: usize> {
    /// ABC-SMC runs are filtered by a threshold value.
    Threshold((&'a E, T)),

    /// ABC-SMC runs are filtered by a fixed acceptance rate.
    AcceptanceRate((&'a E, T)),
}

/// A trait that enables the use of approximate Bayesian computation (ABC) particle filter methods
/// for a [`FEVM`](crate::fevm::FEVM).
pub trait ABCParticleFilter<T, const P: usize, const N: usize, FS, GS>:
    ParticleFilter<T, P, N, FS, GS>
where
    T: for<'x> AddAssign<&'x T>
        + AsPrimitive<usize>
        + Copy
        + Default
        + Float
        + RealField
        + SampleUniform
        + Scalar
        + Serialize
        + Sum<T>
        + for<'x> Sum<&'x T>,
    FS: Clone + Default + Serialize + Send,
    GS: Clone + Default + Serialize + Send,
    StandardNormal: Distribution<T>,
{
    /// Basic ABC-SMC algorithm (single iteration) with fixed acceptance ratio.
    fn abcpf_run<E>(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &FEVMData<T, P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        mode: ABCParticleFilterMode<T, E, N>,
        settings: &mut ParticleFilterSettings<T, N>,
    ) -> Result<ParticleFilterResults<T, P, N, FS, GS>, FEVMError<T>>
    where
        E: Fn(&DVectorView<ObserVec<T, N>>, &ScObsSeries<T, ObserVec<T, N>>) -> T + Send + Sync,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;

        let mut target_data = FEVMData::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), ensemble_size);
        let mut target_filter_values = Vec::<T>::with_capacity(ensemble_size);

        let mut temp_data = FEVMData::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle PDF from input and multiply by the exploration factor.
        let mut density_old = PDFParticles::from_particles(
            fevmd.params.as_view(),
            self.model_prior().valid_range(),
            fevmd.weights.clone(),
        )? * settings.exploration_factor;

        while counter != ensemble_size {
            self.fevm_initialize(
                series,
                &mut temp_data,
                Some(&density_old),
                settings.rseed + 23 * iteration as u64,
            )?;

            let mut flags = self.fevm_simulate(
                series,
                &mut temp_data,
                &mut temp_output,
                Some(&mut settings.noise),
            )?;

            let filter_values = match mode {
                ABCParticleFilterMode::AcceptanceRate((filter, accrate)) => {
                    if !(t_from!(0.001)..t_from!(0.5)).contains(&accrate) {
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
                                        T::infinity()
                                    }
                                })
                                .collect::<Vec<T>>()
                        })
                        .flatten()
                        .collect::<Vec<T>>();

                    let values_sorted = values
                        .iter()
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .copied()
                        .collect::<Vec<T>>();

                    let mut epsilon =
                        values_sorted[(T::from_usize(sim_ensemble_size).unwrap() * accrate).as_()];

                    if !epsilon.is_finite() {
                        epsilon = Float::max_value();
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
                                    T::nan()
                                }
                            })
                            .collect::<Vec<T>>()
                    })
                    .flatten()
                    .collect::<Vec<T>>(),
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

            if t_from!(start.elapsed().as_millis() as f64 / 1e3) > settings.simulation_time_limit {
                return Err(FEVMError::ParticleFilter(
                    ParticleFilterError::TimeLimitExceeded {
                        elapsed: t_from!(start.elapsed().as_millis() as f64 / 1e3),
                        limit: settings.simulation_time_limit,
                    },
                ));
            }
        }

        // Create new density using importance weights.
        let density_new = PDFParticles::from_particles_and_ptpdf(
            target_data.params.as_view(),
            &density_old,
            &self.model_prior(),
        )?;

        // Reset the covariance matrix in the old density.
        density_old *= T::one() / settings.exploration_factor;

        let kld = density_new.kullback_leibler_divergence_mvpdf_estimate(&density_old);

        // Compute the effective sample size.
        let effective_sample_size = T::one()
            / density_new
                .weights()
                .iter()
                .map(|value| Float::powi(*value, 2))
                .sum::<T>();

        if effective_sample_size
            < T::from_usize(ensemble_size).unwrap()
                * settings.effective_sample_size_threshold_factor
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

        let mode_string = match mode {
            ABCParticleFilterMode::AcceptanceRate((_, accrate)) => {
                format!("accept rate: {:.3}", accrate)
            }
            ABCParticleFilterMode::Threshold((_, epsilon)) => format!("epsilon: {:.3}", epsilon),
        };

        info!(
            "abcpf_run ({})\n\tKL delta: {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
            mode_string,
            kld,
            eps_1,
            eps_2,
            eps_3,
            (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            effective_sample_size,
            ensemble_size,
        );

        settings.rseed += 1;

        Ok(ParticleFilterResults::ByMetric(
            target_data,
            target_output,
            target_filter_values,
            [eps_1, eps_2, eps_3],
        ))
    }
}
