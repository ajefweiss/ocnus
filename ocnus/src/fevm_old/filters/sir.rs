use crate::{
    T,
    fevm::{
        FEVMEnsbl, FEVMError,
        filters::{
            ParticleFilter, ParticleFilterError, ParticleFilterResults, ParticleFilterSettings,
        },
    },
    obser::{ObserVec, ScObsSeries},
    prodef::{OcnusProDeF, ParticlesND},
};
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, RealField, Scalar};
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use serde::Serialize;
use std::{cmp::Ordering, iter::Sum, ops::AddAssign, time::Instant};

/// A trait that enables the use of a sequential importance resampling (SIR) particle filter method
/// for a [`FEVM`](crate::fevm::FEVM).
pub trait SIRParticleFilter<T, const P: usize, const N: usize, FS, GS>:
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
    /// Basic SIR filter (single iteration) with multivariate likelihood.
    fn sirpf_run(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &FEVMEnsbl<T, P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        settings: &mut ParticleFilterSettings<T, N>,
    ) -> Result<ParticleFilterResults<T, P, N, FS, GS>, FEVMError<T>> {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;

        let mut target_data = FEVMEnsbl::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), ensemble_size);

        let mut temp_data = FEVMEnsbl::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), sim_ensemble_size);

        let mut interim_data = FEVMEnsbl::<T, P, FS, GS>::new(sim_ensemble_size);
        let mut interim_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle OcnusProDeF from input and multiply by the exploration factor.
        let density_old = ParticlesND::from_particles(
            fevmd.params.as_view(),
            self.model_prior().get_valid_range(),
            fevmd.weights.clone(),
        )? * settings.exploration_factor;

        while counter != sim_ensemble_size {
            self.fevm_initialize(
                series,
                &mut temp_data,
                Some(&density_old),
                settings.rseed + 27 * iteration as u64,
            )?;

            let mut indices_valid = self
                .fevm_simulate(
                    series,
                    &mut temp_data,
                    &mut temp_output,
                    Some(&mut settings.noise),
                )?
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            debug!("valid: {}", indices_valid.len());

            // Remove excessive ensemble members.
            if counter + indices_valid.len() > temp_data.size() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - interim_data.size()
                );
                indices_valid.drain((interim_data.size() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                interim_data
                    .params
                    .set_column(counter + edx, &temp_data.params.column(*idx));

                interim_output.set_column(counter + edx, &temp_output.column(*idx));
            });

            counter += indices_valid.len();

            iteration += 1;

            if T!(start.elapsed().as_millis() as f64 / 1e3) > settings.simulation_time_limit {
                return Err(FEVMError::ParticleFilter(
                    ParticleFilterError::TimeLimitExceeded {
                        elapsed: T!(start.elapsed().as_millis() as f64 / 1e3),
                        limit: settings.simulation_time_limit,
                    },
                ));
            }
        }

        let likelihoods = interim_output
            .par_column_iter()
            .chunks(Self::RCS)
            .map(|mut chunks| {
                chunks
                    .iter_mut()
                    .map(|out| settings.noise.multivariate_likelihood(out, series))
                    .collect::<Vec<T>>()
            })
            .flatten()
            .collect::<Vec<T>>();

        let lh_max = *likelihoods
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
            .unwrap();

        let mut weights = likelihoods
            .iter()
            .map(|lh| exp!(*lh - lh_max))
            .collect::<Vec<T>>();

        weights
            .par_iter_mut()
            .zip(temp_data.params.par_column_iter())
            .for_each(|(weight, params)| *weight *= self.model_prior().density_rel(&params));

        let weights_total = weights.iter().sum::<T>();

        weights.iter_mut().for_each(|w| *w /= weights_total);

        // Create a Particle OcnusProDeF from the temporary simulations.
        let density_new = ParticlesND::from_particles(
            interim_data.params.as_view(),
            self.model_prior().get_valid_range(),
            weights,
        )?;

        let kld = density_new.kullback_leibler_divergence_mvpdf_estimate(&density_old);

        let effective_sample_size = T::one()
            / density_new
                .weights()
                .iter()
                .map(|v| powi!(*v, 2))
                .sum::<T>();

        self.fevm_initialize_resample(series, &mut target_data, &density_new, settings.rseed + 21)?;

        let uniques = target_data
            .params
            .row(0)
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .copied()
            .collect::<Vec<T>>()
            .len();

        self.fevm_simulate(series, &mut target_data, &mut target_output, None)?;

        target_data.weights = vec![T::one() / T::from_usize(ensemble_size).unwrap(); ensemble_size];

        info!(
            "sirpf_run\n\tKL delta: {:.3} \n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}\n\tunique samples = {}",
            kld,
            (series.len() * sim_ensemble_size) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            effective_sample_size,
            ensemble_size,
            uniques,
        );

        settings.rseed += 1;

        Ok(ParticleFilterResults::ByLikelihood(
            target_data,
            target_output,
            likelihoods,
        ))
    }
}
