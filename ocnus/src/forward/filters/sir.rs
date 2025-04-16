use crate::{
    OcnusError, fXX,
    forward::{
        FSMEnsbl,
        filters::{
            ParticleFilter, ParticleFilterError, ParticleFilterResults, ParticleFilterSettings,
        },
    },
    math::{T, exp, powi},
    obser::{ObserVec, ObserVecNoise, ScObsSeries},
    prodef::{OcnusProDeF, ParticlesND},
};
use itertools::Itertools;
use log::{debug, info};
use nalgebra::DMatrix;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::{cmp::Ordering, time::Instant};

/// A trait that enables the use of a sequential importance resampling (SIR) particle filter method
/// for a [`OcnusFSM`](crate::forward::OcnusFSM) with a [`ObserVec`] observable.
pub trait SIRParticleFilter<T, const P: usize, const N: usize, FMST, CSST>:
    ParticleFilter<T, ObserVec<T, N>, P, FMST, CSST>
where
    T: fXX + SampleUniform,
    FMST: Default + Clone + Send,
    CSST: Default + Clone + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
    /// Basic SIR filter (single iteration) with multivariate likelihood.
    fn pf_sir_iter(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        ensbl: &FSMEnsbl<T, P, FMST, CSST>,
        ensemble_sizes: (usize, usize),
        settings: &mut ParticleFilterSettings<T>,
        noise: &mut ObserVecNoise<T, N>,
    ) -> Result<ParticleFilterResults<T, ObserVec<T, N>, P, FMST, CSST>, OcnusError<T>> {
        let start = Instant::now();

        let (ensemble_size, sim_ensemble_size) = ensemble_sizes;

        let mut counter = 0;
        let mut iteration = 0;

        let mut target_ensbl = FSMEnsbl::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), ensemble_size);

        let mut temp_ensbl = FSMEnsbl::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), sim_ensemble_size);

        let mut interim_ensbl = FSMEnsbl::<T, P, FMST, CSST>::new(sim_ensemble_size);
        let mut interim_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle OcnusProDeF from input and multiply by the exploration factor.
        let density_old = ParticlesND::from_particles(
            ensbl.params_array.as_view(),
            self.model_prior().get_valid_range(),
            ensbl.weights.clone(),
        )? * settings.expl_factor;

        while counter != sim_ensemble_size {
            self.fsm_initialize_ensbl(
                series,
                &mut temp_ensbl,
                Some(&density_old),
                settings.rseed + 27 * iteration as u64,
            )?;

            let mut indices_valid = self
                .fsm_simulate_ensbl(
                    series,
                    &mut temp_ensbl,
                    &mut temp_output.as_view_mut(),
                    None::<&mut ObserVecNoise<T, N>>,
                )?
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            debug!("valid: {}", indices_valid.len());

            // Remove excessive ensemble members.
            if counter + indices_valid.len() > temp_ensbl.len() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - interim_ensbl.len()
                );
                indices_valid.drain((interim_ensbl.len() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                interim_ensbl
                    .params_array
                    .set_column(counter + edx, &temp_ensbl.params_array.column(*idx));

                interim_output.set_column(counter + edx, &temp_output.column(*idx));
            });

            counter += indices_valid.len();

            iteration += 1;

            if T!(start.elapsed().as_millis() as f64 / 1e3) > settings.sim_time_limit {
                info!(
                    "pf_sir_iter aborted, time limit exceeded\n\tran {:2.3}M evaluations in {:.2} sec\n\tsamples = {:.1} / {}",
                    (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    ensemble_size,
                );

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: T!(start.elapsed().as_millis() as f64 / 1e3),
                    limit: settings.sim_time_limit,
                }
                .into());
            }
        }

        let likelihoods = interim_output
            .par_column_iter()
            .chunks(Self::RCS)
            .map(|mut chunks| {
                chunks
                    .iter_mut()
                    .map(|out| noise.multivariate_likelihood(out, series))
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
            .zip(temp_ensbl.params_array.par_column_iter())
            .for_each(|(weight, params)| *weight *= self.model_prior().density_rel(&params));

        let weights_total = weights.iter().sum::<T>();

        weights.iter_mut().for_each(|w| *w /= weights_total);

        // Create a Particle OcnusProDeF from the temporary simulations.
        let density_new = ParticlesND::from_particles(
            interim_ensbl.params_array.as_view(),
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

        self.fsm_resample_ensbl(series, &mut target_ensbl, &density_new, settings.rseed + 21)?;

        let uniques = target_ensbl
            .params_array
            .row(0)
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .copied()
            .collect::<Vec<T>>()
            .len();

        self.fsm_simulate_ensbl(
            series,
            &mut target_ensbl,
            &mut target_output.as_view_mut(),
            None::<&mut ObserVecNoise<T, N>>,
        )?;

        target_ensbl.weights =
            vec![T::one() / T::from_usize(ensemble_size).unwrap(); ensemble_size];

        info!(
            "pf_sir_iter success\n\tKL delta: {:.3} \n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}\n\tunique samples = {}",
            kld,
            (series.len() * sim_ensemble_size) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            effective_sample_size,
            ensemble_size,
            uniques,
        );

        settings.rseed += 1;

        Ok(ParticleFilterResults::SIR(
            target_ensbl,
            target_output,
            likelihoods,
        ))
    }
}
