use crate::{
    base::{OcnusEnsbl, OcnusModel, OcnusModelError, ScObs},
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    obser::{NullNoise, ObserVec},
    stats::{Density, ParticleDensity},
};
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, SVector, U1};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Mul, Sub},
    time::Instant,
};

impl<M, T, const D: usize, FMST, CSST, const N: usize>
    ParticleFilter<M, T, D, FMST, CSST, ObserVec<T, N>>
where
    M: OcnusModel<T, D, FMST, CSST>,
    T: Copy
        + for<'x> Mul<&'x T, Output = T>
        + RealField
        + SampleUniform
        + for<'x> Sub<&'x T, Output = T>
        + Sum
        + for<'x> Sum<&'x T>,
    for<'x> &'x T: Mul<&'x T, Output = T>,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an sequential importance resampling particle filter algorithm.
    pub fn pf_sir_iter<LF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        obs_func: &OF,
        llh_func: &LF,
    ) -> Result<(T, T), ParticleFilterError<T>>
    where
        LF: Fn(&DVectorView<ObserVec<T, N>>) -> T + Sync,
        OF: Fn(
                &M,
                &ScObs<T>,
                &SVector<T, D>,
                &FMST,
                &CSST,
            ) -> Result<ObserVec<T, N>, OcnusModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        let ensemble_size = settings.ensemble_size;
        let series = &settings.series;
        let sim_ensemble_size = settings.simulation_ensemble_size;

        let mut counter = 0;
        let mut iteration: usize = 0;

        let mut target_ensbl = self.ensbl.take().expect("ensemble is not initialized");
        let mut target_output = self
            .ensbl_output
            .take()
            .unwrap_or(DMatrix::zeros(series.len(), ensemble_size));
        let mut target_filter_values = Vec::<T>::with_capacity(ensemble_size);

        let mut temp_ensbl = OcnusEnsbl::new(
            sim_ensemble_size,
            self.model.as_ref().unwrap().model_prior().get_range(),
        );
        let mut temp_output = DMatrix::zeros(series.len(), sim_ensemble_size);

        let mut interim_ensbl: OcnusEnsbl<T, D, FMST, CSST> = OcnusEnsbl::new(
            sim_ensemble_size,
            self.model.as_ref().unwrap().model_prior().get_range(),
        );
        let mut interim_likelihoods = Vec::<T>::with_capacity(sim_ensemble_size);

        // Copy the density and increase the size of the multivariate normal density estimate.
        let mut density_old = target_ensbl.ptpdf.clone() * settings.expl_factor;

        while counter != sim_ensemble_size {
            self.model
                .as_ref()
                .unwrap()
                .initialize_ensbl::<100, ParticleDensity<T, D>>(
                    &mut temp_ensbl,
                    Some(&density_old),
                    3 + self.rseed + 27 * iteration as u64,
                )?;

            self.model.as_ref().unwrap().simulate_ensbl(
                series,
                &mut temp_ensbl,
                obs_func,
                &mut temp_output.as_view_mut(),
                None::<&mut NullNoise<T>>,
            )?;

            let mut flags = vec![true; sim_ensemble_size];
            let filter_values = temp_output
                .par_column_iter()
                .zip(flags.par_iter_mut())
                .chunks(M::RCS)
                .map(|mut chunks| {
                    chunks
                        .iter_mut()
                        .map(|(out, flag)| {
                            let value = llh_func(&out.as_view::<Dyn, U1, U1, Dyn>());

                            **flag = value.is_finite();

                            value
                        })
                        .collect::<Vec<T>>()
                })
                .flatten()
                .collect::<Vec<T>>();

            let mut indices_valid = flags
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            debug!("valid: {}", indices_valid.len());

            // Remove excessive ensemble members.
            if counter + indices_valid.len() > interim_ensbl.len() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - interim_ensbl.len()
                );
                indices_valid.drain((interim_ensbl.len() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                interim_ensbl
                    .ptpdf
                    .particles_mut()
                    .set_column(counter + edx, &temp_ensbl.ptpdf.particles().column(*idx));

                interim_likelihoods.push(filter_values[*idx]);
            });

            counter += indices_valid.len();

            iteration += 1;

            if start.elapsed().as_millis() as f64 / 1e3 > settings.simulation_time_limit {
                info!(
                    "pf_sir_iter aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tsamples = {:.1} / {}",
                    (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    ensemble_size,
                );

                self.ensbl = Some(target_ensbl);
                self.ensbl_output = Some(target_output);

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    limit: settings.simulation_time_limit,
                });
            }
        }

        let llh_max = *interim_likelihoods
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
            .unwrap();

        let mut weights = interim_likelihoods
            .iter()
            .map(|llh| (*llh - llh_max).exp())
            .collect::<Vec<T>>();

        weights
            .par_iter_mut()
            .zip(interim_ensbl.ptpdf.particles().par_column_iter())
            .for_each(|(weight, params)| {
                *weight *= self
                    .model
                    .as_ref()
                    .unwrap()
                    .model_prior()
                    .relative_density(&params)
            });

        // Importance function weights.
        weights
            .par_iter_mut()
            .zip(interim_ensbl.ptpdf.particles().par_column_iter())
            .for_each(|(weight, params_new)| {
                *weight *= T::one()
                    / density_old
                        .particles()
                        .column_iter()
                        .zip(density_old.weights().iter())
                        .map(|(params_old, weight_old)| {
                            let delta = params_new - params_old;

                            (weight_old.ln()
                                - (delta.transpose()
                                    * density_old.covmatrix().pseudo_inverse()
                                    * delta)[(0, 0)])
                                .exp()
                        })
                        .sum::<T>();
            });

        let weights_total = weights.iter().sum::<T>();

        // Update interim weights.
        interim_ensbl
            .ptpdf
            .weights_mut()
            .iter_mut()
            .zip(weights.iter())
            .for_each(|(w_target, w_source)| *w_target = *w_source / weights_total);

        density_old *= T::one() / settings.expl_factor;

        let kld = interim_ensbl
            .ptpdf
            .kullback_leibler_divergence(&density_old)
            .expect("failed to compute the kl div");

        self.model.as_ref().unwrap().resample_ensbl(
            &mut target_ensbl,
            &interim_ensbl.ptpdf,
            self.rseed + 37,
        )?;

        // Find an index that is not constant
        let udx = 11;

        let uniques = target_ensbl
            .ptpdf
            .particles()
            .row(udx)
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .copied()
            .collect::<Vec<T>>()
            .len();

        self.model.as_ref().unwrap().simulate_ensbl(
            series,
            &mut target_ensbl,
            obs_func,
            &mut target_output.as_view_mut(),
            None::<&mut NullNoise<T>>,
        )?;

        target_filter_values
            .par_iter_mut()
            .zip(temp_output.par_column_iter())
            .chunks(M::RCS)
            .for_each(|mut chunks| {
                chunks.iter_mut().for_each(|(target, out)| {
                    **target = llh_func(&out.as_view::<Dyn, U1, U1, Dyn>());
                });
            });

        target_ensbl
            .ptpdf
            .weights_mut()
            .iter_mut()
            .for_each(|weight| *weight = T::one() / T::from_usize(ensemble_size).unwrap());

        target_ensbl.ptpdf.update_mvpdf();

        info!(
            "pf_sir_iter\n\tKL delta: {:.3} \n\tran {:2.3}M evaluations in {:.2} sec\n\tunique samples = {:.1} / {}",
            kld,
            T::from_f64((iteration * sim_ensemble_size * series.len()) as f64 / 1e6).unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            uniques,
            ensemble_size,
        );

        self.rseed += 1;

        self.ensbl = Some(target_ensbl);
        self.ensbl_output = Some(target_output);
        self.ensbl_errors = Some(target_filter_values);

        self.iter += 1;
        self.truns += iteration * sim_ensemble_size;

        Ok((T::zero(), T::zero()))
    }

    /// A loop of sequential importance re-sampling steps with various aborting criteria.
    pub fn pf_sir_loop<LF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        obs_func: &OF,
        llh_func: &LF,
    ) -> Result<(Vec<T>, Vec<T>), ParticleFilterError<T>>
    where
        LF: Fn(&DVectorView<ObserVec<T, N>>) -> T + Sync,
        OF: Fn(
                &M,
                &ScObs<T>,
                &SVector<T, D>,
                &FMST,
                &CSST,
            ) -> Result<ObserVec<T, N>, OcnusModelError<T>>
            + Sync,
    {
        let mut ess = Vec::new();
        let mut kld = Vec::new();

        for _ in 0..settings.max_iterations {
            let result = self.pf_sir_iter(settings, obs_func, llh_func);

            match result {
                Ok((new_ess, new_kld)) => {
                    ess.push(new_ess);
                    kld.push(new_kld);
                }
                Err(err) => return Err(err),
            }
        }

        Ok((ess, kld))
    }
}
