use crate::{
    base::{OcnusEnsbl, OcnusModel, OcnusModelError, ScObs},
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    obser::{NullNoise, OcnusNoise, OcnusObser},
    stats::{Density, ParticleDensity},
};
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, SVector, Scalar, U1};
use num_traits::{AsPrimitive, Zero};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{AddAssign, Mul, Sub},
    time::Instant,
};

impl<M, T, const D: usize, FMST, CSST, OT> ParticleFilter<M, T, D, FMST, CSST, OT>
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
    OT: AddAssign + OcnusObser + Scalar + Zero,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an approximate Bayesian Computation particle filter algorithm.
    pub fn pf_abc_iter<NM, EF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        noise: &mut NM,
        obs_func: &OF,
        err_func: (&EF, T),
    ) -> Result<(T, T), ParticleFilterError<T>>
    where
        M: OcnusModel<T, D, FMST, CSST>,
        NM: OcnusNoise<T, OT> + Sync,
        EF: Fn(&DVectorView<OT>) -> T + Sync,
        OF: Fn(&M, &ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>
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
        let mut temp_output = DMatrix::<OT>::zeros(series.len(), sim_ensemble_size);

        // Copy the density and increase the size of the multivariate normal density estimate.
        let mut density_old = target_ensbl.ptpdf.clone() * settings.expl_factor;

        while counter != target_ensbl.len() {
            self.model
                .as_ref()
                .unwrap()
                .initialize_ensbl::<500, ParticleDensity<T, D>>(
                    &mut temp_ensbl,
                    Some(&density_old),
                    1 + self.rseed + 23 * iteration as u64,
                )?;

            self.model.as_ref().unwrap().simulate_ensbl(
                series,
                &mut temp_ensbl,
                obs_func,
                &mut temp_output.as_view_mut(),
                Some(noise),
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
                            let value = err_func.0(&out.as_view::<Dyn, U1, U1, Dyn>());

                            **flag = value < err_func.1;

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
            if counter + indices_valid.len() > target_ensbl.len() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - target_ensbl.len()
                );
                indices_valid.drain((target_ensbl.len() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                target_ensbl
                    .ptpdf
                    .particles_mut()
                    .set_column(counter + edx, &temp_ensbl.ptpdf.particles().column(*idx));

                target_filter_values.push(filter_values[*idx]);
            });

            counter += indices_valid.len();

            iteration += 1;

            if start.elapsed().as_millis() as f64 / 1e3 > settings.simulation_time_limit {
                info!(
                    "pf_abc_iter aborted\n\tran {:2.3}M evaluations in {:.2} sec\n\tsamples = {:.1} / {}",
                    (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    ensemble_size,
                );

                self.ensbl = Some(target_ensbl);
                self.ensbl_output = Some(target_output);
                self.ensbl_errors = Some(target_filter_values);

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    limit: settings.simulation_time_limit,
                });
            }
        }

        if !target_filter_values.is_empty() {
            let filter_values_sorted = target_filter_values
                .iter()
                .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .collect::<Vec<T>>();

            // Compute quantiles for logging purposes.
            let eps_1 = filter_values_sorted[(ensemble_size as f64 * 0.34) as usize];
            let eps_2 = filter_values_sorted[(ensemble_size as f64 * 0.50) as usize];
            let eps_3 = filter_values_sorted[(ensemble_size as f64 * 0.68) as usize];

            // Reset the multivariate normal density estimate of the old density.
            target_ensbl.ptpdf.from_transition(
                &density_old,
                Some(&self.model.as_ref().unwrap().model_prior()),
            );

            density_old *= T::one() / settings.expl_factor;

            // Compute the effective sample size.
            let ess = T::one()
                / target_ensbl
                    .ptpdf
                    .weights()
                    .iter()
                    .map(|value| value.powi(2))
                    .sum::<T>();

            let kld = target_ensbl
                .ptpdf
                .kullback_leibler_divergence(&density_old)
                .expect("failed to compute the kl div");

            info!(
                "pf_abc_iter\n\tKL delta: {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
                kld,
                eps_1,
                eps_2,
                eps_3,
                T::from_f64((iteration * sim_ensemble_size * series.len()) as f64 / 1e6).unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
                ess,
                ensemble_size,
            );

            self.rseed += 1;

            self.model
                .as_ref()
                .unwrap()
                .initialize_states_ensbl(&mut target_ensbl)?;

            self.model.as_ref().unwrap().simulate_ensbl(
                series,
                &mut target_ensbl,
                obs_func,
                &mut target_output.as_view_mut(),
                None::<&mut NullNoise<T>>,
            )?;

            self.ensbl = Some(target_ensbl);
            self.ensbl_output = Some(target_output);
            self.ensbl_errors = Some(target_filter_values);

            self.iter += 1;
            self.truns += iteration * sim_ensemble_size;

            Ok((ess, kld))
        } else {
            info!(
                "pf_abc_iter\n\tKL delta: {:.3}\n\tran {:2.3}M evaluations in {:.2} sec",
                0.0,
                T::from_f64((iteration * sim_ensemble_size * series.len()) as f64 / 1e6).unwrap(),
                T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            );

            self.rseed += 1;

            self.ensbl = Some(target_ensbl);
            self.ensbl_output = Some(target_output);

            Err(ParticleFilterError::Nothing)
        }
    }

    /// A loop of approximate Bayesian Computation particle filtering steps with various aborting criteria.
    pub fn pf_abc_loop<NM, EF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        noise: &mut NM,
        obs_func: &OF,
        err_func: &EF,
    ) -> Result<(Vec<T>, Vec<T>), ParticleFilterError<T>>
    where
        M: OcnusModel<T, D, FMST, CSST>,
        T: AsPrimitive<usize>,
        NM: OcnusNoise<T, OT> + Sync,
        EF: Fn(&DVectorView<OT>) -> T + Sync,
        OF: Fn(&M, &ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>
            + Sync,
    {
        let mut ess = Vec::new();
        let mut kld = Vec::new();

        for _ in 0..settings.max_iterations {
            let threshold = self.error_quantile(settings.error_quantile).unwrap();

            let result = self.pf_abc_iter(settings, noise, obs_func, (err_func, threshold));

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
