use crate::{
    OcnusEnsbl, OcnusError,
    methods::{ParticleFilter, ParticleFilterError, ParticleFilterResults, ParticleFilterSettings},
    obser::{OcnusNoise, OcnusObser, ScObsSeries},
};
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, Scalar, U1};
use num_traits::{AsPrimitive, Zero};
use ocnus_stats::{Density, ParticlesND};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{AddAssign, Deref, DerefMut},
    time::Instant,
};

/// ABC particle filter settings.
#[derive(Clone, Debug)]
pub enum ABCSettings<'a, T, E>
where
    T: RealField,
{
    /// ABC runs are filtered by a fixed acceptance rate.
    AcceptanceRate((ParticleFilterSettings<T>, &'a E, f64)),

    /// ABC runs are filtered by a threshold value.
    Threshold((ParticleFilterSettings<T>, &'a E, T)),
}

impl<'a, T, E> Deref for ABCSettings<'a, T, E>
where
    T: RealField,
{
    type Target = ParticleFilterSettings<T>;

    fn deref(&self) -> &Self::Target {
        match self {
            ABCSettings::AcceptanceRate((pf_settings, ..)) => pf_settings,
            ABCSettings::Threshold((pf_settings, ..)) => pf_settings,
        }
    }
}

impl<'a, T, E> DerefMut for ABCSettings<'a, T, E>
where
    T: RealField,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            ABCSettings::AcceptanceRate((pf_settings, ..)) => pf_settings,
            ABCSettings::Threshold((pf_settings, ..)) => pf_settings,
        }
    }
}

/// A trait that enables the use of approximate Bayesian computation (ABC) particle filter methods
/// for a [`OcnusModel`](`crate::OcnusModel`) with an [`ObserVec`](`crate::obser::ObserVec`) observable.
pub trait ABCParticleFilter<T, O, const P: usize, FMST, CSST>:
    ParticleFilter<T, O, P, FMST, CSST>
where
    T: AsPrimitive<usize> + Copy + RealField + SampleUniform + Sum + for<'x> Sum<&'x T>,
    O: AddAssign + OcnusObser + Scalar + Zero,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
    /// Single iteration of a basic approximate bayesian computation particle filter algorithm.
    fn pf_abc_iter<E, N>(
        &self,
        series: &ScObsSeries<T, O>,
        ensbl: &OcnusEnsbl<T, P, FMST, CSST>,
        ensemble_sizes: (usize, usize),
        settings: &mut ABCSettings<T, E>,
        noise: &mut N,
    ) -> Result<ParticleFilterResults<T, O, P, FMST, CSST>, OcnusError<T>>
    where
        E: Fn(&DVectorView<O>, &ScObsSeries<T, O>) -> T + Send + Sync,
        N: OcnusNoise<T, O> + Sync,
    {
        let start = Instant::now();

        let (ensemble_size, sim_ensemble_size) = ensemble_sizes;

        let mut counter = 0;
        let mut iteration = 0;

        let mut target_ensbl = OcnusEnsbl::new(ensemble_size);
        let mut target_output = DMatrix::<O>::zeros(series.len(), ensemble_size);
        let mut target_filter_values = Vec::<T>::with_capacity(ensemble_size);

        let mut temp_ensbl = OcnusEnsbl::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<O>::zeros(series.len(), sim_ensemble_size);

        let ensbl_params_view = ensbl.params_array.as_view();

        // Create a Particle Density from input and multiply by the exploration factor.
        let mut density_old = ParticlesND::from_particles(
            &ensbl_params_view,
            self.model_prior().get_valid_range(),
            ensbl.weights.clone(),
        )
        .unwrap()
            * settings.expl_factor;

        while counter != ensemble_size {
            self.initialize_ensbl(
                series,
                &mut temp_ensbl,
                Some(&density_old),
                settings.rseed + 23 * iteration as u64,
            )?;

            let mut flags = self.simulate_ensbl(
                series,
                &mut temp_ensbl,
                &mut temp_output.as_view_mut(),
                Some(noise),
            )?;

            let filter_values = match settings {
                ABCSettings::AcceptanceRate((_, metric, accrate)) => {
                    if !(0.001..0.5).contains(accrate) {
                        panic!("acceptance rate is too low")
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
                                        metric(&out.as_view::<Dyn, U1, U1, Dyn>(), series)
                                    } else {
                                        T::one() / T::zero()
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

                    let mut epsilon = values_sorted[(sim_ensemble_size as f64 * *accrate) as usize];

                    if !epsilon.is_finite() {
                        epsilon = T::max_value().unwrap()
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
                ABCSettings::Threshold((.., metric, epsilon)) => temp_output
                    .par_column_iter()
                    .zip(flags.par_iter_mut())
                    .chunks(Self::RCS)
                    .map(|mut chunks| {
                        chunks
                            .iter_mut()
                            .map(|(out, flag)| {
                                if **flag {
                                    let value = metric(&out.as_view::<Dyn, U1, U1, Dyn>(), series);
                                    **flag = value < *epsilon;

                                    value
                                } else {
                                    (-T::one()).sqrt()
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
                    .params_array
                    .set_column(counter + edx, &temp_ensbl.params_array.column(*idx));

                target_output.set_column(counter + edx, &temp_output.column(*idx));

                target_filter_values.push(filter_values[*idx]);
            });

            counter += indices_valid.len();

            iteration += 1;

            if start.elapsed().as_millis() as f64 / 1e3 > settings.sim_time_limit {
                info!(
                    "pf_abc_iter aborted, time limit exceeded\n\tran {:2.3}M evaluations in {:.2} sec\n\tsamples = {:.1} / {}",
                    (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                    start.elapsed().as_millis() as f64 / 1e3,
                    counter,
                    ensemble_size,
                );

                return Err(ParticleFilterError::TimeLimitExceeded {
                    elapsed: start.elapsed().as_millis() as f64 / 1e3,
                    limit: settings.sim_time_limit,
                }
                .into());
            }
        }

        let target_params_view = target_ensbl.params_array.as_view();

        // Create new density using importance weights.
        let density_new = ParticlesND::from_particles_and_ptpdf(
            &target_params_view,
            &density_old,
            &self.model_prior(),
        )
        .unwrap();

        // Reset the covariance matrix in the old density.
        density_old *= T::one() / settings.expl_factor;

        let kld = density_new.kullback_leibler_divergence_estimate(&density_old);

        // Compute the effective sample size.
        let effective_sample_size = T::one()
            / density_new
                .weights()
                .iter()
                .map(|value| value.powi(2))
                .sum::<T>();

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

        if effective_sample_size < T::from_usize(ensemble_size).unwrap() * settings.ess_factor {
            info!(
                "pf_abc_iter aborted, too small sample size\n\tKL delta: {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
                kld,
                eps_1,
                eps_2,
                eps_3,
                (iteration * sim_ensemble_size * series.len()) as f64 / 1e6,
                start.elapsed().as_millis() as f64 / 1e3,
                effective_sample_size,
                ensemble_size,
            );

            return Err(ParticleFilterError::SmallSampleSize {
                effective_sample_size,
                ensemble_size,
            }
            .into());
        }

        let mode_string = match settings {
            ABCSettings::AcceptanceRate((.., accrate)) => {
                format!("accept rate: {:.3}", accrate)
            }
            ABCSettings::Threshold((.., epsilon)) => {
                format!("epsilon: {:.3}", epsilon)
            }
        };

        info!(
            "pf_abc_iter success ({})\n\tKL delta: {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
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

        Ok(ParticleFilterResults::ABC(
            target_ensbl,
            target_output,
            target_filter_values,
            [eps_1, eps_2, eps_3],
        ))
    }
}
