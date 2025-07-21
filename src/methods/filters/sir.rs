use crate::{
    base::{Model, ModelEnsbl, ModelError, Obser, ScConf},
    math::normalize,
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    obsty::{NullNoise, ObserVec},
    stats::Density,
};
use itertools::Itertools;
use log::info;
use nalgebra::{Dyn, RealField, SVector};
use num_traits::AsPrimitive;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{cmp::Ordering, iter::Sum, time::Instant};

impl<T, M, const D: usize, const N: usize> ParticleFilter<T, M, D, ObserVec<T, N>>
where
    T: Copy + RealField + SampleUniform + Sum,
    M: Clone + Model<T, D> + Sync,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an sequential importance resampling particle filter algorithm.
    pub fn pf_sir_iter<LF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        obs_func: &OF,
        llh_func: &LF,
    ) -> Result<(usize, T), ParticleFilterError<T>>
    where
        LF: Fn(&[ObserVec<T, N>], &[ObserVec<T, N>]) -> T + Sync,
        OF: Fn(
                &M,
                &ScConf<T>,
                &SVector<T, D>,
                &M::FMST,
                &M::CSST,
            ) -> Result<ObserVec<T, N>, ModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        // Create an interim [`ParticleFilter`] with a larger ensemble size.
        let mut sub_pf = self.clone();

        sub_pf.ensbl = ModelEnsbl::<T, M, D>::new(
            self.ensbl.len() * settings.simulation_ensemble_size_factor,
            Some(&self.model.get_range()),
        );
        sub_pf.obser = Obser::<T, ObserVec<T, N>>::new(
            self.obser.scobs().clone(),
            self.ensbl.len() * settings.simulation_ensemble_size_factor,
        );

        // Copy the density and increase the size of the multivariate normal density estimate.
        let mut density_old = self.ensbl.ptpdf.clone() * settings.expl_factor;

        let flt_func = |arg1: &[ObserVec<T, N>], arg2: &[ObserVec<T, N>]| {
            let value = llh_func(arg1, arg2);

            (value.is_finite(), value)
        };

        let (interim_likelihood_values, iterations) = sub_pf.pf_filter(
            settings,
            &flt_func,
            obs_func,
            Some(&density_old),
            &mut None::<&mut NullNoise<T>>,
            settings.max_attempts,
            27,
        )?;

        // Offset log-likelihood values to reduce precision issues.
        let llh_max = *interim_likelihood_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
            .unwrap();

        // Convert log-likelihood to likelihood and apply prior and importance weight.
        let interim_weights = normalize(
            &interim_likelihood_values
                .par_iter()
                .zip(sub_pf.ensbl.ptpdf.par_iter())
                .map(|(llh, params)| {
                    (*llh - llh_max).exp() * self.model.model_prior().relative_density(&params)
                        / density_old
                            .iter()
                            .zip(density_old.weights().iter())
                            .map(|(params_old, weight_old)| {
                                let delta = params - params_old;

                                (weight_old.ln()
                                    - density_old
                                        .covmatrix()
                                        .mahalanobis_distance::<Dyn, Dyn>(&delta.as_view()))
                                .exp()
                            })
                            .sum::<T>()
                })
                .collect::<Vec<T>>(),
        );

        // Compute the effective sample size from the interim weights
        let ess = T::one() / interim_weights.iter().map(|value| value.powi(2)).sum::<T>();
        info!("pf_sir_iter interim ess = {}", ess);

        // Update interim weights.
        sub_pf
            .ensbl
            .ptpdf
            .update_weights(interim_weights.as_slice());

        density_old *= T::one() / settings.expl_factor;

        let kld = sub_pf
            .ensbl
            .ptpdf
            .kullback_leibler_divergence(&density_old)
            .expect("failed to compute the kl div");

        self.model
            .resample_ensbl(&mut self.ensbl, &sub_pf.ensbl.ptpdf, self.rseed + 37)?;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.rseed);

        let constants = (&self.ensbl.ptpdf)
            .get_constants()
            .iter()
            .map(|c| c.is_finite())
            .collect::<Vec<bool>>();

        let mut udx = rng.random_range(0..D);

        // Select a dimension that is not fixed.
        while constants[udx] {
            udx = rng.random_range(0..D);
        }

        let uniques = self
            .ensbl
            .ptpdf
            .get_param_values(udx)
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .copied()
            .collect::<Vec<T>>()
            .len();

        self.model.simulate_ensbl(
            &mut self.ensbl,
            &mut self.obser,
            obs_func,
            &mut None::<&mut NullNoise<T>>,
        )?;

        self.ensbl.ptpdf.update_weights(&vec![
            T::one() / T::from_usize(self.ensbl.len()).unwrap();
            self.ensbl.len()
        ]);

        self.ensbl.ptpdf.update_mvpdf();

        info!(
            "pf_sir_iter\n\tKL delta: {:.3} \n\tran {:2.3}M evaluations in {:.2} sec\n\tunique samples = {:.1} / {}",
            kld,
            T::from_f64(
                (iterations
                    * self.ensbl.len()
                    * settings.simulation_ensemble_size_factor
                    * self.obser.len()) as f64
                    / 1e6
            )
            .unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            uniques,
            self.ensbl.len(),
        );

        self.rseed += 1;

        self.errors = self.obser.errors(llh_func);

        self.iter += 1;
        self.truns += iterations * self.ensbl.len() * settings.simulation_ensemble_size_factor;

        Ok((uniques, kld))
    }

    /// A loop of sequential importance re-sampling steps with various aborting criteria.
    pub fn pf_sir_loop<LF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        obs_func: &OF,
        llh_func: &LF,
    ) -> Result<(Vec<usize>, Vec<T>), ParticleFilterError<T>>
    where
        LF: Fn(&[ObserVec<T, N>], &[ObserVec<T, N>]) -> T + Sync,
        OF: Fn(
                &M,
                &ScConf<T>,
                &SVector<T, D>,
                &M::FMST,
                &M::CSST,
            ) -> Result<ObserVec<T, N>, ModelError<T>>
            + Sync,
    {
        let mut uniques = Vec::new();
        let mut kld = Vec::new();

        for _ in 0..settings.max_iterations {
            let result = self.pf_sir_iter(settings, obs_func, llh_func);

            match result {
                Ok((new_uniques, new_kld)) => {
                    uniques.push(new_uniques);
                    kld.push(new_kld);
                }
                Err(err) => return Err(err),
            }
        }

        Ok((uniques, kld))
    }
}
