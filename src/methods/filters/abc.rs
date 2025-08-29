use crate::{
    base::{Model, ModelError, ScConf},
    math::{normalize, quantiles},
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    obsty::{NoiseModel, NullNoise, Observable},
    stats::Density,
};
use log::info;
use nalgebra::{RealField, SVector, Scalar};
use num_traits::{AsPrimitive, Zero};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::{iter::Sum, ops::AddAssign, time::Instant};

impl<T, M, const D: usize, OT> ParticleFilter<T, M, D, OT>
where
    M: Model<T, D> + Sync,
    T: Copy + RealField + SampleUniform + Sum,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    OT: AddAssign + Observable + Scalar + Zero,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an approximate Bayesian Computation particle filter algorithm.
    pub fn pf_abc<NM, EF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        noise: &mut NM,
        obs_func: &OF,
        err_func: (&EF, T),
    ) -> Result<(T, T), ParticleFilterError<T>>
    where
        M: Model<T, D>,
        T: AsPrimitive<f64>,
        NM: NoiseModel<T, OT> + Sync,
        EF: Fn(&[OT], &[OT]) -> T + Sync,
        OF: Fn(&M, &ScConf<T>, &SVector<T, D>, &M::FMST, &M::CSST) -> Result<OT, ModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        // Copy the density and increase the size of the multivariate normal density estimate.
        let mut density_old = self.ensbl.ptpdf.clone() * settings.expl_factor;

        let flt_func = |arg1: &[OT], arg2: &[OT]| {
            let value = err_func.0(arg1, arg2);

            (value < err_func.1, value)
        };

        let (filter_values, iterations) = match self.pf_filter(
            settings,
            &flt_func,
            obs_func,
            Some(&density_old),
            &mut Some(noise),
            settings.max_attempts,
            6361,
        ) {
            Ok(result) => result,
            Err(err) => {
                // Replace underlying particle density with previous particle density.
                density_old *= T::one() / settings.expl_factor;
                self.ensbl.ptpdf = density_old.clone();

                return Err(err);
            }
        };

        // Calculate new weights using prior and importance weights.
        let new_weights = normalize(
            &self
                .ensbl
                .ptpdf
                .par_iter()
                .map(|params| {
                    self.model.model_prior().relative_density(&params)
                        / density_old
                            .iter()
                            .zip(density_old.weights().iter())
                            .map(|(params_old, weight_old)| {
                                let delta = params - params_old;

                                (weight_old.ln()
                                    - (delta.transpose()
                                        * density_old.covmatrix().pseudo_inverse()
                                        * delta)[(0, 0)])
                                    .exp()
                            })
                            .sum::<T>()
                })
                .collect::<Vec<T>>(),
        );

        self.ensbl.ptpdf.update_weights(new_weights.as_slice());

        self.ensbl.ptpdf.update_mvpdf();

        density_old *= T::one() / settings.expl_factor;

        // Compute the effective sample size.
        let ess = T::one()
            / self
                .ensbl
                .ptpdf
                .weights()
                .iter()
                .map(|value| value.powi(2))
                .sum::<T>();

        if ess < T::from_usize(self.ensbl.len()).unwrap() * settings.eff_particle_threshold_factor {
            // Replace underlying particle density with previous particle density.
            self.ensbl.ptpdf = density_old.clone();

            return Err(ParticleFilterError::InsufficientParticles(ess));
        }

        let kld = self
            .ensbl
            .ptpdf
            .kullback_leibler_divergence(&density_old)
            .expect("failed to compute the kl div");

        // Compute quantiles for logging purposes.
        let quantiles = quantiles(
            &filter_values,
            &[
                T::from_f64(0.34).unwrap(),
                T::from_f64(0.50).unwrap(),
                T::from_f64(0.68).unwrap(),
            ],
        );

        let (q_low, q_mid, q_hgh) = (quantiles[0], quantiles[1], quantiles[2]);

        info!(
            "pf_abc_iter\n\tKL delta: {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
            kld,
            q_low,
            q_mid,
            q_hgh,
            T::from_f64(
                (iterations
                    * self.ensbl.len()
                    * settings.simulation_ensemble_size_factor
                    * self.obser.len()) as f64
                    / 1e6
            )
            .unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            ess,
            self.ensbl.len(),
        );

        self.model.initialize_states_ensbl(&mut self.ensbl)?;

        self.model.simulate_ensbl(
            &mut self.ensbl,
            &mut self.obser,
            obs_func,
            &mut None::<&mut NullNoise<T>>,
        )?;

        self.errors = self.obser.errors(err_func.0);

        self.iter += 1;
        self.truns += iterations * self.ensbl.len() * settings.simulation_ensemble_size_factor;

        Ok((ess, kld))
    }

    /// A loop of approximate Bayesian Computation particle filtering steps with various aborting criteria.
    #[allow(clippy::type_complexity)]
    pub fn pf_abc_loop<NM, EF, OF>(
        &mut self,
        error_quantile: T,
        settings: &ParticleFilterSettings<T>,
        noise: &mut NM,
        obs_func: &OF,
        err_func: &EF,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>), ParticleFilterError<T>>
    where
        M: Model<T, D>,
        T: AsPrimitive<f64> + AsPrimitive<usize>,
        NM: NoiseModel<T, OT> + Sync,
        EF: Fn(&[OT], &[OT]) -> T + Sync,
        OF: Fn(&M, &ScConf<T>, &SVector<T, D>, &M::FMST, &M::CSST) -> Result<OT, ModelError<T>>
            + Sync,
    {
        let mut ess = Vec::new();
        let mut kld = Vec::new();
        let mut eps = Vec::new();

        info!(
            "pf_abc_loop starting, maximum {} iterations",
            settings.max_iterations
        );

        for _ in 0..settings.max_iterations {
            let threshold = self.error_quantile(error_quantile).unwrap();

            let result = self.pf_abc(settings, noise, obs_func, (err_func, threshold));

            match result {
                Ok((new_ess, new_kld)) => {
                    ess.push(new_ess);
                    kld.push(new_kld);
                    eps.push(threshold)
                }
                Err(err) => return Err(err),
            }
        }

        Ok((ess, kld, eps))
    }
}
