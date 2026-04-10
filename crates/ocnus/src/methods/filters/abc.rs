use crate::{
    base::{Model, ModelError},
    math::{normalize, quantiles},
    methods::filters::{FilterError, FilterObject},
    obs::{
        conf::ObsTime,
        data::ObsVec,
        noise::{NullNoise, ObsNoise},
    },
};
use log::debug;
use nalgebra::{Const, RealField, SVector, SVectorView, U1};
use num_traits::AsPrimitive;
use prodef::{
    Density, domain::UDomain, multinormal::MultiNormalDensity, particle::ParticleDensity,
};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::{iter::Sum, time::Instant};

impl<T, OC, M, const D: usize, const P: usize, const N: usize>
    FilterObject<T, OC, ObsVec<T, N>, M, D, P>
where
    M: Model<T, D, P> + Sync,
    T: Copy + RealField + SampleUniform + Sum,
    OC: ObsTime<T> + Sync,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an approximate Bayesian Computation particle filter algorithm using a multinormal kernel.
    pub fn abc_mvnk<EF, OF, NM>(
        &mut self,
        err_func: (&EF, T),
        obs_func: &OF,
        noise: &mut NM,
    ) -> Result<T, FilterError<T>>
    where
        T: AsPrimitive<f64>,
        EF: Fn(&[ObsVec<T, N>], &[ObsVec<T, N>]) -> T + Sync,
        OF: Fn(
                &M,
                &OC,
                &SVectorView<T, P>,
                &M::FMST,
                &M::CSST,
            ) -> Result<ObsVec<T, N>, ModelError<T>>
            + Sync,
        NM: ObsNoise<T, OC, ObsVec<T, N>> + Sync,
    {
        let start = Instant::now();

        let flt_func = |arg1: &[ObsVec<T, N>], arg2: &[ObsVec<T, N>]| {
            let value = err_func.0(arg1, arg2);

            (value < err_func.1, value)
        };

        // Preserve the previous particles.
        let old_input = self.model_ensbl.input.clone();
        let old_weights = self.model_ensbl.opt_weights.clone().unwrap_or(vec![
            T::one()
                / T::from_usize(
                    self.model_ensbl.len()
                )
                .unwrap();
            self.model_ensbl
                .len()
        ]);

        let mut mvnk: MultiNormalDensity<T, Const<P>, UDomain<Const<P>>> =
            MultiNormalDensity::from_view::<U1, Const<P>>(
                &old_input.as_view(),
                UDomain::new(Const::<P>),
                self.model_ensbl.opt_weights.as_deref(),
            )
            .unwrap()
                * self.settings.exploration_factor;

        mvnk.mean = SVector::zeros();

        let ptpdf = ParticleDensity::from_view::<U1, Const<P>>(
            &old_input.as_view(),
            self.model.domain(),
            Some(&old_weights),
            Some(mvnk),
        )
        .unwrap();

        let (filter_values, iterations) =
            match self.filter(&flt_func, obs_func, &ptpdf, &mut Some(noise), 6361) {
                Ok(result) => result,
                Err(err) => {
                    // Restore previous particles.
                    self.model_ensbl.input = old_input;

                    return Err(err);
                }
            };

        let transitions = ptpdf.transition_weights(&self.model_ensbl.input);
        let new_weights = normalize(
            &self
                .model_ensbl
                .input
                .par_column_iter()
                .zip(transitions.par_iter())
                .map(|(params, transition)| {
                    self.model.prior().density(&params).unwrap() * *transition
                })
                .collect::<Vec<T>>(),
        );

        // Compute the effective sample size.
        let ess = T::one() / new_weights.iter().map(|value| value.powi(2)).sum::<T>();

        self.model_ensbl.opt_weights = Some(new_weights);

        if ess
            < T::from_usize(self.model_ensbl.len()).unwrap()
                * self.settings.effective_particle_threshold_factor
        {
            // Replace underlying particle density with previous particle density.
            self.model_ensbl.input = old_input.clone();
            self.model_ensbl.opt_weights = Some(old_weights);

            return Err(FilterError::EffectiveParticles(ess));
        }

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

        debug!(
            "abc_iter\n\teps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\teffective sample size = {:.1} / {}",
            q_low,
            q_mid,
            q_hgh,
            T::from_f64(
                (iterations
                    * self.model_ensbl.len()
                    * self.settings.simulation_ensemble_size_factor
                    * self.obs_ensbl.len()) as f64
                    / 1e6
            )
            .unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            ess,
            self.model_ensbl.len(),
        );

        self.model.initialize_states_ensbl(&mut self.model_ensbl)?;

        self.model.simulate_ensbl(
            &mut self.model_ensbl,
            &mut self.obs_ensbl,
            obs_func,
            &mut None::<&mut NullNoise<T>>,
        )?;

        self.errors = self.obs_ensbl.errors_func(err_func.0);

        self.iterations += 1;
        self.total_runs +=
            iterations * self.model_ensbl.len() * self.settings.simulation_ensemble_size_factor;

        Ok(ess)
    }

    /// A loop of approximate Bayesian Computation particle filtering steps with various aborting criteria.
    pub fn abc_mvnk_loop<NM, EF, OF>(
        &mut self,
        err_func: &EF,
        obs_func: &OF,
        error_quantile: T,
        noise: &mut NM,
    ) -> Result<(Vec<T>, Vec<T>), FilterError<T>>
    where
        T: AsPrimitive<f64> + AsPrimitive<usize>,
        NM: ObsNoise<T, OC, ObsVec<T, N>> + Sync,
        EF: Fn(&[ObsVec<T, N>], &[ObsVec<T, N>]) -> T + Sync,
        OF: Fn(
                &M,
                &OC,
                &SVectorView<T, P>,
                &M::FMST,
                &M::CSST,
            ) -> Result<ObsVec<T, N>, ModelError<T>>
            + Sync,
    {
        let mut ess = Vec::new();
        let mut eps = Vec::new();

        debug!(
            "abc_loop starting, maximum {} iterations",
            self.settings.max_iterations
        );

        for _ in 0..self.settings.max_iterations {
            let threshold = self.error_quantile(error_quantile).unwrap();

            let result = self.abc_mvnk((err_func, threshold), obs_func, noise);

            match result {
                Ok(new_ess) => {
                    ess.push(new_ess);
                    eps.push(threshold)
                }
                Err(err) => return Err(err),
            }
        }

        Ok((ess, eps))
    }
}
