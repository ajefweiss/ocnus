use crate::{
    fevms::{FEVMEnsbl, FEVModelError, ForwardEnsembleVectorModel, ModelObserArray, ModelObserVec},
    model::OcnusState,
    scobs::ScObs,
    stats::{CovMatrix, ParticlePDF, ParticleRefPDF},
    Fp, PMatrix,
};
use derive_builder::Builder;
use derive_more::derive::Deref;
use itertools::zip_eq;
use log::debug;
use log::info;
use nalgebra::{Const, DVector, DimAdd, Dyn, SVector, ToTypenum};
use ndarray::{Array2, ArrayViewMut2, Axis};
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, mem::replace, time::Instant};

/// A data structure that holds the results and settings for arbitrary particle filters.
#[derive(Builder, Debug, Default, Deref, Deserialize, Serialize)]
pub struct FEVMFilter<S, const P: usize, const N: usize>
where
    S: OcnusState,
{
    /// The underlying [`FEVMEnsbl`] object that is being filtered.
    #[deref]
    #[builder(setter(skip))]
    pub fevme: Option<FEVMEnsbl<S, P>>,

    // ------------------------------ Basic algorithm settings ------------------------------
    /// Size of the output FEVM ensemble. This value does not necessarily
    /// need to match the size of the underlying [`FEVMEnsbl`] ensemble.
    pub target_size: usize,

    /// Size of the FEVM ensemble that is used for temporary simulation runs.
    #[builder(default = "self.target_size.unwrap() * 8")]
    pub simulation_size: usize,

    /// Multiplier for the transition kernel (covariance matrix),
    /// a higher value leads to a better exploration of the parameter
    /// space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = 2.0)]
    pub exploration_factor: f32,

    /// Initial and running random seed value.
    pub seed_init: u64,

    // ------------------------------ Statistic and diagnostic fields ------------------------------
    #[builder(setter(skip))]
    seed: Option<u64>,

    /// Step counter,
    #[builder(setter(skip))]
    step: usize,

    /// Run counter,
    #[builder(setter(skip))]
    total_runs: usize,

    // ------------------------------ Temporary data structures ------------------------------
    #[builder(setter(skip))]
    fevme_output: Option<ModelObserArray<N>>,
    #[builder(setter(skip))]
    fevme_simulation: Option<FEVMEnsbl<S, P>>,
    #[builder(setter(skip))]
    fevme_simulation_output: Option<ModelObserArray<N>>,
}

/// A trait that enables shared behaviour of particle filtering methods for implementations of [`ForwardEnsembleVectorModel`].
pub trait ParticleFiltering<S, const P: usize, const N: usize>:
    ForwardEnsembleVectorModel<S, P, N>
where
    S: OcnusState,
    Self: Sync,
{
    /// Return a builder object for [`ParticleFilterData`].
    fn fevmfilter_builder() -> FEVMFilterBuilder<S, P, N> {
        FEVMFilterBuilder::default()
    }

    /// Run and filter simulations in the given ensemble with respect to the given [`ScObs`].
    ///
    /// Simulation runs are only valid if, and only if, output values are Some/None when they are Some/None in `scobs` respectively.
    fn pf_filter_simulations(
        &self,
        scobs: &ScObs<ModelObserVec<N>>,
        fevme: &mut FEVMEnsbl<S, P>,
        output: &mut ModelObserArray<N>,
    ) -> Result<Vec<bool>, FEVModelError> {
        self.fevm_simulate(scobs.as_scconf_slice(), fevme, &mut output.view_mut())?;

        let mut valid_indices_flags = vec![false; fevme.len()];

        // Collect indices that produce valid results and add random noise.
        output
            .axis_chunks_iter_mut(Axis(1), Self::RCS)
            .into_par_iter()
            .zip(valid_indices_flags.par_chunks_mut(Self::RCS))
            .for_each(|(mut chunks_out, flag_chunks)| {
                chunks_out
                    .axis_iter_mut(Axis(1))
                    .zip(flag_chunks.iter_mut())
                    .for_each(|(out, is_valid)| {
                        *is_valid =
                            out.iter()
                                .zip(scobs.as_obser_slice())
                                .fold(true, |acc, (o, r)| {
                                    acc & ((o.is_some() && r.is_some())
                                        || (o.is_none() && r.is_none()))
                                });
                    });
            });

        Ok(valid_indices_flags)
    }
}

/// A trait that enables the use of approximate Bayesian computation (ABC) particle filter methods for implementations of [`ForwardEnsembleVectorModel`].
pub trait ABCParticleFilter<S, const P: usize, const N: usize>:
    ForwardEnsembleVectorModel<S, P, N> + ParticleFiltering<S, P, N>
where
    S: OcnusState,
    Self: Sync,
{
    /// Basic ABC-SMC algorithm that uses the covariance matrix of the ensemble particles to produce the transition kernel.
    fn abc_filter(
        &self,
        scobs: &ScObs<ModelObserVec<N>>,
        filter: &mut FEVMFilter<S, P, N>,
        cov_noise: Option<&CovMatrix<Dyn>>,
        epsilon: f32,
    ) -> Result<Vec<f32>, FEVModelError> {
        Ok(Vec::new())
    }

    /// Compute weights according to the ABC particle filter algorithm.
    fn abc_weights(&self, pdf: &ParticleRefPDF<P>, pdf_old: &ParticleRefPDF<P>) -> Vec<f64> {
        let covm_inv = pdf_old.covm_inverse();

        let mut weights = prtd_new
            .particles
            .par_iter()
            .map(|new| {
                let mut w = 0.0;

                prtd_old
                    .particles
                    .iter()
                    .zip(prtd_old.weights.as_ref().unwrap())
                    .for_each(|(old, old_weight)| {
                        let diff = new - old;

                        let arg = (diff.transpose() * inverse * diff)[(0, 0)] as f64;

                        w += (old_weight.ln() - arg).exp();
                    });

                1.0 / w
            })
            .collect::<Vec<f64>>();

        // Apply prior distributions to the weights.
        self.model_prior()
            .as_uvnd_ref()
            .unwrap()
            .0
            .iter()
            .enumerate()
            .for_each(|(idx, prior)| match prior {
                UnivariateType::Normal { mean, variance, .. } => weights
                    .par_iter_mut()
                    .zip(prtd_new.particles.par_iter())
                    .for_each(|(w, p)| {
                        *w *= ((p[idx] - mean).powi(2) / 2.0 / variance).exp() as f64
                    }),
                UnivariateType::Reciprocal { .. } => weights
                    .par_iter_mut()
                    .zip(prtd_new.particles.par_iter())
                    .for_each(|(w, p)| *w *= 1.0 / p[idx] as f64),
                _ => (),
            });

        let total_weights = weights.iter().sum::<f64>();
        weights.iter_mut().for_each(|value| *value /= total_weights);

        weights
    }
}
