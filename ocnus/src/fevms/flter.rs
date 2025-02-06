use crate::{
    fevms::{FEVMEnsbl, FEVModelError, ForwardEnsembleVectorModel, ModelObserArray, ModelObserVec},
    model::OcnusState,
    scobs::ScObs,
    stats::{CovMatrix, ParticlePDF, ParticleRefPDF, PDF},
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
use rand_xoshiro::Xoshiro256PlusPlus;
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
        self.fevm_simulate(scobs.as_scconf_slice(), fevme, output)?;

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

    fn abc_filter_by_threshold(
        &self,
        scobs: &ScObs<ModelObserVec<N>>,
        fevme: &mut FEVMEnsbl<S, P>,
        output: &mut ModelObserArray<N>,
        cov_noise: Option<&CovMatrix<Dyn>>,
        epsilon: Fp,
        seed: u64,
    ) -> Vec<(usize, Fp)> {
        self.fevm_initialize_states_only(scobs.as_scconf_slice(), fevme);
        self.fevm_simulate(scobs.as_scconf_slice(), fevme, output);

        let mut valid_indices_flags = vec![false; fevme.len()];

        // Collect indices that produce valid results and add random noise.
        output
            .axis_chunks_iter_mut(Axis(1), Self::RCS)
            .into_par_iter()
            .zip(valid_indices_flags.par_chunks_mut(Self::RCS))
            .enumerate()
            .for_each(|(cdx, (mut ensbl_chunks, flag_chunks))| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + (cdx as u64 * 517));
                let norm = Normal::new(0.0, 1.0).unwrap();

                ensbl_chunks
                    .axis_iter_mut(Axis(1))
                    .zip(flag_chunks.iter_mut())
                    .for_each(|(mut obs, is_valid)| {
                        *is_valid = obs.iter().fold(true, |acc, o| acc & o.is_some());

                        // Only add noise for valid ensemble members and only if a covariance matrix is given.
                        if let Some(covm) = cov_noise {
                            if *is_valid {
                                for n in 0..N {
                                    let noise_vector = DVector::<Fp>::from_iterator(
                                        scobs.len(),
                                        (0..scobs.len()).map(|_| rng.sample(norm)),
                                    )
                                    .transpose()
                                        * covm.cholesky_ltm();

                                    zip_eq(obs.iter_mut(), noise_vector.iter()).for_each(
                                        |(obs, noise)| {
                                            let obs_ref = obs.as_mut().unwrap();
                                            obs_ref.0[n] += *noise;
                                        },
                                    )
                                }
                            }
                        }
                    });
            });

        let mut valid_indices_rmses = valid_indices_flags
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| match flag {
                true => Some((idx, 0.0)),
                false => None,
            })
            .collect::<Vec<(usize, Fp)>>();

        // Only accept results that fall below the rmse threshold.
        valid_indices_rmses.retain_mut(|(idx, rmse_value)| {
            let rmse = vecobs.rmse(
                &output
                    .column(*idx)
                    .iter()
                    .map(|v| v.clone().unwrap())
                    .collect::<Vec<ModelObserVec<N>>>(),
            );

            *rmse_value = rmse;

            matches!(
                rmse.partial_cmp(&epsilon)
                    .unwrap_or(std::cmp::Ordering::Less),
                std::cmp::Ordering::Less
            )
        });

        valid_indices_rmses
    }

    /// Compute weights according to the ABC particle filter algorithm.
    fn abc_weights<'a>(&self, pdf: &ParticleRefPDF<'a, P>, pdf_old: &ParticleRefPDF<P>) -> Vec<Fp> {
        let covm_inv = pdf_old.covm().inverse();

        let mut weights = pdf
            .particles()
            .par_column_iter()
            .map(|params_new| {
                let value = pdf_old
                    .particles()
                    .par_column_iter()
                    .zip(pdf_old.weights())
                    .map(|(params_old, weight_old)| {
                        let delta = params_new - params_old;

                        (weight_old.ln() - (delta.transpose() * covm_inv * delta)[(0, 0)]).exp()
                    })
                    .sum::<Fp>();

                1.0 / value
            })
            .collect::<Vec<Fp>>();

        let total = weights.iter().sum::<Fp>();

        weights
            .par_iter_mut()
            .zip(pdf.particles().par_column_iter())
            .for_each(|(weight, params)| {
                *weight *= self.model_prior().relative_density(&params) / total
            });

        weights
    }
}
