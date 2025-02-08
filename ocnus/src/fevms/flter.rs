use crate::{
    fevms::{FEVMEnsbl, FEVModelError, ForwardEnsembleVectorModel, ModelObserArray, ModelObserVec},
    model::OcnusState,
    scobs::ScObs,
    stats::{CovMatrix, PUnivariatePDF},
    Fp, PMatrix,
};
use derive_builder::Builder;
use derive_more::derive::Deref;
use itertools::{zip_eq, Itertools};
use log::info;
use nalgebra::{Const, DVector};
use ndarray::Axis;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, mem::replace, time::Instant};

/// A data structure that holds the results and settings for arbitrary particle filters.
#[derive(Builder, Debug, Default, Deref, Deserialize, Serialize)]
pub struct FEVMData<S, const P: usize, const N: usize> {
    /// The underlying [`FEVMEnsbl`] object that is being filtered.
    #[deref]
    #[builder(setter(skip))]
    pub fevme: Option<FEVMEnsbl<S, P>>,

    /// The running seed.
    #[builder(setter(skip))]
    seed: Option<u64>,

    /// Step counter,
    #[builder(setter(skip))]
    step: usize,

    /// Run counter,
    #[builder(setter(skip))]
    total_runs: usize,

    // Temporary data structures.
    // Stored here to guarantee that we do not need
    // to re-initialize large arrays every time a filtering
    // function is called.
    #[builder(setter(skip))]
    fevme_output: Option<ModelObserArray<N>>,
    #[builder(setter(skip))]
    fevme_simulation: Option<FEVMEnsbl<S, P>>,
    #[builder(setter(skip))]
    fevme_simulation_output: Option<ModelObserArray<N>>,
}

/// A data structure holding the results of any particle filtering method.
pub struct ParticleFilteringResults {
    /// Log-determinant of the covariance matrix of the outut probability density functions.
    pub ln_determinant: Fp,

    /// Effective sample size of the output ensemble.
    pub effective_sample_size: Fp,

    /// Estimated multivariate normal Kullback-Leibler divergence with respect to the input and output probability density functions.
    pub kullback_leibler_div: Fp,

    /// Depending on the particular PF method this vector contains either the errors or likelihoods of the filtering result.
    pub result: Vec<Fp>,

    /// Total runs performed.
    pub total_runs: usize,

    /// Total time elapsed (in milliseconds).
    pub total_runtime: Fp,
}

/// A trait that enables shared behaviour of particle filtering methods for implementations of [`ForwardEnsembleVectorModel`].
pub trait ParticleFiltering<S, const P: usize, const N: usize>:
    ForwardEnsembleVectorModel<S, P, N>
where
    S: OcnusState,
    Self: Sync,
{
    /// Return a default builder object.
    fn fevmfilter_builder(&self) -> FEVMDataBuilder<S, P, N> {
        FEVMDataBuilder::default()
    }

    /// Run and filter simulations in the given ensemble with respect to the given [`ScObs`].
    ///
    /// Simulation runs are only valid if, and only if, output values are Some/None when they are Some/None in `scobs` respectively.
    fn pf_filter_simulations(
        &self,
        scobs: &ScObs<ModelObserVec<N>>,
        fevmd: &mut FEVMData<S, P, N>,
        output: &mut ModelObserArray<N>,
    ) -> Result<Vec<bool>, FEVModelError> {
        self.fevm_simulate(
            scobs.as_scconf_slice(),
            fevmd.fevme.as_mut().unwrap(),
            output,
        )?;

        let mut valid_indices_flags = vec![false; fevmd.fevme.as_ref().unwrap().len()];

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

/// ABC-SMC settings structure.
#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
pub struct ABCSettings {
    /// Multiplier for the transition kernel (covariance matrix),
    /// a higher value leads to a better exploration of the parameter
    /// space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = 2.0)]
    pub exploration_factor: Fp,

    /// Initial random seed.
    #[builder(default = 42)]
    pub initial_seed: u64,

    /// The mode that is used to gather accepted simulations.
    pub mode: ABCMode,

    /// The covariance matrix that is used to generate the multivariate noise
    /// that is super imposed on top of the model outputs.
    pub noise_covm: Option<CovMatrix>,

    /// Size of the FEVM ensemble that is used for temporary simulation runs
    /// at each iteration.
    #[builder(default = "self.target_size.unwrap() * 8")]
    pub simulation_size: usize,

    /// The type of summary statistic that is used to compare
    /// with the threshold epsilon values.
    pub summary_statistic: ABCSummaryStatistic,

    /// Size of the output FEVM ensemble. Does not necessarily
    /// need to match the size of the input ensemble.
    pub target_size: usize,
}

/// ABC-SMC algorithm mode.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ABCMode {
    Threshold(Fp),
    AcceptanceRate(Fp),
}

impl Default for ABCMode {
    fn default() -> Self {
        Self::AcceptanceRate(0.02)
    }
}

/// ABC-SMC summary statistic.
#[warn(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub enum ABCSummaryStatistic {
    #[default]
    MeanSquareError,
    RootMeanSquareError,
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
        fevmd: &mut FEVMData<S, P, N>,
        abcfs: &ABCSettings,
    ) -> Result<ParticleFilteringResults, FEVModelError> {
        let start = Instant::now();

        let (step, seed) = match fevmd.step.partial_cmp(&0).unwrap() {
            Ordering::Greater => (fevmd.step, fevmd.seed.expect("no running seed value")),
            _ => (0, abcfs.initial_seed),
        };

        // Take() or create temporary data structures.
        let mut fevme_sim = fevmd.fevme_simulation.take().unwrap_or(FEVMEnsbl {
            ensbl: PMatrix::<Const<P>>::zeros(abcfs.simulation_size),
            states: vec![S::default(); abcfs.simulation_size],
            weights: vec![1.0 / abcfs.simulation_size as Fp; abcfs.simulation_size],
        });
        let mut fevme_sim_out = fevmd
            .fevme_simulation_output
            .take()
            .unwrap_or(ModelObserArray::new(scobs, abcfs.simulation_size));

        // Uninitialized variables.
        let (det, ess, kld);

        // Initialized variables.
        let mut counter = 0;
        let mut iteration = 0;
        let mut result = Vec::with_capacity(abcfs.target_size);

        let fevme_new = if fevmd.fevme.is_some() && (fevmd.step > 0) {
            // N -th iteration.

            // Copy existing ensemble as particle pdf.
            let pdf_old = fevmd
                .fevme
                .as_ref()
                .unwrap()
                .clone()
                .into_ptpdf(self.valid_range())
                .unwrap();

            // Take() FEVMEnsbl.
            let mut fevme_new = fevmd.fevme.take().unwrap();

            // Create a ParticleRefPDF object where we multiply the covariance matrix by exploration_factor.
            let pdf_old_ef = match pdf_old.mul_covm(abcfs.exploration_factor) {
                Ok(result) => result,
                Err(err) => return Err(FEVModelError::Stats(err)),
            };

            while counter != abcfs.target_size {
                self.fevm_initialize(
                    scobs.as_scconf_slice(),
                    &mut fevme_sim,
                    Some(&pdf_old_ef),
                    (37 + iteration * 43) as u64 + seed,
                )?;

                let mut valid_indices = self.abc_filter_simulations(
                    scobs,
                    &mut fevme_sim,
                    &mut fevme_sim_out,
                    abcfs,
                    137 + seed * 151 + iteration as u64,
                )?;

                // Remove excessive ensemble members.
                if counter + valid_indices.len() > abcfs.target_size {
                    valid_indices.drain((abcfs.target_size - counter)..valid_indices.len());
                }

                // Copy valid params into result.
                valid_indices
                    .iter()
                    .enumerate()
                    .for_each(|(idx, (vdx, rmse))| {
                        fevme_new
                            .ensbl
                            .column_mut(counter + idx)
                            .iter_mut()
                            .zip(fevme_sim.ensbl.column(*vdx).iter())
                            .for_each(|(a, b)| *a = *b);
                        result.push(*rmse);
                    });

                counter += valid_indices.len();

                iteration += 1;
            }

            let mut pdf_new = fevme_new.into_ptpdf(self.valid_range())?;

            pdf_new.compute_importance_weights(&pdf_old_ef, &self.model_prior());

            det = pdf_new.covm().determinant();
            ess = 1.0
                / pdf_new
                    .weights_ref()
                    .iter()
                    .map(|value| value.powi(2))
                    .sum::<Fp>();
            kld = Fp::NAN;

            pdf_new.into_fevme()
        } else if fevmd.step == 0 {
            // First iteration.
            let mut particles = PMatrix::<Const<P>>::zeros(abcfs.target_size);

            while counter != abcfs.target_size {
                self.fevm_initialize(
                    scobs.as_scconf_slice(),
                    &mut fevme_sim,
                    None::<&PUnivariatePDF<P>>,
                    (61 + iteration * 67) as u64 + seed,
                )?;

                let mut valid_indices = self.abc_filter_simulations(
                    scobs,
                    &mut fevme_sim,
                    &mut fevme_sim_out,
                    abcfs,
                    131 + seed * 167 + iteration as u64,
                )?;

                // Remove excessive ensemble members.
                if counter + valid_indices.len() > abcfs.target_size {
                    valid_indices.drain((abcfs.target_size - counter)..valid_indices.len());
                }

                // Copy valid params into result.
                valid_indices
                    .iter()
                    .enumerate()
                    .for_each(|(idx, (vdx, rmse))| {
                        particles
                            .column_mut(counter + idx)
                            .iter_mut()
                            .zip(fevme_sim.ensbl.column(*vdx))
                            .for_each(|(a, b)| *a = *b);
                        result.push(*rmse);
                    });

                counter += valid_indices.len();

                iteration += 1;
            }

            det = Fp::NAN;
            ess = abcfs.target_size as Fp;
            kld = Fp::NAN;

            FEVMEnsbl {
                ensbl: particles,
                states: vec![S::default(); abcfs.target_size],
                weights: vec![1.0 / abcfs.target_size as Fp; abcfs.target_size],
            }
        } else {
            return Err(FEVModelError::InvalidArgument((
                "fevme field is not initialized and step counter is non-zero",
                step as Fp,
            )));
        };

        // Insert our FEVMEnsbl into the fevme field.
        let _ = replace(&mut fevmd.fevme, Some(fevme_new));

        // Also re-insert the temporary data structures.
        let _ = replace(&mut fevmd.fevme_simulation, Some(fevme_sim));
        let _ = replace(&mut fevmd.fevme_simulation_output, Some(fevme_sim_out));

        // Increase seed and step values.
        fevmd.seed = Some(seed + 1);
        fevmd.step = step + 1;

        let res_sorted = result
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .collect::<Vec<Fp>>();

        let eps_25 = res_sorted[abcfs.target_size / 4];
        let eps_50 = res_sorted[abcfs.target_size / 2];
        let eps_75 = res_sorted[3 * abcfs.target_size / 4];

        info!(
            "abc pf step {}\n\tKL delta: {:.3} | ln det {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M simulations in {:.2} sec\n\teffective sample size = {:.0} / {}",
            step,
            kld, det.ln(), eps_25,eps_50,eps_75,
            (iteration * abcfs.simulation_size) as Fp / 1e6,
            start.elapsed().as_millis() as Fp / 1e3,
            ess,
            abcfs.target_size,
        );

        Ok(ParticleFilteringResults {
            ln_determinant: det.ln(),
            effective_sample_size: ess,
            kullback_leibler_div: kld,
            result,
            total_runs: iteration * abcfs.simulation_size,
            total_runtime: start.elapsed().as_nanos() as Fp / 1e3,
        })
    }

    // This function performs an ensemble forward simulation and returns the indices and rmse values in a vector
    /// for results that fall below the given epsilon threshold.
    fn abc_filter_simulations(
        &self,
        scobs: &ScObs<ModelObserVec<N>>,
        fevme: &mut FEVMEnsbl<S, P>,
        output: &mut ModelObserArray<N>,
        abcfs: &ABCSettings,
        seed: u64,
    ) -> Result<Vec<(usize, Fp)>, FEVModelError> {
        self.fevm_initialize_states_only(scobs.as_scconf_slice(), fevme)?;
        self.fevm_simulate(scobs.as_scconf_slice(), fevme, output)?;

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
                        if let Some(covm) = abcfs.noise_covm.as_ref() {
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

        let mut valid_indices = match abcfs.summary_statistic {
            ABCSummaryStatistic::MeanSquareError => valid_indices_flags
                .iter()
                .zip(output.mse_maximum(scobs, Some(&valid_indices_flags)))
                .enumerate()
                .filter_map(|(idx, (flag, rmse))| match flag {
                    true => Some((idx, rmse)),
                    false => None,
                })
                .collect::<Vec<(usize, Fp)>>(),
            ABCSummaryStatistic::RootMeanSquareError => unimplemented!(),
        };

        let epsilon = match abcfs.mode {
            ABCMode::AcceptanceRate(rate) => {
                if !(0.01..0.99).contains(&rate) {
                    return Err(FEVModelError::InvalidArgument((
                        "rate must be in (0.01, 0.99)",
                        rate,
                    )));
                }

                // Compute epsilon threshold that satisfies the constraint.
                let rmse_sorted = valid_indices
                    .iter()
                    .map(|(_, rmse)| *rmse)
                    .sorted_by(|x, y| x.partial_cmp(y).unwrap())
                    .collect::<Vec<Fp>>();

                let target_index = (abcfs.simulation_size as Fp * rate) as usize;

                if target_index > rmse_sorted.len() {
                    Fp::INFINITY
                } else {
                    rmse_sorted[target_index]
                }
            }
            ABCMode::Threshold(epsilon) => epsilon,
        };

        // Only accept results that fall below the rmse threshold.
        valid_indices.retain(|(_, rmse)| {
            matches!(
                rmse.partial_cmp(&epsilon)
                    .unwrap_or(std::cmp::Ordering::Less),
                std::cmp::Ordering::Less
            )
        });

        Ok(valid_indices)
    }
}
