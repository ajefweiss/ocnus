use crate::{
    ScObsSeries,
    fevm::{FEVM, FEVMData, FEVMError, FEVMNoiseGen},
    obser::ObserVec,
    stats::{PDF, PDFParticles, ptpdf_importance_weighting},
};
use derive_builder::Builder;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, U1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A data structure that holds particle filter diagnostics and settings.
#[derive(Builder, Debug, Default, Deserialize, Serialize)]
pub struct ParticleFilterSettings<const N: usize, NG>
where
    NG: FEVMNoiseGen<N>,
{
    /// Multiplier for the transition kernel (covariance matrix),
    /// a higher value leads to a better exploration of the parameter
    /// space but slower convergence. The "optimal" value is 2.0
    /// (see Filippi et al. 2013).
    #[builder(default = 2.0)]
    pub exploration_factor: f64,

    /// Iteration counter,
    #[builder(default = 0, setter(skip))]
    pub iteration: usize,

    /// Noise generator.
    pub noise: NG,

    /// Random seed (initial & running).
    #[builder(default = 42)]
    pub rseed: u64,

    /// Total simulation runs counter,
    #[builder(default = 0, setter(skip))]
    pub truns: usize,
}

/// A data structure holding the results of any particle filtering method.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticleFilterResults<const P: usize, const N: usize, FS, GS> {
    /// FEVM data object.
    pub fevmd: FEVMData<P, FS, GS>,

    /// Output array.
    pub output: DMatrix<ObserVec<N>>,

    /// Error values.
    pub errors: Vec<f64>,
}

/// ABC-SMC algorithm mode.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ABCPFMode<const N: usize, F>
where
    F: Send + Sync + Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f64,
{
    /// ABC-SMC runs are filtered by a threshold value.
    Threshold((f64, F)),

    /// ABC-SMC runs are filtered by a fixed acceptance rate.
    AcceptanceRate((f64, F)),
}

/// A trait that enables the use of approximate Bayesian computation (ABC) particle filter methods
/// for a [`FEVM`].
pub trait ABCParticleFilter<const P: usize, const N: usize, FS, GS>: FEVM<P, N, FS, GS>
where
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Serialize + Send,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Serialize + Send,
{
    /// Basic ABC-SMC algorithm (single iteration) with fixed acceptance ratio.
    fn abcpf_run<F, NG>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        mut fevmd: FEVMData<P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        settings: &mut ParticleFilterSettings<N, NG>,
        mode: ABCPFMode<N, F>,
    ) -> Result<ParticleFilterResults<P, N, FS, GS>, FEVMError>
    where
        F: Send + Sync + Fn(&DVectorView<ObserVec<N>>, &ScObsSeries<ObserVec<N>>) -> f64,
        NG: FEVMNoiseGen<N>,
    {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;
        let mut errors = Vec::<f64>::with_capacity(ensemble_size);

        let mut target_data = FEVMData::<P, FS, GS>::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        let mut temp_data = FEVMData::<P, FS, GS>::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle PDF from input and multiply by the exploration factor.
        let mut density_old = PDFParticles::from_particles(
            fevmd.params.as_view_mut(),
            self.model_prior().valid_range(),
            fevmd.weights,
        )? * settings.exploration_factor;

        while counter != ensemble_size {
            self.fevm_initialize(
                series,
                &mut temp_data,
                Some(&density_old),
                settings.rseed + 23 * iteration,
            )?;

            let mut indices = self.fevm_simulate(
                series,
                &mut temp_data,
                &mut temp_output,
                Some((&settings.noise, settings.rseed + iteration * 17)),
            )?;

            let filter_values = match mode {
                ABCPFMode::AcceptanceRate((accrate, ref _ranking)) => {
                    if !(0.01..0.99).contains(&accrate) {
                        return Err(FEVMError::InvalidParameter {
                            name: "acceptance rate",
                            value: accrate,
                        });
                    }

                    unimplemented!()
                }
                ABCPFMode::Threshold((epsilon, ref filter)) => temp_output
                    .par_column_iter()
                    .zip(indices.par_iter_mut())
                    .chunks(Self::RCS)
                    .map(|mut chunks| {
                        chunks
                            .iter_mut()
                            .map(|(out, flag)| {
                                if **flag {
                                    let value = filter(&out.as_view::<Dyn, U1, U1, Dyn>(), series);
                                    **flag = value < epsilon;

                                    value
                                } else {
                                    f64::NAN
                                }
                            })
                            .collect::<Vec<f64>>()
                    })
                    .flatten()
                    .collect::<Vec<f64>>(),
            };

            let mut indices_valid = indices
                .into_iter()
                .enumerate()
                .filter_map(|(idx, flag)| if flag { Some(idx) } else { None })
                .collect::<Vec<usize>>();

            debug!("valid: {}", indices_valid.len());

            // Remove excessive ensemble members.
            if counter + indices_valid.len() > target_data.size() {
                debug!(
                    "removing excessive ensemble members simulations n={}",
                    counter + indices_valid.len() - target_data.size()
                );
                indices_valid.drain((target_data.size() - counter)..indices_valid.len());
            }

            // Copy over results.
            indices_valid.iter().enumerate().for_each(|(edx, idx)| {
                target_data
                    .params
                    .set_column(counter + edx, &temp_data.params.column(*idx));

                target_output.set_column(counter + edx, &temp_output.column(*idx));

                errors.push(filter_values[*idx]);
            });

            counter += indices_valid.len();

            iteration += 1;
        }

        // Create a Particle PDF from input and multiply by the exploration factor.
        let mut density_new = PDFParticles::from_particles(
            target_data.params.as_view_mut(),
            self.model_prior().valid_range(),
            target_data.weights.clone(),
        )?;

        // density_old *= 1.0 / settings.exploration_factor;

        ptpdf_importance_weighting(&mut density_new, &density_old, &self.model_prior());

        let ess = 1.0
            / density_new
                .weights()
                .iter()
                .map(|value| value.powi(2))
                .sum::<f64>();

        let errors_sorted = errors
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .collect::<Vec<f64>>();

        let eps_25 = errors_sorted[ensemble_size / 4];
        let eps_50 = errors_sorted[ensemble_size / 2];
        let eps_75 = errors_sorted[3 * ensemble_size / 4];

        info!(
            "abc pf step\n\tKL delta: {:.3} | ln det {:.3} | eps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M simulations in {:.2} sec\n\teffective sample size = {:.0} / {}",
            0.0,
            2.0,
            eps_25,
            eps_50,
            eps_75,
            (iteration as usize * sim_ensemble_size) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            ess,
            ensemble_size,
        );

        Ok(ParticleFilterResults {
            fevmd: target_data,
            output: target_output,
            errors,
        })
    }
}
