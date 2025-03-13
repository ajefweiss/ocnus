use crate::{
    ScObsSeries,
    fevm::{
        FEVMData, FEVMError,
        filters::{ParticleFilter, ParticleFilterResults, ParticleFilterSettings},
        noise::{FEVMNoiseMultivariate, FEVMNoiseNull},
    },
    obser::ObserVec,
    stats::{PDF, PDFParticles},
};
use log::{error, info};
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, time::Instant};

use super::ParticleFilterError;

/// A trait that enables the use of a sequential Monte Carlo (SMC) particle filter methods
/// for a [`FEVM`].
pub trait MVLLParticleFilter<const P: usize, const N: usize, FS, GS>:
    ParticleFilter<P, N, FS, GS>
where
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
{
    /// Basic SMC algorithm (single iteration) with multivariate likelihood.
    fn mvllpf_run(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &FEVMData<P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        settings: &mut ParticleFilterSettings<N, FEVMNoiseMultivariate>,
    ) -> Result<ParticleFilterResults<P, N, FS, GS>, FEVMError> {
        let start = Instant::now();

        let mut target_data = FEVMData::<P, FS, GS>::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<N>>::zeros(series.len(), ensemble_size);

        let mut temp_data = FEVMData::<P, FS, GS>::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle PDF from input and multiply by the exploration factor.
        let density_old = PDFParticles::from_particles(
            fevmd.params.as_view(),
            self.model_prior().valid_range(),
            &fevmd.weights,
        )?;

        self.fevm_initialize(
            series,
            &mut temp_data,
            Some(&density_old),
            settings.rseed + 29,
        )?;

        let flags = self.fevm_simulate(
            series,
            &mut temp_data,
            &mut temp_output,
            Some((&settings.noise, settings.rseed + 71)),
        )?;

        if flags.iter().map(|flag| *flag as usize).sum::<usize>() < ensemble_size {
            error!(
                "inefficient sampling {}",
                flags.iter().map(|flag| *flag as usize).sum::<usize>()
            );
            return Err(FEVMError::ParticleFilter(
                ParticleFilterError::InefficientSampling,
            ));
        }

        let mut weights = temp_output
            .par_column_iter()
            .zip(flags.par_iter())
            .chunks(Self::RCS)
            .map(|mut chunks| {
                chunks
                    .iter_mut()
                    .map(|(out, flag)| {
                        if **flag {
                            settings.noise.mvll(out.as_slice(), series)
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<f64>>()
            })
            .flatten()
            .collect::<Vec<f64>>();

        let weights_max = weights.iter().fold(f64::NEG_INFINITY, |acc, next| {
            match acc.partial_cmp(next).unwrap() {
                Ordering::Less => *next,
                _ => acc,
            }
        });

        weights
            .iter_mut()
            .for_each(|w| *w = (*w + weights_max).exp());

        let weights_total = weights.iter().sum::<f64>();

        weights.iter_mut().for_each(|w| *w /= weights_total);

        let effective_sample_size = 1.0 / weights.iter().map(|value| value.powi(2)).sum::<f64>();

        temp_data.weights = weights;

        // Create a Particle PDF from the temporary simulations.
        let density_new: PDFParticles<'_, P> = PDFParticles::from_particles(
            temp_data.params.as_view(),
            self.model_prior().valid_range(),
            &temp_data.weights,
        )? * settings.exploration_factor;

        self.fevm_initialize(
            series,
            &mut target_data,
            Some(&density_new),
            settings.rseed + 13,
        )?;
        self.fevm_simulate(
            series,
            &mut target_data,
            &mut target_output,
            None::<(&FEVMNoiseNull, u64)>,
        )?;

        target_data.weights = vec![1.0 / ensemble_size as f64; ensemble_size];

        info!(
            "mvllpf_run\n\tKL delta: {:.3} | ln det {:.3} \n\tran {:2.3}M simulations in {:.2} sec\n\teffective sample size = {:.0} / {}",
            0.0,
            2.0,
            sim_ensemble_size as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            effective_sample_size,
            ensemble_size,
        );

        settings.rseed += 1;

        Ok(ParticleFilterResults {
            fevmd: target_data,
            output: target_output,
            errors: Vec::new(),
            error_quantiles: [0.0, 0.0, 0.0],
        })
    }
}
