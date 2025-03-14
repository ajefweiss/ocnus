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
use core::f32;
use itertools::Itertools;
use log::{error, info};
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::ParticleFilterError;

/// A trait that enables the use of a bootstrap particle filter method
/// for a [`FEVM`].
pub trait BSParticleFilter<const P: usize, const N: usize, FS, GS>:
    ParticleFilter<P, N, FS, GS>
where
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
{
    /// Basic bootstrap filter (single iteration) with multivariate likelihood.
    fn bspf_run(
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
        )? * settings.exploration_factor;

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

        // Compute likelihoods.
        let likelihoods = temp_output
            .par_column_iter()
            .zip(flags.par_iter())
            .chunks(Self::RCS)
            .map(|mut chunks| {
                chunks
                    .iter_mut()
                    .map(|(out, flag)| {
                        if **flag {
                            settings.noise.mvlh(out.as_slice(), series)
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<f32>>()
            })
            .flatten()
            .collect::<Vec<f32>>();

        let lh_max = *likelihoods.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        let mut weights = likelihoods
            .iter()
            .map(|lh| (lh - lh_max).exp())
            .collect::<Vec<f32>>();

        weights
            .par_iter_mut()
            .zip(temp_data.params.par_column_iter())
            .for_each(|(weight, params)| *weight *= self.model_prior().relative_density(&params));

        let weights_total = weights.iter().sum::<f32>();

        weights.iter_mut().for_each(|w| *w /= weights_total);

        // Create a Particle PDF from the temporary simulations.
        let density_new: PDFParticles<'_, P> = PDFParticles::from_particles(
            temp_data.params.as_view(),
            self.model_prior().valid_range(),
            &weights,
        )?;

        self.fevm_initialize_resample(series, &mut target_data, &density_new, settings.rseed + 21)?;

        let uniques = target_data
            .params
            .row(0)
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .copied()
            .collect::<Vec<f32>>()
            .len();

        self.fevm_simulate(
            series,
            &mut target_data,
            &mut target_output,
            None::<(&FEVMNoiseNull, u64)>,
        )?;

        target_data.weights = vec![1.0 / ensemble_size as f32; ensemble_size];

        info!(
            "bspf_run\n\tKL delta: {:.3} | ln det {:.3} \n\tran {:2.3}M evaluations in {:.2} sec\n\t unique samples = {} / {}",
            0.0,
            2.0,
            (series.len() * sim_ensemble_size) as f32 / 1e6,
            start.elapsed().as_millis() as f32 / 1e3,
            uniques,
            ensemble_size,
        );

        settings.rseed += 1;

        Ok(ParticleFilterResults {
            fevmd: target_data,
            output: target_output,
            errors: Some(likelihoods),
            error_quantiles: None,
        })
    }
}
