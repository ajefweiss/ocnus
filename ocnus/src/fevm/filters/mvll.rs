use crate::{
    ScObsSeries,
    fevm::{
        FEVM, FEVMData, FEVMError, FEVMNoiseGenerator,
        filters::{ParticleFilter, ParticleFilterResults, ParticleFilterSettings},
        noise::FEVMNoiseMultivariate,
    },
    obser::ObserVec,
    stats::{PDF, PDFParticles, StatsError, ptpdf_importance_weighting},
};
use core::f64;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::{DMatrix, DVectorView, Dyn, U1};
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
        mut fevmd: FEVMData<P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        settings: &mut ParticleFilterSettings<N, FEVMNoiseMultivariate>,
    ) -> Result<ParticleFilterResults<P, N, FS, GS>, FEVMError> {
        let start = Instant::now();

        let mut counter = 0;
        let mut iteration = 0;

        let mut target_data = FEVMData::<P, FS, GS>::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);
        let mut target_filter_values = Vec::<f32>::with_capacity(ensemble_size);

        let mut temp_data = FEVMData::<P, FS, GS>::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<N>>::zeros(series.len(), sim_ensemble_size);

        // Create a Particle PDF from input and multiply by the exploration factor.
        let density_old = PDFParticles::from_particles(
            fevmd.params.as_view_mut(),
            self.model_prior().valid_range(),
            &mut fevmd.weights,
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
            Some((&settings.noise, settings.rseed + iteration * 17)),
        )?;

        if flags.iter().map(|flag| *flag as usize).sum::<usize>() < ensemble_size {
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
                            f32::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<f32>>()
            })
            .flatten()
            .collect::<Vec<f32>>();

        let weights_max = weights.iter().fold(f32::NEG_INFINITY, |acc, next| {
            match acc.partial_cmp(next).unwrap() {
                Ordering::Less => *next,
                _ => acc,
            }
        });

        weights
            .iter_mut()
            .for_each(|w| *w = (*w + weights_max).exp());

        let weights_total = weights.iter().sum::<f32>();

        weights.iter_mut().for_each(|w| *w /= weights_total);

        let effective_sample_size = 1.0 / weights.iter().map(|value| value.powi(2)).sum::<f32>();

        dbg!(effective_sample_size);

        Ok(ParticleFilterResults {
            fevmd: target_data,
            output: target_output,
            errors: Vec::new(),
            error_quantiles: [0.0, 0.0, 0.0],
        })
    }
}
