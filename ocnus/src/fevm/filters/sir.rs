use crate::{
    OFloat, OState,
    fevm::{
        FEVMData, FEVMError, ParticleFilterError,
        filters::{ParticleFilter, ParticleFilterResults, ParticleFilterSettings},
    },
    obser::{ObserVec, ScObsSeries},
    stats::{PDF, PDFParticles},
};
use itertools::Itertools;
use log::{error, info};
use nalgebra::DMatrix;
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::time::Instant;

/// A trait that enables the use of a bootstrap particle filter method
/// for a [`FEVM`].
pub trait SIRParticleFilter<T, const P: usize, const N: usize, FS, GS>:
    ParticleFilter<T, P, N, FS, GS>
where
    T: OFloat,
    FS: OState,
    GS: OState,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
    StandardNormal: Distribution<T>,
{
    /// Basic bootstrap filter (single iteration) with multivariate likelihood.
    fn sirpf_run(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        fevmd: &FEVMData<T, P, FS, GS>,
        ensemble_size: usize,
        sim_ensemble_size: usize,
        settings: &mut ParticleFilterSettings<T, N>,
    ) -> Result<ParticleFilterResults<T, P, N, FS, GS>, FEVMError<T>> {
        let start = Instant::now();

        let mut target_data = FEVMData::new(ensemble_size);
        let mut target_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), ensemble_size);

        let mut temp_data = FEVMData::new(sim_ensemble_size);
        let mut temp_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), sim_ensemble_size);

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

        let flags = self.fevm_simulate(series, &mut temp_data, &mut temp_output, None)?;

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
                            settings.noise.likelihood(out, series)
                        } else {
                            T::neg_infinity()
                        }
                    })
                    .collect::<Vec<T>>()
            })
            .flatten()
            .collect::<Vec<T>>();

        let lh_max = *likelihoods
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let mut weights = likelihoods
            .iter()
            .map(|lh| Float::exp(*lh - lh_max))
            .collect::<Vec<T>>();

        weights
            .par_iter_mut()
            .zip(temp_data.params.par_column_iter())
            .for_each(|(weight, params)| *weight *= self.model_prior().relative_density(&params));

        let weights_total = weights.iter().sum::<T>();

        weights.iter_mut().for_each(|w| *w /= weights_total);

        // Create a Particle PDF from the temporary simulations.
        let density_new = PDFParticles::from_particles(
            temp_data.params.as_view(),
            self.model_prior().valid_range(),
            &weights,
        )?;

        let effective_sample_size =
            T::one() / weights.iter().map(|v| Float::powi(*v, 2)).sum::<T>();

        self.fevm_initialize_resample(series, &mut target_data, &density_new, settings.rseed + 21)?;

        let uniques = target_data
            .params
            .row(0)
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .dedup()
            .copied()
            .collect::<Vec<T>>()
            .len();

        self.fevm_simulate(series, &mut target_data, &mut target_output, None)?;

        target_data.weights = vec![T::one() / ensemble_size.as_(); ensemble_size];

        info!(
            "bootpf_run\n\tKL delta: {:.3} | ln det {:.3} \n\tran {:2.3}M evaluations in {:.2} sec\n\tunique samples = {} ({:.1}) / {}",
            0.0,
            2.0,
            (series.len() * sim_ensemble_size) as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3,
            uniques,
            effective_sample_size,
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
