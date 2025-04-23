use crate::{
    base::{OcnusModel, OcnusModelError, ScObs},
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    obser::{NullNoise, OcnusObser},
    stats::Density,
};

use itertools::Itertools;
use log::info;
use nalgebra::{DMatrix, DVectorView, Dyn, RealField, SVector, Scalar, U1};
use num_traits::{AsPrimitive, Zero};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{AddAssign, Mul, Sub},
    time::Instant,
};

impl<M, T, const D: usize, FMST, CSST, OT> ParticleFilter<M, T, D, FMST, CSST, OT>
where
    M: OcnusModel<T, D, FMST, CSST>,
    T: Copy
        + for<'x> Mul<&'x T, Output = T>
        + RealField
        + SampleUniform
        + for<'x> Sub<&'x T, Output = T>
        + Sum
        + for<'x> Sum<&'x T>,
    for<'x> &'x T: Mul<&'x T, Output = T>,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    OT: AddAssign + OcnusObser + Scalar + Zero,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an differential evolution algorithm (not a particle filter).
    ///
    /// The algorithm assumes that `ensbl_errors` field is appropriately filled.
    pub fn diff_ev_iter<EF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        (mutation, recombination): (T, T),
        obs_func: &OF,
        err_func: &EF,
    ) -> Result<usize, ParticleFilterError<T>>
    where
        M: OcnusModel<T, D, FMST, CSST>,
        EF: Fn(&DVectorView<OT>) -> T + Sync,
        OF: Fn(&M, &ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        let ensemble_size = settings.ensemble_size;
        let series = &settings.series;

        let mut target_ensbl = self.ensbl.take().expect("ensemble is not initialized");
        let mut target_output = self.ensbl_output.take().unwrap();
        let mut target_errors = self.ensbl_errors.take().unwrap();

        let mut temp_ensbl = target_ensbl.clone();
        let mut temp_output = DMatrix::<OT>::zeros(series.len(), ensemble_size);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.rseed);

        let constants = (&target_ensbl.ptpdf)
            .get_constants()
            .iter()
            .map(|c| c.is_finite())
            .collect::<Vec<bool>>();

        let mut ddx = rng.random_range(0..D);

        // Select a dimension that is not fixed.
        while constants[ddx] {
            ddx = rng.random_range(0..D);
        }

        // Choose indices for recombination particles.
        let ddx_a = vec![rng.random_range(0..settings.ensemble_size); settings.ensemble_size];
        let ddx_b = vec![rng.random_range(0..settings.ensemble_size); settings.ensemble_size];
        let thresholds =
            vec![T::from_f64(rng.random_range(0.0..1.0)).unwrap(); settings.ensemble_size];

        temp_ensbl
            .ptpdf
            .particles_mut()
            .par_column_iter_mut()
            .enumerate()
            .chunks(128)
            .for_each(|mut chunks| {
                chunks.iter_mut().for_each(|(idx, new_col)| {
                    new_col[(ddx, 0)] += mutation
                        * (target_ensbl.ptpdf.particles()[(ddx, ddx_a[*idx])]
                            - target_ensbl.ptpdf.particles()[(ddx, ddx_b[*idx])])
                });
            });

        self.model
            .as_ref()
            .unwrap()
            .initialize_states_ensbl(&mut temp_ensbl)?;

        self.model.as_ref().unwrap().simulate_ensbl(
            series,
            &mut temp_ensbl,
            obs_func,
            &mut temp_output.as_view_mut(),
            None::<&mut NullNoise<T>>,
        )?;

        let mutated = temp_output
            .par_column_iter()
            .zip(target_errors.par_iter_mut())
            .zip(thresholds.par_iter())
            .zip(target_ensbl.ptpdf.particles_mut().par_column_iter_mut())
            .zip(temp_ensbl.ptpdf.particles().par_column_iter())
            .zip(target_output.par_column_iter_mut())
            .chunks(128)
            .map(|mut chunks| {
                chunks
                    .iter_mut()
                    .map(|(((((out, error), threshold), col), temp_col), temp_out)| {
                        let value = err_func(&out.as_view::<Dyn, U1, U1, Dyn>());

                        if ((value < **error) && (**threshold < recombination))
                            && self
                                .model
                                .as_ref()
                                .unwrap()
                                .model_prior()
                                .validate_sample(temp_col)
                        {
                            col[(ddx, 0)] = temp_col[(ddx, 0)];
                            **error = value;
                            temp_out.set_column(0, out);
                            1
                        } else {
                            0
                        }
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();

        target_ensbl
            .ptpdf
            .weights_mut()
            .iter_mut()
            .for_each(|value| *value = T::one() / T::from_usize(ensemble_size).unwrap());

        // Update covariance matrix.
        target_ensbl.ptpdf.update_mvpdf();

        let target_errors_sorted = target_errors
            .iter()
            .sorted_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .collect::<Vec<T>>();

        // Compute quantiles for logging purposes.
        let eps_1 = target_errors_sorted[(ensemble_size as f64 * 0.34) as usize];
        let eps_2 = target_errors_sorted[(ensemble_size as f64 * 0.50) as usize];
        let eps_3 = target_errors_sorted[(ensemble_size as f64 * 0.68) as usize];

        info!(
            "diff_ev_iter\n\teps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\tmutated = {:.1} / {}",
            eps_1,
            eps_2,
            eps_3,
            T::from_f64((ensemble_size * series.len()) as f64 / 1e6).unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            mutated,
            ensemble_size
        );

        self.rseed += 1;

        self.ensbl = Some(target_ensbl);
        self.ensbl_output = Some(target_output);
        self.ensbl_errors = Some(target_errors);

        self.iter += 1;
        self.truns += ensemble_size;

        Ok(mutated)
    }

    /// A loop of differential evolution steps with various aborting criteria.
    pub fn diff_ev_loop<EF, OF>(
        &mut self,
        settings: &ParticleFilterSettings<T>,
        (mutation, recombination): (T, T),
        obs_func: &OF,
        err_func: &EF,
    ) -> Result<Vec<usize>, ParticleFilterError<T>>
    where
        M: OcnusModel<T, D, FMST, CSST>,
        T: AsPrimitive<usize>,
        EF: Fn(&DVectorView<OT>) -> T + Sync,
        OF: Fn(&M, &ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<OT, OcnusModelError<T>>
            + Sync,
    {
        let mut mutated = Vec::new();

        for _ in 0..settings.max_iterations {
            let result = self.diff_ev_iter(settings, (mutation, recombination), obs_func, err_func);

            match result {
                Ok(new_mutated) => {
                    mutated.push(new_mutated);
                }
                Err(err) => return Err(err),
            }
        }

        Ok(mutated)
    }
}
