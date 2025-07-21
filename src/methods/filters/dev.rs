use crate::{
    base::{Model, ModelError, ScConf},
    math::quantiles,
    methods::filters::ParticleFilter,
    obsty::{NullNoise, Observable},
    stats::Density,
};
use log::info;
use nalgebra::{RealField, SVector, Scalar};
use num_traits::{AsPrimitive, Zero};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{iter::Sum, ops::AddAssign, time::Instant};

impl<T, M, const D: usize, OT> ParticleFilter<T, M, D, OT>
where
    M: Clone + Model<T, D> + Sync,
    T: Copy + RealField + SampleUniform + Sum,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    OT: AddAssign + Observable + Scalar + Zero,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an differential evolution algorithm (not a particle filter!).
    ///
    /// The algorithm assumes that `errors` field is appropriately filled so that
    /// a comparison with the previous generation can be made.
    pub fn diff_ev_iter<EF, OF>(
        &mut self,
        (mutation, recombination): (T, T),
        obs_func: &OF,
        err_func: &EF,
    ) -> Result<usize, ModelError<T>>
    where
        M: Model<T, D>,
        T: AsPrimitive<f64>,
        EF: Fn(&[OT], &[OT]) -> T + Sync,
        OF: Fn(&M, &ScConf<T>, &SVector<T, D>, &M::FMST, &M::CSST) -> Result<OT, ModelError<T>>
            + Sync,
    {
        let start = Instant::now();

        let mut temp_ensbl = self.ensbl.clone();
        let mut temp_obser = self.obser.clone();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.rseed);

        let constants = (&self.ensbl.ptpdf)
            .get_constants()
            .iter()
            .map(|c| c.is_finite())
            .collect::<Vec<bool>>();

        let mut ddx = rng.random_range(0..D);

        // Select a dimension that is not fixed.
        while constants[ddx] {
            ddx = rng.random_range(0..D);
        }

        // Choose indices for recombining the particles.
        let ddx_a = vec![rng.random_range(0..self.ensbl.len()); self.ensbl.len()];
        let ddx_b = vec![rng.random_range(0..self.ensbl.len()); self.ensbl.len()];
        let thresholds = vec![T::from_f64(rng.random_range(0.0..1.0)).unwrap(); self.ensbl.len()];

        temp_ensbl
            .ptpdf
            .par_iter_mut()
            .enumerate()
            .chunks(128)
            .for_each(|mut chunks| {
                chunks.iter_mut().for_each(|(idx, new_col)| {
                    new_col[(ddx, 0)] += mutation
                        * (self.ensbl.ptpdf.get_particle(ddx_a[*idx])[ddx]
                            - self.ensbl.ptpdf.get_particle(ddx_b[*idx])[ddx])
                });
            });

        self.model.initialize_states_ensbl(&mut temp_ensbl)?;

        self.model.simulate_ensbl(
            &mut temp_ensbl,
            &mut temp_obser,
            obs_func,
            &mut None::<&mut NullNoise<T>>,
        )?;

        let refdt = Vec::<OT>::from_iter(self.obser.refdt().iter().cloned());

        let mutated = temp_obser
            .par_ensbl_iter()
            .zip(self.errors.par_iter_mut())
            .zip(thresholds.par_iter())
            .zip(self.ensbl.ptpdf.par_iter_mut())
            .zip(temp_ensbl.ptpdf.par_iter())
            .zip(self.obser.par_ensbl_iter_mut())
            .chunks(128)
            .map(|mut chunks| {
                chunks
                    .iter_mut()
                    .map(
                        |((((((_, temp_out), error), threshold), pt), temp_pt), (_, out))| {
                            let value = err_func(&refdt, temp_out.as_slice());

                            if ((value < **error) && (**threshold < recombination))
                                && self.model.model_prior().validate_sample(temp_pt)
                            {
                                pt[(ddx, 0)] = temp_pt[(ddx, 0)];
                                **error = value;
                                out.set_column(0, temp_out);
                                1
                            } else {
                                0
                            }
                        },
                    )
                    .sum::<usize>()
            })
            .sum::<usize>();

        self.ensbl.ptpdf.update_weights(&vec![
            T::one() / T::from_usize(self.ensbl.len()).unwrap();
            self.ensbl.len()
        ]);

        // Update covariance matrix.
        self.ensbl.ptpdf.update_mvpdf();

        // Compute quantiles for logging purposes.
        let quantiles = quantiles(
            &self.errors,
            &[
                T::from_f64(0.34).unwrap(),
                T::from_f64(0.50).unwrap(),
                T::from_f64(0.68).unwrap(),
            ],
        );

        let (q_low, q_mid, q_hgh) = (quantiles[0], quantiles[1], quantiles[2]);

        info!(
            "diff_ev_iter\n\teps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\tmutated = {:.1} / {}",
            q_low,
            q_mid,
            q_hgh,
            T::from_f64((self.ensbl.len() * self.obser.len()) as f64 / 1e6).unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            mutated,
            self.ensbl.len()
        );

        self.rseed += 1;

        self.iter += 1;
        self.truns += self.ensbl.len();

        Ok(mutated)
    }

    /// A loop of differential evolution steps with various aborting criteria.
    pub fn diff_ev_loop<EF, OF>(
        &mut self,
        max_iterations: usize,
        (mutation, recombination): (T, T),
        obs_func: &OF,
        err_func: &EF,
    ) -> Result<Vec<usize>, ModelError<T>>
    where
        M: Model<T, D>,
        T: AsPrimitive<f64> + AsPrimitive<usize>,
        EF: Fn(&[OT], &[OT]) -> T + Sync,
        OF: Fn(&M, &ScConf<T>, &SVector<T, D>, &M::FMST, &M::CSST) -> Result<OT, ModelError<T>>
            + Sync,
    {
        let mut mutated = Vec::new();

        for _ in 0..max_iterations {
            let result = self.diff_ev_iter((mutation, recombination), obs_func, err_func);

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
