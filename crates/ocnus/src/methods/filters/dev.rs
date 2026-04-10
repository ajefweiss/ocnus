use crate::{
    base::{Model, ModelEnsbl, ModelError},
    math::quantiles,
    methods::filters::FilterObject,
    obs::{conf::ObsTime, data::ObsData, noise::NullNoise},
};
use log::debug;
use nalgebra::{Const, RealField, SVectorView, U1};
use num_traits::AsPrimitive;
use prodef::{Density, domain::Domain};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{iter::Sum, ops::AddAssign, time::Instant};

impl<T, OC, OD, M, const P: usize, const D: usize> FilterObject<T, OC, OD, M, P, D>
where
    M: Model<T, P, D> + Sync,
    T: Copy + RealField + SampleUniform + Sum,
    OC: ObsTime<T> + Sync,
    OD: AddAssign + ObsData,
    M::FMST: std::fmt::Debug + Clone + Default + Send,
    M::CSST: std::fmt::Debug + Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    /// A single iteration of an differential evolution algorithm (not a particle filter!).
    ///
    /// The algorithm assumes that `errors` field is appropriately filled so that
    /// a comparison with the previous generation can be made.
    pub fn dev<EF, OF>(
        &mut self,
        err_func: &EF,
        obs_func: &OF,
        (mutation, recombination): (T, T),
    ) -> Result<usize, ModelError<T>>
    where
        T: AsPrimitive<f64>,
        EF: Fn(&[OD], &[OD]) -> T + Sync,
        OF: Fn(&M, &OC, &SVectorView<T, D>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
    {
        let start = Instant::now();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.random_seed);

        let constants = self
            .model
            .prior()
            .domain()
            .size()
            .iter()
            .map(|size| match size {
                Some(value) => value.partial_cmp(&T::zero()).unwrap() == std::cmp::Ordering::Equal,
                None => false,
            })
            .collect::<Vec<bool>>();

        let mut ddx = rng.random_range(0..D);

        // Select a dimension that is not fixed.
        while constants[ddx] {
            ddx = rng.random_range(0..D);
        }

        // Choose indices for recombining the particles.
        let ddx_a = vec![rng.random_range(0..self.model_ensbl.len()); self.model_ensbl.len()];
        let ddx_b = vec![rng.random_range(0..self.model_ensbl.len()); self.model_ensbl.len()];
        let thresholds =
            vec![T::from_f64(rng.random_range(0.0..1.0)).unwrap(); self.model_ensbl.len()];

        let mut input = self.model_ensbl.input.clone();

        input
            .par_column_iter_mut()
            .enumerate()
            .chunks(128)
            .for_each(|mut chunk| {
                chunk.iter_mut().for_each(|(idx, new_col)| {
                    new_col[(ddx, 0)] += mutation
                        * (self.model_ensbl.input.column(ddx_a[*idx])[ddx]
                            - self.model_ensbl.input.column(ddx_b[*idx])[ddx]);

                    let view = (*new_col).as_view::<Const<D>, U1, U1, Const<D>>();

                    // Clip values if they go beyond the valid boundaries.
                    new_col.set_column(0, &self.model.domain().clamp(&view));
                });
            });

        let mut temp_ensbl = ModelEnsbl::new(input, None, None);
        let mut temp_obs_ensbl = self.obs_ensbl.clone();

        self.model.initialize_states_ensbl(&mut temp_ensbl)?;

        self.model.simulate_ensbl(
            &mut temp_ensbl,
            &mut temp_obs_ensbl,
            obs_func,
            &mut None::<&mut NullNoise<T>>,
        )?;

        let ref_data = Vec::from(
            self.obs_ensbl
                .ref_data()
                .expect("reference data missing")
                .as_slice(),
        );

        let mutated = temp_obs_ensbl
            .par_ensbl_iter()
            .zip(self.errors.par_iter_mut())
            .zip(thresholds.par_iter())
            .zip(self.model_ensbl.input.par_column_iter_mut())
            .zip(temp_ensbl.input.par_column_iter())
            .zip(self.obs_ensbl.par_ensbl_iter_mut())
            .chunks(128)
            .map(|mut chunk| {
                chunk
                    .iter_mut()
                    .map(
                        |((((((_, temp_out), error), threshold), pt), temp_pt), (_, out))| {
                            let value = err_func(&ref_data, temp_out.as_slice());

                            if ((value < **error) && (**threshold < recombination))
                                && self.model.domain().contains(temp_pt)
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

        debug!(
            "dev_iter\n\teps: {:.3} -- {:.3} -- {:.3}\n\tran {:2.3}M evaluations in {:.2} sec\n\tmutated = {:.1} / {}",
            q_low,
            q_mid,
            q_hgh,
            T::from_f64((self.model_ensbl.len() * self.obs_ensbl.len()) as f64 / 1e6).unwrap(),
            T::from_f64(start.elapsed().as_millis() as f64 / 1e3).unwrap(),
            mutated,
            self.model_ensbl.len()
        );

        self.iterations += 1;
        self.random_seed += 1;
        self.total_runs += self.model_ensbl.len();

        Ok(mutated)
    }

    /// A loop of differential evolution steps with various aborting criteria.
    pub fn dev_loop<EF, OF>(
        &mut self,
        err_func: &EF,
        obs_func: &OF,
        (mutation, recombination): (T, T),
    ) -> Result<Vec<usize>, ModelError<T>>
    where
        T: AsPrimitive<f64> + AsPrimitive<usize>,
        EF: Fn(&[OD], &[OD]) -> T + Sync,
        OF: Fn(&M, &OC, &SVectorView<T, D>, &M::FMST, &M::CSST) -> Result<OD, ModelError<T>> + Sync,
    {
        let mut mutated = Vec::new();

        for _ in 0..self.settings.max_iterations * P {
            let result = self.dev(err_func, obs_func, (mutation, recombination));

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
