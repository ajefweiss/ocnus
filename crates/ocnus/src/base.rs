//! Core model traits and implementations for the **ocnus** framework.

use crate::{
    coords::Coordinates,
    obs::{
        Obs, ObsEnsbl,
        conf::{ObsPosition, ObsTime},
        data::{ICSBasis, ObsData},
        noise::ObsNoise,
    },
};
use itertools::zip_eq;
use log::debug;
use nalgebra::{
    Const, DVector, MatrixView, OMatrix, RealField, SVectorView, SVectorViewMut, Scalar,
};
use nalgebra::{Dyn, Matrix, VecStorage};
use num_traits::AsPrimitive;
use prodef::{Density, domain::Domain, particle::ParticleDensity};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{io::Write, iter::Sum};
use std::{ops::AddAssign, time::Instant};
use thiserror::Error;

/// Error types associated with the [`Model`] trait.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ModelError<T> {
    #[error("failed to convert external to internal coords")]
    Coordinates(Vec<T>),
    #[error("failed to evolve the model state (dt={0:.2}sec)")]
    Evolution(T),
    #[error("failed to sample from the probability density function")]
    Sampling,
}

/// A trait that is shared by all forward models within the **ocnus** framework.
pub trait Model<T, const D: usize, const P: usize>: Coordinates<T, D, P>
where
    T: Copy + RealField,
    Self: Sized,
{
    /// The default rayon chunk size that is used for any parallel iterators.
    ///
    /// Certain operations may use multiples of this value.
    const RCS: usize;

    /// Forward modeling state type.
    type FMST;

    /// Return the prior domain.
    fn domain(&self) -> impl Domain<T, Const<P>> + 'static;

    /// Evolve a model state forward in time.
    fn evolve_state(
        &self,
        time_step: T,
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>>;

    /// Evolve a model ensemble forward in time.
    fn evolve_ensbl(
        &self,
        time_step: T,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
    ) -> Result<(), ModelError<T>>
    where
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        model_ensbl
            .input
            .par_column_iter()
            .zip(model_ensbl.states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunk| {
                chunk
                    .iter_mut()
                    .try_for_each(|(params, (fm_state, cs_state))| {
                        self.evolve_state(time_step, params, fm_state, cs_state)
                    })
            })
    }

    /// Initialize the model parameters, and both the coordinate system and forward model states.
    fn initialize<G>(
        &self,
        params: &mut SVectorViewMut<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
        prior: G,
        max_attempts: usize,
        rng: &mut impl RngExt,
    ) -> Result<(), ModelError<T>>
    where
        G: Density<T, Const<P>>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        self.initialize_params::<G>(params, prior, max_attempts, rng)?;
        self.initialize_states(&params.as_view(), fm_state, cs_state)?;

        Ok(())
    }

    /// Initialize the model parameters, and both the coordinate system and forward model states for an ensemble.
    fn initialize_ensbl<G>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
        prior: G,
        max_attempts: usize,
        rseed: u64,
    ) -> Result<(), ModelError<T>>
    where
        G: Density<T, Const<P>> + Sync,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        assert!(
            !model_ensbl.is_empty(),
            "cannot initialize an empty model ensemble"
        );

        model_ensbl
            .input
            .par_column_iter_mut()
            .zip(model_ensbl.states.par_iter_mut())
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunk)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 310248241) as u64);

                chunk
                    .iter_mut()
                    .try_for_each(|(params, (fm_state, cs_state))| {
                        self.initialize::<G>(
                            params,
                            fm_state,
                            cs_state,
                            prior.clone(),
                            max_attempts,
                            &mut rng,
                        )?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "initialize_ensbl: {:2.1}k evaluations in {:.0}ms",
            model_ensbl.len() as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Initialize the model parameters.
    fn initialize_params<G>(
        &self,
        params: &mut SVectorViewMut<T, P>,
        prior: G,
        max_attempts: usize,
        rng: &mut impl RngExt,
    ) -> Result<(), ModelError<T>>
    where
        G: Density<T, Const<P>>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let opt_col = prior.sample(rng, &prodef::SamplingMode::UntilValid { max_attempts });

        if let Some(col) = opt_col {
            params.set_column(0, &col);

            Ok(())
        } else {
            Err(ModelError::Sampling)
        }
    }

    /// Initialize the model parameters for an ensemble.
    fn initialize_params_ensbl<G>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
        prior: G,
        max_attempts: usize,
        rseed: u64,
    ) -> Result<(), ModelError<T>>
    where
        G: Density<T, Const<P>> + Sync,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        model_ensbl
            .input
            .par_column_iter_mut()
            .chunks(Self::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunk)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 213161503) as u64);

                chunk.iter_mut().try_for_each(|params| {
                    self.initialize_params(params, prior.clone(), max_attempts, &mut rng)?;

                    Ok::<(), ModelError<T>>(())
                })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "initialize_params_ensbl: {:2.1}k evaluations in {:.0}ms",
            model_ensbl.len() as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Initialize the coordinate system and forward model states.
    fn initialize_states(
        &self,
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<(), ModelError<T>>;

    /// Initialize the the coordinate system and forward model states for an ensemble.
    fn initialize_states_ensbl(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
    ) -> Result<(), ModelError<T>>
    where
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();

        model_ensbl
            .input
            .par_column_iter()
            .zip(model_ensbl.states.par_iter_mut())
            .chunks(Self::RCS)
            .try_for_each(|mut chunk| {
                chunk
                    .iter_mut()
                    .try_for_each(|(params, (fm_state, cs_state))| {
                        self.initialize_states(params, fm_state, cs_state)?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "initialize_states_ensbl: {:2.1}k evaluations in {:.0}ms",
            model_ensbl.len() as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Return internal coordinates and the basis vectors at the location of the observer.
    fn observe_icsbasis(
        &self,
        coordinates: &SVectorView<T, D>,
        params: &SVectorView<T, P>,
        cs_state: &Self::CSST,
    ) -> Result<ICSBasis<T, D>, ModelError<T>> {
        let q = match Self::transform_ecs_to_ics(coordinates, params, cs_state) {
            Some(value) => value,
            None => {
                return Err(ModelError::Coordinates(coordinates.as_slice().to_vec()));
            }
        };

        let basis = Self::contravariant_basis(&q.as_view(), params, cs_state)
            .expect("failed to construct contravariant basis");

        Ok(ICSBasis::new(q, &basis))
    }

    /// Returns a reference to the underlying model prior.
    fn prior(&self) -> impl Density<T, Const<P>> + 'static;

    /// Perform a forward simulation and generate synthetic observables `OD` for the
    /// given spacecraft observers, with configuration type `OC`, for a given generating function `OF`.
    fn simulate<OC, OD, OF>(
        &self,
        obs: &Obs<T, OC>,
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
        obs_func: &OF,
    ) -> Result<DVector<OD>, ModelError<T>>
    where
        OC: ObsTime<T>,
        OD: ObsData,
        OF: Fn(
            &Self,
            &OC,
            &SVectorView<T, P>,
            &Self::FMST,
            &Self::CSST,
        ) -> Result<OD, ModelError<T>>,
    {
        let mut timer = T::zero();

        let mut obs_ensbl_vector = DVector::zeros(obs.len());

        zip_eq(obs, obs_ensbl_vector.iter_mut()).try_for_each(|(conf, obs)| {
            // Compute time step to next observation.
            let time_step = conf.timestamp() - timer;
            timer = conf.timestamp();

            if time_step < T::zero() {
                return Err(ModelError::Evolution(time_step));
            } else {
                self.evolve_state(time_step, params, fm_state, cs_state)?;

                *obs = obs_func(self, conf, params, fm_state, cs_state)?;
            }

            Ok::<(), ModelError<T>>(())
        })?;

        Ok(obs_ensbl_vector)
    }

    /// Perform an ensemble forward simulation and generate synthetic observables `OD` for the
    /// given spacecraft observers for a given generating function `OF` and noise model `NM`.
    fn simulate_ensbl<OC, OD, OF, NM>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
        obs_ensbl: &mut ObsEnsbl<T, OC, OD>,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: ObsTime<T>,
        OD: AddAssign + ObsData,
        OF: Fn(
                &Self,
                &OC,
                &SVectorView<T, P>,
                &Self::FMST,
                &Self::CSST,
            ) -> Result<OD, ModelError<T>>
            + Sync,
        NM: ObsNoise<T, OC, OD> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        obs_ensbl
            .time_iter_mut()
            .try_for_each(|(conf, mut obs_row)| {
                // Compute time step to next observation.
                let time_step = conf.timestamp() - timer;
                timer = conf.timestamp();

                if time_step < T::zero() {
                    return Err(ModelError::Evolution(time_step));
                } else {
                    model_ensbl
                        .input
                        .column_iter()
                        .zip(model_ensbl.states.iter_mut())
                        .zip(obs_row.column_iter_mut())
                        .try_for_each(|((params, (fm_state, cs_state)), mut obs)| {
                            self.evolve_state(time_step, &params, fm_state, cs_state)?;

                            obs[(0, 0)] = obs_func(self, conf, &params, fm_state, cs_state)?;

                            Ok::<(), ModelError<T>>(())
                        })?;
                }

                Ok::<(), ModelError<T>>(())
            })?;

        if let Some(noise) = opt_noise {
            let mut rng = noise.initialize_rng(37, 23);

            obs_ensbl.ensbl_iter_mut().for_each(|(_, mut col)| {
                noise.generate_noise(&mut col, &mut rng);
            });

            noise.increment_random_seed()
        }

        debug!(
            "simulate_ensbl: {:2.1}k evaluations in {:.0}ms",
            (obs_ensbl.len() * model_ensbl.len()) as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Perform an ensemble forward simulation, in parallel, and generate synthetic observables `OD` for the
    /// given spacecraft observers for a given generating function `OF` and noise model `NM`.
    fn simulate_ensbl_par<OC, OD, OF, NM>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
        obs_ensbl: &mut ObsEnsbl<T, OC, OD>,
        obs_func: &OF,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: ObsTime<T> + Sync,
        OD: AddAssign + ObsData + Scalar,
        OF: Fn(
                &Self,
                &OC,
                &SVectorView<T, P>,
                &Self::FMST,
                &Self::CSST,
            ) -> Result<OD, ModelError<T>>
            + Sync,
        NM: ObsNoise<T, OC, OD> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        obs_ensbl
            .time_iter_mut()
            .try_for_each(|(conf, mut obs_row)| {
                // Compute time step to next observation.
                let time_step = conf.timestamp() - timer;
                timer = conf.timestamp();

                if time_step < T::zero() {
                    return Err(ModelError::Evolution(time_step));
                } else {
                    model_ensbl
                        .input
                        .par_column_iter()
                        .zip(model_ensbl.states.par_iter_mut())
                        .zip(obs_row.par_column_iter_mut())
                        .chunks(Self::RCS)
                        .try_for_each(|mut chunk| {
                            chunk.iter_mut().try_for_each(
                                |((params, (fm_state, cs_state)), obs)| {
                                    self.evolve_state(time_step, params, fm_state, cs_state)?;

                                    obs[(0, 0)] = obs_func(self, conf, params, fm_state, cs_state)?;

                                    Ok::<(), ModelError<T>>(())
                                },
                            )?;

                            Ok::<(), ModelError<T>>(())
                        })?;
                }

                Ok::<(), ModelError<T>>(())
            })?;

        if let Some(noise) = opt_noise {
            obs_ensbl
                .par_ensbl_iter_mut()
                .chunks(Self::RCS)
                .enumerate()
                .for_each(|(cdx, mut chunk)| {
                    let mut rng = noise.initialize_rng(29 * cdx as u64, 23);

                    chunk.iter_mut().for_each(|(_, col)| {
                        noise.generate_noise(col, &mut rng);
                    });
                });

            noise.increment_random_seed()
        }

        debug!(
            "simulate_ensbl_par: {:2.1}k evaluations in {:.0}ms",
            (obs_ensbl.len() * model_ensbl.len()) as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );

        Ok(())
    }

    /// Perform a forward simulation and return the internal coordinates and basis vectors for
    /// the given spacecraft observers.
    fn simulate_icsbasis<OC>(
        &self,
        obs: &Obs<T, OC>,
        params: &SVectorView<T, P>,
        fm_state: &mut Self::FMST,
        cs_state: &mut Self::CSST,
    ) -> Result<DVector<ICSBasis<T, D>>, ModelError<T>>
    where
        OC: ObsPosition<T, D>,
    {
        let mut timer = T::zero();

        let mut obs_ensbl_vector = DVector::zeros(obs.len());

        zip_eq(obs, obs_ensbl_vector.iter_mut()).try_for_each(|(conf, obs)| {
            // Compute time step to next observation.
            let time_step = conf.timestamp() - timer;
            timer = conf.timestamp();

            if time_step < T::zero() {
                return Err(ModelError::Evolution(time_step));
            } else {
                self.evolve_state(time_step, params, fm_state, cs_state)?;
                *obs = self.observe_icsbasis(&conf.position().as_view(), params, cs_state)?;
            }

            Ok::<(), ModelError<T>>(())
        })?;

        Ok(obs_ensbl_vector)
    }

    /// Perform an ensemble forward simulation and return the internal coords and basis
    /// vectors for the given spacecraft observers.
    fn simulate_icsbasis_ensbl<OC>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
        obs_ensbl: &mut ObsEnsbl<T, OC, ICSBasis<T, D>>,
    ) -> Result<(), ModelError<T>>
    where
        OC: ObsPosition<T, D> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        let start = Instant::now();
        let mut timer = T::zero();

        obs_ensbl
            .time_iter_mut()
            .try_for_each(|(conf, mut out_row)| {
                // Compute time step to next observation.
                let time_step = conf.timestamp() - timer;
                timer = conf.timestamp();

                if time_step < T::zero() {
                    return Err(ModelError::Evolution(time_step));
                } else {
                    model_ensbl
                        .input
                        .par_column_iter()
                        .zip(model_ensbl.states.par_iter_mut())
                        .zip(out_row.par_column_iter_mut())
                        .chunks(Self::RCS)
                        .try_for_each(|mut chunk| {
                            chunk.iter_mut().try_for_each(
                                |((params, (fm_state, cs_state)), obs)| {
                                    self.evolve_state(time_step, params, fm_state, cs_state)?;

                                    obs[(0, 0)] = self.observe_icsbasis(
                                        &conf.position().as_view(),
                                        params,
                                        cs_state,
                                    )?;

                                    Ok::<(), ModelError<T>>(())
                                },
                            )?;

                            Ok::<(), ModelError<T>>(())
                        })?;
                }

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "simulate_icsbasis_ensbl: {:2.1}k evaluations in {:.0}ms",
            (obs_ensbl.len() * model_ensbl.len()) as f64 / 1e3,
            start.elapsed().as_millis() as f64
        );
        Ok(())
    }
}

/// A model ensemble object.
///
/// Holds the input parameters and states for an ensemble of model simulations.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize,
    M::FMST: Serialize,
    M::CSST: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    M::FMST: Deserialize<'de>,
    M::CSST: Deserialize<'de>"))]
pub struct ModelEnsbl<T, M, const D: usize, const P: usize>
where
    T: Copy + RealField,
    M: Model<T, D, P>,
{
    /// Input parameters for each ensemble member.
    pub input: OMatrix<T, Const<P>, Dyn>,

    /// Coordinate and forward system states for each ensemble member.
    states: Vec<(M::FMST, M::CSST)>,

    /// Optional ensemble weights for the input parameters.
    pub opt_weights: Option<Vec<T>>,
}

impl<T, M, const P: usize, const D: usize> ModelEnsbl<T, M, D, P>
where
    T: Copy + RealField,
    M: Model<T, D, P>,
{
    /// Create a new [`ModelEnsbl`] from a view.
    pub fn from_view(
        input: MatrixView<T, Const<P>, Dyn>,
        opt_states: Option<Vec<(M::FMST, M::CSST)>>,
        opt_weights: Option<Vec<T>>,
    ) -> Self
    where
        T: Copy + RealField + Sum,
        M::FMST: Clone + Default,
        M::CSST: Clone + Default,
    {
        let size = input.ncols();

        Self {
            input: input.clone_owned(),
            states: opt_states.unwrap_or(vec![(M::FMST::default(), M::CSST::default()); size]),
            opt_weights,
        }
    }

    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of members in the ensemble.
    pub fn len(&self) -> usize {
        self.input.ncols()
    }

    /// Create a new [`ModelEnsbl`] from an existing array .
    pub fn new(
        input: OMatrix<T, Const<P>, Dyn>,
        opt_states: Option<Vec<(M::FMST, M::CSST)>>,
        opt_weights: Option<Vec<T>>,
    ) -> Self
    where
        T: Copy + RealField + Sum,
        M::FMST: Clone + Default,
        M::CSST: Clone + Default,
    {
        let size = input.ncols();

        Self {
            input,
            states: opt_states.unwrap_or(vec![(M::FMST::default(), M::CSST::default()); size]),
            opt_weights,
        }
    }

    /// Create a new [`ModelEnsbl`] from an existing array and initialize the states.
    pub fn new_initialize(
        model: &M,
        input: Matrix<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>,
        opt_weights: Option<Vec<T>>,
    ) -> Result<Self, ModelError<T>>
    where
        T: Copy + RealField + Sum,
        M: Sync,
        M::FMST: Clone + Default + Send,
        M::CSST: Clone + Default + Send,
    {
        let size = input.ncols();

        let mut obj = Self {
            input,
            states: vec![(M::FMST::default(), M::CSST::default()); size],
            opt_weights,
        };

        model.initialize_states_ensbl(&mut obj)?;

        Ok(obj)
    }

    /// Resample the model input parameters from a particle density, and re-initialize the states.
    pub fn new_resampled<GB, GK>(
        model: &M,
        size: usize,
        ptpdf: &ParticleDensity<T, Const<P>, GB, GK>,
        rseed: u64,
    ) -> Result<Self, ModelError<T>>
    where
        T: SampleUniform + Sum,
        M: Sync,
        M::CSST: Clone + Default + Send,
        M::FMST: Clone + Default + Send,
        GB: Sync,
        GK: Sync,
    {
        let start = Instant::now();

        let mut obj = Self::new(
            Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::zeros(size),
            None,
            None,
        );

        obj.input
            .par_column_iter_mut()
            .zip(obj.states.par_iter_mut())
            .chunks(M::RCS)
            .enumerate()
            .try_for_each(|(cdx, mut chunk)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rseed + (cdx * 679389209) as u64);

                chunk
                    .iter_mut()
                    .try_for_each(|(params, (fm_state, cs_state))| {
                        params.set_column(0, &ptpdf.sample_particle(&mut rng));

                        M::initialize_states(model, &params.as_view(), fm_state, cs_state)?;

                        Ok::<(), ModelError<T>>(())
                    })?;

                Ok::<(), ModelError<T>>(())
            })?;

        debug!(
            "new_resampled: {:2.3}M evaluations in {:.2} sec",
            obj.len() as f64 / 1e6,
            start.elapsed().as_millis() as f64 / 1e3
        );

        obj.opt_weights = None;

        Ok(obj)
    }

    /// Serialize this data structure to a file using the JSON5 format.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }

    /// Returns a reference to the indexed state tuple.
    pub fn state(&self, index: usize) -> &(M::FMST, M::CSST) {
        &self.states[index]
    }
}
