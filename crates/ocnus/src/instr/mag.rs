use crate::{
    base::{Model, ModelEnsbl, ModelError},
    methods::fisher_information_matrix,
    obs::{Obs, ObsEnsbl, conf::ObsPosition, data::ObsVec, noise::ObsNoise},
};
use nalgebra::{DMatrix, DVector, Dyn, RealField, SMatrix, SVector, SVectorView, Vector4};
use num_traits::{AsPrimitive, Float};
use prodef::{domain::UDomain, multinormal::MultiNormalDensity};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

use std::iter::Sum;

/// A trait that is shared by all models that can measure in situ magnetic fields.
pub trait Magnetometer<T, OC, const P: usize>: Model<T, 3, P>
where
    T: Copy + RealField + SampleUniform + Sum,
    OC: ObsPosition<T, 3>,
{
    /// Compute the fisher information matrix (FIM) using magnetic field vector observations.
    fn fisher_mag(
        &self,
        obs: &Obs<T, OC>,
        params: &SVectorView<T, P>,
        covariance_matrix: &DMatrix<T>,
    ) -> Result<SMatrix<T, P, P>, ModelError<T>>
    where
        T: Float,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
        Self: Sync,
        Self::CSST: Clone + Default + Send,
        Self::FMST: Clone + Default + Send,
    {
        assert!(obs.len() == covariance_matrix.nrows());
        assert!(obs.len() == covariance_matrix.ncols());

        let likelihood = MultiNormalDensity::from_matrix(
            covariance_matrix.clone(),
            DVector::zeros(obs.len()),
            UDomain::new(Dyn(obs.len())),
        )
        .unwrap();

        fisher_information_matrix(self, obs, params, &Self::observe_mag3, &likelihood)
    }

    /// Returns an in situ magnetic field vector observation.
    fn observe_mag3(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsVec<T, 3>, ModelError<T>> {
        let position = conf.position();

        let q = match Self::transform_ecs_to_ics(&position.as_view(), params, cs_state) {
            Some(value) => value,
            None => {
                return Err(ModelError::Coordinates(position.as_slice().to_vec()));
            }
        };

        match self.observe_mag3_ics(&q.as_view(), params, fm_state, cs_state) {
            Some(b_q) => {
                let b_s =
                    Self::contravariant_vector(&q.as_view(), &b_q.as_view(), params, cs_state)
                        .expect("failed to construct contravariant basis");

                Ok(ObsVec::<T, 3>::from(b_s))
            }
            None => Ok(ObsVec::<T, 3>::from([
                (-T::one()).sqrt(),
                (-T::one()).sqrt(),
                (-T::one()).sqrt(),
            ])),
        }
    }

    /// Returns the in situ magnetic field vector in internal coordinates.
    fn observe_mag3_ics(
        &self,
        ics: &SVectorView<T, 3>,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, 3>>;

    /// Returns an in situ magnetic field vector observation with magnitude.
    fn observe_mag4(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsVec<T, 4>, ModelError<T>> {
        let measurement = Self::observe_mag3(self, conf, params, fm_state, cs_state)?;

        Ok(ObsVec::<T, 4>::from(Vector4::from([
            measurement.sum_of_squares().sqrt(),
            measurement[0],
            measurement[1],
            measurement[2],
        ])))
    }

    /// Perform an ensemble forward simulation for a magnetic field measurement, in parallel, for the given spacecraft observers
    /// and noise model `NM`.
    fn simulate_mag3<NM>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, 3, P>,
        obs_ensbl: &mut ObsEnsbl<T, OC, ObsVec<T, 3>>,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: Sync,
        NM: ObsNoise<T, OC, ObsVec<T, 3>> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        self.simulate_ensbl_par(model_ensbl, obs_ensbl, &Self::observe_mag3, opt_noise)
    }

    /// Perform an ensemble forward simulation for a magnetic field measurement, in parallel, for the given spacecraft observers
    /// and noise model `NM`.
    fn simulate_mag4<NM>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, 3, P>,
        obs_ensbl: &mut ObsEnsbl<T, OC, ObsVec<T, 4>>,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: Sync,
        NM: ObsNoise<T, OC, ObsVec<T, 4>> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        self.simulate_ensbl_par(model_ensbl, obs_ensbl, &Self::observe_mag4, opt_noise)
    }
}
