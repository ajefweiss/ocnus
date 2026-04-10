use crate::{
    base::{Model, ModelEnsbl, ModelError},
    obs::{
        ObsEnsbl,
        conf::{ObsPosition, ObsTime},
        data::ObsVec,
        noise::ObsNoise,
    },
};
use nalgebra::{Const, RealField, SVectorView, U1};

/// A trait that is shared by all models that can measure the in situ plasma properties.
pub trait Plasma<T, OC, const D: usize, const P: usize>: Model<T, D, P>
where
    T: Copy + RealField,
    OC: ObsTime<T>,
{
    /// Returns the in situ plasma bulk speed.
    fn observe_pbs(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsVec<T, 1>, ModelError<T>>
    where
        OC: ObsPosition<T, D>,
    {
        let q = match Self::transform_ecs_to_ics::<U1, Const<P>>(
            &conf.position().as_view(),
            params,
            cs_state,
        ) {
            Some(value) => value,
            None => {
                return Err(ModelError::Coordinates(conf.position().as_slice().to_vec()));
            }
        };

        Ok(ObsVec::from([self.observe_pbs_ics(
            &q.as_view(),
            params,
            fm_state,
            cs_state,
        )]))
    }

    /// Returns the in situ plasma density.
    fn observe_rho(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsVec<T, 1>, ModelError<T>>
    where
        OC: ObsPosition<T, D>,
    {
        let q = match Self::transform_ecs_to_ics::<U1, Const<P>>(
            &conf.position().as_view(),
            params,
            cs_state,
        ) {
            Some(value) => value,
            None => {
                return Err(ModelError::Coordinates(conf.position().as_slice().to_vec()));
            }
        };

        Ok(ObsVec::from([self.observe_rho_ics(
            &q.as_view(),
            params,
            fm_state,
            cs_state,
        )]))
    }

    /// Returns the in situ plasma temperature.
    fn observe_temp(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsVec<T, 1>, ModelError<T>>
    where
        OC: ObsPosition<T, D>,
    {
        let q = match Self::transform_ecs_to_ics::<U1, Const<P>>(
            &conf.position().as_view(),
            params,
            cs_state,
        ) {
            Some(value) => value,
            None => {
                return Err(ModelError::Coordinates(conf.position().as_slice().to_vec()));
            }
        };

        Ok(ObsVec::from([self.observe_temp_ics(
            &q.as_view(),
            params,
            fm_state,
            cs_state,
        )]))
    }

    /// Returns the in situ plasma bulk speed using internal coordinates.
    fn observe_pbs_ics(
        &self,
        ics: &SVectorView<T, D>,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> T;

    /// Returns the in situ plasma density using internal coordinates.
    fn observe_rho_ics(
        &self,
        ics: &SVectorView<T, D>,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> T;

    /// Returns the in situ plasma temperature using internal coordinates.
    fn observe_temp_ics(
        &self,
        ics: &SVectorView<T, D>,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> T;

    /// Perform an ensemble forward simulation for a plasma density, in parallel, for the given spacecraft observers
    /// and noise model `NM`.
    fn simulate_rho<NM>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
        obs_ensbl: &mut ObsEnsbl<T, OC, ObsVec<T, 1>>,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: ObsPosition<T, D> + Sync,
        NM: ObsNoise<T, OC, ObsVec<T, 1>> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        self.simulate_ensbl_par(model_ensbl, obs_ensbl, &Self::observe_rho, opt_noise)
    }

    /// Perform an ensemble forward simulation for a plasma bulk speed measurement, in parallel, for the given spacecraft observers
    /// and noise model `NM`.
    fn simulate_pbs<NM>(
        &self,
        model_ensbl: &mut ModelEnsbl<T, Self, D, P>,
        obs_ensbl: &mut ObsEnsbl<T, OC, ObsVec<T, 1>>,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: ObsPosition<T, D> + Sync,
        NM: ObsNoise<T, OC, ObsVec<T, 1>> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sync,
    {
        self.simulate_ensbl_par(model_ensbl, obs_ensbl, &Self::observe_pbs, opt_noise)
    }
}
