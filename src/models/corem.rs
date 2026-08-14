use crate::geometry::{AGCSState, XTState};
use bayesfm::{
    ModelError,
    conf::{ConfCamera, ConfPosition},
    geometry::{Geometry, param_value},
    model_impl_coords,
};
use nalgebra::{Const, RealField, SVector, SVectorView, Scalar, VectorView, VectorView3};
use prodef::Density;
use rand_distr::uniform::SampleUniform;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, iter::Sum, marker::PhantomData, ops::Sub};

/// Forward model cs_state type for the CORE models.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct COREState<T> {
    /// Timestamp (sec)
    pub time: T,

    /// Speed (km/s)
    pub speed: T,

    /// Magnetic field scale factor (nT)
    pub magnetic_field: T,
}

impl<T> Default for COREState<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            time: T::zero(),
            speed: T::zero(),
            magnetic_field: T::zero(),
        }
    }
}

/// Magnetic field components for the CORE model.
pub fn core_obs<T, const D: usize>(
    q: &VectorView3<T>,
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
    fm_state: &COREState<T>,
    _cs_state: &XTState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: RealField,
{
    // Extract parameters using their identifiers.
    let tau = param_value("tau", names, params);

    let magnetic_field = fm_state.magnetic_field.clone();

    let (mu, _nu, _s) = (q[0].clone(), q[1].clone(), q[2].clone());

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                let chi = mu.clone() * magnetic_field.clone() * tau.clone()
                    / (T::one() + (tau.clone() * mu.clone()).powi(2));

                let xi = magnetic_field.clone() / (T::one() + (tau * mu).powi(2));

                Ok((chi, xi))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Magnetic field components for the AGCS model.
pub fn agcs_obs<T, const D: usize>(
    q: &VectorView3<T>,
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
    fm_state: &COREState<T>,
    _cs_state: &AGCSState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: RealField,
{
    // Extract parameters using their identifiers.
    let tau = param_value("tau", names, params);

    let magnetic_field = fm_state.magnetic_field.clone();

    let (mu, _nu, _s) = (q[0].clone(), q[1].clone(), q[2].clone());

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                let chi = mu.clone() * magnetic_field.clone() * tau.clone()
                    / (T::one() + (tau.clone() * mu.clone()).powi(2));

                let xi = magnetic_field.clone() / (T::one() + (tau * mu).powi(2));

                Ok((chi, xi))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

macro_rules! impl_corem {
    ($model: ident, $docs: literal, $($coords: ident)::+, $csty: ty, $mag: expr, $params: expr) => {
        #[doc=$docs]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $model<T, G>(G, PhantomData<T>)
        where
            T: RealField;

        impl<T, G> $model<T, G>
        where
            T: RealField,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: G) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        impl<T, OC, G> crate::mag::Magnetometer<T, OC, { $($coords)::+::<f32>::NPARAMS + $params.len() }>
            for $model<T, G>
        where
            T: RealField + rand_distr::uniform::SampleUniform + std::iter::Sum,
            G: 'static + prodef::Density<T, nalgebra::Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>> + Sync,
            OC: bayesfm::conf::ConfPosition<T, 3> + nalgebra::Scalar + Sync,
            for<'a> &'a OC: std::ops::Sub<&'a OC, Output=T>,
        {
            fn observe_mag3_ics(
                &self,
                ics: &SVectorView<T, 3>,
                params: &SVectorView<T, { $($coords)::+::<f32>::NPARAMS + $params.len() }>,
                fm_state: &Self::FMST,
                cs_state: &Self::CSST,
            ) -> Option<SVector<T, 3>> {
                let (chi, xi) = $mag(ics, &Self::PARAM_NAMES, params, fm_state, cs_state).ok()?;

                Some(SVector::<T, 3>::from_column_slice(&[
                    T::zero(),
                    chi,
                    xi,
                ]))
            }
        }

        model_impl_coords!($model, $($coords)::+, $params);

        impl<T, G> bayesfm::Model<T, 3, { $($coords)::+::<f32>::NPARAMS + $params.len() }>
            for $model<T, G>
        where
            T: RealField,
            G: 'static + Density<T, Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>> + Sync,
        {
            type FMST = COREState<T>;

            fn evolve_fmst(
                &self,
                time_step: T,
                params: &VectorView<T, Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>>,
                fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                // Extract parameters using their identifiers.
                let distance_0 =
                    param_value("r_0rs", &Self::PARAM_NAMES, params) * T::from_f64(695510.0).unwrap();
                let diameter_1au = param_value("d_1au", &Self::PARAM_NAMES, params);

                let b_scale = param_value("b_scale", &Self::PARAM_NAMES, params);
                let v_0 = param_value("speed", &Self::PARAM_NAMES, params);
                let v_sw = param_value("sw_speed", &Self::PARAM_NAMES, params);
                let gamma =
                    param_value("sw_gamma", &Self::PARAM_NAMES, params) * T::from_f64(1e-7).unwrap();

                fm_state.time += time_step;

                let delta_v = v_0 - v_sw.clone();

                let sign = match delta_v.partial_cmp(&T::zero()).unwrap() {
                    Ordering::Greater => T::one(),
                    _ => T::neg(T::one()),
                };

                let rt = (sign.clone() / gamma.clone() * (T::one() + sign.clone() * gamma.clone() * delta_v.clone() * fm_state.time.clone()).ln()
                    + v_sw.clone() * fm_state.time.clone()
                    + distance_0)
                    / T::from_f64(1.496e8).unwrap();
                let vt = delta_v.clone() / (T::one() + sign.clone() * gamma.clone() * delta_v.clone() * fm_state.time.clone()) + v_sw;

                let expansion_factor = T::from_f64(1.14).unwrap();
                let b_decay_factor = T::from_f64(-1.68).unwrap();

                // // This sets the expansion and decay to zero, used for debugging purposes.
                // let expansion_factor = T::from_f64(1.0).unwrap();
                // let b_decay_factor = T::from_f64(0.0).unwrap();

                cs_state.rp =
                    diameter_1au * rt.clone().powf(expansion_factor) / T::from_usize(2).unwrap();
                cs_state.rt = (rt.clone() - cs_state.rp.clone()) / T::from_usize(2).unwrap();

                fm_state.magnetic_field = b_scale
                    * (T::from_usize(2).unwrap() * cs_state.rt.clone()).powf(b_decay_factor);
                fm_state.speed = vt;

                Ok(())
            }

            fn initialize_states(
                &self,
                params: &VectorView<T, Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>>,
                fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                Self::initialize_csst(params, cs_state);

                fm_state.time = T::zero();

                Ok(())
            }

            fn prior(&self) -> impl prodef::Density<T, nalgebra::Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>> + 'static {
                self.0.clone()
            }

            fn prior_domain(&self) -> prodef::Domain<T, nalgebra::Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>> {
                (&self.0).domain().clone()
            }

            fn prior_density(&self, params: &nalgebra::SVectorView<T, { $($coords)::+::<f32>::NPARAMS + $params.len() }>) -> Option<T> {
                (&self.0).density(params)
            }
        }

        impl<T, G> bayesfm::EnsembleModel<T, 3, { $($coords)::+::<f32>::NPARAMS + $params.len() }> for $model<T, G>
        where
            T:  nalgebra::RealField,
            G: 'static + prodef::Density<T, nalgebra::Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>> + Sync,
        {
            const RAYON_CHUNK_SIZE: usize = 128;
        }
    };
}

impl_corem!(
    COREModel,
    "The standard 3DCORE magnetic flux rope model.",
    crate::geometry::TTGeometry,
    XTState<T>,
    core_obs,
    ["speed", "b_scale", "tau", "sw_speed", "sw_gamma"]
);

impl_corem!(
    AGCSModel,
    "The analgous GCS magnetic flux rope model.",
    crate::geometry::AGCSGeometry,
    AGCSState<T>,
    agcs_obs,
    ["speed", "b_scale", "tau", "sw_speed", "sw_gamma"]
);

impl<T, OC, G> crate::rho::ElectronDensity<T, OC, 13> for AGCSModel<T, G>
where
    T: RealField + SampleUniform + Sum,
    G: 'static + Density<T, Const<13>> + Sync,
    OC: ConfPosition<T, 3> + Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    Self: Sized,
{
    fn observe_electron_density_ics(
        &self,
        ics: &VectorView3<T>,
        _params: &SVectorView<T, 13>,
        _fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, 1>> {
        // A very simple electron density model.
        Some(SVector::from([
            T::one() / cs_state.rt.clone() / cs_state.rp.clone().powi(2)
            * (T::pi() * ics[2].clone()).sin().powi(2)
            // * (T::two_pi() * ics[1]).cos().powi(2)
            * match ics[0].partial_cmp(&T::one()).unwrap() {
                Ordering::Less => {
                    ics[0].clone().powi(2) * T::from_f64(0.75).unwrap() + T::from_f64(0.25).unwrap()
                }
                Ordering::Equal => T::one(),
                Ordering::Greater => match ics[0].partial_cmp(&T::from_f64(1.25).unwrap()).unwrap()
                {
                    Ordering::Less => T::from_f64(2.0).unwrap() - ics[0].clone().powi(3),
                    _ => T::zero(),
                },
            },
        ]))
    }
}

impl<T, OC, G> crate::rho::ElectronCamera<T, OC, 13> for AGCSModel<T, G>
where
    T: RealField + SampleUniform + Sum,
    G: 'static + Density<T, Const<13>> + Sync,
    OC: ConfCamera<T> + Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    Self: Sized,
{
}

impl<T, OC, G> crate::rho::ElectronDensity<T, OC, 11> for COREModel<T, G>
where
    T: RealField + SampleUniform + Sum,
    G: 'static + Density<T, Const<11>> + Sync,
    OC: ConfPosition<T, 3> + Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    Self: Sized,
{
    fn observe_electron_density_ics(
        &self,
        ics: &VectorView3<T>,
        _params: &SVectorView<T, 11>,
        _fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, 1>> {
        // A very simple electron density model.
        Some(SVector::from([
            T::one() / cs_state.rt.clone() / cs_state.rp.clone().powi(2)
            * (T::pi() * ics[2].clone()).sin().powi(2)
            // * (T::two_pi() * ics[1]).cos().powi(2)
            * match ics[0].partial_cmp(&T::one()).unwrap() {
                Ordering::Less => {
                    ics[0].clone().powi(2) * T::from_f64(0.75).unwrap() + T::from_f64(0.25).unwrap()
                }
                Ordering::Equal => T::one(),
                Ordering::Greater => match ics[0].partial_cmp(&T::from_f64(1.25).unwrap()).unwrap()
                {
                    Ordering::Less => T::from_f64(2.0).unwrap() - ics[0].clone().powi(3),
                    _ => T::zero(),
                },
            },
        ]))
    }
}

impl<T, OC, G> crate::rho::ElectronCamera<T, OC, 11> for COREModel<T, G>
where
    T: RealField + SampleUniform + Sum,
    G: 'static + Density<T, Const<11>> + Sync,
    OC: ConfCamera<T> + Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    Self: Sized,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mag::Magnetometer;
    use approx::ulps_eq;
    use bayesfm::{
        EnsembleModel, EnsembleObservations, EnsembleState,
        conf::{ConfSeries, Location},
        noise::NullNoise,
        obs::ObsCoordBasis,
    };
    use nalgebra::{Dyn, OMatrix, SVector, U11, Vector3};
    use prodef::{ConstantDensity, MultivariateDensity, UniformDensity};

    #[test]
    fn test_core_model() {
        let prior = MultivariateDensity::new(SVector::from([
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(0.5, 1.0).unwrap().into(),
            UniformDensity::new(-0.5, 0.5).unwrap().into(),
            ConstantDensity::new(20.0).into(),
            UniformDensity::new(0.05, 0.25).unwrap().into(),
            ConstantDensity::new(1.0).into(),
            ConstantDensity::new(1125.0).into(),
            UniformDensity::new(5.0, 100.0).unwrap().into(),
            UniformDensity::new(-10.0, 10.0).unwrap().into(),
            ConstantDensity::new(400.0).into(),
            ConstantDensity::new(1.0).into(),
        ]));

        let model = COREModel::new(prior);

        let conf = ConfSeries::from_iter((0..10).map(|i| {
            Location::from((
                72.0 * 3600.0 + i as f32 * 2.0 * 3600.0,
                Vector3::new(1.0, 0.0, 0.0),
            ))
        }));

        let mut input = OMatrix::<f32, U11, Dyn>::zeros(1);
        input.set_column(
            0,
            &SVector::<f32, 11>::from([
                0.0_f32.to_radians(),
                1.0_f32.to_radians(),
                0.0_f32.to_radians(),
                20.0,
                0.15,
                1.0,
                1300.0,
                20.0,
                0.6,
                400.0,
                1.0,
            ]),
        );

        let mut model_ensbl = EnsembleState::new(input, None, None);
        let mut obs_ensbl =
            EnsembleObservations::new(Location::default(), conf.clone(), 1, None).unwrap();
        let mut obs_ensbl_diag = EnsembleObservations::<_, ObsCoordBasis<f32, 3>>::new(
            Location::default(),
            conf.clone(),
            1,
            None,
        )
        .unwrap();

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut model_ensbl,
                &mut obs_ensbl,
                &COREModel::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_basis_ensbl_par(&mut model_ensbl, &mut obs_ensbl_diag)
            .expect("simulation failed");

        assert!(ulps_eq!(
            obs_ensbl.output(0)[1][1],
            -18.194891,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[2][1],
            -19.41521,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[4][2],
            -0.12136115,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[2].coordinates()[0],
            0.52137023,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[3].coordinates()[1],
            0.87713426,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[4].coordinates()[0],
            0.21290788,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[5].coordinates()[2],
            0.5,
            max_ulps = 5,
            epsilon = 1e-5
        ));
    }

    #[test]
    fn test_core_elliptic_model() {
        let prior = MultivariateDensity::<_, U11>::new(SVector::from([
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(0.5, 1.0).unwrap().into(),
            UniformDensity::new(-0.5, 0.5).unwrap().into(),
            ConstantDensity::new(20.0).into(),
            UniformDensity::new(0.05, 0.25).unwrap().into(),
            ConstantDensity::new(1.0).into(),
            ConstantDensity::new(1125.0).into(),
            UniformDensity::new(5.0, 100.0).unwrap().into(),
            UniformDensity::new(-10.0, 10.0).unwrap().into(),
            ConstantDensity::new(400.0).into(),
            ConstantDensity::new(1.0).into(),
        ]));

        let model = COREModel::new(prior);

        let conf = ConfSeries::from_iter((0..10).map(|i| {
            Location::from((
                72.0 * 3600.0 + i as f32 * 2.0 * 3600.0,
                Vector3::new(1.0, 0.0, 0.0),
            ))
        }));

        let mut input = OMatrix::<f32, U11, Dyn>::zeros(1);
        input.set_column(
            0,
            &SVector::<f32, 11>::from([
                0.0_f32.to_radians(),
                1.0_f32.to_radians(),
                0.0_f32.to_radians(),
                20.0,
                0.15,
                0.99,
                1300.0,
                20.0,
                0.6,
                400.0,
                1.0,
            ]),
        );

        let mut model_ensbl = EnsembleState::new(input, None, None);
        let mut obs_ensbl =
            EnsembleObservations::new(Location::default(), conf.clone(), 1, None).unwrap();
        let mut obs_ensbl_diag = EnsembleObservations::<_, ObsCoordBasis<f32, 3>>::new(
            Location::default(),
            conf.clone(),
            1,
            None,
        )
        .unwrap();

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut model_ensbl,
                &mut obs_ensbl,
                &COREModel::<f32, _>::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_basis_ensbl_par(&mut model_ensbl, &mut obs_ensbl_diag)
            .expect("simulation failed");
    }
}
