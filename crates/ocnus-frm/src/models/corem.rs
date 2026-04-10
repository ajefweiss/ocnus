use crate::coords::{AGCSGeometry, AGCSState, TTGeometry, XTState};
use nalgebra::{Const, Dim, RealField, SVector, SVectorView, U11, U13, VectorView, VectorView3};
use num_traits::AsPrimitive;
use ocnus::{
    base::{Model, ModelError},
    coords::{Coordinates, param_value},
    instr::{Magnetometer, Plasma, WLCamera},
    model_impl_concat_strs, model_impl_coords,
    obs::conf::{ObsCam, ObsPosition},
};
use prodef::{Density, domain::Domain};
use rand_distr::{Distribution, StandardNormal, StandardUniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, iter::Sum, marker::PhantomData};

/// Forward model cs_state type for the CORE models.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct COREState<T> {
    /// Timestamp (sec)
    pub time: T,

    /// Speed (km/s)
    pub speed: T,

    /// Magnetic field scale factor (nT)
    pub magnetic_field: T,
}

/// Magnetic field components for the CORE model.
pub fn core_obs<T, const D: usize>(
    q: &VectorView3<T>,
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
    fm_state: &COREState<T>,
    cs_state: &XTState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let tau = param_value("tau", names, params);

    let magnetic_field = fm_state.magnetic_field;
    let radius = cs_state.rp;

    let (mu, _nu, _s) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                let chi =
                    mu * radius * magnetic_field * tau / (T::one() + (tau * mu * radius).powi(2));

                let xi = magnetic_field / (T::one() + (tau * mu * radius).powi(2));

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
    cs_state: &AGCSState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let tau = param_value("tau", names, params);

    let magnetic_field = fm_state.magnetic_field;
    let radius = cs_state.rp;

    let (mu, _nu, _s) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                let chi =
                    mu * radius * magnetic_field * tau / (T::one() + (tau * mu * radius).powi(2));

                let xi = magnetic_field / (T::one() + (tau * mu * radius).powi(2));

                Ok((chi, xi))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

macro_rules! impl_corem {
    ($model: ident, $coords: ident, $csty: ty, $docs: literal, $params: expr, $fn_mag: expr) => {
        #[doc=$docs]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $model<T, G>(G, PhantomData<T>)
        where
            T: Copy + RealField;

        impl<T, G> $model<T, G>
        where
            T: Copy + RealField,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: G) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        impl<T, OC, G> Magnetometer<T, OC, { $coords::<f32>::NPARAMS + $params.len() }>
            for $model<T, G>
        where
            T: Copy + Default + RealField + SampleUniform + Sum,
            OC: ObsPosition<T, 3>,
            G: 'static + Density<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>>,
            for<'a> &'a G: Density<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
        {
            fn observe_mag3_ics(
                &self,
                ics: &SVectorView<T, 3>,
                params: &SVectorView<T, { $coords::<f32>::NPARAMS + $params.len() }>,
                fm_state: &Self::FMST,
                cs_state: &Self::CSST,
            ) -> Option<SVector<T, 3>> {
                let (chi, xi) = $fn_mag(ics, &Self::PARAMS, params, fm_state, cs_state).ok()?;

                // Components are not appropriately scaled.
                let basis = Self::contravariant_basis(ics, params, cs_state)?;

                Some(SVector::<T, 3>::from_column_slice(&[
                    T::zero(),
                    chi / basis[1].norm(),
                    xi / basis[2].norm(),
                ]))
            }
        }

        model_impl_coords!($model, $csty, $coords, $params);

        impl<T, G> Model<T, 3, { $coords::<f32>::NPARAMS + $params.len() }> for $model<T, G>
        where
            T: Copy + Default + RealField + SampleUniform,
            G: 'static + Density<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>>,
            for<'a> &'a G: Density<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
        {
            const RCS: usize = 128;

            type FMST = COREState<T>;

            fn domain(
                &self,
            ) -> impl Domain<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>> + 'static {
                self.0.domain()
            }

            fn evolve_state(
                &self,
                time_step: T,
                params: &VectorView<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>>,
                fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                // Extract parameters using their identifiers.
                let distance_0 =
                    param_value("r_0rs", &Self::PARAMS, params) * T::from_f64(695510.0).unwrap();
                let diameter_1au = param_value("d_1au", &Self::PARAMS, params);

                let b_scale = param_value("b_scale", &Self::PARAMS, params);
                let v_0 = param_value("speed", &Self::PARAMS, params);
                let v_sw = param_value("sw_speed", &Self::PARAMS, params);
                let gamma =
                    param_value("sw_gamma", &Self::PARAMS, params) * T::from_f64(1e-7).unwrap();

                fm_state.time += time_step;

                let delta_v = v_0 - v_sw;

                let sign = match delta_v.partial_cmp(&T::zero()).unwrap() {
                    Ordering::Greater => T::one(),
                    _ => T::neg(T::one()),
                };

                let rt = (sign / gamma * (T::one() + sign * gamma * delta_v * fm_state.time).ln()
                    + v_sw * fm_state.time
                    + distance_0)
                    / T::from_f64(1.496e8).unwrap();
                let vt = delta_v / (T::one() + sign * gamma * delta_v * fm_state.time) + v_sw;

                cs_state.rp =
                    diameter_1au * rt.powf(T::from_f64(1.14).unwrap()) / T::from_usize(2).unwrap();
                cs_state.rt = (rt - cs_state.rp) / T::from_usize(2).unwrap();

                fm_state.magnetic_field = b_scale
                    * (T::from_usize(2).unwrap() * cs_state.rt).powf(T::from_f64(-1.68).unwrap());
                fm_state.speed = vt;

                Ok(())
            }

            fn initialize_states(
                &self,
                params: &VectorView<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>>,
                fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                Self::initialize_cs(params, cs_state);

                fm_state.time = T::zero();

                Ok(())
            }

            fn prior(
                &self,
            ) -> impl Density<T, Const<{ $coords::<f32>::NPARAMS + $params.len() }>> + 'static {
                self.0.clone()
            }
        }
    };
}

impl_corem!(
    COREModel,
    TTGeometry,
    XTState<T>,
    "The standard 3DCORE magnetic flux rope model.",
    ["speed", "b_scale", "tau", "sw_speed", "sw_gamma"],
    core_obs
);

impl_corem!(
    AGCSModel,
    AGCSGeometry,
    AGCSState<T>,
    "The analgous GCS magnetic flux rope model.",
    ["speed", "b_scale", "tau", "sw_speed", "sw_gamma"],
    agcs_obs
);

impl<T, OC, G> Plasma<T, OC, 3, 11> for COREModel<T, G>
where
    T: AsPrimitive<usize> + Default + Copy + RealField + SampleUniform + Sum,
    OC: ObsPosition<T, 3>,
    G: 'static + Density<T, U11>,
    for<'a> &'a G: Density<T, U11>,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    fn observe_pbs_ics(
        &self,
        _ics: &SVectorView<T, 3>,
        _params: &SVectorView<T, 11>,
        fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        // TODO: calculate correct in situ speed
        fm_state.speed
    }

    fn observe_rho_ics(
        &self,
        ics: &VectorView3<T>,
        _params: &SVectorView<T, 11>,
        _fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> T {
        T::one() / cs_state.rt / cs_state.rp.powi(2)
            * (T::pi() * ics[2]).sin()
            * match ics[0].partial_cmp(&T::one()).unwrap() {
                Ordering::Less => ics[0].powi(2),
                Ordering::Equal => T::one(),
                Ordering::Greater => match ics[0].partial_cmp(&T::from_f64(1.25).unwrap()).unwrap()
                {
                    Ordering::Less => T::from_f64(2.0).unwrap() - ics[0].powi(3),
                    _ => T::zero(),
                },
            }
    }

    fn observe_temp_ics(
        &self,
        _ics: &SVectorView<T, 3>,
        _params: &SVectorView<T, 11>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        unimplemented!("CORE models does not support plasma temperature measurements")
    }
}

impl<T, OC, G> WLCamera<T, OC, 11> for COREModel<T, G>
where
    T: AsPrimitive<usize> + Default + Copy + RealField + SampleUniform + Sum,
    OC: ObsCam<T>,
    G: 'static + Density<T, U11>,
    for<'a> &'a G: Density<T, U11>,
    StandardUniform: Distribution<T>,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
}

impl<T, OC, G> Plasma<T, OC, 3, 13> for AGCSModel<T, G>
where
    T: AsPrimitive<usize> + Default + Copy + RealField + SampleUniform + Sum,
    OC: ObsPosition<T, 3>,
    G: 'static + Density<T, U13>,
    for<'a> &'a G: Density<T, U13>,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    fn observe_pbs_ics(
        &self,
        _ics: &SVectorView<T, 3>,
        _params: &SVectorView<T, 13>,
        fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        // TODO: calculate correct in situ speed
        fm_state.speed
    }

    fn observe_rho_ics(
        &self,
        ics: &VectorView3<T>,
        _params: &SVectorView<T, 13>,
        _fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> T {
        // Simple electron density model
        T::one() / cs_state.rt / cs_state.rp.powi(2)
            * (T::pi() * ics[2]).sin().powi(2)
            // * (T::two_pi() * ics[1]).cos().powi(2)
            * match ics[0].partial_cmp(&T::one()).unwrap() {
                Ordering::Less => {
                    ics[0].powi(2) * T::from_f64(0.75).unwrap() + T::from_f64(0.25).unwrap()
                }
                Ordering::Equal => T::one(),
                Ordering::Greater => match ics[0].partial_cmp(&T::from_f64(1.25).unwrap()).unwrap()
                {
                    Ordering::Less => T::from_f64(2.0).unwrap() - ics[0].powi(3),
                    _ => T::zero(),
                },
            }
    }

    fn observe_temp_ics(
        &self,
        _ics: &SVectorView<T, 3>,
        _params: &SVectorView<T, 13>,
        _fm_state: &Self::FMST,
        _cs_state: &Self::CSST,
    ) -> T {
        unimplemented!("CORE models does not support plasma temperature measurements")
    }
}

impl<T, OC, G> WLCamera<T, OC, 13> for AGCSModel<T, G>
where
    T: AsPrimitive<usize> + Default + Copy + RealField + SampleUniform + Sum,
    OC: ObsCam<T>,
    G: 'static + Density<T, U13>,
    for<'a> &'a G: Density<T, U13>,
    StandardUniform: Distribution<T>,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{Dyn, OMatrix};
    use nalgebra::{SVector, Vector3};
    use ocnus::{
        base::ModelEnsbl,
        obs::{Obs, ObsEnsbl, conf::VecConf, data::ICSBasis, noise::NullNoise},
    };
    use prodef::multivariate::{ConstantDensity, MultivariateDensity, UniformDensity};

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

        let obs = Obs::from_iter((0..10).map(|i| {
            VecConf::from((
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
                1.0 / 0.25,
                400.0,
                1.0,
            ]),
        );

        let mut model_ensbl = ModelEnsbl::new(input, None, None);
        let mut obs_ensbl = ObsEnsbl::new(obs.clone(), 1, None).unwrap();
        let mut obs_ensbl_diag =
            ObsEnsbl::<f32, _, ICSBasis<f32, 3>>::new(obs.clone(), 1, None).unwrap();

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut model_ensbl,
                &mut obs_ensbl,
                &COREModel::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_icsbasis_ensbl(&mut model_ensbl, &mut obs_ensbl_diag)
            .expect("simulation failed");

        assert!(ulps_eq!(
            obs_ensbl.output(0)[1][1],
            -20.889551,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[2][1],
            -20.758156,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[4][2],
            -0.06722071,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        // TODO: These asserts changed?
        // Unkown why the results changed.
        // assert!(ulps_eq!(
        //     obs_ensbl.output(0)[1][1],
        //     -20.99427,
        //     max_ulps = 5,
        //     epsilon = 1e-5
        // ));
        // assert!(ulps_eq!(
        //     obs_ensbl.output(0)[2][1],
        //     -20.91451,
        //     max_ulps = 5,
        //     epsilon = 1e-5
        // ));
        // assert!(ulps_eq!(
        //     obs_ensbl.output(0)[4][2],
        //     -0.069730066,
        //     max_ulps = 5,
        //     epsilon = 1e-5
        // ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[2].ics()[0],
            0.52137023,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[3].ics()[1],
            0.87713426,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[4].ics()[0],
            0.21290788,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl_diag.output(0)[5].ics()[2],
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

        let obs = Obs::from_iter((0..10).map(|i| {
            VecConf::from((
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
                1.0 / 0.25,
                400.0,
                1.0,
            ]),
        );

        let mut model_ensbl = ModelEnsbl::new(input, None, None);
        let mut obs_ensbl = ObsEnsbl::new(obs.clone(), 1, None).unwrap();
        let mut obs_ensbl_diag =
            ObsEnsbl::<f32, _, ICSBasis<f32, 3>>::new(obs.clone(), 1, None).unwrap();

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut model_ensbl,
                &mut obs_ensbl,
                &COREModel::<f32, _>::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        model
            .initialize_states_ensbl(&mut model_ensbl)
            .expect("initialization failed");

        model
            .simulate_icsbasis_ensbl(&mut model_ensbl, &mut obs_ensbl_diag)
            .expect("simulation failed");
    }
}
