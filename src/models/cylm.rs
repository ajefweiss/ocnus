use crate::geometry::XCState;
use bayesfm::{
    ModelError,
    geometry::{Geometry, param_value},
    math::bessel_jn,
    model_impl_coords,
};
use nalgebra::{Const, RealField, SVector, SVectorView, VectorView, VectorView3};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, marker::PhantomData};

/// Linear force-free magnetic field components.
pub fn cc_lff_chi_xi<T, const D: usize>(
    q: &VectorView3<T>,
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
    _fm_state: &(),
    _cs_state: &XCState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: RealField,
{
    // Extract parameters using their identifiers.
    let b = param_value("b_scale", names, params);
    let alpha_signed = param_value("alpha", names, params);
    let radius = param_value("radius", names, params);

    let (alpha, sign) = match alpha_signed
        .partial_cmp(&T::zero())
        .expect("alpha value is NaN")
    {
        Ordering::Less => (-alpha_signed, T::neg(T::one())),
        _ => (alpha_signed, T::one()),
    };

    let (mu, _nu, _z) = (q[0].clone(), q[1].clone(), q[2].clone());

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => match mu.partial_cmp(&T::zero()).unwrap() {
                Ordering::Equal => Ok((T::zero(), b)),
                Ordering::Greater => {
                    let chi_lff = b.clone()
                        * sign
                        * bessel_jn(alpha.clone() * mu.clone() * radius.clone(), 1)
                        / mu.clone()
                        / radius.clone()
                        / T::two_pi();

                    let xi_lff: T = b * bessel_jn(alpha * mu * radius, 0);

                    Ok((chi_lff, xi_lff))
                }
                Ordering::Less => {
                    panic!("mu value is less than zero");
                }
            },
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Magnetic field components for the model from Nieves-Chinchilla et al. 2016
pub fn cc_nc16_chi_xi<T, const D: usize>(
    q: &VectorView3<T>,
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
    _fm_state: &(),
    _cs_state: &XCState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: RealField,
{
    // Extract parameters using their identifiers.
    let b = param_value("b_star", names, params);

    let tau = param_value("tau", names, params);

    let c10 = param_value("c10", names, params);

    let radius = param_value("radius", names, params);

    let (mu, _nu, _z) = (q[0].clone(), q[1].clone(), q[2].clone());

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => match mu.partial_cmp(&T::zero()).unwrap() {
                Ordering::Equal => Ok((T::zero(), b)),
                Ordering::Greater => {
                    let chi_nc16 = -b.clone() / radius / c10 / T::two_pi();

                    let xi_nc16: T = b * (tau.clone() - mu.powi(2)) / tau;

                    Ok((chi_nc16, xi_nc16))
                }
                Ordering::Less => {
                    panic!("mu value is less than zero");
                }
            },
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Uniform twist magnetic field components.
pub fn cc_ut_chi_xi<T, const D: usize>(
    q: &VectorView3<T>,
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
    _fm_state: &(),
    _cs_state: &XCState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: RealField,
{
    // Extract parameters using their identifiers.
    let b = param_value("b_scale", names, params);
    let tau = param_value("tau", names, params);
    let radius = param_value("radius", names, params);

    let (mu, _nu, _z) = (q[0].clone(), q[1].clone(), q[2].clone());

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => match mu.partial_cmp(&T::zero()).unwrap() {
                Ordering::Equal => Ok((T::zero(), b)),
                Ordering::Greater => {
                    let chi_ut = b.clone() * tau.clone()
                        / (T::one() + (tau.clone() * mu.clone() * radius.clone()).powi(2))
                        / T::two_pi();
                    let xi_ut = b / (T::one() + (tau * mu * radius).powi(2));

                    Ok((chi_ut, xi_ut))
                }
                Ordering::Less => {
                    panic!("mu value is less than zero");
                }
            },
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Magnetic field components for the elliptic-circular hybrid model.
pub fn ec_hybrid_obs<T, const D: usize>(
    q: &VectorView3<T>,
    names: &SVector<&'static str, D>,
    params: &SVectorView<T, D>,
    _fm_state: &(),
    _cs_state: &XCState<T>,
) -> Result<(T, T), ModelError<T>>
where
    T: RealField,
{
    // Extract parameters using their identifiers.
    let radius = param_value("radius", names, params);
    let b = param_value("b_scale", names, params);
    let lambda = param_value("lambda", names, params);
    let alpha_signed = param_value("alpha", names, params);
    let tau = param_value("tau", names, params);
    let delta = param_value("cs_delta", names, params);

    let (alpha, sign) = match alpha_signed
        .partial_cmp(&T::zero())
        .expect("alpha value is NaN")
    {
        Ordering::Less => (-alpha_signed, T::neg(T::one())),
        _ => (alpha_signed, T::one()),
    };

    let (mu, _nu, _z) = (q[0].clone(), q[1].clone(), q[2].clone());

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                match mu.partial_cmp(&T::zero()).unwrap() {
                    Ordering::Equal => Ok((T::zero(), b / delta.powi(2))),
                    Ordering::Greater => {
                        // LFF terms.
                        let chi_lff = b.clone()
                            * sign
                            * bessel_jn(alpha.clone() * mu.clone() * radius.clone(), 1)
                            / mu.clone()
                            / radius.clone()
                            / T::two_pi()
                            / delta.clone().powi(2);

                        let xi_lff: T = b.clone()
                            * bessel_jn(alpha.clone() * mu.clone() * radius.clone(), 0)
                            / delta.clone().powi(2);

                        // UT terms.
                        let chi_ut = b.clone() * tau.clone()
                            / (T::one() + (tau.clone() * mu.clone() * radius.clone()).powi(2))
                            / T::two_pi()
                            / delta.clone().powi(2);
                        let xi_ut = b / (T::one() + (tau * mu * radius).powi(2)) / delta.powi(2);

                        Ok((
                            chi_lff * lambda.clone() + (T::one() - lambda.clone()) * chi_ut,
                            xi_lff * lambda.clone() + (T::one() - lambda) * xi_ut,
                        ))
                    }
                    Ordering::Less => {
                        panic!("mu value is less than zero");
                    }
                }
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

macro_rules! impl_cylm_model {
    ($model: ident, $docs: literal, $($coords: ident)::+, $mag: expr, $params: expr) => {
        #[doc=$docs]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $model<T, G>(G, PhantomData<T>)
        where
            T: RealField;

        impl<T, G> $model<T, G>
        where
            T: Default + RealField,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: G) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        model_impl_coords!($model, $($coords)::+, $params);

        impl<T, OC, G> crate::mag::Magnetometer<T, OC, { $($coords)::+::<f32>::NPARAMS + $params.len() }>
            for $model<T, G>
        where
            T: RealField + rand_distr::uniform::SampleUniform + std::iter::Sum,
            G: 'static + prodef::Density<T, nalgebra::Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>> + Sync,
            OC: bayesfm::conf::ConfPosition<T, 3> + nalgebra::Scalar + Sync,
            for<'a> &'a OC: std::ops::Sub<&'a OC, Output=T>

        {
            fn observe_mag3_ics(
                &self,
                ics: &SVectorView<T, 3>,
                params: &SVectorView<T, { $($coords)::+::<f32>::NPARAMS + $params.len() }>,
                fm_state: &Self::FMST,
                cs_state: &Self::CSST,
            ) -> Option<SVector<T, 3>> {
                let (chi, xi) = $mag(ics, &Self::PARAM_NAMES, params, fm_state, cs_state).ok()?;

                Some(SVector::<T, 3>::from_column_slice(&[T::zero(), chi, xi]))
            }
        }

        impl<T, G> bayesfm::Model<T, 3, { $($coords)::+::<f32>::NPARAMS + $params.len() }> for $model<T, G>
        where
            T: nalgebra::RealField,
            G: 'static + prodef::Density<T, nalgebra::Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>> + Sync,
        {
            type FMST = ();

            fn evolve_fmst(
                &self,
                time_step: T,
                params: &VectorView<T, Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>>,
                _fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                // Extract parameters using their identifiers.
                let vel = param_value("speed", &Self::PARAM_NAMES, params)
                    / T::from_f64(1.496e8).unwrap();

                cs_state.x += vel * time_step as T;

                Ok(())
            }

            fn initialize_states(
                &self,
                params: &VectorView<T, Const<{ $($coords)::+::<f32>::NPARAMS + $params.len() }>>,
                _fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                Self::initialize_csst(params, cs_state);

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

impl_cylm_model!(
    CCLFFModel,
    "Circular-cylindrical linear force-free magnetic flux rope model.",
    crate::geometry::CCGeometry,
    cc_lff_chi_xi,
    ["speed", "b_scale", "alpha"]
);

impl_cylm_model!(
    NC16Model,
    "Circular-cylindrical model from Nieves-Chinchilla et al. 2016.",
    crate::geometry::CCGeometry,
    cc_nc16_chi_xi,
    ["speed", "b_star", "tau", "c10"]
);

impl_cylm_model!(
    CCUTModel,
    "Circular-cylindrical uniform twist magnetic flux rope model.",
    crate::geometry::CCGeometry,
    cc_ut_chi_xi,
    ["speed", "b_scale", "tau"]
);

impl_cylm_model!(
    ECHModel,
    "Elliptic-cylindrical hybrid flux rope model.",
    crate::geometry::ECGeometry,
    ec_hybrid_obs,
    ["speed", "b_scale", "lambda", "alpha", "tau"]
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mag::Magnetometer;
    use approx::ulps_eq;
    use bayesfm::{
        EnsembleModel, EnsembleObservations, EnsembleState,
        conf::{BasicConf, ConfSeries},
        noise::NullNoise,
    };
    use nalgebra::{DMatrix, Dyn, OMatrix, SVector, U8, U12, Vector3};
    use prodef::{ConstantDensity, MultivariateDensity, UniformDensity};

    #[test]
    fn test_cclff_model() {
        let prior = MultivariateDensity::new(SVector::from([
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(0.5, 1.0).unwrap().into(),
            UniformDensity::new(0.05, 0.1).unwrap().into(),
            UniformDensity::new(0.1, 0.5).unwrap().into(),
            ConstantDensity::new(1125.0).into(),
            UniformDensity::new(5.0, 100.0).unwrap().into(),
            UniformDensity::new(-2.4, 2.4).unwrap().into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
        ]));

        let model = CCLFFModel::new(prior);

        let conf: ConfSeries<BasicConf<f32, 3>> = ConfSeries::from_iter((0..10).map(|i| {
            BasicConf::from((
                224640.0 + i as f32 * 3600.0 * 2.0,
                Vector3::new(1.0, 0.0, 0.0),
            ))
        }));

        let mut input = OMatrix::<f32, U8, Dyn>::zeros(1);
        input.set_column(
            0,
            &SVector::from([
                5.0_f32.to_radians(),
                -3.0_f32.to_radians(),
                0.1,
                0.25,
                0.0,
                600.0,
                20.0,
                1.0 / 0.25,
            ]),
        );

        let mut ensbl = EnsembleState::new(input, None, None);
        let mut obs_ensbl =
            EnsembleObservations::new(BasicConf::default(), conf.clone(), 1, None).unwrap();

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obs_ensbl,
                &CCLFFModel::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obs_ensbl.output(0)[0][1],
            19.200111,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[2][1],
            19.712679,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[4][2],
            -1.6972067,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            ensbl.state(0).1.z,
            -0.025034571,
            max_ulps = 5,
            epsilon = 1e-5
        ));
    }

    #[test]
    fn test_cclff_model_twist() {
        let prior = MultivariateDensity::new(SVector::from([
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(0.5, 1.0).unwrap().into(),
            UniformDensity::new(0.05, 0.1).unwrap().into(),
            UniformDensity::new(0.1, 0.5).unwrap().into(),
            ConstantDensity::new(1125.0).into(),
            UniformDensity::new(5.0, 100.0).unwrap().into(),
            UniformDensity::new(-2.4, 2.4).unwrap().into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
        ]));

        let model = CCLFFModel::new(prior);

        let conf = ConfSeries::from_iter(
            (0..2).map(|i| BasicConf::from((0.0, Vector3::new(i as f32, 0.0, 0.0)))),
        );

        let mut input = OMatrix::<f32, U8, Dyn>::zeros(1);
        input.set_column(
            0,
            &SVector::from([
                0_f32.to_radians(),
                0_f32.to_radians(),
                0.0,
                1.56,
                0.0,
                600.0,
                20.0,
                2.4048254f32,
            ]),
        );

        let mut ensbl = EnsembleState::new(input, None, None);
        let mut obs_ensbl =
            EnsembleObservations::new(BasicConf::default(), conf.clone(), 1, None).unwrap();

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obs_ensbl,
                &CCLFFModel::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(obs_ensbl.output(0)[0][1], 20.0));
        assert!(ulps_eq!(obs_ensbl.output(0)[0][2], 0.0));

        assert!(ulps_eq!(
            obs_ensbl.output(0)[1][1],
            0.0,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[1][2] / bessel_jn(2.4048254f32, 1),
            20.0,
            max_ulps = 5,
            epsilon = 1e-5
        ));
    }

    #[test]
    fn test_ccut_model() {
        let prior = MultivariateDensity::new(SVector::from([
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(0.5, 1.0).unwrap().into(),
            UniformDensity::new(0.05, 0.1).unwrap().into(),
            UniformDensity::new(0.1, 0.5).unwrap().into(),
            ConstantDensity::new(1125.0).into(),
            UniformDensity::new(5.0, 100.0).unwrap().into(),
            UniformDensity::new(-2.4, 2.4).unwrap().into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
        ]));

        let model = CCUTModel::new(prior);

        let conf = ConfSeries::from_iter((0..8).map(|i| {
            BasicConf::from((
                224640.0 + i as f32 * 3600.0 * 2.0,
                Vector3::new(1.0, 0.0, 0.0),
            ))
        }));

        let mut input = OMatrix::<f32, U8, Dyn>::zeros(1);
        input.set_column(
            0,
            &SVector::from([
                5.0_f32.to_radians(),
                -3.0_f32.to_radians(),
                0.1,
                0.25,
                0.0,
                600.0,
                20.0,
                1.0 / 0.25,
            ]),
        );

        let mut ensbl = EnsembleState::new(input, None, None);
        let mut obs_ensbl =
            EnsembleObservations::new(BasicConf::default(), conf.clone(), 1, None).unwrap();

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obs_ensbl,
                &CCUTModel::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obs_ensbl.output(0)[0][1],
            17.279219,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[2][1],
            19.186895,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[4][2],
            -2.3241665,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            ensbl.state(0).1.z,
            -0.025034571,
            max_ulps = 5,
            epsilon = 1e-5
        ));
    }

    #[test]
    fn test_ccut_model_twist() {
        let prior = MultivariateDensity::new(SVector::from([
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(0.5, 1.0).unwrap().into(),
            UniformDensity::new(0.05, 0.1).unwrap().into(),
            UniformDensity::new(0.1, 0.5).unwrap().into(),
            ConstantDensity::new(1125.0).into(),
            UniformDensity::new(5.0, 100.0).unwrap().into(),
            UniformDensity::new(-2.4, 2.4).unwrap().into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
        ]));

        let model = CCUTModel::new(prior);

        let conf = ConfSeries::from_iter(
            (0..2).map(|i| BasicConf::from((0.0, Vector3::new(i as f32, 0.0, 0.0)))),
        );

        let mut input = OMatrix::<f32, U8, Dyn>::zeros(1);
        input.set_column(
            0,
            &SVector::from([
                0_f32.to_radians(),
                0_f32.to_radians(),
                0.0,
                1.56,
                0.0,
                600.0,
                20.0,
                1.0,
            ]),
        );

        let mut ensbl = EnsembleState::new(input, None, None);
        let mut obs_ensbl =
            EnsembleObservations::new(BasicConf::default(), conf.clone(), 1, None).unwrap();

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obs_ensbl,
                &CCUTModel::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obs_ensbl.output(0)[0][1],
            20.0,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[0][2],
            0.0,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(obs_ensbl.output(0)[1][1], 10.0, max_ulps = 5,));
        assert!(ulps_eq!(obs_ensbl.output(0)[1][2], 10.0, epsilon = 1e-5));
    }

    #[test]
    fn test_ech_model() {
        let prior = MultivariateDensity::new(SVector::from([
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(-1.0, 1.0).unwrap().into(),
            UniformDensity::new(0.05, 0.1).unwrap().into(),
            UniformDensity::new(0.1, 1.0).unwrap().into(),
            UniformDensity::new(0.1, 0.5).unwrap().into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
            ConstantDensity::new(1125.0).into(),
            UniformDensity::new(5.0, 100.0).unwrap().into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
            UniformDensity::new(-10.0, 10.0).unwrap().into(),
            UniformDensity::new(-10.0, 10.0).unwrap().into(),
        ]));

        let model = ECHModel::new(prior);

        let conf = ConfSeries::from_iter((0..8).map(|i| {
            BasicConf::from((
                224640.0 + i as f32 * 3600.0 * 2.0,
                Vector3::new(1.0, 0.0, 0.0),
            ))
        }));

        let mut input = OMatrix::<f32, U12, Dyn>::zeros(1);

        // UT
        input.set_column(
            0,
            &SVector::from([
                5.0_f32.to_radians(),
                -3.0_f32.to_radians(),
                0.0_f32.to_radians(),
                0.1,
                1.0,
                0.25,
                0.0,
                600.0,
                20.0,
                0.0,
                1.0 / 0.25,
                1.0 / 0.25,
            ]),
        );

        let mut ensbl = EnsembleState::new(input, None, None);
        let mut obs_ensbl =
            EnsembleObservations::new(BasicConf::default(), conf.clone(), 1, None).unwrap();

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obs_ensbl,
                &ECHModel::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obs_ensbl.output(0)[0][1],
            17.279219,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl.output(0)[2][1],
            19.186895,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            obs_ensbl.output(0)[4][2],
            -2.3241665,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        assert!(ulps_eq!(
            ensbl.state(0).1.z,
            -0.025034571,
            max_ulps = 5,
            epsilon = 1e-5
        ));

        let mut input = OMatrix::<f32, U12, Dyn>::zeros(1);

        // LFF
        input.set_column(
            0,
            &SVector::from([
                5.0_f32.to_radians(),
                -3.0_f32.to_radians(),
                0.0_f32.to_radians(),
                0.1,
                1.0,
                0.25,
                0.0,
                600.0,
                20.0,
                1.0,
                1.0 / 0.25,
                1.0 / 0.25,
            ]),
        );

        let mut ensbl = EnsembleState::new(input, None, None);

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obs_ensbl,
                &ECHModel::observe_mag3,
                &mut None::<&mut NullNoise>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obs_ensbl.output(0)[0][1],
            19.200111,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[2][1],
            19.712679,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            obs_ensbl.output(0)[4][2],
            -1.6972067,
            max_ulps = 5,
            epsilon = 1e-5
        ));
        assert!(ulps_eq!(
            ensbl.state(0).1.z,
            -0.025034571,
            max_ulps = 5,
            epsilon = 1e-5
        ));
    }

    #[test]
    fn test_ech_model_fisher() {
        let prior = MultivariateDensity::new(SVector::from([
            UniformDensity::new(-1.0_f32, 1.0).unwrap().into(),
            UniformDensity::new(f32::pi() - 1.25, f32::pi() + 1.25)
                .unwrap()
                .into(),
            ConstantDensity::new(0.0).into(),
            UniformDensity::new(-0.65, 0.65).unwrap().into(),
            ConstantDensity::new(1.0).into(),
            ConstantDensity::new(0.11).into(),
            UniformDensity::new(0.25, 1.0).unwrap().into(),
            ConstantDensity::new(500.0).into(),
            UniformDensity::new(10.0, 25.0).unwrap().into(),
            ConstantDensity::new(0.0).into(),
            ConstantDensity::new(0.0).into(),
            UniformDensity::new(0.0, 20.0).unwrap().into(),
        ]));

        let model = ECHModel::new(prior);

        let conf = ConfSeries::from_iter((0..5).map(|i| {
            BasicConf::from((
                14400.0 + i as f32 * 4.0 * 3600.0,
                Vector3::new(1.0, 0.0, 0.0),
            ))
        }));

        let params = SVector::from([
            -0.22_f32, 3.06, 0.0, -0.11, 1.0, 0.11, 0.87, 500.0, 15.4, 0.0, 0.0, 5.2,
        ]);

        let fisher_info = model
            .fisher_mag(
                (&BasicConf::default(), &conf),
                &params.as_view(),
                &DMatrix::from_diagonal_element(5, 5, 0.1_f32),
            )
            .expect("fisher information computation failed");

        assert!(ulps_eq!(
            fisher_info[(0, 0)],
            9593.1,
            max_ulps = 1,
            epsilon = 1e-1
        ));
    }
}
