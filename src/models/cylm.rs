use crate::{
    base::{Model, ModelError, ScConf},
    coords::{CCGeometry, Coordinates, ECGeometry, XCState, param_value, param_value_static},
    math::bessel_jn,
    models::{concat_strs, reimpl_coords},
    obsty::{InSituMagnetometer, ObserVec},
    stats::{Density, DensityRange},
};
use nalgebra::{Const, Dim, RealField, SVector, U1, Vector3, VectorView, VectorView3};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, iter::Sum, marker::PhantomData};

/// Linear force-free magnetic field Chi (nu) and Xi (s) functions.
pub fn cc_lff_chi_xi<T, const D: usize>(
    q: Vector3<T>,
    names: &SVector<&'static str, D>,
    params: &SVector<T, D>,
) -> Result<(T, T), ModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let b = param_value(
        "b_scale",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();
    let alpha_signed = param_value(
        "alpha",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();
    let radius = param_value(
        "radius",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();

    let (alpha, sign) = match alpha_signed
        .partial_cmp(&T::zero())
        .expect("alpha value is NaN")
    {
        Ordering::Less => (-alpha_signed, T::neg(T::one())),
        _ => (alpha_signed, T::one()),
    };

    let (mu, _nu, _z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                // Bessel function evaluation uses 11 terms.
                let chi_lff = b * radius * sign * bessel_jn(alpha * mu * radius, 1);

                let xi_lff: T =
                    T::two_pi() * mu * radius.powi(2) * b * bessel_jn(alpha * mu * radius, 0);

                Ok((chi_lff, xi_lff))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Magnetic field Chi (nu) and Xi (s) functions for the model from Nieves-Chinchilla et al. 2016
pub fn cc_nc16_chi_xi<T, const D: usize>(
    q: Vector3<T>,
    names: &SVector<&'static str, D>,
    params: &SVector<T, D>,
) -> Result<(T, T), ModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let b = param_value_static("b_scale", names, params).unwrap();

    let tau = param_value_static("tau", names, params).unwrap();

    let c10 = param_value_static("c10", names, params).unwrap();

    let radius = param_value_static("radius", names, params).unwrap();

    let (mu, _nu, _z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                let chi_nc16 = -b * radius / c10 * mu;

                let xi_nc16: T = T::two_pi() * mu * radius.powi(2) * b * (tau - mu.powi(2));

                Ok((chi_nc16, xi_nc16))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Uniform twist magnetic field observable.
pub fn cc_ut_chi_xi<T, const D: usize>(
    q: Vector3<T>,
    names: &SVector<&'static str, D>,
    params: &SVector<T, D>,
) -> Result<(T, T), ModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let b = param_value(
        "b_scale",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();
    let tau = param_value(
        "tau",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();
    let radius = param_value(
        "radius",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();

    let (mu, _nu, _z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                let chi_ut =
                    mu * radius.powi(2) * b * tau / (T::one() + (tau * mu * radius).powi(2));
                let xi_ut = T::two_pi() * mu * radius.powi(2) * b
                    / (T::one() + (tau * mu * radius).powi(2));

                Ok((chi_ut, xi_ut))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

/// Magnetic field configuration for the elliptic-circular hybrid model.
pub fn ec_hybrid_obs<T, const D: usize>(
    q: Vector3<T>,
    names: &SVector<&'static str, D>,
    params: &SVector<T, D>,
) -> Result<(T, T), ModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let radius = param_value_static("radius", names, params).unwrap();
    let b = param_value_static("b_scale", names, params).unwrap();
    let lambda = param_value_static("lambda", names, params).unwrap();
    let alpha_signed = param_value_static("alpha", names, params).unwrap();
    let tau = param_value_static("tau", names, params).unwrap();

    let (alpha, sign) = match alpha_signed
        .partial_cmp(&T::zero())
        .expect("alpha value is NaN")
    {
        Ordering::Less => (-alpha_signed, T::neg(T::one())),
        _ => (alpha_signed, T::one()),
    };

    let (mu, _nu, _z) = (q[0], q[1], q[2]);

    match mu.partial_cmp(&T::one()) {
        Some(ord) => match ord {
            Ordering::Greater => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
            _ => {
                // Bessel function evaluation uses 11 terms.
                let chi_lff = b * radius * sign * bessel_jn(alpha * mu * radius, 1);
                let xi_lff: T =
                    T::two_pi() * mu * radius.powi(2) * b * bessel_jn(alpha * mu * radius, 0);

                // UT terms.
                let chi_ut =
                    mu * radius.powi(2) * b * tau / (T::one() + (tau * mu * radius).powi(2));
                let xi_ut = T::two_pi() * mu * radius.powi(2) * b
                    / (T::one() + (tau * mu * radius).powi(2));

                Ok((
                    chi_lff * lambda + (T::one() - lambda) * chi_ut,
                    xi_lff * lambda + (T::one() - lambda) * xi_ut,
                ))
            }
        },
        None => Ok(((-T::one()).sqrt(), (-T::one()).sqrt())),
    }
}

macro_rules! impl_cylm {
    ($model: ident, $coords: ident, $docs: literal, $params: expr, $fn_mag: expr) => {
        #[doc=$docs]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $model<T, P>(P, PhantomData<T>)
        where
            T: Copy + RealField;

        impl<T, P> $model<T, P>
        where
            T: Copy + RealField,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: P) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        impl<T, P> InSituMagnetometer<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>
            for $model<T, P>
        where
            T: Copy + Default + RealField + SampleUniform + Sum,
            for<'x> &'x P: Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
        {
            fn observe_mag3(
                &self,
                scconf: &ScConf<T>,
                params: &SVector<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                _fm_state: &Self::FMST,
                cs_state: &Self::CSST,
            ) -> Result<ObserVec<T, 3>, ModelError<T>> {
                let sc_pos = scconf.position();

                let q = match Self::transform_ecs_to_ics(
                    &sc_pos.as_view(),
                    &params.generic_view(
                        (0, 0),
                        (
                            Const::<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                            Const::<1>,
                        ),
                    ),
                    cs_state,
                ) {
                    Some(value) => value,
                    None => {
                        return Err(ModelError::CoordinateTransform(sc_pos.into_owned()));
                    }
                };

                let params_view = &params.generic_view(
                    (0, 0),
                    (
                        Const::<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                        Const::<1>,
                    ),
                );

                match q[0].partial_cmp(&T::zero()) {
                    Some(ord) => match ord {
                        Ordering::Equal => {
                            let b_q = Vector3::new(
                                T::zero(),
                                T::zero(),
                                param_value("b_scale", &Self::PARAMS, params_view).unwrap(),
                            );

                            let b_s = Self::contravariant_vector(
                                &q.as_view(),
                                &b_q.as_view(),
                                params_view,
                                cs_state,
                            )
                            .expect("failed to construct contravariant basis");

                            Ok(ObserVec::<T, 3>::from(b_s))
                        }
                        _ => {
                            let (chi, xi) = $fn_mag(q, &Self::PARAMS, params)?;

                            let chi_xi_det = chi
                                / Self::detg(&q.as_view(), params_view, cs_state)
                                    .expect("failed to construct contravariant basis");

                            let xi_lff_det = xi
                                / Self::detg(&q.as_view(), params_view, cs_state)
                                    .expect("failed to construct contravariant basis");

                            let b_q = Vector3::new(T::zero(), chi_xi_det, xi_lff_det);

                            let b_s = Self::contravariant_vector(
                                &q.as_view(),
                                &b_q.as_view(),
                                params_view,
                                cs_state,
                            )
                            .expect("failed to construct contravariant basis");

                            Ok(ObserVec::<T, 3>::from(b_s))
                        }
                    },
                    None => Ok(ObserVec::<T, 3>::from([
                        (-T::one()).sqrt(),
                        (-T::one()).sqrt(),
                        (-T::one()).sqrt(),
                    ])),
                }
            }
        }

        reimpl_coords!($model, $coords, $params);

        impl<T, P> Model<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }> for $model<T, P>
        where
            T: Copy + Default + RealField + SampleUniform,
            for<'x> &'x P: Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
        {
            const RCS: usize = 128;

            type FMST = ();

            fn forward(
                &self,
                time_step: T,
                params: &VectorView<T, Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>>,
                _fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                // Extract parameters using their identifiers.
                let vel = param_value("velocity", &Self::PARAMS, params).unwrap()
                    / T::from_f32(1.496e8).unwrap();

                cs_state.x += vel * time_step as T;

                Ok(())
            }

            fn get_range(
                &self,
            ) -> SVector<DensityRange<T>, { $coords::<f32>::PARAMS_COUNT + $params.len() }> {
                (&self.0).get_range()
            }

            fn initialize_states(
                &self,
                params: &VectorView<T, Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>>,
                _fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                Self::initialize_cs(params, cs_state);

                Ok(())
            }

            fn model_prior(
                &self,
            ) -> impl Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }> {
                &self.0
            }
        }
    };
}

impl_cylm!(
    CCLFFModel,
    CCGeometry,
    "Circular-cylindrical linear force-free magnetic flux rope model.",
    ["velocity", "b_scale", "alpha"],
    cc_lff_chi_xi
);

impl_cylm!(
    NC16Model,
    CCGeometry,
    "Circular-cylindrical model from Nieves-Chinchilla et al. 2016.",
    ["velocity", "b_scale", "tau", "c10"],
    cc_nc16_chi_xi
);

impl_cylm!(
    CCUTModel,
    CCGeometry,
    "Circular-cylindrical uniform twist magnetic flux rope model.",
    ["velocity", "b_scale", "tau"],
    cc_ut_chi_xi
);

impl_cylm!(
    ECHModel,
    ECGeometry,
    "Elliptic-cylindrical hybrid flux rope model.",
    ["velocity", "b_scale", "lambda", "alpha", "tau"],
    ec_hybrid_obs
);

#[cfg(test)]
mod tests {
    use std::f32;

    use super::*;
    use crate::{
        base::{ModelEnsbl, Obser, ScObs},
        obsty::NullNoise,
        stats::{ConstantDensity, MultivariateDensity, UniformDensity},
    };
    use approx::ulps_eq;
    use nalgebra::SVector;

    #[test]
    fn test_cclff_model() {
        let prior = MultivariateDensity::<_, 8>::new(&[
            UniformDensity::new(-1.0, 1.0).unwrap(),
            UniformDensity::new(0.5, 1.0).unwrap(),
            UniformDensity::new(0.05, 0.1).unwrap(),
            UniformDensity::new(0.1, 0.5).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new(5.0, 100.0).unwrap(),
            UniformDensity::new(-2.4, 2.4).unwrap(),
            UniformDensity::new(0.0, 1.0).unwrap(),
        ]);

        let model = CCLFFModel::new(prior);

        let sc = ScObs::from_iterator((0..8).map(|i| {
            (
                224640.0 + i as f32 * 3600.0 * 2.0,
                ScConf::Position(Vector3::new(1.0, 0.0, 0.0)),
            )
        }));

        let mut ensbl = ModelEnsbl::new(1, None);
        let mut obser = Obser::<f32, ObserVec<f32, 3>>::new(sc.clone(), 1);

        ensbl.ptpdf.set_particle(
            0,
            &SVector::<f32, 8>::from([
                5.0_f32.to_radians(),
                -3.0_f32.to_radians(),
                0.1,
                0.25,
                0.0,
                600.0,
                20.0,
                1.0 / 0.25,
            ])
            .as_view(),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obser,
                &CCLFFModel::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obser.get_output(0)[0][1],
            19.554552,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[2][1],
            20.083601,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[4][2],
            -1.7143152,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            ensbl.cs_states[0].z,
            0.025129674,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
    }

    #[test]
    fn test_cclff_model_twist() {
        let prior = MultivariateDensity::<_, 8>::new(&[
            UniformDensity::new(-1.0, 1.0).unwrap(),
            UniformDensity::new(0.5, 1.0).unwrap(),
            UniformDensity::new(0.05, 0.1).unwrap(),
            UniformDensity::new(0.1, 0.5).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new(5.0, 100.0).unwrap(),
            UniformDensity::new(-2.4, 2.4).unwrap(),
            UniformDensity::new(0.0, 1.0).unwrap(),
        ]);

        let model = CCLFFModel::new(prior);

        let sc = ScObs::from_iterator(
            (0..2).map(|i| (0.0, ScConf::Position(Vector3::new(i as f32, 0.0, 0.0)))),
        );

        let mut ensbl = ModelEnsbl::new(1, None);
        let mut obser = Obser::<f32, ObserVec<f32, 3>>::new(sc.clone(), 1);

        ensbl.ptpdf.set_particle(
            0,
            &SVector::<f32, 8>::from([
                0_f32.to_radians(),
                0_f32.to_radians(),
                0.0,
                1.56,
                0.0,
                600.0,
                20.0,
                2.4048254f32,
            ])
            .as_view(),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obser,
                &CCLFFModel::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(obser.get_output(0)[0][1], 20.0));
        assert!(ulps_eq!(obser.get_output(0)[0][2], 0.0));

        assert!(ulps_eq!(
            obser.get_output(0)[1][1],
            0.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[1][2] / bessel_jn(2.4048254f32, 1),
            20.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
    }

    #[test]
    fn test_ccut_model() {
        let prior = MultivariateDensity::<_, 8>::new(&[
            UniformDensity::new(-1.0, 1.0).unwrap(),
            UniformDensity::new(0.5, 1.0).unwrap(),
            UniformDensity::new(0.05, 0.1).unwrap(),
            UniformDensity::new(0.1, 0.5).unwrap(),
            UniformDensity::new(0.0, 1.0).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new(5.0, 100.0).unwrap(),
            UniformDensity::new(-10.0, 10.0).unwrap(),
        ]);

        let model = CCUTModel::new(prior);

        let sc = ScObs::from_iterator((0..8).map(|i| {
            (
                224640.0 + i as f32 * 3600.0 * 2.0,
                ScConf::Position(Vector3::new(1.0, 0.0, 0.0)),
            )
        }));

        let mut ensbl = ModelEnsbl::new(1, None);
        let mut obser = Obser::<f32, ObserVec<f32, 3>>::new(sc.clone(), 1);

        ensbl.ptpdf.set_particle(
            0,
            &SVector::<f32, 8>::from([
                5.0_f32.to_radians(),
                -3.0_f32.to_radians(),
                0.1,
                0.25,
                0.0,
                600.0,
                20.0,
                1.0 / 0.25,
            ])
            .as_view(),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obser,
                &CCUTModel::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obser.get_output(0)[0][1],
            17.718813,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[2][1],
            19.706617,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[4][2],
            -2.3474102,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            ensbl.cs_states[0].z,
            0.025129674,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
    }

    #[test]
    fn test_ccut_model_twist() {
        let prior = MultivariateDensity::<_, 8>::new(&[
            UniformDensity::new(-1.0, 1.0).unwrap(),
            UniformDensity::new(0.5, 1.0).unwrap(),
            UniformDensity::new(0.05, 0.1).unwrap(),
            UniformDensity::new(0.1, 0.5).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new(5.0, 100.0).unwrap(),
            UniformDensity::new(-2.4, 2.4).unwrap(),
            UniformDensity::new(0.0, 1.0).unwrap(),
        ]);

        let model = CCUTModel::new(prior);

        let sc = ScObs::from_iterator(
            (0..2).map(|i| (0.0, ScConf::Position(Vector3::new(i as f32, 0.0, 0.0)))),
        );

        let mut ensbl = ModelEnsbl::new(1, None);
        let mut obser = Obser::<f32, ObserVec<f32, 3>>::new(sc.clone(), 1);

        ensbl.ptpdf.set_particle(
            0,
            &SVector::<f32, 8>::from([
                0_f32.to_radians(),
                0_f32.to_radians(),
                0.0,
                1.56,
                0.0,
                600.0,
                20.0,
                1.0,
            ])
            .as_view(),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obser,
                &CCUTModel::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obser.get_output(0)[0][1],
            20.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[0][2],
            0.0,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(obser.get_output(0)[1][1], 10.0, max_ulps = 5,));
        assert!(ulps_eq!(
            obser.get_output(0)[1][2],
            10.0,
            epsilon = 32.0 * f32::EPSILON
        ));
    }

    #[test]
    fn test_ech_model() {
        let prior = MultivariateDensity::<_, 12>::new(&[
            UniformDensity::new(-1.0, 1.0).unwrap(),
            UniformDensity::new(-1.0, 1.0).unwrap(),
            UniformDensity::new(-1.0, 1.0).unwrap(),
            UniformDensity::new(0.05, 0.1).unwrap(),
            UniformDensity::new(0.1, 1.0).unwrap(),
            UniformDensity::new(0.1, 0.5).unwrap(),
            UniformDensity::new(0.0, 1.0).unwrap(),
            ConstantDensity::new(1125.0),
            UniformDensity::new(5.0, 100.0).unwrap(),
            UniformDensity::new(0.0, 1.0).unwrap(),
            UniformDensity::new(-10.0, 10.0).unwrap(),
            UniformDensity::new(-10.0, 10.0).unwrap(),
        ]);

        let model = ECHModel::new(prior);

        let sc = ScObs::from_iterator((0..8).map(|i| {
            (
                224640.0 + i as f32 * 3600.0 * 2.0,
                ScConf::Position(Vector3::new(1.0, 0.0, 0.0)),
            )
        }));

        let mut ensbl = ModelEnsbl::new(1, None);
        let mut obser = Obser::<f32, ObserVec<f32, 3>>::new(sc.clone(), 1);

        // UT
        ensbl.ptpdf.set_particle(
            0,
            &SVector::<f32, 12>::from([
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
            ])
            .as_view(),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obser,
                &ECHModel::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obser.get_output(0)[0][1],
            17.718813,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[2][1],
            19.70662,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(
            obser.get_output(0)[4][2],
            -2.3474102,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            ensbl.cs_states[0].z,
            0.025129674,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        // LFF
        ensbl.ptpdf.set_particle(
            0,
            &SVector::<f32, 12>::from([
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
            ])
            .as_view(),
        );

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &mut ensbl,
                &mut obser,
                &ECHModel::observe_mag3,
                &mut None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        assert!(ulps_eq!(
            obser.get_output(0)[0][1],
            19.554552,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[2][1],
            20.083601,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            obser.get_output(0)[4][2],
            -1.7143152,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            ensbl.cs_states[0].z,
            0.025129674,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
    }
}
