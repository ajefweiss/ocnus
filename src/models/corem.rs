use crate::{
    base::{OcnusModel, OcnusModelError, ScObs, ScObsConf},
    coords::{OcnusCoords, TTGeometry, TTState, param_value},
    models::concat_strs,
    obser::{MeasureInSituMagneticFields, ObserVec},
    stats::{Density, DensityRange},
};
use nalgebra::{Const, Dim, RealField, SVector, U1, Vector3, VectorView, VectorView3};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, iter::Sum, marker::PhantomData};

/// Forward model cs_state type for the CORE models.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[allow(missing_docs)]
pub struct COREState<T> {
    pub time: T,
    pub velocity: T,
    pub magnetic_field: T,
}

/// Magnetic field for the CORE model.
pub fn core_obs<T, const D: usize>(
    q: Vector3<T>,
    names: &SVector<&'static str, D>,
    params: &SVector<T, D>,
    fm_state: &COREState<T>,
    cs_state: &TTState<T>,
) -> Result<(T, T), OcnusModelError<T>>
where
    T: Copy + RealField,
{
    // Extract parameters using their identifiers.
    let tau = param_value(
        "tau",
        names,
        &params.as_view::<Const<D>, U1, U1, Const<D>>(),
    )
    .unwrap();

    let magnetic_field = fm_state.magnetic_field;
    let radius = cs_state.minor_radius;

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

macro_rules! impl_core_forward_model {
    ($model: ident, $coords: ident, $params: expr, $fn_obs: tt, $docs: literal) => {
        #[doc=$docs]
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $model<T, P>(P, PhantomData<T>)
        where
            T: Copy + RealField;

        impl<T, P> $model<T, P>
        where
            T: Copy + RealField,
            for<'x> &'x P: Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`]")]
            pub fn new(pdf: P) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        impl<T, P>
            MeasureInSituMagneticFields<
                T,
                { $coords::<f32>::PARAMS_COUNT + $params.len() },
                COREState<T>,
                TTState<T>,
            > for $model<T, P>
        where
            T: Copy + RealField + Sum,
            Self: Send + Sync,
        {
            fn observe_mag3(
                &self,
                scobs: &ScObs<T>,
                params: &SVector<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                fm_state: &COREState<T>,
                cs_state: &TTState<T>,
            ) -> Result<ObserVec<T, 3>, OcnusModelError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Position(r) => *r,
                });

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
                        return Err(OcnusModelError::CoordinateTransform(sc_pos.into_owned()));
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
                            let (chi, xi) = core_obs(q, &Self::PARAMS, params, fm_state, cs_state)?;

                            let chi_det = chi
                                * Self::detg(
                                    &Vector3::new(q[0], T::zero(), q[2]).as_view(),
                                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>(0),
                                    cs_state,
                                )
                                .unwrap()
                                / Self::detg(&q.as_view(), params_view, cs_state)
                                    .expect("failed to compute determinant");

                            let xi_det = if ((q[1] % T::one()) < T::from_f32(0.25).unwrap())
                                || ((q[1] % T::one()) > T::from_f32(0.75).unwrap())
                            {
                                xi * Self::detg(
                                    &Vector3::new(q[0], T::zero(), T::from_f32(0.5).unwrap())
                                        .as_view(),
                                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>(0),
                                    cs_state,
                                )
                                .unwrap()
                            } else {
                                xi * Self::detg(
                                    &Vector3::new(
                                        q[0],
                                        T::from_f32(0.5).unwrap(),
                                        T::from_f32(0.5).unwrap(),
                                    )
                                    .as_view(),
                                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>(0),
                                    cs_state,
                                )
                                .unwrap()
                            } / Self::detg(&q.as_view(), params_view, cs_state).expect("failed to compute determinant");

                            let b_q = Vector3::new(T::zero(), chi_det, xi_det);

                            let b_s = Self::contravariant_vector_normalized(
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

        // Re-implement the OcnusCoords trait because we have no inheritance.
        // Here we make use of the fact that the parameters for the coordinates are at the front
        // and we pass on smaller fixed views of each parameter vector.
        impl<T, P> OcnusCoords<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, TTState<T>>
            for $model<T, P>
        where
            T: Copy + RealField,
            Self: Send + Sync,
        {
            const PARAMS: SVector<&'static str, { $coords::<f32>::PARAMS_COUNT + $params.len() }> =
                SVector::from_array_storage(concat_strs!($coords::<f32>::PARAMS, $params));

            fn contravariant_basis<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &TTState<T>,
            ) -> Option<[Vector3<T>; 3]> {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn detg<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &TTState<T>,
            ) -> Option<T> {
                $coords::detg(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<RStride: Dim, CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &mut TTState<T>,
            ) {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &TTState<T>,
            ) -> Option<Vector3<T>> {
                $coords::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &TTState<T>,
            ) -> Option<Vector3<T>> {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }
        }

        impl<T, P>
            OcnusModel<
                T,
                { $coords::<f32>::PARAMS_COUNT + $params.len() },
                COREState<T>,
                TTState<T>,
            > for $model<T, P>
        where
            T: Copy + Default + RealField + SampleUniform,
            P: for<'x> Deserialize<'x> + Serialize,
            for<'x> &'x P: Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }>,
            StandardNormal: Distribution<T>,
            usize: AsPrimitive<T>,
            Self: OcnusCoords<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }, TTState<T>>,
        {
            const RCS: usize = 128;

            fn forward(
                &self,
                time_step: T,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    U1,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                >,
                fm_state: &mut COREState<T>,
                cs_state: &mut TTState<T>,
            ) -> Result<(), OcnusModelError<T>> {
                // Extract parameters using their identifiers.
                let distance_0 = param_value("distance_0", &Self::PARAMS, params).unwrap()
                    * T::from_f32(695510.0).unwrap();
                let diameter_1au = param_value("diameter_1au", &Self::PARAMS, params).unwrap();

                let b_scale = param_value("b_scale", &Self::PARAMS, params).unwrap();
                let v_0 = param_value("velocity", &Self::PARAMS, params).unwrap();
                let v_sw = param_value("sw_velocity", &Self::PARAMS, params).unwrap();
                let gamma = param_value("sw_gamma", &Self::PARAMS, params).unwrap()
                    * T::from_f32(1e-7).unwrap();

                fm_state.time += time_step;

                let delta_v = v_0 - v_sw;

                let sign = match delta_v.partial_cmp(&T::zero()).unwrap() {
                    Ordering::Greater => T::one(),
                    _ => T::neg(T::one()),
                };

                let rt = (sign / gamma * (T::one() + sign * gamma * delta_v * fm_state.time).ln()
                    + v_sw * fm_state.time
                    + distance_0)
                    / T::from_f32(1.496e8).unwrap();
                let vt = delta_v / (T::one() + sign * gamma * delta_v * fm_state.time) + v_sw;

                cs_state.minor_radius =
                    diameter_1au * rt.powf(T::from_f32(1.14).unwrap()) / T::from_usize(2).unwrap();
                cs_state.major_radius = (rt - cs_state.minor_radius) / T::from_usize(2).unwrap();

                fm_state.magnetic_field = b_scale
                    * (T::from_usize(2).unwrap() * cs_state.major_radius)
                        .powf(T::from_f32(-1.68).unwrap());
                fm_state.velocity = vt;

                Ok(())
            }

            fn get_range(
                &self,
            ) -> SVector<DensityRange<T>, { $coords::<f32>::PARAMS_COUNT + $params.len() }> {
                (&self.0).get_range()
            }

            fn initialize_states(
                &self,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    U1,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                >,
                fm_state: &mut COREState<T>,
                cs_state: &mut TTState<T>,
            ) -> Result<(), OcnusModelError<T>> {
                // Extract parameters using their identifiers.
                let b_scale = param_value("b_scale", &Self::PARAMS, params).unwrap();
                let v_0 = param_value("velocity", &Self::PARAMS, params).unwrap();

                Self::initialize_cs(params, cs_state);

                fm_state.time = T::zero();
                fm_state.magnetic_field = b_scale
                    * (T::from_usize(2).unwrap() * cs_state.major_radius)
                        .powf(T::from_f32(-1.68).unwrap());
                fm_state.velocity = v_0;

                Ok(())
            }

            fn observe_ics_basis(
                &self,
                scobs: &ScObs<T>,
                params: &VectorView<T, Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>>,
                _fm_state: &COREState<T>,
                cs_state: &TTState<T>,
            ) -> Result<ObserVec<T, 12>, OcnusModelError<T>> {
                let sc_pos = Vector3::from(match scobs.configuration() {
                    ScObsConf::Position(r) => *r,
                });

                let q = match Self::transform_ecs_to_ics(&sc_pos.as_view(), params, cs_state) {
                    Some(value) => value,
                    None => {
                        return Err(OcnusModelError::CoordinateTransform(sc_pos.into_owned()));
                    }
                };

                let (mu, nu, s) = (q[0], q[1], q[2]);

                let [e1, e2, e3] = Self::contravariant_basis(
                    &Vector3::from([mu, nu, s]).as_view(),
                    params,
                    cs_state,
                )
                .expect("failed to construct contravariant basis");

                Ok(ObserVec::<T, 12>::from([
                    mu, nu, s, e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
                ]))
            }

            fn model_prior(
                &self,
            ) -> impl Density<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }> {
                &self.0
            }
        }
    };
}

impl_core_forward_model!(
    COREModel,
    TTGeometry,
    ["velocity", "b_scale", "tau", "sw_velocity", "sw_gamma"],
    core_obs,
    "Standard 3DCORE magnetic flux rope model."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        base::{OcnusEnsbl, ScObsSeries},
        obser::NullNoise,
        stats::{ConstantDensity, MultivariateDensity, UniformDensity},
    };
    use approx::ulps_eq;
    use nalgebra::{DMatrix, SVector};

    #[test]
    fn test_core_circ_model() {
        let prior = MultivariateDensity::<_, 11>::new(&[
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((0.5, 1.0)).unwrap(),
            UniformDensity::new((-0.5, 0.5)).unwrap(),
            ConstantDensity::new(20.0),
            UniformDensity::new((0.05, 0.25)).unwrap(),
            ConstantDensity::new(1.0),
            ConstantDensity::new(1125.0),
            UniformDensity::new((5.0, 100.0)).unwrap(),
            UniformDensity::new((-10.0, 10.0)).unwrap(),
            ConstantDensity::new(400.0),
            ConstantDensity::new(1.0),
        ]);

        let range = (&prior).get_range();
        let model = COREModel::new(prior);

        let sc = ScObsSeries::<f32>::from_iterator((0..10).map(|i| {
            ScObs::new(
                72.0 * 3600.0 + i as f32 * 2.0 * 3600.0,
                ScObsConf::Position([1.0, 0.0, 0.0]),
            )
        }));

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);
        let mut output_diag = DMatrix::<ObserVec<f32, 12>>::zeros(sc.len(), 1);

        ensbl.ptpdf.particles_mut().set_column(
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

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &COREModel::<f32, MultivariateDensity<f32, 11>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ics_basis_ensbl(&sc, &mut ensbl, &mut output_diag.as_view_mut())
            .expect("simulation failed");

        assert!(ulps_eq!(
            output[(1, 0)][1],
            -20.99427,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(2, 0)][1],
            -20.91451,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
        assert!(ulps_eq!(
            output[(4, 0)][2],
            -0.069730066,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(
            output_diag[(2, 0)][0],
            0.52137023,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(
            output_diag[(3, 0)][1],
            0.87713426,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(
            output_diag[(4, 0)][0],
            0.21290788,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));

        assert!(ulps_eq!(
            output_diag[(5, 0)][2],
            0.5,
            max_ulps = 5,
            epsilon = 32.0 * f32::EPSILON
        ));
    }

    #[test]
    fn test_core_elliptic_model() {
        let prior = MultivariateDensity::<_, 11>::new(&[
            UniformDensity::new((-1.0, 1.0)).unwrap(),
            UniformDensity::new((0.5, 1.0)).unwrap(),
            UniformDensity::new((-0.5, 0.5)).unwrap(),
            ConstantDensity::new(20.0),
            UniformDensity::new((0.05, 0.25)).unwrap(),
            ConstantDensity::new(1.0),
            ConstantDensity::new(1125.0),
            UniformDensity::new((5.0, 100.0)).unwrap(),
            UniformDensity::new((-10.0, 10.0)).unwrap(),
            ConstantDensity::new(400.0),
            ConstantDensity::new(1.0),
        ]);

        let range = (&prior).get_range();
        let model = COREModel::new(prior);

        let sc = ScObsSeries::<f32>::from_iterator((0..10).map(|i| {
            ScObs::new(
                72.0 * 3600.0 + i as f32 * 2.0 * 3600.0,
                ScObsConf::Position([1.0, 0.0, 0.0]),
            )
        }));

        let mut ensbl = OcnusEnsbl::new(1, range);
        let mut output = DMatrix::<ObserVec<f32, 3>>::zeros(sc.len(), 1);
        let mut output_diag = DMatrix::<ObserVec<f32, 12>>::zeros(sc.len(), 1);

        ensbl.ptpdf.particles_mut().set_column(
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

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ensbl(
                &sc,
                &mut ensbl,
                &COREModel::<f32, MultivariateDensity<f32, 11>>::observe_mag3,
                &mut output.as_view_mut(),
                None::<&mut NullNoise<f32>>,
            )
            .expect("simulation failed");

        model
            .initialize_states_ensbl(&mut ensbl)
            .expect("initialization failed");

        model
            .simulate_ics_basis_ensbl(&sc, &mut ensbl, &mut output_diag.as_view_mut())
            .expect("simulation failed");
    }
}
